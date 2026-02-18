#!/usr/bin/env python3
"""
Build embedding-based ontology-topic linkage for the science data lake.

Uses a sentence embedding model to compute semantic similarity between
OpenAlex topics and ontology terms. This complements the string-based
matching in build_ontology_linkage.py with higher-quality semantic matches.

Pipeline:
  1. Load OpenAlex topics (4,516) and ontology terms (1.3M+) from DuckDB
  2. Embed all texts on GPU using sentence-transformers
  3. Build FAISS index for fast nearest-neighbor search
  4. Filter by cosine similarity threshold
  5. Output to Parquet (datasets/xref/topic_ontology_map/)

Models (trade-off: speed vs quality):
  - all-MiniLM-L6-v2 (default): 22M params, 384d, ~5000 texts/s, STS≈80
  - BAAI/bge-base-en-v1.5:     110M params, 768d, ~1000 texts/s, STS≈82
  - BAAI/bge-large-en-v1.5:    335M params, 1024d, ~80 texts/s, STS≈83

Output:
  datasets/xref/topic_ontology_map/      — Combined best matches per (topic, ontology)
  datasets/xref/topic_ontology_embeddings/ — Cached embeddings for reuse

Usage:
    python scripts/build_embedding_linkage.py
    python scripts/build_embedding_linkage.py --model BAAI/bge-large-en-v1.5
    python scripts/build_embedding_linkage.py --threshold 0.70
    python scripts/build_embedding_linkage.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Resolve data lake root ───────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

DB_PATH = ROOT / "datalake.duckdb"
TOPIC_MAP_DIR = ROOT / "datasets" / "xref" / "topic_ontology_map"
EMBEDDINGS_DIR = ROOT / "datasets" / "xref" / "topic_ontology_embeddings"

# Default model — BGE-large-en-v1.5: best quality for concept matching, 1024-dim, 335M params
# ~840 texts/s on RTX A4500 with batch_size=256 (~6 min for 291K terms)
# For faster runs: sentence-transformers/all-MiniLM-L6-v2 (22M, ~5000 texts/s, STS≈80)
DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"

# Ontology schemas that have terms tables
ONTOLOGY_SCHEMAS = [
    "cso", "doid", "go", "mesh", "chebi", "ncit", "hpo",
    "edam", "agrovoc", "unesco", "stw", "msc2020", "physh",
]

# Ontologies too large for embedding (>100K primary terms).
# These contain mostly entity names (chemicals, species) that won't match
# general academic topics. For these, we fall back to exact label matching.
EMBEDDING_SKIP_ONTOLOGIES = {"mesh", "chebi", "ncit"}

# Threshold for including synonyms: only embed synonyms for ontologies
# with fewer than this many primary terms (to keep embedding set manageable)
SYNONYM_SIZE_THRESHOLD = 50000

# Batch size for encoding (adjust based on GPU memory)
ENCODE_BATCH_SIZE = 256


def load_topics(conn):
    """Load OpenAlex topics from DuckDB."""
    rows = conn.execute("""
        SELECT
            id AS topic_id,
            display_name AS topic_name,
            subfield_display_name AS subfield,
            field_display_name AS field,
            domain_display_name AS domain
        FROM openalex.topics
        ORDER BY id
    """).fetchall()
    topics = []
    for r in rows:
        topics.append({
            "topic_id": r[0],
            "topic_name": r[1],
            "subfield": r[2],
            "field": r[3],
            "domain": r[4],
        })
    return topics


def load_ontology_terms(conn, skip_large=True):
    """Load ontology terms from DuckDB for embedding.

    For large ontologies (MeSH, ChEBI, NCIT) with >100K entity-level terms
    (chemicals, species, etc.), embedding is skipped — these use exact label
    matching instead. For remaining ontologies, primary labels are always
    included; synonyms are included for ontologies below SYNONYM_SIZE_THRESHOLD.

    Returns list of dicts with: ontology, term_id, label, is_synonym.
    """
    available = set()
    for row in conn.execute(
        "SELECT table_schema || '.' || table_name "
        "FROM information_schema.tables"
    ).fetchall():
        available.add(row[0])

    all_terms = []
    exact_only = []  # Track ontologies handled by exact matching
    for onto in ONTOLOGY_SCHEMAS:
        table = f"{onto}.{onto}_terms"
        if table not in available:
            continue

        # Skip large entity-level ontologies for embedding
        if skip_large and onto in EMBEDDING_SKIP_ONTOLOGIES:
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE label IS NOT NULL AND obsolete = false"
            ).fetchone()[0]
            exact_only.append((onto, n))
            continue

        # Check if synonyms column exists
        cols = [r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()]
        has_synonyms = "synonyms" in cols

        # Count primary terms to decide on synonym inclusion
        n_primary = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE label IS NOT NULL AND LENGTH(label) >= 3 AND obsolete = false"
        ).fetchone()[0]
        include_synonyms = has_synonyms and n_primary < SYNONYM_SIZE_THRESHOLD

        if include_synonyms:
            rows = conn.execute(f"""
                SELECT id, label, synonyms
                FROM {table}
                WHERE label IS NOT NULL AND LENGTH(label) >= 3
                  AND obsolete = false
            """).fetchall()
        else:
            rows = conn.execute(f"""
                SELECT id, label, NULL
                FROM {table}
                WHERE label IS NOT NULL AND LENGTH(label) >= 3
                  AND obsolete = false
            """).fetchall()

        for r in rows:
            term_id, label, synonyms = r[0], r[1], r[2]
            # Primary label entry
            all_terms.append({
                "ontology": onto,
                "term_id": term_id,
                "label": label,
                "is_synonym": False,
            })
            # Synonym entries (if included)
            if include_synonyms and synonyms and isinstance(synonyms, list):
                seen = {label.lower()}
                for syn in synonyms:
                    if syn and isinstance(syn, str) and len(syn) >= 3:
                        syn_lower = syn.lower()
                        if syn_lower not in seen:
                            seen.add(syn_lower)
                            all_terms.append({
                                "ontology": onto,
                                "term_id": term_id,
                                "label": syn,
                                "is_synonym": True,
                            })

    if exact_only:
        print(f"  Skipped for embedding (exact-match only):")
        for onto, n in exact_only:
            print(f"    {onto:10s}: {n:>8,} terms")

    return all_terms


def load_exact_match_terms(conn):
    """Load terms from large ontologies for exact label matching only."""
    available = set()
    for row in conn.execute(
        "SELECT table_schema || '.' || table_name "
        "FROM information_schema.tables"
    ).fetchall():
        available.add(row[0])

    exact_terms = []
    for onto in EMBEDDING_SKIP_ONTOLOGIES:
        table = f"{onto}.{onto}_terms"
        if table not in available:
            continue
        rows = conn.execute(f"""
            SELECT id, label
            FROM {table}
            WHERE label IS NOT NULL AND LENGTH(label) >= 3
              AND obsolete = false
        """).fetchall()
        for r in rows:
            exact_terms.append({
                "ontology": onto,
                "term_id": r[0],
                "label": r[1],
            })
    return exact_terms


def embed_texts(model, texts, batch_size=ENCODE_BATCH_SIZE, desc="Encoding"):
    """Encode texts using sentence-transformers model. Returns numpy array."""
    import torch

    print(f"  {desc}: {len(texts):,} texts, batch_size={batch_size}")
    t0 = time.time()

    # BGE models benefit from a query prefix for retrieval
    # But for symmetric similarity between concepts, we skip the prefix
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2 normalize for cosine sim = dot product
        convert_to_numpy=True,
    )

    elapsed = time.time() - t0
    rate = len(texts) / elapsed if elapsed > 0 else 0
    print(f"  Done: {elapsed:.1f}s ({rate:.0f} texts/s), shape={embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings):
    """Build FAISS index for fast nearest-neighbor search."""
    import faiss

    dim = embeddings.shape[1]
    # Use inner product (= cosine similarity since embeddings are L2-normalized)
    index = faiss.IndexFlatIP(dim)
    # Use GPU if available
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        print(f"  FAISS: GPU index, dim={dim}")
    except Exception:
        print(f"  FAISS: CPU index, dim={dim}")
    index.add(embeddings.astype(np.float32))
    return index


def search_matches(topic_embeddings, faiss_index, k=20):
    """Search for top-k nearest ontology terms for each topic."""
    print(f"  Searching top-{k} matches for {len(topic_embeddings):,} topics...")
    t0 = time.time()
    similarities, indices = faiss_index.search(
        topic_embeddings.astype(np.float32), k
    )
    elapsed = time.time() - t0
    print(f"  Search done: {elapsed:.1f}s")
    return similarities, indices


def main():
    parser = argparse.ArgumentParser(
        description="Build embedding-based ontology-topic linkage"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Sentence-transformers model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65,
        help="Cosine similarity threshold (default: 0.65)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--batch-size", type=int, default=ENCODE_BATCH_SIZE,
        help=f"Encoding batch size (default: {ENCODE_BATCH_SIZE})"
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of nearest neighbors to search per topic (default: 20)"
    )
    parser.add_argument(
        "--cache-embeddings", action="store_true", default=True,
        help="Cache embeddings to disk for reuse (default: True)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable embedding cache"
    )
    parser.add_argument(
        "--string-baseline", action="store_true",
        help="Also run string-based matching for comparison"
    )
    args = parser.parse_args()

    if args.no_cache:
        args.cache_embeddings = False

    print(f"Embedding-based ontology-topic matching")
    print(f"  Model: {args.model}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Top-k: {args.top_k}")
    t_start = time.time()

    # ── Step 1: Load data from DuckDB ────────────────────────────────────────

    if not DB_PATH.exists():
        print(f"ERROR: Database not found: {DB_PATH}")
        return 1

    import duckdb
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute(f"SET threads={args.threads}")

    print("\n── Loading data ──")
    topics = load_topics(conn)
    print(f"  OpenAlex topics: {len(topics):,}")

    terms = load_ontology_terms(conn)
    n_primary = sum(1 for t in terms if not t["is_synonym"])
    n_synonym = sum(1 for t in terms if t["is_synonym"])
    print(f"  Embedding terms: {len(terms):,} ({n_primary:,} primary + {n_synonym:,} synonyms)")

    # Per-ontology breakdown
    from collections import Counter
    onto_counts = Counter(t["ontology"] for t in terms)
    for onto in ONTOLOGY_SCHEMAS:
        if onto in onto_counts:
            n_prim = sum(1 for t in terms if t["ontology"] == onto and not t["is_synonym"])
            n_syn = onto_counts[onto] - n_prim
            print(f"    {onto:10s}: {n_prim:>8,} terms + {n_syn:>6,} synonyms = {onto_counts[onto]:>8,}")

    # Load exact-match terms for large ontologies
    exact_terms = load_exact_match_terms(conn)
    print(f"  Exact-match terms: {len(exact_terms):,}")

    # Build exact matches now (fast hash-join in Python)
    print("\n── Exact matching for large ontologies ──")
    topic_name_lower = {t["topic_name"].lower(): t for t in topics}
    exact_matches = []
    for et in exact_terms:
        label_lower = et["label"].lower()
        if label_lower in topic_name_lower:
            topic = topic_name_lower[label_lower]
            exact_matches.append({
                "topic_id": topic["topic_id"],
                "topic_name": topic["topic_name"],
                "subfield": topic["subfield"],
                "field": topic["field"],
                "domain": topic["domain"],
                "ontology_term_id": et["term_id"],
                "ontology_term_label": et["label"],
                "ontology": et["ontology"],
                "similarity": 1.0,
                "match_type": "exact",
            })
    print(f"  Exact matches from large ontologies: {len(exact_matches):,}")

    conn.close()

    if args.dry_run:
        print("\nDry run — stopping before embedding.")
        return 0

    # ── Step 2: Embed all texts ──────────────────────────────────────────────

    print("\n── Loading model ──")
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = SentenceTransformer(args.model, device=device)
    dim = model.get_sentence_embedding_dimension()
    print(f"  Embedding dim: {dim}")

    # Prepare text lists
    topic_texts = [t["topic_name"] for t in topics]
    term_texts = [t["label"] for t in terms]

    # Check for cached embeddings
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_")
    topic_cache = EMBEDDINGS_DIR / f"topic_embeddings_{model_slug}.npy"
    term_cache = EMBEDDINGS_DIR / f"term_embeddings_{model_slug}.npy"
    meta_cache = EMBEDDINGS_DIR / f"embedding_meta_{model_slug}.json"

    topic_embeddings = None
    term_embeddings = None

    if args.cache_embeddings and topic_cache.exists() and term_cache.exists():
        # Verify cache validity
        if meta_cache.exists():
            with open(meta_cache) as f:
                meta = json.load(f)
            if (meta.get("n_topics") == len(topics) and
                    meta.get("n_terms") == len(terms) and
                    meta.get("model") == args.model):
                print(f"\n  Loading cached embeddings...")
                topic_embeddings = np.load(topic_cache)
                term_embeddings = np.load(term_cache)
                print(f"  Topics: {topic_embeddings.shape}, Terms: {term_embeddings.shape}")

    if topic_embeddings is None:
        print("\n── Embedding topics ──")
        topic_embeddings = embed_texts(
            model, topic_texts, batch_size=args.batch_size, desc="Topics"
        )

        print("\n── Embedding ontology terms ──")
        term_embeddings = embed_texts(
            model, term_texts, batch_size=args.batch_size, desc="Terms"
        )

        # Cache embeddings
        if args.cache_embeddings:
            print(f"\n  Caching embeddings to {EMBEDDINGS_DIR}/")
            np.save(topic_cache, topic_embeddings)
            np.save(term_cache, term_embeddings)
            with open(meta_cache, "w") as f:
                json.dump({
                    "model": args.model,
                    "dim": dim,
                    "n_topics": len(topics),
                    "n_terms": len(terms),
                    "n_primary_terms": n_primary,
                    "n_synonym_terms": n_synonym,
                    "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, f, indent=2)

    # Free model from GPU
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Step 3: FAISS search ─────────────────────────────────────────────────

    print("\n── Building FAISS index ──")
    try:
        import faiss
    except ImportError:
        # Fallback: use numpy dot product (slower but no faiss dependency)
        print("  FAISS not available, using numpy fallback")
        faiss = None

    if faiss is not None:
        index = build_faiss_index(term_embeddings)
        similarities, indices = search_matches(
            topic_embeddings, index, k=args.top_k
        )
    else:
        # Numpy fallback — compute full similarity matrix in chunks
        print(f"  Computing similarity matrix (numpy)...")
        t0 = time.time()
        k = args.top_k
        n_topics = len(topics)
        similarities = np.zeros((n_topics, k), dtype=np.float32)
        indices = np.zeros((n_topics, k), dtype=np.int64)

        chunk_size = 100  # Process 100 topics at a time
        for i in range(0, n_topics, chunk_size):
            end = min(i + chunk_size, n_topics)
            # Dot product = cosine sim since normalized
            sims = topic_embeddings[i:end] @ term_embeddings.T
            top_k_idx = np.argpartition(-sims, k, axis=1)[:, :k]
            for j in range(end - i):
                sorted_idx = top_k_idx[j][np.argsort(-sims[j][top_k_idx[j]])]
                indices[i + j] = sorted_idx
                similarities[i + j] = sims[j][sorted_idx]

        elapsed = time.time() - t0
        print(f"  Done: {elapsed:.1f}s")

    # ── Step 4: Build match table ────────────────────────────────────────────

    print(f"\n── Building match table (threshold={args.threshold}) ──")
    matches = []
    for i, topic in enumerate(topics):
        # Track best match per ontology for this topic
        best_per_ontology = {}
        for j in range(args.top_k):
            term_idx = int(indices[i][j])
            sim = float(similarities[i][j])

            if sim < args.threshold:
                continue

            term = terms[term_idx]
            onto = term["ontology"]

            # Keep best match per (topic, ontology)
            if onto not in best_per_ontology or sim > best_per_ontology[onto]["similarity"]:
                best_per_ontology[onto] = {
                    "topic_id": topic["topic_id"],
                    "topic_name": topic["topic_name"],
                    "subfield": topic["subfield"],
                    "field": topic["field"],
                    "domain": topic["domain"],
                    "ontology_term_id": term["term_id"],
                    "ontology_term_label": term["label"],
                    "ontology": onto,
                    "similarity": sim,
                    "match_type": "synonym" if term["is_synonym"] else "label",
                }

        matches.extend(best_per_ontology.values())

    # Merge exact matches from large ontologies
    print(f"  Embedding matches: {len(matches):,}")
    print(f"  Exact matches (large ontologies): {len(exact_matches):,}")

    # Combine, keeping best per (topic_id, ontology)
    combined = {}
    for m in matches + exact_matches:
        key = (m["topic_id"], m["ontology"])
        if key not in combined or m["similarity"] > combined[key]["similarity"]:
            combined[key] = m
    matches = list(combined.values())

    n_topics_matched = len(set(m["topic_id"] for m in matches))
    print(f"  Combined matches: {len(matches):,}")
    print(f"  Topics matched: {n_topics_matched:,} / {len(topics):,}")

    if not matches:
        print("  No matches found! Try lowering the threshold.")
        return 1

    # ── Step 5: Write Parquet output ─────────────────────────────────────────

    print(f"\n── Writing Parquet output ──")
    import duckdb

    conn = duckdb.connect(":memory:")

    # Create table from match list
    conn.execute("""
        CREATE TABLE matches (
            topic_id VARCHAR,
            topic_name VARCHAR,
            subfield VARCHAR,
            field VARCHAR,
            domain VARCHAR,
            ontology_term_id VARCHAR,
            ontology_term_label VARCHAR,
            ontology VARCHAR,
            similarity DOUBLE,
            match_type VARCHAR
        )
    """)

    # Insert in batches
    insert_sql = """
        INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch = []
    for m in matches:
        batch.append((
            m["topic_id"], m["topic_name"], m["subfield"], m["field"],
            m["domain"], m["ontology_term_id"], m["ontology_term_label"],
            m["ontology"], m["similarity"], m["match_type"],
        ))
        if len(batch) >= 10000:
            conn.executemany(insert_sql, batch)
            batch = []
    if batch:
        conn.executemany(insert_sql, batch)

    # Write to Parquet
    TOPIC_MAP_DIR.mkdir(parents=True, exist_ok=True)
    for f in TOPIC_MAP_DIR.glob("*.parquet"):
        f.unlink()

    conn.execute(f"""
        COPY (
            SELECT * FROM matches
            ORDER BY topic_id, similarity DESC
        ) TO '{TOPIC_MAP_DIR}/'
        (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION zstd)
    """)

    parquet_files = list(TOPIC_MAP_DIR.glob("*.parquet"))
    n_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{TOPIC_MAP_DIR}/*.parquet')"
    ).fetchone()[0]

    # ── Summary ──────────────────────────────────────────────────────────────

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Embedding-based topic-ontology map")
    print(f"  Model: {args.model}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Mappings: {n_rows:,}")
    print(f"  Topics matched: {n_topics_matched:,} / {len(topics):,} ({100*n_topics_matched/len(topics):.1f}%)")
    print(f"  Files: {len(parquet_files)}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'='*60}")

    # Per-ontology breakdown
    print(f"\n  Matches per ontology:")
    rows = conn.execute(f"""
        SELECT ontology, COUNT(*) as matches,
               ROUND(AVG(similarity), 3) AS avg_sim,
               ROUND(MIN(similarity), 3) AS min_sim,
               ROUND(MAX(similarity), 3) AS max_sim,
               COUNT(CASE WHEN similarity >= 0.90 THEN 1 END) AS high_quality,
               COUNT(CASE WHEN match_type = 'synonym' THEN 1 END) AS synonym_matches
        FROM read_parquet('{TOPIC_MAP_DIR}/*.parquet')
        GROUP BY ontology
        ORDER BY matches DESC
    """).fetchall()
    print(f"  {'Ontology':10s}  {'Matches':>7s}  {'Avg':>5s}  {'Min':>5s}  {'Max':>5s}  {'HQ≥.9':>6s}  {'Synonym':>7s}")
    for r in rows:
        print(f"  {r[0]:10s}  {r[1]:>7,}  {r[2]:>5.3f}  {r[3]:>5.3f}  {r[4]:>5.3f}  {r[5]:>6,}  {r[6]:>7,}")

    # Quality sample
    print(f"\n  Top 15 highest-similarity matches:")
    samples = conn.execute(f"""
        SELECT topic_name, ontology, ontology_term_label, similarity, match_type
        FROM read_parquet('{TOPIC_MAP_DIR}/*.parquet')
        ORDER BY similarity DESC
        LIMIT 15
    """).fetchall()
    for s in samples:
        print(f"    {s[3]:.3f} [{s[1]:8s}] '{s[0]}' → '{s[2]}' ({s[4]})")

    # Random sample from mid-range
    print(f"\n  15 random mid-range matches ({args.threshold:.2f}-{args.threshold+0.10:.2f}):")
    samples = conn.execute(f"""
        SELECT topic_name, ontology, ontology_term_label, similarity, match_type
        FROM read_parquet('{TOPIC_MAP_DIR}/*.parquet')
        WHERE similarity BETWEEN {args.threshold} AND {args.threshold + 0.10}
        USING SAMPLE 15
    """).fetchall()
    for s in samples:
        print(f"    {s[3]:.3f} [{s[1]:8s}] '{s[0]}' → '{s[2]}' ({s[4]})")

    # Save metadata
    meta_path = TOPIC_MAP_DIR / "embedding_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "method": "embedding_similarity + exact_match",
            "model": args.model,
            "embedding_dim": int(topic_embeddings.shape[1]),
            "cosine_threshold": args.threshold,
            "top_k_search": args.top_k,
            "n_topics": len(topics),
            "n_embedded_terms": len(terms),
            "n_embedded_primary": n_primary,
            "n_embedded_synonyms": n_synonym,
            "n_exact_match_terms": len(exact_terms),
            "exact_match_ontologies": list(EMBEDDING_SKIP_ONTOLOGIES),
            "n_mappings": n_rows,
            "n_topics_matched": n_topics_matched,
            "coverage_pct": round(100 * n_topics_matched / len(topics), 2),
            "per_ontology": {
                r[0]: {
                    "matches": r[1], "avg_sim": r[2], "min_sim": r[3],
                    "max_sim": r[4], "high_quality": r[5], "synonym_matches": r[6],
                }
                for r in rows
            },
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)
    print(f"\n  Metadata: {meta_path}")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
