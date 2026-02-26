#!/usr/bin/env python3
"""
Run baseline comparisons for ontology alignment evaluation.

Implements three baselines against the BGE-large embedding method:
  1. TF-IDF cosine similarity (char n-gram based)
  2. BM25 (Okapi BM25 over tokenized labels)
  3. Jaro-Winkler (string similarity at threshold 0.90)

All methods operate on the same 4,516 OpenAlex topics and 291K ontology terms
(the 10 smaller ontologies used for embedding). Large ontologies (MeSH, ChEBI,
NCIT) use exact matching in all methods for consistency.

Output:
  paper/evaluation/baseline_tfidf.parquet
  paper/evaluation/baseline_bm25.parquet
  paper/evaluation/baseline_jaro_winkler.parquet
  paper/evaluation/baseline_comparison.json

Usage:
    python scripts/run_baseline_comparisons.py
    python scripts/run_baseline_comparisons.py --gold-standard paper/evaluation/gold_standard_sample.tsv
"""

import argparse
import json
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
EVAL_DIR = ROOT / "paper" / "evaluation"
TOPIC_MAP_DIR = ROOT / "datasets" / "xref" / "topic_ontology_map"

# Same constants as build_embedding_linkage.py
ONTOLOGY_SCHEMAS = [
    "cso", "doid", "go", "mesh", "chebi", "ncit", "hpo",
    "edam", "agrovoc", "unesco", "stw", "msc2020", "physh",
]
EMBEDDING_SKIP_ONTOLOGIES = {"mesh", "chebi", "ncit"}
SYNONYM_SIZE_THRESHOLD = 50000
SIM_THRESHOLD = 0.65


def load_data(conn):
    """Load topics and ontology terms from DuckDB (reuses build_embedding_linkage logic)."""
    # Load topics
    rows = conn.execute("""
        SELECT id AS topic_id, display_name AS topic_name,
               subfield_display_name AS subfield,
               field_display_name AS field,
               domain_display_name AS domain
        FROM openalex.topics ORDER BY id
    """).fetchall()
    topics = [{"topic_id": r[0], "topic_name": r[1], "subfield": r[2],
               "field": r[3], "domain": r[4]} for r in rows]

    # Available tables
    available = set()
    for row in conn.execute(
        "SELECT table_schema || '.' || table_name FROM information_schema.tables"
    ).fetchall():
        available.add(row[0])

    # Load embedding terms (10 smaller ontologies)
    all_terms = []
    for onto in ONTOLOGY_SCHEMAS:
        table = f"{onto}.{onto}_terms"
        if table not in available or onto in EMBEDDING_SKIP_ONTOLOGIES:
            continue
        cols = [r[0] for r in conn.execute(f"DESCRIBE {table}").fetchall()]
        has_synonyms = "synonyms" in cols
        n_primary = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE label IS NOT NULL AND LENGTH(label) >= 3 AND obsolete = false"
        ).fetchone()[0]
        include_synonyms = has_synonyms and n_primary < SYNONYM_SIZE_THRESHOLD

        if include_synonyms:
            rows = conn.execute(f"""
                SELECT id, label, synonyms FROM {table}
                WHERE label IS NOT NULL AND LENGTH(label) >= 3 AND obsolete = false
            """).fetchall()
        else:
            rows = conn.execute(f"""
                SELECT id, label, NULL FROM {table}
                WHERE label IS NOT NULL AND LENGTH(label) >= 3 AND obsolete = false
            """).fetchall()

        for r in rows:
            term_id, label, synonyms = r[0], r[1], r[2]
            all_terms.append({"ontology": onto, "term_id": term_id, "label": label, "is_synonym": False})
            if include_synonyms and synonyms and isinstance(synonyms, list):
                seen = {label.lower()}
                for syn in synonyms:
                    if syn and isinstance(syn, str) and len(syn) >= 3:
                        syn_lower = syn.lower()
                        if syn_lower not in seen:
                            seen.add(syn_lower)
                            all_terms.append({"ontology": onto, "term_id": term_id, "label": syn, "is_synonym": True})

    # Load exact-match terms (large ontologies)
    exact_terms = []
    for onto in EMBEDDING_SKIP_ONTOLOGIES:
        table = f"{onto}.{onto}_terms"
        if table not in available:
            continue
        rows = conn.execute(f"""
            SELECT id, label FROM {table}
            WHERE label IS NOT NULL AND LENGTH(label) >= 3 AND obsolete = false
        """).fetchall()
        for r in rows:
            exact_terms.append({"ontology": onto, "term_id": r[0], "label": r[1]})

    return topics, all_terms, exact_terms


def get_exact_matches(topics, exact_terms):
    """Exact string matching for large ontologies (shared across all methods)."""
    topic_name_lower = {t["topic_name"].lower(): t for t in topics}
    matches = []
    for et in exact_terms:
        label_lower = et["label"].lower()
        if label_lower in topic_name_lower:
            topic = topic_name_lower[label_lower]
            matches.append({
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
    return matches


def keep_best_per_topic_ontology(matches):
    """Keep best match per (topic_id, ontology) pair."""
    combined = {}
    for m in matches:
        key = (m["topic_id"], m["ontology"])
        if key not in combined or m["similarity"] > combined[key]["similarity"]:
            combined[key] = m
    return list(combined.values())


def run_tfidf_baseline(topics, terms, exact_matches):
    """TF-IDF cosine similarity baseline using character n-grams."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

    print("\n── TF-IDF Baseline ──")
    t0 = time.time()

    topic_texts = [t["topic_name"].lower() for t in topics]
    term_texts = [t["label"].lower() for t in terms]

    print(f"  Vectorizing {len(topic_texts):,} topics + {len(term_texts):,} terms...")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    all_texts = topic_texts + term_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    topic_matrix = tfidf_matrix[:len(topic_texts)]
    term_matrix = tfidf_matrix[len(topic_texts):]

    print(f"  Computing cosine similarity in batches...")
    matches = []
    batch_size = 200  # Process 200 topics at a time
    for i in range(0, len(topics), batch_size):
        end = min(i + batch_size, len(topics))
        sims = sklearn_cosine(topic_matrix[i:end], term_matrix)

        for j in range(end - i):
            topic = topics[i + j]
            # Get indices of terms above threshold
            row = sims[j]
            above_thresh = np.where(row >= SIM_THRESHOLD)[0]
            if len(above_thresh) == 0:
                continue

            # Keep best per ontology
            best_per_ontology = {}
            for idx in above_thresh:
                term = terms[idx]
                onto = term["ontology"]
                sim_val = float(row[idx])
                if onto not in best_per_ontology or sim_val > best_per_ontology[onto]["similarity"]:
                    best_per_ontology[onto] = {
                        "topic_id": topic["topic_id"],
                        "topic_name": topic["topic_name"],
                        "subfield": topic["subfield"],
                        "field": topic["field"],
                        "domain": topic["domain"],
                        "ontology_term_id": term["term_id"],
                        "ontology_term_label": term["label"],
                        "ontology": term["ontology"],
                        "similarity": sim_val,
                        "match_type": "synonym" if term["is_synonym"] else "label",
                    }
            matches.extend(best_per_ontology.values())

        if (i + batch_size) % 1000 == 0 or end == len(topics):
            print(f"    Processed {end}/{len(topics)} topics, {len(matches)} matches so far")

    # Combine with exact matches
    all_matches = keep_best_per_topic_ontology(matches + exact_matches)
    elapsed = time.time() - t0
    n_topics_matched = len(set(m["topic_id"] for m in all_matches))
    print(f"  TF-IDF: {len(all_matches):,} mappings, "
          f"{n_topics_matched:,} topics covered, {elapsed:.1f}s")
    return all_matches


def run_bm25_baseline(topics, terms, exact_matches):
    """BM25 baseline over tokenized labels."""
    from rank_bm25 import BM25Okapi

    print("\n── BM25 Baseline ──")
    t0 = time.time()

    # Tokenize term labels
    print(f"  Building BM25 index over {len(terms):,} terms...")
    tokenized_terms = [t["label"].lower().split() for t in terms]
    bm25 = BM25Okapi(tokenized_terms)

    print(f"  Querying {len(topics):,} topics (top-20 each)...")
    matches = []
    for i, topic in enumerate(topics):
        query = topic["topic_name"].lower().split()
        scores = bm25.get_scores(query)

        # Get top-20 indices
        top_k = min(20, len(scores))
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # Min-max normalize scores to [0, 1]
        min_s = scores.min()
        max_s = scores.max()
        if max_s > min_s:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(scores)

        # Keep best per ontology above threshold
        best_per_ontology = {}
        for idx in top_indices:
            term = terms[idx]
            onto = term["ontology"]
            sim_val = float(norm_scores[idx])
            if sim_val < SIM_THRESHOLD:
                continue
            if onto not in best_per_ontology or sim_val > best_per_ontology[onto]["similarity"]:
                best_per_ontology[onto] = {
                    "topic_id": topic["topic_id"],
                    "topic_name": topic["topic_name"],
                    "subfield": topic["subfield"],
                    "field": topic["field"],
                    "domain": topic["domain"],
                    "ontology_term_id": term["term_id"],
                    "ontology_term_label": term["label"],
                    "ontology": term["ontology"],
                    "similarity": sim_val,
                    "match_type": "synonym" if term["is_synonym"] else "label",
                }
        matches.extend(best_per_ontology.values())

        if (i + 1) % 500 == 0 or i == len(topics) - 1:
            print(f"    Processed {i+1}/{len(topics)} topics, {len(matches)} matches so far")

    # Combine with exact matches
    all_matches = keep_best_per_topic_ontology(matches + exact_matches)
    elapsed = time.time() - t0
    n_topics_matched = len(set(m["topic_id"] for m in all_matches))
    print(f"  BM25: {len(all_matches):,} mappings, "
          f"{n_topics_matched:,} topics covered, {elapsed:.1f}s")
    return all_matches


def run_jaro_winkler_baseline(conn, topics, exact_matches, threshold=0.90):
    """Jaro-Winkler string similarity baseline (via DuckDB)."""
    print(f"\n── Jaro-Winkler Baseline (threshold={threshold}) ──")
    t0 = time.time()

    available = set()
    for row in conn.execute(
        "SELECT table_schema || '.' || table_name FROM information_schema.tables"
    ).fetchall():
        available.add(row[0])

    # Create temp table with topics
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE oa_topics AS
        SELECT id AS topic_id, display_name AS topic_name,
               subfield_display_name AS subfield,
               field_display_name AS field,
               domain_display_name AS domain
        FROM openalex.topics
    """)

    matches = []
    for onto in ONTOLOGY_SCHEMAS:
        table = f"{onto}.{onto}_terms"
        if table not in available or onto in EMBEDDING_SKIP_ONTOLOGIES:
            continue

        n_terms = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE label IS NOT NULL AND obsolete = false"
        ).fetchone()[0]

        print(f"  Matching against {onto} ({n_terms:,} terms)...", end=" ", flush=True)
        try:
            rows = conn.execute(f"""
                SELECT t.topic_id, t.topic_name, t.subfield, t.field, t.domain,
                       o.id AS ontology_term_id, o.label AS ontology_term_label,
                       '{onto}' AS ontology,
                       jaro_winkler_similarity(LOWER(t.topic_name), LOWER(o.label)) AS similarity
                FROM oa_topics t
                CROSS JOIN {table} o
                WHERE o.label IS NOT NULL AND LENGTH(o.label) >= 3
                  AND o.obsolete = false
                  AND jaro_winkler_similarity(LOWER(t.topic_name), LOWER(o.label)) >= {threshold}
            """).fetchall()
            print(f"{len(rows)} matches")
            for r in rows:
                matches.append({
                    "topic_id": r[0], "topic_name": r[1], "subfield": r[2],
                    "field": r[3], "domain": r[4], "ontology_term_id": r[5],
                    "ontology_term_label": r[6], "ontology": r[7],
                    "similarity": float(r[8]), "match_type": "label",
                })
        except Exception as e:
            print(f"ERROR: {e}")

    # Combine with exact matches
    all_matches = keep_best_per_topic_ontology(matches + exact_matches)
    elapsed = time.time() - t0
    n_topics_matched = len(set(m["topic_id"] for m in all_matches))
    print(f"  Jaro-Winkler: {len(all_matches):,} mappings, "
          f"{n_topics_matched:,} topics covered, {elapsed:.1f}s")
    return all_matches


def write_parquet(matches, output_path):
    """Write matches to Parquet file."""
    import duckdb as _duckdb
    mem = _duckdb.connect(":memory:")
    mem.execute("""
        CREATE TABLE matches (
            topic_id VARCHAR, topic_name VARCHAR, subfield VARCHAR,
            field VARCHAR, domain VARCHAR, ontology_term_id VARCHAR,
            ontology_term_label VARCHAR, ontology VARCHAR,
            similarity DOUBLE, match_type VARCHAR
        )
    """)
    insert_sql = "INSERT INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    batch = []
    for m in matches:
        batch.append((
            m["topic_id"], m["topic_name"], m["subfield"], m["field"],
            m["domain"], m["ontology_term_id"], m["ontology_term_label"],
            m["ontology"], m["similarity"], m["match_type"],
        ))
        if len(batch) >= 10000:
            mem.executemany(insert_sql, batch)
            batch = []
    if batch:
        mem.executemany(insert_sql, batch)
    mem.execute(f"""
        COPY (SELECT * FROM matches ORDER BY topic_id, similarity DESC)
        TO '{output_path}' (FORMAT PARQUET, COMPRESSION zstd)
    """)
    mem.close()


def evaluate_on_gold_standard(method_matches, gold_path):
    """Evaluate a method's matches against gold-standard annotations.

    Returns dict with per-stratum and aggregate metrics, plus a PR curve.
    """
    if not gold_path.exists():
        return None

    # Load gold standard
    annotations = []
    with open(gold_path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                continue
            d = dict(zip(header, fields))
            d["similarity"] = float(d["similarity"])
            annotations.append(d)

    # Check if annotated
    unannotated = [a for a in annotations if a.get("label", "").strip() == ""]
    if unannotated:
        return None

    # Build lookup of method's matches: (topic_name, ontology) -> best similarity
    method_lookup = {}
    for m in method_matches:
        key = (m["topic_name"], m["ontology"])
        if key not in method_lookup or m["similarity"] > method_lookup[key]:
            method_lookup[key] = m["similarity"]

    # For each gold-standard pair, check if method found it
    total_correct = sum(1 for a in annotations if a["label"].strip().lower() == "correct")
    results_at_threshold = {}

    for thresh_int in range(60, 101):
        thresh = thresh_int / 100.0
        # Gold-standard pairs above this threshold (from the BGE-large system)
        gold_above = [a for a in annotations if a["similarity"] >= thresh]
        if not gold_above:
            results_at_threshold[thresh] = {"precision": 1.0, "recall": 0.0, "n": 0}
            continue

        # For each gold pair above threshold, check if this method also finds it
        method_tp = 0
        method_found = 0
        for a in gold_above:
            key = (a["topic_name"], a["ontology"])
            method_sim = method_lookup.get(key, 0.0)
            if method_sim >= SIM_THRESHOLD:  # Method found this pair at all
                method_found += 1
                if a["label"].strip().lower() == "correct":
                    method_tp += 1

        precision = method_tp / method_found if method_found > 0 else 0
        recall = method_tp / total_correct if total_correct > 0 else 0
        results_at_threshold[thresh] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n": method_found,
        }

    # Build PR curve
    pr_curve = []
    for thresh_int in range(60, 101):
        thresh = thresh_int / 100.0
        r = results_at_threshold[thresh]
        pr_curve.append({
            "threshold": round(thresh, 2),
            "precision": r["precision"],
            "recall": r["recall"],
            "n": r["n"],
        })

    # Aggregate metrics at the baseline threshold
    n_found = sum(1 for key in method_lookup if method_lookup[key] >= SIM_THRESHOLD)
    n_topics = len(set(m["topic_id"] for m in method_matches))

    return {
        "n_mappings": len(method_matches),
        "n_topics_covered": n_topics,
        "pr_curve": pr_curve,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline comparisons for ontology alignment"
    )
    parser.add_argument("--gold-standard", type=str,
                        default=str(EVAL_DIR / "gold_standard_sample.tsv"),
                        help="Path to annotated gold standard TSV")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    gold_path = Path(args.gold_standard)
    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────────
    import duckdb
    print("Loading data from DuckDB...")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute(f"SET threads={args.threads}")

    topics, terms, exact_terms = load_data(conn)
    print(f"  Topics: {len(topics):,}")
    print(f"  Embedding terms: {len(terms):,}")
    print(f"  Exact-match terms: {len(exact_terms):,}")

    exact_matches = get_exact_matches(topics, exact_terms)
    print(f"  Exact matches (large ontologies): {len(exact_matches):,}")

    # ── Run baselines ────────────────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. TF-IDF
    tfidf_matches = run_tfidf_baseline(topics, terms, exact_matches)
    tfidf_path = EVAL_DIR / "baseline_tfidf.parquet"
    write_parquet(tfidf_matches, tfidf_path)
    print(f"  Saved to {tfidf_path}")

    # 2. BM25
    bm25_matches = run_bm25_baseline(topics, terms, exact_matches)
    bm25_path = EVAL_DIR / "baseline_bm25.parquet"
    write_parquet(bm25_matches, bm25_path)
    print(f"  Saved to {bm25_path}")

    # 3. Jaro-Winkler
    jw_conn = duckdb.connect(str(DB_PATH), read_only=True)
    jw_conn.execute(f"SET threads={args.threads}")
    jw_matches = run_jaro_winkler_baseline(jw_conn, topics, exact_matches)
    jw_conn.close()
    jw_path = EVAL_DIR / "baseline_jaro_winkler.parquet"
    write_parquet(jw_matches, jw_path)
    print(f"  Saved to {jw_path}")

    conn.close()

    # ── Load BGE-large matches for comparison ────────────────────────────────
    bge_conn = duckdb.connect(":memory:")
    bge_matches_raw = bge_conn.execute(f"""
        SELECT topic_id, topic_name, subfield, field, domain,
               ontology_term_id, ontology_term_label, ontology,
               similarity, match_type
        FROM read_parquet('{TOPIC_MAP_DIR}/*.parquet')
    """).fetchall()
    bge_conn.close()
    bge_columns = ["topic_id", "topic_name", "subfield", "field", "domain",
                   "ontology_term_id", "ontology_term_label", "ontology",
                   "similarity", "match_type"]
    bge_matches = [dict(zip(bge_columns, r)) for r in bge_matches_raw]

    # ── Evaluate on gold standard (if annotated) ─────────────────────────────
    comparison = {}
    methods = {
        "tfidf": tfidf_matches,
        "bm25": bm25_matches,
        "jaro_winkler": jw_matches,
        "bge_large": bge_matches,
    }

    for method_name, method_matches in methods.items():
        n_mappings = len(method_matches)
        n_topics = len(set(m["topic_id"] for m in method_matches))
        comparison[method_name] = {
            "n_mappings": n_mappings,
            "n_topics_covered": n_topics,
        }

        gold_eval = evaluate_on_gold_standard(method_matches, gold_path)
        if gold_eval:
            comparison[method_name].update(gold_eval)

    # ── Print summary ────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Baseline Comparison Summary")
    print(f"{'='*60}")
    print(f"  {'Method':<16s} {'Mappings':>10s} {'Topics':>8s}")
    print(f"  {'-'*36}")
    for method_name in ["jaro_winkler", "tfidf", "bm25", "bge_large"]:
        m = comparison[method_name]
        print(f"  {method_name:<16s} {m['n_mappings']:>10,} {m['n_topics_covered']:>8,}")
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Save comparison JSON ─────────────────────────────────────────────────
    comparison_path = EVAL_DIR / "baseline_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Comparison saved to {comparison_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
