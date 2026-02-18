#!/usr/bin/env python3
"""
Build ontology-paper linkage tables for the science data lake.

NOTE: For Tier 2 (topic-ontology alignment), prefer build_embedding_linkage.py
which uses BGE-large embeddings for much better results (16K matches vs 937).
This script's Tier 2 is kept as a string-based baseline / fallback.

Creates three types of linkage:
  Tier 2: OpenAlex topic → ontology term alignment via label similarity (baseline)
  Tier 3: Cross-ontology bridges via shared external IDs (UMLS, Wikidata, etc.)

Note: Tier 1 (PWC→CSO direct paper mapping) is not available — PWC's cso_topics
table is a vocabulary list, not a paper-to-topic mapping.

Output:
  datasets/xref/topic_ontology_map/  — OpenAlex topic ↔ ontology term alignment
  datasets/xref/ontology_bridges/    — Cross-ontology bridges via shared xrefs

Usage:
    python scripts/build_ontology_linkage.py
    python scripts/build_ontology_linkage.py --dry-run
    python scripts/build_ontology_linkage.py --threshold 0.85  # Similarity threshold
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import duckdb

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
BRIDGES_DIR = ROOT / "datasets" / "xref" / "ontology_bridges"

# Ontology schemas that have terms tables
ONTOLOGY_SCHEMAS = [
    "cso", "doid", "go", "mesh", "chebi", "ncit", "hpo",
    "edam", "agrovoc", "unesco", "stw", "msc2020", "physh",
]

# Ontologies with >50K terms: use exact/prefix matching instead of full CROSS JOIN
# with jaro_winkler to avoid multi-billion row comparisons
LARGE_ONTOLOGY_THRESHOLD = 50000

# Domain mapping: which OpenAlex domains should match which ontologies
DOMAIN_ONTOLOGY_MAP = {
    # OpenAlex domain → list of (ontology_schema, terms_table) to match against
    "Physical Sciences": ["physh", "msc2020"],
    "Life Sciences": ["go", "chebi", "doid", "mesh", "ncit", "hpo"],
    "Social Sciences": ["unesco", "stw", "agrovoc"],
    "Health Sciences": ["doid", "mesh", "ncit", "hpo"],
    # CS topics match CSO regardless of domain
}


def check_available(conn):
    """Check which ontology tables exist."""
    available = set()
    for row in conn.execute(
        "SELECT table_schema || '.' || table_name "
        "FROM information_schema.tables"
    ).fetchall():
        available.add(row[0])
    return available


def build_topic_ontology_map(conn, output_dir: Path, threshold: float = 0.80,
                              dry_run: bool = False):
    """Tier 2: Match OpenAlex topics to ontology terms by label similarity.

    Uses DuckDB's jaro_winkler_similarity() for fuzzy matching.
    """
    available = check_available(conn)

    if "openalex.topics" not in available:
        print("  OpenAlex topics not available, skipping.")
        return False

    print("Tier 2: OpenAlex topic → ontology term alignment")
    print(f"  Similarity threshold: {threshold}")
    t0 = time.time()

    # Load OpenAlex topics into temp table
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE oa_topics AS
        SELECT
            id AS topic_id,
            display_name AS topic_name,
            subfield_display_name AS subfield,
            field_display_name AS field,
            domain_display_name AS domain
        FROM openalex.topics
    """)
    n_topics = conn.execute("SELECT COUNT(*) FROM oa_topics").fetchone()[0]
    print(f"  OpenAlex topics: {n_topics}")

    # For each available ontology, match topic labels
    all_matches = []

    for onto in ONTOLOGY_SCHEMAS:
        terms_table = f"{onto}.{onto}_terms"
        if terms_table not in available:
            continue

        # Check ontology size to choose matching strategy
        n_terms = conn.execute(
            f"SELECT COUNT(*) FROM {terms_table} WHERE label IS NOT NULL AND obsolete = false"
        ).fetchone()[0]

        if n_terms > LARGE_ONTOLOGY_THRESHOLD:
            # Large ontology: exact topic_name match only (via hash join)
            print(f"  Matching against {onto} ({n_terms:,} terms, exact)...",
                  end=" ", flush=True)
            match_sql = f"""
            SELECT t.topic_id, t.topic_name, t.subfield, t.field, t.domain,
                   o.id AS ontology_term_id, o.label AS ontology_term_label,
                   '{onto}' AS ontology,
                   1.0 AS name_similarity
            FROM oa_topics t
            JOIN {terms_table} o
                ON LOWER(o.label) = LOWER(t.topic_name)
            WHERE o.label IS NOT NULL AND LENGTH(o.label) >= 3
              AND o.obsolete = false
            """
        else:
            # Small ontology: fuzzy match on topic_name only (not subfield/field)
            print(f"  Matching against {onto} ({n_terms:,} terms, fuzzy)...",
                  end=" ", flush=True)
            match_sql = f"""
            SELECT
                t.topic_id,
                t.topic_name,
                t.subfield,
                t.field,
                t.domain,
                o.id AS ontology_term_id,
                o.label AS ontology_term_label,
                '{onto}' AS ontology,
                jaro_winkler_similarity(LOWER(t.topic_name), LOWER(o.label)) AS name_similarity
            FROM oa_topics t
            CROSS JOIN {terms_table} o
            WHERE o.label IS NOT NULL AND LENGTH(o.label) >= 3
              AND o.obsolete = false
              AND jaro_winkler_similarity(LOWER(t.topic_name), LOWER(o.label)) >= {threshold}
            """

        if dry_run:
            print(f"(dry run)")
            continue

        try:
            rows = conn.execute(match_sql).fetchall()
            print(f"{len(rows)} matches")
            all_matches.append((onto, match_sql, len(rows)))
        except Exception as e:
            print(f"ERROR: {e}")

    if dry_run:
        return True

    # Build combined table with best match per (topic, ontology) pair
    print("  Building combined mapping table...")

    union_parts = []
    for onto in ONTOLOGY_SCHEMAS:
        terms_table = f"{onto}.{onto}_terms"
        if terms_table not in available:
            continue

        n_terms = conn.execute(
            f"SELECT COUNT(*) FROM {terms_table} WHERE label IS NOT NULL AND obsolete = false"
        ).fetchone()[0]

        if n_terms > LARGE_ONTOLOGY_THRESHOLD:
            # Large ontology: exact topic_name match only
            union_parts.append(f"""
            SELECT t.topic_id, t.topic_name, t.subfield, t.field, t.domain,
                   o.id AS ontology_term_id, o.label AS ontology_term_label,
                   '{onto}' AS ontology,
                   1.0 AS name_similarity,
                   1.0 AS best_similarity
            FROM oa_topics t
            JOIN {terms_table} o ON LOWER(o.label) = LOWER(t.topic_name)
            WHERE o.label IS NOT NULL AND LENGTH(o.label) >= 3 AND o.obsolete = false
            """)
        else:
            # Small ontology: fuzzy match on topic_name only
            union_parts.append(f"""
            SELECT
                t.topic_id,
                t.topic_name,
                t.subfield,
                t.field,
                t.domain,
                o.id AS ontology_term_id,
                o.label AS ontology_term_label,
                '{onto}' AS ontology,
                jaro_winkler_similarity(LOWER(t.topic_name), LOWER(o.label)) AS name_similarity,
                jaro_winkler_similarity(LOWER(t.topic_name), LOWER(o.label)) AS best_similarity
            FROM oa_topics t
            CROSS JOIN {terms_table} o
            WHERE o.label IS NOT NULL AND LENGTH(o.label) >= 3
              AND o.obsolete = false
              AND jaro_winkler_similarity(LOWER(t.topic_name), LOWER(o.label)) >= {threshold}
            """)

    if not union_parts:
        print("  No ontologies available for matching.")
        return False

    combined_sql = " UNION ALL ".join(union_parts)

    # Keep only the best match per (topic, ontology) pair
    final_sql = f"""
    SELECT * FROM (
        {combined_sql}
    ) all_matches
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY topic_id, ontology
        ORDER BY best_similarity DESC
    ) = 1
    ORDER BY topic_id, best_similarity DESC
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("*.parquet"):
        f.unlink()

    conn.execute(f"""
        COPY ({final_sql}) TO '{output_dir}/'
        (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION zstd)
    """)

    # Count results
    parquet_files = list(output_dir.glob("*.parquet"))
    n_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_dir}/*.parquet')"
    ).fetchone()[0]
    n_topics_matched = conn.execute(
        f"SELECT COUNT(DISTINCT topic_id) FROM read_parquet('{output_dir}/*.parquet')"
    ).fetchone()[0]

    elapsed = time.time() - t0
    print(f"\n  Topic-ontology map:")
    print(f"    Mappings: {n_rows:,}")
    print(f"    Topics matched: {n_topics_matched:,} / {n_topics}")
    print(f"    Files: {len(parquet_files)}")
    print(f"    Time: {elapsed:.1f}s")

    # Quality summary by ontology
    print("\n  Matches per ontology:")
    rows = conn.execute(f"""
        SELECT ontology, COUNT(*) as matches,
               ROUND(AVG(best_similarity), 3) AS avg_sim,
               ROUND(MIN(best_similarity), 3) AS min_sim,
               COUNT(CASE WHEN best_similarity >= 0.95 THEN 1 END) AS exact_matches
        FROM read_parquet('{output_dir}/*.parquet')
        GROUP BY ontology
        ORDER BY matches DESC
    """).fetchall()
    for r in rows:
        print(f"    {r[0]:10s}: {r[1]:6,} matches (avg={r[2]}, min={r[3]}, exact={r[4]})")

    return True


def build_ontology_bridges(conn, output_dir: Path, dry_run: bool = False):
    """Tier 3: Cross-ontology bridges via shared external IDs.

    Finds ontology terms that share xref IDs (e.g., UMLS CUIs, Wikidata IDs).
    """
    available = check_available(conn)
    print("\nTier 3: Cross-ontology bridges via shared xrefs")
    t0 = time.time()

    # Find which ontologies have xrefs tables
    xref_tables = []
    for onto in ONTOLOGY_SCHEMAS:
        table = f"{onto}.{onto}_xrefs"
        if table in available:
            xref_tables.append(onto)

    print(f"  Ontologies with xrefs: {', '.join(xref_tables)}")

    if len(xref_tables) < 2:
        print("  Need at least 2 ontologies with xrefs for bridges.")
        return False

    # Normalize xref_db names across ontologies for bridge matching
    # Different ontologies use different names for the same database
    normalize_xref_db = """
        CASE
            WHEN xref_db IN ('UMLS_CUI', 'UMLS') THEN 'UMLS'
            WHEN xref_db LIKE 'SNOMEDCT%' THEN 'SNOMEDCT'
            WHEN xref_db IN ('NCI', 'NCIT') THEN 'NCIT'
            WHEN xref_db IN ('MESH', 'MSH') THEN 'MESH'
            WHEN xref_db IN ('MIM', 'OMIM') THEN 'OMIM'
            WHEN xref_db IN ('exactMatch', 'closeMatch', 'relatedMatch', 'sameAs') THEN xref_db
            ELSE xref_db
        END
    """

    # Build pairwise bridges using normalized xref_db names
    union_parts = []
    for i, onto1 in enumerate(xref_tables):
        for onto2 in xref_tables[i+1:]:
            union_parts.append(f"""
            SELECT
                '{onto1}' AS ontology_1,
                x1.term_id AS term_id_1,
                '{onto2}' AS ontology_2,
                x2.term_id AS term_id_2,
                ({normalize_xref_db.replace('xref_db', 'x1.xref_db')}) AS bridge_type,
                x1.xref_id AS bridge_id
            FROM {onto1}.{onto1}_xrefs x1
            JOIN {onto2}.{onto2}_xrefs x2
                ON ({normalize_xref_db.replace('xref_db', 'x1.xref_db')}) =
                   ({normalize_xref_db.replace('xref_db', 'x2.xref_db')})
                AND x1.xref_id = x2.xref_id
            WHERE x1.xref_id IS NOT NULL AND x1.xref_id != ''
              AND x2.xref_id IS NOT NULL AND x2.xref_id != ''
            """)

    if not union_parts:
        print("  No bridge combinations available.")
        return False

    bridge_sql = " UNION ALL ".join(union_parts)

    if dry_run:
        print(f"  Would execute {len(union_parts)} pairwise bridge queries.")
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("*.parquet"):
        f.unlink()

    print(f"  Computing {len(union_parts)} pairwise bridges...")
    conn.execute(f"""
        COPY ({bridge_sql}) TO '{output_dir}/'
        (FORMAT PARQUET, PER_THREAD_OUTPUT true, COMPRESSION zstd)
    """)

    parquet_files = list(output_dir.glob("*.parquet"))
    n_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_dir}/*.parquet')"
    ).fetchone()[0]

    elapsed = time.time() - t0
    print(f"\n  Ontology bridges:")
    print(f"    Bridges: {n_rows:,}")
    print(f"    Files: {len(parquet_files)}")
    print(f"    Time: {elapsed:.1f}s")

    # Summary by pair
    print("\n  Bridges per ontology pair:")
    rows = conn.execute(f"""
        SELECT ontology_1, ontology_2, bridge_type, COUNT(*) as count
        FROM read_parquet('{output_dir}/*.parquet')
        GROUP BY ontology_1, ontology_2, bridge_type
        ORDER BY count DESC
        LIMIT 20
    """).fetchall()
    for r in rows:
        print(f"    {r[0]:8s} ↔ {r[1]:8s} via {r[2]:20s}: {r[3]:>8,}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build ontology-paper linkage tables"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Jaro-Winkler similarity threshold for topic matching (default: 0.90)"
    )
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument(
        "--skip-bridges", action="store_true",
        help="Skip cross-ontology bridge computation"
    )
    parser.add_argument(
        "--skip-topics", action="store_true",
        help="Skip OpenAlex topic matching"
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found: {DB_PATH}")
        return 1

    conn = duckdb.connect(str(DB_PATH), read_only=args.dry_run)
    conn.execute(f"SET threads={args.threads}")
    conn.execute("SET memory_limit='200GB'")

    try:
        if not args.skip_topics:
            build_topic_ontology_map(
                conn, TOPIC_MAP_DIR,
                threshold=args.threshold,
                dry_run=args.dry_run,
            )

        if not args.skip_bridges:
            build_ontology_bridges(
                conn, BRIDGES_DIR,
                dry_run=args.dry_run,
            )
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
