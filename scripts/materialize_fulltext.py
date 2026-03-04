#!/usr/bin/env python3
"""
Deduplicate full-text papers across sources into a unified dataset.

For each DOI, keeps the best version based on source priority:
    PMC > S2ORC > peS2o > arXiv > bioRxiv > CORE

Tie-breaks by text_length DESC (longest text wins).

Output: datasets/fulltext/parquet/unified/

Usage:
    python scripts/materialize_fulltext.py
    python scripts/materialize_fulltext.py --dry-run
    python scripts/materialize_fulltext.py --summary
"""

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb

# ── Resolve data lake root ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import find_datalake_root

ROOT = find_datalake_root()
FULLTEXT_PARQUET = ROOT / "datasets" / "fulltext" / "parquet"
OUTPUT_DIR = FULLTEXT_PARQUET / "unified"

# Source priority: lower number = higher priority
SOURCE_PRIORITY = {
    "pmc": 1,
    "s2orc": 2,
    "pes2o": 3,
    "arxiv": 4,
    "biorxiv": 5,
    "core": 6,
}

ALL_SOURCES = list(SOURCE_PRIORITY.keys())


def find_available_sources() -> list[str]:
    """Discover which fulltext sources have parquet data."""
    available = []
    for source in ALL_SOURCES:
        source_dir = FULLTEXT_PARQUET / source
        if source_dir.exists() and list(source_dir.glob("*.parquet")):
            available.append(source)
    return available


def materialize(dry_run: bool = False, threads: int = 16):
    """Build the deduplicated unified fulltext dataset."""
    available = find_available_sources()

    if not available:
        print("ERROR: No fulltext sources found. Run convert_fulltext.py first.")
        return False

    print(f"Available sources: {', '.join(available)}")
    t0 = time.time()

    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={threads}")
    conn.execute("SET memory_limit='150GB'")
    conn.execute("SET preserve_insertion_order=false")

    # Build UNION ALL across all available sources
    union_parts = []
    for source in available:
        source_dir = FULLTEXT_PARQUET / source
        path = source_dir / "*.parquet"
        union_parts.append(
            f"SELECT * FROM read_parquet('{path}')"
        )

    union_sql = " UNION ALL ".join(union_parts)

    # Build source priority CASE expression
    priority_cases = " ".join(
        f"WHEN source = '{s}' THEN {p}" for s, p in SOURCE_PRIORITY.items()
    )
    priority_expr = f"CASE {priority_cases} ELSE 99 END"

    # Dedup query: per DOI, keep best source (lowest priority number),
    # tie-break by text_length DESC.
    # Also clean any remaining malformed DOIs (defense-in-depth).
    query = f"""
    SELECT
        clean_doi AS doi, source, title, abstract, text, license, year, source_id,
        text_length, language, has_full_text
    FROM (
        SELECT *,
            CASE
                WHEN doi LIKE '%doi.org/%'
                THEN LOWER(REGEXP_EXTRACT(doi, 'doi\\.org/(.+)$', 1))
                ELSE doi
            END AS clean_doi,
            ROW_NUMBER() OVER (
                PARTITION BY
                    CASE
                        WHEN doi LIKE '%doi.org/%'
                        THEN LOWER(REGEXP_EXTRACT(doi, 'doi\\.org/(.+)$', 1))
                        ELSE doi
                    END
                ORDER BY {priority_expr} ASC, text_length DESC
            ) AS rn
        FROM ({union_sql})
        WHERE doi IS NOT NULL AND doi != '' AND LENGTH(doi) >= 5
    )
    WHERE rn = 1
    """

    if dry_run:
        print("\n-- Deduplication query:")
        print(query)
        conn.close()
        return True

    print("\nStep 1: Building deduplicated dataset...")

    # Export to parquet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing output
    for f in OUTPUT_DIR.glob("*.parquet"):
        f.unlink()

    print(f"Step 2: Exporting to {OUTPUT_DIR}/")
    conn.execute(f"""
        COPY ({query}) TO '{OUTPUT_DIR}/'
        (FORMAT PARQUET, PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 10000,
         COMPRESSION zstd, COMPRESSION_LEVEL 3)
    """)

    # Stats
    parquet_files = sorted(OUTPUT_DIR.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)

    stats = conn.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT doi) AS unique_dois,
            COUNT(*) FILTER (WHERE has_full_text) AS full_text_count,
            COUNT(*) FILTER (WHERE NOT has_full_text) AS abstract_only_count
        FROM read_parquet('{OUTPUT_DIR}/*.parquet')
    """).fetchone()

    # Per-source breakdown (after dedup — shows which source won)
    source_stats = conn.execute(f"""
        SELECT source, COUNT(*) AS count,
               COUNT(*) FILTER (WHERE has_full_text) AS full_text,
               APPROX_QUANTILE(text_length, 0.5) AS median_length
        FROM read_parquet('{OUTPUT_DIR}/*.parquet')
        GROUP BY source ORDER BY count DESC
    """).fetchall()

    conn.close()

    elapsed = time.time() - t0

    print(f"\nDone! Unified fulltext dataset:")
    print(f"  Total rows:     {stats[0]:>12,}")
    print(f"  Unique DOIs:    {stats[1]:>12,}")
    print(f"  Full-text:      {stats[2]:>12,}")
    print(f"  Abstract-only:  {stats[3]:>12,}")
    print(f"  Files:          {len(parquet_files):>12}")
    print(f"  Size:           {total_size / (1024**3):>11.2f} GB")
    print(f"  Time:           {elapsed:>11.1f}s ({elapsed/60:.1f}m)")
    print(f"  Path:           {OUTPUT_DIR}/")

    print(f"\n  Per-source winners:")
    for source, count, full_text, median_len in source_stats:
        print(f"    {source:10s}: {count:>12,} rows, "
              f"{full_text:>12,} full-text, "
              f"median length {int(median_len):,}")

    # Save summary JSON
    summary = {
        "total_rows": stats[0],
        "unique_dois": stats[1],
        "full_text_count": stats[2],
        "abstract_only_count": stats[3],
        "sources_used": available,
        "per_source_winners": {
            row[0]: {"count": row[1], "full_text": row[2]}
            for row in source_stats
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
    }
    summary_path = FULLTEXT_PARQUET / "unified_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")

    return True


def print_summary():
    """Print summary of the unified dataset."""
    summary_path = FULLTEXT_PARQUET / "unified_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print("\n=== Unified Fulltext Summary ===\n")
        print(f"  Total rows:     {summary['total_rows']:>12,}")
        print(f"  Unique DOIs:    {summary['unique_dois']:>12,}")
        print(f"  Full-text:      {summary['full_text_count']:>12,}")
        print(f"  Abstract-only:  {summary['abstract_only_count']:>12,}")
        print(f"  Generated at:   {summary['generated_at']}")
        if "per_source_winners" in summary:
            print(f"\n  Per-source winners:")
            for source, info in summary["per_source_winners"].items():
                print(f"    {source:10s}: {info['count']:>12,} rows, "
                      f"{info['full_text']:>12,} full-text")
        return

    # Fall back to scanning parquet
    if not OUTPUT_DIR.exists() or not list(OUTPUT_DIR.glob("*.parquet")):
        print("  Unified dataset not yet materialized.")
        print("  Run: python scripts/materialize_fulltext.py")
        return

    conn = duckdb.connect(":memory:")
    stats = conn.execute(f"""
        SELECT
            source,
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE has_full_text) AS full_text,
            COUNT(DISTINCT doi) AS dois
        FROM read_parquet('{OUTPUT_DIR}/*.parquet')
        GROUP BY source ORDER BY total DESC
    """).fetchall()
    conn.close()

    print("\n=== Unified Fulltext Summary ===\n")
    total = sum(r[1] for r in stats)
    print(f"  Total: {total:,} papers")
    for source, count, full_text, dois in stats:
        print(f"    {source:10s}: {count:>12,} rows, {full_text:>12,} full-text, {dois:>12,} DOIs")


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate full-text papers into unified dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--dry-run", action="store_true",
                        help="Print SQL queries without executing")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing unified data")
    parser.add_argument("--threads", type=int, default=16,
                        help="Number of DuckDB threads (default: 16)")

    args = parser.parse_args()

    print("=== Full-Text Papers Deduplication ===")
    print(f"Source dir: {FULLTEXT_PARQUET}")
    print(f"Output dir: {OUTPUT_DIR}")

    if args.summary:
        print_summary()
        return 0

    ok = materialize(dry_run=args.dry_run, threads=args.threads)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
