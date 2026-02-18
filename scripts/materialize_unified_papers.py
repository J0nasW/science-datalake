#!/usr/bin/env python3
"""
Materialize the unified_papers cross-reference table as Parquet.

Joins S2AG, OpenAlex, SciSciNet, PWC, Retraction Watch, and Reliance on Science
via normalized DOI to produce one row per unique DOI with coverage flags and
key metrics from each source.

Expected runtime: 10-30 minutes on reference hardware (24-core, 251GB RAM).

Usage:
    python scripts/materialize_unified_papers.py
    python scripts/materialize_unified_papers.py --dry-run   # Print query only
    python scripts/materialize_unified_papers.py --coverage   # Coverage stats only
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
OUTPUT_DIR = ROOT / "datasets" / "xref" / "unified_papers"
COVERAGE_DIR = ROOT / "datasets" / "xref" / "coverage_stats"


def check_available_datasets(conn):
    """Check which datasets are available in the database."""
    available = set()
    for row in conn.execute(
        "SELECT table_schema || '.' || table_name "
        "FROM information_schema.tables"
    ).fetchall():
        available.add(row[0])

    datasets = {
        "openalex": "openalex.works" in available,
        "s2ag": "s2ag.papers" in available,
        "sciscinet": "sciscinet.papers" in available,
        "pwc": "pwc.papers_fulltexts" in available,
        "retwatch": "retwatch.retraction_watch" in available,
        "ros": "ros.patent_paper_pairs" in available,
        "p2p": "p2p.preprint_to_paper" in available,
    }

    print("Available datasets:")
    for name, avail in datasets.items():
        print(f"  {name}: {'YES' if avail else 'NO'}")
    print()

    return datasets


def materialize(conn, output_dir: Path, dry_run: bool = False):
    """Build and export the unified_papers table."""
    datasets = check_available_datasets(conn)

    if not any(datasets.values()):
        print("ERROR: No datasets available. Run create_unified_db.py first.")
        return False

    t0 = time.time()

    # ── Step 1: Collect all distinct DOIs ────────────────────────────────

    print("Step 1: Collecting distinct DOIs from all sources...")
    doi_unions = []

    if datasets["openalex"]:
        doi_unions.append(
            "SELECT LOWER(REPLACE(doi, 'https://doi.org/', '')) AS doi "
            "FROM openalex.works WHERE doi IS NOT NULL"
        )
    if datasets["s2ag"]:
        doi_unions.append(
            "SELECT LOWER(externalids.DOI) AS doi "
            "FROM s2ag.papers WHERE externalids.DOI IS NOT NULL"
        )
    if datasets["sciscinet"]:
        doi_unions.append(
            "SELECT LOWER(REPLACE(doi, 'https://doi.org/', '')) AS doi "
            "FROM sciscinet.papers WHERE doi IS NOT NULL"
        )
    if datasets["pwc"]:
        doi_unions.append(
            "SELECT LOWER(doi) AS doi "
            "FROM pwc.papers_fulltexts WHERE doi IS NOT NULL AND doi != ''"
        )
    if datasets["retwatch"]:
        doi_unions.append(
            "SELECT original_paper_doi AS doi "
            "FROM retwatch.retraction_watch "
            "WHERE original_paper_doi IS NOT NULL AND original_paper_doi != ''"
        )

    doi_sql = " UNION ALL ".join(doi_unions)
    create_dois = (
        f"CREATE OR REPLACE TEMP TABLE all_dois AS "
        f"SELECT DISTINCT doi FROM ({doi_sql}) sub "
        f"WHERE doi IS NOT NULL AND doi != '' AND LENGTH(doi) >= 5"
    )

    if dry_run:
        print(f"\n-- Step 1: Distinct DOIs\n{create_dois};\n")
    else:
        conn.execute(create_dois)
        n_dois = conn.execute("SELECT COUNT(*) FROM all_dois").fetchone()[0]
        print(f"  {n_dois:,} distinct DOIs ({time.time()-t0:.1f}s)")

    # ── Step 2: Build per-source temp tables ─────────────────────────────

    print("Step 2: Building per-source key fields...")

    # OpenAlex (dedup by DOI, keep highest cited)
    if datasets["openalex"]:
        oa_sql = """
        CREATE OR REPLACE TEMP TABLE oa_keyed AS
        SELECT
            LOWER(REPLACE(doi, 'https://doi.org/', '')) AS doi,
            id AS openalex_id,
            display_name AS title,
            publication_year AS year,
            cited_by_count AS oa_cited_by_count,
            type AS oa_type,
            language AS oa_language,
            is_retracted AS oa_is_retracted,
            fwci AS oa_fwci
        FROM openalex.works
        WHERE doi IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY LOWER(REPLACE(doi, 'https://doi.org/', ''))
            ORDER BY cited_by_count DESC NULLS LAST
        ) = 1
        """
        if dry_run:
            print(f"\n-- OpenAlex\n{oa_sql};\n")
        else:
            t1 = time.time()
            conn.execute(oa_sql)
            n = conn.execute("SELECT COUNT(*) FROM oa_keyed").fetchone()[0]
            print(f"  OpenAlex: {n:,} rows ({time.time()-t1:.1f}s)")

    # S2AG (dedup by DOI)
    if datasets["s2ag"]:
        s2_sql = """
        CREATE OR REPLACE TEMP TABLE s2_keyed AS
        SELECT
            LOWER(externalids.DOI) AS doi,
            corpusid AS s2ag_corpusid,
            title AS s2ag_title,
            year AS s2ag_year,
            citationcount AS s2ag_citationcount,
            influentialcitationcount AS s2ag_influentialcitationcount,
            isopenaccess AS s2ag_isopenaccess
        FROM s2ag.papers
        WHERE externalids.DOI IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY LOWER(externalids.DOI)
            ORDER BY citationcount DESC NULLS LAST
        ) = 1
        """
        if dry_run:
            print(f"\n-- S2AG\n{s2_sql};\n")
        else:
            t1 = time.time()
            conn.execute(s2_sql)
            n = conn.execute("SELECT COUNT(*) FROM s2_keyed").fetchone()[0]
            print(f"  S2AG: {n:,} rows ({time.time()-t1:.1f}s)")

    # SciSciNet (dedup by DOI)
    if datasets["sciscinet"]:
        sci_sql = """
        CREATE OR REPLACE TEMP TABLE sci_keyed AS
        SELECT
            LOWER(REPLACE(doi, 'https://doi.org/', '')) AS doi,
            paperid AS sciscinet_paperid,
            year AS sciscinet_year,
            disruption AS sciscinet_disruption,
            Atyp_Median_Z AS sciscinet_atypicality,
            Atyp_10pct_Z AS sciscinet_atypicality_10pct,
            citation_count AS sciscinet_citation_count,
            team_size AS sciscinet_team_size,
            patent_count AS sciscinet_patent_count,
            C3 AS sciscinet_c3,
            C5 AS sciscinet_c5,
            C10 AS sciscinet_c10
        FROM sciscinet.papers
        WHERE doi IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY LOWER(REPLACE(doi, 'https://doi.org/', ''))
            ORDER BY citation_count DESC NULLS LAST
        ) = 1
        """
        if dry_run:
            print(f"\n-- SciSciNet\n{sci_sql};\n")
        else:
            t1 = time.time()
            conn.execute(sci_sql)
            n = conn.execute("SELECT COUNT(*) FROM sci_keyed").fetchone()[0]
            print(f"  SciSciNet: {n:,} rows ({time.time()-t1:.1f}s)")

    # PWC distinct DOIs (for existence flag)
    if datasets["pwc"]:
        pwc_sql = """
        CREATE OR REPLACE TEMP TABLE pwc_dois AS
        SELECT DISTINCT LOWER(doi) AS doi
        FROM pwc.papers_fulltexts
        WHERE doi IS NOT NULL AND doi != ''
        """
        if dry_run:
            print(f"\n-- PWC\n{pwc_sql};\n")
        else:
            t1 = time.time()
            conn.execute(pwc_sql)
            n = conn.execute("SELECT COUNT(*) FROM pwc_dois").fetchone()[0]
            print(f"  PWC: {n:,} DOIs ({time.time()-t1:.1f}s)")

    # Retraction Watch distinct DOIs
    if datasets["retwatch"]:
        rw_sql = """
        CREATE OR REPLACE TEMP TABLE rw_dois AS
        SELECT DISTINCT original_paper_doi AS doi
        FROM retwatch.retraction_watch
        WHERE original_paper_doi IS NOT NULL AND original_paper_doi != ''
        """
        if dry_run:
            print(f"\n-- RetWatch\n{rw_sql};\n")
        else:
            t1 = time.time()
            conn.execute(rw_sql)
            n = conn.execute("SELECT COUNT(*) FROM rw_dois").fetchone()[0]
            print(f"  RetWatch: {n:,} DOIs ({time.time()-t1:.1f}s)")

    # RoS: papers with patent citations (keyed by OpenAlex ID)
    if datasets["ros"]:
        ros_sql = """
        CREATE OR REPLACE TEMP TABLE ros_papers AS
        SELECT DISTINCT
            'https://openalex.org/' || paperid AS openalex_id
        FROM ros.patent_paper_pairs
        """
        if dry_run:
            print(f"\n-- RoS\n{ros_sql};\n")
        else:
            t1 = time.time()
            conn.execute(ros_sql)
            n = conn.execute("SELECT COUNT(*) FROM ros_papers").fetchone()[0]
            print(f"  RoS: {n:,} papers with patents ({time.time()-t1:.1f}s)")

    # ── Step 3: Build the unified join ───────────────────────────────────

    print("Step 3: Joining all sources...")

    # Build SELECT columns and JOINs dynamically
    select_cols = ["d.doi"]
    joins = []

    # Title and year: prefer OpenAlex > S2AG > SciSciNet
    title_parts = []
    year_parts = []

    if datasets["openalex"]:
        select_cols.extend([
            "oa.openalex_id",
            "oa.oa_cited_by_count",
            "oa.oa_type",
            "oa.oa_language",
            "oa.oa_is_retracted",
            "oa.oa_fwci",
        ])
        joins.append("LEFT JOIN oa_keyed oa ON d.doi = oa.doi")
        title_parts.append("oa.title")
        year_parts.append("oa.year")
    else:
        select_cols.extend([
            "NULL::VARCHAR AS openalex_id",
            "NULL::BIGINT AS oa_cited_by_count",
            "NULL::VARCHAR AS oa_type",
            "NULL::VARCHAR AS oa_language",
            "NULL::BOOLEAN AS oa_is_retracted",
            "NULL::DOUBLE AS oa_fwci",
        ])

    if datasets["s2ag"]:
        select_cols.extend([
            "s2.s2ag_corpusid",
            "s2.s2ag_citationcount",
            "s2.s2ag_influentialcitationcount",
            "s2.s2ag_isopenaccess",
        ])
        joins.append("LEFT JOIN s2_keyed s2 ON d.doi = s2.doi")
        title_parts.append("s2.s2ag_title")
        year_parts.append("s2.s2ag_year")
    else:
        select_cols.extend([
            "NULL::BIGINT AS s2ag_corpusid",
            "NULL::BIGINT AS s2ag_citationcount",
            "NULL::BIGINT AS s2ag_influentialcitationcount",
            "NULL::BOOLEAN AS s2ag_isopenaccess",
        ])

    if datasets["sciscinet"]:
        select_cols.extend([
            "sci.sciscinet_paperid",
            "sci.sciscinet_disruption",
            "sci.sciscinet_atypicality",
            "sci.sciscinet_atypicality_10pct",
            "sci.sciscinet_citation_count",
            "sci.sciscinet_team_size",
            "sci.sciscinet_patent_count",
            "sci.sciscinet_c3",
            "sci.sciscinet_c5",
            "sci.sciscinet_c10",
        ])
        joins.append("LEFT JOIN sci_keyed sci ON d.doi = sci.doi")
        title_parts.append("sci.sciscinet_paperid")  # Not used for title; just placeholder
        year_parts.append("sci.sciscinet_year")
    else:
        select_cols.extend([
            "NULL::VARCHAR AS sciscinet_paperid",
            "NULL::DOUBLE AS sciscinet_disruption",
            "NULL::DOUBLE AS sciscinet_atypicality",
            "NULL::DOUBLE AS sciscinet_atypicality_10pct",
            "NULL::UINTEGER AS sciscinet_citation_count",
            "NULL::UINTEGER AS sciscinet_team_size",
            "NULL::UINTEGER AS sciscinet_patent_count",
            "NULL::UINTEGER AS sciscinet_c3",
            "NULL::UINTEGER AS sciscinet_c5",
            "NULL::UINTEGER AS sciscinet_c10",
        ])

    # Build COALESCE for title (OA > S2AG only - SciSciNet has no title in papers table)
    title_sources = [p for p in title_parts if "sciscinet" not in p]
    if title_sources:
        select_cols.insert(1, f"COALESCE({', '.join(title_sources)}) AS title")
    else:
        select_cols.insert(1, "NULL::VARCHAR AS title")

    # Build COALESCE for year (OA > S2AG > SciSciNet)
    if year_parts:
        select_cols.insert(2, f"COALESCE({', '.join(year_parts)}) AS year")
    else:
        select_cols.insert(2, "NULL::BIGINT AS year")

    # Coverage flags
    flags = []
    if datasets["s2ag"]:
        flags.append("s2.s2ag_corpusid IS NOT NULL AS has_s2ag")
    else:
        flags.append("false AS has_s2ag")

    if datasets["openalex"]:
        flags.append("oa.openalex_id IS NOT NULL AS has_openalex")
    else:
        flags.append("false AS has_openalex")

    if datasets["sciscinet"]:
        flags.append("sci.sciscinet_paperid IS NOT NULL AS has_sciscinet")
    else:
        flags.append("false AS has_sciscinet")

    if datasets["pwc"]:
        flags.append("pwc.doi IS NOT NULL AS has_pwc")
        joins.append("LEFT JOIN pwc_dois pwc ON d.doi = pwc.doi")
    else:
        flags.append("false AS has_pwc")

    if datasets["retwatch"]:
        flags.append("rw.doi IS NOT NULL AS has_retraction")
        joins.append("LEFT JOIN rw_dois rw ON d.doi = rw.doi")
    else:
        flags.append("false AS has_retraction")

    if datasets["ros"] and datasets["openalex"]:
        flags.append("ros.openalex_id IS NOT NULL AS has_patent")
        joins.append("LEFT JOIN ros_papers ros ON oa.openalex_id = ros.openalex_id")
    else:
        flags.append("false AS has_patent")

    select_cols.extend(flags)

    # Build final query
    select_str = ",\n    ".join(select_cols)
    join_str = "\n".join(joins)

    query = f"""
    SELECT
        {select_str}
    FROM all_dois d
    {join_str}
    """

    if dry_run:
        print(f"\n-- Step 3: Final query\n{query};\n")
        return True

    # ── Step 4: Export to Parquet ────────────────────────────────────────

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing output
    for f in output_dir.glob("*.parquet"):
        f.unlink()

    output_path = output_dir / "*.parquet"
    print(f"Step 4: Exporting to {output_dir}/")

    t1 = time.time()
    conn.execute(f"""
        COPY ({query}) TO '{output_dir}/'
        (FORMAT PARQUET, PER_THREAD_OUTPUT true, ROW_GROUP_SIZE 250000,
         COMPRESSION zstd)
    """)

    # Count output files and total rows
    parquet_files = sorted(output_dir.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    n_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_dir}/*.parquet')"
    ).fetchone()[0]

    elapsed = time.time() - t0
    print(f"\nDone! Materialized unified_papers:")
    print(f"  Rows: {n_rows:,}")
    print(f"  Files: {len(parquet_files)}")
    print(f"  Size: {total_size / (1024**3):.2f} GB")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  Path: {output_dir}/")

    return True


def compute_coverage_stats(conn, output_dir: Path):
    """Compute coverage overlap statistics for all source combinations."""
    unified_path = OUTPUT_DIR / "*.parquet"
    if not list(OUTPUT_DIR.glob("*.parquet")):
        print("ERROR: unified_papers not materialized yet. Run without --coverage first.")
        return False

    print("Computing coverage statistics from unified_papers...\n")
    t0 = time.time()

    # Register the unified_papers as a view for easier querying
    conn.execute(
        f"CREATE OR REPLACE TEMP VIEW unified AS "
        f"SELECT * FROM read_parquet('{unified_path}')"
    )

    # Total rows
    total = conn.execute("SELECT COUNT(*) FROM unified").fetchone()[0]
    print(f"Total papers (unique DOIs): {total:,}\n")

    # Per-source counts
    print("Per-source coverage:")
    sources = ["s2ag", "openalex", "sciscinet", "pwc", "retraction", "patent"]
    counts = {}
    for src in sources:
        col = f"has_{src}"
        n = conn.execute(f"SELECT COUNT(*) FROM unified WHERE {col}").fetchone()[0]
        counts[src] = n
        pct = 100.0 * n / total if total > 0 else 0
        print(f"  {src:12s}: {n:>13,} ({pct:5.1f}%)")

    # Pairwise overlaps
    print("\nPairwise overlaps:")
    for i, s1 in enumerate(sources):
        for s2 in sources[i+1:]:
            n = conn.execute(
                f"SELECT COUNT(*) FROM unified WHERE has_{s1} AND has_{s2}"
            ).fetchone()[0]
            pct = 100.0 * n / total if total > 0 else 0
            print(f"  {s1:12s} & {s2:12s}: {n:>13,} ({pct:5.1f}%)")

    # Three-way overlap for the big three
    n_all3 = conn.execute(
        "SELECT COUNT(*) FROM unified "
        "WHERE has_s2ag AND has_openalex AND has_sciscinet"
    ).fetchone()[0]
    pct = 100.0 * n_all3 / total if total > 0 else 0
    print(f"\n  All three (S2AG & OA & SSN): {n_all3:>13,} ({pct:5.1f}%)")

    # Full UpSet-style: all 2^6 combinations for the 6 boolean flags
    print("\nFull set intersection counts (UpSet data):")
    output_dir.mkdir(parents=True, exist_ok=True)

    upset_query = f"""
    SELECT
        has_s2ag, has_openalex, has_sciscinet, has_pwc, has_retraction, has_patent,
        COUNT(*) AS count
    FROM unified
    GROUP BY has_s2ag, has_openalex, has_sciscinet, has_pwc, has_retraction, has_patent
    ORDER BY count DESC
    """
    rows = conn.execute(upset_query).fetchall()

    # Print top combinations
    for row in rows[:20]:
        labels = []
        for flag, name in zip(row[:6], sources):
            if flag:
                labels.append(name)
        label = " + ".join(labels) if labels else "(none)"
        print(f"  {label:50s}: {row[6]:>13,}")
    if len(rows) > 20:
        print(f"  ... ({len(rows)} total combinations)")

    # Export as JSON for plotting
    upset_data = []
    for row in rows:
        upset_data.append({
            "has_s2ag": bool(row[0]),
            "has_openalex": bool(row[1]),
            "has_sciscinet": bool(row[2]),
            "has_pwc": bool(row[3]),
            "has_retraction": bool(row[4]),
            "has_patent": bool(row[5]),
            "count": row[6],
        })

    json_path = output_dir / "upset_data.json"
    with open(json_path, "w") as f:
        json.dump(upset_data, f, indent=2)
    print(f"\nUpSet data saved to: {json_path}")

    # Export per-source summary
    summary = {
        "total_unique_dois": total,
        "per_source": counts,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_path = output_dir / "coverage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print(f"\nElapsed: {time.time()-t0:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Materialize unified_papers cross-reference table"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print SQL queries without executing"
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Compute coverage statistics from existing materialized data"
    )
    parser.add_argument(
        "--threads", type=int, default=16,
        help="Number of DuckDB threads (default: 16)"
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found: {DB_PATH}")
        print("Run: python scripts/create_unified_db.py")
        return 1

    conn = duckdb.connect(str(DB_PATH), read_only=args.dry_run or args.coverage)
    conn.execute(f"SET threads={args.threads}")
    conn.execute("SET memory_limit='200GB'")

    try:
        if args.coverage:
            ok = compute_coverage_stats(conn, COVERAGE_DIR)
        else:
            ok = materialize(conn, OUTPUT_DIR, dry_run=args.dry_run)
    finally:
        conn.close()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
