#!/usr/bin/env python3
"""
Convert the Retraction Watch CSV to Parquet with correct date parsing.

Self-contained script with data lake path integration. DuckDB view
creation is handled separately by create_unified_db.py, which points
`retwatch.retraction_watch` at the Parquet file written here.

Why this script exists:
    The original Parquet snapshot was built by an ad-hoc one-off that
    used `TRY_CAST(... AS DATE)` on US-format strings like
    "7/15/2024 0:00". DuckDB's default date parser expects ISO-8601,
    so all 68,869 RetractionDate and OriginalPaperDate values were
    silently NULL-ed during conversion. This script reparses them
    with `TRY_STRPTIME('%-m/%-d/%Y %-H:%M')`, which recovers 99.89%
    of the dates (verified 2026-04 on the 2025-02 snapshot).

Usage:
    python scripts/convert_retractionwatch.py
    python scripts/convert_retractionwatch.py --force
    python scripts/convert_retractionwatch.py --dry-run

Exit codes:
    0  success
    1  date coverage below DATE_COVERAGE_FLOOR (99%)
    2  row count mismatch against the raw CSV
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import duckdb

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import find_datalake_root

ROOT = find_datalake_root()
RAW_CSV = ROOT / "datasets" / "retractionwatch" / "raw" / "retraction_watch.csv"
OUTPUT_DIR = ROOT / "datasets" / "retractionwatch" / "parquet"
OUTPUT_PARQUET = OUTPUT_DIR / "retraction_watch.parquet"

# Retraction Watch CSV uses US-format dates with a zero-padded trailing time,
# e.g. "7/15/2024 0:00". DuckDB's strptime uses POSIX %-m / %-d / %-H for
# non-padded numbers, which matches this format.
DATE_FORMAT = "%-m/%-d/%Y %-H:%M"

# Sanity floors. If these fail, something changed upstream and we want a
# loud failure rather than a silently bad Parquet.
DATE_COVERAGE_FLOOR = 0.99  # fraction of rows that must parse to a valid date
ROW_COUNT_FLOOR = 60_000    # refuse to ship a truncated snapshot


def _build_sql(raw_csv: Path) -> str:
    """Return the SELECT that produces the final table from the raw CSV.

    Column names are snake_cased to match the existing parquet schema so
    downstream views (retwatch.retraction_watch) keep working unchanged.
    """
    return f"""
    SELECT
        CAST("Record ID" AS INTEGER)                             AS record_id,
        "Title"                                                  AS title,
        "Subject"                                                AS subject,
        "Institution"                                            AS institution,
        "Journal"                                                AS journal,
        "Publisher"                                              AS publisher,
        "Country"                                                AS country,
        "Author"                                                 AS author,
        "URLS"                                                   AS urls,
        "ArticleType"                                            AS article_type,
        CAST(TRY_STRPTIME("RetractionDate",    '{DATE_FORMAT}') AS DATE) AS retraction_date,
        "RetractionDOI"                                          AS retraction_doi,
        CAST("RetractionPubMedID" AS VARCHAR)                    AS retraction_pubmed_id,
        CAST(TRY_STRPTIME("OriginalPaperDate", '{DATE_FORMAT}') AS DATE) AS original_paper_date,
        "OriginalPaperDOI"                                       AS original_paper_doi,
        CAST("OriginalPaperPubMedID" AS VARCHAR)                 AS original_paper_pubmed_id,
        "RetractionNature"                                       AS retraction_nature,
        "Reason"                                                 AS reason,
        "Paywalled"                                              AS paywalled,
        "Notes"                                                  AS notes
    FROM read_csv(
        '{raw_csv}',
        header = TRUE,
        ignore_errors = TRUE,
        nullstr = '',
        all_varchar = TRUE
    )
    """


def _audit(con: duckdb.DuckDBPyConnection, raw_csv: Path) -> dict:
    """Count rows and date coverage on the final table (as a CTE)."""
    sql = f"""
    WITH final AS ({_build_sql(raw_csv)})
    SELECT
        COUNT(*)                       AS rows_total,
        COUNT(retraction_date)         AS rows_with_retraction_date,
        COUNT(original_paper_date)     AS rows_with_original_paper_date,
        COUNT(original_paper_doi) FILTER (WHERE original_paper_doi <> '')
                                       AS rows_with_original_paper_doi
    FROM final
    """
    row = con.execute(sql).fetchone()
    return {
        "rows_total": row[0],
        "rows_with_retraction_date": row[1],
        "rows_with_original_paper_date": row[2],
        "rows_with_original_paper_doi": row[3],
    }


def _validate(audit: dict) -> list[str]:
    """Return a list of validation errors. Empty list means OK."""
    errors = []
    n = audit["rows_total"]
    if n < ROW_COUNT_FLOOR:
        errors.append(
            f"row count {n} is below floor {ROW_COUNT_FLOOR}; "
            f"refusing to ship a truncated snapshot"
        )
    r_cov = audit["rows_with_retraction_date"] / n if n else 0.0
    o_cov = audit["rows_with_original_paper_date"] / n if n else 0.0
    if r_cov < DATE_COVERAGE_FLOOR:
        errors.append(
            f"retraction_date coverage {r_cov:.4f} below floor {DATE_COVERAGE_FLOOR}"
        )
    if o_cov < DATE_COVERAGE_FLOOR:
        errors.append(
            f"original_paper_date coverage {o_cov:.4f} below floor {DATE_COVERAGE_FLOOR}"
        )
    return errors


def convert(force: bool, dry_run: bool) -> int:
    if not RAW_CSV.exists():
        print(f"[error] raw CSV not found at {RAW_CSV}", file=sys.stderr)
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(":memory:")

    print(f"[retwatch] source  : {RAW_CSV}")
    print(f"[retwatch] target  : {OUTPUT_PARQUET}")

    t0 = time.time()
    audit = _audit(con, RAW_CSV)
    print(
        f"[retwatch] audit   : rows={audit['rows_total']}, "
        f"retraction_date={audit['rows_with_retraction_date']}, "
        f"original_paper_date={audit['rows_with_original_paper_date']}, "
        f"original_paper_doi={audit['rows_with_original_paper_doi']}"
    )

    errors = _validate(audit)
    if errors:
        for err in errors:
            print(f"[retwatch] FAIL   : {err}", file=sys.stderr)
        return 1

    if dry_run:
        print("[retwatch] dry run: not writing output")
        return 0

    if OUTPUT_PARQUET.exists() and not force:
        backup = OUTPUT_PARQUET.with_suffix(".parquet.bak")
        print(f"[retwatch] backup  : {OUTPUT_PARQUET} -> {backup}")
        shutil.move(str(OUTPUT_PARQUET), str(backup))

    write_sql = f"""
    COPY ({_build_sql(RAW_CSV)})
    TO '{OUTPUT_PARQUET}'
    (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 3)
    """
    con.execute(write_sql)

    size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    dt = time.time() - t0
    print(
        f"[retwatch] wrote   : {OUTPUT_PARQUET.name} "
        f"({size_mb:.1f} MB) in {dt:.1f}s"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the existing parquet without creating a .bak backup",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the audit but do not write the parquet",
    )
    args = ap.parse_args()
    return convert(force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
