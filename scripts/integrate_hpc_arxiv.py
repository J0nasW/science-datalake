#!/usr/bin/env python3
"""
Integrate HPC arXiv extraction results into the fulltext datalake.

Uses PyArrow for fast JSONL→parquet conversion, then DuckDB for dedup merge.
"""

import json
import sys
import time
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/mnt/nvme03/science_datalake")
ARXIV_DIR = ROOT / "datasets" / "fulltext" / "parquet" / "arxiv"
RAW_JSONL = ROOT / "datasets" / "fulltext" / "raw" / "arxiv" / "arxiv_fulltext.jsonl"

SCHEMA = pa.schema([
    ("doi", pa.string()),
    ("source", pa.string()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("text", pa.string()),
    ("license", pa.string()),
    ("year", pa.int32()),
    ("source_id", pa.string()),
    ("text_length", pa.int32()),
    ("language", pa.string()),
    ("has_full_text", pa.bool_()),
])

CHUNK_SIZE = 50_000  # rows per chunk


def jsonl_to_parquet(jsonl_path: Path, out_path: Path):
    """Stream JSONL → parquet using PyArrow, chunk by chunk."""
    print(f"  Reading: {jsonl_path}")
    print(f"  Writing: {out_path}")
    print(f"  Chunk size: {CHUNK_SIZE:,} rows")

    writer = None
    total = 0
    t0 = time.time()
    chunk_rows = []

    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            chunk_rows.append(row)

            if len(chunk_rows) >= CHUNK_SIZE:
                batch = rows_to_batch(chunk_rows)
                if writer is None:
                    writer = pq.ParquetWriter(
                        out_path, SCHEMA,
                        compression="zstd",
                    )
                writer.write_table(batch)
                total += len(chunk_rows)
                elapsed = time.time() - t0
                rate = total / elapsed
                print(f"    [{total:,}] {rate:.0f} rows/s, "
                      f"{elapsed:.0f}s elapsed", flush=True)
                chunk_rows = []

    # Write remaining
    if chunk_rows:
        batch = rows_to_batch(chunk_rows)
        if writer is None:
            writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
        writer.write_table(batch)
        total += len(chunk_rows)

    if writer:
        writer.close()

    elapsed = time.time() - t0
    size_gb = out_path.stat().st_size / (1024**3)
    print(f"  Done: {total:,} rows, {size_gb:.2f} GB, {elapsed:.0f}s")
    return total


def rows_to_batch(rows: list[dict]) -> pa.Table:
    """Convert list of dicts to Arrow table matching SCHEMA."""
    arrays = {}
    for field in SCHEMA:
        name = field.name
        values = [r.get(name) for r in rows]
        arrays[name] = values

    return pa.table(arrays, schema=SCHEMA)


def main():
    print("=" * 60)
    print("Integrating HPC arXiv extraction into datalake")
    print("=" * 60)
    sys.stdout.flush()

    assert RAW_JSONL.exists(), f"JSONL not found: {RAW_JSONL}"
    print(f"\nSource JSONL: {RAW_JSONL}")
    print(f"  Size: {RAW_JSONL.stat().st_size / (1024**3):.1f} GB")

    existing_parquets = sorted(ARXIV_DIR.glob("*.parquet"))
    # Skip leftover tmp files
    real_parquets = [p for p in existing_parquets
                     if "tmp" not in p.name and "merged" not in p.name]
    print(f"\nExisting parquet files:")
    for p in real_parquets:
        print(f"  {p.name}: {p.stat().st_size / (1024**3):.2f} GB")
    sys.stdout.flush()

    # --- Step 1: Convert JSONL → parquet ---
    print(f"\n--- Step 1: JSONL → Parquet (PyArrow) ---")
    sys.stdout.flush()

    tmp_parquet = ARXIV_DIR / "hpc_missing_tmp.parquet"
    if tmp_parquet.exists() and tmp_parquet.stat().st_size > 1_000_000_000:
        # Reuse previous conversion
        import duckdb as _db
        _c = _db.connect()
        hpc_count = _c.execute(
            f"SELECT COUNT(*) FROM read_parquet('{tmp_parquet}')"
        ).fetchone()[0]
        _c.close()
        print(f"  Reusing existing: {tmp_parquet.name} ({hpc_count:,} rows, "
              f"{tmp_parquet.stat().st_size / (1024**3):.2f} GB)")
    else:
        hpc_count = jsonl_to_parquet(RAW_JSONL, tmp_parquet)
    sys.stdout.flush()

    # --- Step 2: Merge and deduplicate with DuckDB ---
    print(f"\n--- Step 2: Merge + Dedup (DuckDB) ---")
    sys.stdout.flush()

    all_parquets = real_parquets + [tmp_parquet]
    parquet_list = ", ".join(f"'{p}'" for p in all_parquets)

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    con.execute("SET threads = 8")
    con.execute("SET memory_limit = '100GB'")
    # Spill to disk when memory is insufficient
    con.execute(f"SET temp_directory = '{ARXIV_DIR / 'tmp'}'")
    (ARXIV_DIR / "tmp").mkdir(exist_ok=True)

    # Count before
    existing_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet([{', '.join(repr(str(p)) for p in real_parquets)}])"
    ).fetchone()[0] if real_parquets else 0
    print(f"  Existing: {existing_count:,}")
    print(f"  New HPC: {hpc_count:,}")
    sys.stdout.flush()

    t2 = time.time()
    merged_path = ARXIV_DIR / "merged_arxiv.parquet"

    # Simple DISTINCT ON dedup (DuckDB will spill to disk if needed)
    print("  Dedup: DISTINCT ON source_id ORDER BY text_length DESC...")
    sys.stdout.flush()
    con.execute(f"""
        COPY (
            SELECT DISTINCT ON (source_id)
                doi, source, title, abstract, text, license, year,
                source_id, text_length, language, has_full_text
            FROM read_parquet([{parquet_list}])
            ORDER BY source_id, text_length DESC NULLS LAST
        ) TO '{merged_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 10000)
    """)

    elapsed2 = time.time() - t2
    merged_size = merged_path.stat().st_size / (1024**3)
    merged_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{merged_path}')"
    ).fetchone()[0]
    ft_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{merged_path}') WHERE has_full_text = true"
    ).fetchone()[0]
    print(f"  Merged: {merged_count:,} unique papers ({ft_count:,} full-text)")
    print(f"  Size: {merged_size:.2f} GB, {elapsed2:.0f}s")
    sys.stdout.flush()

    # Year distribution
    print("\n  Year distribution (top 15):")
    years = con.execute(f"""
        SELECT year, COUNT(*) AS n
        FROM read_parquet('{merged_path}')
        WHERE year IS NOT NULL
        GROUP BY year ORDER BY year DESC LIMIT 15
    """).fetchall()
    for year, n in years:
        print(f"    {year}: {n:,}")
    sys.stdout.flush()

    # --- Step 3: Swap files ---
    print(f"\n--- Step 3: Replacing old files ---")

    for p in real_parquets:
        print(f"  Removing: {p.name}")
        p.unlink()

    # Remove leftover files from previous failed attempts
    for leftover in ARXIV_DIR.glob("*tmp*"):
        if leftover != tmp_parquet:
            print(f"  Removing leftover: {leftover.name}")
            leftover.unlink()
    for leftover in ARXIV_DIR.glob("merged_all*"):
        print(f"  Removing leftover: {leftover.name}")
        leftover.unlink()

    tmp_parquet.unlink()
    print(f"  Removed: {tmp_parquet.name}")

    final_path = ARXIV_DIR / "data_0.parquet"
    merged_path.rename(final_path)
    print(f"  Final: {final_path.name}")

    # --- Step 4: Verify ---
    print(f"\n--- Step 4: Verification ---")
    verify_count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{final_path}')"
    ).fetchone()[0]
    verify_ids = con.execute(
        f"SELECT COUNT(DISTINCT source_id) FROM read_parquet('{final_path}')"
    ).fetchone()[0]
    print(f"  Total rows: {verify_count:,}")
    print(f"  Unique source_ids: {verify_ids:,}")
    assert verify_count == verify_ids, "Duplicate source_ids!"
    print("  PASSED")

    total_elapsed = time.time() - time.time()  # recompute from t0... use a global
    con.close()

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Before: {existing_count:,} papers")
    print(f"  After:  {merged_count:,} papers (+{merged_count - existing_count:,})")
    print(f"  Full-text: {ft_count:,}")
    print(f"  File: {final_path}")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
