#!/usr/bin/env python3
"""
Convert S2AG JSON.gz datasets to Parquet format.

Self-contained script with data lake path integration.
DuckDB view creation is handled separately by create_unified_db.py.

Usage:
    python scripts/convert_s2ag.py --all
    python scripts/convert_s2ag.py --dataset papers
    python scripts/convert_s2ag.py --summary
    python scripts/convert_s2ag.py --all --workers 8
    python scripts/convert_s2ag.py --all --force
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb

# ── Resolve data lake root ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import find_datalake_root

ROOT = find_datalake_root()
SOURCE_DIR = ROOT / "datasets" / "s2ag" / "raw" / "latest"
OUTPUT_DIR = ROOT / "datasets" / "s2ag" / "parquet"

# ── Dataset configurations ──────────────────────────────────────────────────

DATASETS = {
    "publication-venues": {
        "row_group_size": 10_000,
        "description": "Journal and conference metadata",
    },
    "tldrs": {
        "row_group_size": 100_000,
        "description": "Short paper summaries",
    },
    "authors": {
        "row_group_size": 100_000,
        "description": "Author profiles and metrics",
    },
    "paper-ids": {
        "row_group_size": 500_000,
        "description": "SHA to corpus ID mappings",
    },
    "abstracts": {
        "row_group_size": 100_000,
        "description": "Paper abstracts with open access info",
    },
    "papers": {
        "row_group_size": 50_000,
        "description": "Paper metadata (complex nested schema)",
    },
    "citations": {
        "row_group_size": 500_000,
        "description": "Citation relationships with contexts",
    },
    "s2orc_v2": {
        "row_group_size": 10_000,
        "description": "Full paper text content",
    },
}


# ── Conversion functions ────────────────────────────────────────────────────

def convert_file(args: tuple) -> dict:
    """Convert a single JSON.gz file to Parquet using DuckDB.

    This function runs in a separate process.
    """
    input_path, output_path, row_group_size = args

    conn = duckdb.connect(":memory:")
    conn.execute("SET threads=2")
    conn.execute("SET memory_limit='8GB'")

    try:
        start_time = time.time()

        query = f"""
            COPY (
                SELECT * FROM read_json_auto(
                    '{input_path}',
                    format='newline_delimited',
                    compression='gzip',
                    maximum_object_size=104857600,
                    ignore_errors=true
                )
            ) TO '{output_path}' (
                FORMAT PARQUET,
                COMPRESSION 'zstd',
                COMPRESSION_LEVEL 3,
                ROW_GROUP_SIZE {row_group_size}
            )
        """

        conn.execute(query)

        row_count = conn.execute(f"""
            SELECT COUNT(*) FROM read_parquet('{output_path}')
        """).fetchone()[0]

        elapsed = time.time() - start_time
        output_size = output_path.stat().st_size / (1024 * 1024)  # MB

        return {
            "input": str(input_path),
            "output": str(output_path),
            "rows": row_count,
            "size_mb": output_size,
            "elapsed": elapsed,
            "success": True,
        }

    except Exception as e:
        return {
            "input": str(input_path),
            "output": str(output_path),
            "error": str(e),
            "success": False,
        }
    finally:
        conn.close()


def convert_dataset_with_paths(
    dataset_name: str,
    source_dir: Path,
    output_dir: Path,
    workers: int = 4,
    skip_existing: bool = True,
) -> list:
    """Convert all files for a dataset with explicit paths."""
    config = DATASETS.get(dataset_name)
    if not config:
        print(f"Unknown dataset: {dataset_name}")
        return []

    input_dir = source_dir / dataset_name
    dataset_output_dir = output_dir / dataset_name

    if not input_dir.exists():
        print(f"Source directory not found: {input_dir}")
        return []

    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.gz"))
    if not input_files:
        print(f"No .gz files found in {input_dir}")
        return []

    tasks = []
    skipped = 0
    for input_file in input_files:
        output_file = dataset_output_dir / (input_file.stem + ".parquet")

        if skip_existing and output_file.exists():
            skipped += 1
            continue

        tasks.append((input_file, output_file, config["row_group_size"]))

    if skipped > 0:
        print(f"  Skipping {skipped} already converted files")

    if not tasks:
        print(f"  All files already converted")
        return []

    print(f"  Converting {len(tasks)} files with {workers} workers...")

    results = []
    completed = 0
    total_rows = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_file, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            input_file = Path(task[0])

            try:
                result = future.result()
                results.append(result)
                completed += 1

                if result["success"]:
                    total_rows += result["rows"]
                    print(
                        f"    [{completed}/{len(tasks)}] {input_file.name}: "
                        f"{result['rows']:,} rows, {result['size_mb']:.1f}MB, "
                        f"{result['elapsed']:.1f}s"
                    )
                else:
                    print(
                        f"    [{completed}/{len(tasks)}] {input_file.name}: "
                        f"ERROR - {result['error']}"
                    )

            except Exception as e:
                print(f"    [{completed}/{len(tasks)}] {input_file.name}: EXCEPTION - {e}")
                results.append({
                    "input": str(input_file),
                    "error": str(e),
                    "success": False,
                })

    success_count = sum(1 for r in results if r.get("success"))
    print(f"  Completed: {success_count}/{len(tasks)} files, {total_rows:,} total rows")

    return results


def convert_all_datasets_with_paths(
    source_dir: Path,
    output_dir: Path,
    workers: int = 4,
    skip_existing: bool = True,
) -> dict:
    """Convert all datasets with explicit paths."""
    all_results = {}

    dataset_order = [
        "publication-venues",
        "tldrs",
        "authors",
        "paper-ids",
        "abstracts",
        "papers",
        "citations",
        "s2orc_v2",
    ]

    for dataset_name in dataset_order:
        if (source_dir / dataset_name).exists():
            print(f"\n[{dataset_name}] {DATASETS[dataset_name]['description']}")
            results = convert_dataset_with_paths(
                dataset_name, source_dir, output_dir, workers, skip_existing
            )
            all_results[dataset_name] = results
        else:
            print(f"\n[{dataset_name}] SKIPPED - directory not found")

    return all_results


def print_summary(parquet_dir: Path = OUTPUT_DIR):
    """Print summary of existing parquet files."""
    print("\n=== Parquet Files Summary ===\n")

    total_files = 0
    total_size = 0

    for dataset_name in DATASETS:
        dataset_dir = parquet_dir / dataset_name
        if not dataset_dir.exists():
            print(f"{dataset_name}: not converted")
            continue

        files = list(dataset_dir.glob("*.parquet"))
        if not files:
            print(f"{dataset_name}: no parquet files")
            continue

        size = sum(f.stat().st_size for f in files)
        total_files += len(files)
        total_size += size

        print(f"{dataset_name}: {len(files)} files, {size / (1024**3):.2f} GB")

    print(f"\nTotal: {total_files} files, {total_size / (1024**3):.2f} GB")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert S2AG JSON.gz datasets to Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()),
                        help="Convert specific dataset")
    parser.add_argument("--all", action="store_true",
                        help="Convert all datasets")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing parquet files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert files even if parquet already exists")
    parser.add_argument("--source-dir", type=str, default=str(SOURCE_DIR),
                        help=f"Source directory (default: {SOURCE_DIR})")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    skip_existing = not args.force

    print("=== S2AG to Parquet Converter ===")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")

    if args.summary:
        print_summary(output_dir)
        return 0

    if args.dataset:
        print(f"\nConverting dataset: {args.dataset}")
        results = convert_dataset_with_paths(
            args.dataset, source_dir, output_dir, args.workers, skip_existing
        )

        if results:
            success = sum(1 for r in results if r.get("success"))
            failed = len(results) - success
            if failed > 0:
                print(f"\nWARNING: {failed} files failed to convert")

        return 0 if not results or all(r.get("success") for r in results) else 1

    if args.all:
        print(f"\nConverting all datasets with {args.workers} workers")
        results = convert_all_datasets_with_paths(
            source_dir, output_dir, args.workers, skip_existing
        )

        print("\n=== Conversion Summary ===")
        total_success = 0
        total_failed = 0
        for dataset, dataset_results in results.items():
            success = sum(1 for r in dataset_results if r.get("success"))
            failed = len(dataset_results) - success
            total_success += success
            total_failed += failed
            status = "OK" if failed == 0 else f"{failed} FAILED"
            print(f"  {dataset}: {success} files converted, {status}")

        print(f"\nTotal: {total_success} success, {total_failed} failed")

        return 0 if total_failed == 0 else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
