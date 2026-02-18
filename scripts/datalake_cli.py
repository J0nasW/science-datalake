#!/usr/bin/env python3
"""
Master CLI for the Science Data Lake.

Usage:
    python scripts/datalake_cli.py status          # Disk usage, versions, row counts
    python scripts/datalake_cli.py info             # LLM-friendly dataset descriptions
    python scripts/datalake_cli.py download openalex
    python scripts/datalake_cli.py convert openalex [--workers 8]
    python scripts/datalake_cli.py update openalex  # Download + convert + views
    python scripts/datalake_cli.py update           # Update all datasets
    python scripts/datalake_cli.py views            # Regenerate DuckDB views
    python scripts/datalake_cli.py query "SELECT COUNT(*) FROM s2ag.papers"
    python scripts/datalake_cli.py shell            # Open DuckDB CLI
    python scripts/datalake_cli.py catalog          # Regenerate CATALOG.md from meta.json
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import duckdb

# Resolve data lake root
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

DB_PATH = ROOT / "datalake.duckdb"
DATASETS_DIR = ROOT / "datasets"


def cmd_status(args):
    """Show data lake status: disk usage, versions, row counts."""
    print("=" * 60)
    print("  Science Data Lake Status")
    print("=" * 60)
    print(f"\n  Root: {ROOT}")
    print(f"  DB:   {DB_PATH}")

    # Disk usage
    print(f"\n--- Disk Usage ---\n")

    # Auto-discover all datasets
    all_datasets = sorted(
        d.name for d in DATASETS_DIR.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    ) if DATASETS_DIR.exists() else []

    total_size = 0
    for ds_name in all_datasets:
        ds_dir = DATASETS_DIR / ds_name
        size = sum(
            f.stat().st_size for f in ds_dir.rglob("*") if f.is_file()
        )
        total_size += size
        if size > 1024**3:
            print(f"  {ds_name:25s}  {size / (1024**3):8.1f} GB")
        else:
            print(f"  {ds_name:25s}  {size / (1024**2):8.1f} MB")

    print(f"  {'─' * 40}")
    print(f"  {'TOTAL':25s}  {total_size / (1024**3):8.1f} GB")

    # Dataset versions from meta.json
    print(f"\n--- Dataset Versions ---\n")
    for ds_name in all_datasets:
        meta_path = DATASETS_DIR / ds_name / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            release = meta.get("release", "")
            updated = meta.get("last_updated", "")
            version_parts = []
            if release:
                version_parts.append(f"release={release}")
            if updated:
                version_parts.append(f"updated={updated}")
            if version_parts:
                print(f"  {ds_name:25s}  {' '.join(version_parts)}")

    # Row counts (if DB exists)
    if DB_PATH.exists():
        print(f"\n--- Key Row Counts ---\n")
        conn = duckdb.connect(str(DB_PATH), read_only=True)
        conn.execute("SET threads=8")

        queries = [
            ("s2ag.papers", "SELECT COUNT(*) FROM s2ag.papers"),
            ("s2ag.citations", "SELECT COUNT(*) FROM s2ag.citations"),
            ("s2ag.authors", "SELECT COUNT(*) FROM s2ag.authors"),
            ("sciscinet.papers", "SELECT COUNT(*) FROM sciscinet.papers"),
            ("sciscinet.paper_refs", "SELECT COUNT(*) FROM sciscinet.paper_refs"),
        ]

        # Check if openalex exists
        try:
            conn.execute("SELECT 1 FROM openalex.works LIMIT 0")
            queries.append(("openalex.works", "SELECT COUNT(*) FROM openalex.works"))
        except Exception:
            pass

        for name, sql in queries:
            try:
                count = conn.execute(sql).fetchone()[0]
                print(f"  {name:25s}  {count:>15,} rows")
            except Exception as e:
                print(f"  {name:25s}  ERROR: {e}")

        conn.close()

    print()


def cmd_info(args):
    """Show dataset descriptions in various formats."""
    fmt = getattr(args, "format", "summary") or "summary"
    ds_filter = getattr(args, "dataset", None)
    if ds_filter:
        datasets = [ds_filter]
    elif DATASETS_DIR.exists():
        datasets = sorted(
            d.name for d in DATASETS_DIR.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        )
    else:
        datasets = []

    if fmt == "json":
        _info_json(datasets)
    elif fmt == "schema":
        _info_schema(datasets)
    else:
        _info_summary(datasets)


def _info_summary(datasets):
    """Default human-readable summary format."""
    for ds_name in datasets:
        meta_path = DATASETS_DIR / ds_name / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        print(f"\n{'=' * 60}")
        print(f"  {meta.get('full_name', ds_name)}")
        print(f"{'=' * 60}")
        print(f"  {meta.get('description', 'No description')}")
        print(f"\n  License: {meta.get('license', 'unknown')}")
        print(f"  Source:  {meta.get('source_url', 'unknown')}")
        print(f"  Format:  {meta.get('format', 'unknown')}")
        print(f"  Size:    {meta.get('total_size_gb', '?')} GB")

        tables = meta.get("tables", {})
        if tables:
            print(f"\n  Tables ({len(tables)}):")
            for table_name, table_info in tables.items():
                rows = table_info.get("row_count", "?")
                if isinstance(rows, int):
                    rows = f"{rows:,}"
                elif rows is None:
                    rows = "derived"
                else:
                    rows = str(rows)
                desc = table_info.get("description", "")[:60]
                print(f"    {table_name:30s}  {rows:>15s}  {desc}")

    print()


def _info_json(datasets):
    """Dump full meta.json content for specified datasets."""
    result = {}
    for ds_name in datasets:
        meta_path = DATASETS_DIR / ds_name / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                result[ds_name] = json.load(f)
    print(json.dumps(result, indent=2))


def _info_schema(datasets):
    """Compact column-level schema for LLM system prompts."""
    for ds_name in datasets:
        meta_path = DATASETS_DIR / ds_name / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        print(f"# {ds_name} ({meta.get('full_name', ds_name)})")
        tables = meta.get("tables", {})
        for table_name, table_info in tables.items():
            rows = table_info.get("row_count", "?")
            if isinstance(rows, int):
                rows = f"{rows:,}"
            tier = table_info.get("performance_tier", "")
            print(f"\n## {ds_name}.{table_name} ({rows} rows, {tier})")
            desc = table_info.get("description", "")
            if desc:
                print(f"  {desc}")

            columns = table_info.get("columns", {})
            if isinstance(columns, dict):
                for col_name, col_info in columns.items():
                    if isinstance(col_info, dict):
                        ctype = col_info.get("type", "?")
                        cdesc = col_info.get("description", "")
                        pk = " PK" if col_info.get("is_primary_key") else ""
                        null = " NULL" if col_info.get("nullable") else ""
                        print(f"  - {col_name}: {ctype}{pk}{null} -- {cdesc}")
                    else:
                        print(f"  - {col_name}: {col_info}")
            elif isinstance(columns, str):
                print(f"  Columns: {columns}")
        print()


def _discover_scripts(prefix: str) -> dict[str, Path]:
    """Discover available dataset scripts by prefix (e.g., 'download', 'convert')."""
    scripts = {}
    for s in SCRIPT_DIR.glob(f"{prefix}_*.py"):
        name = s.stem.replace(f"{prefix}_", "")
        scripts[name] = s
    return scripts


def cmd_download(args):
    """Download a dataset."""
    extra = args.extra_args or []
    script = SCRIPT_DIR / f"download_{args.dataset}.py"
    if not script.exists():
        available = [s.stem.replace("download_", "") for s in SCRIPT_DIR.glob("download_*.py")]
        print(f"No download script for: {args.dataset}")
        print(f"Available: {available}")
        return 1
    return subprocess.run([sys.executable, str(script), "--all", *extra]).returncode


def cmd_convert(args):
    """Convert a dataset to Parquet."""
    extra = args.extra_args or []
    script = SCRIPT_DIR / f"convert_{args.dataset}.py"
    if not script.exists():
        available = [s.stem.replace("convert_", "") for s in SCRIPT_DIR.glob("convert_*.py")]
        print(f"No convert script for: {args.dataset}")
        print(f"Available: {available}")
        return 1
    return subprocess.run([sys.executable, str(script), "--all", *extra]).returncode


def cmd_update(args):
    """Run full update pipeline: download -> convert -> views."""
    datasets = [args.dataset] if args.dataset else ["openalex", "s2ag", "sciscinet"]
    extra = args.extra_args or []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"  Updating: {dataset}")
        print(f"{'='*60}")

        if dataset == "ontologies":
            # Special handling: ontologies have their own download/convert scripts
            print(f"\n--- [ontologies] Download ---")
            rc = subprocess.run([
                sys.executable, str(SCRIPT_DIR / "download_ontologies.py"), "--all", *extra
            ]).returncode
            if rc != 0:
                print(f"WARNING: ontologies download returned {rc}")

            print(f"\n--- [ontologies] Convert ---")
            rc = subprocess.run([
                sys.executable, str(SCRIPT_DIR / "convert_ontologies.py"), "--all", *extra
            ]).returncode
            if rc != 0:
                print(f"WARNING: ontologies conversion returned {rc}")
            continue

        # Step 1: Download
        dl_script = SCRIPT_DIR / f"download_{dataset}.py"
        if dl_script.exists():
            print(f"\n--- [{dataset}] Download ---")
            # SciSciNet: default to --core (large files are usually pre-processed)
            dl_flag = "--core" if dataset == "sciscinet" else "--all"
            rc = subprocess.run([sys.executable, str(dl_script), dl_flag, *extra]).returncode
            if rc != 0:
                print(f"WARNING: {dataset} download returned {rc}")

        # Step 2: Convert (if applicable)
        conv_script = SCRIPT_DIR / f"convert_{dataset}.py"
        if conv_script.exists():
            print(f"\n--- [{dataset}] Convert ---")
            rc = subprocess.run([sys.executable, str(conv_script), "--all", *extra]).returncode
            if rc != 0:
                print(f"WARNING: {dataset} conversion returned {rc}")

    # Step 3: Regenerate views (once, after all datasets)
    print(f"\n--- Regenerating DuckDB views ---")
    subprocess.run([sys.executable, str(SCRIPT_DIR / "create_unified_db.py")])
    print("\nUpdate complete.")


def cmd_views(args):
    """Regenerate DuckDB views."""
    cmd = [sys.executable, str(SCRIPT_DIR / "create_unified_db.py")]
    if args.materialize_xref:
        cmd.append("--materialize-xref")
    return subprocess.run(cmd).returncode


def cmd_query(args):
    """Execute a DuckDB query."""
    sql = args.sql

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: python scripts/datalake_cli.py views")
        return 1

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute("SET threads=16")

    try:
        start = time.time()
        result = conn.execute(sql)

        if sql.strip().upper().startswith("SELECT") or sql.strip().upper().startswith("WITH"):
            df = result.df()
            elapsed = time.time() - start
            print(df.to_string())
            print(f"\n({len(df)} rows, {elapsed:.2f}s)")
        else:
            elapsed = time.time() - start
            print(f"OK ({elapsed:.2f}s)")

    except Exception as e:
        print(f"Query error: {e}")
        return 1
    finally:
        conn.close()

    return 0


def cmd_shell(args):
    """Open DuckDB CLI shell."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: python scripts/datalake_cli.py views")
        return 1

    # Try duckdb CLI first, fall back to Python
    try:
        return subprocess.run(["duckdb", str(DB_PATH), "-readonly"]).returncode
    except FileNotFoundError:
        print("DuckDB CLI not found. Starting Python-based shell...")
        print(f"Connected to {DB_PATH}")
        print("Type SQL queries, or 'quit' to exit.\n")

        conn = duckdb.connect(str(DB_PATH), read_only=True)
        conn.execute("SET threads=16")

        while True:
            try:
                sql = input("datalake> ").strip()
                if not sql:
                    continue
                if sql.lower() in ("quit", "exit", ".quit", ".exit"):
                    break
                if sql == ".tables":
                    sql = "SELECT table_schema || '.' || table_name AS view FROM information_schema.tables ORDER BY 1"

                start = time.time()
                result = conn.execute(sql)

                if sql.upper().startswith("SELECT") or sql.upper().startswith("WITH") or sql.startswith("."):
                    df = result.df()
                    elapsed = time.time() - start
                    print(df.to_string())
                    print(f"\n({len(df)} rows, {elapsed:.2f}s)\n")
                else:
                    elapsed = time.time() - start
                    print(f"OK ({elapsed:.2f}s)\n")

            except KeyboardInterrupt:
                print()
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}\n")

        conn.close()
        return 0


def cmd_catalog(args):
    """Regenerate CATALOG.md from meta.json files."""
    print("CATALOG.md regeneration is manual for now.")
    print(f"Edit: {ROOT / 'CATALOG.md'}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Science Data Lake CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # status
    sub = subparsers.add_parser("status", help="Show data lake status")
    sub.set_defaults(func=cmd_status)

    # info
    sub = subparsers.add_parser("info", help="LLM-friendly dataset descriptions")
    sub.add_argument("--format", choices=["summary", "json", "schema"], default="summary",
                     help="Output format: summary (default), json (full meta.json), schema (compact columns)")
    sub.add_argument("--dataset", type=str, default=None,
                     help="Filter to specific dataset: s2ag, sciscinet, openalex")
    sub.set_defaults(func=cmd_info)

    # download
    sub = subparsers.add_parser("download", help="Download a dataset")
    sub.add_argument("dataset", type=str, help="Dataset name: openalex, s2ag, sciscinet")
    sub.add_argument("extra_args", nargs="*", help="Extra arguments for the download script")
    sub.set_defaults(func=cmd_download)

    # convert
    sub = subparsers.add_parser("convert", help="Convert dataset to Parquet")
    sub.add_argument("dataset", type=str, help="Dataset name: openalex, s2ag")
    sub.add_argument("extra_args", nargs="*", help="Extra arguments for the converter")
    sub.set_defaults(func=cmd_convert)

    # update
    sub = subparsers.add_parser("update", help="Full pipeline: download -> convert -> views")
    sub.add_argument("dataset", type=str, nargs="?", default=None,
                     help="Dataset name (default: all datasets)")
    sub.add_argument("extra_args", nargs="*", help="Extra arguments passed to scripts")
    sub.set_defaults(func=cmd_update)

    # views
    sub = subparsers.add_parser("views", help="Regenerate DuckDB views")
    sub.add_argument("--materialize-xref", action="store_true",
                     help="Materialize xref.doi_index (slow)")
    sub.set_defaults(func=cmd_views)

    # query
    sub = subparsers.add_parser("query", help="Execute a DuckDB query")
    sub.add_argument("sql", type=str, help="SQL query to execute")
    sub.set_defaults(func=cmd_query)

    # shell
    sub = subparsers.add_parser("shell", help="Open DuckDB shell")
    sub.set_defaults(func=cmd_shell)

    # catalog
    sub = subparsers.add_parser("catalog", help="Regenerate CATALOG.md")
    sub.set_defaults(func=cmd_catalog)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
