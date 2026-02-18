#!/usr/bin/env python3
"""
Download OpenAlex snapshot data from S3.

Dynamically discovers all available entities from the S3 bucket - no hardcoded
list, so new entities added by OpenAlex are automatically picked up.

Uses `aws s3 sync` with --no-sign-request (free, no API key needed).
Downloads entity-by-entity for progress tracking. The snapshot uses
partition-based directories (updated_date=YYYY-MM-DD/) so subsequent runs
only download new/changed partitions.

Prerequisites:
    aws-cli installed: `sudo dnf install awscli` or `pip install awscli`

Usage:
    python scripts/download_openalex.py --all
    python scripts/download_openalex.py --entity works
    python scripts/download_openalex.py --entity awards
    python scripts/download_openalex.py --list          # Discover entities from S3
    python scripts/download_openalex.py --status        # Local download status
    python scripts/download_openalex.py --entity works --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ── Resolve data lake root ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

S3_BUCKET = "s3://openalex"
S3_DATA_PREFIX = f"{S3_BUCKET}/data/"
SNAPSHOT_DIR = ROOT / "datasets" / "openalex" / "snapshot"


# ── AWS CLI helpers ─────────────────────────────────────────────────────────

def check_aws_cli() -> bool:
    """Check if aws-cli is installed."""
    try:
        result = subprocess.run(
            ["aws", "--version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def discover_s3_entities() -> list[str]:
    """Discover all entity directories under s3://openalex/data/ dynamically."""
    cmd = [
        "aws", "s3", "ls", S3_DATA_PREFIX,
        "--no-sign-request",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"ERROR listing S3: {result.stderr.strip()}")
            return []

        entities = []
        for line in result.stdout.strip().splitlines():
            # Lines look like: "                           PRE authors/"
            line = line.strip()
            if line.startswith("PRE ") and line.endswith("/"):
                name = line[4:].rstrip("/")
                entities.append(name)

        return sorted(entities)

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"ERROR discovering entities: {e}")
        return []


def get_s3_entity_size(entity: str) -> tuple[int, int]:
    """Get file count and total size for an S3 entity.

    Returns (file_count, total_bytes). Uses `aws s3 ls --summarize`.
    """
    cmd = [
        "aws", "s3", "ls", f"{S3_DATA_PREFIX}{entity}/",
        "--no-sign-request", "--recursive", "--summarize",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return (0, 0)

        total_objects = 0
        total_size = 0
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("Total Objects:"):
                total_objects = int(line.split(":")[-1].strip())
            elif line.startswith("Total Size:"):
                # Parse human-readable or raw bytes
                size_str = line.split(":")[-1].strip()
                total_size = _parse_size(size_str)

        return (total_objects, total_size)

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return (0, 0)


def _parse_size(s: str) -> int:
    """Parse size string like '2.6 GiB' or '124.5 KiB' to bytes."""
    s = s.strip()
    parts = s.split()
    if len(parts) == 2:
        try:
            num = float(parts[0])
            unit = parts[1].upper()
            multipliers = {
                "B": 1, "BYTES": 1,
                "KIB": 1024, "KB": 1000,
                "MIB": 1024**2, "MB": 1000**2,
                "GIB": 1024**3, "GB": 1000**3,
                "TIB": 1024**4, "TB": 1000**4,
            }
            return int(num * multipliers.get(unit, 1))
        except (ValueError, KeyError):
            pass
    # Try plain integer
    try:
        return int(s)
    except ValueError:
        return 0


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


# ── Download logic ──────────────────────────────────────────────────────────

def download_entity(entity: str, dry_run: bool = False, delete: bool = False) -> bool:
    """Download a single entity from the OpenAlex S3 bucket."""
    s3_path = f"{S3_DATA_PREFIX}{entity}/"
    local_path = SNAPSHOT_DIR / "data" / entity
    local_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "aws", "s3", "sync",
        s3_path, str(local_path) + "/",
        "--no-sign-request",
    ]

    if dry_run:
        cmd.append("--dryrun")
    if delete:
        cmd.append("--delete")

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Syncing: {entity}")
    print(f"  S3: {s3_path}")
    print(f"  Local: {local_path}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(cmd, timeout=86400)  # 24h timeout
        elapsed = time.time() - start_time

        if result.returncode == 0:
            if local_path.exists():
                total_size = sum(
                    f.stat().st_size for f in local_path.rglob("*") if f.is_file()
                )
                print(f"\n  Completed in {elapsed:.0f}s ({total_size / (1024**3):.2f} GB)")
            return True
        else:
            print(f"\n  FAILED (exit code {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT after 24 hours")
        return False
    except KeyboardInterrupt:
        print(f"\n  Interrupted by user")
        return False


def download_release_notes():
    """Download the RELEASE_NOTES.txt file."""
    s3_path = f"{S3_BUCKET}/RELEASE_NOTES.txt"
    local_path = SNAPSHOT_DIR / "RELEASE_NOTES.txt"
    cmd = [
        "aws", "s3", "cp",
        s3_path, str(local_path),
        "--no-sign-request",
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
    except Exception:
        pass


# ── Status & listing ────────────────────────────────────────────────────────

def show_status():
    """Show local download status. Discovers local entities dynamically."""
    print("=== OpenAlex Download Status ===\n")

    data_dir = SNAPSHOT_DIR / "data"
    if not data_dir.exists():
        print("  No data downloaded yet.")
        print(f"  Run: python scripts/download_openalex.py --all")
        return

    total_size = 0
    total_files = 0
    entities = sorted(d.name for d in data_dir.iterdir() if d.is_dir())

    for entity in entities:
        entity_dir = data_dir / entity
        files = list(entity_dir.rglob("*.gz"))
        size = sum(f.stat().st_size for f in files) if files else 0
        total_size += size
        total_files += len(files)

        partitions = [
            d for d in entity_dir.iterdir()
            if d.is_dir() and d.name.startswith("updated_date=")
        ]

        print(
            f"  {entity:20s}  {len(files):6d} files  "
            f"{len(partitions):4d} partitions  "
            f"{size / (1024**3):8.2f} GB"
        )

    print(f"\n  {'TOTAL':20s}  {total_files:6d} files  "
          f"{'':4s}             "
          f"{total_size / (1024**3):8.2f} GB")

    rn_path = SNAPSHOT_DIR / "RELEASE_NOTES.txt"
    if rn_path.exists():
        print(f"\n  Release notes: {rn_path}")


def list_entities():
    """Discover and list all entities from S3 with their sizes."""
    print("=== OpenAlex S3 Entities (live from s3://openalex/data/) ===\n")
    print("Discovering entities...")

    entities = discover_s3_entities()
    if not entities:
        print("  Could not list S3 bucket. Check aws-cli and network.")
        return

    print(f"\n  {'Entity':20s}  {'Files':>8s}  {'Size':>10s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}")

    total_size = 0
    total_files = 0
    for entity in entities:
        n_files, size = get_s3_entity_size(entity)
        total_files += n_files
        total_size += size
        print(f"  {entity:20s}  {n_files:>8,}  {_format_size(size):>10s}")

    print(f"  {'-'*20}  {'-'*8}  {'-'*10}")
    print(f"  {'TOTAL':20s}  {total_files:>8,}  {_format_size(total_size):>10s}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download OpenAlex snapshot data from S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--entity", type=str, help="Download a specific entity")
    parser.add_argument("--all", action="store_true", help="Download all entities")
    parser.add_argument("--list", action="store_true",
                        help="Discover and list all S3 entities with sizes")
    parser.add_argument("--status", action="store_true", help="Show local download status")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded")
    parser.add_argument("--delete", action="store_true",
                        help="Delete local files not in S3 (for clean sync)")

    args = parser.parse_args()

    if args.status:
        show_status()
        return 0

    if args.list:
        if not check_aws_cli():
            print("ERROR: aws-cli required for --list. Install with: pip install awscli")
            return 1
        list_entities()
        return 0

    if not check_aws_cli():
        print("ERROR: aws-cli not found. Install with:")
        print("  sudo dnf install awscli")
        print("  # or")
        print("  pip install awscli")
        return 1

    if args.entity:
        # Accept any entity name - no validation against hardcoded list
        download_release_notes()
        success = download_entity(args.entity, dry_run=args.dry_run, delete=args.delete)
        return 0 if success else 1

    if args.all:
        print("=== Downloading All OpenAlex Entities ===")
        print(f"Destination: {SNAPSHOT_DIR}")
        print("Discovering entities from S3...")

        entities = discover_s3_entities()
        if not entities:
            print("ERROR: Could not discover entities from S3.")
            return 1

        # Sort: works and authors last (largest), everything else alphabetically
        large_entities = {"works", "authors"}
        small_first = [e for e in entities if e not in large_entities]
        large_last = [e for e in entities if e in large_entities]
        ordered = small_first + sorted(large_last)

        print(f"Found {len(entities)} entities: {', '.join(ordered)}")
        print()

        download_release_notes()

        results = {}
        for entity in ordered:
            success = download_entity(entity, dry_run=args.dry_run, delete=args.delete)
            results[entity] = success

            if not success and entity in large_entities:
                print(f"\nWARNING: {entity} download failed. Resume with:")
                print(f"  python scripts/download_openalex.py --entity {entity}")

        # Summary
        print("\n=== Download Summary ===")
        for entity, success in results.items():
            status = "OK" if success else "FAILED"
            print(f"  {entity:20s}  {status}")

        # Update meta.json
        if not args.dry_run and all(results.values()):
            meta_path = ROOT / "datasets" / "openalex" / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["format"] = "ndjson_gz"
                meta["last_updated"] = time.strftime("%Y-%m-%d")
                meta["entities_downloaded"] = list(results.keys())
                meta["notes"] = "Snapshot downloaded. Run convert_openalex.py to create Parquet files."
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"\nUpdated {meta_path}")

        failed = sum(1 for s in results.values() if not s)
        return 0 if failed == 0 else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
