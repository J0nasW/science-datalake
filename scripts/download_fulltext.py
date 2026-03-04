#!/usr/bin/env python3
"""
Download raw full-text paper data from external sources.

Supports peS2o (HuggingFace), PMC (FTP), and stubs for future sources
(arXiv, bioRxiv, CORE). All downloads are free — no paid APIs.

Usage:
    python scripts/download_fulltext.py --source pes2o
    python scripts/download_fulltext.py --source pmc
    python scripts/download_fulltext.py --all
    python scripts/download_fulltext.py --status
"""

import argparse
import ftplib
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Resolve data lake root ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import find_datalake_root

ROOT = find_datalake_root()
RAW_DIR = ROOT / "datasets" / "fulltext" / "raw"

SOURCES = ["pes2o", "pmc", "arxiv", "biorxiv", "core"]


# ── peS2o (HuggingFace) ────────────────────────────────────────────────────

def download_pes2o(dry_run: bool = False):
    """Download peS2o V2 dataset from HuggingFace.

    Uses huggingface_hub to download allenai/peS2o.
    The dataset contains ~39M papers in JSONL format.
    """
    output_dir = RAW_DIR / "pes2o"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ERROR: huggingface_hub required. Install with: pip install huggingface_hub")
        return False

    repo_id = "allenai/peS2o"

    if dry_run:
        print(f"  [DRY RUN] Would download {repo_id} to {output_dir}")
        return True

    print(f"  Downloading {repo_id} to {output_dir}")
    print("  This is ~50 GB and may take a while...")

    try:
        t0 = time.time()
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_dir),
            # Only download v2 data files (json.gz, ~81 GB)
            allow_patterns=["data/v2/*.json.gz"],
        )
        elapsed = time.time() - t0

        # Count downloaded files
        jsonl_files = list(output_dir.rglob("*.json.gz"))
        total_size = sum(f.stat().st_size for f in jsonl_files)
        print(f"  Done: {len(jsonl_files)} files, "
              f"{total_size / (1024**3):.2f} GB, {elapsed:.0f}s")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ── PMC Open Access (FTP) ──────────────────────────────────────────────────

def download_pmc(dry_run: bool = False):
    """Download PMC Open Access bulk archives from NCBI FTP.

    Downloads .tar.gz archives from ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/
    Each archive contains JATS XML files for a batch of articles.
    """
    output_dir = RAW_DIR / "pmc"
    output_dir.mkdir(parents=True, exist_ok=True)

    ftp_host = "ftp.ncbi.nlm.nih.gov"
    ftp_dir = "/pub/pmc/oa_bulk/oa_comm/xml/"  # Commercial-use OA subset

    print(f"  Connecting to {ftp_host}...")

    try:
        ftp = ftplib.FTP(ftp_host)
        ftp.login()  # anonymous
        ftp.cwd(ftp_dir)

        # List available .tar.gz files
        files = []
        ftp.retrlines("NLST", files.append)
        tar_files = [f for f in files if f.endswith(".tar.gz")]

        if not tar_files:
            # Try alternative directory structure
            ftp.cwd("/pub/pmc/oa_bulk/")
            files = []
            ftp.retrlines("NLST", files.append)
            tar_files = [f for f in files if f.endswith(".tar.gz")]

        print(f"  Found {len(tar_files)} archives on FTP")

        if dry_run:
            for f in tar_files[:10]:
                print(f"    {f}")
            if len(tar_files) > 10:
                print(f"    ... and {len(tar_files) - 10} more")
            ftp.quit()
            return True

        downloaded = 0
        skipped = 0
        failed = 0

        for filename in tar_files:
            local_path = output_dir / filename
            if local_path.exists():
                skipped += 1
                continue

            print(f"    Downloading {filename}...", end=" ", flush=True)
            t0 = time.time()

            try:
                with open(local_path, "wb") as f:
                    ftp.retrbinary(f"RETR {filename}", f.write)
                elapsed = time.time() - t0
                size = local_path.stat().st_size / (1024 * 1024)
                print(f"{size:.1f} MB, {elapsed:.0f}s")
                downloaded += 1
            except Exception as e:
                print(f"FAILED: {e}")
                if local_path.exists():
                    local_path.unlink()
                failed += 1

        ftp.quit()

        print(f"\n  Summary: {downloaded} downloaded, {skipped} skipped, {failed} failed")
        return failed == 0

    except Exception as e:
        print(f"  ERROR connecting to FTP: {e}")
        return False


# ── Stubs for future sources ────────────────────────────────────────────────

def download_arxiv(dry_run: bool = False):
    """Download arXiv source files. (Phase 4 - deferred)"""
    print("  arXiv download not yet implemented (Phase 4)")
    print("  Will use: gsutil -m cp -r gs://arxiv-dataset/arxiv/ ./")
    print("  Or: archive.org mirror for LaTeX source tarballs")
    return False


def download_biorxiv(dry_run: bool = False):
    """Download bioRxiv/medRxiv papers. (Phase 5 - deferred)"""
    print("  bioRxiv download not yet implemented (Phase 5)")
    print("  Will use: aws s3 sync s3://biorxiv-src-monthly/ --requester-pays")
    return False


def download_core(dry_run: bool = False):
    """Download CORE bulk dataset. (Phase 6 - deferred)"""
    print("  CORE download not yet implemented (Phase 6)")
    print("  Will use: CORE bulk download API")
    return False


# ── Status ──────────────────────────────────────────────────────────────────

def show_status():
    """Show local download status for all sources."""
    print("=== Full-Text Download Status ===\n")

    if not RAW_DIR.exists():
        print("  No data downloaded yet.")
        print(f"  Run: python scripts/download_fulltext.py --all")
        return

    total_size = 0
    total_files = 0

    for source in SOURCES:
        source_dir = RAW_DIR / source
        if not source_dir.exists():
            print(f"  {source:10s}: not downloaded")
            continue

        all_files = list(source_dir.rglob("*"))
        files = [f for f in all_files if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        total_size += size
        total_files += len(files)

        print(f"  {source:10s}: {len(files):6d} files, {size / (1024**3):8.2f} GB")

    print(f"\n  {'TOTAL':10s}: {total_files:6d} files, {total_size / (1024**3):8.2f} GB")


# ── CLI ─────────────────────────────────────────────────────────────────────

DOWNLOADERS = {
    "pes2o": download_pes2o,
    "pmc": download_pmc,
    "arxiv": download_arxiv,
    "biorxiv": download_biorxiv,
    "core": download_core,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download raw full-text paper data from external sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--source", type=str, choices=SOURCES,
                        help="Download a specific source")
    parser.add_argument("--all", action="store_true",
                        help="Download all available sources (Phases 2-3)")
    parser.add_argument("--status", action="store_true",
                        help="Show local download status")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded")

    args = parser.parse_args()

    print("=== Full-Text Papers Downloader ===")
    print(f"Download dir: {RAW_DIR}")

    if args.status:
        show_status()
        return 0

    if args.source:
        downloader = DOWNLOADERS.get(args.source)
        if not downloader:
            print(f"  Unknown source: {args.source}")
            return 1
        print(f"\n[{args.source}]")
        ok = downloader(dry_run=args.dry_run)
        return 0 if ok else 1

    if args.all:
        # Only run Phase 2-3 sources (pes2o and pmc)
        active_sources = ["pes2o", "pmc"]
        results = {}
        for source in active_sources:
            print(f"\n[{source}]")
            results[source] = DOWNLOADERS[source](dry_run=args.dry_run)

        print("\n=== Download Summary ===")
        for source, ok in results.items():
            print(f"  {source}: {'OK' if ok else 'FAILED/SKIPPED'}")

        show_status()
        return 0 if all(results.values()) else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
