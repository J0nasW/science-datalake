#!/usr/bin/env python3
"""
Download SciSciNet v2 data from Google Cloud Storage.

Self-contained script with data lake path integration.

Note: This script disables gsutil integrity checks (check_hashes=never) to avoid
crcmod performance issues. While this speeds up downloads, it skips hash verification.

Usage:
    python scripts/download_sciscinet.py --core
    python scripts/download_sciscinet.py --large
    python scripts/download_sciscinet.py --all
    python scripts/download_sciscinet.py --status
    python scripts/download_sciscinet.py --list
    python scripts/download_sciscinet.py --bulk
"""

import subprocess
import sys
from pathlib import Path

# ── Resolve data lake root ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import find_datalake_root

ROOT = find_datalake_root()
LOCAL_BASE = ROOT / "datasets" / "sciscinet"

# ── GCS configuration ──────────────────────────────────────────────────────

GCS_BASE = "gs://sciscinet-neo/v2"

CORE_FILES = [
    "sciscinet_papers.parquet",
    "sciscinet_authors.parquet",
    "sciscinet_affiliations.parquet",
    "sciscinet_fields.parquet",
    "sciscinet_sources.parquet",
    "sciscinet_paper_author_affiliation.parquet",
    "sciscinet_paperrefs.parquet",
    "sciscinet_papersources.parquet",
    "sciscinet_paperfields.parquet",
    # Author details
    "sciscinet_author_details.parquet",
    "sciscinet_authors_paperid.parquet",
    # Funding & linkage files
    "sciscinet_link_nsf.parquet",
    "sciscinet_link_nih.parquet",
    "sciscinet_link_clinicaltrials.parquet",
    "sciscinet_link_patents.parquet",
    "sciscinet_link_twitter.parquet",
    "sciscinet_link_newsfeed.parquet",
    "sciscinet_link_nobellaureates.parquet",
    # Metadata files
    "sciscinet_nsf_metadata.parquet",
    "sciscinet_nih_metadata.parquet",
    "sciscinet_clinicaltrials_metadata.parquet",
    "sciscinet_newsfeed_metadata.parquet",
    "sciscinet_twitter_metadata.parquet",
    # Other useful files
    "sciscinet_affl_assoc_affl.parquet",
    "sciscinet_papers_pmid_pmcid.parquet",
    "funders.parquet",
    "hit_papers_level0.parquet",
    "hit_papers_level1.parquet",
    "normalized_citations_level0.parquet",
    "normalized_citations_level1.parquet",
]

LARGE_FILES = [
    "sciscinet_paperdetails.parquet",       # 117GB
    "sciscinet_papertitleabstract.parquet",  # 92GB
]


# ── Helper functions ────────────────────────────────────────────────────────

def get_remote_size(gcs_path: str) -> int:
    """Get the size of a remote file in bytes."""
    result = subprocess.run(
        ["gsutil", "ls", "-l", gcs_path],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split()
        if len(parts) >= 1 and parts[0].isdigit():
            return int(parts[0])
    return -1


def file_exists_and_complete(local_path: Path, gcs_path: str) -> bool:
    """Check if local file exists and matches remote size."""
    if not local_path.exists():
        return False

    local_size = local_path.stat().st_size
    remote_size = get_remote_size(gcs_path)

    if remote_size == -1:
        print(f"  [SKIP] {local_path.name} exists (couldn't verify size)")
        return True

    if local_size == remote_size:
        print(f"  [SKIP] {local_path.name} ({local_size / 1e9:.2f} GB) - already complete")
        return True
    else:
        print(f"  [INCOMPLETE] {local_path.name}: local={local_size}, remote={remote_size}")
        return False


def cleanup_temp_files(dest: Path):
    """Remove any .gstmp temporary files from failed downloads."""
    for tmp_file in dest.glob("*.gstmp"):
        try:
            tmp_file.unlink()
            print(f"  [CLEANUP] Removed temp file: {tmp_file.name}")
        except Exception as e:
            print(f"  [WARNING] Could not remove {tmp_file.name}: {e}")


def run_gsutil(source: str, dest: Path, parallel: bool = True, max_retries: int = 2):
    """Run gsutil copy command with retry logic."""
    dest.mkdir(parents=True, exist_ok=True)

    cleanup_temp_files(dest)

    filename = source.split("/")[-1]
    local_path = dest / filename

    if file_exists_and_complete(local_path, source):
        return True

    cmd = ["gsutil"]
    if parallel:
        cmd.append("-m")
    cmd.append("-o")
    cmd.append("GSUtil:check_hashes=never")
    cmd.extend(["cp", source, str(dest) + "/"])

    print(f"Downloading: {source}")
    print(f"       -> {dest}/")

    for attempt in range(max_retries + 1):
        if attempt > 0:
            print(f"  [RETRY {attempt}/{max_retries}] Retrying download...")
            cleanup_temp_files(dest)

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            if local_path.exists():
                return True

        cleanup_temp_files(dest)

    print(f"Failed to download {source} after {max_retries + 1} attempts")
    return False


# ── Download commands ───────────────────────────────────────────────────────

def download_core():
    """Download core parquet files."""
    print("\n=== Downloading Core Files ===\n")
    dest = LOCAL_BASE / "core"

    downloaded = 0
    skipped = 0
    failed = 0

    for filename in CORE_FILES:
        source = f"{GCS_BASE}/{filename}"
        local_path = dest / filename

        if file_exists_and_complete(local_path, source):
            skipped += 1
            continue

        if run_gsutil(source, dest):
            downloaded += 1
        else:
            failed += 1

    print(f"\nCore files: {downloaded} downloaded, {skipped} skipped, {failed} failed")


def download_large():
    """Download large parquet files (~210GB)."""
    print("\n=== Downloading Large Files ===\n")
    dest = LOCAL_BASE / "large"

    downloaded = 0
    skipped = 0
    failed = 0

    for filename in LARGE_FILES:
        source = f"{GCS_BASE}/{filename}"
        local_path = dest / filename

        if file_exists_and_complete(local_path, source):
            skipped += 1
            continue

        if run_gsutil(source, dest, parallel=False):
            downloaded += 1
        else:
            failed += 1

    print(f"\nLarge files: {downloaded} downloaded, {skipped} skipped, {failed} failed")


def download_embeddings():
    """Download embeddings (~1.7TB)."""
    print("\n=== Downloading Embeddings ===\n")
    dest = LOCAL_BASE / "embeddings"
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["gsutil", "-m", "rsync", "-r", f"{GCS_BASE}/embeddings/", str(dest) + "/"]

    print(f"Syncing embeddings to {dest}")
    print("Using rsync for incremental download (skips existing files)")
    print("Warning: This is ~1.7TB and will take a long time!")

    subprocess.run(cmd)
    print("\nEmbeddings sync complete!")


def download_all_parquet():
    """Download all parquet files in the bucket (easier approach)."""
    print("\n=== Downloading All Parquet Files ===\n")
    dest = LOCAL_BASE / "all"
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["gsutil", "-m", "rsync", "-x", "embeddings/.*", f"{GCS_BASE}/", str(dest) + "/"]

    print(f"Syncing all parquet files to {dest}")
    print("Using rsync for incremental download (skips existing files)")

    subprocess.run(cmd)
    print("\nSync complete!")


def list_remote_files():
    """List available files in GCS bucket."""
    print("\n=== Available Files in GCS ===\n")
    subprocess.run(["gsutil", "ls", "-lh", f"{GCS_BASE}/"])


def list_local_files():
    """List downloaded files and their sizes."""
    print("\n=== Downloaded Files ===\n")

    for subdir in ["core", "large", "all", "embeddings"]:
        path = LOCAL_BASE / subdir
        if path.exists():
            files = list(path.glob("*.parquet"))
            if files:
                print(f"\n{subdir}/:")
                total_size = 0
                for f in sorted(files):
                    size = f.stat().st_size
                    total_size += size
                    print(f"  {f.name}: {size / 1e9:.2f} GB")
                print(f"  Total: {total_size / 1e9:.2f} GB")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python download_sciscinet.py [OPTIONS]")
        print("")
        print("Options:")
        print("  --list       List available files in GCS bucket")
        print("  --status     Show downloaded files and sizes")
        print("  --core       Download core parquet files")
        print("  --large      Download large files (paperdetails, abstracts ~210GB)")
        print("  --all        Download core + large files")
        print("  --bulk       Download all parquet files at once (simpler)")
        print("  --embeddings Download embeddings (~1.7TB)")
        print("")
        print("Files already downloaded will be automatically skipped.")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--list":
        list_remote_files()
    elif arg == "--status":
        list_local_files()
    elif arg == "--core":
        download_core()
    elif arg == "--large":
        download_large()
    elif arg == "--all":
        download_core()
        download_large()
    elif arg == "--bulk":
        download_all_parquet()
    elif arg == "--embeddings":
        download_embeddings()
    else:
        print(f"Unknown option: {arg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
