#!/usr/bin/env python3
"""
Download Semantic Scholar Academic Graph (S2AG) datasets.

Self-contained script with data lake path integration.

Usage:
    python scripts/download_s2ag.py --list-releases
    python scripts/download_s2ag.py --list-datasets
    python scripts/download_s2ag.py --all
    python scripts/download_s2ag.py --dataset papers
    python scripts/download_s2ag.py --dataset papers --workers 8
    python scripts/download_s2ag.py --status
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# ── Resolve data lake root ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import find_datalake_root

ROOT = find_datalake_root()  # also loads .env
OUTPUT_DIR = ROOT / "datasets" / "s2ag" / "raw"

# ── API configuration ───────────────────────────────────────────────────────

BASE_URL = "https://api.semanticscholar.org/datasets/v1"
DEFAULT_WORKERS = 4
CHUNK_SIZE = 8 * 1024 * 1024  # 8MB chunks


class S2AGDownloader:
    """Client for downloading S2AG datasets."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        output_dir: str = None,
        workers: int = DEFAULT_WORKERS,
    ):
        self.api_key = api_key or os.environ.get("S2_API_KEY")
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.workers = workers
        self.session = requests.Session()
        headers = {"User-Agent": "S2AG-Downloader/1.0"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        self.session.headers.update(headers)

    def _request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make a GET request to the API."""
        url = f"{BASE_URL}{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limited. Waiting 60 seconds...")
                time.sleep(60)
                return self._request(endpoint, params)
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

    def list_releases(self) -> list[str]:
        """List all available dataset releases."""
        return self._request("/release/")

    def get_release_info(self, release_id: str = "latest") -> dict:
        """Get metadata for a specific release."""
        return self._request(f"/release/{release_id}")

    def get_dataset_info(self, dataset_name: str, release_id: str = "latest") -> dict:
        """Get download URLs for a specific dataset."""
        return self._request(f"/release/{release_id}/dataset/{dataset_name}")

    def list_datasets(self, release_id: str = "latest") -> list[dict]:
        """List all datasets in a release."""
        release_info = self.get_release_info(release_id)
        return release_info.get("datasets", [])

    def download_file(
        self,
        url: str,
        output_path: Path,
        description: str = "",
        resume: bool = True,
    ) -> bool:
        """Download a single file with progress bar and resume support."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        downloaded_bytes = 0
        mode = "wb"
        headers = {}

        if resume and output_path.exists():
            downloaded_bytes = output_path.stat().st_size
            headers["Range"] = f"bytes={downloaded_bytes}-"
            mode = "ab"

        try:
            response = self.session.get(url, headers=headers, stream=True, timeout=300)

            if response.status_code == 416:
                print(f"  {description}: Already downloaded")
                return True
            elif response.status_code == 206:
                pass  # Resuming partial download
            elif response.status_code == 200:
                downloaded_bytes = 0
                mode = "wb"
            else:
                response.raise_for_status()

            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) + downloaded_bytes if content_length else None

            with open(output_path, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=downloaded_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=description[:40],
                    leave=True,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            return True

        except requests.exceptions.RequestException as e:
            print(f"  Error downloading {description}: {e}")
            return False

    def download_dataset_parallel(
        self,
        dataset_name: str,
        release_id: str = "latest",
        resume: bool = True,
    ) -> bool:
        """Download all files for a dataset using parallel downloads."""
        print(f"\nFetching download URLs for dataset: {dataset_name}")
        dataset_info = self.get_dataset_info(dataset_name, release_id)

        files = dataset_info.get("files", [])
        if not files:
            print(f"No files found for dataset: {dataset_name}")
            return False

        print(f"Found {len(files)} files to download (using {self.workers} workers)")

        actual_release = release_id if release_id != "latest" else "latest"
        dataset_dir = self.output_dir / actual_release / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        tasks = []
        for i, file_url in enumerate(files, 1):
            parsed = urlparse(file_url)
            filename = Path(parsed.path).name
            if not filename:
                filename = f"part_{i:05d}.gz"
            output_path = dataset_dir / filename
            tasks.append((file_url, output_path, f"[{i}/{len(files)}] {filename}"))

        success_count = 0
        failed_files = []

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(
                    self.download_file, url, path, desc, resume
                ): (url, path, desc)
                for url, path, desc in tasks
            }

            for future in as_completed(futures):
                _, path, _ = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed_files.append(path.name)
                except Exception as e:
                    print(f"Error downloading {path.name}: {e}")
                    failed_files.append(path.name)

        print(f"\nDownload complete: {success_count}/{len(files)} files")
        if failed_files:
            print(f"Failed files: {', '.join(failed_files)}")
            return False

        return True

    def download_dataset(
        self,
        dataset_name: str,
        release_id: str = "latest",
        resume: bool = True,
    ) -> bool:
        """Download all files for a dataset sequentially."""
        print(f"\nFetching download URLs for dataset: {dataset_name}")
        dataset_info = self.get_dataset_info(dataset_name, release_id)

        files = dataset_info.get("files", [])
        if not files:
            print(f"No files found for dataset: {dataset_name}")
            return False

        print(f"Found {len(files)} files to download")

        actual_release = release_id if release_id != "latest" else "latest"
        dataset_dir = self.output_dir / actual_release / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(dataset_info, f, indent=2)

        success_count = 0
        failed_files = []

        for i, file_url in enumerate(files, 1):
            parsed = urlparse(file_url)
            filename = Path(parsed.path).name
            if not filename:
                filename = f"part_{i:05d}.gz"

            output_path = dataset_dir / filename
            description = f"[{i}/{len(files)}] {filename}"

            if self.download_file(file_url, output_path, description, resume):
                success_count += 1
            else:
                failed_files.append(filename)

        print(f"\nDownload complete: {success_count}/{len(files)} files")
        if failed_files:
            print(f"Failed files: {', '.join(failed_files)}")
            return False

        return True

    def download_all_datasets(
        self,
        release_id: str = "latest",
        resume: bool = True,
        parallel: bool = True,
    ) -> bool:
        """Download all datasets from a release."""
        datasets = self.list_datasets(release_id)
        if not datasets:
            print("No datasets found in release")
            return False

        print(f"Found {len(datasets)} datasets to download:")
        for ds in datasets:
            print(f"  - {ds.get('name', 'unknown')}: {ds.get('description', '')}")

        all_success = True
        for ds in datasets:
            dataset_name = ds.get("name")
            if not dataset_name:
                continue

            if parallel:
                success = self.download_dataset_parallel(dataset_name, release_id, resume)
            else:
                success = self.download_dataset(dataset_name, release_id, resume)

            if not success:
                all_success = False

        return all_success


def show_status():
    """Show download status: file counts and sizes per dataset."""
    print("\n=== S2AG Download Status ===\n")
    print(f"Output directory: {OUTPUT_DIR}\n")

    if not OUTPUT_DIR.exists():
        print("  No downloads found.")
        return

    total_files = 0
    total_size = 0

    for release_dir in sorted(OUTPUT_DIR.iterdir()):
        if not release_dir.is_dir():
            continue
        print(f"  Release: {release_dir.name}")
        for dataset_dir in sorted(release_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            files = list(dataset_dir.glob("*.gz"))
            size = sum(f.stat().st_size for f in files)
            total_files += len(files)
            total_size += size
            print(f"    {dataset_dir.name:25s}  {len(files):5d} files  {size / (1024**3):8.2f} GB")

    print(f"\n  Total: {total_files} files, {total_size / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Download Semantic Scholar Academic Graph (S2AG) datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--list-releases", action="store_true",
                        help="List all available releases")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List all datasets in a release")
    parser.add_argument("--dataset", type=str,
                        help="Name of the dataset to download (e.g., 'papers', 'authors')")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets in the release")
    parser.add_argument("--release", type=str, default="latest",
                        help="Release ID to use (default: 'latest')")
    parser.add_argument("--status", action="store_true",
                        help="Show download status (file counts, sizes)")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("S2_API_KEY"),
                        help="Semantic Scholar API key (or set S2_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel download workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume for partial downloads")
    parser.add_argument("--sequential", action="store_true",
                        help="Download files sequentially instead of in parallel")

    args = parser.parse_args()

    resume = not args.no_resume
    parallel = not args.sequential

    if args.status:
        show_status()
        return 0

    downloader = S2AGDownloader(
        api_key=args.api_key,
        output_dir=args.output_dir,
        workers=args.workers,
    )

    try:
        if args.list_releases:
            print("Available releases:")
            releases = downloader.list_releases()
            for release in releases:
                print(f"  - {release}")
            print(f"\nTotal: {len(releases)} releases")
            return 0

        if args.list_datasets:
            print(f"Datasets in release '{args.release}':")
            datasets = downloader.list_datasets(args.release)
            for ds in datasets:
                name = ds.get("name", "unknown")
                desc = ds.get("description", "")
                print(f"  - {name}: {desc}")
            print(f"\nTotal: {len(datasets)} datasets")
            return 0

        if args.dataset:
            print(f"Downloading dataset: {args.dataset}")
            print(f"Release: {args.release}")
            print(f"Output directory: {args.output_dir}")

            if parallel:
                success = downloader.download_dataset_parallel(
                    args.dataset, args.release, resume
                )
            else:
                success = downloader.download_dataset(
                    args.dataset, args.release, resume
                )
            return 0 if success else 1

        if args.all:
            print(f"Downloading all datasets from release: {args.release}")
            print(f"Output directory: {args.output_dir}")
            success = downloader.download_all_datasets(args.release, resume, parallel)
            return 0 if success else 1

        parser.print_help()
        return 1

    except KeyboardInterrupt:
        print("\nDownload cancelled by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
