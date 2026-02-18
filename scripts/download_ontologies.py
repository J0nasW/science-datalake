#!/usr/bin/env python3
"""
Download scientific ontologies for the datalake.

Usage:
    python scripts/download_ontologies.py --all
    python scripts/download_ontologies.py --ontology mesh,go,doid
    python scripts/download_ontologies.py --list
    python scripts/download_ontologies.py --status
"""

import argparse
import gzip
import json
import shutil
import sys
import zipfile
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

from ontology_registry import ALL_ONTOLOGY_NAMES, ONTOLOGIES

ONTOLOGIES_DIR = ROOT / "ontologies"


def _meta_path(ont_dir: Path) -> Path:
    return ont_dir / ".download_meta.json"


def _load_meta(ont_dir: Path) -> dict:
    mp = _meta_path(ont_dir)
    if mp.exists():
        with open(mp) as f:
            return json.load(f)
    return {}


def _save_meta(ont_dir: Path, meta: dict):
    with open(_meta_path(ont_dir), "w") as f:
        json.dump(meta, f, indent=2)


def _download_file(url: str, dest: Path, meta: dict) -> bool:
    """Download a file with ETag/Last-Modified caching. Returns True if new data."""
    headers = {}
    if meta.get("etag"):
        headers["If-None-Match"] = meta["etag"]
    if meta.get("last_modified"):
        headers["If-Modified-Since"] = meta["last_modified"]

    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=300)
    except requests.RequestException as e:
        print(f"    ERROR: {e}")
        return False

    if resp.status_code == 304:
        print(f"    Not modified (cached): {dest.name}")
        return False

    if resp.status_code != 200:
        print(f"    ERROR: HTTP {resp.status_code} for {url}")
        return False

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                print(f"\r    Downloading {dest.name}: {downloaded / (1024*1024):.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r    Downloading {dest.name}: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
    print()

    # Update cache headers
    if resp.headers.get("ETag"):
        meta["etag"] = resp.headers["ETag"]
    if resp.headers.get("Last-Modified"):
        meta["last_modified"] = resp.headers["Last-Modified"]
    meta["url"] = url
    meta["size_bytes"] = dest.stat().st_size

    return True


def _extract_gz(gz_path: Path, ont_dir: Path) -> Path:
    """Extract a .gz file, return the extracted path."""
    out_path = ont_dir / gz_path.stem  # removes .gz
    # Check if file is actually gzip (magic bytes \x1f\x8b)
    with open(gz_path, "rb") as f:
        magic = f.read(2)
    if magic != b'\x1f\x8b':
        # Server already decompressed (e.g. Content-Encoding: x-gzip)
        print(f"    {gz_path.name} is already decompressed, copying -> {out_path.name}")
        shutil.copy2(gz_path, out_path)
        return out_path
    print(f"    Extracting {gz_path.name} -> {out_path.name}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


def _extract_zip(zip_path: Path, ont_dir: Path) -> list[Path]:
    """Extract a .zip file, return list of extracted paths."""
    print(f"    Extracting {zip_path.name}")
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            zf.extract(name, ont_dir)
            extracted.append(ont_dir / name)
            print(f"      -> {name}")
    return extracted


def download_ontology(name: str, force: bool = False) -> bool:
    """Download a single ontology. Returns True on success."""
    if name not in ONTOLOGIES:
        print(f"  Unknown ontology: {name}")
        return False

    info = ONTOLOGIES[name]
    dl = info["download"]
    ont_dir = ONTOLOGIES_DIR / name
    ont_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [{name}] {info['full_name']}")

    # Manual download: just check the file exists
    if dl.get("method") == "manual":
        dest = ont_dir / dl["filename"]
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"    {dest.name}: {size_mb:.1f} MB (manual download, already present)")
            return True
        else:
            print(f"    MANUAL DOWNLOAD REQUIRED: {dl['url']}")
            print(f"    Place the file at: {dest}")
            return False

    meta = _load_meta(ont_dir)
    if force:
        meta.pop("etag", None)
        meta.pop("last_modified", None)

    url = dl["url"]
    filename = dl["filename"]
    dest = ont_dir / filename

    downloaded = _download_file(url, dest, meta)

    if not downloaded and not dest.exists():
        # Try fallback URL
        fallback = dl.get("fallback_url")
        if fallback:
            fb_filename = dl.get("fallback_filename", filename)
            fb_dest = ont_dir / fb_filename
            print(f"    Trying fallback: {fallback}")
            downloaded = _download_file(fallback, fb_dest, meta)

    if not downloaded and not dest.exists():
        print(f"    FAILED: No data downloaded for {name}")
        return False

    # Extract compressed files
    extract_type = dl.get("extract")
    if downloaded and extract_type == "gz" and dest.suffix == ".gz":
        _extract_gz(dest, ont_dir)
    elif downloaded and extract_type == "zip" and dest.suffix == ".zip":
        _extract_zip(dest, ont_dir)

    # Download extra files (e.g., PhySH SKOS compat)
    for extra in dl.get("extra_files", []):
        extra_dest = ont_dir / extra["filename"]
        _download_file(extra["url"], extra_dest, meta)

    _save_meta(ont_dir, meta)

    # Report file sizes
    for f in sorted(ont_dir.iterdir()):
        if f.name.startswith("."):
            continue
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name}: {size_mb:.1f} MB")

    return True


def cmd_list():
    """List all available ontologies."""
    print(f"\n{'Name':12s}  {'Full Name':45s}  {'Domain':20s}  {'Format':8s}  {'Est. Terms':>10s}")
    print("-" * 105)
    for name, info in ONTOLOGIES.items():
        print(
            f"{name:12s}  {info['full_name']:45s}  {info['domain']:20s}  "
            f"{info['format']:8s}  {info['estimated_terms']:>10,}"
        )
    print(f"\nTotal: {len(ONTOLOGIES)} ontologies")


def cmd_status():
    """Show download status of all ontologies."""
    print(f"\n{'Name':12s}  {'Status':12s}  {'Size':>10s}  {'Files':>5s}")
    print("-" * 50)
    total_size = 0
    for name in ALL_ONTOLOGY_NAMES:
        ont_dir = ONTOLOGIES_DIR / name
        if ont_dir.exists():
            files = [f for f in ont_dir.iterdir() if not f.name.startswith(".")]
            size = sum(f.stat().st_size for f in files)
            total_size += size
            status = "downloaded"
            size_str = f"{size / (1024*1024):.1f} MB"
            print(f"{name:12s}  {status:12s}  {size_str:>10s}  {len(files):>5d}")
        else:
            print(f"{name:12s}  {'missing':12s}  {'':>10s}  {'':>5s}")
    print(f"\nTotal downloaded: {total_size / (1024*1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download scientific ontologies")
    parser.add_argument("--all", action="store_true", help="Download all ontologies")
    parser.add_argument("--ontology", type=str, help="Comma-separated list of ontologies to download")
    parser.add_argument("--list", action="store_true", help="List available ontologies")
    parser.add_argument("--status", action="store_true", help="Show download status")
    parser.add_argument("--force", action="store_true", help="Force re-download (ignore cache)")
    args = parser.parse_args()

    if args.list:
        cmd_list()
        return 0

    if args.status:
        cmd_status()
        return 0

    if args.all:
        names = ALL_ONTOLOGY_NAMES
    elif args.ontology:
        names = [n.strip() for n in args.ontology.split(",")]
    else:
        parser.print_help()
        return 1

    print(f"=== Downloading {len(names)} ontologies ===")

    success = 0
    failed = []
    for name in names:
        try:
            if download_ontology(name, force=args.force):
                success += 1
            else:
                failed.append(name)
        except Exception as e:
            print(f"    ERROR: {e}")
            failed.append(name)

    print(f"\n=== Download complete: {success}/{len(names)} succeeded ===")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
