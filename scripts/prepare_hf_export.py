#!/usr/bin/env python3
"""
Prepare Parquet export directory for HuggingFace upload.

Creates a structured directory layout:
  parquet_export/{schema}/{table}/*.parquet

This script creates symlinks (not copies) to avoid doubling disk usage.
For actual upload, use `huggingface-cli upload-large-folder`.

Usage:
    python scripts/prepare_hf_export.py
    python scripts/prepare_hf_export.py --dry-run
    python scripts/prepare_hf_export.py --copy  # Copy instead of symlink
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

EXPORT_DIR = ROOT / "parquet_export"

# ── Source mapping ─────────────────────────────────────────────────────────────
# Maps export schema/table → source directory of Parquet files

def get_source_mappings():
    """Build mapping of export paths to source directories."""
    ds = ROOT / "datasets"
    mappings = {}

    # S2AG
    s2ag_dir = ds / "s2ag" / "parquet"
    if s2ag_dir.exists():
        for sub in sorted(s2ag_dir.iterdir()):
            if sub.is_dir() and list(sub.glob("*.parquet")):
                table_name = sub.name.replace("-", "_")
                mappings[f"s2ag/{table_name}"] = sub

    # OpenAlex
    oa_dir = ds / "openalex" / "parquet"
    if oa_dir.exists():
        for sub in sorted(oa_dir.iterdir()):
            if sub.is_dir() and list(sub.glob("*.parquet")):
                mappings[f"openalex/{sub.name}"] = sub

    # SciSciNet — has core/ and large/ subdirs
    ssn_dir = ds / "sciscinet"
    if ssn_dir.exists():
        for sub in ["core", "large"]:
            sub_dir = ssn_dir / sub
            if sub_dir.is_dir() and list(sub_dir.glob("*.parquet")):
                mappings[f"sciscinet/{sub}"] = sub_dir

    # Papers With Code — flat directory with many parquet files
    pwc_dir = ds / "paperswithcode" / "parquet"
    if pwc_dir.exists() and list(pwc_dir.glob("*.parquet")):
        # Group by filename prefix (papers, methods, datasets, etc.)
        pwc_files = {}
        for f in sorted(pwc_dir.glob("*.parquet")):
            # Remove .parquet extension to get table name
            table_name = f.stem
            pwc_files[table_name] = f
        # Each file becomes its own "table"
        for table_name, filepath in pwc_files.items():
            mappings[f"pwc/{table_name}"] = filepath

    # Retraction Watch
    rw_dir = ds / "retractionwatch" / "parquet"
    if rw_dir.exists() and list(rw_dir.glob("*.parquet")):
        mappings["retwatch/retraction_watch"] = rw_dir

    # Reliance on Science
    ros_dir = ds / "reliance_on_science" / "parquet"
    if ros_dir.exists() and list(ros_dir.glob("*.parquet")):
        mappings["ros/patent_paper_pairs"] = ros_dir

    # Preprint to Paper
    p2p_dir = ds / "preprint_to_paper" / "parquet"
    if p2p_dir.exists() and list(p2p_dir.glob("*.parquet")):
        mappings["p2p/preprint_to_paper"] = p2p_dir

    # Cross-reference tables
    xref_dir = ds / "xref"
    for sub_name in ["unified_papers", "topic_ontology_map", "ontology_bridges"]:
        sub_dir = xref_dir / sub_name
        if sub_dir.exists() and list(sub_dir.glob("*.parquet")):
            mappings[f"xref/{sub_name}"] = sub_dir

    # Ontologies (13)
    for onto_name in [
        "cso", "doid", "go", "mesh", "chebi", "ncit", "hpo",
        "edam", "agrovoc", "unesco", "stw", "msc2020", "physh",
    ]:
        onto_dir = ds / onto_name / "parquet"
        if onto_dir.exists():
            for f in sorted(onto_dir.glob("*.parquet")):
                table_name = f.stem  # e.g., cso_terms, cso_hierarchy, cso_xrefs
                mappings[f"ontologies/{table_name}"] = f

    return mappings


def prepare_export(dry_run=False, use_copy=False):
    """Create the export directory structure."""
    mappings = get_source_mappings()

    print(f"Export directory: {EXPORT_DIR}")
    print(f"Mode: {'dry run' if dry_run else 'copy' if use_copy else 'symlink'}")
    print(f"Mappings: {len(mappings)}")
    print()

    total_files = 0
    total_bytes = 0

    for export_path, source in sorted(mappings.items()):
        dest_dir = EXPORT_DIR / export_path
        source_path = Path(source)

        # Collect files to link/copy
        if source_path.is_file():
            files = [source_path]
        elif source_path.is_dir():
            files = sorted(source_path.glob("*.parquet"))
        else:
            print(f"  SKIP {export_path}: source not found ({source_path})")
            continue

        if not files:
            print(f"  SKIP {export_path}: no parquet files")
            continue

        n_files = len(files)
        size = sum(f.stat().st_size for f in files)
        total_files += n_files
        total_bytes += size

        size_str = _format_size(size)
        print(f"  {export_path:50s}: {n_files:4d} files, {size_str:>8s}")

        if dry_run:
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            dest_file = dest_dir / f.name
            if dest_file.exists() or dest_file.is_symlink():
                dest_file.unlink()

            if use_copy:
                shutil.copy2(f, dest_file)
            else:
                # Create relative symlink
                rel_path = os.path.relpath(f, dest_dir)
                dest_file.symlink_to(rel_path)

    print(f"\nTotal: {total_files} files, {_format_size(total_bytes)}")

    if not dry_run:
        # Write a manifest
        manifest_path = EXPORT_DIR / "MANIFEST.txt"
        with open(manifest_path, "w") as mf:
            mf.write(f"# Science Data Lake Parquet Export\n")
            mf.write(f"# Files: {total_files}\n")
            mf.write(f"# Total size: {_format_size(total_bytes)}\n\n")
            for export_path in sorted(mappings.keys()):
                mf.write(f"{export_path}/\n")
        print(f"Manifest: {manifest_path}")

    return total_files, total_bytes


def _format_size(n_bytes):
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Parquet export for HuggingFace upload"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show what would be done")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of symlinking")
    args = parser.parse_args()

    prepare_export(dry_run=args.dry_run, use_copy=args.copy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
