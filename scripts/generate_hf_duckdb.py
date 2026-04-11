#!/usr/bin/env python3
"""
Generate a HuggingFace-ready DuckDB database file.

Takes the local datalake.duckdb and rewrites all view definitions to point
to hf://datasets/J0nasW/science-datalake/... paths instead of local filesystem
paths. Only includes views for datasets that are actually hosted on HuggingFace
(excludes S2AG, fulltext, USPTO, EPO, CPC taxonomy, Lens).

Usage:
    python scripts/generate_hf_duckdb.py
    python scripts/generate_hf_duckdb.py --output parquet_export/datalake.duckdb
"""

import argparse
import re
import sys
from pathlib import Path

import duckdb

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

LOCAL_DB = ROOT / "datalake.duckdb"
HF_REPO = "hf://datasets/J0nasW/science-datalake"

# Mapping from local path prefixes to HF path prefixes
# Only datasets that are hosted on HuggingFace
LOCAL_TO_HF = {
    "datasets/openalex/parquet": "openalex",
    "datasets/sciscinet/core": "sciscinet/core",
    "datasets/sciscinet/large": "sciscinet/large",
    "datasets/paperswithcode/parquet": "pwc",
    "datasets/retractionwatch/parquet": "retwatch/retraction_watch",
    "datasets/reliance_on_science/parquet": "ros",
    "datasets/preprint_to_paper/parquet": "p2p/preprint_to_paper",
    "datasets/xref/unified_papers": "xref/unified_papers",
    "datasets/xref/topic_ontology_map": "xref/topic_ontology_map",
    "datasets/xref/ontology_bridges": "xref/ontology_bridges",
}

# Ontologies: datasets/{name}/parquet → ontologies/{name}_*
ONTOLOGY_NAMES = [
    "mesh", "go", "doid", "chebi", "hpo", "ncit", "edam",
    "physh", "msc2020", "agrovoc", "unesco", "stw", "cso",
]
for name in ONTOLOGY_NAMES:
    LOCAL_TO_HF[f"datasets/{name}/parquet"] = f"ontologies"

# Schemas to exclude (not on HuggingFace)
EXCLUDED_SCHEMAS = {"s2ag", "fulltext", "uspto", "epo", "cpc_taxonomy", "lens", "main"}


def rewrite_path(local_path: str) -> str | None:
    """Rewrite a local Parquet path to an HF path. Returns None if not on HF."""
    # Strip the root prefix
    rel = local_path.replace(str(ROOT) + "/", "")

    for local_prefix, hf_prefix in LOCAL_TO_HF.items():
        if rel.startswith(local_prefix):
            suffix = rel[len(local_prefix):]
            # For ontologies, the files are in ontologies/ flat dir
            if hf_prefix == "ontologies":
                # datasets/cso/parquet/cso_terms.parquet → ontologies/cso_terms/cso_terms.parquet
                filename = Path(suffix).name  # e.g., cso_terms.parquet
                stem = Path(suffix).stem      # e.g., cso_terms
                if "*" in filename:
                    # Glob pattern: *.parquet
                    return f"{HF_REPO}/{hf_prefix}/{stem}/{filename}"
                return f"{HF_REPO}/{hf_prefix}/{stem}/{filename}"
            # For PWC, files go into pwc/{table_name}/
            if hf_prefix == "pwc":
                filename = Path(suffix).name
                stem = Path(suffix).stem
                if "*" in filename:
                    return f"{HF_REPO}/{hf_prefix}/{stem}/{filename}"
                return f"{HF_REPO}/{hf_prefix}/{stem}/{filename}"
            # For RoS, files go into ros/{table_name}/
            if hf_prefix == "ros":
                filename = Path(suffix).name
                stem = Path(suffix).stem
                if "*" in filename:
                    return f"{HF_REPO}/{hf_prefix}/{stem}/{filename}"
                return f"{HF_REPO}/{hf_prefix}/{stem}/{filename}"
            return f"{HF_REPO}/{hf_prefix}{suffix}"
    return None


def generate_hf_db(output_path: Path):
    """Generate HF-ready DuckDB database."""
    # Read all views from local DB
    local_con = duckdb.connect(str(LOCAL_DB), read_only=True)
    views = local_con.execute("""
        SELECT schema_name, view_name, sql FROM duckdb_views()
        ORDER BY schema_name, view_name
    """).fetchall()
    local_con.close()

    # Create new DB
    if output_path.exists():
        output_path.unlink()
    hf_con = duckdb.connect(str(output_path))
    hf_con.execute("INSTALL httpfs; LOAD httpfs;")

    created = 0
    skipped = 0
    schemas_created = set()

    for schema, name, sql in views:
        # Skip excluded schemas
        if schema in EXCLUDED_SCHEMAS:
            skipped += 1
            continue

        # Rewrite all local paths to HF paths
        new_sql = sql

        def replace_path(match):
            local_path = match.group(1)
            hf_path = rewrite_path(local_path)
            if hf_path is None:
                return None  # Signal to skip this view
            return f"read_parquet('{hf_path}')"

        # Find all read_parquet calls and rewrite
        paths_ok = True
        def do_replace(match):
            nonlocal paths_ok
            result = replace_path(match)
            if result is None:
                paths_ok = False
                return match.group(0)
            return result

        new_sql = re.sub(r"read_parquet\('([^']+)'\)", do_replace, new_sql)

        if not paths_ok:
            skipped += 1
            continue

        # Ensure schema exists
        if schema not in schemas_created:
            hf_con.execute(f"CREATE SCHEMA IF NOT EXISTS \"{schema}\"")
            schemas_created.add(schema)

        # Replace CREATE VIEW with CREATE OR REPLACE VIEW
        new_sql = new_sql.replace("CREATE VIEW", "CREATE OR REPLACE VIEW", 1)

        try:
            hf_con.execute(new_sql)
            created += 1
        except Exception as e:
            print(f"  WARN: {schema}.{name}: {e}")
            skipped += 1

    hf_con.close()

    size = output_path.stat().st_size
    print(f"Generated: {output_path}")
    print(f"  Views: {created} created, {skipped} skipped")
    print(f"  Schemas: {len(schemas_created)}")
    print(f"  Size: {size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "parquet_export" / "datalake.duckdb",
                        help="Output path for the HF-ready DuckDB file")
    args = parser.parse_args()

    generate_hf_db(args.output)


if __name__ == "__main__":
    main()
