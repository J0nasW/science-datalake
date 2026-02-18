"""
Data lake path resolution and portability logic.

Resolves the data lake root at runtime so paths work regardless
of where the NVME is mounted. Used by all other scripts.

Resolution order:
1. Walk up from this script's location to find datalake.json
2. DATALAKE_ROOT environment variable
3. Common mount points
"""

import json
import os
from pathlib import Path

# Common mount points to try as fallback
COMMON_MOUNT_POINTS = [
    "/mnt/data/science_datalake",
    "/data/science_datalake",
]

MANIFEST_FILE = "datalake.json"


def _load_dotenv(root: Path):
    """Load .env file from data lake root into os.environ (setdefault, won't override)."""
    env_file = root / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())


def find_datalake_root() -> Path:
    """Find the data lake root directory.

    Resolution order:
    1. Walk up from this script's location to find datalake.json
    2. DATALAKE_ROOT environment variable
    3. Common mount points

    Also loads ROOT/.env into os.environ (if present).
    """
    root = None

    # Strategy 1: Walk up from script location
    current = Path(__file__).resolve().parent
    for _ in range(5):  # max 5 levels up
        manifest = current / MANIFEST_FILE
        if manifest.exists():
            root = current
            break
        current = current.parent

    # Strategy 2: Environment variable
    if root is None:
        env_root = os.environ.get("DATALAKE_ROOT")
        if env_root:
            candidate = Path(env_root)
            if candidate.exists() and (candidate / MANIFEST_FILE).exists():
                root = candidate

    # Strategy 3: Common mount points
    if root is None:
        for mount in COMMON_MOUNT_POINTS:
            candidate = Path(mount)
            if candidate.exists() and (candidate / MANIFEST_FILE).exists():
                root = candidate
                break

    if root is None:
        raise FileNotFoundError(
            "Cannot find data lake root. Ensure datalake.json exists or set DATALAKE_ROOT."
        )

    _load_dotenv(root)
    return root


def load_manifest(root: Path = None) -> dict:
    """Load the datalake.json manifest."""
    if root is None:
        root = find_datalake_root()
    manifest_path = root / MANIFEST_FILE
    with open(manifest_path) as f:
        return json.load(f)


def resolve_paths(root: Path = None) -> dict:
    """Resolve all dataset paths relative to the data lake root.

    Returns a dict mapping dataset names to their absolute paths.
    """
    if root is None:
        root = find_datalake_root()
    manifest = load_manifest(root)

    paths = {"root": root}
    for ds_name, ds_info in manifest.get("datasets", {}).items():
        ds_base = root / ds_info["path"]
        paths[ds_name] = ds_base
        if "parquet_path" in ds_info:
            paths[f"{ds_name}_parquet"] = root / ds_info["parquet_path"]
        if "snapshot_path" in ds_info:
            paths[f"{ds_name}_snapshot"] = root / ds_info["snapshot_path"]

    paths["db"] = root / manifest.get("db_path", "datalake.duckdb")
    return paths


# Module-level convenience
ROOT = None
PATHS = None


def init():
    """Initialize module-level ROOT and PATHS."""
    global ROOT, PATHS
    ROOT = find_datalake_root()
    PATHS = resolve_paths(ROOT)
    return ROOT, PATHS
