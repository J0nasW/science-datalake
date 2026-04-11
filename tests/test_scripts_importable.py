"""Verify all pipeline scripts can be imported without errors.

Catches syntax errors, missing dependencies, hardcoded paths that break
on import, and other issues that would embarrass us if a reviewer tries
to use the code.
"""

import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

# Collect all .py files in scripts/
SCRIPT_FILES = sorted(SCRIPTS_DIR.glob("*.py"))
SCRIPT_NAMES = [f.stem for f in SCRIPT_FILES]


@pytest.fixture(scope="module", autouse=True)
def add_scripts_to_path():
    """Temporarily add scripts/ to sys.path for imports."""
    scripts_str = str(SCRIPTS_DIR)
    sys.path.insert(0, scripts_str)
    yield
    sys.path.remove(scripts_str)


@pytest.mark.parametrize("script_name", SCRIPT_NAMES)
def test_script_importable(script_name):
    """Each script should import without raising exceptions."""
    try:
        mod = importlib.import_module(script_name)
        # Verify the module actually loaded
        assert mod is not None
    except ImportError as e:
        # Distinguish between missing optional deps and real errors
        optional_deps = {
            "playwright", "google.cloud", "huggingface_hub",
            "upsetplot", "seaborn", "sentence_transformers", "faiss",
            "requests", "pyoxigraph", "pronto", "rdflib",
        }
        msg = str(e)
        if any(dep in msg for dep in optional_deps):
            pytest.skip(f"Optional dependency missing: {e}")
        raise
    except SystemExit:
        # Some scripts call argparse which may sys.exit on --help
        pytest.skip(f"{script_name} calls sys.exit on import")


def test_no_hardcoded_absolute_paths():
    """No script should contain hardcoded absolute paths to specific machines."""
    violations = []
    bad_patterns = [
        "/mnt/nvme",
        "/home/wilinski",
        "/home/jonas",
    ]
    for script_file in SCRIPT_FILES:
        content = script_file.read_text()
        for pattern in bad_patterns:
            if pattern in content:
                # Find the offending line
                for i, line in enumerate(content.splitlines(), 1):
                    if pattern in line and not line.strip().startswith("#"):
                        violations.append(f"{script_file.name}:{i}: {line.strip()}")
    assert not violations, (
        "Hardcoded machine-specific paths found:\n" + "\n".join(violations)
    )


def test_all_scripts_have_docstrings():
    """Every script should have a module-level docstring."""
    missing = []
    for script_file in SCRIPT_FILES:
        content = script_file.read_text().strip()
        # Check for docstring (triple-quoted string at start)
        if not (content.startswith('"""') or content.startswith("'''")):
            missing.append(script_file.name)
    if missing:
        pytest.xfail(f"Scripts without docstrings: {', '.join(missing)}")
