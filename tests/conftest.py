"""Shared fixtures for the Science Data Lake test suite."""

import pytest
import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT / "datalake.duckdb")


@pytest.fixture(scope="session")
def con():
    """Read-only DuckDB connection shared across all tests in a session."""
    c = duckdb.connect(DB_PATH, read_only=True)
    yield c
    c.close()
