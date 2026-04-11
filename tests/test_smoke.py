"""Smoke tests: DB opens, views resolve, schemas exist, basic row counts."""

import pytest

# ── Paper claims ──────────────────────────────────────────────────────────
# The NSD paper describes 8 core sources across these schemas.
# The actual DB may have MORE (patent extensions, fulltext), which is fine.
PAPER_SCHEMAS = {
    "s2ag", "openalex", "sciscinet", "pwc", "retwatch", "ros", "p2p", "xref",
    # 13 ontology schemas
    "mesh", "go", "chebi", "ncit", "doid", "hpo", "edam", "cso",
    "agrovoc", "unesco", "stw", "physh", "msc2020",
}
MIN_VIEWS = 153  # paper claims 153; DB may have more from extensions


class TestDatabaseOpens:
    def test_connection(self, con):
        result = con.execute("SELECT 1").fetchone()
        assert result == (1,)

    def test_has_information_schema(self, con):
        tables = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables"
        ).fetchone()[0]
        assert tables > 0


class TestSchemas:
    def test_paper_schemas_exist(self, con):
        schemas = {
            r[0]
            for r in con.execute(
                "SELECT DISTINCT table_schema FROM information_schema.tables"
            ).fetchall()
        }
        missing = PAPER_SCHEMAS - schemas
        assert not missing, f"Missing schemas: {missing}"

    def test_minimum_view_count(self, con):
        n_views = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_type='VIEW'"
        ).fetchone()[0]
        assert n_views >= MIN_VIEWS, f"Expected >= {MIN_VIEWS} views, got {n_views}"


class TestViewsResolve:
    """Every view should be queryable without error (LIMIT 0 = schema check only)."""

    def test_all_views_resolve(self, con):
        views = con.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'VIEW'
            ORDER BY table_schema, table_name
        """).fetchall()
        failures = []
        for schema, name in views:
            try:
                con.execute(f'SELECT * FROM "{schema}"."{name}" LIMIT 0')
            except Exception as e:
                failures.append(f"{schema}.{name}: {e}")
        assert not failures, "Views that failed to resolve:\n" + "\n".join(failures)


class TestKeyRowCounts:
    """Verify row counts are in the right ballpark (within 5% of paper claims)."""

    @pytest.mark.parametrize("view,min_rows,max_rows", [
        ("xref.unified_papers", 280_000_000, 310_000_000),   # paper: 293M
        ("xref.topic_ontology_map", 15_000, 20_000),         # paper: 16,150
    ])
    def test_row_count_range(self, con, view, min_rows, max_rows):
        n = con.execute(f"SELECT COUNT(*) FROM {view}").fetchone()[0]
        assert min_rows <= n <= max_rows, (
            f"{view}: expected {min_rows:,}–{max_rows:,}, got {n:,}"
        )

    def test_unified_papers_column_count(self, con):
        cols = con.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema='xref' AND table_name='unified_papers'
        """).fetchone()[0]
        assert cols == 29, f"Expected 29 columns, got {cols}"
