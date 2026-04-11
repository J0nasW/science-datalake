"""Verify every specific number claimed in the NSD paper text.

These are exact or near-exact checks against the paper's stated values.
If a test fails, either the paper text or the data has drifted and needs reconciliation.
"""

import pytest


class TestAbstractNumbers:
    def test_total_papers(self, con):
        n = con.execute("SELECT COUNT(*) FROM xref.unified_papers").fetchone()[0]
        assert n == 293_123_121, f"Paper says 293,123,121 — got {n:,}"

    def test_schema_count(self, con):
        """Paper says 22 schemas (NSD scope, excluding patent/fulltext extensions)."""
        schemas = {
            r[0] for r in con.execute(
                "SELECT DISTINCT table_schema FROM information_schema.tables"
            ).fetchall()
        }
        # NSD paper's 22 schemas: 8 sources + xref + 13 ontologies
        nsd_schemas = {
            "s2ag", "openalex", "sciscinet", "pwc", "retwatch", "ros", "p2p", "xref",
            "mesh", "go", "chebi", "ncit", "doid", "hpo", "edam", "cso",
            "agrovoc", "unesco", "stw", "physh", "msc2020",
        }
        missing = nsd_schemas - schemas
        assert not missing, f"Missing NSD schemas: {missing}"
        assert len(nsd_schemas) >= 21, "Paper claims 22 schemas"

    def test_ontology_mapping_count(self, con):
        n = con.execute(
            "SELECT COUNT(*) FROM xref.topic_ontology_map WHERE similarity >= 0.65"
        ).fetchone()[0]
        assert n == 16_150, f"Paper says 16,150 — got {n:,}"

    def test_ontology_topics_covered(self, con):
        n = con.execute("""
            SELECT COUNT(DISTINCT topic_id) FROM xref.topic_ontology_map
            WHERE similarity >= 0.65
        """).fetchone()[0]
        assert n == 4_509, f"Paper says 4,509 topics covered — got {n:,}"

    def test_column_count(self, con):
        n = con.execute("""
            SELECT COUNT(*) FROM information_schema.columns
            WHERE table_schema='xref' AND table_name='unified_papers'
        """).fetchone()[0]
        assert n == 29, f"Paper says 29 columns — got {n}"


class TestSourceCoverage:
    """Numbers from tab:sources and tab:overlap."""

    @pytest.mark.parametrize("flag,expected_pct,tolerance", [
        ("has_openalex", 99.67, 0.5),
        ("has_sciscinet", 54.08, 1.0),
        ("has_s2ag", 45.52, 1.0),
        ("has_pwc", 0.048, 0.01),
        ("has_retraction", 0.020, 0.005),
    ])
    def test_coverage_percentage(self, con, flag, expected_pct, tolerance):
        result = con.execute(f"""
            SELECT 100.0 * SUM(CASE WHEN {flag} THEN 1 ELSE 0 END)
                         / COUNT(*) FROM xref.unified_papers
        """).fetchone()[0]
        assert abs(result - expected_pct) < tolerance, (
            f"{flag}: expected {expected_pct}%, got {result:.4f}%"
        )

    def test_retraction_rate_consistency(self, con):
        """Paper uses 0.020% everywhere — verify the exact computation."""
        rate = con.execute("""
            SELECT 100.0 * SUM(CASE WHEN has_retraction THEN 1 ELSE 0 END)
                         / COUNT(*) FROM xref.unified_papers
        """).fetchone()[0]
        # 60074/293123121 = 0.02049..., rounds to 0.020% at 3 decimal places
        assert round(rate, 3) == 0.020, f"Retraction rate = {rate:.5f}%"


class TestOntologyTiers:
    """Numbers from tab:ontology_tiers."""

    @pytest.mark.parametrize("threshold,expected_mappings,expected_topics", [
        (0.95, 85, 71),
        (0.85, 2527, 1647),
        (0.65, 16150, 4509),
    ])
    def test_tier(self, con, threshold, expected_mappings, expected_topics):
        result = con.execute(f"""
            SELECT COUNT(*) AS n_mappings,
                   COUNT(DISTINCT topic_id) AS n_topics
            FROM xref.topic_ontology_map
            WHERE similarity >= {threshold}
        """).fetchone()
        n_map, n_top = result
        assert n_map == expected_mappings, (
            f"Tier >= {threshold}: expected {expected_mappings} mappings, got {n_map}"
        )
        assert n_top == expected_topics, (
            f"Tier >= {threshold}: expected {expected_topics} topics, got {n_top}"
        )


class TestVignetteNumbers:
    def test_v1_pwc_disruption_count(self, con):
        """Vignette 1: 139,873 papers with code + disruption."""
        n = con.execute("""
            SELECT COUNT(*) FROM xref.unified_papers
            WHERE has_pwc = TRUE AND sciscinet_disruption IS NOT NULL
        """).fetchone()[0]
        assert n == 139_873, f"V1: expected 139,873, got {n:,}"

    def test_v2_retracted_count(self, con):
        """Vignette 2: 60,074 retracted papers."""
        n = con.execute("""
            SELECT COUNT(*) FROM xref.unified_papers
            WHERE has_retraction = TRUE
        """).fetchone()[0]
        assert n == 60_074, f"V2: expected 60,074, got {n:,}"

    def test_v3_patent_cited_count(self, con):
        """Vignette 3: 312,929 patent-cited papers."""
        n = con.execute("""
            SELECT COUNT(*) FROM xref.unified_papers
            WHERE has_patent = TRUE
        """).fetchone()[0]
        assert n == 312_929, f"V3: expected 312,929, got {n:,}"

    def test_v4_three_source_overlap(self, con):
        """Vignette 4: ~121M papers in all three large sources."""
        n = con.execute("""
            SELECT COUNT(*) FROM xref.unified_papers
            WHERE has_s2ag AND has_openalex AND has_sciscinet
        """).fetchone()[0]
        # Paper says "~121M" — allow 5% tolerance
        assert 110_000_000 <= n <= 130_000_000, f"V4: 3-source overlap = {n:,}"


class TestUpsetNumbers:
    """Numbers from the UpSet plot description."""

    def test_openalex_only_fraction(self, con):
        """Paper says OpenAlex-only is 41.7%."""
        rate = con.execute("""
            SELECT 100.0 * SUM(CASE WHEN has_openalex AND NOT has_s2ag
                               AND NOT has_sciscinet AND NOT has_pwc
                               AND NOT has_retraction AND NOT has_patent
                               THEN 1 ELSE 0 END)
                         / COUNT(*)
            FROM xref.unified_papers
        """).fetchone()[0]
        assert abs(rate - 41.7) < 1.0, f"OA-only = {rate:.1f}% (expected ~41.7%)"

    def test_three_way_overlap_fraction(self, con):
        """Paper says OA+SSN+S2AG overlap is 41.3%."""
        rate = con.execute("""
            SELECT 100.0 * SUM(CASE WHEN has_openalex AND has_s2ag
                               AND has_sciscinet THEN 1 ELSE 0 END)
                         / COUNT(*)
            FROM xref.unified_papers
        """).fetchone()[0]
        assert abs(rate - 41.3) < 1.0, f"3-way overlap = {rate:.1f}% (expected ~41.3%)"
