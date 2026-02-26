#!/usr/bin/env python3
"""
Create unified DuckDB database with views across all datasets.

Creates schema-namespaced views (s2ag.*, sciscinet.*, openalex.*, pwc.*, ontologies.*, xref.*)
pointing to Parquet files. The database file is tiny (~500KB) since it
stores only view definitions, not data.

Run after mounting on a new workstation to regenerate paths:
    python scripts/create_unified_db.py

Usage:
    python scripts/create_unified_db.py              # Create full database
    python scripts/create_unified_db.py --summary     # Print current view info
"""

import argparse
import sys
from pathlib import Path

import duckdb

# Resolve data lake root
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# Import config for path resolution
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass  # fallback to script-relative

DB_PATH = ROOT / "datalake.duckdb"
S2AG_PARQUET = ROOT / "datasets" / "s2ag" / "parquet"
SCISCINET_CORE = ROOT / "datasets" / "sciscinet" / "core"
SCISCINET_LARGE = ROOT / "datasets" / "sciscinet" / "large"
OPENALEX_PARQUET = ROOT / "datasets" / "openalex" / "parquet"
PWC_PARQUET = ROOT / "datasets" / "paperswithcode" / "parquet"
RETWATCH_PARQUET = ROOT / "datasets" / "retractionwatch" / "parquet"
ROS_PARQUET = ROOT / "datasets" / "reliance_on_science" / "parquet"
P2P_PARQUET = ROOT / "datasets" / "preprint_to_paper" / "parquet"

try:
    from ontology_registry import ALL_ONTOLOGY_NAMES as ONTOLOGY_NAMES
except ImportError:
    ONTOLOGY_NAMES = [
        "mesh", "go", "doid", "chebi", "hpo", "ncit", "edam",
        "physh", "msc2020", "agrovoc", "unesco", "stw", "cso",
    ]


def create_s2ag_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create S2AG schema and views."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS s2ag")
    views = []

    # Papers
    papers_path = S2AG_PARQUET / "papers" / "*.parquet"
    if (S2AG_PARQUET / "papers").exists():
        conn.execute(f"""
            CREATE VIEW s2ag.papers AS
            SELECT
                corpusid,
                title,
                url,
                venue,
                publicationvenueid,
                year,
                referencecount,
                citationcount,
                influentialcitationcount,
                isopenaccess,
                TRY_CAST(publicationdate AS DATE) AS publicationdate,
                publicationtypes,
                authors,
                s2fieldsofstudy,
                journal.name AS journal_name,
                journal.pages AS journal_pages,
                journal.volume AS journal_volume,
                externalids.DOI AS doi,
                externalids.MAG AS mag_id,
                externalids.PubMed AS pubmed_id,
                externalids.ArXiv AS arxiv_id,
                externalids.ACL AS acl_id,
                externalids.DBLP AS dblp_id,
                externalids.PubMedCentral AS pmc_id,
                externalids
            FROM read_parquet('{papers_path}')
        """)
        views.append("s2ag.papers")

        # Paper authors (unnested)
        conn.execute(f"""
            CREATE VIEW s2ag.paper_authors AS
            SELECT
                corpusid,
                author.authorId AS authorid,
                author.name AS author_name,
                row_number() OVER (PARTITION BY corpusid ORDER BY (SELECT NULL)) AS author_position
            FROM read_parquet('{papers_path}'),
            LATERAL UNNEST(authors) AS t(author)
            WHERE authors IS NOT NULL
        """)
        views.append("s2ag.paper_authors")

        # Paper fields (unnested)
        conn.execute(f"""
            CREATE VIEW s2ag.paper_fields AS
            SELECT DISTINCT
                corpusid,
                field.category AS field_category,
                field.source AS field_source
            FROM read_parquet('{papers_path}'),
            LATERAL UNNEST(s2fieldsofstudy) AS t(field)
            WHERE s2fieldsofstudy IS NOT NULL
        """)
        views.append("s2ag.paper_fields")

    # Abstracts
    path = S2AG_PARQUET / "abstracts" / "*.parquet"
    if (S2AG_PARQUET / "abstracts").exists():
        conn.execute(f"""
            CREATE VIEW s2ag.abstracts AS
            SELECT
                corpusid,
                abstract,
                openaccessinfo.disclaimer AS oa_disclaimer,
                openaccessinfo.license AS oa_license,
                openaccessinfo.url AS oa_url,
                openaccessinfo.status AS oa_status,
                openaccessinfo.externalids.DOI AS doi,
                openaccessinfo.externalids.Medline AS medline_id,
                openaccessinfo.externalids.PubMedCentral AS pmc_id,
                openaccessinfo.externalids.MAG AS mag_id,
                openaccessinfo.externalids.ArXiv AS arxiv_id,
                openaccessinfo.externalids.MedRxiv AS medrxiv_id
            FROM read_parquet('{path}')
        """)
        views.append("s2ag.abstracts")

    # Authors
    path = S2AG_PARQUET / "authors" / "*.parquet"
    if (S2AG_PARQUET / "authors").exists():
        conn.execute(f"""
            CREATE VIEW s2ag.authors AS
            SELECT
                authorid, name, url, aliases, affiliations,
                homepage, papercount, citationcount, hindex
            FROM read_parquet('{path}')
        """)
        views.append("s2ag.authors")

    # Citations
    path = S2AG_PARQUET / "citations" / "*.parquet"
    if (S2AG_PARQUET / "citations").exists():
        conn.execute(f"""
            CREATE VIEW s2ag.citations AS
            SELECT
                citationid, citingcorpusid, citedcorpusid,
                isinfluential, contexts, intents
            FROM read_parquet('{path}')
        """)
        views.append("s2ag.citations")

    # Paper IDs
    path = S2AG_PARQUET / "paper-ids" / "*.parquet"
    if (S2AG_PARQUET / "paper-ids").exists():
        conn.execute(f"""
            CREATE VIEW s2ag.paper_ids AS
            SELECT sha, corpusid, "primary" AS is_primary
            FROM read_parquet('{path}')
        """)
        views.append("s2ag.paper_ids")

    # Publication venues
    path = S2AG_PARQUET / "publication-venues" / "*.parquet"
    if (S2AG_PARQUET / "publication-venues").exists():
        conn.execute(f"""
            CREATE VIEW s2ag.publication_venues AS
            SELECT id, name, type, issn, url,
                   alternate_names, alternate_issns, alternate_urls
            FROM read_parquet('{path}')
        """)
        views.append("s2ag.publication_venues")

    # TLDRs
    path = S2AG_PARQUET / "tldrs" / "*.parquet"
    if (S2AG_PARQUET / "tldrs").exists():
        conn.execute(f"""
            CREATE VIEW s2ag.tldrs AS
            SELECT corpusid, model, text
            FROM read_parquet('{path}')
        """)
        views.append("s2ag.tldrs")

    # S2ORC
    path = S2AG_PARQUET / "s2orc_v2" / "*.parquet"
    if (S2AG_PARQUET / "s2orc_v2").exists():
        conn.execute(f"""
            CREATE VIEW s2ag.s2orc AS
            SELECT
                corpusid, title, authors,
                body.text AS body_text,
                bibliography.text AS bibliography_text,
                openaccessinfo.license AS license,
                openaccessinfo.status AS oa_status,
                openaccessinfo.url AS oa_url
            FROM read_parquet('{path}')
        """)
        views.append("s2ag.s2orc")

    return views


def create_sciscinet_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create SciSciNet schema and views."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS sciscinet")
    views = []

    # Define view mappings: (view_name, parquet_path, sql_override)
    core_views = [
        ("papers", "sciscinet_papers.parquet", None),
        ("authors", "sciscinet_authors.parquet", None),
        ("author_details", "sciscinet_author_details.parquet", None),
        ("affiliations", "sciscinet_affiliations.parquet", None),
        ("sources", "sciscinet_sources.parquet", None),
        ("fields", "sciscinet_fields.parquet", None),
        ("funders", "funders.parquet", None),
        ("paper_refs", "sciscinet_paperrefs.parquet", None),
        ("paper_fields", "sciscinet_paperfields.parquet", None),
        ("paper_authors", "sciscinet_paper_author_affiliation.parquet", None),
        ("author_papers", "sciscinet_authors_paperid.parquet", None),
        ("paper_sources", "sciscinet_papersources.parquet", None),
        ("papers_pmid_pmcid", "sciscinet_papers_pmid_pmcid.parquet", None),
        ("affl_assoc_affl", "sciscinet_affl_assoc_affl.parquet", None),
        # External links
        ("link_nih", "sciscinet_link_nih.parquet", None),
        ("link_nsf", "sciscinet_link_nsf.parquet", None),
        ("link_patents", "sciscinet_link_patents.parquet", None),
        ("link_twitter", "sciscinet_link_twitter.parquet", None),
        ("link_clinicaltrials", "sciscinet_link_clinicaltrials.parquet", None),
        ("link_newsfeed", "sciscinet_link_newsfeed.parquet", None),
        ("link_nobellaureates", "sciscinet_link_nobellaureates.parquet", None),
        # Metadata
        ("nih_metadata", "sciscinet_nih_metadata.parquet", None),
        ("nsf_metadata", "sciscinet_nsf_metadata.parquet", None),
        ("clinicaltrials_metadata", "sciscinet_clinicaltrials_metadata.parquet", None),
        ("newsfeed_metadata", "sciscinet_newsfeed_metadata.parquet", None),
        ("twitter_metadata", "sciscinet_twitter_metadata.parquet", None),
        # Scientometrics
        ("hit_papers", "hit_papers_level0.parquet", None),
        ("hit_papers_level1", "hit_papers_level1.parquet", None),
        ("normalized_citations", "normalized_citations_level0.parquet", None),
        ("normalized_citations_level1", "normalized_citations_level1.parquet", None),
    ]

    for view_name, filename, sql_override in core_views:
        path = SCISCINET_CORE / filename
        if path.exists():
            if sql_override:
                conn.execute(f"CREATE VIEW sciscinet.{view_name} AS {sql_override}")
            else:
                conn.execute(
                    f"CREATE VIEW sciscinet.{view_name} AS "
                    f"SELECT * FROM read_parquet('{path}')"
                )
            views.append(f"sciscinet.{view_name}")

    # Paper details (large file)
    clean_path = SCISCINET_LARGE / "sciscinet_paperdetails_clean.parquet"
    if clean_path.exists():
        conn.execute(
            f"CREATE VIEW sciscinet.paper_details AS "
            f"SELECT * FROM read_parquet('{clean_path}')"
        )
        views.append("sciscinet.paper_details")

        # Convenience: English papers with valid abstracts
        conn.execute(
            "CREATE VIEW sciscinet.papers_english AS "
            "SELECT * FROM sciscinet.paper_details WHERE valid_title_abstract = true"
        )
        views.append("sciscinet.papers_english")

    # Convenience views
    if "sciscinet.papers" in views:
        conn.execute("""
            CREATE VIEW sciscinet.recent_papers AS
            SELECT * FROM sciscinet.papers WHERE year >= 2020
        """)
        views.append("sciscinet.recent_papers")

    if "sciscinet.papers" in views and "sciscinet.hit_papers" in views:
        conn.execute("""
            CREATE VIEW sciscinet.high_impact_papers AS
            SELECT DISTINCT p.*, h.fieldid, h.Hit_1pct, h.Hit_5pct
            FROM sciscinet.papers p
            INNER JOIN sciscinet.hit_papers h ON p.paperid = h.paperid
            WHERE h.Hit_1pct = 1
        """)
        views.append("sciscinet.high_impact_papers")

    if "sciscinet.paper_refs" in views:
        conn.execute("""
            CREATE VIEW sciscinet.citation_edges AS
            SELECT citing_paperid AS source, cited_paperid AS target, year_diff
            FROM sciscinet.paper_refs
        """)
        views.append("sciscinet.citation_edges")

    if "sciscinet.affiliations" in views:
        conn.execute("""
            CREATE VIEW sciscinet.us_institutions AS
            SELECT * FROM sciscinet.affiliations WHERE country_code = 'US'
        """)
        views.append("sciscinet.us_institutions")

    return views


def create_openalex_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create OpenAlex schema and views (if parquet data exists)."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS openalex")
    views = []

    if not OPENALEX_PARQUET.exists():
        return views

    # Auto-discover all parquet subdirectories
    for table_dir in sorted(OPENALEX_PARQUET.iterdir()):
        if not table_dir.is_dir():
            continue
        if not list(table_dir.glob("*.parquet")):
            continue
        table_name = table_dir.name
        path = table_dir / "*.parquet"
        conn.execute(
            f"CREATE VIEW openalex.{table_name} AS "
            f"SELECT * FROM read_parquet('{path}')"
        )
        views.append(f"openalex.{table_name}")

    return views


def create_pwc_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create Papers With Code schema and views (auto-discover parquet files)."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS pwc")
    views = []

    if not PWC_PARQUET.exists():
        return views

    for pf in sorted(PWC_PARQUET.glob("*.parquet")):
        table_name = pf.stem
        conn.execute(
            f"CREATE VIEW pwc.{table_name} AS "
            f"SELECT * FROM read_parquet('{pf}')"
        )
        views.append(f"pwc.{table_name}")

    # Convenience: pre-formatted OpenAlex join view
    if "pwc.paper_has_openalexWorkID" in views:
        conn.execute("""
            CREATE VIEW pwc.papers_openalex AS
            SELECT paper_id,
                   openalex_work_id AS openalex_work_id_bare,
                   'https://openalex.org/' || openalex_work_id AS openalex_work_id
            FROM pwc.paper_has_openalexWorkID
        """)
        views.append("pwc.papers_openalex")

    return views


def create_retractionwatch_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create Retraction Watch schema and views."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS retwatch")
    views = []

    if not RETWATCH_PARQUET.exists():
        return views

    for pf in sorted(RETWATCH_PARQUET.glob("*.parquet")):
        table_name = pf.stem
        conn.execute(
            f"CREATE VIEW retwatch.{table_name} AS "
            f"SELECT * FROM read_parquet('{pf}')"
        )
        views.append(f"retwatch.{table_name}")

    return views


def create_ros_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create Reliance on Science schema and views."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS ros")
    views = []

    if not ROS_PARQUET.exists():
        return views

    for pf in sorted(ROS_PARQUET.glob("*.parquet")):
        table_name = pf.stem
        conn.execute(
            f"CREATE VIEW ros.{table_name} AS "
            f"SELECT * FROM read_parquet('{pf}')"
        )
        views.append(f"ros.{table_name}")

    return views


def create_p2p_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Create PreprintToPaper schema and views."""
    conn.execute("CREATE SCHEMA IF NOT EXISTS p2p")
    views = []

    if not P2P_PARQUET.exists():
        return views

    for pf in sorted(P2P_PARQUET.glob("*.parquet")):
        table_name = pf.stem
        conn.execute(
            f"CREATE VIEW p2p.{table_name} AS "
            f"SELECT * FROM read_parquet('{pf}')"
        )
        views.append(f"p2p.{table_name}")

    return views


def create_ontology_views(conn: duckdb.DuckDBPyConnection) -> list[str]:
    """Auto-discover ontology parquet dirs, create one schema per ontology."""
    all_views = []
    for name in ONTOLOGY_NAMES:
        parquet_dir = ROOT / "datasets" / name / "parquet"
        if not parquet_dir.exists():
            continue
        parquet_files = sorted(parquet_dir.glob("*.parquet"))
        if not parquet_files:
            continue
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {name}")
        for pf in parquet_files:
            view_name = pf.stem
            conn.execute(
                f"CREATE VIEW {name}.{view_name} AS "
                f"SELECT * FROM read_parquet('{pf}')"
            )
            all_views.append(f"{name}.{view_name}")
    return all_views


def create_xref_views(
    conn: duckdb.DuckDBPyConnection,
    s2ag_views: list[str],
    sciscinet_views: list[str],
    openalex_views: list[str],
    pwc_views: list[str],
    retwatch_views: list[str],
    p2p_views: list[str],
    materialize: bool = False,
) -> list[str]:
    """Create cross-reference schema for DOI-based linking.

    Args:
        materialize: If True, create a materialized doi_index table (slow, scans all papers).
                     If False, only create the doi_map view (fast, lazy evaluation).
    """
    conn.execute("CREATE SCHEMA IF NOT EXISTS xref")
    views = []

    # Build DOI map from available datasets
    union_parts = []

    if "s2ag.papers" in s2ag_views:
        union_parts.append(
            "SELECT 's2ag' AS source, LOWER(externalids.DOI) AS doi, "
            "CAST(corpusid AS VARCHAR) AS source_id "
            "FROM s2ag.papers WHERE externalids.DOI IS NOT NULL"
        )

    if "sciscinet.papers" in sciscinet_views:
        union_parts.append(
            "SELECT 'sciscinet' AS source, "
            "LOWER(REPLACE(doi, 'https://doi.org/', '')) AS doi, "
            "paperid AS source_id "
            "FROM sciscinet.papers WHERE doi IS NOT NULL"
        )

    if "openalex.works" in openalex_views:
        union_parts.append(
            "SELECT 'openalex' AS source, "
            "LOWER(REPLACE(doi, 'https://doi.org/', '')) AS doi, "
            "id AS source_id "
            "FROM openalex.works WHERE doi IS NOT NULL"
        )

    if "pwc.papers_fulltexts" in pwc_views:
        union_parts.append(
            "SELECT 'pwc' AS source, "
            "LOWER(doi) AS doi, "
            "paper_id AS source_id "
            "FROM pwc.papers_fulltexts WHERE doi IS NOT NULL AND doi != ''"
        )

    if "retwatch.retraction_watch" in retwatch_views:
        union_parts.append(
            "SELECT 'retwatch' AS source, "
            "original_paper_doi AS doi, "
            "CAST(record_id AS VARCHAR) AS source_id "
            "FROM retwatch.retraction_watch WHERE original_paper_doi IS NOT NULL"
        )

    if "p2p.preprint_to_paper" in p2p_views:
        union_parts.append(
            "SELECT 'p2p_preprint' AS source, "
            "LOWER(biorxiv_doi) AS doi, "
            "biorxiv_doi AS source_id "
            "FROM p2p.preprint_to_paper WHERE biorxiv_doi IS NOT NULL"
        )
        union_parts.append(
            "SELECT 'p2p_published' AS source, "
            "LOWER(biorxiv_published_doi) AS doi, "
            "biorxiv_doi AS source_id "
            "FROM p2p.preprint_to_paper WHERE biorxiv_published_doi IS NOT NULL"
        )

    if union_parts:
        union_sql = " UNION ALL ".join(union_parts)
        conn.execute(f"CREATE VIEW xref.doi_map AS {union_sql}")
        views.append("xref.doi_map")

        if materialize:
            print("  Building xref.doi_index (this may take several minutes)...")
            conn.execute("CREATE TABLE xref.doi_index AS SELECT * FROM xref.doi_map")
            conn.execute("CREATE INDEX idx_doi ON xref.doi_index(doi)")
            views.append("xref.doi_index")

    # Unified papers (materialized Parquet, created by materialize_unified_papers.py)
    unified_dir = ROOT / "datasets" / "xref" / "unified_papers"
    if unified_dir.exists() and list(unified_dir.glob("*.parquet")):
        unified_path = unified_dir / "*.parquet"
        conn.execute(
            f"CREATE VIEW xref.unified_papers AS "
            f"SELECT * FROM read_parquet('{unified_path}')"
        )
        views.append("xref.unified_papers")

    # Coverage stats (materialized Parquet, created by materialize_unified_papers.py)
    coverage_dir = ROOT / "datasets" / "xref" / "coverage_stats"
    if coverage_dir.exists() and list(coverage_dir.glob("*.parquet")):
        for pf in sorted(coverage_dir.glob("*.parquet")):
            view_name = f"coverage_{pf.stem}"
            conn.execute(
                f"CREATE VIEW xref.{view_name} AS "
                f"SELECT * FROM read_parquet('{pf}')"
            )
            views.append(f"xref.{view_name}")

    # Topic-ontology map (created by build_ontology_linkage.py)
    topic_map_dir = ROOT / "datasets" / "xref" / "topic_ontology_map"
    if topic_map_dir.exists() and list(topic_map_dir.glob("*.parquet")):
        topic_map_path = topic_map_dir / "*.parquet"
        conn.execute(
            f"CREATE VIEW xref.topic_ontology_map AS "
            f"SELECT * FROM read_parquet('{topic_map_path}')"
        )
        views.append("xref.topic_ontology_map")

    # Ontology bridges (created by build_ontology_linkage.py)
    bridges_dir = ROOT / "datasets" / "xref" / "ontology_bridges"
    if bridges_dir.exists() and list(bridges_dir.glob("*.parquet")):
        bridges_path = bridges_dir / "*.parquet"
        conn.execute(
            f"CREATE VIEW xref.ontology_bridges AS "
            f"SELECT * FROM read_parquet('{bridges_path}')"
        )
        views.append("xref.ontology_bridges")

    # Temporal coverage metadata per source
    conn.execute("""
        CREATE VIEW xref.source_temporal_coverage AS
        SELECT * FROM (VALUES
            ('openalex',  1500, 2025, 'works',           'Broadest coverage; trails real-time by weeks to months'),
            ('s2ag',      1900, 2025, 'papers',          'Concentrated in recent decades; CS emphasis'),
            ('sciscinet', 1900, 2022, 'metrics',         'Disruption/atypicality metrics end ~2022'),
            ('pwc',       2012, 2024, 'papers',          'ML papers with code; platform ceased active operations'),
            ('retwatch',  1927, 2024, 'retracted_papers', 'Retraction events; ongoing curation'),
            ('ros',       1947, 2023, 'patent_pairs',    'Patent-to-paper citations; patent processing lag'),
            ('p2p',       2013, 2024, 'preprint_maps',   'bioRxiv/medRxiv preprint-to-published DOI mappings'),
            ('crossref',  NULL, NULL, 'doi_metadata',    'DOI metadata and reference lists; no temporal bound')
        ) AS t(source, year_min, year_max, coverage_type, note)
    """)
    views.append("xref.source_temporal_coverage")

    # Per-paper temporal flags (requires unified_papers)
    unified_dir = ROOT / "datasets" / "xref" / "unified_papers"
    if unified_dir.exists() and list(unified_dir.glob("*.parquet")):
        conn.execute("""
            CREATE VIEW xref.paper_temporal_flags AS
            SELECT
                doi,
                year,
                (year > 2022 AND has_sciscinet) AS sciscinet_metrics_stale,
                (year > 2023 AND has_patent)    AS ros_coverage_incomplete,
                (year IS NULL)                  AS year_missing
            FROM xref.unified_papers
        """)
        views.append("xref.paper_temporal_flags")

    return views


def create_backward_compat_views(
    conn: duckdb.DuckDBPyConnection,
    s2ag_views: list[str],
    sciscinet_views: list[str],
) -> list[str]:
    """Create unschema'd aliases for backward compatibility with existing notebooks."""
    # These are created in the default (main) schema
    aliases = []

    compat_map = {
        # Old name -> new schema.view
        "papers": "s2ag.papers",
        "abstracts": "s2ag.abstracts",
        "authors": "s2ag.authors",
        "citations": "s2ag.citations",
        "paper_ids": "s2ag.paper_ids",
        "publication_venues": "s2ag.publication_venues",
        "tldrs": "s2ag.tldrs",
        "s2orc": "s2ag.s2orc",
    }

    for alias, source in compat_map.items():
        if source in s2ag_views:
            try:
                conn.execute(f"CREATE VIEW {alias} AS SELECT * FROM {source}")
                aliases.append(alias)
            except Exception:
                pass  # skip if conflict

    return aliases


def create_database(materialize: bool = False):
    """Create the unified DuckDB database."""
    print(f"=== Creating Unified DuckDB Database ===")
    print(f"Root: {ROOT}")
    print(f"Database: {DB_PATH}")
    print()

    # Remove existing database
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = duckdb.connect(str(DB_PATH))
    conn.execute("SET threads=16")

    # Create views for each dataset
    print("[S2AG]")
    s2ag_views = create_s2ag_views(conn)
    print(f"  Created {len(s2ag_views)} views")

    print("[SciSciNet]")
    sciscinet_views = create_sciscinet_views(conn)
    print(f"  Created {len(sciscinet_views)} views")

    print("[OpenAlex]")
    openalex_views = create_openalex_views(conn)
    if openalex_views:
        print(f"  Created {len(openalex_views)} views")
    else:
        print("  No parquet data found (download and convert first)")

    print("[Papers With Code]")
    pwc_views = create_pwc_views(conn)
    if pwc_views:
        print(f"  Created {len(pwc_views)} views")
    else:
        print("  No parquet data found")

    print("[Scientific Ontologies]")
    ontology_views = create_ontology_views(conn)
    if ontology_views:
        schemas = set(v.split(".")[0] for v in ontology_views)
        print(f"  Created {len(ontology_views)} views across {len(schemas)} ontologies: {', '.join(sorted(schemas))}")
    else:
        print("  No ontology parquet data found")

    print("[Retraction Watch]")
    retwatch_views = create_retractionwatch_views(conn)
    if retwatch_views:
        print(f"  Created {len(retwatch_views)} views")
    else:
        print("  No parquet data found")

    print("[Reliance on Science]")
    ros_views = create_ros_views(conn)
    if ros_views:
        print(f"  Created {len(ros_views)} views")
    else:
        print("  No parquet data found")

    print("[PreprintToPaper]")
    p2p_views = create_p2p_views(conn)
    if p2p_views:
        print(f"  Created {len(p2p_views)} views")
    else:
        print("  No parquet data found")

    print("[Cross-references]")
    xref_views = create_xref_views(
        conn, s2ag_views, sciscinet_views, openalex_views, pwc_views,
        retwatch_views, p2p_views,
        materialize=materialize,
    )
    print(f"  Created {len(xref_views)} views/tables")

    print("[Backward compatibility]")
    compat_views = create_backward_compat_views(conn, s2ag_views, sciscinet_views)
    print(f"  Created {len(compat_views)} aliases")

    conn.close()

    all_views = [s2ag_views, sciscinet_views, openalex_views, pwc_views,
                 ontology_views, retwatch_views, ros_views, p2p_views,
                 xref_views, compat_views]
    total = sum(len(v) for v in all_views)
    db_size = DB_PATH.stat().st_size / 1024
    print(f"\nTotal: {total} views/tables")
    print(f"Database size: {db_size:.1f} KB")
    print(f"Saved to: {DB_PATH}")


def print_summary():
    """Print summary of views in the database."""
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        print("Run: python scripts/create_unified_db.py")
        return

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    conn.execute("SET threads=16")

    # List all schemas
    schemas = conn.execute(
        "SELECT DISTINCT schema_name FROM information_schema.schemata "
        "WHERE schema_name NOT IN ('information_schema', 'pg_catalog') "
        "ORDER BY schema_name"
    ).fetchall()

    for (schema,) in schemas:
        views = conn.execute(
            f"SELECT table_name FROM information_schema.tables "
            f"WHERE table_schema = '{schema}' ORDER BY table_name"
        ).fetchall()

        if views:
            print(f"\n[{schema}] ({len(views)} views)")
            for (view_name,) in views:
                full_name = f"{schema}.{view_name}" if schema != "main" else view_name
                print(f"  {full_name}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create unified DuckDB database with views across all datasets"
    )
    parser.add_argument("--summary", action="store_true", help="Print current view info")
    parser.add_argument(
        "--materialize-xref", action="store_true",
        help="Materialize xref.doi_index table (slow, scans all papers)"
    )
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return 0

    create_database(materialize=args.materialize_xref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
