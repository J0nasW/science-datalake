#!/usr/bin/env python3
"""
Convert OpenAlex NDJSON.gz snapshot to flattened Parquet tables.

Optimized for a 24-core / 251GB RAM workstation with external NVME.

Key design decisions:
- DuckDB does all heavy lifting (JSON parsing, flattening, compression)
- Works: each .gz file is materialized into a TEMP TABLE once, then 10 sub-tables
  are extracted from that single in-memory copy (avoids re-parsing JSON 10x)
- Abstract conversion is done inline during works processing via Python UDF
  registered in each worker process, not as a serial post-processing step
- Thread/memory budgets scale to available hardware
- Checkpoint/resume per file for crash recovery
- Incremental updates: works files are tracked by size+mtime so re-downloaded
  partitions are detected even if the checkpoint says "completed"
- Compaction merges per-file shards into single optimized Parquet files

Incremental workflow:
    python scripts/download_openalex.py --all       # s3 sync: only new/changed files
    python scripts/convert_openalex.py --all        # convert only new/changed (checkpoint-aware)
    python scripts/convert_openalex.py --compact    # merge shards -> single optimized files
    python scripts/create_unified_db.py             # recreate DuckDB views

Usage:
    python scripts/convert_openalex.py --all                    # Everything
    python scripts/convert_openalex.py --entity works --workers 6
    python scripts/convert_openalex.py --entity topics          # Single entity
    python scripts/convert_openalex.py --status                 # Check progress
    python scripts/convert_openalex.py --entity works --sample 3  # Test run
    python scripts/convert_openalex.py --compact                # Merge shards into single files
"""

import argparse
import gzip
import json
import logging
import os
import re
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb

# ── Resolve data lake root ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

SNAPSHOT_DIR = ROOT / "datasets" / "openalex" / "snapshot" / "data"
PARQUET_DIR = ROOT / "datasets" / "openalex" / "parquet"
CHECKPOINT_FILE = ROOT / "datasets" / "openalex" / ".conversion_checkpoint.json"
LOG_FILE = ROOT / "datasets" / "openalex" / "conversion.log"

# ── Hardware detection ──────────────────────────────────────────────────────

TOTAL_CORES = os.cpu_count() or 8
TOTAL_RAM_GB = 32  # conservative default
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                TOTAL_RAM_GB = int(line.split()[1]) // (1024 * 1024)
                break
except Exception:
    pass

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
log = logging.getLogger(__name__)

# ── Abstract processing (ported from convert_abstracts.py) ──────────────────

MIN_TITLE_LENGTH = 10
MIN_ABSTRACT_LENGTH = 50
MIN_ASCII_RATIO = 0.80
MIN_WORD_COUNT = 10
MAX_AVG_WORD_LENGTH = 25
PRINTABLE_ASCII = set(string.printable)


def inverted_index_to_text(inv_idx) -> str:
    """Convert OpenAlex abstract_inverted_index to plain text."""
    if not inv_idx:
        return ""
    try:
        if isinstance(inv_idx, str):
            inv_idx = json.loads(inv_idx)
        if not isinstance(inv_idx, dict):
            return ""
        words = []
        for word, positions in inv_idx.items():
            if isinstance(positions, list):
                for pos in positions:
                    words.append((pos, word))
        words.sort(key=lambda x: x[0])
        return " ".join(word for _, word in words)
    except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
        return ""


def is_readable_text(text: str) -> bool:
    """Check if text is readable English-like content."""
    if not text:
        return False
    ascii_count = sum(1 for c in text if c in PRINTABLE_ASCII)
    if ascii_count / len(text) < MIN_ASCII_RATIO:
        return False
    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False
    total_word_len = sum(len(w) for w in words)
    if total_word_len / len(words) > MAX_AVG_WORD_LENGTH:
        return False
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / len(text) < 0.5:
        return False
    return True


def is_valid_title_abstract(language, title, abstract) -> bool:
    """Check if record meets validity criteria for NLP use."""
    if language != "en":
        return False
    if not title or len(str(title).strip()) < MIN_TITLE_LENGTH:
        return False
    if not abstract or len(str(abstract).strip()) < MIN_ABSTRACT_LENGTH:
        return False
    return is_readable_text(str(abstract))


# ── Entity SQL configurations ──────────────────────────────────────────────
#
# Each entity defines the DuckDB SQL to extract columns.
# Validated against the actual 2026-02-03 snapshot schemas.

SIMPLE_ENTITY_CONFIGS = {
    "domains": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(original_id AS BIGINT) AS original_id,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(description AS VARCHAR) AS description,
                TRY_CAST(display_name_alternatives AS VARCHAR[]) AS display_name_alternatives,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(ids.wikidata AS VARCHAR) AS wikidata_id,
                TRY_CAST(ids.wikipedia AS VARCHAR) AS wikipedia_url,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 10000,
    },
    "fields": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(original_id AS BIGINT) AS original_id,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(description AS VARCHAR) AS description,
                TRY_CAST(display_name_alternatives AS VARCHAR[]) AS display_name_alternatives,
                TRY_CAST(domain.id AS VARCHAR) AS domain_id,
                TRY_CAST(domain.display_name AS VARCHAR) AS domain_display_name,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(ids.wikidata AS VARCHAR) AS wikidata_id,
                TRY_CAST(ids.wikipedia AS VARCHAR) AS wikipedia_url,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 10000,
    },
    "subfields": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(original_id AS BIGINT) AS original_id,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(description AS VARCHAR) AS description,
                TRY_CAST(display_name_alternatives AS VARCHAR[]) AS display_name_alternatives,
                TRY_CAST(field.id AS VARCHAR) AS field_id,
                TRY_CAST(field.display_name AS VARCHAR) AS field_display_name,
                TRY_CAST(domain.id AS VARCHAR) AS domain_id,
                TRY_CAST(domain.display_name AS VARCHAR) AS domain_display_name,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(ids.wikidata AS VARCHAR) AS wikidata_id,
                TRY_CAST(ids.wikipedia AS VARCHAR) AS wikipedia_url,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 10000,
    },
    "topics": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(original_id AS BIGINT) AS original_id,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(description AS VARCHAR) AS description,
                TRY_CAST(keywords AS VARCHAR[]) AS keywords,
                TRY_CAST(subfield.id AS VARCHAR) AS subfield_id,
                TRY_CAST(subfield.display_name AS VARCHAR) AS subfield_display_name,
                TRY_CAST(field.id AS VARCHAR) AS field_id,
                TRY_CAST(field.display_name AS VARCHAR) AS field_display_name,
                TRY_CAST(domain.id AS VARCHAR) AS domain_id,
                TRY_CAST(domain.display_name AS VARCHAR) AS domain_display_name,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(ids.wikipedia AS VARCHAR) AS wikipedia_url,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 10000,
    },
    "publishers": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(alternate_titles AS VARCHAR[]) AS alternate_titles,
                TRY_CAST(country_codes AS VARCHAR[]) AS country_codes,
                TRY_CAST(lineage AS VARCHAR[]) AS lineage,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(ids.ror AS VARCHAR) AS ror_id,
                TRY_CAST(ids.wikidata AS VARCHAR) AS wikidata_id,
                TRY_CAST(ror_id AS VARCHAR) AS ror_id_top,
                TRY_CAST(wikidata_id AS VARCHAR) AS wikidata_id_top,
                TRY_CAST(homepage_url AS VARCHAR) AS homepage_url,
                TRY_CAST(image_url AS VARCHAR) AS image_url,
                TRY_CAST(image_thumbnail_url AS VARCHAR) AS image_thumbnail_url,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(summary_stats."2yr_mean_citedness" AS DOUBLE) AS mean_citedness_2yr,
                TRY_CAST(summary_stats.h_index AS BIGINT) AS h_index,
                TRY_CAST(summary_stats.i10_index AS BIGINT) AS i10_index,
                TRY_CAST(sources_api_url AS VARCHAR) AS sources_api_url,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 10000,
    },
    "funders": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(alternate_titles AS VARCHAR[]) AS alternate_titles,
                TRY_CAST(country_code AS VARCHAR) AS country_code,
                TRY_CAST(description AS VARCHAR) AS description,
                TRY_CAST(homepage_url AS VARCHAR) AS homepage_url,
                TRY_CAST(image_url AS VARCHAR) AS image_url,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(ids.ror AS VARCHAR) AS ror_id,
                TRY_CAST(ids.wikidata AS VARCHAR) AS wikidata_id,
                TRY_CAST(CAST(ids.crossref AS VARCHAR) AS VARCHAR) AS crossref_id,
                TRY_CAST(ids.doi AS VARCHAR) AS doi,
                TRY_CAST(awards_count AS BIGINT) AS awards_count,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(summary_stats."2yr_mean_citedness" AS DOUBLE) AS mean_citedness_2yr,
                TRY_CAST(summary_stats.h_index AS BIGINT) AS h_index,
                TRY_CAST(summary_stats.i10_index AS BIGINT) AS i10_index,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 10000,
    },
    "sources": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(CAST(host_organization AS VARCHAR) AS VARCHAR) AS host_organization,
                TRY_CAST(host_organization_name AS VARCHAR) AS host_organization_name,
                TRY_CAST(type AS VARCHAR) AS type,
                TRY_CAST(issn_l AS VARCHAR) AS issn_l,
                TRY_CAST(issn AS VARCHAR[]) AS issn,
                TRY_CAST(is_oa AS BOOLEAN) AS is_oa,
                TRY_CAST(is_in_doaj AS BOOLEAN) AS is_in_doaj,
                TRY_CAST(is_core AS BOOLEAN) AS is_core,
                TRY_CAST(CAST(homepage_url AS VARCHAR) AS VARCHAR) AS homepage_url,
                TRY_CAST(CAST(country_code AS VARCHAR) AS VARCHAR) AS country_code,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(ids.issn_l AS VARCHAR) AS ids_issn_l,
                TRY_CAST(CAST(ids.mag AS VARCHAR) AS VARCHAR) AS mag_id,
                TRY_CAST(CAST(ids.wikidata AS VARCHAR) AS VARCHAR) AS wikidata_id,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(oa_works_count AS BIGINT) AS oa_works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(summary_stats."2yr_mean_citedness" AS DOUBLE) AS mean_citedness_2yr,
                TRY_CAST(summary_stats.h_index AS BIGINT) AS h_index,
                TRY_CAST(summary_stats.i10_index AS BIGINT) AS i10_index,
                TRY_CAST(first_publication_year AS BIGINT) AS first_publication_year,
                TRY_CAST(last_publication_year AS BIGINT) AS last_publication_year,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 50000,
    },
    "institutions": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(ror AS VARCHAR) AS ror,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(country_code AS VARCHAR) AS country_code,
                TRY_CAST(type AS VARCHAR) AS type,
                TRY_CAST(type_id AS VARCHAR) AS type_id,
                TRY_CAST(is_super_system AS BOOLEAN) AS is_super_system,
                TRY_CAST(homepage_url AS VARCHAR) AS homepage_url,
                TRY_CAST(image_url AS VARCHAR) AS image_url,
                TRY_CAST(display_name_acronyms AS VARCHAR[]) AS display_name_acronyms,
                TRY_CAST(display_name_alternatives AS VARCHAR[]) AS display_name_alternatives,
                TRY_CAST(lineage AS VARCHAR[]) AS lineage,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(ids.ror AS VARCHAR) AS ror_id,
                TRY_CAST(ids.wikidata AS VARCHAR) AS wikidata_id,
                TRY_CAST(ids.wikipedia AS VARCHAR) AS wikipedia_url,
                TRY_CAST(geo.city AS VARCHAR) AS geo_city,
                TRY_CAST(geo.region AS VARCHAR) AS geo_region,
                TRY_CAST(geo.country AS VARCHAR) AS geo_country,
                TRY_CAST(geo.country_code AS VARCHAR) AS geo_country_code,
                TRY_CAST(geo.latitude AS DOUBLE) AS geo_latitude,
                TRY_CAST(geo.longitude AS DOUBLE) AS geo_longitude,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(summary_stats."2yr_mean_citedness" AS DOUBLE) AS mean_citedness_2yr,
                TRY_CAST(summary_stats.h_index AS BIGINT) AS h_index,
                TRY_CAST(summary_stats.i10_index AS BIGINT) AS i10_index,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 50000,
    },
    "concepts": {
        "sql": """
            SELECT
                TRY_CAST(id AS VARCHAR) AS id,
                TRY_CAST(wikidata AS VARCHAR) AS wikidata,
                TRY_CAST(display_name AS VARCHAR) AS display_name,
                TRY_CAST(level AS BIGINT) AS level,
                TRY_CAST(description AS VARCHAR) AS description,
                TRY_CAST(image_url AS VARCHAR) AS image_url,
                TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
                TRY_CAST(CAST(ids.mag AS VARCHAR) AS VARCHAR) AS mag_id,
                TRY_CAST(ids.wikipedia AS VARCHAR) AS wikipedia_url,
                TRY_CAST(ids.wikidata AS VARCHAR) AS wikidata_id,
                TRY_CAST(works_count AS BIGINT) AS works_count,
                TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
                TRY_CAST(updated_date AS VARCHAR) AS updated_date,
                TRY_CAST(created_date AS VARCHAR) AS created_date
            FROM source_data
        """,
        "row_group_size": 10000,
    },
}

# ── Authors: main + sub-tables in one pass ──────────────────────────────────

AUTHORS_MAIN_SQL = """
    SELECT
        TRY_CAST(id AS VARCHAR) AS id,
        TRY_CAST(orcid AS VARCHAR) AS orcid,
        TRY_CAST(display_name AS VARCHAR) AS display_name,
        TRY_CAST(display_name_alternatives AS VARCHAR[]) AS display_name_alternatives,
        TRY_CAST(works_count AS BIGINT) AS works_count,
        TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
        TRY_CAST(summary_stats."2yr_mean_citedness" AS DOUBLE) AS mean_citedness_2yr,
        TRY_CAST(summary_stats.h_index AS BIGINT) AS h_index,
        TRY_CAST(summary_stats.i10_index AS BIGINT) AS i10_index,
        TRY_CAST(last_known_institutions AS JSON) AS last_known_institutions,
        TRY_CAST(ids.openalex AS VARCHAR) AS openalex_id,
        TRY_CAST(ids.orcid AS VARCHAR) AS orcid_url,
        TRY_CAST(updated_date AS VARCHAR) AS updated_date,
        TRY_CAST(created_date AS VARCHAR) AS created_date
    FROM source_data
"""

AUTHORS_SUBTABLE_SQLS = {
    "authors_ids": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS author_id,
            TRY_CAST(s.ids.openalex AS VARCHAR) AS openalex,
            TRY_CAST(s.ids.orcid AS VARCHAR) AS orcid,
            TRY_CAST(s.orcid AS VARCHAR) AS orcid_top
        FROM source_data s
        WHERE s.ids IS NOT NULL
    """,
    "authors_counts_by_year": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS author_id,
            TRY_CAST(c.year AS INTEGER) AS year,
            TRY_CAST(c.works_count AS BIGINT) AS works_count,
            TRY_CAST(c.oa_works_count AS BIGINT) AS oa_works_count,
            TRY_CAST(c.cited_by_count AS BIGINT) AS cited_by_count
        FROM source_data s, LATERAL UNNEST(s.counts_by_year) AS t(c)
        WHERE s.counts_by_year IS NOT NULL
    """,
}

# ── Works: main + sub-tables ────────────────────────────────────────────────

WORKS_MAIN_SQL = """
    SELECT
        TRY_CAST(id AS VARCHAR) AS id,
        TRY_CAST(doi AS VARCHAR) AS doi,
        TRY_CAST(title AS VARCHAR) AS title,
        TRY_CAST(display_name AS VARCHAR) AS display_name,
        TRY_CAST(publication_year AS INTEGER) AS publication_year,
        TRY_CAST(publication_date AS VARCHAR) AS publication_date,
        TRY_CAST(language AS VARCHAR) AS language,
        TRY_CAST(type AS VARCHAR) AS type,
        TRY_CAST(cited_by_count AS BIGINT) AS cited_by_count,
        TRY_CAST(fwci AS DOUBLE) AS fwci,
        TRY_CAST(is_retracted AS BOOLEAN) AS is_retracted,
        TRY_CAST(is_paratext AS BOOLEAN) AS is_paratext,
        TRY_CAST(has_fulltext AS BOOLEAN) AS has_fulltext,
        TRY_CAST(abstract AS VARCHAR) AS abstract,
        TRY_CAST(referenced_works_count AS BIGINT) AS referenced_works_count,
        TRY_CAST(updated_date AS VARCHAR) AS updated_date,
        TRY_CAST(created_date AS VARCHAR) AS created_date
    FROM source_data
"""

WORKS_SUBTABLE_SQLS = {
    "works_authorships": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(a.author.id AS VARCHAR) AS author_id,
            TRY_CAST(a.author.display_name AS VARCHAR) AS author_display_name,
            TRY_CAST(a.author.orcid AS VARCHAR) AS author_orcid,
            TRY_CAST(a.raw_author_name AS VARCHAR) AS raw_author_name,
            TRY_CAST(a.is_corresponding AS BOOLEAN) AS is_corresponding,
            TRY_CAST(a.raw_affiliation_strings AS VARCHAR[]) AS raw_affiliation_strings,
            TRY_CAST(a.institutions AS JSON) AS institutions,
            TRY_CAST(a.countries AS VARCHAR[]) AS countries
        FROM source_data s, LATERAL UNNEST(s.authorships) AS t(a)
        WHERE s.authorships IS NOT NULL
    """,
    "works_topics": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(tp.id AS VARCHAR) AS topic_id,
            TRY_CAST(tp.display_name AS VARCHAR) AS topic_display_name,
            TRY_CAST(tp.score AS DOUBLE) AS score
        FROM source_data s, LATERAL UNNEST(s.topics) AS t(tp)
        WHERE s.topics IS NOT NULL
    """,
    "works_referenced_works": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(rw AS VARCHAR) AS referenced_work_id
        FROM source_data s, LATERAL UNNEST(s.referenced_works) AS t(rw)
        WHERE s.referenced_works IS NOT NULL
    """,
    "works_locations": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(loc.source.id AS VARCHAR) AS source_id,
            TRY_CAST(loc.source.display_name AS VARCHAR) AS source_display_name,
            TRY_CAST(loc.pdf_url AS VARCHAR) AS pdf_url,
            TRY_CAST(loc.landing_page_url AS VARCHAR) AS landing_page_url,
            TRY_CAST(loc.is_oa AS BOOLEAN) AS is_oa,
            TRY_CAST(loc.license AS VARCHAR) AS license,
            TRY_CAST(loc.version AS VARCHAR) AS version
        FROM source_data s, LATERAL UNNEST(s.locations) AS t(loc)
        WHERE s.locations IS NOT NULL
    """,
    "works_ids": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            json_extract_string(CAST(s.ids AS JSON), '$.openalex') AS openalex,
            json_extract_string(CAST(s.ids AS JSON), '$.doi') AS doi,
            json_extract_string(CAST(s.ids AS JSON), '$.pmid') AS pmid,
            json_extract_string(CAST(s.ids AS JSON), '$.mag') AS mag
        FROM source_data s
        WHERE s.ids IS NOT NULL
    """,
    "works_open_access": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(s.open_access.is_oa AS BOOLEAN) AS is_oa,
            TRY_CAST(s.open_access.oa_status AS VARCHAR) AS oa_status,
            TRY_CAST(s.open_access.oa_url AS VARCHAR) AS oa_url,
            TRY_CAST(s.open_access.any_repository_has_fulltext AS BOOLEAN) AS any_repository_has_fulltext
        FROM source_data s
        WHERE s.open_access IS NOT NULL
    """,
    "works_biblio": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(s.biblio.volume AS VARCHAR) AS volume,
            TRY_CAST(s.biblio.issue AS VARCHAR) AS issue,
            TRY_CAST(s.biblio.first_page AS VARCHAR) AS first_page,
            TRY_CAST(s.biblio.last_page AS VARCHAR) AS last_page
        FROM source_data s
        WHERE s.biblio IS NOT NULL
    """,
    "works_counts_by_year": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(c.year AS INTEGER) AS year,
            TRY_CAST(c.cited_by_count AS BIGINT) AS cited_by_count
        FROM source_data s, LATERAL UNNEST(s.counts_by_year) AS t(c)
        WHERE s.counts_by_year IS NOT NULL
    """,
    "works_best_oa_location": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(s.best_oa_location.source.id AS VARCHAR) AS source_id,
            TRY_CAST(s.best_oa_location.pdf_url AS VARCHAR) AS pdf_url,
            TRY_CAST(s.best_oa_location.landing_page_url AS VARCHAR) AS landing_page_url,
            TRY_CAST(s.best_oa_location.is_oa AS BOOLEAN) AS is_oa,
            TRY_CAST(s.best_oa_location.license AS VARCHAR) AS license,
            TRY_CAST(s.best_oa_location.version AS VARCHAR) AS version
        FROM source_data s
        WHERE s.best_oa_location IS NOT NULL
    """,
    "works_concepts": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(c.id AS VARCHAR) AS concept_id,
            TRY_CAST(c.display_name AS VARCHAR) AS display_name,
            TRY_CAST(c.level AS INTEGER) AS level,
            TRY_CAST(c.score AS DOUBLE) AS score
        FROM source_data s, LATERAL UNNEST(s.concepts) AS t(c)
        WHERE s.concepts IS NOT NULL
    """,
    "works_keywords": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(kw.id AS VARCHAR) AS keyword_id,
            TRY_CAST(kw.display_name AS VARCHAR) AS display_name,
            TRY_CAST(kw.score AS DOUBLE) AS score
        FROM source_data s, LATERAL UNNEST(s.keywords) AS t(kw)
        WHERE s.keywords IS NOT NULL
    """,
    "works_related_works": """
        SELECT
            TRY_CAST(s.id AS VARCHAR) AS work_id,
            TRY_CAST(t.rw AS VARCHAR) AS related_work_id
        FROM source_data s, LATERAL UNNEST(s.related_works) AS t(rw)
        WHERE s.related_works IS NOT NULL
    """,
}


def _extract_sql_source_columns(sql: str) -> set[str]:
    """Extract top-level source column names from a SQL SELECT statement.

    Parses patterns like TRY_CAST(colname AS ...) to find simple column
    references. Ignores struct access (s.foo.bar) and nested expressions.
    This allows dynamic detection of which columns the SQL expects.
    """
    # Match TRY_CAST(colname AS ...) where colname is a bare identifier (no dots)
    return set(re.findall(r'TRY_CAST\((\w+)\s+AS\s', sql, re.IGNORECASE))


def ensure_source_columns(conn, sql: str):
    """Add missing columns to source_data that the SQL references.

    OpenAlex schema evolves across partitions — older partitions lack columns
    added later (e.g. abstract, doi, fwci). This dynamically detects which
    columns the SQL needs but source_data lacks, and adds them as NULL.
    Works for any future schema changes without hardcoded column lists.
    """
    expected = _extract_sql_source_columns(sql)
    if not expected:
        return
    actual = {row[0] for row in conn.execute("DESCRIBE source_data").fetchall()}
    for col in expected - actual:
        conn.execute(f'ALTER TABLE source_data ADD COLUMN "{col}" VARCHAR')


def works_file_key(gz_path) -> str:
    """Compute a unique checkpoint key for a works .gz file.

    All works partition files are named part_0000.gz, so we include the
    partition directory to make the key unique, e.g. 'updated_date=2016-06-24/part_0000.gz'.
    """
    return f"{gz_path.parent.name}/{gz_path.name}"


# ── Checkpoint management ──────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_files": {}}


def save_checkpoint(checkpoint: dict):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def is_file_completed(checkpoint: dict, entity: str, filename: str) -> bool:
    return f"{entity}/{filename}" in checkpoint.get("completed_files", {})


def mark_file_completed(checkpoint: dict, entity: str, filename: str, info: dict):
    key = f"{entity}/{filename}"
    checkpoint.setdefault("completed_files", {})[key] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **info,
    }
    save_checkpoint(checkpoint)


def is_works_file_changed(checkpoint: dict, gz_path: Path) -> bool:
    """Check if a works .gz file has changed since it was last converted.

    Compares current st_size and st_mtime against checkpoint values.
    Returns True (changed) if:
    - File is not in checkpoint at all
    - Checkpoint entry lacks file_size/file_mtime (old format)
    - Current file stats don't match checkpoint values
    """
    key = f"works/{works_file_key(gz_path)}"
    entry = checkpoint.get("completed_files", {}).get(key)
    if not entry:
        return True
    if "file_size" not in entry or "file_mtime" not in entry:
        return True  # old checkpoint format, treat as changed
    stat = gz_path.stat()
    return stat.st_size != entry["file_size"] or stat.st_mtime != entry["file_mtime"]


# ── Meta.json auto-update ──────────────────────────────────────────────────

META_FILE = ROOT / "datasets" / "openalex" / "meta.json"


def update_meta_json():
    """Update meta.json with actual row counts, sizes, and status from parquet files.

    Scans all parquet output directories and updates only the machine-computable
    fields (row_count, size_gb, file_count, status, last_updated), preserving
    all hand-written descriptions, column schemas, and relationships.
    """
    if not META_FILE.exists():
        log.warning("meta.json not found, skipping update")
        return

    with open(META_FILE) as f:
        meta = json.load(f)

    tables = meta.setdefault("tables", {})
    updated_any = False

    # Auto-discover tables on disk that aren't yet in meta.json
    if PARQUET_DIR.exists():
        for table_dir in sorted(PARQUET_DIR.iterdir()):
            if table_dir.is_dir() and table_dir.name not in tables:
                tables[table_dir.name] = {
                    "description": f"Auto-discovered table: {table_dir.name}",
                    "status": "pending_conversion",
                }

    for table_name, table_info in tables.items():
        table_dir = PARQUET_DIR / table_name
        if not table_dir.exists():
            continue

        parquet_files = list(table_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        # Compute actual stats from parquet files
        total_size = sum(f.stat().st_size for f in parquet_files)
        size_gb = round(total_size / (1024**3), 3)

        # Get row count from parquet metadata (fast, no full scan)
        try:
            import pyarrow.parquet as pq
            total_rows = sum(pq.read_metadata(str(f)).num_rows for f in parquet_files)
        except Exception:
            total_rows = None

        if total_rows is not None and total_rows > 0:
            old_count = table_info.get("row_count")
            table_info["row_count"] = total_rows
            table_info["size_gb"] = size_gb
            table_info["status"] = "complete"
            table_info["file_count"] = len(parquet_files)
            if len(parquet_files) == 1 and parquet_files[0].name == f"{table_name}.parquet":
                table_info["compacted"] = True
            else:
                table_info.pop("compacted", None)

            if old_count != total_rows:
                updated_any = True

    if updated_any:
        meta["last_updated"] = time.strftime("%Y-%m-%d")

        # Update total_size_gb
        total = 0
        for table_info in tables.values():
            if isinstance(table_info.get("size_gb"), (int, float)):
                total += table_info["size_gb"]
        if total > 0:
            meta["total_size_gb"] = round(total, 2)

        with open(META_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        log.info(f"meta.json updated (total: {meta['total_size_gb']} GB)")
    else:
        log.info("meta.json: no changes needed")


# ── Helper: compute thread/memory budget per DuckDB process ────────────────

def duckdb_settings_for_workers(n_workers: int) -> tuple[int, str]:
    """Return (threads, memory_limit) for each DuckDB subprocess."""
    threads_per = max(2, TOTAL_CORES // n_workers)
    mem_per_gb = max(4, (TOTAL_RAM_GB * 3 // 4) // max(n_workers, 1))
    return threads_per, f"{mem_per_gb}GB"


# ── Simple entity conversion ──────────────────────────────────────────────

def convert_simple_entity(entity: str, force: bool = False, sample: int = 0) -> dict:
    """Convert a non-works, non-authors entity to Parquet."""
    config = SIMPLE_ENTITY_CONFIGS[entity]
    input_dir = SNAPSHOT_DIR / entity
    output_dir = PARQUET_DIR / entity
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        log.warning(f"{entity}: snapshot dir not found at {input_dir}")
        return {"entity": entity, "success": False, "error": "not_found"}

    gz_files = sorted(input_dir.rglob("*.gz"))
    if sample > 0:
        gz_files = gz_files[:sample]
    if not gz_files:
        log.warning(f"{entity}: no .gz files found")
        return {"entity": entity, "success": False, "error": "no_files"}

    checkpoint = load_checkpoint()
    output_file = output_dir / f"{entity}.parquet"

    # Compute input signature to detect new/changed files
    input_signature = f"{len(gz_files)}:{sum(f.stat().st_size for f in gz_files)}"
    checkpoint_info = checkpoint.get("completed_files", {}).get(f"{entity}/all", {})

    if not force and output_file.exists() and checkpoint_info:
        if checkpoint_info.get("input_signature") == input_signature:
            log.info(f"{entity}: up to date ({len(gz_files)} files, signature matches)")
            return {"entity": entity, "success": True, "skipped": True}
        else:
            log.info(f"{entity}: input changed (was {checkpoint_info.get('input_signature')}, "
                     f"now {input_signature}), reconverting...")

    log.info(f"{entity}: converting {len(gz_files)} files")

    # Use all cores for single-entity conversion
    threads, mem = duckdb_settings_for_workers(1)
    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={threads}")
    conn.execute(f"SET memory_limit='{mem}'")

    try:
        file_list_sql = ", ".join(f"'{f}'" for f in gz_files)
        conn.execute(f"""
            CREATE VIEW source_data AS
            SELECT * FROM read_json_auto(
                [{file_list_sql}],
                format='newline_delimited',
                compression='gzip',
                maximum_object_size=104857600,
                ignore_errors=true,
                union_by_name=true
            )
        """)

        conn.execute(f"""
            COPY ({config['sql']})
            TO '{output_file}' (
                FORMAT PARQUET, COMPRESSION 'zstd',
                COMPRESSION_LEVEL 3, ROW_GROUP_SIZE {config['row_group_size']}
            )
        """)

        row_count = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_file}')"
        ).fetchone()[0]
        file_size = output_file.stat().st_size / (1024**3)

        log.info(f"{entity}: {row_count:,} rows, {file_size:.3f} GB")
        mark_file_completed(checkpoint, entity, "all", {
            "row_count": row_count, "size_gb": round(file_size, 3),
            "input_signature": input_signature,
        })
        return {"entity": entity, "success": True, "row_count": row_count, "size_gb": file_size}

    except Exception as e:
        log.error(f"{entity}: conversion failed: {e}")
        return {"entity": entity, "success": False, "error": str(e)}
    finally:
        conn.close()


# ── Generic entity conversion (auto-discover schema) ──────────────────────

def convert_generic_entity(entity: str, force: bool = False, sample: int = 0) -> dict:
    """Convert any entity to Parquet using SELECT * (auto-discovered schema).

    Used as a fallback for entities not in SIMPLE_ENTITY_CONFIGS. Preserves
    the full schema from the JSON without any column filtering.
    """
    input_dir = SNAPSHOT_DIR / entity
    output_dir = PARQUET_DIR / entity
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        log.warning(f"{entity}: snapshot dir not found at {input_dir}")
        return {"entity": entity, "success": False, "error": "not_found"}

    gz_files = sorted(input_dir.rglob("*.gz"))
    if sample > 0:
        gz_files = gz_files[:sample]
    if not gz_files:
        log.warning(f"{entity}: no .gz files found")
        return {"entity": entity, "success": False, "error": "no_files"}

    checkpoint = load_checkpoint()
    output_file = output_dir / f"{entity}.parquet"

    # Compute input signature to detect new/changed files
    input_signature = f"{len(gz_files)}:{sum(f.stat().st_size for f in gz_files)}"
    checkpoint_info = checkpoint.get("completed_files", {}).get(f"{entity}/all", {})

    if not force and output_file.exists() and checkpoint_info:
        if checkpoint_info.get("input_signature") == input_signature:
            log.info(f"{entity}: up to date ({len(gz_files)} files, signature matches)")
            return {"entity": entity, "success": True, "skipped": True}
        else:
            log.info(f"{entity}: input changed (was {checkpoint_info.get('input_signature')}, "
                     f"now {input_signature}), reconverting...")

    log.info(f"{entity}: converting {len(gz_files)} files (generic SELECT *)")

    threads, mem = duckdb_settings_for_workers(1)
    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={threads}")
    conn.execute(f"SET memory_limit='{mem}'")

    try:
        file_list_sql = ", ".join(f"'{f}'" for f in gz_files)
        conn.execute(f"""
            COPY (
                SELECT * FROM read_json_auto(
                    [{file_list_sql}],
                    format='newline_delimited',
                    compression='gzip',
                    maximum_object_size=104857600,
                    ignore_errors=true,
                    union_by_name=true
                )
            )
            TO '{output_file}' (
                FORMAT PARQUET, COMPRESSION 'zstd',
                COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 50000
            )
        """)

        row_count = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_file}')"
        ).fetchone()[0]
        file_size = output_file.stat().st_size / (1024**3)

        log.info(f"{entity}: {row_count:,} rows, {file_size:.3f} GB")
        mark_file_completed(checkpoint, entity, "all", {
            "row_count": row_count, "size_gb": round(file_size, 3),
            "input_signature": input_signature,
        })
        return {"entity": entity, "success": True, "row_count": row_count, "size_gb": file_size}

    except Exception as e:
        log.error(f"{entity}: conversion failed: {e}")
        return {"entity": entity, "success": False, "error": str(e)}
    finally:
        conn.close()


def discover_downloaded_entities() -> list[str]:
    """Discover all entities that have been downloaded (have .gz files)."""
    if not SNAPSHOT_DIR.exists():
        return []
    data_dir = SNAPSHOT_DIR / "data" if (SNAPSHOT_DIR / "data").exists() else SNAPSHOT_DIR
    entities = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and list(d.rglob("*.gz")):
            entities.append(d.name)
    return entities


# ── Authors conversion (main + sub-tables in one scan) ─────────────────────

def convert_authors(force: bool = False, sample: int = 0) -> dict:
    """Convert authors with sub-tables using chunked processing.

    Processes input files in chunks to stay within memory limits.
    Each chunk materializes a TEMP TABLE, extracts all sub-tables,
    then frees memory before the next chunk. Output is multiple
    parquet files per sub-table (chunk_0000.parquet, etc.) which
    DuckDB reads via *.parquet glob.
    """
    input_dir = SNAPSHOT_DIR / "authors"
    if not input_dir.exists():
        log.warning("authors: snapshot dir not found")
        return {"entity": "authors", "success": False, "error": "not_found"}

    gz_files = sorted(input_dir.rglob("*.gz"))
    if sample > 0:
        gz_files = gz_files[:sample]
    if not gz_files:
        return {"entity": "authors", "success": False, "error": "no_files"}

    checkpoint = load_checkpoint()

    # Compute input signature to detect new/changed files
    input_signature = f"{len(gz_files)}:{sum(f.stat().st_size for f in gz_files)}"
    author_tables = ["authors", "authors_ids", "authors_counts_by_year"]
    all_done = all(
        is_file_completed(checkpoint, t, "all")
        for t in author_tables
    )
    # Check if input signature matches for the main authors table
    authors_checkpoint = checkpoint.get("completed_files", {}).get("authors/all", {})
    signature_matches = authors_checkpoint.get("input_signature") == input_signature

    if not force and all_done:
        if signature_matches:
            log.info("authors: all tables up to date (signature matches)")
            return {"entity": "authors", "success": True, "skipped": True}
        else:
            log.info(f"authors: input changed (was {authors_checkpoint.get('input_signature')}, "
                     f"now {input_signature}), reconverting...")

    # Determine which sub-tables still need conversion
    tables_to_convert = {}
    all_table_configs = {
        "authors": (AUTHORS_MAIN_SQL, "authors", 100000),
    }
    all_table_configs.update({
        name: (sql, name, 100000)
        for name, sql in AUTHORS_SUBTABLE_SQLS.items()
    })

    for table_name, config in all_table_configs.items():
        if not force and signature_matches and is_file_completed(checkpoint, table_name, "all"):
            log.info(f"  {table_name}: already done, skipping")
        else:
            tables_to_convert[table_name] = config

    if not tables_to_convert:
        log.info("authors: all sub-tables up to date")
        return {"entity": "authors", "success": True, "skipped": True}

    # Calculate chunk size based on available RAM
    # Rule of thumb: each .gz file expands ~300x in DuckDB memory with nested structs
    # 413 files, 62GB compressed → ~180GB in memory = ~435MB per file in memory
    # Target: use at most 40% of RAM per chunk to leave headroom
    max_chunk_mem_gb = TOTAL_RAM_GB * 0.4
    est_mem_per_file_gb = 0.5  # conservative estimate
    chunk_size = max(10, int(max_chunk_mem_gb / est_mem_per_file_gb))
    chunk_size = min(chunk_size, len(gz_files))  # don't exceed total

    n_chunks = (len(gz_files) + chunk_size - 1) // chunk_size
    log.info(f"authors: converting {len(gz_files)} files in {n_chunks} chunks "
             f"of {chunk_size} files ({len(tables_to_convert)} sub-tables to extract)")

    # Clean up old single-file outputs for tables we're reconverting
    for table_name in tables_to_convert:
        out_dir = PARQUET_DIR / tables_to_convert[table_name][1]
        out_dir.mkdir(parents=True, exist_ok=True)
        # Remove old files (single-file format or previous chunks)
        for old_file in out_dir.glob("*.parquet"):
            old_file.unlink()

    threads, mem = duckdb_settings_for_workers(1)
    # Cap memory to leave headroom for OS and other processes
    mem_cap_gb = min(int(mem.replace("GB", "")), int(TOTAL_RAM_GB * 0.5))
    mem = f"{mem_cap_gb}GB"

    total_rows = {t: 0 for t in tables_to_convert}
    total_size = {t: 0.0 for t in tables_to_convert}

    try:
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_files = gz_files[chunk_start:chunk_start + chunk_size]

            log.info(f"  chunk {chunk_idx + 1}/{n_chunks}: "
                     f"files {chunk_start + 1}-{chunk_start + len(chunk_files)} "
                     f"of {len(gz_files)}")

            conn = duckdb.connect(":memory:")
            conn.execute(f"SET threads={threads}")
            conn.execute(f"SET memory_limit='{mem}'")

            try:
                file_list_sql = ", ".join(f"'{f}'" for f in chunk_files)
                conn.execute(f"""
                    CREATE TEMP TABLE source_data AS
                    SELECT * FROM read_json_auto(
                        [{file_list_sql}],
                        format='newline_delimited',
                        compression='gzip',
                        maximum_object_size=104857600,
                        ignore_errors=true,
                        union_by_name=true
                    )
                """)
                chunk_rows = conn.execute(
                    "SELECT COUNT(*) FROM source_data"
                ).fetchone()[0]
                log.info(f"    materialized {chunk_rows:,} rows")

                for table_name, (sql, out_name, rgs) in tables_to_convert.items():
                    out_dir = PARQUET_DIR / out_name
                    out_file = out_dir / f"{out_name}_chunk{chunk_idx:04d}.parquet"

                    conn.execute(f"""
                        COPY ({sql})
                        TO '{out_file}' (FORMAT PARQUET, COMPRESSION 'zstd',
                            COMPRESSION_LEVEL 3, ROW_GROUP_SIZE {rgs})
                    """)
                    rows = conn.execute(
                        f"SELECT COUNT(*) FROM read_parquet('{out_file}')"
                    ).fetchone()[0]
                    size = out_file.stat().st_size / (1024**3)
                    total_rows[table_name] += rows
                    total_size[table_name] += size

                log.info(f"    chunk {chunk_idx + 1} done: " +
                         ", ".join(f"{t}={total_rows[t]:,}" for t in tables_to_convert))

            finally:
                conn.close()  # Frees all memory for this chunk

        # Log final totals and update checkpoints
        for table_name in tables_to_convert:
            log.info(f"  {table_name}: {total_rows[table_name]:,} rows, "
                     f"{total_size[table_name]:.3f} GB")
            mark_file_completed(checkpoint, table_name, "all", {
                "row_count": total_rows[table_name],
                "size_gb": round(total_size[table_name], 3),
                "input_signature": input_signature,
            })

        return {"entity": "authors", "success": True, "tables": total_rows}

    except Exception as e:
        log.error(f"authors: conversion failed: {e}")
        return {"entity": "authors", "success": False, "error": str(e)}


# ── Works conversion (parallel, materialize-once, inline abstracts) ────────

def convert_single_works_file(args: tuple) -> dict:
    """Convert one works .gz file to main + sub-table Parquets.

    KEY OPTIMIZATION: The JSON is read into a TEMP TABLE once.  All 10 sub-table
    extractions then read from that in-memory table instead of re-parsing JSON.
    """
    gz_path_str, parquet_dir_str, file_idx, total_files, threads, mem = args
    gz_path = Path(gz_path_str)
    parquet_dir = Path(parquet_dir_str)
    gz_name = gz_path.name
    # Create unique stem from partition dir + filename to avoid collisions
    # e.g. updated_date=2016-06-24/part_0000.gz -> 2016-06-24_part_0000
    partition_dir = gz_path.parent.name  # e.g. "updated_date=2016-06-24"
    partition_tag = partition_dir.split("=", 1)[-1] if "=" in partition_dir else partition_dir
    stem = f"{partition_tag}_{gz_path.stem}"

    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={threads}")
    conn.execute(f"SET memory_limit='{mem}'")

    results = {"file": gz_name, "success": True, "tables": {}}
    start = time.time()

    try:
        # ── Step 1: Materialize the full JSON into a temp table ──
        conn.execute(f"""
            CREATE TEMP TABLE source_data AS
            SELECT * FROM read_json_auto(
                '{gz_path}',
                format='newline_delimited',
                compression='gzip',
                maximum_object_size=104857600,
                ignore_errors=true
            )
        """)

        row_count_raw = conn.execute("SELECT COUNT(*) FROM source_data").fetchone()[0]

        # ── Step 1b: Add missing columns (schema varies across partitions) ──
        ensure_source_columns(conn, WORKS_MAIN_SQL)

        # ── Step 2: Main works table ──
        works_out = parquet_dir / "works" / f"{stem}.parquet"
        works_out.parent.mkdir(parents=True, exist_ok=True)

        conn.execute(f"""
            COPY ({WORKS_MAIN_SQL})
            TO '{works_out}' (FORMAT PARQUET, COMPRESSION 'zstd',
                COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 50000)
        """)
        results["tables"]["works"] = row_count_raw

        # ── Step 3: All sub-tables from the materialized temp table ──
        for subtable_name, sql in WORKS_SUBTABLE_SQLS.items():
            sub_out = parquet_dir / subtable_name / f"{stem}.parquet"
            sub_out.parent.mkdir(parents=True, exist_ok=True)

            try:
                conn.execute(f"""
                    COPY ({sql})
                    TO '{sub_out}' (FORMAT PARQUET, COMPRESSION 'zstd',
                        COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 100000)
                """)
                sub_rows = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{sub_out}')"
                ).fetchone()[0]
                results["tables"][subtable_name] = sub_rows
            except Exception as e:
                results["tables"][subtable_name] = f"ERROR: {e}"

        results["elapsed"] = round(time.time() - start, 1)

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
    finally:
        conn.close()

    return results


def process_works_validity_parallel(parquet_dir: Path, workers: int = 6,
                                     force: bool = False):
    """Add valid_title_abstract flag to works Parquets.

    The snapshot already includes plain-text abstracts. This step just adds
    a boolean flag indicating which records have high-quality English abstracts
    suitable for NLP use.

    Parallelized across files with ProcessPoolExecutor.
    """
    import pyarrow.parquet as pq

    works_dir = parquet_dir / "works"
    if not works_dir.exists():
        return

    parquet_files = sorted(works_dir.glob("*.parquet"))

    # Filter to only files needing processing
    pending = []
    for pf in parquet_files:
        try:
            schema = pq.read_schema(pf)
            if force or "valid_title_abstract" not in schema.names:
                pending.append(pf)
        except Exception:
            continue

    if not pending:
        log.info("validity: all works files already processed")
        return

    log.info(f"validity: processing {len(pending)} / {len(parquet_files)} works files "
             f"with {workers} workers")

    tasks = [(str(f), force) for f in pending]

    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_one_validity_file, t): t
            for t in tasks
        }
        for future in as_completed(futures):
            completed += 1
            try:
                fname, n_rows, n_valid = future.result()
                pct = n_valid / n_rows * 100 if n_rows > 0 else 0
                log.info(f"  [{completed}/{len(pending)}] {fname}: "
                         f"{n_rows:,} rows, {n_valid:,} valid ({pct:.1f}%)")
            except Exception as e:
                task = futures[future]
                log.error(f"  [{completed}/{len(pending)}] {Path(task[0]).name}: ERROR {e}")


def _process_one_validity_file(args: tuple) -> tuple:
    """Add valid_title_abstract flag to a single works parquet file. Runs in subprocess."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq_path_str, force = args
    pq_file = Path(pq_path_str)

    table = pq.read_table(pq_file)
    schema = table.schema
    n_rows = len(table)

    abstract_col = table["abstract"].to_pylist() if "abstract" in schema.names else [None] * n_rows
    title_col = table["title"].to_pylist() if "title" in schema.names else [None] * n_rows
    language_col = table["language"].to_pylist() if "language" in schema.names else [None] * n_rows

    valid_flags = []
    for j in range(n_rows):
        valid_flags.append(is_valid_title_abstract(language_col[j], title_col[j], abstract_col[j]))

    # Drop old column if re-processing
    if "valid_title_abstract" in schema.names:
        table = table.drop("valid_title_abstract")

    table = table.append_column("valid_title_abstract", pa.array(valid_flags, type=pa.bool_()))

    pq.write_table(table, pq_file, compression="zstd", compression_level=3,
                    row_group_size=50000)

    return pq_file.name, n_rows, sum(valid_flags)


def _check_compacted_tables() -> bool:
    """Check if any works tables have been compacted (single-file state).

    Returns True if any compacted tables are found. This matters for
    incremental updates: compacted tables have merged all shards into one
    file, so re-converting individual changed files would create duplicates.
    """
    table_names = ["works"] + list(WORKS_SUBTABLE_SQLS.keys())
    for table_name in table_names:
        table_dir = PARQUET_DIR / table_name
        if not table_dir.exists():
            continue
        compacted_file = table_dir / f"{table_name}.parquet"
        if compacted_file.exists():
            all_parquet = list(table_dir.glob("*.parquet"))
            if len(all_parquet) == 1:
                return True
    return False


def convert_works(workers: int = 6, force: bool = False, sample: int = 0,
                  skip_abstracts: bool = False) -> dict:
    """Convert works entity with parallel processing."""
    input_dir = SNAPSHOT_DIR / "works"
    if not input_dir.exists():
        log.warning("works: snapshot dir not found")
        return {"entity": "works", "success": False, "error": "not_found"}

    gz_files = sorted(input_dir.rglob("*.gz"))
    if sample > 0:
        gz_files = gz_files[:sample]
    if not gz_files:
        return {"entity": "works", "success": False, "error": "no_files"}

    checkpoint = load_checkpoint()

    if not force:
        pending = []
        skipped = 0
        changed = 0
        for f in gz_files:
            if is_file_completed(checkpoint, "works", works_file_key(f)):
                if is_works_file_changed(checkpoint, f):
                    pending.append(f)
                    changed += 1
                else:
                    skipped += 1
            else:
                pending.append(f)
        if skipped:
            log.info(f"works: skipping {skipped} already-converted files")
        if changed:
            if _check_compacted_tables():
                # Tables have been compacted (shards merged into single files).
                # Re-converting only changed files would create duplicates
                # because the compacted file already contains old data for
                # those files and we can't surgically remove it.
                # Must escalate to full re-conversion.
                log.warning(
                    f"works: {changed} source files changed but tables are compacted. "
                    f"Escalating to full re-conversion to avoid duplicates. "
                    f"Run --compact after conversion to re-merge."
                )
                # Re-convert ALL files, not just changed ones
                pending = list(gz_files)
                # Clear works checkpoint entries
                completed = checkpoint.get("completed_files", {})
                for k in [k for k in completed if k.startswith("works/")]:
                    del completed[k]
                save_checkpoint(checkpoint)
                # Delete all existing parquet outputs
                for table_name in ["works"] + list(WORKS_SUBTABLE_SQLS.keys()):
                    table_dir = PARQUET_DIR / table_name
                    if table_dir.exists():
                        for pf in table_dir.glob("*.parquet"):
                            pf.unlink()
                        log.info(f"  Cleared {table_name}/ for full re-conversion")
            else:
                log.warning(
                    f"works: {changed} previously-converted files have changed on disk "
                    f"and will be re-converted (tables not compacted, safe to add shards)")
        gz_files = pending

    if not gz_files:
        log.info("works: all files already converted")
        if not skip_abstracts:
            process_works_validity_parallel(PARQUET_DIR, workers=workers, force=force)
        return {"entity": "works", "success": True, "skipped": True}

    threads_per, mem_per = duckdb_settings_for_workers(workers)
    log.info(f"works: converting {len(gz_files)} files | {workers} workers | "
             f"{threads_per} threads/worker | {mem_per} mem/worker")

    tasks = [
        (str(f), str(PARQUET_DIR), i, len(gz_files), threads_per, mem_per)
        for i, f in enumerate(gz_files)
    ]

    total_rows = 0
    success_count = 0
    failed_files = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_single_works_file, t): t for t in tasks}

        for future in as_completed(futures):
            task = futures[future]
            gz_path = Path(task[0])
            completed += 1

            try:
                result = future.result()
                if result["success"]:
                    rows = result["tables"].get("works", 0)
                    total_rows += rows if isinstance(rows, int) else 0
                    success_count += 1
                    file_key = works_file_key(gz_path)
                    gz_stat = gz_path.stat()
                    mark_file_completed(checkpoint, "works", file_key, {
                        "tables": {k: v for k, v in result["tables"].items()
                                   if isinstance(v, int)},
                        "file_size": gz_stat.st_size,
                        "file_mtime": gz_stat.st_mtime,
                    })
                    elapsed = result.get("elapsed", "?")
                    log.info(f"  [{completed}/{len(gz_files)}] {gz_path.name}: "
                             f"{rows:,} works ({elapsed}s)")
                else:
                    failed_files.append(gz_path.name)
                    log.error(f"  [{completed}/{len(gz_files)}] {gz_path.name}: "
                              f"FAILED - {result.get('error')}")
            except Exception as e:
                failed_files.append(gz_path.name)
                log.error(f"  [{completed}/{len(gz_files)}] {gz_path.name}: EXCEPTION - {e}")

    log.info(f"works: {success_count}/{completed} files, {total_rows:,} total rows")
    if failed_files:
        log.warning(f"works: {len(failed_files)} files failed: {failed_files[:5]}")

    # ── Abstract processing (parallelized) ──
    if not skip_abstracts:
        log.info("works: starting validity flag processing...")
        process_works_validity_parallel(PARQUET_DIR, workers=workers, force=force)

    return {
        "entity": "works",
        "success": len(failed_files) == 0,
        "total_rows": total_rows,
        "files_converted": success_count,
        "files_failed": len(failed_files),
        "failed_files": failed_files,
    }


# ── Compaction ──────────────────────────────────────────────────────────────

def compact_table(table_dir: Path) -> dict:
    """Compact all parquet shards in a table directory into a single file.

    Steps:
    1. Read all shards via union_by_name (handles schema differences)
    2. Write to a .tmp file (invisible to *.parquet glob queries)
    3. Verify row count matches
    4. Delete old shard files
    5. Rename .tmp -> .parquet

    Returns dict with result info.
    """
    table_name = table_dir.name
    parquet_files = sorted(table_dir.glob("*.parquet"))

    if len(parquet_files) <= 1:
        if len(parquet_files) == 1:
            return {"table": table_name, "status": "already_compacted"}
        return {"table": table_name, "status": "empty"}

    compact_tmp = table_dir / f"{table_name}_compact.tmp"
    compact_out = table_dir / f"{table_name}.parquet"

    log.info(f"  {table_name}: compacting {len(parquet_files)} shards...")

    threads, mem = duckdb_settings_for_workers(1)
    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={threads}")
    conn.execute(f"SET memory_limit='{mem}'")

    try:
        glob_pattern = str(table_dir / "*.parquet")

        # Count source rows
        source_rows = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{glob_pattern}', union_by_name=true)"
        ).fetchone()[0]
        log.info(f"  {table_name}: {source_rows:,} rows across {len(parquet_files)} shards")

        # Write compacted file to .tmp
        conn.execute(f"""
            COPY (
                SELECT * FROM read_parquet('{glob_pattern}', union_by_name=true)
            )
            TO '{compact_tmp}' (
                FORMAT PARQUET, COMPRESSION 'zstd',
                COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 122880
            )
        """)

        # Verify row count
        compact_rows = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{compact_tmp}')"
        ).fetchone()[0]

        if compact_rows != source_rows:
            compact_tmp.unlink(missing_ok=True)
            msg = (f"{table_name}: row count mismatch! "
                   f"source={source_rows:,} compact={compact_rows:,}")
            log.error(msg)
            return {"table": table_name, "status": "error", "error": msg}

        conn.close()
        conn = None

        # Delete old shard files
        for pf in parquet_files:
            pf.unlink()

        # Rename .tmp -> .parquet
        compact_tmp.rename(compact_out)

        size_gb = compact_out.stat().st_size / (1024**3)
        log.info(f"  {table_name}: compacted to {size_gb:.3f} GB ({compact_rows:,} rows)")

        return {
            "table": table_name,
            "status": "compacted",
            "rows": compact_rows,
            "size_gb": round(size_gb, 3),
        }

    except Exception as e:
        log.error(f"  {table_name}: compaction failed: {e}")
        # Clean up .tmp on failure
        compact_tmp.unlink(missing_ok=True)
        return {"table": table_name, "status": "error", "error": str(e)}
    finally:
        if conn is not None:
            conn.close()


def run_compaction():
    """Compact all table directories under PARQUET_DIR.

    Processes tables sequentially (each gets full machine resources).
    Handles crash recovery: if .tmp exists from a previous interrupted run,
    decides whether to resume or clean up based on state.
    """
    if not PARQUET_DIR.exists():
        log.error("No parquet directory found. Run conversion first.")
        return 1

    table_dirs = sorted(
        d for d in PARQUET_DIR.iterdir()
        if d.is_dir()
    )

    if not table_dirs:
        log.info("No tables to compact.")
        return 0

    log.info(f"=== Compaction: {len(table_dirs)} tables ===")

    # Crash recovery: check for leftover .tmp files
    for td in table_dirs:
        table_name = td.name
        compact_tmp = td / f"{table_name}_compact.tmp"
        if compact_tmp.exists():
            parquet_files = list(td.glob("*.parquet"))
            if not parquet_files:
                # .tmp exists, no .parquet files → previous run completed deletion
                # but crashed before rename. Rename .tmp to .parquet.
                compact_out = td / f"{table_name}.parquet"
                log.info(f"  {table_name}: recovering .tmp -> .parquet (crash recovery)")
                compact_tmp.rename(compact_out)
            else:
                # .tmp exists alongside .parquet files → previous run crashed
                # during write. Delete .tmp and restart compaction.
                log.info(f"  {table_name}: deleting stale .tmp file (crash recovery)")
                compact_tmp.unlink()

    results = []
    for td in table_dirs:
        result = compact_table(td)
        results.append(result)

    # Summary
    print("\n=== Compaction Summary ===")
    for r in results:
        status = r["status"]
        table = r["table"]
        if status == "compacted":
            print(f"  {table:30s}  {r['rows']:>12,} rows  {r['size_gb']:.3f} GB")
        elif status == "already_compacted":
            print(f"  {table:30s}  already compacted")
        elif status == "empty":
            print(f"  {table:30s}  empty (no files)")
        else:
            print(f"  {table:30s}  ERROR: {r.get('error', 'unknown')}")

    errors = sum(1 for r in results if r["status"] == "error")

    # Update meta.json with new compacted sizes
    compacted_any = any(r["status"] == "compacted" for r in results)
    if compacted_any:
        update_meta_json()

    return 1 if errors else 0


# ── Add missing sub-tables ─────────────────────────────────────────────────

def _extract_subtables_from_file(args: tuple) -> dict:
    """Extract specific sub-tables from one works .gz file."""
    gz_path_str, parquet_dir_str, subtable_names, file_idx, total_files, threads, mem = args
    gz_path = Path(gz_path_str)
    parquet_dir = Path(parquet_dir_str)

    partition_dir = gz_path.parent.name
    partition_tag = partition_dir.split("=", 1)[-1] if "=" in partition_dir else partition_dir
    stem = f"{partition_tag}___{gz_path.stem}"

    results = {"file": gz_path.name, "success": True, "tables": {}}
    start = time.time()

    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={threads}")
    conn.execute(f"SET memory_limit='{mem}'")

    try:
        conn.execute(f"""
            CREATE TEMP TABLE source_data AS
            SELECT * FROM read_json('{gz_path}',
                format='newline_delimited',
                compression='gzip',
                maximum_object_size=104857600,
                ignore_errors=true
            )
        """)

        for subtable_name in subtable_names:
            sql = WORKS_SUBTABLE_SQLS.get(subtable_name)
            if not sql:
                results["tables"][subtable_name] = f"ERROR: unknown sub-table"
                continue

            sub_out = parquet_dir / subtable_name / f"{stem}.parquet"
            sub_out.parent.mkdir(parents=True, exist_ok=True)

            try:
                ensure_source_columns(conn, sql)
                conn.execute(f"""
                    COPY ({sql})
                    TO '{sub_out}' (FORMAT PARQUET, COMPRESSION 'zstd',
                        COMPRESSION_LEVEL 3, ROW_GROUP_SIZE 100000)
                """)
                sub_rows = conn.execute(
                    f"SELECT COUNT(*) FROM read_parquet('{sub_out}')"
                ).fetchone()[0]
                results["tables"][subtable_name] = sub_rows
            except Exception as e:
                results["tables"][subtable_name] = f"ERROR: {e}"

        results["elapsed"] = round(time.time() - start, 1)

    except Exception as e:
        results["success"] = False
        results["error"] = str(e)
    finally:
        conn.close()

    return results


def add_subtables(subtable_names: list[str], workers: int = 6,
                  sample: int = 0) -> int:
    """Extract specific sub-tables from existing works source files.

    Used to add new sub-tables (e.g. works_concepts, works_keywords,
    works_related_works) without re-converting everything.
    """
    # Validate sub-table names
    for name in subtable_names:
        if name not in WORKS_SUBTABLE_SQLS:
            log.error(f"Unknown sub-table: {name}")
            log.info(f"Available: {', '.join(sorted(WORKS_SUBTABLE_SQLS.keys()))}")
            return 1

    input_dir = SNAPSHOT_DIR / "works"
    if not input_dir.exists():
        log.error("works: snapshot dir not found")
        return 1

    gz_files = sorted(input_dir.rglob("*.gz"))
    if sample > 0:
        gz_files = gz_files[:sample]

    # Skip files that already have output for ALL requested sub-tables
    pending = []
    for f in gz_files:
        partition_dir = f.parent.name
        partition_tag = partition_dir.split("=", 1)[-1] if "=" in partition_dir else partition_dir
        stem = f"{partition_tag}___{f.stem}"
        missing = False
        for sub_name in subtable_names:
            out = PARQUET_DIR / sub_name / f"{stem}.parquet"
            if not out.exists():
                missing = True
                break
        if missing:
            pending.append(f)

    if not pending:
        log.info(f"add-subtables: all {len(gz_files)} files already have "
                 f"{', '.join(subtable_names)}")
        return 0

    threads_per, mem_per = duckdb_settings_for_workers(workers)
    log.info(f"=== Adding sub-tables: {', '.join(subtable_names)} ===")
    log.info(f"    {len(pending)} files to process ({len(gz_files) - len(pending)} skipped)")
    log.info(f"    {workers} workers, {threads_per} threads/worker, {mem_per} mem/worker")

    tasks = [
        (str(f), str(PARQUET_DIR), subtable_names, i, len(pending), threads_per, mem_per)
        for i, f in enumerate(pending)
    ]

    success_count = 0
    failed_files = []
    completed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract_subtables_from_file, t): t for t in tasks}

        for future in as_completed(futures):
            completed += 1
            task = futures[future]
            gz_name = Path(task[0]).name

            try:
                result = future.result()
                if result["success"]:
                    success_count += 1
                    table_summary = ", ".join(
                        f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}"
                        for k, v in result["tables"].items()
                    )
                    if completed % 100 == 0 or completed == len(pending):
                        log.info(f"  [{completed}/{len(pending)}] {gz_name}: {table_summary} "
                                 f"({result.get('elapsed', '?')}s)")
                else:
                    failed_files.append(gz_name)
                    log.error(f"  [{completed}/{len(pending)}] {gz_name}: "
                              f"FAILED - {result.get('error')}")
            except Exception as e:
                failed_files.append(gz_name)
                log.error(f"  [{completed}/{len(pending)}] {gz_name}: EXCEPTION - {e}")

    log.info(f"\n=== Add Sub-Tables Summary ===")
    log.info(f"  {success_count}/{completed} files processed successfully")
    if failed_files:
        log.warning(f"  {len(failed_files)} files failed: {failed_files[:5]}")

    # Update meta.json
    update_meta_json()

    if not failed_files:
        log.info(f"\nDone! Run these to finalize:")
        log.info(f"  python scripts/convert_openalex.py --compact     # merge shards")
        log.info(f"  python scripts/create_unified_db.py              # rebuild views")

    return 1 if failed_files else 0


# ── Status display ──────────────────────────────────────────────────────────

def convert_entity(entity: str, force: bool = False, sample: int = 0,
                   workers: int = 6, skip_abstracts: bool = False) -> dict:
    """Route an entity to its appropriate converter.

    Uses specialized converters for works/authors, custom SQL for known
    entities, and generic SELECT * for any unknown entity.
    """
    if entity == "works":
        result = convert_works(workers=workers, force=force, sample=sample,
                               skip_abstracts=skip_abstracts)
    elif entity == "authors":
        result = convert_authors(force=force, sample=sample)
    elif entity in SIMPLE_ENTITY_CONFIGS:
        result = convert_simple_entity(entity, force=force, sample=sample)
    else:
        log.info(f"{entity}: no custom SQL config, using generic SELECT * conversion")
        result = convert_generic_entity(entity, force=force, sample=sample)

    if result.get("success") and not result.get("skipped"):
        update_meta_json()

    return result


def show_status():
    print("=== OpenAlex Conversion Status ===\n")
    checkpoint = load_checkpoint()
    total_size = 0

    # Discover all parquet output dirs dynamically
    known_tables = (
        list(SIMPLE_ENTITY_CONFIGS.keys()) +
        ["authors", "authors_ids", "authors_counts_by_year"] +
        ["works"] + list(WORKS_SUBTABLE_SQLS.keys())
    )

    # Add any additional parquet dirs not in the known list
    extra_tables = []
    if PARQUET_DIR.exists():
        for d in sorted(PARQUET_DIR.iterdir()):
            if d.is_dir() and d.name not in known_tables:
                extra_tables.append(d.name)

    all_tables = known_tables + extra_tables

    # Also show downloaded-but-not-converted entities
    downloaded = discover_downloaded_entities()
    not_converted = [
        e for e in downloaded
        if e not in all_tables and e not in {"works", "authors"}
        and not (PARQUET_DIR / e).exists()
    ]

    for table in all_tables:
        tdir = PARQUET_DIR / table
        if not tdir.exists():
            print(f"  {table:30s}  NOT CONVERTED")
            continue

        files = list(tdir.rglob("*.parquet"))
        size = sum(f.stat().st_size for f in files)
        total_size += size

        ckpt_entries = sum(1 for k in checkpoint.get("completed_files", {})
                          if k.startswith(f"{table}/"))

        # Determine compaction state
        if len(files) == 1 and files[0].name == f"{table}.parquet":
            compact_state = "compacted"
        elif len(files) == 1:
            compact_state = "1 shard"
        else:
            compact_state = f"{len(files)} shards"

        print(f"  {table:30s}  {compact_state:>12s}  "
              f"{size / (1024**3):8.2f} GB  ({ckpt_entries} checkpointed)")

    for entity in not_converted:
        print(f"  {entity:30s}  DOWNLOADED (not yet converted)")

    print(f"\n  {'TOTAL':30s}  {total_size / (1024**3):21.2f} GB")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenAlex NDJSON.gz snapshot to flattened Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--entity", type=str,
                        help="Convert a specific entity (any downloaded entity accepted)")
    parser.add_argument("--all", action="store_true",
                        help="Convert all downloaded entities")
    parser.add_argument("--status", action="store_true", help="Show conversion status")
    parser.add_argument("--workers", type=int, default=6,
                        help=f"Workers for parallel conversion (default: 6, machine has {TOTAL_CORES} cores)")
    parser.add_argument("--force", action="store_true", help="Force reconversion")
    parser.add_argument("--sample", type=int, default=0,
                        help="Process first N files only (for testing)")
    parser.add_argument("--skip-abstracts", action="store_true",
                        help="Skip valid_title_abstract flag computation for works")
    parser.add_argument("--compact", action="store_true",
                        help="Compact parquet shards into single optimized files per table")
    parser.add_argument("--add-subtables", nargs="+", metavar="TABLE",
                        help="Extract specific sub-tables from already-converted works "
                             "(e.g. --add-subtables works_concepts works_keywords works_related_works)")

    args = parser.parse_args()

    if args.status:
        show_status()
        return 0

    if args.compact:
        return run_compaction()

    if args.add_subtables:
        return add_subtables(args.add_subtables, workers=args.workers,
                             sample=args.sample)

    if args.entity:
        result = convert_entity(
            args.entity, force=args.force, sample=args.sample,
            workers=args.workers, skip_abstracts=args.skip_abstracts)

        if not result.get("success"):
            log.error(f"Conversion failed: {result}")
        return 0 if result.get("success") else 1

    if args.all:
        # Discover all downloaded entities
        downloaded = discover_downloaded_entities()
        if not downloaded:
            print("No downloaded entities found. Run download_openalex.py first.")
            return 1

        print(f"=== Converting All OpenAlex Entities ===")
        print(f"    Machine: {TOTAL_CORES} cores, {TOTAL_RAM_GB} GB RAM")
        print(f"    Workers: {args.workers}")
        print(f"    Discovered: {', '.join(downloaded)}")
        print()

        # Order: small entities first, authors second-last, works last
        large = {"works", "authors"}
        small_entities = [e for e in downloaded if e not in large]
        ordered = small_entities + (["authors"] if "authors" in downloaded else []) + \
                  (["works"] if "works" in downloaded else [])

        results = {}
        for entity in ordered:
            log.info(f"\n--- {entity} ---")
            results[entity] = convert_entity(
                entity, force=args.force, sample=args.sample,
                workers=args.workers, skip_abstracts=args.skip_abstracts)

        # Summary
        print("\n=== Conversion Summary ===")
        for entity, result in results.items():
            status = "OK" if result.get("success") else "FAILED"
            rows = result.get("row_count", result.get("total_rows", "?"))
            if isinstance(rows, int):
                rows = f"{rows:,}"
            print(f"  {entity:20s}  {status:8s}  {rows} rows")

        failed = sum(1 for r in results.values() if not r.get("success"))
        return 0 if failed == 0 else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
