---
language:
- en
license:
- cc0-1.0
- cc-by-4.0
- cc-by-sa-4.0
- cc-by-nc-sa-4.0
size_categories:
- 100M<n<1B
task_categories:
- text-classification
- feature-extraction
tags:
- scholarly
- academic
- citations
- bibliometrics
- science-of-science
- openalex
- sciscinet
- papers-with-code
- duckdb
- parquet
- ontologies
- knowledge-graph
pretty_name: Science Data Lake
configs:
  # Cross-reference tables
  - config_name: unified_papers
    data_files: "xref/unified_papers/*.parquet"
  - config_name: topic_ontology_map
    data_files: "xref/topic_ontology_map/*.parquet"
  - config_name: ontology_bridges
    data_files: "xref/ontology_bridges/*.parquet"
  # OpenAlex (CC0 1.0)
  - config_name: openalex_works
    data_files: "openalex/works/*.parquet"
  - config_name: openalex_authors
    data_files: "openalex/authors/*.parquet"
  - config_name: openalex_topics
    data_files: "openalex/topics/*.parquet"
  - config_name: openalex_works_topics
    data_files: "openalex/works_topics/*.parquet"
  - config_name: openalex_works_authorships
    data_files: "openalex/works_authorships/*.parquet"
  - config_name: openalex_works_referenced_works
    data_files: "openalex/works_referenced_works/*.parquet"
  - config_name: openalex_works_keywords
    data_files: "openalex/works_keywords/*.parquet"
  - config_name: openalex_institutions
    data_files: "openalex/institutions/*.parquet"
  # SciSciNet (CC BY 4.0)
  - config_name: sciscinet_core
    data_files: "sciscinet/core/*.parquet"
  - config_name: sciscinet_large
    data_files: "sciscinet/large/*.parquet"
  # Papers With Code (CC BY-SA 4.0)
  - config_name: pwc_papers
    data_files: "pwc/papers/*.parquet"
  - config_name: pwc_paper_has_code
    data_files: "pwc/paper_has_code/*.parquet"
  - config_name: pwc_methods
    data_files: "pwc/methods/*.parquet"
  - config_name: pwc_paper_has_task
    data_files: "pwc/paper_has_task/*.parquet"
  - config_name: pwc_datasets
    data_files: "pwc/datasets/*.parquet"
  # Other sources
  - config_name: retwatch
    data_files: "retwatch/retraction_watch/*.parquet"
  - config_name: p2p_preprint_to_paper
    data_files: "p2p/preprint_to_paper/*.parquet"
  # Ontologies (various licenses, see below)
  - config_name: ontology_terms
    data_files: "ontologies/*_terms.parquet"
  - config_name: ontology_hierarchy
    data_files: "ontologies/*_hierarchy.parquet"
  - config_name: ontology_xrefs
    data_files: "ontologies/*_xrefs.parquet"
---

# Science Data Lake

A unified, portable science data lake integrating **6 scholarly datasets** (~523 GB Parquet) with cross-dataset DOI normalization, **13 scientific ontologies** (1.3M terms), and a reproducible ETL pipeline.

> **Note:** Two additional sources (Semantic Scholar S2AG and Reliance on Science) are supported by the pipeline but are **not redistributed here** pending license clarification. See [Not Included in This Upload](#not-included-in-this-upload) below.

## What's Unique

This dataset enables queries that are **impossible with any single source**:

```sql
-- "Top disruptive papers with open-source code, checking for retractions"
SELECT doi, title, year,
       sciscinet_disruption,      -- from SciSciNet
       oa_cited_by_count,         -- from OpenAlex
       has_pwc,                   -- from Papers With Code
       has_retraction             -- from Retraction Watch
FROM unified_papers
WHERE has_pwc AND sciscinet_disruption > 0.5
ORDER BY oa_cited_by_count DESC
LIMIT 20
```

## Datasets Included

| Dataset | Papers/Records | License | Key Contribution |
|---------|---------------|---------|-----------------|
| **OpenAlex** | 479M works | **CC0 1.0** (public domain) | Broadest coverage, topics, FWCI |
| **SciSciNet** v2 | 159M papers | **CC BY 4.0** | Disruption index, atypicality, team size |
| **Papers With Code** | 513K papers | **CC BY-SA 4.0** | Method-task-dataset-code links |
| **Retraction Watch** | 69K records | **Open** (via Crossref) | Retraction flags + reasons |
| **Preprint-to-Paper** | 146K pairs | **CC BY 4.0** | bioRxiv preprint to published paper |
| **13 Ontologies** | 1.3M terms | Various (see below) | CSO, MeSH, GO, DOID, ChEBI, NCIT, HPO, EDAM, AGROVOC, UNESCO, STW, MSC2020, PhySH |

### Ontology Licenses

| Ontology | License |
|----------|---------|
| MeSH | Public Domain (US government work) |
| GO, ChEBI, NCIT, EDAM, CSO, PhySH, STW | CC BY 4.0 |
| DOID | CC0 1.0 |
| AGROVOC | CC BY 3.0 IGO |
| UNESCO Thesaurus | CC BY-SA 3.0 IGO |
| HPO | Custom (free for research use) |
| MSC2020 | **CC BY-NC-SA 4.0** (non-commercial) |

### Not Included in This Upload

The following sources are supported by the full pipeline ([GitHub](https://github.com/J0nasW/science-datalake)) but are **not redistributed here** due to license restrictions or pending clarification:

| Dataset | Reason | How to obtain |
|---------|--------|---------------|
| **S2AG** (Semantic Scholar, 231M papers) | License requires individual agreement with Semantic Scholar | [Semantic Scholar Datasets API](https://api.semanticscholar.org/api-docs/datasets) |
| **Reliance on Science** (548K patent-paper pairs) | CC BY-NC 4.0 — non-commercial restriction | [Zenodo record](https://zenodo.org/records/8278104) |

After downloading these sources locally, run the full pipeline to integrate them.

## Key Tables

### `unified_papers` (293M rows)
The headline table: one row per unique DOI, joining all sources.

| Column | Type | Description |
|--------|------|-------------|
| `doi` | VARCHAR | Normalized DOI (lowercase, no prefix) |
| `title` | VARCHAR | Best available title (OpenAlex > S2AG) |
| `year` | BIGINT | Publication year |
| `openalex_id` | VARCHAR | OpenAlex work ID |
| `sciscinet_paperid` | VARCHAR | SciSciNet paper ID |
| `has_openalex` | BOOLEAN | Present in OpenAlex |
| `has_sciscinet` | BOOLEAN | Present in SciSciNet |
| `has_pwc` | BOOLEAN | Has code on Papers With Code |
| `has_retraction` | BOOLEAN | Flagged in Retraction Watch |
| `oa_cited_by_count` | BIGINT | OpenAlex citation count |
| `sciscinet_disruption` | DOUBLE | Disruption index (CD index) |
| `sciscinet_atypicality` | DOUBLE | Atypicality score |
| `oa_fwci` | DOUBLE | Field-Weighted Citation Impact |

> **Note:** The locally-built version of `unified_papers` includes additional columns from S2AG and RoS (`s2ag_corpusid`, `s2ag_citationcount`, `has_s2ag`, `has_patent`). These columns are present in the uploaded file but will contain NULL values for users who have not integrated those sources locally.

### `topic_ontology_map`
Maps OpenAlex's 4,516 topics to terms in 13 scientific ontologies via embedding-based semantic similarity (BGE-large-en-v1.5, 1024-dim) + exact matching for large ontologies (MeSH, ChEBI, NCIT). 16,150 mappings covering 99.8% of topics. Columns include `similarity` (cosine, 0-1) and `match_type` (label/synonym/exact) for quality filtering.

### `ontology_bridges`
Cross-ontology links discovered via shared external IDs (UMLS, Wikidata, MESH, etc.).

## Usage with DuckDB

```python
import duckdb

# Query directly from HuggingFace
con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

df = con.execute("""
    SELECT doi, title, year, sciscinet_disruption, oa_cited_by_count
    FROM 'hf://datasets/J0nasW/science-datalake/xref/unified_papers/*.parquet'
    WHERE sciscinet_disruption IS NOT NULL
    ORDER BY sciscinet_disruption DESC
    LIMIT 100
""").df()
```

## Building the Full Instance (All 8 Sources)

Clone the GitHub repository and run the pipeline to integrate all sources including S2AG and RoS:

```bash
git clone https://github.com/J0nasW/science-datalake
cd science-datalake
python scripts/datalake_cli.py download --all
python scripts/datalake_cli.py convert --all
python scripts/create_unified_db.py
python scripts/materialize_unified_papers.py
```

## Citation

```bibtex
@dataset{wilinski2026sciencedatalake,
  title={Science Data Lake: A Unified, Portable Data Lake for Full-Lifecycle Scholarly Analysis},
  author={Wilinski, Piotr},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/J0nasW/science-datalake},
  doi={10.5281/zenodo.TODO}
}
```

## License

This dataset aggregates multiple sources, each with its own license. **Users must comply with the most restrictive license applicable to the sources they use.**

| Component | License |
|-----------|---------|
| Integration code (scripts, pipeline) | MIT |
| OpenAlex data | CC0 1.0 (public domain) |
| SciSciNet v2 data | CC BY 4.0 |
| Papers With Code data | CC BY-SA 4.0 |
| Retraction Watch data | Open (via Crossref) |
| Preprint-to-Paper data | CC BY 4.0 |
| Cross-reference tables (`unified_papers`, `topic_ontology_map`) | Derived work — most restrictive source license applies |
| Ontologies | Various — see table above; note **MSC2020 is CC BY-NC-SA 4.0** |
