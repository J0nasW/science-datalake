---
language:
- en
license: cc-by-4.0
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
- semantic-scholar
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
  # OpenAlex
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
  # S2AG
  - config_name: s2ag_papers
    data_files: "s2ag/papers/*.parquet"
  - config_name: s2ag_abstracts
    data_files: "s2ag/abstracts/*.parquet"
  - config_name: s2ag_citations
    data_files: "s2ag/citations/*.parquet"
  - config_name: s2ag_authors
    data_files: "s2ag/authors/*.parquet"
  - config_name: s2ag_tldrs
    data_files: "s2ag/tldrs/*.parquet"
  - config_name: s2ag_paper_ids
    data_files: "s2ag/paper_ids/*.parquet"
  # SciSciNet
  - config_name: sciscinet_core
    data_files: "sciscinet/core/*.parquet"
  - config_name: sciscinet_large
    data_files: "sciscinet/large/*.parquet"
  # Papers With Code
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
  - config_name: ros_patent_paper_pairs
    data_files: "ros/patent_paper_pairs/*.parquet"
  - config_name: p2p_preprint_to_paper
    data_files: "p2p/preprint_to_paper/*.parquet"
  # Ontologies (all 13 combined)
  - config_name: ontology_terms
    data_files: "ontologies/*_terms.parquet"
  - config_name: ontology_hierarchy
    data_files: "ontologies/*_hierarchy.parquet"
  - config_name: ontology_xrefs
    data_files: "ontologies/*_xrefs.parquet"
---

# Science Data Lake

A unified, portable science data lake integrating **8 complementary scholarly datasets** (293M+ unique DOIs, ~960 GB Parquet) with cross-dataset DOI normalization, **13 scientific ontologies** (1.3M terms), and a reproducible ETL pipeline.

## What's Unique

This dataset enables queries that are **impossible with any single source**:

```sql
-- "Top disruptive papers with open-source code, checking for retractions"
SELECT doi, title, year,
       sciscinet_disruption,      -- from SciSciNet
       oa_cited_by_count,         -- from OpenAlex
       s2ag_citationcount,        -- from S2AG
       has_pwc,                   -- from Papers With Code
       has_retraction             -- from Retraction Watch
FROM unified_papers
WHERE has_pwc AND sciscinet_disruption > 0.5
ORDER BY oa_cited_by_count DESC
LIMIT 20
```

## Datasets Integrated

| Dataset | Papers/Records | Key Contribution |
|---------|---------------|-----------------|
| **OpenAlex** | 479M works | Broadest coverage, topics, FWCI |
| **S2AG** (Semantic Scholar) | 231M papers | Citation context/intent, TLDRs |
| **SciSciNet** | 159M papers | Disruption index, atypicality, patent/grant linkage |
| **Papers With Code** | 513K papers | Method-task-dataset-code links |
| **Retraction Watch** | 69K records | Retraction flags + reasons |
| **Reliance on Science** | 548K pairs | Patent-paper citation links |
| **Preprint-to-Paper** | 146K pairs | bioRxiv preprint ↔ published paper |
| **13 Ontologies** | 1.3M terms | CSO, MeSH, GO, DOID, ChEBI, NCIT, HPO, EDAM, AGROVOC, UNESCO, STW, MSC2020, PhySH |

## Key Tables

### `unified_papers` (293M rows)
The headline table: one row per unique DOI, joining all sources.

| Column | Type | Description |
|--------|------|-------------|
| `doi` | VARCHAR | Normalized DOI (lowercase, no prefix) |
| `title` | VARCHAR | Best available title (OpenAlex > S2AG) |
| `year` | BIGINT | Publication year |
| `openalex_id` | VARCHAR | OpenAlex work ID |
| `s2ag_corpusid` | BIGINT | S2AG corpus ID |
| `sciscinet_paperid` | VARCHAR | SciSciNet paper ID |
| `has_s2ag` | BOOLEAN | Present in S2AG |
| `has_openalex` | BOOLEAN | Present in OpenAlex |
| `has_sciscinet` | BOOLEAN | Present in SciSciNet |
| `has_pwc` | BOOLEAN | Has code on Papers With Code |
| `has_retraction` | BOOLEAN | Flagged in Retraction Watch |
| `has_patent` | BOOLEAN | Cited in patents (via RoS) |
| `oa_cited_by_count` | BIGINT | OpenAlex citation count |
| `s2ag_citationcount` | BIGINT | S2AG citation count |
| `sciscinet_disruption` | DOUBLE | Disruption index (CD index) |
| `sciscinet_atypicality` | DOUBLE | Atypicality score |
| `oa_fwci` | DOUBLE | Field-Weighted Citation Impact |

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

## Building Your Own Instance

Clone the GitHub repository and run the pipeline:

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
  author={Wilinski, Jonas},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/J0nasW/science-datalake},
  doi={10.5281/zenodo.TODO}
}
```

## License

The integration scripts are MIT licensed. Individual datasets retain their original licenses:
- OpenAlex: CC0
- S2AG: ODC-BY
- SciSciNet: CC BY 4.0
- Papers With Code: CC BY-SA
- Retraction Watch: Custom (research use)
- Reliance on Science: CC BY-NC 4.0
