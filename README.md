<p align="center">
  <img src="sdl_banner.jpg" alt="Science Data Lake" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/J0nasW/science-datalake"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Dataset-yellow" alt="HuggingFace Dataset"></a>
  <a href="https://doi.org/10.57967/hf/7850"><img src="https://img.shields.io/badge/DOI-10.57967%2Fhf%2F7850-blue" alt="DOI"></a>
  <a href="https://github.com/J0nasW/science-datalake/blob/main/LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-green" alt="License: MIT"></a>
  <a href="https://github.com/J0nasW/science-datalake/blob/main/SCHEMA.md"><img src="https://img.shields.io/badge/LLM--Ready-SCHEMA.md-purple" alt="LLM-Ready"></a>
  <a href="https://x.com/Jonas_H_W"><img src="https://img.shields.io/badge/Follow-%40Jonas__H__W-black?logo=x" alt="Follow on X"></a>
</p>

# Science Data Lake

Portable data lake of 480M+ scientific publications from eight complementary datasets, plus 13 scientific ontologies (1.3M terms), queryable via DuckDB.

**Links:** [HuggingFace Dataset](https://huggingface.co/datasets/J0nasW/science-datalake) | [DOI: 10.57967/hf/7850](https://doi.org/10.57967/hf/7850) | [SCHEMA.md (LLM reference)](SCHEMA.md) | [Author website](https://wilinski.me)

## What This Is

Eight scholarly datasets and 13 scientific ontologies unified under a single DuckDB interface with 148 views across 20+ schemas:

- **S2AG** (Semantic Scholar) - 231M papers, 2.9B citation edges with context sentences, full text for 12M papers
- **SciSciNet v2** - 250M papers with disruption index, atypicality, sleeping beauty, patent/funding links
- **OpenAlex** - 479M works (CC0), broadest coverage, topic/institution hierarchies, funding awards
- **Papers With Code** - 513K ML papers with method-task-dataset-code mappings (archived snapshot)
- **Retraction Watch** - 69K retraction/correction records for data quality flagging
- **Reliance on Science** - 47.8M patent-to-paper citations (global)
- **PreprintToPaper** - 146K bioRxiv/medRxiv preprint-to-publication mappings
- **Scientific Ontologies** - 13 ontologies with 1.3M terms: MeSH (721K biomedical), ChEBI (205K chemistry), NCIT (204K cancer), GO (48K biology), AGROVOC (42K agriculture), HPO (20K phenotypes), CSO (15K CS), DOID (15K diseases), STW (8K economics), MSC2020 (7K math), UNESCO (4.5K), PhySH (4K physics), EDAM (3.5K bioinformatics)

All data lives as Parquet files on disk (~960 GB). The `datalake.duckdb` file (~268KB) stores only view definitions pointing to those files. This makes it fully portable: mount the drive, regenerate views, query.

## Snapshot Dates

Each source was downloaded at a specific point in time. The data in this release reflects these snapshots:

| Dataset | Snapshot / Release | Notes |
|---------|-------------------|-------|
| OpenAlex | 2026-02-03 | S3 snapshot `s3://openalex/data/` |
| S2AG (Semantic Scholar) | 2025-12-05 | Datasets API bulk download |
| SciSciNet v2 | 2024-11-01 | GCS bucket `gs://sciscinet-neo/v2` |
| Papers With Code | 2025-07 | Archived JSON snapshot |
| Retraction Watch | 2025-02 | Crossref-sourced CSV |
| Reliance on Science | v64 | Zenodo record |
| PreprintToPaper | 2025-06 | Zenodo record |
| 13 Ontologies | 2026-02 | Downloaded from official sources |

All snapshots can be refreshed using the update pipeline — see [Keeping the Data Lake Current](#keeping-the-data-lake-current) below.

## Architecture

```
datalake.duckdb (268KB, view definitions only)
  ├── s2ag.*          → datasets/s2ag/parquet/**/*.parquet          (437 GB)
  ├── sciscinet.*     → datasets/sciscinet/{core,large}/*.parquet   (151 GB)
  ├── openalex.*      → datasets/openalex/parquet/**/*.parquet      (262 GB)
  ├── pwc.*           → datasets/paperswithcode/parquet/*.parquet   (6.2 GB)
  ├── retwatch.*      → datasets/retractionwatch/parquet/*.parquet  (70 MB)
  ├── ros.*           → datasets/reliance_on_science/parquet/*.parquet (2.7 GB)
  ├── p2p.*           → datasets/preprint_to_paper/parquet/*.parquet (735 MB)
  ├── {ont}.*         → datasets/{ont}/parquet/*.parquet (13 ontologies, 56 MB)
  └── xref.doi_map    → UNION ALL across datasets (normalized DOIs, 588M rows)
```

## Quick Start

### Python

```python
import duckdb
conn = duckdb.connect('datalake.duckdb', read_only=True)
conn.sql("SELECT title, citationcount FROM s2ag.papers WHERE doi = '10.1038/nature12373'").show()
```

### CLI

```bash
python scripts/datalake_cli.py status                    # Disk usage, versions, row counts
python scripts/datalake_cli.py info                      # Dataset descriptions
python scripts/datalake_cli.py query "SELECT COUNT(*) FROM s2ag.papers"
python scripts/datalake_cli.py shell                     # Interactive DuckDB shell
```

## Which Dataset for What?

| Need | Best Dataset | Why |
|------|-------------|-----|
| Citation contexts & intents | S2AG | Only dataset with in-text citation sentences |
| Full paper text | S2AG (s2orc) | 12M open access papers with body text |
| AI paper summaries | S2AG (tldrs) | 70M one-sentence TLDRs |
| Disruption / novelty metrics | SciSciNet | CD index, atypicality, sleeping beauty |
| Normalized citation impact | SciSciNet | Field-year normalized scores, hit paper flags |
| Patent / funding links | SciSciNet + RoS | SciSciNet: NIH/NSF links; RoS: 47.8M global patent-paper citations |
| Broadest paper coverage | OpenAlex | 479M works, CC0 license |
| Topic / field hierarchy | OpenAlex | 4-level: domain -> field -> subfield -> topic |
| Institution geocoding | OpenAlex | 121K institutions with lat/long |
| Funding awards with amounts | OpenAlex | 11.7M awards with dollar amounts and PIs |
| ML methods, tasks, code | Papers With Code | Method-task-dataset mappings, GitHub repos |
| Retraction flagging | Retraction Watch | 69K retraction records joinable via DOI |
| Preprint tracking | PreprintToPaper | 146K preprint-to-publication mappings with timing |
| Biomedical terminology | MeSH | 721K terms, the standard medical vocabulary |
| Chemical compounds | ChEBI | 205K chemical entities with hierarchy |
| Cancer terms | NCIT | 204K cancer/biomedical terms |
| Biological processes | GO | 48K gene ontology terms (MF, BP, CC) |
| Disease classification | DOID | 14.5K disease terms with xrefs |
| CS topic ontology | CSO | 14.6K CS topics with hierarchy |
| Human phenotypes | HPO | 20K phenotype terms |
| All ontologies | 13 schemas | `{ont}.{ont}_terms` — search, browse, cross-link |

## Cross-Dataset Linking

The datasets can be linked via **DOI**, but formats differ:

| Dataset | DOI Column | Format | Example |
|---------|-----------|--------|---------|
| S2AG | `papers.doi` | lowercase, no prefix | `10.1038/nature12373` |
| SciSciNet | `papers.doi` | lowercase, WITH prefix | `https://doi.org/10.1038/nature12373` |
| OpenAlex | `works.doi` | lowercase, WITH prefix | `https://doi.org/10.1038/nature12373` |
| Retraction Watch | `original_paper_doi` | lowercase, no prefix | `10.1038/nature12373` |
| PreprintToPaper | `biorxiv_doi` | lowercase, no prefix | `10.1101/2020.01.01.123456` |

SciSciNet and OpenAlex both store DOIs with `https://doi.org/` prefix. To join with S2AG, strip the prefix: `REPLACE(doi, 'https://doi.org/', '')`.

The `xref.doi_map` view normalizes DOIs from all datasets (588M rows across 7 sources) to no-prefix format:

```sql
SELECT * FROM xref.doi_map WHERE doi = '10.1038/nature12373';
```

Additional join keys:
- SciSciNet `paperid` (e.g., `W2100837269`) equals the OpenAlex work ID (without URL prefix)
- PWC `openalex_work_id` needs prefix: `'https://openalex.org/' || openalex_work_id`
- RoS `oaid` is bare numeric: `'W' || CAST(oaid AS VARCHAR)` for SciSciNet joins

### Pre-Built Cross-Reference Tables

| Table | Rows | Description |
|-------|------|-------------|
| `xref.unified_papers` | 293M | Pre-joined table with coverage flags across all sources |
| `xref.topic_ontology_map` | 16.2K | OpenAlex topics → ontology terms via BGE-large-en-v1.5 embeddings (99.8% topic coverage) |
| `xref.ontology_bridges` | 1.8K | Cross-ontology links via shared external IDs (UMLS, Wikidata, etc.) |

## Keeping the Data Lake Current

The pipeline is designed for repeated updates. When upstream sources release new snapshots, a single command re-downloads, re-converts, and regenerates views:

```bash
# Update a single dataset (download fresh snapshot + convert to Parquet + regenerate views)
python scripts/datalake_cli.py update openalex

# Update all datasets at once
python scripts/datalake_cli.py update

# Re-materialize the unified cross-reference table after any update
python scripts/materialize_unified_papers.py
```

Each source has its own download + convert script that the CLI orchestrates. The pipeline is idempotent — re-running it replaces the old snapshot cleanly.

## Day-to-Day Operations

```bash
# Check status: disk usage, row counts, snapshot dates
python scripts/datalake_cli.py status

# Run a query
python scripts/datalake_cli.py query "SELECT COUNT(*) FROM sciscinet.papers WHERE disruption > 0.5"

# Interactive DuckDB shell
python scripts/datalake_cli.py shell

# Compact OpenAlex shards into single files per table
python scripts/convert_openalex.py --compact

# Regenerate views after mounting on a new machine
python scripts/create_unified_db.py

# Export JSON metadata
python scripts/datalake_cli.py info --format=json --dataset=s2ag
```

## LLM & AI Agent Integration

This data lake is designed to be queryable by LLM-based coding agents (Claude Code, Cursor, Copilot, etc.). The key enabler is **[SCHEMA.md](SCHEMA.md)** — a single, structured reference file containing:

- Every table and column with types, row counts, and descriptions
- Cross-dataset join strategies with ready-to-use SQL examples
- DOI format differences and normalization patterns
- Performance tiers (S/M/L/VL) so agents know when to add WHERE/LIMIT
- Common task recipes (find paper by DOI, get disruption + citations, etc.)

**How to use it:** Point your AI coding assistant at `SCHEMA.md` (e.g., add it to context or reference it in your prompt). The agent can then write correct DuckDB SQL queries across all 8 datasets and 13 ontologies without prior knowledge of the schema.

```
# Example prompt for an AI agent:
"Using the schema in SCHEMA.md, write a query that finds the top 20
 most disruptive papers in computer science that have open-source code
 on GitHub, including their retraction status."
```

## Documentation Map

| File | Purpose | Audience |
|------|---------|----------|
| **README.md** | Overview and operations | Humans, new users |
| **[SCHEMA.md](SCHEMA.md)** | Complete table/column reference | **LLMs, AI agents**, developers |
| **[CATALOG.md](CATALOG.md)** | Detailed queries, quirks, narrative docs | Researchers, analysts |
| **datasets/*/meta.json** | Machine-readable metadata per dataset | Scripts, automation |

## Prerequisites

- Python 3.12+ with venv at `.venv/`
- `pip install duckdb pyarrow` (see `requirements.txt`)
- A local or external SSD with sufficient storage

## Licenses

| Dataset | License | Notes |
|---------|---------|-------|
| S2AG | [Semantic Scholar Dataset License](https://api.semanticscholar.org/corpus/legal/) | Non-commercial research use |
| SciSciNet | CC BY 4.0 | Attribution required |
| OpenAlex | CC0 1.0 | Public domain, no restrictions |
| Papers With Code | CC BY-SA 4.0 | Attribution + share-alike |
| MeSH | Public Domain | US government work |
| GO, ChEBI, NCIT, EDAM, CSO, PhySH | CC BY 4.0 | Attribution required |
| DOID | CC0 1.0 | Public domain |
| AGROVOC | CC BY 3.0 IGO | FAO thesaurus |
| UNESCO Thesaurus | CC BY-SA 3.0 IGO | UNESCO |
| STW | CC BY 4.0 | Economics thesaurus |
| HPO | Custom | Free for research |
| MSC 2020 | CC BY-NC-SA 4.0 | Non-commercial |
| Retraction Watch | Open (via Crossref) | Open access |
| Reliance on Science | CC BY-NC 4.0 | Non-commercial use |
| PreprintToPaper | Open Access | Open access |
