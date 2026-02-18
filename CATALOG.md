# Science Data Lake Catalog

> **New here?** Start with [README.md](README.md) for overview and operations.
> **LLM/Agent?** See [SCHEMA.md](SCHEMA.md) for complete table/column reference.
> This file provides detailed example queries, data quirks, and narrative documentation.

A unified, portable data lake of scientific publication data.
Three core bibliometric datasets (~250M+ papers each), plus Papers With Code (ML method-task-dataset-code mappings),
the Computer Science Ontology (14K+ CS topics), Retraction Watch (69K retraction records), Reliance on Science
(47.8M patent→paper citations), and PreprintToPaper (146K preprint→publication mappings), all queryable via DuckDB.

## Quick Start

```python
import duckdb
conn = duckdb.connect('datalake.duckdb', read_only=True)

# Query any dataset using schema-qualified names
conn.sql("SELECT * FROM s2ag.papers WHERE year = 2024 LIMIT 5").show()
conn.sql("SELECT * FROM sciscinet.papers WHERE disruption > 0.5 LIMIT 5").show()
conn.sql("SELECT * FROM openalex.authors WHERE h_index > 100 LIMIT 5").show()

# Cross-dataset lookup by DOI
conn.sql("""
    SELECT * FROM xref.doi_map
    WHERE doi = '10.1038/nature12373'
""").show()
```

Or from the command line:
```bash
python scripts/datalake_cli.py query "SELECT COUNT(*) FROM s2ag.papers"
python scripts/datalake_cli.py shell  # interactive DuckDB CLI
python scripts/datalake_cli.py status # disk usage, versions, row counts
python scripts/datalake_cli.py update openalex  # download + convert + views
```

---

## Datasets Overview

| Dataset | Papers | Focus | Key Strengths |
|---------|--------|-------|---------------|
| **S2AG** | 231M | Metadata, abstracts, full text, citations | Rich citation contexts & intents, TLDRs, full text (S2ORC), 2.9B citation edges |
| **SciSciNet** | 250M | Science of Science metrics | Disruption index, atypicality, sleeping beauty, normalized citations, patent/funding links |
| **OpenAlex** | 479M | Comprehensive scholarly catalog | Broadest coverage, CC0 license, institution/funder/topic hierarchies, incremental updates |
| **Papers With Code** | 513K | ML methods, tasks, datasets, code | Method-task-dataset mappings, GitHub repos, patent citations, embeddings, OpenAlex/S2AG links |
| **Scientific Ontologies** | ~33K terms | 3 domain ontologies (converted) | CSO (14.6K CS topics), DOID (14.5K diseases), EDAM (3.5K bioinformatics) |
| **Retraction Watch** | 69K | Retracted/corrected papers | Retraction reasons, dates, journals, data quality flag layer |
| **Reliance on Science** | 47.8M | Patent→paper citations | Global patent-science linkages with confidence scores, institution types |
| **PreprintToPaper** | 146K | Preprint→publication mapping | bioRxiv/medRxiv to journal DOIs, timing, publication status |

### How the Datasets Relate

All three datasets overlap substantially (~200M+ papers appear in all three). They can be linked via **DOI**:

- **S2AG**: DOI is at `papers.doi` (flattened from externalids, lowercase, no prefix). Primary key is `corpusid` (BIGINT).
- **SciSciNet**: DOI is at `papers.doi` (lowercase, **with** `https://doi.org/` prefix). Primary key is `paperid` (VARCHAR, OpenAlex work ID like `W2100837269`).
- **OpenAlex**: DOI is at `works.doi` (lowercase, **with** `https://doi.org/` prefix). Primary key is `id` (VARCHAR, OpenAlex work ID).

The `xref.doi_map` view normalizes DOIs across all three for easy cross-dataset joins.

---

## S2AG (Semantic Scholar Academic Graph)

**Release:** 2025-12-05 | **Size:** 437 GB | **Format:** Parquet

### Schema: `s2ag`

| View | Rows | Description |
|------|------|-------------|
| `papers` | 230.9M | Core metadata: title, year, venue, citations, DOI, authors |
| `abstracts` | 36.7M | Plain text abstracts |
| `authors` | 112.3M | Author profiles: name, affiliations, h-index, paper/citation counts |
| `citations` | 2.9B | Citation edges with context sentences and intents |
| `paper_ids` | 519.3M | SHA-to-corpusid mappings |
| `publication_venues` | 194K | Journals and conferences |
| `tldrs` | 70.4M | AI-generated one-sentence summaries |
| `s2orc` | 12.0M | Full paper text (open access) |
| `paper_authors` | (derived) | Unnested paper-author pairs with position |
| `paper_fields` | (derived) | Unnested paper-field pairs |

### Key Columns (papers)

| Column | Type | Description |
|--------|------|-------------|
| `corpusid` | BIGINT | **Primary key** |
| `title` | VARCHAR | Paper title |
| `year` | BIGINT | Publication year |
| `citationcount` | BIGINT | Total citations |
| `influentialcitationcount` | BIGINT | Influential citations |
| `isopenaccess` | BOOLEAN | Open access status |
| `doi` | VARCHAR | DOI (lowercase, no prefix) - flattened from externalids |
| `venue` | VARCHAR | Venue name |
| `publicationtypes` | VARCHAR[] | JournalArticle, Conference, etc. |
| `authors` | STRUCT[] | [{authorId, name}, ...] |

### Common Queries

```sql
-- Find a paper by DOI
SELECT * FROM s2ag.papers WHERE doi = '10.1038/nature12373';

-- Top cited papers in 2024
SELECT title, citationcount FROM s2ag.papers
WHERE year = 2024 ORDER BY citationcount DESC LIMIT 20;

-- Papers with abstracts
SELECT p.title, a.abstract FROM s2ag.papers p
JOIN s2ag.abstracts a ON p.corpusid = a.corpusid
WHERE p.year = 2023 LIMIT 10;

-- Citation count distribution
SELECT citationcount, COUNT(*) AS n
FROM s2ag.papers WHERE year = 2020
GROUP BY citationcount ORDER BY citationcount;
```

---

## SciSciNet v2

**Release:** 2024-11-01 | **Size:** 151 GB | **Format:** Parquet

### Schema: `sciscinet`

| View | Rows | Description |
|------|------|-------------|
| `papers` | 249.8M | Scientometric data: disruption, atypicality, sleeping beauty |
| `paper_details` | 249.8M | Title, abstract, language, validity flag |
| `papers_english` | ~162M | Filtered: valid English papers with readable abstracts |
| `authors` | 100.4M | Author h-index, productivity, avg citations |
| `author_details` | 100.4M | ORCID, institution, alternative names |
| `affiliations` | 110K | Institution metadata with country, type, metrics |
| `fields` | 303 | OpenAlex field hierarchy |
| `sources` | 261K | Publication venues |
| `funders` | 32K | Funding organizations |
| `paper_refs` | 2.5B | Citation edges with year difference |
| `paper_fields` | 1.3B | Paper-field mappings with scores |
| `paper_authors` | 773M | Paper-author-affiliation relationships |
| `paper_sources` | 204M | Paper-venue mappings with OA info |
| `hit_papers` | 570M | Top 1%/5%/10% papers by field-year |
| `normalized_citations` | 570M | Field-year normalized citations |
| `link_nih` | 6.5M | Paper-NIH funding links |
| `link_nsf` | 1.8M | Paper-NSF funding links |
| `link_patents` | 47.8M | Paper-patent citation links |
| `link_twitter` | 55.8M | Paper-tweet links |

### Key Columns (papers)

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | **Primary key** (OpenAlex work ID, e.g., `W2100837269`) |
| `doi` | VARCHAR | DOI (lowercase, no prefix) |
| `year` | BIGINT | Publication year |
| `disruption` | DOUBLE | CD disruption index (-1 to 1, higher = more disruptive) |
| `Atyp_Median_Z` | DOUBLE | Atypicality median z-score |
| `C3`, `C5`, `C10` | UINTEGER | Citations within 3/5/10 years |
| `SB_B` | DOUBLE | Sleeping beauty coefficient |
| `team_size` | UINTEGER | Number of authors |
| `patent_count` | UINTEGER | Citing patents |

### Common Queries

```sql
-- Most disruptive papers in 2020
SELECT p.paperid, d.title, p.disruption, p.cited_by_count
FROM sciscinet.papers p
JOIN sciscinet.paper_details d ON p.paperid = d.paperid
WHERE p.year = 2020 AND p.disruption IS NOT NULL
ORDER BY p.disruption DESC LIMIT 20;

-- Papers with both high disruption and high citations
SELECT paperid, year, disruption, citation_count
FROM sciscinet.papers
WHERE disruption > 0.5 AND citation_count > 100;

-- English papers with abstracts for NLP
SELECT paperid, title, abstract
FROM sciscinet.paper_details
WHERE valid_title_abstract = true AND year >= 2020;

-- NIH-funded papers and their impact
SELECT p.paperid, p.year, p.citation_count, l.award_id
FROM sciscinet.papers p
JOIN sciscinet.link_nih l ON p.paperid = l.paperid;
```

---

## OpenAlex

**Release:** 2026-02-03 | **Size:** 262 GB Parquet (757 GB raw snapshots) | **Format:** Parquet
**License:** CC0 1.0 | **Source:** s3://openalex (free, no API key)

### Schema: `openalex`

| View | Rows | Description |
|------|------|-------------|
| `works` | 479.3M | Paper metadata: title, DOI, abstract, publication year, citations, type |
| `works_authorships` | 1.32B | Author-paper relationships with institutions and positions |
| `works_referenced_works` | 3.01B | Citation edges (outgoing references) |
| `works_topics` | 909.9M | Topic assignments with scores |
| `works_locations` | 612.0M | Hosting venues with OA status |
| `works_ids` | 479.3M | External IDs (DOI, PMID, PMC, MAG) per work |
| `works_biblio` | 479.3M | Bibliographic info: volume, issue, pages |
| `works_open_access` | 479.3M | OA status, OA URL, license |
| `works_best_oa_location` | 210.0M | Best open access location per work |
| `works_counts_by_year` | 442.9M | Per-year citation counts per work |
| `authors` | 107.5M | Author profiles: name, ORCID, h-index, institutional affiliations |
| `authors_ids` | 107.5M | Author external identifiers (ORCID, OpenAlex ID) |
| `authors_counts_by_year` | 302.4M | Per-year publication and citation counts per author |
| `awards` | 11.7M | Funding awards with amounts, investigators, linked outputs |
| `topics` | 4,516 | Topic hierarchy (topic -> subfield -> field -> domain) |
| `domains` | 4 | Top-level: Life Sciences, Physical Sciences, Health Sciences, Social Sciences |
| `fields` | 26 | Mid-level: Computer Science, Medicine, Physics, etc. |
| `subfields` | 252 | Fine-grained classification under each field |
| `publishers` | 10,703 | Publishing organizations with ROR IDs |
| `funders` | 32,437 | Funding organizations with country codes and metrics |
| `concepts` | 65,026 | Concept taxonomy (deprecated, use topics) |
| `sources` | 255,250 | Journals, conferences, repositories with ISSN and OA info |
| `institutions` | 120,658 | Research institutions with ROR, geo coordinates, metrics |

### Key Columns (works)

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | **Primary key** (OpenAlex work ID, URL format) |
| `doi` | VARCHAR | DOI with `https://doi.org/` prefix |
| `title` | VARCHAR | Paper title |
| `publication_year` | INTEGER | Publication year |
| `cited_by_count` | BIGINT | Total citations |
| `type` | VARCHAR | Work type (article, book-chapter, etc.) |
| `abstract` | VARCHAR | Plain text abstract |
| `valid_title_abstract` | BOOLEAN | Quality flag: English, title>=10 chars, abstract>=50 chars |

### Key Columns (authors)

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | **Primary key** (OpenAlex author ID) |
| `orcid` | VARCHAR | ORCID identifier |
| `display_name` | VARCHAR | Author name |
| `works_count` | BIGINT | Total works |
| `cited_by_count` | BIGINT | Total citations |
| `h_index` | BIGINT | h-index |
| `i10_index` | BIGINT | i10-index |
| `last_known_institutions` | JSON | Most recent affiliations |

### Common Queries

```sql
-- Find a paper by DOI
SELECT id, title, publication_year, cited_by_count
FROM openalex.works
WHERE doi = 'https://doi.org/10.1038/nature12373';

-- Top-cited works in 2024
SELECT title, cited_by_count, type
FROM openalex.works
WHERE publication_year = 2024
ORDER BY cited_by_count DESC LIMIT 20;

-- Works with their topics
SELECT w.title, w.cited_by_count, t.display_name AS topic, wt.score
FROM openalex.works w
JOIN openalex.works_topics wt ON w.id = wt.work_id
JOIN openalex.topics t ON wt.topic_id = t.id
WHERE w.doi = 'https://doi.org/10.1038/nature12373';

-- Top-cited authors in OpenAlex
SELECT display_name, works_count, cited_by_count, h_index
FROM openalex.authors
ORDER BY cited_by_count DESC LIMIT 20;

-- Topics in Computer Science
SELECT t.display_name, t.works_count, t.cited_by_count
FROM openalex.topics t
WHERE t.field_display_name = 'Computer Science'
ORDER BY t.works_count DESC LIMIT 20;

-- Institutions by country
SELECT country_code, COUNT(*) as n, SUM(works_count) as total_works
FROM openalex.institutions
GROUP BY country_code ORDER BY total_works DESC LIMIT 20;

-- Topic hierarchy: domains -> fields -> subfields -> topics
SELECT d.display_name AS domain, f.display_name AS field,
       sf.display_name AS subfield, t.display_name AS topic
FROM openalex.topics t
JOIN openalex.subfields sf ON t.subfield_id = sf.id
JOIN openalex.fields f ON sf.field_id = f.id
JOIN openalex.domains d ON f.domain_id = d.id
LIMIT 20;
```

### Update Pipeline

```bash
# Full update: download new data + convert + regenerate views
python scripts/datalake_cli.py update openalex

# Or step by step:
python scripts/download_openalex.py --all     # S3 sync (incremental)
python scripts/convert_openalex.py --all      # NDJSON → Parquet (checkpoint/resume)
python scripts/convert_openalex.py --compact  # Merge shards into single files per table
python scripts/create_unified_db.py           # Regenerate DuckDB views
```

---

## Papers With Code (PWC)

**Release:** 2025-07 | **Size:** 6.6 GB | **Format:** Parquet | **License:** CC BY-SA 4.0

Structured ML research data: 512K papers with method→task→dataset→code mappings, patent citations, and embeddings. Archived snapshot (PWC shut down July 2025).

### Schema: `pwc`

| View | Rows | Description |
|------|------|-------------|
| `papers` | 513K | Core paper metadata: title, abstract, arXiv ID, URLs |
| `methods` | 2.3K | ML/AI method catalog with descriptions |
| `tasks` | 5.3K | Research task catalog |
| `datasets` | 11K | Dataset catalog with modalities, languages |
| `paper_uses_method` | 621K | Paper→method usage links |
| `paper_has_task` | 951K | Paper→task assignment links |
| `paper_has_code` | 269K | Paper→GitHub repo links |
| `code_repos` | 215K | Repo metadata: framework, is_official |
| `paper_has_openalexWorkID` | 407K | Paper→OpenAlex work ID (bare `W...` format) |
| `paper_has_openalexAuthorID` | 1.4M | Paper→OpenAlex author ID |
| `paper_has_embedding` | 513K | Dense paper embeddings (2.4 GB) |
| `papers_fulltexts` | 203K | Full text with S2ORC corpus ID and DOI |
| `paper_links` | 512K | Paper→SemOpenAlex URL mappings |
| `patent_cites_paper` | 156K | Patent→paper citations with confidence |
| `patents` | 81K | Patent metadata |
| `patent_paper_pairs` | 3K | Patent-paper similarity pairs |
| `paper_introduces_method` | 2.1K | Which paper first introduced a method |
| `paper_introduces_dataset` | 8.6K | Which paper first introduced a dataset |
| `dataset_has_task` | 18K | Dataset→task links |
| `cso_topics` | 6.4K | CSO topic vocabulary (topic names only) |
| `areas` | 16 | Top-level research areas |
| `collections` | 362 | Method collections |
| `collection_isin_area` | 362 | Collection→area mapping |
| `method_isin_collection` | 2.8K | Method→collection mapping |
| `conferences` | 1.8K | Conference proceedings |
| `paper_isin_conference` | 103K | Paper→conference links |
| `book_indices` | 3.2K | Textbook keyword index |

### Cross-Dataset Join Keys

- **PWC → OpenAlex**: `'https://openalex.org/' || pwc.paper_has_openalexWorkID.openalex_work_id = openalex.works.id`
- **PWC → S2AG**: `pwc.papers_fulltexts.s2orc_corpus_id = s2ag.papers.corpusid`
- **PWC → S2AG (DOI)**: `pwc.papers_fulltexts.doi = s2ag.papers.doi` (both lowercase, no prefix)
- **PWC → CSO**: `pwc.cso_topics.cso_topics` matches `cso.cso_hierarchy.child_topic` (topic name join)

### Common Queries

```sql
-- Methods with most papers
SELECT m.name, m.full_name, m.num_papers
FROM pwc.methods m ORDER BY m.num_papers DESC LIMIT 20;

-- Papers with code repositories (official implementations)
SELECT p.title, c.repo_url, c.framework
FROM pwc.papers p
JOIN pwc.paper_has_code pc ON p.paper_id = pc.paper_id
JOIN pwc.code_repos c ON pc.repo_url = c.repo_url
WHERE c.is_official = true LIMIT 20;

-- Method diffusion: which methods span the most tasks?
SELECT m.name AS method, COUNT(DISTINCT pt.task_id) AS task_count
FROM pwc.paper_uses_method pm
JOIN pwc.methods m ON pm.method_id = m.method_id
JOIN pwc.paper_has_task pt ON pm.paper_id = pt.paper_id
GROUP BY m.name ORDER BY task_count DESC LIMIT 20;

-- Cross-dataset: PWC papers matched to S2AG via corpus ID
SELECT p.title, ft.doi, ft.s2orc_corpus_id
FROM pwc.papers p
JOIN pwc.papers_fulltexts ft ON p.paper_id = ft.paper_id
WHERE ft.s2orc_corpus_id IS NOT NULL LIMIT 10;
```

---

## Scientific Ontologies (13 Converted)

**Format:** Parquet (flat tables) | **Total:** ~1.29M terms, ~56 MB Parquet

All 13 scientific ontologies are converted and queryable via DuckDB. Each has up to three tables: `{name}_terms` (always present), `{name}_hierarchy` (parent-child edges), and `{name}_xrefs` (cross-references to other databases). Some ontologies lack hierarchy or xref data — see the "—" entries below.

| Ontology | Schema | Domain | Terms | Hierarchy Edges | Xrefs | License |
|----------|--------|--------|-------|-----------------|-------|---------|
| MeSH | `mesh` | Biomedical | 720,801 | — | — | Public Domain |
| ChEBI | `chebi` | Chemistry | 205,317 | 379,841 | 389,000 | CC BY 4.0 |
| NCI Thesaurus | `ncit` | Cancer/Biomedical | 203,668 | 293,258 | 2,250 | CC BY 4.0 |
| Gene Ontology | `go` | Biology | 47,856 | 81,173 | 25,868 | CC BY 4.0 |
| AGROVOC | `agrovoc` | Agriculture | 41,699 | 42,132 | 49,762 | CC BY 3.0 IGO |
| Human Phenotype Ontology | `hpo` | Phenotypes | 19,934 | 23,765 | 18,099 | Custom (free for research) |
| CSO | `cso` | Computer Science | 14,636 | 93,491 | 28,100 | CC BY 4.0 |
| Disease Ontology | `doid` | Disease | 14,521 | 16,916 | 38,653 | CC0 |
| STW | `stw` | Economics | 7,858 | 28,702 | — | CC BY 4.0 |
| MSC 2020 | `msc2020` | Mathematics | 6,603 | 6,603 | — | CC BY-NC-SA 4.0 |
| UNESCO Thesaurus | `unesco` | Education/Science | 4,498 | 8,682 | — | CC BY-SA 3.0 IGO |
| PhySH | `physh` | Physics | 3,925 | 8,844 | — | CC BY 4.0 |
| EDAM | `edam` | Bioinformatics | 3,524 | 5,219 | 266 | CC BY 4.0 |

### SQL Example Queries

```sql
-- Search disease terms
SELECT id, label, definition FROM doid.doid_terms
WHERE label ILIKE '%alzheimer%';

-- Navigate DOID hierarchy
SELECT h.child_id, t.label
FROM doid.doid_hierarchy h
JOIN doid.doid_terms t ON h.child_id = t.id
WHERE h.parent_id = 'DOID:14566';

-- CSO topic hierarchy
SELECT h.child_id, t.label
FROM cso.cso_hierarchy h
JOIN cso.cso_terms t ON h.child_id = t.id
WHERE h.parent_id ILIKE '%machine_learning%' AND h.relation = 'superTopicOf';

-- Search across multiple ontologies
SELECT 'doid' AS src, label FROM doid.doid_terms WHERE label ILIKE '%cancer%'
UNION ALL
SELECT 'mesh', label FROM mesh.mesh_terms WHERE label ILIKE '%cancer%'
UNION ALL
SELECT 'ncit', label FROM ncit.ncit_terms WHERE label ILIKE '%cancer%';
```

---

## Retraction Watch

**Release:** 2025-02 | **Size:** 8.5 MB | **Format:** Parquet (converted from CSV) | **License:** Open (via Crossref)

Comprehensive database of retracted and corrected scholarly publications. Use as a data quality layer to flag retracted papers across the datalake.

### Schema: `retwatch`

| View | Rows | Description |
|------|------|-------------|
| `retraction_watch` | 68,869 | Retraction/correction records with DOIs, reasons, dates |

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | INTEGER | Unique record ID |
| `original_paper_doi` | VARCHAR | DOI of retracted paper (lowercase, no prefix) |
| `retraction_doi` | VARCHAR | DOI of retraction notice |
| `retraction_nature` | VARCHAR | Retraction, Correction, Expression of Concern, etc. |
| `reason` | VARCHAR | Semicolon-separated retraction reasons |
| `retraction_date` | DATE | Date of retraction action |
| `original_paper_date` | DATE | Date of original paper |
| `subject` | VARCHAR | Subject area |
| `journal` | VARCHAR | Journal name |
| `publisher` | VARCHAR | Publisher name |
| `country` | VARCHAR | Country of origin |
| `author` | VARCHAR | Author names |
| `paywalled` | VARCHAR | Whether retraction notice is paywalled (YES/NO) |

### Common Queries

```sql
-- Retracted papers by reason
SELECT reason, COUNT(*) AS n FROM retwatch.retraction_watch
GROUP BY reason ORDER BY n DESC LIMIT 20;

-- Flag retracted papers in S2AG results
SELECT p.title, p.citationcount, rw.retraction_nature, rw.reason
FROM s2ag.papers p
JOIN retwatch.retraction_watch rw ON p.doi = rw.original_paper_doi
WHERE p.year >= 2020 ORDER BY p.citationcount DESC LIMIT 20;

-- Retraction trends over time
SELECT YEAR(retraction_date) AS yr, COUNT(*) AS retractions
FROM retwatch.retraction_watch
WHERE retraction_date IS NOT NULL
GROUP BY yr ORDER BY yr;
```

---

## Reliance on Science (Marx & Fuegi)

**Release:** v64 | **Size:** 1.0 GB | **Format:** Parquet (converted from CSV) | **License:** Open Access

Global patent-to-paper citations. The most comprehensive open dataset linking patents worldwide to the scientific papers they cite.

### Schema: `ros`

| View | Rows | Description |
|------|------|-------------|
| `pcs_oa` | 47.8M | Patent citation-to-science pairs with confidence scores |
| `patent_paper_pairs` | 548K | Curated patent-paper similarity pairs |
| `patent_paper_pairs_plus` | 548K | Extended pairs with institutional and commercialization metadata |

### Key Columns (pcs_oa)

| Column | Type | Description |
|--------|------|-------------|
| `patent` | VARCHAR | Patent ID (e.g., `US-10000036`) |
| `oaid` | BIGINT | OpenAlex numeric ID (bare, no `W` prefix). Join: `'W' \|\| CAST(oaid AS VARCHAR)` |
| `reftype` | VARCHAR | Citation type |
| `confscore` | INTEGER | Confidence score |
| `uspto` | INTEGER | USPTO flag |
| `self_cite` | VARCHAR | Self-citation flag |

### Key Columns (patent_paper_pairs)

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | OpenAlex work ID with `W` prefix (e.g., `W2100837269`) |
| `patent` | VARCHAR | Patent ID |
| `ppp_score` | DOUBLE | Patent-paper pair similarity score |
| `daysdiffcont` | DOUBLE | Days difference (continuous) |

### Cross-Dataset Join Keys

- **RoS → OpenAlex**: `'https://openalex.org/W' || CAST(ros.pcs_oa.oaid AS VARCHAR) = openalex.works.id`
- **RoS → SciSciNet**: `'W' || CAST(ros.pcs_oa.oaid AS VARCHAR) = sciscinet.papers.paperid`
- **RoS (pairs) → SciSciNet**: `ros.patent_paper_pairs.paperid = sciscinet.papers.paperid`

### Common Queries

```sql
-- Most-cited papers by patents
SELECT oaid, COUNT(*) AS patent_citations
FROM ros.pcs_oa
GROUP BY oaid ORDER BY patent_citations DESC LIMIT 20;

-- Patent-paper pairs with SciSciNet impact metrics
SELECT pp.paperid, pp.patent, pp.ppp_score,
       sc.disruption, sc.cited_by_count
FROM ros.patent_paper_pairs pp
JOIN sciscinet.papers sc ON pp.paperid = sc.paperid
WHERE sc.disruption IS NOT NULL
ORDER BY sc.disruption DESC LIMIT 20;

-- Commercialized patents from university research
SELECT patent, paperid, ppp_score
FROM ros.patent_paper_pairs_plus
WHERE paperuniv = true AND commercialized = true
LIMIT 20;
```

---

## PreprintToPaper

**Release:** 2025-06 | **Size:** 11 MB | **Format:** Parquet (converted from CSV) | **License:** Open Access

Links bioRxiv/medRxiv preprints to their published journal versions with timing and status metadata.

### Schema: `p2p`

| View | Rows | Description |
|------|------|-------------|
| `preprint_to_paper` | 145,517 | Preprint→publication mappings with status and timing |
| `preprint_to_paper_grayzone` | 299 | Manually annotated suspected matches |

### Publication Status Breakdown

| Status | Count | Description |
|--------|-------|-------------|
| published | 90,614 | Confirmed published in a journal |
| preprint_only | 35,813 | Remained as preprint only |
| gray_zone | 19,090 | Uncertain publication status |

### Key Columns (preprint_to_paper)

| Column | Type | Description |
|--------|------|-------------|
| `biorxiv_doi` | VARCHAR | Preprint DOI (e.g., `10.1101/2020.01.01.123456`) |
| `biorxiv_published_doi` | VARCHAR | Published version DOI (nullable) |
| `custom_status` | VARCHAR | `published`, `preprint_only`, or `gray_zone` |
| `crossref_journal_name` | VARCHAR | Journal name from Crossref |
| `biorxiv_category` | VARCHAR | bioRxiv subject category |
| `first_submission_date` | VARCHAR | First preprint submission date |
| `first_pub_date` | VARCHAR | Published version date |
| `submission_pub_date_diff_days` | DOUBLE | Days from submission to publication |

### Common Queries

```sql
-- Publication rate by category
SELECT biorxiv_category,
       COUNT(*) AS total,
       SUM(CASE WHEN custom_status = 'published' THEN 1 ELSE 0 END) AS published,
       ROUND(100.0 * SUM(CASE WHEN custom_status = 'published' THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_published
FROM p2p.preprint_to_paper
GROUP BY biorxiv_category ORDER BY total DESC LIMIT 20;

-- Time to publication distribution
SELECT ROUND(submission_pub_date_diff_days / 30) AS months_to_pub,
       COUNT(*) AS n
FROM p2p.preprint_to_paper
WHERE custom_status = 'published' AND submission_pub_date_diff_days IS NOT NULL
GROUP BY months_to_pub ORDER BY months_to_pub;

-- Cross-reference: preprints in S2AG with citation data
SELECT p2p.biorxiv_doi, p2p.biorxiv_published_doi, p2p.custom_status,
       s.citationcount
FROM p2p.preprint_to_paper p2p
JOIN s2ag.papers s ON p2p.biorxiv_published_doi = s.doi
WHERE p2p.custom_status = 'published'
ORDER BY s.citationcount DESC LIMIT 20;
```

---

## Cross-Dataset Queries (xref schema)

### DOI-Based Lookup

```sql
-- Find a paper across all datasets by DOI
SELECT * FROM xref.doi_map WHERE doi = '10.1038/nature12373';

-- Get S2AG + SciSciNet data for the same paper
SELECT s.title, s.citationcount, sc.disruption, sc.C10
FROM s2ag.papers s
JOIN sciscinet.papers sc ON s.doi = sc.doi
WHERE s.doi = '10.1038/nature12373';

-- Link SciSciNet to OpenAlex via paperid = work ID
SELECT sc.paperid, sc.disruption, oa.title, oa.cited_by_count
FROM sciscinet.papers sc
JOIN openalex.works oa ON 'https://openalex.org/' || sc.paperid = oa.id
WHERE sc.year = 2020 AND sc.disruption > 0.5
LIMIT 20;
```

### Topic → Ontology Alignment

The `xref.topic_ontology_map` table maps OpenAlex's 4,516 topics to terms in 10 scientific ontologies using embedding-based semantic similarity (BGE-large-en-v1.5, 1024-dim, cosine threshold ≥ 0.65). Three large entity-level ontologies (MeSH, ChEBI, NCIT) use exact label matching instead.

```sql
-- Find ontology terms aligned to an OpenAlex topic
SELECT ontology, ontology_term_label, ROUND(similarity, 3) AS sim, match_type
FROM xref.topic_ontology_map
WHERE topic_name = 'Machine Learning Algorithms'
ORDER BY similarity DESC;

-- Find all topics linked to a specific ontology
SELECT topic_name, ontology_term_label, ROUND(similarity, 3) AS sim
FROM xref.topic_ontology_map
WHERE ontology = 'cso' AND similarity >= 0.85
ORDER BY similarity DESC;

-- Bridge from paper → topic → ontology term
SELECT w.title, t.display_name AS topic, m.ontology, m.ontology_term_label
FROM openalex.works_topics wt
JOIN openalex.works w ON wt.work_id = w.id
JOIN openalex.topics t ON wt.topic_id = t.id
JOIN xref.topic_ontology_map m ON t.id = m.topic_id
WHERE m.similarity >= 0.85
LIMIT 20;
```

Quality tiers by cosine similarity:
- **≥ 0.95**: Exact/near-exact matches (85 mappings)
- **0.85–0.95**: High-quality semantic matches (2,442 mappings)
- **0.75–0.85**: Reasonable domain matches (8,778 mappings)
- **0.65–0.75**: Borderline, use with caution (4,845 mappings)

### Cross-Ontology Bridges

The `xref.ontology_bridges` table links terms across ontologies via shared external IDs (UMLS CUIs, Wikidata, SNOMEDCT, etc.):

```sql
SELECT ontology_1, term_id_1, ontology_2, term_id_2, bridge_type, bridge_id
FROM xref.ontology_bridges
LIMIT 10;
```

---

## Data Quirks & Gotchas

1. **DOI formats differ**: S2AG stores lowercase DOIs without prefix. SciSciNet and OpenAlex store them with `https://doi.org/` prefix. Always normalize with `LOWER(REPLACE(doi, 'https://doi.org/', ''))`. The `xref.doi_map` view handles this automatically.

2. **SciSciNet paperid = OpenAlex work ID**: SciSciNet's `paperid` (e.g., `W2100837269`) is the same as OpenAlex's `id`. You can join directly.

3. **S2AG uses corpusid, not DOI**: S2AG's primary key is `corpusid` (BIGINT). DOI is in `papers.doi` and may be NULL.

4. **Citation counts differ**: Each dataset computes citation counts independently. S2AG counts are based on S2AG's corpus, SciSciNet on OpenAlex, etc.

5. **Large tables**: `citations` (2.9B), `paper_refs` (2.5B), `paper_fields` (1.3B) are very large. Always use WHERE clauses and LIMIT.

6. **Abstract coverage**: S2AG abstracts cover only ~37M papers (16%). SciSciNet has abstracts for most papers but requires `valid_title_abstract = true` for quality filtering. OpenAlex has plain text abstracts for most works with a similar `valid_title_abstract` quality flag.

7. **Views, not tables**: The DuckDB file contains views pointing to Parquet files, not materialized tables. Queries read directly from Parquet. This means the `.duckdb` file is tiny (~268KB) but queries are I/O bound.

8. **Portability**: After mounting on a new workstation, run `./remount.sh` or `python scripts/create_unified_db.py` to regenerate views with correct paths.

9. **OpenAlex schema evolution**: OpenAlex snapshots evolve over time. The conversion scripts use `TRY_CAST`, `ignore_errors=true`, and `union_by_name=true` to handle schema differences across partitions (e.g. 2016 vs 2025 data).

10. **PWC OpenAlex ID format**: PWC stores OpenAlex work IDs as bare `W2803600120`, while OpenAlex uses `https://openalex.org/W2803600120`. Prepend prefix for joins.

---

## File Structure

```
science_datalake/
├── CATALOG.md                    # This file
├── SCHEMA.md                     # LLM-optimized schema reference
├── datalake.json                 # Machine-readable manifest (16 datasets)
├── datalake.duckdb               # Unified views (~268KB, 121 views)
├── remount.sh                    # Run after mounting on new machine
├── requirements.txt
├── .venv/
├── datasets/
│   ├── s2ag/
│   │   ├── parquet/{abstracts,authors,citations,...}/*.parquet  (437 GB)
│   │   └── meta.json
│   ├── sciscinet/
│   │   ├── core/*.parquet        # 30 tables
│   │   ├── large/*.parquet       # Paper details with abstracts  (151 GB)
│   │   └── meta.json
│   ├── openalex/
│   │   ├── snapshot/             # Raw NDJSON from S3 (~757 GB)
│   │   ├── parquet/              # 23 converted tables (262 GB)
│   │   └── meta.json
│   ├── paperswithcode/
│   │   ├── parquet/*.parquet     # 24 tables (6.2 GB)
│   │   └── meta.json
│   ├── cso/
│   │   ├── parquet/*.parquet     # 3 tables (<1 MB)
│   │   └── meta.json
│   ├── doid/
│   │   └── parquet/*.parquet     # 3 tables (<1 MB)
│   ├── edam/
│   │   └── parquet/*.parquet     # 3 tables (<1 MB)
│   ├── retractionwatch/
│   │   ├── parquet/*.parquet     # 1 table (70 MB)
│   │   └── meta.json
│   ├── reliance_on_science/
│   │   ├── parquet/*.parquet     # 3 tables (2.7 GB)
│   │   └── meta.json
│   └── preprint_to_paper/
│       ├── parquet/*.parquet     # 2 tables (735 MB)
│       └── meta.json
└── scripts/
    ├── config.py                 # Path resolution
    ├── create_unified_db.py      # Generates datalake.duckdb
    ├── datalake_cli.py           # Master CLI
    ├── download_openalex.py      # S3 sync downloader
    ├── convert_openalex.py       # NDJSON → Parquet converter (with --compact)
    ├── download_s2ag.py          # S2AG API downloader
    ├── convert_s2ag.py           # S2AG conversion
    └── download_sciscinet.py     # GCS downloader
```

---

## Update History

| Date | Dataset | Action |
|------|---------|--------|
| 2025-12-05 | S2AG | Full download and parquet conversion |
| 2024-11-01 | SciSciNet | Full download from GCS |
| 2026-02-14 | All | Restructured into unified data lake |
| 2026-02-16 | OpenAlex | Snapshot download (release 2026-02-03), reference entities + authors converted |
| 2026-02-16 | PWC + CSO | Integrated Papers With Code (24 views) and CSO 3.5 (3 views) into datalake |
| 2026-02-16 | Retraction Watch | Downloaded and converted 68.9K retraction records |
| 2026-02-16 | Reliance on Science | Downloaded v64, converted 47.8M patent-paper citations + 548K curated pairs |
| 2026-02-16 | PreprintToPaper | Downloaded and converted 145.5K preprint-publication mappings |
| 2026-02-16 | DOID + EDAM | Converted Disease Ontology (14.5K terms) and EDAM (3.5K terms) to parquet |
| 2026-02-17 | OpenAlex | All 23 tables converted (479M works, 3B refs, 1.3B authorships). Compaction in progress. |
