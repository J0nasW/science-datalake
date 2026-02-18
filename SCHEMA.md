# Science Data Lake - Schema Reference

> LLM-optimized reference for querying the science data lake.
> Connection: `duckdb.connect('datalake.duckdb', read_only=True)`
> Schemas: `s2ag`, `sciscinet`, `openalex`, `pwc`, 13 ontology schemas (`mesh`, `go`, `chebi`, `ncit`, `hpo`, `agrovoc`, `cso`, `doid`, `edam`, `stw`, `msc2020`, `physh`, `unesco`), `retwatch`, `ros`, `p2p`, `xref` | 148 views (including 8 backward-compat aliases in `main`)

---

## Table Index

| Schema.View | Rows | Size | Purpose |
|-------------|------|------|---------|
| **s2ag.papers** | 231M | VL | Core paper metadata with DOI, citations, venue, authors |
| **s2ag.abstracts** | 37M | L | Paper abstracts (16% coverage) |
| **s2ag.authors** | 112M | VL | Author profiles with h-index, affiliations |
| **s2ag.citations** | 2.9B | VL | Citation edges with context sentences and intents |
| **s2ag.paper_ids** | 519M | VL | SHA hash to corpusid mappings |
| **s2ag.publication_venues** | 195K | S | Journal/conference metadata |
| **s2ag.tldrs** | 70M | L | AI-generated one-sentence summaries |
| **s2ag.s2orc** | 12M | L | Full paper text (open access) |
| **s2ag.paper_authors** | derived | VL | Unnested paper-author pairs |
| **s2ag.paper_fields** | derived | VL | Unnested paper-field pairs |
| **sciscinet.papers** | 250M | VL | Scientometric indicators: disruption, atypicality, sleeping beauty |
| **sciscinet.paper_details** | 250M | VL | Title, abstract, language, validity flag |
| **sciscinet.papers_english** | ~162M | VL | Convenience: English papers with valid abstracts |
| **sciscinet.recent_papers** | varies | VL | Convenience: papers from 2020+ |
| **sciscinet.high_impact_papers** | varies | VL | Convenience: top 1% papers |
| **sciscinet.authors** | 100M | VL | Author h-index, productivity, avg citations |
| **sciscinet.author_details** | 100M | VL | ORCID, institution, alternative names |
| **sciscinet.author_papers** | 773M | VL | Author-paper mapping |
| **sciscinet.affiliations** | 111K | S | Institution metadata |
| **sciscinet.affl_assoc_affl** | 39K | S | Institution relationships |
| **sciscinet.us_institutions** | 31K | S | Convenience: US institutions only |
| **sciscinet.fields** | 303 | S | Field hierarchy (level 0 and 1) |
| **sciscinet.sources** | 261K | S | Publication venues |
| **sciscinet.funders** | 32K | S | Funding organizations |
| **sciscinet.paper_refs** | 2.5B | VL | Citation edges with year difference |
| **sciscinet.citation_edges** | derived | VL | Convenience: source/target format |
| **sciscinet.paper_fields** | 1.3B | VL | Paper-field mappings with scores |
| **sciscinet.paper_authors** | 773M | VL | Paper-author-affiliation |
| **sciscinet.paper_sources** | 204M | VL | Paper-venue mappings with OA info |
| **sciscinet.papers_pmid_pmcid** | 6.3M | M | PubMed/PMC identifier mappings |
| **sciscinet.hit_papers** | 570M | VL | Top 1%/5%/10% by field-year (level 0) |
| **sciscinet.hit_papers_level1** | 702M | VL | Top 1%/5%/10% by sub-field-year |
| **sciscinet.normalized_citations** | 570M | VL | Field-year normalized citations (level 0) |
| **sciscinet.normalized_citations_level1** | 702M | VL | Sub-field-year normalized citations |
| **sciscinet.link_nih** | 6.5M | M | Paper-NIH funding links |
| **sciscinet.link_nsf** | 1.8M | M | Paper-NSF funding links |
| **sciscinet.link_patents** | 47.8M | L | Paper-patent links |
| **sciscinet.link_twitter** | 55.8M | L | Paper-tweet links |
| **sciscinet.link_clinicaltrials** | 613K | S | Paper-clinical trial links |
| **sciscinet.link_newsfeed** | 1.4M | M | Paper-news mention links |
| **sciscinet.link_nobellaureates** | 87K | S | Paper-Nobel laureate links |
| **sciscinet.nih_metadata** | 534K | S | NIH award metadata |
| **sciscinet.nsf_metadata** | 460K | S | NSF award metadata |
| **sciscinet.clinicaltrials_metadata** | 273K | S | Clinical trial reference metadata |
| **sciscinet.newsfeed_metadata** | 1.4M | M | News mention metadata |
| **sciscinet.twitter_metadata** | 59.6M | L | Tweet metadata |
| **openalex.works** | 479M | VL | Paper metadata: title, DOI, abstract, year, citations, type |
| **openalex.works_authorships** | 1.32B | VL | Author-paper relationships with institutions |
| **openalex.works_referenced_works** | 3.01B | VL | Citation edges (outgoing references) |
| **openalex.works_topics** | 910M | VL | Topic assignments with scores |
| **openalex.works_locations** | 612M | VL | Hosting venues with OA status |
| **openalex.works_ids** | 479M | VL | External IDs (DOI, PMID, PMC, MAG) per work |
| **openalex.works_biblio** | 479M | VL | Bibliographic info: volume, issue, pages |
| **openalex.works_open_access** | 479M | VL | OA status, OA URL, license |
| **openalex.works_best_oa_location** | 210M | VL | Best open access location per work |
| **openalex.works_concepts** | 5.04B | VL | Concept assignments with scores (deprecated, use topics) |
| **openalex.works_keywords** | 2.98B | VL | Author-assigned keyword tags per work |
| **openalex.works_related_works** | 2.49B | VL | Related-works edges per work |
| **openalex.works_counts_by_year** | 443M | VL | Per-year citation counts per work |
| **openalex.authors** | 108M | VL | Author profiles with h-index, institutions |
| **openalex.authors_ids** | 108M | VL | Author external IDs (ORCID) |
| **openalex.authors_counts_by_year** | 302M | VL | Per-year author metrics |
| **openalex.awards** | 11.7M | L | Funding awards with amounts, PIs |
| **openalex.topics** | 4,516 | S | Topic hierarchy (topic->subfield->field->domain) |
| **openalex.domains** | 4 | S | Top-level: Life/Physical/Health/Social Sciences |
| **openalex.fields** | 26 | S | Mid-level fields (Computer Science, Medicine, etc.) |
| **openalex.subfields** | 252 | S | Fine-grained subfields |
| **openalex.publishers** | 10.7K | S | Publishing organizations |
| **openalex.funders** | 32K | S | Funding organizations |
| **openalex.concepts** | 65K | S | Concept taxonomy (deprecated, use topics) |
| **openalex.sources** | 255K | S | Journals, conferences, repositories |
| **openalex.institutions** | 121K | S | Institutions with geo coordinates |
| **pwc.papers** | 513K | S | ML paper metadata: title, abstract, arXiv ID |
| **pwc.methods** | 2.3K | S | ML/AI method catalog |
| **pwc.tasks** | 5.3K | S | Research task catalog |
| **pwc.datasets** | 11K | S | Dataset catalog with modalities |
| **pwc.paper_uses_method** | 621K | S | Paper→method usage links |
| **pwc.paper_has_task** | 951K | S | Paper→task links |
| **pwc.paper_has_code** | 269K | S | Paper→GitHub repo links |
| **pwc.code_repos** | 215K | S | Repo metadata: framework, official status |
| **pwc.paper_has_openalexWorkID** | 407K | S | Paper→OpenAlex work ID mapping |
| **pwc.paper_has_openalexAuthorID** | 1.4M | M | Paper→OpenAlex author ID |
| **pwc.paper_has_embedding** | 513K | S | Dense paper embeddings (2.4 GB) |
| **pwc.papers_fulltexts** | 203K | S | Full text with S2ORC corpus ID and DOI |
| **pwc.paper_links** | 512K | S | Paper→SemOpenAlex URL mapping |
| **pwc.patent_cites_paper** | 156K | S | Patent→paper citations |
| **pwc.patents** | 81K | S | Patent metadata |
| **pwc.paper_introduces_method** | 2.1K | S | Method provenance |
| **pwc.paper_introduces_dataset** | 8.6K | S | Dataset provenance |
| **pwc.dataset_has_task** | 18K | S | Dataset→task links |
| **pwc.cso_topics** | 6.4K | S | CSO topic vocabulary |
| **pwc.areas** | 16 | S | Top-level research areas |
| **pwc.collections** | 362 | S | Method collections |
| **pwc.collection_isin_area** | 362 | S | Collection→area mapping |
| **pwc.method_isin_collection** | 2.8K | S | Method→collection mapping |
| **pwc.conferences** | 1.8K | S | Conference proceedings |
| **pwc.paper_isin_conference** | 103K | S | Paper→conference links |
| **pwc.patent_paper_pairs** | 3K | S | Patent-paper similarity pairs |
| **pwc.papers_openalex** | 407K | S | Papers with OpenAlex work IDs (convenience view) |
| **pwc.book_indices** | 3.2K | S | Textbook keyword index |
| **mesh.mesh_terms** | 720,801 | M | MeSH biomedical vocabulary (terms only, no hierarchy/xrefs) |
| **chebi.chebi_terms** | 205,317 | S | Chemical Entities of Biological Interest |
| **chebi.chebi_hierarchy** | 379,841 | S | ChEBI parent-child edges |
| **chebi.chebi_xrefs** | 389,000 | S | ChEBI cross-references |
| **ncit.ncit_terms** | 203,668 | S | NCI Thesaurus cancer/biomedical terms |
| **ncit.ncit_hierarchy** | 293,258 | S | NCIT parent-child edges |
| **ncit.ncit_xrefs** | 2,250 | S | NCIT cross-references |
| **go.go_terms** | 47,856 | S | Gene Ontology (MF, BP, CC) |
| **go.go_hierarchy** | 81,173 | S | GO parent-child edges |
| **go.go_xrefs** | 25,868 | S | GO cross-references |
| **agrovoc.agrovoc_terms** | 41,699 | S | AGROVOC agriculture thesaurus |
| **agrovoc.agrovoc_hierarchy** | 42,132 | S | AGROVOC parent-child edges |
| **agrovoc.agrovoc_xrefs** | 49,762 | S | AGROVOC cross-references |
| **hpo.hpo_terms** | 19,934 | S | Human Phenotype Ontology |
| **hpo.hpo_hierarchy** | 23,765 | S | HPO parent-child edges |
| **hpo.hpo_xrefs** | 18,099 | S | HPO cross-references |
| **cso.cso_terms** | 14,636 | S | CSO topic catalog (standardized schema) |
| **cso.cso_hierarchy** | 93,491 | S | CSO parent->child edges (superTopicOf + contributesTo) |
| **cso.cso_xrefs** | 28,100 | S | External links: sameAs, Wikidata, Wikipedia |
| **doid.doid_terms** | 14,521 | S | Disease Ontology term catalog |
| **doid.doid_hierarchy** | 16,916 | S | DOID parent-child edges |
| **doid.doid_xrefs** | 38,653 | S | DOID cross-references |
| **stw.stw_terms** | 7,858 | S | STW economics thesaurus |
| **stw.stw_hierarchy** | 28,702 | S | STW parent-child edges |
| **msc2020.msc2020_terms** | 6,603 | S | Mathematics Subject Classification 2020 |
| **msc2020.msc2020_hierarchy** | 6,603 | S | MSC2020 parent-child edges |
| **unesco.unesco_terms** | 4,498 | S | UNESCO education/science thesaurus |
| **unesco.unesco_hierarchy** | 8,682 | S | UNESCO parent-child edges |
| **physh.physh_terms** | 3,925 | S | Physics Subject Headings |
| **physh.physh_hierarchy** | 8,844 | S | PhySH parent-child edges |
| **edam.edam_terms** | 3,524 | S | EDAM bioinformatics ontology catalog |
| **edam.edam_hierarchy** | 5,219 | S | EDAM parent-child edges |
| **edam.edam_xrefs** | 266 | S | EDAM cross-references |
| **retwatch.retraction_watch** | 69K | S | Retraction/correction records with DOIs and reasons |
| **ros.pcs_oa** | 47.8M | L | Patent citation-to-science pairs (global) |
| **ros.patent_paper_pairs** | 548K | S | Curated patent-paper similarity pairs |
| **ros.patent_paper_pairs_plus** | 548K | S | Extended pairs with institutional metadata |
| **p2p.preprint_to_paper** | 146K | S | Preprint→publication DOI mapping with status |
| **p2p.preprint_to_paper_grayzone** | 299 | S | Manually annotated suspected matches |
| **xref.doi_map** | 588M | VL | Cross-dataset DOI lookup (normalized, no prefix) |
| **xref.unified_papers** | 293M | VL | Pre-joined cross-source paper table with coverage flags |
| **xref.topic_ontology_map** | 16.2K | S | OpenAlex topic → ontology term alignment via BGE-large-en-v1.5 embeddings (cosine sim ≥ 0.65) + exact matching; 99.8% topic coverage |
| **xref.ontology_bridges** | 1.8K | S | Cross-ontology term links via shared external IDs (UMLS, Wikidata, etc.) |

Size tiers: S=<1M, M=1-10M, L=10-100M, VL=>100M rows

---

## Cross-Dataset Join Strategies

### Strategy 1: DOI Join (S2AG <-> SciSciNet)

S2AG stores DOIs without prefix, SciSciNet stores them with `https://doi.org/` prefix. Strip the prefix to join:

```sql
SELECT s.title, s.citationcount AS s2ag_cites,
       sc.disruption, sc.C10, sc.patent_count
FROM s2ag.papers s
JOIN sciscinet.papers sc ON s.doi = REPLACE(sc.doi, 'https://doi.org/', '')
WHERE s.doi = '10.1038/nature12373';
```

### Strategy 2: paperid = OpenAlex work ID (SciSciNet <-> OpenAlex)

SciSciNet `paperid` (e.g. `W2100837269`) is the short form of OpenAlex `id` (e.g. `https://openalex.org/W2100837269`). To join:

```sql
SELECT sc.disruption, oa.cited_by_count, oa.title
FROM sciscinet.papers sc
JOIN openalex.works oa ON 'https://openalex.org/' || sc.paperid = oa.id
WHERE sc.doi = 'https://doi.org/10.1038/nature12373';
```

### Strategy 3: xref.doi_map (any combination)

The `xref.doi_map` view normalizes DOIs from all datasets to lowercase, no-prefix format:

```sql
-- Find all dataset IDs for a DOI
SELECT * FROM xref.doi_map WHERE doi = '10.1038/nature12373';
-- Returns: source='s2ag', doi='10.1038/nature12373', source_id='12345678'
--          source='sciscinet', doi='10.1038/nature12373', source_id='W2100837269'

-- Three-way join via DOI
SELECT s.title, s.citationcount, sc.disruption
FROM xref.doi_map xs
JOIN xref.doi_map xsc ON xs.doi = xsc.doi AND xsc.source = 'sciscinet'
JOIN s2ag.papers s ON xs.source_id = CAST(s.corpusid AS VARCHAR)
JOIN sciscinet.papers sc ON xsc.source_id = sc.paperid
WHERE xs.source = 's2ag' AND xs.doi = '10.1038/nature12373';
```

**Performance note:** `xref.doi_map` is a UNION ALL view that scans papers tables lazily. For repeated lookups, filter early with WHERE doi = ... to avoid full scans.

### Strategy 4: PWC → OpenAlex (via work ID)

```sql
-- PWC papers matched to OpenAlex works
SELECT p.title, w.cited_by_count, w.publication_year
FROM pwc.papers p
JOIN pwc.paper_has_openalexWorkID poa ON p.paper_id = poa.paper_id
JOIN openalex.works w ON 'https://openalex.org/' || poa.openalex_work_id = w.id
LIMIT 20;
```

### Strategy 5: PWC → S2AG (via S2ORC corpus ID)

```sql
-- PWC papers matched to S2AG via corpus ID (most reliable)
SELECT p.title AS pwc_title, s.citationcount, s.year
FROM pwc.papers p
JOIN pwc.papers_fulltexts ft ON p.paper_id = ft.paper_id
JOIN s2ag.papers s ON ft.s2orc_corpus_id = s.corpusid;
```

### Strategy 6: PWC → CSO (topic name join)

```sql
-- Find which CSO subtopics are referenced in PWC
-- PWC has cso_topics table with topic names, CSO uses URI-based IDs
SELECT ct.cso_topics AS topic, t.label, h.parent_id
FROM pwc.cso_topics ct
JOIN cso.cso_terms t ON t.label = ct.cso_topics
JOIN cso.cso_hierarchy h ON h.child_id = t.id AND h.relation = 'superTopicOf';
```

### Strategy 7: Retraction Watch → Any Dataset (DOI)

```sql
-- Flag retracted papers (DOI is already lowercase, no prefix)
SELECT p.title, rw.retraction_nature, rw.reason
FROM s2ag.papers p
JOIN retwatch.retraction_watch rw ON p.doi = rw.original_paper_doi;
```

### Strategy 8: Reliance on Science → SciSciNet/OpenAlex (OpenAlex ID)

```sql
-- pcs_oa → SciSciNet (bare numeric oaid needs W prefix)
SELECT sc.paperid, sc.disruption, COUNT(*) AS patent_cites
FROM ros.pcs_oa r
JOIN sciscinet.papers sc ON 'W' || CAST(r.oaid AS VARCHAR) = sc.paperid
GROUP BY sc.paperid, sc.disruption ORDER BY patent_cites DESC LIMIT 20;

-- pcs_oa → OpenAlex works (needs full URL)
SELECT oa.title, oa.cited_by_count, COUNT(*) AS patent_cites
FROM ros.pcs_oa r
JOIN openalex.works oa ON 'https://openalex.org/W' || CAST(r.oaid AS VARCHAR) = oa.id
GROUP BY oa.title, oa.cited_by_count ORDER BY patent_cites DESC LIMIT 20;

-- patent_paper_pairs → SciSciNet (W-prefixed, direct join)
SELECT pp.patent, sc.disruption, sc.cited_by_count
FROM ros.patent_paper_pairs pp
JOIN sciscinet.papers sc ON pp.paperid = sc.paperid;
```

### Strategy 9: PreprintToPaper → Any Dataset (DOI)

```sql
-- Link preprints to their published versions in S2AG
SELECT p2p.biorxiv_doi, p2p.custom_status, s.citationcount
FROM p2p.preprint_to_paper p2p
JOIN s2ag.papers s ON p2p.biorxiv_published_doi = s.doi
WHERE p2p.custom_status = 'published';
```

---

## S2AG (Semantic Scholar Academic Graph)

**Primary key:** `corpusid` (BIGINT) | **DOI format:** lowercase, no prefix | **License:** Proprietary

### s2ag.papers (231M rows, VERY_LARGE)

Core paper metadata. The view flattens `externalids` and `journal` structs into top-level columns.

| Column | Type | Description |
|--------|------|-------------|
| `corpusid` | BIGINT | **Primary key.** Unique S2AG paper identifier. |
| `title` | VARCHAR | Paper title |
| `year` | BIGINT | Publication year. NULL for some old papers. |
| `citationcount` | BIGINT | Total incoming citations in S2AG. Differs from SciSciNet/OpenAlex counts. |
| `influentialcitationcount` | BIGINT | Influential citations per S2AG model (~5% are influential). |
| `isopenaccess` | BOOLEAN | Whether paper is open access |
| `doi` | VARCHAR | DOI (lowercase, NO prefix). Flattened from `externalids.DOI`. Use for cross-dataset joins. |
| `venue` | VARCHAR | Venue name (nullable) |
| `publicationvenueid` | UUID | FK to `publication_venues.id` (nullable, many papers lack this) |
| `publicationdate` | DATE | Full publication date (nullable) |
| `publicationtypes` | VARCHAR[] | JournalArticle, Conference, Review, CaseReport, etc. |
| `referencecount` | BIGINT | Number of outgoing references |
| `authors` | STRUCT[] | Array of {authorId, name}. Use `paper_authors` view for unnested access. |
| `s2fieldsofstudy` | STRUCT[] | Fields with source. Use `paper_fields` view for unnested access. |
| `journal_name` | VARCHAR | Journal name (flattened from journal struct) |
| `journal_pages` | VARCHAR | Page range |
| `journal_volume` | VARCHAR | Volume number |
| `mag_id` | VARCHAR | Microsoft Academic Graph ID |
| `pubmed_id` | VARCHAR | PubMed ID |
| `arxiv_id` | VARCHAR | ArXiv ID |
| `acl_id` | VARCHAR | ACL Anthology ID |
| `dblp_id` | VARCHAR | DBLP ID |
| `pmc_id` | VARCHAR | PubMed Central ID |
| `externalids` | STRUCT | Original external IDs struct. Prefer flattened columns. |

```sql
-- Find paper by DOI
SELECT corpusid, title, year, citationcount FROM s2ag.papers
WHERE doi = '10.1038/nature12373';

-- Top cited papers in a year
SELECT title, citationcount FROM s2ag.papers
WHERE year = 2024 ORDER BY citationcount DESC LIMIT 20;
```

### s2ag.abstracts (37M rows, LARGE)

Only ~16% of papers have abstracts. View flattens `openaccessinfo` struct.

| Column | Type | Description |
|--------|------|-------------|
| `corpusid` | BIGINT | FK to `papers.corpusid` |
| `abstract` | VARCHAR | Full abstract text |
| `oa_disclaimer` | VARCHAR | OA disclaimer (nullable) |
| `oa_license` | VARCHAR | License string (nullable) |
| `oa_url` | VARCHAR | OA URL (nullable) |
| `oa_status` | VARCHAR | OA status (nullable) |
| `doi`, `medline_id`, `pmc_id`, `mag_id`, `arxiv_id`, `medrxiv_id` | VARCHAR | External IDs from OA info (all nullable) |

```sql
-- Papers with abstracts
SELECT p.title, a.abstract FROM s2ag.papers p
JOIN s2ag.abstracts a ON p.corpusid = a.corpusid
WHERE p.year = 2023 LIMIT 10;
```

### s2ag.citations (2.9B rows, VERY_LARGE)

Citation edges with context sentences and intents. **Always filter with WHERE.**

| Column | Type | Description |
|--------|------|-------------|
| `citationid` | BIGINT | Unique edge ID |
| `citingcorpusid` | BIGINT | FK to papers. The paper doing the citing. |
| `citedcorpusid` | BIGINT | FK to papers. The paper being cited. |
| `isinfluential` | BOOLEAN | Influential citation flag (~5% are true) |
| `contexts` | VARCHAR[] | Sentences surrounding the citation. **Unique to S2AG.** |
| `intents` | VARCHAR[][] | Per-context intents: Background, Methodology, ResultComparison. **Unique to S2AG.** |

```sql
-- Get citation contexts for a specific paper
SELECT c.citingcorpusid, c.contexts, c.intents
FROM s2ag.citations c
WHERE c.citedcorpusid = 12345678 AND c.isinfluential = true;
```

### s2ag.authors (112M rows, VERY_LARGE)

| Column | Type | Description |
|--------|------|-------------|
| `authorid` | VARCHAR | **Primary key.** Links to `paper_authors.authorid`. |
| `name` | VARCHAR | Display name |
| `aliases` | VARCHAR[] | Alternative spellings |
| `affiliations` | VARCHAR[] | Current affiliations |
| `papercount` | BIGINT | Total papers |
| `citationcount` | BIGINT | Total citations |
| `hindex` | BIGINT | H-index |

### s2ag.tldrs (70M rows, LARGE)

AI-generated one-sentence summaries. 30% coverage.

| Column | Type | Description |
|--------|------|-------------|
| `corpusid` | BIGINT | FK to papers |
| `model` | VARCHAR | Model used |
| `text` | VARCHAR | One-sentence summary |

### s2ag.s2orc (12M rows, LARGE)

Full paper text for open access papers. View flattens body/bibliography structs.

| Column | Type | Description |
|--------|------|-------------|
| `corpusid` | BIGINT | FK to papers |
| `body_text` | VARCHAR | Full body text (can be very long) |
| `bibliography_text` | VARCHAR | Full bibliography text |
| `license` | VARCHAR | License |

### s2ag.paper_authors (derived, VERY_LARGE)

Unnested from `papers.authors`. Computed on-the-fly (slower than base tables).

| Column | Type | Description |
|--------|------|-------------|
| `corpusid` | BIGINT | FK to papers |
| `authorid` | VARCHAR | FK to authors |
| `author_name` | VARCHAR | Name on this paper |
| `author_position` | BIGINT | 1-based position in author list |

### s2ag.paper_fields (derived, VERY_LARGE)

Unnested from `papers.s2fieldsofstudy` with DISTINCT.

| Column | Type | Description |
|--------|------|-------------|
| `corpusid` | BIGINT | FK to papers |
| `field_category` | VARCHAR | Field name (e.g., 'Computer Science') |
| `field_source` | VARCHAR | 's2-fos-model' or 'external' |

### s2ag.publication_venues (195K rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | **Primary key.** Papers link via `publicationvenueid`. |
| `name` | VARCHAR | Venue name |
| `type` | VARCHAR | journal, conference, etc. |
| `issn` | VARCHAR | ISSN (nullable) |

### s2ag.paper_ids (519M rows, VERY_LARGE)

SHA hash to corpusid mapping.

| Column | Type | Description |
|--------|------|-------------|
| `sha` | VARCHAR | PDF content hash |
| `corpusid` | BIGINT | FK to papers |
| `is_primary` | BOOLEAN | Primary hash for paper |

---

## SciSciNet v2

**Primary key:** `paperid` (VARCHAR, OpenAlex work ID like `W2100837269`) | **DOI format:** lowercase, WITH `https://doi.org/` prefix | **License:** CC BY 4.0

### sciscinet.papers (250M rows, VERY_LARGE)

Core scientometric data. Every column serves a specific analytical purpose.

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | **Primary key.** OpenAlex work ID. Same as `openalex.works.id` (without URL prefix). |
| `doi` | VARCHAR | DOI (lowercase, WITH `https://doi.org/` prefix). To match S2AG: `REPLACE(doi, 'https://doi.org/', '')`. |
| `year` | BIGINT | Publication year |
| `cited_by_count` | BIGINT | Citations from OpenAlex. Differs from S2AG `citationcount`. |
| `citation_count` | UINTEGER | SciSciNet-computed citation count (may differ from `cited_by_count`) |
| `C3` | UINTEGER | Citations within 3 years. Early impact. |
| `C5` | UINTEGER | Citations within 5 years. Standard short-term. |
| `C10` | UINTEGER | Citations within 10 years. Standard long-term. |
| `disruption` | DOUBLE | CD disruption index (-1 to 1). Positive=disruptive, negative=consolidating. NULL if insufficient data. |
| `Atyp_Median_Z` | DOUBLE | Atypicality median z-score. Higher=more novel journal pair combinations. |
| `Atyp_10pct_Z` | DOUBLE | Atypicality 10th percentile. Captures most unusual pair. |
| `Atyp_Pairs` | BIGINT | Number of journal pairs for atypicality |
| `WSB_mu` | DOUBLE | Sleeping Beauty: mean annual citations before awakening |
| `WSB_sigma` | DOUBLE | Sleeping Beauty: std dev of citations |
| `WSB_Cinf` | DOUBLE | Sleeping Beauty: asymptotic citation level |
| `SB_B` | DOUBLE | Beauty coefficient. Higher=more delayed recognition. |
| `SB_T` | BIGINT | Awakening time in years after publication |
| `team_size` | UINTEGER | Number of authors |
| `institution_count` | UINTEGER | Number of distinct institutions |
| `patent_count` | UINTEGER | Citing patents. Technological impact. |
| `newsfeed_count` | UINTEGER | News mentions. Media attention. |
| `nct_count` | UINTEGER | Clinical trial links. Clinical translation. |
| `nih_count` | UINTEGER | NIH funding links |
| `nsf_count` | UINTEGER | NSF funding links |

```sql
-- Most disruptive papers in 2020
SELECT p.paperid, d.title, p.disruption, p.cited_by_count
FROM sciscinet.papers p
JOIN sciscinet.paper_details d ON p.paperid = d.paperid
WHERE p.year = 2020 AND p.disruption IS NOT NULL
ORDER BY p.disruption DESC LIMIT 20;

-- Sleeping beauties: high beauty coefficient, published before 2000
SELECT paperid, year, SB_B, SB_T, cited_by_count
FROM sciscinet.papers
WHERE SB_B > 10 AND year < 2000 AND cited_by_count > 50
ORDER BY SB_B DESC LIMIT 20;
```

### sciscinet.paper_details (250M rows, VERY_LARGE)

Title, abstract, language. JOIN with `papers` for full picture.

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | **Primary key** |
| `title` | VARCHAR | Paper title |
| `abstract` | VARCHAR | Plain text abstract (converted from inverted index, nullable) |
| `language` | VARCHAR | ISO 639-1 code |
| `valid_title_abstract` | BOOLEAN | True if: English, title>=10 chars, abstract>=50 chars, >80% ASCII, >=10 words |
| `display_name` | VARCHAR | Display name |
| `is_paratext` | BOOLEAN | Editorial, TOC, etc. |
| `is_retracted` | BOOLEAN | Retracted flag |

```sql
-- English papers with quality abstracts for NLP
SELECT paperid, title, abstract FROM sciscinet.paper_details
WHERE valid_title_abstract = true AND year >= 2020
LIMIT 100;
```

### sciscinet.paper_refs (2.5B rows, VERY_LARGE)

Citation network edges. **Always filter.**

| Column | Type | Description |
|--------|------|-------------|
| `citing_paperid` | VARCHAR | FK to papers |
| `cited_paperid` | VARCHAR | FK to papers |
| `year` | BIGINT | Year of citing paper |
| `ref_year` | BIGINT | Year of cited paper |
| `year_diff` | BIGINT | citing year - cited year |

### sciscinet.hit_papers (570M rows, VERY_LARGE)

Top 1%/5%/10% papers by field-year at level 0.

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | FK to papers |
| `fieldid` | VARCHAR | FK to `fields.fieldid` |
| `Hit_1pct` | INTEGER | 1 if top 1% in field-year |
| `Hit_5pct` | INTEGER | 1 if top 5% |
| `Hit_10pct` | INTEGER | 1 if top 10% |

### sciscinet.normalized_citations (570M rows, VERY_LARGE)

Field-year normalized citation scores.

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | FK to papers |
| `fieldid` | VARCHAR | FK to fields |
| `normalized_citations` | DOUBLE | Citations / field-year average. >1 = above average. |
| `c0` | DOUBLE | Baseline for field-year |

### sciscinet.paper_authors (773M rows, VERY_LARGE)

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | FK to papers |
| `authorid` | VARCHAR | FK to authors |
| `author_position` | VARCHAR | 'first', 'middle', or 'last' |
| `institutionid` | VARCHAR | FK to affiliations (nullable) |
| `raw_affiliation_string` | VARCHAR | Raw affiliation text |

### sciscinet.paper_fields (1.3B rows, VERY_LARGE)

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | FK to papers |
| `fieldid` | VARCHAR | FK to `fields.fieldid` |
| `score_openalex` | DOUBLE | Relevance score (0-1) |

### sciscinet.authors (100M rows, VERY_LARGE)

| Column | Type | Description |
|--------|------|-------------|
| `authorid` | VARCHAR | **Primary key.** OpenAlex author ID. |
| `display_name` | VARCHAR | Name |
| `h_index` | UINTEGER | H-index |
| `productivity` | UINTEGER | Paper count |
| `avg_c10` | DOUBLE | Average 10-year citations |
| `P(gf)` | DOUBLE | Gender inference probability (female) |

### sciscinet.link_* tables (funding, patents, social)

| Table | Rows | Key Columns |
|-------|------|-------------|
| `link_nih` | 6.5M | paperid, award_id |
| `link_nsf` | 1.8M | paperid, award_id |
| `link_patents` | 47.8M | paperid, patent, uspto |
| `link_twitter` | 55.8M | paperid, tweet_id |
| `link_clinicaltrials` | 613K | paperid, nct_id |
| `link_newsfeed` | 1.4M | paperid, newsfeed_id |
| `link_nobellaureates` | 87K | paperid, laureate_id |

Metadata tables: `nih_metadata`, `nsf_metadata`, `clinicaltrials_metadata`, `newsfeed_metadata`, `twitter_metadata` provide details for the link IDs.

### Convenience Views

| View | Filter | Source |
|------|--------|--------|
| `papers_english` | `valid_title_abstract = true` | paper_details |
| `recent_papers` | `year >= 2020` | papers |
| `high_impact_papers` | `Hit_1pct = 1` | papers JOIN hit_papers |
| `citation_edges` | renames to source/target | paper_refs |
| `us_institutions` | `country_code = 'US'` | affiliations |

---

## OpenAlex

**Primary key:** `id` (VARCHAR, URL format) | **DOI format:** lowercase, WITH `https://doi.org/` prefix | **License:** CC0

### openalex.works (479M rows, VERY_LARGE)

Core paper metadata. 23 tables total spanning works, authors, and reference entities.

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | **Primary key.** OpenAlex work ID (URL format, e.g. `https://openalex.org/W2100837269`). |
| `doi` | VARCHAR | DOI with `https://doi.org/` prefix (nullable). |
| `title` | VARCHAR | Paper title |
| `publication_year` | INTEGER | Publication year |
| `publication_date` | DATE | Full publication date (nullable) |
| `type` | VARCHAR | Work type: article, book-chapter, dissertation, etc. |
| `cited_by_count` | BIGINT | Total citations in OpenAlex |
| `is_retracted` | BOOLEAN | Retraction flag |
| `is_paratext` | BOOLEAN | Editorial, TOC, etc. |
| `language` | VARCHAR | ISO 639-1 language code |
| `abstract` | VARCHAR | Plain text abstract (nullable) |
| `valid_title_abstract` | BOOLEAN | True if: English, title>=10 chars, abstract>=50 chars, >80% ASCII, >=10 words |
| `created_date` | DATE | Date record was created in OpenAlex |
| `updated_date` | DATE | Date record was last updated |

```sql
-- Find paper by DOI
SELECT id, title, publication_year, cited_by_count FROM openalex.works
WHERE doi = 'https://doi.org/10.1038/nature12373';

-- Top-cited works in a year
SELECT title, cited_by_count FROM openalex.works
WHERE publication_year = 2024 ORDER BY cited_by_count DESC LIMIT 20;
```

### openalex.works_authorships (1.32B rows, VERY_LARGE)

Author-paper relationships. **Always filter.**

| Column | Type | Description |
|--------|------|-------------|
| `work_id` | VARCHAR | FK to `works.id` |
| `author_id` | VARCHAR | FK to `authors.id` |
| `raw_author_name` | VARCHAR | Name as it appears on the paper |
| `is_corresponding` | BOOLEAN | Whether this is the corresponding author |
| `institution_ids` | VARCHAR[] | Affiliated institutions |
| `raw_affiliation_strings` | VARCHAR[] | Raw affiliation text |

### openalex.works_referenced_works (3.01B rows, VERY_LARGE)

Citation edges (outgoing references). **Always filter.**

| Column | Type | Description |
|--------|------|-------------|
| `work_id` | VARCHAR | FK to `works.id` (the citing paper) |
| `referenced_work_id` | VARCHAR | FK to `works.id` (the cited paper) |

### openalex.works_topics (910M rows, VERY_LARGE)

| Column | Type | Description |
|--------|------|-------------|
| `work_id` | VARCHAR | FK to `works.id` |
| `topic_id` | VARCHAR | FK to `topics.id` |
| `score` | DOUBLE | Relevance score (0-1) |
| `display_name` | VARCHAR | Topic name |
| `subfield_display_name` | VARCHAR | Parent subfield |
| `field_display_name` | VARCHAR | Parent field |
| `domain_display_name` | VARCHAR | Parent domain |

### openalex.works_locations (612M rows, VERY_LARGE)

| Column | Type | Description |
|--------|------|-------------|
| `work_id` | VARCHAR | FK to `works.id` |
| `source_id` | VARCHAR | FK to `sources.id` |
| `is_oa` | BOOLEAN | Whether this location is open access |
| `landing_page_url` | VARCHAR | Landing page URL |
| `pdf_url` | VARCHAR | Direct PDF URL (nullable) |
| `license` | VARCHAR | License string (nullable) |
| `version` | VARCHAR | publishedVersion, acceptedVersion, submittedVersion |

### Other OpenAlex Works Tables

| Table | Rows | Key Columns | Purpose |
|-------|------|-------------|---------|
| `works_concepts` | 5.04B | work_id, concept_id, score | Concept assignments (deprecated, use topics) |
| `works_keywords` | 2.98B | work_id, keyword, score | Author-assigned keyword tags |
| `works_related_works` | 2.49B | work_id, related_work_id | Related-works edges |
| `works_ids` | 479M | work_id, doi, pmid, pmcid, mag | External ID crosswalk per work |
| `works_biblio` | 479M | work_id, volume, issue, first_page, last_page | Bibliographic details |
| `works_open_access` | 479M | work_id, is_oa, oa_status, oa_url | Open access status per work |
| `works_best_oa_location` | 210M | work_id, source_id, pdf_url, license | Best OA location (only for OA works) |
| `works_counts_by_year` | 443M | work_id, year, cited_by_count | Per-year citation counts |

> **Warning**: `works_concepts`, `works_keywords`, and `works_related_works` are among the largest tables in the data lake (2.5-5B rows each). Always filter by `work_id` before scanning.

### openalex.authors (108M rows, VERY_LARGE)

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | **Primary key.** OpenAlex author ID (URL format). |
| `display_name` | VARCHAR | Author name |
| `orcid` | VARCHAR | ORCID (nullable) |
| `display_name_alternatives` | VARCHAR[] | Alternative spellings |
| `works_count` | BIGINT | Total works |
| `cited_by_count` | BIGINT | Total citations |
| `h_index` | BIGINT | H-index |
| `i10_index` | BIGINT | i10-index |
| `mean_citedness_2yr` | DOUBLE | 2-year mean citedness |
| `last_known_institutions` | JSON | Most recent affiliations (JSON array) |

```sql
-- Top cited authors
SELECT display_name, works_count, cited_by_count, h_index
FROM openalex.authors ORDER BY cited_by_count DESC LIMIT 20;
```

### openalex.awards (11.7M rows, LARGE)

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | **Primary key** |
| `display_name` | VARCHAR | Award title |
| `amount` | DOUBLE | Award amount (nullable) |
| `currency` | VARCHAR | Currency code (nullable) |
| `funder` | STRUCT | {id, display_name, ror_id, doi} |
| `funded_outputs` | VARCHAR[] | Work IDs funded by this award |
| `funded_outputs_count` | BIGINT | Number of funded outputs |
| `start_date` | DATE | Start date (nullable) |
| `end_date` | DATE | End date (nullable) |
| `lead_investigator` | STRUCT | PI info (given_name, family_name, orcid, affiliation) |

### openalex.topics (4,516 rows, SMALL)

Topic hierarchy: domain -> field -> subfield -> topic.

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | Topic ID |
| `display_name` | VARCHAR | Topic name |
| `description` | VARCHAR | Description |
| `keywords` | VARCHAR[] | Associated keywords |
| `subfield_id`, `subfield_display_name` | VARCHAR | Parent subfield |
| `field_id`, `field_display_name` | VARCHAR | Parent field |
| `domain_id`, `domain_display_name` | VARCHAR | Parent domain |
| `works_count` | BIGINT | Works in this topic |

```sql
-- Topic hierarchy traversal
SELECT d.display_name AS domain, f.display_name AS field,
       sf.display_name AS subfield, t.display_name AS topic
FROM openalex.topics t
JOIN openalex.subfields sf ON t.subfield_id = sf.id
JOIN openalex.fields f ON sf.field_id = f.id
JOIN openalex.domains d ON f.domain_id = d.id
LIMIT 20;
```

### Other OpenAlex Reference Tables

| Table | Rows | Key Columns | Use For |
|-------|------|-------------|---------|
| `domains` | 4 | id, display_name | Top-level classification |
| `fields` | 26 | id, display_name, domain_id | Mid-level classification |
| `subfields` | 252 | id, display_name, field_id | Fine-grained classification |
| `publishers` | 10.7K | id, display_name, country_codes, works_count | Publisher lookup |
| `funders` | 32K | id, display_name, country_code, awards_count, works_count | Funder lookup |
| `concepts` | 65K | id, display_name, level (0-5) | Legacy concept taxonomy (deprecated) |
| `sources` | 255K | id, display_name, type, issn_l, is_oa | Journal/venue lookup |
| `institutions` | 121K | id, display_name, ror, country_code, geo_latitude, geo_longitude | Institution lookup with geocoding |
| `authors_ids` | 108M | author_id, openalex, orcid | Author ID crosswalk |
| `authors_counts_by_year` | 302M | author_id, year, works_count, cited_by_count | Author productivity trends |

---

## Papers With Code (PWC)

**Primary key:** `paper_id` (VARCHAR, slug like `attention-is-all-you-need`) | **DOI format:** lowercase, no prefix (in papers_fulltexts) | **License:** CC BY-SA 4.0

28 auto-discovered views. Key tables below.

### pwc.papers (513K rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `paper_id` | VARCHAR | **Primary key.** URL slug identifier. |
| `arxiv_id` | VARCHAR | ArXiv ID (nullable) |
| `title` | VARCHAR | Paper title |
| `abstract` | VARCHAR | Abstract text |
| `url_abs` | VARCHAR | Abstract page URL |
| `url_pdf` | VARCHAR | PDF URL |
| `date` | VARCHAR | Publication date |
| `proceeding` | VARCHAR | Conference proceeding (nullable) |

### pwc.methods (2.3K rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `method_id` | VARCHAR | **Primary key.** |
| `name` | VARCHAR | Short method name (e.g., "Batch Normalization") |
| `full_name` | VARCHAR | Full method name |
| `description` | VARCHAR | Method description |
| `introduced_year` | INTEGER | Year first introduced (nullable) |
| `num_papers` | INTEGER | Number of papers using this method |

### pwc.tasks (5.3K rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `task_id` | VARCHAR | **Primary key.** |
| `task_name` | VARCHAR | Task display name |
| `task_description` | VARCHAR | Task description |

### pwc.datasets (11K rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `dataset_id` | VARCHAR | **Primary key.** |
| `name` | VARCHAR | Dataset name |
| `full_name` | VARCHAR | Full name |
| `description` | VARCHAR | Description |
| `num_papers` | INTEGER | Papers using this dataset |
| `modalities` | VARCHAR | Data modalities |
| `languages` | VARCHAR | Supported languages |

### pwc.papers_fulltexts (203K rows, SMALL)

Key cross-dataset join table with S2ORC corpus IDs and DOIs.

| Column | Type | Description |
|--------|------|-------------|
| `s2orc_corpus_id` | BIGINT | FK to `s2ag.papers.corpusid`. **Primary S2AG join key.** |
| `paper_id` | VARCHAR | FK to `pwc.papers.paper_id` |
| `doi` | VARCHAR | DOI (lowercase, no prefix). Matches S2AG format. |
| `arxiv` | VARCHAR | ArXiv ID |
| `full_text` | VARCHAR | Full paper text (can be very long) |

### pwc.paper_has_openalexWorkID (407K rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `paper_id` | VARCHAR | FK to `pwc.papers.paper_id` |
| `openalex_work_id` | VARCHAR | OpenAlex work ID in bare format (`W2803600120`). To join: `'https://openalex.org/' \|\| openalex_work_id = openalex.works.id` |

### Key Relationship Tables

| Table | Rows | Columns | Purpose |
|-------|------|---------|---------|
| `paper_uses_method` | 621K | paper_id, method_id | Which methods each paper uses |
| `paper_has_task` | 951K | paper_id, task_id | Which tasks each paper addresses |
| `paper_has_code` | 269K | paper_id, repo_url | Paper→GitHub repo (FK to code_repos) |
| `code_repos` | 215K | repo_url, framework, is_official | Repo metadata |
| `paper_introduces_method` | 2.1K | method_id, paper_id | Provenance: first paper for method |
| `paper_introduces_dataset` | 8.6K | dataset_id, paper_id | Provenance: first paper for dataset |
| `patent_cites_paper` | 156K | paper_id, patent, confscore | Patent→paper citations |
| `dataset_has_task` | 18K | dataset_id, task_id | Which tasks use which datasets |

```sql
-- Method diffusion: which methods span the most tasks?
SELECT m.name, COUNT(DISTINCT pt.task_id) AS task_count
FROM pwc.paper_uses_method pm
JOIN pwc.methods m ON pm.method_id = m.method_id
JOIN pwc.paper_has_task pt ON pm.paper_id = pt.paper_id
GROUP BY m.name ORDER BY task_count DESC LIMIT 10;

-- Papers with official code implementations
SELECT p.title, c.repo_url, c.framework
FROM pwc.papers p
JOIN pwc.paper_has_code pc ON p.paper_id = pc.paper_id
JOIN pwc.code_repos c ON pc.repo_url = c.repo_url
WHERE c.is_official = true LIMIT 10;
```

---

## Scientific Ontologies (13 converted)

All 13 ontologies are converted. Each has up to 3 tables in its own DuckDB schema. All ontologies have a `_terms` table; most also have `_hierarchy` and `_xrefs` tables (see table below for availability).

### Standard Ontology Tables

Every ontology `{name}` has up to three tables:

**`{name}.{name}_terms`** — Term catalog

| Column | Type | Description |
|--------|------|-------------|
| `id` | VARCHAR | Term identifier (ontology-specific format) |
| `label` | VARCHAR | Preferred English label |
| `definition` | VARCHAR | Term definition (nullable) |
| `synonyms` | VARCHAR[] | Alternative labels |
| `namespace` | VARCHAR | Ontology namespace/category |
| `obsolete` | BOOLEAN | Whether the term is deprecated |

**`{name}.{name}_hierarchy`** — Parent-child edges

| Column | Type | Description |
|--------|------|-------------|
| `parent_id` | VARCHAR | Parent term ID |
| `child_id` | VARCHAR | Child term ID |
| `relation` | VARCHAR | Relation type (is_a, broader, part_of, superTopicOf, etc.) |

**`{name}.{name}_xrefs`** — Cross-references

| Column | Type | Description |
|--------|------|-------------|
| `term_id` | VARCHAR | Term ID |
| `xref_db` | VARCHAR | External database name or match type |
| `xref_id` | VARCHAR | External identifier |

### Available Ontologies

| Schema | Full Name | Domain | Terms | Hierarchy | Xrefs |
|--------|-----------|--------|-------|-----------|-------|
| `mesh` | Medical Subject Headings | Biomedical | 720,801 | — | — |
| `chebi` | Chemical Entities of Biological Interest | Chemistry | 205,317 | 379,841 | 389,000 |
| `ncit` | NCI Thesaurus | Cancer/Biomedical | 203,668 | 293,258 | 2,250 |
| `go` | Gene Ontology | Biology | 47,856 | 81,173 | 25,868 |
| `agrovoc` | AGROVOC Multilingual Thesaurus | Agriculture | 41,699 | 42,132 | 49,762 |
| `hpo` | Human Phenotype Ontology | Phenotypes | 19,934 | 23,765 | 18,099 |
| `cso` | Computer Science Ontology | CS | 14,636 | 93,491 | 28,100 |
| `doid` | Disease Ontology | Disease | 14,521 | 16,916 | 38,653 |
| `stw` | STW Thesaurus for Economics | Economics | 7,858 | 28,702 | — |
| `msc2020` | Mathematics Subject Classification 2020 | Mathematics | 6,603 | 6,603 | — |
| `unesco` | UNESCO Thesaurus | Education/Science | 4,498 | 8,682 | — |
| `physh` | Physics Subject Headings | Physics | 3,925 | 8,844 | — |
| `edam` | EDAM Ontology | Bioinformatics | 3,524 | 5,219 | 266 |

### Example SQL Queries

```sql
-- Search for a term across available ontologies
SELECT 'cso' AS source, id, label FROM cso.cso_terms WHERE label ILIKE '%machine_learning%'
UNION ALL
SELECT 'doid', id, label FROM doid.doid_terms WHERE label ILIKE '%diabetes%'
UNION ALL
SELECT 'edam', id, label FROM edam.edam_terms WHERE label ILIKE '%sequence%';

-- Navigate CSO hierarchy: find subtopics
SELECT h.child_id, t.label
FROM cso.cso_hierarchy h
JOIN cso.cso_terms t ON h.child_id = t.id
WHERE h.parent_id ILIKE '%deep_learning%' AND h.relation = 'superTopicOf';

-- Find DOID cross-references for a disease
SELECT x.xref_db, x.xref_id
FROM doid.doid_xrefs x
JOIN doid.doid_terms t ON x.term_id = t.id
WHERE t.label ILIKE '%alzheimer%';
```

---

## Retraction Watch

**Primary key:** `record_id` (INTEGER) | **DOI format:** lowercase, no prefix | **License:** Open (via Crossref)

### retwatch.retraction_watch (69K rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | INTEGER | **Primary key.** Unique record identifier. |
| `title` | VARCHAR | Title of the retracted/corrected paper |
| `subject` | VARCHAR | Subject area |
| `institution` | VARCHAR | Institution |
| `journal` | VARCHAR | Journal name |
| `publisher` | VARCHAR | Publisher name |
| `country` | VARCHAR | Country of origin |
| `author` | VARCHAR | Author names |
| `urls` | VARCHAR | Related URLs |
| `article_type` | VARCHAR | Article type |
| `retraction_date` | DATE | Date of retraction action |
| `retraction_doi` | VARCHAR | DOI of the retraction notice |
| `retraction_pubmed_id` | VARCHAR | PubMed ID of retraction notice |
| `original_paper_date` | DATE | Publication date of original paper |
| `original_paper_doi` | VARCHAR | DOI of retracted paper (lowercase, no prefix). **Join key.** |
| `original_paper_pubmed_id` | VARCHAR | PubMed ID of original paper |
| `retraction_nature` | VARCHAR | Retraction, Correction, Expression of Concern, etc. |
| `reason` | VARCHAR | Semicolon-separated retraction reasons |
| `paywalled` | VARCHAR | YES/NO whether retraction notice is paywalled |
| `notes` | VARCHAR | Additional notes |

```sql
-- Flag retracted papers in any query
SELECT p.title, rw.retraction_nature, rw.reason
FROM s2ag.papers p
JOIN retwatch.retraction_watch rw ON p.doi = rw.original_paper_doi
WHERE p.year >= 2020;
```

---

## Reliance on Science (Marx & Fuegi)

**Primary key:** varies by table | **License:** Open Access

### ros.pcs_oa (47.8M rows, LARGE)

Patent citation-to-science pairs. Each row is one patent citing one paper.

| Column | Type | Description |
|--------|------|-------------|
| `patent` | VARCHAR | Patent ID (e.g., `US-10000036`) |
| `oaid` | BIGINT | OpenAlex numeric ID (bare numeric, no `W` prefix). **Join:** `'W' \|\| CAST(oaid AS VARCHAR) = sciscinet.papers.paperid` |
| `reftype` | VARCHAR | Citation type |
| `confscore` | INTEGER | Confidence score |
| `uspto` | INTEGER | USPTO flag |
| `wherefound` | VARCHAR | Where citation was found |
| `self_cite` | VARCHAR | Self-citation flag |

### ros.patent_paper_pairs (548K rows, SMALL)

Curated patent-paper similarity pairs.

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | OpenAlex work ID with `W` prefix (e.g., `W2100837269`). Matches `sciscinet.papers.paperid`. |
| `patent` | VARCHAR | Patent ID |
| `ppp_score` | DOUBLE | Patent-paper similarity score |
| `daysdiffcont` | DOUBLE | Days difference (continuous) |
| `all_patents_for_the_same_paper` | INTEGER | Count of patents citing same paper |

### ros.patent_paper_pairs_plus (548K rows, SMALL)

Extended pairs with institutional and commercialization metadata. Same rows as `patent_paper_pairs` plus extra columns.

| Column | Type | Description |
|--------|------|-------------|
| `paperid` | VARCHAR | OpenAlex work ID with `W` prefix |
| `patent` | VARCHAR | Patent ID |
| `ppp_score` | DOUBLE | Similarity score |
| `concepts` | VARCHAR | Concept annotations |
| `citationbased` | VARCHAR | Citation-based flag |
| `paperuniv` | BOOLEAN | Paper from university |
| `paperfirm` | BOOLEAN | Paper from firm |
| `papergovn` | BOOLEAN | Paper from government |
| `paperunk` | BOOLEAN | Paper from unknown institution |
| `patentuniv` | BOOLEAN | Patent from university |
| `patentfirm` | BOOLEAN | Patent from firm |
| `patentgovn` | BOOLEAN | Patent from government |
| `patentlone` | BOOLEAN | Patent from lone inventor |
| `reassigned` | BOOLEAN | Patent reassigned |
| `commercialized` | BOOLEAN | Patent commercialized |
| `prerecommercialized` | BOOLEAN | Pre-commercialized |

```sql
-- Papers with most patent citations
SELECT oaid, COUNT(*) AS patent_cites
FROM ros.pcs_oa GROUP BY oaid ORDER BY patent_cites DESC LIMIT 20;

-- University papers that led to commercialized patents
SELECT paperid, patent, ppp_score
FROM ros.patent_paper_pairs_plus
WHERE paperuniv = true AND commercialized = true;
```

---

## PreprintToPaper

**Primary key:** `biorxiv_doi` | **DOI format:** lowercase, no prefix | **License:** Open Access

### p2p.preprint_to_paper (146K rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `biorxiv_doi` | VARCHAR | **Primary key.** Preprint DOI (e.g., `10.1101/2020.01.01.123456`) |
| `biorxiv_published_doi` | VARCHAR | Published version DOI (nullable, lowercase, no prefix). **Join key to other datasets.** |
| `custom_status` | VARCHAR | `published` (90.6K), `preprint_only` (35.8K), or `gray_zone` (19.1K) |
| `crossref_journal_name` | VARCHAR | Journal name from Crossref (nullable) |
| `biorxiv_category` | VARCHAR | bioRxiv subject category |
| `first_submission_date` | VARCHAR | Date of first preprint submission |
| `first_pub_date` | VARCHAR | Published version date |
| `version_date_diff_days` | DOUBLE | Days between preprint versions |
| `submission_pub_date_diff_days` | DOUBLE | Days from submission to publication |

### p2p.preprint_to_paper_grayzone (299 rows, SMALL)

| Column | Type | Description |
|--------|------|-------------|
| `year` | BIGINT | Year |
| `biorxiv_doi` | VARCHAR | Preprint DOI |
| `suspected_published_doi` | VARCHAR | Suspected published DOI |
| `author_match_score` | DOUBLE | Author overlap score (0-1) |
| `annotator1` | VARCHAR | Manual annotation (TRUE/FALSE/NA) |
| `annotator2` | VARCHAR | Manual annotation (TRUE/FALSE/NA) |

```sql
-- Publication rate by category
SELECT biorxiv_category, COUNT(*) AS total,
       ROUND(100.0 * SUM(CASE WHEN custom_status='published' THEN 1 ELSE 0 END)/COUNT(*), 1) AS pct
FROM p2p.preprint_to_paper GROUP BY biorxiv_category ORDER BY total DESC;

-- Time to publication
SELECT ROUND(submission_pub_date_diff_days/30) AS months, COUNT(*) AS n
FROM p2p.preprint_to_paper
WHERE custom_status='published' AND submission_pub_date_diff_days IS NOT NULL
GROUP BY months ORDER BY months;
```

---

## xref (Cross-Reference)

### xref.doi_map (VERY_LARGE)

Normalizes DOIs from all datasets to lowercase, no-prefix format.

| Column | Type | Description |
|--------|------|-------------|
| `source` | VARCHAR | Dataset: 'openalex' (293M), 'sciscinet' (159M), 's2ag' (136M), 'p2p_preprint' (146K), 'pwc' (141K), 'p2p_published' (110K), 'retwatch' (66K) |
| `doi` | VARCHAR | Normalized DOI (lowercase, no prefix) |
| `source_id` | VARCHAR | Dataset-specific paper ID (corpusid for S2AG, paperid for SciSciNet, paper_id for PWC) |

**Note:** This is a UNION ALL view over papers tables. Each query scans the underlying parquet. Always filter by `doi = ...` to avoid full table scans.

---

## Performance Guide

### Size Tiers

**SMALL (<1M rows):** No filtering needed. Full scans are fast.
- publication_venues, affiliations, fields, sources, funders, topics, domains, subfields, publishers, concepts, institutions, link_clinicaltrials, link_nobellaureates, metadata tables
- All pwc.* tables, all cso.* tables, doid.*, edam.*, retwatch.*, ros.patent_paper_pairs*, p2p.*

**MEDIUM (1-10M rows):** Generally fast. Light filtering recommended.
- papers_pmid_pmcid, link_nih, link_nsf, link_newsfeed, newsfeed_metadata, pwc.paper_has_openalexAuthorID, openalex.awards (11.7M)

**LARGE (10-100M rows):** Always use WHERE clauses. LIMIT recommended.
- abstracts, tldrs, s2orc, link_patents, link_twitter, twitter_metadata, ros.pcs_oa (47.8M)

**VERY_LARGE (>100M rows):** Always filter aggressively. Never SELECT * without WHERE + LIMIT.
- s2ag: papers (231M), citations (2.9B), paper_ids (519M), authors (112M)
- sciscinet: papers (250M), paper_details (250M), paper_refs (2.5B), paper_fields (1.3B), paper_authors (773M), hit_papers (570M, 702M), normalized_citations (570M, 702M), paper_sources (204M), authors (100M)
- openalex: works (479M), works_referenced_works (3.01B), works_authorships (1.32B), works_topics (910M), works_locations (612M), works_ids (479M), works_biblio (479M), works_open_access (479M), works_counts_by_year (443M), works_best_oa_location (210M), authors (108M), authors_counts_by_year (302M)
- xref: doi_map (588M)

### Thread Settings

```sql
SET threads=16;  -- Good default for this machine (24-core Threadripper)
```

### Performance Tips

1. **Filter before joining.** Narrow down with WHERE on the larger table first.
2. **Use LIMIT** for exploratory queries.
3. **Avoid derived views for bulk operations.** `paper_authors` and `paper_fields` in S2AG are computed on-the-fly from UNNEST. For large-scale author analysis, prefer `sciscinet.paper_authors` (materialized).
4. **DOI lookups on xref.doi_map** scan all papers tables. For a single DOI, query the specific dataset's papers table directly.
5. **Parquet is columnar.** SELECT only the columns you need.

---

## Data Quality Notes

1. **Abstract coverage:** S2AG ~16% (37M/231M), SciSciNet ~65% with valid_title_abstract, OpenAlex has the broadest coverage (479M works with abstracts where available).
2. **Citation counts differ:** Each dataset counts independently. S2AG citationcount != SciSciNet cited_by_count != OpenAlex cited_by_count for the same paper.
3. **DOI availability:** Not all papers have DOIs. ~70% of S2AG papers have DOIs. OpenAlex has 293M works with DOIs out of 479M.
4. **Disruption NULL:** SciSciNet disruption is NULL when insufficient citation data (too few references or citations).
5. **Atypicality NULL:** Requires journal-level reference pairs. Papers without journal references will have NULL atypicality.
6. **valid_title_abstract** is a strict filter: English only, readable abstract, sufficient length. ~65% of SciSciNet papers pass. OpenAlex works have a similar flag.
7. **PWC OpenAlex ID mismatch:** PWC stores bare work IDs (`W2803600120`), OpenAlex uses full URLs (`https://openalex.org/W2803600120`). Always prepend prefix when joining.

---

## Common Task Recipes

### 1. Find a Paper by DOI Across All Datasets

```sql
-- Quick: check which datasets have it
SELECT source, source_id FROM xref.doi_map
WHERE doi = '10.1038/nature12373';

-- Full details from S2AG + SciSciNet
SELECT s.title, s.year, s.citationcount AS s2ag_cites,
       sc.disruption, sc.C10, sc.patent_count
FROM s2ag.papers s
JOIN sciscinet.papers sc ON s.doi = REPLACE(sc.doi, 'https://doi.org/', '')
WHERE s.doi = '10.1038/nature12373';
```

### 2. Get Disruption + Citation Context for a Paper

```sql
-- SciSciNet for disruption, S2AG for citation contexts
SELECT sc.disruption, sc.Atyp_Median_Z,
       c.contexts, c.intents
FROM sciscinet.papers sc
JOIN s2ag.papers s ON REPLACE(sc.doi, 'https://doi.org/', '') = s.doi
JOIN s2ag.citations c ON s.corpusid = c.citedcorpusid
WHERE s.doi = '10.1038/nature12373'
  AND c.isinfluential = true;
```

### 3. Find an Author Across Datasets

```sql
-- By ORCID in SciSciNet
SELECT ad.authorid, ad.display_name, ad.orcid, a.h_index, a.productivity
FROM sciscinet.author_details ad
JOIN sciscinet.authors a ON ad.authorid = a.authorid
WHERE ad.orcid = 'https://orcid.org/0000-0002-1234-5678';

-- Same author in OpenAlex
SELECT id, display_name, h_index, works_count, cited_by_count
FROM openalex.authors
WHERE orcid = 'https://orcid.org/0000-0002-1234-5678';
```

### 4. NIH-Funded Papers and Their Impact

```sql
SELECT p.paperid, d.title, p.disruption, p.cited_by_count,
       nm.award_id, nm.funder_display_name
FROM sciscinet.papers p
JOIN sciscinet.paper_details d ON p.paperid = d.paperid
JOIN sciscinet.link_nih l ON p.paperid = l.paperid
JOIN sciscinet.nih_metadata nm ON l.award_id = nm.award_id AND l.paperid = nm.paperid
WHERE p.year BETWEEN 2015 AND 2020
ORDER BY p.disruption DESC LIMIT 20;
```

### 5. Topic Classification Hierarchy

```sql
-- All topics in Computer Science with work counts
SELECT t.display_name AS topic, sf.display_name AS subfield,
       t.works_count, t.cited_by_count
FROM openalex.topics t
JOIN openalex.subfields sf ON t.subfield_id = sf.id
WHERE sf.field_display_name = 'Computer Science'
ORDER BY t.works_count DESC;
```

### 6. Top 1% Papers with Abstracts

```sql
SELECT p.paperid, d.title, d.abstract, p.disruption, p.C10
FROM sciscinet.high_impact_papers p
JOIN sciscinet.paper_details d ON p.paperid = d.paperid
WHERE d.valid_title_abstract = true AND p.year = 2015
LIMIT 100;
```

### 7. Three-Way Join: S2AG + SciSciNet + OpenAlex

```sql
-- Get citation context, disruption, and topic for a paper
SELECT s.title, s.citationcount AS s2ag_cites,
       sc.disruption, sc.C10,
       oa.cited_by_count AS oa_cites,
       wt.display_name AS topic
FROM s2ag.papers s
JOIN sciscinet.papers sc ON s.doi = REPLACE(sc.doi, 'https://doi.org/', '')
JOIN openalex.works oa ON 'https://openalex.org/' || sc.paperid = oa.id
LEFT JOIN openalex.works_topics wt ON oa.id = wt.work_id AND wt.score > 0.9
WHERE s.doi = '10.1038/nature12373';
```
