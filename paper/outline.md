# The Science Data Lake: A Unified Open Infrastructure Integrating 293 Million Papers Across Eight Scholarly Sources with Embedding-Based Ontology Alignment

## Target Venue

**Primary: Nature Scientific Data** — "Data Descriptor" format; direct precedent with SciSciNet (Lin et al. 2023); IF ~7-10.
**Strong backup: Quantitative Science Studies (QSS)** — Published MAG and Crossref descriptor papers; IF ~4; scientometrics community home.

---

## Novelty Assessment

### Closest Competitors

| System | Venue | What it does | Key difference from us |
|--------|-------|-------------|----------------------|
| **SciSciNet** (Lin 2023) | Nat. Sci. Data | "Data lake" with 134M papers + metrics | Single-source (MAG/OpenAlex), no cross-source integration |
| **PubGraph** (Ahrabian 2023) | arXiv | KG unifying Wikidata+OpenAlex+S2AG | Merges into single RDF schema, loses source-level detail |
| **SemOpenAlex** (Farber 2023) | ISWC | OpenAlex as 26B RDF triples | Single-source, different format |
| **Dimensions BigQuery** (Hook 2021) | Frontiers | SQL-queryable Dimensions | Commercial, single-source |
| **BERTMap** (He 2022) | AAAI | Embedding-based ontology alignment | General biomedical, not bibliometric topic-to-ontology |

### Three Novelty Pillars

**1. Multi-source preserving architecture.** No published system integrates 8 scholarly sources (S2AG + OpenAlex + SciSciNet + PWC + Retraction Watch + RoS + P2P + Crossref) into a single DuckDB-queryable lake while *preserving source-level schemas* for cross-source comparison. PubGraph merges into a single schema; SciSciNet is single-source; Dimensions is commercial.

**2. Embedding-based topic-to-ontology bridging.** The OAEI community does ontology-to-ontology alignment; we do topic-taxonomy-to-formal-ontology mapping using BGE-large (1024-dim). No direct precedent in bibliometrics. The 17x improvement over string matching (16,150 vs. 937 matches) is a concrete methodological contribution.

**3. Cross-source record-level comparison.** Existing comparison studies use API queries or aggregate statistics. Our `xref.unified_papers` (293M rows, 32 columns) enables record-level joins across all sources simultaneously — enabling analyses like Vignette 4 (citation reliability) that no single source or pairwise comparison supports.

### What We Cannot Claim

- "First unified bibliometric database" (PubGraph, SciSciNet exist)
- "First cross-database comparison" (many existing papers)
- "First embedding-based ontology alignment" (BERTMap, OAEI)
- "First portable data lake" (concept exists broadly)

---

## Paper Structure (Nature Scientific Data "Data Descriptor" Format)

### Abstract (~250 words)

- **Problem:** Scholarly data is fragmented across siloed databases with incompatible identifiers, divergent citation counts, and no ontology layer connecting topic taxonomies to formal scientific ontologies.
- **Solution:** A locally-deployable data lake built on DuckDB/Parquet that unifies 8 open sources via DOI normalization and transformer-based ontology bridging, while preserving source-level schemas for direct cross-source comparison.
- **Results:** 293M unified papers across 22 schemas and 151 views; 16,150 topic-ontology mappings achieving 99.8% topic coverage; 4 cross-domain vignettes demonstrating analyses impossible with any single source.
- **Availability:** Open source, runs on a single NVME drive or via HuggingFace.

### Background & Summary (~1 page)

- The fragmentation problem in science-of-science research
  - Each source captures different facets: S2AG has influential citations, OpenAlex has FWCI and topics, SciSciNet has disruption/atypicality, PWC has code links, Retraction Watch has integrity flags, RoS has patent citations
  - No single API covers everything (cite: Pons-Aranguren et al. 2023)
  - Liu et al. (2023) NHB review: "the science of science has become a field"
- Why cross-source integration is needed
  - Citation counts vary across sources (our V4: S2AG-OA Pearson r=0.76, mean abs. diff=4.14)
  - Retraction status, code availability, patent impact each come from different sources
  - Researchers currently must write ad-hoc scripts to merge sources
- Position relative to existing systems
  - SciSciNet: single-source, same venue — we extend with 7 more sources
  - PubGraph: multi-source KG — but merges into single schema, losing provenance
  - Dimensions BigQuery: commercial, not reproducible
- Our contribution: 3 novelty pillars (architecture, ontology bridging, record-level comparison)

### Methods (~3-4 pages)

#### Data Sources

**Table 1: Dataset Overview**

| Source | Records | License | Version | Key Metrics |
|--------|---------|---------|---------|-------------|
| Semantic Scholar (S2AG) | 133M papers | ODC-BY | 2024-09 | citations, influential citations, open access |
| OpenAlex | 292M works | CC0 | 2025-01 | FWCI, topics, types, languages |
| SciSciNet | 159M papers | CC BY-NC | v1 | disruption, atypicality, team size, patent count |
| Papers With Code | 141K papers | CC BY-SA | 2024-10 | code repositories, tasks, datasets |
| Retraction Watch | 60K records | open | 2024-09 | retraction reasons, dates |
| Reliance on Science (RoS) | 548K links | open | v64 | patent-paper pairs, confidence scores |
| Preprint-to-Published (P2P) | ~4M links | open | 2024 | biorxiv DOI to published DOI |
| Crossref | metadata | open | 2024 | DOI metadata, references |

#### Architecture

- **DuckDB views over Parquet** — 22 schemas, 151 views, ~1.7 TB on disk
  - Schema design: each source retains its native schema (e.g., `s2ag.papers`, `openalex.works`)
  - Cross-referencing via `xref` schema: `unified_papers`, `doi_map`, `topic_ontology_map`
  - View-only database (~500 KB) — all data in Parquet files
- **Dual-mode access** — local NVME for full-speed queries; HuggingFace-hosted for cloud access
- **Reproducible pipeline** — `datalake_cli.py` orchestrates download, convert, create views, materialize

#### DOI Normalization & Record Linkage

**Table 2: DOI Format Differences**

| Source | Raw DOI Format | Normalization |
|--------|---------------|---------------|
| S2AG | lowercase, no prefix (`10.1038/...`) | — (canonical) |
| OpenAlex | lowercase, `https://doi.org/` prefix | strip prefix |
| SciSciNet | lowercase, `https://doi.org/` prefix | strip prefix |
| PWC | lowercase, no prefix | — |
| Retraction Watch | lowercase, no prefix | — |
| Crossref | mixed case | lowercase |

- All DOIs normalized to lowercase, no-prefix format
- `xref.doi_map`: union of 7 sub-queries with per-source normalization
- `xref.unified_papers`: 293,123,121 unique DOIs with 32 columns from all sources
- Coverage matrix: OpenAlex 99.67%, SciSciNet 54.08%, S2AG 45.55%

#### Embedding-Based Ontology Alignment

- **13 scientific ontologies** converted to Parquet + Oxigraph RDF store
  - Large: MeSH (721K terms), ChEBI (205K), NCIT (204K)
  - Medium: GO (48K), AGROVOC (42K), CSO (26K)
  - Small: DOID, HPO, EDAM, UNESCO, STW, PhySH, MSC2020
- **Hybrid approach:**
  - BGE-large-en-v1.5 (1024-dim, 335M params) for 10 smaller ontologies (291K terms incl. synonyms)
  - Exact string matching for 3 large ontologies (MeSH, ChEBI, NCIT)
  - FAISS index for fast nearest-neighbor search on GPU (RTX A4500)
- **Quality tiers:**
  - Exact match (sim >= 0.95): 85 mappings
  - High-quality (sim >= 0.85): 2,527 mappings
  - All (sim >= 0.65): 16,150 mappings
  - Topic coverage: 4,509/4,516 (99.84%)
- **Comparison to string baseline:**
  - Jaro-Winkler at threshold 0.90: only 937 matches
  - Embedding approach: 16,150 matches — **17x improvement**

### Technical Validation (~2 pages)

**Table 5: Sanity Check Results**

| # | Check | Result | Detail |
|---|-------|--------|--------|
| 1 | DOI format (no http prefix, all lowercase) | PASS | 0 violations / 293M |
| 2 | Coverage flags match data presence | PASS | 0 mismatches (OA, S2AG, SciSciNet) |
| 3 | Primary key uniqueness (no duplicate DOIs) | PASS | 293,123,121 unique = total |
| 4 | OpenAlex ID format + joinability to works_topics | PASS | 0 format violations; 69% join match |
| 5 | topic_ontology_map to openalex.topics (no orphans) | PASS | 0 orphan topic_ids |
| 6 | RoS oaid to OpenAlex join (10K sample) | PASS | 86% match rate |
| 7 | Citation cross-source correlation | PASS | S2AG-OA: r=0.76, S2AG-SSN: r=0.87, OA-SSN: r=0.86 |
| 8 | Year distribution (NULL/invalid) | PASS | NULL: 0.53%, invalid: 0.002% |
| 9 | Spot-check known papers (Wakefield retraction) | PASS | Retraction flags correct |
| 10 | Vignette count reproducibility (V1-V4) | PASS | All 4 counts match exactly |

Additional validation points:
- Citation count cross-source agreement: 2 of 3 pairwise Pearson r > 0.8; mean absolute difference 2.3-4.1 citations
- Ontology linkage precision: manual inspection of high-similarity mappings confirms semantic relevance
- Temporal coverage notes: RoS has citation lag (patents through late 2023); SciSciNet metrics end ~2022

### Usage Notes (~2 pages)

- **Setup:** Clone repo, run pipeline (`datalake_cli.py`), connect DuckDB
- **4 vignettes demonstrating cross-domain analyses:**

  **Vignette 1: Disruption x Code Adoption x Ontology Landscape**
  - Papers with code: 139,873 (0.048%) — avg disruption -0.0005 vs. +0.0026 without code
  - Ontology mapping reveals domain-specific code adoption patterns across CSO, EDAM, UNESCO
  - *Only possible because:* disruption (SciSciNet), code flags (PWC), topics (OpenAlex), ontology bridging (our linkage)

  **Vignette 2: Retraction Profiles x Ontology Enrichment**
  - 58,775 retracted papers with SciSciNet metrics; avg disruption 0.0035 vs. 0.0025 non-retracted
  - Top retracted: COVID paper with 8,062 citations; DNA methylation paper with 6,811
  - *Only possible because:* retraction flags (RetWatch), disruption scores (SciSciNet), citation counts (OpenAlex)

  **Vignette 3: Patent Impact x Multi-Ontology Footprint**
  - 312,929 patent-cited papers (0.107%); avg citations 94.3 vs. 16.1 non-cited
  - Patent-cited papers have 5.8x higher citations and 3.1x higher FWCI
  - *Only possible because:* patent links (RoS), FWCI (OpenAlex), disruption (SciSciNet), team size (SciSciNet)

  **Vignette 4: Cross-Source Citation Reliability x Field Variation**
  - 121M papers in all 3 sources; S2AG-OA correlation r=0.76, mean diff 4.14
  - Divergence varies by citation magnitude: low-cited papers show 20% relative difference
  - Largest disagreement: single paper with 257,887 S2AG citations vs. 0 in OpenAlex
  - *Only possible because:* parallel citation counts from 3 independent sources in one table

- **HuggingFace fallback** for users without local NVME storage

### Data Records (~0.5 page)

- Repository: Zenodo (frozen snapshots) + HuggingFace (live access)
- File structure: 22 schema directories, each containing Parquet files
- Total size: ~1.7 TB compressed Parquet
- License: each source retains its original license (CC0, CC BY-NC, ODC-BY, etc.)
- Row counts: See Table 1

### Code Availability (~0.5 page)

- GitHub repo with all scripts
- Pipeline stages: download -> convert -> create views -> materialize unified -> build ontology linkage
- Key scripts:
  - `datalake_cli.py` — master CLI orchestrating all stages
  - `materialize_unified_papers.py` — DOI normalization and record linkage
  - `build_embedding_linkage.py` — transformer-based ontology alignment
  - `convert_ontologies.py` — 5 parsers for 13 ontology formats
  - `create_unified_db.py` — DuckDB view generation
- Python 3.12, DuckDB 1.4.2, PyArrow 22.0, sentence-transformers 5.2.2

---

## Key Tables

1. **Table 1:** Dataset overview (8 sources x 5 columns: source, records, license, version, key metrics)
2. **Table 2:** DOI format differences and normalization strategy per source
3. **Table 3:** Cross-source overlap matrix (coverage percentages for top-3 sources)
4. **Table 4:** Ontology linkage quality by tier (similarity thresholds x match counts)
5. **Table 5:** Sanity check summary (10 checks, all PASS)
6. **Table 6:** Comparison with existing systems (SciSciNet, PubGraph, SemOpenAlex, Dimensions BigQuery)

## Key Figures

1. **Architecture diagram** — Data flow: 8 sources -> Parquet -> DuckDB views -> unified_papers -> ontology layer (NEW)
2. **Coverage UpSet plot** — Multi-source overlap across S2AG, OpenAlex, SciSciNet, PWC, RetWatch, RoS (NEW)
3. **Ontology linkage quality** — Similarity distribution histogram + comparison to Jaro-Winkler baseline (NEW)
4. **Vignette highlights** — 1 key figure from each of the 4 vignettes (EXISTING, from notebooks)

---

## References

### Core citations (our sources & competitors)

- Lin, Z. et al. (2023). SciSciNet: A large-scale open data lake for the science of science research. *Scientific Data*, 10, 315.
- Ahrabian, K. & Du, Y. (2023). PubGraph: A large-scale scientific knowledge graph. arXiv:2302.02231.
- Farber, M. (2023). SemOpenAlex: The scientific landscape in 26 billion RDF triples. *ISWC 2023*.
- Hook, D. & Porter, S. (2021). Dimensions: Building context for search and evaluation. *Frontiers in Research Metrics and Analytics*, 6, 620557.
- He, Y. et al. (2022). BERTMap: A BERT-based ontology alignment system. *AAAI 2022*.
- Priem, J. et al. (2022). OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv:2205.01833.
- Kinney, R. et al. (2023). The Semantic Scholar Open Data Platform. arXiv:2301.10140.

### Motivation & context

- Pons-Aranguren, F. et al. (2023). A systematic comparison of bibliometric databases. *Information Processing & Management*, 60(6), 103532.
- Liu, L. et al. (2023). Data, measurement and empirical methods in the science of science. *Nature Human Behaviour*, 7, 1046-1058.
- Rodrigues, L. et al. (2025). Cross-database citation detection in scholarly literature. *Journal of Informetrics*, 19(1), 101612.
- Gusenbauer, M. (2024). Citation coverage of 59 databases: A comprehensive evaluation. *Research Synthesis Methods*, 15, 1-19.
- Salatino, A. et al. (2020). The Computer Science Ontology: A comprehensive automatically-generated ontology of research areas. *Data Intelligence*, 2(3), 379-416.

### Methods

- Xiao, S. et al. (2023). C-Pack: Packaged resources to advance general Chinese embedding. arXiv:2309.07597. [BGE model]
- Johnson, J. et al. (2019). Billion-scale similarity search with GPUs. *IEEE TPAMI*, 42(2), 401-416. [FAISS]
