#!/usr/bin/env python3
"""
Convert full-text paper sources to unified Parquet schema.

Normalizes papers from S2ORC, peS2o, PMC (and later arXiv, bioRxiv, CORE)
into a common schema with DOI-based linking back to the datalake.

Unified schema per row:
    doi, source, title, abstract, text, license, year, source_id,
    text_length, language, has_full_text

Usage:
    python scripts/convert_fulltext.py --source s2orc
    python scripts/convert_fulltext.py --source pes2o
    python scripts/convert_fulltext.py --source pmc
    python scripts/convert_fulltext.py --all
    python scripts/convert_fulltext.py --summary
"""

import argparse
import glob as globmod
import gzip
import json
import os
import re
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import duckdb

# ── Resolve data lake root ──────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import find_datalake_root

ROOT = find_datalake_root()
OUTPUT_DIR = ROOT / "datasets" / "fulltext" / "parquet"
RAW_DIR = ROOT / "datasets" / "fulltext" / "raw"
S2AG_PARQUET = ROOT / "datasets" / "s2ag" / "parquet"

SOURCES = ["s2orc", "pes2o", "pmc", "arxiv", "biorxiv", "core"]
ROW_GROUP_SIZE = 10_000  # Large text per row → smaller row groups

# SQL expression to normalize DOIs: lowercase, strip embedded doi.org URLs
# Handles malformed DOIs like "10.1002/https://doi.org/10.6028/nist.ir.7942"
DOI_CLEAN_SQL = """
    CASE
        WHEN {doi_expr} LIKE '%doi.org/%'
        THEN LOWER(REGEXP_EXTRACT({doi_expr}, 'doi\\.org/(.+)$', 1))
        ELSE LOWER({doi_expr})
    END
"""

# peS2o text is a single field combining title+abstract+body.
# Papers with text_length <= 5000 are typically just abstracts.
PES2O_FULLTEXT_THRESHOLD = 5000


# ── Text cleaning utilities ─────────────────────────────────────────────────

def clean_text(text: str | None) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Normalize whitespace: collapse runs of spaces/tabs, keep newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str | None:
    """Detect language of text. Returns ISO 639-1 code or None."""
    try:
        from langdetect import detect
        if len(text) < 50:
            return None
        # Use first 2000 chars for speed
        return detect(text[:2000])
    except Exception:
        return None


# ── S2ORC converter ─────────────────────────────────────────────────────────

def convert_s2orc(workers: int = 4, force: bool = False):
    """Convert S2ORC v2 data to fulltext schema using DuckDB.

    Joins s2orc_v2 (body text) + papers (DOIs, year) + abstracts.
    """
    source_dir = S2AG_PARQUET / "s2orc_v2"
    output_dir = OUTPUT_DIR / "s2orc"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists() or not list(source_dir.glob("*.parquet")):
        print("  ERROR: S2ORC v2 parquet data not found at", source_dir)
        return False

    # Check if already converted
    existing = list(output_dir.glob("*.parquet"))
    if existing and not force:
        print(f"  Already converted ({len(existing)} files). Use --force to reconvert.")
        return True

    # Clean existing output
    for f in existing:
        f.unlink()

    print("  Converting S2ORC v2 → fulltext schema via DuckDB...")
    t0 = time.time()

    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={workers}")
    conn.execute("SET memory_limit='64GB'")

    s2orc_path = source_dir / "*.parquet"
    papers_path = S2AG_PARQUET / "papers" / "*.parquet"
    abstracts_path = S2AG_PARQUET / "abstracts" / "*.parquet"

    # Check required data exists
    for name, path in [("papers", S2AG_PARQUET / "papers"), ("abstracts", S2AG_PARQUET / "abstracts")]:
        if not path.exists() or not list(path.glob("*.parquet")):
            print(f"  ERROR: s2ag.{name} parquet not found at {path}")
            conn.close()
            return False

    doi_expr = DOI_CLEAN_SQL.format(doi_expr="p.externalids.DOI")
    query = f"""
    COPY (
        SELECT
            {doi_expr} AS doi,
            's2orc' AS source,
            COALESCE(s.title, p.title) AS title,
            COALESCE(a.abstract, '') AS abstract,
            COALESCE(s.body.text, '') AS text,
            s.openaccessinfo.license AS license,
            CAST(p.year AS INTEGER) AS year,
            CAST(s.corpusid AS VARCHAR) AS source_id,
            CAST(LENGTH(COALESCE(s.body.text, '')) AS INTEGER) AS text_length,
            CAST(NULL AS VARCHAR) AS language,
            LENGTH(COALESCE(s.body.text, '')) > 0 AS has_full_text
        FROM read_parquet('{s2orc_path}') s
        JOIN read_parquet('{papers_path}') p ON s.corpusid = p.corpusid
        LEFT JOIN read_parquet('{abstracts_path}') a ON s.corpusid = a.corpusid
        WHERE p.externalids.DOI IS NOT NULL
    ) TO '{output_dir}/' (
        FORMAT PARQUET,
        PER_THREAD_OUTPUT true,
        ROW_GROUP_SIZE {ROW_GROUP_SIZE},
        COMPRESSION zstd,
        COMPRESSION_LEVEL 3
    )
    """

    try:
        conn.execute(query)

        parquet_files = sorted(output_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)
        n_rows = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_dir}/*.parquet')"
        ).fetchone()[0]

        elapsed = time.time() - t0
        print(f"  Done: {n_rows:,} rows, {len(parquet_files)} files, "
              f"{total_size / (1024**3):.2f} GB, {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        conn.close()


# ── peS2o converter ─────────────────────────────────────────────────────────

def convert_pes2o(workers: int = 4, force: bool = False):
    """Convert peS2o v2 json.gz files to fulltext schema.

    Uses a single DuckDB process with multiple threads (not per-file workers)
    because each file needs to join against the full s2ag.papers table. A
    per-file ProcessPoolExecutor would OOM loading 231M papers per worker.

    Strategy:
      1. Build a lightweight corpusid→(doi, title, year) lookup table
      2. Build a corpusid→abstract lookup table
      3. Stream all peS2o files through a join against these small tables

    peS2o v2 schema: id (corpusid), text, source, version, added, created
    """
    raw_dir = RAW_DIR / "pes2o"
    output_dir = OUTPUT_DIR / "pes2o"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print("  ERROR: peS2o raw data not found at", raw_dir)
        print("  Run: python scripts/download_fulltext.py --source pes2o")
        return False

    # Find json.gz files (peS2o v2 uses .json.gz)
    input_files = sorted(raw_dir.rglob("*.json.gz"))
    if not input_files:
        print("  ERROR: No .json.gz files found in", raw_dir)
        return False

    # Check if already converted
    existing = list(output_dir.glob("*.parquet"))
    if existing and not force:
        print(f"  Already converted ({len(existing)} files). Use --force to reconvert.")
        return True

    # Clean existing output
    for f in existing:
        f.unlink()

    papers_path = S2AG_PARQUET / "papers" / "*.parquet"
    abstracts_path = S2AG_PARQUET / "abstracts" / "*.parquet"
    for name, path in [("papers", S2AG_PARQUET / "papers"), ("abstracts", S2AG_PARQUET / "abstracts")]:
        if not path.exists() or not list(path.glob("*.parquet")):
            print(f"  ERROR: s2ag.{name} not found at {path}")
            return False

    print(f"  Found {len(input_files)} peS2o json.gz files")

    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={workers}")
    conn.execute("SET memory_limit='128GB'")

    t0 = time.time()

    # Step 1: Build lightweight DOI lookup from papers (only rows with DOIs)
    print("  Step 1: Building corpusid → DOI lookup from s2ag.papers...")
    t1 = time.time()
    doi_expr = DOI_CLEAN_SQL.format(doi_expr="externalids.DOI")
    conn.execute(f"""
        CREATE TABLE doi_lookup AS
        SELECT
            corpusid,
            {doi_expr} AS doi,
            title,
            CAST(year AS INTEGER) AS year
        FROM read_parquet('{papers_path}')
        WHERE externalids.DOI IS NOT NULL
    """)
    n_lookup = conn.execute("SELECT COUNT(*) FROM doi_lookup").fetchone()[0]
    print(f"    {n_lookup:,} papers with DOIs ({time.time()-t1:.1f}s)")

    # Step 2: Build abstract lookup
    print("  Step 2: Building corpusid → abstract lookup...")
    t1 = time.time()
    conn.execute(f"""
        CREATE TABLE abstract_lookup AS
        SELECT corpusid, abstract
        FROM read_parquet('{abstracts_path}')
    """)
    n_abs = conn.execute("SELECT COUNT(*) FROM abstract_lookup").fetchone()[0]
    print(f"    {n_abs:,} abstracts ({time.time()-t1:.1f}s)")

    # Step 3: Join all peS2o files against the lookup tables
    # Build a glob pattern for all input files
    pes2o_glob = str(raw_dir / "**" / "*.json.gz")
    # DuckDB can read multiple gzip JSON files via glob
    pes2o_paths = [str(f) for f in input_files]
    pes2o_list = ", ".join(f"'{p}'" for p in pes2o_paths)

    print(f"  Step 3: Joining peS2o against lookups and writing parquet...")
    t1 = time.time()

    query = f"""
    COPY (
        SELECT
            dl.doi,
            'pes2o' AS source,
            COALESCE(dl.title, '') AS title,
            COALESCE(al.abstract, '') AS abstract,
            COALESCE(pe.text, '') AS text,
            CAST(NULL AS VARCHAR) AS license,
            dl.year,
            pe.id AS source_id,
            CAST(LENGTH(COALESCE(pe.text, '')) AS INTEGER) AS text_length,
            CAST(NULL AS VARCHAR) AS language,
            LENGTH(COALESCE(pe.text, '')) > {PES2O_FULLTEXT_THRESHOLD} AS has_full_text
        FROM read_json(
            [{pes2o_list}],
            format='newline_delimited',
            columns={{
                id: 'VARCHAR',
                text: 'VARCHAR',
                source: 'VARCHAR',
                version: 'VARCHAR',
                added: 'VARCHAR',
                created: 'VARCHAR'
            }},
            compression='gzip',
            maximum_object_size=104857600,
            ignore_errors=true
        ) pe
        JOIN doi_lookup dl ON CAST(pe.id AS BIGINT) = dl.corpusid
        LEFT JOIN abstract_lookup al ON CAST(pe.id AS BIGINT) = al.corpusid
    ) TO '{output_dir}/' (
        FORMAT PARQUET,
        PER_THREAD_OUTPUT true,
        ROW_GROUP_SIZE {ROW_GROUP_SIZE},
        COMPRESSION zstd,
        COMPRESSION_LEVEL 3
    )
    """

    try:
        conn.execute(query)
        print(f"    Parquet written ({time.time()-t1:.1f}s)")

        parquet_files = sorted(output_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in parquet_files)
        n_rows = conn.execute(
            f"SELECT COUNT(*) FROM read_parquet('{output_dir}/*.parquet')"
        ).fetchone()[0]

        elapsed = time.time() - t0
        print(f"  Done: {n_rows:,} rows, {len(parquet_files)} files, "
              f"{total_size / (1024**3):.2f} GB, {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False
    finally:
        conn.close()


# ── PMC converter ───────────────────────────────────────────────────────────

def _parse_jats_xml(xml_bytes: bytes) -> dict | None:
    """Parse a JATS XML article into the fulltext schema fields.

    Used for both PMC and bioRxiv (same XML format).
    """
    try:
        from lxml import etree
    except ImportError:
        raise ImportError("lxml is required for PMC/bioRxiv conversion: pip install lxml")

    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return None

    # Handle namespace
    nsmap = root.nsmap
    ns = nsmap.get(None, "")
    if ns:
        ns_prefix = f"{{{ns}}}"
    else:
        ns_prefix = ""

    def find(path):
        """Find element, trying with and without namespace."""
        el = root.find(f".//{ns_prefix}{path}")
        if el is None:
            el = root.find(f".//{path}")
        return el

    def findall(path):
        els = root.findall(f".//{ns_prefix}{path}")
        if not els:
            els = root.findall(f".//{path}")
        return els

    def get_text(el) -> str:
        """Recursively get text content of an element."""
        if el is None:
            return ""
        return "".join(el.itertext()).strip()

    # Extract DOI
    doi = None
    for article_id in findall("article-id"):
        if article_id.get("pub-id-type") == "doi":
            doi = article_id.text
            break
    if not doi:
        return None  # Skip papers without DOI

    doi = doi.strip().lower()
    # Strip standard prefixes
    if doi.startswith("https://doi.org/"):
        doi = doi[16:]
    elif doi.startswith("http://doi.org/"):
        doi = doi[15:]
    elif doi.startswith("http://dx.doi.org/"):
        doi = doi[18:]
    # Handle malformed embedded DOIs like "10.1002/https://doi.org/10.6028/..."
    if "doi.org/" in doi:
        doi = doi.split("doi.org/")[-1]

    # Title
    title_el = find("article-title")
    title = get_text(title_el) if title_el is not None else ""

    # Abstract
    abstract_el = find("abstract")
    abstract = get_text(abstract_el) if abstract_el is not None else ""

    # Body text: concatenate all <p> elements within <body>
    body_el = find("body")
    body_parts = []
    if body_el is not None:
        for sec in body_el.iter():
            if sec.tag.endswith("p") or sec.tag == "p":
                text = get_text(sec)
                if text:
                    body_parts.append(text)
    body_text = "\n\n".join(body_parts)

    # License
    license_el = find("license")
    license_str = ""
    if license_el is not None:
        license_str = license_el.get("{http://www.w3.org/1999/xlink}href", "")
        if not license_str:
            license_str = get_text(license_el)

    # Year
    year = None
    for pub_date in findall("pub-date"):
        year_el = pub_date.find(f"{ns_prefix}year")
        if year_el is None:
            year_el = pub_date.find("year")
        if year_el is not None and year_el.text:
            try:
                year = int(year_el.text)
                break
            except ValueError:
                pass

    # PMC ID
    pmc_id = None
    for article_id in findall("article-id"):
        if article_id.get("pub-id-type") == "pmc":
            pmc_id = article_id.text
            break

    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "text": body_text,
        "license": license_str,
        "year": year,
        "source_id": f"PMC{pmc_id}" if pmc_id else "",
    }


def _convert_pmc_archive(args: tuple) -> dict:
    """Process a single PMC .tar.gz archive containing JATS XML files."""
    tar_path, output_path = args

    try:
        from lxml import etree  # noqa: F401
    except ImportError:
        return {"input": str(tar_path), "error": "lxml not installed", "success": False}

    records = []
    errors = 0

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".xml"):
                    continue
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    xml_bytes = f.read()
                    parsed = _parse_jats_xml(xml_bytes)
                    if parsed:
                        records.append(parsed)
                except Exception:
                    errors += 1
    except Exception as e:
        return {"input": str(tar_path), "error": str(e), "success": False}

    if not records:
        return {
            "input": str(tar_path),
            "output": str(output_path),
            "rows": 0,
            "errors": errors,
            "success": True,
        }

    # Write to parquet via DuckDB
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("SET threads=1")

        # Build values for insert
        conn.execute("""
            CREATE TABLE articles (
                doi VARCHAR,
                source VARCHAR,
                title VARCHAR,
                abstract VARCHAR,
                text VARCHAR,
                license VARCHAR,
                year INTEGER,
                source_id VARCHAR,
                text_length INTEGER,
                language VARCHAR,
                has_full_text BOOLEAN
            )
        """)

        conn.executemany(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["doi"],
                    "pmc",
                    r["title"],
                    r["abstract"],
                    r["text"],
                    r["license"],
                    r["year"],
                    r["source_id"],
                    len(r["text"]),
                    None,
                    len(r["text"]) > 0,
                )
                for r in records
            ],
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        conn.execute(f"""
            COPY articles TO '{output_path}' (
                FORMAT PARQUET,
                ROW_GROUP_SIZE {ROW_GROUP_SIZE},
                COMPRESSION zstd,
                COMPRESSION_LEVEL 3
            )
        """)

        return {
            "input": str(tar_path),
            "output": str(output_path),
            "rows": len(records),
            "errors": errors,
            "size_mb": output_path.stat().st_size / (1024 * 1024),
            "success": True,
        }
    except Exception as e:
        return {"input": str(tar_path), "error": str(e), "success": False}
    finally:
        conn.close()


def convert_pmc(workers: int = 4, force: bool = False):
    """Convert PMC Open Access .tar.gz archives to fulltext parquet."""
    raw_dir = RAW_DIR / "pmc"
    output_dir = OUTPUT_DIR / "pmc"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print("  ERROR: PMC raw data not found at", raw_dir)
        print("  Run: python scripts/download_fulltext.py --source pmc")
        return False

    # PMC bulk download is organized as .tar.gz archives
    tar_files = sorted(list(raw_dir.rglob("*.tar.gz")))
    if not tar_files:
        print("  ERROR: No .tar.gz files found in", raw_dir)
        return False

    tasks = []
    skipped = 0
    for tar_file in tar_files:
        output_file = output_dir / (tar_file.stem.replace(".tar", "") + ".parquet")
        if not force and output_file.exists():
            skipped += 1
            continue
        tasks.append((tar_file, output_file))

    if skipped:
        print(f"  Skipping {skipped} already converted archives")
    if not tasks:
        print(f"  All {len(tar_files)} archives already converted")
        return True

    print(f"  Converting {len(tasks)} PMC archives with {workers} workers...")

    completed = 0
    total_rows = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_convert_pmc_archive, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["success"]:
                rows = result.get("rows", 0)
                total_rows += rows
                size_mb = result.get("size_mb", 0)
                print(
                    f"    [{completed}/{len(tasks)}] "
                    f"{Path(result['input']).name}: "
                    f"{rows:,} articles, {size_mb:.1f}MB"
                )
            else:
                failed += 1
                print(
                    f"    [{completed}/{len(tasks)}] "
                    f"{Path(result['input']).name}: ERROR - {result['error']}"
                )

    print(f"  Done: {total_rows:,} total articles, {failed} failures")
    return failed == 0


# ── arXiv (unarXive) converter ──────────────────────────────────────────────

# Regex to strip unarXive inline markers like {{cite:abc123}} and {{formula:uuid}}
_UNARXIVE_MARKER_RE = re.compile(r"\{\{(?:cite|formula|figure|table):[^}]+\}\}")


def _clean_unarxive_text(body_text: list[dict], ref_entries: dict | None = None) -> str:
    """Concatenate unarXive body_text sections into clean plain text.

    Strips {{cite:...}} and {{formula:...}} markers. Optionally replaces
    formula markers with their LaTeX representation from ref_entries.
    """
    parts = []
    for section in body_text:
        sec_heading = section.get("section", "")
        text = section.get("text", "")
        if not text:
            continue

        # Replace formula markers with LaTeX if ref_entries available
        if ref_entries:
            def _replace_formula(m):
                key = m.group(0)[2:-2]  # strip {{ and }}
                if key.startswith("formula:"):
                    ref_id = key[len("formula:"):]
                    entry = ref_entries.get(ref_id, {})
                    latex = entry.get("latex", "")
                    if latex:
                        return latex
                return ""
            text = re.sub(r"\{\{formula:[^}]+\}\}", _replace_formula, text)

        # Strip remaining markers (citations, figures, tables)
        text = _UNARXIVE_MARKER_RE.sub("", text)

        # Clean up whitespace artifacts from marker removal
        text = re.sub(r"  +", " ", text)
        text = text.strip()

        if text:
            if sec_heading and section.get("sec_number"):
                parts.append(f"{section['sec_number']} {sec_heading}")
            elif sec_heading:
                parts.append(sec_heading)
            parts.append(text)

    return "\n\n".join(parts)


def _process_unarxive_file(args: tuple) -> dict:
    """Process a single unarXive JSONL file into records for parquet."""
    jsonl_path, doi_lookup, arxiv_meta_lookup = args
    jsonl_path = Path(jsonl_path)

    records = []
    errors = 0

    open_fn = gzip.open if str(jsonl_path).endswith(".gz") else open

    try:
        with open_fn(jsonl_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    errors += 1
                    continue

                paper_id = paper.get("paper_id", "")
                if not paper_id:
                    errors += 1
                    continue

                # Extract body text
                body_text = paper.get("body_text", [])
                ref_entries = paper.get("ref_entries", {})
                text = _clean_unarxive_text(body_text, ref_entries)

                if not text or len(text) < 100:
                    continue  # Skip empty/trivial entries

                # Get metadata
                metadata = paper.get("metadata", {})
                title = metadata.get("title", "")
                abstract_data = paper.get("abstract", {})
                if isinstance(abstract_data, dict):
                    # Abstract is structured like body_text
                    abstract_text = abstract_data.get("text", "")
                    if not abstract_text and "body_text" in abstract_data:
                        abstract_text = " ".join(
                            s.get("text", "") for s in abstract_data.get("body_text", [])
                        )
                elif isinstance(abstract_data, str):
                    abstract_text = abstract_data
                elif isinstance(abstract_data, list):
                    abstract_text = _clean_unarxive_text(abstract_data, ref_entries)
                else:
                    abstract_text = ""

                # Clean abstract markers too
                abstract_text = _UNARXIVE_MARKER_RE.sub("", abstract_text).strip()

                # Resolve DOI: S2AG lookup (best) → arXiv metadata → synthetic arXiv DOI
                doi = None
                year = None

                # Try S2AG lookup first (has publisher DOIs)
                s2ag_info = doi_lookup.get(paper_id)
                if s2ag_info:
                    doi = s2ag_info["doi"]
                    year = s2ag_info.get("year")
                    if not title and s2ag_info.get("title"):
                        title = s2ag_info["title"]

                # Fallback to arXiv metadata
                if not doi and arxiv_meta_lookup:
                    meta_info = arxiv_meta_lookup.get(paper_id)
                    if meta_info:
                        if meta_info.get("doi"):
                            doi = meta_info["doi"].lower()
                        if not year and meta_info.get("year"):
                            year = meta_info["year"]
                        if not title and meta_info.get("title"):
                            title = meta_info["title"]
                        if not abstract_text and meta_info.get("abstract"):
                            abstract_text = meta_info["abstract"].strip()

                # Final fallback: arXiv-issued DOI
                if not doi:
                    doi = f"10.48550/arxiv.{paper_id}"

                # Clean DOI
                doi = doi.lower().strip()
                if "doi.org/" in doi:
                    doi = doi.split("doi.org/")[-1]

                # Try to get year from paper_id (YYMM format)
                if not year:
                    try:
                        yymm = paper_id.split(".")[0] if "." in paper_id else paper_id[:4]
                        yy = int(yymm[:2])
                        year = 2000 + yy if yy < 90 else 1900 + yy
                    except (ValueError, IndexError):
                        pass

                records.append({
                    "doi": doi,
                    "title": title or "",
                    "abstract": abstract_text or "",
                    "text": text,
                    "year": year,
                    "source_id": paper_id,
                    "text_length": len(text),
                })

    except Exception as e:
        return {
            "input": str(jsonl_path),
            "error": str(e),
            "success": False,
            "records": [],
        }

    return {
        "input": str(jsonl_path),
        "rows": len(records),
        "errors": errors,
        "success": True,
        "records": records,
    }


def convert_arxiv(workers: int = 4, force: bool = False):
    """Convert arXiv data (unarXive format) to fulltext schema.

    Uses unarXive JSONL files which contain pre-extracted structured text
    from LaTeX sources. Links to DOIs via S2AG papers table and arXiv
    metadata. Papers without a publisher DOI get the arXiv-issued
    DOI (10.48550/arXiv.{id}).
    """
    raw_dir = RAW_DIR / "arxiv"
    output_dir = OUTPUT_DIR / "arxiv"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already converted
    existing = list(output_dir.glob("*.parquet"))
    if existing and not force:
        print(f"  Already converted ({len(existing)} files). Use --force to reconvert.")
        return True

    # Find unarXive JSONL files (after extraction from tar.xz)
    # Try both extracted directory and any JSONL files
    unarxive_dir = raw_dir
    jsonl_files = sorted(unarxive_dir.rglob("*.jsonl"))

    if not jsonl_files:
        # Check if tar.xz needs extraction
        tar_xz = list(raw_dir.glob("*.tar.xz"))
        if tar_xz:
            print(f"  Found {len(tar_xz)} tar.xz archive(s). Extracting first...")
            import subprocess
            for txz in tar_xz:
                print(f"    Extracting {txz.name}...")
                subprocess.run(
                    ["tar", "-xJf", str(txz), "-C", str(raw_dir)],
                    check=True,
                )
            jsonl_files = sorted(unarxive_dir.rglob("*.jsonl"))

    if not jsonl_files:
        print("  ERROR: No unarXive JSONL files found in", raw_dir)
        print("  Download unarXive first, then extract the tar.xz")
        return False

    print(f"  Found {len(jsonl_files)} unarXive JSONL files")

    # Clean existing output
    for f in existing:
        f.unlink()

    t0 = time.time()

    # Step 1: Build DOI lookup from S2AG papers (arxiv_id → doi)
    print("  Step 1: Building arXiv ID → DOI lookup from S2AG...")
    t1 = time.time()
    doi_lookup = {}

    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={workers}")
    papers_path = S2AG_PARQUET / "papers" / "*.parquet"
    if (S2AG_PARQUET / "papers").exists():
        doi_expr = DOI_CLEAN_SQL.format(doi_expr="externalids.DOI")
        rows = conn.execute(f"""
            SELECT
                externalids.ArXiv AS arxiv_id,
                {doi_expr} AS doi,
                title,
                CAST(year AS INTEGER) AS year
            FROM read_parquet('{papers_path}')
            WHERE externalids.ArXiv IS NOT NULL AND externalids.ArXiv != ''
        """).fetchall()
        for arxiv_id, doi, title, year in rows:
            doi_lookup[arxiv_id] = {
                "doi": doi if doi else None,
                "title": title,
                "year": year,
            }
        print(f"    {len(doi_lookup):,} arXiv papers from S2AG ({time.time()-t1:.1f}s)")
    else:
        print("    WARNING: S2AG papers not available, DOI lookup will be limited")
    conn.close()

    # Step 2: Build metadata lookup from arXiv JSON (for DOI fallback + year)
    print("  Step 2: Building arXiv metadata lookup...")
    t1 = time.time()
    arxiv_meta_lookup = {}
    meta_file = raw_dir / "arxiv-metadata-oai.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    aid = d.get("id", "")
                    if aid:
                        entry = {}
                        if d.get("doi"):
                            entry["doi"] = d["doi"]
                        if d.get("title"):
                            entry["title"] = d["title"]
                        if d.get("abstract"):
                            entry["abstract"] = d["abstract"]
                        # Parse year from versions or update_date
                        update_date = d.get("update_date", "")
                        if update_date and len(update_date) >= 4:
                            try:
                                entry["year"] = int(update_date[:4])
                            except ValueError:
                                pass
                        if entry:
                            arxiv_meta_lookup[aid] = entry
                except json.JSONDecodeError:
                    pass
        print(f"    {len(arxiv_meta_lookup):,} papers from arXiv metadata ({time.time()-t1:.1f}s)")
    else:
        print("    WARNING: arXiv metadata JSON not found, using S2AG only")

    # Step 3: Process unarXive files with ProcessPoolExecutor
    print(f"  Step 3: Processing {len(jsonl_files)} JSONL files with {workers} workers...")
    t1 = time.time()

    total_rows = 0
    total_errors = 0
    all_records = []

    # For ProcessPoolExecutor, we pass the lookups to each worker
    # Since lookups can be large, process files sequentially with shared lookups
    # (ProcessPoolExecutor would copy the dicts per worker → memory explosion)
    for i, jsonl_file in enumerate(jsonl_files):
        result = _process_unarxive_file((str(jsonl_file), doi_lookup, arxiv_meta_lookup))

        if result["success"]:
            rows = result.get("rows", 0)
            errors = result.get("errors", 0)
            total_rows += rows
            total_errors += errors
            all_records.extend(result["records"])
            print(f"    [{i+1}/{len(jsonl_files)}] {jsonl_file.name}: "
                  f"{rows:,} papers, {errors} errors")
        else:
            print(f"    [{i+1}/{len(jsonl_files)}] {jsonl_file.name}: "
                  f"ERROR - {result['error']}")

    print(f"    Parsed {total_rows:,} papers ({total_errors} errors, {time.time()-t1:.1f}s)")

    if not all_records:
        print("  ERROR: No records extracted")
        return False

    # Step 4: Write to parquet via DuckDB
    print(f"  Step 4: Writing {len(all_records):,} records to parquet...")
    t1 = time.time()

    conn = duckdb.connect(":memory:")
    conn.execute(f"SET threads={workers}")
    conn.execute("SET memory_limit='64GB'")

    conn.execute("""
        CREATE TABLE articles (
            doi VARCHAR,
            source VARCHAR,
            title VARCHAR,
            abstract VARCHAR,
            text VARCHAR,
            license VARCHAR,
            year INTEGER,
            source_id VARCHAR,
            text_length INTEGER,
            language VARCHAR,
            has_full_text BOOLEAN
        )
    """)

    # Insert in batches to avoid huge memory spike
    BATCH_SIZE = 50_000
    for batch_start in range(0, len(all_records), BATCH_SIZE):
        batch = all_records[batch_start:batch_start + BATCH_SIZE]
        conn.executemany(
            "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["doi"],
                    "arxiv",
                    r["title"],
                    r["abstract"],
                    r["text"],
                    None,  # license (unarXive open subset is permissive)
                    r["year"],
                    r["source_id"],
                    r["text_length"],
                    None,  # language
                    True,  # all unarXive entries have body text
                )
                for r in batch
            ],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    conn.execute(f"""
        COPY articles TO '{output_dir}/' (
            FORMAT PARQUET,
            PER_THREAD_OUTPUT true,
            ROW_GROUP_SIZE {ROW_GROUP_SIZE},
            COMPRESSION zstd,
            COMPRESSION_LEVEL 3
        )
    """)

    parquet_files = sorted(output_dir.glob("*.parquet"))
    total_size = sum(f.stat().st_size for f in parquet_files)
    n_rows = conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{output_dir}/*.parquet')"
    ).fetchone()[0]

    # Stats
    doi_stats = conn.execute("""
        SELECT
            COUNT(*) FILTER (WHERE doi LIKE '10.48550/arxiv.%') as arxiv_dois,
            COUNT(*) FILTER (WHERE doi NOT LIKE '10.48550/arxiv.%') as publisher_dois
        FROM articles
    """).fetchone()

    conn.close()

    elapsed = time.time() - t0
    print(f"  Done: {n_rows:,} rows, {len(parquet_files)} files, "
          f"{total_size / (1024**3):.2f} GB, {elapsed:.1f}s")
    print(f"    Publisher DOIs: {doi_stats[1]:,}")
    print(f"    arXiv DOIs (10.48550): {doi_stats[0]:,}")

    return True


# ── Summary ─────────────────────────────────────────────────────────────────

def print_summary():
    """Print summary of all fulltext parquet data."""
    print("\n=== Fulltext Parquet Summary ===\n")

    total_files = 0
    total_size = 0
    total_rows = 0

    conn = duckdb.connect(":memory:")

    for source in SOURCES:
        source_dir = OUTPUT_DIR / source
        if not source_dir.exists():
            print(f"  {source:10s}: not converted")
            continue

        files = list(source_dir.glob("*.parquet"))
        if not files:
            print(f"  {source:10s}: no parquet files")
            continue

        size = sum(f.stat().st_size for f in files)
        total_files += len(files)
        total_size += size

        try:
            n_rows = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{source_dir}/*.parquet')"
            ).fetchone()[0]
            total_rows += n_rows

            n_fulltext = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{source_dir}/*.parquet') "
                f"WHERE has_full_text"
            ).fetchone()[0]

            n_dois = conn.execute(
                f"SELECT COUNT(DISTINCT doi) FROM read_parquet('{source_dir}/*.parquet') "
                f"WHERE doi IS NOT NULL AND doi != ''"
            ).fetchone()[0]

            print(
                f"  {source:10s}: {len(files):4d} files, {size / (1024**3):7.2f} GB, "
                f"{n_rows:>12,} rows, {n_fulltext:>12,} full-text, "
                f"{n_dois:>12,} DOIs"
            )
        except Exception as e:
            print(f"  {source:10s}: {len(files)} files, {size / (1024**3):.2f} GB (error counting: {e})")

    # Unified (deduplicated)
    unified_dir = OUTPUT_DIR / "unified"
    if unified_dir.exists() and list(unified_dir.glob("*.parquet")):
        files = list(unified_dir.glob("*.parquet"))
        size = sum(f.stat().st_size for f in files)
        try:
            n_rows = conn.execute(
                f"SELECT COUNT(*) FROM read_parquet('{unified_dir}/*.parquet')"
            ).fetchone()[0]
            print(
                f"\n  {'unified':10s}: {len(files):4d} files, {size / (1024**3):7.2f} GB, "
                f"{n_rows:>12,} rows (deduplicated)"
            )
        except Exception:
            pass

    conn.close()

    print(f"\n  Total (per-source): {total_files} files, {total_size / (1024**3):.2f} GB, "
          f"{total_rows:,} rows")


# ── CLI ─────────────────────────────────────────────────────────────────────

CONVERTERS = {
    "s2orc": convert_s2orc,
    "pes2o": convert_pes2o,
    "pmc": convert_pmc,
    "arxiv": convert_arxiv,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert full-text paper sources to unified Parquet schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--source", type=str, choices=SOURCES,
                        help="Convert a specific source")
    parser.add_argument("--all", action="store_true",
                        help="Convert all available sources")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary of existing parquet files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if output already exists")

    args = parser.parse_args()

    print("=== Full-Text Papers Converter ===")
    print(f"Output: {OUTPUT_DIR}")

    if args.summary:
        print_summary()
        return 0

    if args.source:
        converter = CONVERTERS.get(args.source)
        if not converter:
            print(f"  Converter for '{args.source}' not yet implemented (deferred phase)")
            return 1
        print(f"\n[{args.source}]")
        ok = converter(workers=args.workers, force=args.force)
        return 0 if ok else 1

    if args.all:
        results = {}
        for source, converter in CONVERTERS.items():
            print(f"\n[{source}]")
            results[source] = converter(workers=args.workers, force=args.force)

        print("\n=== Conversion Summary ===")
        for source, ok in results.items():
            print(f"  {source}: {'OK' if ok else 'FAILED'}")

        print_summary()
        return 0 if all(results.values()) else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
