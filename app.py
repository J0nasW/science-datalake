#!/usr/bin/env python3
"""
Science Data Lake Explorer — Interactive Gradio demo.

Provides a SQL interface to query the unified science data lake via DuckDB,
reading Parquet files from HuggingFace Datasets or a local path.

Designed for HuggingFace Spaces (free CPU tier: 2-core, 16GB RAM).

Usage:
    Local:    python app.py --local /path/to/science_datalake
    HF:       python app.py --hf-dataset username/science-datalake
    Default:  python app.py  (auto-detects local or HF)
"""

import argparse
import os
import re
import time
from pathlib import Path

import duckdb
import gradio as gr

# ── Configuration ────────────────────────────────────────────────────────────

MAX_ROWS = 5000
QUERY_TIMEOUT_SEC = 120
HF_DATASET = os.environ.get("HF_DATASET", "")
LOCAL_PATH = os.environ.get("LOCAL_PATH", "")

# SQL safety: block DDL/DML and dangerous statements
BLOCKED_PATTERNS = re.compile(
    r"\b(CREATE|DROP|ALTER|INSERT|UPDATE|DELETE|TRUNCATE|GRANT|REVOKE|"
    r"COPY|ATTACH|DETACH|LOAD|INSTALL|SET|PRAGMA|CALL|EXECUTE|EXPORT|IMPORT)\b",
    re.IGNORECASE,
)

EXAMPLE_QUERIES = {
    "Quick overview: row counts per dataset": """
SELECT 'OpenAlex works' AS dataset, COUNT(*) AS rows FROM openalex.works
UNION ALL SELECT 'S2AG papers', COUNT(*) FROM s2ag.papers
UNION ALL SELECT 'SciSciNet papers', COUNT(*) FROM sciscinet.papers
UNION ALL SELECT 'PWC papers', COUNT(*) FROM pwc.papers
UNION ALL SELECT 'Retraction Watch', COUNT(*) FROM retwatch.retraction_watch
UNION ALL SELECT 'RoS patent pairs', COUNT(*) FROM ros.patent_paper_pairs
UNION ALL SELECT 'Unified papers (xref)', COUNT(*) FROM xref.unified_papers
ORDER BY rows DESC
""",

    "Top 10 most-cited papers with code on GitHub": """
SELECT
    u.doi,
    u.title,
    u.year,
    u.oa_cited_by_count,
    u.sciscinet_disruption,
    u.has_retraction
FROM xref.unified_papers u
WHERE u.has_pwc AND u.oa_cited_by_count IS NOT NULL
ORDER BY u.oa_cited_by_count DESC
LIMIT 10
""",

    "Retracted papers by year and average disruption": """
SELECT
    u.year,
    COUNT(*) AS retracted_count,
    ROUND(AVG(u.sciscinet_disruption), 4) AS avg_disruption,
    ROUND(AVG(u.oa_cited_by_count), 1) AS avg_citations
FROM xref.unified_papers u
WHERE u.has_retraction AND u.year BETWEEN 2000 AND 2024
GROUP BY u.year
ORDER BY u.year
""",

    "Cross-source citation count comparison (sample)": """
SELECT
    u.doi,
    u.title,
    u.year,
    u.s2ag_citationcount AS s2ag_citations,
    u.oa_cited_by_count AS openalex_citations,
    u.sciscinet_citation_count AS sciscinet_citations,
    ABS(COALESCE(u.s2ag_citationcount, 0) - COALESCE(u.oa_cited_by_count, 0)) AS s2ag_oa_diff
FROM xref.unified_papers u
WHERE u.has_s2ag AND u.has_openalex AND u.has_sciscinet
    AND u.oa_cited_by_count > 100
ORDER BY s2ag_oa_diff DESC
LIMIT 20
""",

    "Coverage statistics: source overlaps": """
SELECT
    has_s2ag, has_openalex, has_sciscinet, has_pwc, has_retraction, has_patent,
    COUNT(*) AS papers
FROM xref.unified_papers
GROUP BY has_s2ag, has_openalex, has_sciscinet, has_pwc, has_retraction, has_patent
ORDER BY papers DESC
LIMIT 20
""",

    "Most disruptive papers with patent citations": """
SELECT
    u.doi,
    u.title,
    u.year,
    ROUND(u.sciscinet_disruption, 4) AS disruption,
    u.oa_cited_by_count AS citations,
    u.sciscinet_team_size AS team_size
FROM xref.unified_papers u
WHERE u.has_patent AND u.sciscinet_disruption IS NOT NULL
ORDER BY u.sciscinet_disruption DESC
LIMIT 15
""",

    "OpenAlex topic distribution (top 20)": """
SELECT
    t.display_name AS topic,
    t.subfield_display_name AS subfield,
    t.field_display_name AS field,
    t.domain_display_name AS domain,
    COUNT(*) AS paper_count
FROM openalex.works_topics wt
JOIN openalex.topics t ON wt.topic_id = t.id
GROUP BY t.display_name, t.subfield_display_name, t.field_display_name, t.domain_display_name
ORDER BY paper_count DESC
LIMIT 20
""",

    "Ontology term lookup (CSO — Computer Science)": """
SELECT id, label, description
FROM cso.cso_terms
ORDER BY label
LIMIT 20
""",

    "Topic-ontology alignment (embedding-based)": """
SELECT
    m.topic_name,
    m.ontology,
    m.ontology_term_label,
    ROUND(m.similarity, 3) AS cosine_sim,
    m.match_type,
    m.domain
FROM xref.topic_ontology_map m
WHERE m.similarity >= 0.85
ORDER BY m.similarity DESC
LIMIT 25
""",
}


def get_connection(local_path: str = "", hf_dataset: str = ""):
    """Create a DuckDB connection with appropriate data sources."""
    conn = duckdb.connect(":memory:")
    conn.execute("SET threads=2")
    conn.execute("SET memory_limit='12GB'")

    if local_path:
        # Local mode: open the pre-built DuckDB with views
        db_path = Path(local_path) / "datalake.duckdb"
        if db_path.exists():
            conn.close()
            conn = duckdb.connect(str(db_path), read_only=True)
            conn.execute("SET threads=2")
            conn.execute("SET memory_limit='12GB'")
            return conn
        else:
            raise FileNotFoundError(f"Database not found: {db_path}")

    if hf_dataset:
        # HuggingFace mode: create views over hf:// Parquet paths
        base = f"hf://datasets/{hf_dataset}"
        conn.execute("INSTALL httpfs; LOAD httpfs;")

        # Auto-discover configs by listing known schemas
        schemas = {
            "s2ag": ["papers", "abstracts", "authors", "citations", "tldrs"],
            "openalex": ["works", "authors", "topics", "works_topics", "works_authorships"],
            "sciscinet": ["papers"],
            "pwc": ["papers", "papers_fulltexts", "cso_topics"],
            "retwatch": ["retraction_watch"],
            "ros": ["patent_paper_pairs"],
            "xref": ["unified_papers"],
        }
        for schema, tables in schemas.items():
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
            for table in tables:
                path = f"{base}/{schema}/{table}/*.parquet"
                try:
                    conn.execute(
                        f"CREATE VIEW {schema}.{table} AS "
                        f"SELECT * FROM read_parquet('{path}')"
                    )
                except Exception:
                    pass  # table may not exist in the dataset

        return conn

    raise ValueError("Specify --local or --hf-dataset (or set LOCAL_PATH / HF_DATASET env vars)")


def validate_query(sql: str) -> str | None:
    """Validate a SQL query for safety. Returns error message or None if OK."""
    sql_stripped = sql.strip().rstrip(";")
    if not sql_stripped:
        return "Empty query"

    if BLOCKED_PATTERNS.search(sql_stripped):
        return "Only SELECT queries are allowed. DDL/DML statements are blocked."

    # Basic check: must start with SELECT or WITH
    first_word = sql_stripped.split()[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return f"Query must start with SELECT or WITH, got: {first_word}"

    return None


def run_query(sql: str, conn):
    """Execute a SQL query and return results as a dataframe."""
    error = validate_query(sql)
    if error:
        return None, error, ""

    sql = sql.strip().rstrip(";")
    # Add LIMIT if not present
    if "LIMIT" not in sql.upper():
        sql += f"\nLIMIT {MAX_ROWS}"

    t0 = time.time()
    try:
        result = conn.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()
        elapsed = time.time() - t0

        if not rows:
            return None, "", f"No results ({elapsed:.2f}s)"

        # Build a list of dicts for Gradio dataframe
        import pandas as pd
        df = pd.DataFrame(rows, columns=columns)
        status = f"{len(rows):,} rows ({elapsed:.2f}s)"
        if len(rows) >= MAX_ROWS:
            status += f" [truncated to {MAX_ROWS}]"
        return df, "", status

    except duckdb.Error as e:
        elapsed = time.time() - t0
        return None, f"SQL Error: {e}", f"({elapsed:.2f}s)"
    except Exception as e:
        return None, f"Error: {e}", ""


def build_app(conn):
    """Build the Gradio interface."""
    with gr.Blocks(
        title="Science Data Lake Explorer",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("""
# Science Data Lake Explorer

Query 480M+ scholarly papers across 8 integrated datasets using SQL.
**Datasets**: OpenAlex, Semantic Scholar (S2AG), SciSciNet, Papers With Code,
Retraction Watch, Reliance on Science, Preprint-to-Paper, and 13 scientific ontologies.
        """)

        with gr.Row():
            with gr.Column(scale=3):
                sql_input = gr.Code(
                    label="SQL Query",
                    language="sql",
                    value=list(EXAMPLE_QUERIES.values())[0].strip(),
                    lines=10,
                )
            with gr.Column(scale=1):
                example_dropdown = gr.Dropdown(
                    choices=list(EXAMPLE_QUERIES.keys()),
                    label="Example Queries",
                    value=list(EXAMPLE_QUERIES.keys())[0],
                )
                run_btn = gr.Button("Run Query", variant="primary", size="lg")

        with gr.Row():
            error_output = gr.Textbox(label="Errors", visible=True, max_lines=3)
            status_output = gr.Textbox(label="Status", visible=True, max_lines=1)

        results_output = gr.Dataframe(
            label="Results",
            interactive=False,
            wrap=True,
        )

        # Event handlers
        def on_example_change(example_name):
            return EXAMPLE_QUERIES.get(example_name, "").strip()

        def on_run(sql):
            df, error, status = run_query(sql, conn)
            return df, error, status

        example_dropdown.change(on_example_change, example_dropdown, sql_input)
        run_btn.click(on_run, sql_input, [results_output, error_output, status_output])
        sql_input.submit(on_run, sql_input, [results_output, error_output, status_output])

        gr.Markdown("""
---
**Available schemas**: `s2ag`, `openalex`, `sciscinet`, `pwc`, `retwatch`, `ros`, `p2p`, `xref`,
plus 13 ontology schemas (`cso`, `mesh`, `go`, `doid`, `chebi`, `ncit`, `hpo`, `edam`, `agrovoc`, `unesco`, `stw`, `msc2020`, `physh`).

**Key tables**: `xref.unified_papers` (cross-dataset join), `xref.doi_map` (DOI normalization),
`openalex.works`, `s2ag.papers`, `sciscinet.papers`, `pwc.papers`.

Built with DuckDB + Parquet. Only SELECT queries allowed.
        """)

    return app


def main():
    parser = argparse.ArgumentParser(description="Science Data Lake Explorer")
    parser.add_argument("--local", type=str, default=LOCAL_PATH,
                        help="Path to local data lake root")
    parser.add_argument("--hf-dataset", type=str, default=HF_DATASET,
                        help="HuggingFace dataset ID (e.g., username/science-datalake)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    local = args.local or LOCAL_PATH
    hf = args.hf_dataset or HF_DATASET

    # Auto-detect: if local datalake.duckdb exists in common locations, use it
    if not local and not hf:
        for candidate in [
            Path(__file__).resolve().parent,
            Path.home() / "science_datalake",
            Path("."),
        ]:
            if (candidate / "datalake.duckdb").exists():
                local = str(candidate)
                break

    if not local and not hf:
        parser.error("Specify --local or --hf-dataset")

    print(f"Connecting to {'local: ' + local if local else 'HuggingFace: ' + hf}...")
    conn = get_connection(local_path=local, hf_dataset=hf)
    print("Connected!")

    app = build_app(conn)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
