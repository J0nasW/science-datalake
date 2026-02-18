#!/usr/bin/env python3
"""
CLI for querying the Oxigraph RDF store with SPARQL.

Usage:
    python scripts/sparql_query.py "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10"
    python scripts/sparql_query.py --graph mesh "SELECT ?term ?label WHERE { ?term <http://www.w3.org/2004/02/skos/core#prefLabel> ?label } LIMIT 10"
    python scripts/sparql_query.py --file query.rq
    python scripts/sparql_query.py --graphs   # List available named graphs
"""

import argparse
import sys
import time
from pathlib import Path

import pyoxigraph as ox

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

OXIGRAPH_DIR = ROOT / "ontologies.oxigraph"

# Common prefixes auto-injected if not present
COMMON_PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#>
PREFIX mesh: <http://id.nlm.nih.gov/mesh/>
PREFIX obo: <http://purl.obolibrary.org/obo/>
"""


def get_store() -> ox.Store:
    if not OXIGRAPH_DIR.exists():
        print(f"Oxigraph store not found: {OXIGRAPH_DIR}")
        print("Run: python scripts/convert_ontologies.py --all")
        sys.exit(1)
    return ox.Store(str(OXIGRAPH_DIR))


def inject_prefixes(query: str) -> str:
    """Add common prefixes if not already declared."""
    query_upper = query.upper()
    if "PREFIX" not in query_upper:
        return COMMON_PREFIXES + query
    return query


def wrap_graph(query: str, graph_name: str) -> str:
    """Wrap a query to restrict it to a specific named graph."""
    import re
    graph_iri = f"urn:ontology:{graph_name}"
    # Find WHERE { ... } and wrap the body in GRAPH <iri> { ... }
    # The closing brace of WHERE is the last } before optional modifiers (LIMIT, ORDER BY, etc.)
    pattern = r"(WHERE\s*\{)(.*)\}(\s*(?:LIMIT|ORDER|GROUP|HAVING|OFFSET|VALUES).*)?$"
    match = re.search(pattern, query, re.DOTALL | re.IGNORECASE)
    if match:
        before = query[:match.start()]
        body = match.group(2)
        after = match.group(3) or ""
        return f"{before}WHERE {{ GRAPH <{graph_iri}> {{{body}}} }}{after}"
    return query


def format_term(term) -> str:
    """Format an RDF term for display."""
    if isinstance(term, ox.NamedNode):
        return str(term.value)
    elif isinstance(term, ox.Literal):
        return str(term.value)
    elif isinstance(term, ox.BlankNode):
        return f"_:{term.value}"
    elif term is None:
        return ""
    return str(term)


def run_query(store: ox.Store, query: str):
    """Execute a SPARQL query and print results."""
    t0 = time.time()
    try:
        results = store.query(query)
    except Exception as e:
        print(f"SPARQL error: {e}")
        return

    elapsed = time.time() - t0

    if isinstance(results, bool):
        # ASK query
        print(f"Result: {results}")
        print(f"({elapsed:.3f}s)")
        return

    # SELECT query - results is an iterator of QuerySolution
    rows = list(results)
    if not rows:
        print("(no results)")
        print(f"({elapsed:.3f}s)")
        return

    # Get variable names from first result
    variables = list(results.variables)
    var_names = [str(v.value) for v in variables]

    # Calculate column widths
    col_data = []
    for row in rows:
        row_data = []
        for var in variables:
            val = row[var]
            row_data.append(format_term(val))
        col_data.append(row_data)

    widths = [len(name) for name in var_names]
    for row_data in col_data:
        for i, val in enumerate(row_data):
            widths[i] = max(widths[i], min(len(val), 80))

    # Print header
    header = " | ".join(name.ljust(widths[i]) for i, name in enumerate(var_names))
    print(header)
    print("-" * len(header))

    # Print rows
    for row_data in col_data:
        line = " | ".join(
            val[:80].ljust(widths[i]) for i, val in enumerate(row_data)
        )
        print(line)

    print(f"\n({len(rows)} rows, {elapsed:.3f}s)")


def list_graphs(store: ox.Store):
    """List all named graphs with triple counts."""
    query = """
    SELECT ?g (COUNT(*) AS ?triples)
    WHERE { GRAPH ?g { ?s ?p ?o } }
    GROUP BY ?g
    ORDER BY ?g
    """
    print("Named graphs in Oxigraph store:\n")
    run_query(store, query)


def main():
    parser = argparse.ArgumentParser(
        description="Query the Oxigraph RDF store with SPARQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query", nargs="?", help="SPARQL query string")
    parser.add_argument("--file", "-f", type=str, help="Read SPARQL from file")
    parser.add_argument("--graph", "-g", type=str, help="Restrict to named graph (e.g., mesh, go)")
    parser.add_argument("--graphs", action="store_true", help="List all named graphs")
    parser.add_argument("--no-prefixes", action="store_true", help="Don't auto-inject common prefixes")
    args = parser.parse_args()

    store = get_store()

    if args.graphs:
        list_graphs(store)
        return 0

    if args.file:
        with open(args.file) as f:
            query = f.read()
    elif args.query:
        query = args.query
    else:
        parser.print_help()
        return 1

    if not args.no_prefixes:
        query = inject_prefixes(query)

    if args.graph:
        query = wrap_graph(query, args.graph)

    run_query(store, query)
    return 0


if __name__ == "__main__":
    sys.exit(main())
