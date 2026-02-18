#!/usr/bin/env python3
"""
Convert downloaded ontologies to Oxigraph (RDF graph store) and Parquet (flat tables).

For each ontology, produces:
  - Oxigraph named graph (urn:ontology:{name}) in ontologies.oxigraph/
  - datasets/{name}/parquet/{name}_terms.parquet
  - datasets/{name}/parquet/{name}_hierarchy.parquet
  - datasets/{name}/parquet/{name}_xrefs.parquet

Usage:
    python scripts/convert_ontologies.py --all
    python scripts/convert_ontologies.py --ontology mesh,go
    python scripts/convert_ontologies.py --oxigraph-only
    python scripts/convert_ontologies.py --parquet-only
    python scripts/convert_ontologies.py --status
"""

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyoxigraph as ox

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
try:
    from config import find_datalake_root
    ROOT = find_datalake_root()
except Exception:
    pass

from ontology_registry import ALL_ONTOLOGY_NAMES, ONTOLOGIES

ONTOLOGIES_DIR = ROOT / "ontologies"
OXIGRAPH_DIR = ROOT / "ontologies.oxigraph"
DATASETS_DIR = ROOT / "datasets"

# Parquet write options
PARQUET_OPTS = {"compression": "zstd", "compression_level": 3}

# ------------------------------------------------------------------
# Oxigraph loading
# ------------------------------------------------------------------

RDF_FORMATS = {
    "nt": ox.RdfFormat.N_TRIPLES,
    "nt_gz": ox.RdfFormat.N_TRIPLES,
    "ttl": ox.RdfFormat.TURTLE,
    "rdf": ox.RdfFormat.RDF_XML,
    "owl": ox.RdfFormat.RDF_XML,
}


def _get_store() -> ox.Store:
    """Open (or create) the persistent Oxigraph store."""
    OXIGRAPH_DIR.mkdir(parents=True, exist_ok=True)
    return ox.Store(str(OXIGRAPH_DIR))


def _graph_iri(name: str) -> ox.NamedNode:
    return ox.NamedNode(f"urn:ontology:{name}")


def _find_raw_file(name: str, info: dict) -> Path | None:
    """Find the raw ontology file to load."""
    ont_dir = ONTOLOGIES_DIR / name
    if not ont_dir.exists():
        return None

    dl = info["download"]
    extract = dl.get("extract")

    # For compressed files, look for the extracted version first
    if extract == "gz":
        extracted_name = dl["filename"].rsplit(".gz", 1)[0]
        extracted = ont_dir / extracted_name
        if extracted.exists():
            return extracted
    elif extract == "zip":
        extracted_name = dl["filename"].rsplit(".zip", 1)[0]
        extracted = ont_dir / extracted_name
        if extracted.exists():
            return extracted

    # Try the main download file
    main_file = ont_dir / dl["filename"]
    if main_file.exists():
        return main_file

    # Try fallback filename
    fb = dl.get("fallback_filename")
    if fb:
        fbp = ont_dir / fb
        if fbp.exists():
            return fbp

    return None


def load_to_oxigraph(name: str, info: dict, store: ox.Store) -> bool:
    """Load an ontology into Oxigraph. Returns True on success."""
    fmt = info["format"]
    parser_type = info["parser"]
    graph = _graph_iri(name)

    # Clear existing data for this graph
    for quad in store.quads_for_pattern(None, None, None, graph):
        store.remove(quad)

    raw_file = _find_raw_file(name, info)
    if raw_file is None:
        print(f"    No raw file found for {name}")
        return False

    if parser_type in ("obo_pronto", "msc_csv", "cso_csv"):
        # These need conversion to N-Triples via rdflib, then load
        nt_data = _convert_to_ntriples(name, info, raw_file)
        if nt_data:
            store.load(
                io.BytesIO(nt_data),
                ox.RdfFormat.N_TRIPLES,
                to_graph=graph,
            )
            return True
        return False

    # Native RDF formats: load directly
    rdf_fmt = RDF_FORMATS.get(fmt)
    if rdf_fmt is None:
        print(f"    Unknown format: {fmt}")
        return False

    # PhySH: also load the SKOS compat file
    if name == "physh":
        compat = ONTOLOGIES_DIR / name / "physh_skos_compat.ttl"
        if compat.exists():
            with open(compat, "rb") as f:
                store.load(f, ox.RdfFormat.TURTLE, to_graph=graph)

    with open(raw_file, "rb") as f:
        store.load(f, rdf_fmt, to_graph=graph)

    return True


def _convert_to_ntriples(name: str, info: dict, raw_file: Path) -> bytes | None:
    """Convert OBO/CSV to N-Triples via rdflib for Oxigraph ingestion."""
    import rdflib

    g = rdflib.Graph()

    parser_type = info["parser"]
    if parser_type == "obo_pronto":
        # Use pronto to parse OBO/OWL, then serialize terms as RDF
        import pronto
        try:
            ont = pronto.Ontology(str(raw_file))
        except Exception as e:
            print(f"    pronto parse error: {e}")
            return None

        SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
        OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
        g.bind("skos", SKOS)
        g.bind("owl", OWL)

        for term in ont.terms():
            uri = rdflib.URIRef(str(term.id) if ":" not in str(term.id) or str(term.id).startswith("http") else f"http://purl.obolibrary.org/obo/{str(term.id).replace(':', '_')}")
            g.add((uri, rdflib.RDF.type, OWL.Class))
            if term.name:
                g.add((uri, SKOS.prefLabel, rdflib.Literal(term.name)))
            if term.definition:
                g.add((uri, SKOS.definition, rdflib.Literal(str(term.definition))))
            if term.obsolete:
                g.add((uri, OWL.deprecated, rdflib.Literal(True)))
            for parent in term.superclasses(distance=1, with_self=False):
                parent_uri = rdflib.URIRef(str(parent.id) if str(parent.id).startswith("http") else f"http://purl.obolibrary.org/obo/{str(parent.id).replace(':', '_')}")
                g.add((uri, SKOS.broader, parent_uri))
            for syn in term.synonyms:
                g.add((uri, SKOS.altLabel, rdflib.Literal(syn.description)))
            for xref in term.xrefs:
                g.add((uri, SKOS.exactMatch, rdflib.Literal(str(xref.id))))

    elif parser_type == "msc_csv":
        SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")
        g.bind("skos", SKOS)
        base = "http://msc2020.org/resources/MSC/2020/"

        with open(raw_file, "r", encoding="latin-1") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 2:
                    continue
                code = row[0].strip()
                label = row[1].strip() if len(row) > 1 else ""
                if not code:
                    continue
                uri = rdflib.URIRef(f"{base}{code}")
                g.add((uri, rdflib.RDF.type, SKOS.Concept))
                g.add((uri, SKOS.notation, rdflib.Literal(code)))
                if label:
                    g.add((uri, SKOS.prefLabel, rdflib.Literal(label)))
                # Hierarchy from code structure
                if len(code) == 5:
                    parent_code = code[:3]
                    g.add((uri, SKOS.broader, rdflib.URIRef(f"{base}{parent_code}")))
                elif len(code) == 3:
                    parent_code = code[:2]
                    g.add((uri, SKOS.broader, rdflib.URIRef(f"{base}{parent_code}")))

    elif parser_type == "cso_csv":
        # CSO CSV is triples: "<subject>","<predicate>","<object>"
        # Already has full URIs, just need to convert to proper N-Triples
        with open(raw_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3:
                    continue
                s = row[0].strip().strip("<>").strip('"').strip("<>")
                p = row[1].strip().strip("<>").strip('"').strip("<>")
                o = row[2].strip().strip("<>").strip('"').strip("<>")
                if not s or not p or not o:
                    continue
                subj = rdflib.URIRef(s)
                pred = rdflib.URIRef(p)
                # Object: if it looks like a URI, treat as URIRef; otherwise Literal
                if o.startswith("http://") or o.startswith("https://"):
                    obj = rdflib.URIRef(o)
                else:
                    obj = rdflib.Literal(o)
                g.add((subj, pred, obj))

    else:
        return None

    nt_data = g.serialize(format="nt")
    return nt_data.encode("utf-8") if isinstance(nt_data, str) else nt_data


# ------------------------------------------------------------------
# Parquet export
# ------------------------------------------------------------------

def export_parquet_obo(name: str, info: dict) -> int:
    """Export OBO/OWL ontology to Parquet using pronto. Returns term count."""
    import pronto

    raw_file = _find_raw_file(name, info)
    if raw_file is None:
        return 0

    try:
        ont = pronto.Ontology(str(raw_file))
    except Exception as e:
        print(f"    pronto parse error: {e}")
        return 0

    terms_rows = []
    hier_rows = []
    xrefs_rows = []

    for term in ont.terms():
        tid = str(term.id)
        label = term.name or ""
        definition = str(term.definition) if term.definition else ""
        synonyms = [s.description for s in term.synonyms]
        namespace = str(term.namespace) if term.namespace else ""
        obsolete = bool(term.obsolete)

        terms_rows.append({
            "id": tid,
            "label": label,
            "definition": definition,
            "synonyms": synonyms,
            "namespace": namespace,
            "obsolete": obsolete,
        })

        for parent in term.superclasses(distance=1, with_self=False):
            hier_rows.append({
                "parent_id": str(parent.id),
                "child_id": tid,
                "relation": "is_a",
            })

        try:
            for rel in term.relationships:
                for target in term.relationships[rel]:
                    hier_rows.append({
                        "parent_id": str(target.id),
                        "child_id": tid,
                        "relation": rel.id if hasattr(rel, "id") else str(rel),
                    })
        except KeyError:
            pass  # Some OWL ontologies have relationship types pronto can't resolve

        for xref in term.xrefs:
            xref_str = str(xref.id)
            if ":" in xref_str:
                parts = xref_str.split(":", 1)
                xrefs_rows.append({
                    "term_id": tid,
                    "xref_db": parts[0],
                    "xref_id": parts[1],
                })
            else:
                xrefs_rows.append({
                    "term_id": tid,
                    "xref_db": "",
                    "xref_id": xref_str,
                })

    _write_parquet_tables(name, terms_rows, hier_rows, xrefs_rows)
    return len(terms_rows)


def export_parquet_skos(name: str, info: dict) -> int:
    """Export SKOS/RDF ontology to Parquet using rdflib. Returns term count."""
    import rdflib

    raw_file = _find_raw_file(name, info)
    if raw_file is None:
        return 0

    g = rdflib.Graph()

    # PhySH: load both files
    if name == "physh":
        compat = ONTOLOGIES_DIR / name / "physh_skos_compat.ttl"
        if compat.exists():
            g.parse(str(compat), format="turtle")

    fmt_map = {"nt": "nt", "ttl": "turtle", "rdf": "xml", "owl": "xml"}
    rdf_fmt = fmt_map.get(info["format"], "xml")
    try:
        g.parse(str(raw_file), format=rdf_fmt)
    except Exception as e:
        print(f"    rdflib parse error: {e}")
        return 0

    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

    terms_rows = []
    hier_rows = []
    xrefs_rows = []

    # Collect all SKOS concepts
    concepts = set(g.subjects(rdflib.RDF.type, SKOS.Concept))
    # Also try owl:Class for some ontologies
    OWL = rdflib.Namespace("http://www.w3.org/2002/07/owl#")
    concepts |= set(g.subjects(rdflib.RDF.type, OWL.Class))

    for concept in concepts:
        tid = str(concept)

        # Get label (prefer English)
        label = ""
        for _, _, o in g.triples((concept, SKOS.prefLabel, None)):
            lbl = str(o)
            lang = getattr(o, "language", None)
            if lang == "en" or not label:
                label = lbl
                if lang == "en":
                    break

        # Get definition
        definition = ""
        for pred in [SKOS.definition, SKOS.scopeNote]:
            for _, _, o in g.triples((concept, pred, None)):
                defn = str(o)
                lang = getattr(o, "language", None)
                if lang == "en" or not definition:
                    definition = defn
                    if lang == "en":
                        break
            if definition:
                break

        # Get synonyms (altLabels)
        synonyms = []
        for _, _, o in g.triples((concept, SKOS.altLabel, None)):
            lang = getattr(o, "language", None)
            if lang is None or lang == "en":
                synonyms.append(str(o))

        terms_rows.append({
            "id": tid,
            "label": label,
            "definition": definition,
            "synonyms": synonyms,
            "namespace": "",
            "obsolete": False,
        })

        # Hierarchy (broader)
        for _, _, parent in g.triples((concept, SKOS.broader, None)):
            hier_rows.append({
                "parent_id": str(parent),
                "child_id": tid,
                "relation": "broader",
            })

        # Also check broaderTransitive, narrower (inverted)
        for _, _, parent in g.triples((concept, SKOS.broaderTransitive, None)):
            hier_rows.append({
                "parent_id": str(parent),
                "child_id": tid,
                "relation": "broaderTransitive",
            })
        for _, child, _ in g.triples((None, SKOS.narrower, concept)):
            hier_rows.append({
                "parent_id": tid,
                "child_id": str(child),
                "relation": "narrower",
            })

        # Cross-references
        for pred in [SKOS.exactMatch, SKOS.closeMatch, SKOS.relatedMatch]:
            rel = str(pred).split("#")[-1]
            for _, _, match in g.triples((concept, pred, None)):
                match_str = str(match)
                xrefs_rows.append({
                    "term_id": tid,
                    "xref_db": rel,
                    "xref_id": match_str,
                })

    _write_parquet_tables(name, terms_rows, hier_rows, xrefs_rows)
    return len(terms_rows)


def export_parquet_mesh(name: str, info: dict) -> int:
    """Export MeSH N-Triples to Parquet using rdflib. Returns term count."""
    import rdflib

    raw_file = _find_raw_file(name, info)
    if raw_file is None:
        return 0

    g = rdflib.Graph()
    print(f"    Parsing MeSH N-Triples (this may take a minute)...")
    try:
        g.parse(str(raw_file), format="nt")
    except Exception as e:
        print(f"    rdflib parse error: {e}")
        return 0

    MESHV = rdflib.Namespace("http://id.nlm.nih.gov/mesh/vocab#")
    SKOS = rdflib.Namespace("http://www.w3.org/2004/02/skos/core#")

    terms_rows = []
    hier_rows = []
    xrefs_rows = []

    # MeSH descriptors
    descriptors = set(g.subjects(rdflib.RDF.type, MESHV.Descriptor))
    # Also include SCR (supplementary concepts) and qualifiers
    descriptors |= set(g.subjects(rdflib.RDF.type, MESHV.SCR_Chemical))
    descriptors |= set(g.subjects(rdflib.RDF.type, MESHV.SCR_Disease))
    descriptors |= set(g.subjects(rdflib.RDF.type, MESHV.SCR_Protocol))
    descriptors |= set(g.subjects(rdflib.RDF.type, MESHV.Qualifier))
    # Include Concepts too
    concepts = set(g.subjects(rdflib.RDF.type, MESHV.Concept))
    all_entities = descriptors | concepts

    for entity in all_entities:
        tid = str(entity)

        label = ""
        for _, _, o in g.triples((entity, rdflib.RDFS.label, None)):
            label = str(o)
            break
        if not label:
            for _, _, o in g.triples((entity, MESHV.prefLabel, None)):
                label = str(o)
                break

        definition = ""
        for _, _, o in g.triples((entity, MESHV.scopeNote, None)):
            definition = str(o)
            break
        if not definition:
            for _, _, o in g.triples((entity, SKOS.scopeNote, None)):
                definition = str(o)
                break

        synonyms = []
        for _, _, o in g.triples((entity, MESHV.altLabel, None)):
            synonyms.append(str(o))

        # Active status
        active = True
        for _, _, o in g.triples((entity, MESHV.active, None)):
            active = str(o).lower() in ("true", "1")

        namespace = "descriptor" if entity in descriptors else "concept"

        terms_rows.append({
            "id": tid,
            "label": label,
            "definition": definition,
            "synonyms": synonyms,
            "namespace": namespace,
            "obsolete": not active,
        })

        # Hierarchy
        for _, _, parent in g.triples((entity, MESHV.broaderDescriptor, None)):
            hier_rows.append({
                "parent_id": str(parent),
                "child_id": tid,
                "relation": "broaderDescriptor",
            })
        for _, _, parent in g.triples((entity, SKOS.broader, None)):
            hier_rows.append({
                "parent_id": str(parent),
                "child_id": tid,
                "relation": "broader",
            })

        # Cross-references
        for _, _, match in g.triples((entity, SKOS.exactMatch, None)):
            xrefs_rows.append({
                "term_id": tid,
                "xref_db": "exactMatch",
                "xref_id": str(match),
            })
        for _, _, match in g.triples((entity, SKOS.closeMatch, None)):
            xrefs_rows.append({
                "term_id": tid,
                "xref_db": "closeMatch",
                "xref_id": str(match),
            })

    _write_parquet_tables(name, terms_rows, hier_rows, xrefs_rows)
    return len(terms_rows)


def export_parquet_msc(name: str, info: dict) -> int:
    """Export MSC2020 CSV to Parquet. Returns term count."""
    raw_file = _find_raw_file(name, info)
    if raw_file is None:
        return 0

    terms_rows = []
    hier_rows = []

    with open(raw_file, "r", encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            code = row[0].strip()
            label = row[1].strip() if len(row) > 1 else ""
            if not code:
                continue

            terms_rows.append({
                "id": f"MSC:{code}",
                "label": label,
                "definition": row[2].strip() if len(row) > 2 else "",
                "synonyms": [],
                "namespace": f"level_{len(code)}",
                "obsolete": False,
            })

            # Hierarchy from code structure
            if len(code) == 5:
                hier_rows.append({
                    "parent_id": f"MSC:{code[:3]}",
                    "child_id": f"MSC:{code}",
                    "relation": "is_a",
                })
            elif len(code) == 3:
                hier_rows.append({
                    "parent_id": f"MSC:{code[:2]}",
                    "child_id": f"MSC:{code}",
                    "relation": "is_a",
                })

    _write_parquet_tables(name, terms_rows, hier_rows, [])
    return len(terms_rows)


def export_parquet_cso(name: str, info: dict) -> int:
    """Export CSO triples CSV to Parquet. Returns term count."""
    raw_file = _find_raw_file(name, info)
    if raw_file is None:
        return 0

    # CSO predicates
    SUPER_TOPIC_OF = "http://cso.kmi.open.ac.uk/schema/cso#superTopicOf"
    CONTRIBUTES_TO = "http://cso.kmi.open.ac.uk/schema/cso#contributesTo"
    RELATED_EQUIV = "http://cso.kmi.open.ac.uk/schema/cso#relatedEquivalent"
    PREF_EQUIV = "http://cso.kmi.open.ac.uk/schema/cso#preferentialEquivalent"
    LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
    SAME_AS = "http://www.w3.org/2002/07/owl#sameAs"
    RELATED_LINK = "http://schema.org/relatedLink"
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    # Collect data from triples
    labels = {}          # uri -> label
    types = {}           # uri -> type
    children = []        # (parent, child)
    contributes = []     # (source, target)
    synonyms = {}        # uri -> [synonym uris]
    xrefs_list = []      # (uri, link_type, external_uri)

    def _strip(s: str) -> str:
        """Strip angle brackets, quotes, and N-Triples language tags."""
        import re
        s = s.strip().strip('"').strip("<>")
        # Handle N-Triples language tag: value@en . or value@en
        s = re.sub(r'@\w+\s*\.?\s*$', '', s)
        return s

    def _topic_name(uri: str) -> str:
        """Extract human-readable topic name from URI."""
        from urllib.parse import unquote
        if "/topics/" in uri:
            return unquote(uri.split("/topics/")[-1]).replace("_", " ")
        return uri

    with open(raw_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            s = _strip(row[0])
            p = _strip(row[1])
            o = _strip(row[2])

            if p == LABEL:
                labels[s] = o
            elif p == RDF_TYPE:
                types[s] = o
            elif p == SUPER_TOPIC_OF:
                children.append((s, o))
            elif p == CONTRIBUTES_TO:
                contributes.append((s, o))
            elif p == RELATED_EQUIV or p == PREF_EQUIV:
                synonyms.setdefault(s, []).append(o)
            elif p == SAME_AS:
                xrefs_list.append((s, "sameAs", o))
            elif p == RELATED_LINK:
                # Detect link type from URI
                if "wikidata.org" in o:
                    xrefs_list.append((s, "wikidata", o))
                elif "dbpedia.org" in o:
                    xrefs_list.append((s, "dbpedia", o))
                elif "wikipedia.org" in o:
                    xrefs_list.append((s, "wikipedia", o))
                elif "freebase.com" in o:
                    xrefs_list.append((s, "freebase", o))
                else:
                    xrefs_list.append((s, "relatedLink", o))

    # Build term catalog: all unique topic URIs
    all_topics = set()
    for parent, child in children:
        all_topics.add(parent)
        all_topics.add(child)
    for src, tgt in contributes:
        all_topics.add(src)
        all_topics.add(tgt)
    for uri in labels:
        all_topics.add(uri)
    for uri in synonyms:
        all_topics.add(uri)
        for syn_uri in synonyms[uri]:
            all_topics.add(syn_uri)

    terms_rows = []
    for uri in sorted(all_topics):
        label = labels.get(uri, _topic_name(uri))
        syn_labels = []
        for syn_uri in synonyms.get(uri, []):
            syn_labels.append(labels.get(syn_uri, _topic_name(syn_uri)))

        terms_rows.append({
            "id": uri,
            "label": label,
            "definition": "",
            "synonyms": syn_labels,
            "namespace": "",
            "obsolete": False,
        })

    hier_rows = []
    for parent, child in children:
        hier_rows.append({
            "parent_id": parent,
            "child_id": child,
            "relation": "superTopicOf",
        })
    for src, tgt in contributes:
        hier_rows.append({
            "parent_id": tgt,
            "child_id": src,
            "relation": "contributesTo",
        })

    xrefs_rows = []
    for uri, link_type, external_uri in xrefs_list:
        xrefs_rows.append({
            "term_id": uri,
            "xref_db": link_type,
            "xref_id": external_uri,
        })

    _write_parquet_tables(name, terms_rows, hier_rows, xrefs_rows)
    return len(terms_rows)


def _write_parquet_tables(name: str, terms: list, hierarchy: list, xrefs: list):
    """Write the standard 3 Parquet tables for an ontology."""
    out_dir = DATASETS_DIR / name / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Terms table
    if terms:
        schema = pa.schema([
            ("id", pa.string()),
            ("label", pa.string()),
            ("definition", pa.string()),
            ("synonyms", pa.list_(pa.string())),
            ("namespace", pa.string()),
            ("obsolete", pa.bool_()),
        ])
        table = pa.table(
            {col: [row[col] for row in terms] for col in schema.names},
            schema=schema,
        )
        pq.write_table(table, out_dir / f"{name}_terms.parquet", **PARQUET_OPTS)
        print(f"    {name}_terms.parquet: {len(terms):,} rows")

    # Hierarchy table
    if hierarchy:
        schema = pa.schema([
            ("parent_id", pa.string()),
            ("child_id", pa.string()),
            ("relation", pa.string()),
        ])
        table = pa.table(
            {col: [row[col] for row in hierarchy] for col in schema.names},
            schema=schema,
        )
        pq.write_table(table, out_dir / f"{name}_hierarchy.parquet", **PARQUET_OPTS)
        print(f"    {name}_hierarchy.parquet: {len(hierarchy):,} rows")

    # Xrefs table
    if xrefs:
        schema = pa.schema([
            ("term_id", pa.string()),
            ("xref_db", pa.string()),
            ("xref_id", pa.string()),
        ])
        table = pa.table(
            {col: [row[col] for row in xrefs] for col in schema.names},
            schema=schema,
        )
        pq.write_table(table, out_dir / f"{name}_xrefs.parquet", **PARQUET_OPTS)
        print(f"    {name}_xrefs.parquet: {len(xrefs):,} rows")

    # Write meta.json
    meta = {
        "full_name": ONTOLOGIES[name]["full_name"],
        "description": f"{ONTOLOGIES[name]['full_name']} - {ONTOLOGIES[name]['domain']} ontology",
        "license": ONTOLOGIES[name]["license"],
        "source_url": ONTOLOGIES[name]["source_url"],
        "format": "parquet",
        "tables": {},
    }
    if terms:
        meta["tables"][f"{name}_terms"] = {
            "row_count": len(terms),
            "description": f"{ONTOLOGIES[name]['full_name']} term catalog",
            "performance_tier": "SMALL",
        }
    if hierarchy:
        meta["tables"][f"{name}_hierarchy"] = {
            "row_count": len(hierarchy),
            "description": "Parent-child edges",
            "performance_tier": "SMALL",
        }
    if xrefs:
        meta["tables"][f"{name}_xrefs"] = {
            "row_count": len(xrefs),
            "description": "Cross-references to other ontologies/databases",
            "performance_tier": "SMALL",
        }

    with open(DATASETS_DIR / name / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ------------------------------------------------------------------
# Main conversion orchestration
# ------------------------------------------------------------------

PARSER_TO_EXPORT = {
    "obo_pronto": export_parquet_obo,
    "skos_rdflib": export_parquet_skos,
    "mesh_nt": export_parquet_mesh,
    "msc_csv": export_parquet_msc,
    "cso_csv": export_parquet_cso,
}


def convert_ontology(
    name: str,
    do_oxigraph: bool = True,
    do_parquet: bool = True,
) -> bool:
    """Convert a single ontology. Returns True on success."""
    if name not in ONTOLOGIES:
        print(f"  Unknown ontology: {name}")
        return False

    info = ONTOLOGIES[name]
    print(f"\n  [{name}] {info['full_name']}")

    raw_file = _find_raw_file(name, info)
    if raw_file is None:
        print(f"    No downloaded file found. Run download_ontologies.py first.")
        return False
    print(f"    Source: {raw_file.name} ({raw_file.stat().st_size / (1024*1024):.1f} MB)")

    success = True

    if do_oxigraph:
        t0 = time.time()
        store = _get_store()
        if load_to_oxigraph(name, info, store):
            # Count triples in this graph
            count = 0
            for _ in store.quads_for_pattern(None, None, None, _graph_iri(name)):
                count += 1
            elapsed = time.time() - t0
            print(f"    Oxigraph: {count:,} triples loaded ({elapsed:.1f}s)")
            store.flush()
        else:
            print(f"    Oxigraph: FAILED")
            success = False

    if do_parquet:
        t0 = time.time()
        export_fn = PARSER_TO_EXPORT.get(info["parser"])
        if export_fn:
            term_count = export_fn(name, info)
            elapsed = time.time() - t0
            if term_count:
                print(f"    Parquet: {term_count:,} terms exported ({elapsed:.1f}s)")
            else:
                print(f"    Parquet: no terms exported")
                success = False
        else:
            print(f"    No Parquet exporter for parser: {info['parser']}")
            success = False

    return success


def cmd_status():
    """Show conversion status."""
    print(f"\n{'Name':12s}  {'Oxigraph':>10s}  {'Terms':>10s}  {'Hierarchy':>10s}  {'Xrefs':>10s}")
    print("-" * 65)
    for name in ALL_ONTOLOGY_NAMES:
        # Check Oxigraph
        ox_count = "-"
        if OXIGRAPH_DIR.exists():
            try:
                store = _get_store()
                count = 0
                for _ in store.quads_for_pattern(None, None, None, _graph_iri(name)):
                    count += 1
                if count > 0:
                    ox_count = f"{count:,}"
            except Exception:
                pass

        # Check Parquet
        pq_dir = DATASETS_DIR / name / "parquet"
        terms_count = hierarchy_count = xrefs_count = "-"
        if pq_dir.exists():
            for pf in pq_dir.glob("*.parquet"):
                try:
                    meta = pq.read_metadata(pf)
                    rows = meta.num_rows
                    if "terms" in pf.stem:
                        terms_count = f"{rows:,}"
                    elif "hierarchy" in pf.stem:
                        hierarchy_count = f"{rows:,}"
                    elif "xrefs" in pf.stem:
                        xrefs_count = f"{rows:,}"
                except Exception:
                    pass

        print(f"{name:12s}  {ox_count:>10s}  {terms_count:>10s}  {hierarchy_count:>10s}  {xrefs_count:>10s}")


def main():
    parser = argparse.ArgumentParser(description="Convert ontologies to Oxigraph + Parquet")
    parser.add_argument("--all", action="store_true", help="Convert all downloaded ontologies")
    parser.add_argument("--ontology", type=str, help="Comma-separated list of ontologies")
    parser.add_argument("--oxigraph-only", action="store_true", help="Only load into Oxigraph")
    parser.add_argument("--parquet-only", action="store_true", help="Only export to Parquet")
    parser.add_argument("--status", action="store_true", help="Show conversion status")
    args = parser.parse_args()

    if args.status:
        cmd_status()
        return 0

    if args.all:
        names = ALL_ONTOLOGY_NAMES
    elif args.ontology:
        names = [n.strip() for n in args.ontology.split(",")]
    else:
        parser.print_help()
        return 1

    do_ox = not args.parquet_only
    do_pq = not args.oxigraph_only

    print(f"=== Converting {len(names)} ontologies ===")
    if do_ox:
        print(f"  Oxigraph store: {OXIGRAPH_DIR}")
    if do_pq:
        print(f"  Parquet output: {DATASETS_DIR}/{{name}}/parquet/")

    success = 0
    failed = []
    for name in names:
        try:
            if convert_ontology(name, do_oxigraph=do_ox, do_parquet=do_pq):
                success += 1
            else:
                failed.append(name)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed.append(name)

    print(f"\n=== Conversion complete: {success}/{len(names)} succeeded ===")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
