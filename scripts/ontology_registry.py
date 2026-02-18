#!/usr/bin/env python3
"""
Central registry of all scientific ontologies in the datalake.

Each entry defines download source, format, parser type, and metadata.
Used by download_ontologies.py and convert_ontologies.py.
"""

from datetime import datetime

CURRENT_YEAR = datetime.now().year

ONTOLOGIES = {
    "mesh": {
        "full_name": "Medical Subject Headings",
        "domain": "Biomedical",
        "license": "Public Domain",
        "source_url": "https://www.nlm.nih.gov/mesh/",
        "download": {
            "url": f"https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/{CURRENT_YEAR}/mesh{CURRENT_YEAR}.nt.gz",
            "filename": f"mesh{CURRENT_YEAR}.nt.gz",
            "method": "http",
            "extract": "gz",
        },
        "format": "nt_gz",
        "parser": "mesh_nt",
        "estimated_terms": 30000,
        "id_prefix": "http://id.nlm.nih.gov/mesh/",
    },
    "go": {
        "full_name": "Gene Ontology",
        "domain": "Biology",
        "license": "CC BY 4.0",
        "source_url": "http://geneontology.org/",
        "download": {
            "url": "https://release.geneontology.org/2024-06-17/ontology/go-basic.obo",
            "filename": "go-basic.obo",
            "method": "http",
        },
        "format": "obo",
        "parser": "obo_pronto",
        "estimated_terms": 45000,
        "id_prefix": "GO:",
    },
    "doid": {
        "full_name": "Disease Ontology",
        "domain": "Disease",
        "license": "CC0",
        "source_url": "https://disease-ontology.org/",
        "download": {
            "url": "http://purl.obolibrary.org/obo/doid.obo",
            "filename": "doid.obo",
            "method": "http",
        },
        "format": "obo",
        "parser": "obo_pronto",
        "estimated_terms": 12000,
        "id_prefix": "DOID:",
    },
    "chebi": {
        "full_name": "Chemical Entities of Biological Interest",
        "domain": "Chemistry",
        "license": "CC BY 4.0",
        "source_url": "https://www.ebi.ac.uk/chebi/",
        "download": {
            "url": "http://purl.obolibrary.org/obo/chebi.obo",
            "filename": "chebi.obo",
            "method": "http",
        },
        "format": "obo",
        "parser": "obo_pronto",
        "estimated_terms": 195000,
        "id_prefix": "CHEBI:",
    },
    "hpo": {
        "full_name": "Human Phenotype Ontology",
        "domain": "Phenotypes",
        "license": "Custom (free for research)",
        "source_url": "https://hpo.jax.org/",
        "download": {
            "url": "http://purl.obolibrary.org/obo/hp.obo",
            "filename": "hp.obo",
            "method": "http",
        },
        "format": "obo",
        "parser": "obo_pronto",
        "estimated_terms": 18000,
        "id_prefix": "HP:",
    },
    "ncit": {
        "full_name": "NCI Thesaurus",
        "domain": "Cancer/Biomedical",
        "license": "CC BY 4.0",
        "source_url": "https://ncithesaurus.nci.nih.gov/",
        "download": {
            "url": "http://purl.obolibrary.org/obo/ncit.obo",
            "filename": "ncit.obo",
            "method": "http",
        },
        "format": "obo",
        "parser": "obo_pronto",
        "estimated_terms": 160000,
        "id_prefix": "NCIT:",
    },
    "edam": {
        "full_name": "EDAM Ontology",
        "domain": "Bioinformatics",
        "license": "CC BY 4.0",
        "source_url": "https://edamontology.org/",
        "download": {
            "url": "http://edamontology.org/EDAM.owl",
            "filename": "EDAM.owl",
            "method": "http",
        },
        "format": "owl",
        "parser": "obo_pronto",
        "estimated_terms": 3000,
        "id_prefix": "http://edamontology.org/",
    },
    "physh": {
        "full_name": "Physics Subject Headings",
        "domain": "Physics",
        "license": "CC BY 4.0",
        "source_url": "https://physh.org/",
        "download": {
            "url": "https://raw.githubusercontent.com/physh-org/PhySH/master/physh.ttl",
            "filename": "physh.ttl",
            "method": "http",
            "extra_files": [
                {
                    "url": "https://raw.githubusercontent.com/physh-org/PhySH/master/physh_skos_compat.ttl",
                    "filename": "physh_skos_compat.ttl",
                }
            ],
        },
        "format": "ttl",
        "parser": "skos_rdflib",
        "estimated_terms": 3500,
        "id_prefix": "https://physh.org/concepts/",
    },
    "msc2020": {
        "full_name": "Mathematics Subject Classification 2020",
        "domain": "Mathematics",
        "license": "CC BY-NC-SA 4.0",
        "source_url": "https://msc2020.org/",
        "download": {
            "url": "https://msc2020.org/MSC_2020.csv",
            "filename": "MSC_2020.csv",
            "method": "http",
        },
        "format": "csv",
        "parser": "msc_csv",
        "estimated_terms": 6500,
        "id_prefix": "MSC:",
    },
    "agrovoc": {
        "full_name": "AGROVOC Multilingual Thesaurus",
        "domain": "Agriculture",
        "license": "CC BY 3.0 IGO",
        "source_url": "https://agrovoc.fao.org/",
        "download": {
            "url": "https://agrovoc.fao.org/latestAgrovoc/agrovoc_core.nt.zip",
            "filename": "agrovoc_core.nt.zip",
            "method": "http",
            "extract": "zip",
        },
        "format": "nt",
        "parser": "skos_rdflib",
        "estimated_terms": 42000,
        "id_prefix": "http://aims.fao.org/aos/agrovoc/",
    },
    "unesco": {
        "full_name": "UNESCO Thesaurus",
        "domain": "General Science / Education",
        "license": "CC BY-SA 3.0 IGO",
        "source_url": "https://vocabularies.unesco.org/",
        "download": {
            "url": "https://vocabularies.unesco.org/exports/thesaurus/latest/unesco-thesaurus.rdf",
            "filename": "unesco-thesaurus.rdf",
            "method": "http",
        },
        "format": "rdf",
        "parser": "skos_rdflib",
        "estimated_terms": 4400,
        "id_prefix": "http://vocabularies.unesco.org/thesaurus/",
    },
    "stw": {
        "full_name": "STW Thesaurus for Economics",
        "domain": "Economics",
        "license": "CC BY 4.0",
        "source_url": "https://zbw.eu/stw/",
        "download": {
            "url": "https://zbw.eu/stw/version/latest/download/stw.rdf.zip",
            "filename": "stw.rdf.zip",
            "method": "http",
            "extract": "zip",
            "fallback_url": "https://zbw.eu/stw/version/latest/download/stw.nt.gz",
            "fallback_filename": "stw.nt.gz",
        },
        "format": "rdf",
        "parser": "skos_rdflib",
        "estimated_terms": 6000,
        "id_prefix": "http://zbw.eu/stw/descriptor/",
    },
    "cso": {
        "full_name": "Computer Science Ontology",
        "domain": "Computer Science",
        "license": "CC BY 4.0",
        "source_url": "https://cso.kmi.open.ac.uk/",
        "download": {
            # Portal requires login; file must be downloaded manually once.
            # Place CSO.3.5.csv in ontologies/cso/
            "url": "https://cso.kmi.open.ac.uk/download/CSO.csv",
            "filename": "CSO.3.5.csv",
            "method": "manual",
        },
        "format": "cso_csv",
        "parser": "cso_csv",
        "estimated_terms": 14000,
        "id_prefix": "https://cso.kmi.open.ac.uk/topics/",
    },
}

# Short names for CLI
ALL_ONTOLOGY_NAMES = list(ONTOLOGIES.keys())
