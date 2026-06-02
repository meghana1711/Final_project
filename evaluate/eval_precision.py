import argparse
import csv
import os
import random
from typing import Dict, List, Set, Tuple, Optional

from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD, SKOS


# ============================================================
# Helpers
# ============================================================

META_PREDICATES = {
    RDF.type,
    RDFS.subClassOf,
    RDFS.subPropertyOf,
    RDFS.label,
    RDFS.comment,
    RDFS.domain,
    RDFS.range,
    OWL.equivalentClass,
    OWL.equivalentProperty,
    OWL.disjointWith,
    OWL.inverseOf,
    OWL.sameAs,
    OWL.differentFrom,
    OWL.imports,
    OWL.versionInfo,
    SKOS.prefLabel,
    SKOS.altLabel,
    SKOS.definition,
    SKOS.broader,
    SKOS.narrower,
    SKOS.related,
}

CLASS_TYPE_OBJECTS = {
    OWL.Class,
    RDFS.Class,
}

PROPERTY_TYPE_OBJECTS = {
    RDF.Property,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    OWL.FunctionalProperty,
    OWL.TransitiveProperty,
    OWL.SymmetricProperty,
    OWL.AsymmetricProperty,
    OWL.InverseFunctionalProperty,
}


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_rows(rows: List[Dict], sample_size: int, seed: int) -> List[Dict]:
    if sample_size <= 0:
        return []
    if sample_size >= len(rows):
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, sample_size)


def normalize_text(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def is_uri_in_namespace(uri: URIRef, prefixes: Tuple[str, ...]) -> bool:
    s = str(uri)
    return any(s.startswith(p) for p in prefixes)


def is_builtin_resource(node) -> bool:
    if not isinstance(node, URIRef):
        return False

    return is_uri_in_namespace(
        node,
        (
            str(RDF),
            str(RDFS),
            str(OWL),
            str(XSD),
            str(SKOS),
        ),
    )


def short_form(graph: Graph, node) -> str:
    """
    Convert URI/BNode/Literal into a readable string.
    """
    if isinstance(node, Literal):
        return str(node)

    if isinstance(node, BNode):
        return f"_:{node}"

    if isinstance(node, URIRef):
        s = str(node)

        # URI with fragment
        if "#" in s:
            return s.rsplit("#", 1)[-1]

        # URI with slash
        if "/" in s:
            return s.rstrip("/").rsplit("/", 1)[-1]

        if ":" in s:
            return s.rsplit(":", 1)[-1]

        return s
    return str(node)


def is_named_resource(node) -> bool:
    return isinstance(node, URIRef)


def is_class_like(graph: Graph, node) -> bool:
    if not isinstance(node, URIRef):
        return False

    for o in graph.objects(node, RDF.type):
        if o in CLASS_TYPE_OBJECTS:
            return True

    # Also allow nodes that participate in subclass hierarchy
    if (node, RDFS.subClassOf, None) in graph or (None, RDFS.subClassOf, node) in graph:
        return True

    return False


def is_property_like(graph: Graph, node) -> bool:
    if not isinstance(node, URIRef):
        return False

    for o in graph.objects(node, RDF.type):
        if o in PROPERTY_TYPE_OBJECTS:
            return True

    return False


def is_individual_assertion_predicate(graph: Graph, pred: URIRef) -> bool:
    """
    True for predicates that can be considered non-taxonomic relation predicates.
    Excludes schema/meta predicates.
    """
    if pred in META_PREDICATES:
        return False

    if is_builtin_resource(pred):
        return False

    # Prefer explicitly declared object properties,
    # but allow custom predicates not declared in weak ontologies.
    return True


def is_schema_type_triple(s, p, o) -> bool:
    if p != RDF.type:
        return False

    if o in CLASS_TYPE_OBJECTS:
        return True
    if o in PROPERTY_TYPE_OBJECTS:
        return True
    if o in {OWL.Ontology, OWL.Restriction, RDFS.Datatype}:
        return True

    return False


# ============================================================
# Extraction
# ============================================================

def extract_concepts(graph: Graph) -> List[Dict]:
    """
    Concepts = classes / class-like resources.
    """
    concepts: List[Dict] = []
    seen: Set[str] = set()

    candidates: Set[URIRef] = set()

    # Explicit class declarations
    for s in graph.subjects(RDF.type, OWL.Class):
        if isinstance(s, URIRef):
            candidates.add(s)

    for s in graph.subjects(RDF.type, RDFS.Class):
        if isinstance(s, URIRef):
            candidates.add(s)

    # Classes participating in taxonomy
    for s, _, o in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef):
            candidates.add(s)
        if isinstance(o, URIRef):
            candidates.add(o)

    for node in candidates:
        if is_builtin_resource(node):
            continue

        label = short_form(graph, node).strip()
        if not label:
            continue

        key = str(node).lower()
        if key in seen:
            continue
        seen.add(key)

        concepts.append({
            "term": label,
            "manual_label": "",
            "manual_notes": "",
        })

    concepts.sort(key=lambda x: x["term"].lower())
    return concepts


def extract_taxonomy(graph: Graph) -> List[Dict]:
    """
    Taxonomy = rdfs:subClassOf(child, parent), excluding blank-node parents/children.
    """
    rows: List[Dict] = []
    seen: Set[Tuple[str, str]] = set()

    for child, _, parent in graph.triples((None, RDFS.subClassOf, None)):
        if not isinstance(child, URIRef):
            continue
        if not isinstance(parent, URIRef):
            # skip anonymous restrictions / blank-node superclass expressions
            continue

        if is_builtin_resource(child) or is_builtin_resource(parent):
            continue

        child_txt = short_form(graph, child).strip()
        parent_txt = short_form(graph, parent).strip()

        if not child_txt or not parent_txt:
            continue

        key = (str(child).lower(), str(parent).lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "child": child_txt,
            "parent": parent_txt,
            "manual_label": "",
            "manual_notes": "",
        })

    rows.sort(key=lambda x: (x["child"].lower(), x["parent"].lower()))
    return rows


def extract_non_taxonomy(graph: Graph) -> List[Dict]:
    """
    Non-taxonomic relations = assertion triples such as:
        subject --predicate--> object
    excluding:
    - rdf:type schema declarations
    - rdfs:subClassOf
    - schema/meta predicates
    - built-in vocabulary resources
    - literal objects (to keep this focused on entity-entity relations)
    """
    rows: List[Dict] = []
    seen: Set[Tuple[str, str, str]] = set()

    for s, p, o in graph:
        # Exclude taxonomy
        if p == RDFS.subClassOf:
            continue

        # Exclude schema-type declarations
        if is_schema_type_triple(s, p, o):
            continue

        # Exclude generic rdf:type assertions; these are not non-tax relations
        if p == RDF.type:
            continue

        # Only keep resource-resource assertions
        if not isinstance(s, URIRef):
            continue
        if not isinstance(p, URIRef):
            continue
        if not isinstance(o, URIRef):
            continue

        if is_builtin_resource(s) or is_builtin_resource(p) or is_builtin_resource(o):
            continue

        if not is_individual_assertion_predicate(graph, p):
            continue

        subj_txt = short_form(graph, s).strip()
        rel_txt = short_form(graph, p).strip()
        obj_txt = short_form(graph, o).strip()

        if not subj_txt or not rel_txt or not obj_txt:
            continue

        key = (str(s).lower(), str(p).lower(), str(o).lower())
        if key in seen:
            continue
        seen.add(key)

        rows.append({
            "subject": subj_txt,
            "relation": rel_txt,
            "object": obj_txt,
            "manual_label": "",
            "manual_notes": "",
        })

    rows.sort(key=lambda x: (
        x["relation"].lower(),
        x["subject"].lower(),
        x["object"].lower(),
    ))
    return rows


# ============================================================
# Export wrappers
# ============================================================

def export_concepts(graph: Graph, out_path: str, sample_size: int, seed: int) -> None:
    rows = extract_concepts(graph)
    sampled = sample_rows(rows, sample_size, seed)
    write_csv(out_path, sampled, ["term", "manual_label", "manual_notes"])
    print(f"[OK] Concepts exported: {len(sampled)} -> {out_path}")
    print(f"[INFO] Total concept candidates found: {len(rows)}")


def export_taxonomy(graph: Graph, out_path: str, sample_size: int, seed: int) -> None:
    rows = extract_taxonomy(graph)
    sampled = sample_rows(rows, sample_size, seed)
    write_csv(out_path, sampled, ["child", "parent", "manual_label", "manual_notes"])
    print(f"[OK] Taxonomy exported: {len(sampled)} -> {out_path}")
    print(f"[INFO] Total taxonomy candidates found: {len(rows)}")


def export_non_taxonomy(graph: Graph, out_path: str, sample_size: int, seed: int) -> None:
    rows = extract_non_taxonomy(graph)
    sampled = sample_rows(rows, sample_size, seed)
    write_csv(out_path, sampled, ["subject", "relation", "object", "manual_label", "manual_notes"])
    print(f"[OK] Non-taxonomy exported: {len(sampled)} -> {out_path}")
    print(f"[INFO] Total non-taxonomy candidates found: {len(rows)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Randomly export concepts, taxonomy relations, and non-taxonomic relations from a TTL ontology."
    )
    parser.add_argument("--ttl", required=True, help="Path to TTL file")
    parser.add_argument("--out_dir", required=True, help="Output folder")
    parser.add_argument("--concept_sample", type=int, default=50)
    parser.add_argument("--taxonomy_sample", type=int, default=50)
    parser.add_argument("--non_tax_sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    safe_mkdir(args.out_dir)

    graph = Graph()
    graph.parse(args.ttl, format="turtle")

    export_concepts(
        graph=graph,
        out_path=os.path.join(args.out_dir, "concepts.csv"),
        sample_size=args.concept_sample,
        seed=args.seed,
    )

    export_taxonomy(
        graph=graph,
        out_path=os.path.join(args.out_dir, "taxonomy.csv"),
        sample_size=args.taxonomy_sample,
        seed=args.seed,
    )

    export_non_taxonomy(
        graph=graph,
        out_path=os.path.join(args.out_dir, "non_taxonomy.csv"),
        sample_size=args.non_tax_sample,
        seed=args.seed,
    )

    print(f"[OK] All exports written to: {args.out_dir}")


if __name__ == "__main__":
    main()