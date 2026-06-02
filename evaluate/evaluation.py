from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD
from rdflib.plugins.parsers.notation3 import BadSyntax


SCHEMA_DECLARATION_TYPES = {
    OWL.Class,
    RDFS.Class,
    OWL.ObjectProperty,
    OWL.DatatypeProperty,
    OWL.AnnotationProperty,
    RDF.Property,
    OWL.NamedIndividual,
    OWL.Ontology,
}

SCHEMA_PREDICATES = {
    RDF.type,
    RDFS.subClassOf,
    RDFS.subPropertyOf,
    RDFS.domain,
    RDFS.range,
    RDFS.label,
    RDFS.comment,
    OWL.equivalentClass,
    OWL.equivalentProperty,
    OWL.disjointWith,
    OWL.inverseOf,
    SKOS.prefLabel,
    SKOS.altLabel,
    SKOS.definition,
}

XSD_DATATYPES: Set[URIRef] = {
    XSD.string,
    XSD.boolean,
    XSD.integer,
    XSD.decimal,
    XSD.float,
    XSD.double,
    XSD.date,
    XSD.dateTime,
    XSD.time,
    XSD.duration,
    XSD.anyURI,
}


@dataclass
class PrecisionResult:
    available: bool
    total: int = 0
    valid: int = 0
    precision: Optional[float] = None
    invalid_rows: int = 0
    ambiguous_rows: int = 0
    note: str = ""


@dataclass
class ParseResult:
    ok: bool
    graph: Optional[Graph] = None
    error: Optional[str] = None


def is_named(node) -> bool:
    return isinstance(node, URIRef)


def is_literal(node) -> bool:
    return isinstance(node, Literal)


def node_label(g: Graph, node) -> str:
    try:
        if isinstance(node, URIRef):
            return g.namespace_manager.normalizeUri(node)
        if isinstance(node, BNode):
            return f"_:{str(node)}"
        if isinstance(node, Literal):
            return node.n3(g.namespace_manager)
    except Exception:
        pass
    return str(node)


def fmt_pct(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{100.0 * x:.2f}%"


def count_duplicates(items: Sequence[Tuple]) -> int:
    seen: Set[Tuple] = set()
    dup = 0
    for item in items:
        if item in seen:
            dup += 1
        else:
            seen.add(item)
    return dup


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_ttl(ttl_path: Path) -> ParseResult:
    g = Graph()
    try:
        g.parse(ttl_path.as_posix(), format="turtle")
        return ParseResult(ok=True, graph=g)
    except (BadSyntax, Exception) as e:
        return ParseResult(ok=False, error=str(e))


def get_declared_classes(g: Graph) -> Set[URIRef]:
    classes = set(g.subjects(RDF.type, OWL.Class)) | set(g.subjects(RDF.type, RDFS.Class))
    return {c for c in classes if is_named(c)}


def get_declared_object_properties(g: Graph) -> Set[URIRef]:
    return {p for p in g.subjects(RDF.type, OWL.ObjectProperty) if is_named(p)}


def get_declared_datatype_properties(g: Graph) -> Set[URIRef]:
    return {p for p in g.subjects(RDF.type, OWL.DatatypeProperty) if is_named(p)}


def get_declared_annotation_properties(g: Graph) -> Set[URIRef]:
    return {p for p in g.subjects(RDF.type, OWL.AnnotationProperty) if is_named(p)}


def get_named_taxonomy_edges(g: Graph) -> List[Tuple[URIRef, URIRef]]:
    edges: List[Tuple[URIRef, URIRef]] = []
    for c, _, p in g.triples((None, RDFS.subClassOf, None)):
        if is_named(c) and is_named(p):
            edges.append((c, p))
    return edges


def get_non_taxonomy_triples(g: Graph) -> List[Tuple]:
    out: List[Tuple] = []
    for s, p, o in g:
        if p in SCHEMA_PREDICATES:
            continue
        if p == RDF.type and o in SCHEMA_DECLARATION_TYPES:
            continue
        out.append((s, p, o))
    return out


def build_adj(edges: Sequence[Tuple[URIRef, URIRef]]) -> Dict[URIRef, List[URIRef]]:
    adj: Dict[URIRef, List[URIRef]] = defaultdict(list)
    for child, parent in edges:
        adj[child].append(parent)
    return adj


def find_cycles(edges: Sequence[Tuple[URIRef, URIRef]]) -> List[List[URIRef]]:
    adj = build_adj(edges)
    nodes: Set[URIRef] = set()
    for child, parent in edges:
        nodes.add(child)
        nodes.add(parent)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[URIRef, int] = {n: WHITE for n in nodes}
    parent_map: Dict[URIRef, Optional[URIRef]] = {n: None for n in nodes}
    cycles: List[List[URIRef]] = []
    seen_norm: Set[Tuple[str, ...]] = set()

    def normalize_cycle(cycle: List[URIRef]) -> Tuple[str, ...]:
        labels = [str(n) for n in cycle[:-1]]
        if not labels:
            return tuple()
        mins = min(range(len(labels)), key=lambda i: labels[i])
        rotated = labels[mins:] + labels[:mins]
        return tuple(rotated)

    sys.setrecursionlimit(max(10000, len(nodes) + 100))

    def dfs(u: URIRef) -> None:
        color[u] = GRAY
        for v in adj.get(u, []):
            if color[v] == WHITE:
                parent_map[v] = u
                dfs(v)
            elif color[v] == GRAY:
                cycle = [v]
                x = u
                while x is not None and x != v:
                    cycle.append(x)
                    x = parent_map.get(x)
                cycle.append(v)
                cycle.reverse()
                if len(cycle) >= 3:
                    key = normalize_cycle(cycle)
                    if key and key not in seen_norm:
                        seen_norm.add(key)
                        cycles.append(cycle)
        color[u] = BLACK

    for n in nodes:
        if color[n] == WHITE:
            dfs(n)
    return cycles


def taxonomy_depth(edges: Sequence[Tuple[URIRef, URIRef]]) -> int:
    if not edges:
        return 0

    children = {c for c, _ in edges}
    parents = {p for _, p in edges}
    roots = parents - children

    children_of: Dict[URIRef, List[URIRef]] = defaultdict(list)
    nodes: Set[URIRef] = set()
    for child, parent in edges:
        children_of[parent].append(child)
        nodes.add(child)
        nodes.add(parent)

    if not roots:
        roots = nodes

    max_depth = 0
    stack: List[Tuple[URIRef, int, Set[URIRef]]] = [(r, 1, {r}) for r in roots]
    while stack:
        node, depth, path = stack.pop()
        max_depth = max(max_depth, depth)
        for child in children_of.get(node, []):
            if child in path:
                continue
            stack.append((child, depth + 1, path | {child}))
    return max_depth


def taxonomy_roots_and_leaves(edges: Sequence[Tuple[URIRef, URIRef]]) -> Tuple[Set[URIRef], Set[URIRef]]:
    children = {c for c, _ in edges}
    parents = {p for _, p in edges}
    roots = parents - children
    leaves = children - parents
    return roots, leaves


def connected_components_undirected(edges: Sequence[Tuple[URIRef, URIRef]]) -> List[Set[URIRef]]:
    nbrs: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    nodes: Set[URIRef] = set()
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
        nodes.add(a)
        nodes.add(b)

    comps: List[Set[URIRef]] = []
    seen: Set[URIRef] = set()

    for n in nodes:
        if n in seen:
            continue
        comp: Set[URIRef] = set()
        q = deque([n])
        seen.add(n)
        while q:
            u = q.popleft()
            comp.add(u)
            for v in nbrs.get(u, set()):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)

    comps.sort(key=lambda c: (-len(c), sorted(map(str, c))[0] if c else ""))
    return comps


def transitive_superclasses(edges: Sequence[Tuple[URIRef, URIRef]]) -> Dict[URIRef, Set[URIRef]]:
    parents_of: Dict[URIRef, List[URIRef]] = defaultdict(list)
    nodes: Set[URIRef] = set()
    for c, p in edges:
        parents_of[c].append(p)
        nodes.add(c)
        nodes.add(p)

    memo: Dict[URIRef, Set[URIRef]] = {}

    def dfs(n: URIRef, visiting: Set[URIRef]) -> Set[URIRef]:
        if n in memo:
            return memo[n]
        if n in visiting:
            return set()
        visiting.add(n)
        supers: Set[URIRef] = set()
        for p in parents_of.get(n, []):
            supers.add(p)
            supers |= dfs(p, visiting)
        visiting.remove(n)
        memo[n] = supers
        return supers

    for n in nodes:
        dfs(n, set())
    return memo


def schema_checks(
    g: Graph,
    classes: Set[URIRef],
    obj_props: Set[URIRef],
    data_props: Set[URIRef],
) -> Dict[str, object]:
    anno_props = get_declared_annotation_properties(g)
    rdf_props = {p for p in g.subjects(RDF.type, RDF.Property) if is_named(p)}
    all_props = obj_props | data_props | anno_props | rdf_props

    class_property_overlap = sorted(classes & all_props, key=str)
    obj_data_overlap = sorted(obj_props & data_props, key=str)

    obj_prop_with_literal_range: List[Tuple[URIRef, URIRef]] = []
    data_prop_with_nonliteral_range: List[Tuple[URIRef, URIRef]] = []

    for p, _, rng in g.triples((None, RDFS.range, None)):
        if not is_named(p):
            continue
        if p in obj_props and rng in XSD_DATATYPES:
            obj_prop_with_literal_range.append((p, rng))
        if p in data_props and is_named(rng) and rng not in XSD_DATATYPES:
            data_prop_with_nonliteral_range.append((p, rng))

    undeclared_predicates = sorted(
        {
            p
            for _, p, _ in g
            if is_named(p)
            and p not in SCHEMA_PREDICATES
            and p not in obj_props
            and p not in data_props
            and p not in anno_props
            and p != RDF.type
        },
        key=str,
    )

    return {
        "class_property_overlap": class_property_overlap,
        "object_datatype_overlap": obj_data_overlap,
        "object_property_with_literal_range": obj_prop_with_literal_range,
        "datatype_property_with_nonliteral_range": data_prop_with_nonliteral_range,
        "undeclared_predicates": undeclared_predicates,
    }


def domain_range_checks(
    g: Graph,
    class_supers: Dict[URIRef, Set[URIRef]],
    obj_props: Set[URIRef],
    data_props: Set[URIRef],
) -> Dict[str, object]:
    domains: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    ranges: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    for p, _, d in g.triples((None, RDFS.domain, None)):
        if is_named(p) and is_named(d):
            domains[p].add(d)
    for p, _, r in g.triples((None, RDFS.range, None)):
        if is_named(p) and is_named(r):
            ranges[p].add(r)

    types_of: Dict[URIRef, Set[URIRef]] = defaultdict(set)
    for ind, _, cls in g.triples((None, RDF.type, None)):
        if is_named(ind) and is_named(cls) and cls not in SCHEMA_DECLARATION_TYPES:
            types_of[ind].add(cls)
            types_of[ind] |= class_supers.get(cls, set())

    subj_domain_violations: List[Tuple] = []
    obj_range_violations: List[Tuple] = []
    datatype_object_violations: List[Tuple] = []
    object_property_literal_violations: List[Tuple] = []

    for s, p, o in g:
        if not is_named(p):
            continue
        if p in SCHEMA_PREDICATES:
            continue

        required_domains = domains.get(p, set())
        if required_domains and is_named(s):
            subj_types = types_of.get(s, set())
            if subj_types and subj_types.isdisjoint(required_domains):
                subj_domain_violations.append(
                    (s, p, tuple(sorted(required_domains, key=str)), tuple(sorted(subj_types, key=str)))
                )

        if p in data_props:
            if not is_literal(o):
                datatype_object_violations.append((s, p, o))
        elif p in obj_props:
            if is_literal(o):
                object_property_literal_violations.append((s, p, o))

        required_ranges = ranges.get(p, set())
        if required_ranges:
            if is_literal(o):
                if p in obj_props:
                    obj_range_violations.append((s, p, o, tuple(sorted(required_ranges, key=str))))
            elif is_named(o):
                obj_types = types_of.get(o, set())
                non_xsd_ranges = {r for r in required_ranges if r not in XSD_DATATYPES}
                if non_xsd_ranges and obj_types and obj_types.isdisjoint(non_xsd_ranges):
                    obj_range_violations.append(
                        (s, p, o, tuple(sorted(non_xsd_ranges, key=str)), tuple(sorted(obj_types, key=str)))
                    )

    return {
        "subject_domain_violations": subj_domain_violations,
        "object_range_violations": obj_range_violations,
        "datatype_property_nonliteral_objects": datatype_object_violations,
        "object_property_literal_objects": object_property_literal_violations,
    }


def read_precision_csv(path: Optional[Path]) -> PrecisionResult:
    """
    Expected CSV column:
    - manual_label

    Expected values in manual_label:
    - Valid
    - Invalid
    - Ambiguous

    Precision is computed as:
        Valid / (Valid + Invalid)

    Ambiguous rows are excluded from the denominator.
    """
    if path is None:
        return PrecisionResult(available=False, note="No annotation CSV provided.")
    if not path.exists():
        return PrecisionResult(available=False, note=f"Annotation CSV not found: {path}")

    total = 0
    valid = 0
    invalid_rows = 0
    ambiguous_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}

        if "manual_label" not in cols:
            return PrecisionResult(
                available=False,
                note=f"CSV missing required 'manual_label' column: {path}"
            )

        label_col = cols["manual_label"]

        for row in reader:
            val = (row.get(label_col) or "").strip().lower()
            if not val:
                continue

            if val == "valid":
                valid += 1
                total += 1
            elif val == "invalid":
                invalid_rows += 1
                total += 1
            elif val == "ambiguous":
                ambiguous_rows += 1
                total +=1
                
    return PrecisionResult(
        available=True,
        total=total,
        valid=valid,
        precision=(valid / total) if total else None,
        invalid_rows=invalid_rows,
        ambiguous_rows=ambiguous_rows,
    )


def summarize_precision(name: str, pr: PrecisionResult) -> str:
    if not pr.available:
        return f"- {name}: N/A ({pr.note})"
    return (
        f"- {name}: {fmt_pct(pr.precision)} "
        f"(valid rows:{pr.valid}; "
        f"ambiguous rows: {pr.ambiguous_rows}; "
        f"inalid rows: {pr.invalid_rows})"
    )


def report_for_parse_failure(ttl_path: Path, parse_error: str) -> str:
    return f"""# Ontology Evaluation Report

## Input
- TTL file: `{ttl_path}`

## Parse status
- Status: **FAILED**

## Why evaluation did not run
The ontology could not be parsed as valid Turtle by `rdflib`, so no RDF graph was created. Because of that, coverage, structural, schema, logical, and precision metrics could not be computed.

## Parse error
```text
{parse_error} """

def render_report(
    ttl_path: Path,
    g: Graph,
    concept_precision: PrecisionResult,
    taxonomy_precision: PrecisionResult,
    relation_precision: PrecisionResult,
) -> str:
    classes = get_declared_classes(g)
    obj_props = get_declared_object_properties(g)
    data_props = get_declared_datatype_properties(g)
    tax_edges = get_named_taxonomy_edges(g)
    non_tax = get_non_taxonomy_triples(g)
    tax_self_loops = [(c, p) for c, p in tax_edges if c == p]
    tax_cycles = find_cycles([(c, p) for c, p in tax_edges if c != p])
    tax_dup = count_duplicates(tax_edges)
    non_tax_dup = count_duplicates(non_tax)
    depth = taxonomy_depth(tax_edges)

    roots, leaves = taxonomy_roots_and_leaves(tax_edges)
    comps = connected_components_undirected(tax_edges)

    class_supers = transitive_superclasses(tax_edges)
    schema = schema_checks(g, classes, obj_props, data_props)
    dom_rng = domain_range_checks(g, class_supers, obj_props, data_props)

    predicate_freq = Counter([p for _, p, _ in non_tax])

    lines: List[str] = []
    lines.append("# Ontology Evaluation Report")
    lines.append("")
    lines.append("## Input")
    lines.append(f"- TTL file: `{ttl_path}`")
    lines.append("- Parse status: **PASS**")
    lines.append(f"- Total RDF triples: **{len(g)}**")
    lines.append("")

    lines.append("## 1. Coverage")
    lines.append(f"- Declared classes: **{len(classes)}**")
    lines.append(f"- Declared object properties: **{len(obj_props)}**")
    lines.append(f"- Declared datatype properties: **{len(data_props)}**")
    lines.append(f"- Named taxonomy edges (`rdfs:subClassOf`): **{len(tax_edges)}**")
    lines.append(f"- Non-taxonomy triples: **{len(non_tax)}**")
    lines.append(f"- Approximate taxonomy depth: **{depth}**")
    lines.append(f"- Taxonomy roots: **{len(roots)}**")
    lines.append(f"- Taxonomy leaves: **{len(leaves)}**")
    lines.append(f"- Undirected taxonomy connected components: **{len(comps)}**")
    if roots:
        lines.append(f"- Sample roots: {', '.join(node_label(g, x) for x in sorted(list(roots), key=str)[:10])}")
    if leaves:
        lines.append(f"- Sample leaves: {', '.join(node_label(g, x) for x in sorted(list(leaves), key=str)[:10])}")
    if predicate_freq:
        lines.append("- Top non-taxonomy predicates by frequency:")
        for p, cnt in predicate_freq.most_common(10):
            lines.append(f"  - {node_label(g, p)}: {cnt}")
    lines.append("")

    lines.append("## 2. Structural validity")
    lines.append(f"- Taxonomy self-loops (`A ⊑ A`): **{len(tax_self_loops)}**")
    lines.append(f"- Taxonomy cycles: **{len(tax_cycles)}**")
    lines.append(f"- Duplicate taxonomy edges: **{tax_dup}**")
    lines.append(f"- Duplicate non-taxonomy triples: **{non_tax_dup}**")
    if tax_cycles:
        lines.append("- Sample taxonomy cycles:")
        for cyc in tax_cycles[:5]:
            lines.append(f"  - {' -> '.join(node_label(g, x) for x in cyc)}")
    lines.append("")

    lines.append("## 3. Schema quality")
    lines.append(f"- Resources declared both as class and property: **{len(schema['class_property_overlap'])}**")
    if schema["class_property_overlap"]:
        lines.append(f"  - Sample: {', '.join(node_label(g, x) for x in schema['class_property_overlap'][:10])}")

    lines.append(f"- Resources declared both as object and datatype property: **{len(schema['object_datatype_overlap'])}**")
    if schema["object_datatype_overlap"]:
        lines.append(f"  - Sample: {', '.join(node_label(g, x) for x in schema['object_datatype_overlap'][:10])}")

    lines.append(f"- Object properties with XSD/literal ranges: **{len(schema['object_property_with_literal_range'])}**")
    if schema["object_property_with_literal_range"]:
        for p, r in schema["object_property_with_literal_range"][:10]:
            lines.append(f"  - {node_label(g, p)} -> range {node_label(g, r)}")

    lines.append(f"- Datatype properties with non-XSD/class-like ranges: **{len(schema['datatype_property_with_nonliteral_range'])}**")
    if schema["datatype_property_with_nonliteral_range"]:
        for p, r in schema["datatype_property_with_nonliteral_range"][:10]:
            lines.append(f"  - {node_label(g, p)} -> range {node_label(g, r)}")

    lines.append(f"- Undeclared predicates used in assertions: **{len(schema['undeclared_predicates'])}**")
    if schema["undeclared_predicates"]:
        lines.append(f"  - Sample: {', '.join(node_label(g, x) for x in schema['undeclared_predicates'][:10])}")
    lines.append("")

    lines.append("## 4. Logical / usage quality")
    lines.append(f"- Datatype properties used with non-literal objects: **{len(dom_rng['datatype_property_nonliteral_objects'])}**")
    lines.append(f"- Object properties used with literal objects: **{len(dom_rng['object_property_literal_objects'])}**")
    lines.append(f"- Subject domain violations: **{len(dom_rng['subject_domain_violations'])}**")
    lines.append(f"- Object range violations: **{len(dom_rng['object_range_violations'])}**")
    lines.append("")

    lines.append("## 5. Sample-based precision")
    lines.append("Precision cannot be computed from the ontology file alone. It requires manually validated samples.")
    lines.append("The CSV must contain a column named `manual_label` with values `Valid`, `Invalid`, or `Ambiguous`.")
    lines.append(summarize_precision("Concept precision", concept_precision))
    lines.append(summarize_precision("Taxonomy precision", taxonomy_precision))
    lines.append(summarize_precision("Relation precision", relation_precision))
    lines.append("")

    lines.append("## 6. Short interpretation")
    lines.append("This report focuses on the main evaluation dimensions only: coverage, structural validity, schema quality, logical/usage quality, and manual precision.")
    lines.append("Coverage shows how much ontology content was produced.")
    lines.append("Taxonomy roots, leaves, and connected components help interpret whether the hierarchy is broad, shallow, or fragmented.")
    lines.append("Top predicates show which relations dominate the ontology.")
    lines.append("Structural validity checks whether the hierarchy and assertion graph are clean.")
    lines.append("Schema quality checks whether ontology elements are modeled consistently.")
    lines.append("Logical/usage quality checks whether properties are used with appropriate subjects and objects.")
    lines.append("Precision evaluates whether extracted concepts and relations are semantically correct based on manual validation.")
    lines.append("")

    return "\n".join(lines)

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a TTL ontology and write a simple Markdown report.")
    ap.add_argument("--ttl", required=True, help="Path to input TTL ontology file")
    ap.add_argument("--report", required=True, help="Path to output Markdown report file")
    ap.add_argument("--concept-eval-csv", default=None, help="Optional CSV with a 'manual_label' column for concept precision")
    ap.add_argument("--taxonomy-eval-csv", default=None, help="Optional CSV with a 'manual_label' column for taxonomy precision")
    ap.add_argument("--relation-eval-csv", default=None, help="Optional CSV with a 'manual_label' column for relation precision")
    args = ap.parse_args()
    ttl_path = Path(args.ttl)
    report_path = Path(args.report)

    parse_res = parse_ttl(ttl_path)
    if not parse_res.ok:
        write_report(report_path, report_for_parse_failure(ttl_path, parse_res.error or "Unknown parse error"))
        print(f"Report written: {report_path}")
        return

    concept_precision = read_precision_csv(Path(args.concept_eval_csv)) if args.concept_eval_csv else PrecisionResult(False, note="No annotation CSV provided.")
    taxonomy_precision = read_precision_csv(Path(args.taxonomy_eval_csv)) if args.taxonomy_eval_csv else PrecisionResult(False, note="No annotation CSV provided.")
    relation_precision = read_precision_csv(Path(args.relation_eval_csv)) if args.relation_eval_csv else PrecisionResult(False, note="No annotation CSV provided.")

    report = render_report(
        ttl_path=ttl_path,
        g=parse_res.graph,  # type: ignore[arg-type]
        concept_precision=concept_precision,
        taxonomy_precision=taxonomy_precision,
        relation_precision=relation_precision,
    )
    write_report(report_path, report)
    print(f"Report written: {report_path}")

if __name__ == "__main__":
    main()