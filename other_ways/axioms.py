from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


# -----------------------------
# Helpers: normalization
# -----------------------------

def normalize_term(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# SQLite introspection
# -----------------------------

def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    )
    return cur.fetchone() is not None


def get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cur.fetchall()]


def pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def detect_taxonomy_columns(conn: sqlite3.Connection, taxonomy_table: str) -> Tuple[str, str]:
    cols = get_table_columns(conn, taxonomy_table)

    # UPDATED: includes your schema names
    child_candidates = [
        "child_canonical_term", "child_term", "child", "child_text", "child_label",
        "child_canonical", "child_canonical_id",
        "sub", "subject", "term", "source", "head_canonical_term", "head_term", "head_text"
    ]
    parent_candidates = [
        "parent_canonical_term", "parent_term", "parent", "parent_text", "parent_label",
        "parent_canonical", "parent_canonical_id",
        "sup", "object", "target", "tail_canonical_term", "tail_term", "tail_text"
    ]

    child_col = pick_column(cols, child_candidates)
    parent_col = pick_column(cols, parent_candidates)

    if not child_col or not parent_col:
        raise RuntimeError(
            f"Could not detect child/parent columns for taxonomy_table='{taxonomy_table}'. "
            f"Columns found: {cols}. "
            f"Expected something like (child,parent) or similar."
        )
    return child_col, parent_col


def detect_triple_columns(conn: sqlite3.Connection, triple_table: str) -> Tuple[str, str, str]:
    cols = get_table_columns(conn, triple_table)

    # UPDATED: includes common OLAF/HPC schemas
    subj_candidates = [
        "subject", "subj", "arg1", "head", "head_term", "head_text", "s", "source", "term1",
        "head_canonical_term", "child_canonical_term", "child_term"
    ]
    rel_candidates = [
        "rel_text", "rel", "relation", "predicate", "p", "verb", "rel_phrase", "relation_text"
    ]
    obj_candidates = [
        "object", "obj", "arg2", "tail", "tail_term", "tail_text", "o", "target", "term2",
        "tail_canonical_term", "parent_canonical_term", "parent_term"
    ]

    s_col = pick_column(cols, subj_candidates)
    r_col = pick_column(cols, rel_candidates)
    o_col = pick_column(cols, obj_candidates)

    if not s_col or not r_col or not o_col:
        raise RuntimeError(
            f"Could not detect (subject, relation, object) columns for triple_table='{triple_table}'. "
            f"Columns found: {cols}. "
            f"Expected columns like (subject, rel_text, object) or similar."
        )
    return s_col, r_col, o_col


def detect_types_columns(conn: sqlite3.Connection, types_table: str) -> Tuple[str, str]:
    cols = get_table_columns(conn, types_table)

    term_candidates = [
        "canonical_term", "term", "label", "surface", "concept", "name",
        "child_canonical_term"
    ]
    type_candidates = [
        "coarse_type", "type", "term_type", "category", "class", "semantic_type"
    ]

    t_col = pick_column(cols, term_candidates)
    ty_col = pick_column(cols, type_candidates)

    if not t_col or not ty_col:
        raise RuntimeError(
            f"Could not detect (term,type) columns for types_table='{types_table}'. "
            f"Columns found: {cols}. "
            f"Expected something like (canonical_term, coarse_type) or similar."
        )
    return t_col, ty_col


# -----------------------------
# Taxonomy graph utilities
# -----------------------------

def build_taxonomy_graph(edges: Iterable[Tuple[str, str]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """parents[node] = set of parents. children[node] = set of children."""
    parents: Dict[str, Set[str]] = defaultdict(set)
    children: Dict[str, Set[str]] = defaultdict(set)

    for child, parent in edges:
        c = normalize_term(child)
        p = normalize_term(parent)
        if not c or not p:
            continue
        parents[c].add(p)
        children[p].add(c)

    nodes = set(parents.keys()) | set(children.keys())
    for n in list(nodes):
        parents.setdefault(n, set())
        children.setdefault(n, set())

    return parents, children


def detect_cycle(parents: Dict[str, Set[str]]) -> Optional[List[str]]:
    """
    Return a cycle path if found, else None.
    DFS on directed edges child -> parent.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in parents}
    parent_in_dfs: Dict[str, Optional[str]] = {n: None for n in parents}

    def dfs(u: str) -> Optional[List[str]]:
        color[u] = GRAY
        for v in parents.get(u, set()):
            if v not in color:
                color[v] = WHITE
                parent_in_dfs[v] = None
            if color[v] == WHITE:
                parent_in_dfs[v] = u
                cyc = dfs(v)
                if cyc:
                    return cyc
            elif color[v] == GRAY:
                # Found back-edge u -> v, reconstruct
                cycle = [v]
                cur = u
                while cur != v and cur is not None:
                    cycle.append(cur)
                    cur = parent_in_dfs[cur]
                cycle.append(v)
                cycle.reverse()
                return cycle
        color[u] = BLACK
        return None

    for n in list(parents.keys()):
        if color[n] == WHITE:
            cyc = dfs(n)
            if cyc:
                return cyc
    return None


def compute_depths(parents: Dict[str, Set[str]]) -> Dict[str, int]:
    """
    Approx depth estimate = longest distance from any root.
    Works for DAG; call detect_cycle() before this.
    """
    nodes = set(parents.keys())
    depth = {n: 0 for n in nodes}

    for _ in range(max(1, len(nodes))):
        changed = False
        for n in nodes:
            ps = parents.get(n, set())
            if not ps:
                if depth[n] != 0:
                    depth[n] = 0
                    changed = True
            else:
                best = max(depth.get(p, 0) for p in ps) + 1
                if best > depth[n]:
                    depth[n] = best
                    changed = True
        if not changed:
            break

    return depth


def ancestors_of(node: str, parents: Dict[str, Set[str]], cache: Dict[str, Set[str]]) -> Set[str]:
    if node in cache:
        return cache[node]
    seen: Set[str] = set()
    stack = [node]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for p in parents.get(u, set()):
            if p not in seen:
                stack.append(p)
    cache[node] = seen
    return seen


def lca(nodes: List[str], parents: Dict[str, Set[str]], depth: Dict[str, int]) -> Optional[str]:
    """Lowest common ancestor (deepest common ancestor by depth)."""
    nodes = [n for n in (normalize_term(x) for x in nodes) if n]
    if not nodes:
        return None

    cache: Dict[str, Set[str]] = {}
    common: Optional[Set[str]] = None
    for n in nodes:
        anc = ancestors_of(n, parents, cache)
        common = anc if common is None else (common & anc)
        if not common:
            return None

    return max(common, key=lambda x: depth.get(x, 0))


# -----------------------------
# Typing utilities
# -----------------------------

def load_types(conn: sqlite3.Connection, types_table: str) -> Dict[str, str]:
    term_col, type_col = detect_types_columns(conn, types_table)
    cur = conn.execute(f"SELECT {term_col}, {type_col} FROM {types_table};")
    mapping: Dict[str, str] = {}
    for t, ty in cur.fetchall():
        t2, ty2 = normalize_term(t), normalize_term(ty)
        if t2 and ty2:
            mapping[t2] = ty2
    return mapping


def infer_type_from_ancestors(
    term: str,
    types: Dict[str, str],
    parents: Dict[str, Set[str]],
    max_hops: int = 10,
) -> Optional[str]:
    """
    If term has no direct type, walk up taxonomy parents to find the nearest typed ancestor.
    """
    if term in types:
        return types[term]

    visited: Set[str] = set()
    q = deque([(term, 0)])
    while q:
        node, d = q.popleft()
        if node in visited:
            continue
        visited.add(node)
        if d > max_hops:
            continue
        for p in parents.get(node, set()):
            if p in types:
                return types[p]
            q.append((p, d + 1))
    return None


# -----------------------------
# Axiom dataclasses
# -----------------------------

@dataclass
class SubClassOfAxiom:
    child: str
    parent: str
    status: str = "accepted"
    confidence: float = 1.0


@dataclass
class PropertyAxiom:
    property: str
    kind: str  # ObjectProperty
    status: str = "accepted"
    confidence: float = 1.0


@dataclass
class DomainRangeAxiom:
    property: str
    domain: Optional[str]
    range: Optional[str]
    support: int
    typed_subjects: int
    typed_objects: int
    subject_type_coverage: float
    object_type_coverage: float
    purity_domain: float
    purity_range: float
    confidence: float
    status: str
    method: str
    evidence_examples: List[Tuple[str, str, str]]
    subject_type_hist: Dict[str, int]
    object_type_hist: Dict[str, int]


# -----------------------------
# Loaders with overrides
# -----------------------------

def load_taxonomy_edges(
    conn: sqlite3.Connection,
    taxonomy_table: str,
    child_override: Optional[str] = None,
    parent_override: Optional[str] = None,
) -> List[Tuple[str, str]]:
    if child_override and parent_override:
        child_col, parent_col = child_override, parent_override
    else:
        child_col, parent_col = detect_taxonomy_columns(conn, taxonomy_table)

    cur = conn.execute(f"SELECT {child_col}, {parent_col} FROM {taxonomy_table};")
    edges: List[Tuple[str, str]] = []
    for row in cur.fetchall():
        c, p = normalize_term(row[0]), normalize_term(row[1])
        if c and p:
            edges.append((c, p))
    return edges


def load_triples(
    conn: sqlite3.Connection,
    triple_table: str,
    subj_override: Optional[str] = None,
    rel_override: Optional[str] = None,
    obj_override: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    if subj_override and rel_override and obj_override:
        s_col, r_col, o_col = subj_override, rel_override, obj_override
    else:
        s_col, r_col, o_col = detect_triple_columns(conn, triple_table)

    cur = conn.execute(f"SELECT {s_col}, {r_col}, {o_col} FROM {triple_table};")
    triples: List[Tuple[str, str, str]] = []
    for s, r, o in cur.fetchall():
        s2, r2, o2 = normalize_term(s), normalize_term(r), normalize_term(o)
        if s2 and r2 and o2:
            triples.append((s2, r2, o2))
    return triples


# -----------------------------
# Domain/Range induction
# -----------------------------

def compute_domain_range_for_relation(
    relation: str,
    triples: List[Tuple[str, str, str]],
    types: Optional[Dict[str, str]],
    parents: Dict[str, Set[str]],
    depth: Dict[str, int],
    min_support: int,
    min_purity: float,
    evidence_k: int = 5,
) -> DomainRangeAxiom:
    rel_triples = [(s, r, o) for (s, r, o) in triples if r == relation]
    support = len(rel_triples)
    examples = rel_triples[:evidence_k]

    subjects = [s for (s, _, _) in rel_triples]
    objects = [o for (_, _, o) in rel_triples]

    subj_types: List[str] = []
    obj_types: List[str] = []

    typed_subjects = 0
    typed_objects = 0

    if types:
        for s in subjects:
            ty = types.get(s) or infer_type_from_ancestors(s, types, parents)
            if ty:
                subj_types.append(ty)
                typed_subjects += 1
        for o in objects:
            ty = types.get(o) or infer_type_from_ancestors(o, types, parents)
            if ty:
                obj_types.append(ty)
                typed_objects += 1

    subj_hist = Counter(subj_types)
    obj_hist = Counter(obj_types)

    # Domain
    if types and subj_types:
        domain = subj_hist.most_common(1)[0][0]
        purity_domain = subj_hist[domain] / len(subj_types)
        domain_method = "mode(inferred_subject_types)"
    else:
        domain = lca(subjects, parents, depth)
        purity_domain = 1.0 if domain else 0.0
        domain_method = "lca(subject_terms)"

    # Range
    if types and obj_types:
        range_ = obj_hist.most_common(1)[0][0]
        purity_range = obj_hist[range_] / len(obj_types)
        range_method = "mode(inferred_object_types)"
    else:
        range_ = lca(objects, parents, depth)
        purity_range = 1.0 if range_ else 0.0
        range_method = "lca(object_terms)"

    method = f"{domain_method} + {range_method}"

    # Coverage
    subj_cov = (typed_subjects / support) if support else 0.0
    obj_cov = (typed_objects / support) if support else 0.0

    # Confidence: support + purity + coverage
    support_score = min(1.0, math.log10(support + 1) / 2.0)  # ~0..1
    purity_score = min(purity_domain, purity_range)
    coverage_score = min(subj_cov, obj_cov) if types else 1.0

    confidence = round(0.10 + 0.40 * support_score + 0.35 * purity_score + 0.15 * coverage_score, 4)

    status = "accepted" if (
        support >= min_support
        and purity_domain >= min_purity
        and purity_range >= min_purity
        and (coverage_score >= 0.30 if types else True)   # require some typing coverage if using types
    ) else "candidate"

    return DomainRangeAxiom(
        property=relation,
        domain=domain,
        range=range_,
        support=support,
        typed_subjects=typed_subjects,
        typed_objects=typed_objects,
        subject_type_coverage=round(subj_cov, 4),
        object_type_coverage=round(obj_cov, 4),
        purity_domain=round(purity_domain, 4),
        purity_range=round(purity_range, 4),
        confidence=confidence,
        status=status,
        method=method,
        evidence_examples=examples,
        subject_type_hist=dict(subj_hist.most_common(10)),
        object_type_hist=dict(obj_hist.most_common(10)),
    )


# -----------------------------
# Optional OWL export
# -----------------------------

def export_to_owl(
    out_path: str,
    classes: Set[str],
    subclass_axioms: List[SubClassOfAxiom],
    properties: Set[str],
    domain_range_axioms: List[DomainRangeAxiom],
    base_iri: str = "http://example.org/hpc-onto#",
) -> None:
    try:
        from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal
    except Exception as e:
        print("[WARN] rdflib not installed; skipping OWL export.")
        print(f"       Install with: pip install rdflib\n       Error: {e}")
        return

    g = Graph()
    NS = Namespace(base_iri)

    def iri(local: str) -> URIRef:
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in local.strip())
        if not safe:
            safe = "Unnamed"
        return NS[safe]

    for c in classes:
        u = iri(c)
        g.add((u, RDF.type, OWL.Class))
        g.add((u, RDFS.label, Literal(c)))

    for p in properties:
        u = iri(p)
        g.add((u, RDF.type, OWL.ObjectProperty))
        g.add((u, RDFS.label, Literal(p)))

    for ax in subclass_axioms:
        if ax.status != "accepted":
            continue
        g.add((iri(ax.child), RDFS.subClassOf, iri(ax.parent)))

    for ax in domain_range_axioms:
        if ax.status != "accepted":
            continue
        prop_u = iri(ax.property)
        if ax.domain:
            g.add((prop_u, RDFS.domain, iri(ax.domain)))
        if ax.range:
            g.add((prop_u, RDFS.range, iri(ax.range)))

    g.serialize(destination=out_path, format="turtle")
    print(f"[OK] OWL/Turtle exported to: {out_path}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite DB")
    ap.add_argument("--out_dir", required=True, help="Output directory for JSON/OWL")

    ap.add_argument("--taxonomy_table", default="term_is_a", help="Table containing is-a edges")
    ap.add_argument("--triple_table", default="non_taxonomic_edges_clean", help="Table containing (s,rel,o) triples")
    ap.add_argument("--types_table", default=None, help="Optional table containing (term,type) mapping")

    # UPDATED: column overrides
    ap.add_argument("--tax_child_col", default=None, help="Override taxonomy child column name")
    ap.add_argument("--tax_parent_col", default=None, help="Override taxonomy parent column name")
    ap.add_argument("--triple_subj_col", default=None, help="Override triple subject column name")
    ap.add_argument("--triple_rel_col", default=None, help="Override triple relation column name")
    ap.add_argument("--triple_obj_col", default=None, help="Override triple object column name")

    ap.add_argument("--min_support", type=int, default=10, help="Min triple count to accept domain/range")
    ap.add_argument("--min_purity", type=float, default=0.75, help="Min purity to accept domain/range")
    ap.add_argument("--evidence_k", type=int, default=5, help="How many example triples to store per relation")

    ap.add_argument("--export_owl", action="store_true", help="Export accepted axioms to OWL/Turtle (requires rdflib)")
    ap.add_argument("--base_iri", default="http://example.org/hpc-onto#", help="Base IRI for OWL export")

    args = ap.parse_args()

    safe_mkdir(args.out_dir)

    conn = sqlite3.connect(args.db)

    if not table_exists(conn, args.taxonomy_table):
        raise RuntimeError(f"taxonomy_table '{args.taxonomy_table}' not found in DB.")
    if not table_exists(conn, args.triple_table):
        raise RuntimeError(f"triple_table '{args.triple_table}' not found in DB.")
    if args.types_table and not table_exists(conn, args.types_table):
        raise RuntimeError(f"types_table '{args.types_table}' not found in DB.")

    print(f"[INFO] Loading taxonomy edges from: {args.taxonomy_table}")
    tax_edges = load_taxonomy_edges(conn, args.taxonomy_table, args.tax_child_col, args.tax_parent_col)

    parents, children = build_taxonomy_graph(tax_edges)
    cyc = detect_cycle(parents)
    if cyc:
        raise RuntimeError(
            "Cycle detected in taxonomy (is-a). Fix this before axiom generation.\n"
            f"Cycle example: {' -> '.join(cyc)}"
        )
    depth = compute_depths(parents)

    print(f"[INFO] Loading triples from: {args.triple_table}")
    triples = load_triples(conn, args.triple_table, args.triple_subj_col, args.triple_rel_col, args.triple_obj_col)

    types = None
    if args.types_table:
        print(f"[INFO] Loading types from: {args.types_table}")
        types = load_types(conn, args.types_table)
        print(f"[INFO] Types loaded: {len(types)} term->type mappings")
    else:
        print("[INFO] No types_table provided. Will use taxonomy LCA fallback when needed.")

    # Phase 1: Classes + SubClassOf axioms
    classes: Set[str] = set()
    subclass_axioms: List[SubClassOfAxiom] = []
    for c, p in tax_edges:
        classes.add(c)
        classes.add(p)
        subclass_axioms.append(SubClassOfAxiom(child=c, parent=p))

    # Include subjects/objects as classes too (helps coverage)
    for s, r, o in triples:
        classes.add(s)
        classes.add(o)

    # Also include type labels as classes (if types_table present)
    if types:
        for ty in set(types.values()):
            classes.add(ty)

    # Phase 2: Properties
    properties = sorted({r for (_, r, _) in triples if normalize_term(r)})

    # Phase 3: Domain/Range induction
    print(f"[INFO] Computing domain/range for {len(properties)} relations...")
    dr_axioms: List[DomainRangeAxiom] = []
    for rel in properties:
        ax = compute_domain_range_for_relation(
            relation=rel,
            triples=triples,
            types=types,
            parents=parents,
            depth=depth,
            min_support=args.min_support,
            min_purity=args.min_purity,
            evidence_k=args.evidence_k,
        )
        dr_axioms.append(ax)

    # Outputs
    write_json(os.path.join(args.out_dir, "classes.json"), {
        "count": len(classes),
        "classes": sorted(classes),
    })

    write_json(os.path.join(args.out_dir, "properties.json"), {
        "count": len(properties),
        "object_properties": properties,
    })

    write_json(os.path.join(args.out_dir, "subclass_axioms.json"), {
        "count": len(subclass_axioms),
        "axioms": [asdict(a) for a in subclass_axioms],
    })

    write_json(os.path.join(args.out_dir, "domain_range_axioms.json"), {
        "min_support": args.min_support,
        "min_purity": args.min_purity,
        "accepted_count": sum(1 for a in dr_axioms if a.status == "accepted"),
        "candidate_count": sum(1 for a in dr_axioms if a.status == "candidate"),
        "axioms": [asdict(a) for a in dr_axioms],
    })

    # Simple quality report
    generic_relations = {
        "use", "include", "provide", "get", "set", "show", "create", "require", "support",
        "contain", "add", "take", "enable", "specify", "represent", "receive", "assign"
    }

    flagged = []
    for a in dr_axioms:
        if a.property.lower() in generic_relations:
            flagged.append({"relation": a.property, "reason": "generic_relation_phrase"})
        if a.support < args.min_support:
            flagged.append({"relation": a.property, "reason": f"low_support<{args.min_support}", "support": a.support})
        if a.status == "candidate":
            flagged.append({"relation": a.property, "reason": "candidate_domain_range", "confidence": a.confidence})
        if args.types_table:
            if a.subject_type_coverage < 0.30 or a.object_type_coverage < 0.30:
                flagged.append({
                    "relation": a.property,
                    "reason": "low_type_coverage",
                    "subj_cov": a.subject_type_coverage,
                    "obj_cov": a.object_type_coverage
                })

    write_json(os.path.join(args.out_dir, "axiom_quality_report.json"), {
        "taxonomy_edges": len(subclass_axioms),
        "taxonomy_nodes": len(classes),
        "triple_count": len(triples),
        "relation_count": len(properties),
        "flagged_count": len(flagged),
        "flagged": flagged[:2000],
    })

    print(f"[OK] Wrote outputs to: {args.out_dir}")
    print("     - classes.json")
    print("     - properties.json")
    print("     - subclass_axioms.json")
    print("     - domain_range_axioms.json")
    print("     - axiom_quality_report.json")

    if args.export_owl:
        out_owl = os.path.join(args.out_dir, "ontology_lightweight.ttl")
        export_to_owl(
            out_path=out_owl,
            classes=classes,
            subclass_axioms=subclass_axioms,
            properties=set(properties),
            domain_range_axioms=dr_axioms,
            base_iri=args.base_iri,
        )

    conn.close()


if __name__ == "__main__":
    main()
