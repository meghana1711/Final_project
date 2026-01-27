from __future__ import annotations

import argparse
import json
import math
import os
import re
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

    child_candidates = [
        "child_canonical_term",
        "child_term",
        "child",
        "child_text",
        "child_label",
        "head_canonical_term",
        "head_term",
        "head_text",
        "source",
        "sub",
        "subject",
        "term",
        "child_canonical_id",
    ]

    parent_candidates = [
        "parent_canonical_term",
        "parent_term",
        "parent",
        "parent_text",
        "parent_label",
        "tail_canonical_term",
        "tail_term",
        "tail_text",
        "target",
        "object",
        "sup",
        "parent_canonical_id",
        # OLAF taxonomy tables
        "proposed_parent_canonical_term",
        "proposed_parent_canonical_id",
        "llm_best_parent_canonical_term",
        "llm_best_parent_canonical_id",
    ]

    child_col = pick_column(cols, child_candidates)
    parent_col = pick_column(cols, parent_candidates)

    if not child_col or not parent_col:
        raise RuntimeError(
            f"Could not detect child/parent columns for taxonomy_table='{taxonomy_table}'. "
            f"Columns found: {cols}."
        )
    return child_col, parent_col


def detect_triple_columns(conn: sqlite3.Connection, triple_table: str) -> Tuple[str, str, str]:
    cols = get_table_columns(conn, triple_table)

    subj_candidates = [
        "subj_canonical_term",
        "subj_text",
        "subject",
        "subj",
        "arg1",
        "s",
        "head",
        "head_text",
        "head_term",
        "source",
    ]
    rel_candidates = [
        "rel_text",
        "rel",
        "relation",
        "predicate",
        "p",
        "verb",
        "relation_text",
        "rel_phrase",
    ]
    obj_candidates = [
        "obj_canonical_term",
        "obj_text",
        "object",
        "obj",
        "arg2",
        "o",
        "tail",
        "tail_text",
        "tail_term",
        "target",
    ]

    s_col = pick_column(cols, subj_candidates)
    r_col = pick_column(cols, rel_candidates)
    o_col = pick_column(cols, obj_candidates)

    if not s_col or not r_col or not o_col:
        raise RuntimeError(
            f"Could not detect (subject, relation, object) columns for triple_table='{triple_table}'. "
            f"Columns found: {cols}."
        )
    return s_col, r_col, o_col


def detect_enrichment_columns(conn: sqlite3.Connection, types_table: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Your term_enrichment_extension table example columns:
      scheduler  category  ontology_role  dul_bucket
    We want:
      term_col         (e.g., scheduler/canonical_term/term)
      ontology_role_col (optional)
      dul_bucket_col    (optional)
    """
    cols = get_table_columns(conn, types_table)

    term_candidates = [
        "canonical_term", "term", "label", "surface", "concept", "name",
        "child_canonical_term",
        # your enrichment extension example:
        "scheduler",
    ]
    ontology_role_candidates = [
        "ontology_role", "role", "owl_role", "ont_role"
    ]
    dul_bucket_candidates = [
        "dul_bucket", "dul", "bucket", "coarse_type", "term_type", "type", "category"
    ]

    t_col = pick_column(cols, term_candidates)
    r_col = pick_column(cols, ontology_role_candidates)
    d_col = pick_column(cols, dul_bucket_candidates)

    if not t_col:
        raise RuntimeError(
            f"Could not detect term column for types_table='{types_table}'. Columns found: {cols}"
        )
    return t_col, r_col, d_col


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
# Term typing from enrichment extension
# -----------------------------

@dataclass
class TermTypeInfo:
    term: str
    ontology_role: Optional[str] = None   # e.g., "class", "property"
    dul_bucket: Optional[str] = None      # e.g., "object", "situation", "process", "description"


def load_term_type_info(conn: sqlite3.Connection, types_table: str) -> Dict[str, TermTypeInfo]:
    term_col, role_col, dul_col = detect_enrichment_columns(conn, types_table)

    cols = [term_col]
    if role_col:
        cols.append(role_col)
    if dul_col and dul_col not in cols:
        cols.append(dul_col)

    cur = conn.execute(f"SELECT {', '.join(cols)} FROM {types_table};")
    mapping: Dict[str, TermTypeInfo] = {}

    for row in cur.fetchall():
        term = normalize_term(row[0])
        if not term:
            continue
        ontology_role = None
        dul_bucket = None

        idx = 1
        if role_col:
            ontology_role = normalize_term(row[idx])
            idx += 1
        if dul_col and (dul_col != role_col):
            dul_bucket = normalize_term(row[idx]) if idx < len(row) else None

        mapping[term] = TermTypeInfo(term=term, ontology_role=ontology_role, dul_bucket=dul_bucket)

    return mapping


# -----------------------------
# Rules: action-like terms, taxonomy filtering, relation normalization
# -----------------------------

VERB_PREFIXES = {
    "access", "grant", "accept", "use", "create", "start", "stop", "run",
    "set", "get", "show", "enable", "disable", "configure", "request",
    "assign", "allocate", "cancel", "submit", "launch", "kill"
}

SECTION_ID_RE = re.compile(r"^\s*(\d+(\.\d+)*[a-zA-Z]?)\b")

GENERIC_PARENTS = {
    "system", "information", "data", "thing", "value", "parameter", "option",
    "slurm", "lsf", "scheduler"
}

GENERIC_RELATIONS = {
    "use", "include", "provide", "get", "set", "show", "create", "require", "support",
    "contain", "add", "take", "enable", "specify", "represent", "receive", "assign"
}


def looks_like_section_id(term: str) -> bool:
    t = term.strip()
    return bool(SECTION_ID_RE.match(t))


def is_action_like(term: str) -> bool:
    t = term.strip().lower().replace("-", " ")
    parts = t.split()
    if not parts:
        return False
    first = parts[0]
    if first in VERB_PREFIXES:
        return True
    # Heuristic: common verb+object phrase shape
    if len(parts) >= 2 and first in VERB_PREFIXES:
        return True
    return False


def normalize_relation(r: str) -> str:
    """
    Normalize relation strings to stable property local names.
    Examples:
      "accept as" -> "accept_as"
      "Runs-On"   -> "runs_on"
    """
    r = r.strip().lower()
    r = r.replace("-", " ")
    r = re.sub(r"\s+", " ", r)
    r = r.replace(" ", "_")
    r = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in r)
    r = re.sub(r"_+", "_", r).strip("_")
    return r or "related_to"


def keep_as_class(term: str, type_info: Optional[Dict[str, TermTypeInfo]] = None) -> bool:
    """
    Decide if a term should be declared as owl:Class.
    - Do not add action-like terms as Classes.
    - If enrichment types exist:
        ontology_role == "property" -> not a class
        dul_bucket indicates non-entity buckets -> not a class
    """
    if not term:
        return False

    if type_info and term in type_info:
        ti = type_info[term]
        if ti.ontology_role and ti.ontology_role.strip().lower() in {"property", "relation", "predicate", "objectproperty"}:
            return False
        if ti.dul_bucket and ti.dul_bucket.strip().lower() in {"process", "event", "action"}:
            return False

    if is_action_like(term):
        return False

    if looks_like_section_id(term):
        return False

    return True


def accept_tax_edge(child: str, parent: str, type_info: Optional[Dict[str, TermTypeInfo]]) -> bool:
    """
    Filter taxonomy edges:
      - drop section IDs / doc headings
      - drop action-like child terms
      - drop very-generic parents unless typed info suggests it's OK
    """
    if not child or not parent:
        return False

    if looks_like_section_id(child) or looks_like_section_id(parent):
        return False

    if is_action_like(child):
        return False

    # If parent is generic, require typed evidence or non-action child + non-generic bucket
    p_low = parent.strip().lower()
    if p_low in GENERIC_PARENTS:
        if type_info and child in type_info:
            # allow if child is explicitly marked as class/object/situation etc.
            return True
        return False

    # If we have DUL buckets for both, reject obvious mismatches (very lightweight)
    if type_info and child in type_info and parent in type_info:
        cb = (type_info[child].dul_bucket or "").strip().lower()
        pb = (type_info[parent].dul_bucket or "").strip().lower()
        if cb and pb:
            # Example: situation should not be subclass of object (usually)
            if cb == "situation" and pb == "object":
                return False
            # object should not be subclass of situation
            if cb == "object" and pb == "situation":
                return False

    return True


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


@dataclass
class InversePropertyAxiom:
    property: str
    inverse_of: str
    support: int
    ratio: float
    confidence: float
    status: str
    evidence_examples: List[Tuple[str, str, str]]


# -----------------------------
# Loaders with WHERE filters
# -----------------------------

def _select_sql(table: str, cols: List[str], where: Optional[str]) -> str:
    c = ", ".join(cols)
    sql = f"SELECT {c} FROM {table}"
    if where and where.strip():
        sql += f" WHERE {where}"
    sql += ";"
    return sql


def load_taxonomy_edges(
    conn: sqlite3.Connection,
    taxonomy_table: str,
    child_override: Optional[str] = None,
    parent_override: Optional[str] = None,
    where: Optional[str] = None,
) -> List[Tuple[str, str]]:
    if child_override and parent_override:
        child_col, parent_col = child_override, parent_override
    else:
        child_col, parent_col = detect_taxonomy_columns(conn, taxonomy_table)

    sql = _select_sql(taxonomy_table, [child_col, parent_col], where)
    cur = conn.execute(sql)

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
    where: Optional[str] = None,
    normalize_relations: bool = True,
) -> List[Tuple[str, str, str]]:
    if subj_override and rel_override and obj_override:
        s_col, r_col, o_col = subj_override, rel_override, obj_override
    else:
        s_col, r_col, o_col = detect_triple_columns(conn, triple_table)

    sql = _select_sql(triple_table, [s_col, r_col, o_col], where)
    cur = conn.execute(sql)

    triples: List[Tuple[str, str, str]] = []
    for s, r, o in cur.fetchall():
        s2, r2, o2 = normalize_term(s), normalize_term(r), normalize_term(o)
        if not (s2 and r2 and o2):
            continue
        if normalize_relations:
            r2 = normalize_relation(r2)
        triples.append((s2, r2, o2))
    return triples


# -----------------------------
# Domain/Range induction
# -----------------------------

def infer_type_from_term_info(term: str, type_info: Dict[str, TermTypeInfo]) -> Optional[str]:
    """
    We use DUL buckets as "types" when present.
    This lets domain/range induction work even if you don't have a separate types_table.
    """
    ti = type_info.get(term)
    if not ti:
        return None
    if ti.dul_bucket:
        return ti.dul_bucket.strip().lower()
    if ti.ontology_role:
        return ti.ontology_role.strip().lower()
    return None


def infer_type_from_ancestors(
    term: str,
    type_info: Dict[str, TermTypeInfo],
    parents: Dict[str, Set[str]],
    max_hops: int = 10,
) -> Optional[str]:
    """
    If term isn't typed directly, walk up taxonomy to find an ancestor with dul_bucket.
    """
    direct = infer_type_from_term_info(term, type_info)
    if direct:
        return direct

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
            ty = infer_type_from_term_info(p, type_info)
            if ty:
                return ty
            q.append((p, d + 1))
    return None


def compute_domain_range_for_relation(
    relation: str,
    triples: List[Tuple[str, str, str]],
    type_info: Optional[Dict[str, TermTypeInfo]],
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

    if type_info:
        for s in subjects:
            ty = infer_type_from_term_info(s, type_info) or infer_type_from_ancestors(s, type_info, parents)
            if ty:
                subj_types.append(ty)
                typed_subjects += 1
        for o in objects:
            ty = infer_type_from_term_info(o, type_info) or infer_type_from_ancestors(o, type_info, parents)
            if ty:
                obj_types.append(ty)
                typed_objects += 1

    subj_hist = Counter(subj_types)
    obj_hist = Counter(obj_types)

    # Domain
    if type_info and subj_types:
        domain = subj_hist.most_common(1)[0][0]
        purity_domain = subj_hist[domain] / len(subj_types)
        domain_method = "mode(dul_bucket_subject)"
    else:
        domain = lca(subjects, parents, depth)
        purity_domain = 1.0 if domain else 0.0
        domain_method = "lca(subject_terms)"

    # Range
    if type_info and obj_types:
        range_ = obj_hist.most_common(1)[0][0]
        purity_range = obj_hist[range_] / len(obj_types)
        range_method = "mode(dul_bucket_object)"
    else:
        range_ = lca(objects, parents, depth)
        purity_range = 1.0 if range_ else 0.0
        range_method = "lca(object_terms)"

    method = f"{domain_method} + {range_method}"

    subj_cov = (typed_subjects / support) if support else 0.0
    obj_cov = (typed_objects / support) if support else 0.0

    support_score = min(1.0, math.log10(support + 1) / 2.0)
    purity_score = min(purity_domain, purity_range)
    coverage_score = min(subj_cov, obj_cov) if type_info else 1.0

    confidence = round(0.10 + 0.40 * support_score + 0.35 * purity_score + 0.15 * coverage_score, 4)

    status = "accepted" if (
        support >= min_support
        and purity_domain >= min_purity
        and purity_range >= min_purity
        and (coverage_score >= 0.30 if type_info else True)
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
# Inverse property induction (optional)
# -----------------------------

def compute_inverse_properties(
    triples: List[Tuple[str, str, str]],
    min_support: int = 10,
    min_ratio: float = 0.6,
    evidence_k: int = 5,
    drop_generic_relations: bool = True,
) -> List[InversePropertyAxiom]:
    """
    Evidence-based inverses:
      If relation A has many pairs (s,o) and relation B has many pairs (o,s),
      and overlap is high, we declare A inverseOf B.

    overlap = |pairsA ∩ swapped(pairsB)|
    ratio   = overlap / min(|pairsA|, |pairsB|)
    """
    rel_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for s, r, o in triples:
        if drop_generic_relations and r in GENERIC_RELATIONS:
            continue
        rel_pairs[r].add((s, o))

    rels = list(rel_pairs.keys())
    inv_axioms: List[InversePropertyAxiom] = []

    # Precompute swapped sets lazily
    swapped_cache: Dict[str, Set[Tuple[str, str]]] = {}

    def swapped(rel: str) -> Set[Tuple[str, str]]:
        if rel not in swapped_cache:
            swapped_cache[rel] = {(o, s) for (s, o) in rel_pairs[rel]}
        return swapped_cache[rel]

    # For each rel, find best inverse candidate
    for ra in rels:
        pairs_a = rel_pairs[ra]
        if len(pairs_a) < min_support:
            continue

        best = None  # (overlap, ratio, rb)
        for rb in rels:
            if ra == rb:
                continue
            pairs_b_swapped = swapped(rb)
            overlap = len(pairs_a & pairs_b_swapped)
            if overlap < min_support:
                continue
            denom = min(len(pairs_a), len(rel_pairs[rb]))
            ratio = overlap / denom if denom else 0.0
            if ratio >= min_ratio:
                if (best is None) or (overlap > best[0]) or (overlap == best[0] and ratio > best[1]):
                    best = (overlap, ratio, rb)

        if best:
            overlap, ratio, rb = best
            # Confidence is a simple function of overlap+ratio (bounded)
            conf = round(min(0.99, 0.20 + 0.60 * min(1.0, math.log10(overlap + 1) / 2.0) + 0.20 * ratio), 4)
            # evidence examples: pick overlapping ones and show the swapped match (as triples)
            ev = []
            count = 0
            for (s, o) in list(pairs_a):
                if (s, o) in swapped(rb):
                    ev.append((s, ra, o))
                    count += 1
                    if count >= evidence_k:
                        break

            inv_axioms.append(InversePropertyAxiom(
                property=ra,
                inverse_of=rb,
                support=overlap,
                ratio=round(ratio, 4),
                confidence=conf,
                status="accepted",
                evidence_examples=ev,
            ))

    # Deduplicate symmetric duplicates by keeping higher support
    best_by_pair: Dict[Tuple[str, str], InversePropertyAxiom] = {}
    for ax in inv_axioms:
        key = tuple(sorted([ax.property, ax.inverse_of]))
        cur = best_by_pair.get(key)
        if (cur is None) or (ax.support > cur.support) or (ax.support == cur.support and ax.ratio > cur.ratio):
            best_by_pair[key] = ax

    return list(best_by_pair.values())


# -----------------------------
# OWL export (with annotations + inverses)
# -----------------------------

def export_to_owl(
    out_path: str,
    classes: Set[str],
    subclass_axioms: List[SubClassOfAxiom],
    properties: Set[str],
    domain_range_axioms: List[DomainRangeAxiom],
    inverse_axioms: Optional[List[InversePropertyAxiom]] = None,
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
    ANNO = Namespace(base_iri.rstrip("#") + "/anno#")  # simple annotation namespace

    def iri(local: str) -> URIRef:
        safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in local.strip())
        safe = re.sub(r"_+", "_", safe).strip("_")
        if not safe:
            safe = "Unnamed"
        return NS[safe]

    # classes
    for c in classes:
        u = iri(c)
        g.add((u, RDF.type, OWL.Class))
        g.add((u, RDFS.label, Literal(c)))

    # properties
    for p in properties:
        u = iri(p)
        g.add((u, RDF.type, OWL.ObjectProperty))
        g.add((u, RDFS.label, Literal(p)))

    # subclass edges
    for ax in subclass_axioms:
        if ax.status not in {"accepted", "candidate"}:
            continue
        g.add((iri(ax.child), RDFS.subClassOf, iri(ax.parent)))

    # domain/range + confidence annotations
    for ax in domain_range_axioms:
        #if not (ax.domain and ax.range):
          #  continue
        prop_u = iri(ax.property)
        if ax.domain:
            g.add((prop_u, RDFS.domain, iri(ax.domain)))
        if ax.range:
            g.add((prop_u, RDFS.range, iri(ax.range)))

        # annotations (confidence, support, method, purity, coverage)
        g.add((prop_u, ANNO.confidence, Literal(ax.confidence)))
        g.add((prop_u, ANNO.support, Literal(ax.support)))
        g.add((prop_u, ANNO.method, Literal(ax.method)))
        g.add((prop_u, ANNO.purityDomain, Literal(ax.purity_domain)))
        g.add((prop_u, ANNO.purityRange, Literal(ax.purity_range)))
        g.add((prop_u, ANNO.subjectTypeCoverage, Literal(ax.subject_type_coverage)))
        g.add((prop_u, ANNO.objectTypeCoverage, Literal(ax.object_type_coverage)))

    # inverse properties (optional)
    if inverse_axioms:
        for ax in inverse_axioms:
            if ax.status != "accepted":
                continue
            p = iri(ax.property)
            q = iri(ax.inverse_of)
            g.add((p, OWL.inverseOf, q))
            # annotate inverse confidence/support too
            g.add((p, ANNO.inverseSupport, Literal(ax.support)))
            g.add((p, ANNO.inverseRatio, Literal(ax.ratio)))
            g.add((p, ANNO.inverseConfidence, Literal(ax.confidence)))

    g.serialize(destination=out_path, format="turtle")
    print(f"[OK] OWL/Turtle exported to: {out_path}")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite DB")
    ap.add_argument("--out_dir", required=True, help="Output directory for JSON/OWL")

    ap.add_argument("--taxonomy_table", default="taxonomy_is_a", help="Table containing is-a edges")
    ap.add_argument("--triple_table", default="non_taxonomic_edges_clean", help="Table containing (s,rel,o) triples")

    # Use term_enrichment_extension (or similar) for typing constraints
    ap.add_argument(
        "--types_table",
        default=None,
        help="Optional table containing term typing info (e.g., term_enrichment_extension with ontology_role + dul_bucket).",
    )

    # Column overrides
    ap.add_argument("--tax_child_col", default=None, help="Override taxonomy child column name")
    ap.add_argument("--tax_parent_col", default=None, help="Override taxonomy parent column name")
    ap.add_argument("--triple_subj_col", default=None, help="Override triple subject column name")
    ap.add_argument("--triple_rel_col", default=None, help="Override triple relation column name")
    ap.add_argument("--triple_obj_col", default=None, help="Override triple object column name")

    # WHERE filters
    ap.add_argument("--taxonomy_where", default=None, help="Optional SQL WHERE for taxonomy table (e.g., 'llm_accept=1')")
    ap.add_argument("--triple_where", default=None, help="Optional SQL WHERE for triple table")

    ap.add_argument("--min_support", type=int, default=10, help="Min triple count to accept domain/range")
    ap.add_argument("--min_purity", type=float, default=0.75, help="Min purity to accept domain/range")
    ap.add_argument("--evidence_k", type=int, default=5, help="How many example triples to store per relation")

    ap.add_argument("--drop_generic_relations", action="store_true",
                    help="Drop generic relation phrases from inverse mining + quality report.")

    # Inverse options
    ap.add_argument("--infer_inverses", action="store_true", help="Infer inverse properties from evidence.")
    ap.add_argument("--min_inverse_support", type=int, default=10, help="Min overlap support for inverseOf.")
    ap.add_argument("--min_inverse_ratio", type=float, default=0.60, help="Min overlap ratio for inverseOf.")

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

    # Load typing info from term_enrichment_extension if provided
    type_info: Optional[Dict[str, TermTypeInfo]] = None
    if args.types_table:
        print(f"[INFO] Loading term typing info from: {args.types_table}")
        type_info = load_term_type_info(conn, args.types_table)
        print(f"[INFO] Term typing rows loaded: {len(type_info)}")
        # This answers your question: YES, we can use ontology_role + dul_bucket to constrain axioms.
        # - ontology_role=property -> do NOT create owl:Class for that term
        # - dul_bucket=process/event/action -> do NOT create owl:Class
        # - dul_bucket=object/situation -> helps domain/range induction

    print(f"[INFO] Loading taxonomy edges from: {args.taxonomy_table}")
    if args.taxonomy_where:
        print(f"[INFO] taxonomy_where: {args.taxonomy_where}")

    tax_edges_raw = load_taxonomy_edges(
        conn,
        args.taxonomy_table,
        args.tax_child_col,
        args.tax_parent_col,
        where=args.taxonomy_where,
    )
    print(f"[INFO] Taxonomy edges loaded (raw): {len(tax_edges_raw)}")

    if not tax_edges_raw:
        raise RuntimeError(
            "No taxonomy edges loaded. "
            "If using taxonomy_is_a_validated, try: --taxonomy_where 'llm_accept=1 AND llm_best_parent_canonical_term IS NOT NULL'"
        )

    # Filter taxonomy edges (section IDs + action child + generic parent)
    tax_edges: List[Tuple[str, str]] = []
    dropped_tax = 0
    for c, p in tax_edges_raw:
        if accept_tax_edge(c, p, type_info):
            tax_edges.append((c, p))
        else:
            dropped_tax += 1
    print(f"[INFO] Taxonomy edges kept: {len(tax_edges)} ; dropped: {dropped_tax}")

    parents, _children = build_taxonomy_graph(tax_edges)
    cyc = detect_cycle(parents)
    if cyc:
        raise RuntimeError(
            "Cycle detected in taxonomy (is-a). Fix this before axiom generation.\n"
            f"Cycle example: {' -> '.join(cyc)}"
        )
    depth = compute_depths(parents)

    print(f"[INFO] Loading triples from: {args.triple_table}")
    if args.triple_where:
        print(f"[INFO] triple_where: {args.triple_where}")

    triples = load_triples(
        conn,
        args.triple_table,
        args.triple_subj_col,
        args.triple_rel_col,
        args.triple_obj_col,
        where=args.triple_where,
        normalize_relations=True,  # Normalize relation strings
    )
    print(f"[INFO] Triples loaded: {len(triples)}")

    # Phase 1: Classes + SubClassOf axioms (DO NOT add action-like terms as Classes)
    classes: Set[str] = set()
    subclass_axioms: List[SubClassOfAxiom] = []

    for c, p in tax_edges:
        if keep_as_class(c, type_info):
            classes.add(c)
        if keep_as_class(p, type_info):
            classes.add(p)
        subclass_axioms.append(SubClassOfAxiom(child=c, parent=p))

    # Include subjects/objects as classes too, but filtered
    dropped_class_terms = 0
    for s, _r, o in triples:
        if keep_as_class(s, type_info):
            classes.add(s)
        else:
            dropped_class_terms += 1
        if keep_as_class(o, type_info):
            classes.add(o)
        else:
            dropped_class_terms += 1

    if dropped_class_terms:
        print(f"[INFO] Dropped {dropped_class_terms} subject/object terms from owl:Class due to action/section/type filters.")

    # If you want DUL buckets present as Classes (optional), keep them:
    if type_info:
        for ti in type_info.values():
            if ti.dul_bucket:
                classes.add(ti.dul_bucket.strip().lower())

    # Phase 2: Properties (relations)
    properties = sorted({r for (_s, r, _o) in triples if normalize_term(r)})

    # Phase 3: Domain/Range induction
    print(f"[INFO] Computing domain/range for {len(properties)} relations...")
    dr_axioms: List[DomainRangeAxiom] = []
    for rel in properties:
        ax = compute_domain_range_for_relation(
            relation=rel,
            triples=triples,
            type_info=type_info,
            parents=parents,
            depth=depth,
            min_support=args.min_support,
            min_purity=args.min_purity,
            evidence_k=args.evidence_k,
        )
        dr_axioms.append(ax)

    # Phase 4 (optional): Inverse properties when evidence supports it
    inverse_axioms: List[InversePropertyAxiom] = []
    if args.infer_inverses:
        print("[INFO] Inferring inverse properties from evidence...")
        inverse_axioms = compute_inverse_properties(
            triples=triples,
            min_support=args.min_inverse_support,
            min_ratio=args.min_inverse_ratio,
            evidence_k=args.evidence_k,
            drop_generic_relations=args.drop_generic_relations,
        )
        print(f"[INFO] Inverse axioms inferred: {len(inverse_axioms)} (accepted only)")

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

    if args.infer_inverses:
        write_json(os.path.join(args.out_dir, "inverse_property_axioms.json"), {
            "min_inverse_support": args.min_inverse_support,
            "min_inverse_ratio": args.min_inverse_ratio,
            "accepted_count": sum(1 for a in inverse_axioms if a.status == "accepted"),
            "axioms": [asdict(a) for a in inverse_axioms],
        })

    # Quality report
    flagged = []
    for a in dr_axioms:
        if a.property.lower() in GENERIC_RELATIONS:
            flagged.append({"relation": a.property, "reason": "generic_relation_phrase"})
        if a.support < args.min_support:
            flagged.append({"relation": a.property, "reason": f"low_support<{args.min_support}", "support": a.support})
        if a.status == "candidate":
            flagged.append({"relation": a.property, "reason": "candidate_domain_range", "confidence": a.confidence})
        if type_info:
            if a.subject_type_coverage < 0.30 or a.object_type_coverage < 0.30:
                flagged.append({
                    "relation": a.property,
                    "reason": "low_type_coverage",
                    "subj_cov": a.subject_type_coverage,
                    "obj_cov": a.object_type_coverage
                })

    write_json(os.path.join(args.out_dir, "axiom_quality_report.json"), {
        "taxonomy_edges_raw": len(tax_edges_raw),
        "taxonomy_edges_kept": len(tax_edges),
        "taxonomy_edges_dropped": dropped_tax,
        "taxonomy_nodes": len(classes),
        "triple_count": len(triples),
        "relation_count": len(properties),
        "inverse_axioms_count": len(inverse_axioms) if args.infer_inverses else 0,
        "flagged_count": len(flagged),
        "flagged": flagged[:2000],
    })

    print(f"[OK] Wrote outputs to: {args.out_dir}")
    print("     - classes.json")
    print("     - properties.json")
    print("     - subclass_axioms.json")
    print("     - domain_range_axioms.json")
    if args.infer_inverses:
        print("     - inverse_property_axioms.json")
    print("     - axiom_quality_report.json")
    print("\n[DEBUG] Domain/Range Axiom Sample:")
    for ax in dr_axioms[:10]:
        print(f"  {ax.property}: domain={ax.domain}, range={ax.range}, status={ax.status}, conf={ax.confidence}")

    if args.export_owl:
        out_owl = os.path.join(args.out_dir, "ontology_lightweight.ttl")
        
        # FILTER to only export axioms with reasonable confidence
        # even if they're "candidate" status
        filtered_dr = [ax for ax in dr_axioms if ax.confidence >= 0.50 and ax.domain and ax.range]
        
        export_to_owl(
            out_path=out_owl,
            classes=classes,
            subclass_axioms=subclass_axioms,
            properties=set(properties),
            domain_range_axioms=filtered_dr,  # Use filtered list
            inverse_axioms=inverse_axioms if args.infer_inverses else None,
            base_iri=args.base_iri,
        )
    conn.close()


if __name__ == "__main__":
    main()
