"""
olaf/axiom.py (UPDATED v4)

Fixes zero non-taxonomic relations by exporting actual non-taxonomic assertions
(subject, predicate, object) in addition to schema-level property axioms.

Still includes:
- cycle handling with --break_cycles
- explicit default table/column names
- safe WHERE filters
- OWL/RDF/XSD meta-vocabulary filtering
- GraphDB-friendly Turtle export
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ============================================================
# Basic helpers
# ============================================================

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


# ============================================================
# SQLite introspection
# ============================================================

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


def prefer_column_if_exists(columns: List[str], preferred: Optional[str], fallbacks: List[str]) -> Optional[str]:
    if preferred:
        for c in columns:
            if c.lower() == preferred.lower():
                return c
    return pick_column(columns, fallbacks)


def _rewrite_where(where: Optional[str], child_col: str, parent_col: str) -> Optional[str]:
    if not where or not where.strip():
        return where
    w = where
    w = re.sub(r"\bchild\b", child_col, w, flags=re.I)
    w = re.sub(r"\bparent\b", parent_col, w, flags=re.I)
    return w


# ============================================================
# Column detection
# ============================================================

def detect_taxonomy_columns(
    conn: sqlite3.Connection,
    taxonomy_table: str,
    child_preferred: Optional[str] = None,
    parent_preferred: Optional[str] = None,
) -> Tuple[str, str]:
    cols = get_table_columns(conn, taxonomy_table)

    child_candidates = [
        "child",
        "child_canonical_term",
        "subj_canonical_term",
        "head_canonical_term",
        "head_term",
        "subject",
        "subj",
        "term",
    ]
    parent_candidates = [
        "llm_best_parent",
        "llm_best_parent_canonical_term",
        "parent",
        "parent_canonical_term",
        "obj_canonical_term",
        "tail_canonical_term",
        "object",
        "obj",
    ]

    child_col = prefer_column_if_exists(cols, child_preferred, child_candidates)
    parent_col = prefer_column_if_exists(cols, parent_preferred, parent_candidates)

    if not child_col or not parent_col:
        raise RuntimeError(
            f"Could not detect child/parent columns for taxonomy_table='{taxonomy_table}'. "
            f"Columns found: {cols}. Try passing --tax_child_col/--tax_parent_col."
        )
    return child_col, parent_col


def detect_triple_columns(
    conn: sqlite3.Connection,
    triple_table: str,
    subj_preferred: Optional[str] = None,
    rel_preferred: Optional[str] = None,
    obj_preferred: Optional[str] = None,
) -> Tuple[str, str, str]:
    cols = get_table_columns(conn, triple_table)

    subj_candidates = [
        "subj_canonical_term",
        "subj_text",
        "subj",
        "subject",
        "head",
        "head_term",
    ]
    rel_candidates = [
        "rel_key",
        "rel_text_raw",
        "rel_text",
        "predicate",
        "relation",
        "rel",
    ]
    obj_candidates = [
        "obj_canonical_term",
        "obj_text",
        "obj",
        "object",
        "tail",
        "tail_term",
    ]

    s_col = prefer_column_if_exists(cols, subj_preferred, subj_candidates)
    r_col = prefer_column_if_exists(cols, rel_preferred, rel_candidates)
    o_col = prefer_column_if_exists(cols, obj_preferred, obj_candidates)

    if not s_col or not r_col or not o_col:
        raise RuntimeError(
            f"Could not detect (subject, relation, object) columns for triple_table='{triple_table}'. "
            f"Columns found: {cols}. Try passing --triple_subj_col/--triple_rel_col/--triple_obj_col."
        )
    return s_col, r_col, o_col


def detect_enrichment_columns(
    conn: sqlite3.Connection,
    types_table: str,
    term_preferred: Optional[str] = None,
    role_preferred: Optional[str] = None,
    dul_preferred: Optional[str] = None,
    cat_preferred: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    cols = get_table_columns(conn, types_table)

    term_candidates = ["canonical_term", "term", "label", "name"]
    role_candidates = ["ontology_role", "role", "owl_role", "ont_role"]
    dul_candidates = ["dul_bucket", "dul", "bucket"]
    cat_candidates = ["category", "term_type", "type"]

    t_col = prefer_column_if_exists(cols, term_preferred, term_candidates)
    r_col = prefer_column_if_exists(cols, role_preferred, role_candidates)
    d_col = prefer_column_if_exists(cols, dul_preferred, dul_candidates)
    c_col = prefer_column_if_exists(cols, cat_preferred, cat_candidates)

    if not t_col:
        raise RuntimeError(f"Could not detect term column for types_table='{types_table}'. Columns found: {cols}")
    return t_col, r_col, d_col, c_col


# ============================================================
# Taxonomy graph utilities
# ============================================================

def build_taxonomy_graph(edges: Iterable[Tuple[str, str]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
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


# ============================================================
# Cycle breaking
# ============================================================

def _is_more_specific(a: str, b: str) -> bool:
    a0 = a.strip().lower()
    b0 = b.strip().lower()
    if a0 == b0:
        return False
    if b0 in a0 and len(a0) > len(b0):
        return True
    if len(a0) >= len(b0) + 4:
        return True
    return False


def break_taxonomy_cycles(edges: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    kept = edges[:]
    removed: List[Tuple[str, str]] = []

    while True:
        parents, _ = build_taxonomy_graph(kept)
        cyc = detect_cycle(parents)
        if not cyc:
            break

        cycle_nodes = cyc[:-1]
        if len(cycle_nodes) < 2:
            to_remove = kept[-1]
            kept.remove(to_remove)
            removed.append(to_remove)
            continue

        cycle_edges = [(cycle_nodes[i], cycle_nodes[i + 1]) for i in range(len(cycle_nodes) - 1)]
        cycle_edges.append((cycle_nodes[-1], cycle_nodes[0]))

        to_remove = None
        for c, p in cycle_edges:
            if _is_more_specific(p, c):
                to_remove = (c, p)
                break

        if to_remove is None:
            to_remove = cycle_edges[-1]

        if to_remove in kept:
            kept.remove(to_remove)
            removed.append(to_remove)
        else:
            fallback = None
            for e in cycle_edges:
                if e in kept:
                    fallback = e
                    break
            if fallback is None and kept:
                fallback = kept[-1]
            if fallback:
                kept.remove(fallback)
                removed.append(fallback)

    return kept, removed


# ============================================================
# Typing from term enrichment
# ============================================================

@dataclass
class TermTypeInfo:
    term: str
    ontology_role: Optional[str] = None
    dul_bucket: Optional[str] = None
    category: Optional[str] = None


def load_term_type_info(
    conn: sqlite3.Connection,
    types_table: str,
    term_col_pref: Optional[str] = None,
    role_col_pref: Optional[str] = None,
    dul_col_pref: Optional[str] = None,
    cat_col_pref: Optional[str] = None,
) -> Dict[str, TermTypeInfo]:
    term_col, role_col, dul_col, cat_col = detect_enrichment_columns(
        conn,
        types_table,
        term_preferred=term_col_pref,
        role_preferred=role_col_pref,
        dul_preferred=dul_col_pref,
        cat_preferred=cat_col_pref,
    )

    cols = [term_col]
    if role_col:
        cols.append(role_col)
    if dul_col and dul_col not in cols:
        cols.append(dul_col)
    if cat_col and cat_col not in cols:
        cols.append(cat_col)

    cur = conn.execute(f"SELECT {', '.join(cols)} FROM {types_table};")
    mapping: Dict[str, TermTypeInfo] = {}

    for row in cur.fetchall():
        term = normalize_term(row[0])
        if not term:
            continue
        idx = 1
        role = normalize_term(row[idx]) if role_col and idx < len(row) else None
        if role_col:
            idx += 1
        dul = normalize_term(row[idx]) if dul_col and idx < len(row) else None
        if dul_col:
            idx += 1
        cat = normalize_term(row[idx]) if cat_col and idx < len(row) else None

        mapping[term] = TermTypeInfo(term=term, ontology_role=role, dul_bucket=dul, category=cat)

    return mapping


# ============================================================
# Filtering rules
# ============================================================

VERB_PREFIXES = {
    "access", "grant", "accept", "use", "create", "start", "stop", "run",
    "set", "get", "show", "enable", "disable", "configure", "request",
    "assign", "allocate", "cancel", "submit", "launch", "kill",
}
SECTION_ID_RE = re.compile(r"^\s*(\d+(\.\d+)*[a-zA-Z]?)\b")

GENERIC_PARENTS = {"system", "information", "data", "thing", "value", "parameter", "option", "slurm", "lsf", "scheduler"}

INSTANCE_LIKE_PARENTS = {
    "slurmdbd", "slurmctld", "slurmd", "munged",
    "srun", "sbatch", "salloc", "sacct", "scontrol", "squeue", "sinfo",
}

RESERVED_OWL_TERMS = {
    "thing", "nothing", "literal",
    "resource", "class", "property",
    "objectproperty", "datatypeproperty",
    "functionalproperty", "transitiveproperty", "symmetricproperty", "asymmetricproperty", "irreflexiveproperty",
    "topdataproperty", "bottomdataproperty",
    "topobjectproperty", "bottomobjectproperty",
    "nonnegativeinteger", "integer", "decimal", "boolean", "string",
    "owl:thing", "owl:nothing",
    "rdf:property", "rdfs:class", "rdfs:resource",
    "xsd:nonnegativeinteger", "xsd:integer", "xsd:decimal", "xsd:boolean", "xsd:string",
}

DUL_BUCKET_MAP = {
    "object": "Entity",
    "situation": "Situation",
    "process": "Process",
    "event": "Event",
    "description": "Description",
    "role": "Role",
    "informationobject": "InformationObject",
}


def _term_key(term: str) -> str:
    return re.sub(r"[\s_]+", "", term.strip().lower())


def _strip_prefixes(t: str) -> str:
    s = t.strip().lower()
    s = s.replace("owl:", "").replace("rdf:", "").replace("rdfs:", "").replace("xsd:", "")
    return s


def looks_like_section_id(term: str) -> bool:
    return bool(SECTION_ID_RE.match(term.strip()))


def is_action_like(term: str) -> bool:
    t = term.strip().lower().replace("-", " ")
    parts = t.split()
    return bool(parts) and (parts[0] in VERB_PREFIXES)


def normalize_relation_key(rel_key: str) -> str:
    r = rel_key.strip().lower()
    r = r.replace("-", "_")
    r = re.sub(r"\s+", "_", r)
    r = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in r)
    r = re.sub(r"_+", "_", r).strip("_")
    return r or "related_to"


def keep_as_class(term: str, type_info: Optional[Dict[str, TermTypeInfo]] = None) -> bool:
    if not term:
        return False

    t = term.strip()
    k = _strip_prefixes(_term_key(t))
    reserved_keys = {_term_key(_strip_prefixes(x)) for x in RESERVED_OWL_TERMS}
    if k in reserved_keys:
        return False
    if looks_like_section_id(t):
        return False
    if is_action_like(t):
        return False

    if type_info and t in type_info:
        role = (type_info[t].ontology_role or "").strip().lower()
        if role in {"property", "predicate", "relation", "objectproperty", "datatypeproperty", "datatype_property"}:
            return False

    return True


def accept_tax_edge(child: str, parent: str, type_info: Optional[Dict[str, TermTypeInfo]]) -> bool:
    if not child or not parent:
        return False
    if looks_like_section_id(child) or looks_like_section_id(parent):
        return False
    if is_action_like(child):
        return False

    ck = _strip_prefixes(_term_key(child))
    pk = _strip_prefixes(_term_key(parent))
    reserved_keys = {_term_key(_strip_prefixes(x)) for x in RESERVED_OWL_TERMS}
    if ck in reserved_keys or pk in reserved_keys:
        return False

    if parent.strip().lower() in INSTANCE_LIKE_PARENTS:
        return False

    if parent.strip().lower() in GENERIC_PARENTS:
        if type_info and child in type_info:
            return True
        return False

    return True


# ============================================================
# Axiom dataclasses
# ============================================================

@dataclass
class SubClassOfAxiom:
    child: str
    parent: str
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
    property_kind: str
    datatype_iri: Optional[str] = None


# ============================================================
# Loaders
# ============================================================

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
) -> Tuple[List[Tuple[str, str]], str, str, Optional[str]]:
    child_col, parent_col = detect_taxonomy_columns(conn, taxonomy_table, child_override, parent_override)
    where2 = _rewrite_where(where, child_col, parent_col)

    cur = conn.execute(_select_sql(taxonomy_table, [child_col, parent_col], where2))
    edges: List[Tuple[str, str]] = []
    for c, p in cur.fetchall():
        c2, p2 = normalize_term(c), normalize_term(p)
        if not (c2 and p2):
            continue
        if p2.strip().lower() in {"none", "null", "unknown"}:
            continue
        if c2.strip().lower() in {"none", "null", "unknown"}:
            continue
        edges.append((c2, p2))
    return edges, child_col, parent_col, where2


def load_triples(
    conn: sqlite3.Connection,
    triple_table: str,
    subj_override: Optional[str] = None,
    rel_override: Optional[str] = None,
    obj_override: Optional[str] = None,
    where: Optional[str] = None,
) -> Tuple[List[Tuple[str, str, str]], str, str, str]:
    s_col, r_col, o_col = detect_triple_columns(conn, triple_table, subj_override, rel_override, obj_override)

    cur = conn.execute(_select_sql(triple_table, [s_col, r_col, o_col], where))
    triples: List[Tuple[str, str, str]] = []
    for s, r, o in cur.fetchall():
        s2, r2, o2 = normalize_term(s), normalize_term(r), normalize_term(o)
        if not (s2 and r2 and o2):
            continue
        r2 = normalize_relation_key(r2)
        triples.append((s2, r2, o2))
    return triples, s_col, r_col, o_col


# ============================================================
# Datatype detection + domain/range induction
# ============================================================

_LITERAL_NUM_RE = re.compile(r"^[+-]?\d+(\.\d+)?$")
_LITERAL_BOOL_RE = re.compile(r"^(true|false|yes|no|on|off)$", re.I)

DATATYPE_REL_HINTS = {
    "path", "dir", "file", "filename", "location", "port", "ip", "address",
    "time", "timeout", "duration", "interval", "date",
    "count", "num", "number", "size", "limit", "max", "min",
    "memory", "mem", "cpu", "cpus", "gpu", "gpus",
    "version", "format", "type", "mode", "state",
    "ratio", "factor", "usage", "shares", "priority",
}

XSD = {
    "string": "http://www.w3.org/2001/XMLSchema#string",
    "integer": "http://www.w3.org/2001/XMLSchema#integer",
    "decimal": "http://www.w3.org/2001/XMLSchema#decimal",
    "boolean": "http://www.w3.org/2001/XMLSchema#boolean",
}


def _guess_literal_type(obj: str) -> Optional[str]:
    s = obj.strip().strip('"').strip("'")
    if not s:
        return None
    if _LITERAL_BOOL_RE.match(s):
        return "boolean"
    if _LITERAL_NUM_RE.match(s):
        is_int = s.isdigit() or (s.startswith(("+", "-")) and s[1:].isdigit())
        return "integer" if is_int else "decimal"
    return None


def infer_bucket(term: str, type_info: Optional[Dict[str, TermTypeInfo]]) -> Optional[str]:
    if not type_info:
        return None
    ti = type_info.get(term)
    if not ti:
        return None
    if ti.dul_bucket:
        b = ti.dul_bucket.strip().lower()
        return DUL_BUCKET_MAP.get(b, b)
    return None


def infer_bucket_from_ancestors(
    term: str,
    type_info: Optional[Dict[str, TermTypeInfo]],
    parents: Dict[str, Set[str]],
    max_hops: int = 10,
) -> Optional[str]:
    direct = infer_bucket(term, type_info)
    if direct:
        return direct
    if not type_info:
        return None

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
            ty = infer_bucket(p, type_info)
            if ty:
                return ty
            q.append((p, d + 1))
    return None


def decide_property_kind_and_datatype(
    relation: str,
    rel_triples: List[Tuple[str, str, str]],
    datatype_ratio_threshold: float = 0.55,
) -> Tuple[str, Optional[str]]:
    rel_low = relation.lower()
    hint = any(h in rel_low for h in DATATYPE_REL_HINTS)

    guessed: List[str] = []
    for (_, _, o) in rel_triples:
        t = _guess_literal_type(o)
        if t:
            guessed.append(t)

    if guessed:
        hist = Counter(guessed)
        best_type, best_n = hist.most_common(1)[0]
        ratio = best_n / max(1, len(rel_triples))
        if ratio >= datatype_ratio_threshold or hint:
            return "datatype", XSD.get(best_type, XSD["string"])

    if hint and len(rel_triples) >= 2:
        return "datatype", XSD["string"]

    return "object", None


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

    prop_kind, datatype_iri = decide_property_kind_and_datatype(relation, rel_triples)

    subj_types: List[str] = []
    obj_types: List[str] = []
    typed_subjects = 0
    typed_objects = 0

    if type_info:
        for s in subjects:
            ty = infer_bucket(s, type_info) or infer_bucket_from_ancestors(s, type_info, parents)
            if ty:
                subj_types.append(ty)
                typed_subjects += 1

        if prop_kind == "object":
            for o in objects:
                ty = infer_bucket(o, type_info) or infer_bucket_from_ancestors(o, type_info, parents)
                if ty:
                    obj_types.append(ty)
                    typed_objects += 1

    subj_hist = Counter(subj_types)
    obj_hist = Counter(obj_types)

    if type_info and subj_types:
        domain = subj_hist.most_common(1)[0][0]
        purity_domain = subj_hist[domain] / len(subj_types)
        domain_method = "mode(dul_bucket_subject)"
    else:
        domain = lca(subjects, parents, depth)
        purity_domain = 1.0 if domain else 0.0
        domain_method = "lca(subject_terms)"

    if prop_kind == "datatype":
        range_ = "Literal"
        purity_range = 1.0
        range_method = "datatype_heuristic(objects)"
        typed_objects = support
    else:
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
        and purity_range >= (min_purity if prop_kind == "object" else 0.0)
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
        property_kind=prop_kind,
        datatype_iri=datatype_iri,
    )


# ============================================================
# NEW: Triple assertion filtering
# ============================================================

GENERIC_OBJECTS = {
    "number", "value", "thing", "data", "information", "message", "result",
    "request", "maximum", "minimum", "system", "option", "parameter"
}


def keep_non_taxonomic_triple(
    s: str,
    r: str,
    o: str,
    dr_by_property: Dict[str, DomainRangeAxiom],
    type_info: Optional[Dict[str, TermTypeInfo]],
) -> bool:
    if not s or not r or not o:
        return False
    if s.strip().lower() == o.strip().lower():
        return False
    if r not in dr_by_property:
        return False

    ax = dr_by_property[r]
    if ax.status != "accepted":
        return False

    if not keep_as_class(s, type_info):
        return False

    if ax.property_kind == "object":
        if o.strip().lower() in GENERIC_OBJECTS:
            return False
        if not keep_as_class(o, type_info):
            return False

    return True


# ============================================================
# OWL export
# ============================================================

def export_to_owl(
    out_path: str,
    classes: Set[str],
    subclass_axioms: List[SubClassOfAxiom],
    dr_axioms: List[DomainRangeAxiom],
    triples: List[Tuple[str, str, str]],
    type_info: Optional[Dict[str, TermTypeInfo]],
    base_iri: str,
    use_hash_iris: bool,
) -> Dict[str, int]:
    try:
        from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal
    except Exception as e:
        print("[WARN] rdflib not installed; skipping OWL export.")
        print(f"       Install with: pip install rdflib\n       Error: {e}")
        return {"exported_non_taxonomic_assertions": 0}

    g = Graph()
    NS = Namespace(base_iri)
    XSD_NS = Namespace("http://www.w3.org/2001/XMLSchema#")

    g.bind("hpc", NS)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)
    g.bind("xsd", XSD_NS)

    ONTO = URIRef(base_iri.rstrip("#/"))
    g.add((ONTO, RDF.type, OWL.Ontology))

    def pascalize(label: str) -> str:
        s = label.strip()
        if not s:
            return "Unnamed"
        parts = re.split(r"[^A-Za-z0-9]+", s)
        parts = [p for p in parts if p]
        if not parts:
            return "Unnamed"
        return "".join(p[:1].upper() + p[1:] for p in parts)

    def iri_for_class(label: str) -> URIRef:
        frag = pascalize(label)
        return NS[frag]

    def iri_for_property(rel_key: str) -> URIRef:
        frag = pascalize(rel_key.replace("_", " "))
        return NS[frag]

    def declare_class(label: str) -> URIRef:
        u = iri_for_class(label)
        g.add((u, RDF.type, OWL.Class))
        g.add((u, RDFS.label, Literal(label)))
        return u

    def declare_obj_prop(rel_key: str) -> URIRef:
        u = iri_for_property(rel_key)
        g.add((u, RDF.type, OWL.ObjectProperty))
        g.add((u, RDFS.label, Literal(rel_key)))
        return u

    def declare_data_prop(rel_key: str) -> URIRef:
        u = iri_for_property(rel_key)
        g.add((u, RDF.type, OWL.DatatypeProperty))
        g.add((u, RDFS.label, Literal(rel_key)))
        return u

    def literal_for_object(obj: str, datatype_iri: Optional[str]):
        t = _guess_literal_type(obj)
        raw = obj.strip().strip('"').strip("'")

        if datatype_iri == XSD["integer"] or t == "integer":
            try:
                return Literal(int(raw), datatype=URIRef(XSD["integer"]))
            except Exception:
                return Literal(raw, datatype=URIRef(XSD["string"]))

        if datatype_iri == XSD["decimal"] or t == "decimal":
            try:
                return Literal(float(raw), datatype=URIRef(XSD["decimal"]))
            except Exception:
                return Literal(raw, datatype=URIRef(XSD["string"]))

        if datatype_iri == XSD["boolean"] or t == "boolean":
            norm = raw.lower()
            if norm in {"true", "yes", "on", "1"}:
                return Literal(True, datatype=URIRef(XSD["boolean"]))
            if norm in {"false", "no", "off", "0"}:
                return Literal(False, datatype=URIRef(XSD["boolean"]))
            return Literal(raw, datatype=URIRef(XSD["string"]))

        return Literal(raw, datatype=URIRef(datatype_iri or XSD["string"]))

    for c in sorted(classes):
        if keep_as_class(c, None):
            declare_class(c)

    for ax in subclass_axioms:
        if ax.status != "accepted":
            continue
        if not keep_as_class(ax.child, None) or not keep_as_class(ax.parent, None):
            continue
        g.add((declare_class(ax.child), RDFS.subClassOf, declare_class(ax.parent)))

    dr_by_property: Dict[str, DomainRangeAxiom] = {ax.property: ax for ax in dr_axioms}

    for ax in dr_axioms:
        if ax.status != "accepted":
            continue

        dom = ax.domain
        ran = ax.range

        if dom and dom.strip().lower() in DUL_BUCKET_MAP:
            dom = DUL_BUCKET_MAP[dom.strip().lower()]
        if ran and ran.strip().lower() in DUL_BUCKET_MAP:
            ran = DUL_BUCKET_MAP[ran.strip().lower()]

        if ax.property_kind == "datatype":
            pu = declare_data_prop(ax.property)
            if dom and keep_as_class(dom, None):
                g.add((pu, RDFS.domain, declare_class(dom)))
            g.add((pu, RDFS.range, URIRef(ax.datatype_iri or XSD_NS.string)))
        else:
            pu = declare_obj_prop(ax.property)
            if dom and keep_as_class(dom, None):
                g.add((pu, RDFS.domain, declare_class(dom)))
            if ran and ran != "Literal" and keep_as_class(ran, None):
                g.add((pu, RDFS.range, declare_class(ran)))

    exported_non_tax_assertions = 0
    skipped_non_tax_assertions = 0

    for s, r, o in triples:
        if not keep_non_taxonomic_triple(s, r, o, dr_by_property, type_info):
            skipped_non_tax_assertions += 1
            continue

        ax = dr_by_property[r]
        su = iri_for_class(s)
        pu = iri_for_property(r)

        if ax.property_kind == "datatype":
            obj_node = literal_for_object(o, ax.datatype_iri)
            g.add((su, pu, obj_node))
            exported_non_tax_assertions += 1
        else:
            ou = iri_for_class(o)
            g.add((su, pu, ou))
            exported_non_tax_assertions += 1

    g.serialize(destination=out_path, format="turtle")
    print(f"[OK] OWL/Turtle exported to: {out_path}")
    print(f"[INFO] Exported non-taxonomic assertions: {exported_non_tax_assertions}")
    print(f"[INFO] Skipped non-taxonomic assertions: {skipped_non_tax_assertions}")

    return {
        "exported_non_taxonomic_assertions": exported_non_tax_assertions,
        "skipped_non_taxonomic_assertions": skipped_non_tax_assertions,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--db", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--taxonomy_table", default="taxonomy_is_a_final")
    ap.add_argument("--triple_table", default="non_taxonomic_edges_accept")
    ap.add_argument("--types_table", default="term_enrichment_exten")

    ap.add_argument("--tax_child_col", default="child")
    ap.add_argument("--tax_parent_col", default="llm_best_parent")

    ap.add_argument("--triple_subj_col", default="subj_canonical_term")
    ap.add_argument("--triple_rel_col", default="rel_key")
    ap.add_argument("--triple_obj_col", default="obj_canonical_term")

    ap.add_argument(
        "--taxonomy_where",
        default=(
            "child IS NOT NULL AND TRIM(child) != '' AND LOWER(child) NOT IN ('none','null','unknown') "
            "AND llm_best_parent IS NOT NULL AND TRIM(llm_best_parent) != '' AND LOWER(llm_best_parent) NOT IN ('none','null','unknown') "
            "AND (llm_accept = 1 OR LOWER(llm_accept) IN ('true','yes','accept'))"
        )
    )
    ap.add_argument(
        "--triple_where",
        default="decision IS NULL OR LOWER(decision) IN ('accept','accepted','yes','true','1')"
    )

    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--min_purity", type=float, default=0.55)
    ap.add_argument("--evidence_k", type=int, default=5)

    ap.add_argument("--export_owl", action="store_true")
    ap.add_argument("--base_iri", default="http://example.org/hpc#")
    ap.add_argument("--no_hash_iris", action="store_true")
    ap.add_argument("--break_cycles", action="store_true",
                    help="Automatically remove cycle-causing taxonomy edges instead of crashing.")

    args = ap.parse_args()

    safe_mkdir(args.out_dir)
    conn = sqlite3.connect(args.db)

    if not table_exists(conn, args.taxonomy_table):
        raise RuntimeError(f"taxonomy_table '{args.taxonomy_table}' not found in DB.")
    if not table_exists(conn, args.triple_table):
        raise RuntimeError(f"triple_table '{args.triple_table}' not found in DB.")
    if args.types_table and not table_exists(conn, args.types_table):
        raise RuntimeError(f"types_table '{args.types_table}' not found in DB.")

    type_info: Optional[Dict[str, TermTypeInfo]] = None
    if args.types_table:
        print(f"[INFO] Loading types from: {args.types_table}")
        type_info = load_term_type_info(conn, args.types_table)
        print(f"[INFO] Types loaded: {len(type_info)}")

    print(f"[INFO] Loading taxonomy edges from: {args.taxonomy_table}")
    tax_edges_raw, tax_child_col, tax_parent_col, tax_where_used = load_taxonomy_edges(
        conn,
        args.taxonomy_table,
        child_override=args.tax_child_col,
        parent_override=args.tax_parent_col,
        where=args.taxonomy_where,
    )
    print(f"[INFO] Using taxonomy columns: child='{tax_child_col}', parent='{tax_parent_col}'")
    if tax_where_used:
        print(f"[INFO] taxonomy_where (effective): {tax_where_used}")
    print(f"[INFO] Tax edges raw: {len(tax_edges_raw)}")

    tax_edges: List[Tuple[str, str]] = []
    dropped_tax = 0
    for c, p in tax_edges_raw:
        if accept_tax_edge(c, p, type_info):
            tax_edges.append((c, p))
        else:
            dropped_tax += 1
    print(f"[INFO] Tax edges kept: {len(tax_edges)} ; dropped: {dropped_tax}")

    if not tax_edges:
        raise RuntimeError(
            "No taxonomy edges kept. Likely causes:\n"
            "1) Wrong table\n"
            "2) Wrong columns\n"
            "3) taxonomy_where removed everything\n"
            "Fix: set --tax_child_col/--tax_parent_col and adjust --taxonomy_where."
        )

    if args.break_cycles:
        tax_edges, removed_cycle_edges = break_taxonomy_cycles(tax_edges)
        if removed_cycle_edges:
            print(f"[WARN] Removed {len(removed_cycle_edges)} cycle-causing taxonomy edges.")
            write_json(os.path.join(args.out_dir, "removed_taxonomy_cycle_edges.json"), removed_cycle_edges)

    parents, _children = build_taxonomy_graph(tax_edges)
    cyc = detect_cycle(parents)
    if cyc:
        raise RuntimeError("Cycle detected in taxonomy: " + " -> ".join(cyc))
    depth = compute_depths(parents)

    print(f"[INFO] Loading triples from: {args.triple_table}")
    triples, s_col, r_col, o_col = load_triples(
        conn,
        args.triple_table,
        subj_override=args.triple_subj_col,
        rel_override=args.triple_rel_col,
        obj_override=args.triple_obj_col,
        where=args.triple_where,
    )
    print(f"[INFO] Using triple columns: subj='{s_col}', rel='{r_col}', obj='{o_col}'")
    if args.triple_where:
        print(f"[INFO] triple_where: {args.triple_where}")
    print(f"[INFO] Triples loaded: {len(triples)}")

    classes: Set[str] = set()
    subclass_axioms: List[SubClassOfAxiom] = []

    for c, p in tax_edges:
        if keep_as_class(c, type_info):
            classes.add(c)
        if keep_as_class(p, type_info):
            classes.add(p)
        subclass_axioms.append(SubClassOfAxiom(child=c, parent=p))

    dropped = 0
    for s, _r, o in triples:
        if keep_as_class(s, type_info):
            classes.add(s)
        else:
            dropped += 1
        if keep_as_class(o, type_info):
            classes.add(o)
        else:
            dropped += 1
    if dropped:
        print(f"[INFO] Dropped {dropped} subject/object terms from owl:Class due to filters/meta-vocab/literals.")

    for upper in set(DUL_BUCKET_MAP.values()):
        classes.add(upper)

    properties = sorted({r for (_s, r, _o) in triples if r})
    print(f"[INFO] Computing domain/range axioms for {len(properties)} relations...")

    dr_axioms: List[DomainRangeAxiom] = []
    for rel in properties:
        dr_axioms.append(compute_domain_range_for_relation(
            relation=rel,
            triples=triples,
            type_info=type_info,
            parents=parents,
            depth=depth,
            min_support=args.min_support,
            min_purity=args.min_purity,
            evidence_k=args.evidence_k,
        ))

    accepted_dr = [a for a in dr_axioms if a.status == "accepted"]
    print(f"[INFO] Domain/Range accepted: {len(accepted_dr)} ; candidate: {len(dr_axioms) - len(accepted_dr)}")

    write_json(os.path.join(args.out_dir, "classes.json"),
               {"count": len(classes), "classes": sorted(classes)})
    write_json(os.path.join(args.out_dir, "subclass_axioms.json"),
               {"count": len(subclass_axioms), "axioms": [asdict(a) for a in subclass_axioms]})
    write_json(os.path.join(args.out_dir, "domain_range_axioms.json"), {
        "min_support": args.min_support,
        "min_purity": args.min_purity,
        "accepted_count": len(accepted_dr),
        "candidate_count": len(dr_axioms) - len(accepted_dr),
        "axioms": [asdict(a) for a in dr_axioms],
    })

    if args.export_owl:
        out_ttl = os.path.join(args.out_dir, "hpc_ontology.ttl")
        export_stats = export_to_owl(
            out_path=out_ttl,
            classes=classes,
            subclass_axioms=subclass_axioms,
            dr_axioms=dr_axioms,
            triples=triples,
            type_info=type_info,
            base_iri=args.base_iri,
            use_hash_iris=(not args.no_hash_iris),
        )
        write_json(os.path.join(args.out_dir, "export_stats.json"), export_stats)

    conn.close()
    print(f"[OK] Done. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()