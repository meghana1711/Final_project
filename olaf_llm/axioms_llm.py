from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# -----------------------------
# Small utilities
# -----------------------------

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def norm(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def snake_ok(s: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_]{2,80}", s)) and any(c.isalpha() for c in s)


def to_safe_localname(s: str) -> str:
    """
    Convert a label/canonical into an IRI-safe localname.
    Predictable + stable. Collision handled separately.
    """
    t = (s or "").strip()
    if not t:
        return "thing"
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "thing"
    if t[0].isdigit():
        t = "t_" + t
    return t[:80]


def ttl_escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


def brace_match_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue

        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i + 1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml  # type: ignore
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML config must be a mapping. Got: {type(data)}")
    return data


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class EvidenceRow:
    subj: str
    pred: str
    obj: str
    doc_id: Optional[str]
    chunk_id: Optional[str]
    relation_type: Optional[str]
    justification: Optional[str]


@dataclass
class TermInfo:
    term_id: int
    term: str
    canonical: str
    scheduler: str
    ontology_role: str
    category: str
    dul_bucket: str
    is_hpc_domain: int
    definition: str
    freq_total: int


# -----------------------------
# LLM (transformers only)
# -----------------------------

class TransformersClient:
    """
    Local HF transformers backend.
    pip install transformers accelerate torch
    """
    def __init__(self, model_path: str, max_new_tokens: int = 700):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError(
                "Missing transformers/torch. Install with: pip install transformers accelerate torch"
            ) from e

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.max_new_tokens = max_new_tokens

    def complete(self, system: str, user: str) -> str:
        prompt = f"### System\n{system}\n\n### User\n{user}\n\n### Assistant\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("### Assistant", 1)[-1].strip()


# -----------------------------
# DB helpers
# -----------------------------

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (name,),
    ).fetchone()
    return row is not None


def init_axiom_tables(conn: sqlite3.Connection, runs_table: str, out_table: str) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {runs_table} (
            predicate      TEXT PRIMARY KEY,
            status         TEXT NOT NULL, -- 'done' | 'skipped' | 'error'
            processed_at   TEXT,
            error          TEXT
        );
        """
    )
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            predicate      TEXT PRIMARY KEY,
            axiom_json     TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );
        """
    )
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{runs_table}_status ON {runs_table}(status);")
    conn.commit()


def mark_run(conn: sqlite3.Connection, runs_table: str, pred: str, status: str, error: str = "") -> None:
    conn.execute(
        f"""
        INSERT INTO {runs_table}(predicate, status, processed_at, error)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(predicate) DO UPDATE SET
            status=excluded.status,
            processed_at=excluded.processed_at,
            error=excluded.error;
        """,
        (pred, status, now_iso(), error[:2000]),
    )
    conn.commit()


def store_axiom_json(conn: sqlite3.Connection, out_table: str, pred: str, obj: Dict[str, Any]) -> None:
    conn.execute(
        f"""
        INSERT INTO {out_table}(predicate, axiom_json, created_at)
        VALUES(?, ?, ?)
        ON CONFLICT(predicate) DO UPDATE SET
            axiom_json=excluded.axiom_json,
            created_at=excluded.created_at;
        """,
        (pred, json.dumps(obj, ensure_ascii=False), now_iso()),
    )
    conn.commit()


# -----------------------------
# Load term enrichment -> vocab + role maps
# -----------------------------

def load_term_info(
    conn: sqlite3.Connection,
    enrich_table: str,
    *,
    canonical_col: str = "canonical",
    term_col: str = "term",
    role_col: str = "ontology_role",
    is_domain_col: str = "is_hpc_domain",
    dul_col: str = "dul_bucket",
    category_col: str = "category",
    scheduler_col: str = "scheduler",
    definition_col: str = "definition",
    freq_col: str = "freq_total",
    id_col: str = "term_id",
) -> Tuple[Dict[str, TermInfo], Dict[str, str], Set[str]]:
    q = f"""
    SELECT
      {id_col}, {term_col}, {canonical_col},
      COALESCE({scheduler_col}, 'unknown'),
      COALESCE({role_col}, 'unknown'),
      COALESCE({category_col}, 'other_hpc'),
      COALESCE({dul_col}, 'unknown'),
      COALESCE({is_domain_col}, 1),
      COALESCE({definition_col}, ''),
      COALESCE({freq_col}, 0)
    FROM {enrich_table};
    """
    canonical_map: Dict[str, TermInfo] = {}
    surface_to_canonical: Dict[str, str] = {}
    class_set: Set[str] = set()

    rows = conn.execute(q).fetchall()
    for (term_id, term, canonical, scheduler, role, category, dul, is_dom, definition, freq_total) in rows:
        t = norm(term) or ""
        c = (norm(canonical) or t).strip().lower()
        if not c:
            continue
        info = TermInfo(
            term_id=int(term_id) if term_id is not None else -1,
            term=t,
            canonical=c,
            scheduler=str(scheduler or "unknown"),
            ontology_role=str(role or "unknown"),
            category=str(category or "other_hpc"),
            dul_bucket=str(dul or "unknown"),
            is_hpc_domain=int(is_dom or 0),
            definition=str(definition or ""),
            freq_total=int(freq_total or 0),
        )
        canonical_map[c] = info
        if t:
            surface_to_canonical[t.strip().lower()] = c

        if info.is_hpc_domain == 1 and info.ontology_role == "class":
            class_set.add(c)

    return canonical_map, surface_to_canonical, class_set


def resolve_to_canonical(
    s: str,
    canonical_map: Dict[str, TermInfo],
    surface_to_canonical: Dict[str, str],
) -> str:
    t = (s or "").strip()
    if not t:
        return ""
    key = t.lower()
    if key in canonical_map:
        return key
    if key in surface_to_canonical:
        return surface_to_canonical[key]
    return key


# -----------------------------
# Taxonomy parents + export
# -----------------------------

def load_parents(
    conn: sqlite3.Connection,
    taxonomy_table: str,
    child_col: str,
    parent_col: str,
    canonical_map: Dict[str, TermInfo],
    surface_to_canonical: Dict[str, str],
) -> Dict[str, List[str]]:
    q = f"SELECT {child_col}, {parent_col} FROM {taxonomy_table};"
    parents: Dict[str, List[str]] = {}
    for c, p in conn.execute(q).fetchall():
        c2 = resolve_to_canonical(norm(c) or "", canonical_map, surface_to_canonical)
        p2 = resolve_to_canonical(norm(p) or "", canonical_map, surface_to_canonical)
        if not c2 or not p2 or c2 == p2:
            continue
        parents.setdefault(c2, []).append(p2)

    for k, vals in list(parents.items()):
        seen = set()
        out = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        parents[k] = out
    return parents


def load_taxonomy_edges_terms(
    conn: sqlite3.Connection,
    taxonomy_table: str,
    child_col: str,
    parent_col: str,
    canonical_map: Dict[str, TermInfo],
    surface_to_canonical: Dict[str, str],
) -> List[Tuple[str, str]]:
    q = f"SELECT {child_col}, {parent_col} FROM {taxonomy_table};"
    out: List[Tuple[str, str]] = []
    for c, p in conn.execute(q).fetchall():
        c2 = resolve_to_canonical(norm(c) or "", canonical_map, surface_to_canonical).strip().lower()
        p2 = resolve_to_canonical(norm(p) or "", canonical_map, surface_to_canonical).strip().lower()
        if not c2 or not p2 or c2 == p2:
            continue
        out.append((c2, p2))
    return out


def ancestors_bfs(term: str, parents: Dict[str, List[str]], max_hops: int = 4, max_out: int = 12) -> List[str]:
    seen = set([term])
    q: List[Tuple[str, int]] = [(term, 0)]
    out: List[str] = []
    while q:
        node, d = q.pop(0)
        if d >= max_hops:
            continue
        for p in parents.get(node, []):
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            q.append((p, d + 1))
            if len(out) >= max_out:
                return out
    return out


# -----------------------------
# Class/predicate filtering
# -----------------------------

BANNED_CLASS_KEYS = {
    "owl:thing", "owl:nothing",
    "thing", "nothing",
    "class",
    "entity", "unknown", "informationobject", "information_object",
    "action", "ability", "resource", "literal",
    "--nodes", "--clusters",
}

VERB_STOP = {
    "add", "get", "set", "run", "use", "make", "create", "apply", "allocate", "bind",
    "submit", "request", "show", "list", "print", "see", "do", "does", "did",
    "start", "stop", "enable", "disable", "configure", "register", "save", "load",
    "read", "write", "execute", "launch", "kill", "cancel", "take", "have",
}

PREP_SUFFIXES = {"to", "from", "in", "on", "with", "for", "into", "via", "as", "by"}

BANNED_PREDICATES = {
    "related_to", "associated_with",
    "has", "have", "do", "does", "did",
    "make", "made", "create", "created",
    "add", "add_to", "use", "uses",
    "set", "sets", "enable", "enabled", "disable", "disabled",
}

GENERIC_OBJECTS = {
    "number", "value", "thing", "data", "information", "message", "result",
    "request", "maximum", "minimum", "system", "option", "parameter"
}


def base_verb_of_pred(pred: str) -> str:
    p = (pred or "").strip().lower()
    return p.split("_", 1)[0] if p else ""


def has_prep_suffix(pred: str) -> bool:
    p = (pred or "").strip().lower()
    parts = p.split("_")
    return bool(len(parts) >= 2 and parts[-1] in PREP_SUFFIXES)


# -----------------------------
# Evidence + predicate inventory
# -----------------------------

def predicate_counts(conn: sqlite3.Connection, non_tax_table: str, pred_col: str) -> Dict[str, int]:
    q = f"SELECT {pred_col}, COUNT(*) FROM {non_tax_table} GROUP BY {pred_col};"
    out: Dict[str, int] = {}
    for pred, cnt in conn.execute(q).fetchall():
        p = norm(pred)
        if p:
            out[p] = int(cnt or 0)
    return out


def load_predicates(conn: sqlite3.Connection, non_tax_table: str, pred_col: str) -> List[str]:
    q = f"SELECT DISTINCT {pred_col} FROM {non_tax_table} ORDER BY {pred_col};"
    preds = []
    for (p,) in conn.execute(q).fetchall():
        p2 = norm(p)
        if p2:
            preds.append(p2)
    return preds


def load_evidence_for_predicate(
    conn: sqlite3.Connection,
    non_tax_table: str,
    subj_col: str,
    pred_col: str,
    obj_col: str,
    doc_col: str,
    chunk_col: str,
    reltype_col: str,
    just_col: str,
    predicate: str,
    k: int,
    *,
    random_seed: int = 13,
) -> List[EvidenceRow]:
    rng = random.Random(random_seed)
    pool = max(k * 6, 80)

    q = f"""
    SELECT {subj_col}, {pred_col}, {obj_col}, {doc_col}, {chunk_col}, {reltype_col}, {just_col}
    FROM {non_tax_table}
    WHERE {pred_col} = ?
    LIMIT ?;
    """
    rows = conn.execute(q, (predicate, pool)).fetchall()
    if not rows:
        return []

    rows_list = list(rows)
    rng.shuffle(rows_list)

    out: List[EvidenceRow] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    for s, p, o, doc, chunk, rt, just in rows_list:
        s0 = norm(s) or ""
        o0 = norm(o) or ""
        pair = (s0.strip().lower(), o0.strip().lower())
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        out.append(EvidenceRow(
            subj=s0,
            pred=norm(p) or "",
            obj=o0,
            doc_id=norm(doc),
            chunk_id=norm(chunk),
            relation_type=norm(rt),
            justification=norm(just),
        ))
        if len(out) >= k:
            break

    return out


def load_assertion_triples(
    conn: sqlite3.Connection,
    non_tax_table: str,
    subj_col: str,
    pred_col: str,
    obj_col: str,
    *,
    allowed_predicates: Set[str],
) -> List[Tuple[str, str, str]]:
    q = f"""
    SELECT {subj_col}, {pred_col}, {obj_col}
    FROM {non_tax_table};
    """
    out: List[Tuple[str, str, str]] = []
    for s, p, o in conn.execute(q).fetchall():
        s0 = norm(s) or ""
        p0 = (norm(p) or "").strip().lower()
        o0 = norm(o) or ""
        if not s0 or not p0 or not o0:
            continue
        if p0 not in allowed_predicates:
            continue
        out.append((s0, p0, o0))
    return out


def build_evidence_block(evs: List[EvidenceRow], max_chars: int = 4500) -> str:
    lines = []
    for e in evs:
        just = (e.justification or "").replace("\n", " ").strip()
        just = just[:260]
        lines.append(
            f'- ({e.subj}) {e.pred} ({e.obj}) | doc={e.doc_id} chunk={e.chunk_id} type={e.relation_type} | just="{just}"'
        )
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n... (truncated)"
    return block


# -----------------------------
# Output validation
# -----------------------------

def validate_axiom_json(obj: Dict[str, Any]) -> Tuple[bool, str]:
    need = ["predicate", "property", "domain_class", "range_class",
            "subPropertyOf", "inverseOf", "confidence", "rationale", "evidence"]
    for k in need:
        if k not in obj:
            return False, f"Missing key: {k}"
    if not isinstance(obj["predicate"], str):
        return False, "predicate must be string"
    if not isinstance(obj["property"], dict):
        return False, "property must be object"
    prop = obj["property"]
    for k in ["name", "label", "kind"]:
        if k not in prop or not isinstance(prop[k], str):
            return False, f"property.{k} must be string"
    if prop["kind"] not in ("ObjectProperty", "DatatypeProperty"):
        return False, "property.kind must be ObjectProperty|DatatypeProperty"
    if not isinstance(obj["domain_class"], str) or not isinstance(obj["range_class"], str):
        return False, "domain_class and range_class must be strings"
    if not isinstance(obj["subPropertyOf"], list) or not isinstance(obj["inverseOf"], list):
        return False, "subPropertyOf and inverseOf must be lists"
    if not isinstance(obj["confidence"], (int, float)):
        return False, "confidence must be number"
    if not isinstance(obj["rationale"], str):
        return False, "rationale must be string"
    if not isinstance(obj["evidence"], list):
        return False, "evidence must be list"
    return True, "ok"


# -----------------------------
# Datatype inference
# -----------------------------

_LITERAL_NUM_RE = re.compile(r'^[+-]?\d+(\.\d+)?$')
_LITERAL_BOOL_RE = re.compile(r'^(true|false|yes|no|on|off)$', re.I)

DATATYPE_HINTS = {
    "id", "uid", "gid", "pid", "port",
    "path", "dir", "file", "filename", "location", "address", "ip",
    "time", "timeout", "duration", "interval", "date",
    "count", "num", "number", "size", "limit", "max", "min",
    "memory", "mem", "cpu", "cpus", "gpu", "gpus",
    "version", "format", "mode", "state", "status",
    "priority", "ratio", "factor", "usage", "shares",
}

XSD_IRI = {
    "string": "http://www.w3.org/2001/XMLSchema#string",
    "integer": "http://www.w3.org/2001/XMLSchema#integer",
    "decimal": "http://www.w3.org/2001/XMLSchema#decimal",
    "boolean": "http://www.w3.org/2001/XMLSchema#boolean",
}


def _guess_literal_type(s: str) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().strip('"').strip("'")
    if not t:
        return None
    if _LITERAL_BOOL_RE.match(t):
        return "boolean"
    if _LITERAL_NUM_RE.match(t):
        is_int = t.isdigit() or (t.startswith(("+", "-")) and t[1:].isdigit())
        return "integer" if is_int else "decimal"
    if "/" in t or "\\" in t:
        return "string"
    return None


def decide_property_kind_from_evidence(predicate: str, evidence: List[EvidenceRow]) -> Tuple[str, Optional[str]]:
    pred_low = (predicate or "").lower()
    hint = any(h in pred_low for h in DATATYPE_HINTS)

    guesses: List[str] = []
    for e in evidence:
        g = _guess_literal_type(e.obj)
        if g:
            guesses.append(g)

    if guesses:
        hist: Dict[str, int] = {}
        for g in guesses:
            hist[g] = hist.get(g, 0) + 1
        best = max(hist.items(), key=lambda kv: kv[1])[0]
        ratio = hist[best] / max(1, len(evidence))
        if ratio >= 0.55 or hint:
            return "DatatypeProperty", XSD_IRI.get(best, XSD_IRI["string"])

    if hint:
        return "DatatypeProperty", XSD_IRI["string"]

    return "ObjectProperty", None


# -----------------------------
# Prompting
# -----------------------------

def build_prompts_from_yaml(
    yaml_cfg: Dict[str, Any],
    *,
    predicate: str,
    evidence: List[EvidenceRow],
    subj_ancestors: Dict[str, List[str]],
    obj_ancestors: Dict[str, List[str]],
) -> Tuple[str, str]:
    system = yaml_cfg.get("system_prompt")
    user_template = yaml_cfg.get("user_template")
    if not isinstance(system, str) or not system.strip():
        raise RuntimeError("YAML missing system_prompt (string).")
    if not isinstance(user_template, str) or not user_template.strip():
        raise RuntimeError("YAML missing user_template (string).")

    user = user_template.format(
        predicate=predicate,
        evidence_block=build_evidence_block(evidence),
        subj_ancestors_json=json.dumps(subj_ancestors, ensure_ascii=False),
        obj_ancestors_json=json.dumps(obj_ancestors, ensure_ascii=False),
    )
    return system, user


def llm_axioms_for_predicate(
    client: TransformersClient,
    yaml_cfg: Dict[str, Any],
    predicate: str,
    evidence: List[EvidenceRow],
    parents: Dict[str, List[str]],
    canonical_map: Dict[str, TermInfo],
    surface_to_canonical: Dict[str, str],
    *,
    max_retries: int = 2,
) -> Dict[str, Any]:
    subj_terms = [resolve_to_canonical(e.subj, canonical_map, surface_to_canonical) for e in evidence][:6]
    obj_terms = [resolve_to_canonical(e.obj, canonical_map, surface_to_canonical) for e in evidence][:6]
    subj_anc = {t: ancestors_bfs(t, parents, max_hops=3, max_out=8) for t in subj_terms if t}
    obj_anc = {t: ancestors_bfs(t, parents, max_hops=3, max_out=8) for t in obj_terms if t}

    system, user = build_prompts_from_yaml(
        yaml_cfg,
        predicate=predicate,
        evidence=evidence,
        subj_ancestors=subj_anc,
        obj_ancestors=obj_anc,
    )

    last = ""
    for _ in range(max_retries + 1):
        text = client.complete(system=system, user=user)
        last = text
        obj = brace_match_first_json_object(text)
        if obj is None:
            user += "\n\nYour response was not valid JSON. Return ONLY one valid JSON object."
            continue
        ok, msg = validate_axiom_json(obj)
        if not ok:
            user += f"\n\nYour JSON did not match schema: {msg}. Return corrected JSON ONLY."
            continue

        obj["predicate"] = predicate
        obj.setdefault("created_at", now_iso())
        obj.setdefault("raw_output", last)
        return obj

    return {
        "predicate": predicate,
        "property": {"name": predicate, "label": predicate, "kind": "ObjectProperty"},
        "domain_class": "owl:Thing",
        "range_class": "owl:Thing",
        "datatype_iri": None,
        "subPropertyOf": [],
        "inverseOf": [],
        "confidence": 0.0,
        "rationale": "LLM failed to produce valid JSON after retries; using owl:Thing fallback.",
        "evidence": [
            {
                "subj": e.subj,
                "obj": e.obj,
                "doc_id": e.doc_id,
                "chunk_id": e.chunk_id,
                "justification": e.justification,
            } for e in evidence[:3]
        ],
        "error": "invalid_json",
        "raw_output": last,
        "created_at": now_iso(),
    }


def clamp_to_class(
    term_canonical: str,
    canonical_map: Dict[str, TermInfo],
    class_set: Set[str],
    parents: Dict[str, List[str]],
) -> Optional[str]:
    if not term_canonical:
        return None
    key = term_canonical.strip().lower()
    if key in BANNED_CLASS_KEYS:
        return None

    info = canonical_map.get(key)
    if info and info.ontology_role == "class" and info.is_hpc_domain == 1:
        return key

    for a in ancestors_bfs(key, parents, max_hops=4, max_out=20):
        akey = a.strip().lower()
        if akey in BANNED_CLASS_KEYS:
            continue
        ai = canonical_map.get(akey)
        if ai and ai.ontology_role == "class" and ai.is_hpc_domain == 1:
            return akey
        if akey in class_set:
            return akey
    return None


def keep_llm_assertion_triple(
    subj: str,
    pred: str,
    obj: str,
    predicate_axiom_map: Dict[str, Dict[str, Any]],
    canonical_map: Dict[str, TermInfo],
    surface_to_canonical: Dict[str, str],
    class_set: Set[str],
) -> bool:
    s = (subj or "").strip()
    p = (pred or "").strip().lower()
    o = (obj or "").strip()

    if not s or not p or not o:
        return False
    if s.lower() == o.lower():
        return False
    if p not in predicate_axiom_map:
        return False

    ax = predicate_axiom_map[p]
    prop = ax.get("property", {}) if isinstance(ax.get("property"), dict) else {}
    kind = str(prop.get("kind") or "ObjectProperty").strip()

    s_can = resolve_to_canonical(s, canonical_map, surface_to_canonical)
    if s_can not in class_set:
        return False

    if kind == "ObjectProperty":
        o_can = resolve_to_canonical(o, canonical_map, surface_to_canonical)
        if o_can not in class_set:
            return False
        if o_can in GENERIC_OBJECTS:
            return False

    return True


# -----------------------------
# TTL export (DEFAULT GRAPH)
#   - Omits domain/range when unknown (no owl:Thing hubs)
#   - Supports DatatypeProperty with xsd ranges
#   - Writes rdfs:subClassOf edges
#   - Writes actual non-taxonomic assertions
#   - Collision-safe localnames
# -----------------------------

def _allocate_localname(label: str, used: Dict[str, str]) -> str:
    base = to_safe_localname(label)
    if base not in used:
        used[base] = label
        return base
    if used[base] == label:
        return base
    i = 2
    while True:
        cand = f"{base}_{i}"
        if cand not in used:
            used[cand] = label
            return cand
        if used[cand] == label:
            return cand
        i += 1


def write_ttl(
    out_path: str,
    *,
    base_iri: str,
    class_terms: Sequence[str],
    canonical_map: Dict[str, TermInfo],
    surface_to_canonical: Dict[str, str],
    taxonomy_edges: Sequence[Tuple[str, str]],
    predicate_axioms: Sequence[Dict[str, Any]],
    assertion_triples: Sequence[Tuple[str, str, str]],
    known_predicates: Set[str],
) -> None:
    if not (base_iri.endswith("#") or base_iri.endswith("/")):
        base_iri = base_iri + "#"

    used_locals: Dict[str, str] = {}

    def iri_class(term: str) -> str:
        return _allocate_localname(term, used_locals)

    def iri_prop(term: str) -> str:
        return _allocate_localname(term, used_locals)

    lines: List[str] = []
    lines.append(f"@prefix hpc: <{base_iri}> .")
    lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    lines.append("@prefix owl: <http://www.w3.org/2002/07/owl#> .")
    lines.append("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .")
    lines.append("")
    lines.append("### Classes")
    for c in sorted(set(class_terms)):
        ck = (c or "").strip().lower()
        if not ck or ck in BANNED_CLASS_KEYS:
            continue
        info = canonical_map.get(ck)
        label = info.term if info and info.term else ck
        local = iri_class(ck)
        lines.append(f"hpc:{local} a owl:Class ;")
        lines.append(f'  rdfs:label "{ttl_escape(label)}" .')
        lines.append("")

    lines.append("### Taxonomy (rdfs:subClassOf)")
    class_set_norm = {c.strip().lower() for c in class_terms if c and c.strip().lower() not in BANNED_CLASS_KEYS}
    seen_sc: Set[Tuple[str, str]] = set()
    for child, parent in taxonomy_edges:
        c = (child or "").strip().lower()
        p = (parent or "").strip().lower()
        if not c or not p or c == p:
            continue
        if c in BANNED_CLASS_KEYS or p in BANNED_CLASS_KEYS:
            continue
        if c not in class_set_norm or p not in class_set_norm:
            continue
        key = (c, p)
        if key in seen_sc:
            continue
        seen_sc.add(key)
        lines.append(f"hpc:{iri_class(c)} rdfs:subClassOf hpc:{iri_class(p)} .")
    lines.append("")

    lines.append("### Properties (axioms)")
    for ax in predicate_axioms:
        pred = str(ax.get("predicate") or "").strip().lower()
        if not pred:
            continue

        prop = ax.get("property", {}) if isinstance(ax.get("property"), dict) else {}
        pname = str(prop.get("name") or pred).strip().lower()
        plabel = str(prop.get("label") or pred).strip()
        kind = str(prop.get("kind") or "ObjectProperty").strip()
        dt_iri = ax.get("datatype_iri")

        local_p = iri_prop(pname)
        rdf_kind = "owl:ObjectProperty" if kind == "ObjectProperty" else "owl:DatatypeProperty"
        lines.append(f"hpc:{local_p} a {rdf_kind} ;")
        lines.append(f'  rdfs:label "{ttl_escape(plabel)}"')

        dom = str(ax.get("domain_class") or "").strip()
        rng = str(ax.get("range_class") or "").strip()
        dom_k = dom.strip().lower()
        rng_k = rng.strip().lower()

        if dom and dom != "owl:Thing" and dom_k not in BANNED_CLASS_KEYS:
            if dom_k in class_set_norm:
                lines.append(f"  ; rdfs:domain hpc:{iri_class(dom_k)}")

        if kind == "DatatypeProperty":
            if isinstance(dt_iri, str) and dt_iri.startswith("http://www.w3.org/2001/XMLSchema#"):
                dt_local = dt_iri.rsplit("#", 1)[-1]
                lines.append(f"  ; rdfs:range xsd:{dt_local}")
            else:
                lines.append("  ; rdfs:range xsd:string")
        else:
            if rng and rng != "owl:Thing" and rng_k not in BANNED_CLASS_KEYS:
                if rng_k in class_set_norm:
                    lines.append(f"  ; rdfs:range hpc:{iri_class(rng_k)}")

        lines.append(" .")

        subs = ax.get("subPropertyOf", [])
        invs = ax.get("inverseOf", [])
        if isinstance(subs, list):
            good = [s for s in subs if isinstance(s, str) and s.strip().lower() in known_predicates]
            for s in good:
                lines.append(f"hpc:{local_p} rdfs:subPropertyOf hpc:{iri_prop(s.strip().lower())} .")
        if isinstance(invs, list):
            good = [s for s in invs if isinstance(s, str) and s.strip().lower() in known_predicates]
            for inv in good:
                lines.append(f"hpc:{local_p} owl:inverseOf hpc:{iri_prop(inv.strip().lower())} .")

        lines.append("")

    lines.append("### Non-taxonomic assertions")
    predicate_axiom_map = {
        str(ax.get("predicate") or "").strip().lower(): ax
        for ax in predicate_axioms
        if str(ax.get("predicate") or "").strip()
    }

    seen_assertions: Set[Tuple[str, str, str]] = set()

    for subj, pred, obj in assertion_triples:
        p = pred.strip().lower()
        if p not in predicate_axiom_map:
            continue

        ax = predicate_axiom_map[p]
        prop = ax.get("property", {}) if isinstance(ax.get("property"), dict) else {}
        kind = str(prop.get("kind") or "ObjectProperty").strip()

        s_can = resolve_to_canonical(subj, canonical_map, surface_to_canonical)
        if s_can not in class_set_norm:
            continue

        local_s = iri_class(s_can)
        local_p = iri_prop(p)

        if kind == "DatatypeProperty":
            lit_type = _guess_literal_type(obj)
            raw = obj.strip().strip('"').strip("'")
            key = (s_can, p, raw)
            if key in seen_assertions:
                continue
            seen_assertions.add(key)

            if lit_type == "integer":
                lines.append(f'hpc:{local_s} hpc:{local_p} "{raw}"^^xsd:integer .')
            elif lit_type == "decimal":
                lines.append(f'hpc:{local_s} hpc:{local_p} "{raw}"^^xsd:decimal .')
            elif lit_type == "boolean":
                norm_bool = raw.lower()
                if norm_bool in {"true", "yes", "on", "1"}:
                    lines.append(f'hpc:{local_s} hpc:{local_p} "true"^^xsd:boolean .')
                elif norm_bool in {"false", "no", "off", "0"}:
                    lines.append(f'hpc:{local_s} hpc:{local_p} "false"^^xsd:boolean .')
                else:
                    lines.append(f'hpc:{local_s} hpc:{local_p} "{ttl_escape(raw)}"^^xsd:string .')
            else:
                lines.append(f'hpc:{local_s} hpc:{local_p} "{ttl_escape(raw)}"^^xsd:string .')
        else:
            o_can = resolve_to_canonical(obj, canonical_map, surface_to_canonical)
            if o_can not in class_set_norm:
                continue
            key = (s_can, p, o_can)
            if key in seen_assertions:
                continue
            seen_assertions.add(key)
            local_o = iri_class(o_can)
            lines.append(f"hpc:{local_s} hpc:{local_p} hpc:{local_o} .")

    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate TBox + non-tax assertions (TTL) using LOCAL transformers.")

    ap.add_argument("--db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_iri", default="http://example.org/hpc#")

    ap.add_argument("--enrich_table", default="llm_enrich_final")
    ap.add_argument("--taxonomy_table", default="llm_is_a_edges")
    ap.add_argument("--non_tax_table", default="llm_non_taxonomy_edges")

    ap.add_argument("--tax_child_col", default="child_term")
    ap.add_argument("--tax_parent_col", default="parent_term")

    ap.add_argument("--subj_col", default="subject")
    ap.add_argument("--pred_col", default="predicate")
    ap.add_argument("--obj_col", default="object")
    ap.add_argument("--doc_col", default="doc_id")
    ap.add_argument("--chunk_col", default="chunk_id")
    ap.add_argument("--reltype_col", default="relation_type")
    ap.add_argument("--just_col", default="justification")

    ap.add_argument("--runs_table", default="axioms_llm_runs")
    ap.add_argument("--axioms_table", default="axioms_llm_predicates")

    ap.add_argument("--min_predicate_count", type=int, default=5)
    ap.add_argument("--examples_per_predicate", type=int, default=12)
    ap.add_argument("--max_predicates", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--prompt_config", default="prompts/axioms_llm.yaml")

    ap.add_argument("--schema_config", default="", help="Optional YAML with allowed_predicates list/map.")
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--model_path", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--max_new_tokens", type=int, default=700)

    ap.add_argument("--debug_one", default="", help="Only run for this predicate (exact match).")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    try:
        yaml_cfg = load_yaml(args.prompt_config)
    except ModuleNotFoundError:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML prompt_config: {e}") from e

    allow_predicates: Optional[Set[str]] = None
    if args.schema_config:
        cfg = load_yaml(args.schema_config)
        apreds = cfg.get("allowed_predicates")
        if isinstance(apreds, list):
            allow_predicates = {str(x).strip().lower() for x in apreds if str(x).strip()}
        elif isinstance(apreds, dict):
            allow_predicates = {str(k).strip().lower() for k in apreds.keys() if str(k).strip()}
        else:
            raise RuntimeError("--schema_config must contain allowed_predicates as list or map.")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    for t in (args.enrich_table, args.taxonomy_table, args.non_tax_table):
        if not table_exists(conn, t):
            raise RuntimeError(f"Missing required table: {t}")

    init_axiom_tables(conn, args.runs_table, args.axioms_table)

    canonical_map, surface_to_canonical, class_set = load_term_info(conn, args.enrich_table)
    parents = load_parents(conn, args.taxonomy_table, args.tax_child_col, args.tax_parent_col, canonical_map, surface_to_canonical)
    taxonomy_edges = load_taxonomy_edges_terms(conn, args.taxonomy_table, args.tax_child_col, args.tax_parent_col, canonical_map, surface_to_canonical)

    client = TransformersClient(model_path=args.model_path, max_new_tokens=args.max_new_tokens)

    counts = predicate_counts(conn, args.non_tax_table, args.pred_col)
    preds = load_predicates(conn, args.non_tax_table, args.pred_col)

    if args.debug_one:
        preds = [p for p in preds if p.strip().lower() == args.debug_one.strip().lower()]
        if not preds:
            raise RuntimeError(f"--debug_one predicate not found: {args.debug_one}")

    filtered: List[str] = []
    for p in preds:
        p_norm = p.strip().lower()
        cnt = counts.get(p, 0)

        if not snake_ok(p_norm):
            continue
        if p_norm in BANNED_PREDICATES:
            continue
        if cnt < args.min_predicate_count:
            continue

        if allow_predicates is not None:
            if p_norm not in allow_predicates:
                continue
        else:
            base = base_verb_of_pred(p_norm)
            if base in VERB_STOP:
                continue
            if has_prep_suffix(p_norm):
                continue

        filtered.append(p_norm)

    filtered = sorted(set(filtered))
    if args.max_predicates and args.max_predicates > 0:
        filtered = filtered[: args.max_predicates]

    print(f"[INFO] Predicates after filtering: {len(filtered)} (min_count={args.min_predicate_count})")
    if allow_predicates is not None:
        print(f"[INFO] Using schema allowlist: {len(allow_predicates)} allowed predicates")

    if args.resume:
        done = set(r[0] for r in conn.execute(
            f"SELECT predicate FROM {args.runs_table} WHERE status IN ('done','skipped');"
        ).fetchall())
        filtered = [p for p in filtered if p not in done]
        print(f"[INFO] After resume-skip: {len(filtered)} remaining")

    known_predicates: Set[str] = set(k.strip().lower() for k in counts.keys()) | set(filtered)

    for i, pred in enumerate(filtered, 1):
        try:
            ev = load_evidence_for_predicate(
                conn=conn,
                non_tax_table=args.non_tax_table,
                subj_col=args.subj_col,
                pred_col=args.pred_col,
                obj_col=args.obj_col,
                doc_col=args.doc_col,
                chunk_col=args.chunk_col,
                reltype_col=args.reltype_col,
                just_col=args.just_col,
                predicate=pred,
                k=args.examples_per_predicate,
                random_seed=args.seed + i,
            )
            if len(ev) < 2:
                mark_run(conn, args.runs_table, pred, "skipped", "too_few_evidence")
                continue

            inferred_kind, inferred_dt = decide_property_kind_from_evidence(pred, ev)

            obj = llm_axioms_for_predicate(
                client=client,
                yaml_cfg=yaml_cfg,
                predicate=pred,
                evidence=ev,
                parents=parents,
                canonical_map=canonical_map,
                surface_to_canonical=surface_to_canonical,
                max_retries=2,
            )

            prop = obj.get("property", {}) if isinstance(obj.get("property"), dict) else {}
            pname = str(prop.get("name") or pred).strip().lower()
            if not snake_ok(pname):
                pname = pred
            prop["name"] = pname
            prop["label"] = str(prop.get("label") or pred).strip()
            prop["kind"] = inferred_kind
            obj["property"] = prop
            obj["datatype_iri"] = inferred_dt if inferred_kind == "DatatypeProperty" else None

            suggested_dom = str(obj.get("domain_class") or "").strip()
            suggested_rng = str(obj.get("range_class") or "").strip()

            dom_can = resolve_to_canonical(suggested_dom, canonical_map, surface_to_canonical)
            rng_can = resolve_to_canonical(suggested_rng, canonical_map, surface_to_canonical)

            dom_class = clamp_to_class(dom_can, canonical_map, class_set, parents)

            rng_class: Optional[str] = None
            if inferred_kind == "ObjectProperty":
                rng_class = clamp_to_class(rng_can, canonical_map, class_set, parents)

            if dom_class is None:
                for sc in [resolve_to_canonical(e.subj, canonical_map, surface_to_canonical) for e in ev]:
                    dom_class = clamp_to_class(sc, canonical_map, class_set, parents)
                    if dom_class:
                        break

            if inferred_kind == "ObjectProperty" and rng_class is None:
                for oc in [resolve_to_canonical(e.obj, canonical_map, surface_to_canonical) for e in ev]:
                    rng_class = clamp_to_class(oc, canonical_map, class_set, parents)
                    if rng_class:
                        break

            obj["domain_class"] = dom_class if dom_class else "owl:Thing"
            obj["range_class"] = (rng_class if rng_class else "owl:Thing") if inferred_kind == "ObjectProperty" else "owl:Thing"

            subs = obj.get("subPropertyOf", [])
            invs = obj.get("inverseOf", [])
            obj["subPropertyOf"] = [str(s).strip().lower() for s in subs if isinstance(s, str) and s.strip().lower() in known_predicates] if isinstance(subs, list) else []
            obj["inverseOf"] = [str(s).strip().lower() for s in invs if isinstance(s, str) and s.strip().lower() in known_predicates] if isinstance(invs, list) else []

            store_axiom_json(conn, args.axioms_table, pred, obj)
            mark_run(conn, args.runs_table, pred, "done", "")

            if i % 10 == 0:
                print(f"[INFO] {i}/{len(filtered)} predicates processed")

        except Exception as e:
            mark_run(conn, args.runs_table, pred, "error", str(e))
            print(f"[WARN] predicate={pred} failed: {e}")

    all_axioms: List[Dict[str, Any]] = []
    for (_pred, ax_json) in conn.execute(f"SELECT predicate, axiom_json FROM {args.axioms_table} ORDER BY predicate;").fetchall():
        try:
            obj = json.loads(ax_json)
            if isinstance(obj, dict):
                all_axioms.append(obj)
        except Exception:
            continue

    predicate_axiom_map = {
        str(ax.get("predicate") or "").strip().lower(): ax
        for ax in all_axioms
        if str(ax.get("predicate") or "").strip()
    }

    raw_assertion_triples = load_assertion_triples(
        conn=conn,
        non_tax_table=args.non_tax_table,
        subj_col=args.subj_col,
        pred_col=args.pred_col,
        obj_col=args.obj_col,
        allowed_predicates=set(predicate_axiom_map.keys()),
    )

    assertion_triples: List[Tuple[str, str, str]] = []
    for s, p, o in raw_assertion_triples:
        if keep_llm_assertion_triple(
            s, p, o,
            predicate_axiom_map=predicate_axiom_map,
            canonical_map=canonical_map,
            surface_to_canonical=surface_to_canonical,
            class_set=class_set,
        ):
            assertion_triples.append((s, p, o))

    print(f"[INFO] Assertion triples kept for export: {len(assertion_triples)}")

    class_terms = sorted(class_set)

    ttl_path = os.path.join(args.out_dir, "hpc_ontology.ttl")
    write_ttl(
        ttl_path,
        base_iri=args.base_iri,
        class_terms=class_terms,
        canonical_map=canonical_map,
        surface_to_canonical=surface_to_canonical,
        taxonomy_edges=taxonomy_edges,
        predicate_axioms=all_axioms,
        assertion_triples=assertion_triples,
        known_predicates=known_predicates,
    )

    merged_path = os.path.join(args.out_dir, "tbox_axioms_llm_merged.json")
    merged = {
        "created_at": now_iso(),
        "db": os.path.abspath(args.db),
        "base_iri": args.base_iri,
        "enrich_table": args.enrich_table,
        "taxonomy_table": args.taxonomy_table,
        "non_tax_table": args.non_tax_table,
        "prompt_config": os.path.abspath(args.prompt_config),
        "schema_config": os.path.abspath(args.schema_config) if args.schema_config else "",
        "model_path": args.model_path,
        "min_predicate_count": args.min_predicate_count,
        "examples_per_predicate": args.examples_per_predicate,
        "axiom_count": len(all_axioms),
        "class_count": len(class_terms),
        "assertion_triple_count": len(assertion_triples),
        "axioms": all_axioms,
    }
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    conn.close()
    print(f"[OK] Wrote TTL (default graph): {ttl_path}")
    print(f"[OK] Wrote audit JSON: {merged_path}")


if __name__ == "__main__":
    main()