from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


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
    # allow a-z0-9_ with at least one letter
    return bool(re.fullmatch(r"[a-z0-9_]{2,80}", s)) and any(c.isalpha() for c in s)


def to_safe_localname(s: str) -> str:
    """
    Convert a label/canonical into an IRI-safe localname.
    We keep it predictable and stable (no hashes) for prototype work.
    """
    t = (s or "").strip()
    if not t:
        return "Thing"
    t = t.lower()
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    if not t:
        t = "thing"
    if t[0].isdigit():
        t = "t_" + t
    return t[:80]


def brace_match_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Find first {...} JSON object in a string using brace matching.
    Returns dict or None.
    """
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
    """
    Option B: YAML prompt config (external).
    Requires PyYAML.
    """
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


# -----------------------------
# LLM clients
# -----------------------------

class LLMClient:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """
    Works with OpenAI API and OpenAI-compatible servers (vLLM, LM Studio OpenAI mode, etc.)
    pip install openai
    """
    def __init__(self, model: str, api_key: str, api_base: Optional[str] = None):
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Missing dependency 'openai'. Install with: pip install openai") from e

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=api_base) if api_base else OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


class TransformersClient(LLMClient):
    """
    Local HF transformers backend.
    pip install transformers accelerate torch
    """
    def __init__(self, model_path: str, max_new_tokens: int = 700):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception as e:
            raise RuntimeError("Missing transformers/torch. Install with: pip install transformers accelerate torch") from e

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
# DB helpers: schema + indexes + resume
# -----------------------------

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (name,),
    ).fetchone()
    return row is not None


def init_axiom_tables(conn: sqlite3.Connection, runs_table: str, out_table: str) -> None:
    cur = conn.cursor()

    # Resume table: one row per predicate
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

    # Store parsed JSON per predicate (for audit/repro)
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            predicate      TEXT PRIMARY KEY,
            axiom_json     TEXT NOT NULL,
            created_at     TEXT NOT NULL
        );
        """
    )

    # Useful indexes
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
    """
    Returns:
      - canonical_map: canonical_lower -> TermInfo
      - surface_to_canonical: lower(term) -> canonical_lower
      - class_set: set of canonical_lower that are safe classes
    """
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

        # safe class criterion
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
    # fallback: lowercase original
    return key


# -----------------------------
# Taxonomy parents (for ancestor lookup)
# -----------------------------

def load_parents(
    conn: sqlite3.Connection,
    taxonomy_table: str,
    child_col: str,
    parent_col: str,
    canonical_map: Dict[str, TermInfo],
    surface_to_canonical: Dict[str, str],
) -> Dict[str, List[str]]:
    """
    parents[child_canonical] = [parent_canonical,...]
    """
    q = f"SELECT {child_col}, {parent_col} FROM {taxonomy_table};"
    parents: Dict[str, List[str]] = {}
    for c, p in conn.execute(q).fetchall():
        c2 = resolve_to_canonical(norm(c) or "", canonical_map, surface_to_canonical)
        p2 = resolve_to_canonical(norm(p) or "", canonical_map, surface_to_canonical)
        if not c2 or not p2 or c2 == p2:
            continue
        parents.setdefault(c2, []).append(p2)

    # dedupe preserve order
    for k, vals in list(parents.items()):
        seen = set()
        out = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        parents[k] = out
    return parents


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


def clamp_to_class(
    term_canonical: str,
    canonical_map: Dict[str, TermInfo],
    class_set: Set[str],
    parents: Dict[str, List[str]],
) -> Optional[str]:
    """
    Return a canonical that is a class:
    - if term itself is a class, return it
    - else return nearest class ancestor
    - else None
    """
    if not term_canonical:
        return None
    info = canonical_map.get(term_canonical)
    if info and info.ontology_role == "class" and info.is_hpc_domain == 1:
        return term_canonical
    # search ancestors
    for a in ancestors_bfs(term_canonical, parents, max_hops=4, max_out=20):
        ai = canonical_map.get(a)
        if ai and ai.ontology_role == "class" and ai.is_hpc_domain == 1:
            return a
        if a in class_set:
            return a
    return None


# -----------------------------
# Non-tax evidence + predicate inventory (deterministic)
# -----------------------------

def predicate_counts(
    conn: sqlite3.Connection,
    non_tax_table: str,
    pred_col: str,
) -> Dict[str, int]:
    q = f"SELECT {pred_col}, COUNT(*) FROM {non_tax_table} GROUP BY {pred_col};"
    out: Dict[str, int] = {}
    for pred, cnt in conn.execute(q).fetchall():
        p = norm(pred)
        if p:
            out[p] = int(cnt or 0)
    return out


def load_predicates(
    conn: sqlite3.Connection,
    non_tax_table: str,
    pred_col: str,
) -> List[str]:
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
) -> List[EvidenceRow]:
    """
    Deterministic evidence: stable ordering and LIMIT.
    """
    q = f"""
    SELECT {subj_col}, {pred_col}, {obj_col}, {doc_col}, {chunk_col}, {reltype_col}, {just_col}
    FROM {non_tax_table}
    WHERE {pred_col} = ?
    ORDER BY COALESCE({doc_col}, ''), COALESCE({chunk_col}, ''), COALESCE({subj_col}, ''), COALESCE({obj_col}, '')
    LIMIT ?;
    """
    rows = conn.execute(q, (predicate, k)).fetchall()
    out: List[EvidenceRow] = []
    for s, p, o, doc, chunk, rt, just in rows:
        out.append(EvidenceRow(
            subj=norm(s) or "",
            pred=norm(p) or "",
            obj=norm(o) or "",
            doc_id=norm(doc),
            chunk_id=norm(chunk),
            relation_type=norm(rt),
            justification=norm(just),
        ))
    return out


# -----------------------------
# Prompting + strict output validation
# -----------------------------

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


def validate_axiom_json(obj: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Strict-ish schema:
    {
      "predicate": "...",
      "property": {"name": "...", "label": "...", "kind": "ObjectProperty"},
      "domain_class": "...",
      "range_class": "...",
      "subPropertyOf": [...],
      "inverseOf": [...],
      "confidence": 0.0,
      "rationale": "...",
      "evidence": [{"subj":..,"obj":..,"doc_id":..,"chunk_id":..,"justification":..}, ...]
    }
    """
    need = ["predicate", "property", "domain_class", "range_class", "subPropertyOf", "inverseOf", "confidence", "rationale", "evidence"]
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


def build_prompts_from_yaml(
    yaml_cfg: Dict[str, Any],
    *,
    predicate: str,
    evidence: List[EvidenceRow],
    subj_ancestors: Dict[str, List[str]],
    obj_ancestors: Dict[str, List[str]],
) -> Tuple[str, str]:
    """
    YAML expected keys:
      - system_prompt: str
      - user_template: str  (uses {predicate}, {evidence_block}, {subj_ancestors_json}, {obj_ancestors_json})
    """
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
    client: LLMClient,
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

        # force predicate match
        obj["predicate"] = predicate
        obj.setdefault("created_at", now_iso())
        obj.setdefault("raw_output", last)
        return obj

    # hard fallback
    return {
        "predicate": predicate,
        "property": {"name": predicate, "label": predicate, "kind": "ObjectProperty"},
        "domain_class": "owl:Thing",
        "range_class": "owl:Thing",
        "subPropertyOf": [],
        "inverseOf": [],
        "confidence": 0.0,
        "rationale": "LLM failed to produce valid JSON after retries; using owl:Thing/Thing fallback.",
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


# -----------------------------
# TTL export (DEFAULT GRAPH)
# -----------------------------

def ttl_escape(s: str) -> str:
    return (s or "").replace("\\", "\\\\").replace('"', '\\"')


def write_ttl(
    out_path: str,
    *,
    base_iri: str,
    class_terms: Sequence[str],
    canonical_map: Dict[str, TermInfo],
    predicate_axioms: Sequence[Dict[str, Any]],
    known_predicates: Set[str],
) -> None:
    """
    Writes a single TTL file for default graph load:
      - declares classes
      - declares properties + domain/range + optional subPropertyOf/inverseOf
    """
    # prefix IRI must end with # or /
    if not (base_iri.endswith("#") or base_iri.endswith("/")):
        base_iri = base_iri + "#"

    lines: List[str] = []
    lines.append(f"@prefix hpc: <{base_iri}> .")
    lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    lines.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    lines.append("@prefix owl: <http://www.w3.org/2002/07/owl#> .")
    lines.append("")
    lines.append("### Classes (from llm_enrich_final)")
    for c in sorted(set(class_terms)):
        info = canonical_map.get(c)
        label = info.term if info and info.term else c
        local = to_safe_localname(c)
        lines.append(f"hpc:{local} a owl:Class ;")
        lines.append(f'  rdfs:label "{ttl_escape(label)}" .')
        lines.append("")

    lines.append("### Properties (from non-tax predicates + LLM axiom suggestions clamped to classes)")
    for ax in predicate_axioms:
        pred = ax.get("predicate", "")
        prop = ax.get("property", {}) if isinstance(ax.get("property"), dict) else {}
        pname = prop.get("name") or pred
        plabel = prop.get("label") or pred
        kind = prop.get("kind") or "ObjectProperty"

        local_p = to_safe_localname(pname)
        rdf_kind = "owl:ObjectProperty" if kind == "ObjectProperty" else "owl:DatatypeProperty"
        lines.append(f"hpc:{local_p} a {rdf_kind} ;")
        lines.append(f'  rdfs:label "{ttl_escape(str(plabel))}" ;')

        dom = str(ax.get("domain_class") or "owl:Thing")
        rng = str(ax.get("range_class") or "owl:Thing")

        # allow owl:Thing explicitly
        if dom == "owl:Thing":
            lines.append("  rdfs:domain owl:Thing ;")
        else:
            lines.append(f"  rdfs:domain hpc:{to_safe_localname(dom)} ;")
        if rng == "owl:Thing":
            lines.append("  rdfs:range owl:Thing")
        else:
            lines.append(f"  rdfs:range hpc:{to_safe_localname(rng)}")

        # optional: subPropertyOf / inverseOf (only if they point to known predicates)
        subs = ax.get("subPropertyOf", [])
        invs = ax.get("inverseOf", [])
        if isinstance(subs, list) and subs:
            good = [s for s in subs if isinstance(s, str) and s in known_predicates]
            if good:
                # append as extra triples after main statement
                lines.append(" .")
                for s in good:
                    lines.append(f"hpc:{local_p} rdfs:subPropertyOf hpc:{to_safe_localname(s)} .")
                # inverseOf handled similarly below
            else:
                lines.append(" .")
        else:
            lines.append(" .")

        if isinstance(invs, list) and invs:
            good = [s for s in invs if isinstance(s, str) and s in known_predicates]
            for inv in good:
                lines.append(f"hpc:{local_p} owl:inverseOf hpc:{to_safe_localname(inv)} .")
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Main pipeline: predicate -> LLM -> clamp -> DB -> TTL
# -----------------------------

BANNED_PREDICATES = {
    "related_to", "associated_with", "has", "have", "do", "does", "did",
    "make", "made", "create", "created", "add", "add_to", "use", "uses",
    "set", "sets", "enable", "enabled", "disable", "disabled",
}

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate TBox axioms (TTL) from LLM tables with strong guards.")

    ap.add_argument("--db", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_iri", default="http://example.org/hpc#")

    # Input tables
    ap.add_argument("--enrich_table", default="llm_enrich_final")
    ap.add_argument("--taxonomy_table", default="llm_is_a_edges")
    ap.add_argument("--non_tax_table", default="llm_non_taxonomy_edges")

    # Column overrides (taxonomy)
    ap.add_argument("--tax_child_col", default="child_term")
    ap.add_argument("--tax_parent_col", default="parent_term")

    # Column overrides (non-tax)
    ap.add_argument("--subj_col", default="subject_term")
    ap.add_argument("--pred_col", default="predicate")
    ap.add_argument("--obj_col", default="object_term")
    ap.add_argument("--doc_col", default="doc_id")
    ap.add_argument("--chunk_col", default="chunk_id")
    ap.add_argument("--reltype_col", default="relation_type")
    ap.add_argument("--just_col", default="justification")

    # Output/Resume tables
    ap.add_argument("--runs_table", default="axioms_llm_runs")
    ap.add_argument("--axioms_table", default="axioms_llm_predicates")

    # Controls
    ap.add_argument("--min_predicate_count", type=int, default=5, help="Skip predicates that appear < N times.")
    ap.add_argument("--examples_per_predicate", type=int, default=12)
    ap.add_argument("--max_predicates", type=int, default=0, help="0 = all")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")

    # Prompt config (Option B)
    ap.add_argument("--prompt_config", default="prompts/non_tax_llm.yaml")

    # Backend
    ap.add_argument("--backend", choices=["openai_compatible", "transformers"], required=True)
    ap.add_argument("--model", default=None, help="Model name for openai_compatible backend")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key (or OPENAI_API_KEY)")
    ap.add_argument("--api_base", default=None, help="e.g., http://localhost:8000/v1 for vLLM")
    ap.add_argument("--model_path", default=None, help="Local HF model path for transformers backend")

    ap.add_argument("--debug_one", default="", help="Only run for this predicate (exact match).")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # Load YAML prompt config
    try:
        yaml_cfg = load_yaml(args.prompt_config)
    except ModuleNotFoundError:
        raise RuntimeError("PyYAML not installed. Install with: pip install pyyaml")
    except Exception as e:
        raise RuntimeError(f"Failed to load YAML prompt_config: {e}") from e

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Validate required tables
    for t in (args.enrich_table, args.taxonomy_table, args.non_tax_table):
        if not table_exists(conn, t):
            raise RuntimeError(f"Missing required table: {t}")

    init_axiom_tables(conn, args.runs_table, args.axioms_table)

    canonical_map, surface_to_canonical, class_set = load_term_info(conn, args.enrich_table)
    parents = load_parents(conn, args.taxonomy_table, args.tax_child_col, args.tax_parent_col, canonical_map, surface_to_canonical)

    # Backend client
    if args.backend == "openai_compatible":
        if not args.model:
            raise RuntimeError("--model is required for openai_compatible backend")
        api_key = args.api_key or "EMPTY"
        client: LLMClient = OpenAICompatibleClient(model=args.model, api_key=api_key, api_base=args.api_base)
    else:
        if not args.model_path:
            raise RuntimeError("--model_path is required for transformers backend")
        client = TransformersClient(model_path=args.model_path)

    # Predicate inventory + counts
    counts = predicate_counts(conn, args.non_tax_table, args.pred_col)
    preds = load_predicates(conn, args.non_tax_table, args.pred_col)

    if args.debug_one:
        preds = [p for p in preds if p == args.debug_one]
        if not preds:
            raise RuntimeError(f"--debug_one predicate not found: {args.debug_one}")

    # Filter + order
    filtered: List[str] = []
    for p in preds:
        p_norm = p.strip().lower()
        cnt = counts.get(p, 0)

        # validate predicate form (prefer snake_case)
        if not snake_ok(p_norm):
            continue
        if p_norm in BANNED_PREDICATES:
            continue
        if cnt < args.min_predicate_count:
            continue

        filtered.append(p_norm)

    # deterministic order
    filtered = sorted(set(filtered))

    if args.max_predicates and args.max_predicates > 0:
        filtered = filtered[: args.max_predicates]

    print(f"[INFO] Predicates after filtering: {len(filtered)} (min_count={args.min_predicate_count})")

    # If resume: skip predicates already done/skipped
    if args.resume:
        done = set(r[0] for r in conn.execute(
            f"SELECT predicate FROM {args.runs_table} WHERE status IN ('done','skipped');"
        ).fetchall())
        filtered = [p for p in filtered if p not in done]
        print(f"[INFO] After resume-skip: {len(filtered)} remaining")

    # Run LLM for each predicate, then clamp domain/range to classes
    axiom_objs: List[Dict[str, Any]] = []
    known_predicates: Set[str] = set(filtered) | set(counts.keys())

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
            )
            if len(ev) < 2:
                mark_run(conn, args.runs_table, pred, "skipped", "too_few_evidence")
                continue

            # Ask LLM for suggestions (property name/label + domain/range)
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

            # ---- Clamp domain/range to CLASSES only (super important) ----
            # Resolve suggested domain/range to canonical, then clamp to class/ancestor class
            suggested_dom = str(obj.get("domain_class") or "").strip()
            suggested_rng = str(obj.get("range_class") or "").strip()

            dom_can = resolve_to_canonical(suggested_dom, canonical_map, surface_to_canonical)
            rng_can = resolve_to_canonical(suggested_rng, canonical_map, surface_to_canonical)

            dom_class = clamp_to_class(dom_can, canonical_map, class_set, parents)
            rng_class = clamp_to_class(rng_can, canonical_map, class_set, parents)

            # If LLM suggestion fails, fall back to evidence-driven clamping
            if dom_class is None:
                # try from subjects
                subj_cans = [resolve_to_canonical(e.subj, canonical_map, surface_to_canonical) for e in ev]
                for sc in subj_cans:
                    dom_class = clamp_to_class(sc, canonical_map, class_set, parents)
                    if dom_class:
                        break
            if rng_class is None:
                # try from objects
                obj_cans = [resolve_to_canonical(e.obj, canonical_map, surface_to_canonical) for e in ev]
                for oc in obj_cans:
                    rng_class = clamp_to_class(oc, canonical_map, class_set, parents)
                    if rng_class:
                        break

            obj["domain_class"] = dom_class if dom_class else "owl:Thing"
            obj["range_class"] = rng_class if rng_class else "owl:Thing"

            # normalize property.name to snake_case if missing/invalid
            prop = obj.get("property", {}) if isinstance(obj.get("property"), dict) else {}
            pname = (prop.get("name") or pred).strip().lower()
            if not snake_ok(pname):
                pname = pred
            prop["name"] = pname
            prop["kind"] = "ObjectProperty"  # safe default (edges are term-term)
            obj["property"] = prop

            # keep only known predicates for subPropertyOf/inverseOf
            subs = obj.get("subPropertyOf", [])
            invs = obj.get("inverseOf", [])
            if isinstance(subs, list):
                obj["subPropertyOf"] = [s for s in subs if isinstance(s, str) and s in known_predicates]
            else:
                obj["subPropertyOf"] = []
            if isinstance(invs, list):
                obj["inverseOf"] = [s for s in invs if isinstance(s, str) and s in known_predicates]
            else:
                obj["inverseOf"] = []

            # Store in DB + mark run
            store_axiom_json(conn, args.axioms_table, pred, obj)
            mark_run(conn, args.runs_table, pred, "done", "")

            axiom_objs.append(obj)

            if i % 10 == 0:
                print(f"[INFO] {i}/{len(filtered)} predicates processed")

        except Exception as e:
            mark_run(conn, args.runs_table, pred, "error", str(e))
            print(f"[WARN] predicate={pred} failed: {e}")

    # If resume mode and we want TTL for ALL processed ever, load from DB table
    all_axioms: List[Dict[str, Any]] = []
    for (pred, ax_json) in conn.execute(f"SELECT predicate, axiom_json FROM {args.axioms_table} ORDER BY predicate;").fetchall():
        try:
            obj = json.loads(ax_json)
            if isinstance(obj, dict):
                all_axioms.append(obj)
        except Exception:
            continue

    # Declare classes from llm_enrich_final (safe classes only)
    class_terms = sorted(class_set)

    ttl_path = os.path.join(args.out_dir, "tbox_axioms_llm.ttl")
    write_ttl(
        ttl_path,
        base_iri=args.base_iri,
        class_terms=class_terms,
        canonical_map=canonical_map,
        predicate_axioms=all_axioms,
        known_predicates=set(counts.keys()),
    )

    # Also dump merged JSON for audit
    merged_path = os.path.join(args.out_dir, "tbox_axioms_llm_merged.json")
    merged = {
        "created_at": now_iso(),
        "db": os.path.abspath(args.db),
        "base_iri": args.base_iri,
        "enrich_table": args.enrich_table,
        "taxonomy_table": args.taxonomy_table,
        "non_tax_table": args.non_tax_table,
        "prompt_config": os.path.abspath(args.prompt_config),
        "backend": args.backend,
        "model": args.model if args.backend == "openai_compatible" else args.model_path,
        "min_predicate_count": args.min_predicate_count,
        "examples_per_predicate": args.examples_per_predicate,
        "axiom_count": len(all_axioms),
        "class_count": len(class_terms),
        "axioms": all_axioms,
    }
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    conn.close()
    print(f"[OK] Wrote TTL (default graph): {ttl_path}")
    print(f"[OK] Wrote audit JSON: {merged_path}")


if __name__ == "__main__":
    main()
