#!/usr/bin/env python3
"""
olaf_llm/axioms_llm.py

LLM-only lightweight axiom extraction from *LLM-generated* edge tables in SQLite.

DEFAULT INPUT TABLES (your schema):
- taxonomy: llm_is_a_edges
    columns: child_term, parent_term, doc_id, chunk_id, justification
- non-taxonomy: llm_non_taxonomy_edges
    columns: subject_term, predicate, object_term, doc_id, chunk_id, relation_type, justification, raw_json

For each distinct predicate, the LLM proposes:
- ObjectProperty name/label
- domain class + range class (best guess using only provided evidence + taxonomy context)
- optional: subPropertyOf, inverseOf
- optional candidate: disjointness and restrictions
Writes:
- axioms_llm.jsonl (one JSON object per predicate)
- axioms_llm_merged.json (full combined)

Backend options:
- openai_compatible (OpenAI API OR vLLM OpenAI server, LMStudio, etc.)
- transformers (local HF model)

NOTE:
This is "LLM-only" w.r.t. axiom decisions.
The code only samples evidence rows + taxonomy context and enforces JSON schema.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


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


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Find and parse first JSON object in a string (brace matching)."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i+1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
                    return None
    return None


def validate_output(obj: Dict[str, Any]) -> Tuple[bool, str]:
    # Minimal strict schema validation
    required_top = ["relation", "axioms", "confidence", "evidence"]
    for k in required_top:
        if k not in obj:
            return False, f"Missing top-level key '{k}'"

    if not isinstance(obj["relation"], str):
        return False, "relation must be a string"
    if not isinstance(obj["axioms"], dict):
        return False, "axioms must be an object"
    if not isinstance(obj["confidence"], (int, float)):
        return False, "confidence must be a number"
    if not isinstance(obj["evidence"], list):
        return False, "evidence must be a list"

    axioms = obj["axioms"]
    for k in ["property", "domain", "range", "subPropertyOf", "inverseOf",
              "disjointness_candidates", "restriction_candidates"]:
        if k not in axioms:
            return False, f"Missing axioms.{k}"

    prop = axioms["property"]
    for k in ["name", "label", "kind"]:
        if k not in prop:
            return False, f"Missing axioms.property.{k}"
        if not isinstance(prop[k], str):
            return False, f"axioms.property.{k} must be a string"

    for side in ["domain", "range"]:
        if "class" not in axioms[side] or "rationale" not in axioms[side]:
            return False, f"axioms.{side} must have 'class' and 'rationale'"
        if not isinstance(axioms[side]["class"], str):
            return False, f"axioms.{side}.class must be a string"
        if not isinstance(axioms[side]["rationale"], str):
            return False, f"axioms.{side}.rationale must be a string"

    if not isinstance(axioms["subPropertyOf"], list):
        return False, "axioms.subPropertyOf must be a list"
    if not isinstance(axioms["inverseOf"], list):
        return False, "axioms.inverseOf must be a list"
    if not isinstance(axioms["disjointness_candidates"], list):
        return False, "axioms.disjointness_candidates must be a list"
    if not isinstance(axioms["restriction_candidates"], list):
        return False, "axioms.restriction_candidates must be a list"

    return True, "ok"


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class EvidenceRow:
    subj: str
    rel: str
    obj: str
    doc_id: Optional[str]
    chunk_id: Optional[str]
    relation_type: Optional[str]
    justification: Optional[str]
    raw_json: Optional[str]


# -----------------------------
# DB access (your schema defaults)
# -----------------------------

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (name,),
    ).fetchone()
    return row is not None


def load_taxonomy_parents(
    conn: sqlite3.Connection,
    taxonomy_table: str,
    child_col: str,
    parent_col: str,
) -> Dict[str, List[str]]:
    """
    Returns parents[child] = [parent1, parent2,...]
    """
    q = f"SELECT {child_col}, {parent_col} FROM {taxonomy_table};"
    parents: Dict[str, List[str]] = defaultdict(list)
    for c, p in conn.execute(q).fetchall():
        c2, p2 = norm(c), norm(p)
        if c2 and p2:
            parents[c2].append(p2)
    return parents


def top_ancestors(term: str, parents: Dict[str, List[str]], max_hops: int = 3, max_out: int = 8) -> List[str]:
    seen = set()
    frontier = [(term, 0)]
    out: List[str] = []
    while frontier:
        node, d = frontier.pop(0)
        if node in seen:
            continue
        seen.add(node)
        if d >= max_hops:
            continue
        for p in parents.get(node, []):
            if p not in seen:
                out.append(p)
                frontier.append((p, d + 1))
        if len(out) >= max_out:
            break
    # unique-preserve
    uniq: List[str] = []
    for x in out:
        if x not in uniq:
            uniq.append(x)
    return uniq[:max_out]


def load_relation_inventory(
    conn: sqlite3.Connection,
    non_tax_table: str,
    rel_col: str,
) -> List[str]:
    q = f"SELECT DISTINCT {rel_col} FROM {non_tax_table} ORDER BY {rel_col};"
    return [norm(r[0]) for r in conn.execute(q).fetchall() if norm(r[0])]


def load_evidence_for_relation(
    conn: sqlite3.Connection,
    non_tax_table: str,
    subj_col: str,
    rel_col: str,
    obj_col: str,
    doc_col: str,
    chunk_col: str,
    reltype_col: str,
    just_col: str,
    rawjson_col: str,
    relation: str,
    k: int,
) -> List[EvidenceRow]:
    q = f"""
    SELECT {subj_col}, {rel_col}, {obj_col},
           {doc_col}, {chunk_col}, {reltype_col}, {just_col}, {rawjson_col}
    FROM {non_tax_table}
    WHERE {rel_col} = ?
    """
    rows = conn.execute(q, (relation,)).fetchall()
    if not rows:
        return []
    if len(rows) > k:
        rows = random.sample(rows, k)

    out: List[EvidenceRow] = []
    for s, r, o, doc_id, chunk_id, reltype, just, rawj in rows:
        out.append(EvidenceRow(
            subj=norm(s) or "",
            rel=norm(r) or "",
            obj=norm(o) or "",
            doc_id=norm(doc_id),
            chunk_id=norm(chunk_id),
            relation_type=norm(reltype),
            justification=norm(just),
            raw_json=norm(rawj),
        ))
    return out


# -----------------------------
# LLM clients
# -----------------------------

class LLMClient:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    """
    Works with OpenAI API *and* OpenAI-compatible servers (vLLM, LM Studio OpenAI mode, etc.)
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
        return resp.choices[0].message.content or ""


class TransformersClient(LLMClient):
    """
    Local HF transformers backend.
    pip install transformers accelerate torch
    """
    def __init__(self, model_path: str, max_new_tokens: int = 900):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except Exception as e:
            raise RuntimeError("Missing transformers/torch. Install with: pip install transformers accelerate torch") from e

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
# Prompting (LLM-only axioms)
# -----------------------------

SYSTEM_PROMPT = """You are an expert ontology engineer for HPC documentation (SLURM + IBM LSF domain).
Your job: generate LIGHTWEIGHT ontology axioms for ONE relation based ONLY on the provided evidence.

Return ONLY valid JSON (single object). No markdown. No commentary.
Be conservative: if unsure, set low confidence and explain rationale.

Do NOT invent evidence. Use only the supplied triples + their justifications and taxonomy context.
"""

USER_PROMPT_TEMPLATE = """Generate axioms for relation (predicate): "{relation}"

EVIDENCE (triples + LLM justifications / raw context):
{evidence_block}

TAXONOMY CONTEXT (ancestors sampled from llm_is_a_edges):
- subject ancestors examples: {subj_ancestors}
- object ancestors examples: {obj_ancestors}

Output JSON schema (must match exactly):
{{
  "relation": "{relation}",
  "axioms": {{
    "property": {{
      "name": "<normalized property name>",
      "label": "<human label>",
      "kind": "ObjectProperty"
    }},
    "domain": {{
      "class": "<best domain class>",
      "rationale": "<evidence-based rationale>"
    }},
    "range": {{
      "class": "<best range class>",
      "rationale": "<evidence-based rationale>"
    }},
    "subPropertyOf": ["<optional super-property names>"],
    "inverseOf": ["<optional inverse property names>"],
    "disjointness_candidates": [
      {{
        "classes": ["<A>", "<B>"],
        "rationale": "<why candidate disjoint>",
        "confidence": 0.0
      }}
    ],
    "restriction_candidates": [
      {{
        "class": "<some class>",
        "restriction": "<e.g., Class ⊑ ∃property.RangeClass>",
        "rationale": "<why candidate>",
        "confidence": 0.0
      }}
    ]
  }},
  "confidence": 0.0,
  "evidence": [
    {{
      "subj": "<subj term>",
      "obj": "<obj term>",
      "doc_id": "<doc_id or null>",
      "chunk_id": "<chunk_id or null>",
      "justification": "<justification or null>"
    }}
  ]
}}

Constraints:
- evidence list must include at least 3 items (unless fewer were provided).
- Keep disjointness_candidates / restriction_candidates empty if you have no strong support.
- If you propose a domain/range class not present in taxonomy context, mention that explicitly.
"""


def normalize_property_name(rel_text: str) -> str:
    s = (rel_text or "").strip().lower()
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    s = "_".join([p for p in s.split("_") if p])
    return s[:80] if s else "related_to"


def build_evidence_block(evs: List[EvidenceRow], max_chars: int = 4200) -> str:
    lines: List[str] = []
    for e in evs:
        just = (e.justification or e.raw_json or "").replace("\n", " ").strip()
        just = just[:300]
        lines.append(
            f'- ({e.subj}) {e.rel} ({e.obj}) | doc={e.doc_id} chunk={e.chunk_id} rel_type={e.relation_type} | just="{just}"'
        )
    block = "\n".join(lines)
    if len(block) > max_chars:
        block = block[:max_chars] + "\n... (truncated)"
    return block


def build_prompt(
    relation: str,
    evidence: List[EvidenceRow],
    parents: Dict[str, List[str]],
) -> Tuple[str, str]:
    subj_terms = [e.subj for e in evidence][:6]
    obj_terms = [e.obj for e in evidence][:6]
    subj_anc = {t: top_ancestors(t, parents, max_hops=3, max_out=6) for t in subj_terms}
    obj_anc = {t: top_ancestors(t, parents, max_hops=3, max_out=6) for t in obj_terms}

    user = USER_PROMPT_TEMPLATE.format(
        relation=relation,
        evidence_block=build_evidence_block(evidence),
        subj_ancestors=json.dumps(subj_anc, ensure_ascii=False),
        obj_ancestors=json.dumps(obj_anc, ensure_ascii=False),
    )
    return SYSTEM_PROMPT, user


def llm_extract_for_relation(
    client: LLMClient,
    relation: str,
    evidence: List[EvidenceRow],
    parents: Dict[str, List[str]],
    max_retries: int = 2,
) -> Dict[str, Any]:
    system, user = build_prompt(relation, evidence, parents)
    last_text = ""

    for attempt in range(max_retries + 1):
        text = client.complete(system=system, user=user)
        last_text = text

        obj = extract_first_json_object(text)
        if obj is None:
            user += "\n\nYour response was not valid JSON. Return ONLY a single valid JSON object."
            continue

        ok, msg = validate_output(obj)
        if not ok:
            user += f"\n\nYour JSON did not match schema: {msg}. Return corrected JSON ONLY."
            continue

        # Force relation exact match
        obj["relation"] = relation

        # Ensure evidence list
        ev_out = []
        for e in evidence[: max(3, min(len(evidence), 10))]:
            ev_out.append({
                "subj": e.subj,
                "obj": e.obj,
                "doc_id": e.doc_id,
                "chunk_id": e.chunk_id,
                "justification": e.justification,
            })
        if not obj.get("evidence"):
            obj["evidence"] = ev_out
        else:
            # keep model evidence but ensure at least 3 if possible
            if len(obj["evidence"]) < 3 and len(ev_out) >= 3:
                obj["evidence"] = ev_out

        # Fill defaults if missing
        obj.setdefault("confidence", 0.5)
        obj.setdefault("created_at", now_iso())
        return obj

    # fallback
    return {
        "relation": relation,
        "axioms": {
            "property": {"name": normalize_property_name(relation), "label": relation, "kind": "ObjectProperty"},
            "domain": {"class": "UNKNOWN", "rationale": "LLM output invalid; see raw_output"},
            "range": {"class": "UNKNOWN", "rationale": "LLM output invalid; see raw_output"},
            "subPropertyOf": [],
            "inverseOf": [],
            "disjointness_candidates": [],
            "restriction_candidates": [],
        },
        "confidence": 0.0,
        "evidence": [
            {
                "subj": e.subj,
                "obj": e.obj,
                "doc_id": e.doc_id,
                "chunk_id": e.chunk_id,
                "justification": e.justification,
            } for e in evidence[:3]
        ],
        "raw_output": last_text,
        "error": "Failed to produce valid JSON after retries",
        "created_at": now_iso(),
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--db", required=True)
    ap.add_argument("--out_dir", required=True)

    # Defaults aligned to your LLM tables
    ap.add_argument("--taxonomy_table", default="llm_is_a_edges")
    ap.add_argument("--non_tax_table", default="llm_non_taxonomy_edges")

    # Column overrides (in case schema changes)
    ap.add_argument("--tax_child_col", default="child_term")
    ap.add_argument("--tax_parent_col", default="parent_term")

    ap.add_argument("--subj_col", default="subject_term")
    ap.add_argument("--rel_col", default="predicate")
    ap.add_argument("--obj_col", default="object_term")
    ap.add_argument("--doc_col", default="doc_id")
    ap.add_argument("--chunk_col", default="chunk_id")
    ap.add_argument("--reltype_col", default="relation_type")
    ap.add_argument("--just_col", default="justification")
    ap.add_argument("--rawjson_col", default="raw_json")

    ap.add_argument("--examples_per_relation", type=int, default=12)
    ap.add_argument("--max_relations", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--sleep_s", type=float, default=0.0)

    ap.add_argument("--backend", choices=["openai_compatible", "transformers"], required=True)
    ap.add_argument("--model", default=None, help="Model name for openai_compatible backend")
    ap.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key (or OPENAI_API_KEY)")
    ap.add_argument("--api_base", default=None, help="e.g., http://localhost:8000/v1 for vLLM")
    ap.add_argument("--model_path", default=None, help="Local HF model path for transformers backend")
    args = ap.parse_args()

    random.seed(args.seed)
    ensure_dir(args.out_dir)

    conn = sqlite3.connect(args.db)

    if not table_exists(conn, args.taxonomy_table):
        raise RuntimeError(f"Missing taxonomy table: {args.taxonomy_table}")
    if not table_exists(conn, args.non_tax_table):
        raise RuntimeError(f"Missing non-taxonomy table: {args.non_tax_table}")

    parents = load_taxonomy_parents(conn, args.taxonomy_table, args.tax_child_col, args.tax_parent_col)

    # Backend
    if args.backend == "openai_compatible":
        if not args.model:
            raise RuntimeError("--model is required for openai_compatible backend")
        api_key = args.api_key or "EMPTY"
        client: LLMClient = OpenAICompatibleClient(model=args.model, api_key=api_key, api_base=args.api_base)
    else:
        if not args.model_path:
            raise RuntimeError("--model_path is required for transformers backend")
        client = TransformersClient(model_path=args.model_path)

    relations = load_relation_inventory(conn, args.non_tax_table, args.rel_col)
    if args.max_relations and args.max_relations > 0:
        relations = relations[: args.max_relations]

    print(f"[INFO] Relations to process: {len(relations)}")

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, rel in enumerate(relations, 1):
        evidence = load_evidence_for_relation(
            conn=conn,
            non_tax_table=args.non_tax_table,
            subj_col=args.subj_col,
            rel_col=args.rel_col,
            obj_col=args.obj_col,
            doc_col=args.doc_col,
            chunk_col=args.chunk_col,
            reltype_col=args.reltype_col,
            just_col=args.just_col,
            rawjson_col=args.rawjson_col,
            relation=rel,
            k=args.examples_per_relation,
        )

        out = llm_extract_for_relation(client, rel, evidence, parents, max_retries=2)
        out["created_at"] = now_iso()
        results.append(out)

        if args.sleep_s > 0:
            time.sleep(args.sleep_s)

        if i % 10 == 0:
            print(f"[INFO] {i}/{len(relations)} done")

    conn.close()

    jsonl_path = os.path.join(args.out_dir, "axioms_llm.jsonl")
    write_jsonl(jsonl_path, results)

    merged = {
        "created_at": now_iso(),
        "db": os.path.abspath(args.db),
        "taxonomy_table": args.taxonomy_table,
        "non_tax_table": args.non_tax_table,
        "backend": args.backend,
        "model": args.model if args.backend == "openai_compatible" else args.model_path,
        "examples_per_relation": args.examples_per_relation,
        "count": len(results),
        "axioms": results,
    }
    json_path = os.path.join(args.out_dir, "axioms_llm_merged.json")
    write_json(json_path, merged)

    print(f"[OK] Wrote: {jsonl_path}")
    print(f"[OK] Wrote: {json_path}")
    print(f"[INFO] Total time: {round(time.time() - t0, 2)}s")


if __name__ == "__main__":
    main()
