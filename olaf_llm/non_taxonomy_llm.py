from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# =============================================================================
# Enums / validation
# =============================================================================

RELATION_TYPE_ENUM: Set[str] = {
    "part_of",
    "configuration",
    "provision",
    "usage",
    "data_flow",
    "logging",
    "scheduling",
    "other",
}

PREDICATE_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")  # lower_snake_case-ish


# =============================================================================
# Prompt config (STRICT: YAML required)
# =============================================================================

@dataclass(frozen=True)
class PromptConfig:
    system_prompt: str
    few_shots: List[Dict[str, Any]]
    max_terms_per_chunk: int
    max_new_tokens: int
    allowed_predicates: Optional[Set[str]]


def load_prompt_config(path: str) -> PromptConfig:
    if not path:
        raise RuntimeError("--prompt-config is required (YAML prompt file).")
    if not os.path.exists(path):
        raise RuntimeError(f"Prompt config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}

    if not isinstance(obj, dict):
        raise RuntimeError("Invalid YAML prompt config: expected a mapping/object at top level.")

    system_prompt = obj.get("system_prompt")
    few_shots = obj.get("few_shots") or obj.get("few_shot_examples")

    if not system_prompt or not isinstance(system_prompt, str):
        raise RuntimeError("YAML prompt config must include 'system_prompt' as a non-empty string.")

    if few_shots is None or not isinstance(few_shots, list):
        raise RuntimeError("YAML prompt config must include 'few_shots' as a list (can be empty).")

    # Validate few-shots structure strictly
    for i, ex in enumerate(few_shots, start=1):
        if not isinstance(ex, dict):
            raise RuntimeError(f"few_shots[{i}] must be a mapping with keys: text, terms, json.")
        if "text" not in ex or "terms" not in ex or "json" not in ex:
            raise RuntimeError(f"few_shots[{i}] must contain keys: text, terms, json.")
        if not isinstance(ex["text"], str):
            raise RuntimeError(f"few_shots[{i}].text must be a string.")
        if not isinstance(ex["terms"], list) or not all(isinstance(t, str) for t in ex["terms"]):
            raise RuntimeError(f"few_shots[{i}].terms must be a list of strings.")
        if not isinstance(ex["json"], dict):
            raise RuntimeError(f"few_shots[{i}].json must be a dict (structured JSON object).")

    max_terms_per_chunk = int(obj.get("max_terms_per_chunk") or 18)
    max_new_tokens = int(obj.get("max_new_tokens") or 256)

    allowed_predicates = obj.get("allowed_predicates")
    if allowed_predicates is not None:
        if not isinstance(allowed_predicates, list) or not all(isinstance(p, str) for p in allowed_predicates):
            raise RuntimeError("'allowed_predicates' must be a list of strings when provided.")
        allowed_predicates_set: Optional[Set[str]] = {p.strip() for p in allowed_predicates if p.strip()}
    else:
        allowed_predicates_set = None

    return PromptConfig(
        system_prompt=system_prompt.strip(),
        few_shots=few_shots,
        max_terms_per_chunk=max_terms_per_chunk,
        max_new_tokens=max_new_tokens,
        allowed_predicates=allowed_predicates_set,
    )


# =============================================================================
# JSON brace-matching extraction
# =============================================================================

def extract_first_json_object(text: str) -> Optional[str]:
    """
    Extract the first top-level JSON object by brace matching.
    Handles leading/trailing junk (LLM sometimes prints extra text).
    """
    if not text:
        return None

    in_str = False
    esc = False
    depth = 0
    start = None

    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    return text[start : i + 1]

    return None


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    return tok, model, device


# =============================================================================
# DB schema + indexes
# =============================================================================

def ensure_tables(conn: sqlite3.Connection, out_table: str, runs_table: str) -> None:
    cur = conn.cursor()

    # Output table with provenance (doc_id, chunk_id) + raw_json for auditability
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            edge_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id         TEXT NOT NULL,
            chunk_id       TEXT NOT NULL,
            subject        TEXT NOT NULL,
            predicate      TEXT NOT NULL,
            object         TEXT NOT NULL,
            relation_type  TEXT NOT NULL,
            justification  TEXT,
            raw_json       TEXT,
            UNIQUE(doc_id, chunk_id, subject, predicate, object)
        )
        """
    )

    # Runs table: mark each (doc_id,chunk_id) as done even if 0 edges.
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {runs_table} (
            doc_id        TEXT NOT NULL,
            chunk_id      TEXT NOT NULL,
            rowid_src     INTEGER,
            status        TEXT NOT NULL,   -- 'done' | 'error'
            processed_at  TEXT NOT NULL,
            error_msg     TEXT,
            PRIMARY KEY (doc_id, chunk_id)
        )
        """
    )

    # Indexes for stability/perf
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{out_table}_doc_chunk ON {out_table}(doc_id, chunk_id)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{out_table}_pred ON {out_table}(predicate)")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{runs_table}_status ON {runs_table}(status)")
    conn.commit()


# =============================================================================
# Fetch chunks that are not processed (resume)
# =============================================================================

def fetch_unprocessed_chunks(
    conn: sqlite3.Connection,
    input_table: str,
    doc_col: str,
    chunk_col: str,
    text_col: str,
    runs_table: str,
    offset_rowid: int,
    max_chunks: int,
) -> List[Tuple[int, str, str, str]]:
    """
    Returns rows: (rowid, doc_id, chunk_id, text) for chunks not yet in runs_table (status=done/error).
    Using runs_table is safer than checking out_table existence, because a chunk may legitimately yield 0 edges.
    """
    cur = conn.cursor()

    lim_sql = ""
    params: List[Any] = [offset_rowid]
    if max_chunks and max_chunks > 0:
        lim_sql = "LIMIT ?"
        params.append(max_chunks)

    sql = f"""
    SELECT cc.rowid, cc.{doc_col}, cc.{chunk_col}, cc.{text_col}
    FROM {input_table} AS cc
    WHERE cc.rowid > ?
      AND NOT EXISTS (
          SELECT 1 FROM {runs_table} r
          WHERE r.doc_id = cc.{doc_col} AND r.chunk_id = cc.{chunk_col}
      )
    ORDER BY cc.rowid
    {lim_sql}
    """
    cur.execute(sql, params)
    return cur.fetchall()


# =============================================================================
# Candidate terms per chunk (from llm_terms_final + term_enrich_final)
# =============================================================================

def fetch_terms_for_chunk(
    conn: sqlite3.Connection,
    terms_table: str,
    enrich_table: str,
    doc_id: str,
    chunk_id: str,
    max_terms: int,
) -> List[Dict[str, str]]:
    """
    Return term list with enrichment metadata.
    Filters:
      - is_hpc_domain = 1
      - ontology_role != 'drop'
      - category != 'non_domain'
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            e.term,
            COALESCE(e.category, 'other_hpc') as category,
            COALESCE(e.definition, '') as definition,
            COALESCE(e.ontology_role, 'unknown') as ontology_role,
            COALESCE(e.dul_bucket, 'unknown') as dul_bucket
        FROM {terms_table} t
        JOIN {enrich_table} e
          ON e.term_id = t.term_id
        WHERE t.doc_id = ?
          AND t.chunk_id = ?
          AND COALESCE(e.is_hpc_domain, 0) = 1
          AND COALESCE(e.category, '') <> 'non_domain'
          AND COALESCE(e.ontology_role, 'unknown') <> 'drop'
        ORDER BY t.freq_total DESC, e.term ASC
        """,
        (doc_id, chunk_id),
    )
    rows = cur.fetchall()

    out: List[Dict[str, str]] = []
    seen = set()
    for term, cat, defin, role, dul in rows:
        term = (term or "").strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "term": term,
                "category": (cat or "other_hpc").strip(),
                "definition": (defin or "").strip(),
                "ontology_role": (role or "unknown").strip(),
                "dul_bucket": (dul or "unknown").strip(),
            }
        )
        if max_terms and len(out) >= max_terms:
            break

    return out


# =============================================================================
# Prompt building
# =============================================================================

def build_prompt(cfg: PromptConfig, chunk_text: str, terms: List[Dict[str, str]]) -> str:
    # Few-shot blocks from YAML
    ex_blocks: List[str] = []
    for i, ex in enumerate(cfg.few_shots, start=1):
        ex_json = json.dumps(ex["json"], ensure_ascii=False, indent=2)
        ex_terms = ", ".join(ex["terms"])
        ex_blocks.append(
            f"Example {i}:\n"
            f"TEXT:\n{ex['text']}\n"
            f"TERMS_IN_CHUNK:\n{ex_terms}\n"
            f"CORRECT_JSON:\n{ex_json}\n"
        )
    examples_str = "\n".join(ex_blocks)

    term_lines: List[str] = []
    for t in terms:
        line = f"- {t['term']} [category={t['category']}, role={t['ontology_role']}, dul={t['dul_bucket']}]"
        if t["definition"]:
            line += f" – {t['definition']}"
        term_lines.append(line)
    terms_block = "\n".join(term_lines)

    user_content = (
        "You will receive a chunk of HPC scheduler documentation and a list of DOMAIN TERMS.\n"
        "Each term includes category/role/bucket and a short definition.\n\n"
        "Extract ONLY NON-TAXONOMIC relations using ONLY the provided terms as subject/object.\n\n"
        f"{examples_str}\n"
        "Now process the NEW chunk.\n\n"
        f"NEW_TEXT:\n{chunk_text}\n\n"
        "DOMAIN TERMS (use ONLY these as subject/object):\n"
        f"{terms_block}\n\n"
        "Return ONLY one JSON object with a single key 'relations'."
    )

    return (
        f"<s>[INST] <<SYS>>\n{cfg.system_prompt}\n<</SYS>>\n\n"
        f"{user_content}\n"
        "[/INST]"
    )


# =============================================================================
# LLM call
# =============================================================================

def call_llm(tok, model, device: str, prompt: str, max_new_tokens: int) -> str:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )

    gen_only = out[0, input_ids.shape[-1] :]
    return tok.decode(gen_only, skip_special_tokens=True)


# =============================================================================
# Parse + validate output
# =============================================================================

def validate_predicate(pred: str, allowed_predicates: Optional[Set[str]]) -> bool:
    pred = pred.strip()
    if not pred:
        return False
    if not PREDICATE_RE.match(pred):
        return False
    if allowed_predicates is not None and pred not in allowed_predicates:
        return False
    return True


def parse_relations(
    raw_output: str,
    allowed_terms: Set[str],
    allowed_predicates: Optional[Set[str]],
) -> List[Dict[str, str]]:
    """
    Parse LLM output using brace-matching extraction, then strict validation:
    - subject/object must be exactly from allowed_terms
    - predicate passes validation (regex + optional allowlist)
    - relation_type must be in RELATION_TYPE_ENUM
    - drop duplicates, drop self-edges
    """
    json_obj = extract_first_json_object(raw_output)
    if not json_obj:
        return []

    try:
        data = json.loads(json_obj)
    except Exception:
        return []

    if not isinstance(data, dict):
        return []
    rels = data.get("relations", [])
    if not isinstance(rels, list):
        return []

    out: List[Dict[str, str]] = []
    seen: Set[Tuple[str, str, str]] = set()

    for item in rels:
        if not isinstance(item, dict):
            continue

        subj = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        rtype = str(item.get("relation_type", "")).strip() or "other"
        just = str(item.get("justification", "")).strip()

        if not subj or not pred or not obj:
            continue

        # exact match constraint
        if subj not in allowed_terms or obj not in allowed_terms:
            continue

        if subj.lower() == obj.lower():
            continue

        if rtype not in RELATION_TYPE_ENUM:
            continue

        if not validate_predicate(pred, allowed_predicates):
            continue

        key = (subj.lower(), pred.lower(), obj.lower())
        if key in seen:
            continue
        seen.add(key)

        out.append(
            {
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "relation_type": rtype,
                "justification": just,
            }
        )

    return out


# =============================================================================
# Main loop (runs table resume)
# =============================================================================

def mark_run(
    conn: sqlite3.Connection,
    runs_table: str,
    doc_id: str,
    chunk_id: str,
    rowid_src: int,
    status: str,
    error_msg: Optional[str] = None,
) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT OR REPLACE INTO {runs_table} (doc_id, chunk_id, rowid_src, status, processed_at, error_msg)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            chunk_id,
            rowid_src,
            status,
            datetime.utcnow().isoformat(timespec="seconds"),
            error_msg,
        ),
    )


def insert_edges(
    conn: sqlite3.Connection,
    out_table: str,
    doc_id: str,
    chunk_id: str,
    rels: List[Dict[str, str]],
    raw_json: str,
) -> None:
    cur = conn.cursor()
    for r in rels:
        cur.execute(
            f"""
            INSERT OR IGNORE INTO {out_table}
              (doc_id, chunk_id, subject, predicate, object, relation_type, justification, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                chunk_id,
                r["subject"],
                r["predicate"],
                r["object"],
                r["relation_type"],
                r["justification"],
                raw_json,
            ),
        )


def run(
    db_path: str,
    model_id: str,
    prompt_config_path: str,
    input_table: str,
    doc_id_col: str,
    chunk_id_col: str,
    text_col: str,
    terms_table: str,
    enrich_table: str,
    out_table: str,
    runs_table: str,
    max_chunks: int,
    offset_rowid: int,
    debug_first_chunk: bool,
    require_gpu: bool,
    commit_every: int,
) -> None:
    if require_gpu and not torch.cuda.is_available():
        raise SystemExit("[ERROR] --require-gpu set but CUDA is not available.")

    cfg = load_prompt_config(prompt_config_path)

    conn = sqlite3.connect(db_path)
    try:
        ensure_tables(conn, out_table=out_table, runs_table=runs_table)
        tok, model, device = load_model(model_id)

        rows = fetch_unprocessed_chunks(
            conn=conn,
            input_table=input_table,
            doc_col=doc_id_col,
            chunk_col=chunk_id_col,
            text_col=text_col,
            runs_table=runs_table,
            offset_rowid=offset_rowid,
            max_chunks=max_chunks,
        )

        print(f"[INFO] Unprocessed chunks: {len(rows)} (offset_rowid={offset_rowid})")

        processed = 0
        for idx, (rowid, doc_id, chunk_id, text) in enumerate(rows, start=1):
            terms = fetch_terms_for_chunk(
                conn=conn,
                terms_table=terms_table,
                enrich_table=enrich_table,
                doc_id=str(doc_id),
                chunk_id=str(chunk_id),
                max_terms=cfg.max_terms_per_chunk,
            )

            # even if <2 terms, mark done (important for resume correctness)
            if len(terms) < 2:
                mark_run(conn, runs_table, str(doc_id), str(chunk_id), int(rowid), "done", None)
                processed += 1
                if commit_every <= 1 or processed % commit_every == 0:
                    conn.commit()
                if debug_first_chunk:
                    print(f"[DEBUG] rowid={rowid} has <2 terms; marked done.")
                    return
                continue

            prompt = build_prompt(cfg, chunk_text=str(text), terms=terms)
            raw = call_llm(tok, model, device, prompt, max_new_tokens=cfg.max_new_tokens)

            allowed_terms = {t["term"] for t in terms}
            rels = parse_relations(raw, allowed_terms=allowed_terms, allowed_predicates=cfg.allowed_predicates)

            if debug_first_chunk:
                print(f"\n[DEBUG] rowid={rowid}, doc_id={doc_id}, chunk_id={chunk_id}")
                print("\n=== RAW OUTPUT (first 900 chars) ===")
                print(raw[:900])
                print("\n=== KEPT RELATIONS ===")
                if not rels:
                    print("(none)")
                else:
                    for r in rels:
                        print(f"- {r['subject']} --{r['predicate']}--> {r['object']} [{r['relation_type']}] ({r['justification']})")
                print("\n[DEBUG] Not writing to DB in --debug-first-chunk mode.")
                return

            try:
                json_obj = extract_first_json_object(raw) or ""
                insert_edges(conn, out_table, str(doc_id), str(chunk_id), rels, raw_json=json_obj)
                mark_run(conn, runs_table, str(doc_id), str(chunk_id), int(rowid), "done", None)
            except Exception as e:
                mark_run(conn, runs_table, str(doc_id), str(chunk_id), int(rowid), "error", str(e))

            processed += 1
            if commit_every <= 1 or processed % commit_every == 0:
                conn.commit()

            if idx == 1 or idx % 10 == 0:
                print(f"[INFO] chunk {idx}/{len(rows)} rowid={rowid} terms={len(terms)} kept_rels={len(rels)}")

        conn.commit()
        print("[DONE] Non-taxonomy LLM extraction completed.")
    finally:
        conn.close()


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="OLAF-LLM: non-taxonomic relation extraction.")

    ap.add_argument("--db", required=True, help="SQLite DB path")
    ap.add_argument("--model-id", required=True, help="HF model id (e.g., mistralai/Mistral-7B-Instruct-v0.3)")
    ap.add_argument("--prompt-config", default="prompts/non_tax_llm.yaml")

    ap.add_argument("--input-table", default="contextual_chunk")
    ap.add_argument("--doc-id-col", default="doc_id")
    ap.add_argument("--chunk-id-col", default="chunk_id")
    ap.add_argument("--text-col", default="text")

    ap.add_argument("--terms-table", default="llm_terms_final")
    ap.add_argument("--enrich-table", default="llm_enrich_final")

    ap.add_argument("--out-table", default="non_tax_llm")
    ap.add_argument("--runs-table", default="non_tax_llm_runs")

    ap.add_argument("--max-chunks", type=int, default=0, help="0=all; else limit")
    ap.add_argument("--offset-rowid", type=int, default=0)
    ap.add_argument("--debug-first-chunk", action="store_true")

    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--commit-every", type=int, default=50)

    args = ap.parse_args()

    run(
        db_path=args.db,
        model_id=args.model_id,
        prompt_config_path=args.prompt_config,
        input_table=args.input_table,
        doc_id_col=args.doc_id_col,
        chunk_id_col=args.chunk_id_col,
        text_col=args.text_col,
        terms_table=args.terms_table,
        enrich_table=args.enrich_table,
        out_table=args.out_table,
        runs_table=args.runs_table,
        max_chunks=args.max_chunks,
        offset_rowid=args.offset_rowid,
        debug_first_chunk=args.debug_first_chunk,
        require_gpu=args.require_gpu,
        commit_every=args.commit_every,
    )


if __name__ == "__main__":
    main()
