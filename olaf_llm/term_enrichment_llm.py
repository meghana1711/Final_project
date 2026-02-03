from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import yaml  # PyYAML


# =============================================================================
# Prompt config (YAML)
# =============================================================================

@dataclass
class PromptConfig:
    system_prompt: str
    prompt_mode: str = "few-shot"  # "few-shot" or "zero-shot"
    max_new_tokens: int = 240
    few_shot_examples: List[Dict[str, Any]] = None  # examples list

    @staticmethod
    def from_yaml(path: str) -> "PromptConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML prompt config must be a mapping/dict at top level.")

        system_prompt = data.get("system_prompt")
        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("YAML prompt config must contain a string key: system_prompt")

        prompt_mode = data.get("prompt_mode", "few-shot")
        if prompt_mode not in ("few-shot", "zero-shot"):
            raise ValueError("prompt_mode must be 'few-shot' or 'zero-shot'")

        few_shot_examples = data.get("few_shot_examples", []) or []
        if not isinstance(few_shot_examples, list):
            raise ValueError("few_shot_examples must be a list (or omitted).")

        # Light validation
        cleaned: List[Dict[str, Any]] = []
        for ex in few_shot_examples:
            if not isinstance(ex, dict):
                continue
            term = ex.get("term", "")
            ctx = ex.get("context", "")
            js = ex.get("json", None)
            if isinstance(term, str) and isinstance(ctx, str) and isinstance(js, dict):
                cleaned.append({"term": term, "context": ctx, "json": js})

        return PromptConfig(
            system_prompt=system_prompt,
            prompt_mode=prompt_mode,
            max_new_tokens=int(data.get("max_new_tokens", 240)),
            few_shot_examples=cleaned,
        )


# =============================================================================
# Enums + validation
# =============================================================================

SCHEDULER_ALLOWED = {"slurm", "lsf", "both", "generic", "unknown"}

CATEGORY_ALLOWED = {
    "scheduler",
    "command",
    "option_flag",
    "config_param",
    "config_file",
    "log_or_state_path",
    "queue_or_partition",
    "resource",
    "job_state",
    "user_role",
    "other_hpc",
    "non_domain",
}

ONTOLOGY_ROLE_ALLOWED = {"class", "object_property", "datatype_property", "individual", "drop", "unknown"}

DUL_BUCKET_ALLOWED = {"unknown", "information_object", "object", "description", "event"}


def clamp_enum(val: Any, allowed: set[str], default: str) -> str:
    s = str(val).strip().lower() if val is not None else ""
    return s if s in allowed else default


def to_bool_int(val: Any, default: int = 1) -> int:
    if val is None:
        return default
    if isinstance(val, bool):
        return 1 if val else 0
    if isinstance(val, (int, float)):
        return 1 if val != 0 else 0
    if isinstance(val, str):
        t = val.strip().lower()
        if t in ("true", "yes", "1", "y", "accept"):
            return 1
        if t in ("false", "no", "0", "n", "reject"):
            return 0
    return default


def canonicalize(term: str) -> str:
    # Conservative canonical form: lowercase + trim + collapse spaces
    t = (term or "").strip().lower()
    t = " ".join(t.split())
    return t


def clean_aliases(x: Any) -> str:
    # store JSON list string
    if not isinstance(x, list):
        x = []
    out: List[str] = []
    for a in x:
        s = str(a).strip()
        if s:
            out.append(s)
    return json.dumps(out, ensure_ascii=False)


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model, device


# =============================================================================
# Prompt building
# =============================================================================

def build_prefix(cfg: PromptConfig) -> str:
    if cfg.prompt_mode == "zero-shot":
        return f"<s>[INST] <<SYS>>\n{cfg.system_prompt}\n<</SYS>>\n\n"

    blocks: List[str] = []
    for i, ex in enumerate(cfg.few_shot_examples or [], start=1):
        blocks.append(
            "Example {}:\nTERM: {}\nCONTEXT:\n{}\nJSON:\n{}\n".format(
                i,
                ex["term"],
                ex["context"],
                json.dumps(ex["json"], ensure_ascii=False),
            )
        )
    examples_str = "\n".join(blocks)

    return (
        f"<s>[INST] <<SYS>>\n{cfg.system_prompt}\n<</SYS>>\n\n"
        "You will see examples of correct enrichments.\n"
        "Follow the same schema and strictness for the NEW term.\n\n"
        f"{examples_str}\n"
        "Now enrich ONLY the following NEW term.\n"
    )


def build_prompt(prefix: str, term: str, context: str) -> str:
    user = (
        "TERM:\n"
        f"{term}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "Respond ONLY with a single JSON object and nothing else."
    )
    return f"{prefix}{user}[/INST]"


def call_llm(tokenizer, model, device: str, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_only_ids = generated[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_only_ids, skip_special_tokens=True)


# =============================================================================
# Robust brace-matching JSON extraction
# =============================================================================

def extract_json_objects(text: str) -> List[str]:
    objs: List[str] = []
    start_positions: List[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            start_positions.append(i)

    for start in start_positions:
        depth = 0
        end = None
        for j in range(start, len(text)):
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end is not None:
            objs.append(text[start:end])

    return objs


def parse_enrich_output(raw: str, fallback_term: str) -> Dict[str, Any]:
    """
    Returns normalized dict ready for DB insert.
    """
    data: Dict[str, Any] = {}

    candidates = extract_json_objects(raw)
    for js in reversed(candidates):
        parsed: Any = None
        try:
            parsed = json.loads(js)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(js)
            except Exception:
                parsed = None

        if isinstance(parsed, dict):
            data = parsed
            break

    # Pull fields (with fallbacks)
    term = str(data.get("term", fallback_term)).strip() or fallback_term
    canonical = str(data.get("canonical", "")).strip()
    canonical = canonicalize(canonical) if canonical else canonicalize(term)

    is_hpc_domain = to_bool_int(data.get("is_hpc_domain", data.get("is_hpc_domain_term", 1)), default=1)

    scheduler = clamp_enum(data.get("scheduler", "unknown"), SCHEDULER_ALLOWED, "unknown")
    category = clamp_enum(data.get("category", "other_hpc"), CATEGORY_ALLOWED, "other_hpc")
    ontology_role = clamp_enum(data.get("ontology_role", "unknown"), ONTOLOGY_ROLE_ALLOWED, "unknown")
    dul_bucket = clamp_enum(data.get("dul_bucket", "unknown"), DUL_BUCKET_ALLOWED, "unknown")

    definition = str(
        data.get("definition", data.get("short_definition", ""))
    ).strip()

    aliases_json = clean_aliases(data.get("aliases", []))

    # If clearly non-domain, force conservative outputs
    if is_hpc_domain == 0 or category == "non_domain":
        is_hpc_domain = 0
        category = "non_domain"
        scheduler = "unknown" if scheduler not in {"slurm", "lsf", "both", "generic"} else scheduler
        ontology_role = "drop" if ontology_role == "unknown" else ontology_role
        dul_bucket = "unknown" if dul_bucket not in DUL_BUCKET_ALLOWED else dul_bucket

    return {
        "term": term,
        "canonical": canonical,
        "scheduler": scheduler,
        "ontology_role": ontology_role,
        "category": category,
        "dul_bucket": dul_bucket,
        "is_hpc_domain": is_hpc_domain,
        "definition": definition,
        "aliases_json": aliases_json,
        "raw_json": raw.strip(),
    }


# =============================================================================
# DB schema
# =============================================================================

def init_enrich_table(conn: sqlite3.Connection, table: str) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            term_id        INTEGER PRIMARY KEY,
            term           TEXT    NOT NULL,
            canonical      TEXT    NOT NULL,
            scheduler      TEXT    NOT NULL,
            ontology_role  TEXT    NOT NULL,
            category       TEXT    NOT NULL,
            dul_bucket     TEXT    NOT NULL,
            is_hpc_domain  INTEGER NOT NULL,
            definition     TEXT,
            aliases_json   TEXT,
            raw_json       TEXT,
            freq_total     INTEGER NOT NULL,
            FOREIGN KEY(term_id) REFERENCES llm_terms_final(term_id)
        )
        """
    )
    conn.commit()


def fetch_terms_to_enrich(
    conn: sqlite3.Connection,
    terms_table: str,
    enrich_table: str,
    min_freq: int,
    max_rows: int,
    offset_term_id: int,
) -> List[Tuple[int, str, str, str, int]]:
    """
    Returns list of (term_id, term, doc_id, chunk_id, freq_total) for terms that are not yet enriched.
    """
    cur = conn.cursor()

    sql = f"""
        SELECT t.term_id, t.term, t.doc_id, t.chunk_id, t.freq_total
        FROM {terms_table} t
        LEFT JOIN {enrich_table} e
          ON e.term_id = t.term_id
        WHERE e.term_id IS NULL
          AND t.term_id > ?
    """
    params: List[Any] = [offset_term_id]

    if min_freq and min_freq > 1:
        sql += " AND t.freq_total >= ?"
        params.append(min_freq)

    sql += " ORDER BY t.term_id"

    if max_rows and max_rows > 0:
        sql += " LIMIT ?"
        params.append(max_rows)

    cur.execute(sql, params)
    return cur.fetchall()


def fetch_context(conn: sqlite3.Connection, chunks_table: str, doc_id: str, chunk_id: str, text_col: str) -> str:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT {text_col}
        FROM {chunks_table}
        WHERE doc_id = ? AND chunk_id = ?
        """,
        (doc_id, chunk_id),
    )
    row = cur.fetchone()
    return (row[0] if row and row[0] else "") or ""


def insert_enrichment(
    conn: sqlite3.Connection,
    enrich_table: str,
    term_id: int,
    term: str,
    canonical: str,
    scheduler: str,
    ontology_role: str,
    category: str,
    dul_bucket: str,
    is_hpc_domain: int,
    definition: str,
    aliases_json: str,
    raw_json: str,
    freq_total: int,
) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT OR REPLACE INTO {enrich_table} (
            term_id, term, canonical, scheduler, ontology_role, category, dul_bucket,
            is_hpc_domain, definition, aliases_json, raw_json, freq_total
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            term_id, term, canonical, scheduler, ontology_role, category, dul_bucket,
            is_hpc_domain, definition, aliases_json, raw_json, freq_total
        ),
    )


# =============================================================================
# Main loop
# =============================================================================

def enrich_terms(
    conn: sqlite3.Connection,
    tokenizer,
    model,
    device: str,
    cfg: PromptConfig,
    model_id: str,
    terms_table: str,
    enrich_table: str,
    chunks_table: str,
    chunks_text_col: str,
    min_freq: int,
    max_rows: int,
    offset_term_id: int,
    debug_first: bool,
    commit_every: int,
) -> None:
    init_enrich_table(conn, enrich_table)

    prefix = build_prefix(cfg)

    rows = fetch_terms_to_enrich(
        conn=conn,
        terms_table=terms_table,
        enrich_table=enrich_table,
        min_freq=min_freq,
        max_rows=(1 if debug_first else max_rows),
        offset_term_id=offset_term_id,
    )

    total = len(rows)
    print(f"Enriching {total} terms from {terms_table} -> {enrich_table} (min_freq={min_freq})")
    if total == 0:
        return

    for i, (term_id, term, doc_id, chunk_id, freq_total) in enumerate(rows, start=1):
        if i == 1 or i % 10 == 0:
            print(f"  -> {i}/{total}: term_id={term_id}, term='{term}', freq_total={freq_total}")

        context = fetch_context(conn, chunks_table, doc_id, chunk_id, chunks_text_col)

        prompt = build_prompt(prefix, term=term, context=context)
        raw = call_llm(tokenizer, model, device, prompt, max_new_tokens=cfg.max_new_tokens)
        parsed = parse_enrich_output(raw, fallback_term=term)

        if debug_first:
            print("\n=== RAW OUTPUT (first 900 chars) ===")
            print(raw[:900])
            print("\n=== PARSED ===")
            print(parsed)
            return

        insert_enrichment(
            conn=conn,
            enrich_table=enrich_table,
            term_id=term_id,
            term=term,
            canonical=parsed["canonical"],
            scheduler=parsed["scheduler"],
            ontology_role=parsed["ontology_role"],
            category=parsed["category"],
            dul_bucket=parsed["dul_bucket"],
            is_hpc_domain=parsed["is_hpc_domain"],
            definition=parsed["definition"],
            aliases_json=parsed["aliases_json"],
            raw_json=parsed["raw_json"],
            freq_total=freq_total,
        )

        if commit_every > 0 and (i % commit_every == 0):
            conn.commit()

    conn.commit()
    print("Done.")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="OLAF-LLM: term enrichment from llm_terms_final into llm_enrich_final (YAML prompt).")

    ap.add_argument("--db", required=True, help="Path to SQLite DB.")
    ap.add_argument("--prompt-config", default="prompts/term_enrich_llm.yaml")
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF model id.")

    ap.add_argument("--terms-table", default="llm_terms_final", help="Input terms table")
    ap.add_argument("--enrich-table", default="llm_enrich_final", help="Output enrichment table.")
    ap.add_argument("--chunks-table", default="contextual_chunk", help="Chunks table for context lookups.")
    ap.add_argument("--chunks-text-col", default="text", help="Text column in chunks table.")

    ap.add_argument("--min-freq", type=int, default=2, help="Only enrich terms with freq_total >= min_freq (set 0/1 to disable).")
    ap.add_argument("--max-rows", type=int, default=0, help="0=all; otherwise limit number of terms enriched this run.")
    ap.add_argument("--offset-term-id", type=int, default=0, help="Only enrich terms with term_id > offset-term-id.")
    ap.add_argument("--commit-every", type=int, default=50, help="Commit every N terms (0 disables periodic commits).")

    ap.add_argument("--debug-first", action="store_true", help="Enrich only 1 term; print raw+parsed; no DB writes.")

    args = ap.parse_args()

    cfg = PromptConfig.from_yaml(args.prompt_config)

    conn = sqlite3.connect(args.db)
    try:
        tokenizer, model, device = load_model(args.model_id)

        enrich_terms(
            conn=conn,
            tokenizer=tokenizer,
            model=model,
            device=device,
            cfg=cfg,
            model_id=args.model_id,
            terms_table=args.terms_table,
            enrich_table=args.enrich_table,
            chunks_table=args.chunks_table,
            chunks_text_col=args.chunks_text_col,
            min_freq=args.min_freq,
            max_rows=args.max_rows,
            offset_term_id=args.offset_term_id,
            debug_first=args.debug_first,
            commit_every=args.commit_every,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
