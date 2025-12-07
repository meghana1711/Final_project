import sqlite3
import json
import ast
import re
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Basic config
# ---------------------------------------------------------------------------

DB_PATH = "onto_db/olaf_sample_llm.db"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Make GPU math a bit faster (safe on L4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


SYSTEM_PROMPT = """\
You are an expert in High Performance Computing (HPC) and job schedulers like SLURM and IBM LSF.
Your task is STRICT TERM EXTRACTION.

Given a technical chunk of documentation, extract the most important DOMAIN TERMS.
Focus on:
- scheduler names, job states, partitions, queues
- configuration parameters and options (e.g. Slurm options, LSF keywords)
- CLI commands and flags (sbatch, srun, bsub, --partition, -N, -n)
- resource types (nodes, GPUs, CPUs, memory, cores)
- log file names and config file names

Rules:
- You MUST answer in valid JSON only (no explanations, no prose).
- JSON schema:
  { "terms": [ { "term": "string", "reason": "very short reason" } ] }
- Use each unique term only once per chunk.
- Ignore pure numbers, timestamps, and random punctuation.
"""


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def init_llm_terms_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_terms (
            term_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id    TEXT    NOT NULL,
            chunk_id  TEXT    NOT NULL,
            term      TEXT    NOT NULL,
            reason    TEXT,
            UNIQUE(doc_id, chunk_id, term)
        )
        """
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_mistral():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Ensure pad token exists and is configured
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model, device


# ---------------------------------------------------------------------------
# Prompt building (no chat template)
# ---------------------------------------------------------------------------

def build_prompt(text: str, max_terms: int = 20) -> str:
    """
    Build a single prompt string in Mistral [INST] format.
    We do NOT use apply_chat_template.
    """
    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"Read the following HPC documentation chunk and extract up to {max_terms} domain terms. "
        f"Return ONLY a JSON object following the schema above.\n\n"
        f"Text:\n{text}\n\n"
        "Remember: output only valid JSON, no extra text.\n"
        "[/INST]"
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(tokenizer, model, device: str, text: str) -> str:
    prompt = build_prompt(text)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=96,  
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only NEW tokens (exclude the prompt part)
    gen_only_ids = generated_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_only_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_terms(raw_output: str) -> List[Dict[str, Any]]:
    """
    Extract terms from the model output.

    Strategy:
    1) Try to find JSON objects that contain a "terms" key and parse them using
       json / ast.literal_eval (handles proper JSON or Python dict style).
    2) If that fails (e.g., truncated / invalid JSON), fall back to a simple
       regex that extracts all quoted strings after "terms" and treats them as terms.

    Accepts:
      { "terms": [ "Slurm", "sacct", ... ] }
      and
      { "terms": [ {"term": "...", "reason": "..."}, ... ] }
    """

    # ----------------------
    # 1) JSON-style parsing
    # ----------------------
    candidates: List[str] = []
    search_pos = 0
    key = '"terms"'  # just the key, not '{"terms"'

    while True:
        idx = raw_output.find(key, search_pos)
        if idx == -1:
            break

        # Find nearest '{' before "terms"
        start = raw_output.rfind("{", 0, idx)
        if start == -1:
            search_pos = idx + len(key)
            continue

        # Walk forward from that '{' and match braces
        depth = 0
        end = None
        for i, ch in enumerate(raw_output[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            # no matching closing brace; JSON is incomplete
            break

        candidates.append(raw_output[start:end])
        search_pos = end

    if candidates:
        # Try from the last candidate backwards (most likely the actual answer)
        for json_str in reversed(candidates):
            data = None

            # Try strict JSON first
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Try Python dict-style (single quotes, etc.)
                try:
                    data = ast.literal_eval(json_str)
                except Exception:
                    data = None

            if not isinstance(data, dict):
                continue

            items = data.get("terms", [])
            cleaned: List[Dict[str, Any]] = []
            seen = set()

            for item in items:
                if isinstance(item, str):
                    term = item.strip()
                    reason = ""
                elif isinstance(item, dict):
                    term = str(item.get("term", "")).strip()
                    reason = str(item.get("reason", "")).strip()
                else:
                    continue

                if not term:
                    continue

                # drop pure numbers
                t = term.replace(".", "").replace(",", "")
                if t.isdigit():
                    continue

                key_term = term.lower()
                if key_term in seen:
                    continue

                seen.add(key_term)
                cleaned.append({"term": term, "reason": reason})

            if cleaned:
                return cleaned

    # -----------------------------------------
    # 2) Fallback: regex over quoted strings
    # -----------------------------------------
    idx = raw_output.find('"terms"')
    if idx == -1:
        return []

    # Only look at the substring starting from "terms"
    sub = raw_output[idx:]

    # Grab all quoted strings
    matches = re.findall(r'"([^"]+)"', sub)
    if not matches:
        return []

    # First match should be the key "terms", skip it
    values = matches[1:]

    cleaned: List[Dict[str, Any]] = []
    seen = set()

    for term in values:
        term = term.strip()
        if not term:
            continue

        t = term.replace(".", "").replace(",", "")
        if t.isdigit():
            continue

        key_term = term.lower()
        if key_term in seen:
            continue

        seen.add(key_term)
        cleaned.append({"term": term, "reason": ""})

    return cleaned


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_chunks(
    conn: sqlite3.Connection,
    tokenizer,
    model,
    device: str,
    max_chunks: Optional[int] = None,
) -> None:
    init_llm_terms_table(conn)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT c.chunk_id, c.doc_id, c.text
        FROM contextual_chunk AS c
        LEFT JOIN llm_terms AS t
          ON t.chunk_id = c.chunk_id
        WHERE t.chunk_id IS NULL
        ORDER BY c.doc_id, c.chunk_id
        """
    )

    rows = cur.fetchall()
    if max_chunks is not None:
        rows = rows[:max_chunks]

    total = len(rows)
    print(f"Processing {total} chunks...")

    for idx, (chunk_id, doc_id, chunk_text) in enumerate(rows, start=1):
        if idx % 10 == 0 or idx == 1:
            print(f"  -> chunk {idx}/{total} (doc_id={doc_id}, chunk_id={chunk_id})")

        raw = call_llm(tokenizer, model, device, chunk_text)
        terms = parse_terms(raw)

        # Debug: show raw + parsed for the first chunk if parsing failed
        if idx == 1:
            print("\n=== DEBUG FIRST CHUNK RAW (first 600 chars) ===")
            print(raw[:600])
            print("\n=== DEBUG FIRST CHUNK PARSED TERMS ===")
            print(terms)
            print("------\n")

        for t in terms:
            cur.execute(
                """
                INSERT OR IGNORE INTO llm_terms (doc_id, chunk_id, term, reason)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, chunk_id, t["term"], t["reason"]),
            )
        conn.commit()

    print("Done.")


# ---------------------------------------------------------------------------
# Entry point with debug mode
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based term extraction over contextual_chunk.")
    parser.add_argument(
        "--debug-first-chunk",
        action="store_true",
        help="Run on a single chunk and print raw + parsed output.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit the number of chunks processed in this run.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        tokenizer, model, device = load_mistral()

        if args.debug_first_chunk:
            cur = conn.cursor()
            cur.execute(
                "SELECT chunk_id, doc_id, text FROM contextual_chunk ORDER BY doc_id, chunk_id LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                print("No rows in contextual_chunk.")
            else:
                chunk_id, doc_id, chunk_text = row
                print(f"DEBUG chunk_id={chunk_id}, doc_id={doc_id}")
                raw = call_llm(tokenizer, model, device, chunk_text)

                print("\n=== RAW OUTPUT (first 800 chars) ===")
                print(raw[:800])
                print("\n=== PARSED TERMS ===")
                terms = parse_terms(raw)
                for t in terms:
                    print("-", t)
                if not terms:
                    print("\n(No terms parsed)")
        else:
            process_chunks(conn, tokenizer, model, device, max_chunks=args.max_chunks)
    finally:
        conn.close()
