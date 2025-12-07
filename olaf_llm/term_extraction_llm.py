# file: llm_term_extraction_run.py

import sqlite3
import json
import re
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DB_PATH = "onto_db/olaf_sample_llm.db"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# choose "zero-shot" or "few-shot"
PROMPT_MODE = "few-shot"

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
- Return ONLY valid JSON. No explanations, no prose.
- JSON format:
  {
    "terms": [
      {"term": "string", "reason": "very short reason"},
      ...
    ]
  }
- Use each unique term only once per chunk.
- Ignore pure numbers, timestamps, and random punctuation.
"""

# --- FEW-SHOT EXAMPLES -------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "text": "Submit SLURM jobs with sbatch. Use the --partition=gpu queue to request GPU nodes.",
        "json": {
            "terms": [
                {"term": "SLURM", "reason": "job scheduler name"},
                {"term": "sbatch", "reason": "SLURM command to submit batch jobs"},
                {"term": "--partition", "reason": "command-line option selecting a queue"},
                {"term": "gpu queue", "reason": "partition configured for GPU jobs"},
                {"term": "GPU nodes", "reason": "nodes with GPU resources"},
            ]
        },
    },
    {
        "text": "The bsub command in IBM LSF submits jobs to queues. Use -q normal or -q gpu to select the queue.",
        "json": {
            "terms": [
                {"term": "IBM LSF", "reason": "HPC job scheduler"},
                {"term": "bsub", "reason": "LSF command to submit jobs"},
                {"term": "-q", "reason": "bsub option to choose the queue"},
                {"term": "normal queue", "reason": "default LSF job queue"},
                {"term": "gpu queue", "reason": "LSF queue configured for GPU jobs"},
            ]
        },
    },
]


def init_llm_terms_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_terms (
            term_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id    INTEGER NOT NULL,
            chunk_id  INTEGER NOT NULL,
            term      TEXT    NOT NULL,
            reason    TEXT,
            UNIQUE(doc_id, chunk_id, term)
        )
        """
    )
    conn.commit()


def load_mistral():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # ✅ ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # good practice for causal LMs

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # ✅ let the model know about the pad token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model, device



# --- PROMPT BUILDING ---------------------------------------------------------

def build_messages_zero_shot(text: str, max_terms: int = 40):
    user_prompt = f"Extract up to {max_terms} domain terms from the following text:\n\n{text}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_messages_few_shot(text: str, max_terms: int = 40):
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for ex in FEW_SHOT_EXAMPLES:
        messages.append(
            {
                "role": "user",
                "content": f"Text:\n{ex['text']}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(ex["json"]),
            }
        )

    # Strong instruction for the real chunk
    messages.append(
        {
            "role": "user",
            "content": (
                f"Now process ONLY the following NEW text. "
                f"Return a SINGLE JSON object in the same format, and nothing else.\n\n"
                f"Text:\n{text}"
            ),
        }
    )
    return messages



def build_messages(text: str, max_terms: int = 40):
    if PROMPT_MODE == "few-shot":
        return build_messages_few_shot(text, max_terms=max_terms)
    else:
        return build_messages_zero_shot(text, max_terms=max_terms)


# --- LLM CALL (with attention_mask) -----------------------------------------

def call_llm(tokenizer, model, device, text: str) -> str:
    messages = build_messages(text)

    # 1) Get chat text via template
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 2) Tokenize with padding (now valid because pad_token is set)
    encoded = tokenizer(
        chat_text,
        return_tensors="pt",
        padding=True,
        truncation=True,      # optional, but safe
        max_length=4096,      # safe upper bound per chunk
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


# --- PARSING -----------------------------------------------------------------

def parse_terms(raw_output: str) -> List[Dict[str, Any]]:
    """
    Extract the *last* valid JSON object that starts with {"terms": ...}
    from the raw model output, then return its terms list.
    """

    candidates: List[str] = []
    search_pos = 0
    key = '{"terms"'

    while True:
        start = raw_output.find(key, search_pos)
        if start == -1:
            break

        # walk forward and match braces
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
            # no closing brace, stop
            break

        candidates.append(raw_output[start:end])
        search_pos = end

    if not candidates:
        return []

    # Try from the last candidate backwards (most likely the real answer)
    for json_str in reversed(candidates):
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        items = data.get("terms", [])
        cleaned: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            term = str(item.get("term", "")).strip()
            reason = str(item.get("reason", "")).strip()
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

    # no valid JSON with terms found
    return []


# --- MAIN LOOP ---------------------------------------------------------------

def process_chunks(conn: sqlite3.Connection, tokenizer, model, device) -> None:
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
    print(f"Processing {len(rows)} chunks...")

    for idx, (chunk_id, doc_id, chunk_text) in enumerate(rows, start=1):
        # small debug: print progress every 10 chunks
        if idx % 10 == 0:
            print(f"  -> chunk {idx}/{len(rows)} (doc_id={doc_id}, chunk_id={chunk_id})")

        raw = call_llm(tokenizer, model, device, chunk_text)
        terms = parse_terms(raw)

        # debug: if first few chunks give no terms, show the output once
        if idx <= 3 and not terms:
            print(f"[DEBUG] No terms parsed for first chunk (id={chunk_id}). Raw output snippet:")
            print(raw[:500])
            print("------")

        for t in terms:
            cur.execute(
                """
                INSERT OR IGNORE INTO llm_terms (doc_id, chunk_id, term, reason)
                VALUES (?, ?, ?, ?)
                """,
                (doc_id, chunk_id, t["term"], t["reason"]),
            )
        conn.commit()


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    try:
        tokenizer, model, device = load_mistral()
        process_chunks(conn, tokenizer, model, device)
    finally:
        conn.close()
