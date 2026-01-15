# file: olaf_llm/taxonomy_is_a_llm.py

from __future__ import annotations

import sqlite3
import json
import ast
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================================================================
# CONFIG
# =============================================================================

DB_PATH = "onto_db/onto_new.db"

# Recommended first try: Qwen 2.5 7B Instruct
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Alternative (smaller/faster): uncomment below and comment out Qwen if you want
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# GPU settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Max number of candidate terms per chunk to send to LLM
MAX_TERMS_PER_CHUNK = 16

# Justification phrases that indicate a NON-taxonomic relation
BAD_JUSTIFICATION_PATTERNS = [
    "consequence of",
    "result of",
    "caused by",
    "due to",
    "used in",
    "used by",
    "stored in",
    "written in",
    "written to",
    "logged to",
    "example of",
    "for example",
    "e.g.",
    "illustrate",
    "illustration",
    "sample of",
    "part of",
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert in High Performance Computing (HPC) and schedulers like SLURM and IBM LSF.
Your task is to identify ONLY TRUE "is-a" (subclass) relations between HPC terms.

You will be given:
- A short documentation CHUNK (text).
- A list of CANDIDATE TERMS that appear in or are relevant to that chunk.

Your job:
- Look for statements where one term is a more specific KIND of another term.
- Extract ONLY "X is a type of Y" relations (subclass / hyponym).
- Express these as "is_a_edges".

DEFINITION OF TRUE IS-A (SUBCLASS):
- "child" is-a "parent" if and only if, in HPC context, the sentence
  "Every CHILD is a PARENT" is generally true.
  Examples:
    - "Partition QOS is a specific type of QOS."
      → Every partition QOS is a QOS. This IS an is-a relation.
    - "Job QOS is another type of QOS."
      → Every job QOS is a QOS. This IS an is-a relation.
    - "The default partition is a partition."
      → Every default partition is a partition. This IS an is-a relation.

WHAT IS NOT IS-A (MUST BE EXCLUDED):
- part-of:
    - "Suspended jobs are part of the job queue."
      → This is a part_of relation, NOT is-a. Do NOT output.
    - "The active bitmap is maintained inside the gang scheduler logic."
      → storage/maintenance, NOT is-a. Do NOT output.
- used-in / used-by / stored-in / logged-to:
    - "Square brackets are used in node range expressions."
      → usage, NOT is-a. Do NOT output.
    - "slurmctld.log is written under /var/log/slurm."
      → logging, NOT is-a. Do NOT output.
- effect / consequence:
    - "Non-negligible delays are a consequence of lock contention."
      → effect_of, NOT is-a. Do NOT output.
- example-of / instance-of:
    - "Singularity is an example of an hpcng container runtime."
      → instance/example, NOT a subclass for this task. Do NOT output.
- purely numeric / value statements:
    - "0.5 seconds is an example timeout value."
      → numeric example, NOT an is-a relation.

If the text expresses ONLY part-of, used-in, example-of, consequence-of, or other
non-taxonomic relations, then you MUST output an empty list of is_a_edges.

OUTPUT FORMAT (STRICT):
You MUST output exactly ONE JSON object and nothing else.

The JSON schema is:

{
  "is_a_edges": [
    {
      "child": "more specific term",
      "parent": "more general term",
      "justification": "one or two sentences explaining why this is a true subclass relation"
    },
    ...
  ]
}

- child and parent MUST come from the candidate terms list (or obvious close variants).
- justification MUST explicitly reflect an "X is a type of Y" reading.
- If no valid is-a relations exist, return: { "is_a_edges": [] }.
"""


# =============================================================================
# FEW-SHOT EXAMPLES (GOOD AND BAD)
# =============================================================================

IS_A_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    # Example 1 – good is-a: Partition QOS and QOS
    {
        "chunk": (
            "Partition QOS is a specific type of QOS assigned to a partition. "
            "Job QOS is another type of QOS associated with individual jobs."
        ),
        "candidate_terms": [
            "Partition QOS",
            "QOS",
            "Job QOS",
            "partition",
            "job",
        ],
        "json": {
            "is_a_edges": [
                {
                    "child": "Partition QOS",
                    "parent": "QOS",
                    "justification": "The text explicitly states that Partition QOS is a specific type of QOS."
                },
                {
                    "child": "Job QOS",
                    "parent": "QOS",
                    "justification": "The text describes Job QOS as another type of QOS."
                }
            ]
        },
    },
    # Example 2 – good is-a: default partition, partition
    {
        "chunk": (
            "The default partition is the partition used when users do not specify one explicitly. "
            "Each partition represents a group of compute nodes with shared limits and QOS."
        ),
        "candidate_terms": [
            "default partition",
            "partition",
            "compute nodes",
            "QOS",
        ],
        "json": {
            "is_a_edges": [
                {
                    "child": "default partition",
                    "parent": "partition",
                    "justification": "The default partition is described as a particular partition used when none is specified, so it is a specific kind of partition."
                }
            ]
        },
    },
    # Example 3 – good is-a: slurmctld daemon, slurmctld
    {
        "chunk": (
            "The slurmctld daemon is the main Slurm controller process responsible for managing job queues and "
            "distributing work to slurmd on compute nodes."
        ),
        "candidate_terms": [
            "slurmctld daemon",
            "slurmctld",
            "slurmd",
            "compute nodes",
            "job queues",
        ],
        "json": {
            "is_a_edges": [
                {
                    "child": "slurmctld daemon",
                    "parent": "slurmctld",
                    "justification": "The text refers to the slurmctld daemon as the controller process, making it a specific form of slurmctld."
                }
            ]
        },
    },
    # Example 4 – BAD example: part-of and usage only (must produce empty list)
    {
        "chunk": (
            "Suspended jobs are part of the job queue, as they are tracked within it. "
            "The active bitmap is maintained inside the gang scheduler logic, which itself is part of the job queue."
        ),
        "candidate_terms": [
            "suspended jobs",
            "job queue",
            "active bitmap",
            "gang scheduler logic",
        ],
        "json": {
            "is_a_edges": []
        },
    },
    # Example 5 – BAD example: consequence-of (must produce empty list)
    {
        "chunk": (
            "Non-negligible delays are a consequence of increased lock contention on the slurmctld. "
            "These delays affect how quickly jobs move from pending to running."
        ),
        "candidate_terms": [
            "non-negligible delays",
            "lock contention",
            "slurmctld",
            "pending",
            "running",
        ],
        "json": {
            "is_a_edges": []
        },
    },
    # Example 6 – BAD example: used-in / syntax (must produce empty list)
    {
        "chunk": (
            "Square brackets are used in node range expressions to specify multiple nodes, such as node[01-08]. "
            "This syntax helps users submit jobs to many nodes at once."
        ),
        "candidate_terms": [
            "square brackets",
            "node range expressions",
            "nodes",
            "jobs",
        ],
        "json": {
            "is_a_edges": []
        },
    },
]


# =============================================================================
# DB HELPERS
# =============================================================================

def init_is_a_table(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_is_a_qwen (
            edge_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id        TEXT NOT NULL,
            chunk_id      TEXT NOT NULL,
            child_term    TEXT NOT NULL,
            parent_term   TEXT NOT NULL,
            justification TEXT
        )
        """
    )
    conn.commit()


def load_llm():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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


def fetch_chunks_for_is_a(
    conn: sqlite3.Connection,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
) -> List[tuple]:
    """
    Fetch contextual_chunk rows that still need is-a extraction.
    Skip chunks where we already have at least one is-a edge for that (doc_id, chunk_id).
    """
    init_is_a_table(conn)
    cur = conn.cursor()

    sql = """
        SELECT cc.rowid, cc.doc_id, cc.chunk_id, cc.text
        FROM contextual_chunk AS cc
        WHERE cc.rowid > ?
          AND NOT EXISTS (
              SELECT 1 FROM llm_is_a_qwen e
              WHERE e.doc_id = cc.doc_id AND e.chunk_id = cc.chunk_id
          )
        ORDER BY cc.rowid
    """
    params = [offset_rowid]
    if max_chunks is not None:
        sql += " LIMIT ?"
        params.append(max_chunks)

    cur.execute(sql, params)
    return cur.fetchall()


def fetch_candidate_terms_for_chunk(
    conn: sqlite3.Connection,
    doc_id: str,
    chunk_id: str,
) -> List[str]:
    """
    Get candidate terms for a given chunk from llm_terms + llm_enrich.

    Heuristics:
    - Use llm_terms.term for that (doc_id, chunk_id).
    - LEFT JOIN llm_enrich on canonical_term.
    - Keep only is_hpc_domain=1 and category != 'non_domain' where enrichment exists.
    - Skip obviously tiny junk terms (< 3 letters after removing non-letters).
    """
    cur = conn.cursor()

    cur.execute(
        """
        SELECT DISTINCT
            COALESCE(e.example_term, t.term) AS term
        FROM llm_terms t
        LEFT JOIN llm_enrich e
          ON LOWER(TRIM(t.term)) = e.canonical_term
        WHERE t.doc_id = ?
          AND t.chunk_id = ?
          AND (
              e.canonical_term IS NULL
              OR e.is_hpc_domain = 1
          )
          AND (
              e.category IS NULL
              OR e.category <> 'non_domain'
          )
        """,
        (doc_id, chunk_id),
    )

    raw_terms = [row[0] for row in cur.fetchall()]

    cleaned: List[str] = []
    seen = set()
    for term in raw_terms:
        t = (term or "").strip()
        if not t:
            continue

        letters = "".join(ch for ch in t if ch.isalpha())
        if len(letters) < 3:
            continue

        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)

    return cleaned[:MAX_TERMS_PER_CHUNK]


# =============================================================================
# PROMPT + LLM CALL (USING CHAT TEMPLATE)
# =============================================================================

def build_is_a_messages(chunk_text: str, candidate_terms: List[str]) -> List[Dict[str, str]]:
    """
    Build messages for chat-style models (Qwen, DeepSeek-R1-distill, Mistral, etc.).
    """
    example_blocks = []
    for i, ex in enumerate(IS_A_FEW_SHOT_EXAMPLES, start=1):
        example_blocks.append(
            "Example {}:\nCHUNK:\n{}\n\nCANDIDATE_TERMS:\n{}\n\nJSON:\n{}\n".format(
                i,
                ex["chunk"],
                ", ".join(ex["candidate_terms"]),
                json.dumps(ex["json"], ensure_ascii=False, indent=2),
            )
        )
    examples_str = "\n\n".join(example_blocks)

    terms_bullet = "\n".join(f"- {t}" for t in candidate_terms)

    user_content = (
        "You are given a short HPC documentation CHUNK and a list of CANDIDATE TERMS.\n"
        "Your job is to output ONLY true is-a (subclass) relations between those terms.\n\n"
        "Here are some examples of CORRECT and INCORRECT behavior:\n\n"
        f"{examples_str}\n\n"
        "Now process the NEW CHUNK below.\n\n"
        "CHUNK:\n"
        f"{chunk_text}\n\n"
        "CANDIDATE_TERMS:\n"
        f"{terms_bullet}\n\n"
        "Return exactly one JSON object with the key \"is_a_edges\" as described in the system prompt.\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


def call_is_a_llm(
    tokenizer,
    model,
    device: str,
    chunk_text: str,
    candidate_terms: List[str],
) -> str:
    messages = build_is_a_messages(chunk_text, candidate_terms)

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_only = generated_ids[0, encoded["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True)


# =============================================================================
# PARSING + HEURISTIC FILTERING
# =============================================================================

def _is_lexical_subtype(child: str, parent: str) -> bool:
    """
    Heuristic: parent should look like the head / more general noun of the child.

    - parent appearing as a substring in child (case-insensitive) but not identical
    - or child ends with parent

    This strongly favors patterns like:
      "default partition" -> "partition"
      "Partition QOS"     -> "QOS"
      "slurmctld daemon"  -> "slurmctld"
    """
    c = child.lower().strip()
    p = parent.lower().strip()
    if not c or not p or c == p:
        return False

    if p in c:
        return True
    if c.endswith(p):
        return True
    return False


def _fails_justification_rules(justification: str) -> bool:
    j = justification.lower()
    for pattern in BAD_JUSTIFICATION_PATTERNS:
        if pattern in j:
            return True
    return False


def parse_is_a_output(raw_output: str) -> List[Dict[str, str]]:
    """
    Parse the LLM JSON and apply heuristic filters.
    Returns list of dicts with keys: child, parent, justification.
    """
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []

    json_str = raw_output[start : end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(json_str)
        except Exception:
            return []

    if not isinstance(data, dict):
        return []

    edges = data.get("is_a_edges", [])
    if not isinstance(edges, list):
        return []

    cleaned: List[Dict[str, str]] = []
    seen_pairs = set()

    for item in edges:
        if not isinstance(item, dict):
            continue
        child = str(item.get("child", "")).strip()
        parent = str(item.get("parent", "")).strip()
        justification = str(item.get("justification", "")).strip()

        if not child or not parent:
            continue
        if child.lower() == parent.lower():
            continue

        # reject obviously non-taxonomic based on justification
        if _fails_justification_rules(justification):
            continue

        # enforce lexical subtype pattern
        if not _is_lexical_subtype(child, parent):
            continue

        key = (child.lower(), parent.lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        cleaned.append(
            {
                "child": child,
                "parent": parent,
                "justification": justification,
            }
        )

    return cleaned


# =============================================================================
# MAIN PROCESSING LOOP
# =============================================================================

def process_chunks(
    conn: sqlite3.Connection,
    tokenizer,
    model,
    device: str,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
    debug_first: bool = False,
) -> None:
    init_is_a_table(conn)

    rows = fetch_chunks_for_is_a(conn, max_chunks=max_chunks, offset_rowid=offset_rowid)
    total = len(rows)
    print(f"Processing {total} chunks for is-a taxonomy (offset_rowid={offset_rowid})...")

    if total == 0:
        return

    cur = conn.cursor()

    for idx, (rowid, doc_id, chunk_id, chunk_text) in enumerate(rows, start=1):
        candidate_terms = fetch_candidate_terms_for_chunk(conn, doc_id, chunk_id)

        if len(candidate_terms) < 2:
            continue

        if debug_first and idx > 1:
            break

        if idx == 1 or idx % 10 == 0:
            print(
                f"  -> chunk {idx}/{total} "
                f"(rowid={rowid}, doc_id={doc_id}, chunk_id={chunk_id}, {len(candidate_terms)} candidate terms)"
            )

        raw = call_is_a_llm(tokenizer, model, device, chunk_text, candidate_terms)
        edges = parse_is_a_output(raw)

        if debug_first:
            print(
                f"\nDEBUG rowid={rowid}, doc_id={doc_id}, chunk_id={chunk_id}, "
                f"{len(candidate_terms)} candidate terms"
            )
            print("\n=== RAW OUTPUT (first 800 chars) ===")
            print(raw[:800])
            print("\n=== PARSED IS-A EDGES ===")
            if edges:
                for e in edges:
                    print(f"- {e['child']}  ->  {e['parent']}  ({e['justification']})")
            else:
                print("(No is-a edges kept after filtering)")
            return

        for e in edges:
            cur.execute(
                """
                INSERT INTO llm_is_a_qwen (
                    doc_id, chunk_id, child_term, parent_term, justification
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc_id, chunk_id, e["child"], e["parent"], e["justification"]),
            )
        conn.commit()

    print("is-a taxonomy extraction completed.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based is-a taxonomy extraction over contextual_chunk (HPC docs)."
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit the number of chunks processed in this run.",
    )
    parser.add_argument(
        "--offset-rowid",
        type=int,
        default=0,
        help="Start from contextual_chunk.rowid > offset_rowid (for job arrays / resume).",
    )
    parser.add_argument(
        "--debug-first-chunk",
        action="store_true",
        help="Run on a single (first) unprocessed chunk and print raw + parsed output (no DB writes).",
    )

    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        tokenizer, model, device = load_llm()
        process_chunks(
            conn,
            tokenizer,
            model,
            device,
            max_chunks=args.max_chunks,
            offset_rowid=args.offset_rowid,
            debug_first=args.debug_first_chunk,
        )
    finally:
        conn.close()
