# file: olaf_llm/non_taxonomy_llm.py

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

# Use the SAME model as for taxonomy; change this to your HF model id
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Or, for a smaller model:
# MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MAX_TERMS_PER_CHUNK = 16  # how many candidate terms to send per chunk

# Relations we *don’t* want to see as outputs (too generic / useless)
FORBIDDEN_RELATION_STRINGS = {
    "is", "are", "be", "is related to", "are related to",
    "has", "have", "contains", "exists", "is described in", "is mentioned in"
}


# =============================================================================
# SYSTEM PROMPT (NON-TAXONOMIC RELATIONS)
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert in High Performance Computing (HPC) and job schedulers like SLURM and IBM LSF.
Your task is to extract NON-TAXONOMIC relations between HPC terms from documentation chunks.

You are given:
- A short documentation CHUNK.
- A list of CANDIDATE TERMS that appear in or are relevant to that chunk.

Your job:
- Find meaningful relations between these terms that are NOT is-a (subclass) relations.
- A non-taxonomic relation describes how two concepts interact or relate in practice, for example:
  - "slurmctld reads its configuration from slurm.conf"
  - "sbatch submits jobs to a partition"
  - "Burst Buffer accelerates I/O for jobs"
  - "jobs are scheduled onto compute nodes"
  - "jobs are logged in the accounting database"

You MUST obey the following rules:

1. WHAT TO EXTRACT (GOOD NON-TAXONOMIC RELATIONS)
   Relations such as:
   - usage: X uses Y, X submits jobs to Y, X runs on Y, X connects to Y
   - configuration: X reads configuration from Y, X is configured by Y
   - logging / accounting: X logs to Y, X writes records to Y
   - data flow / control: X sends data to Y, X controls Y, X monitors Y
   - scheduling: scheduler schedules jobs onto nodes/partitions, jobs run in partitions
   - resource usage: jobs consume resources on nodes, jobs use GPUs, jobs use Burst Buffer

   Examples:
   - subject: "slurmctld", relation: "reads_config_from", object: "slurm.conf"
   - subject: "sbatch",    relation: "submits_jobs_to",  object: "partition"
   - subject: "Burst Buffer", relation: "accelerates_io_for", object: "jobs"
   - subject: "jobs", relation: "run_on", object: "compute nodes"

2. WHAT TO AVOID (NOT NON-TAXONOMIC)
   DO NOT output:
   - is-a (subclass) relations (e.g., "Partition QOS is a type of QOS").
   - part-of relations (e.g., "Suspended jobs are part of the job queue").
   - pure numeric or time values (e.g., "0.5 seconds", "100 jobs") as subjects/objects.
   - vague or meaningless relations like "is related to", "has", "is described in".
   - meta-text relations about the document itself ("this section describes...", "the table shows...").

3. RELATION TEXT FORMAT
   - "relation" MUST be a short, informative phrase (1–4 words).
   - Prefer verb-like phrases such as "submits_jobs_to", "runs_on", "reads_config_from",
     "logs_to", "configured_by", "allocates_resources_on".
   - Use lowercase and underscores instead of spaces where appropriate
     (e.g., "runs_on", not "Runs On").
   - Do NOT include question marks or full sentences as the relation text.

4. SUBJECT AND OBJECT
   - Both subject and object MUST come from the candidate term list (or obvious close variants).
   - subject != object (they must be different terms).
   - Prefer pairs that are strongly supported by the text.

5. OUTPUT FORMAT (STRICT)
   You MUST output exactly ONE JSON object and nothing else.

   The JSON schema is:

   {
     "relations": [
       {
         "subject": "term from the candidate list",
         "relation": "short relation phrase, e.g. reads_config_from",
         "object": "term from the candidate list",
         "justification": "one or two sentences explaining the relation in context"
       },
       ...
     ]
   }

   - If no good non-taxonomic relations are found, return:
     { "relations": [] }

   - Do NOT output any other keys.
   - Do NOT output is-a relations here; they are handled in a separate step.
"""


# =============================================================================
# FEW-SHOT EXAMPLES
# =============================================================================

NON_TAX_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    # Example 1 – config + logs_to
    {
        "chunk": (
            "The slurmctld daemon reads its configuration from slurm.conf and logs messages to "
            "/var/log/slurm/slurmctld.log. Changes to slurm.conf require a restart of slurmctld."
        ),
        "candidate_terms": [
            "slurmctld daemon",
            "slurmctld",
            "slurm.conf",
            "/var/log/slurm/slurmctld.log",
            "messages",
        ],
        "json": {
            "relations": [
                {
                    "subject": "slurmctld daemon",
                    "relation": "reads_config_from",
                    "object": "slurm.conf",
                    "justification": "The text says the slurmctld daemon reads its configuration from slurm.conf."
                },
                {
                    "subject": "slurmctld daemon",
                    "relation": "logs_to",
                    "object": "/var/log/slurm/slurmctld.log",
                    "justification": "The text says slurmctld logs messages to /var/log/slurm/slurmctld.log."
                }
            ]
        },
    },
    # Example 2 – sbatch submits jobs to partition
    {
        "chunk": (
            "Users submit jobs with sbatch, and the jobs are placed into the debug partition "
            "if no other partition is specified."
        ),
        "candidate_terms": [
            "users",
            "jobs",
            "sbatch",
            "debug partition",
            "partition",
        ],
        "json": {
            "relations": [
                {
                    "subject": "sbatch",
                    "relation": "submits_jobs_to",
                    "object": "debug partition",
                    "justification": "The text explains that sbatch is used to submit jobs which are placed into the debug partition."
                }
            ]
        },
    },
    # Example 3 – Burst Buffer accelerates IO for jobs
    {
        "chunk": (
            "A Burst Buffer provides high-speed intermediate storage to accelerate I/O for jobs "
            "that read or write large volumes of data."
        ),
        "candidate_terms": [
            "Burst Buffer",
            "jobs",
            "I/O",
            "data",
        ],
        "json": {
            "relations": [
                {
                    "subject": "Burst Buffer",
                    "relation": "accelerates_io_for",
                    "object": "jobs",
                    "justification": "The text states that a Burst Buffer accelerates I/O for jobs."
                }
            ]
        },
    },
    # Example 4 – BAD: is-a only (must return empty list)
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
            "relations": []
        },
    },
    # Example 5 – BAD: part-of and storage (must return empty list)
    {
        "chunk": (
            "Suspended jobs are part of the job queue, as they are tracked within it. "
            "The active bitmap is maintained inside the gang scheduler logic."
        ),
        "candidate_terms": [
            "suspended jobs",
            "job queue",
            "active bitmap",
            "gang scheduler logic",
        ],
        "json": {
            "relations": []
        },
    },
]


# =============================================================================
# DB HELPERS
# =============================================================================

def init_non_tax_table(conn: sqlite3.Connection) -> None:
    """
    Table for non-taxonomic LLM edges.
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_non_taxonomy_qwen (
            edge_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id        TEXT NOT NULL,
            chunk_id      TEXT NOT NULL,
            subj_term     TEXT NOT NULL,
            rel_text      TEXT NOT NULL,
            obj_term      TEXT NOT NULL,
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


def fetch_chunks_for_non_taxonomy(
    conn: sqlite3.Connection,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
) -> List[tuple]:
    """
    Fetch contextual_chunk rows that still need non-taxonomic relation extraction.
    Skip chunks that already have at least one non-tax edge for that (doc_id, chunk_id).
    """
    init_non_tax_table(conn)
    cur = conn.cursor()

    sql = """
        SELECT cc.rowid, cc.doc_id, cc.chunk_id, cc.text
        FROM contextual_chunk AS cc
        WHERE cc.rowid > ?
          AND NOT EXISTS (
              SELECT 1 FROM llm_non_taxonomy_qwen e
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

    Same logic as taxonomy:
      - start from llm_terms.term for this (doc_id, chunk_id)
      - join llm_enrich on canonical term
      - keep only is_hpc_domain=1, category != 'non_domain' where enrichment exists
      - dedupe & filter very short ones
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
# PROMPT + LLM CALL
# =============================================================================

def build_non_tax_messages(chunk_text: str, candidate_terms: List[str]) -> List[Dict[str, str]]:
    """
    Build messages for chat-style models (Qwen, DeepSeek, etc.).
    """
    example_blocks = []
    for i, ex in enumerate(NON_TAX_FEW_SHOT_EXAMPLES, start=1):
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
        "Your job is to output ONLY good non-taxonomic relations between these terms.\n\n"
        "Here are some examples of CORRECT and INCORRECT behavior:\n\n"
        f"{examples_str}\n\n"
        "Now process the NEW CHUNK below.\n\n"
        "CHUNK:\n"
        f"{chunk_text}\n\n"
        "CANDIDATE_TERMS:\n"
        f"{terms_bullet}\n\n"
        "Return exactly ONE JSON object with the key \"relations\" as described in the system prompt.\n"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return messages


def call_non_tax_llm(
    tokenizer,
    model,
    device: str,
    chunk_text: str,
    candidate_terms: List[str],
) -> str:
    messages = build_non_tax_messages(chunk_text, candidate_terms)

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
# PARSING + FILTERING
# =============================================================================

def parse_non_tax_output(raw_output: str) -> List[Dict[str, str]]:
    """
    Parse the LLM JSON and apply simple filters.
    Returns list of dicts: {subject, relation, object, justification}.
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

    relations = data.get("relations", [])
    if not isinstance(relations, list):
        return []

    cleaned: List[Dict[str, str]] = []
    seen = set()

    for item in relations:
        if not isinstance(item, dict):
            continue

        subj = str(item.get("subject", "")).strip()
        rel = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        just = str(item.get("justification", "")).strip()

        if not subj or not rel or not obj:
            continue
        if subj.lower() == obj.lower():
            continue
        if "?" in rel or "?" in just:
            # avoid questions
            continue

        rel_lower = rel.lower()
        if rel_lower in FORBIDDEN_RELATION_STRINGS:
            continue

        key = (subj.lower(), rel_lower, obj.lower())
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(
            {
                "subject": subj,
                "relation": rel,
                "object": obj,
                "justification": just,
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
    init_non_tax_table(conn)

    rows = fetch_chunks_for_non_taxonomy(conn, max_chunks=max_chunks, offset_rowid=offset_rowid)
    total = len(rows)
    print(f"Processing {total} chunks for non-taxonomic relations (offset_rowid={offset_rowid})...")

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

        raw = call_non_tax_llm(tokenizer, model, device, chunk_text, candidate_terms)
        edges = parse_non_tax_output(raw)

        if debug_first:
            print(
                f"\nDEBUG rowid={rowid}, doc_id={doc_id}, chunk_id={chunk_id}, "
                f"{len(candidate_terms)} candidate terms"
            )
            print("\n=== RAW OUTPUT (first 800 chars) ===")
            print(raw[:800])
            print("\n=== PARSED NON-TAX RELATIONS ===")
            if edges:
                for e in edges:
                    print(f"- {e['subject']}  --{e['relation']}-->  {e['object']}  ({e['justification']})")
            else:
                print("(No non-taxonomic relations kept after filtering)")
            return

        for e in edges:
            cur.execute(
                """
                INSERT INTO llm_non_taxonomy_qwen (
                    doc_id, chunk_id, subj_term, rel_text, obj_term, justification
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, chunk_id, e["subject"], e["relation"], e["object"], e["justification"]),
            )
        conn.commit()

    print("Non-taxonomic relation extraction completed.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based non-taxonomic relation extraction over contextual_chunk (HPC docs)."
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
