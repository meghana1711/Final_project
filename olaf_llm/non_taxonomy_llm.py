# file: olaf_llm/non_taxonomy_llm.py
from __future__ import annotations

import sqlite3
import json
import ast
import re
from typing import List, Dict, Any, Optional, Set

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DB_PATH = "onto_db/onto_new.db"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# SYSTEM PROMPT (NON-TAXONOMIC RELATIONS)
# -----------------------------------------------------------------------------

SYSTEM_PROMPT_NON_TAX = """\
You are an expert in High Performance Computing (HPC) and job schedulers such as SLURM and IBM LSF.
Your task is STRICT NON-TAXONOMIC RELATION EXTRACTION for ONTOLOGY BUILDING.

You are given:
- A short documentation CHUNK (HPC scheduler text).
- A list of DOMAIN TERMS that occur in that chunk.
- For each term: its CATEGORY and a short DEFINITION from a previous enrichment step.

You must propose only high-quality NON-TAXONOMIC relations between these terms.

WHAT COUNTS AS NON-TAXONOMIC:
These are relations that are NOT IS-A (type-of) hypernyms. Examples include:
- part_of / component_of:
  - "suspended jobs are part of the job queue"
  - "slurmctld is a component of the Slurm controller"
  - "task/rank is part of an application"
  - "CPU cores are part of a node"
- provided_by / exposed_via:
  - "REST API is provided by Slurm"
  - "Slurm's REST API is exposed via the slurmrestd daemon"
- configured_by / configured_via:
  - "slurmctld provides configuration to slurmd"
  - "a floating partition is configured with a Partition QOS"
- uses / consumes / depends_on:
  - "jobs use GPU resources"
  - "the Burst Buffer plugin uses a base path"
- stored_in / logged_to / maintained_in:
  - "state is stored under /var/spool/slurm"
  - "active bitmap is maintained inside the gang scheduler logic"
  - "logs are written to slurmctld.log"
- other clear structural or functional relations (runs_on, submitted_via, scheduled_by, communicates_with, etc.).

WHAT YOU MUST *NOT* OUTPUT HERE:
- IS-A / type-of relations:
  - "job preemption is a type of preemption"
  - "Partition QOS is a type of QOS"
  - "Singularity is a type of hpcng container runtime"
  These belong to the TAXONOMY component and MUST NOT be output by this script.
- ALIASES / SYNONYMS:
  - "slurmd (compute nodes)" vs "slurmd"
  - "pmi2" vs "pmi-2"
  - "mem_per_cpu" vs "memory_per_cpu"
  Do NOT output alias_of / same_as edges here. Synonyms are handled elsewhere.
- PURELY NUMERIC OR EXAMPLE-ONLY content:
  - "500 simple batch jobs", "1,024 nodes", "0.5 seconds"
  These are examples, not stable relations between domain concepts.

CATEGORY HINTS:
Each term belongs to a category such as:
- scheduler, command, option_flag, config_param, config_file, log_or_state_path,
  queue_or_partition, resource, job_state, other_hpc, non_domain.
Use these to guide reasonable relations, e.g.:
- scheduler / component  → provides_configuration_to → daemon
- job / queue_or_partition → submitted_to / runs_in
- job / resource → uses_resource / requests_resource
- plugin / resource → manages / exposes / allocates

HARD CONSTRAINT (VERY IMPORTANT):
- Both "subject" and "object" MUST be EXACTLY one of the DOMAIN TERMS listed.
- You are NOT allowed to invent or introduce new subjects or objects such as
  "flavor", "feature", "man page", "factor", "modifier", etc., unless that exact string
  appears as a DOMAIN TERM.
- If there are NO valid NON-TAXONOMIC relations using ONLY the provided terms,
  you MUST return:
  { "relations": [] }.

PREDICATE FORMAT:
- Use short, lower_snake_case predicates that describe the relation, such as:
  - "part_of", "component_of", "provided_by", "exposed_via",
    "configured_by", "configured_via", "uses_resource", "depends_on",
    "submitted_to", "runs_in", "scheduled_by", "logs_to",
    "stored_in", "maintained_in", "communicates_with", "runs_on", "uses_plugin".
- Avoid extremely vague predicates like "related_to" or "associated_with" unless you
  genuinely cannot be more specific.

RELATION TYPE:
Set "relation_type" to a coarse label summarising the relation:
- "part_of"
- "configuration"
- "provision"
- "usage"
- "data_flow"
- "logging"
- "scheduling"
- "other"

OUTPUT FORMAT (STRICT):
You MUST output EXACTLY one JSON object and nothing else.

The JSON schema is:

{
  "relations": [
    {
      "subject": "term_from_the_list",
      "predicate": "lower_snake_case_relation",
      "object": "term_from_the_list",
      "relation_type": "part_of | configuration | provision | usage | data_flow | logging | scheduling | other",
      "justification": "one short sentence explaining why this relation holds in the text"
    },
    ...
  ]
}

Rules:
- Every subject and object MUST be taken from the provided term list ONLY.
- Do NOT invent new concepts.
- Do NOT output is-a / type-of edges.
- Do NOT output alias/synonym relations.
- Do NOT output duplicate edges.
- If NO valid non-taxonomic relations exist, return:
  { "relations": [] }.
"""

# -----------------------------------------------------------------------------
# FEW-SHOT EXAMPLES (HPC NON-TAX RELATIONS)
# -----------------------------------------------------------------------------

NON_TAX_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    # Example 1 – suspended jobs, job queue, active bitmap
    {
        "text": (
            "Suspended jobs are part of the job queue, as they are tracked within it. "
            "The active bitmap is maintained inside the gang scheduler logic, which itself is part of the job queue."
        ),
        "terms": [
            "suspended jobs",
            "job queue",
            "active bitmap",
            "gang scheduler logic",
        ],
        "json": {
            "relations": [
                {
                    "subject": "suspended jobs",
                    "predicate": "part_of",
                    "object": "job queue",
                    "relation_type": "part_of",
                    "justification": "The text explicitly states that suspended jobs are part of the job queue.",
                },
                {
                    "subject": "active bitmap",
                    "predicate": "maintained_in",
                    "object": "gang scheduler logic",
                    "relation_type": "data_flow",
                    "justification": "The active bitmap is described as being maintained inside the gang scheduler logic.",
                },
                {
                    "subject": "gang scheduler logic",
                    "predicate": "part_of",
                    "object": "job queue",
                    "relation_type": "part_of",
                    "justification": "The gang scheduler logic is described as part of the job queue.",
                },
            ]
        },
    },
    # Example 2 – Slurm, REST API, slurmrestd
    {
        "text": (
            "Slurm provides a REST API that allows external tools to submit and inspect jobs over HTTP. "
            "This REST API is exposed via the slurmrestd daemon, which runs as a service on the controller node."
        ),
        "terms": [
            "Slurm",
            "REST API",
            "slurmrestd daemon",
            "controller node",
        ],
        "json": {
            "relations": [
                {
                    "subject": "REST API",
                    "predicate": "provided_by",
                    "object": "Slurm",
                    "relation_type": "provision",
                    "justification": "The text says that Slurm provides a REST API.",
                },
                {
                    "subject": "REST API",
                    "predicate": "exposed_via",
                    "object": "slurmrestd daemon",
                    "relation_type": "provision",
                    "justification": "The REST API is described as being exposed via the slurmrestd daemon.",
                },
                {
                    "subject": "slurmrestd daemon",
                    "predicate": "runs_on",
                    "object": "controller node",
                    "relation_type": "other",
                    "justification": "The slurmrestd daemon is said to run as a service on the controller node.",
                },
            ]
        },
    },
    # Example 3 – slurmctld, slurmd, configuration
    {
        "text": (
            "The slurmctld daemon is responsible for providing configuration and job information to slurmd on "
            "each compute node. slurmd uses this configuration to launch and manage job steps locally."
        ),
        "terms": [
            "slurmctld",
            "slurmd",
            "compute node",
            "job steps",
        ],
        "json": {
            "relations": [
                {
                    "subject": "slurmctld",
                    "predicate": "provides_configuration_to",
                    "object": "slurmd",
                    "relation_type": "configuration",
                    "justification": "slurmctld is described as providing configuration to slurmd.",
                },
                {
                    "subject": "slurmd",
                    "predicate": "runs_on",
                    "object": "compute node",
                    "relation_type": "other",
                    "justification": "slurmd runs on each compute node according to the text.",
                },
                {
                    "subject": "slurmd",
                    "predicate": "manages",
                    "object": "job steps",
                    "relation_type": "scheduling",
                    "justification": "slurmd is said to launch and manage job steps locally.",
                },
            ]
        },
    },
    # Example 4 – task/rank, application, CPUs and nodes
    {
        "text": (
            "Each task or rank in an MPI application is bound to specific CPU cores on a node. "
            "These tasks or ranks are part of the application, which may span multiple nodes."
        ),
        "terms": [
            "task/rank",
            "application",
            "CPU cores",
            "node",
        ],
        "json": {
            "relations": [
                {
                    "subject": "task/rank",
                    "predicate": "part_of",
                    "object": "application",
                    "relation_type": "part_of",
                    "justification": "The text states that each task or rank is part of the application.",
                },
                {
                    "subject": "task/rank",
                    "predicate": "bound_to",
                    "object": "CPU cores",
                    "relation_type": "usage",
                    "justification": "Tasks or ranks are described as being bound to specific CPU cores.",
                },
                {
                    "subject": "CPU cores",
                    "predicate": "part_of",
                    "object": "node",
                    "relation_type": "part_of",
                    "justification": "CPU cores are implicitly part of a node, as they are cores on a node.",
                },
            ]
        },
    },
    # Example 5 – negative example: only is-a (no non-tax relations)
    {
        "text": (
            "Job preemption is a specific type of preemption in Slurm. Partition QOS is a specific type of QOS "
            "assigned to a partition. Singularity is an example of an hpcng container runtime."
        ),
        "terms": [
            "job preemption",
            "preemption",
            "Partition QOS",
            "QOS",
            "Singularity",
            "hpcng container runtime",
        ],
        "json": {
            "relations": []
        },
    },
]

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

def init_llm_non_tax_table(conn: sqlite3.Connection) -> None:
    """
    Ensure llm_non_taxonomy_edges exists.

    One row per NON-TAX relation, with doc + chunk context and raw JSON.
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_non_taxonomy_edges (
            edge_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id        TEXT    NOT NULL,
            chunk_id      TEXT    NOT NULL,
            subject_term  TEXT    NOT NULL,
            predicate     TEXT    NOT NULL,
            object_term   TEXT    NOT NULL,
            relation_type TEXT,
            justification TEXT,
            raw_json      TEXT,
            UNIQUE(doc_id, chunk_id, subject_term, predicate, object_term)
        )
        """
    )
    conn.commit()


def load_mistral():
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


# -----------------------------------------------------------------------------
# Fetch chunks + candidate terms
# -----------------------------------------------------------------------------

def fetch_chunks_for_non_taxonomy(
    conn: sqlite3.Connection,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
) -> List[tuple]:
    """
    Fetch chunks that still need non-taxonomic relation extraction.

    - Use contextual_chunk.rowid for stable ordering.
    - Skip chunks that already have at least one entry in llm_non_taxonomy_edges.
    """
    init_llm_non_tax_table(conn)
    cur = conn.cursor()

    sql = """
        SELECT rowid, doc_id, chunk_id, text
        FROM contextual_chunk
        WHERE rowid > ?
          AND NOT EXISTS (
              SELECT 1
              FROM llm_non_taxonomy_edges e
              WHERE e.chunk_id = contextual_chunk.chunk_id
          )
        ORDER BY rowid
    """
    params = [offset_rowid]
    if max_chunks is not None:
        sql += " LIMIT ?"
        params.append(max_chunks)

    cur.execute(sql, params)
    return cur.fetchall()


def get_candidate_terms_for_chunk(
    conn: sqlite3.Connection, doc_id: str, chunk_id: str
) -> List[Dict[str, str]]:
    """
    Return domain terms for a given (doc_id, chunk_id) with metadata from llm_enrich:
      - canonical_term
      - category
      - short_definition

    Filter:
      - is_hpc_domain = 1
      - category != 'non_domain'
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT
            e.canonical_term,
            e.category,
            COALESCE(e.short_definition, '')
        FROM llm_terms t
        JOIN llm_enrich e
          ON LOWER(TRIM(t.term)) = e.canonical_term
        WHERE t.doc_id = ?
          AND t.chunk_id = ?
          AND e.is_hpc_domain = 1
          AND e.category != 'non_domain'
        ORDER BY e.canonical_term
        """,
        (doc_id, chunk_id),
    )
    rows = cur.fetchall()
    return [
        {
            "term": canonical,
            "category": category or "other_hpc",
            "definition": short_def.strip(),
        }
        for (canonical, category, short_def) in rows
    ]


# -----------------------------------------------------------------------------
# Prompt building
# -----------------------------------------------------------------------------

def build_non_taxonomy_prompt(chunk_text: str, candidate_terms: List[Dict[str, str]]) -> str:
    """
    Build a Mistral [INST] style prompt with positive AND negative few-shot examples,
    including category + short definition for each term.
    """
    example_blocks = []
    for i, ex in enumerate(NON_TAX_FEW_SHOT_EXAMPLES, start=1):
        ex_json = json.dumps(ex["json"], indent=2, ensure_ascii=False)
        example_blocks.append(
            f"Example {i}:\n"
            f"TEXT:\n{ex['text']}\n"
            f"TERMS_IN_CHUNK: {', '.join(ex['terms'])}\n"
            f"CORRECT_JSON:\n{ex_json}\n"
        )
    examples_str = "\n".join(example_blocks)

    term_lines = []
    for t in candidate_terms:
        line = f"- {t['term']} [category={t['category']}]"
        if t["definition"]:
            line += f" – {t['definition']}"
        term_lines.append(line)
    terms_block = "\n".join(term_lines)

    user_content = (
        "You will receive a chunk of HPC scheduler documentation and a list of DOMAIN TERMS.\n"
        "Each term has a category and a short definition from a previous enrichment step.\n\n"
        "Your job is to propose ONLY NON-TAXONOMIC relations between these terms, following the rules in the system prompt.\n\n"
        "Here are examples of CORRECT behaviour (including a case where no non-taxonomic relations exist):\n\n"
        f"{examples_str}\n"
        "Now process the NEW chunk.\n\n"
        f"NEW_TEXT:\n{chunk_text}\n\n"
        "DOMAIN TERMS IN THIS CHUNK (with category and brief meaning):\n"
        f"{terms_block}\n\n"
        "From ONLY the terms listed above, propose NON-TAXONOMIC relations where:\n"
        "- subject and object describe a structural or functional relationship (part_of, provided_by, configured_by, uses_resource, stored_in, runs_on, etc.).\n"
        "- you AVOID is-a/type-of edges (those are handled separately).\n"
        "- you AVOID alias/synonym edges.\n\n"
        "Return ONLY one JSON object with a single key 'relations', as in the examples above."
    )

    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT_NON_TAX}\n<</SYS>>\n\n"
        f"{user_content}\n"
        "[/INST]"
    )


# -----------------------------------------------------------------------------
# LLM call
# -----------------------------------------------------------------------------

def call_non_taxonomy_llm(
    tokenizer,
    model,
    device: str,
    chunk_text: str,
    candidate_terms: List[Dict[str, str]],
) -> str:
    prompt = build_non_taxonomy_prompt(chunk_text, candidate_terms)

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
            max_new_tokens=256,
            do_sample=False,  # deterministic
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_only_ids = generated_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_only_ids, skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def parse_non_tax_output(raw_output: str, allowed_terms: Set[str]) -> List[Dict[str, str]]:
    """
    Parse the LLM output for non-taxonomic relations.

    expected JSON:
    {
      "relations": [
        {
          "subject": "...",
          "predicate": "...",
          "object": "...",
          "relation_type": "...",
          "justification": "..."
        },
        ...
      ]
    }

    - Only keep edges where subject and object are in allowed_terms.
    - Accept either 'justification' or 'reason' as the explanation key.
    - Drop duplicates and self-edges.
    """

    def _from_structured(obj: Any) -> List[Dict[str, str]]:
        if not isinstance(obj, dict):
            return []
        rels_raw = obj.get("relations", [])
        if not isinstance(rels_raw, list):
            return []

        result: List[Dict[str, str]] = []
        seen_keys: Set[tuple] = set()

        for item in rels_raw:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject", "")).strip()
            predicate = str(item.get("predicate", "")).strip()
            obj_term = str(item.get("object", "")).strip()
            relation_type = str(item.get("relation_type", "")).strip() or "other"
            justification = str(
                item.get("justification") or item.get("reason") or ""
            ).strip()

            if not subject or not predicate or not obj_term:
                continue

            if subject not in allowed_terms or obj_term not in allowed_terms:
                continue

            if subject.lower() == obj_term.lower():
                continue

            key = (subject.lower(), predicate.lower(), obj_term.lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)

            result.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj_term,
                    "relation_type": relation_type,
                    "justification": justification,
                }
            )
        return result

    # ---------------------------
    # 1) Try full JSON / dict parse
    # ---------------------------
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    data: Any = {}

    if start != -1 and end != -1 and end > start:
        json_str = raw_output[start : end + 1]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(json_str)
            except Exception:
                data = {}

    rels = _from_structured(data)
    if rels:
        return rels

    # --------------------------------
    # 2) Fallback: regex extraction for truncated/messy JSON
    # --------------------------------
    idx = raw_output.find('"relations"')
    if idx == -1:
        return []

    sub = raw_output[idx:]

    # Look for objects with subject/predicate/object(/relation_type)(/justification|reason)
    pattern = re.compile(
        r'"subject"\s*:\s*"([^"]+)"\s*,\s*'
        r'"predicate"\s*:\s*"([^"]+)"\s*,\s*'
        r'"object"\s*:\s*"([^"]+)"'
        r'(?:\s*,\s*"relation_type"\s*:\s*"([^"]*)")?'
        r'(?:\s*,\s*"(?:justification|reason)"\s*:\s*"([^"]*)")?',
        re.DOTALL,
    )

    result: List[Dict[str, str]] = []
    seen_keys: Set[tuple] = set()

    for match in pattern.finditer(sub):
        subject = match.group(1).strip()
        predicate = match.group(2).strip()
        obj_term = match.group(3).strip()
        relation_type = (match.group(4) or "").strip() or "other"
        justification = (match.group(5) or "").strip()

        if not subject or not predicate or not obj_term:
            continue

        if subject not in allowed_terms or obj_term not in allowed_terms:
            continue

        if subject.lower() == obj_term.lower():
            continue

        key = (subject.lower(), predicate.lower(), obj_term.lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)

        result.append(
            {
                "subject": subject,
                "predicate": predicate,
                "object": obj_term,
                "relation_type": relation_type,
                "justification": justification,
            }
        )

    return result


# -----------------------------------------------------------------------------
# Main processing loop
# -----------------------------------------------------------------------------

def process_chunks_for_non_taxonomy(
    conn: sqlite3.Connection,
    tokenizer,
    model,
    device: str,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
) -> None:
    init_llm_non_tax_table(conn)

    rows = fetch_chunks_for_non_taxonomy(conn, max_chunks=max_chunks, offset_rowid=offset_rowid)
    total = len(rows)
    print(f"Processing {total} chunks for NON-TAXONOMIC relations (offset_rowid={offset_rowid})...")

    if total == 0:
        return

    cur = conn.cursor()

    for idx, (rowid, doc_id, chunk_id, text) in enumerate(rows, start=1):
        candidate_terms = get_candidate_terms_for_chunk(conn, doc_id, chunk_id)
        if len(candidate_terms) < 2:
            # need at least two terms to form a relation
            continue

        if idx == 1 or idx % 10 == 0:
            print(
                f"  -> chunk {idx}/{total} (rowid={rowid}, doc_id={doc_id}, "
                f"chunk_id={chunk_id}, terms={len(candidate_terms)})"
            )

        raw = call_non_taxonomy_llm(tokenizer, model, device, text, candidate_terms)
        allowed_terms = {t["term"] for t in candidate_terms}
        rels = parse_non_tax_output(raw, allowed_terms=allowed_terms)

        # Debug for the first processed chunk in this run
        if idx == 1:
            print("\n=== DEBUG FIRST CHUNK RAW (first 600 chars) ===")
            print(raw[:600])
            print("\n=== DEBUG FIRST CHUNK PARSED RELATIONS ===")
            for r in rels:
                print(
                    f"- {r['subject']}  --{r['predicate']}-->  {r['object']}  "
                    f"[type={r['relation_type']}]  ({r['justification']})"
                )
            if not rels:
                print("(No non-taxonomic relations parsed)")
            print("------\n")

        for r in rels:
            cur.execute(
                """
                INSERT OR IGNORE INTO llm_non_taxonomy_edges (
                    doc_id,
                    chunk_id,
                    subject_term,
                    predicate,
                    object_term,
                    relation_type,
                    justification,
                    raw_json
                )
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
                    raw.strip(),
                ),
            )
        conn.commit()

    print("Non-taxonomic relation extraction done.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based non-taxonomic relation extraction over contextual_chunk (HPC docs)."
    )
    parser.add_argument(
        "--debug-first-chunk",
        action="store_true",
        help="Run on a single first eligible chunk and print raw + parsed output (no DB writes).",
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
        help="Start from contextual_chunk.rowid > offset-rowid (for resuming / job arrays).",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        tokenizer, model, device = load_mistral()

        if args.debug_first_chunk:
            rows = fetch_chunks_for_non_taxonomy(
                conn, max_chunks=1000, offset_rowid=args.offset_rowid
            )
            # find first chunk that actually has >=2 candidate terms
            chosen = None
            for rowid, doc_id, chunk_id, text in rows:
                c_terms = get_candidate_terms_for_chunk(conn, doc_id, chunk_id)
                if len(c_terms) >= 2:
                    chosen = (rowid, doc_id, chunk_id, text, c_terms)
                    break

            if chosen is None:
                print("No eligible chunks (with >=2 domain terms) found.")
            else:
                rowid, doc_id, chunk_id, text, c_terms = chosen
                print(
                    f"DEBUG rowid={rowid}, doc_id={doc_id}, chunk_id={chunk_id}, "
                    f"{len(c_terms)} candidate terms"
                )
                raw = call_non_taxonomy_llm(tokenizer, model, device, text, c_terms)
                allowed_terms = {t["term"] for t in c_terms}
                rels = parse_non_tax_output(raw, allowed_terms=allowed_terms)

                print("\n=== RAW OUTPUT (first 800 chars) ===")
                print(raw[:800])
                print("\n=== PARSED NON-TAXONOMIC RELATIONS ===")
                for r in rels:
                    print(
                        f"- {r['subject']}  --{r['predicate']}-->  {r['object']}  "
                        f"[type={r['relation_type']}]  ({r['justification']})"
                    )
                if not rels:
                    print("(No non-taxonomic relations parsed)")
        else:
            process_chunks_for_non_taxonomy(
                conn,
                tokenizer,
                model,
                device,
                max_chunks=args.max_chunks,
                offset_rowid=args.offset_rowid,
            )
    finally:
        conn.close()
