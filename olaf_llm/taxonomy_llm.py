# file: olaf_llm/taxonomy_is_a_llm.py
from __future__ import annotations

import sqlite3
import re
import json
import ast
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
# SYSTEM PROMPT (TAXONOMY IS-A, WITH CATEGORIES + HARD CONSTRAINT)
# -----------------------------------------------------------------------------

SYSTEM_PROMPT_TAXONOMY = """\
You are an expert in High Performance Computing (HPC) and job schedulers such as SLURM and IBM LSF.
Your task is STRICT TAXONOMY (IS-A) EXTRACTION for ONTOLOGY BUILDING.

You are given:
- A short documentation CHUNK (HPC scheduler text).
- A list of DOMAIN TERMS that occur in that chunk.
- For each term: its CATEGORY and a short DEFINITION taken from a previous enrichment step.

You must propose only high-quality IS-A relations between these terms.

DEFINITION OF IS-A:
- "X is-a Y" means: every X is a kind of Y (X is a subtype / more specific concept of Y).
- Examples:
  - "gpu partition" is-a "partition".
  - "debug queue" is-a "queue".
  - "job preemption" is-a "preemption".
  - "knl_generic plugin" is-a "node features plugin".
  - "knl_generic" is-a "plugin".

WHAT IS *NOT* IS-A (MUST BE REJECTED):
- PART-OF relations:
  - "slurmctld" is part of the Slurm controller → NOT is-a.
  - "active bitmap" is part of gang scheduler logic → NOT is-a.
  - "slurmctld provides configuration to slurmd" → NOT is-a.
- ROLE or FUNCTION relations:
  - "job step manager" manages jobs → NOT is-a job.
  - "slurmstepd daemon" manages job steps → NOT is-a job.
- CONSTRAINTS or LIMITS:
  - "maximum allocation" is a limit applied to resources → NOT is-a resource.
  - "time limit" is a constraint on jobs → NOT is-a job.
- ALIASES, SYNONYMS, OR TRIVIAL RENAMINGS:
  - "slurmd (compute nodes)" vs "slurmd" → alias, NOT is-a.
  - Parameter spelling variants like "mem_per_cpu" vs "memory_per_cpu" → NOT is-a.
  - Version-like aliases such as "pmi2" vs "pmi-2" → NOT is-a.
- GENERIC EXAMPLES:
  - Numeric-heavy phrases like "500 simple batch jobs" → no is-a edges at all.
  - Example-only descriptions that do not define a stable concept.
- SIBLING FACTORS OR MODES:
  - When two terms are both described as separate factors, modes, or settings under the same concept
    (for example, 'bonus' and 'malus' as different fair-share factors), they are SIBLINGS.
  - Do NOT create an is-a edge such as 'bonus is-a malus' or 'malus is-a bonus'.
  - Instead, they would both be children of some more general concept (e.g. 'fair-share modifier'),
    and if that more general concept is not in the term list, you MUST output no is-a edges.


CATEGORY INFORMATION:
Each term has a category from the enrichment step, for example:
- "scheduler"          : scheduler or major components (Slurmctld, SlurmDBD, IBM LSF).
- "command"            : CLI commands or subcommands.
- "option_flag"        : command-line options or flags.
- "config_param"       : configuration parameters or plugin-type keys.
- "config_file"        : configuration/include files.
- "log_or_state_path"  : log or state file paths and directories.
- "queue_or_partition" : queues, partitions, QoS names.
- "resource"           : resource concepts (CPU cores, node features, burst buffer, etc.).
- "job_state"          : job states and similar status labels.
- "other_hpc"          : other HPC-specific concepts.
- "non_domain"         : non-domain or low-value terms (already filtered out before this step).

You MUST prefer IS-A edges where parent and child are compatible in category, such as:
- resource           → resource
- queue_or_partition → queue_or_partition
- config_param       → config_param
- job_state          → job_state
- a more specific scheduler component → a more general scheduler concept

If categories clearly do not match (e.g., a constraint vs resource, a role vs job),
you MUST NOT create an is-a edge.

HARD CONSTRAINT (VERY IMPORTANT):
- Both "child" and "parent" MUST be EXACTLY one of the DOMAIN TERMS listed for the chunk.
- You are NOT allowed to invent or introduce a new parent or child such as "flavor",
  "man page", "parameter", "feature", "option", "resource", etc., unless that exact
  string appears in the provided DOMAIN TERMS list.
- If no valid is-a edges can be formed using ONLY the provided terms, you MUST return:
  { "is_a_edges": [] }.

OUTPUT FORMAT (STRICT):
You MUST output EXACTLY one JSON object and nothing else.

The JSON schema is:

{
  "is_a_edges": [
    {
      "child": "child_term_from_the_list",
      "parent": "parent_term_from_the_list",
      "justification": "one short sentence explaining why this is a valid is-a"
    },
    ...
  ]
}

Rules:
- Every "child" and "parent" MUST be taken from the provided term list ONLY.
- Do NOT invent new terms.
- Do NOT output duplicate edges.
- If there are NO valid is-a edges, return:
  { "is_a_edges": [] }
- Do NOT include any extra keys.
- Do NOT write anything outside the JSON.
"""

# -----------------------------------------------------------------------------
# FEW-SHOT EXAMPLES
# -----------------------------------------------------------------------------

TAXONOMY_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    # Example 1 – licenses & plugin hierarchy (GOOD edges)
    {
        "text": (
            "Licenses in Slurm can be configured as local or remote. Local licenses are defined directly "
            "in slurm.conf, while remote licenses are served by the accounting database and managed via "
            "the sacctmgr command. The knl_generic plugin is a node features plugin used on Intel KNL "
            "nodes to expose KNL-specific capabilities."
        ),
        "terms": [
            "licenses",
            "local licenses",
            "remote licenses",
            "knl_generic plugin",
            "node features plugin",
            "knl_generic",
            "plugin",
        ],
        "json": {
            "is_a_edges": [
                {
                    "child": "local licenses",
                    "parent": "licenses",
                    "justification": "Local licenses are a specific type of licenses configured directly on the cluster.",
                },
                {
                    "child": "remote licenses",
                    "parent": "licenses",
                    "justification": "Remote licenses are a specific type of licenses served by the accounting database.",
                },
                {
                    "child": "knl_generic plugin",
                    "parent": "node features plugin",
                    "justification": "The knl_generic plugin is described as a node features plugin for Intel KNL systems.",
                },
                {
                    "child": "knl_generic",
                    "parent": "plugin",
                    "justification": "knl_generic is introduced as a specific plugin name.",
                },
            ]
        },
    },
    # Example 2 – preemption and QOS hierarchy (GOOD edges)
    {
        "text": (
            "Job preemption in Slurm is a specific type of preemption where higher-priority jobs can "
            "preempt lower-priority ones. Partition QOS is a Quality of Service value assigned to a "
            "partition. Partition QOS is therefore a specific type of QOS used in the context of Slurm partitions."
        ),
        "terms": [
            "preemption",
            "job preemption",
            "Quality of Service",
            "QOS",
            "Partition QOS",
        ],
        "json": {
            "is_a_edges": [
                {
                    "child": "job preemption",
                    "parent": "preemption",
                    "justification": "Job preemption is described as a specific kind of preemption for jobs in Slurm.",
                },
                {
                    "child": "Partition QOS",
                    "parent": "QOS",
                    "justification": "Partition QOS is presented as a specific QOS value assigned to a partition.",
                },
            ]
        },
    },
    # Example 3 – REST API, Singularity, process affinity (GOOD edges)
    {
        "text": (
            "Slurm provides a REST API via the slurmrestd daemon, allowing external tools to submit and "
            "inspect jobs over HTTP. Singularity is one of the hpcng container runtimes commonly used on "
            "HPC clusters. Process affinity is an explicit configuration option controlling how processes "
            "are bound to CPU cores."
        ),
        "terms": [
            "REST API",
            "API",
            "slurmrestd daemon",
            "Singularity",
            "hpcng container runtime",
            "process affinity",
            "configuration option",
        ],
        "json": {
            "is_a_edges": [
                {
                    "child": "REST API",
                    "parent": "API",
                    "justification": "The REST API is introduced as a specific API provided by Slurm.",
                },
                {
                    "child": "Singularity",
                    "parent": "hpcng container runtime",
                    "justification": "Singularity is explicitly mentioned as an example of an hpcng container runtime.",
                },
                {
                    "child": "process affinity",
                    "parent": "configuration option",
                    "justification": "Process affinity is described as an explicit configuration option.",
                },
            ]
        },
    },
    # Example 4 – roles, part-of, aliases (NO is-a edges)
    {
        "text": (
            "The slurmctld daemon provides configuration to slurmd on each compute node. The active bitmap "
            "is maintained inside the gang scheduler logic as part of the job queue. The string 'slurmd "
            "(compute nodes)' is simply another way to refer to the slurmd daemons running on compute nodes."
        ),
        "terms": [
            "slurmctld",
            "slurmd",
            "compute nodes",
            "active bitmap",
            "gang scheduler",
            "job queue",
            "slurmd (compute nodes)",
        ],
        "json": {
            "is_a_edges": []
        },
    },

    #Example 5: Not to add a sibbling term as a is_a
    {
        "text": (
            "In the depth-oblivious fair-share policy, each account can have both a bonus and a malus. "
            "The bonus factor increases the account's effective fair-share, while the malus factor penalizes it "
            "based on the usage of ancestor accounts. These are two different multipliers applied to the same fair-share "
            "calculation, not types of each other."
        ),
        "terms": [
            "depth_oblivious",
            "fair-share factor",
            "bonus",
            "malus",
        ],
        "json": {
            "is_a_edges": []
        },
    },
]

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

def init_llm_is_a_table(conn: sqlite3.Connection) -> None:
    """
    Ensure llm_is_a_edges exists.

    One row per IS-A edge, with doc + chunk context and raw JSON from the LLM.
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_is_a_edges (
            edge_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id       TEXT    NOT NULL,
            chunk_id     TEXT    NOT NULL,
            child_term   TEXT    NOT NULL,
            parent_term  TEXT    NOT NULL,
            justification TEXT,
            raw_json     TEXT,
            UNIQUE(doc_id, chunk_id, child_term, parent_term)
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

def fetch_chunks_for_taxonomy(
    conn: sqlite3.Connection,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
) -> List[tuple]:
    """
    Fetch chunks that still need IS-A taxonomy extraction.

    - Use contextual_chunk.rowid for stable ordering and partitioning.
    - Skip chunks that already have at least one entry in llm_is_a_edges.
    """
    init_llm_is_a_table(conn)
    cur = conn.cursor()

    sql = """
        SELECT rowid, doc_id, chunk_id, text
        FROM contextual_chunk
        WHERE rowid > ?
          AND NOT EXISTS (
              SELECT 1
              FROM llm_is_a_edges e
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
    Return domain terms for a given (doc_id, chunk_id), with metadata from llm_enrich:
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

def build_taxonomy_prompt(chunk_text: str, candidate_terms: List[Dict[str, str]]) -> str:
    """
    Build a Mistral [INST] style prompt with positive AND negative few-shot examples,
    including category + short definition for each term.
    """
    example_blocks = []
    for i, ex in enumerate(TAXONOMY_FEW_SHOT_EXAMPLES, start=1):
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
        "Your job is to propose ONLY valid IS-A edges between these terms, following the rules in the system prompt.\n\n"
        "Here are examples of CORRECT behaviour (including cases where no is-a edges exist):\n\n"
        f"{examples_str}\n"
        "Now process the NEW chunk.\n\n"
        f"NEW_TEXT:\n{chunk_text}\n\n"
        "DOMAIN TERMS IN THIS CHUNK (with category and brief meaning):\n"
        f"{terms_block}\n\n"
        "From ONLY the terms listed above, propose IS-A edges where:\n"
        "- child and parent are compatible in category (e.g. resource→resource, config_param→config_param, "
        "queue_or_partition→queue_or_partition, job_state→job_state, or a more specific scheduler component "
        "→ a more general scheduler concept).\n"
        "- child is NOT merely an alias, version, feature, configuration, or part-of the parent.\n"
        "- child is NOT just a role or manager of the parent.\n"
        "- constraints or limits applied to a concept (like 'maximum allocation') are NOT is-a edges to that concept.\n\n"
        "Return ONLY one JSON object with a single key 'is_a_edges', as in the examples above."
    )

    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT_TAXONOMY}\n<</SYS>>\n\n"
        f"{user_content}\n"
        "[/INST]"
    )


# -----------------------------------------------------------------------------
# LLM call
# -----------------------------------------------------------------------------

def call_taxonomy_llm(
    tokenizer,
    model,
    device: str,
    chunk_text: str,
    candidate_terms: List[Dict[str, str]],
) -> str:
    prompt = build_taxonomy_prompt(chunk_text, candidate_terms)

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

def parse_is_a_output(raw_output: str, allowed_terms: Set[str]) -> List[Dict[str, str]]:
    """
    Parse the LLM output for is-a edges.

    expected JSON:
    {
      "is_a_edges": [
        {"child": "...", "parent": "...", "justification": "..."},
        ...
      ]
    }

    - Only keep edges where child and parent are in allowed_terms.
    - Accept either 'justification' or 'reason' as the explanation key.
    - Drop duplicates and self-edges.
    """

    def _from_structured(obj: Any) -> List[Dict[str, str]]:
        if not isinstance(obj, dict):
            return []
        edges_raw = obj.get("is_a_edges", [])
        if not isinstance(edges_raw, list):
            return []

        result: List[Dict[str, str]] = []
        seen_pairs: Set[tuple] = set()

        for item in edges_raw:
            if not isinstance(item, dict):
                continue
            child = str(item.get("child", "")).strip()
            parent = str(item.get("parent", "")).strip()
            justification = str(
                item.get("justification") or item.get("reason") or ""
            ).strip()

            if not child or not parent:
                continue

            # must refer only to allowed terms
            if child not in allowed_terms or parent not in allowed_terms:
                continue

            # drop self-edges
            if child.lower() == parent.lower():
                continue

            key = (child.lower(), parent.lower())
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            result.append(
                {
                    "child": child,
                    "parent": parent,
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

    edges = _from_structured(data)
    if edges:
        return edges

    # --------------------------------
    # 2) Fallback: regex extraction
    #    (handles truncated / messy JSON)
    # --------------------------------
    idx = raw_output.find('"is_a_edges"')
    if idx == -1:
        return []

    sub = raw_output[idx:]

    # This pattern looks for objects with child/parent/(justification|reason)
    pattern = re.compile(
        r'"child"\s*:\s*"([^"]+)"\s*,\s*'
        r'"parent"\s*:\s*"([^"]+)"'
        r'(?:\s*,\s*"(?:justification|reason)"\s*:\s*"([^"]*)")?',
        re.DOTALL,
    )

    result: List[Dict[str, str]] = []
    seen_pairs: Set[tuple] = set()

    for match in pattern.finditer(sub):
        child = match.group(1).strip()
        parent = match.group(2).strip()
        justification = (match.group(3) or "").strip()

        if not child or not parent:
            continue

        if child not in allowed_terms or parent not in allowed_terms:
            continue

        if child.lower() == parent.lower():
            continue

        key = (child.lower(), parent.lower())
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        result.append(
            {
                "child": child,
                "parent": parent,
                "justification": justification,
            }
        )

    return result


# -----------------------------------------------------------------------------
# Main processing loop
# -----------------------------------------------------------------------------

def process_chunks_for_taxonomy(
    conn: sqlite3.Connection,
    tokenizer,
    model,
    device: str,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
) -> None:
    init_llm_is_a_table(conn)

    rows = fetch_chunks_for_taxonomy(conn, max_chunks=max_chunks, offset_rowid=offset_rowid)
    total = len(rows)
    print(f"Processing {total} chunks for IS-A taxonomy (offset_rowid={offset_rowid})...")

    if total == 0:
        return

    cur = conn.cursor()

    for idx, (rowid, doc_id, chunk_id, text) in enumerate(rows, start=1):
        candidate_terms = get_candidate_terms_for_chunk(conn, doc_id, chunk_id)
        if len(candidate_terms) < 2:
            # need at least two terms to form an is-a relation
            continue

        if idx == 1 or idx % 10 == 0:
            print(
                f"  -> chunk {idx}/{total} (rowid={rowid}, doc_id={doc_id}, "
                f"chunk_id={chunk_id}, terms={len(candidate_terms)})"
            )

        raw = call_taxonomy_llm(tokenizer, model, device, text, candidate_terms)
        allowed_terms = {t["term"] for t in candidate_terms}
        edges = parse_is_a_output(raw, allowed_terms=allowed_terms)

        # Debug for the first processed chunk in this run
        if idx == 1:
            print("\n=== DEBUG FIRST CHUNK RAW (first 600 chars) ===")
            print(raw[:600])
            print("\n=== DEBUG FIRST CHUNK PARSED EDGES ===")
            for e in edges:
                print(f"- {e['child']}  ->  {e['parent']}  ({e['justification']})")
            if not edges:
                print("(No is-a edges parsed)")
            print("------\n")

        for e in edges:
            cur.execute(
                """
                INSERT OR IGNORE INTO llm_is_a_edges (
                    doc_id,
                    chunk_id,
                    child_term,
                    parent_term,
                    justification,
                    raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    chunk_id,
                    e["child"],
                    e["parent"],
                    e["justification"],
                    raw.strip(),
                ),
            )
        conn.commit()

    print("IS-A taxonomy extraction done.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based IS-A taxonomy extraction over contextual_chunk (HPC docs)."
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
            rows = fetch_chunks_for_taxonomy(
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
                raw = call_taxonomy_llm(tokenizer, model, device, text, c_terms)
                allowed_terms = {t["term"] for t in c_terms}
                edges = parse_is_a_output(raw, allowed_terms=allowed_terms)

                print("\n=== RAW OUTPUT (first 800 chars) ===")
                print(raw[:800])
                print("\n=== PARSED IS-A EDGES ===")
                for e in edges:
                    print(f"- {e['child']}  ->  {e['parent']}  ({e['justification']})")
                if not edges:
                    print("(No is-a edges parsed)")
        else:
            process_chunks_for_taxonomy(
                conn,
                tokenizer,
                model,
                device,
                max_chunks=args.max_chunks,
                offset_rowid=args.offset_rowid,
            )
    finally:
        conn.close()
