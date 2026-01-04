# file: olaf_llm/term_enrichment_llm.py

from __future__ import annotations

import sqlite3
import json
import ast
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# Adjust DB_PATH if your DB filename is different
DB_PATH = "onto_db/onto_new.db"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# SYSTEM PROMPT (TERM ENRICHMENT, PRECISE)
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in High Performance Computing (HPC) and job schedulers such as SLURM and IBM LSF.
Your task is PRECISE TERM ENRICHMENT for ONTOLOGY BUILDING.

For each input, you receive:
- a TERM string (already extracted as a candidate HPC/scheduler term), and
- a SHORT CONTEXT snippet (a few sentences of documentation where the term appears).

You must decide:
1) whether this is truly a meaningful HPC / scheduler domain term,
2) which scheduler(s) it belongs to,
3) which category it falls into, and
4) provide a short, accurate definition and optional aliases.

Be conservative. If the term does not clearly refer to an HPC / scheduling concept,
classify it as non-domain.

ALLOWED VALUES FOR "scheduler":
- "slurm"     : specific to SLURM configuration, commands, or components.
- "lsf"       : specific to IBM LSF configuration, commands, or components.
- "both"      : used in both SLURM and LSF with similar meaning.
- "generic"   : generic HPC resource / job concept, not tied to a single scheduler
                (e.g. compute node, GPU resources, memory limit, job array).
- "unknown"   : you genuinely cannot tell from the term and context.

ALLOWED VALUES FOR "category":
- "scheduler"          : names of schedulers or major components (SLURM, IBM LSF,
                         Slurmctld, SlurmDBD, JobScheduler).
- "command"            : CLI commands or subcommands (sbatch, srun, sacct, squeue,
                         bsub, bjobs, lsload).
- "option_flag"        : command-line options or flags (--partition, --time, -q, --gres, etc.).
- "config_param"       : configuration parameters or plugin-type keys
                         (AccountingStorageType, JobAcctGatherType, JobCompType,
                          SelectType, GresTypes, SchedulerType, SlurmctldPort).
- "config_file"        : configuration or include files (slurm.conf, gres.conf,
                         lsb.queues, lsf.cluster, slurmdbd.conf).
- "log_or_state_path"  : log or state file paths, or state directories that are clearly
                         scheduler-related (/var/log/slurm/slurmctld.log,
                         /var/spool/slurm/statesave/jwt_hs256.key).
- "queue_or_partition" : queues, partitions, or QoS names (gpu partition, debug queue,
                         normal queue, qos-high, qos-debug).
- "resource"           : resource concepts (GPU resources, CPU cores, memory limit,
                         node features, GRES types, burst buffer).
- "job_state"          : job states and similar status labels (PENDING, RUNNING,
                         COMPLETED, FAILED, CANCELLED).
- "other_hpc"          : other HPC-specific concepts that do not fit cleanly into the above
                         but are clearly domain terms (e.g. job submission script,
                         login node, compute node, scheduler API).
- "non_domain"         : clearly not a useful HPC/scheduler term.

HOW TO DECIDE is_hpc_domain_term:
- Set is_hpc_domain_term = true when the term denotes:
  - a scheduler, scheduler component, or scheduler command,
  - a CLI option or flag related to job submission or control,
  - a configuration parameter or plugin type,
  - a config/log/state file or directory tied to the scheduler,
  - a resource, queue/partition, QoS, job state, or clearly HPC-specific concept.
- Set is_hpc_domain_term = false when the term is:
  - a generic English phrase not tied to HPC/scheduling,
  - a pure example value or numeric quantity (time, counts, ranges),
  - a generic filesystem path unrelated to SLURM/LSF (e.g. /tmp used only as an example),
  - any token that appears to be noise from parsing or formatting.

SHORT DEFINITION:
- Provide 1–2 concise sentences describing the term in its HPC / scheduler context.
- If the term is marked non-domain, you may return an empty definition ("") or a brief
  explanation such as "Not an HPC/scheduler-specific domain term.".

ALIASES:
- "aliases" should list alternative spellings, abbreviations, closely related names,
  or obvious variants that might appear in documentation.
- Example: for "SLURM", aliases might include ["Slurm", "Simple Linux Utility for Resource Management"].
- If you do not know any aliases, use an empty list [].

OUTPUT FORMAT (STRICT):
You MUST respond with EXACTLY one JSON object and nothing else.

The JSON schema is:

{
  "term": "original term string",
  "canonical": "lowercased, trimmed canonical form",
  "is_hpc_domain_term": true or false,
  "scheduler": "slurm | lsf | both | generic | unknown",
  "category": "scheduler | command | option_flag | config_param | config_file | log_or_state_path | queue_or_partition | resource | job_state | other_hpc | non_domain",
  "short_definition": "one or two short sentences in HPC/scheduler context",
  "aliases": ["optional", "aliases", "may", "be", "empty"]
}

- Do NOT add any extra keys.
- Do NOT output any text before or after the JSON.
- If you are unsure about scheduler or category, choose "unknown" (scheduler) or "other_hpc"/"non_domain" (category)
  instead of guessing wildly.
"""

# -----------------------------------------------------------------------------
# Few-shot examples (built from your extracted terms)
# -----------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    # 1. Option flag: --root
    {
        "term": "--root",
        "context": "When launching containerized jobs from the cluster, use the --root option to set the container's root filesystem path.",
        "json": {
            "term": "--root",
            "canonical": "--root",
            "is_hpc_domain_term": True,
            "scheduler": "generic",
            "category": "option_flag",
            "short_definition": "--root is a command-line option used to set the root filesystem path for a container or job environment on the cluster.",
            "aliases": []
        },
    },
    # 2. Option flag: --security-opt
    {
        "term": "--security-opt",
        "context": "The --security-opt option can be passed in job scripts to configure additional security settings for containerized workloads.",
        "json": {
            "term": "--security-opt",
            "canonical": "--security-opt",
            "is_hpc_domain_term": True,
            "scheduler": "generic",
            "category": "option_flag",
            "short_definition": "--security-opt is a command-line option used to configure extra security options for containerized jobs on the cluster.",
            "aliases": []
        },
    },
    # 3. Path used by plugins: /BasePath
    {
        "term": "/BasePath",
        "context": "The Burst Buffer plugin stores its metadata under /BasePath, which must be accessible from all compute nodes.",
        "json": {
            "term": "/BasePath",
            "canonical": "basepath",
            "is_hpc_domain_term": True,
            "scheduler": "generic",
            "category": "log_or_state_path",
            "short_definition": "/BasePath is a filesystem location used by the Burst Buffer or storage subsystem to store persistent state or metadata for jobs.",
            "aliases": ["/basepath"]
        },
    },
    # 4. Binary commonly used in job scripts: /bin/hostname
    {
        "term": "/bin/hostname",
        "context": "Many job scripts invoke /bin/hostname to log which compute node the job is running on.",
        "json": {
            "term": "/bin/hostname",
            "canonical": "/bin/hostname",
            "is_hpc_domain_term": True,
            "scheduler": "generic",
            "category": "other_hpc",
            "short_definition": "/bin/hostname is the system binary often called in job scripts to print the name of the compute node.",
            "aliases": ["hostname"]
        },
    },
    # 5. API family
    {
        "term": "API",
        "context": "The scheduler's API allows applications to submit and monitor jobs programmatically instead of using only command-line tools.",
        "json": {
            "term": "API",
            "canonical": "api",
            "is_hpc_domain_term": True,
            "scheduler": "generic",
            "category": "other_hpc",
            "short_definition": "The scheduler API is a programmatic interface that lets applications submit, control, and query jobs on the cluster.",
            "aliases": [
                "APIs",
                "API call",
                "API calls",
                "API Functions",
                "Application Program Interfaces",
                "Application Programming Interfaces (APIs)"
            ]
        },
    },
    # 6. API call
    {
        "term": "API call",
        "context": "Each API call to the workload manager returns a status code indicating whether the job submission or query succeeded.",
        "json": {
            "term": "API call",
            "canonical": "api",
            "is_hpc_domain_term": True,
            "scheduler": "generic",
            "category": "other_hpc",
            "short_definition": "An API call is a single request made to the scheduler's API, such as submitting, cancelling, or querying a job.",
            "aliases": ["API calls"]
        },
    },
    # 7. Burst Buffer as a resource
    {
        "term": "Burst Buffer",
        "context": "A Burst Buffer provides high-speed intermediate storage to accelerate I/O for jobs that read or write large volumes of data.",
        "json": {
            "term": "Burst Buffer",
            "canonical": "burst buffer",
            "is_hpc_domain_term": True,
            "scheduler": "generic",
            "category": "resource",
            "short_definition": "A Burst Buffer is a high-performance storage layer used on HPC systems to stage or absorb I/O for data-intensive jobs.",
            "aliases": [
                "Burst Buffers",
                "Burst buffers",
                "Burst Buffer Resources",
                "Burst Buffer States",
                "Burst Buffer plugin"
            ]
        },
    },
    # 8. Non-domain example: numeric-heavy phrase
    {
        "term": "500 simple batch jobs",
        "context": "This example shows 500 simple batch jobs used only to illustrate the scheduler's scaling behavior.",
        "json": {
            "term": "500 simple batch jobs",
            "canonical": "500 simple batch jobs",
            "is_hpc_domain_term": False,
            "scheduler": "unknown",
            "category": "non_domain",
            "short_definition": "Not an HPC/scheduler-specific term; this is just a numeric example used in the documentation.",
            "aliases": []
        },
    },
]

# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

def init_llm_enrich_table(conn: sqlite3.Connection) -> None:
    """
    Ensure llm_enrich exists.

    One row per canonical_term with LLM-enriched information +
    frequency statistics and aliases.
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_enrich (
            enrich_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_term   TEXT    NOT NULL UNIQUE,
            example_term     TEXT    NOT NULL,
            scheduler        TEXT,
            category         TEXT,
            short_definition TEXT,
            is_hpc_domain    INTEGER NOT NULL DEFAULT 1,
            freq_total       INTEGER,
            doc_count        INTEGER,
            aliases_json     TEXT,
            raw_json         TEXT
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
# Fetch terms to enrich (resume-safe)
# -----------------------------------------------------------------------------

def fetch_terms_to_enrich(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Fetch ALL distinct canonical terms from llm_terms that are not yet enriched in llm_enrich,
    together with an example doc_id, chunk_id, context text, and frequency stats.

    Resume logic:
    - A term is "already enriched" if its canonical form exists in llm_enrich.canonical_term.
    - If a job is killed, all rows already inserted into llm_enrich are preserved.
    - On the next run, those terms are skipped automatically, so we continue where we stopped.
    """
    init_llm_enrich_table(conn)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            LOWER(TRIM(t.term))          AS canonical_term,
            MIN(t.term)                  AS example_term,
            MIN(t.doc_id)                AS example_doc_id,
            MIN(t.chunk_id)              AS example_chunk_id,
            MIN(c.text)                  AS context_text,
            COUNT(*)                     AS freq_total,
            COUNT(DISTINCT t.doc_id)     AS doc_count
        FROM llm_terms t
        JOIN contextual_chunk c
          ON c.doc_id = t.doc_id AND c.chunk_id = t.chunk_id
        LEFT JOIN llm_enrich e
          ON e.canonical_term = LOWER(TRIM(t.term))
        WHERE e.canonical_term IS NULL
        GROUP BY LOWER(TRIM(t.term))
        ORDER BY canonical_term
        """
    )

    rows = cur.fetchall()
    terms: List[Dict[str, Any]] = []
    for canonical, example_term, doc_id, chunk_id, ctx, freq_total, doc_count in rows:
        terms.append(
            {
                "canonical_term": canonical,
                "example_term": example_term,
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "context_text": ctx or "",
                "freq_total": freq_total,
                "doc_count": doc_count,
            }
        )
    return terms


# -----------------------------------------------------------------------------
# Prompt + LLM call (few-shot)
# -----------------------------------------------------------------------------

def build_enrich_prompt(term: str, context: str) -> str:
    """
    Build a Mistral [INST] style prompt for enriching a single term, with few-shot examples.
    """
    example_blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        example_blocks.append(
            "Example {}:\n"
            "TERM: {}\n"
            "CONTEXT:\n{}\n"
            "JSON:\n{}\n".format(
                i,
                ex["term"],
                ex["context"],
                json.dumps(ex["json"], ensure_ascii=False),
            )
        )
    examples_str = "\n".join(example_blocks)

    user_content = (
        "You will be given a single term and a short context snippet from HPC scheduler documentation.\n"
        "Your job is to enrich the term STRICTLY following the JSON schema and rules in the system prompt.\n\n"
        "Here are some examples of CORRECT enrichments:\n\n"
        f"{examples_str}\n"
        "Now enrich the NEW term below. Follow exactly the same JSON structure and style.\n\n"
        f"TERM:\n{term}\n\n"
        "CONTEXT (excerpt where the term appears):\n"
        f"{context}\n\n"
        "Respond ONLY with a single JSON object and nothing else."
    )

    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_content}\n"
        "[/INST]"
    )


def call_enrich_llm(tokenizer, model, device: str, term: str, context: str) -> str:
    prompt = build_enrich_prompt(term, context)

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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # decode only the completion (new tokens)
    gen_only_ids = generated_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_only_ids, skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Parsing enrichment JSON
# -----------------------------------------------------------------------------

def parse_enrich_output(raw_output: str, fallback_canonical: str, fallback_term: str) -> Dict[str, Any]:
    """
    Parse the LLM enrichment JSON safely.

    Returns a dict with:
      canonical_term, example_term, scheduler, category,
      short_definition, is_hpc_domain, aliases_json, raw_json
    """
    # Try to isolate the outermost JSON object
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        data = {}
    else:
        json_str = raw_output[start : end + 1]
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(json_str)
            except Exception:
                data = {}

    if not isinstance(data, dict):
        data = {}

    # Normalise fields
    term = str(data.get("term", fallback_term)).strip() or fallback_term
    canonical = str(data.get("canonical", fallback_canonical)).strip().lower()
    if not canonical:
        canonical = fallback_canonical

    is_domain = data.get("is_hpc_domain_term", True)
    if isinstance(is_domain, str):
        is_domain_lower = is_domain.strip().lower()
        is_hpc_domain = 1 if is_domain_lower in ("true", "yes", "1") else 0
    else:
        is_hpc_domain = 1 if bool(is_domain) else 0

    scheduler = str(data.get("scheduler", "unknown")).strip() or "unknown"
    category = str(data.get("category", "other_hpc")).strip() or "other_hpc"
    short_def = str(data.get("short_definition", "")).strip()

    # Aliases: list of strings → JSON-encoded
    aliases = data.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []
    cleaned_aliases = []
    for a in aliases:
        s = str(a).strip()
        if s:
            cleaned_aliases.append(s)
    aliases_json = json.dumps(cleaned_aliases, ensure_ascii=False)

    return {
        "canonical_term": canonical,
        "example_term": term,
        "scheduler": scheduler,
        "category": category,
        "short_definition": short_def,
        "is_hpc_domain": is_hpc_domain,
        "aliases_json": aliases_json,
        "raw_json": raw_output.strip(),
    }


# -----------------------------------------------------------------------------
# Main processing loop (process all pending terms; resume via DB)
# -----------------------------------------------------------------------------

def enrich_terms(
    conn: sqlite3.Connection,
    tokenizer,
    model,
    device: str,
) -> None:
    init_llm_enrich_table(conn)

    candidates = fetch_terms_to_enrich(conn)
    total = len(candidates)
    print(f"Enriching {total} terms...")

    if total == 0:
        return

    cur = conn.cursor()

    for idx, item in enumerate(candidates, start=1):
        canonical = item["canonical_term"]
        term = item["example_term"]
        ctx = item["context_text"]
        freq_total = item["freq_total"]
        doc_count = item["doc_count"]

        if idx == 1 or idx % 10 == 0:
            print(
                f"  -> term {idx}/{total}: '{term}' "
                f"(canonical='{canonical}', freq_total={freq_total}, doc_count={doc_count})"
            )

        raw = call_enrich_llm(tokenizer, model, device, term, ctx)
        parsed = parse_enrich_output(raw, fallback_canonical=canonical, fallback_term=term)

        # Debug for first term in this run
        if idx == 1:
            print("\n=== DEBUG FIRST TERM RAW (first 600 chars) ===")
            print(raw[:600])
            print("\n=== DEBUG FIRST TERM PARSED ===")
            print(parsed)
            print("------\n")

        cur.execute(
            """
            INSERT OR IGNORE INTO llm_enrich (
                canonical_term,
                example_term,
                scheduler,
                category,
                short_definition,
                is_hpc_domain,
                freq_total,
                doc_count,
                aliases_json,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parsed["canonical_term"],
                parsed["example_term"],
                parsed["scheduler"],
                parsed["category"],
                parsed["short_definition"],
                parsed["is_hpc_domain"],
                freq_total,
                doc_count,
                parsed["aliases_json"],
                parsed["raw_json"],
            ),
        )
        conn.commit()  # commit after each term so progress is saved even if the job is killed

    print("Term enrichment done.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based term enrichment over llm_terms (HPC docs)."
    )
    parser.add_argument(
        "--debug-first",
        action="store_true",
        help="Enrich only the first pending term and print raw + parsed output (no DB writes).",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        tokenizer, model, device = load_mistral()

        if args.debug_first:
            candidates = fetch_terms_to_enrich(conn)
            if not candidates:
                print("No terms left to enrich.")
            else:
                item = candidates[0]
                canonical = item["canonical_term"]
                term = item["example_term"]
                ctx = item["context_text"]
                print(f"DEBUG term='{term}', canonical='{canonical}'")

                raw = call_enrich_llm(tokenizer, model, device, term, ctx)
                print("\n=== RAW OUTPUT (first 800 chars) ===")
                print(raw[:800])

                parsed = parse_enrich_output(raw, fallback_canonical=canonical, fallback_term=term)
                print("\n=== PARSED ENRICHMENT ===")
                print(parsed)
                if not parsed["short_definition"]:
                    print("(Warning: empty short_definition)")
        else:
            enrich_terms(
                conn,
                tokenizer,
                model,
                device,
            )

    finally:
        conn.close()
