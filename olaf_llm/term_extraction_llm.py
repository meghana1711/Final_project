from __future__ import annotations

import sqlite3
import json
import ast
import re
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DB_PATH = "onto_db/onto_new.db"   # adjust if your DB path is different
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Use few-shot prompting by default ("few-shot" or "zero-shot")
PROMPT_MODE = "few-shot"
MAX_TERMS_PER_CHUNK = 25

# Make GPU math a bit faster
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SYSTEM_PROMPT = """\
You are an expert in High Performance Computing (HPC) and job schedulers like SLURM and IBM LSF.
Your task is STRICT DOMAIN TERM EXTRACTION.

You are given small chunks of HPC scheduler documentation.
For EACH chunk, extract a list of IMPORTANT HPC DOMAIN TERMS that are useful for building
an ontology or knowledge graph.

You MUST follow these rules exactly:

1. WHAT TO EXTRACT (GOOD TERMS)
   Extract ONLY terms that are clearly HPC / scheduler concepts, for example:

   1.1 Schedulers and components
       - SLURM, IBM LSF, Slurmctld, Slurmdbd, JobScheduler.

   1.2 Commands and subcommands
       - sbatch, srun, sacct, squeue, sreport, scancel, bsub, bjobs, lsload.

   1.3 Command-line options and flags (REAL options only)
       - Long options starting with '--' and containing letters:
         --partition, --time, --gres, --begin, --constraint, --cluster, --nodes, --mem.
       - Short options that are at least 3 letters long after removing non-letter symbols,
         and that look like real flags or parameters (for example: -gpu, -debug).

   1.4 Configuration parameters and plugin types
       - AccountingStorageType, AccountingStorageHost, JobAcctGatherType,
         JobCompType, SelectType, GresTypes, SchedulerType.

   1.5 Configuration, log, and state files (scheduler-specific)
       - slurm.conf, gres.conf, lsb.queues, lsf.cluster, lsf.shared,
         slurmdbd.conf, /var/log/slurm/slurmctld.log,
         /var/spool/slurm/statesave/jwt_hs256.key.

   1.6 Queues, partitions, QoS names
       - gpu partition, cpu partition, debug queue, normal queue, gpu-long,
         qos-high, qos-debug.

   1.7 Resources and job concepts
       - job array, job step, pending jobs, GPU resources, CPU cores,
         memory limit, node features, submission host, compute node.

2. WHAT TO IGNORE (BAD TERMS – DO NOT OUTPUT)
   The following MUST NOT be output as terms:

   2.1 Pure symbols or punctuation
       - "#", "*", "**", "--", "-", "=", "-=", "+=", "/*", "*/", "[]", "{}".

   2.2 Too-short terms (based on letters only)
       - ANY candidate whose LETTER-ONLY part is shorter than 3 characters
         MUST NOT be output as a term.
         Examples:
         - "p", "h", "k", "ff", "bb", "N", "K", "%K", "%N", "%j", "%a".
       - If a candidate contains symbols, conceptually REMOVE all non-letter characters
         and look at what remains. If the remaining letters are fewer than 3, discard it.

   2.3 Non-option garbage starting with dashes
       - Tokens like "--bbf", "--bb", "--ff", or any sequence of dashes and
         random letters that are not documented options MUST NOT be output.

   2.4 Pure numeric examples, time values, and ranges
       - "0.5 seconds", "0–255 range", "100 jobs", "1,024 nodes",
         "1154-node cluster", "1024M", "8 hours", "10 minutes".

   2.5 Generic phrases with no scheduler-specific meaning
       - "memory size", "memory requirements", "large number of jobs",
         "example value", "this section", "configuration example".

   2.6 Generic file system locations used only as examples
       - "/tmp", "/var/tmp", "/home/user/job", unless explicitly described
         as a scheduler state or log directory.

3. LENGTH AND SYMBOL RULES
   - Every term MUST contain at least one alphabetic character (a–z or A–Z).
   - A term MUST NOT be only punctuation characters.
   - For the purpose of deciding if something is a valid term, conceptually strip
     all non-letter characters (such as %, #, -, =, /, *, etc.).
       - If the remaining letters are fewer than 3 → DO NOT output it.
       - Only terms whose letter-only part has length >= 3 may be included.

4. IF YOU ARE UNSURE
   - If you are not clearly sure that a token is a meaningful HPC / scheduler term,
     DO NOT output it.
   - It is better to output FEWER, HIGH-QUALITY domain terms than to guess.

5. OUTPUT FORMAT (STRICT)
   - You MUST output ONLY one JSON object and nothing else.
   - JSON schema:
     {
       "terms": [
         "term1",
         "term2",
         "term3",
         ...
       ]
     }
   - Do NOT include explanations, reasons, or any extra keys.
   - Do NOT include duplicate terms within one chunk.
   - Do NOT include empty strings.
"""


# -----------------------------------------------------------------------------
# Few-shot examples (HPC domain)
# -----------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    # Example 1 – preemption, gang scheduling, priorities
    {
        "text": (
            "The preempted job will be cancelled. Enables gang scheduling (time slicing) of jobs in the same partition, "
            "and allows the resuming of suspended jobs. In order to use gang scheduling, the GANG option must be specified "
            "at the cluster level. If GANG scheduling is enabled with PreemptType=preempt/partition_prio, the controller "
            "will ignore PreemptExemptTime and the following PreemptParameters: reorder_count, strict_order, and "
            "youngest_first. Gang scheduling is performed independently for each partition, so if you only want time-slicing "
            "by OverSubscribe, without any preemption, then configuring partitions with overlapping nodes is not recommended. "
            "On the other hand, if you want to use PreemptType=preempt/partition_prio to allow jobs from higher PriorityTier "
            "partitions to Suspend jobs from lower PriorityTier partitions, then you will need overlapping partitions, and "
            "PreemptMode=SUSPEND,GANG to use Gang scheduler to resume the suspended job(s). In either case, time-slicing won't "
            "happen between jobs on different partitions. Preempts jobs by requeuing them (if possible) or canceling them. "
            "For jobs to be requeued they must have the \"--requeue\" sbatch option set or the cluster wide JobRequeue "
            "parameter in slurm.conf must be set to 1. The preempted jobs will be suspended, and later the Gang scheduler "
            "will resume them. Therefore, the SUSPEND preemption mode always needs the GANG option to be specified at the "
            "cluster level. Also, because the suspended jobs will still use memory on the allocated nodes, Slurm needs to be "
            "able to track memory resources to be able to suspend jobs. Because gang scheduling is performed independently "
            "for each partition, if using PreemptType=preempt/partition_prio then jobs in higher PriorityTier partitions "
            "will suspend jobs in lower PriorityTier partitions to run on the released resources. Only when the preemptor "
            "job ends then the suspended jobs will be resumed by the Gang scheduler. If PreemptType=preempt/qos is configured "
            "and if the preempted job(s) and the preemptor job from are on the same partition, then they will share resources "
            "with the Gang scheduler (time-slicing). If not (i.e. if the preemptees and preemptor are on different partitions) "
            "then the preempted jobs will remain suspended until the preemptor ends. PreemptType: Specifies the plugin used "
            "to identify which jobs can be preempted in order to start a pending job."
        ),
        "terms": [
            "gang scheduling",
            "time slicing",
            "GANG option",
            "PreemptType",
            "partition_prio",
            "preempt",
            "qos",
            "Gang scheduler",
            "PreemptExemptTime",
            "PreemptParameters",
            "reorder_count",
            "strict_order",
            "youngest_first",
            "sbatch",
            "PriorityTier",
            "PreemptMode",
            "SUSPEND",
            "--requeue",
            "JobRequeue",
            "slurm.conf",
            "plugin",
        ],
    },

    # Example 2 – fair-share formula and effective usage
    {
        "text": (
            "= 0.25\n"
            "Account F effective usage: 0.0 + ((0.25 - 0.0) * 35 / 60) = 0.1458\n"
            "The effective usage of each user is calculated using the same formula:\n"
            "User 1 effective usage: 0.2 + ((0.3875 - 0.2) * 1 / 1) = 0.3875\n"
            "User 2 effective usage: 0.25 + ((0.3 - 0.25) * 1 / 2) = 0.275\n"
            "User 3 effective usage: 0.0 + ((0.3 - 0.0) * 1 / 2) = 0.15\n"
            "User 4 effective usage: 0.25 + ((0.25 - 0.25) * 1 / 1) = 0.25\n"
            "User 5 effective usage: 0.0 + ((.1458 - 0.0) * 1 / 1) = 0.1458\n"
            "Using the Slurm fair-share formula,\n"
            " F = 2**(-UE/S)\n"
            "the fair-share factor for each user is:\n"
            "User 1 fair-share factor: 2**(-.3875 / .3) = 0.408479\n"
            "User 2 fair-share factor: 2**(-.275 / .05) = 0.022097\n"
            "User 3 fair-share factor: 2**(-.15 / .05) = 0.125000\n"
            "User 4 fair-share factor: 2**(-.25 / .25) = 0.500000\n"
            "User 5 fair-share factor: 2**(-.1458 / .35) = 0.749154\n"
            "From this example, once can see that users 1,2, and 3 are over-serviced while user 5 is under-serviced. "
            "Even though user 3 has yet to submit a job, his/her fair-share factor is negatively influenced by the "
            "jobs users 1 and 2 have run. Based on the fair-share factor alone, if all 5 users were to submit a job "
            "charging their respective accounts, user 5's job would be granted the highest scheduling priority."
        ),
        "terms": [
            "User",
            "job",
            "Slurm fair-share formula",
            "fair-share factor",
            "scheduling priority",
        ],
    },

    # Example 3 – TRES accounting commands and parameters
    {
        "text": (
            "If a Billing TRES is defined as a weight, it is ignored. sacct\n"
            "sacct can be used to view the TRES of each job by adding \"tres\" to the --format option. sacctmgr\n"
            "sacctmgr is used to view the various TRES available globally in the system. sacctmgr show tres will do this. sreport\n"
            "sreport reports on different TRES. Simply using the comma separated input option --tres= will have sreport generate "
            "reports available for the requested TRES types. More information about these reports can be found on the sreport "
            "manpage. In sreport, the \"Reported\" Billing TRES is calculated from the largest Billing TRES of each node "
            "multiplied by the time frame. For example, if a node is part of multiple partitions and each has a different "
            "TRESBillingWeights defined the Billing TRES for the node will be the highest of the partitions. If "
            "TRESBillingWeights is not defined on any partition for a node then the Billing TRES will be equal to the number "
            "of CPUs on the node."
        ),
        "terms": [
            "TRES",
            "Billing TRES",
            "sacct",
            "sacctmgr",
            "sreport",
            "--tres",
            "TRES types",
            "TRESBillingWeights",
        ],
    },

    # Example 4 – duplicate of TRES text, kept short but consistent
    {
        "text": (
            "If a Billing TRES is defined as a weight, it is ignored. sacct\n"
            "sacct can be used to view the TRES of each job by adding \"tres\" to the --format option. sacctmgr "
            "is used to view the various TRES available globally in the system. sacctmgr show tres will do this. "
            "sreport reports on different TRES and uses the --tres option to select TRES types. If TRESBillingWeights "
            "is not defined on any partition for a node then the Billing TRES will be equal to the number of CPUs on the node."
        ),
        "terms": [
            "TRES",
            "Billing TRES",
            "sacct",
            "sacctmgr",
            "sreport",
            "--tres",
            "TRESBillingWeights",
            "CPUs",
            "node",
        ],
    },

    # Example 5 – Slurm job_submit plugin build/install
    {
        "text": (
            "-g -O2 -pthread -fno-gcse -Werror "
            "-fno-strict-aliasing -MT job_submit_mine.lo "
            "-MD -MP -MF .deps/job_submit_mine.Tpo "
            "-c job_submit_mine.c -o .libs/job_submit_mine.o "
            "# Some clean up "
            "mv -f .deps/job_submit_mine.Tpo .deps/job_submit_mine.Plo "
            "rm -fr .libs/job_submit_mine.a .libs/job_submit_mine.la "
            ".libs/job_submit_mine.lai job_submit_mine.so "
            "# Link "
            "gcc -shared -fPIC -DPIC .libs/job_submit_mine.o -O2 "
            "-pthread -O0 -pthread -Wl,-soname -Wl,job_submit_mine.so "
            "-o job_submit_mine.so "
            "# Install "
            "cp job_submit_mine.so file "
            " /usr/local/lib/slurm/job_submit_mine.so"
        ),
        "terms": [
            "job_submit",
            "-pthread",
            "job_submit_mine",
        ],
    },
]



# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------

def init_llm_terms_table(conn: sqlite3.Connection) -> None:
    """
    Ensure llm_terms exists. One row per (doc_id, chunk_id, term).
    """
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_terms (
            term_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id    TEXT    NOT NULL,
            chunk_id  TEXT    NOT NULL,
            term      TEXT    NOT NULL,
            UNIQUE(doc_id, chunk_id, term)
        )
        """
    )
    conn.commit()


def dedupe_llm_terms(conn: sqlite3.Connection) -> None:
    """
    Build/update a deduplicated view of terms in llm_terms.

    Creates/overwrites llm_terms_unique with:
      - one row per unique term
      - example_doc_id, example_chunk_id
      - freq_total  = total occurrences across llm_terms
      - doc_count   = in how many distinct docs the term appears
    """
    print("Building llm_terms_unique from llm_terms...")
    init_llm_terms_table(conn)

    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_terms_unique (
            term              TEXT PRIMARY KEY,
            example_doc_id    TEXT,
            example_chunk_id  TEXT,
            freq_total        INTEGER,
            doc_count         INTEGER
        )
        """
    )
    # Clear previous contents so we rebuild from scratch
    #cur.execute("DELETE FROM llm_terms_unique")

    cur.execute(
        """
        INSERT INTO llm_terms_unique (term, example_doc_id, example_chunk_id, freq_total, doc_count)
        SELECT
            term,
            MIN(doc_id)  AS example_doc_id,
            MIN(chunk_id) AS example_chunk_id,
            COUNT(*)     AS freq_total,
            COUNT(DISTINCT doc_id) AS doc_count
        FROM llm_terms
        GROUP BY term
        """
    )
    conn.commit()
    print("Deduplication complete: llm_terms_unique rebuilt.")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Prompt building
# -----------------------------------------------------------------------------

def build_zero_shot_prompt(text: str, max_terms: int = MAX_TERMS_PER_CHUNK) -> str:
    """
    Simple instruction prompt: no examples, just schema + rules.
    """
    user_instructions = (
        "You will see examples of HPC documentation and the JSON terms extracted from them.\n"
        "Notice that the extracted terms are STABLE scheduler/configuration concepts, not raw numbers,\n"
        "example durations, or generic phrases like '100 jobs' or '0.5 seconds'.\n"
        "Follow the same behaviour for the NEW text.\n\n"
        f"{examples_str}"
        f"Now process ONLY the following NEW text.\n"
        f"Extract up to {max_terms} important HPC domain terms.\n"
        "Return ONLY one JSON object of the form:\n"
        '{ \"terms\": [\"term1\", \"term2\", \"term3\", ...] }\n'
        "Do NOT include pure numeric examples, time durations, or generic words as terms.\n\n"
        f"Text:\n{text}\n"
    )

    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_instructions}\n"
        "[/INST]"
    )


def build_few_shot_prompt(text: str, max_terms: int = MAX_TERMS_PER_CHUNK) -> str:
    """
    Few-shot prompt in Mistral [INST] style.
    The model sees HPC examples, then your new chunk.
    Expect format: { "terms": ["...", "...", ...] }
    """
    examples_str = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        ex_json = json.dumps({"terms": ex["terms"]}, indent=2)
        examples_str += (
            f"Example {i}:\n"
            f"Text:\n{ex['text']}\n"
            f"Valid JSON:\n{ex_json}\n\n"
        )

    user_instructions = (
        "You will see examples of HPC documentation and the JSON terms extracted from them.\n"
        "Follow the same behaviour for the NEW text.\n\n"
        f"{examples_str}"
        f"Now process ONLY the following NEW text.\n"
        f"Extract up to {max_terms} important HPC domain terms.\n"
        'Return ONLY one JSON object of the form { "terms": ["term1", "term2", ...] } and nothing else.\n\n'
        f"Text:\n{text}\n"
    )

    return (
        f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n"
        f"{user_instructions}\n"
        "[/INST]"
    )


def build_prompt(text: str, max_terms: int = MAX_TERMS_PER_CHUNK) -> str:
    if PROMPT_MODE == "few-shot":
        return build_few_shot_prompt(text, max_terms=max_terms)
    else:
        return build_zero_shot_prompt(text, max_terms=max_terms)


# -----------------------------------------------------------------------------
# LLM call
# -----------------------------------------------------------------------------

def call_llm(tokenizer, model, device: str, text: str) -> str:
    """
    Build the prompt, call Mistral, and return ONLY the completion text.
    """
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
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only NEW tokens (exclude the prompt part)
    gen_only_ids = generated_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_only_ids, skip_special_tokens=True)


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def _clean_term(term: str, seen: set[str]) -> Optional[str]:
    # Trim whitespace
    term = term.strip()
    if not term:
        return None

    # Must contain at least one alphanumeric character
    if not any(ch.isalnum() for ch in term):
        return None

    # For the "length < 3" rule, consider only LETTERS (ignore digits and symbols)
    letters_only = "".join(ch for ch in term if ch.isalpha())

    # If there are fewer than 3 letters total, reject the term
    if len(letters_only) < 3:
        return None

    # Drop pure numbers (after removing punctuation)
    numeric_candidate = term.replace(".", "").replace(",", "")
    if numeric_candidate.isdigit():
        return None

    # Optional extra safety: if more than half the characters are punctuation, it's probably junk
    non_alnum = sum(1 for ch in term if not ch.isalnum())
    if non_alnum > len(term) / 2:
        return None

    # Deduplicate within the chunk (case-insensitive)
    key = term.lower()
    if key in seen:
        return None
    seen.add(key)

    return term



def parse_terms(raw_output: str) -> List[str]:
    """
    Extract term strings from the model output.

    Strategy:
    1) Try to find JSON objects that contain a "terms" key and parse them using
       json / ast.literal_eval (handles proper JSON or Python dict style).
    2) If that fails (e.g., truncated / invalid JSON), fall back to a simple
       regex that extracts all quoted strings after "terms" and treats them as terms.

    Accepts:
      { "terms": [ "Slurm", "sacct", ... ] }
      and
      { "terms": [ {"term": "...", "reason": "..."}, ... ] }
    Returns:
      List[str] of unique, cleaned terms.
    """

    # ----------------------
    # 1) JSON-style parsing
    # ----------------------
    candidates: List[str] = []
    search_pos = 0
    key = '"terms"'  # just the key

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
            data: Any = None

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
            seen: set[str] = set()
            cleaned_terms: List[str] = []

            for item in items:
                if isinstance(item, str):
                    candidate = item
                elif isinstance(item, dict):
                    candidate = str(item.get("term", ""))
                else:
                    continue

                term = _clean_term(candidate, seen)
                if term is not None:
                    cleaned_terms.append(term)

            if cleaned_terms:
                return cleaned_terms

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

    # First match should be the key "terms", skip it if so
    if matches[0] == "terms":
        values = matches[1:]
    else:
        values = matches

    seen: set[str] = set()
    cleaned_terms: List[str] = []

    for candidate in values:
        term = _clean_term(candidate, seen)
        if term is not None:
            cleaned_terms.append(term)

    return cleaned_terms


# -----------------------------------------------------------------------------
# Chunk fetching
# -----------------------------------------------------------------------------

def fetch_chunks(
    conn: sqlite3.Connection,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
) -> List[tuple]:
    """
    Fetch chunks that still need LLM term extraction.

    - Use contextual_chunk.rowid for stable ordering and partitioning.
    - Skip chunks that already have entries in llm_terms (resume-safe).
    """
    # Ensure llm_terms exists before we reference it in the NOT EXISTS clause
    init_llm_terms_table(conn)

    cur = conn.cursor()

    sql = """
        SELECT rowid, chunk_id, doc_id, text
        FROM contextual_chunk
        WHERE rowid > ?
          AND NOT EXISTS (
              SELECT 1 FROM llm_terms t
              WHERE t.chunk_id = contextual_chunk.chunk_id
          )
        ORDER BY rowid
    """
    params = [offset_rowid]
    if max_chunks is not None:
        sql += " LIMIT ?"
        params.append(max_chunks)

    cur.execute(sql, params)
    return cur.fetchall()


# -----------------------------------------------------------------------------
# Main processing loop
# -----------------------------------------------------------------------------

def process_chunks(
    conn: sqlite3.Connection,
    tokenizer,
    model,
    device: str,
    max_chunks: Optional[int] = None,
    offset_rowid: int = 0,
) -> None:
    init_llm_terms_table(conn)

    rows = fetch_chunks(conn, max_chunks=max_chunks, offset_rowid=offset_rowid)
    total = len(rows)
    print(f"Processing {total} chunks (offset_rowid={offset_rowid})...")

    if total == 0:
        return

    cur = conn.cursor()

    for idx, (rowid, chunk_id, doc_id, chunk_text) in enumerate(rows, start=1):
        if idx == 1 or idx % 10 == 0:
            print(f"  -> chunk {idx}/{total} (rowid={rowid}, doc_id={doc_id}, chunk_id={chunk_id})")

        raw = call_llm(tokenizer, model, device, chunk_text)
        terms = parse_terms(raw)

        # Debug: for the very first processed chunk in this run, print terms
        if idx == 1:
            print("\n=== DEBUG FIRST CHUNK RAW (first 600 chars) ===")
            print(raw[:600])
            print("\n=== DEBUG FIRST CHUNK PARSED TERMS ===")
            print(terms)
            print("------\n")

        for term in terms:
            cur.execute(
                """
                INSERT OR IGNORE INTO llm_terms (doc_id, chunk_id, term)
                VALUES (?, ?, ?)
                """,
                (doc_id, chunk_id, term),
            )
        conn.commit()  # commit after each chunk so progress is never lost

    print("Done.")


# -----------------------------------------------------------------------------
# Entry point with debug + partial processing + dedupe
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based term extraction over contextual_chunk (HPC docs)."
    )
    parser.add_argument(
        "--debug-first-chunk",
        action="store_true",
        help="Run on a single (first) unprocessed chunk and print raw + parsed output (no DB writes).",
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
        help="Start from contextual_chunk.rowid > offset-rowid (for job arrays / resuming).",
    )
    parser.add_argument(
        "--dedupe-only",
        action="store_true",
        help="Do not run LLM; just build/update llm_terms_unique from existing llm_terms.",
    )
    parser.add_argument(
        "--dedupe-after",
        action="store_true",
        help="After processing chunks, build/update llm_terms_unique with unique terms.",
    )
    args = parser.parse_args()

    conn = sqlite3.connect(DB_PATH)
    try:
        if args.dedupe_only:
            # No GPU needed for dedupe-only mode
            dedupe_llm_terms(conn)
        else:
            tokenizer, model, device = load_mistral()

            if args.debug_first_chunk:
                # Just inspect one chunk (first unprocessed after offset_rowid)
                rows = fetch_chunks(conn, max_chunks=1, offset_rowid=args.offset_rowid)
                if not rows:
                    print("No unprocessed chunks found.")
                else:
                    rowid, chunk_id, doc_id, chunk_text = rows[0]
                    print(f"DEBUG rowid={rowid}, chunk_id={chunk_id}, doc_id={doc_id}")

                    raw = call_llm(tokenizer, model, device, chunk_text)
                    print("\n=== RAW OUTPUT (first 800 chars) ===")
                    print(raw[:800])

                    terms = parse_terms(raw)
                    print("\n=== PARSED TERMS ===")
                    for t in terms:
                        print("-", t)
                    if not terms:
                        print("(No terms parsed)")

            else:
                process_chunks(
                    conn,
                    tokenizer,
                    model,
                    device,
                    max_chunks=args.max_chunks,
                    offset_rowid=args.offset_rowid,
                )
                if args.dedupe_after:
                    dedupe_llm_terms(conn)

    finally:
        conn.close()
