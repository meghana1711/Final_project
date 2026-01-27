import argparse
import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# --- Optional torch/transformers imports (only needed when running LLM) ---
def load_textgen(model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return tok, model


# -----------------------------
# Heuristics (same idea as earlier taxonomy typed parents)
# -----------------------------
JOB_STATES = {
    "pending", "running", "completed", "failed", "cancelled", "configuring",
    "completing", "suspended", "timeout", "node_fail", "preempted"
}

GENERIC_HEADS_DEFAULT = {
    "list", "type", "value", "system", "default", "name", "number", "limit", "array",
    "file", "files", "path", "directory", "mode", "size", "time"
}

def infer_typed_parent(term: str) -> Optional[str]:
    """Return one of: option_flag, config_param, config_file, log_or_state_path, job_state, command, resource, other_hpc"""
    if not term:
        return None
    t = term.strip()
    lower = t.lower()

    # option flag
    if lower.startswith("--") or re.fullmatch(r"-[A-Za-z]\b", t):
        return "option_flag"

    # config/env style var
    if re.fullmatch(r"[A-Z0-9_]+", t) and "_" in t:
        return "config_param"

    # config file
    if lower.endswith((".conf", ".cfg", ".ini")):
        return "config_file"

    # path-ish
    if lower.startswith("/") or "\\" in t:
        return "log_or_state_path"

    # job state
    if lower in JOB_STATES:
        return "job_state"

    # very small command whitelist (extend anytime)
    if lower in {
        "sbatch","srun","salloc","squeue","sacct","scancel","sinfo","scontrol",
        "bsub","bjobs","bqueues","bhosts","lsload","lsid","bhist"
    }:
        return "command"

    # resources (light)
    if lower in {"cpu","cpus","core","cores","gpu","gpus","memory","mem","node","nodes"}:
        return "resource"

    return None


def is_special_token(term: str) -> bool:
    if not term:
        return False
    t = term.strip()
    return (
        t.startswith("--")
        or re.fullmatch(r"-[A-Za-z]\b", t) is not None
        or (re.fullmatch(r"[A-Z0-9_]+", t) is not None and "_" in t)
        or t.lower().endswith((".conf", ".cfg", ".ini"))
        or t.startswith("/")
    )


# -----------------------------
# Prompting (bounded choice)
# -----------------------------
SYSTEM_PROMPT = """You are an expert in HPC schedulers (SLURM/LSF) and ontology taxonomy building.

You must VALIDATE/IMPROVE a proposed is_a taxonomy edge for a child term.

You will receive:
- child_term: the canonical child label
- proposed_parent: current parent (string)
- candidates: a list of allowed parent choices (strings) INCLUDING the proposed parent
- evidence: short documentation snippets where the child term appears (may be empty)

Your job:
1) Decide if an is_a edge should exist at all.
2) If yes, pick the best_parent strictly from candidates.
3) Be conservative. If evidence is weak and candidates are generic, prefer a typed parent like option_flag/config_param/config_file/resource/job_state.

Rules:
- You MUST pick best_parent from candidates, or "none".
- If the edge is wrong/noisy, return accept=false and best_parent="none".
- Output JSON only, exactly these keys:
  {"accept": true/false, "best_parent": "...", "confidence": 0.0-1.0, "reason": "..."}

No extra keys. No extra text.
"""

FEW_SHOTS = [
    {
        "child_term": "--nodes",
        "proposed_parent": "nodes",
        "candidates": ["nodes", "option_flag", "resource", "none"],
        "evidence": ["Use --nodes to request a specific number of nodes for the job allocation."],
        "out": {"accept": True, "best_parent": "option_flag", "confidence": 0.92, "reason": "Starts with '--' and used as a CLI option; typed parent is correct."}
    },
    {
        "child_term": "slurm.conf",
        "proposed_parent": "conf",
        "candidates": ["conf", "config_file", "file", "none"],
        "evidence": ["The slurm.conf file controls core SLURM configuration parameters."],
        "out": {"accept": True, "best_parent": "config_file", "confidence": 0.90, "reason": "This is a scheduler configuration file; config_file is the right parent."}
    },
    {
        "child_term": "node list",
        "proposed_parent": "list",
        "candidates": ["list", "resource", "option_flag", "other_hpc", "none"],
        "evidence": ["Specify the node list using --nodelist or via job constraints."],
        "out": {"accept": True, "best_parent": "other_hpc", "confidence": 0.65, "reason": "Head 'list' is generic; concept relates to node selection; other_hpc is safer than list."}
    },
]


def format_user_prompt(child_term: str, proposed_parent: str, candidates: List[str], evidence: List[str]) -> str:
    # Keep it small and deterministic
    payload = {
        "child_term": child_term,
        "proposed_parent": proposed_parent,
        "candidates": candidates,
        "evidence": evidence[:3],
    }
    return json.dumps(payload, ensure_ascii=False)


def generate_json(tok, model, system: str, user: str, max_new_tokens: int = 220) -> str:
    """
    Uses chat template if available; otherwise concatenates.
    Temperature fixed low for stability.
    """
    import torch

    # Try chat template
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = system + "\n\nUSER:\n" + user + "\n\nASSISTANT:\n"

    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)

    # Extract the last JSON object from decoded text
    # (model may echo prompt; we grab the last {...})
    m = re.findall(r"\{[\s\S]*\}", decoded)
    return m[-1] if m else ""


def safe_parse_llm_json(s: str) -> Optional[Dict]:
    try:
        obj = json.loads(s)
    except Exception:
        return None
    # strict keys
    if not isinstance(obj, dict):
        return None
    keys = set(obj.keys())
    if keys != {"accept", "best_parent", "confidence", "reason"}:
        return None
    if not isinstance(obj["accept"], bool):
        return None
    if not isinstance(obj["best_parent"], str):
        return None
    try:
        obj["confidence"] = float(obj["confidence"])
    except Exception:
        return None
    if not isinstance(obj["reason"], str):
        return None
    return obj


# -----------------------------
# DB: validated table
# -----------------------------
def init_validated_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            src_taxonomy_id          INTEGER,
            child_canonical_id       INTEGER NOT NULL,
            child_canonical_term     TEXT    NOT NULL,

            proposed_parent_head_text      TEXT NOT NULL,
            proposed_parent_canonical_id   INTEGER,
            proposed_parent_canonical_term TEXT NOT NULL,

            llm_accept               INTEGER NOT NULL,
            llm_best_parent          TEXT    NOT NULL,
            llm_best_parent_canonical_id   INTEGER,
            llm_best_parent_canonical_term TEXT,
            llm_confidence           REAL    NOT NULL,
            llm_reason               TEXT    NOT NULL,

            candidates_json          TEXT,
            evidence_json            TEXT,
            created_at               TEXT NOT NULL
        )
        """
    )
    conn.commit()


def fetch_evidence_snippets(
    conn: sqlite3.Connection,
    chunks_table: str,
    term: str,
    limit: int = 2
) -> List[str]:
    """
    Simple evidence retrieval: LIKE match in chunk text.
    This is cheap and works well enough for validation.
    """
    term = (term or "").strip()
    if not term:
        return []
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT text
        FROM {chunks_table}
        WHERE text LIKE ?
        LIMIT ?
        """,
        (f"%{term}%", limit),
    )
    return [r[0] for r in cur.fetchall() if r and r[0]]


def top_global_heads(conn: sqlite3.Connection, parent_candidates_table: str, k: int = 10) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT head_text
        FROM {parent_candidates_table}
        ORDER BY frequency DESC
        LIMIT ?
        """,
        (k,),
    )
    return [(r[0] or "").strip().lower() for r in cur.fetchall() if r and r[0]]


def validate_taxonomy_with_llm(
    db_path: str,
    taxonomy_table: str,
    enrichment_table: str,
    parent_candidates_table: str,
    chunks_table: str,
    out_table: str,
    llm_model: str,
    max_rows: int,
    evidence_k: int,
    global_heads_k: int,
    generic_heads: List[str],
) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    init_validated_table(conn, out_table)

    # preload global strong heads (adds better alternatives for generic heads)
    global_heads = top_global_heads(conn, parent_candidates_table, k=global_heads_k)

    # preload canonical term map for quick id lookup
    id2term: Dict[int, str] = {}
    label2id: Dict[str, int] = {}
    rows = conn.execute(
        f"""
        SELECT canonical_id, canonical_term
        FROM {enrichment_table}
        WHERE canonical_term IS NOT NULL AND TRIM(canonical_term) != ''
        """
    ).fetchall()
    for r in rows:
        cid = int(r["canonical_id"])
        lab = (r["canonical_term"] or "").strip()
        if not lab:
            continue
        id2term[cid] = lab
        label2id[lab.lower()] = cid

    # load taxonomy edges
    edges = conn.execute(
        f"""
        SELECT id, child_canonical_id, child_canonical_term,
               parent_head_text, parent_canonical_id, parent_canonical_term
        FROM {taxonomy_table}
        ORDER BY id
        LIMIT ?
        """,
        (max_rows,),
    ).fetchall()

    print(f"[INFO] Loaded {len(edges)} taxonomy edges from {taxonomy_table}.")
    if not edges:
        conn.close()
        return

    # load LLM
    print(f"[INFO] Loading HF model: {llm_model}")
    tok, model = load_textgen(llm_model)

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    inserted = 0
    processed = 0

    for e in edges:
        processed += 1
        tax_id = int(e["id"])
        child_id = int(e["child_canonical_id"])
        child_term = (e["child_canonical_term"] or "").strip()

        proposed_parent = (e["parent_canonical_term"] or e["parent_head_text"] or "").strip()
        proposed_head = (e["parent_head_text"] or "").strip().lower()
        proposed_parent_id = e["parent_canonical_id"]
        proposed_parent_id = int(proposed_parent_id) if proposed_parent_id is not None else None
        proposed_parent_term = (e["parent_canonical_term"] or proposed_head).strip()

        # ---- hard case filter ----
        hard = False
        if proposed_head in set(generic_heads):
            hard = True
        if proposed_parent_id is None:
            hard = True
        if is_special_token(child_term):
            hard = True

        if not hard:
            continue

        typed = infer_typed_parent(child_term)

        # candidate parents (bounded!)
        candidates = []
        if proposed_parent_term:
            candidates.append(proposed_parent_term)

        # typed parents are strong “ontology correct” parents
        if typed:
            candidates.append(typed)

        # add a few global strong heads as alternatives (avoid exploding candidates)
        # keep only non-generic heads
        for h in global_heads:
            if h and h not in set(generic_heads) and h != proposed_head:
                candidates.append(h)
            if len(candidates) >= 8:
                break

        # always allow "none"
        candidates.append("none")

        # dedupe while preserving order
        seen = set()
        cand_final = []
        for c in candidates:
            c = (c or "").strip()
            if not c:
                continue
            if c.lower() in seen:
                continue
            seen.add(c.lower())
            cand_final.append(c)

        evidence = fetch_evidence_snippets(conn, chunks_table, child_term, limit=evidence_k)

        # ---- Few-shot block (inline) ----
        # Keep prompt small: system + 2-3 few shots + current user payload
        prompt_system = SYSTEM_PROMPT
        prompt_user_parts = []

        for ex in FEW_SHOTS[:2]:
            prompt_user_parts.append("EXAMPLE_INPUT:\n" + format_user_prompt(
                ex["child_term"], ex["proposed_parent"], ex["candidates"], ex["evidence"]
            ))
            prompt_user_parts.append("EXAMPLE_OUTPUT:\n" + json.dumps(ex["out"], ensure_ascii=False))

        prompt_user_parts.append("INPUT:\n" + format_user_prompt(child_term, proposed_parent_term, cand_final, evidence))
        user_prompt = "\n\n".join(prompt_user_parts)

        raw = generate_json(tok, model, prompt_system, user_prompt)
        parsed = safe_parse_llm_json(raw)

        if parsed is None:
            # fail-safe: keep original edge if typed parent exists and generic head, else reject
            accept = True
            best_parent = typed if typed else proposed_parent_term
            conf = 0.25
            reason = "LLM parse failed; fallback decision."
        else:
            accept = bool(parsed["accept"])
            best_parent = parsed["best_parent"].strip()
            conf = float(parsed["confidence"])
            reason = parsed["reason"].strip()

        if best_parent.lower() not in {c.lower() for c in cand_final}:
            # enforce boundedness
            best_parent = "none"
            accept = False
            conf = min(conf, 0.3)
            reason = (reason + " (best_parent not in candidates; forced none)").strip()

        best_parent_id = label2id.get(best_parent.lower()) if best_parent.lower() != "none" else None
        best_parent_term = id2term.get(best_parent_id) if best_parent_id is not None else (best_parent if best_parent != "none" else None)

        conn.execute(
            f"""
            INSERT INTO {out_table} (
                src_taxonomy_id,
                child_canonical_id, child_canonical_term,
                proposed_parent_head_text, proposed_parent_canonical_id, proposed_parent_canonical_term,
                llm_accept, llm_best_parent, llm_best_parent_canonical_id, llm_best_parent_canonical_term,
                llm_confidence, llm_reason,
                candidates_json, evidence_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tax_id,
                child_id, child_term,
                proposed_head, proposed_parent_id, proposed_parent_term,
                1 if accept else 0,
                best_parent,
                best_parent_id,
                best_parent_term,
                conf,
                reason,
                json.dumps(cand_final, ensure_ascii=False),
                json.dumps(evidence, ensure_ascii=False),
                now,
            ),
        )
        inserted += 1

        if inserted % 25 == 0:
            conn.commit()
            print(f"[INFO] validated {inserted} edges...")

    conn.commit()
    conn.close()
    print(f"[INFO] Done. Inserted {inserted} validated rows into {out_table} (processed {processed} total edges, only hard-cases were validated).")


def main():
    ap = argparse.ArgumentParser(description="LLM validate/rerank taxonomy edges (bounded candidates).")
    ap.add_argument("--db", required=True)

    ap.add_argument("--taxonomy_table", default="taxonomy_is_a")
    ap.add_argument("--enrichment_table", default="term_enrichment")
    ap.add_argument("--parent_candidates_table", default="taxonomy_parent_candidates")
    ap.add_argument("--chunks_table", default="contextual_chunk")
    ap.add_argument("--out_table", default="taxonomy_is_a_validated")

    ap.add_argument("--model", required=True, help="HF model name/path, e.g. mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--max_rows", type=int, default=50000, help="Upper bound edges scanned; only hard cases validated.")
    ap.add_argument("--evidence_k", type=int, default=2, help="How many chunk snippets to attach.")
    ap.add_argument("--global_heads_k", type=int, default=10, help="Top heads to include as alternative parents.")
    ap.add_argument("--generic_heads", default=",".join(sorted(GENERIC_HEADS_DEFAULT)),
                    help="Comma-separated generic head stoplist for hard-case detection.")

    args = ap.parse_args()
    generic_heads = [h.strip().lower() for h in args.generic_heads.split(",") if h.strip()]

    validate_taxonomy_with_llm(
        db_path=args.db,
        taxonomy_table=args.taxonomy_table,
        enrichment_table=args.enrichment_table,
        parent_candidates_table=args.parent_candidates_table,
        chunks_table=args.chunks_table,
        out_table=args.out_table,
        llm_model=args.model,
        max_rows=args.max_rows,
        evidence_k=args.evidence_k,
        global_heads_k=args.global_heads_k,
        generic_heads=generic_heads,
    )


if __name__ == "__main__":
    main()
