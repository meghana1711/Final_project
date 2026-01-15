import argparse
import json
import math
import re
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Your SYSTEM prompt (keep it exactly as you like)
# -----------------------------

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
- "slurm"
- "lsf"
- "both"
- "generic"
- "unknown"

ALLOWED VALUES FOR "category":
- "scheduler"
- "command"
- "option_flag"
- "config_param"
- "config_file"
- "log_or_state_path"
- "queue_or_partition"
- "resource"
- "job_state"
- "user_role"
- "other_hpc"
- "non_domain"

OUTPUT FORMAT (STRICT):
You MUST respond with EXACTLY one JSON object and nothing else.

The JSON schema is:

{
  "term": "original term string",
  "canonical": "lowercased, trimmed canonical form",
  "is_hpc_domain_term": true or false,
  "scheduler": "slurm | lsf | both | generic | unknown",
  "category": "scheduler | command | option_flag | config_param | config_file | log_or_state_path | queue_or_partition | resource | job_state | user_role | other_hpc | non_domain",
  "short_definition": "one or two short sentences in HPC/scheduler context",
  "aliases": ["optional", "aliases", "may", "be", "empty"]
}

- Do NOT add any extra keys.
- Do NOT output any text before or after the JSON.
"""


# -----------------------------
# Minimal hard-case selection (optional, but recommended)
# -----------------------------

HARDCASE_RE = re.compile(r"(--|^-{1,2}\w|=|/etc/|\.conf$|\.cfg$|\.ini$|/)", re.IGNORECASE)


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


def is_hard_case(term_text: str, tf_idf: float, low: float, high: float) -> bool:
    if HARDCASE_RE.search(term_text or ""):
        return True
    return low <= tf_idf <= high


# -----------------------------
# Evidence from DB: term_occurrences -> sentence_lemmatized(sentence)
# -----------------------------

def fetch_evidence_sentences(
    cur: sqlite3.Cursor,
    term_id: int,
    term_occurrences_table: str,
    sentences_table: str,
    cleaned_version: int,
    max_sents: int,
) -> List[str]:
    cur.execute(
        f"""
        SELECT o.doc_id, o.sent_idx
        FROM {term_occurrences_table} o
        WHERE o.term_id = ? AND o.cleaned_version = ?
        ORDER BY o.doc_id, o.sent_idx
        LIMIT ?
        """,
        (term_id, cleaned_version, max_sents * 3),
    )
    pairs = cur.fetchall()

    evidence: List[str] = []
    seen = set()

    for doc_id, sent_idx in pairs:
        key = (doc_id, sent_idx)
        if key in seen:
            continue
        seen.add(key)

        cur.execute(
            f"""
            SELECT sentence
            FROM {sentences_table}
            WHERE doc_id = ? AND sent_idx = ? AND cleaned_version = ?
            """,
            (doc_id, sent_idx, cleaned_version),
        )
        row = cur.fetchone()
        if row and row[0]:
            s = str(row[0]).strip()
            if s:
                evidence.append(s)
        if len(evidence) >= max_sents:
            break

    return evidence


# -----------------------------
# HF model runner
# -----------------------------

@dataclass
class HFLLM:
    model_id: str
    dtype: str = "auto"      # auto|float16|bfloat16
    device: str = "auto"     # device_map
    max_new_tokens: int = 350
    temperature: float = 0.0
    top_p: float = 1.0

    def load(self):
        torch_dtype = None
        if self.dtype == "float16":
            torch_dtype = torch.float16
        elif self.dtype == "bfloat16":
            torch_dtype = torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = self.temperature > 0.0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                top_p=self.top_p if do_sample else None,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(txt: str) -> Dict[str, Any]:
    txt = (txt or "").strip()
    m = JSON_RE.search(txt)
    if not m:
        return {"_parse_error": "no_json_found", "_raw": txt}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception as e:
        return {"_parse_error": str(e), "_raw": txt, "_json_candidate": blob}


ALLOWED_SCHED = {"slurm", "lsf", "both", "generic", "unknown"}
ALLOWED_CAT = {
    "scheduler", "command", "option_flag", "config_param", "config_file", "log_or_state_path",
    "queue_or_partition", "resource", "job_state", "user_role", "other_hpc", "non_domain",
}


def validate_schema(out: Dict[str, Any], term: str) -> Dict[str, Any]:
    """
    Enforce exact schema; on failure, downgrade to conservative non_domain.
    """
    if "_parse_error" in out:
        return {
            "term": term,
            "canonical": term.strip().lower(),
            "is_hpc_domain_term": False,
            "scheduler": "unknown",
            "category": "non_domain",
            "short_definition": "",
            "aliases": [],
            "_error": out["_parse_error"],
        }

    # strict keys
    required = {"term", "canonical", "is_hpc_domain_term", "scheduler", "category", "short_definition", "aliases"}
    if not required.issubset(set(out.keys())):
        return {
            "term": term,
            "canonical": term.strip().lower(),
            "is_hpc_domain_term": False,
            "scheduler": "unknown",
            "category": "non_domain",
            "short_definition": "",
            "aliases": [],
            "_error": "missing_keys",
        }

    scheduler = str(out["scheduler"]).strip().lower()
    category = str(out["category"]).strip()

    if scheduler not in ALLOWED_SCHED:
        scheduler = "unknown"
    if category not in ALLOWED_CAT:
        category = "other_hpc" if bool(out.get("is_hpc_domain_term")) else "non_domain"

    aliases = out.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []

    canonical = str(out.get("canonical", "")).strip().lower()
    if not canonical:
        canonical = term.strip().lower()

    return {
        "term": str(out.get("term", term)),
        "canonical": canonical,
        "is_hpc_domain_term": bool(out.get("is_hpc_domain_term")),
        "scheduler": scheduler,
        "category": category,
        "short_definition": str(out.get("short_definition", "")).strip(),
        "aliases": [str(a).strip() for a in aliases if str(a).strip()],
    }


# -----------------------------
# Few-shot formatting
# -----------------------------

def format_fewshot_block(fewshot: List[Dict[str, Any]]) -> str:
    if not fewshot:
        return ""
    lines = ["FEW-SHOT EXAMPLES:"]
    for i, ex in enumerate(fewshot, 1):
        term = ex.get("term", "")
        context = ex.get("context", "")
        js = ex.get("json", {})
        lines.append(f"\nExample {i} TERM:\n{term}")
        lines.append(f"Example {i} CONTEXT:\n{context}")
        lines.append(f"Example {i} OUTPUT:\n{json.dumps(js, ensure_ascii=False, indent=2)}")
    lines.append("\nEND EXAMPLES.\n")
    return "\n".join(lines)


def build_prompt(term: str, context: str, fewshot: List[Dict[str, Any]]) -> str:
    fewshot_block = format_fewshot_block(fewshot)
    return (
        SYSTEM_PROMPT
        + "\n"
        + fewshot_block
        + "\n"
        + "NOW PROCESS:\n"
        + f"TERM:\n{term}\n\nCONTEXT:\n{context}\n"
        + "\nOUTPUT JSON ONLY:\n"
    )


# -----------------------------
# Destination table schema (rich)
# -----------------------------

def init_dst_table(db_path: str, term_candidates_table: str, dst_table: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {dst_table} (
            canonical_id          INTEGER PRIMARY KEY,
            canonical_term        TEXT NOT NULL,
            is_hpc_domain_term    INTEGER NOT NULL,  -- 0/1
            scheduler             TEXT NOT NULL,
            category              TEXT NOT NULL,
            short_definition      TEXT,
            aliases_json          TEXT,
            member_term_ids_json  TEXT,
            freq_total            INTEGER,
            freq_docs             INTEGER,
            llm_json              TEXT,
            evidence_json         TEXT,
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL,
            FOREIGN KEY (canonical_id)
                REFERENCES {term_candidates_table}(term_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


# -----------------------------
# Main routine
# -----------------------------

def run(
    db_path: str,
    term_candidates_table: str,
    term_occurrences_table: str,
    sentences_table: str,
    dst_table: str,
    cleaned_version: int,
    min_tf_idf_keep: float,
    hardcase_p_low: float,
    hardcase_p_high: float,
    max_terms_llm: int,
    max_evidence_sents: int,
    hf_model: Optional[HFLLM],
    fewshot: List[Dict[str, Any]],
    reset_dst: bool,
) -> None:
    init_dst_table(db_path, term_candidates_table, dst_table)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    if reset_dst:
        cur.execute(f"DELETE FROM {dst_table};")
        conn.commit()

    cur.execute(
        f"""
        SELECT term_id, term_text, term_lemma, freq_total, freq_docs, COALESCE(tf_idf, 0.0)
        FROM {term_candidates_table}
        """
    )
    rows = cur.fetchall()
    print(f"[INFO] Loaded {len(rows)} rows from {term_candidates_table}.")

    tfidf_vals = [float(r[5] or 0.0) for r in rows]
    low = percentile(tfidf_vals, hardcase_p_low)
    high = percentile(tfidf_vals, hardcase_p_high)
    print(f"[INFO] Hard-case TF-IDF band: [{low:.4f}, {high:.4f}]")

    # base keep: tfidf >= min_tf_idf_keep OR hardcase punct (so we don't miss flags)
    kept = []
    for term_id, term_text, term_lemma, freq_total, freq_docs, tf_idf in rows:
        term_text = (term_text or "").strip()
        tf_idf = float(tf_idf or 0.0)
        if tf_idf < min_tf_idf_keep and not HARDCASE_RE.search(term_text):
            # NOTE: still filtered; this stage isn't your main "drop all low tfidf"
            continue

        kept.append({
            "term_id": int(term_id),
            "term_text": term_text,
            "freq_total": int(freq_total or 0),
            "freq_docs": int(freq_docs or 0),
            "tf_idf": tf_idf,
            "hard": is_hard_case(term_text, tf_idf, low, high),
        })

    # only send limited hard cases to LLM, others will be grouped by rule canonical
    hard_cases = [x for x in kept if x["hard"]]
    hard_cases.sort(key=lambda x: (0 if HARDCASE_RE.search(x["term_text"]) else 1, -x["tf_idf"]))
    hard_cases = hard_cases[:max_terms_llm] if (hf_model and max_terms_llm > 0) else []
    hard_ids = {x["term_id"] for x in hard_cases}
    print(f"[INFO] Terms after base keep: {len(kept)} ; hard cases for LLM: {len(hard_cases)}")

    # LLM classify hard cases
    llm_out_by_term_id: Dict[int, Dict[str, Any]] = {}
    if hf_model and hard_cases:
        for i, item in enumerate(hard_cases, 1):
            tid = item["term_id"]
            term = item["term_text"]
            evidence = fetch_evidence_sentences(
                cur, tid, term_occurrences_table, sentences_table, cleaned_version, max_evidence_sents
            )
            context = " ".join(evidence) if evidence else ""

            prompt = build_prompt(term=term, context=context, fewshot=fewshot)
            gen = hf_model.generate(prompt)
            parsed = parse_json(gen)
            validated = validate_schema(parsed, term=term)

            llm_out_by_term_id[tid] = {
                "validated": validated,
                "raw": parsed if "_parse_error" not in parsed else {"raw": parsed},
                "evidence": evidence,
            }

            print(f"[LLM {i}/{len(hard_cases)}] term_id={tid} category={validated['category']} scheduler={validated['scheduler']}")

    # Grouping strategy:
    # - If LLM said non-domain => drop from enriched table
    # - Else canonical_term = validated.canonical
    # - For non-LLM terms: canonical = lower(term_text)
    groups: Dict[str, Dict[str, Any]] = {}

    for item in kept:
        tid = item["term_id"]
        term = item["term_text"]

        if tid in llm_out_by_term_id:
            v = llm_out_by_term_id[tid]["validated"]
            if not v["is_hpc_domain_term"] or v["category"] == "non_domain":
                continue
            canonical = v["canonical"]
            scheduler = v["scheduler"]
            category = v["category"]
            definition = v["short_definition"]
            aliases = v["aliases"]
            evidence = llm_out_by_term_id[tid]["evidence"]
            raw_llm = llm_out_by_term_id[tid]["raw"]
        else:
            canonical = term.lower().strip()
            scheduler = "unknown"
            category = "other_hpc"
            definition = ""
            aliases = []
            evidence = []
            raw_llm = None

        if not canonical:
            continue

        g = groups.setdefault(canonical, {
            "ids": set(),
            "terms": set(),
            "freq_total": 0,
            "freq_docs": 0,
            "scheduler": scheduler,
            "category": category,
            "definition": definition,
            "aliases": set(aliases),
            "llm_json_list": [],
            "evidence_list": [],
        })

        g["ids"].add(tid)
        g["terms"].add(term)
        g["freq_total"] += item["freq_total"]
        g["freq_docs"] = max(g["freq_docs"], item["freq_docs"])

        # if LLM available, store
        if raw_llm is not None:
            g["llm_json_list"].append({"term_id": tid, "out": raw_llm})
            g["evidence_list"].append({"term_id": tid, "evidence": evidence})
            for a in aliases:
                if a.strip():
                    g["aliases"].add(a.strip())

    print(f"[INFO] Final groups to write: {len(groups)}")

    now = datetime.utcnow().isoformat(timespec="seconds")
    insert_sql = f"""
        INSERT INTO {dst_table}
            (canonical_id, canonical_term, is_hpc_domain_term,
             scheduler, category, short_definition, aliases_json,
             member_term_ids_json, freq_total, freq_docs,
             llm_json, evidence_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    for canonical, g in groups.items():
        member_ids = sorted(g["ids"])
        canonical_id = member_ids[0]

        cur.execute(
            insert_sql,
            (
                canonical_id,
                canonical,
                1,
                g["scheduler"],
                g["category"],
                g["definition"],
                json.dumps(sorted(g["aliases"]), ensure_ascii=False),
                json.dumps(member_ids, ensure_ascii=False),
                int(g["freq_total"]),
                int(g["freq_docs"]),
                json.dumps(g["llm_json_list"], ensure_ascii=False) if g["llm_json_list"] else None,
                json.dumps(g["evidence_list"], ensure_ascii=False) if g["evidence_list"] else None,
                now,
                now,
            )
        )

    conn.commit()
    conn.close()
    print(f"[DONE] Wrote {len(groups)} rows into {dst_table}.")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="HF Mistral term enrichment using strict schema prompt -> term_enrichment_exten")

    ap.add_argument("--db", required=True)
    ap.add_argument("--term_candidates_table", default="term_candidates")
    ap.add_argument("--term_occurrences_table", default="term_occurrences")
    ap.add_argument("--sentences_table", default="sentence_lemmatized")
    ap.add_argument("--dst_table", default="term_enrichment_exten")

    ap.add_argument("--cleaned_version", type=int, default=1)
    ap.add_argument("--min_tf_idf_keep", type=float, default=9.0)

    ap.add_argument("--hardcase_p_low", type=float, default=60.0)
    ap.add_argument("--hardcase_p_high", type=float, default=85.0)
    ap.add_argument("--max_terms_llm", type=int, default=200)
    ap.add_argument("--max_evidence_sents", type=int, default=3)

    ap.add_argument("--hf_model", default=None, help="HF model id/path, e.g. mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max_new_tokens", type=int, default=350)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--fewshot_json", default=None, help="JSON file containing FEW_SHOT_EXAMPLES (optional)")
    ap.add_argument("--no_reset", action="store_true")

    args = ap.parse_args()

    fewshot = []
    if args.fewshot_json:
        with open(args.fewshot_json, "r", encoding="utf-8") as f:
            fewshot = json.load(f)
        if not isinstance(fewshot, list):
            raise ValueError("--fewshot_json must be a JSON list of examples")

    hf = None
    if args.hf_model:
        hf = HFLLM(
            model_id=args.hf_model,
            dtype=args.dtype,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"[INFO] Loading HF model: {args.hf_model}")
        hf.load()
        print("[INFO] HF model loaded.")
    else:
        print("[INFO] No --hf_model provided. This script is intended for LLM usage; provide --hf_model.")

    run(
        db_path=args.db,
        term_candidates_table=args.term_candidates_table,
        term_occurrences_table=args.term_occurrences_table,
        sentences_table=args.sentences_table,
        dst_table=args.dst_table,
        cleaned_version=args.cleaned_version,
        min_tf_idf_keep=args.min_tf_idf_keep,
        hardcase_p_low=args.hardcase_p_low,
        hardcase_p_high=args.hardcase_p_high,
        max_terms_llm=args.max_terms_llm,
        max_evidence_sents=args.max_evidence_sents,
        hf_model=hf,
        fewshot=fewshot,
        reset_dst=(not args.no_reset),
    )


if __name__ == "__main__":
    main()
