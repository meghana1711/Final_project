import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple, Optional
from contextlib import nullcontext

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# Allowed output values (keep validation so DB is consistent)
ALLOWED_SCHED = {"slurm", "lsf", "both", "generic", "unknown"}
ALLOWED_CAT = {
    "scheduler", "command", "option_flag", "config_param", "config_file", "log_or_state_path",
    "queue_or_partition", "resource", "job_state", "user_role", "other_hpc", "non_domain",
}
ALLOWED_ROLE = {"class", "object_property", "datatype_property", "individual", "drop"}
ALLOWED_DUL = {"object", "information_object", "description", "situation", "event", "role", "unknown"}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# -----------------------------
# Prompt config (SYSTEM + fewshot)
# -----------------------------

def load_prompt_config(path: str) -> Tuple[str, List[Dict[str, Any]]]:
    cfg = json.load(open(path, "r", encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("prompt_config must be a JSON object")

    system = str(cfg.get("system_prompt", "")).strip()
    if not system:
        raise ValueError("prompt_config missing 'system_prompt'")

    fewshot = cfg.get("fewshot", []) or []
    if not isinstance(fewshot, list):
        raise ValueError("'fewshot' must be a list")

    cleaned: List[Dict[str, Any]] = []
    for ex in fewshot:
        if not isinstance(ex, dict):
            continue
        term = str(ex.get("term", "")).strip()
        js = ex.get("json", {})
        if term and isinstance(js, dict):
            cleaned.append({
                "term": term,
                "context": str(ex.get("context", "")).strip(),
                "json": js
            })

    return system, cleaned


def format_fewshot(fewshot: List[Dict[str, Any]]) -> str:
    if not fewshot:
        return ""
    lines = ["FEW-SHOT EXAMPLES:"]
    for i, ex in enumerate(fewshot, 1):
        lines.append(f"\nExample {i} TERM:\n{ex.get('term','')}")
        lines.append(f"Example {i} CONTEXT:\n{ex.get('context','')}")
        lines.append(f"Example {i} OUTPUT:\n{json.dumps(ex.get('json',{}), ensure_ascii=False, indent=2)}")
    lines.append("\nEND EXAMPLES.\n")
    return "\n".join(lines)


def build_prompt(system: str, fewshot: List[Dict[str, Any]], term: str, context: str) -> str:
    return (
        system
        + "\n"
        + format_fewshot(fewshot)
        + "\nNOW PROCESS:\n"
        + f"TERM:\n{term}\n\nCONTEXT:\n{context}\n\nOUTPUT JSON ONLY:\n"
    )


# -----------------------------
# LLM runner
# -----------------------------

@dataclass
class HFLLM:
    model_id: str
    dtype: str = "auto"
    device: str = "auto"
    max_new_tokens: int = 200
    max_input_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0

    tokenizer: Any = None
    model: Any = None

    def load(self):
        torch_dtype = torch.float16 if self.dtype == "float16" else (
            torch.bfloat16 if self.dtype == "bfloat16" else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.device in ("cuda", "cuda:0"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch_dtype
            ).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch_dtype, device_map=self.device
            )

        self.model.eval()

    def generate_batch(self, prompts: List[str], batch_size: int) -> List[str]:
        if not prompts:
            return []
        res: List[str] = []
        do_sample = self.temperature > 0.0

        use_cuda = hasattr(self.model, "device") and getattr(self.model.device, "type", "") == "cuda"
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inp = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens
            )
            inp = {k: v.to(self.model.device) for k, v in inp.items()}

            with torch.no_grad(), autocast_ctx:
                out = self.model.generate(
                    **inp,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else None,
                    top_p=self.top_p if do_sample else None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            dec = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            for p, t in zip(batch, dec):
                res.append(t[len(p):].strip() if t.startswith(p) else t.strip())
        return res


# -----------------------------
# JSON parsing + validation
# -----------------------------

def parse_json(txt: str) -> Dict[str, Any]:
    m = JSON_RE.search((txt or "").strip())
    if not m:
        return {"_parse_error": "no_json_found", "_raw": txt}
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception as e:
        return {"_parse_error": str(e), "_raw": txt, "_json_candidate": blob}


def validate_schema(out: Dict[str, Any], term: str) -> Dict[str, Any]:
    """
    Keep strict validation so downstream tables don’t break.
    No deterministic overrides; LLM decides.
    """
    fallback = {
        "term": term,
        "canonical": term.strip().lower(),
        "is_hpc_domain_term": False,
        "scheduler": "unknown",
        "category": "non_domain",
        "ontology_role": "drop",
        "dul_bucket": "unknown",
        "confidence": 0.0,
        "short_definition": "",
        "aliases": [],
    }

    if "_parse_error" in out:
        fallback["_error"] = out["_parse_error"]
        return fallback

    required = {
        "term", "canonical", "is_hpc_domain_term", "scheduler", "category",
        "ontology_role", "dul_bucket", "confidence", "short_definition", "aliases"
    }
    if not required.issubset(set(out.keys())):
        fallback["_error"] = "missing_keys"
        return fallback

    scheduler = str(out["scheduler"]).strip().lower()
    category = str(out["category"]).strip()
    role = str(out["ontology_role"]).strip().lower()
    dul = str(out["dul_bucket"]).strip().lower()

    if scheduler not in ALLOWED_SCHED:
        scheduler = "unknown"
    if category not in ALLOWED_CAT:
        category = "other_hpc" if bool(out.get("is_hpc_domain_term")) else "non_domain"
    if role not in ALLOWED_ROLE:
        role = "class" if bool(out.get("is_hpc_domain_term")) else "drop"
    if dul not in ALLOWED_DUL:
        dul = "unknown"

    try:
        conf = clamp01(float(out.get("confidence", 0.0)))
    except Exception:
        conf = 0.0

    aliases = out.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []

    canonical = str(out.get("canonical", "")).strip().lower() or term.strip().lower()
    is_domain = bool(out.get("is_hpc_domain_term"))

    return {
        "term": term,
        "canonical": canonical,
        "is_hpc_domain_term": is_domain,
        "scheduler": scheduler,
        "category": category,
        "ontology_role": role,
        "dul_bucket": dul,
        "confidence": conf,
        "short_definition": str(out.get("short_definition", "")).strip(),
        "aliases": [str(a).strip() for a in aliases if str(a).strip()],
    }


# -----------------------------
# DB helpers
# -----------------------------

def init_dst_table(db_path: str, dst: str, src_table: str) -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {dst} (
      canonical_id INTEGER PRIMARY KEY,
      canonical_term TEXT NOT NULL,
      is_hpc_domain_term INTEGER NOT NULL,
      scheduler TEXT NOT NULL,
      category TEXT NOT NULL,
      ontology_role TEXT NOT NULL,
      dul_bucket TEXT NOT NULL,
      confidence REAL NOT NULL,
      short_definition TEXT,
      aliases_json TEXT,
      llm_json TEXT,
      evidence_json TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY (canonical_id)
        REFERENCES {src_table}(canonical_id)
        ON UPDATE CASCADE ON DELETE CASCADE
    );
    """)
    con.commit()
    con.close()


def fetch_evidence(
    cur: sqlite3.Cursor,
    tid: int,
    occ_table: str,
    sent_table: str,
    cleaned_version: int,
    max_sents: int
) -> List[str]:
    cur.execute(f"""
      SELECT o.doc_id, o.sent_idx
      FROM {occ_table} o
      WHERE o.term_id=? AND o.cleaned_version=?
      ORDER BY o.doc_id, o.sent_idx
      LIMIT ?
    """, (tid, cleaned_version, max_sents * 3))
    pairs = cur.fetchall()

    seen = set()
    out: List[str] = []
    for doc_id, sent_idx in pairs:
        if (doc_id, sent_idx) in seen:
            continue
        seen.add((doc_id, sent_idx))
        cur.execute(f"""
          SELECT sentence FROM {sent_table}
          WHERE doc_id=? AND sent_idx=? AND cleaned_version=?
        """, (doc_id, sent_idx, cleaned_version))
        row = cur.fetchone()
        if row and row[0]:
            s = str(row[0]).strip()
            if s:
                out.append(s)
        if len(out) >= max_sents:
            break
    return out


# -----------------------------
# Main run
# -----------------------------

def run(
    db_path: str,
    src_table: str,
    dst_table: str,
    occ_table: str,
    sent_table: str,
    cleaned_version: int,
    prompt_config: str,
    hf_model: str,
    dtype: str,
    device: str,
    batch_size: int,
    max_terms_llm: int,
    max_evidence_sents: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    log_every: int,
    reset_dst: bool,
    generic_score_max: float,
) -> None:
    system, fewshot = load_prompt_config(prompt_config)

    llm = HFLLM(
        model_id=hf_model,
        dtype=dtype,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    print(f"[INFO] Loading HF model: {hf_model} (device={device}, dtype={dtype})")
    llm.load()
    print("[INFO] HF model loaded.")

    init_dst_table(db_path, dst_table, src_table)

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    if reset_dst:
        cur.execute(f"DELETE FROM {dst_table};")
        con.commit()

    # -----------------------------------------
    # ONLY USE NON-GENERIC TERMS FROM term_enrichment
    # -----------------------------------------
    cur.execute(
        f"""
        SELECT canonical_id, canonical_term
        FROM {src_table}
        WHERE canonical_term IS NOT NULL
          AND TRIM(canonical_term) != ''
          AND generic_score < ?
          AND is_generic = 0
        ORDER BY canonical_id
        """,
        (generic_score_max,),
    )
    rows = [(int(r[0]), str(r[1] or "").strip()) for r in cur.fetchall() if r[1]]
    print(f"[INFO] After generic filter: {len(rows)} terms (generic_score < {generic_score_max}, is_generic=0)")

    # Optionally cap total number sent to LLM (for speed)
    if max_terms_llm > 0:
        rows = rows[:max_terms_llm]
        print(f"[INFO] Capped to max_terms_llm={max_terms_llm} terms.")

    # Build prompts (SYSTEM + FEWSHOT + EVIDENCE CONTEXT)
    prompts: List[str] = []
    meta: List[Tuple[int, str, List[str]]] = []
    for tid, term in rows:
        ev = fetch_evidence(cur, tid, occ_table, sent_table, cleaned_version, max_evidence_sents)
        ctx = " ".join(ev) if ev else ""
        prompts.append(build_prompt(system, fewshot, term, ctx))
        meta.append((tid, term, ev))

    # Generate
    gens = llm.generate_batch(prompts, batch_size=batch_size)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    insert_sql = f"""
    INSERT INTO {dst_table}
      (canonical_id, canonical_term, is_hpc_domain_term, scheduler, category, ontology_role, dul_bucket,
       confidence, short_definition, aliases_json, llm_json, evidence_json, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    written = 0
    total = len(meta)

    for i, (txt, (tid, term, ev)) in enumerate(zip(gens, meta), 1):
        parsed = parse_json(txt)
        validated = validate_schema(parsed, term)

        dbg = {"validated": validated, "raw": parsed, "evidence": ev}

        cur.execute(
            insert_sql,
            (
                tid, term,
                1 if validated["is_hpc_domain_term"] else 0,
                validated["scheduler"], validated["category"], validated["ontology_role"], validated["dul_bucket"],
                float(validated["confidence"]),
                validated["short_definition"],
                json.dumps(validated["aliases"], ensure_ascii=False),
                json.dumps(dbg, ensure_ascii=False),
                json.dumps({"evidence": ev}, ensure_ascii=False),
                now, now,
            )
        )
        written += 1

        if (i == 1) or (i % max(1, log_every) == 0) or (i == total):
            print(f"[PROGRESS] {i}/{total} last_id={tid} role={validated['ontology_role']} "
                  f"cat={validated['category']} conf={validated['confidence']:.2f}")

    con.commit()
    con.close()
    print(f"[DONE] Wrote {written} rows into {dst_table} (LLM-only labeling; no hardcore heuristics).")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Term enrichment extension (LLM-only labeling using improved prompts; generic-filtered input)"
    )

    ap.add_argument("--db", required=True)
    ap.add_argument("--src_table", default="term_enrichment")
    ap.add_argument("--dst_table", default="term_enrichment_exten")
    ap.add_argument("--term_occurrences_table", default="term_occurrences")
    ap.add_argument("--sentences_table", default="sentence_lemmatized")
    ap.add_argument("--cleaned_version", type=int, default=1)

    ap.add_argument("--prompt_config", required=True, help="JSON with {system_prompt, fewshot}")

    ap.add_argument("--hf_model", required=True)
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=8)

    ap.add_argument("--max_terms_llm", type=int, default=0,
                    help="Cap how many terms to send to LLM (0 = no cap).")
    ap.add_argument("--max_evidence_sents", type=int, default=3)

    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--no_reset", action="store_true")

    # NEW generic filter knob
    ap.add_argument("--generic_score_max", type=float, default=3.0,
                    help="Only consider src_table rows where generic_score < this and is_generic=0")
    ap.add_argument("--classify_all", action="store_true",
                help="Compatibility flag (ignored). LLM-only mode already classifies all selected terms.")

    args = ap.parse_args()

    run(
        db_path=args.db,
        src_table=args.src_table,
        dst_table=args.dst_table,
        occ_table=args.term_occurrences_table,
        sent_table=args.sentences_table,
        cleaned_version=args.cleaned_version,
        prompt_config=args.prompt_config,
        hf_model=args.hf_model,
        dtype=args.dtype,
        device=args.device,
        batch_size=args.batch_size,
        max_terms_llm=args.max_terms_llm,
        max_evidence_sents=args.max_evidence_sents,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        log_every=args.log_every,
        reset_dst=(not args.no_reset),
        generic_score_max=args.generic_score_max,
    )


if __name__ == "__main__":
    main()
