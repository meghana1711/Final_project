import argparse
import json
import math
import re
import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# SYSTEM prompt
# -----------------------------

SYSTEM_PROMPT = """\
You are an expert in High Performance Computing (HPC) and job schedulers such as SLURM and IBM LSF.
Your task is PRECISE TERM ENRICHMENT for ONTOLOGY BUILDING.

For each input, you receive:
- a TERM string (already extracted and canonicalized), and
- a SHORT CONTEXT snippet (a few sentences of documentation where the term appears).

You must decide:
1) whether this is truly a meaningful HPC / scheduler domain term,
2) which scheduler(s) it belongs to,
3) which category it falls into,
4) what OWL/ontology role it should have (class/property/value/drop),
5) which DUL/DnS bucket it best fits (lightweight alignment),
6) provide a short, accurate definition and optional aliases.

Be conservative. If the term does not clearly refer to an HPC / scheduling concept,
classify it as non-domain and set ontology_role="drop".

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

ALLOWED VALUES FOR "ontology_role":
- "class"              (real domain concepts: node, partition, job, daemon)
- "object_property"    (relations/verbs: runsOn, requests, belongsTo)
- "datatype_property"  (literal-valued attributes: hasMemoryMB, hasTimeLimit, hasValue)
- "individual"         (fixed named values/states: YES/NO, RUNNING, PENDING, etc.)
- "drop"               (noise, doc artifacts, generic English, headings)

ALLOWED VALUES FOR "dul_bucket":
- "object"             (physical/logical entities: node, gpu, daemon)
- "information_object" (manual, log, config file, script text)
- "description"        (specifications: job script/spec, configuration specification)
- "situation"          (runtime situations: running job, queued job)
- "event"              (job submission, job start/finish; only if clearly event-like)
- "role"               (user/admin roles)
- "unknown"

OUTPUT FORMAT (STRICT):
You MUST respond with EXACTLY one JSON object and nothing else.

The JSON schema is:

{
  "term": "original term string",
  "canonical": "lowercased, trimmed canonical form",
  "is_hpc_domain_term": true or false,
  "scheduler": "slurm | lsf | both | generic | unknown",
  "category": "scheduler | command | option_flag | config_param | config_file | log_or_state_path | queue_or_partition | resource | job_state | user_role | other_hpc | non_domain",
  "ontology_role": "class | object_property | datatype_property | individual | drop",
  "dul_bucket": "object | information_object | description | situation | event | role | unknown",
  "confidence": 0.0 to 1.0,
  "short_definition": "one or two short sentences in HPC/scheduler context",
  "aliases": ["optional", "aliases", "may", "be", "empty"]
}

- Do NOT add any extra keys.
- Do NOT output any text before or after the JSON.
"""


# -----------------------------
# Helpers / heuristics
# -----------------------------

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
HARDCASE_RE = re.compile(r"(--|^-{1,2}\w|=|/etc/|\.conf$|\.cfg$|\.ini$|/)", re.IGNORECASE)
HEADING_MARK_RE = re.compile(r"^(?:\d+(?:\.\d+){0,4}[a-z]?|[A-Z](?:\.\d+){1,4}[a-z]?)$", re.IGNORECASE)
DOC_META_RE = re.compile(r"\b(example|examples|note|notes|section|table|figure|chapter|see|man page|manual|usage)\b", re.IGNORECASE)
LIKELY_VERB_SINGLE_RE = re.compile(r"^[a-z]{3,20}$", re.IGNORECASE)

ALLOWED_SCHED = {"slurm", "lsf", "both", "generic", "unknown"}
ALLOWED_CAT = {
    "scheduler", "command", "option_flag", "config_param", "config_file", "log_or_state_path",
    "queue_or_partition", "resource", "job_state", "user_role", "other_hpc", "non_domain",
}
ALLOWED_ROLE = {"class", "object_property", "datatype_property", "individual", "drop"}
ALLOWED_DUL = {"object", "information_object", "description", "situation", "event", "role", "unknown"}


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def is_risky_for_typing(term_text: str) -> bool:
    t = (term_text or "").strip()
    if not t:
        return True
    if HEADING_MARK_RE.match(t):
        return True
    if DOC_META_RE.search(t):
        return True
    if HARDCASE_RE.search(t):
        return True
    if " " not in t and LIKELY_VERB_SINGLE_RE.match(t) and not t.isupper():
        return True
    return False


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


def validate_schema(out: Dict[str, Any], term: str) -> Dict[str, Any]:
    """
    Enforce schema; on failure, downgrade to conservative non_domain + drop.
    """
    base_fallback = {
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
        base_fallback["_error"] = out["_parse_error"]
        return base_fallback

    required = {
        "term", "canonical", "is_hpc_domain_term", "scheduler", "category",
        "ontology_role", "dul_bucket", "confidence", "short_definition", "aliases"
    }
    if not required.issubset(set(out.keys())):
        base_fallback["_error"] = "missing_keys"
        return base_fallback

    scheduler = str(out["scheduler"]).strip().lower()
    category = str(out["category"]).strip()
    role = str(out["ontology_role"]).strip().lower()
    dul_bucket = str(out["dul_bucket"]).strip().lower()

    if scheduler not in ALLOWED_SCHED:
        scheduler = "unknown"
    if category not in ALLOWED_CAT:
        category = "other_hpc" if bool(out.get("is_hpc_domain_term")) else "non_domain"

    if role not in ALLOWED_ROLE:
        role = "drop" if category == "non_domain" or (not bool(out.get("is_hpc_domain_term"))) else "class"
    if dul_bucket not in ALLOWED_DUL:
        dul_bucket = "unknown"

    # confidence clamp
    try:
        conf = float(out.get("confidence", 0.0))
    except Exception:
        conf = 0.0
    conf = clamp01(conf)

    aliases = out.get("aliases", [])
    if not isinstance(aliases, list):
        aliases = []

    canonical = str(out.get("canonical", "")).strip().lower()
    if not canonical:
        canonical = term.strip().lower()

    is_domain = bool(out.get("is_hpc_domain_term"))

    # Conservative safety: if non-domain, always drop
    if (not is_domain) or category == "non_domain":
        role = "drop"
        dul_bucket = "unknown"
        conf = min(conf, 0.4)

    return {
        "term": str(out.get("term", term)),
        "canonical": canonical,
        "is_hpc_domain_term": is_domain,
        "scheduler": scheduler,
        "category": category,
        "ontology_role": role,
        "dul_bucket": dul_bucket,
        "confidence": conf,
        "short_definition": str(out.get("short_definition", "")).strip(),
        "aliases": [str(a).strip() for a in aliases if str(a).strip()],
    }


def compute_hybrid_confidence(
    llm_conf: float,
    evidence_count: int,
    freq_total: int,
    freq_docs: int,
    parse_error: bool,
    is_drop_or_nondomain: bool,
) -> float:
    """
    Hybrid confidence (0..1), deterministic:
    + llm_conf (0..1)
    + evidence bonus: up to +0.15 (0.05 per evidence sentence)
    + frequency bonus: up to +0.15 (log1p(freq_total)/20 capped)
    + docs bonus: up to +0.10 (log1p(freq_docs)/10 capped)
    - parse penalty: -0.30 if JSON parse failed
    Then clamp to [0,1]
    If drop/non-domain => cap at 0.4
    """
    llm_conf = clamp01(llm_conf)

    ev_bonus = min(0.15, 0.05 * max(0, int(evidence_count)))
    freq_bonus = min(0.15, math.log1p(max(0, int(freq_total))) / 20.0)
    docs_bonus = min(0.10, math.log1p(max(0, int(freq_docs))) / 10.0)
    parse_penalty = 0.30 if parse_error else 0.0

    hybrid = llm_conf + ev_bonus + freq_bonus + docs_bonus - parse_penalty
    hybrid = clamp01(hybrid)

    if is_drop_or_nondomain:
        hybrid = min(hybrid, 0.4)

    return hybrid


# -----------------------------
# Evidence from DB
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


def build_prompt(
    term: str,
    context: str,
    fewshot: List[Dict[str, Any]],
    term_type_hint: str = "",
    synonyms_hint: Optional[List[str]] = None,
    abbreviations_hint: Optional[List[str]] = None,
) -> str:
    fewshot_block = format_fewshot_block(fewshot)

    hints = []
    if term_type_hint:
        hints.append(f"- term_type_hint: {term_type_hint}")
    if synonyms_hint:
        hints.append(f"- synonyms_hint: {', '.join(synonyms_hint[:10])}")
    if abbreviations_hint:
        hints.append(f"- abbreviations_hint: {', '.join(abbreviations_hint[:10])}")

    hint_block = ""
    if hints:
        hint_block = "\nHINTS_FROM_V1:\n" + "\n".join(hints) + "\n"

    return (
        SYSTEM_PROMPT
        + "\n"
        + fewshot_block
        + "\n"
        + "NOW PROCESS:\n"
        + f"TERM:\n{term}\n\nCONTEXT:\n{context}\n"
        + hint_block
        + "\nOUTPUT JSON ONLY:\n"
    )


# -----------------------------
# HF model runner (GPU + batching)
# -----------------------------

@dataclass
class HFLLM:
    model_id: str
    dtype: str = "auto"       # auto|float16|bfloat16
    device: str = "auto"      # "cuda"|"cuda:0"|"cpu"|"auto"
    max_new_tokens: int = 200
    temperature: float = 0.0
    top_p: float = 1.0
    max_input_tokens: int = 2048

    tokenizer: Any = None
    model: Any = None

    def _torch_dtype(self):
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return None

    def load(self):
        torch_dtype = self._torch_dtype()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if self.device in ("cuda", "cuda:0"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available, but --device cuda was requested.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
            ).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device,
            )

        self.model.eval()

    def generate_batch(self, prompts: List[str], batch_size: int = 8) -> List[str]:
        if not prompts:
            return []

        results: List[str] = []
        do_sample = self.temperature > 0.0

        use_cuda = hasattr(self.model, "device") and getattr(self.model.device, "type", "") == "cuda"
        autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad(), autocast_ctx:
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else None,
                    top_p=self.top_p if do_sample else None,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            for prompt, text in zip(batch, decoded):
                results.append(text[len(prompt):].strip() if text.startswith(prompt) else text.strip())

        return results


# -----------------------------
# Output table schema (v2)
# -----------------------------

def init_dst_table(db_path: str, dst_table: str) -> None:
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
            ontology_role         TEXT NOT NULL,
            dul_bucket            TEXT NOT NULL,
            confidence            REAL NOT NULL,

            short_definition      TEXT,
            aliases_json          TEXT,

            term_type             TEXT,
            synonyms              TEXT,
            abbreviations         TEXT,
            member_term_ids       TEXT,

            freq_total            INTEGER,
            freq_docs             INTEGER,

            llm_json              TEXT,
            evidence_json         TEXT,

            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL,

            FOREIGN KEY (canonical_id)
                REFERENCES term_enrichment(canonical_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


# -----------------------------
# Pipeline
# -----------------------------

def load_term_enrichment_rows(cur: sqlite3.Cursor, src_table: str) -> List[Dict[str, Any]]:
    cur.execute(
        f"""
        SELECT
            canonical_id,
            canonical_term,
            term_type,
            synonyms_json,
            abbreviations_json,
            member_term_ids_json,
            COALESCE(freq_total, 0),
            COALESCE(freq_docs, 0)
        FROM {src_table}
        """
    )
    out: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        canonical_id = int(r[0])
        canonical_term = str(r[1] or "").strip()
        term_type = str(r[2] or "").strip()
        synonyms_json = r[3]
        abbreviations_json = r[4]
        member_ids_json = r[5]
        freq_total = int(r[6] or 0)
        freq_docs = int(r[7] or 0)

        syn: List[str] = []
        abbr: List[str] = []
        try:
            if synonyms_json:
                syn = json.loads(synonyms_json)
                if not isinstance(syn, list):
                    syn = []
        except Exception:
            syn = []
        try:
            if abbreviations_json:
                abbr = json.loads(abbreviations_json)
                if not isinstance(abbr, list):
                    abbr = []
        except Exception:
            abbr = []

        out.append({
            "canonical_id": canonical_id,
            "canonical_term": canonical_term,
            "term_type": term_type,
            "synonyms": [str(x).strip() for x in syn if str(x).strip()],
            "abbreviations": [str(x).strip() for x in abbr if str(x).strip()],
            "member_term_ids_json": member_ids_json,
            "freq_total": freq_total,
            "freq_docs": freq_docs,
        })
    return out


def run(
    db_path: str,
    src_table: str,
    term_occurrences_table: str,
    sentences_table: str,
    dst_table: str,
    cleaned_version: int,
    max_terms_llm: int,
    max_evidence_sents: int,
    log_every: int,
    hf_model: HFLLM,
    fewshot: List[Dict[str, Any]],
    reset_dst: bool,
    classify_all: bool,
    batch_size: int,
    only_risky: bool,
    keep_drop_rows: bool,
) -> None:
    init_dst_table(db_path, dst_table)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    if reset_dst:
        cur.execute(f"DELETE FROM {dst_table};")
        conn.commit()

    rows = load_term_enrichment_rows(cur, src_table=src_table)
    print(f"[INFO] Loaded {len(rows)} canonical terms from {src_table}.")

    kept = [r for r in rows if r["canonical_term"]]
    if only_risky:
        kept = [r for r in kept if is_risky_for_typing(r["canonical_term"])]

    if classify_all:
        llm_candidates = kept
    else:
        llm_candidates = [r for r in kept if is_risky_for_typing(r["canonical_term"])]

    if max_terms_llm > 0:
        llm_candidates = llm_candidates[:max_terms_llm]

    print(f"[INFO] Terms kept: {len(kept)} ; terms for LLM: {len(llm_candidates)} (batch_size={batch_size})")

    prompts: List[str] = []
    meta: List[Tuple[Dict[str, Any], List[str]]] = []

    for r in llm_candidates:
        tid = r["canonical_id"]
        term = r["canonical_term"]

        evidence = fetch_evidence_sentences(
            cur, tid, term_occurrences_table, sentences_table, cleaned_version, max_evidence_sents
        )
        context = " ".join(evidence) if evidence else ""

        prompt = build_prompt(
            term=term,
            context=context,
            fewshot=fewshot,
            term_type_hint=r.get("term_type", ""),
            synonyms_hint=r.get("synonyms", []),
            abbreviations_hint=r.get("abbreviations", []),
        )
        prompts.append(prompt)
        meta.append((r, evidence))

    gens = hf_model.generate_batch(prompts, batch_size=max(1, int(batch_size)))

    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    insert_sql = f"""
        INSERT INTO {dst_table}
            (canonical_id, canonical_term,
             is_hpc_domain_term, scheduler, category, ontology_role, dul_bucket, confidence,
             short_definition, aliases_json,
             term_type, synonyms, abbreviations, member_term_ids,
             freq_total, freq_docs,
             llm_json, evidence_json,
             created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    written = 0
    total = len(meta)

    for i, (gen, (r, evidence)) in enumerate(zip(gens, meta), 1):
        tid = r["canonical_id"]
        term = r["canonical_term"]

        parsed = parse_json(gen)
        parse_error = "_parse_error" in parsed
        validated = validate_schema(parsed, term=term)

        is_drop_or_nondomain = (
            validated["ontology_role"] == "drop"
            or validated["category"] == "non_domain"
            or (not validated["is_hpc_domain_term"])
        )

        # hybrid confidence override
        hybrid_conf = compute_hybrid_confidence(
            llm_conf=float(validated.get("confidence", 0.0)),
            evidence_count=len(evidence),
            freq_total=int(r.get("freq_total", 0)),
            freq_docs=int(r.get("freq_docs", 0)),
            parse_error=parse_error,
            is_drop_or_nondomain=is_drop_or_nondomain,
        )
        validated["confidence"] = hybrid_conf

        # Progress log (NO timing / ETA)
        if (i == 1) or (i % max(1, int(log_every)) == 0) or (i == total):
            print(
                f"[PROGRESS] {i}/{total} processed | "
                f"last_id={tid} conf={validated['confidence']:.2f} "
                f"role={validated['ontology_role']} cat={validated['category']}"
            )

        if (not keep_drop_rows) and is_drop_or_nondomain:
            continue

        cur.execute(
            insert_sql,
            (
                tid,
                validated["canonical"],
                1 if validated["is_hpc_domain_term"] else 0,
                validated["scheduler"],
                validated["category"],
                validated["ontology_role"],
                validated["dul_bucket"],
                float(validated["confidence"]),
                validated["short_definition"],
                json.dumps(validated["aliases"], ensure_ascii=False),

                r.get("term_type") or None,
                json.dumps(r.get("synonyms", []), ensure_ascii=False) if r.get("synonyms") else None,
                json.dumps(r.get("abbreviations", []), ensure_ascii=False) if r.get("abbreviations") else None,
                r.get("member_term_ids_json"),

                int(r.get("freq_total", 0)),
                int(r.get("freq_docs", 0)),

                json.dumps(
                    {
                        "validated": validated,
                        "raw": parsed,
                        "hybrid_conf_components": {
                            "llm_conf_raw": float(out_conf(parsed)),
                            "evidence_count": len(evidence),
                            "freq_total": int(r.get("freq_total", 0)),
                            "freq_docs": int(r.get("freq_docs", 0)),
                            "parse_error": parse_error,
                        }
                    },
                    ensure_ascii=False
                ),
                json.dumps({"evidence": evidence}, ensure_ascii=False) if evidence else None,

                now_iso,
                now_iso,
            )
        )
        written += 1

    conn.commit()
    conn.close()
    print(f"[DONE] Wrote {written} rows into {dst_table}.")


def out_conf(parsed: Dict[str, Any]) -> float:
    """
    Helper: try to extract original LLM confidence from raw parsed JSON (before validate/hybrid).
    Not critical; used only for llm_json bookkeeping.
    """
    try:
        if isinstance(parsed, dict) and "confidence" in parsed:
            return clamp01(float(parsed.get("confidence", 0.0)))
    except Exception:
        pass
    return 0.0


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Term enrichment extension v2: input term_enrichment, LLM refine, output term_enrichment_exten (GPU batching, no timing logs)"
    )

    ap.add_argument("--db", required=True)
    ap.add_argument("--src_table", default="term_enrichment")
    ap.add_argument("--dst_table", default="term_enrichment_exten")

    ap.add_argument("--term_occurrences_table", default="term_occurrences")
    ap.add_argument("--sentences_table", default="sentence_lemmatized")
    ap.add_argument("--cleaned_version", type=int, default=1)

    ap.add_argument("--hf_model", required=True, help="HF model id/path")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    ap.add_argument("--device", default="auto", help='Use "cuda" to force GPU-only, or "auto" for device_map.')
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--max_input_tokens", type=int, default=2048)

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_terms_llm", type=int, default=1000)
    ap.add_argument("--max_evidence_sents", type=int, default=3)
    ap.add_argument("--log_every", type=int, default=100, help="Log progress every N processed LLM terms (no timing).")

    ap.add_argument("--fewshot_json", default=None, help="Optional JSON list for few-shot examples")
    ap.add_argument("--no_reset", action="store_true", help="Do not clear dst table before writing")

    ap.add_argument("--classify_all", action="store_true",
                    help="Send ALL terms to LLM. If not set, default is risky-only.")
    ap.add_argument("--only_risky", action="store_true",
                    help="Hard filter: only process risky terms (even if --classify_all set).")
    ap.add_argument("--keep_drop_rows", action="store_true",
                    help="Keep rows even if LLM says drop/non-domain (default: skip drops).")

    args = ap.parse_args()

    fewshot: List[Dict[str, Any]] = []
    if args.fewshot_json:
        with open(args.fewshot_json, "r", encoding="utf-8") as f:
            fewshot = json.load(f)
        if not isinstance(fewshot, list):
            raise ValueError("--fewshot_json must be a JSON list")

    hf = HFLLM(
        model_id=args.hf_model,
        dtype=args.dtype,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_input_tokens=args.max_input_tokens,
    )
    print(f"[INFO] Loading HF model: {args.hf_model} (device={args.device}, dtype={args.dtype})")
    hf.load()
    print("[INFO] HF model loaded.")
    if hasattr(hf.model, "device"):
        print(f"[INFO] Model device: {hf.model.device}")

    run(
        db_path=args.db,
        src_table=args.src_table,
        term_occurrences_table=args.term_occurrences_table,
        sentences_table=args.sentences_table,
        dst_table=args.dst_table,
        cleaned_version=args.cleaned_version,
        max_terms_llm=args.max_terms_llm,
        max_evidence_sents=args.max_evidence_sents,
        hf_model=hf,
        log_every=args.log_every,
        fewshot=fewshot,
        reset_dst=(not args.no_reset),
        classify_all=args.classify_all,
        batch_size=args.batch_size,
        only_risky=args.only_risky,
        keep_drop_rows=args.keep_drop_rows,
    )


if __name__ == "__main__":
    main()
