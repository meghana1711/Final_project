import argparse
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# -----------------------------
# Defaults (safe + bounded)
# -----------------------------

DEFAULT_RELATION_BLACKLIST = {
    "use", "have", "make", "do", "get", "set",
    "include", "contain", "provide", "allow",
}

DEFAULT_ALLOWED_RELATIONS = [
    # keep small + ontology-friendly
    "uses",
    "requires",
    "configures",
    "runs_on",
    "writes_to",
    "reads_from",
    "submits",
    "allocates",
    "limits",
    "sets",
    "enables",
    "disables",
    "reports",
    "stores",
    "manages",
    "depends_on",
    "unknown",  # model may choose if truly unclear
]

SYSTEM_PROMPT = """\
You are an expert in High Performance Computing (HPC) and job schedulers such as SLURM and IBM LSF.

Task: RELATION NORMALIZATION + TRIPLE VALIDATION for knowledge graph building.

You are given:
- subject (canonical term)
- relation phrase (free text extracted by OpenIE)
- object (canonical term)
- sentence evidence (the exact sentence it came from)

You must return JSON ONLY with:
{
  "accept": true|false,
  "normalized_relation": "<must be one of ALLOWED_RELATIONS>",
  "reason": "<short reason tied to the sentence evidence>",
  "evidence": "<short quote (<=20 words) from the sentence that supports your decision>"
}

Rules:
- Be conservative. If the triple is not clearly supported by the sentence, accept=false.
- "normalized_relation" MUST be exactly one of ALLOWED_RELATIONS.
- If relation is too vague ("use", "have", "do") and sentence does not specify a clear relation, accept=false.
- Do NOT hallucinate facts not present in the sentence.
- Output MUST be valid JSON and nothing else.
"""

def build_user_prompt(
    subj: str,
    rel_text: str,
    obj: str,
    sentence: str,
    allowed_relations: List[str],
) -> str:
    allowed = ", ".join(allowed_relations)
    return f"""\
ALLOWED_RELATIONS: [{allowed}]

SUBJECT: {subj}
RELATION: {rel_text}
OBJECT: {obj}

SENTENCE: {sentence}

Return JSON only.
"""


# -----------------------------
# DB helpers
# -----------------------------

def connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def ensure_llm_table(
    conn: sqlite3.Connection,
    out_table: str,
) -> None:
    """
    Stores LLM decisions (normalized relation + accept/reject + evidence).
    Does NOT drop the table.
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,

            -- reference to original edge (if present)
            source_edge_id      INTEGER,

            doc_id              TEXT,
            sentence_id         TEXT,
            sentence_text       TEXT,

            subj_canonical_id   INTEGER,
            subj_canonical_term TEXT,
            rel_text            TEXT,
            obj_canonical_id    INTEGER,
            obj_canonical_term  TEXT,

            -- computed features (for debugging/filtering)
            rel_len             INTEGER,
            has_prep            INTEGER,
            is_generic_rel      INTEGER,
            rel_support_total   INTEGER,
            rel_support_subj    INTEGER,
            rel_support_obj     INTEGER,

            -- LLM outputs
            accept              INTEGER,
            normalized_relation TEXT,
            reason              TEXT,
            evidence            TEXT,

            model_name          TEXT,
            created_at          TEXT NOT NULL,

            UNIQUE(subj_canonical_id, rel_text, obj_canonical_id, sentence_id, model_name)
        )
        """
    )
    conn.commit()


def compute_relation_support(
    conn: sqlite3.Connection,
    edges_table: str,
) -> Dict[str, Tuple[int, int, int]]:
    """
    Return rel_text -> (n_total, n_subj_distinct, n_obj_distinct)
    """
    rows = conn.execute(
        f"""
        SELECT
            rel_text AS rel,
            COUNT(*) AS n_total,
            COUNT(DISTINCT subj_canonical_id) AS n_subj,
            COUNT(DISTINCT obj_canonical_id) AS n_obj
        FROM {edges_table}
        GROUP BY rel_text
        """
    ).fetchall()

    out: Dict[str, Tuple[int, int, int]] = {}
    for r in rows:
        out[str(r["rel"])] = (int(r["n_total"]), int(r["n_subj"]), int(r["n_obj"]))
    return out


# -----------------------------
# Hard-case selection features
# -----------------------------

_PREP_WORDS = {"to", "for", "in", "on", "with", "from", "into", "over", "under", "by", "as", "at", "via"}

def has_preposition(rel_text: str) -> bool:
    toks = re.findall(r"[A-Za-z]+", (rel_text or "").lower())
    return any(t in _PREP_WORDS for t in toks)

def is_generic_relation(rel_text: str, blacklist: set) -> bool:
    rel = (rel_text or "").strip().lower()
    if not rel:
        return True
    # exact blacklist hit
    if rel in blacklist:
        return True
    # very short, single-verb relations tend to be generic
    toks = re.findall(r"[A-Za-z]+", rel)
    if len(toks) == 1 and len(toks[0]) <= 4:
        return True
    return False


@dataclass
class HardCasePolicy:
    """
    Decide which edges go to LLM.
    """
    validate_generic_rels: bool = True
    validate_low_support: bool = True
    min_support_total: int = 3
    min_support_subj: int = 2
    min_support_obj: int = 2


def should_send_to_llm(
    rel_text: str,
    support: Tuple[int, int, int],
    policy: HardCasePolicy,
    blacklist: set,
) -> bool:
    n_total, n_subj, n_obj = support

    generic = is_generic_relation(rel_text, blacklist)
    if policy.validate_generic_rels and generic:
        return True

    if policy.validate_low_support:
        if n_total < policy.min_support_total:
            return True
        if n_subj < policy.min_support_subj:
            return True
        if n_obj < policy.min_support_obj:
            return True

    return False


# -----------------------------
# HuggingFace model runner
# -----------------------------

def load_hf_textgen(model_name: str, device: str = "auto"):
    """
    Uses transformers pipeline. Works for Mistral Instruct.
    """
    from transformers import pipeline

    # device="auto" will choose GPU if available (accelerate)
    gen = pipeline(
        "text-generation",
        model=model_name,
        device_map=device,
        torch_dtype="auto",
    )
    return gen


def run_llm_one(
    gen,
    model_name: str,
    subj: str,
    rel_text: str,
    obj: str,
    sentence: str,
    allowed_relations: List[str],
    max_new_tokens: int = 220,
    temperature: float = 0.0,
) -> Optional[dict]:
    """
    Returns parsed JSON dict or None if parsing failed.
    """
    user_prompt = build_user_prompt(subj, rel_text, obj, sentence, allowed_relations)

    prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    out = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        return_full_text=False,
    )

    text = out[0]["generated_text"].strip()

    # Try to extract JSON object if model surrounds it with text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None

    json_str = m.group(0)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    # Validate schema keys
    needed = {"accept", "normalized_relation", "reason", "evidence"}
    if not needed.issubset(set(data.keys())):
        return None

    # Enforce allowed list
    nr = str(data.get("normalized_relation", "")).strip()
    if nr not in allowed_relations:
        data["normalized_relation"] = "unknown"

    # Normalize accept to bool-ish
    acc = data.get("accept", False)
    data["accept"] = bool(acc)

    # Clamp evidence length (soft)
    ev = str(data.get("evidence", "")).strip()
    if len(ev.split()) > 25:
        data["evidence"] = " ".join(ev.split()[:25])

    return data


# -----------------------------
# Main routine
# -----------------------------

def process_llm_extension(
    db: str,
    in_edges_table: str,
    out_llm_table: str,
    model_name: str,
    allowed_relations: List[str],
    relation_blacklist: set,
    limit: int = 0,
    only_hard_cases: bool = True,
    policy: HardCasePolicy = HardCasePolicy(),
    device: str = "auto",
    max_new_tokens: int = 220,
    temperature: float = 0.0,
) -> None:
    conn = connect(db)
    try:
        ensure_llm_table(conn, out_llm_table)

        support_map = compute_relation_support(conn, in_edges_table)

        # Load candidate edges
        q = f"""
        SELECT
            id,
            doc_id,
            sentence_id,
            sentence_text,
            subj_canonical_id,
            subj_canonical_term,
            rel_text,
            obj_canonical_id,
            obj_canonical_term
        FROM {in_edges_table}
        """
        if limit and limit > 0:
            q += " LIMIT ?"
            rows = conn.execute(q, (limit,)).fetchall()
        else:
            rows = conn.execute(q).fetchall()

        print(f"[INFO] Loaded {len(rows)} edges from {in_edges_table}.")

        gen = load_hf_textgen(model_name, device=device)

        inserted = 0
        processed = 0
        skipped = 0

        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cur = conn.cursor()

        for r in rows:
            processed += 1

            rel_text = (r["rel_text"] or "").strip()
            subj_term = (r["subj_canonical_term"] or "").strip()
            obj_term = (r["obj_canonical_term"] or "").strip()
            sent = (r["sentence_text"] or "").strip()

            if not rel_text or not subj_term or not obj_term or not sent:
                skipped += 1
                continue

            sup = support_map.get(rel_text, (0, 0, 0))
            generic = is_generic_relation(rel_text, relation_blacklist)
            hp = int(has_preposition(rel_text))

            if only_hard_cases:
                if not should_send_to_llm(rel_text, sup, policy, relation_blacklist):
                    skipped += 1
                    continue

            # Run LLM
            data = run_llm_one(
                gen=gen,
                model_name=model_name,
                subj=subj_term,
                rel_text=rel_text,
                obj=obj_term,
                sentence=sent,
                allowed_relations=allowed_relations,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if data is None:
                skipped += 1
                continue

            accept = 1 if data["accept"] else 0
            norm_rel = str(data["normalized_relation"]).strip()
            reason = str(data.get("reason", "")).strip()
            evidence = str(data.get("evidence", "")).strip()

            rel_len = len(rel_text)
            n_total, n_subj, n_obj = sup

            cur.execute(
                f"""
                INSERT OR IGNORE INTO {out_llm_table} (
                    source_edge_id,
                    doc_id, sentence_id, sentence_text,
                    subj_canonical_id, subj_canonical_term,
                    rel_text,
                    obj_canonical_id, obj_canonical_term,
                    rel_len, has_prep, is_generic_rel,
                    rel_support_total, rel_support_subj, rel_support_obj,
                    accept, normalized_relation, reason, evidence,
                    model_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(r["id"]),
                    r["doc_id"],
                    r["sentence_id"],
                    sent,
                    int(r["subj_canonical_id"]),
                    subj_term,
                    rel_text,
                    int(r["obj_canonical_id"]),
                    obj_term,
                    int(rel_len),
                    int(hp),
                    int(1 if generic else 0),
                    int(n_total),
                    int(n_subj),
                    int(n_obj),
                    int(accept),
                    norm_rel,
                    reason,
                    evidence,
                    model_name,
                    now,
                ),
            )
            inserted += cur.rowcount

            if processed % 50 == 0:
                conn.commit()
                print(f"[INFO] processed {processed}/{len(rows)} | inserted={inserted} | skipped={skipped}")

        conn.commit()
        print(f"[DONE] processed={processed} inserted={inserted} skipped={skipped}")
        print(f"[INFO] LLM outputs stored in table: {out_llm_table}")

    finally:
        conn.close()


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="LLM extension: normalize+validate non-taxonomic edges")
    ap.add_argument("--db", required=True)

    ap.add_argument("--in_edges_table", default="non_taxonomic_edges_clean",
                    help="Input edges table to validate (clean recommended).")
    ap.add_argument("--out_llm_table", default="non_taxonomic_edges_llm",
                    help="Output table that stores LLM decisions (no drop).")

    ap.add_argument("--model", required=True, help="HuggingFace model id, e.g. mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--device", default="auto", help="device_map for HF pipeline: auto/cuda/cpu")

    ap.add_argument("--allowed_relations_json", default="",
                    help="JSON list string of allowed relations. If empty, uses defaults.")
    ap.add_argument("--relation_blacklist_json", default="",
                    help="JSON list string of blacklisted generic relations. If empty, uses defaults.")

    ap.add_argument("--only_hard_cases", action="store_true",
                    help="If set, LLM runs only on hard cases (recommended).")
    ap.add_argument("--limit", type=int, default=0, help="Limit how many edges to scan (0 = all)")

    # Hard-case thresholds
    ap.add_argument("--min_support_total", type=int, default=3)
    ap.add_argument("--min_support_subj", type=int, default=2)
    ap.add_argument("--min_support_obj", type=int, default=2)

    # generation settings
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=0.0)

    return ap.parse_args()


def main():
    args = parse_args()

    allowed = DEFAULT_ALLOWED_RELATIONS
    if args.allowed_relations_json.strip():
        allowed = json.loads(args.allowed_relations_json)

    blacklist = DEFAULT_RELATION_BLACKLIST
    if args.relation_blacklist_json.strip():
        blacklist = set(json.loads(args.relation_blacklist_json))

    policy = HardCasePolicy(
        validate_generic_rels=True,
        validate_low_support=True,
        min_support_total=args.min_support_total,
        min_support_subj=args.min_support_subj,
        min_support_obj=args.min_support_obj,
    )

    process_llm_extension(
        db=args.db,
        in_edges_table=args.in_edges_table,
        out_llm_table=args.out_llm_table,
        model_name=args.model,
        allowed_relations=allowed,
        relation_blacklist=blacklist,
        limit=args.limit,
        only_hard_cases=args.only_hard_cases,
        policy=policy,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
