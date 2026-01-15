import argparse
import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Any, Set


# -------------------------------------------------------------------
# Canonicalization helpers
# -------------------------------------------------------------------

LIGHT_PREFIXES: Set[str] = {
    "a", "an", "the",
    "this", "that", "these", "those",
    "some", "any", "each", "every",
    "either", "neither", "no", "none",
    "most", "many", "much", "few", "several", "various", "multiple",
    "certain", "particular", "specific", "typical", "general", "common",
    "other",
}


def is_technical_token(term_text: str) -> bool:
    """
    Heuristic keep rules for important HPC/config tokens even if tf_idf is low.

    Keeps:
      - ALLCAPS + underscores (env/config vars): SLURM_JOB_ID, EGO_AUDIT_MAX_SIZE
      - Config-ish files: slurm.conf, lsf.conf, *.cfg, *.ini
      - Explicit scheduler mentions: contains 'slurm' or 'lsf'
    """
    if not term_text:
        return False

    t = term_text.strip()
    if not t:
        return False

    # Env/config style: ALLCAPS + '_' (or digits)
    if re.fullmatch(r"[A-Z0-9_]+", t) and "_" in t:
        return True

    lower = t.lower()

    # Config files
    if lower.endswith(".conf") or lower.endswith(".cfg") or lower.endswith(".ini"):
        return True

    # Scheduler mentions
    if "slurm" in lower or "lsf" in lower:
        return True

    return False


def _canonical_key(term_lemma: str, term_text: str) -> str:
    """
    Canonical key based on lemma (preferred), fallback to text.

    Behaviour:
      - Lemma tokens are already lowercased/singularized from extraction.
      - Normalize hyphens and slashes into spaces for non-env tokens.
      - Drop light prefixes at the *front* only.
      - For env-like tokens (ALLCAPS/underscore): keep as a single token.
    """
    lemma_raw = (term_lemma or "").strip()
    text = (term_text or "").strip()
    text_tokens = text.split() if text else []

    first_text = text_tokens[0] if text_tokens else ""
    is_env_like = bool(re.fullmatch(r"[A-Z0-9_]+", first_text))

    if is_env_like:
        # Keep whole token shape (no splitting on _)
        lemma_tokens = [lemma_raw.lower()] if lemma_raw else [first_text.lower()]
    else:
        lemma_norm = lemma_raw.lower()
        lemma_norm = re.sub(r"[-/]+", " ", lemma_norm)
        lemma_tokens = lemma_norm.split() if lemma_norm else []

        # fallback: approximate from text if lemma is missing
        if not lemma_tokens and text_tokens:
            lemma_norm = " ".join(text_tokens).lower()
            lemma_norm = re.sub(r"[-/]+", " ", lemma_norm)
            lemma_tokens = lemma_norm.split()

    if not lemma_tokens:
        return ""

    while lemma_tokens and lemma_tokens[0] in LIGHT_PREFIXES:
        lemma_tokens.pop(0)

    if not lemma_tokens:
        return ""

    return " ".join(lemma_tokens)


def _canonical_term_from_key(key: str) -> str:
    """
    Display label:
      - If ALLCAPS/underscore form -> upper
      - else -> keep lowercase lemma form
    """
    if not key:
        return key
    if re.fullmatch(r"[A-Z0-9_]+", key):
        return key.upper()
    return key


def infer_term_type(term_text: str) -> Optional[str]:
    """
    Very coarse type tagging (rules only):
      - ALLCAPS + underscores -> config
      - endswith .conf/.cfg/.ini -> file
      else None
    """
    if not term_text:
        return None

    t = term_text.strip()
    lower = t.lower()

    if re.fullmatch(r"[A-Z0-9_]+", t) and "_" in t:
        return "config"
    if lower.endswith(".conf") or lower.endswith(".cfg") or lower.endswith(".ini"):
        return "file"
    return None


# -------------------------------------------------------------------
# DB schema
# -------------------------------------------------------------------

def init_term_enrichment_table(
    db_path: str,
    term_candidates_table: str,
    term_enrichment_table: str,
) -> None:
    """
    Create term_enrichment table if missing.

    canonical_id is both the PK and a FK to term_candidates(term_id).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {term_enrichment_table} (
            canonical_id          INTEGER PRIMARY KEY,
            canonical_term        TEXT NOT NULL,
            term_type             TEXT,
            synonyms_json         TEXT,
            abbreviations_json    TEXT,
            member_term_ids_json  TEXT,
            freq_total            INTEGER,
            freq_docs             INTEGER,
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


# -------------------------------------------------------------------
# Main enrichment routine
# -------------------------------------------------------------------

def enrich_terms(
    db_path: str,
    term_candidates_table: str,
    term_enrichment_table: str,
    min_tf_idf: float,
    reset_enrichment: bool = True,
) -> None:
    """
    Read term_candidates, filter, canonicalize, group variants,
    and populate term_enrichment.

    Filters:
      - keep if tf_idf >= min_tf_idf OR is_technical_token(term_text)

    Canonicalization:
      - lemma-based canonical key + light prefix stripping + hyphen/slash normalize

    Grouping:
      - one row per canonical term

    Rebuild:
      - if reset_enrichment=True, deletes all rows from term_enrichment first
    """
    init_term_enrichment_table(db_path, term_candidates_table, term_enrichment_table)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    if reset_enrichment:
        cur.execute(f"DELETE FROM {term_enrichment_table};")

    cur.execute(
        f"""
        SELECT term_id, term_text, term_lemma, freq_total, freq_docs, COALESCE(tf_idf, 0.0)
        FROM {term_candidates_table}
        """
    )
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} rows from {term_candidates_table} for enrichment.")

    groups: Dict[str, Dict[str, Any]] = {}
    kept_count = 0

    for term_id, term_text, term_lemma, freq_total, freq_docs, tf_idf in rows:
        term_text = term_text or ""
        tf_idf = float(tf_idf or 0.0)

        # Filter: keep if statistically important OR technical token
        if tf_idf < min_tf_idf and not is_technical_token(term_text):
            continue
        kept_count += 1

        key = _canonical_key(term_lemma, term_text)
        if not key:
            continue

        canonical_term = _canonical_term_from_key(key)

        if canonical_term not in groups:
            groups[canonical_term] = {
                "ids": set(),
                "texts": set(),
                "freq_total": 0,
                "freq_docs": 0,
            }

        g = groups[canonical_term]
        g["ids"].add(int(term_id))
        if term_text.strip():
            g["texts"].add(term_text.strip())
        g["freq_total"] += int(freq_total or 0)
        g["freq_docs"] = max(g["freq_docs"], int(freq_docs or 0))

    print(f"Kept {kept_count} terms after filtering (tf_idf >= {min_tf_idf} OR technical).")
    print(f"Formed {len(groups)} canonical groups.")

    now = datetime.utcnow().isoformat(timespec="seconds")

    for canonical_term, data in groups.items():
        member_ids = sorted(data["ids"])
        member_texts = sorted(data["texts"])
        freq_total = data["freq_total"]
        freq_docs = data["freq_docs"]

        # canonical_id = smallest term_id in the group (deterministic)
        canonical_id = member_ids[0]

        # synonyms: all member texts except canonical label if present
        if canonical_term in member_texts:
            synonyms = [t for t in member_texts if t != canonical_term]
        else:
            synonyms = member_texts

        # abbreviations: ALLCAPS/underscore tokens only (rule-based)
        abbreviations = [
            t for t in member_texts
            if re.fullmatch(r"[A-Z0-9_]+", t) and len(t) > 1
        ]

        term_type = infer_term_type(canonical_term)
        if term_type is None:
            for t in member_texts:
                term_type = infer_term_type(t)
                if term_type is not None:
                    break

        cur.execute(
            f"""
            INSERT INTO {term_enrichment_table}
                (canonical_id, canonical_term, term_type,
                 synonyms_json, abbreviations_json, member_term_ids_json,
                 freq_total, freq_docs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_id,
                canonical_term,
                term_type,
                json.dumps(synonyms, ensure_ascii=False),
                json.dumps(abbreviations, ensure_ascii=False),
                json.dumps(member_ids, ensure_ascii=False),
                freq_total,
                freq_docs,
                now,
                now,
            ),
        )

    conn.commit()
    conn.close()
    print(f"{term_enrichment_table} rebuilt.")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Rule-based term enrichment (canonicalization + grouping).")

    ap.add_argument("--db", required=True)
    ap.add_argument("--term_candidates_table", default="term_candidates")
    ap.add_argument("--term_enrichment_table", default="term_enrichment")
    ap.add_argument("--min_tf_idf", type=float, default=9.0)
    ap.add_argument("--no_reset", action="store_true", help="Do not delete existing enrichment rows first")

    args = ap.parse_args()

    print(f"Running term enrichment: tf_idf >= {args.min_tf_idf} OR technical token...")
    enrich_terms(
        db_path=args.db,
        term_candidates_table=args.term_candidates_table,
        term_enrichment_table=args.term_enrichment_table,
        min_tf_idf=args.min_tf_idf,
        reset_enrichment=(not args.no_reset),
    )
    print("Done term enrichment.")


if __name__ == "__main__":
    main()
