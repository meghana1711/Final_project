import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Any


# DB schema helpers

def init_term_enrichment_table(db_path: str) -> None:
    """
    Create term_enrichment table if missing.

    One row per canonical_term.

    Columns:
      - canonical_term        : canonical label (PK), derived from term_lemma/text
      - term_type             : coarse type (command, config, file, ...)
      - synonyms_json         : JSON list of non-canonical surface variants
      - abbreviations_json    : JSON list of abbreviations
      - member_term_ids_json  : JSON list of term_ids grouped under this canonical_term
      - freq_total            : sum of freq_total over all member terms
      - freq_docs             : max of freq_docs over all member terms
      - created_at, updated_at
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS term_enrichment (
            canonical_term        TEXT PRIMARY KEY,
            term_type             TEXT,
            synonyms_json         TEXT,
            abbreviations_json    TEXT,
            member_term_ids_json  TEXT,
            freq_total            INTEGER,
            freq_docs             INTEGER,
            created_at            TEXT NOT NULL,
            updated_at            TEXT NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()


# Canonicalization (lemma-driven)

def _canonical_key(term_lemma: str, term_text: str) -> str:
    """
    Compute a canonical key using lemma first, with fallback to text.

    Basic idea:
      - use lemma tokens;
      - if first surface token is ALLCAPS/underscore (LSF, LSF_ENVDIR),
        group by first lemma token;
      - otherwise, drop the last lemma token.

    This merges things like:
      - "LSF management host" / "LSF client"      -> "lsf"
    """
    lemma = (term_lemma or "").strip()
    text = (term_text or "").strip()

    lemma_tokens = lemma.split() if lemma else []
    text_tokens = text.split() if text else []

    # Fallback: if lemma is missing, approximate from text
    if not lemma_tokens and text_tokens:
        lemma_tokens = [t.lower() for t in text_tokens]

    if not lemma_tokens:
        return ""

    # Single-token lemma → key is that lemma
    if len(lemma_tokens) == 1:
        return lemma_tokens[0]

    first_lemma = lemma_tokens[0]
    first_text = text_tokens[0] if text_tokens else first_lemma

    # Env/acro style: FIRST surface token ALLCAPS/underscore/digits
    if re.fullmatch(r"[A-Z0-9_]+", first_text):
        return first_lemma

    # Fallback: drop last lemma token
    return " ".join(lemma_tokens[:-1])


def _canonical_term_from_key(key: str) -> str:
    """
    Turn canonical key into canonical_term (display + group).

    - If it looks like ALLCAPS/underscore (env/config style) → upper-case it.
    - Otherwise keep as lemma form (likely lowercase).
    """
    if not key:
        return key

    if re.fullmatch(r"[A-Z0-9_]+", key):
        return key.upper()

    return key


# Type inference (rules only)

def infer_term_type(term_text: str) -> Optional[str]:
    """
    Very coarse type tagging using rules:

      - ^[A-Z0-9_]+$ and contains _  → "config"
      - ends with .conf/.cfg/.ini    → "file"

    Everything else → None.
    """
    t = term_text.strip()
    lower = t.lower()

    # Config-like variables: ALLCAPS + underscore(s)
    if re.fullmatch(r"[A-Z0-9_]+", t) and "_" in t:
        return "config"

    # Config files
    if lower.endswith(".conf") or lower.endswith(".cfg") or lower.endswith(".ini"):
        return "file"

    return None



# Main enrichment routine
def enrich_terms(db_path: str) -> None:
    """
    Read term_candidates, compute lemma-based canonical_term, synonyms, type,
    and populate term_enrichment.

    One row per canonical_term.

    Aggregates:
      - member_term_ids_json
      - freq_total (sum over members)
      - freq_docs (max over members)
    """
    init_term_enrichment_table(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # Need freq_total and freq_docs from term_candidates
    cur.execute(
        """
        SELECT term_id, term_text, term_lemma, freq_total, freq_docs
        FROM term_candidates
        """
    )
    rows = cur.fetchall() 

    groups: Dict[str, Dict[str, Any]] = {}

    for term_id, term_text, term_lemma, freq_total, freq_docs in rows:
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
        g["ids"].add(term_id)
        g["texts"].add(term_text)
        g["freq_total"] += int(freq_total or 0)
        g["freq_docs"] = max(g["freq_docs"], int(freq_docs or 0))

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"


    # Write one row per canonical_term
    for canonical_term, data in groups.items():
        member_ids = sorted(data["ids"])
        member_texts = sorted(data["texts"])
        freq_total = data["freq_total"]
        freq_docs = data["freq_docs"]

        # Synonyms: all member texts except the canonical label itself (if present)
        if canonical_term in member_texts:
            synonyms = [t for t in member_texts if t != canonical_term]
        else:
            synonyms = member_texts

        # Abbreviations (rules only): any member that is ALLCAPS/underscore
        abbreviations = [
            t for t in member_texts
            if re.fullmatch(r"[A-Z0-9_]+", t) and len(t) > 1
        ]

        # Very coarse type: based on canonical_term first, then fallback to any member
        term_type = infer_term_type(canonical_term)
        if term_type is None:
            for t in member_texts:
                term_type = infer_term_type(t)
                if term_type is not None:
                    break

        cur.execute(
            """
            INSERT INTO term_enrichment
                (canonical_term, term_type,
                 synonyms_json, abbreviations_json,
                 member_term_ids_json,
                 freq_total, freq_docs,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(canonical_term) DO UPDATE SET
                term_type             = COALESCE(excluded.term_type, term_enrichment.term_type),
                synonyms_json         = excluded.synonyms_json,
                abbreviations_json    = excluded.abbreviations_json,
                member_term_ids_json  = excluded.member_term_ids_json,
                freq_total            = excluded.freq_total,
                freq_docs             = excluded.freq_docs,
                updated_at            = excluded.updated_at
            """,
            (
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


def main():
    DB_PATH = r"onto_db/ontology_sample_new.db"  
    print("Running term enrichment (lemma-based canonical_term, rule-based only)...")
    enrich_terms(DB_PATH)
    print("Done term enrichment.")

if __name__ == "__main__":
    main()
