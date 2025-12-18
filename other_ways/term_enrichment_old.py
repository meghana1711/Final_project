import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Any

# -------------------------------------------------------------------
# Config: TF-IDF threshold for enrichment
# -------------------------------------------------------------------

# Primary filter: keep statistically important terms
MIN_TF_IDF = 10.0

DB_PATH = r"onto_db/onto_new.db"


# -------------------------------------------------------------------
# DB schema helpers
# -------------------------------------------------------------------

def init_term_enrichment_table(db_path: str) -> None:
    """
    Create term_enrichment table if missing.

    One row per canonical group.

    Columns:
      - canonical_id         : INTEGER PK, also term_id of the canonical member in term_candidates
      - canonical_term       : canonical label (display string)
      - term_type            : coarse type (command, config, file, ...)
      - synonyms_json        : JSON list of non-canonical surface variants
      - abbreviations_json   : JSON list of abbreviations
      - member_term_ids_json : JSON list of term_ids grouped under this canonical_term
      - freq_total           : sum of freq_total over all member terms
      - freq_docs            : max of freq_docs over all member terms
      - created_at, updated_at
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS term_enrichment (
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
                REFERENCES term_candidates(term_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


# -------------------------------------------------------------------
# Canonicalization helpers
# -------------------------------------------------------------------

# Words we are happy to drop from the *front* of a term
LIGHT_PREFIXES = {
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
    Heuristic: keep rare but important HPC/config tokens even if tf_idf is low.

    Examples:
      - ALLCAPS + underscores: SLURM_JOB_ID, EGO_AUDIT_MAX_SIZE
      - Config files: slurm.conf, lsf.conf
      - Obvious scheduler mentions: contains 'slurm' or 'lsf'
    """
    if not term_text:
        return False

    t = term_text.strip()

    # Env/config style: ALLCAPS + '_' (or digits)
    if re.fullmatch(r"[A-Z0-9_]+", t) and "_" in t:
        return True

    lower = t.lower()

    # Config files
    if lower.endswith(".conf") or lower.endswith(".cfg") or lower.endswith(".ini"):
        return True

    # Explicit scheduler mentions
    if "slurm" in lower or "lsf" in lower:
        return True

    return False


def _canonical_key(term_lemma: str, term_text: str) -> str:
    """
    Compute a canonical key using lemma first, with fallback to text.

    Behaviour:
      - Use lemma tokens (already lowercased, singularised from extraction).
      - Drop only light quantifiers/modifiers at the *front*.
      - Group ALLCAPS/underscore env/acro terms by first *lemma* token.
      - Normalise simple punctuation (hyphens, slashes) for non-env terms.
    """
    lemma_raw = (term_lemma or "").strip()
    text = (term_text or "").strip()

    text_tokens = text.split() if text else []

    # Detect env/acro style based on surface form
    first_text = text_tokens[0] if text_tokens else ""
    is_env_like = bool(re.fullmatch(r"[A-Z0-9_]+", first_text))

    if is_env_like:
        # For env-like tokens, keep the first lemma token as-is (no splitting on '_')
        lemma_tokens = [lemma_raw.lower()] if lemma_raw else [first_text.lower()]
    else:
        # Normalise hyphens/underscores/slashes → spaces
        lemma_norm = lemma_raw.lower()
        lemma_norm = re.sub(r"[-/]+", " ", lemma_norm)
        lemma_tokens = lemma_norm.split() if lemma_norm else []

        # Fallback: if lemma is missing, approximate from text
        if not lemma_tokens and text_tokens:
            lemma_norm = " ".join(text_tokens).lower()
            lemma_norm = re.sub(r"[-/]+", " ", lemma_norm)
            lemma_tokens = lemma_norm.split()

    if not lemma_tokens:
        return ""

    # Drop light prefixes at the FRONT (most, particular, specific, ...)
    while lemma_tokens and lemma_tokens[0] in LIGHT_PREFIXES:
        lemma_tokens.pop(0)

    if not lemma_tokens:
        return ""

    return " ".join(lemma_tokens)


def _canonical_term_from_key(key: str) -> str:
    """
    Turn canonical key into canonical_term for display.

    - ALLCAPS/underscore looking tokens → upper-case it.
    - Otherwise: keep lemma form (lowercase).
    """
    if not key:
        return key

    if re.fullmatch(r"[A-Z0-9_]+", key):
        return key.upper()

    return key


# -------------------------------------------------------------------
# Type inference (rules only)
# -------------------------------------------------------------------

def infer_term_type(term_text: str) -> Optional[str]:
    """
    Very coarse type tagging using rules:

      - ^[A-Z0-9_]+$ and contains _  → "config"
      - ends with .conf/.cfg/.ini    → "file"

    Everything else → None.
    """
    if not term_text:
        return None

    t = term_text.strip()
    lower = t.lower()

    # Config-like variables: ALLCAPS + underscore(s)
    if re.fullmatch(r"[A-Z0-9_]+", t) and "_" in t:
        return "config"

    # Config files
    if lower.endswith(".conf") or lower.endswith(".cfg") or lower.endswith(".ini"):
        return "file"

    return None


# -------------------------------------------------------------------
# Main enrichment routine
# -------------------------------------------------------------------

def enrich_terms(db_path: str) -> None:
    """
    Read term_candidates, compute lemma-based canonical_term, synonyms, type,
    and populate term_enrichment.

    Behaviour:
      - Use tf_idf from term_candidates.
      - Only keep terms that are either:
          * tf_idf >= MIN_TF_IDF  (statistically important), OR
          * is_technical_token(term_text)  (HPC configs/env vars/etc.)
      - Rebuild term_enrichment from scratch each run.

    One row per canonical group.

    Aggregates:
      - member_term_ids_json
      - freq_total (sum over members)
      - freq_docs (max over members)
    """
    init_term_enrichment_table(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # Rebuild from scratch
    cur.execute("DELETE FROM term_enrichment;")

    # Pull everything, filter in Python so we can apply the OR rule
    cur.execute(
        """
        SELECT term_id, term_text, term_lemma, freq_total, freq_docs,
               COALESCE(tf_idf, 0.0)
        FROM term_candidates
        """
    )
    rows = cur.fetchall()
    print(f"Loaded {len(rows)} term_candidates for enrichment filtering.")

    groups: Dict[str, Dict[str, Any]] = {}

    kept_count = 0

    for term_id, term_text, term_lemma, freq_total, freq_docs, tf_idf in rows:
        tf_idf = float(tf_idf or 0.0)
        term_text = term_text or ""

        # Filter: keep if statistically important OR technical pattern
        if tf_idf < MIN_TF_IDF and not is_technical_token(term_text):
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
        g["ids"].add(term_id)
        g["texts"].add(term_text)
        g["freq_total"] += int(freq_total or 0)
        g["freq_docs"] = max(g["freq_docs"], int(freq_docs or 0))

    print(f"Kept {kept_count} terms after tf_idf/technical filtering.")
    print(f"Formed {len(groups)} canonical groups.")

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    cur.execute("BEGIN;")
    for canonical_term, data in groups.items():
        member_ids = sorted(data["ids"])
        member_texts = sorted(data["texts"])
        freq_total = data["freq_total"]
        freq_docs = data["freq_docs"]

        # canonical_id = smallest term_id in the group
        canonical_id = member_ids[0]

        # Synonyms: all member texts except the canonical label itself (if present)
        if canonical_term in member_texts:
            synonyms = [t for t in member_texts if t != canonical_term]
        else:
            synonyms = member_texts

        # Abbreviations: ALLCAPS/underscore tokens
        abbreviations = [
            t for t in member_texts
            if re.fullmatch(r"[A-Z0-9_]+", t) and len(t) > 1
        ]

        # Coarse type from canonical term, fallback to any member
        term_type = infer_term_type(canonical_term)
        if term_type is None:
            for t in member_texts:
                term_type = infer_term_type(t)
                if term_type is not None:
                    break

        cur.execute(
            """
            INSERT INTO term_enrichment
                (canonical_id, canonical_term, term_type,
                 synonyms_json, abbreviations_json,
                 member_term_ids_json,
                 freq_total, freq_docs,
                 created_at, updated_at)
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
    print("term_enrichment rebuilt.")


def main():
    print(f"Running term enrichment (tf_idf >= {MIN_TF_IDF} OR technical)...")
    enrich_terms(DB_PATH)
    print("Done term enrichment.")


if __name__ == "__main__":
    main()
