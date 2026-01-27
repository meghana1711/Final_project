import argparse
import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, Optional, Any, Set, Tuple, List


# -------------------------------------------------------------------
# Canonicalization helpers
# -------------------------------------------------------------------

LIGHT_PREFIXES: Set[str] = {
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every",
    "either", "neither", "most", "many", "much", "few", "several", "various", "multiple",
    "certain", "particular", "specific", "typical", "general", "common", "other","default"
}

# Small “headword-ish” set used ONLY for generic scoring, not for dropping.
# This is not a “parents list”; it is a weak signal.
GENERIC_HEADWORDS: Set[str] = {
    "system", "information", "data", "datum", "value", "values", "thing", "things",
    "file", "files", "directory", "path", "number", "record", "records",
    "parameter", "parameters", "option", "options", "setting", "settings",
    "configuration", "configurations", "name", "names", "type", "types",
}

SLURM_LSF_ANCHORS: Set[str] = {"slurm", "slurmdbd", "slurmctld", "slurmd", "lsf", "bsub", "bjobs", "bqueues"}

COMMAND_LIKE_RE = re.compile(r"^(sacct|sreport|squeue|sinfo|srun|sbatch|salloc|scancel|scontrol|sstat|sprio|sdiag|smap|sattach)$", re.I)


def is_technical_token(term_text: str) -> bool:
    """
    Heuristic keep rules for important HPC/config tokens even if tf_idf is low.

    Keeps:
      - ALLCAPS + underscores (env/config vars): SLURM_JOB_ID, EGO_AUDIT_MAX_SIZE
      - Config-ish files: slurm.conf, lsf.conf, *.cfg, *.ini
      - Explicit scheduler mentions: contains 'slurm' or 'lsf'
      - Common scheduler command tokens: sbatch/srun/sacct/etc.
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

    # Command-like
    if COMMAND_LIKE_RE.fullmatch(lower):
        return True

    return False


def _canonical_key(term_lemma: str, term_text: str) -> str:
    """
    Canonical key based on lemma (preferred), fallback to text.

    Behaviour:
      - Lemma tokens are assumed lowercased/singularized from extraction.
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
        lemma_tokens = [lemma_raw.lower()] if lemma_raw else [first_text.lower()]
    else:
        lemma_norm = lemma_raw.lower()
        lemma_norm = re.sub(r"[-/.]+", " ", lemma_norm)
        lemma_tokens = lemma_norm.split() if lemma_norm else []

        if not lemma_tokens and text_tokens:
            lemma_norm = " ".join(text_tokens).lower()
            lemma_norm = re.sub(r"[-/.]+", " ", lemma_norm)
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
    Coarse type tagging (rules only):
      - ALLCAPS + underscores -> config
      - endswith .conf/.cfg/.ini -> file
      - known command -> command
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
    if COMMAND_LIKE_RE.fullmatch(lower):
        return "command"
    return None


def headword(phrase: str) -> str:
    """
    Simple headword: last token (works well enough for noun phrases).
    """
    toks = (phrase or "").strip().lower().split()
    return toks[-1] if toks else ""


def has_domain_anchor(text: str) -> bool:
    """
    Domain anchor if:
      - contains slurm/lsf family strings OR
      - looks technical (env var / config file / command)
    """
    if not text:
        return False
    low = text.lower()
    if any(a in low for a in SLURM_LSF_ANCHORS):
        return True
    return is_technical_token(text)


def generic_score_for_group(
    canonical_term: str,
    member_texts: List[str],
    tfidf_sum: float,
    freq_docs: int,
    total_docs_hint: Optional[int] = None,
) -> Tuple[float, bool, List[str]]:
    """
    Compute a generic-ness score using weak signals.
    Returns (score, is_generic, reasons).

    Higher score => more generic.
    """
    reasons: List[str] = []
    score = 0.0

    can = (canonical_term or "").strip().lower()
    hw = headword(can)

    # Signal 1: generic headword
    if hw in GENERIC_HEADWORDS:
        score += 2.0
        reasons.append(f"headword='{hw}' is generic-ish")

    # Signal 2: very short single-token common-ish words
    if len(can.split()) == 1 and len(can) <= 6 and hw in GENERIC_HEADWORDS:
        score += 1.0
        reasons.append("short single-token generic")

    # Signal 3: lacks domain anchor across all member texts
    if not any(has_domain_anchor(t) for t in member_texts + [canonical_term]):
        score += 2.0
        reasons.append("no domain anchor in variants")

    # Signal 4: very broad spread across docs (if high) and low tfidf (approx)
    # We don't have per-group tfidf reliably; we use tfidf_sum as a proxy.
    if freq_docs >= 20 and tfidf_sum < 5.0:
        score += 1.5
        reasons.append("high doc spread with low tfidf proxy")

    # Signal 5: “information/data” canonical that tends to become garbage parent
    if hw in {"information", "data", "datum"}:
        score += 2.5
        reasons.append("very common abstract concept (information/data)")

    # Signal 6: explicit command/config/file types are usually NOT generic
    # Reduce score for obviously technical types.
    if infer_term_type(canonical_term) in {"command", "file", "config"}:
        score -= 1.5
        reasons.append("technical type reduces generic score")

    # Decision threshold (tuneable)
    is_generic = score >= 2.5

    return score, is_generic, reasons


def choose_representative_original(member_texts: List[str], canonical_term: str) -> str:
    """
    Choose a representative original surface form for display/debug:
      - prefer one that matches canonical ignoring case
      - else prefer shortest (usually the “clean” one)
    """
    if not member_texts:
        return canonical_term
    # exact match ignoring case
    for t in member_texts:
        if t.strip().lower() == canonical_term.strip().lower():
            return t.strip()
    # else choose shortest non-empty
    member_texts2 = [t.strip() for t in member_texts if t and t.strip()]
    member_texts2.sort(key=lambda x: (len(x), x.lower()))
    return member_texts2[0] if member_texts2 else canonical_term


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

    # Base table (existing)
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {term_enrichment_table} (
            canonical_id          INTEGER PRIMARY KEY,
            canonical_term        TEXT NOT NULL,
            term_type             TEXT,
            synonyms_json         TEXT,
            abbreviations_json    TEXT,
            member_term_ids_json  TEXT,

            -- NEW: original/surface tracking
            representative_original TEXT,
            original_terms_json      TEXT,

            -- NEW: generic scoring/flags
            generic_score          REAL DEFAULT 0.0,
            is_generic             INTEGER DEFAULT 0,
            generic_reasons_json   TEXT,

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

    # If table already existed, add missing columns safely (SQLite style)
    def add_col(col_sql: str) -> None:
        try:
            cur.execute(f"ALTER TABLE {term_enrichment_table} ADD COLUMN {col_sql};")
        except sqlite3.OperationalError:
            pass  # already exists

    add_col("representative_original TEXT")
    add_col("original_terms_json TEXT")
    add_col("generic_score REAL DEFAULT 0.0")
    add_col("is_generic INTEGER DEFAULT 0")
    add_col("generic_reasons_json TEXT")

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
                "texts": set(),         # original surface forms
                "freq_total": 0,
                "freq_docs": 0,
                "tfidf_sum": 0.0,        # proxy for specificity
            }

        g = groups[canonical_term]
        g["ids"].add(int(term_id))
        if term_text.strip():
            g["texts"].add(term_text.strip())
        g["freq_total"] += int(freq_total or 0)
        g["freq_docs"] = max(g["freq_docs"], int(freq_docs or 0))
        g["tfidf_sum"] += float(tf_idf)

    print(f"Kept {kept_count} terms after filtering (tf_idf >= {min_tf_idf} OR technical).")
    print(f"Formed {len(groups)} canonical groups.")

    now = datetime.utcnow().isoformat(timespec="seconds")

    for canonical_term, data in groups.items():
        member_ids = sorted(data["ids"])
        member_texts = sorted(data["texts"])
        freq_total = data["freq_total"]
        freq_docs = data["freq_docs"]
        tfidf_sum = float(data.get("tfidf_sum", 0.0))

        # canonical_id = smallest term_id in the group (deterministic)
        canonical_id = member_ids[0]

        # synonyms: all member texts except canonical label if present
        # synonyms are "aliases", not necessarily “same-case”
        if any(t.lower() == canonical_term.lower() for t in member_texts):
            synonyms = [t for t in member_texts if t.lower() != canonical_term.lower()]
        else:
            synonyms = member_texts

        # abbreviations: ALLCAPS/underscore tokens only (rule-based)
        abbreviations = [
            t for t in member_texts
            if re.fullmatch(r"[A-Z0-9_]+", t) and len(t) > 1
        ]

        # Choose representative original surface form (debug/display)
        representative_original = choose_representative_original(member_texts, canonical_term)

        # term_type (strongest wins)
        term_type = infer_term_type(canonical_term)
        if term_type is None:
            for t in member_texts:
                term_type = infer_term_type(t)
                if term_type is not None:
                    break

        # Generic scoring
        gscore, is_generic, reasons = generic_score_for_group(
            canonical_term=canonical_term,
            member_texts=member_texts,
            tfidf_sum=tfidf_sum,
            freq_docs=freq_docs,
        )

        cur.execute(
            f"""
            INSERT INTO {term_enrichment_table}
                (canonical_id, canonical_term, term_type,
                 synonyms_json, abbreviations_json, member_term_ids_json,
                 representative_original, original_terms_json,
                 generic_score, is_generic, generic_reasons_json,
                 freq_total, freq_docs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                canonical_id,
                canonical_term,
                term_type,
                json.dumps(synonyms, ensure_ascii=False),
                json.dumps(abbreviations, ensure_ascii=False),
                json.dumps(member_ids, ensure_ascii=False),
                representative_original,
                json.dumps(member_texts, ensure_ascii=False),
                float(gscore),
                int(1 if is_generic else 0),
                json.dumps(reasons, ensure_ascii=False),
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
    ap = argparse.ArgumentParser(description="Rule-based term enrichment (canonicalization + grouping + generic flags).")

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
