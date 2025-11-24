import sqlite3
import math
import re
from datetime import datetime
from typing import Dict, List, Tuple


# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------

def init_term_scores_table(db_path: str) -> None:
    """
    Create term_scores table if missing, and ensure it has a 'score' column.
    Stores TF, IDF, TF-IDF, C-value, and combined score per term_id.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # Base table definition (including score)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS term_scores (
            term_id    INTEGER PRIMARY KEY,
            tf         REAL NOT NULL,
            idf        REAL NOT NULL,
            tf_idf     REAL NOT NULL,
            c_value    REAL NOT NULL,
            score      REAL NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (term_id) REFERENCES term_candidates(term_id)
                ON DELETE CASCADE
        );
        """
    )

    # If table already existed without 'score', add it.
    cur.execute("PRAGMA table_info(term_scores);")
    cols = [row[1] for row in cur.fetchall()]
    if "score" not in cols:
        cur.execute(
            "ALTER TABLE term_scores "
            "ADD COLUMN score REAL NOT NULL DEFAULT 0.0;"
        )

    conn.commit()
    conn.close()


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _is_contiguous_subsequence(
    sub_tokens: List[str],
    full_tokens: List[str],
) -> bool:
    """
    Return True if sub_tokens appear as a contiguous subsequence in full_tokens.
    E.g. ["quality", "of", "service"] in ["end", "to", "end", "quality", "of", "service"].
    """
    len_sub = len(sub_tokens)
    len_full = len(full_tokens)
    if len_sub == 0 or len_sub > len_full:
        return False

    for i in range(len_full - len_sub + 1):
        if full_tokens[i:i + len_sub] == sub_tokens:
            return True
    return False


def _shape_boost(term_text: str) -> float:
    """
    Heuristic boost for single-token technical-looking terms.

    - ALL_CAPS_WITH_UNDERSCORES: +2.0  (EGO_AUDIT_MAX_SIZE)
    - CamelCase / mixedCase:     +2.0  (JobAcctGatherFrequency)
    - ALL_CAPS:                  +1.5  (SQL)
    - TitleCase:                 +0.7  (Admin, Entities)
    - otherwise:                 +0.0  (physics, poll, nature)
    """
    if not term_text:
        return 0.0

    # All caps + underscores/digits (ENV VARS, CONFIG KEYS)
    if re.match(r'^[A-Z0-9_]+$', term_text):
        if "_" in term_text:
            return 2.0
        return 1.5  # ALL_CAPS like "SQL"

    # CamelCase / mixedCase: e.g. JobAcctGather, TresUsageInMaxNode
    if re.search(r'[a-z][A-Z]', term_text):
        return 2.0

    # TitleCase: e.g. Admin, Alter, Entities
    if term_text[0].isupper():
        return 0.7

    return 0.0


# -------------------------------------------------------------------
# Main scoring logic
# -------------------------------------------------------------------

def compute_term_scores(db_path: str) -> Tuple[int, int]:
    """
    Compute TF, IDF, TF-IDF, C-value, and combined score for all terms in
    term_candidates and store them in term_scores.

    Scoring (final 'score'):
      - Multiword (len >= 2):
          score = log(1 + tf) * idf * (1.0 + 0.3 * (len - 1))

      - Single word (len == 1):
          score = log(1 + tf) * idf + shape_boost(term_text)

    C-value is still computed and stored, but NOT used in 'score'.
    """
    init_term_scores_table(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # How many documents do we have? (based on term_occurrences)
    cur.execute("SELECT COUNT(DISTINCT doc_id) FROM term_occurrences;")
    row = cur.fetchone()
    total_docs = row[0] if row and row[0] is not None else 0
    if total_docs == 0:
        print("No documents found in term_occurrences; aborting scoring.")
        conn.close()
        return (0, 0)

    print(f"Computing scores using total_docs={total_docs}...")

    # Load all term candidates, including term_text for shape features
    cur.execute(
        """
        SELECT term_id, term_text, term_lemma, length_tokens, freq_total, freq_docs
        FROM term_candidates
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("No term candidates found; nothing to score.")
        conn.close()
        return (0, 0)

    # Prepare in-memory structures
    terms: List[Dict] = []
    for term_id, text, lemma, length_tokens, freq_total, freq_docs in rows:
        lemma = lemma or ""
        text = text or ""
        tokens = lemma.split()
        terms.append(
            {
                "term_id": int(term_id),
                "text": text,
                "lemma": lemma,
                "tokens": tokens,
                "len": int(length_tokens),
                "freq": int(freq_total),
                "df": int(freq_docs),
            }
        )

    scores: Dict[int, Dict[str, float]] = {}

    # Precompute TF, IDF, TF-IDF (raw)
    for t in terms:
        term_id = t["term_id"]
        tf = float(t["freq"])
        df = t["df"] if t["df"] > 0 else 1

        # Smoothed IDF
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        tf_idf = tf * idf

        scores[term_id] = {
            "tf": tf,
            "idf": idf,
            "tf_idf": tf_idf,
            "c_value": 0.0,  # filled below
            "score": 0.0,    # final combined score
        }

    # Compute C-value for multiword terms (for info only)
    for i, t in enumerate(terms):
        term_id = t["term_id"]
        length_tokens = t["len"]
        freq = t["freq"]
        tokens = t["tokens"]

        # Single tokens: C-value = 0 by design
        if length_tokens <= 1 or freq <= 0:
            scores[term_id]["c_value"] = 0.0
            continue

        superterm_freq_sum = 0
        num_superterms = 0

        # Search for longer terms that contain this one as contiguous subsequence
        for j, u in enumerate(terms):
            if j == i:
                continue
            if u["len"] <= length_tokens:
                continue

            if _is_contiguous_subsequence(tokens, u["tokens"]):
                superterm_freq_sum += u["freq"]
                num_superterms += 1

        if num_superterms == 0:
            c_val = math.log2(length_tokens) * freq
        else:
            c_val = math.log2(length_tokens) * (
                freq - (superterm_freq_sum / float(num_superterms))
            )

        scores[term_id]["c_value"] = float(c_val)

    # Compute final score per term (NO global normalization, NO C-value in score)
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    upsert_count = 0

    for t in terms:
        term_id = t["term_id"]
        vals = scores[term_id]
        tf = vals["tf"]
        idf = vals["idf"]
        length_tokens = t["len"]

        # log-based TF-IDF component
        tfidf_log = math.log(1.0 + tf) * idf

        if length_tokens <= 1:
            # Single-word terms: TF-IDF + shape-based boost
            shape = _shape_boost(t["text"])
            score = tfidf_log + shape
        else:
            # Multiword terms: TF-IDF scaled by length factor
            len_factor = 1.0 + 0.3 * (length_tokens - 1)
            score = tfidf_log * len_factor

        vals["score"] = float(score)

        # Upsert immediately
        cur.execute(
            """
            INSERT INTO term_scores (term_id, tf, idf, tf_idf, c_value, score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term_id) DO UPDATE SET
                tf       = excluded.tf,
                idf      = excluded.idf,
                tf_idf   = excluded.tf_idf,
                c_value  = excluded.c_value,
                score    = excluded.score,
                updated_at = excluded.updated_at
            """,
            (
                term_id,
                vals["tf"],
                vals["idf"],
                vals["tf_idf"],
                vals["c_value"],
                vals["score"],
                now,
            ),
        )
        upsert_count += 1

    conn.commit()
    conn.close()

    print(f"Scored {len(scores)} terms; upserted {upsert_count} rows into term_scores.")
    return (len(scores), upsert_count)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    DB_PATH = r"onto_db/ontology_sample_new.db"  # adjust path as needed

    print("Computing TF-IDF and combined score for term candidates...")
    n_terms, n_upserted = compute_term_scores(DB_PATH)
    print(f"Done: {n_terms} terms scored, {n_upserted} rows upserted into term_scores.")


if __name__ == "__main__":
    main()
