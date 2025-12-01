import sqlite3
import math
from datetime import datetime
from typing import Tuple

DB_PATH = r"onto_db/ontology_sample_new.db"

MAX_TOKENS_FOR_TFIDF = 4  # only keep terms with length_tokens <= 4


def init_term_tfidf_table(conn: sqlite3.Connection) -> None:
    """
    Create term_tfidf table if missing.

    Only terms with length_tokens <= MAX_TOKENS_FOR_TFIDF are inserted.
    We also add a CHECK constraint to enforce this at the table level.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS term_tfidf (
            term_id       INTEGER PRIMARY KEY,
            term_text     TEXT    NOT NULL,
            term_lemma    TEXT    NOT NULL,
            length_tokens INTEGER NOT NULL CHECK (length_tokens <= {MAX_TOKENS_FOR_TFIDF}),
            tf            REAL    NOT NULL,
            df            INTEGER NOT NULL,
            idf           REAL    NOT NULL,
            tf_idf        REAL    NOT NULL,
            updated_at    TEXT    NOT NULL,
            FOREIGN KEY (term_id) REFERENCES term_candidates(term_id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()


def compute_term_tfidf(db_path: str) -> Tuple[int, int]:
    """
    Compute TF-IDF scores for terms in term_candidates and store them in term_tfidf.

    Only terms with length_tokens <= MAX_TOKENS_FOR_TFIDF are considered.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    init_term_tfidf_table(conn)

    # Count documents
    cur.execute("SELECT COUNT(DISTINCT doc_id) FROM term_occurrences;")
    row = cur.fetchone()
    total_docs = row[0] if row and row[0] is not None else 0
    if total_docs == 0:
        print("No documents found in term_occurrences; aborting TF-IDF computation.")
        conn.close()
        return (0, 0)

    print(f"Computing TF/IDF/TF-IDF using total_docs={total_docs}...")

    # Only terms with length_tokens <= MAX_TOKENS_FOR_TFIDF
    cur.execute(
        f"""
        SELECT term_id, term_text, term_lemma, length_tokens, freq_total, freq_docs
        FROM term_candidates
        WHERE length_tokens <= ?
        """,
        (MAX_TOKENS_FOR_TFIDF,),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"No term candidates found with length_tokens <= {MAX_TOKENS_FOR_TFIDF}; nothing to score.")
        conn.close()
        return (0, 0)

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    upsert_count = 0

    for term_id, term_text, term_lemma, length_tokens, freq_total, freq_docs in rows:
        term_id = int(term_id)
        term_text = term_text or ""
        term_lemma = term_lemma or ""
        length_tokens = int(length_tokens)

        tf = float(freq_total) if freq_total is not None else 0.0
        df = int(freq_docs) if freq_docs is not None else 0

        if df <= 0:
            df = 1  # avoid log(0)

        # Smoothed IDF
        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        tf_idf = tf * idf

        cur.execute(
            """
            INSERT INTO term_tfidf
                (term_id, term_text, term_lemma, length_tokens,
                 tf, df, idf, tf_idf, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term_id) DO UPDATE SET
                term_text     = excluded.term_text,
                term_lemma    = excluded.term_lemma,
                length_tokens = excluded.length_tokens,
                tf            = excluded.tf,
                df            = excluded.df,
                idf           = excluded.idf,
                tf_idf        = excluded.tf_idf,
                updated_at    = excluded.updated_at
            """,
            (
                term_id,
                term_text,
                term_lemma,
                length_tokens,
                tf,
                df,
                idf,
                tf_idf,
                now,
            ),
        )
        upsert_count += 1

    conn.commit()
    conn.close()

    print(
        f"Computed TF-IDF for {len(rows)} terms with length_tokens <= {MAX_TOKENS_FOR_TFIDF}; "
        f"upserted {upsert_count} rows into term_tfidf."
    )
    return (len(rows), upsert_count)


def main():
    print("Computing TF-IDF scores for extracted terms (length_tokens <= 4)...")
    n_terms, n_upserted = compute_term_tfidf(DB_PATH)
    print(f"Done: {n_terms} terms processed, {n_upserted} rows upserted into term_tfidf.")


if __name__ == "__main__":
    main()
