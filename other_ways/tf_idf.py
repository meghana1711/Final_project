import sqlite3
import math
from datetime import datetime
from typing import Tuple, List
import statistics

DB_PATH = r"onto_db/onto_new.db"  
MAX_TOKENS_FOR_TFIDF = 3           


def init_term_tfidf_table(conn: sqlite3.Connection) -> None:
    """
    Create term_tfidf table.

    IMPORTANT: term_tfidf.term_id is the SAME identifier as term_candidates.term_id.
    We do NOT autogenerate IDs here.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        f"""
        CREATE TABLE term_tfidf (
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


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def compute_term_tfidf(db_path: str) -> Tuple[int, int]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    #  Hard reset the table so schema + IDs are clean
    cur.execute("DROP TABLE IF EXISTS term_tfidf;")
    conn.commit()
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
        """
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
    tfidf_values: List[float] = []
    upsert_count = 0

    for term_id, term_text, term_lemma, length_tokens, freq_total, freq_docs in rows:
        # term_id is taken directly from term_candidates (no new IDs!)
        term_id = int(term_id)
        term_text = term_text or ""
        term_lemma = term_lemma or ""
        length_tokens = int(length_tokens)

        tf = float(freq_total) if freq_total is not None else 0.0
        df = int(freq_docs) if freq_docs is not None else 0
        if df <= 0:
            df = 1  # avoid log(0)

        idf = math.log((total_docs + 1) / (df + 1)) + 1.0
        tf_idf = tf * idf
        tfidf_values.append(tf_idf)

        cur.execute(
            """
            INSERT INTO term_tfidf
                (term_id, term_text, term_lemma, length_tokens,
                 tf, df, idf, tf_idf, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    # ---- stats (unchanged) ----
    if tfidf_values:
        tfidf_values_sorted = sorted(tfidf_values)
        n = len(tfidf_values_sorted)
        tfidf_min = tfidf_values_sorted[0]
        tfidf_max = tfidf_values_sorted[-1]
        tfidf_mean = statistics.mean(tfidf_values_sorted)
        tfidf_median = statistics.median(tfidf_values_sorted)

        p25 = _percentile(tfidf_values_sorted, 25)
        p50 = _percentile(tfidf_values_sorted, 50)
        p75 = _percentile(tfidf_values_sorted, 75)
        p90 = _percentile(tfidf_values_sorted, 90)
        p95 = _percentile(tfidf_values_sorted, 95)
        p99 = _percentile(tfidf_values_sorted, 99)

        print("\n=== TF-IDF statistics (length_tokens <= "
              f"{MAX_TOKENS_FOR_TFIDF}) ===")
        print(f"Total terms: {n}")
        print(f"Min tf_idf: {tfidf_min:.4f}")
        print(f"25th pct : {p25:.4f}")
        print(f"50th pct : {p50:.4f}")
        print(f"75th pct : {p75:.4f}")
        print(f"90th pct : {p90:.4f}")
        print(f"95th pct : {p95:.4f}")
        print(f"99th pct : {p99:.4f}")
        print(f"Max tf_idf: {tfidf_max:.4f}")
        print(f"Mean tf_idf: {tfidf_mean:.4f}")
        print(f"Median tf_idf: {tfidf_median:.4f}")

        bins = [5, 10, 20, 40, 80, 160]
        counts = [0] * (len(bins) + 1)
        for v in tfidf_values_sorted:
            placed = False
            for i, b in enumerate(bins):
                if v < b:
                    counts[i] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1

        print("\nTF-IDF buckets (counts):")
        prev = 0.0
        for i, b in enumerate(bins):
            print(f"  [{prev:>5.1f}, {b:>5.1f}) : {counts[i]}")
            prev = b
        print(f"  [ {bins[-1]:>5.1f}, +inf) : {counts[-1]}")
        print("=========================================\n")

    print(
        f"Computed TF-IDF for {len(rows)} terms with length_tokens <= {MAX_TOKENS_FOR_TFIDF}; "
        f"inserted {upsert_count} rows into term_tfidf."
    )
    return (len(rows), upsert_count)


def main():
    print(f"Computing TF-IDF scores for extracted terms (length_tokens <= {MAX_TOKENS_FOR_TFIDF})...")
    n_terms, n_inserted = compute_term_tfidf(DB_PATH)
    print(f"Done: {n_terms} terms processed, {n_inserted} rows written into term_tfidf.")


if __name__ == "__main__":
    main()
