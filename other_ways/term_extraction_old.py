import sqlite3
import json
from datetime import datetime
from typing import List, Tuple, Dict

# Hard stoplist – adjust as needed
STOP_TERMS = {"section", "figure", "table", "chapter", "example"}


def init_term_tables(db_path: str) -> None:
    """Create term_candidates and term_occurrences tables if missing."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS term_candidates (
            term_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            term_text         TEXT NOT NULL,
            term_lemma        TEXT NOT NULL,
            length_tokens     INTEGER NOT NULL,
            freq_total        INTEGER NOT NULL DEFAULT 0,
            freq_docs         INTEGER NOT NULL DEFAULT 0,
            created_at        TEXT,
            updated_at        TEXT,
            UNIQUE (term_lemma, length_tokens)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS term_occurrences (
            term_id         INTEGER NOT NULL,
            doc_id          TEXT NOT NULL,
            sent_idx        INTEGER NOT NULL,
            token_start     INTEGER NOT NULL,
            token_end       INTEGER NOT NULL,
            cleaned_version INTEGER NOT NULL,
            PRIMARY KEY (term_id, doc_id, sent_idx, token_start, token_end),
            FOREIGN KEY (term_id) REFERENCES term_candidates(term_id)
                ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


def _extract_candidates_from_sentence(
    tokens: List[str],
    lemmas: List[str],
    pos: List[str],
) -> List[Tuple[int, int, str, str, int]]:
    """
    Return list of spans as:
        (start, end, term_text, term_lemma, length_tokens)

    Pattern:
        (ADJ|NOUN|PROPN)+ ending in (NOUN|PROPN)
    """
    candidates = []
    allowed_inside = {"ADJ", "NOUN", "PROPN"}
    allowed_end = {"NOUN", "PROPN"}
    n = len(tokens)
    i = 0

    while i < n:
        if pos[i] in allowed_inside:
            start = i
            j = i + 1
            while j < n and pos[j] in allowed_inside:
                j += 1
            end = j - 1

            if pos[end] in allowed_end:
                term_tokens = tokens[start:end + 1]
                term_lemmas = lemmas[start:end + 1]
                term_text = " ".join(term_tokens)
                term_lemma = " ".join(term_lemmas).lower()
                length_tokens = end - start + 1

                if length_tokens == 1 and len(term_lemma) < 3:
                    pass
                elif term_lemma.isdigit():
                    pass
                elif term_lemma in STOP_TERMS:
                    pass
                else:
                    candidates.append((start, end, term_text, term_lemma, length_tokens))

            i = j
        else:
            i += 1

    return candidates


def extract_term_candidates(db_path: str, cleaned_version: int) -> Tuple[int, int]:
    """
    Read from sentence_lemmatized and populate term_candidates + term_occurrences.

    Returns:
        (number_of_unique_terms_touched, number_of_occurrences_inserted)
    """
    init_term_tables(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute("""
        SELECT doc_id, sent_idx, tokens_json, lemmas_json, pos_tags_json
        FROM sentence_lemmatized
        WHERE cleaned_version = ?
        ORDER BY doc_id, sent_idx
    """, (cleaned_version,))
    rows = cur.fetchall()

    if not rows:
        print(f"No lemmatized sentences found for cleaned_version={cleaned_version}.")
        conn.close()
        return (0, 0)

    print(f"Building term candidates from {len(rows)} sentences (cleaned_version={cleaned_version})...")

    term_stats: Dict[Tuple[str, int], Dict] = {}
    occurrences = []

    for doc_id, sent_idx, tokens_json, lemmas_json, pos_json in rows:
        tokens = json.loads(tokens_json)
        lemmas = json.loads(lemmas_json)
        pos = json.loads(pos_json) if pos_json is not None else ["X"] * len(tokens)

        cands = _extract_candidates_from_sentence(tokens, lemmas, pos)
        for start, end, term_text, term_lemma, length_tokens in cands:
            key = (term_lemma, length_tokens)
            if key not in term_stats:
                term_stats[key] = {
                    "term_text": term_text,
                    "freq_total": 0,
                    "docs": set()
                }
            term_stats[key]["freq_total"] += 1
            term_stats[key]["docs"].add(doc_id)

            occurrences.append((term_lemma, length_tokens, doc_id, sent_idx, start, end))

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    term_key_to_id: Dict[Tuple[str, int], int] = {}

    # upsert candidates
    for (term_lemma, length_tokens), info in term_stats.items():
        term_text = info["term_text"]
        freq_total = info["freq_total"]
        freq_docs = len(info["docs"])

        cur.execute("""
            INSERT INTO term_candidates
                (term_text, term_lemma, length_tokens, freq_total, freq_docs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term_lemma, length_tokens) DO UPDATE SET
                freq_total = term_candidates.freq_total + excluded.freq_total,
                freq_docs  = term_candidates.freq_docs  + excluded.freq_docs,
                updated_at = excluded.updated_at
        """, (term_text, term_lemma, length_tokens, freq_total, freq_docs, now, now))

        cur.execute("""
            SELECT term_id
            FROM term_candidates
            WHERE term_lemma = ? AND length_tokens = ?
        """, (term_lemma, length_tokens))
        term_id = cur.fetchone()[0]
        term_key_to_id[(term_lemma, length_tokens)] = term_id

    # insert occurrences
    occ_count = 0
    for term_lemma, length_tokens, doc_id, sent_idx, start, end in occurrences:
        term_id = term_key_to_id[(term_lemma, length_tokens)]
        cur.execute("""
            INSERT OR IGNORE INTO term_occurrences
                (term_id, doc_id, sent_idx, token_start, token_end, cleaned_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (term_id, doc_id, sent_idx, start, end, cleaned_version))
        occ_count += cur.rowcount

    conn.commit()
    conn.close()

    print(f"Inserted/updated {len(term_stats)} term_candidates and {occ_count} term_occurrences.")
    return (len(term_stats), occ_count)


def main():
    DB_PATH = r"onto_db/ontology_sample.db"
    CLEANED_VERSION = 1

    print("Running term extraction...")
    n_terms, n_occ = extract_term_candidates(DB_PATH, CLEANED_VERSION)
    print(f"Done: {n_terms} unique terms, {n_occ} occurrences.")


if __name__ == "__main__":
    main()

