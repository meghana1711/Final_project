import sqlite3
import re
from typing import Dict, List, Tuple

import nltk
from nltk.corpus import wordnet as wn

DB_PATH = r"onto_db/onto_new.db"

# ---------------------- CONFIG ----------------------

# For which terms do we build relations? (using term_tfidf)
MIN_TF = 1.0          # minimal total frequency
MIN_TFIDF = 0.0       # minimal tf-idf (set higher later if you want pruning)

# For embedding-based "similar" relations (from skipgram_neighbors)
MIN_SIMILARITY = 0.80      # neighbours below this are ignored
TOP_NEIGHBORS_LIMIT = 50   # safety cap per term

# For WordNet-based enrichment
MAX_SYNSETS_PER_HEAD = 3   # limit how many senses per head noun


# ------------------ TABLE INIT ----------------------

def init_term_related_table(conn: sqlite3.Connection) -> None:
    """
    Create term_related table if missing, with term_text and related_text columns.
    """
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS term_related (
            term_id          INTEGER NOT NULL,
            term_text        TEXT   NOT NULL,
            related_term_id  INTEGER,
            related_text     TEXT   NOT NULL,
            relation_type    TEXT   NOT NULL,   -- 'similar', 'synonym', 'hypernym', ...
            source           TEXT   NOT NULL,   -- 'embedding' or 'wordnet'
            similarity       REAL,              -- cosine similarity for embeddings; NULL for WordNet
            notes            TEXT,
            PRIMARY KEY (term_id, related_term_id, relation_type, source, related_text),
            FOREIGN KEY (term_id)         REFERENCES term_candidates(term_id) ON DELETE CASCADE,
            FOREIGN KEY (related_term_id) REFERENCES term_candidates(term_id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


# ------------------ COMMON LOADERS ------------------

def load_all_term_texts(conn: sqlite3.Connection) -> Dict[int, str]:
    """
    term_id -> term_text
    """
    cur = conn.cursor()
    cur.execute("SELECT term_id, term_text FROM term_candidates;")
    return {int(tid): txt for (tid, txt) in cur.fetchall()}


def load_good_terms_from_tfidf(conn: sqlite3.Connection) -> Dict[int, Tuple[str, float, float]]:
    """
    term_id -> (term_text, tf_idf, tf) for 'good' seed terms based on term_tfidf.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.term_id, c.term_text, t.tf_idf, t.tf
        FROM term_candidates c
        JOIN term_tfidf   t USING (term_id)
        WHERE t.tf >= ? AND t.tf_idf >= ?
        """,
        (MIN_TF, MIN_TFIDF),
    )
    rows = cur.fetchall()
    terms = {
        int(term_id): (term_text, float(tf_idf), float(tf))
        for term_id, term_text, tf_idf, tf in rows
    }
    print(f"Loaded {len(terms)} good seed terms from term_tfidf (tf >= {MIN_TF}, tf_idf >= {MIN_TFIDF}).")
    return terms


# ---------------- EMBEDDING RELATIONS ---------------

def build_embedding_relations(conn: sqlite3.Connection) -> None:
    """
    Read skipgram_neighbors and push high-similarity 'similar' relations into term_related.
    Uses term_tfidf to decide which terms to treat as seeds.
    """
    init_term_related_table(conn)
    cur = conn.cursor()

    good_terms = load_good_terms_from_tfidf(conn)
    if not good_terms:
        print("No good terms found (term_tfidf) for embeddings; skipping embedding relations.")
        return

    id2text = load_all_term_texts(conn)

    # Clear previous embedding-based relations
    cur.execute(
        """
        DELETE FROM term_related
        WHERE source = 'embedding';
        """
    )
    conn.commit()

    print("Inserting embedding-based relations into term_related...")
    inserted = 0

    for term_id, (term_text, tf_idf, tf) in good_terms.items():
        # Get neighbours for this term from skipgram_neighbors
        cur.execute(
            """
            SELECT neighbor_term_id, similarity
            FROM skipgram_neighbors
            WHERE term_id = ?
            ORDER BY similarity DESC
            """,
            (term_id,),
        )
        rows = cur.fetchall()
        if not rows:
            continue

        count_for_term = 0
        for neighbor_id, sim in rows:
            neighbor_id = int(neighbor_id)
            sim = float(sim)

            if sim < MIN_SIMILARITY:
                continue

            if neighbor_id not in id2text:
                continue

            neighbor_text = id2text[neighbor_id]

            if neighbor_id == term_id:
                continue

            cur.execute(
                """
                INSERT OR IGNORE INTO term_related
                    (term_id, term_text,
                     related_term_id, related_text,
                     relation_type, source, similarity, notes)
                VALUES (?, ?, ?, ?, 'similar', 'embedding', ?, NULL)
                """,
                (term_id, term_text, neighbor_id, neighbor_text, sim),
            )
            inserted += 1
            count_for_term += 1

            if count_for_term >= TOP_NEIGHBORS_LIMIT:
                break

    conn.commit()
    print(f"Inserted {inserted} embedding-based relations into term_related.")


# ---------------- WORDNET RELATIONS -----------------

def ensure_wordnet_downloaded():
    try:
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def is_env_like(term_text: str) -> bool:
    """
    Heuristic: detect env vars / config keys / non-English tokens we want to skip for WordNet.
    """
    if not term_text:
        return True

    # Contains underscore and mostly uppercase -> likely env var / config key
    if re.match(r'^[A-Z0-9_]+$', term_text):
        return True

    # Contains path / shell characters
    if any(ch in term_text for ch in r"/\:$<>[]{}"):
        return True

    return False


def build_lemma_index(conn: sqlite3.Connection) -> Tuple[Dict[int, str], Dict[str, List[int]]]:
    """
    Returns:
      id2lemma: term_id -> lemma (lowercased)
      lemma2ids: lemma string -> [term_id, ...]
    """
    cur = conn.cursor()
    cur.execute("SELECT term_id, term_lemma FROM term_candidates;")
    id2lemma: Dict[int, str] = {}
    lemma2ids: Dict[str, List[int]] = {}

    for term_id, lemma in cur.fetchall():
        term_id = int(term_id)
        lemma = (lemma or "").strip().lower()
        id2lemma[term_id] = lemma
        if lemma:
            lemma2ids.setdefault(lemma, []).append(term_id)

    return id2lemma, lemma2ids


def build_wordnet_relations(conn: sqlite3.Connection) -> None:
    """
    For each English-like term, use WordNet on its head noun to add:
      - synonym relations
      - hypernym (is-a) relations
    into term_related.
    Uses term_tfidf to pick seed terms.
    """
    ensure_wordnet_downloaded()
    init_term_related_table(conn)
    cur = conn.cursor()

    id2text = load_all_term_texts(conn)
    id2lemma, lemma2ids = build_lemma_index(conn)

    # Which terms do we enrich with WordNet? (use the same TF/TF-IDF filter)
    good_terms = load_good_terms_from_tfidf(conn)
    if not good_terms:
        print("No good terms found for WordNet; skipping WordNet relations.")
        return

    # Clear previous WordNet-based relations
    cur.execute(
        """
        DELETE FROM term_related
        WHERE source = 'wordnet';
        """
    )
    conn.commit()

    print("Inserting WordNet-based relations into term_related...")
    inserted = 0

    for term_id, (term_text, tf_idf, tf) in good_terms.items():
        # Skip env/config-like stuff
        if is_env_like(term_text):
            continue

        lemma = id2lemma.get(term_id, "") or term_text.lower()
        tokens = lemma.split()
        if not tokens:
            continue

        # Simple head noun heuristic: last token of lemma
        head = tokens[-1]

        synsets = wn.synsets(head, pos=wn.NOUN)
        if not synsets:
            continue

        synsets = synsets[:MAX_SYNSETS_PER_HEAD]
        used_pairs = set()  # (relation_type, related_text_lower)

        for syn in synsets:
            # --- Synonyms ---
            for name in syn.lemma_names():
                syn_text = name.replace("_", " ")
                syn_text_lower = syn_text.lower()

                if ("synonym", syn_text_lower) in used_pairs:
                    continue
                used_pairs.add(("synonym", syn_text_lower))

                related_ids = lemma2ids.get(syn_text_lower, [])
                if related_ids:
                    for rid in related_ids:
                        if rid == term_id:
                            continue
                        cur.execute(
                            """
                            INSERT OR IGNORE INTO term_related
                                (term_id, term_text,
                                 related_term_id, related_text,
                                 relation_type, source, similarity, notes)
                            VALUES (?, ?, ?, ?, 'synonym', 'wordnet', NULL, NULL)
                            """,
                            (term_id, term_text, rid, id2text[rid]),
                        )
                        inserted += 1
                else:
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO term_related
                            (term_id, term_text,
                             related_term_id, related_text,
                             relation_type, source, similarity, notes)
                        VALUES (?, ?, NULL, ?, 'synonym', 'wordnet', NULL, NULL)
                        """,
                        (term_id, term_text, syn_text),
                    )
                    inserted += 1

            # --- Hypernyms (parent concepts) ---
            for hyper in syn.hypernyms():
                for name in hyper.lemma_names():
                    hyp_text = name.replace("_", " ")
                    hyp_text_lower = hyp_text.lower()

                    if ("hypernym", hyp_text_lower) in used_pairs:
                        continue
                    used_pairs.add(("hypernym", hyp_text_lower))

                    related_ids = lemma2ids.get(hyp_text_lower, [])
                    if related_ids:
                        for rid in related_ids:
                            if rid == term_id:
                                continue
                            cur.execute(
                                """
                                INSERT OR IGNORE INTO term_related
                                    (term_id, term_text,
                                     related_term_id, related_text,
                                     relation_type, source, similarity, notes)
                                VALUES (?, ?, ?, ?, 'hypernym', 'wordnet', NULL, NULL)
                                """,
                                (term_id, term_text, rid, id2text[rid]),
                            )
                            inserted += 1
                    else:
                        cur.execute(
                            """
                            INSERT OR IGNORE INTO term_related
                                (term_id, term_text,
                                 related_term_id, related_text,
                                 relation_type, source, similarity, notes)
                            VALUES (?, ?, NULL, ?, 'hypernym', 'wordnet', NULL, NULL)
                            """,
                            (term_id, term_text, hyp_text),
                        )
                        inserted += 1

    conn.commit()
    print(f"Inserted {inserted} WordNet-based relations into term_related.")


# -------------------------- MAIN -------------------------

def main():
    conn = sqlite3.connect(DB_PATH)
    try:
        build_embedding_relations(conn)  # from skipgram_neighbors + term_tfidf
        build_wordnet_relations(conn)    # from WordNet + term_tfidf
    finally:
        conn.close()


if __name__ == "__main__":
    main()
