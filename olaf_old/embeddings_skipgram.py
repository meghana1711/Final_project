import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Set

import math
import statistics

import numpy as np
from gensim.models import Word2Vec


# -----------------------------
# DB helpers
# -----------------------------

def init_skipgram_neighbors_table(
    conn: sqlite3.Connection,
    neighbors_table: str,
    term_candidates_table: str,
) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {neighbors_table} (
            term_id            INTEGER NOT NULL,
            neighbor_term_id   INTEGER NOT NULL,
            similarity         REAL    NOT NULL,
            term_text          TEXT,
            neighbor_term_text TEXT,
            term_tf_idf        REAL,
            neighbor_tf_idf    REAL,
            PRIMARY KEY (term_id, neighbor_term_id),
            FOREIGN KEY (term_id)          REFERENCES {term_candidates_table}(term_id) ON DELETE CASCADE,
            FOREIGN KEY (neighbor_term_id) REFERENCES {term_candidates_table}(term_id) ON DELETE CASCADE
        );
        """
    )

    # upgrade older schema if needed
    cur.execute(f"PRAGMA table_info({neighbors_table});")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "term_text" not in existing_cols:
        cur.execute(f"ALTER TABLE {neighbors_table} ADD COLUMN term_text TEXT;")
    if "neighbor_term_text" not in existing_cols:
        cur.execute(f"ALTER TABLE {neighbors_table} ADD COLUMN neighbor_term_text TEXT;")
    if "term_tf_idf" not in existing_cols:
        cur.execute(f"ALTER TABLE {neighbors_table} ADD COLUMN term_tf_idf REAL;")
    if "neighbor_tf_idf" not in existing_cols:
        cur.execute(f"ALTER TABLE {neighbors_table} ADD COLUMN neighbor_tf_idf REAL;")

    conn.commit()


def load_sentences_from_db(
    db_path: str,
    sentences_table: str,
    cleaned_version: int,
    use_lemmas: bool = True,
    lowercase: bool = True,
) -> List[List[str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        f"""
        SELECT tokens_json, lemmas_json
        FROM {sentences_table}
        WHERE cleaned_version = ?
        ORDER BY doc_id, sent_idx
        """,
        (cleaned_version,),
    )

    sentences: List[List[str]] = []
    for tokens_json, lemmas_json in cur:
        tokens = json.loads(tokens_json)
        lemmas = json.loads(lemmas_json)

        seq = lemmas if use_lemmas else tokens

        sent = []
        for t in seq:
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue
            if lowercase:
                t = t.lower()
            sent.append(t)

        if sent:
            sentences.append(sent)

    conn.close()
    return sentences


def get_term_candidates(
    db_path: str,
    term_candidates_table: str,
    min_tfidf: float,
) -> List[Tuple[int, str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT term_id, term_lemma
        FROM {term_candidates_table}
        WHERE COALESCE(tf_idf, 0.0) >= ?
        """,
        (min_tfidf,),
    )
    rows = cur.fetchall()
    conn.close()

    out: List[Tuple[int, str]] = []
    for term_id, lemma in rows:
        lemma = (lemma or "").strip()
        if lemma:
            lemma = lemma.lower()
        out.append((int(term_id), lemma))
    return out


# -----------------------------
# Model training / loading
# -----------------------------

def train_skipgram(
    sentences: List[List[str]],
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    model_path: Path,
) -> Word2Vec:
    print(f"Training Skip-gram Word2Vec on {len(sentences)} sentences...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=1,
    )
    model.save(str(model_path))
    print(f"Saved model to {model_path}")
    return model


def load_skipgram_model(model_path: Path) -> Word2Vec:
    return Word2Vec.load(str(model_path))


# -----------------------------
# Phrase vectors + neighbors
# -----------------------------

def build_phrase_vectors(
    model: Word2Vec,
    term_candidates: List[Tuple[int, str]],
    skipped_terms_path: Path,
) -> Tuple[List[int], np.ndarray]:
    term_ids: List[int] = []
    vecs: List[np.ndarray] = []
    skipped: List[Tuple[int, str, str]] = []

    for term_id, lemma in term_candidates:
        if not lemma:
            skipped.append((term_id, lemma, "empty lemma"))
            continue

        tokens = lemma.split()
        token_vecs = []

        for tok in tokens:
            tok_norm = tok.lower()
            if tok_norm in model.wv:
                token_vecs.append(model.wv[tok_norm])

        if not token_vecs:
            skipped.append((term_id, lemma, "no tokens in vocab"))
            continue

        phrase_vec = np.mean(token_vecs, axis=0)
        term_ids.append(term_id)
        vecs.append(phrase_vec)

    print(f"Built vectors for {len(term_ids)} terms, skipped {len(skipped)} terms.")
    skipped_terms_path.parent.mkdir(parents=True, exist_ok=True)
    with open(skipped_terms_path, "w", encoding="utf-8", newline="") as f:
        f.write("term_id\tlemma\treason\n")
        for tid, lem, reason in skipped:
            f.write(f"{tid}\t{lem}\t{reason}\n")
    print(f"Wrote skipped terms to {skipped_terms_path}")

    if not vecs:
        return [], np.empty((0, model.vector_size), dtype=np.float32)

    matrix = np.vstack(vecs)
    return term_ids, matrix


def compute_and_store_neighbors(
    db_path: str,
    neighbors_table: str,
    term_candidates_table: str,
    term_ids: List[int],
    matrix: np.ndarray,
    top_k: int,
) -> None:
    n_terms = len(term_ids)
    if n_terms == 0:
        print("No term vectors available; cannot compute neighbors.")
        return

    print(f"Computing neighbors for {n_terms} terms (top_k={top_k})...")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix_norm = matrix / norms

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    init_skipgram_neighbors_table(conn, neighbors_table, term_candidates_table)

    # preload term_text + tf_idf
    term_meta: Dict[int, Tuple[str, float]] = {}
    placeholders = ",".join("?" * len(term_ids))
    cur.execute(
        f"""
        SELECT term_id, term_text, COALESCE(tf_idf, 0.0)
        FROM {term_candidates_table}
        WHERE term_id IN ({placeholders})
        """,
        term_ids,
    )
    for tid, ttext, tfidf in cur.fetchall():
        term_meta[int(tid)] = (ttext or "", float(tfidf or 0.0))

    # clear + rebuild
    cur.execute(f"DELETE FROM {neighbors_table};")

    for i in range(n_terms):
        vec_i = matrix_norm[i]
        sims = matrix_norm @ vec_i
        sims[i] = -1.0

        if top_k >= n_terms - 1:
            top_indices = np.argsort(-sims)
        else:
            top_indices = np.argpartition(-sims, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]

        term_id_i = term_ids[i]
        term_text_i, term_tfidf_i = term_meta.get(term_id_i, ("", 0.0))

        for j in top_indices:
            neighbor_id = term_ids[j]
            similarity = float(sims[j])

            neighbor_text, neighbor_tfidf = term_meta.get(neighbor_id, ("", 0.0))

            cur.execute(
                f"""
                INSERT OR REPLACE INTO {neighbors_table}
                    (term_id, neighbor_term_id, similarity,
                     term_text, neighbor_term_text,
                     term_tf_idf, neighbor_tf_idf)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    term_id_i,
                    neighbor_id,
                    similarity,
                    term_text_i,
                    neighbor_text,
                    term_tfidf_i,
                    neighbor_tfidf,
                ),
            )

        if (i + 1) % 1000 == 0 or i == n_terms - 1:
            print(f"  processed {i + 1}/{n_terms} terms...")
            conn.commit()

    conn.commit()
    conn.close()
    print(f"Finished writing {neighbors_table}.")


# -----------------------------
# Optional eval (kept small)
# -----------------------------

def evaluate_embedding_coverage(
    db_path: str,
    term_candidates_table: str,
    embedded_term_ids: List[int],
    min_tfidf: float,
) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    embedded_ids_set: Set[int] = set(embedded_term_ids)

    cur.execute(
        f"""
        SELECT term_id, COALESCE(tf_idf, 0.0)
        FROM {term_candidates_table}
        WHERE COALESCE(tf_idf, 0.0) >= ?
        """,
        (min_tfidf,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"[EVAL] No terms with tf_idf >= {min_tfidf}.")
        return

    total_terms = len(rows)
    embedded = sum(1 for tid, _ in rows if tid in embedded_ids_set)
    print(f"[EVAL] Coverage for tf_idf >= {min_tfidf}: {embedded}/{total_terms} ({100.0*embedded/total_terms:.1f}%)")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Train Skip-gram Word2Vec from DB and store nearest neighbors.")

    ap.add_argument("--db", required=True)
    ap.add_argument("--sentences_table", default="sentence_lemmatized")
    ap.add_argument("--term_candidates_table", default="term_candidates")
    ap.add_argument("--neighbors_table", default="skipgram_neighbors")
    ap.add_argument("--cleaned_version", type=int, default=1)

    ap.add_argument("--use_lemmas", action="store_true", default=True)
    ap.add_argument("--no_lowercase", action="store_true", help="Disable lowercasing")

    ap.add_argument("--vector_size", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)

    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--min_tfidf", type=float, default=10.0)

    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--model_path", default=None, help="Override model path")
    ap.add_argument("--skipped_terms_path", default="output/skipgram_skipped_terms.tsv")

    ap.add_argument("--train", action="store_true", help="Force retrain Word2Vec model")
    ap.add_argument("--eval", action="store_true", help="Run small coverage eval")

    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path) if args.model_path else (model_dir / "word2vec_skipgram.model")
    skipped_terms_path = Path(args.skipped_terms_path)

    lowercase = not args.no_lowercase

    # 1) Load sentences
    print("[INFO] Loading sentences from DB...")
    sentences = load_sentences_from_db(
        args.db,
        args.sentences_table,
        args.cleaned_version,
        use_lemmas=args.use_lemmas,
        lowercase=lowercase,
    )
    print(f"[INFO] Loaded {len(sentences)} sentences.")
    if not sentences:
        raise SystemExit("[ERROR] No sentences found. Did you run lemmatization?")

    # 2) Train or load model
    if args.train or not model_path.exists():
        model = train_skipgram(
            sentences,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            workers=args.workers,
            model_path=model_path,
        )
    else:
        print(f"[INFO] Loading existing model from {model_path}")
        model = load_skipgram_model(model_path)

    # 3) Load term candidates
    print(f"[INFO] Loading term candidates (tf_idf >= {args.min_tfidf})...")
    term_cands = get_term_candidates(args.db, args.term_candidates_table, args.min_tfidf)
    print(f"[INFO] Loaded {len(term_cands)} term candidates.")

    # 4) Build vectors + store neighbors
    term_ids, matrix = build_phrase_vectors(model, term_cands, skipped_terms_path)
    compute_and_store_neighbors(
        args.db,
        args.neighbors_table,
        args.term_candidates_table,
        term_ids,
        matrix,
        top_k=args.top_k,
    )

    if args.eval:
        evaluate_embedding_coverage(args.db, args.term_candidates_table, term_ids, args.min_tfidf)


if __name__ == "__main__":
    main()
