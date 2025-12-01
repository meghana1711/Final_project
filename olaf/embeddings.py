import sqlite3
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from gensim.models import Word2Vec


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

DB_PATH = r"onto_db/ontology_sample_new.db"
CLEANED_VERSION = 1            # match your sentence_lemmatized version

USE_LEMMAS = True              # always use lemmas, not raw tokens
LOWERCASE = True               # normalize to lowercase everywhere

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2                  # ignore words with total freq < 2
WORKERS = 4
TOP_K_NEIGHBORS = 10           # how many neighbors per term to store

OUT_DIR = Path("models")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SKIPGRAM_MODEL_PATH = OUT_DIR / "word2vec_skipgram.model"

SKIPPED_OUT_DIR = Path("output")
SKIPPED_OUT_DIR.mkdir(parents=True, exist_ok=True)
SKIPPED_TERMS_PATH = SKIPPED_OUT_DIR / "skipgram_skipped_terms.tsv"


# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------

def init_skipgram_neighbors_table(db_path: str) -> None:
    """
    Create the skipgram_neighbors table if missing.

    This table links each term_id to its top-K nearest neighbor term_ids
    according to Skip-gram embeddings, with cosine similarity.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS skipgram_neighbors (
            term_id          INTEGER NOT NULL,
            neighbor_term_id INTEGER NOT NULL,
            similarity       REAL    NOT NULL,
            PRIMARY KEY (term_id, neighbor_term_id),
            FOREIGN KEY (term_id)          REFERENCES term_candidates(term_id) ON DELETE CASCADE,
            FOREIGN KEY (neighbor_term_id) REFERENCES term_candidates(term_id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


# -------------------------------------------------------------------
# Load sentences from DB (for training)
# -------------------------------------------------------------------

def load_sentences_from_db(
    db_path: str,
    cleaned_version: int,
    use_lemmas: bool = True,
    lowercase: bool = True,
) -> List[List[str]]:
    """
    Load all sentences as lists of tokens from sentence_lemmatized.

    Uses:
      - lemmas_json if use_lemmas=True
      - tokens_json otherwise

    If lowercase=True, everything is lowercased so we don't distinguish
    between 'SLURM', 'Slurm', 'slurm'.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT tokens_json, lemmas_json
        FROM sentence_lemmatized
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


# -------------------------------------------------------------------
# Train Skip-gram model
# -------------------------------------------------------------------

def train_skipgram(sentences: List[List[str]]) -> Word2Vec:
    """
    Train a Skip-gram Word2Vec model (sg=1) and save it to disk.
    """
    print(f"Training Skip-gram Word2Vec on {len(sentences)} sentences...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1,  # 1 = Skip-gram, 0 = CBOW
    )
    model.save(str(SKIPGRAM_MODEL_PATH))
    print(f"Saved Skip-gram model to {SKIPGRAM_MODEL_PATH}")
    return model


def load_skipgram_model() -> Word2Vec:
    return Word2Vec.load(str(SKIPGRAM_MODEL_PATH))


# -------------------------------------------------------------------
# Load term candidates (lemmas)
# -------------------------------------------------------------------

def get_term_candidates(db_path: str) -> List[Tuple[int, str]]:
    """
    Return list of (term_id, term_lemma) from term_candidates.

    We assume term_lemma is already lowercased, but we force lower() anyway
    to be safe (so 'SLURM', 'Slurm', 'slurm' all become 'slurm').
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT term_id, term_lemma
        FROM term_candidates
        """
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


# -------------------------------------------------------------------
# Build phrase vectors for term_candidates (track skipped)
# -------------------------------------------------------------------

def build_phrase_vectors(
    model: Word2Vec,
    term_candidates: List[Tuple[int, str]],
) -> Tuple[List[int], np.ndarray]:
    """
    For each term_lemma, build a phrase vector as the average of word vectors
    for the tokens that exist in the model's vocabulary.

    Returns:
      term_ids: list of term_ids (only those with vectors)
      matrix: 2D numpy array of shape (n_terms, vector_size)

    Also writes skipped terms (with reasons) to SKIPPED_TERMS_PATH.
    """
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

    # Report and save skipped terms
    print(f"Built vectors for {len(term_ids)} terms, skipped {len(skipped)} terms.")
    with open(SKIPPED_TERMS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write("term_id\tlemma\treason\n")
        for tid, lem, reason in skipped:
            f.write(f"{tid}\t{lem}\t{reason}\n")
    print(f"Wrote skipped terms to {SKIPPED_TERMS_PATH}")

    if not vecs:
        return [], np.empty((0, model.vector_size), dtype=np.float32)

    matrix = np.vstack(vecs)
    return term_ids, matrix


# -------------------------------------------------------------------
# Compute neighbors and write to DB
# -------------------------------------------------------------------

def compute_and_store_neighbors(
    db_path: str,
    term_ids: List[int],
    matrix: np.ndarray,
    top_k: int = TOP_K_NEIGHBORS,
) -> None:
    """
    Given term_ids and their embedding matrix, compute cosine similarity
    between all pairs and store top_k neighbors per term into the DB
    table skipgram_neighbors.
    """
    n_terms = len(term_ids)
    if n_terms == 0:
        print("No term vectors available; cannot compute neighbors.")
        return

    print(f"Computing neighbors for {n_terms} terms (top_k={top_k})...")

    # Normalize each vector to unit length for cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix_norm = matrix / norms

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    init_skipgram_neighbors_table(db_path)

    # Clear existing neighbors (optional; comment out if you want incremental updates)
    cur.execute("DELETE FROM skipgram_neighbors;")

    # For each term, compute similarity to all others via dot product
    for i in range(n_terms):
        vec_i = matrix_norm[i]  # shape (d,)
        sims = matrix_norm @ vec_i  # shape (n_terms,)

        # Exclude self
        sims[i] = -1.0

        if top_k >= n_terms - 1:
            top_indices = np.argsort(-sims)  # all others
        else:
            top_indices = np.argpartition(-sims, top_k)[:top_k]
            top_indices = top_indices[np.argsort(-sims[top_indices])]

        term_id_i = term_ids[i]

        for j in top_indices:
            neighbor_id = term_ids[j]
            similarity = float(sims[j])
            cur.execute(
                """
                INSERT OR REPLACE INTO skipgram_neighbors
                    (term_id, neighbor_term_id, similarity)
                VALUES (?, ?, ?)
                """,
                (term_id_i, neighbor_id, similarity),
            )

        if (i + 1) % 1000 == 0 or i == n_terms - 1:
            print(f"  processed {i + 1}/{n_terms} terms...")
            conn.commit()

    conn.commit()
    conn.close()
    print("Finished writing skipgram_neighbors.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    # 1) Load sentences and train Skip-gram model on lowercased lemmas
    print("Loading sentences from DB...")
    sentences = load_sentences_from_db(
        DB_PATH,
        CLEANED_VERSION,
        use_lemmas=USE_LEMMAS,
        lowercase=LOWERCASE,
    )
    print(f"Loaded {len(sentences)} sentences.")

    if not sentences:
        print("No sentences found; aborting.")
        return

    model = train_skipgram(sentences)

    # 2) Load term candidates (lemmas, lowercased)
    print("Loading term candidates...")
    term_cands = get_term_candidates(DB_PATH)
    print(f"Loaded {len(term_cands)} term candidates.")

    term_ids, matrix = build_phrase_vectors(model, term_cands)
    print(f"Built vectors for {len(term_ids)} terms (see skipped file for the rest).")

    # 3) Compute and store neighbors
    compute_and_store_neighbors(DB_PATH, term_ids, matrix, top_k=TOP_K_NEIGHBORS)


if __name__ == "__main__":
    main()
