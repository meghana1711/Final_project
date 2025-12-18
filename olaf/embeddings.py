import sqlite3
import json
from pathlib import Path
from typing import List, Tuple, Dict, Set

import math
import statistics

import numpy as np
from gensim.models import Word2Vec


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

DB_PATH = r"onto_db/onto_new.db"   # uses your new DB
CLEANED_VERSION = 1                # match sentence_lemmatized version

USE_LEMMAS = True
LOWERCASE = True

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 3
WORKERS = 4
TOP_K_NEIGHBORS = 10

# Only build embeddings / neighbors for terms with tf_idf >= this
MIN_TF_IDF = 10.0

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
    Create/upgrade the skipgram_neighbors table if missing.

    Now includes:
      - term_text            : text of the source term
      - neighbor_term_text   : text of the neighbor term
      - term_tf_idf          : tf_idf of source term
      - neighbor_tf_idf      : tf_idf of neighbor term
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # Base create (for fresh DBs)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS skipgram_neighbors (
            term_id            INTEGER NOT NULL,
            neighbor_term_id   INTEGER NOT NULL,
            similarity         REAL    NOT NULL,
            term_text          TEXT,
            neighbor_term_text TEXT,
            term_tf_idf        REAL,
            neighbor_tf_idf    REAL,
            PRIMARY KEY (term_id, neighbor_term_id),
            FOREIGN KEY (term_id)          REFERENCES term_candidates(term_id) ON DELETE CASCADE,
            FOREIGN KEY (neighbor_term_id) REFERENCES term_candidates(term_id) ON DELETE CASCADE
        );
        """
    )

    # Upgrade path for older tables that lack the new columns
    cur.execute("PRAGMA table_info(skipgram_neighbors);")
    existing_cols = {row[1] for row in cur.fetchall()}

    if "term_text" not in existing_cols:
        cur.execute("ALTER TABLE skipgram_neighbors ADD COLUMN term_text TEXT;")
    if "neighbor_term_text" not in existing_cols:
        cur.execute("ALTER TABLE skipgram_neighbors ADD COLUMN neighbor_term_text TEXT;")
    if "term_tf_idf" not in existing_cols:
        cur.execute("ALTER TABLE skipgram_neighbors ADD COLUMN term_tf_idf REAL;")
    if "neighbor_tf_idf" not in existing_cols:
        cur.execute("ALTER TABLE skipgram_neighbors ADD COLUMN neighbor_tf_idf REAL;")

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
# Load term candidates (lemmas, filtered by tf_idf)
# -------------------------------------------------------------------

def get_term_candidates(db_path: str) -> List[Tuple[int, str]]:
    """
    Return list of (term_id, term_lemma) from term_candidates,
    restricted to terms with tf_idf >= MIN_TF_IDF.

    We assume term_lemma is already lowercased, but we force lower()
    anyway to be safe.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT term_id, term_lemma
        FROM term_candidates
        WHERE tf_idf >= ?
        """,
        (MIN_TF_IDF,),
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

    Now also stores:
      - term_text, neighbor_term_text
      - term_tf_idf, neighbor_tf_idf
      copied from term_candidates for convenience.
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

    # Preload term_text and tf_idf for all embedded term_ids
    term_meta: Dict[int, Tuple[str, float]] = {}
    if term_ids:
        placeholders = ",".join("?" * len(term_ids))
        cur.execute(
            f"""
            SELECT term_id, term_text, tf_idf
            FROM term_candidates
            WHERE term_id IN ({placeholders})
            """,
            term_ids,
        )
        for tid, ttext, tfidf in cur.fetchall():
            term_meta[int(tid)] = (ttext or "", float(tfidf or 0.0))

    # Clear existing neighbors
    cur.execute("DELETE FROM skipgram_neighbors;")

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
        term_text_i, term_tfidf_i = term_meta.get(term_id_i, ("", 0.0))

        for j in top_indices:
            neighbor_id = term_ids[j]
            similarity = float(sims[j])

            neighbor_text, neighbor_tfidf = term_meta.get(neighbor_id, ("", 0.0))

            cur.execute(
                """
                INSERT OR REPLACE INTO skipgram_neighbors
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
    print("Finished writing skipgram_neighbors.")


# -------------------------------------------------------------------
# Evaluation helpers
# -------------------------------------------------------------------

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


def evaluate_embedding_coverage(
    db_path: str,
    embedded_term_ids: List[int],
    min_tfidf: float = MIN_TF_IDF,
) -> None:
    """
    Evaluate how well embeddings cover high-tf_idf terms.

    Prints:
      - total #terms with tf_idf >= min_tfidf
      - #terms that actually got vectors
      - coverage by tf_idf bands: [min,20), [20,40), [40,80), [80,+inf)
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    embedded_ids_set: Set[int] = set(embedded_term_ids)

    cur.execute(
        """
        SELECT term_id, tf_idf
        FROM term_candidates
        WHERE tf_idf >= ?
        """,
        (min_tfidf,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print(f"[EVAL] No term_candidates with tf_idf >= {min_tfidf}.")
        return

    print(f"\n[EVAL] Embedding coverage for tf_idf >= {min_tfidf}:")

    bands = [
        (min_tfidf, 20.0),
        (20.0, 40.0),
        (40.0, 80.0),
        (80.0, float("inf")),
    ]

    band_total = [0] * len(bands)
    band_embedded = [0] * len(bands)

    for term_id, tf_idf in rows:
        tf_idf = float(tf_idf)
        embedded = term_id in embedded_ids_set

        for idx, (lo, hi) in enumerate(bands):
            if lo <= tf_idf < hi:
                band_total[idx] += 1
                if embedded:
                    band_embedded[idx] += 1
                break

    total_terms = sum(band_total)
    total_embedded = sum(band_embedded)

    print(f"  Total high-tf_idf terms   : {total_terms}")
    print(f"  Terms with embeddings     : {total_embedded}")
    print(f"  Overall coverage          : {100.0 * total_embedded / total_terms:.1f}%")

    labels = [
        f"[{bands[0][0]}, {bands[0][1]})",
        f"[{bands[1][0]}, {bands[1][1]})",
        f"[{bands[2][0]}, {bands[2][1]})",
        f"[{bands[3][0]}, +inf)",
    ]

    print("  Coverage by tf_idf band:")
    for label, tot, emb in zip(labels, band_total, band_embedded):
        if tot == 0:
            cov = 0.0
        else:
            cov = 100.0 * emb / tot
        print(f"    {label:>12}: {emb}/{tot} ({cov:4.1f}%)")


def evaluate_similarity_distribution(db_path: str) -> None:
    """
    Look at global distribution of similarity scores in skipgram_neighbors.
    Prints min/max/percentiles and a few buckets.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT similarity FROM skipgram_neighbors;")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("\n[EVAL] No similarity scores found in skipgram_neighbors.")
        return

    sims = sorted(float(r[0]) for r in rows)
    n = len(sims)

    s_min = sims[0]
    s_max = sims[-1]
    s_mean = statistics.mean(sims)
    s_median = statistics.median(sims)

    p25 = _percentile(sims, 25)
    p50 = _percentile(sims, 50)
    p75 = _percentile(sims, 75)
    p90 = _percentile(sims, 90)
    p95 = _percentile(sims, 95)
    p99 = _percentile(sims, 99)

    print("\n[EVAL] Similarity distribution (skipgram_neighbors):")
    print(f"  Count   : {n}")
    print(f"  Min     : {s_min:.4f}")
    print(f"  25th pct: {p25:.4f}")
    print(f"  50th pct: {p50:.4f}")
    print(f"  75th pct: {p75:.4f}")
    print(f"  90th pct: {p90:.4f}")
    print(f"  95th pct: {p95:.4f}")
    print(f"  99th pct: {p99:.4f}")
    print(f"  Max     : {s_max:.4f}")
    print(f"  Mean    : {s_mean:.4f}")
    print(f"  Median  : {s_median:.4f}")

    buckets = [0.5, 0.7, 0.85, 0.95, 0.99]
    counts = [0] * (len(buckets) + 1)

    for v in sims:
        placed = False
        for i, b in enumerate(buckets):
            if v < b:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    labels = [
        "[0.0, 0.5)",
        "[0.5, 0.7)",
        "[0.7, 0.85)",
        "[0.85, 0.95)",
        "[0.95, 0.99)",
        "[0.99, 1.0]",
    ]

    print("\n  Buckets:")
    for label, c in zip(labels, counts):
        print(f"    {label:>11}: {c}")
    print()


def inspect_sample_neighbors(
    db_path: str,
    sample_size: int = 5,
    min_tfidf_sample: float = 40.0,
    neighbors_to_show: int = 5,
) -> None:
    """
    For a few top-tf_idf terms, print their nearest neighbors with
    term_text, tf_idf, and similarity.

    Uses skipgram_neighbors (which now already holds text + tf_idf),
    and computes the AVERAGE tf_idf of the shown neighbors.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Seed terms: high tf_idf, which have neighbors in skipgram_neighbors
    cur.execute(
        """
        SELECT DISTINCT term_id, term_text, term_tf_idf
        FROM skipgram_neighbors
        WHERE term_tf_idf >= ?
        ORDER BY term_tf_idf DESC
        LIMIT ?
        """,
        (min_tfidf_sample, sample_size),
    )
    seeds = cur.fetchall()

    if not seeds:
        print(f"\n[EVAL] No seed terms found with term_tf_idf >= {min_tfidf_sample} in skipgram_neighbors.")
        conn.close()
        return

    print(f"\n[EVAL] Sample neighbor inspection (seed tf_idf >= {min_tfidf_sample}):")
    for term_id, term_text, tfidf in seeds:
        print(f"\n  Term {term_id} | '{term_text}' | tf_idf={tfidf:.2f}")

        cur.execute(
            """
            SELECT neighbor_term_id, similarity,
                   neighbor_term_text, neighbor_tf_idf
            FROM skipgram_neighbors
            WHERE term_id = ?
            ORDER BY similarity DESC
            LIMIT ?
            """,
            (term_id, neighbors_to_show),
        )
        neighbors = cur.fetchall()

        if not neighbors:
            print("    (no neighbors)")
            continue

        neighbor_tfidfs = [float(n_tfidf) for _, _, _, n_tfidf in neighbors]
        avg_neighbor_tfidf = sum(neighbor_tfidfs) / len(neighbor_tfidfs)

        print(f"    Avg neighbor tf_idf (top {len(neighbors)}): {avg_neighbor_tfidf:.2f}")
        for n_id, sim, n_text, n_tfidf in neighbors:
            print(f"    -> {n_id:5d} | sim={sim: .4f} | tf_idf={n_tfidf:7.2f} | {n_text}")

    conn.close()


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

    # 2) Load term candidates (lemmas, filtered by tf_idf)
    print(f"Loading term candidates (tf_idf >= {MIN_TF_IDF})...")
    term_cands = get_term_candidates(DB_PATH)
    print(f"Loaded {len(term_cands)} term candidates.")

    term_ids, matrix = build_phrase_vectors(model, term_cands)
    print(f"Built vectors for {len(term_ids)} terms (see skipped file for the rest).")

    # 3) Compute and store neighbors (with term text + tf_idf)
    compute_and_store_neighbors(DB_PATH, term_ids, matrix, top_k=TOP_K_NEIGHBORS)

    # 4) Evaluation: coverage, similarity distribution, and sample neighbors
    evaluate_embedding_coverage(DB_PATH, term_ids, min_tfidf=MIN_TF_IDF)
    evaluate_similarity_distribution(DB_PATH)
    inspect_sample_neighbors(
        DB_PATH,
        sample_size=2,          # how many seed terms to inspect
        min_tfidf_sample=40.0,  # seeds: high-tf_idf core terms
        neighbors_to_show=5,    # how many neighbors per seed
    )


if __name__ == "__main__":
    main()
