import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional

import numpy as np
from gensim.models import Word2Vec
import re
from collections import defaultdict, Counter


# -----------------------------
# Normalization helpers
# -----------------------------

def norm_text(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


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

        # NOTE: simple mean (works, but can be noisy for long phrases)
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
    top_k_max: int,
) -> None:
    n_terms = len(term_ids)
    if n_terms == 0:
        print("No term vectors available; cannot compute neighbors.")
        return

    print(f"Computing neighbors for {n_terms} terms (top_k_max={top_k_max})...")

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

        k = min(top_k_max, n_terms - 1)
        if k <= 0:
            continue

        # fast top-k selection
        if k >= n_terms - 1:
            top_indices = np.argsort(-sims)
        else:
            top_indices = np.argpartition(-sims, k)[:k]
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
# Evaluation: choosing k (5 vs 10)
# -----------------------------

def eval_similarity_curve(
    conn: sqlite3.Connection,
    neighbors_table: str,
    ranks: List[int],
) -> None:
    """
    Prints avg similarity at selected ranks across all terms.
    Uses SQLite window function ROW_NUMBER (SQLite >= 3.25).
    """
    cur = conn.cursor()
    ranks_sorted = sorted(set(ranks))
    rank_list = ",".join(str(r) for r in ranks_sorted)

    q = f"""
    WITH ranked AS (
      SELECT
        term_id,
        similarity,
        ROW_NUMBER() OVER (PARTITION BY term_id ORDER BY similarity DESC) AS rnk
      FROM {neighbors_table}
    )
    SELECT rnk, AVG(similarity) AS avg_sim, MIN(similarity) AS min_sim, MAX(similarity) AS max_sim
    FROM ranked
    WHERE rnk IN ({rank_list})
    GROUP BY rnk
    ORDER BY rnk;
    """
    rows = cur.execute(q).fetchall()
    print("\n[EVAL] Similarity curve (by rank):")
    for rnk, avg_sim, min_sim, max_sim in rows:
        print(f"  rank={int(rnk):>2d}  avg={avg_sim:.3f}  min={min_sim:.3f}  max={max_sim:.3f}")

    # knee proxy: drop from 5 to 10 if those ranks exist
    if 5 in ranks_sorted and 10 in ranks_sorted:
        q2 = f"""
        WITH ranked AS (
          SELECT term_id, similarity,
                 ROW_NUMBER() OVER (PARTITION BY term_id ORDER BY similarity DESC) AS rnk
          FROM {neighbors_table}
        ),
        p AS (
          SELECT
            term_id,
            MAX(CASE WHEN rnk=5 THEN similarity END) AS sim5,
            MAX(CASE WHEN rnk=10 THEN similarity END) AS sim10
          FROM ranked
          GROUP BY term_id
        )
        SELECT AVG(sim5 - sim10) AS avg_drop_5_to_10, AVG(sim10) AS avg_sim10
        FROM p
        WHERE sim5 IS NOT NULL AND sim10 IS NOT NULL;
        """
        row = cur.execute(q2).fetchone()
        if row and row[0] is not None:
            print(f"\n[EVAL] Knee proxy: avg_drop(sim5-sim10)={row[0]:.3f}, avg_sim@10={row[1]:.3f}")


def load_eligible_terms(
    conn: sqlite3.Connection,
    terms_table: str,
    canonical_col: str,
    role_col: str,
    hpc_col: str,
    role_value: str = "class",
) -> Set[str]:
    """
    Eligible = ontology_role == 'class' AND is_hpc_domain_term != 0
    """
    cur = conn.cursor()
    q = f"""
    SELECT {canonical_col} AS t
    FROM {terms_table}
    WHERE LOWER({role_col}) = ?
      AND COALESCE({hpc_col}, 0) != 0
      AND {canonical_col} IS NOT NULL AND TRIM({canonical_col}) != ''
    """
    out = set()
    for (t,) in cur.execute(q, (role_value,)).fetchall():
        out.add(norm_text(t))
    return out


def eval_eligible_neighbor_rate(
    conn: sqlite3.Connection,
    neighbors_table: str,
    eligible_terms: Set[str],
    k_list: List[int],
) -> None:
    """
    For each term, among top-k neighbors, what fraction are eligible terms?
    Uses neighbor_term_text string matching against canonical terms.
    """
    cur = conn.cursor()
    ks = sorted(set(k_list))

    # Pull ranked neighbors once
    q = f"""
    WITH ranked AS (
      SELECT
        term_id,
        term_text,
        neighbor_term_text,
        similarity,
        ROW_NUMBER() OVER (PARTITION BY term_id ORDER BY similarity DESC) AS rnk
      FROM {neighbors_table}
    )
    SELECT term_id, term_text, neighbor_term_text, rnk
    FROM ranked
    WHERE rnk <= {max(ks)};
    """
    rows = cur.execute(q).fetchall()

    # accumulate per term
    per_term_neighbors: Dict[int, List[str]] = defaultdict(list)
    for term_id, term_text, nb_text, rnk in rows:
        per_term_neighbors[int(term_id)].append(norm_text(nb_text))

    print("\n[EVAL] EligibleNeighborRate@k (neighbors passing class+hpc gate):")
    for k in ks:
        rates = []
        for term_id, nbs in per_term_neighbors.items():
            topk = nbs[:k]
            if not topk:
                continue
            eligible = sum(1 for nb in topk if nb in eligible_terms)
            rates.append(eligible / float(len(topk)))
        if rates:
            avg = sum(rates) / len(rates)
            print(f"  k={k:>2d}: avg_rate={avg:.3f} over {len(rates)} terms")
        else:
            print(f"  k={k:>2d}: no data")


def eval_sibling_hit_rate(
    conn: sqlite3.Connection,
    neighbors_table: str,
    taxonomy_seed_table: str,
    term_candidates_table: str,
    k_list: List[int],
) -> None:
    """
    Correct sibling hit rate:
    - taxonomy uses labels (child,parent)
    - skipgram is by (term_id, neighbor_term_id)
    - we align via term_candidates.term_lemma (lowercased)
    """
    cur = conn.cursor()
    ks = sorted(set(k_list))
    maxk = max(ks)

    # 1) Build lemma -> term_id mapping
    lemma2id: Dict[str, int] = {}
    id2lemma: Dict[int, str] = {}
    for tid, lemma in cur.execute(
        f"SELECT term_id, term_lemma FROM {term_candidates_table} WHERE term_lemma IS NOT NULL AND TRIM(term_lemma) != ''"
    ).fetchall():
        l = norm_text(lemma)
        lemma2id[l] = int(tid)
        id2lemma[int(tid)] = l

    # 2) Read taxonomy seed edges in lemma space
    parent2kids: Dict[str, Set[str]] = defaultdict(set)
    child2parent: Dict[str, str] = {}

    seed_rows = cur.execute(f"SELECT child, parent FROM {taxonomy_seed_table}").fetchall()
    mapped_children = 0
    for child, parent in seed_rows:
        c = norm_text(child)
        p = norm_text(parent)
        if not c or not p:
            continue
        if c not in lemma2id:
            continue
        if p not in lemma2id:
            # parent might still be a valid taxonomy label but not in term_candidates;
            # still allow sibling test using child alignment only
            pass
        child2parent[c] = p
        parent2kids[p].add(c)
        mapped_children += 1

    if mapped_children == 0:
        print("\n[EVAL] SiblingHitRate@k: 0 seed children mapped to term_candidates.term_lemma. Check label alignment.")
        return

    # 3) Pull neighbors up to maxk for the mapped children term_ids
    # Use SQL window ranking by similarity for each term_id
    q = f"""
    WITH ranked AS (
      SELECT
        term_id,
        neighbor_term_id,
        similarity,
        ROW_NUMBER() OVER (PARTITION BY term_id ORDER BY similarity DESC) AS rnk
      FROM {neighbors_table}
    )
    SELECT term_id, neighbor_term_id, rnk
    FROM ranked
    WHERE rnk <= ?;
    """
    rows = cur.execute(q, (maxk,)).fetchall()

    termid2neighbors: Dict[int, List[int]] = defaultdict(list)
    for term_id, nb_id, rnk in rows:
        termid2neighbors[int(term_id)].append(int(nb_id))

    print("\n[EVAL] SiblingHitRate@k (aligned via term_lemma):")
    for k in ks:
        rates = []
        used = 0
        for child_lemma, parent_lemma in child2parent.items():
            tid = lemma2id.get(child_lemma)
            if tid is None:
                continue
            nb_ids = termid2neighbors.get(tid, [])
            if not nb_ids:
                continue
            topk = nb_ids[:k]
            # convert neighbor ids to lemmas
            nb_lemmas = [id2lemma.get(nid, "") for nid in topk]
            siblings = parent2kids.get(parent_lemma, set())
            hits = sum(1 for nb in nb_lemmas if nb in siblings and nb != child_lemma)
            rates.append(hits / float(len(topk)))
            used += 1

        if rates:
            avg = sum(rates) / len(rates)
            print(f"  k={k:>2d}: avg_hit_rate={avg:.3f} over {used} seed-children")
        else:
            print(f"  k={k:>2d}: no comparable terms (unexpected)")

def evaluate_k_choices(
    db_path: str,
    neighbors_table: str,
    taxonomy_seed_table: str,
    term_candidates_table: str,   # <-- ADD THIS
    terms_table: str,
    canonical_col: str,
    role_col: str,
    hpc_col: str,
    k_list: List[int],
) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # quick existence checks
    def exists(name: str) -> bool:
        cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
        return cur.fetchone() is not None

    if not exists(neighbors_table):
        conn.close()
        print(f"[EVAL] Missing neighbors table: {neighbors_table}")
        return

    if not exists(taxonomy_seed_table):
        conn.close()
        print(f"[EVAL] Missing taxonomy seed table: {taxonomy_seed_table} (needed for sibling hit rate)")
        return

    if not exists(terms_table):
        conn.close()
        print(f"[EVAL] Missing terms table: {terms_table} (needed for eligible neighbor rate)")
        return

    # Similarity curve
    eval_similarity_curve(conn, neighbors_table, ranks=[1, 3, 5, 10])

    # Eligible neighbor rate
    eligible = load_eligible_terms(conn, terms_table, canonical_col, role_col, hpc_col, role_value="class")
    print(f"\n[EVAL] Eligible terms loaded: {len(eligible)}")
    eval_eligible_neighbor_rate(conn, neighbors_table, eligible, k_list)

    # Sibling hit rate
    eval_sibling_hit_rate(conn, neighbors_table, taxonomy_seed_table, term_candidates_table, k_list)

    conn.close()


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

def parse_k_list(s: str) -> List[int]:
    out = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or [5, 10]


def main():
    ap = argparse.ArgumentParser(description="Train Skip-gram Word2Vec from DB and store nearest neighbors + evaluate k.")

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

    # IMPORTANT: store more once, evaluate multiple ks later
    ap.add_argument("--top_k_max", type=int, default=5, help="Store top-K neighbors per term (max).")
    ap.add_argument("--min_tfidf", type=float, default=10.0)

    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--model_path", default=None, help="Override model path")
    ap.add_argument("--skipped_terms_path", default="output/skipgram_skipped_terms.tsv")

    ap.add_argument("--train", action="store_true", help="Force retrain Word2Vec model")

    # small coverage eval
    ap.add_argument("--eval", action="store_true", help="Run coverage eval")

    # NEW: k validation eval
    ap.add_argument("--eval_k", action="store_true", help="Evaluate whether k=5 vs 10 is good using neighbors + taxonomy + gates.")
    ap.add_argument("--k_list", default="5", help="Comma-separated k values to evaluate, e.g. '3,5,10,15'")

    ap.add_argument("--taxonomy_seed_table", default="taxonomy_is_a_clean",
                    help="Silver truth table with child,parent (for SiblingHitRate@k).")

    ap.add_argument("--terms_table", default="term_enrichment_exten",
                    help="Table with canonical_term, ontology_role, is_hpc_domain_term (for EligibleNeighborRate@k).")
    ap.add_argument("--canonical_col", default="canonical_term")
    ap.add_argument("--role_col", default="ontology_role")
    ap.add_argument("--hpc_col", default="is_hpc_domain_term")

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
        top_k_max=args.top_k_max,
    )

    if args.eval:
        evaluate_embedding_coverage(args.db, args.term_candidates_table, term_ids, args.min_tfidf)

    if args.eval_k:
        ks = parse_k_list(args.k_list)
        evaluate_k_choices(
            db_path=args.db,
            neighbors_table=args.neighbors_table,
            taxonomy_seed_table=args.taxonomy_seed_table,
            term_candidates_table=args.term_candidates_table,  # <-- ADD THIS
            terms_table=args.terms_table,
            canonical_col=args.canonical_col,
            role_col=args.role_col,
            hpc_col=args.hpc_col,
            k_list=ks,
        )


if __name__ == "__main__":
    main()
