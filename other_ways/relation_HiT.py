import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Dict
import torch
# HiT model for hypernym/subsumption scoring
from hierarchy_transformers import HierarchyTransformer  


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

DB_PATH = "onto_db/onto_new.db"  
HIT_MODEL_NAME = "Hierarchy-Transformers/HiT-MiniLM-L12-WordNetNoun"

TOP_K_PARENTS = 5          # max parents per child
CENTRI_SCORE_WEIGHT = 1.0  # weight in subsumption score formula
MIN_SCORE = None           # e.g. set to -2.0 or 0.0 after inspection, or keep None


# -------------------------------------------------------------------
# Data structure
# -------------------------------------------------------------------

@dataclass
class CanonicalTerm:
    canonical_id: int
    label: str


# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_canonical_terms(conn: sqlite3.Connection) -> List[CanonicalTerm]:
    """
    Load canonical terms from term_enrichment table.
    Assumes:
        term_enrichment(canonical_id INTEGER, canonical_term TEXT, ...)
    """
    rows = conn.execute(
        """
        SELECT canonical_id, canonical_term
        FROM term_enrichment
        WHERE canonical_term IS NOT NULL
          AND TRIM(canonical_term) != ''
        """
    ).fetchall()

    terms = [
        CanonicalTerm(
            canonical_id=row["canonical_id"],
            label=row["canonical_term"].strip(),
        )
        for row in rows
    ]
    print(f"[INFO] Loaded {len(terms)} canonical terms from term_enrichment")
    return terms


def init_taxonomy_edges_table(conn: sqlite3.Connection) -> None:
    """
    Create taxonomy_edges table with both IDs and term labels.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS taxonomy_edges (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            child_canonical_id      INTEGER NOT NULL,
            child_canonical_term    TEXT    NOT NULL,
            parent_canonical_id     INTEGER NOT NULL,
            parent_canonical_term   TEXT    NOT NULL,
            score                   REAL    NOT NULL,
            method                  TEXT    NOT NULL,
            UNIQUE(child_canonical_id, parent_canonical_id, method)
        )
        """
    )
    conn.commit()
    print("[INFO] Ensured taxonomy_edges table exists with term columns.")


def insert_taxonomy_edges(
    conn: sqlite3.Connection,
    edges: List[Tuple[int, int, float]],
    id2term: Dict[int, str],
    method: str = "hit_wordnet",
) -> None:
    """
    Insert edges into taxonomy_edges.

    edges: list of (child_canonical_id, parent_canonical_id, score)
    id2term: mapping from canonical_id -> canonical_term
    """
    cur = conn.cursor()
    count = 0
    for child_id, parent_id, score in edges:
        child_term = id2term.get(child_id)
        parent_term = id2term.get(parent_id)
        if not child_term or not parent_term:
            # Skip if somehow we don't know the labels
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO taxonomy_edges
            (child_canonical_id,
             child_canonical_term,
             parent_canonical_id,
             parent_canonical_term,
             score,
             method)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (child_id, child_term, parent_id, parent_term, float(score), method),
        )
        count += 1

    conn.commit()
    print(f"[INFO] Inserted/updated {count} taxonomy edges (method={method}).")


# -------------------------------------------------------------------
# HiT-based subsumption scoring
# -------------------------------------------------------------------

class HitHypernymScorer:
    """
    Wrapper around HiT-MiniLM-L12-WordNetNoun for subsumption prediction.
    """

    def __init__(self, model_name: str = HIT_MODEL_NAME, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[INFO] Loading HiT model '{model_name}' on device={device}...")
        self.model = HierarchyTransformer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("[INFO] HiT model loaded.")

    def encode_terms(self, labels: List[str]) -> torch.Tensor:
        """
        Encode a list of labels into hyperbolic embeddings.
        Returns: Tensor of shape (N, D) on self.device
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                labels,
                convert_to_tensor=True,
            )
        return embeddings.to(self.device)

    def compute_norms(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Hyperbolic norms (distance from origin) for each embedding.
        """
        with torch.no_grad():
            norms = self.model.manifold.dist0(embeddings)
        return norms  # (N,)

    def subsumption_score(
        self,
        child_emb: torch.Tensor,
        child_norm: torch.Tensor,
        parent_emb: torch.Tensor,
        parent_norm: torch.Tensor,
        centri_score_weight: float = CENTRI_SCORE_WEIGHT,
    ) -> float:
        """
        score = - (dist(child, parent) + w * (parent_norm - child_norm))
        Higher score ⇒ stronger "parent subsumes child".
        """
        with torch.no_grad():
            dist = self.model.manifold.dist(child_emb, parent_emb)
            score = -(dist + centri_score_weight * (parent_norm - child_norm))
        return float(score.item())


# -------------------------------------------------------------------
# Taxonomy induction using HiT
# -------------------------------------------------------------------

def induce_taxonomy_edges_with_hit(
    terms: List[CanonicalTerm],
    top_k_parents: int = TOP_K_PARENTS,
    min_score: float | None = MIN_SCORE,
) -> List[Tuple[int, int, float]]:
    """
    For each canonical term (child), find top-k parent candidates using HiT.

    Returns list of edges: (child_canonical_id, parent_canonical_id, score)
    """
    scorer = HitHypernymScorer(HIT_MODEL_NAME)
    labels = [t.label for t in terms]
    embeddings = scorer.encode_terms(labels)   # (N, D)
    norms = scorer.compute_norms(embeddings)   # (N,)

    n_terms = len(terms)
    print(f"[INFO] Computing subsumption scores for {n_terms} terms...")

    id_by_idx = [t.canonical_id for t in terms]
    all_edges: List[Tuple[int, int, float]] = []

    for child_idx in range(n_terms):
        child_id = id_by_idx[child_idx]
        child_emb = embeddings[child_idx : child_idx + 1]
        child_norm = norms[child_idx : child_idx + 1]

        scores_for_child: List[Tuple[int, float]] = []

        for parent_idx in range(n_terms):
            if parent_idx == child_idx:
                continue

            parent_id = id_by_idx[parent_idx]
            parent_emb = embeddings[parent_idx : parent_idx + 1]
            parent_norm = norms[parent_idx : parent_idx + 1]

            # Only consider parents more "central" (smaller norm)
            if parent_norm.item() >= child_norm.item():
                continue

            score = scorer.subsumption_score(
                child_emb, child_norm, parent_emb, parent_norm
            )
            if min_score is not None and score < min_score:
                continue

            scores_for_child.append((parent_idx, score))

        if not scores_for_child:
            continue

        scores_for_child.sort(key=lambda x: x[1], reverse=True)
        for parent_idx, score in scores_for_child[:top_k_parents]:
            parent_id = id_by_idx[parent_idx]
            all_edges.append((child_id, parent_id, score))

        if (child_idx + 1) % 100 == 0 or child_idx == n_terms - 1:
            print(
                f"[DEBUG] Processed {child_idx + 1}/{n_terms} children; "
                f"edges so far: {len(all_edges)}"
            )

    print(f"[INFO] Induced {len(all_edges)} candidate taxonomy edges with HiT.")
    return all_edges


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    conn = get_connection(DB_PATH)

    # 1) Ensure table exists with the full schema
    init_taxonomy_edges_table(conn)

    # 2) Load canonical terms and build mapping id -> label
    canonical_terms = load_canonical_terms(conn)
    if not canonical_terms:
        print("[WARN] No canonical terms found in term_enrichment. Exiting.")
        return

    id2term = {t.canonical_id: t.label for t in canonical_terms}

    # 3) Induce taxonomy edges
    edges = induce_taxonomy_edges_with_hit(
        canonical_terms,
        top_k_parents=TOP_K_PARENTS,
        min_score=MIN_SCORE,
    )

    # 4) Store edges with both IDs and term strings
    insert_taxonomy_edges(conn, edges, id2term, method="hit_wordnet")

    conn.close()
    print("[INFO] Done building taxonomy_edges with term labels.")


if __name__ == "__main__":
    main()
