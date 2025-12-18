import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Optional: WordNet hypernyms (pip install nltk; run nltk.download("wordnet") once)
try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except Exception:
    WORDNET_AVAILABLE = False


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DB_PATH = r"onto_db/onto_new.db"  

# For WordNet edges
MAX_WN_PARENTS_PER_TERM = 3

# Simple stoplist of "too generic" parents to ignore
GENERIC_PARENTS = {
    "thing", "entity", "object", "whole", "unit", "group", "concept",
    "condition", "situation", "time period", "event", "activity"
}


# -------------------------------------------------------------------
# DB HELPER FUNCTIONS
# -------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_taxonomy_edges_table(conn: sqlite3.Connection) -> None:
    """
    Create taxonomy_edges table if missing.

    One row per candidate `child is-a parent` edge, with provenance.
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
            doc_id                  TEXT,
            sent_id                 INTEGER,
            evidence_text           TEXT,
            UNIQUE(child_canonical_id, parent_canonical_id, method)
        )
        """
    )
    conn.commit()
    print("[INFO] Ensured taxonomy_edges table exists.")


# -------------------------------------------------------------------
# CANONICAL TERM MAPPINGS
# -------------------------------------------------------------------

def load_canonical_terms(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Load canonical terms from term_enrichment.
    Expected schema:
        term_enrichment(canonical_id INTEGER PRIMARY KEY,
                        canonical_term TEXT,
                        member_term_ids_json TEXT, ...)
    """
    rows = conn.execute(
        """
        SELECT canonical_id, canonical_term, member_term_ids_json
        FROM term_enrichment
        WHERE canonical_term IS NOT NULL
          AND TRIM(canonical_term) != ''
        """
    ).fetchall()
    print(f"[INFO] Loaded {len(rows)} canonical terms from term_enrichment.")
    return rows


def build_canonical_maps(conn: sqlite3.Connection) -> Tuple[
    Dict[int, str], Dict[str, int], Dict[str, int]
]:
    """
    Build mappings:

      - id2term: canonical_id   -> canonical_term
      - label2id: normalized canonical_term -> canonical_id
      - surface2canonical: normalized term_text (from term_candidates)
                            -> canonical_id (via member_term_ids_json)

    This allows:
      - Working in canonical space (IDs + labels)
      - Canonicalizing spans from sentences
    """
    canonical_rows = load_canonical_terms(conn)

    id2term: Dict[int, str] = {}
    label2id: Dict[str, int] = {}

    # First: canonical_id -> canonical_term
    for row in canonical_rows:
        cid = int(row["canonical_id"])
        label = row["canonical_term"].strip()
        id2term[cid] = label
        label2id[label.lower()] = cid

    # Build mapping from term_id -> canonical_id using member_term_ids_json
    termid2canonical: Dict[int, int] = {}
    for row in canonical_rows:
        cid = int(row["canonical_id"])
        member_json = row["member_term_ids_json"]
        if not member_json:
            continue
        try:
            member_ids = json.loads(member_json)
        except json.JSONDecodeError:
            continue
        for tid in member_ids:
            termid2canonical[int(tid)] = cid

    # surface2canonical: from term_candidates.term_text to canonical_id
    surface2canonical: Dict[str, int] = {}
    tc_rows = conn.execute(
        "SELECT term_id, term_text FROM term_candidates"
    ).fetchall()
    for tr in tc_rows:
        term_id = int(tr["term_id"])
        text = (tr["term_text"] or "").strip()
        if not text:
            continue
        cid = termid2canonical.get(term_id)
        if cid is None:
            continue
        key = normalize_text(text)
        surface2canonical[key] = cid

    print(
        f"[INFO] Built id2term({len(id2term)}), label2id({len(label2id)}), "
        f"surface2canonical({len(surface2canonical)}) maps."
    )
    return id2term, label2id, surface2canonical


def normalize_text(s: str) -> str:
    """
    Basic normalization for matching:
      - lower-case
      - collapse whitespace
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def canonicalize_span(
    span: str,
    label2id: Dict[str, int],
    surface2canonical: Dict[str, int]
) -> Optional[int]:
    """
    Map a raw text span to a canonical_id, if possible.

    Strategy:
      1) exact match on canonical labels
      2) exact match on term_candidates.surface via surface2canonical
    """
    key = normalize_text(span)
    cid = label2id.get(key)
    if cid is not None:
        return cid
    cid = surface2canonical.get(key)
    return cid


# -------------------------------------------------------------------
# 1) HEAD-BASED HYPERNYM INDUCTION
# -------------------------------------------------------------------

def induce_head_edges(
    conn: sqlite3.Connection,
    id2term: Dict[int, str],
    label2id: Dict[str, int],
    method: str = "head"
) -> None:
    """
    For multi-word canonical terms, use the last token as head:
        "job completion plugin" -> head = "plugin"
    If head exists as a canonical term, add:
        child -> head (is-a)
    """
    edges: List[Tuple[int, int, float]] = []

    for cid, label in id2term.items():
        norm_label = label.strip()
        tokens = norm_label.split()
        if len(tokens) < 2:
            continue  # nothing to do for single-word labels

        head = tokens[-1].lower()
        parent_id = label2id.get(head)
        if parent_id is None:
            continue
        if parent_id == cid:
            continue

        edges.append((cid, parent_id, 1.0))  # high confidence

    print(f"[HEAD] Induced {len(edges)} head-based edges.")
    insert_edges(conn, edges, id2term, method=method)


# -------------------------------------------------------------------
# 2) WORDNET-BASED HYPERNYM INDUCTION
# -------------------------------------------------------------------

def wordnet_hypernyms_for_label(label: str) -> List[str]:
    """
    Get a small set of hypernym lemma names for a given label from WordNet.

    Returns a list of strings like "file", "resource", ...
    """
    if not WORDNET_AVAILABLE:
        return []

    label_norm = label.replace(" ", "_").lower()
    synsets = wn.synsets(label_norm, pos=wn.NOUN)
    if not synsets:
        # Try last token as backup
        tokens = label_norm.split("_")
        if len(tokens) > 1:
            synsets = wn.synsets(tokens[-1], pos=wn.NOUN)

    hyper_names: List[str] = []
    for syn in synsets[:3]:  # limit synsets per label
        for h in syn.hypernyms():
            for lemma in h.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name not in hyper_names:
                    hyper_names.append(name)

    return hyper_names


def induce_wordnet_edges(
    conn: sqlite3.Connection,
    id2term: Dict[int, str],
    label2id: Dict[str, int],
    method: str = "wordnet"
) -> None:
    """
    For canonical terms that have WordNet hypernyms that also exist
    as canonical terms, add child -> hypernym edges.
    """
    if not WORDNET_AVAILABLE:
        print("[WORDNET] NLTK WordNet not available; skipping WordNet edges.")
        return

    edges: List[Tuple[int, int, float]] = []

    for cid, label in id2term.items():
        label_norm = label.lower()
        if label_norm in GENERIC_PARENTS:
            continue  # skip generic "entity", "object", etc.

        wn_hyps = wordnet_hypernyms_for_label(label)
        if not wn_hyps:
            continue

        count = 0
        for hname in wn_hyps:
            if hname in GENERIC_PARENTS:
                continue
            parent_id = label2id.get(hname)
            if parent_id is None:
                continue
            if parent_id == cid:
                continue
            edges.append((cid, parent_id, 0.8))  # moderate confidence
            count += 1
            if count >= MAX_WN_PARENTS_PER_TERM:
                break

    print(f"[WORDNET] Induced {len(edges)} WordNet-based edges.")
    insert_edges(conn, edges, id2term, method=method)


# -------------------------------------------------------------------
# 3) SIMPLE PATTERN-BASED "X is a Y" EDGES
# -------------------------------------------------------------------

P_IS_A_KIND = re.compile(
    r'\b(?P<x>[\w\- ]+?)\s+is\s+a[n]?\s+kind of\s+(?P<y>[\w\- ]+?)\b',
    flags=re.IGNORECASE
)

P_IS_A = re.compile(
    r'\b(?P<x>[\w\- ]+?)\s+is\s+a[n]?\s+(?P<y>[\w\- ]+?)\b',
    flags=re.IGNORECASE
)


def extract_is_a_pairs_from_sentence(sent_text: str) -> List[Tuple[str, str, str]]:
    """
    Return a list of (x, y, pattern_id) from "X is a Y" style patterns
    in a sentence.
    """
    pairs: List[Tuple[str, str, str]] = []

    # Pattern "X is a kind of Y"
    for m in P_IS_A_KIND.finditer(sent_text):
        x = m.group("x").strip()
        y = m.group("y").strip()
        if x and y:
            pairs.append((x, y, "is_a_kind_of"))

    # Pattern "X is a Y"
    for m in P_IS_A.finditer(sent_text):
        x = m.group("x").strip()
        y = m.group("y").strip()
        if x and y:
            pairs.append((x, y, "is_a"))

    return pairs


def induce_pattern_edges(
    conn: sqlite3.Connection,
    id2term: Dict[int, str],
    label2id: Dict[str, int],
    surface2canonical: Dict[str, int],
    method: str = "pattern_is_a"
) -> None:
    """
    Scan sentence_segmentation for "X is a Y" patterns and induce edges.

    Expects table:
        sentence_segmentation(doc_id, sent_id, sent_text, ...)
    Adjust column names if yours differ.
    """
    # Try to guess columns; adapt if schema differs.
    try:
        rows = conn.execute(
            """
            SELECT doc_id, sent_id, sent_text
            FROM sentence_segmentation
            """
        ).fetchall()
    except sqlite3.OperationalError:
        print("[PATTERN] sentence_segmentation table not found or schema mismatch; skipping pattern edges.")
        return

    edges_with_meta: List[Tuple[int, int, float, str, int, str]] = []

    for row in rows:
        doc_id = row["doc_id"]
        sent_id = row["sent_id"]
        text = row["sent_text"] or ""
        text_stripped = text.strip()
        if not text_stripped:
            continue

        pairs = extract_is_a_pairs_from_sentence(text_stripped)
        if not pairs:
            continue

        for x_raw, y_raw, pattern_id in pairs:
            child_cid = canonicalize_span(x_raw, label2id, surface2canonical)
            parent_cid = canonicalize_span(y_raw, label2id, surface2canonical)
            if child_cid is None or parent_cid is None:
                continue
            if child_cid == parent_cid:
                continue

            child_label = id2term.get(child_cid)
            parent_label = id2term.get(parent_cid)
            if not child_label or not parent_label:
                continue

            # Avoid generic parents if possible
            if parent_label.lower() in GENERIC_PARENTS:
                continue

            # Score: high confidence for explicit definitional pattern
            score = 1.0
            evidence = text_stripped

            edges_with_meta.append(
                (child_cid, parent_cid, score, doc_id, int(sent_id), evidence)
            )

    print(f"[PATTERN] Induced {len(edges_with_meta)} pattern-based edges.")
    insert_edges_with_evidence(conn, edges_with_meta, id2term, method=method)


# -------------------------------------------------------------------
# INSERT HELPERS
# -------------------------------------------------------------------

def insert_edges(
    conn: sqlite3.Connection,
    edges: List[Tuple[int, int, float]],
    id2term: Dict[int, str],
    method: str
) -> None:
    """
    Insert edges without per-sentence evidence (head/WordNet).
    edges: list of (child_id, parent_id, score)
    """
    cur = conn.cursor()
    count = 0

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for child_id, parent_id, score in edges:
        child_term = id2term.get(child_id)
        parent_term = id2term.get(parent_id)
        if not child_term or not parent_term:
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO taxonomy_edges
            (child_canonical_id,
             child_canonical_term,
             parent_canonical_id,
             parent_canonical_term,
             score,
             method,
             doc_id,
             sent_id,
             evidence_text)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            """,
            (child_id, child_term, parent_id, parent_term, float(score), method),
        )
        count += 1

    conn.commit()
    print(f"[DB] Inserted/updated {count} edges for method='{method}'.")


def insert_edges_with_evidence(
    conn: sqlite3.Connection,
    edges: List[Tuple[int, int, float, str, int, str]],
    id2term: Dict[int, str],
    method: str
) -> None:
    """
    Insert edges that include doc_id, sent_id, evidence_text.
    edges: (child_id, parent_id, score, doc_id, sent_id, evidence_text)
    """
    cur = conn.cursor()
    count = 0

    for child_id, parent_id, score, doc_id, sent_id, evidence in edges:
        child_term = id2term.get(child_id)
        parent_term = id2term.get(parent_id)
        if not child_term or not parent_term:
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO taxonomy_edges
            (child_canonical_id,
             child_canonical_term,
             parent_canonical_id,
             parent_canonical_term,
             score,
             method,
             doc_id,
             sent_id,
             evidence_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                child_id,
                child_term,
                parent_id,
                parent_term,
                float(score),
                method,
                doc_id,
                sent_id,
                evidence,
            ),
        )
        count += 1

    conn.commit()
    print(f"[DB] Inserted/updated {count} edges with evidence for method='{method}'.")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    conn = get_connection(DB_PATH)
    init_taxonomy_edges_table(conn)

    # Build canonical maps once
    id2term, label2id, surface2canonical = build_canonical_maps(conn)

    # 1) Head-based edges
    induce_head_edges(conn, id2term, label2id, method="head")

    # 2) WordNet-based edges
    induce_wordnet_edges(conn, id2term, label2id, method="wordnet")

    # 3) Pattern-based "X is a Y" edges from corpus
    induce_pattern_edges(conn, id2term, label2id, surface2canonical, method="pattern_is_a")

    conn.close()
    print("[INFO] Relation induction (taxonomy) completed.")


if __name__ == "__main__":
    main()
