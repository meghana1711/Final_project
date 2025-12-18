import sqlite3
from datetime import datetime
from typing import Dict, Tuple

DB_PATH = r"onto_db/onto_new.db"  # adjust if needed


# ---------------------------------------------------------
# DB helpers
# ---------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_taxonomy_is_a_table(conn: sqlite3.Connection) -> None:
    """
    Table for head-based is_a edges.

    parent_canonical_id can be NULL when the head_text has no corresponding
    canonical term (we still keep parent_head_text and parent_canonical_term).

    NOTE: has 'method', but NO 'score' column.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS taxonomy_is_a (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            child_canonical_id     INTEGER NOT NULL,
            child_canonical_term   TEXT    NOT NULL,
            parent_head_text       TEXT    NOT NULL,
            parent_canonical_id    INTEGER,
            parent_canonical_term  TEXT    NOT NULL,
            method                 TEXT    NOT NULL,
            created_at             TEXT    NOT NULL,
            UNIQUE(child_canonical_id, parent_head_text, method)
        )
        """
    )
    conn.commit()
    print("[INFO] Ensured taxonomy_is_a table exists.")


# ---------------------------------------------------------
# Load mappings
# ---------------------------------------------------------

def load_canonical_terms(conn: sqlite3.Connection):
    rows = conn.execute(
        """
        SELECT canonical_id, canonical_term
        FROM term_enrichment
        WHERE canonical_term IS NOT NULL
          AND TRIM(canonical_term) != ''
        """
    ).fetchall()

    id2term: Dict[int, str] = {}
    label2id: Dict[str, int] = {}

    for r in rows:
        cid = int(r["canonical_id"])
        label = (r["canonical_term"] or "").strip()
        if not label:
            continue
        id2term[cid] = label
        label2id[label.lower()] = cid

    print(f"[INFO] Loaded {len(id2term)} canonical terms.")
    return id2term, label2id


def load_parent_heads(conn: sqlite3.Connection, label2id: Dict[str, int]):
    """
    Build mapping from head_text -> (parent_canonical_id, parent_label).

    - head_text comes from taxonomy_parent_candidates.head_text
    - parent_canonical_id is taken from taxonomy_parent_candidates.head_canonical_id
      if present; otherwise we try to map head_text to a canonical_term
      (exact lowercase match).
    - If still not found, parent_canonical_id stays None and
      parent_canonical_term = head_text.
    """
    rows = conn.execute(
        """
        SELECT head_text, head_canonical_id
        FROM taxonomy_parent_candidates
        """
    ).fetchall()

    head2parent: Dict[str, Tuple[int, str]] = {}

    for r in rows:
        head = (r["head_text"] or "").strip().lower()
        if not head:
            continue

        cid = r["head_canonical_id"]
        parent_id = None
        parent_label = None

        if cid is not None:
            crow = conn.execute(
                "SELECT canonical_term FROM term_enrichment WHERE canonical_id = ?",
                (cid,),
            ).fetchone()
            if crow is not None:
                parent_id = int(cid)
                parent_label = (crow["canonical_term"] or "").strip()

        if parent_label is None:
            cid_guess = label2id.get(head)
            if cid_guess is not None:
                parent_id = cid_guess
                prow = conn.execute(
                    "SELECT canonical_term FROM term_enrichment WHERE canonical_id = ?",
                    (cid_guess,),
                ).fetchone()
                parent_label = (prow["canonical_term"] or "").strip() if prow else head

        if parent_label is None:
            parent_label = head

        head2parent[head] = (parent_id, parent_label)

    print(f"[INFO] Loaded {len(head2parent)} parent heads from taxonomy_parent_candidates.")
    return head2parent


# ---------------------------------------------------------
# Build is_a edges
# ---------------------------------------------------------

def build_head_based_is_a(conn: sqlite3.Connection):
    id2term, label2id = load_canonical_terms(conn)
    head2parent = load_parent_heads(conn, label2id)

    init_taxonomy_is_a_table(conn)

    cur = conn.cursor()
    inserted = 0
    method = "head_parent_candidates"
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for child_id, label in id2term.items():
        tokens = label.split()
        if len(tokens) < 2:
            continue  # only multi-word terms

        head = tokens[-1].lower()
        parent_info = head2parent.get(head)
        if parent_info is None:
            continue

        parent_id, parent_label = parent_info

        # avoid trivial self-loops
        if parent_id == child_id or parent_label.lower() == label.lower():
            continue

        cur.execute(
            """
            INSERT OR REPLACE INTO taxonomy_is_a
            (child_canonical_id,
             child_canonical_term,
             parent_head_text,
             parent_canonical_id,
             parent_canonical_term,
             method,
             created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                child_id,
                label,
                head,
                parent_id,
                parent_label,
                method,
                now,
            ),
        )
        inserted += 1

    conn.commit()
    print(f"[INFO] Inserted/updated {inserted} head-based is_a edges into taxonomy_is_a.")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    conn = get_connection(DB_PATH)
    build_head_based_is_a(conn)
    conn.close()
    print("[INFO] Done building taxonomy_is_a from parent head_text.")


if __name__ == "__main__":
    main()
