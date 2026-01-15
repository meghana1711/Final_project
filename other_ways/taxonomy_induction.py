import json
import re
import sqlite3
from typing import Dict, List, Tuple, Optional


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

DB_PATH = r"onto_db/onto_new.db"  # adjust if needed

METHOD_NAME = "pattern_def"
BASE_SCORE = 1.0

# We want these as taxonomy parents even if they don't exist in term_enrichment
SPECIAL_PARENTS = {
    "command": -1001,
    "plugin": -1002,
    "file": -1003,
    "job": -1004,
    "queue": -1005,
    "partition": -1006,
    "node": -1007,
}


# -------------------------------------------------------------------
# DB HELPERS
# -------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_taxonomy_edges_table(conn: sqlite3.Connection) -> None:
    """
    Ensure taxonomy_edges table exists with IDs, terms, score, method, and evidence.

    NOTE: sent_id is TEXT so we can store IDs like 'doc_7ceb_sent_00024'.
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
            sent_id                 TEXT,
            evidence_text           TEXT,
            UNIQUE(child_canonical_id, parent_canonical_id, method)
        )
        """
    )
    conn.commit()
    print("[INFO] Ensured taxonomy_edges table exists.")


# -------------------------------------------------------------------
# CANONICAL TERM MAPPING
# -------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_canonical_maps(conn: sqlite3.Connection):
    """
    Build:
      id2term: canonical_id -> canonical_term
      label2id: normalized canonical_term -> canonical_id
      surface2canonical: normalized term_text -> canonical_id
    """
    rows = conn.execute(
        """
        SELECT canonical_id, canonical_term, member_term_ids_json
        FROM term_enrichment
        WHERE canonical_term IS NOT NULL
          AND TRIM(canonical_term) != ''
        """
    ).fetchall()

    id2term: Dict[int, str] = {}
    label2id: Dict[str, int] = {}
    termid2canonical: Dict[int, int] = {}

    for r in rows:
        cid = int(r["canonical_id"])
        label = (r["canonical_term"] or "").strip()
        if not label:
            continue
        id2term[cid] = label
        label2id[normalize_text(label)] = cid

        mjson = r["member_term_ids_json"]
        if not mjson:
            continue
        try:
            member_ids = json.loads(mjson)
        except json.JSONDecodeError:
            continue
        for tid in member_ids:
            termid2canonical[int(tid)] = cid

    # surface text -> canonical_id via term_candidates
    surface2canonical: Dict[str, int] = {}
    trows = conn.execute(
        "SELECT term_id, term_text FROM term_candidates"
    ).fetchall()
    for tr in trows:
        term_id = int(tr["term_id"])
        text = (tr["term_text"] or "").strip()
        if not text:
            continue
        cid = termid2canonical.get(term_id)
        if cid is None:
            continue
        key = normalize_text(text)
        surface2canonical[key] = cid

    # Also add SPECIAL_PARENTS into id2term so we can print labels
    for lbl, pid in SPECIAL_PARENTS.items():
        id2term[pid] = lbl

    print(
        f"[INFO] id2term={len(id2term)}, "
        f"label2id={len(label2id)}, "
        f"surface2canonical={len(surface2canonical)}"
    )
    return id2term, label2id, surface2canonical


def canonicalize_child(
    span: str,
    label2id: Dict[str, int],
    surface2canonical: Dict[str, int],
) -> Optional[int]:
    key = normalize_text(span)
    cid = label2id.get(key)
    if cid is not None:
        return cid
    return surface2canonical.get(key)


def canonicalize_parent(
    span: str,
    label2id: Dict[str, int],
    surface2canonical: Dict[str, int],
) -> Optional[int]:
    key = normalize_text(span)
    cid = label2id.get(key)
    if cid is not None:
        return cid

    cid = surface2canonical.get(key)
    if cid is not None:
        return cid

    # Fallback: treat generic parents as SPECIAL_PARENTS if not in vocab
    if key in SPECIAL_PARENTS:
        return SPECIAL_PARENTS[key]

    return None


# -------------------------------------------------------------------
# PATTERNS
# -------------------------------------------------------------------

# 1) "X is a command/plugin/file/job/queue/partition/node"
P_IS_A_ROLE = re.compile(
    r"\b(?P<x>[\w\-]+)\s+is\s+a[n]?\s+"
    r"(?P<role>command|plugin|file|job|queue|partition|node)\b",
    flags=re.IGNORECASE,
)

# 2) "X are commands/plugins/files/jobs/queues/partitions/nodes"
P_ARE_ROLE = re.compile(
    r"\b(?P<x>[\w\- ]+?)\s+are\s+"
    r"(?P<role>commands|plugins|files|jobs|queues|partitions|nodes)\b",
    flags=re.IGNORECASE,
)

# 3) "X is a type of Y / are a type of Y"
P_TYPE_OF = re.compile(
    r"\b(?P<x>[\w\- ]+?)\s+(?:is|are)\s+a[n]?\s+type of\s+(?P<y>[\w\- ]+?)\b",
    flags=re.IGNORECASE,
)

# 4) "X is a kind of Y / are a kind of Y"
P_KIND_OF = re.compile(
    r"\b(?P<x>[\w\- ]+?)\s+(?:is|are)\s+a[n]?\s+kind of\s+(?P<y>[\w\- ]+?)\b",
    flags=re.IGNORECASE,
)

# 5) "X is a set of Y / are a set of Y"
P_SET_OF = re.compile(
    r"\b(?P<x>[\w\- ]+?)\s+(?:is|are)\s+a[n]?\s+set of\s+(?P<y>[\w\- ]+?)\b",
    flags=re.IGNORECASE,
)

# 6) "the X command/plugin/file/job/queue/partition/node"
#    e.g. "The sacct command is used to ..."
P_THE_X_ROLE = re.compile(
    r"\b(?:the|The)\s+(?P<x>[\w\-]+)\s+"
    r"(?P<role>command|plugin|file|job|queue|partition|node)\b",
    flags=re.IGNORECASE,
)


def extract_pattern_pairs(sent_text: str) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []

    # 1) X is a ROLE
    for m in P_IS_A_ROLE.finditer(sent_text):
        x = m.group("x").strip()
        role = m.group("role").strip().lower()
        if x:
            pairs.append((x, role, "is_a_role"))

    # 2) X are ROLEs
    for m in P_ARE_ROLE.finditer(sent_text):
        x = m.group("x").strip()
        role = m.group("role").strip().lower()
        singular = {
            "commands": "command",
            "plugins": "plugin",
            "files": "file",
            "jobs": "job",
            "queues": "queue",
            "partitions": "partition",
            "nodes": "node",
        }[role]
        if x:
            pairs.append((x, singular, "are_role"))

    # 3) X is/are a type of Y
    for m in P_TYPE_OF.finditer(sent_text):
        x = m.group("x").strip()
        y = m.group("y").strip()
        if x and y:
            pairs.append((x, y, "type_of"))

    # 4) X is/are a kind of Y
    for m in P_KIND_OF.finditer(sent_text):
        x = m.group("x").strip()
        y = m.group("y").strip()
        if x and y:
            pairs.append((x, y, "kind_of"))

    # 5) X is/are a set of Y
    for m in P_SET_OF.finditer(sent_text):
        x = m.group("x").strip()
        y = m.group("y").strip()
        if x and y:
            pairs.append((x, y, "set_of"))

    # 6) "The sacct command" → child="sacct command", parent="command"
    for m in P_THE_X_ROLE.finditer(sent_text):
        x = m.group("x").strip()
        role = m.group("role").strip().lower()
        if x:
            child = f"{x} {role}"
            pairs.append((child, role, "the_x_role"))

    return pairs


# -------------------------------------------------------------------
# INDUCTION
# -------------------------------------------------------------------

def induce_pattern_edges(
    conn: sqlite3.Connection,
    id2term: Dict[int, str],
    label2id: Dict[str, int],
    surface2canonical: Dict[str, int],
    method: str = METHOD_NAME,
) -> None:
    # 1) Load sentences, but be robust to schema differences
    try:
        sents = conn.execute("SELECT * FROM sentence_segmented").fetchall()
    except sqlite3.OperationalError as e:
        print("[ERROR] Could not read sentence_segmentation:", e)
        return

    print(f"[DEBUG] Loaded {len(sents)} sentences from sentence_segmentation.")

    cur = conn.cursor()
    raw_pair_count = 0
    canon_pair_count = 0
    inserted = 0

    debug_raw_examples = 0
    debug_canon_examples = 0

    for row in sents:
        cols = row.keys()

        # doc_id (if present)
        if "doc_id" in cols:
            doc_id = row["doc_id"]
        elif "document_id" in cols:
            doc_id = row["document_id"]
        else:
            doc_id = None

        # sentence ID: prefer 'sentence_id', fallback to 'sent_id'
        sentence_id = None
        if "sentence_id" in cols:
            sentence_id = row["sentence_id"]
        elif "sent_id" in cols:
            sentence_id = row["sent_id"]

        # text column: try common names
        text = ""
        for cand in ("sent_text", "sentence_text", "sentence", "text"):
            if cand in cols:
                text = row[cand] or ""
                break

        sent = (text or "").strip()
        if not sent:
            continue

        pairs = extract_pattern_pairs(sent)
        if not pairs:
            continue

        raw_pair_count += len(pairs)
        if debug_raw_examples < 10:
            print(f"[RAW MATCH] sentence_id={sentence_id}: {sent}")
            print("           pairs:", pairs)
            debug_raw_examples += 1

        for child_raw, parent_raw, pattern_id in pairs:
            child_cid = canonicalize_child(child_raw, label2id, surface2canonical)
            parent_cid = canonicalize_parent(parent_raw, label2id, surface2canonical)

            if child_cid is None or parent_cid is None:
                continue
            if child_cid == parent_cid:
                continue

            child_label = id2term.get(child_cid)
            parent_label = id2term.get(parent_cid)
            if not child_label or not parent_label:
                continue

            canon_pair_count += 1
            if debug_canon_examples < 10:
                print(
                    f"[CANON MATCH] '{child_raw}'→{child_label}  "
                    f"is-a  '{parent_raw}'→{parent_label}  (pattern={pattern_id})"
                )
                debug_canon_examples += 1

            score = BASE_SCORE
            if pattern_id in {"type_of", "kind_of", "set_of"}:
                score = BASE_SCORE  # or BASE_SCORE * 0.9

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
                    child_cid,
                    child_label,
                    parent_cid,
                    parent_label,
                    float(score),
                    method,
                    doc_id,
                    None if sentence_id is None else str(sentence_id),
                    sent,
                ),
            )
            inserted += 1

    conn.commit()
    print(f"[STATS] raw pairs found:      {raw_pair_count}")
    print(f"[STATS] canonicalized pairs: {canon_pair_count}")
    print(f"[STATS] edges inserted:      {inserted}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    conn = get_connection(DB_PATH)
    init_taxonomy_edges_table(conn)

    id2term, label2id, surface2canonical = load_canonical_maps(conn)

    induce_pattern_edges(conn, id2term, label2id, surface2canonical, method=METHOD_NAME)

    conn.close()
    print("[INFO] Pattern-based taxonomy induction finished.")


if __name__ == "__main__":
    main()
