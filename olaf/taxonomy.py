import argparse
import re
import sqlite3
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# -----------------------------
# Tokenization helpers
# -----------------------------
_CAMEL_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])")


def normalize_spaces(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(text: str) -> List[str]:
    """
    Tokenize a canonical label more robustly than split():
      - '_' and '/' -> spaces
      - split CamelCase
      - normalize spaces
      - lowercase tokens
    """
    if not text:
        return []
    t = _CAMEL_SPLIT_RE.sub(" ", text)
    t = normalize_spaces(t)
    return [tok.lower() for tok in t.split() if tok.strip()]


# -----------------------------
# Type rules (typed parents)
# -----------------------------
JOB_STATES = {
    "pending", "running", "completed", "failed", "cancelled", "configuring",
    "completing", "suspended", "timeout", "node_fail", "preempted"
}

def infer_parent_type(term: str) -> Optional[str]:
    """
    Returns a type-parent label like:
      option_flag | config_param | config_file | log_or_state_path | job_state | command | resource | other_hpc
    or None if no rule matches.
    """
    if not term:
        return None
    t = term.strip()
    lower = t.lower()

    # Option flags
    if lower.startswith("--") or re.fullmatch(r"-[A-Za-z]\b", t):
        return "option_flag"

    # Config params / env vars (ALLCAPS with underscores)
    if re.fullmatch(r"[A-Z0-9_]+", t) and "_" in t:
        return "config_param"

    # Config files
    if lower.endswith((".conf", ".cfg", ".ini")):
        return "config_file"

    # Paths (rough)
    if lower.startswith("/") or "\\" in t:
        return "log_or_state_path"

    # Job states (rough list)
    if lower in JOB_STATES:
        return "job_state"

    # Commands (common HPC scheduler command shape)
    # sbatch/srun/squeue/sacct/bsub/bjobs/lsload etc.
    if re.fullmatch(r"[a-z][a-z0-9_-]{1,20}", lower) and lower in {
        "sbatch", "srun", "salloc", "squeue", "sacct", "scancel", "sinfo", "scontrol",
        "bsub", "bjobs", "bqueues", "bhosts", "lsload", "lsid", "bhist"
    }:
        return "command"

    # Resources (simple heuristic)
    if lower in {"cpu", "cpus", "core", "cores", "gpu", "gpus", "memory", "mem", "node", "nodes"}:
        return "resource"

    return None


# -----------------------------
# DB helpers
# -----------------------------
def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_taxonomy_is_a_table(conn: sqlite3.Connection, out_table: str) -> None:
    """
    Upgraded taxonomy table.

    Adds:
      - score   : numeric confidence signal (e.g. head frequency)
      - support : count of supporting signals (head match + optional type rule)
      - rule    : which rule created it (head_last_token / typed_rule)
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            child_canonical_id     INTEGER NOT NULL,
            child_canonical_term   TEXT    NOT NULL,
            parent_head_text       TEXT    NOT NULL,
            parent_canonical_id    INTEGER,
            parent_canonical_term  TEXT    NOT NULL,
            method                 TEXT    NOT NULL,
            rule                   TEXT    NOT NULL,
            score                  REAL    NOT NULL DEFAULT 0.0,
            support                INTEGER NOT NULL DEFAULT 1,
            created_at             TEXT    NOT NULL,
            UNIQUE(child_canonical_id, parent_head_text, method, rule)
        )
        """
    )

    # lightweight upgrade if table existed without new columns
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({out_table});").fetchall()}
    if "rule" not in cols:
        conn.execute(f"ALTER TABLE {out_table} ADD COLUMN rule TEXT NOT NULL DEFAULT 'head_last_token';")
    if "score" not in cols:
        conn.execute(f"ALTER TABLE {out_table} ADD COLUMN score REAL NOT NULL DEFAULT 0.0;")
    if "support" not in cols:
        conn.execute(f"ALTER TABLE {out_table} ADD COLUMN support INTEGER NOT NULL DEFAULT 1;")

    conn.commit()
    print(f"[INFO] Ensured {out_table} exists (with rule/score/support).")


def load_canonical_terms(conn: sqlite3.Connection, enrichment_table: str):
    rows = conn.execute(
        f"""
        SELECT canonical_id, canonical_term
        FROM {enrichment_table}
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

    print(f"[INFO] Loaded {len(id2term)} canonical terms from {enrichment_table}.")
    return id2term, label2id


def load_parent_heads(
    conn: sqlite3.Connection,
    parent_candidates_table: str,
    enrichment_table: str,
    label2id: Dict[str, int],
) -> Dict[str, Tuple[Optional[int], str, int]]:
    """
    Returns head2parent:
      head_text -> (parent_canonical_id or None, parent_label, frequency)

    Also tries:
      - parent_candidates.head_canonical_id
      - fallback exact match head_text to canonical_term
      - else parent_canonical_id None, label=head_text
    """
    rows = conn.execute(
        f"""
        SELECT head_text, head_canonical_id, frequency
        FROM {parent_candidates_table}
        """
    ).fetchall()

    head2parent: Dict[str, Tuple[Optional[int], str, int]] = {}

    for r in rows:
        head = (r["head_text"] or "").strip().lower()
        if not head:
            continue

        freq = int(r["frequency"] or 0)
        cid = r["head_canonical_id"]

        parent_id: Optional[int] = None
        parent_label: Optional[str] = None

        if cid is not None:
            crow = conn.execute(
                f"SELECT canonical_term FROM {enrichment_table} WHERE canonical_id = ?",
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
                    f"SELECT canonical_term FROM {enrichment_table} WHERE canonical_id = ?",
                    (cid_guess,),
                ).fetchone()
                parent_label = (prow["canonical_term"] or "").strip() if prow else head

        if parent_label is None:
            parent_label = head

        head2parent[head] = (parent_id, parent_label, freq)

    print(f"[INFO] Loaded {len(head2parent)} heads from {parent_candidates_table}.")
    return head2parent


# -----------------------------
# Core taxonomy building
# -----------------------------
def insert_edge(
    cur: sqlite3.Cursor,
    out_table: str,
    child_id: int,
    child_label: str,
    parent_head_text: str,
    parent_id: Optional[int],
    parent_label: str,
    method: str,
    rule: str,
    score: float,
    support: int,
    now: str,
) -> None:
    cur.execute(
        f"""
        INSERT OR REPLACE INTO {out_table}
        (child_canonical_id, child_canonical_term,
         parent_head_text, parent_canonical_id, parent_canonical_term,
         method, rule, score, support, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            child_id,
            child_label,
            parent_head_text,
            parent_id,
            parent_label,
            method,
            rule,
            float(score),
            int(support),
            now,
        ),
    )


def build_taxonomy(
    conn: sqlite3.Connection,
    enrichment_table: str,
    parent_candidates_table: str,
    out_table: str,
    method: str,
    clear_out: bool,
    add_typed_parents: bool,
):
    id2term, label2id = load_canonical_terms(conn, enrichment_table)
    head2parent = load_parent_heads(conn, parent_candidates_table, enrichment_table, label2id)

    init_taxonomy_is_a_table(conn, out_table)

    cur = conn.cursor()
    if clear_out:
        cur.execute(f"DELETE FROM {out_table};")
        conn.commit()
        print(f"[INFO] Cleared existing rows from {out_table}.")

    inserted = 0
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # --- optional typed parent nodes (no canonical_id) ---
    # These are meta-parents you can later map to OWL classes: OptionFlag, ConfigParam, ConfigFile, JobState, etc.
    # We store them as parent_head_text like "type:option_flag" to avoid collisions with normal heads.
    def typed_parent_label(ptype: str) -> str:
        return f"type:{ptype}"

    for child_id, label in id2term.items():
        toks = tokenize(label)
        if len(toks) < 2:
            # still allow typed rules for single-token things like "--nodes"
            pass

        # --------------------------
        # Rule 1: head-last-token
        # --------------------------
        if len(toks) >= 2:
            head = toks[-1]
            parent_info = head2parent.get(head)
            if parent_info is not None:
                parent_id, parent_label, freq = parent_info

                # avoid trivial self loops
                if parent_id == child_id or parent_label.lower() == label.lower():
                    pass
                else:
                    # score = head frequency (simple, strong baseline)
                    score = float(freq)
                    support = 1
                    insert_edge(
                        cur=cur,
                        out_table=out_table,
                        child_id=child_id,
                        child_label=label,
                        parent_head_text=head,
                        parent_id=parent_id,
                        parent_label=parent_label,
                        method=method,
                        rule="head_last_token",
                        score=score,
                        support=support,
                        now=now,
                    )
                    inserted += 1

        # --------------------------
        # Rule 2: typed parents (optional)
        # --------------------------
        if add_typed_parents:
            ptype = infer_parent_type(label)
            if ptype:
                # avoid nonsensical case: if head is literally the same as type token
                # This is a meta-type edge; parent_id is None.
                insert_edge(
                    cur=cur,
                    out_table=out_table,
                    child_id=child_id,
                    child_label=label,
                    parent_head_text=typed_parent_label(ptype),
                    parent_id=None,
                    parent_label=typed_parent_label(ptype),
                    method=method,
                    rule="typed_rule",
                    score=5.0,      # fixed baseline; you can tune later
                    support=1,
                    now=now,
                )
                inserted += 1

    conn.commit()
    print(f"[INFO] Inserted/updated {inserted} edges into {out_table}.")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Taxonomy extraction (head-based + optional typed parents).")
    ap.add_argument("--db", required=True)

    ap.add_argument("--enrichment_table", default="term_enrichment",
                    help="Canonical term table (term_enrichment / term_enrichment_v2 / term_enrichment_exten)")
    ap.add_argument("--parent_candidates_table", default="taxonomy_parent_candidates",
                    help="Head candidates table from parent_terms step")
    ap.add_argument("--out_table", default="taxonomy_is_a")

    ap.add_argument("--method", default="head_parent_candidates_v2",
                    help="Method label stored in taxonomy table")
    ap.add_argument("--clear_out", action="store_true",
                    help="Delete existing taxonomy rows in out_table before inserting")

    ap.add_argument("--add_typed_parents", action="store_true",
                    help="Add extra is_a edges to meta-type parents (type:option_flag, type:config_param, etc.)")

    args = ap.parse_args()

    conn = get_connection(args.db)
    try:
        build_taxonomy(
            conn=conn,
            enrichment_table=args.enrichment_table,
            parent_candidates_table=args.parent_candidates_table,
            out_table=args.out_table,
            method=args.method,
            clear_out=args.clear_out,
            add_typed_parents=args.add_typed_parents,
        )
    finally:
        conn.close()

    print("[INFO] Done building taxonomy.")


if __name__ == "__main__":
    main()
