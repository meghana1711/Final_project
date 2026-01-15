import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Dict, List, Set


# ---------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------

_CAMEL_SPLIT_RE = re.compile(r"(?<!^)(?=[A-Z])")


def normalize_spaces(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize_for_head(text: str) -> List[str]:
    """
    Turn a term label into tokens for head detection:
      - split CamelCase (CpuBind -> 'Cpu Bind')
      - normalize spaces
      - lowercase and split on spaces
    """
    if not text:
        return []
    t = _CAMEL_SPLIT_RE.sub(" ", text)
    t = normalize_spaces(t)
    return [tok.lower() for tok in t.split()]


def normalize_head_token(tok: str) -> str:
    if tok in {"tres", "tress"}:
        return "tres"
    return tok


def valid_head(tok: str, min_head_len: int) -> bool:
    if len(tok) < min_head_len:
        return False
    return any(ch.isalpha() for ch in tok)


# ---------------------------------------------------------
# DB helpers
# ---------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_parent_candidates_table(
    conn: sqlite3.Connection,
    out_table: str,
    enrichment_table: str,
) -> None:
    """
    Store derived candidate parent heads.
    Note: head_canonical_id FK points to enrichment_table(canonical_id).
    """
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            head_text            TEXT PRIMARY KEY,
            head_canonical_id    INTEGER,
            frequency            INTEGER NOT NULL,
            example_terms_json   TEXT,
            FOREIGN KEY (head_canonical_id)
                REFERENCES {enrichment_table}(canonical_id)
        )
        """
    )
    conn.commit()
    print(f"[INFO] Ensured {out_table} table exists.")


# ---------------------------------------------------------
# Core logic
# ---------------------------------------------------------

def derive_parent_candidates(
    conn: sqlite3.Connection,
    enrichment_table: str,
    out_table: str,
    max_examples_per_head: int,
    min_head_len: int,
    min_head_freq: int,
) -> None:
    """
    Derive frequent "head tokens" from canonical_term + synonyms_json.

    Two-pass logic:
      Pass 1: last-token heads only (find strong heads)
      Pass 2: attach labels to any strong head appearing in tokens
    """

    rows = conn.execute(
        f"""
        SELECT canonical_id, canonical_term, synonyms_json
        FROM {enrichment_table}
        WHERE canonical_term IS NOT NULL
          AND TRIM(canonical_term) != ''
        """
    ).fetchall()

    # label -> canonical_id (canonical only)
    label2id: Dict[str, int] = {}
    for r in rows:
        cid = int(r["canonical_id"])
        label = (r["canonical_term"] or "").strip()
        if label:
            label2id[label.lower()] = cid

    # collect labels: canonical + synonyms
    labels: List[str] = []
    for r in rows:
        canon_label = (r["canonical_term"] or "").strip()
        if canon_label:
            labels.append(canon_label)

        syn_json = r["synonyms_json"]
        if syn_json:
            try:
                syns = json.loads(syn_json)
            except json.JSONDecodeError:
                syns = []
            if isinstance(syns, list):
                for s in syns:
                    if isinstance(s, str) and s.strip():
                        labels.append(s.strip())

    print(f"[INFO] Collected {len(labels)} labels (canonical + synonyms).")

    # ----------------------------
    # PASS 1: count last-token heads
    # ----------------------------
    head_counts_initial: Counter = Counter()

    for label in labels:
        tokens = tokenize_for_head(label)
        if len(tokens) < 2:
            continue

        head = normalize_head_token(tokens[-1])
        if not valid_head(head, min_head_len):
            continue

        head_counts_initial[head] += 1

    strong_heads: Set[str] = {h for h, c in head_counts_initial.items() if c >= min_head_freq}

    print(f"[INFO] Found {len(head_counts_initial)} distinct heads in pass 1.")
    print(f"[INFO] Strong heads (freq >= {min_head_freq}): {len(strong_heads)}")

    # ----------------------------
    # PASS 2: attach label to any strong head inside it
    # ----------------------------
    head_counts_final: Counter = Counter()
    head_examples: Dict[str, Set[str]] = defaultdict(set)

    for label in labels:
        tokens = tokenize_for_head(label)
        if len(tokens) < 2:
            continue

        norm_tokens = [normalize_head_token(t) for t in tokens]
        heads_here = {t for t in norm_tokens if t in strong_heads and valid_head(t, min_head_len)}
        if not heads_here:
            continue

        for h in heads_here:
            head_counts_final[h] += 1
            if len(head_examples[h]) < max_examples_per_head:
                head_examples[h].add(label)

    print(f"[INFO] Final heads after pass 2: {len(head_counts_final)}")

    # ----------------------------
    # save to DB
    # ----------------------------
    init_parent_candidates_table(conn, out_table, enrichment_table)

    cur = conn.cursor()
    inserted = 0

    for head, freq in head_counts_final.most_common():
        examples_list = list(head_examples[head])
        examples_json = json.dumps(examples_list, ensure_ascii=False)
        head_canonical_id = label2id.get(head)

        cur.execute(
            f"""
            INSERT INTO {out_table}
                (head_text, head_canonical_id, frequency, example_terms_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(head_text) DO UPDATE SET
                head_canonical_id  = excluded.head_canonical_id,
                frequency          = excluded.frequency,
                example_terms_json = excluded.example_terms_json
            """,
            (head, head_canonical_id, int(freq), examples_json),
        )
        inserted += 1

    conn.commit()
    print(f"[INFO] Inserted/updated {inserted} rows in {out_table}.")

    print("\n[TOP HEAD CANDIDATES]")
    for head, freq in head_counts_final.most_common(30):
        print(f"{head:20s}  freq={freq}")
        for ex in list(head_examples[head])[:5]:
            print(f"   - {ex}")
        print()


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Derive taxonomy parent head candidates from term enrichment table.")
    ap.add_argument("--db", required=True)

    ap.add_argument("--enrichment_table", default="term_enrichment_v2",
                    help="Source enrichment table with canonical_term + synonyms_json")
    ap.add_argument("--out_table", default="taxonomy_parent_candidates",
                    help="Destination table for head candidates")

    ap.add_argument("--max_examples_per_head", type=int, default=15)
    ap.add_argument("--min_head_len", type=int, default=3)
    ap.add_argument("--min_head_freq", type=int, default=3)

    args = ap.parse_args()

    conn = get_connection(args.db)
    try:
        derive_parent_candidates(
            conn,
            enrichment_table=args.enrichment_table,
            out_table=args.out_table,
            max_examples_per_head=args.max_examples_per_head,
            min_head_len=args.min_head_len,
            min_head_freq=args.min_head_freq,
        )
    finally:
        conn.close()

    print("[INFO] Done deriving parent candidates.")


if __name__ == "__main__":
    main()
