import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Dict, List, Set

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

DB_PATH = r"onto_db/onto_new.db"  # <-- change if needed

MAX_EXAMPLES_PER_HEAD = 15  # examples stored per head
MIN_HEAD_LEN = 3             # words shorter than this cannot be heads
MIN_HEAD_FREQ = 3            # strong heads: must appear >= 2 times as last token


# ---------------------------------------------------------
# DB helpers
# ---------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_parent_candidates_table(conn: sqlite3.Connection) -> None:
    """
    Table to store automatically derived candidate parent heads.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS taxonomy_parent_candidates (
            head_text            TEXT PRIMARY KEY,
            head_canonical_id    INTEGER,
            frequency            INTEGER NOT NULL,
            example_terms_json   TEXT,
            FOREIGN KEY (head_canonical_id)
                REFERENCES term_enrichment(canonical_id)
        )
        """
    )
    conn.commit()
    print("[INFO] Ensured taxonomy_parent_candidates table exists.")


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

      - replace '_' and '/' with spaces
      - split CamelCase (CpuBind -> 'Cpu Bind')
      - normalize spaces
      - lowercase and split on spaces

    Returns list of tokens (lowercased).
    """
    if not text:
        return []

    # unify underscores/slashes with spaces
    t = re.sub(r"[_/]+", " ", text)

    # split CamelCase boundaries
    t = _CAMEL_SPLIT_RE.sub(" ", t)

    t = normalize_spaces(t)
    tokens = t.split()
    return [tok.lower() for tok in tokens]


def normalize_head_token(tok: str) -> str:
    """
    Extra normalization for head tokens:
      - map tres/tress/TRES/Tres -> 'tres'
      - other tokens: just return (already lowercased)
    """
    if tok in {"tres", "tress"}:
        return "tres"
    return tok


def valid_head(tok: str) -> bool:
    """
    Check if token can be a head term.
    - length >= MIN_HEAD_LEN
    - has at least one alphabetic character
    """
    if len(tok) < MIN_HEAD_LEN:
        return False
    return any(ch.isalpha() for ch in tok)


# ---------------------------------------------------------
# Core logic
# ---------------------------------------------------------

def derive_parent_candidates(conn: sqlite3.Connection) -> None:
    """
    1) Read canonical terms from term_enrichment (including synonyms_json).
    2) Collect all labels (canonical + synonyms).
    3) Pass 1: for each label, take the last token as candidate head and count frequencies.
    4) Keep only "strong heads" with freq >= MIN_HEAD_FREQ.
    5) Pass 2: for each label, look at ALL tokens; if a token is a strong head,
       attach that label to that head (so 'Cpu bind' will be counted under 'bind'
       if 'bind' is already a strong head).
    6) Map heads back to canonical_id when there is a canonical_term == head.
    7) Write final heads into taxonomy_parent_candidates.
    """
    rows = conn.execute(
        """
        SELECT canonical_id, canonical_term, synonyms_json
        FROM term_enrichment
        WHERE canonical_term IS NOT NULL
          AND TRIM(canonical_term) != ''
        """
    ).fetchall()

    # 1) Build label -> canonical_id for canonical labels only
    label2id: Dict[str, int] = {}
    for r in rows:
        cid = int(r["canonical_id"])
        label = (r["canonical_term"] or "").strip()
        if not label:
            continue
        label2id[label.lower()] = cid

    # 2) Collect all labels we want to analyse (canonical + synonyms)
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

    # -----------------------------------------------------
    # PASS 1: count heads from LAST TOKEN ONLY
    # -----------------------------------------------------
    head_counts_initial: Counter = Counter()

    for label in labels:
        tokens = tokenize_for_head(label)
        if len(tokens) < 2:
            continue  # only multi-word labels contribute

        head = normalize_head_token(tokens[-1])
        if not valid_head(head):
            continue

        head_counts_initial[head] += 1

    print(f"[INFO] Found {len(head_counts_initial)} distinct heads in initial pass.")

    # Only keep strong heads (freq >= MIN_HEAD_FREQ)
    strong_heads: Set[str] = {
        h for h, c in head_counts_initial.items() if c >= MIN_HEAD_FREQ
    }

    print(f"[INFO] Strong heads (freq >= {MIN_HEAD_FREQ}): {len(strong_heads)}")

    # -----------------------------------------------------
    # PASS 2: for each label, attach to any strong head appearing
    #         in its tokens (not just last token)
    # -----------------------------------------------------
    head_counts_final: Counter = Counter()
    head_examples: Dict[str, Set[str]] = defaultdict(set)

    for label in labels:
        tokens = tokenize_for_head(label)
        if len(tokens) < 2:
            continue

        norm_tokens = [normalize_head_token(t) for t in tokens]
        # Heads this label contributes to (avoid counting same head twice per label)
        heads_here = {
            t for t in norm_tokens if valid_head(t) and t in strong_heads
        }
        if not heads_here:
            continue

        for h in heads_here:
            head_counts_final[h] += 1
            if len(head_examples[h]) < MAX_EXAMPLES_PER_HEAD:
                head_examples[h].add(label)

    print(f"[INFO] Final heads after pass 2: {len(head_counts_final)}")

    # -----------------------------------------------------
    # Save into DB
    # -----------------------------------------------------
    init_parent_candidates_table(conn)

    cur = conn.cursor()
    inserted = 0

    for head, freq in head_counts_final.most_common():
        examples_list = list(head_examples[head])
        examples_json = json.dumps(examples_list, ensure_ascii=False)

        head_canonical_id = label2id.get(head)  # if head itself is a canonical term

        cur.execute(
            """
            INSERT INTO taxonomy_parent_candidates
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
    print(f"[INFO] Inserted/updated {inserted} rows in taxonomy_parent_candidates.")

    # Optional: print top heads for quick inspection
    print("\n[TOP HEAD CANDIDATES]")
    for head, freq in head_counts_final.most_common(30):
        print(f"{head:20s}  freq={freq}")
        for ex in list(head_examples[head])[:5]:
            print(f"   - {ex}")
        print()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    conn = get_connection(DB_PATH)
    derive_parent_candidates(conn)
    conn.close()
    print("[INFO] Done deriving parent candidates.")


if __name__ == "__main__":
    main()
