"""
olaf/parent_terms.py

Extract parent term candidates (taxonomy scaffolding) from term_enrichment_exten
using ONLY:
  - ontology_role == 'class'
  - is_hpc_domain_term != 0
  - ontology_role != 'drop'   (redundant but kept explicit)

IMPORTANT UPDATE:
This version ALSO **canonicalizes/abstracts** definition-head phrases so you don't
end up with parents like:
  - "the slurm scheduler"  -> "job scheduler"
  - "partition in slurm"   -> "partition"
  - "queue lsf"            -> "queue"
  - "mbatchd daemon"       -> "daemon"

Output table: parent_terms_extracted (same schema as before)
"""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Dict, List, Optional


# ---------------------------------------------------------
# Definition-head (hypernym) mining helpers
# ---------------------------------------------------------

DEF_HEAD_RE = re.compile(r"^\s*(?:a|an|the)\s+(?P<head>[^.]{1,200})", re.IGNORECASE)

BAD_HEADS = {"thing", "data", "information", "value", "values", "type", "types", "details"}

# remove very common determiners/stop tokens in extracted head phrases
DROP_TOKENS = {"the", "a", "an", "this", "that", "these", "those"}

# scheduler names / products we want to strip from parent phrases
SCHEDULER_TOKENS = {"slurm", "lsf", "pbs", "torque", "sge"}

# clause starters—truncate definition head at these
CLAUSE_CUT = re.compile(r"\b(that|which|used|for|to|with|provides|responsible)\b", re.I)

# preposition tails to strip
TRAILING_PREP = re.compile(r"\b(of|in|on|for|to|at|by|from)\b\s*$", re.I)

# small canonicalization map for stable abstract parents (tiny on purpose)
# (This is NOT hardcoding SLURM commands; it's stabilizing ontology parents.)
CANONICAL_PARENT_MAP = {
    "job queue": "job queue",
    "queue": "job queue",              # unify queue -> job queue (esp. LSF)
    "scheduler": "job scheduler",
    "job scheduler": "job scheduler",
    "slurm scheduler": "job scheduler",
    "daemon": "scheduler daemon",
    "scheduler daemon": "scheduler daemon",
    "service daemon": "scheduler daemon",
    "partition": "partition",
    "job": "job",
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def connect(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con


def table_cols(cur: sqlite3.Cursor, table: str) -> List[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return [r[1] for r in cur.fetchall()]


def pick_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    return None


def normalize_parent_phrase(phrase: str) -> Optional[str]:
    """
    Turn noisy definition-head phrases into reusable parent candidates.
    Examples:
      "the slurm scheduler" -> "job scheduler"
      "partition in slurm"  -> "partition"
      "queue lsf"           -> "job queue" (via map)
      "mbatchd daemon"      -> "scheduler daemon" (via 'daemon' + map)
    """
    if not phrase:
        return None

    p = norm(phrase).lower()
    p = CLAUSE_CUT.split(p, maxsplit=1)[0]
    p = re.sub(r"[^a-z0-9_\- ]", " ", p)
    p = re.sub(r"\s+", " ", p).strip()

    if not p or p in BAD_HEADS:
        return None

    toks = [t for t in p.split() if t and t not in DROP_TOKENS]

    # Remove scheduler/product tokens anywhere
    toks = [t for t in toks if t not in SCHEDULER_TOKENS]

    if not toks:
        return None

    # Drop trailing prepositions (after tokenization, we can also clean tail)
    p2 = " ".join(toks).strip()
    p2 = TRAILING_PREP.sub("", p2).strip()

    if not p2 or p2 in BAD_HEADS:
        return None

    # Keep compact: last 1–3 tokens usually capture the hypernym head
    # e.g. "set of nodes" -> "nodes" (not ideal), "job submission commands" -> "submission commands"
    # We'll prefer last 2–3 for better parent phrase stability.
    toks2 = p2.split()
    if len(toks2) > 3:
        p2 = " ".join(toks2[-3:])

    # apply tiny canonical parent map
    p2 = CANONICAL_PARENT_MAP.get(p2, p2)

    # final sanity checks
    if len(p2) < 3 or p2 in BAD_HEADS:
        return None

    return p2


def extract_parent_from_definition(defn: str) -> Optional[str]:
    """
    Extract and normalize parent candidate from definition text.
    """
    if not defn:
        return None
    m = DEF_HEAD_RE.match(defn)
    if not m:
        return None
    raw_head = m.group("head")
    return normalize_parent_phrase(raw_head)


def ensure_out_table(cur: sqlite3.Cursor, out_table: str) -> None:
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
          parent_term TEXT PRIMARY KEY,
          parent_type TEXT,                -- class_term / definition_head / class_term+definition_head
          score REAL,
          class_support INTEGER,
          def_head_support INTEGER,
          avg_confidence REAL,
          categories_json TEXT,
          evidence_json TEXT
        )
        """
    )


def main():
    ap = argparse.ArgumentParser(
        description="OLAF: Parent term candidates (taxonomy scaffolding) from term_enrichment_exten."
    )
    ap.add_argument("--db", required=True)
    ap.add_argument("--src_table", default="term_enrichment_exten")
    ap.add_argument("--out_table", default="parent_terms_extracted")
    ap.add_argument("--min_conf", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()

    con = connect(args.db)
    cur = con.cursor()

    cols = table_cols(cur, args.src_table)

    term_col = pick_col(cols, ["canonical_term", "term", "label"])
    role_col = pick_col(cols, ["ontology_role"])
    cat_col = pick_col(cols, ["category"])
    conf_col = pick_col(cols, ["confidence"])
    def_col = pick_col(cols, ["short_definition", "definition"])
    hpc_col = pick_col(cols, ["is_hpc_domain_term", "is_hpc_domain", "is_hpc_domian"])

    if not term_col or not role_col:
        raise RuntimeError(
            f"Need term column + ontology_role column in {args.src_table}. Found columns: {cols}"
        )
    if not hpc_col:
        raise RuntimeError(
            f"Could not find is_hpc_domain_term (or is_hpc_domain/is_hpc_domian) in {args.src_table}. Found: {cols}"
        )

    select_cols = [term_col, role_col, hpc_col]
    if cat_col:
        select_cols.append(cat_col)
    if conf_col:
        select_cols.append(conf_col)
    if def_col:
        select_cols.append(def_col)

    cur.execute(f"SELECT {', '.join(select_cols)} FROM {args.src_table}")
    rows = cur.fetchall()

    # ---------------------------------------------------------
    # 1) Parent candidates from CLASS TERMS (already canonical terms)
    # Conditions:
    #   ontology_role = 'class'
    #   is_hpc_domain_term != 0
    #   ontology_role != 'drop'
    # ---------------------------------------------------------
    class_support = Counter()
    class_conf_sum = defaultdict(float)
    class_cat = defaultdict(set)
    class_examples = defaultdict(list)

    for r in rows:
        term = norm(r[term_col])
        role = norm(r[role_col]).lower()

        if not term:
            continue
        if role != "class":
            continue
        if role == "drop":
            continue
        try:
            if int(r[hpc_col]) == 0:
                continue
        except (TypeError, ValueError):
            continue

        conf = float(r[conf_col]) if conf_col and r[conf_col] is not None else 0.0
        if conf < args.min_conf:
            continue

        key = term.lower()
        class_support[key] += 1
        class_conf_sum[key] += conf
        if cat_col:
            c = norm(r[cat_col]).lower()
            if c:
                class_cat[key].add(c)
        if len(class_examples[key]) < 5:
            class_examples[key].append(term)

    # ---------------------------------------------------------
    # 2) Parent candidates from DEFINITION HEADS (normalized/abstracted!)
    # We still restrict to the same trusted rows (class + HPC domain).
    # ---------------------------------------------------------
    head_support = Counter()
    head_examples = defaultdict(list)

    if def_col:
        for r in rows:
            term = norm(r[term_col])
            role = norm(r[role_col]).lower()

            if not term:
                continue
            if role != "class":
                continue
            if role == "drop":
                continue
            try:
                if int(r[hpc_col]) == 0:
                    continue
            except (TypeError, ValueError):
                continue

            defn = r[def_col]
            parent = extract_parent_from_definition(str(defn) if defn else "")
            if not parent:
                continue

            head_support[parent] += 1
            if len(head_examples[parent]) < 6:
                head_examples[parent].append(term)

    # ---------------------------------------------------------
    # Combine into ranked parent candidates
    #   - definition heads weighted higher (more "parent-like")
    # ---------------------------------------------------------
    parent_scores: Dict[str, Dict] = {}

    for k, n in class_support.items():
        avg_conf = class_conf_sum[k] / max(n, 1)
        score = (1.0 * n) + (avg_conf / 100.0)
        parent_scores[k] = {
            "parent_term": k,
            "parent_type": "class_term",
            "score": score,
            "class_support": n,
            "def_head_support": 0,
            "avg_conf": avg_conf,
            "cats": sorted(class_cat[k]),
            "evidence": class_examples[k],
        }

    for h, n in head_support.items():
        score = (2.0 * n)
        if h in parent_scores:
            parent_scores[h]["def_head_support"] = n
            parent_scores[h]["score"] += score
            parent_scores[h]["parent_type"] = "class_term+definition_head"
            parent_scores[h]["evidence"] = (parent_scores[h]["evidence"] + head_examples[h])[:12]
        else:
            parent_scores[h] = {
                "parent_term": h,
                "parent_type": "definition_head",
                "score": score,
                "class_support": 0,
                "def_head_support": n,
                "avg_conf": 0.0,
                "cats": [],
                "evidence": head_examples[h],
            }

    ensure_out_table(cur, args.out_table)

    items = sorted(parent_scores.values(), key=lambda x: x["score"], reverse=True)[: args.top_k]

    for it in items:
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {args.out_table}
              (parent_term, parent_type, score, class_support, def_head_support, avg_confidence, categories_json, evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                it["parent_term"],
                it["parent_type"],
                float(it["score"]),
                int(it["class_support"]),
                int(it["def_head_support"]),
                float(it["avg_conf"]),
                json.dumps(it["cats"], ensure_ascii=False),
                json.dumps(it["evidence"], ensure_ascii=False),
            ),
        )

    con.commit()
    con.close()
    print(f"[OK] Extracted top {len(items)} parent terms into table: {args.out_table}")


if __name__ == "__main__":
    main()
