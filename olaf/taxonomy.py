"""
olaf/taxonomy.py

Final taxonomy builder:
- Seed taxonomy (head-match using extracted parents)
- Embedding expansion using skipgram_neighbors (ID-based) [DEFAULT ON]
- Hearst confirmation from sentence table [DEFAULT ON]

output table: taxonomy_is_a (configurable via --out_table)
"""

from __future__ import annotations

import argparse
import sqlite3
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple


# -----------------------------
# Helpers
# -----------------------------

def norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def norm_l(s: Optional[str]) -> str:
    return norm(s).lower()

def tokenize(text: str) -> List[str]:
    t = norm_l(text)
    t = re.sub(r"[^a-z0-9_\- ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return [x for x in t.split() if x]

def table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def col_exists(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def ensure_out_table(cur: sqlite3.Cursor, out_table: str) -> None:
    # (2) confidence column removed
    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {out_table} (
      child TEXT NOT NULL,
      parent TEXT NOT NULL,
      method TEXT NOT NULL,
      evidence TEXT,
      PRIMARY KEY(child, parent, method)
    )
    """)

def is_identity_edge(child: str, parent: str) -> bool:
    return norm_l(child) == norm_l(parent)


# -----------------------------
# Parent extraction from definitions (simple, stable)
# -----------------------------

DEF_HEAD_RE = re.compile(r"^\s*(?:a|an|the)\s+(?P<head>[^.]{1,240})", re.IGNORECASE)
CLAUSE_CUT_RE = re.compile(r"\b(that|which|used|for|to|with|provides|responsible)\b", re.I)
TRAILING_PREP_RE = re.compile(r"\b(of|in|on|for|to|at|by|from)\b\s*$", re.I)

BAD_PARENTS = {"thing", "type", "types", "value", "values", "property", "properties", "details", "information", "data"}
DROP_TOKENS = {"the", "a", "an", "this", "that", "these", "those"}
SCHEDULER_TOKENS = {"slurm", "lsf", "pbs", "torque", "sge"}

def normalize_parent_phrase(raw_phrase: str) -> Optional[str]:
    if not raw_phrase:
        return None
    p = norm_l(raw_phrase)
    p = CLAUSE_CUT_RE.split(p, maxsplit=1)[0]
    p = re.sub(r"[^a-z0-9_\- ]", " ", p)
    p = re.sub(r"\s+", " ", p).strip()
    if not p:
        return None

    toks = [t for t in p.split() if t and t not in DROP_TOKENS and t not in SCHEDULER_TOKENS]
    if not toks:
        return None

    p = " ".join(toks).strip()
    p = TRAILING_PREP_RE.sub("", p).strip()

    if not p or p in BAD_PARENTS:
        return None

    # keep last up to 3 tokens
    tt = p.split()
    if len(tt) > 3:
        p = " ".join(tt[-3:])

    if p in BAD_PARENTS or len(p) < 3:
        return None
    return p

def parent_from_definition(defn: str) -> Optional[str]:
    if not defn:
        return None
    m = DEF_HEAD_RE.match(defn)
    if not m:
        return None
    return normalize_parent_phrase(m.group("head"))


# -----------------------------
# Seed taxonomy: head match
# -----------------------------

ROLE_TOKENS = {"user", "administrator", "admin", "owner", "operator", "manager"}
FILE_WORDS = {"file", "files", "path", "directory", "dir", "conf", "config", "cfg"}
BAD_CHILD_RE = re.compile(r"\b(one|two|three|[0-9]+)\s+or\s+more\b|\bor\s+more\b", re.I)

def is_bad_child(term: str) -> bool:
    t = term.lower()
    if BAD_CHILD_RE.search(t):
        return True
    toks = set(tokenize(term))
    if toks & ROLE_TOKENS:
        return True
    if toks & FILE_WORDS:
        return True
    if "/etc/" in t or "/var/" in t or "\\" in t:
        return True
    return False

def head_match(child: str, parent: str, max_parent_tokens: int = 3) -> bool:
    ct = tokenize(child)
    pt = tokenize(parent)
    if not ct or not pt:
        return False
    if len(pt) > max_parent_tokens:
        return False
    if len(ct) < len(pt):
        return False
    return ct[-len(pt):] == pt


# -----------------------------
# Hearst patterns (small, high precision)
# -----------------------------

PAT_ISA = re.compile(r"\b(?P<x>[a-z0-9_\- ]{2,80})\s+is\s+(?:a|an)\s+(?P<y>[a-z0-9_\- ]{2,80})\b", re.I)
PAT_OTHER = re.compile(r"\b(?P<x>[a-z0-9_\- ]{2,80})\s+(?:and|or)\s+other\s+(?P<y>[a-z0-9_\- ]{2,80})\b", re.I)
PAT_SUCH_AS = re.compile(r"\b(?P<y>[a-z0-9_\- ]{2,80})\s+such\s+as\s+(?P<x>[a-z0-9_\- ]{2,80})\b", re.I)

def clean_np(s: str) -> str:
    s = norm_l(s)
    s = re.sub(r"[^a-z0-9_\- ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = s.split()
    if len(toks) > 4:
        s = " ".join(toks[-4:])
    return s


# -----------------------------
# (6) Simple duplicate normalization for swapped scheduler phrases
# Example: "host lsf" -> "lsf host"
# Only applied to 2-token phrases where one token is scheduler and the other is a generic head.
# -----------------------------

GENERIC_HEADS = {"host", "job", "daemon", "command", "queue", "plugin"}

def canonicalize_swapped_scheduler_phrase(term: str) -> str:
    t = norm_l(term)
    toks = t.split()
    if len(toks) == 2:
        a, b = toks
        if a in SCHEDULER_TOKENS and b in GENERIC_HEADS:
            return f"{a} {b}"
        if b in SCHEDULER_TOKENS and a in GENERIC_HEADS:
            return f"{b} {a}"
    return t

def build_term_rep_map(terms: List[str]) -> Dict[str, str]:
    """
    Map a "duplicate key" -> chosen representative label.
    duplicate key is based on canonicalize_swapped_scheduler_phrase(lowercase term).
    Representative chosen: prefer scheduler-first form, else keep first seen.
    """
    key2rep: Dict[str, str] = {}
    for original in terms:
        orig_norm = norm(original)
        key = canonicalize_swapped_scheduler_phrase(orig_norm)
        if key not in key2rep:
            # representative label: reconstruct scheduler-first if applicable, else original
            # preserve original casing/spacing where possible
            if key != norm_l(orig_norm):
                # key is normalized lowercase; use original but normalized spacing
                # and also force scheduler-first tokens for display
                key2rep[key] = " ".join(key.split())
            else:
                key2rep[key] = orig_norm
    return key2rep

def to_rep(label: str, key2rep: Dict[str, str]) -> str:
    key = canonicalize_swapped_scheduler_phrase(label)
    return key2rep.get(key, norm(label))


# -----------------------------
# Skipgram expansion (ID-based)
# skipgram_neighbors schema:
#   term_id, neighbor_term_id, similarity ...
# We align canonical_term -> term_candidates.term_lemma -> term_id
# -----------------------------

def load_term_candidates_lemma_maps(
    cur: sqlite3.Cursor,
    term_candidates_table: str,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    lemma2id: Dict[str, int] = {}
    id2lemma: Dict[int, str] = {}
    cur.execute(
        f"""
        SELECT term_id, term_lemma
        FROM {term_candidates_table}
        WHERE term_lemma IS NOT NULL AND TRIM(term_lemma) != ''
        """
    )
    for tid, lem in cur.fetchall():
        l = norm_l(lem)
        lemma2id[l] = int(tid)
        id2lemma[int(tid)] = l
    return lemma2id, id2lemma


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Build taxonomy_is_a (seed + embeddings + hearst). Embeddings/Hearst enabled by default."
    )
    ap.add_argument("--db", required=True)
    ap.add_argument("--terms_table", default="term_enrichment_exten")

    # (4) This script writes taxonomy_is_a by default
    ap.add_argument("--out_table", default="taxonomy_is_a")

    # Seed parent selection
    # (5) removed --top_parents; auto-select parents
    ap.add_argument("--min_children_per_parent", type=int, default=2)

    # Embeddings (3) default ON; disable via flag
    ap.add_argument("--no_embeddings", action="store_true", help="Disable embedding expansion.")
    ap.add_argument("--sim_table", default="skipgram_neighbors")
    ap.add_argument("--term_candidates_table", default="term_candidates")
    ap.add_argument("--top_k_neighbors", type=int, default=5)
    ap.add_argument("--min_cos", type=float, default=0.75)
    ap.add_argument("--require_same_category", action="store_true")

    # Hearst (3) default ON; disable via flag
    ap.add_argument("--no_hearst", action="store_true", help="Disable Hearst extraction.")
    ap.add_argument("--sent_table", default="sentence_lemmatized")
    ap.add_argument("--sent_col", default="sentence")
    ap.add_argument("--max_sents", type=int, default=200000)

    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    if not table_exists(cur, args.terms_table):
        raise RuntimeError(f"Missing terms_table: {args.terms_table}")

    # Required cols in term table
    for c in ["canonical_term", "ontology_role", "is_hpc_domain_term", "confidence"]:
        if not col_exists(cur, args.terms_table, c):
            raise RuntimeError(f"{args.terms_table} missing column: {c}")

    has_cat = col_exists(cur, args.terms_table, "category")
    has_def = col_exists(cur, args.terms_table, "short_definition")

    ensure_out_table(cur, args.out_table)
    cur.execute(f"DELETE FROM {args.out_table}")

    # ---- Load meta (canonical truth) ----
    cur.execute(f"""
      SELECT canonical_term, ontology_role, is_hpc_domain_term, confidence
             {", category" if has_cat else ""}
             {", short_definition" if has_def else ""}
      FROM {args.terms_table}
      WHERE canonical_term IS NOT NULL AND TRIM(canonical_term) != ''
    """)

    meta: Dict[str, Dict] = {}
    all_terms_raw: List[str] = []

    for r in cur.fetchall():
        term = norm(r["canonical_term"])
        if not term:
            continue
        all_terms_raw.append(term)

        role = norm_l(r["ontology_role"])
        try:
            hpc = int(r["is_hpc_domain_term"])
        except Exception:
            hpc = 0
        conf = float(r["confidence"]) if (r["confidence"] is not None) else 0.0
        cat = norm_l(r["category"]) if has_cat else ""
        dfn = norm(r["short_definition"]) if has_def else ""
        meta[term.lower()] = {"term": term, "role": role, "hpc": hpc, "cat": cat, "conf": conf, "defn": dfn}

    # (6) build representative map to collapse swapped duplicates like "host lsf" -> "lsf host"
    key2rep = build_term_rep_map(all_terms_raw)

    # (1) eligibility is hard-coded per your requirement
    def eligible_class(label: str) -> bool:
        m = meta.get(label.lower())
        if not m:
            return False
        if m["role"] != "class":
            return False
        if m["hpc"] != 1:
            return False
        if m["conf"] < 0.7:
            return False
        return True

    eligible_terms_raw = [m["term"] for m in meta.values() if eligible_class(m["term"])]

    # apply representative normalization (drop duplicates)
    eligible_terms = sorted({to_rep(t, key2rep) for t in eligible_terms_raw})

    # ---- Step 1: Build parent list (from defs + frequent class terms) ----
    class_support = Counter()
    def_support = Counter()

    for m in meta.values():
        if not eligible_class(m["term"]):
            continue

        rep = to_rep(m["term"], key2rep).lower()
        if rep in BAD_PARENTS:
            continue
        class_support[rep] += 1

        if has_def and m["defn"]:
            p = parent_from_definition(m["defn"])
            if p:
                p_rep = canonicalize_swapped_scheduler_phrase(p)
                if p_rep and p_rep not in BAD_PARENTS:
                    def_support[p_rep] += 1

    # parent scoring: definitions weighted higher
    score: Dict[str, float] = {}
    for p, n in class_support.items():
        if p in BAD_PARENTS:
            continue
        score[p] = max(score.get(p, 0.0), 1.0 * n)
    for p, n in def_support.items():
        if p in BAD_PARENTS:
            continue
        score[p] = max(score.get(p, 0.0), 2.0 * n)

    # (5) No top-N cut. Keep all parents that have some score and are not bad.
    parents_ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    parent_list = [p for p, sc in parents_ranked if sc > 0.0 and p not in BAD_PARENTS]

    # ---- Step 2: Seed edges by head-match ----
    scaffold: List[Tuple[str, str]] = []
    for child in eligible_terms:
        if is_bad_child(child):
            continue
        for parent in parent_list:
            # parent is lowercase normalized string
            if norm_l(child) == parent:
                continue
            if head_match(child, parent, max_parent_tokens=3):
                scaffold.append((child, parent))

    parent_counts = Counter(p for _, p in scaffold)
    whitelist = {p for p, n in parent_counts.items() if n >= args.min_children_per_parent and p not in BAD_PARENTS}

    seed_edges = 0
    parent2seeds: Dict[str, Set[str]] = defaultdict(set)

    for child, parent in scaffold:
        if parent not in whitelist:
            continue
        # (6) identity removal
        if is_identity_edge(child, parent):
            continue

        # store parent as canonical display (scheduler-first if applicable)
        parent_label = " ".join(parent.split())

        # (2) no confidence column
        cur.execute(
            f"INSERT OR REPLACE INTO {args.out_table}(child,parent,method,evidence) VALUES (?,?,?,?)",
            (child, parent_label, "seed", "head_match"),
        )
        seed_edges += 1
        parent2seeds[parent_label].add(child)

    # ---- Step 3: Embedding expansion (skipgram_neighbors) ----
    embed_added = 0
    use_embeddings = not args.no_embeddings
    if use_embeddings:
        if not table_exists(cur, args.sim_table):
            raise RuntimeError(f"Missing sim_table: {args.sim_table}")
        if not table_exists(cur, args.term_candidates_table):
            raise RuntimeError(f"Missing term_candidates_table: {args.term_candidates_table}")

        for c in ["term_id", "neighbor_term_id", "similarity"]:
            if not col_exists(cur, args.sim_table, c):
                raise RuntimeError(f"{args.sim_table} missing column {c}. Expected skipgram_neighbors schema.")

        lemma2id, id2lemma = load_term_candidates_lemma_maps(cur, args.term_candidates_table)

        for parent_label, seeds in parent2seeds.items():
            if not seeds:
                continue

            target_cat = None
            if args.require_same_category and has_cat:
                cats = []
                for s in seeds:
                    # map to raw meta if possible
                    s_key = to_rep(s, key2rep).lower()
                    cats.append(meta.get(s_key, {}).get("cat", ""))
                cats = [c for c in cats if c]
                if cats:
                    target_cat = max(set(cats), key=cats.count)

            added_children = set(seeds)

            for seed in list(seeds):
                seed_lemma = norm_l(seed)
                seed_id = lemma2id.get(seed_lemma)
                if seed_id is None:
                    continue

                cur.execute(
                    f"""
                    SELECT neighbor_term_id, similarity
                    FROM {args.sim_table}
                    WHERE term_id = ?
                      AND similarity >= ?
                    ORDER BY similarity DESC
                    LIMIT ?
                    """,
                    (seed_id, args.min_cos, args.top_k_neighbors),
                )

                for nb_id, sim in cur.fetchall():
                    nb_lemma = id2lemma.get(int(nb_id))
                    if not nb_lemma:
                        continue

                    # neighbor must exist as canonical term
                    if nb_lemma not in meta:
                        continue
                    nb_term_raw = meta[nb_lemma]["term"]
                    nb_term = to_rep(nb_term_raw, key2rep)

                    if nb_term in added_children:
                        continue
                    if not eligible_class(nb_term) or is_bad_child(nb_term):
                        continue
                    if is_identity_edge(nb_term, parent_label):
                        continue

                    if args.require_same_category and target_cat:
                        nb_cat = meta.get(nb_lemma, {}).get("cat", "")
                        if not nb_cat or nb_cat != target_cat:
                            continue

                    cur.execute(
                        f"INSERT OR REPLACE INTO {args.out_table}(child,parent,method,evidence) VALUES (?,?,?,?)",
                        (nb_term, parent_label, f"embed_expand(k={args.top_k_neighbors},min_cos={args.min_cos})",
                         f"seed={seed};sim={float(sim):.3f}"),
                    )
                    embed_added += 1
                    added_children.add(nb_term)

    # ---- Step 4: Hearst confirmation ----
    hearst_added = 0
    use_hearst = not args.no_hearst
    if use_hearst:
        if not table_exists(cur, args.sent_table):
            raise RuntimeError(f"Missing sent_table: {args.sent_table}")
        if not col_exists(cur, args.sent_table, args.sent_col):
            raise RuntimeError(f"{args.sent_table} missing column {args.sent_col}")

        # known eligible class keys (lowercase, representative-normalized)
        known = {to_rep(v["term"], key2rep).lower() for k, v in meta.items() if eligible_class(v["term"])}

        cur.execute(f"SELECT {args.sent_col} AS s FROM {args.sent_table} LIMIT ?", (args.max_sents,))
        for (s,) in cur.fetchall():
            if not s:
                continue
            text = str(s)

            for pat in (PAT_ISA, PAT_OTHER, PAT_SUCH_AS):
                for m in pat.finditer(text):
                    x = clean_np(m.group("x"))
                    y = clean_np(m.group("y"))
                    if not x or not y or x == y:
                        continue
                    if y in BAD_PARENTS:
                        continue

                    # representative-normalize x,y if they correspond to known terms
                    x_rep = to_rep(x, key2rep).lower()
                    y_rep = to_rep(y, key2rep).lower()

                    if x_rep not in known or y_rep not in known:
                        continue

                    child = to_rep(meta[x_rep]["term"] if x_rep in meta else x_rep, key2rep)
                    parent = to_rep(meta[y_rep]["term"] if y_rep in meta else y_rep, key2rep)

                    if is_bad_child(child):
                        continue
                    if is_identity_edge(child, parent):
                        continue

                    cur.execute(
                        f"INSERT OR REPLACE INTO {args.out_table}(child,parent,method,evidence) VALUES (?,?,?,?)",
                        (child, parent, "hearst", text[:220]),
                    )
                    hearst_added += 1

    con.commit()
    con.close()

    print(f"[OK] Wrote {args.out_table}")
    print(f"     eligible terms (class,hpc,conf>=0.7): {len(eligible_terms)}")
    print(f"     parents scored: {len(parent_list)} (no top-N cut)")
    print(f"     seed edges:     {seed_edges}")
    print(f"     embed added:    {embed_added} (embeddings={'ON' if use_embeddings else 'OFF'})")
    print(f"     hearst added:   {hearst_added} (hearst={'ON' if use_hearst else 'OFF'})")
    print(f"     parents kept:   {len(whitelist)} (min_children_per_parent={args.min_children_per_parent})")


if __name__ == "__main__":
    main()
