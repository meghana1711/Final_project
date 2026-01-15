#!/usr/bin/env python3
import argparse
import re
from difflib import SequenceMatcher
from collections import defaultdict, Counter

import pandas as pd

# ---------------------------
# Regex (fallback if columns missing)
# ---------------------------
CMD_RE = re.compile(
    r"\b(lsadmin|badmin|bsub|bjobs|bstop|bresume|bkill|lsrun|"
    r"sacctmgr|sacct|sreport|scontrol|sinfo|sbatch|srun|salloc|scancel)\b",
    re.I,
)
DAEMON_RE = re.compile(r"\b(lim|res|mbatchd|sbatchd|slurmctld|slurmd|slurmdbd)\b", re.I)
CFG_RE = re.compile(r"\b[A-Za-z0-9_-]+\.(?:conf|lsf)\b|/etc/[A-Za-z0-9._/-]+|--[A-Za-z0-9][A-Za-z0-9_-]{1,40}\b", re.I)
DOMAIN_RE = re.compile(r"\b(job|node|partition|queue|qos|account|user|gres|tres|priority|fair[- ]share)\b", re.I)
BAD_RE = re.compile(r"\bEXCERPT\b|\baccording to\b|\bfunction\b|\breturn\b|\bstruct\b|_t\b|\block\b|\bmutex\b", re.I)
DOCID_RE = re.compile(r"\bdoc_[0-9a-fA-F]{2,}(?:_chunk_[0-9]{3,})?\b", re.I)

REL_RE = re.compile(r"\b(enforces|requires|limits|controls|sets|causes|prevents|affects|relationship|depends on)\b", re.I)
GENERIC_RE = re.compile(r"\b(overview|describe|explain|introduction|key features|in general)\b", re.I)

# ---------------------------
# Bucket tagger (coverage)
# ---------------------------
BUCKETS = [
    ("submission", re.compile(r"\b(sbatch|srun|salloc|scancel|bsub|lsrun|bjobs|bstop|bresume|bkill|submit|job script|interactive)\b", re.I)),
    ("resources", re.compile(r"\b(cpu|cpus|memory|mem|gpu|gres|tres|nodes?|tasks?|ntasks|cpus-per-task|--gres|--mem|--cpus|--nodes)\b", re.I)),
    ("scheduling", re.compile(r"\b(partition|queue|qos|priority|fair[- ]share|multifactor|backfill|preempt|reservation|walltime|time limit)\b", re.I)),
    ("accounting", re.compile(r"\b(slurmdbd|sacctmgr|sacct|sreport|accounting|association|accounts?|organizations?)\b", re.I)),
    ("components", re.compile(r"\b(slurmctld|slurmd|slurmdbd|controller|daemon|mbatchd|sbatchd|lim|res|auth|munge|plugin)\b", re.I)),
    ("troubleshooting", re.compile(r"\b(down|drain|pending|reason|error|fail|failed|rebooted|healthcheck|debug|log|not running|permission denied|cannot|why doesn't)\b", re.I)),
]

BUCKET_PRIORITY = ["troubleshooting", "accounting", "scheduling", "resources", "submission", "components"]


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-/\.]", "", s)
    return s


def starter_class(q: str) -> str:
    ql = (q or "").strip().lower()
    if ql.startswith("what happens"):
        return "what_if"
    m = re.match(r"^\s*([a-z]+)", ql)
    return m.group(1) if m else "other"


def bucket_tag(q: str) -> str:
    hits = []
    for name, rx in BUCKETS:
        if rx.search(q or ""):
            hits.append(name)
    if not hits:
        return "other"
    # pick by priority
    for p in BUCKET_PRIORITY:
        if p in hits:
            return p
    return hits[0]


def is_near_duplicate(a: str, b: str, thr: float = 0.92) -> bool:
    """Cheap near-duplicate check using normalized SequenceMatcher."""
    na, nb = normalize_text(a), normalize_text(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    sim = SequenceMatcher(None, na, nb).ratio()
    return sim >= thr


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """If some scoring columns are missing, compute basic ones."""
    if "cq_text" not in df.columns:
        raise ValueError("Input CSV must have cq_text column.")

    if "auto_score_0_10" not in df.columns:
        # basic score if missing
        def basic_score(q):
            q = q or ""
            bad = bool(BAD_RE.search(q))
            has_cmd = bool(CMD_RE.search(q))
            has_cfg = bool(CFG_RE.search(q))
            has_daemon = bool(DAEMON_RE.search(q))
            has_domain = bool(DOMAIN_RE.search(q))
            grounding = 2 if (has_cmd or has_cfg or has_daemon) else (1 if has_domain else 0)
            specificity = 2 if (has_cfg or has_cmd) else (1 if has_domain else 0)
            mappability = 2 if (has_domain and REL_RE.search(q)) else (1 if has_domain else 0)
            operational = 2 if (has_cmd or has_cfg or re.search(r"\b(why|error|down|drain|fail|pending|reason)\b", q, re.I)) else (1 if has_domain else 0)
            clarity = 2 if q.strip().endswith("?") and len(q.strip()) >= 25 else (1 if q.strip().endswith("?") else 0)
            total = grounding + specificity + mappability + operational + clarity
            if bad:
                total = min(total, 3)
            return total

        df["auto_score_0_10"] = df["cq_text"].apply(basic_score)

    # compute flags if missing
    for col, rx in [
        ("bad_internal", BAD_RE),
        ("has_cmd", CMD_RE),
        ("has_cfg", CFG_RE),
        ("has_daemon", DAEMON_RE),
        ("has_domain", DOMAIN_RE),
    ]:
        if col not in df.columns:
            df[col] = df["cq_text"].apply(lambda q: bool(rx.search(q or "")))

    if "generic_flag" not in df.columns:
        df["generic_flag"] = df["cq_text"].apply(lambda q: bool(GENERIC_RE.search(q or "")))

    if "starter" not in df.columns:
        df["starter"] = df["cq_text"].apply(lambda q: starter_class(q))

    df["starter_cls"] = df["cq_text"].apply(starter_class)
    df["bucket"] = df["cq_text"].apply(bucket_tag)

    # mappability heuristic: domain + (relation or cmd/cfg/daemon)
    df["mappable"] = df.apply(
        lambda r: bool(r["has_domain"]) and (bool(REL_RE.search(r["cq_text"] or "")) or bool(r["has_cmd"]) or bool(r["has_cfg"]) or bool(r["has_daemon"])),
        axis=1
    )

    return df


def hard_filter(df: pd.DataFrame, min_score: int) -> pd.DataFrame:
    def ok(q: str) -> bool:
        if not q or len(q.strip()) < 15:
            return False
        if DOCID_RE.search(q):
            return False
        if BAD_RE.search(q):
            return False
        if "excerpt" in (q or "").lower():
            return False
        return True

    df = df.copy()
    df = df[df["cq_text"].apply(ok)]
    df = df[~df["generic_flag"]]
    df = df[df["auto_score_0_10"] >= min_score]

    # require at least one strong anchor
    df = df[(df["has_cmd"]) | (df["has_cfg"]) | (df["has_daemon"]) | (df["has_domain"])]

    # require mappable (ontology-focused)
    df = df[df["mappable"]]

    return df


def pick_best30(
    df: pd.DataFrame,
    per_bucket: int = 5,
    total: int = 30,
    max_what: int = 10,
    prefer_starters=("how", "why", "which", "what_if"),
) -> pd.DataFrame:
    # rank: score desc, then prefer cfg/cmd/daemon
    df = df.copy()
    df["rank_tuple"] = df.apply(
        lambda r: (int(r["auto_score_0_10"]), int(r["has_cfg"]), int(r["has_cmd"]), int(r["has_daemon"])),
        axis=1,
    )
    df = df.sort_values(by="rank_tuple", ascending=False)

    selected = []
    used_idx = set()

    # global duplicate control
    def can_add(q: str) -> bool:
        for s in selected:
            if is_near_duplicate(q, s["cq_text"]):
                return False
        return True

    # Step A: bucket coverage
    buckets = [b for b in BUCKET_PRIORITY if b != "other"]
    for b in buckets:
        cands = df[df["bucket"] == b]
        taken = 0
        for idx, row in cands.iterrows():
            if idx in used_idx:
                continue
            if not can_add(row["cq_text"]):
                continue
            selected.append(row)
            used_idx.add(idx)
            taken += 1
            if taken >= per_bucket:
                break

    # Step B: fill remaining with best overall
    if len(selected) < total:
        for idx, row in df.iterrows():
            if len(selected) >= total:
                break
            if idx in used_idx:
                continue
            if not can_add(row["cq_text"]):
                continue
            selected.append(row)
            used_idx.add(idx)

    sel = pd.DataFrame(selected).copy()
    if sel.empty:
        return sel

    # Step C: starter diversity control (limit too many "what")
    def improve_diversity(sel_df: pd.DataFrame) -> pd.DataFrame:
        sel_df = sel_df.copy()
        counts = Counter(sel_df["starter_cls"].tolist())

        # too many "what"
        if counts.get("what", 0) > max_what:
            excess = counts["what"] - max_what
            # remove lowest-scoring "what"
            what_rows = sel_df[sel_df["starter_cls"] == "what"].sort_values(by="rank_tuple", ascending=True)
            to_remove = what_rows.head(excess).index.tolist()

            # try replace with non-what from remaining df
            remaining = df.drop(index=sel_df.index, errors="ignore")
            replacements = []
            for idx_r in to_remove:
                found = None
                # prefer desired starters
                for st in prefer_starters:
                    pool = remaining[remaining["starter_cls"] == st]
                    if not pool.empty:
                        for idx2, r2 in pool.iterrows():
                            if can_add(r2["cq_text"]):
                                found = (idx2, r2)
                                break
                    if found:
                        break
                # else any non-what
                if not found:
                    pool = remaining[remaining["starter_cls"] != "what"]
                    for idx2, r2 in pool.iterrows():
                        if can_add(r2["cq_text"]):
                            found = (idx2, r2)
                            break
                if found:
                    replacements.append((idx_r, found[0], found[1]))

            # apply replacements
            for idx_r, idx2, r2 in replacements:
                sel_df = sel_df.drop(index=idx_r)
                sel_df = pd.concat([sel_df, pd.DataFrame([r2])], ignore_index=True)

        return sel_df

    sel2 = improve_diversity(sel)

    # final trim to total
    sel2 = sel2.sort_values(by="rank_tuple", ascending=False).head(total).reset_index(drop=True)
    return sel2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="e.g., det_scored.csv")
    ap.add_argument("--best_csv", default="best30.csv")
    ap.add_argument("--shortlist_csv", default="shortlist.csv")
    ap.add_argument("--bucket_counts_csv", default="bucket_counts.csv")

    ap.add_argument("--min_score", type=int, default=7, help="hard filter minimum auto score")
    ap.add_argument("--per_bucket", type=int, default=5)
    ap.add_argument("--total", type=int, default=30)
    ap.add_argument("--max_what", type=int, default=10)

    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df = ensure_columns(df)

    # Hard filter + shortlist
    shortlist = hard_filter(df, min_score=args.min_score).copy()
    shortlist = shortlist.sort_values(by=["auto_score_0_10", "has_cfg", "has_cmd", "has_daemon"], ascending=False)
    shortlist.to_csv(args.shortlist_csv, index=False)

    # Best 30 selection
    best = pick_best30(
        shortlist,
        per_bucket=args.per_bucket,
        total=args.total,
        max_what=args.max_what,
    )
    best.to_csv(args.best_csv, index=False)

    # Bucket summary
    bucket_counts = best["bucket"].value_counts().rename_axis("bucket").reset_index(name="count")
    bucket_counts.to_csv(args.bucket_counts_csv, index=False)

    print("[DONE]")
    print("Shortlist:", args.shortlist_csv, "rows =", len(shortlist))
    print("Best 30 :", args.best_csv, "rows =", len(best))
    print("\nBucket counts:")
    print(bucket_counts.to_string(index=False))

    print("\nStarter counts:")
    print(best["starter_cls"].value_counts().to_string())


if __name__ == "__main__":
    main()
