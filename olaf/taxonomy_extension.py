"""
olaf/taxonomy_extension.py

LLM taxonomy validator (IS-A):

Input (non-LLM taxonomy):  taxonomy_is_a
Output (LLM-labeled):      taxonomy_is_a_final

What it does:
- Reads ALL edges from taxonomy_is_a (no hard-case filter; no dropping).
- For each (child,parent), asks LLM:
    1) is_a_valid? (0/1)
    2) accept? (0/1)
    3) best_parent (either one of candidates OR "none")
    4) confidence + reason + evidence_sentence (optional)
- Writes ONE row per source edge into taxonomy_is_a_final.
- Never deletes edges. If reject -> llm_accept=0 and llm_best_parent='none'.
- Robust JSON extraction + tolerant parsing (extra keys allowed).
- Stores raw_llm_json so you can debug parse failures.

CLI example:
python -m olaf.taxonomy_extension \
  --db onto_db/sample2.db \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --prompt_config prompts/taxonomy_extension.json

Prompt JSON file format (required keys):
{
  "system_prompt": "...",
  "few_shots": [
     { "input": {...}, "output": {...} },
     ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# -----------------------------
# HF loader
# -----------------------------
def load_textgen(model_name: str):
    import torch  # noqa
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    return tok, model


# -----------------------------
# Prompt config loader
# -----------------------------
def load_prompt_config(path: str) -> Tuple[str, List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    system = (cfg.get("system_prompt") or "").strip()
    few = cfg.get("few_shots") or []
    if not system:
        raise ValueError(f"prompt_config missing system_prompt: {path}")
    if not isinstance(few, list):
        raise ValueError("prompt_config few_shots must be a list")
    for i, ex in enumerate(few[:50]):
        if not isinstance(ex, dict) or "input" not in ex or "output" not in ex:
            raise ValueError(f"few_shots[{i}] must have input/output objects")
    return system, few


# -----------------------------
# Tokenize helpers
# -----------------------------
def tokenize(text: str) -> List[str]:
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z0-9_\- ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return [x for x in t.split() if x]


# -----------------------------
# Evidence retrieval (optional but useful)
# -----------------------------
def table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def col_exists(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())


def fetch_sentence_evidence(
    conn: sqlite3.Connection,
    sent_table: str,
    sent_col: str,
    child: str,
    parent: str,
    limit: int,
) -> List[str]:
    """
    Simple LIKE evidence.
    Prefer sentences containing BOTH child and parent; fallback to child only.
    """
    child = (child or "").strip()
    parent = (parent or "").strip()
    if not child:
        return []
    cur = conn.cursor()

    if parent:
        cur.execute(
            f"""
            SELECT {sent_col}
            FROM {sent_table}
            WHERE {sent_col} LIKE ?
              AND {sent_col} LIKE ?
            LIMIT ?
            """,
            (f"%{child}%", f"%{parent}%", limit),
        )
        both = [r[0] for r in cur.fetchall() if r and r[0]]
        if both:
            return both

    cur.execute(
        f"SELECT {sent_col} FROM {sent_table} WHERE {sent_col} LIKE ? LIMIT ?",
        (f"%{child}%", limit),
    )
    return [r[0] for r in cur.fetchall() if r and r[0]]


def top_parents_from_taxonomy(conn: sqlite3.Connection, taxonomy_table: str, k: int) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT parent, COUNT(*) AS n
        FROM {taxonomy_table}
        WHERE parent IS NOT NULL AND TRIM(parent) != ''
        GROUP BY parent
        ORDER BY n DESC
        LIMIT ?
        """,
        (k,),
    )
    return [(r[0] or "").strip() for r in cur.fetchall() if r and r[0]]


# -----------------------------
# Robust JSON extraction
# -----------------------------
def extract_first_json_object(text: str) -> Optional[str]:
    """
    Finds the first balanced {...} substring that parses as JSON dict.
    Works better than "last { }" when prompts/few-shots contain braces.
    """
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = text[start : i + 1]
                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return cand
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


def generate_raw(tok, model, system: str, user: str, max_new_tokens: int = 240) -> str:
    import torch

    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = system + "\n\nUSER:\n" + user + "\n\nASSISTANT:\n"

    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )
    decoded = tok.decode(out[0], skip_special_tokens=True)
    return decoded


def safe_parse_llm_json(s: str) -> Optional[Dict]:
    """
    Tolerant parser:
    - allows extra keys
    - accepts missing optional evidence_sentence
    Required keys (minimum):
      child_is_class, parent_is_class, is_a_valid, accept, best_parent, confidence, reason
    """
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    required = {
        "child_is_class",
        "parent_is_class",
        "is_a_valid",
        "accept",
        "best_parent",
        "confidence",
        "reason",
    }
    if not required.issubset(set(obj.keys())):
        return None

    # types
    if not isinstance(obj["child_is_class"], bool):
        return None
    if not isinstance(obj["parent_is_class"], bool):
        return None
    if not isinstance(obj["is_a_valid"], bool):
        return None
    if not isinstance(obj["accept"], bool):
        return None

    bp = obj.get("best_parent", "none")
    if bp is None:
        bp = "none"
    obj["best_parent"] = str(bp).strip()

    try:
        obj["confidence"] = float(obj["confidence"])
    except Exception:
        return None

    obj["reason"] = str(obj.get("reason", "")).strip()
    obj["evidence_sentence"] = str(obj.get("evidence_sentence", "") or "").strip()
    return obj


# -----------------------------
# Output table
# -----------------------------
def ensure_out_table(conn: sqlite3.Connection, out_table: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            src_rowid             INTEGER NOT NULL,

            child                 TEXT NOT NULL,
            parent                TEXT NOT NULL,
            method                TEXT,
            src_confidence        REAL,
            src_evidence          TEXT,

            llm_child_is_class    INTEGER NOT NULL,
            llm_parent_is_class   INTEGER NOT NULL,
            llm_is_a_valid        INTEGER NOT NULL,

            llm_accept            INTEGER NOT NULL,
            llm_best_parent       TEXT NOT NULL,
            llm_confidence        REAL NOT NULL,
            llm_reason            TEXT NOT NULL,
            llm_evidence_sentence TEXT,

            candidates_json       TEXT,
            evidence_json         TEXT,
            raw_llm_json          TEXT,

            created_at            TEXT NOT NULL,

            UNIQUE(src_rowid)
        )
        """
    )
    conn.commit()


# -----------------------------
# Main validation run
# -----------------------------
def run(
    db_path: str,
    in_taxonomy_table: str,
    out_table: str,
    llm_model: str,
    prompt_config: str,
    few_shots_k: int,
    max_rows: int,
    sent_table: str,
    sent_col: str,
    evidence_k: int,
    global_parents_k: int,
    max_new_tokens: int,
    debug_print_fail_k: int,
) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if not table_exists(cur, in_taxonomy_table):
        raise RuntimeError(f"Missing taxonomy table: {in_taxonomy_table}")

    # input columns: child,parent,method,confidence,evidence are optional except child,parent
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({in_taxonomy_table})").fetchall()]
    if "child" not in cols or "parent" not in cols:
        raise RuntimeError(
            f"{in_taxonomy_table} must have columns child,parent. Found: {cols}"
        )

    has_method = "method" in cols
    has_conf = "confidence" in cols
    has_ev = "evidence" in cols

    has_sent = table_exists(cur, sent_table) and col_exists(cur, sent_table, sent_col)

    ensure_out_table(conn, out_table)

    system_prompt, few_shots = load_prompt_config(prompt_config)
    few_shots = few_shots[: max(0, few_shots_k)]

    global_parents: List[str] = []
    if global_parents_k and global_parents_k > 0:
        global_parents = top_parents_from_taxonomy(conn, in_taxonomy_table, k=global_parents_k)

    print(f"[INFO] Loading HF model: {llm_model}")
    tok, model = load_textgen(llm_model)

    q = f"SELECT rowid AS src_rowid, child, parent"
    if has_method:
        q += ", method"
    if has_conf:
        q += ", confidence"
    if has_ev:
        q += ", evidence"
    q += f" FROM {in_taxonomy_table} ORDER BY rowid"
    if max_rows and max_rows > 0:
        q += " LIMIT ?"
        edges = conn.execute(q, (max_rows,)).fetchall()
    else:
        edges = conn.execute(q).fetchall()

    print(f"[INFO] Loaded {len(edges)} edges from {in_taxonomy_table}. (processing ALL)")

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    inserted = 0
    parse_failed_printed = 0

    for e in edges:
        rid = int(e["src_rowid"])
        child = (e["child"] or "").strip()
        parent = (e["parent"] or "").strip()
        method = (e["method"] or "").strip() if has_method else ""
        src_conf = float(e["confidence"] or 0.0) if has_conf else 0.0
        src_evidence = (e["evidence"] or "") if has_ev else ""

        if not child or not parent:
            # still write a row (keep traceability) but mark reject
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {out_table} (
                    src_rowid, child, parent, method, src_confidence, src_evidence,
                    llm_child_is_class, llm_parent_is_class, llm_is_a_valid,
                    llm_accept, llm_best_parent, llm_confidence, llm_reason,
                    llm_evidence_sentence, candidates_json, evidence_json, raw_llm_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid, child or "UNKNOWN_CHILD", parent or "UNKNOWN_PARENT", method, src_conf, src_evidence,
                    0, 0, 0,
                    0, "none", 0.0, "Empty child/parent in source edge.",
                    "", "[]", "[]", "", now,
                ),
            )
            inserted += 1
            continue

        # candidates: proposed parent + some globals + none
        candidates: List[str] = [parent]
        for gp in global_parents:
            if gp and gp.lower() != parent.lower():
                candidates.append(gp)
            if len(candidates) >= 8:
                break
        candidates.append("none")

        # dedupe candidates
        seen = set()
        cand_final = []
        for c in candidates:
            c = (c or "").strip()
            if not c:
                continue
            k = c.lower()
            if k in seen:
                continue
            seen.add(k)
            cand_final.append(c)

        evidence: List[str] = []
        if has_sent:
            evidence = fetch_sentence_evidence(conn, sent_table, sent_col, child, parent, limit=evidence_k)

        # prompt
        parts: List[str] = []
        for ex in few_shots:
            parts.append("EXAMPLE_INPUT:\n" + json.dumps(ex["input"], ensure_ascii=False))
            parts.append("EXAMPLE_OUTPUT:\n" + json.dumps(ex["output"], ensure_ascii=False))

        payload = {
            "child_term": child,
            "proposed_parent": parent,
            "candidates": cand_final,
            "evidence": evidence[:3],
            "instruction": (
                "Return JSON only. Decide if (child is-a proposed_parent). "
                "If reject, set accept=false and best_parent='none' OR choose a better best_parent from candidates."
            ),
            "required_output_schema": {
                "child_is_class": "bool",
                "parent_is_class": "bool",
                "is_a_valid": "bool",
                "accept": "bool",
                "best_parent": "string (must be one of candidates)",
                "confidence": "float 0..1",
                "reason": "string",
                "evidence_sentence": "string (optional; can be empty)"
            }
        }
        parts.append("INPUT:\n" + json.dumps(payload, ensure_ascii=False))
        user_prompt = "\n\n".join(parts)

        decoded = generate_raw(tok, model, system_prompt, user_prompt, max_new_tokens=max_new_tokens)

        json_str = extract_first_json_object(decoded)  # robust
        parsed = safe_parse_llm_json(json_str or "")

        if parsed is None:
            if parse_failed_printed < max(0, debug_print_fail_k):
                print("\n=== LLM PARSE FAILED (sample) ===")
                print(f"edge: child='{child}' parent='{parent}'")
                print("decoded_tail:", decoded[-800:])
                parse_failed_printed += 1

            # conservative fallback: do NOT accept; keep none
            child_is_class = True
            parent_is_class = True
            is_a_valid = False
            accept = False
            best_parent = "none"
            conf = 0.10
            reason = "LLM parse failed (no valid JSON). Marked reject with best_parent=none."
            ev_sent = evidence[0][:220] if evidence else ""
            raw_llm_json = json_str or ""
        else:
            child_is_class = bool(parsed["child_is_class"])
            parent_is_class = bool(parsed["parent_is_class"])
            is_a_valid = bool(parsed["is_a_valid"])
            accept = bool(parsed["accept"])
            best_parent = (parsed.get("best_parent") or "none").strip()
            conf = float(parsed["confidence"])
            reason = (parsed.get("reason") or "").strip()
            ev_sent = (parsed.get("evidence_sentence") or "").strip()
            raw_llm_json = json_str or ""

        # enforce: best_parent must be in candidates
        cand_lower = {c.lower() for c in cand_final}
        if best_parent.lower() not in cand_lower:
            best_parent = "none"
            accept = False
            conf = min(conf, 0.30)
            reason = (reason + " (best_parent not in candidates; forced none)").strip()

        # enforce logic: if not valid, accept must be false
        if not (child_is_class and parent_is_class and is_a_valid):
            accept = False
            if best_parent.lower() != "none":
                best_parent = "none"

        conn.execute(
            f"""
            INSERT OR REPLACE INTO {out_table} (
                src_rowid, child, parent, method, src_confidence, src_evidence,
                llm_child_is_class, llm_parent_is_class, llm_is_a_valid,
                llm_accept, llm_best_parent, llm_confidence, llm_reason,
                llm_evidence_sentence,
                candidates_json, evidence_json, raw_llm_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid, child, parent, method, src_conf, src_evidence,
                1 if child_is_class else 0,
                1 if parent_is_class else 0,
                1 if is_a_valid else 0,
                1 if accept else 0,
                best_parent,
                float(conf),
                reason,
                ev_sent,
                json.dumps(cand_final, ensure_ascii=False),
                json.dumps(evidence, ensure_ascii=False),
                raw_llm_json,
                now,
            ),
        )
        inserted += 1

        if inserted % 50 == 0:
            conn.commit()
            print(f"[INFO] processed={inserted}/{len(edges)}")

    conn.commit()
    conn.close()
    print(f"[OK] Wrote {inserted} rows into {out_table} (no dropping).")


def main():
    ap = argparse.ArgumentParser(description="Taxonomy IS-A validation with LLM (process ALL edges; no dropping).")

    ap.add_argument("--db", required=True)

    # Naming you requested:
    # - taxonomy_is_a : non-LLM taxonomy table
    # - taxonomy_is_a_final : taxonomy with LLM labels/decisions
    ap.add_argument("--in_taxonomy_table", default="taxonomy_is_a")
    ap.add_argument("--out_table", default="taxonomy_is_a_final")

    ap.add_argument("--model", required=True, help="HF model name/path")
    ap.add_argument("--prompt_config", default="prompts/taxonomy_extension.json")

    ap.add_argument("--few_shots_k", type=int, default=6)

    ap.add_argument("--max_rows", type=int, default=0, help="0=all rows")
    ap.add_argument("--max_new_tokens", type=int, default=260)

    # evidence is optional but recommended
    ap.add_argument("--sent_table", default="sentence_lemmatized")
    ap.add_argument("--sent_col", default="sentence")
    ap.add_argument("--evidence_k", type=int, default=2)
    ap.add_argument("--global_parents_k", type=int, default=12)

    ap.add_argument("--debug_print_fail_k", type=int, default=3, help="Print K parse-fail samples")

    args = ap.parse_args()

    run(
        db_path=args.db,
        in_taxonomy_table=args.in_taxonomy_table,
        out_table=args.out_table,
        llm_model=args.model,
        prompt_config=args.prompt_config,
        few_shots_k=args.few_shots_k,
        max_rows=args.max_rows,
        sent_table=args.sent_table,
        sent_col=args.sent_col,
        evidence_k=args.evidence_k,
        global_parents_k=args.global_parents_k,
        max_new_tokens=args.max_new_tokens,
        debug_print_fail_k=args.debug_print_fail_k,
    )


if __name__ == "__main__":
    main()
