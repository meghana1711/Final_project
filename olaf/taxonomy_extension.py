"""
olaf/taxonomy_extension.py

LLM taxonomy validator (IS-A):

Input (non-LLM taxonomy):  taxonomy_is_a
Output (LLM-labeled):      taxonomy_is_a_final

"""

from __future__ import annotations

import argparse
import yaml
import json
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


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
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Prompt config must be a YAML object: {path}")

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
# Evidence retrieval
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
    """Find first balanced {...} that parses as a JSON dict."""
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


def extract_last_json_object(text: str) -> Optional[str]:
    """Find last balanced {...} that parses as a JSON dict. Helpful if model outputs multiple JSON blocks."""
    if not text:
        return None
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(starts):
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
    return None


# -----------------------------
# Completion-only generation  ✅ FIX
# -----------------------------
def generate_completion(tok, model, system: str, user: str, max_new_tokens: int = 240) -> str:
    import torch

    if hasattr(tok, "apply_chat_template"):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt_text = system + "\n\nUSER:\n" + user + "\n\nASSISTANT:\n"

    inputs = tok(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )

    # ✅ ONLY decode newly generated tokens
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    completion = tok.decode(gen_ids, skip_special_tokens=True)
    return completion.strip()


# -----------------------------
# Tolerant parsing
# -----------------------------
def to_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        try:
            return bool(int(x))
        except Exception:
            return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "t", "yes", "y", "1"):
            return True
        if s in ("false", "f", "no", "n", "0"):
            return False
    return None


def safe_parse_llm_json(s: str) -> Optional[Dict]:
    """
    Required keys:
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

    b1 = to_bool(obj.get("child_is_class"))
    b2 = to_bool(obj.get("parent_is_class"))
    b3 = to_bool(obj.get("is_a_valid"))
    b4 = to_bool(obj.get("accept"))
    if None in (b1, b2, b3, b4):
        return None

    obj["child_is_class"] = bool(b1)
    obj["parent_is_class"] = bool(b2)
    obj["is_a_valid"] = bool(b3)
    obj["accept"] = bool(b4)

    bp = obj.get("best_parent", "none")
    obj["best_parent"] = str(bp if bp is not None else "none").strip()

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

    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({in_taxonomy_table})").fetchall()]
    if "child" not in cols or "parent" not in cols:
        raise RuntimeError(f"{in_taxonomy_table} must have columns child,parent. Found: {cols}")

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

    q = "SELECT rowid AS src_rowid, child, parent"
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

        # candidates
        candidates: List[str] = [parent]
        for gp in global_parents:
            if gp and gp.lower() != parent.lower():
                candidates.append(gp)
            if len(candidates) >= 8:
                break
        candidates.append("none")

        # dedupe
        seen = set()
        cand_final: List[str] = []
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

        # build prompt
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
                "Return a single JSON object ONLY (no prose, no markdown). "
                "Decide if (child is-a proposed_parent). "
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

        completion = generate_completion(tok, model, system_prompt, user_prompt, max_new_tokens=max_new_tokens)

        # ✅ Extract from completion (not the prompt)
        json_str = extract_first_json_object(completion) or extract_last_json_object(completion)
        parsed = safe_parse_llm_json(json_str or "")

        if parsed is None:
            if parse_failed_printed < max(0, debug_print_fail_k):
                print("\n=== LLM PARSE FAILED (sample) ===")
                print(f"edge: child='{child}' parent='{parent}'")
                print("completion_tail:", completion[-800:])
                parse_failed_printed += 1

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
    ap.add_argument("--in_taxonomy_table", default="taxonomy_is_a")
    ap.add_argument("--out_table", default="taxonomy_is_a_final")

    ap.add_argument("--model", required=True, help="HF model name/path")
    ap.add_argument("--prompt_config", default="prompts/taxonomy_extension.yaml")
    ap.add_argument("--few_shots_k", type=int, default=6)

    ap.add_argument("--max_rows", type=int, default=0, help="0=all rows")
    ap.add_argument("--max_new_tokens", type=int, default=260)

    ap.add_argument("--sent_table", default="sentence_lemmatized")
    ap.add_argument("--sent_col", default="sentence")
    ap.add_argument("--evidence_k", type=int, default=2)
    ap.add_argument("--global_parents_k", type=int, default=12)

    ap.add_argument("--debug_print_fail_k", type=int, default=3)

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
