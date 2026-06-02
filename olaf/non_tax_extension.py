from __future__ import annotations

import argparse
import yaml
import json
import re
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Prompt config loader
# -----------------------------
def load_prompt_config(path: str) -> dict:
    """Load prompt configuration from a YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("prompt_config must be a YAML mapping/object")
    if "system_prompt" not in cfg:
        raise ValueError("prompt_config missing required key: system_prompt")
    return cfg


def _render_template(t: str, mapping: Dict[str, str]) -> str:
    out = t
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def build_prompt(cfg: dict, subj: str, rel: str, obj: str, sentence: str) -> str:
    system_prompt = str(cfg.get("system_prompt", "")).strip()
    user_template = cfg.get(
        "user_template",
        "SUBJECT: {SUBJECT}\nRELATION_KEY: {REL}\nOBJECT: {OBJECT}\nSENTENCE: {SENTENCE}\nReturn JSON only.",
    )
    user_block = _render_template(
        str(user_template),
        {"SUBJECT": subj, "REL": rel, "OBJECT": obj, "SENTENCE": sentence},
    )
    return system_prompt + "\n\n" + user_block.strip() + "\n"


# -----------------------------
# HF runner (batched)
# -----------------------------
def load_hf_textgen(model_name: str, device: str = "auto"):
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model=model_name,
        device_map=device,
        torch_dtype="auto",
    )


_JSON_OBJ_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _as_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""


def _decision_binary(x: Any) -> str:
    s = str(x or "").strip().upper()
    return s if s in {"ACCEPT", "REJECT"} else "REJECT"


def _confidence_01(x: Any) -> float:
    # accept: 0..1 expected; if model outputs 1..10, convert to 0..1
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v > 1.0 and v <= 10.0:
        v = v / 10.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def run_llm_batch(gen, prompts: List[str], max_new_tokens: int, temperature: float) -> List[Optional[dict]]:
    outs = gen(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        return_full_text=False,
    )

    results: List[Optional[dict]] = []
    for out in outs:
        txt = ""
        if isinstance(out, list) and out:
            txt = (out[0].get("generated_text") or "").strip()
        elif isinstance(out, dict):
            txt = (out.get("generated_text") or "").strip()

        data = _extract_json(txt)
        if not isinstance(data, dict):
            results.append(None)
            continue

        needed = {"decision", "confidence", "reason"}
        if not needed.issubset(set(data.keys())):
            results.append(None)
            continue

        results.append(
            {
                "decision": _decision_binary(data.get("decision")),
                "confidence": _confidence_01(data.get("confidence")),
                "reason": _as_str(data.get("reason")),
            }
        )

    if len(results) != len(prompts):
        results = (results + [None] * len(prompts))[:len(prompts)]
    return results


# -----------------------------
# DB helpers / schema
# -----------------------------
def connect(db: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    r = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return r is not None


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cols = set()
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    for r in rows:
        cols.add(str(r["name"]))
    return cols


def ensure_llm_table(conn: sqlite3.Connection, out_table: str) -> None:
    # If you changed schema before, easiest is to DROP TABLE manually once.
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {out_table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            source_edge_id      INTEGER NOT NULL,
            sentence_text       TEXT,

            subj_canonical_term TEXT NOT NULL,
            rel_key             TEXT NOT NULL,
            obj_canonical_term  TEXT NOT NULL,

            decision            TEXT NOT NULL,   -- ACCEPT/REJECT
            confidence          REAL NOT NULL,   -- 0..1
            reason              TEXT NOT NULL,

            model_name          TEXT NOT NULL,
            created_at          TEXT NOT NULL,

            UNIQUE(source_edge_id, model_name)
        )
        """
    )
    conn.commit()


def materialize_accept_only(conn: sqlite3.Connection, llm_table: str, out_accept_table: str) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {out_accept_table}")
    conn.execute(
        f"""
        CREATE TABLE {out_accept_table} AS
        SELECT *
        FROM {llm_table}
        WHERE UPPER(decision)='ACCEPT'
        """
    )
    conn.commit()


# -----------------------------
# Main routine
# -----------------------------
def process_all_edges(
    db: str,
    in_edges_table: str,
    out_llm_table: str,
    out_accept_table: str,
    model_name: str,
    device: str,
    prompt_config: str,
    limit: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    commit_every: int,
    log_every: int,
) -> None:
    cfg = load_prompt_config(prompt_config)

    conn = connect(db)
    try:
        if not table_exists(conn, in_edges_table):
            raise RuntimeError(f"Missing input table: {in_edges_table}")

        cols = table_columns(conn, in_edges_table)
        required = {"id", "subj_canonical_term", "rel_key", "obj_canonical_term"}
        missing = required - cols
        if missing:
            raise RuntimeError(f"{in_edges_table} missing required columns: {sorted(missing)}")

        sent_col = "sentence_text" if "sentence_text" in cols else ("sentence" if "sentence" in cols else None)

        ensure_llm_table(conn, out_llm_table)

        q = f"""
        SELECT
          id,
          subj_canonical_term,
          rel_key,
          obj_canonical_term
          {"," + sent_col + " AS sentence_text" if sent_col else ", '' AS sentence_text"}
        FROM {in_edges_table}
        """
        if limit and limit > 0:
            q += " LIMIT ?"
            rows = conn.execute(q, (limit,)).fetchall()
        else:
            rows = conn.execute(q).fetchall()

        print(f"[INFO] Loaded {len(rows)} edges from {in_edges_table}.")
        if not rows:
            return

        gen = load_hf_textgen(model_name, device=device)
        cur = conn.cursor()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")

        processed = 0
        inserted = 0
        forced_reject = 0  # cases where we store REJECT because JSON parse failed or fields missing

        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            prompts: List[str] = []
            meta: List[Tuple[int, str, str, str, str]] = []

            for r in batch:
                eid = int(r["id"])
                s = str(r["subj_canonical_term"] or "").strip()
                rel = str(r["rel_key"] or "").strip()
                o = str(r["obj_canonical_term"] or "").strip()
                sent = str(r["sentence_text"] or "").strip()

                # We still build a prompt only when fields exist;
                # but we will ALWAYS write an output row (forced reject if missing).
                if s and rel and o:
                    prompts.append(build_prompt(cfg, s, rel, o, sent if sent else "none"))
                else:
                    prompts.append("")  # placeholder

                meta.append((eid, s, rel, o, sent))

            idx = [k for k, p in enumerate(prompts) if p]
            sub_prompts = [prompts[k] for k in idx]
            sub_outs: List[Optional[dict]] = []
            if sub_prompts:
                sub_outs = run_llm_batch(gen, sub_prompts, max_new_tokens=max_new_tokens, temperature=temperature)

            outs: List[Optional[dict]] = [None] * len(prompts)
            for j, k in enumerate(idx):
                outs[k] = sub_outs[j] if j < len(sub_outs) else None

            for (eid, s, rel, o, sent), data in zip(meta, outs):
                processed += 1

                if not (s and rel and o):
                    # missing fields -> forced reject
                    decision, conf, reason = "REJECT", 0.0, "Missing subject/relation/object"
                    forced_reject += 1
                elif data is None:
                    # model output didn't match strict JSON -> forced reject
                    decision, conf, reason = "REJECT", 0.0, "LLM output not valid JSON with required keys"
                    forced_reject += 1
                else:
                    decision = data["decision"]
                    conf = float(data["confidence"])
                    reason = data["reason"] or ""

                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO {out_llm_table} (
                      source_edge_id, sentence_text,
                      subj_canonical_term, rel_key, obj_canonical_term,
                      decision, confidence, reason,
                      model_name, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (eid, sent, s, rel, o, decision, conf, reason, model_name, now),
                )
                inserted += 1

                if processed % log_every == 0:
                    print(f"[INFO] processed={processed} inserted={inserted} forced_reject={forced_reject}")
                if processed % commit_every == 0:
                    conn.commit()

        conn.commit()
        print(f"[DONE] processed={processed} inserted={inserted} forced_reject={forced_reject}")
        print(f"[INFO] LLM decisions table: {out_llm_table}")

        # Optional: accept-only table for downstream KG/axioms
        materialize_accept_only(conn, out_llm_table, out_accept_table)
        print(f"[OK] Accept-only table: {out_accept_table}")

    finally:
        conn.close()


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Binary LLM validation of non-tax edges: ACCEPT/REJECT only, store decision for every edge.")

    ap.add_argument("--db", required=True)

    ap.add_argument("--in_edges_table", default="non_taxonomic_edges_clean")
    ap.add_argument("--out_llm_table", default="non_taxonomic_edges_llm_binary")
    ap.add_argument("--out_accept_table", default="non_taxonomic_edges_accept")

    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--device", default="auto", help="HF device_map: auto/cuda/cpu")

    ap.add_argument("--non_tax_config", default="prompts/non_tax_extension.yaml")

    ap.add_argument("--limit", type=int, default=0, help="Limit edges (0=all)")
    ap.add_argument("--batch_size", type=int, default=6)

    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--commit_every", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=50)

    return ap.parse_args()


def main():
    args = parse_args()
    process_all_edges(
        db=args.db,
        in_edges_table=args.in_edges_table,
        out_llm_table=args.out_llm_table,
        out_accept_table=args.out_accept_table,
        model_name=args.model,
        device=args.device,
        prompt_config=args.non_tax_config,
        limit=int(args.limit),
        batch_size=max(1, int(args.batch_size)),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        commit_every=max(1, int(args.commit_every)),
        log_every=max(1, int(args.log_every)),
    )


if __name__ == "__main__":
    main()
