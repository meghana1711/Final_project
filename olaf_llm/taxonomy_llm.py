from __future__ import annotations

import argparse
import ast
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# GPU knobs
# -----------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# -----------------------------
# Taxonomy input policy (tune this!)
# -----------------------------
# Strongly recommended to keep this tight; broad categories create garbage "is-a".
TAXONOMY_ELIGIBLE_CATEGORIES = {
    "queue_or_partition",
    "resource",
    "scheduler",
    "user_role",
    "other_hpc",
    "config_param",
}


# -----------------------------
# YAML config
# -----------------------------
@dataclass
class PromptCfg:
    system_prompt: str
    prompt_mode: str = "few-shot"  # "few-shot" | "zero-shot"
    max_terms_per_chunk: int = 16
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    few_shots: Optional[List[Dict[str, Any]]] = None  # list of {chunk, indexed_terms, output}


def load_prompt_cfg(path: str) -> PromptCfg:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}

    system_prompt = (obj.get("system_prompt") or "").strip()
    if not system_prompt:
        raise ValueError(f"YAML prompt_config missing system_prompt: {path}")

    return PromptCfg(
        system_prompt=system_prompt,
        prompt_mode=(obj.get("prompt_mode") or "few-shot").strip(),
        max_terms_per_chunk=int(obj.get("max_terms_per_chunk") or 16),
        max_new_tokens=int(obj.get("max_new_tokens") or 256),
        temperature=float(obj.get("temperature") or 0.0),
        top_p=float(obj.get("top_p") or 1.0),
        few_shots=obj.get("few_shots"),
    )


# -----------------------------
# DB schema
# -----------------------------
def init_tables(conn: sqlite3.Connection, edges_table: str, runs_table: str) -> None:
    cur = conn.cursor()

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {edges_table} (
            edge_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id         TEXT NOT NULL,
            chunk_id       TEXT NOT NULL,
            child_term_id  INTEGER,
            parent_term_id INTEGER,
            child_term     TEXT NOT NULL,
            parent_term    TEXT NOT NULL,
            justification  TEXT,
            raw_json       TEXT,
            created_at     TEXT DEFAULT (datetime('now')),
            UNIQUE(doc_id, chunk_id, child_term, parent_term)
        )
        """
    )

    # Runs table fixes resume: mark chunk processed even when zero edges.
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {runs_table} (
            doc_id        TEXT NOT NULL,
            chunk_id      TEXT NOT NULL,
            rowid_src     INTEGER,
            candidate_n   INTEGER,
            kept_edges_n  INTEGER,
            raw_output    TEXT,
            parsed_json   TEXT,
            processed_at  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (doc_id, chunk_id)
        )
        """
    )

    conn.commit()


# -----------------------------
# Model loading
# -----------------------------
def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    return tok, model, device


# -----------------------------
# Resume-safe chunk selection
# -----------------------------
def fetch_chunks(
    conn: sqlite3.Connection,
    chunks_table: str,
    doc_id_col: str,
    chunk_id_col: str,
    text_col: str,
    runs_table: str,
    offset_rowid: int,
    max_chunks: int,
) -> List[Tuple[int, str, str, str]]:
    cur = conn.cursor()

    sql = f"""
        SELECT c.rowid, c.{doc_id_col}, c.{chunk_id_col}, c.{text_col}
        FROM {chunks_table} c
        WHERE c.rowid > ?
          AND NOT EXISTS (
              SELECT 1 FROM {runs_table} r
              WHERE r.doc_id = c.{doc_id_col} AND r.chunk_id = c.{chunk_id_col}
          )
        ORDER BY c.rowid
    """
    params: List[Any] = [offset_rowid]
    if max_chunks and max_chunks > 0:
        sql += " LIMIT ?"
        params.append(max_chunks)

    cur.execute(sql, params)
    return cur.fetchall()


# -----------------------------
# Candidate terms per chunk (UPDATED)
# -----------------------------
def fetch_candidate_terms(
    conn: sqlite3.Connection,
    terms_table: str,
    enrich_table: str,
    doc_id: str,
    chunk_id: str,
    min_freq: int,
    max_terms: int,
) -> List[Dict[str, Any]]:
    """
    Candidate selection:
    - is_hpc_domain = 1
    - ontology_role != drop
    - category in TAXONOMY_ELIGIBLE_CATEGORIES
    - freq_total >= min_freq
    """
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT
            t.term_id,
            t.term,
            COALESCE(e.canonical, LOWER(TRIM(t.term))) AS canonical,
            LOWER(COALESCE(e.ontology_role, 'unknown')) AS ontology_role,
            LOWER(COALESCE(e.category, 'unknown')) AS category,
            COALESCE(e.is_hpc_domain, 1) AS is_hpc_domain,
            t.freq_total
        FROM {terms_table} t
        LEFT JOIN {enrich_table} e
          ON e.term_id = t.term_id
        WHERE t.doc_id = ?
          AND t.chunk_id = ?
          AND t.freq_total >= ?
          AND COALESCE(e.is_hpc_domain, 1) = 1
          AND LOWER(COALESCE(e.ontology_role, 'unknown')) != 'drop'
        ORDER BY t.freq_total DESC, t.term_id ASC
        """,
        (doc_id, chunk_id, max(1, min_freq)),
    )

    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    seen = set()

    for term_id, term, canonical, role, category, is_dom, freq_total in rows:
        s = (term or "").strip()
        if not s:
            continue

        if category not in TAXONOMY_ELIGIBLE_CATEGORIES:
            continue

        k = s.lower()
        if k in seen:
            continue
        seen.add(k)

        out.append(
            {
                "term_id": int(term_id),
                "term": s,
                "canonical": str(canonical or "").strip().lower(),
                "ontology_role": str(role or "unknown").strip().lower(),
                "category": str(category or "unknown").strip().lower(),
                "is_hpc_domain": int(is_dom) if is_dom is not None else 1,
                "freq_total": int(freq_total) if freq_total is not None else 1,
            }
        )

        if max_terms and len(out) >= max_terms:
            break

    return out


def build_idx_maps(candidates: List[Dict[str, Any]]) -> Tuple[Dict[int, str], Dict[int, int]]:
    """
    idx -> surface_term, idx -> term_id
    """
    idx_to_term: Dict[int, str] = {}
    idx_to_term_id: Dict[int, int] = {}
    for i, c in enumerate(candidates, start=1):
        idx_to_term[i] = c["term"]
        idx_to_term_id[i] = int(c["term_id"])
    return idx_to_term, idx_to_term_id


# -----------------------------
# Prompt building (INDEXED)
# -----------------------------
def build_prompt(cfg: PromptCfg, chunk_text: str, candidates: List[Dict[str, Any]]) -> str:
    # Build indexed term list
    idx_lines: List[str] = []
    for i, c in enumerate(candidates, start=1):
        idx_lines.append(f"{i}. {c['term']} [category={c['category']}, role={c['ontology_role']}]")
    indexed_terms_block = "\n".join(idx_lines)

    # Few-shots in indexed style
    if cfg.prompt_mode == "few-shot" and cfg.few_shots:
        blocks: List[str] = []
        for i, ex in enumerate(cfg.few_shots, start=1):
            ex_chunk = ex.get("chunk", "")
            ex_terms = ex.get("indexed_terms", ex.get("candidate_terms", []))  # allow older key
            if isinstance(ex_terms, list):
                ex_terms_block = "\n".join(f"{j+1}. {t}" for j, t in enumerate(ex_terms))
            else:
                ex_terms_block = ""
            ex_out = ex.get("output", {"is_a_edges": []})
            blocks.append(
                f"Example {i}:\n"
                f"CHUNK:\n{ex_chunk}\n\n"
                f"TERM_LIST:\n{ex_terms_block}\n\n"
                f"JSON:\n{json.dumps(ex_out, ensure_ascii=False, indent=2)}\n"
            )
        examples_str = "\n\n".join(blocks)

        user = (
            "You will see examples of HPC documentation CHUNKS and correct is-a outputs.\n"
            "Follow the same behavior for the NEW CHUNK.\n\n"
            f"{examples_str}\n\n"
            "NOW process ONLY this NEW CHUNK:\n\n"
            f"CHUNK:\n{chunk_text}\n\n"
            f"TERM_LIST (use ONLY these indices for child_idx/parent_idx):\n{indexed_terms_block}\n\n"
            "Return ONLY one JSON object with key \"is_a_edges\".\n"
            "Each edge item must have:\n"
            "{\n"
            '  "child_idx": 1,\n'
            '  "parent_idx": 2,\n'
            '  "justification": "short reason"\n'
            "}\n"
        )
    else:
        user = (
            "Extract ONLY true is-a (subclass) relations.\n\n"
            f"CHUNK:\n{chunk_text}\n\n"
            f"TERM_LIST (use ONLY these indices for child_idx/parent_idx):\n{indexed_terms_block}\n\n"
            "Return ONLY one JSON object with key \"is_a_edges\".\n"
            "Each edge item must have: child_idx, parent_idx, justification.\n"
        )

    return (
        f"<s>[INST] <<SYS>>\n{cfg.system_prompt}\n<</SYS>>\n\n"
        f"{user}\n"
        "[/INST]"
    )


# -----------------------------
# LLM call
# -----------------------------
def call_llm(tok, model, device: str, prompt: str, cfg: PromptCfg) -> str:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=(cfg.temperature and cfg.temperature > 0),
            temperature=cfg.temperature if cfg.temperature and cfg.temperature > 0 else None,
            top_p=cfg.top_p,
            pad_token_id=tok.pad_token_id,
        )

    gen_only = out[0, input_ids.shape[-1] :]
    return tok.decode(gen_only, skip_special_tokens=True)


# -----------------------------
# Robust JSON extraction (brace matching): last object
# -----------------------------
def extract_json_object_last(text: str) -> Optional[str]:
    best = None
    stack = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            if stack > 0:
                stack -= 1
                if stack == 0 and start is not None:
                    best = text[start : i + 1]
                    start = None
    return best


# -----------------------------
# Parse + validate (INDEXED)
# -----------------------------
def parse_edges_idx(
    raw_output: str,
    idx_to_term: Dict[int, str],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    Validates:
    - JSON schema key is_a_edges
    - child_idx/parent_idx are valid ints within idx_to_term
    Post-filters:
    - drop self edges
    """
    blob = extract_json_object_last(raw_output)
    if not blob:
        return [], {"error": "no_json_object"}

    data: Any
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(blob)
        except Exception:
            return [], {"error": "json_parse_failed"}

    if not isinstance(data, dict):
        return [], {"error": "not_a_dict"}

    edges = data.get("is_a_edges", [])
    if not isinstance(edges, list):
        return [], {"error": "is_a_edges_not_list"}

    kept: List[Dict[str, str]] = []
    seen = set()

    for e in edges:
        if not isinstance(e, dict):
            continue

        ci = e.get("child_idx")
        pi = e.get("parent_idx")
        just = str(e.get("justification", "")).strip()

        try:
            ci = int(ci)
            pi = int(pi)
        except Exception:
            continue

        if ci not in idx_to_term or pi not in idx_to_term:
            continue

        child = idx_to_term[ci]
        parent = idx_to_term[pi]

        if child.lower() == parent.lower():
            continue

        pair = (child.lower(), parent.lower())
        if pair in seen:
            continue
        seen.add(pair)

        kept.append({"child": child, "parent": parent, "justification": just})

    return kept, {"parsed_json": data}


# -----------------------------
# Main loop
# -----------------------------
def run(
    db: str,
    model_id: str,
    prompt_config: str,
    chunks_table: str,
    doc_id_col: str,
    chunk_id_col: str,
    text_col: str,
    terms_table: str,
    enrich_table: str,
    edges_table: str,
    runs_table: str,
    max_chunks: int,
    offset_rowid: int,
    min_freq: int,
    debug_first: bool,
    commit_every: int,
) -> None:
    cfg = load_prompt_cfg(prompt_config)

    conn = sqlite3.connect(db)
    try:
        init_tables(conn, edges_table=edges_table, runs_table=runs_table)
        tok, model, device = load_model(model_id)

        rows = fetch_chunks(
            conn=conn,
            chunks_table=chunks_table,
            doc_id_col=doc_id_col,
            chunk_id_col=chunk_id_col,
            text_col=text_col,
            runs_table=runs_table,
            offset_rowid=offset_rowid,
            max_chunks=max_chunks,
        )

        print(f"[INFO] Pending chunks for is-a: {len(rows)} (offset_rowid={offset_rowid})")
        if not rows:
            return

        cur = conn.cursor()
        n_since_commit = 0

        for i, (rowid, doc_id, chunk_id, chunk_text) in enumerate(rows, start=1):
            candidates = fetch_candidate_terms(
                conn=conn,
                terms_table=terms_table,
                enrich_table=enrich_table,
                doc_id=doc_id,
                chunk_id=chunk_id,
                min_freq=min_freq,
                max_terms=cfg.max_terms_per_chunk,
            )

            if i == 1 or i % 10 == 0:
                print(
                    f"[INFO] chunk {i}/{len(rows)} rowid={rowid} doc_id={doc_id} "
                    f"chunk_id={chunk_id} taxo_candidates={len(candidates)}"
                )

            if len(candidates) < 2:
                # Mark processed even if no candidates
                if debug_first:
                    print("[DEBUG] <2 taxonomy candidate terms; would mark run with 0 edges.")
                    return
                cur.execute(
                    f"""
                    INSERT OR REPLACE INTO {runs_table}
                      (doc_id, chunk_id, rowid_src, candidate_n, kept_edges_n, raw_output, parsed_json, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                    """,
                    (doc_id, chunk_id, rowid, len(candidates), 0, "", json.dumps({"is_a_edges": []})),
                )
                n_since_commit += 1
                if n_since_commit >= max(1, commit_every):
                    conn.commit()
                    n_since_commit = 0
                continue

            idx_to_term, idx_to_term_id = build_idx_maps(candidates)
            prompt = build_prompt(cfg, chunk_text, candidates)
            raw = call_llm(tok, model, device, prompt, cfg)

            edges, meta = parse_edges_idx(raw, idx_to_term)

            if debug_first:
                print(f"\nDEBUG rowid={rowid} doc_id={doc_id} chunk_id={chunk_id}")
                print("\n=== RAW OUTPUT (first 1200 chars) ===")
                print(raw[:1200])
                print("\n=== KEPT EDGES (after idx validation) ===")
                for e in edges:
                    print(f"- {e['child']} -> {e['parent']} :: {e['justification']}")
                print("\n=== PARSE META ===")
                print(meta)
                return

            # Insert edges
            for e in edges:
                child_id = idx_to_term_id.get(next(k for k, v in idx_to_term.items() if v == e["child"]))
                parent_id = idx_to_term_id.get(next(k for k, v in idx_to_term.items() if v == e["parent"]))

                cur.execute(
                    f"""
                    INSERT OR IGNORE INTO {edges_table}
                      (doc_id, chunk_id, child_term_id, parent_term_id, child_term, parent_term, justification, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        chunk_id,
                        child_id,
                        parent_id,
                        e["child"],
                        e["parent"],
                        e["justification"],
                        extract_json_object_last(raw) or "",
                    ),
                )

            # Mark chunk processed ALWAYS
            cur.execute(
                f"""
                INSERT OR REPLACE INTO {runs_table}
                  (doc_id, chunk_id, rowid_src, candidate_n, kept_edges_n, raw_output, parsed_json, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    doc_id,
                    chunk_id,
                    rowid,
                    len(candidates),
                    len(edges),
                    raw,
                    json.dumps(meta.get("parsed_json", {}), ensure_ascii=False),
                ),
            )

            n_since_commit += 1
            if n_since_commit >= max(1, commit_every):
                conn.commit()
                n_since_commit = 0

        conn.commit()
        print("[DONE] is-a taxonomy extraction complete (INDEXED).")

    finally:
        conn.close()


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="OLAF_LLM: is-a taxonomy extraction (LLM, INDEXED child/parent).")

    ap.add_argument("--db", required=True)
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--prompt-config", default="prompts/taxonomy_llm.yaml")

    ap.add_argument("--chunks-table", default="contextual_chunk")
    ap.add_argument("--doc-id-col", default="doc_id")
    ap.add_argument("--chunk-id-col", default="chunk_id")
    ap.add_argument("--text-col", default="text")

    ap.add_argument("--terms-table", default="llm_terms_final")
    ap.add_argument("--enrich-table", default="llm_enrich_final")

    ap.add_argument("--edges-table", default="llm_is_a_edges_final")
    ap.add_argument("--runs-table", default="llm_is_a_runs")

    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--offset-rowid", type=int, default=0)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--commit-every", type=int, default=25)

    ap.add_argument("--debug-first-chunk", action="store_true")

    args = ap.parse_args()

    run(
        db=args.db,
        model_id=args.model_id,
        prompt_config=args.prompt_config,
        chunks_table=args.chunks_table,
        doc_id_col=args.doc_id_col,
        chunk_id_col=args.chunk_id_col,
        text_col=args.text_col,
        terms_table=args.terms_table,
        enrich_table=args.enrich_table,
        edges_table=args.edges_table,
        runs_table=args.runs_table,
        max_chunks=args.max_chunks,
        offset_rowid=args.offset_rowid,
        min_freq=args.min_freq,
        debug_first=args.debug_first_chunk,
        commit_every=args.commit_every,
    )


if __name__ == "__main__":
    main()
