from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import yaml  # PyYAML


# =============================================================================
# Prompt config (YAML)
# =============================================================================

@dataclass
class PromptConfig:
    system_prompt: str
    prompt_mode: str = "few-shot"  # "few-shot" or "zero-shot"
    max_terms_per_chunk: int = 25
    max_new_tokens: int = 220
    few_shot_examples: List[Dict[str, Any]] = None  # [{text:..., terms:[...]}]

    @staticmethod
    def from_yaml(path: str) -> "PromptConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML prompt config must be a mapping/dict at top level.")

        system_prompt = data.get("system_prompt")
        if not system_prompt or not isinstance(system_prompt, str):
            raise ValueError("YAML prompt config must contain a string key: system_prompt")

        prompt_mode = data.get("prompt_mode", "few-shot")
        if prompt_mode not in ("few-shot", "zero-shot"):
            raise ValueError("prompt_mode must be 'few-shot' or 'zero-shot'")

        few_shot_examples = data.get("few_shot_examples", []) or []
        if not isinstance(few_shot_examples, list):
            raise ValueError("few_shot_examples must be a list (or omitted).")

        cleaned_examples: List[Dict[str, Any]] = []
        for ex in few_shot_examples:
            if not isinstance(ex, dict):
                continue
            txt = ex.get("text", "")
            terms = ex.get("terms", [])
            if isinstance(txt, str) and isinstance(terms, list):
                cleaned_examples.append({"text": txt, "terms": terms})

        return PromptConfig(
            system_prompt=system_prompt,
            prompt_mode=prompt_mode,
            max_terms_per_chunk=int(data.get("max_terms_per_chunk", 25)),
            max_new_tokens=int(data.get("max_new_tokens", 220)),
            few_shot_examples=cleaned_examples,
        )


# =============================================================================
# Stopwords
# =============================================================================

def load_stopwords(path: Optional[str]) -> set[str]:
    if not path:
        return set()
    sw: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            sw.add(s.lower())
    return sw


def is_generic_term(term: str, stopwords: set[str]) -> bool:
    if not stopwords:
        return False
    t = term.strip().lower()
    if not t:
        return True
    if t in stopwords:
        return True
    tokens = [tok for tok in re.split(r"\s+", t) if tok]
    if len(tokens) >= 2 and all(tok in stopwords for tok in tokens):
        return True
    return False


# =============================================================================
# Cleaning rules
# =============================================================================

def is_numeric_heavy(term: str) -> bool:
    t = term.strip()
    if not t:
        return True

    digits = sum(ch.isdigit() for ch in t)
    letters = sum(ch.isalpha() for ch in t)

    if digits == 0:
        return False

    if digits > letters:
        return True

    if re.match(r"^\d", t):
        return True

    if re.search(r"\b\d+(\.\d+)?\s*(sec|secs|seconds|min|mins|minutes|hour|hours|day|days|gb|gib|mb|mib|kb|kib|ms|us)\b", t, re.I):
        return True

    return False


def is_punctuation_heavy(term: str, threshold: float = 0.50) -> bool:
    t = term.strip()
    if not t:
        return True
    punct = sum(1 for ch in t if not ch.isalnum() and not ch.isspace())
    total = len(t)
    return (punct / max(total, 1)) >= threshold


# =============================================================================
# Artifact suppression (paths, APIs, binaries, build outputs)
# =============================================================================

# Common “implementation artifact” file extensions (keep this conservative)
_ARTIFACT_EXTS = {
    ".so", ".a", ".o", ".lo", ".la", ".lai", ".dll", ".exe", ".dylib",
    ".jar", ".class", ".pyc", ".pyo", ".whl",
}

# Common path-ish prefixes found in docs/logs
_PATH_PREFIXES = (
    "/", "./", "../", "~", r"\\",  # unix, relative, home, windows UNC
)

# Windows drive letter path
_WIN_DRIVE_RE = re.compile(r"^[A-Za-z]:\\")
# URL
_URL_RE = re.compile(r"^(https?://|ftp://)", re.I)

# Version-like prefix (v0.0.43 etc.)
_VERSION_PREFIX_RE = re.compile(r"^v\d+(\.\d+){1,4}", re.I)

# OpenAPI-ish tokens and suffixes you mentioned
_OPENAPI_RE = re.compile(r"\bopenapi\b", re.I)
_OPENAPI_SUFFIX_RE = re.compile(r"_(resp|request|response|msg|desc|body|params?)\b", re.I)

# Slurm OpenAPI style you showed: v0.0.43_openapi_kill_job_resp, slurm/v0.0.43, etc.
_OPENAPI_VERSIONED_ID_RE = re.compile(
    r"^v\d+(\.\d+){1,4}[_/].*(openapi|swagger).*$", re.I
)

# CamelCase+digits API method names like slurmV0043DeleteJobs
_SLURM_VERB_API_RE = re.compile(r"^[A-Za-z]+V\d{3,}[A-Za-z].+$")

# Assignment-like config strings: TaskPlugin=task/cgroup, etc.
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*.+$")

# “Identifier-y” tokens with lots of separators, typical of generated symbols
_SYMBOLY_RE = re.compile(r"^[A-Za-z0-9]+([_/.-][A-Za-z0-9]+){2,}$")

# GPU / accelerators that are *domain concepts* (exceptions)
_GPU_ALLOW = {"A100", "H100", "V100", "L4", "P100", "K80", "T4"}
_GPU_ALLOW_RE = re.compile(r"^(A|H|V|P)\d{3}$")


def _has_artifact_extension(t: str) -> bool:
    # only consider last path segment
    seg = t.split("/")[-1].split("\\")[-1]
    seg_lower = seg.lower()
    return any(seg_lower.endswith(ext) for ext in _ARTIFACT_EXTS)


def _looks_like_path(t: str) -> bool:
    s = t.strip()
    if not s:
        return False
    if s.startswith(_PATH_PREFIXES):
        return True
    if _WIN_DRIVE_RE.match(s):
        return True
    if "\\" in s or "/" in s:
        # avoid killing normal “task/cgroup” single slash concepts by checking path-ish density
        parts = [p for p in re.split(r"[\\/]+", s) if p]
        if len(parts) >= 3:
            return True
        # if it has an artifact extension, treat as path/file even with 1-2 separators
        if _has_artifact_extension(s):
            return True
    return False


def _looks_like_url(t: str) -> bool:
    return bool(_URL_RE.match(t.strip()))


def _looks_like_openapi_generated(t: str) -> bool:
    s = t.strip()
    if not s:
        return False

    # v0.0.43_openapi_... or v0.0.43/...openapi...
    if _OPENAPI_VERSIONED_ID_RE.match(s):
        return True

    # contains openapi + typical generated suffixes
    if _OPENAPI_RE.search(s) and (_OPENAPI_SUFFIX_RE.search(s) or _VERSION_PREFIX_RE.match(s)):
        return True

    return False


def _looks_like_api_method_name(t: str) -> bool:
    s = t.strip()
    if not s:
        return False
    # slurmV0043DeleteJobs style
    if _SLURM_VERB_API_RE.match(s):
        return True
    return False


def _looks_like_build_or_binary_artifact(t: str) -> bool:
    s = t.strip()
    if not s:
        return False
    if _has_artifact_extension(s):
        return True
    # common build outputs: *.o, *.so already handled; also "libsomething.so" etc.
    return False


def _looks_like_assignment_string(t: str) -> bool:
    # config assignments are often too specific; we usually want the RHS token(s), not key=value blob
    s = t.strip()
    if not s:
        return False
    return bool(_ASSIGNMENT_RE.match(s))


def _looks_like_symbolic_generated_identifier(t: str) -> bool:
    s = t.strip()
    if not s:
        return False
    # many separators -> likely symbol / generated name
    if _SYMBOLY_RE.match(s) and (len(re.findall(r"[_/.-]", s)) >= 2):
        # exception: keep short domain-ish like "task/cgroup" (one slash), "gres/gpu" etc.
        parts = re.split(r"[_/.-]+", s)
        if len(parts) <= 2:
            return False
        return True
    return False


def is_artifact_term(term: str) -> bool:
    """
    Returns True if term is likely an implementation artifact:
    - file paths and URLs
    - binaries/build outputs (.so, .o, etc.)
    - generated OpenAPI response/request identifiers
    - versioned API method names (slurmV0043DeleteJobs)
    - key=value assignment blobs
    - highly symbolic generated identifiers
    """
    s = (term or "").strip()
    if not s:
        return True

    # Allow-list: keep GPU model tokens as legitimate domain terms
    if s in _GPU_ALLOW or _GPU_ALLOW_RE.match(s):
        return False

    # URLs / paths / binaries
    if _looks_like_url(s):
        return True
    if _looks_like_path(s):
        return True
    if _looks_like_build_or_binary_artifact(s):
        return True

    # OpenAPI / swagger generated identifiers
    if _looks_like_openapi_generated(s):
        return True

    # API method names like slurmV0043DeleteJobs
    if _looks_like_api_method_name(s):
        return True

    # key=value config blobs (often too specific + noisy)
    if _looks_like_assignment_string(s):
        return True

    # very “symbolic” long identifiers
    if _looks_like_symbolic_generated_identifier(s):
        return True

    return False


def clean_term(term: str, stopwords: set[str], seen: set[str], suppress_artifacts: bool = True) -> Optional[str]:
    term = (term or "").strip()
    if not term:
        return None

    # Drop if it is clearly an implementation artifact (paths, openapi ids, binaries, etc.)
    if suppress_artifacts and is_artifact_term(term):
        return None

    if not any(ch.isalpha() for ch in term):
        return None

    letters_only = "".join(ch for ch in term if ch.isalpha())
    if len(letters_only) < 3:
        return None

    if is_numeric_heavy(term):
        return None

    if is_punctuation_heavy(term, threshold=0.50):
        return None

    if is_generic_term(term, stopwords):
        return None

    k = term.lower()
    if k in seen:
        return None
    seen.add(k)

    return term


# =============================================================================
# Parse model output
# =============================================================================

def parse_terms(raw_output: str, stopwords: set[str], suppress_artifacts: bool = True) -> List[str]:
    candidates: List[str] = []
    search_pos = 0
    key = '"terms"'

    while True:
        idx = raw_output.find(key, search_pos)
        if idx == -1:
            break

        start = raw_output.rfind("{", 0, idx)
        if start == -1:
            search_pos = idx + len(key)
            continue

        depth = 0
        end = None
        for i, ch in enumerate(raw_output[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            break

        candidates.append(raw_output[start:end])
        search_pos = end

    seen: set[str] = set()

    if candidates:
        for json_str in reversed(candidates):
            data: Any = None
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    data = ast.literal_eval(json_str)
                except Exception:
                    data = None

            if not isinstance(data, dict):
                continue

            items = data.get("terms", [])
            out: List[str] = []

            for item in items:
                if isinstance(item, str):
                    cand = item
                elif isinstance(item, dict):
                    cand = str(item.get("term", ""))
                else:
                    continue

                cleaned = clean_term(cand, stopwords, seen, suppress_artifacts=suppress_artifacts)
                if cleaned is not None:
                    out.append(cleaned)

            return out

    idx = raw_output.find('"terms"')
    if idx == -1:
        return []

    sub = raw_output[idx:]
    matches = re.findall(r'"([^"]+)"', sub)
    if not matches:
        return []

    values = matches[1:] if matches and matches[0] == "terms" else matches

    out: List[str] = []
    for cand in values:
        cleaned = clean_term(cand, stopwords, seen, suppress_artifacts=suppress_artifacts)
        if cleaned is not None:
            out.append(cleaned)
    return out


# =============================================================================
# Model loading + prompt building
# =============================================================================

def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model, device


def build_prefix(cfg: PromptConfig) -> str:
    if cfg.prompt_mode == "zero-shot":
        return f"<s>[INST] <<SYS>>\n{cfg.system_prompt}\n<</SYS>>\n\n"

    examples_str = ""
    for i, ex in enumerate(cfg.few_shot_examples or [], start=1):
        ex_text = ex.get("text", "")
        ex_terms = ex.get("terms", [])
        ex_json = json.dumps({"terms": ex_terms}, indent=2)
        examples_str += (
            f"Example {i}:\n"
            f"Text:\n{ex_text}\n"
            f"Valid JSON:\n{ex_json}\n\n"
        )

    return (
        f"<s>[INST] <<SYS>>\n{cfg.system_prompt}\n<</SYS>>\n\n"
        "You will see examples of HPC documentation and the JSON terms extracted from them.\n"
        "Follow the same behaviour for the NEW text.\n\n"
        f"{examples_str}"
        "Now process ONLY the following NEW text.\n"
    )


def build_prompt(prefix: str, cfg: PromptConfig, text: str) -> str:
    user_instructions = (
        f"Extract up to {cfg.max_terms_per_chunk} important HPC domain terms.\n"
        'Return ONLY one JSON object of the form { "terms": ["term1", "term2", ...] } and nothing else.\n\n'
        f"Text:\n{text}\n"
    )
    return f"{prefix}{user_instructions}[/INST]"


def call_llm(tokenizer, model, device: str, prompt: str, max_new_tokens: int) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_only_ids = generated_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_only_ids, skip_special_tokens=True)


# =============================================================================
# SINGLE FINAL TERMS TABLE
# =============================================================================

def init_terms_table(conn: sqlite3.Connection, table: str) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            term_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            term        TEXT    NOT NULL,
            doc_id      TEXT    NOT NULL,
            chunk_id    TEXT    NOT NULL,
            freq_total  INTEGER NOT NULL DEFAULT 1,
            UNIQUE(term)
        )
        """
    )
    conn.commit()


def upsert_term(conn: sqlite3.Connection, table: str, term: str, doc_id: str, chunk_id: str) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT INTO {table} (term, doc_id, chunk_id, freq_total)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(term) DO UPDATE SET
            freq_total = {table}.freq_total + 1
        """,
        (term, doc_id, chunk_id),
    )


# =============================================================================
# Runs table (resume-safe using doc_id + chunk_id)
# =============================================================================

def init_runs_table(conn: sqlite3.Connection, runs_table: str) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {runs_table} (
            run_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc    TEXT NOT NULL,
            doc_id    TEXT NOT NULL,
            chunk_id  TEXT NOT NULL,
            rowid     INTEGER,
            model_id  TEXT,
            raw_output TEXT,
            UNIQUE(doc_id, chunk_id)
        )
        """
    )
    conn.commit()


def mark_chunk_done(
    conn: sqlite3.Connection,
    runs_table: str,
    doc_id: str,
    chunk_id: str,
    rowid: int,
    model_id: str,
    raw_output: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT OR IGNORE INTO {runs_table} (ts_utc, doc_id, chunk_id, rowid, model_id, raw_output)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (datetime.utcnow().isoformat(timespec="seconds") + "Z", doc_id, chunk_id, rowid, model_id, raw_output),
    )


def fetch_chunks(
    conn: sqlite3.Connection,
    input_table: str,
    doc_id_col: str,
    chunk_id_col: str,
    text_col: str,
    max_chunks: Optional[int],
    offset_rowid: int,
    runs_table: Optional[str],
) -> List[Tuple[int, str, str, str]]:
    cur = conn.cursor()

    sql = f"""
        SELECT rowid, {chunk_id_col}, {doc_id_col}, {text_col}
        FROM {input_table}
        WHERE rowid > ?
    """
    params: List[Any] = [offset_rowid]

    if runs_table:
        sql += f"""
          AND NOT EXISTS (
              SELECT 1 FROM {runs_table} r
              WHERE r.doc_id = {input_table}.{doc_id_col}
                AND r.chunk_id = {input_table}.{chunk_id_col}
          )
        """

    sql += " ORDER BY rowid"
    if max_chunks is not None:
        sql += " LIMIT ?"
        params.append(max_chunks)

    cur.execute(sql, params)
    return cur.fetchall()


# =============================================================================
# Thresholding
# =============================================================================

def apply_frequency_threshold(conn: sqlite3.Connection, table: str, min_freq: int) -> int:
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE freq_total < ?", (min_freq,))
    to_delete = cur.fetchone()[0]
    cur.execute(f"DELETE FROM {table} WHERE freq_total < ?", (min_freq,))
    conn.commit()
    return int(to_delete)


# =============================================================================
# Main
# =============================================================================

def process_chunks(
    conn: sqlite3.Connection,
    tokenizer,
    model,
    device: str,
    cfg: PromptConfig,
    model_id: str,
    input_table: str,
    doc_id_col: str,
    chunk_id_col: str,
    text_col: str,
    terms_table: str,
    runs_table: Optional[str],
    stopwords: set[str],
    max_chunks: Optional[int],
    offset_rowid: int,
    debug_first_chunk: bool,
    suppress_artifacts: bool,
) -> None:
    init_terms_table(conn, terms_table)
    if runs_table:
        init_runs_table(conn, runs_table)

    prefix = build_prefix(cfg)

    rows = fetch_chunks(
        conn=conn,
        input_table=input_table,
        doc_id_col=doc_id_col,
        chunk_id_col=chunk_id_col,
        text_col=text_col,
        max_chunks=(1 if debug_first_chunk else max_chunks),
        offset_rowid=offset_rowid,
        runs_table=runs_table,
    )

    total = len(rows)
    print(f"Processing {total} chunks from {input_table} (offset_rowid={offset_rowid})...")
    if total == 0:
        return

    for i, (rowid, chunk_id, doc_id, chunk_text) in enumerate(rows, start=1):
        print(f"  -> chunk {i}/{total} (rowid={rowid}, doc_id={doc_id}, chunk_id={chunk_id})")

        prompt = build_prompt(prefix, cfg, chunk_text)
        raw = call_llm(tokenizer, model, device, prompt, max_new_tokens=cfg.max_new_tokens)

        terms = parse_terms(raw, stopwords=stopwords, suppress_artifacts=suppress_artifacts)

        if debug_first_chunk:
            print("\n=== RAW OUTPUT (first 900 chars) ===")
            print(raw[:900])
            print("\n=== PARSED+CLEANED TERMS ===")
            for t in terms:
                print("-", t)
            if not terms:
                print("(No terms parsed)")
            return

        for t in terms:
            upsert_term(conn, terms_table, t, doc_id, chunk_id)

        if runs_table:
            mark_chunk_done(conn, runs_table, doc_id, chunk_id, rowid, model_id, raw)

        conn.commit()

    print("Done.")


def main():
    ap = argparse.ArgumentParser(description="LLM term extraction (YAML config) into a single final table with freq_total.")

    ap.add_argument("--db", required=True, help="Path to SQLite DB.")
    ap.add_argument("--prompt-config", default="prompts/term_extract_llm.yaml")
    ap.add_argument("--model-id", default="mistralai/Mistral-7B-Instruct-v0.3", help="HF model id.")

    ap.add_argument("--input-table", default="contextual_chunk", help="Input chunk table name.")
    ap.add_argument("--doc-id-col", default="doc_id", help="Doc id column name.")
    ap.add_argument("--chunk-id-col", default="chunk_id", help="Chunk id column name.")
    ap.add_argument("--text-col", default="text", help="Chunk text column name.")

    ap.add_argument("--terms-table", default="llm_terms_final", help="Final terms table name.")
    ap.add_argument("--runs-table", default="llm_term_runs", help="Runs table for resume safety.")
    ap.add_argument("--no-runs-table", action="store_true", help="Disable runs table (not recommended).")

    ap.add_argument("--stop-words", default="stop_word/stop_words.txt", help="Path to stop_words.txt")

    ap.add_argument("--max-chunks", type=int, default=None, help="Limit chunks processed this run.")
    ap.add_argument("--offset-rowid", type=int, default=0, help="Start from rowid > offset-rowid.")
    ap.add_argument("--debug-first-chunk", action="store_true", help="Print raw+parsed; no DB writes.")

    ap.add_argument("--min-freq", type=int, default=None,
                    help="After extraction, delete terms with freq_total < min-freq (e.g., 2).")

    # NEW: artifact suppression toggle
    ap.add_argument(
        "--no-artifact-suppression",
        action="store_true",
        help="Disable artifact suppression (paths/OpenAPI/binaries/config blobs).",
    )

    args = ap.parse_args()

    cfg = PromptConfig.from_yaml(args.prompt_config)
    stopwords = load_stopwords(args.stop_words)
    print(f"[INFO] stopwords loaded: {len(stopwords)}")

    runs_table = None if args.no_runs_table else args.runs_table
    suppress_artifacts = not args.no_artifact_suppression
    print(f"[INFO] artifact suppression: {'ON' if suppress_artifacts else 'OFF'}")

    conn = sqlite3.connect(args.db)
    try:
        tokenizer, model, device = load_model(args.model_id)

        process_chunks(
            conn=conn,
            tokenizer=tokenizer,
            model=model,
            device=device,
            cfg=cfg,
            model_id=args.model_id,
            input_table=args.input_table,
            doc_id_col=args.doc_id_col,
            chunk_id_col=args.chunk_id_col,
            text_col=args.text_col,
            terms_table=args.terms_table,
            runs_table=runs_table,
            stopwords=stopwords,
            max_chunks=args.max_chunks,
            offset_rowid=args.offset_rowid,
            debug_first_chunk=args.debug_first_chunk,
            suppress_artifacts=suppress_artifacts,
        )

        if args.min_freq is not None:
            deleted = apply_frequency_threshold(conn, args.terms_table, args.min_freq)
            print(f"[THRESH] deleted {deleted} terms with freq_total < {args.min_freq}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
