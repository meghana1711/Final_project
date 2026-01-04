# text2owl.py
#
# Robust Text2OWL pipeline (SQLite input):
# - Reads chunks from SQLite DB: olaf_sample_llm.db, table contextual_chunk(chunk_id, doc_id, text)
# - Each DB row is treated as one chunk (NO extra chunking)
# - Writes per-chunk fragments to disk immediately (no losing long runs)
# - Resume support: skips chunks already generated
# - Optional sharding: split chunks across multiple jobs/GPUs
# - Merge-only mode to create final TTL from fragments
# - Logs timing per chunk + total runtime
#
# Usage examples:
#   python text2owl.text2owl --db-path olaf_sample_llm.db --output
#   python text2owl_mistral_db.py --db-path olaf_sample_llm.db --num-shards 4 --shard-id 0
#   python text2owl_mistral_db.py --output-dir output --merge-only

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    # Optional, only needed if you use --load-in-4bit
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  # type: ignore


# -----------------------
# DEFAULTS
# -----------------------
DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_DB_PATH = Path("onto_new.db")
DEFAULT_OUTPUT_DIR = Path("output")

DEFAULT_MAX_NEW_TOKENS = 800
DEFAULT_TEMPERATURE = 0.0  # deterministic is usually best for OWL
DEFAULT_TOP_P = 


# -----------------------
# PROMPT
# -----------------------
def build_text2owl_user_message(chunk: str) -> str:
    # Keep instructions tight; prefer stable output over creativity.
    # We provide prefixes in the merged header, so the model can omit them.
    return f"""
You are an ontology engineering assistant.

From the following domain-specific documentation chunk, generate an ontology fragment
in OWL using Turtle syntax.

TEXT:
{chunk}

INSTRUCTIONS:
- Use this base IRI: http://example.org/hpc#
- Use these prefixes (assume they are already declared): hpc:, rdf:, rdfs:, owl:, xsd:
- Output ONLY valid Turtle statements (no explanations, no markdown, no code fences).
- Declare classes with: hpc:ClassName a owl:Class .
- Declare object properties with: hpc:propName a owl:ObjectProperty .
- Declare datatype properties with: hpc:propName a owl:DatatypeProperty .
- Use rdfs:subClassOf for subclass relations.
- Use rdfs:domain and rdfs:range for properties.
- Use concise CamelCase for classes and lowerCamelCase for properties.
- If the text is not useful, output nothing (empty output is acceptable).

OUTPUT:
Return ONLY Turtle triples/statements.
""".strip()


def format_prompt(tokenizer, chunk: str) -> str:
    """
    Prefer chat template if available, otherwise fallback to [INST] format.
    """
    user_msg = build_text2owl_user_message(chunk)

    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback for older tokenizers
    return f"[INST]\n{user_msg}\n[/INST]"


# -----------------------
# CLEANING
# -----------------------
PREFIX_LINE_RE = re.compile(r"^\s*@prefix\s+.*\.\s*$", re.IGNORECASE)

def keep_turtle_only(text: str) -> str:
    """
    Heuristic cleanup:
    - remove leading junk
    - keep from first line that looks like Turtle (@prefix or hpc:)
    """
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("@prefix") or s.startswith("hpc:"):
            start_idx = i
            break
    cleaned = "\n".join(lines[start_idx:]).strip()
    return cleaned


def strip_prefix_declarations(turtle_text: str) -> str:
    """
    Remove @prefix lines from model output since we add a global header when merging.
    """
    out_lines = []
    for line in turtle_text.splitlines():
        if PREFIX_LINE_RE.match(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


# -----------------------
# DB LOADING
# -----------------------
def load_chunks_from_db(db_path: Path) -> List[Tuple[str, str, str]]:
    """
    Load chunks from SQLite DB.

    Returns list of (chunk_id, doc_id, text)
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Ensure table exists early with a helpful message
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='contextual_chunk';"
    )
    if cur.fetchone() is None:
        conn.close()
        raise RuntimeError(
            "Table 'contextual_chunk' not found in DB. "
            "Expected schema: contextual_chunk(chunk_id, doc_id, text)"
        )

    cur.execute(
        """
        SELECT chunk_id, doc_id, text
        FROM contextual_chunk
        ORDER BY doc_id, chunk_id
        """
    )
    rows = cur.fetchall()
    conn.close()

    # Normalize types to str for filenames/logging
    norm_rows: List[Tuple[str, str, str]] = []
    for chunk_id, doc_id, text in rows:
        norm_rows.append((str(chunk_id), str(doc_id), text if text is not None else ""))

    print(f"[INFO] Loaded {len(norm_rows)} chunks from DB: {db_path}")
    return norm_rows


# -----------------------
# IO HELPERS
# -----------------------
def append_jsonl(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def safe_id(s: str) -> str:
    # Make a filesystem-safe id string
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def fragment_path_db(fragments_dir: Path, doc_id: str, chunk_id: str) -> Path:
    # e.g. doc_12_chunk_000345.ttl or doc_foo_chunk_bar.ttl
    return fragments_dir / f"doc_{safe_id(doc_id)}_chunk_{safe_id(chunk_id)}.ttl"


# -----------------------
# MODEL LOADING
# -----------------------
def load_model(
    model_name: str,
    load_in_4bit: bool,
) -> Tuple:
    print(f"[INFO] Loading model: {model_name}")
    device_map = "auto"

    # bfloat16 is great on supported GPUs; fallback to float16 otherwise
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model_kwargs = dict(
        device_map=device_map,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError(
                "BitsAndBytesConfig not available. Install bitsandbytes and a compatible transformers build "
                "or run without --load-in-4bit."
            )
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["quantization_config"] = qconf

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


# -----------------------
# GENERATION
# -----------------------
def generate_chunk(
    tokenizer,
    model,
    chunk: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, float]:
    prompt = format_prompt(tokenizer, chunk)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0

    # Better timing accuracy on GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Decode ONLY generated tokens (not the prompt)
    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, (t1 - t0)


# -----------------------
# MERGE
# -----------------------
TTL_HEADER = """@prefix hpc: <http://example.org/hpc#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"""

def merge_fragments(fragments_dir: Path, output_ttl: Path) -> int:
    frags = sorted(fragments_dir.glob("*.ttl"))
    output_ttl.parent.mkdir(parents=True, exist_ok=True)

    with output_ttl.open("w", encoding="utf-8") as f:
        f.write(TTL_HEADER)
        f.write("\n")
        for p in frags:
            frag = p.read_text(encoding="utf-8", errors="ignore").strip()
            if frag:
                f.write(f"# Fragment: {p.name}\n")
                f.write(frag)
                f.write("\n\n")
    return len(frags)


# -----------------------
# MAIN
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME)
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH, help="SQLite DB containing contextual_chunk table")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--output-ttl", type=Path, default=None)

    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)

    ap.add_argument("--load-in-4bit", action="store_true", help="Use 4-bit quantization (requires bitsandbytes).")

    # Resume behavior:
    ap.add_argument("--force", action="store_true", help="Regenerate even if fragment already exists.")
    ap.add_argument("--merge-only", action="store_true", help="Only merge existing fragments -> final TTL.")

    # Sharding: split workload across multiple jobs
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)

    args = ap.parse_args()

    output_dir: Path = args.output_dir
    fragments_dir = output_dir / "fragments"
    fragments_dir.mkdir(parents=True, exist_ok=True)

    output_ttl = args.output_ttl or (output_dir / "text2owl_db.ttl")

    timings_jsonl = output_dir / "timings.jsonl"
    run_meta_json = output_dir / "run_meta.json"

    if args.merge_only:
        n = merge_fragments(fragments_dir, output_ttl)
        print(f"[INFO] Merged {n} fragments -> {output_ttl}")
        return

    # Validate shard args
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must be in [0, num_shards-1]")

    # Load DB rows (each row is already a chunk)
    rows = load_chunks_from_db(args.db_path)
    if not rows:
        print("[ERROR] No rows found in contextual_chunk. Exiting.")
        return

    tokenizer, model = load_model(args.model, args.load_in_4bit)

    # Run metadata
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    meta = {
        "run_id": run_id,
        "utc_start": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "db_path": str(args.db_path),
        "output_dir": str(output_dir),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "load_in_4bit": bool(args.load_in_4bit),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_db_rows": len(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    run_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Shard: {args.shard_id}/{args.num_shards}")
    print(f"[INFO] Writing fragments to: {fragments_dir}")
    print(f"[INFO] DB rows total: {len(rows)}")

    total_t0 = time.perf_counter()

    # Sharding index over DB rows
    generated_count = 0
    skipped_count = 0
    processed_count = 0

    for global_idx, (chunk_id, doc_id, text) in enumerate(rows):
        # Sharding decision based on global index
        my_turn = (global_idx % args.num_shards) == args.shard_id
        if not my_turn:
            continue

        processed_count += 1
        frag_path = fragment_path_db(fragments_dir, doc_id, chunk_id)

        if frag_path.exists() and not args.force:
            skipped_count += 1
            continue

        # Empty chunks: skip quickly
        if not text.strip():
            frag_path.write_text("", encoding="utf-8")
            skipped_count += 1
            continue

        print(f"[INFO] Generating: doc_id={doc_id} chunk_id={chunk_id} -> {frag_path.name}")

        out_text, gen_seconds = generate_chunk(
            tokenizer, model, text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        cleaned = keep_turtle_only(out_text)
        cleaned = strip_prefix_declarations(cleaned)

        # Write immediately so progress is never lost
        frag_path.write_text(cleaned + "\n", encoding="utf-8")
        generated_count += 1

        # Log timing immediately
        append_jsonl(timings_jsonl, {
            "run_id": run_id,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "global_index": global_idx,
            "fragment": frag_path.name,
            "gen_seconds": gen_seconds,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        })

        print(f"[INFO] Chunk done in {gen_seconds:.2f}s")

    total_t1 = time.perf_counter()
    total_seconds = total_t1 - total_t0

    # Merge everything we currently have (from all shards, if any)
    n_frags = merge_fragments(fragments_dir, output_ttl)

    print("\n[INFO] Summary")
    print(f"[INFO] DB rows assigned to this shard (processed): {processed_count}")
    print(f"[INFO] Generated fragments this run: {generated_count}")
    print(f"[INFO] Skipped existing/empty fragments: {skipped_count}")
    print(f"[INFO] Total fragments present: {n_frags}")
    print(f"[INFO] Wrote merged ontology to: {output_ttl}")
    print(f"[INFO] Total time: {total_seconds:.2f} seconds ({total_seconds/60:.2f} minutes)")


if __name__ == "__main__":
    main()
