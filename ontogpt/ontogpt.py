# text2owl.py
#
# OntoGPT-based robust pipeline (SQLite input):
# - Reads chunks from SQLite DB: table contextual_chunk(chunk_id, doc_id, text)
# - Each DB row is treated as one chunk (NO extra chunking)
# - Calls: ontogpt extract -t TEMPLATE -i <tempfile> --output-format turtle -o <fragment>
# - Writes per-chunk fragments immediately + timing JSONL
# - Resume support: skips chunks already generated
# - Optional sharding: split chunks across multiple jobs/GPUs
# - Merge-only mode to create final TTL from fragments
#
# Requires: ontogpt installed and "ontogpt" available on PATH
# Docs: ontogpt extract, output formats include turtle/owl/json/yaml/etc. :contentReference[oaicite:4]{index=4}

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


# -----------------------
# DEFAULTS
# -----------------------
DEFAULT_DB_PATH = Path("onto_new.db")
DEFAULT_OUTPUT_DIR = Path("output")

# OntoGPT settings
DEFAULT_OUTPUT_FORMAT = "turtle"  # one of: html,json,jsonl,md,owl,pickle,turtle,yaml :contentReference[oaicite:5]{index=5}
DEFAULT_TEMPERATURE = 0.0


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

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='contextual_chunk';")
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
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def fragment_path_db(fragments_dir: Path, doc_id: str, chunk_id: str, ext: str) -> Path:
    return fragments_dir / f"doc_{safe_id(doc_id)}_chunk_{safe_id(chunk_id)}.{ext}"


# -----------------------
# MERGE (for turtle output)
# -----------------------
PREFIX_LINE_RE = re.compile(r"^\s*@prefix\s+.*\.\s*$", re.IGNORECASE)

TTL_HEADER = """@prefix hpc: <http://example.org/hpc#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"""

def strip_prefix_declarations(turtle_text: str) -> str:
    out_lines = []
    for line in turtle_text.splitlines():
        if PREFIX_LINE_RE.match(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def merge_fragments_turtle(fragments_dir: Path, output_ttl: Path) -> int:
    frags = sorted(fragments_dir.glob("*.ttl"))
    output_ttl.parent.mkdir(parents=True, exist_ok=True)

    with output_ttl.open("w", encoding="utf-8") as f:
        f.write(TTL_HEADER)
        f.write("\n")
        for p in frags:
            frag = p.read_text(encoding="utf-8", errors="ignore").strip()
            if frag:
                frag = strip_prefix_declarations(frag)
                if frag:
                    f.write(f"# Fragment: {p.name}\n")
                    f.write(frag)
                    f.write("\n\n")
    return len(frags)


# -----------------------
# OntoGPT runner
# -----------------------
def ensure_ontogpt_available() -> str:
    exe = shutil.which("ontogpt")
    if not exe:
        raise RuntimeError(
            "ontogpt command not found on PATH. Install with: pip install ontogpt "
            "(and make sure your venv is activated)."
        )
    return exe


def run_ontogpt_extract(
    ontogpt_exe: str,
    template: str,
    input_txt_path: Path,
    output_path: Path,
    output_format: str,
    model: str | None,
    temperature: float | None,
    cache_db: Path | None,
    api_base: str | None,
    model_provider: str | None,
    extra_args: List[str],
) -> Tuple[float, str, str, int]:
    """
    Calls:
      ontogpt extract -t TEMPLATE -i INPUT --output-format turtle -o OUTPUT
    OntoGPT output-format supports turtle/owl/json/yaml/etc. :contentReference[oaicite:6]{index=6}
    """
    cmd = [
        ontogpt_exe, "extract",
        "--template", template,
        "--inputfile", str(input_txt_path),
        "--output", str(output_path),
        "--output-format", output_format,
    ]

    if model:
        cmd += ["--model", model]   # model option described in OntoGPT docs :contentReference[oaicite:7]{index=7}
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]  # :contentReference[oaicite:8]{index=8}
    if cache_db:
        cmd += ["--cache-db", str(cache_db)]        # :contentReference[oaicite:9]{index=9}
    if api_base:
        cmd += ["--api-base", api_base]             # :contentReference[oaicite:10]{index=10}
    if model_provider:
        cmd += ["--model-provider", model_provider] # :contentReference[oaicite:11]{index=11}

    cmd += extra_args

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, text=True, capture_output=True)
    t1 = time.perf_counter()

    return (t1 - t0), proc.stdout, proc.stderr, proc.returncode


# -----------------------
# MAIN
# -----------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH,
                    help="SQLite DB containing contextual_chunk(chunk_id, doc_id, text)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--output-ttl", type=Path, default=None)

    # OntoGPT requires a template/schema :contentReference[oaicite:12]{index=12}
    ap.add_argument("--template", required=True,
                    help="OntoGPT template name (see: ontogpt list-templates) or path to custom schema YAML")

    ap.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT,
                    help="One of: html,json,jsonl,md,owl,pickle,turtle,yaml")
    ap.add_argument("--model", default=None,
                    help="OntoGPT model name (e.g., gpt-4o or a provider-specific name)")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)

    # Optional OntoGPT cache DB for completions :contentReference[oaicite:13]{index=13}
    ap.add_argument("--cache-db", type=Path, default=None,
                    help="SQLite path for OntoGPT prompt-completion cache")

    # Optional for OpenAI-compatible proxy servers / endpoints :contentReference[oaicite:14]{index=14}
    ap.add_argument("--api-base", default=None, help="Base URL for OpenAI-compatible API")
    ap.add_argument("--model-provider", default=None, help="Provider name (e.g., openai)")

    # Resume + robustness
    ap.add_argument("--force", action="store_true", help="Regenerate even if fragment already exists.")
    ap.add_argument("--merge-only", action="store_true", help="Only merge existing fragments -> final TTL.")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failed chunk instead of continuing.")

    # Sharding
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)

    # Pass-through for any extra OntoGPT flags you want
    ap.add_argument("--ontogpt-extra", nargs=argparse.REMAINDER, default=[],
                    help="Extra args passed to ontogpt (put this last). Example: --ontogpt-extra --no-recurse")

    args = ap.parse_args()

    output_dir: Path = args.output_dir
    fragments_dir = output_dir / "fragments"
    tmp_dir = output_dir / "tmp_inputs"
    errors_dir = output_dir / "errors"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    errors_dir.mkdir(parents=True, exist_ok=True)

    # For turtle merge output
    output_ttl = args.output_ttl or (output_dir / "ontogpt_db.ttl")
    timings_jsonl = output_dir / "timings.jsonl"
    run_meta_json = output_dir / "run_meta.json"

    # Validate shard args
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must be in [0, num_shards-1]")

    if args.merge_only:
        if args.output_format != "turtle":
            print("[WARN] --merge-only currently merges only .ttl fragments (output-format turtle).")
        n = merge_fragments_turtle(fragments_dir, output_ttl)
        print(f"[INFO] Merged {n} fragments -> {output_ttl}")
        return

    ontogpt_exe = ensure_ontogpt_available()

    rows = load_chunks_from_db(args.db_path)
    if not rows:
        print("[ERROR] No rows found in contextual_chunk. Exiting.")
        return

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    meta = {
        "run_id": run_id,
        "utc_start": datetime.utcnow().isoformat() + "Z",
        "db_path": str(args.db_path),
        "output_dir": str(output_dir),
        "template": args.template,
        "output_format": args.output_format,
        "model": args.model,
        "temperature": args.temperature,
        "cache_db": str(args.cache_db) if args.cache_db else None,
        "api_base": args.api_base,
        "model_provider": args.model_provider,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_db_rows": len(rows),
        "ontogpt_extra": args.ontogpt_extra,
    }
    run_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Shard: {args.shard_id}/{args.num_shards}")
    print(f"[INFO] Template: {args.template}")
    print(f"[INFO] Output format: {args.output_format}")
    print(f"[INFO] Writing fragments to: {fragments_dir}")
    print(f"[INFO] DB rows total: {len(rows)}")

    total_t0 = time.perf_counter()

    generated_count = 0
    skipped_count = 0
    failed_count = 0
    processed_count = 0

    # Choose extension by output format
    ext = "ttl" if args.output_format == "turtle" else args.output_format

    for global_idx, (chunk_id, doc_id, text) in enumerate(rows):
        # Sharding decision
        my_turn = (global_idx % args.num_shards) == args.shard_id
        if not my_turn:
            continue
        processed_count += 1

        out_path = fragment_path_db(fragments_dir, doc_id, chunk_id, ext=ext)

        if out_path.exists() and not args.force:
            skipped_count += 1
            continue

        if not text.strip():
            out_path.write_text("", encoding="utf-8")
            skipped_count += 1
            continue

        # Write chunk into a temp file for OntoGPT input
        tmp_in = tmp_dir / f"doc_{safe_id(doc_id)}_chunk_{safe_id(chunk_id)}.txt"
        tmp_in.write_text(text, encoding="utf-8")

        print(f"[INFO] OntoGPT: doc_id={doc_id} chunk_id={chunk_id} -> {out_path.name}")

        seconds, stdout, stderr, rc = run_ontogpt_extract(
            ontogpt_exe=ontogpt_exe,
            template=args.template,
            input_txt_path=tmp_in,
            output_path=out_path,
            output_format=args.output_format,
            model=args.model,
            temperature=args.temperature,
            cache_db=args.cache_db,
            api_base=args.api_base,
            model_provider=args.model_provider,
            extra_args=args.ontogpt_extra,
        )

        append_jsonl(timings_jsonl, {
            "run_id": run_id,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "global_index": global_idx,
            "fragment": out_path.name,
            "seconds": seconds,
            "returncode": rc,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        })

        if rc != 0:
            failed_count += 1
            # Save stderr/stdout for debugging but keep pipeline progress
            err_file = errors_dir / (out_path.name + ".log")
            err_file.write_text(
                "CMD FAILED\n\n"
                f"returncode: {rc}\n\n"
                "STDOUT:\n" + (stdout or "") + "\n\n"
                "STDERR:\n" + (stderr or "") + "\n",
                encoding="utf-8"
            )
            print(f"[WARN] Failed chunk in {seconds:.2f}s. Log: {err_file}")
            if args.fail_fast:
                raise RuntimeError(f"OntoGPT failed for doc_id={doc_id} chunk_id={chunk_id} (rc={rc})")
            continue

        generated_count += 1
        print(f"[INFO] Chunk done in {seconds:.2f}s")

    total_t1 = time.perf_counter()
    total_seconds = total_t1 - total_t0

    # Merge only if turtle (because merge header is turtle-specific)
    if args.output_format == "turtle":
        n_frags = merge_fragments_turtle(fragments_dir, output_ttl)
        print(f"[INFO] Wrote merged Turtle to: {output_ttl} (fragments: {n_frags})")
    else:
        print("[INFO] Skipping merge because output-format != turtle. Use fragments directly.")

    print("\n[INFO] Summary")
    print(f"[INFO] DB rows assigned to this shard (processed): {processed_count}")
    print(f"[INFO] Generated fragments this run: {generated_count}")
    print(f"[INFO] Skipped existing/empty fragments: {skipped_count}")
    print(f"[INFO] Failed chunks: {failed_count}")
    print(f"[INFO] Total time: {total_seconds:.2f} seconds ({total_seconds/60:.2f} minutes)")


if __name__ == "__main__":
    main()
