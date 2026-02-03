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
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None  

try:
    from rdflib import Graph  
    _RDFLIB_AVAILABLE = True
except Exception:
    Graph = None  
    _RDFLIB_AVAILABLE = False


# -----------------------
# DEFAULTS
# -----------------------
DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_DB_PATH = Path("onto_new.db")
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_PROMPT_CONFIG = Path("prompts/text2owl_prompt.json")

# You already learned this the hard way :)
DEFAULT_MAX_NEW_TOKENS = 250
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0


# -----------------------
# PROMPT CONFIG
# -----------------------
def load_prompt_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"prompt_config not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if "user_template" not in cfg:
        raise ValueError("prompt_config missing required key: user_template")
    if "system_prompt" not in cfg:
        cfg["system_prompt"] = ""
    if "defaults" not in cfg:
        cfg["defaults"] = {}
    return cfg


def render_template(t: str, mapping: Dict[str, str]) -> str:
    out = t
    for k, v in mapping.items():
        out = out.replace("{" + k + "}", v)
    return out


def build_user_message_from_config(
    cfg: Dict,
    chunk: str,
    base_iri: str,
    prefixes: str,
) -> str:
    defaults = cfg.get("defaults", {}) or {}
    base_iri = base_iri or str(defaults.get("base_iri", "http://example.org/hpc#"))
    prefixes = prefixes or str(defaults.get("prefixes", "hpc:, rdf:, rdfs:, owl:, xsd:"))

    mapping = {
        "chunk": chunk,
        "base_iri": base_iri,
        "prefixes": prefixes,
    }
    return render_template(str(cfg["user_template"]), mapping).strip()


def format_prompt(tokenizer, system_prompt: str, user_msg: str) -> str:
    """
    Prefer chat template if available, otherwise fallback to [INST] format.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": user_msg})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    joined = (system_prompt.strip() + "\n\n" + user_msg).strip() if system_prompt.strip() else user_msg
    return f"[INST]\n{joined}\n[/INST]"


# -----------------------
# CLEANING / SANITIZATION
# -----------------------
PREFIX_LINE_RE = re.compile(r"^\s*@prefix\s+.*\.\s*$", re.IGNORECASE)
CODE_FENCE_RE = re.compile(r"```+", re.MULTILINE)
LEADING_JUNK_RE = re.compile(
    r"^\s*(here( is|'s)?|turtle|output|ontology fragment)\s*[:\-]\s*$",
    re.IGNORECASE,
)

# Fix one common model mistake: "hpc:Foo a owl:Class, ..." -> "hpc:Foo a owl:Class ."
BAD_CLASS_TYPE_RE = re.compile(
    r"^(?P<s>\s*hpc:[A-Za-z0-9_]+)\s+a\s+owl:Class\s*,.*$",
    re.MULTILINE,
)

# Rewrite unknown prefixes like hibm:Something -> hpc:Something (safe fallback)
UNKNOWN_PREFIX_RE = re.compile(
    r"\b(?!hpc:|rdf:|rdfs:|owl:|xsd:)([a-zA-Z][\w\-]*):([A-Za-z0-9_]+)\b"
)

# Fix: "hpc:Foo owl:DatatypeProperty ." -> "hpc:Foo a owl:DatatypeProperty ."
MISSING_A_PROP_RE = re.compile(
    r"^(\s*hpc:[A-Za-z0-9_]+)\s+(owl:(?:ObjectProperty|DatatypeProperty))\s*\.\s*$",
    re.MULTILINE,
)

# Fix: "hpc:isOk 'an owl:ObjectProperty" -> "hpc:isOk a owl:ObjectProperty"
BROKEN_AN_RE = re.compile(
    r"(\bhpc:[A-Za-z0-9_]+\b)\s*['\"]?\s*an\s+(owl:(?:ObjectProperty|DatatypeProperty|Class))",
    re.IGNORECASE,
)


def strip_code_fences(text: str) -> str:
    text = CODE_FENCE_RE.sub("", text or "")
    return text.replace("`", "").strip()


def drop_leading_junk_lines(text: str) -> str:
    lines = (text or "").splitlines()
    out = []
    for ln in lines:
        if LEADING_JUNK_RE.match(ln.strip()):
            continue
        out.append(ln)
    return "\n".join(out).strip()


def keep_turtle_only(text: str) -> str:
    """
    Trim anything before the first @prefix/@base/PREFIX/BASE or first hpc: line.
    """
    if not text:
        return ""
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if (
            s.lower().startswith("@prefix")
            or s.lower().startswith("@base")
            or s.startswith("PREFIX")
            or s.startswith("BASE")
            or s.startswith("hpc:")
        ):
            start_idx = i
            break
    return "\n".join(lines[start_idx:]).strip()


def strip_prefix_declarations(turtle_text: str) -> str:
    out_lines = []
    for line in (turtle_text or "").splitlines():
        if PREFIX_LINE_RE.match(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def fix_bad_class_typing(turtle_text: str) -> str:
    def repl(m: re.Match) -> str:
        return f"{m.group('s')} a owl:Class ."

    return BAD_CLASS_TYPE_RE.sub(repl, turtle_text or "")


def rewrite_unknown_prefixes(turtle_text: str) -> str:
    """
    Example: hibm:Cluster -> hpc:Cluster
    """
    def repl(m: re.Match) -> str:
        pref = m.group(1)
        local = m.group(2)
        if pref.lower() in {"hpc", "rdf", "rdfs", "owl", "xsd"}:
            return m.group(0)
        return f"hpc:{local}"

    return UNKNOWN_PREFIX_RE.sub(repl, turtle_text or "")


def fix_common_ttl_grammar(t: str) -> str:
    t = BROKEN_AN_RE.sub(r"\1 a \2", t or "")
    t = MISSING_A_PROP_RE.sub(r"\1 a \2 .", t)
    return t


def sanitize_turtle(text: str) -> str:
    """
    Robust sanitizer for model outputs:
    - Strip code fences/backticks
    - Drop junk lines
    - Trim to turtle-ish content
    - Strip per-fragment @prefix lines (we inject header ourselves)
    - Rewrite unknown prefixes to hpc:
    - Fix common grammar mistakes
    - Fix common owl:Class typing glitch
    """
    t = text or ""
    t = strip_code_fences(t)
    t = drop_leading_junk_lines(t)
    t = keep_turtle_only(t)
    t = strip_prefix_declarations(t)
    t = rewrite_unknown_prefixes(t)
    t = fix_common_ttl_grammar(t)
    t = fix_bad_class_typing(t)
    return t.strip()


# -----------------------
# NAME CANONICALIZATION
# -----------------------
_HPC_QNAME_RE = re.compile(r"\bhpc:([A-Za-z0-9_]+)\b")


def _to_camel_case(s: str) -> str:
    s2 = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_")
    parts = [p for p in s2.split("_") if p]
    if not parts:
        return s
    return "".join(p[:1].upper() + p[1:].lower() if p else "" for p in parts)


def _to_lower_camel_case(s: str) -> str:
    cc = _to_camel_case(s)
    return cc[:1].lower() + cc[1:] if cc else s


def canonicalize_fragment_names(turtle_text: str) -> str:
    lines = (turtle_text or "").splitlines()
    out: List[str] = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            out.append(line)
            continue

        if " a owl:Class" in s:
            def repl(m): return "hpc:" + _to_camel_case(m.group(1))
            out.append(_HPC_QNAME_RE.sub(repl, line))
            continue

        if " a owl:ObjectProperty" in s or " a owl:DatatypeProperty" in s:
            def repl(m): return "hpc:" + _to_lower_camel_case(m.group(1))
            out.append(_HPC_QNAME_RE.sub(repl, line))
            continue

        if "rdfs:subClassOf" in s or "rdfs:domain" in s or "rdfs:range" in s:
            def repl(m): return "hpc:" + _to_camel_case(m.group(1))
            out.append(_HPC_QNAME_RE.sub(repl, line))
            continue

        m = re.match(r"^\s*hpc:([A-Za-z0-9_]+)\s+hpc:([A-Za-z0-9_]+)\b", line)
        if m:
            subj, pred = m.group(1), m.group(2)
            line2 = re.sub(r"\bhpc:" + re.escape(subj) + r"\b", "hpc:" + _to_camel_case(subj), line, count=1)
            line2 = re.sub(r"\bhpc:" + re.escape(pred) + r"\b", "hpc:" + _to_lower_camel_case(pred), line2, count=1)
            out.append(line2)
            continue

        out.append(line)

    return "\n".join(out).strip()


# -----------------------
# TURTLE VALIDATION
# -----------------------
TTL_HEADER = """@prefix hpc: <http://example.org/hpc#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"""


def validate_turtle_fragment(fragment_text: str) -> Tuple[bool, Optional[str]]:
    if not _RDFLIB_AVAILABLE:
        return False, "rdflib not installed (pip install rdflib) — cannot validate"
    frag = (fragment_text or "").strip()
    if not frag:
        return True, None
    g = Graph()
    try:
        g.parse(data=TTL_HEADER + "\n" + frag + "\n", format="turtle")
        return True, None
    except Exception as e:
        return False, str(e)


# -----------------------
# DB LOADING
# -----------------------
def load_chunks_from_db(db_path: Path) -> List[Tuple[str, str, str]]:
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


def fragment_path_db(fragments_dir: Path, _doc_id: str, chunk_id: str) -> Path:
    return fragments_dir / f"{safe_id(chunk_id)}.ttl"


# -----------------------
# MODEL LOADING
# -----------------------
def _pick_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float16
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def load_model(model_name: str, load_in_4bit: bool) -> Tuple:
    print(f"[INFO] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = dict(
        device_map="auto",
        torch_dtype=_pick_dtype(),
        low_cpu_mem_usage=True,
    )

    if load_in_4bit:
        if BitsAndBytesConfig is None:
            raise RuntimeError(
                "BitsAndBytesConfig not available. Install bitsandbytes "
                "or run without --load-in-4bit."
            )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return tokenizer, model


# -----------------------
# GENERATION
# -----------------------
def generate_chunk(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    do_sample = temperature > 0.0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    gen_ids = out[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, (t1 - t0)


# -----------------------
# REPAIR PROMPT
# -----------------------
def build_repair_user_message(
    base_iri: str,
    invalid_ttl: str,
    error_msg: str,
) -> str:
    return f"""
Repairing INVALID Turtle syntax. Return ONLY corrected Turtle (no markdown, no explanations).

Constraints:
- Use ONLY these prefixes: hpc:, rdf:, rdfs:, owl:, xsd:
- Do NOT invent new prefixes like hibm:
- End every statement with '.' and use valid Turtle punctuation.
- Fix missing 'a' in declarations like: hpc:X owl:DatatypeProperty . -> hpc:X a owl:DatatypeProperty .
- Fix broken "'an" or "an" -> "a" in statements like: hpc:rel 'an owl:ObjectProperty -> hpc:rel a owl:ObjectProperty

Base IRI: {base_iri}

Parser error:
{error_msg}

INVALID TURTLE TO REPAIR:
{invalid_ttl}

Return ONLY corrected Turtle fragment (no @prefix lines).
""".strip()


# -----------------------
# MERGE
# -----------------------
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
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--output-ttl", type=Path, default=None)

    ap.add_argument("--prompt-config", type=Path, default=DEFAULT_PROMPT_CONFIG)
    ap.add_argument("--base-iri", type=str, default="")
    ap.add_argument("--prefixes", type=str, default="")

    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)

    ap.add_argument("--load-in-4bit", action="store_true")

    ap.add_argument("--force", action="store_true")
    ap.add_argument("--merge-only", action="store_true")

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)

    validate_group = ap.add_mutually_exclusive_group()
    validate_group.add_argument("--validate-turtle", action="store_true")
    validate_group.add_argument("--no-validate-turtle", action="store_true")

    canon_group = ap.add_mutually_exclusive_group()
    canon_group.add_argument("--canonicalize-names", action="store_true")
    canon_group.add_argument("--no-canonicalize-names", action="store_true")

    # NEW: repair controls
    ap.add_argument("--repair-invalid", action="store_true", help="Attempt LLM repair on invalid Turtle (recommended).")
    ap.add_argument("--max-repair-attempts", type=int, default=1, help="Max repair attempts per chunk (default: 1).")

    # NEW: resume behavior
    ap.add_argument("--resume", action="store_true", help="Skip if fragment already exists OR invalid already exists (unless --force).")

    args = ap.parse_args()

    output_dir: Path = args.output_dir
    fragments_dir = output_dir / "fragments"
    invalid_dir = output_dir / "fragments_invalid"
    fragments_dir.mkdir(parents=True, exist_ok=True)
    invalid_dir.mkdir(parents=True, exist_ok=True)

    output_ttl = args.output_ttl or (output_dir / "text2owl_db.ttl")

    timings_jsonl = output_dir / "timings.jsonl"
    invalid_jsonl = output_dir / "invalid_fragments.jsonl"
    run_meta_json = output_dir / "run_meta.json"

    if args.merge_only:
        n = merge_fragments(fragments_dir, output_ttl)
        print(f"[INFO] Merged {n} fragments -> {output_ttl}")
        return

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard-id must be in [0, num_shards-1]")

    validate = (not args.no_validate_turtle) and (args.validate_turtle or _RDFLIB_AVAILABLE)
    canonicalize = (not args.no_canonicalize_names)

    if validate and not _RDFLIB_AVAILABLE:
        raise RuntimeError("Turtle validation requested but rdflib is not installed. Run: pip install rdflib")

    cfg = load_prompt_config(args.prompt_config)
    system_prompt = str(cfg.get("system_prompt", "")).strip()

    rows = load_chunks_from_db(args.db_path)
    if not rows:
        print("[ERROR] No rows found in contextual_chunk. Exiting.")
        return

    tokenizer, model = load_model(args.model, args.load_in_4bit)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    meta = {
        "run_id": run_id,
        "utc_start": datetime.utcnow().isoformat() + "Z",
        "model": args.model,
        "db_path": str(args.db_path),
        "output_dir": str(output_dir),
        "output_ttl": str(output_ttl),
        "prompt_config": str(args.prompt_config),
        "base_iri": args.base_iri,
        "prefixes": args.prefixes,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "load_in_4bit": bool(args.load_in_4bit),
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "num_db_rows": len(rows),
        "validate_turtle": bool(validate),
        "canonicalize_names": bool(canonicalize),
        "rdflib_available": bool(_RDFLIB_AVAILABLE),
        "repair_invalid": bool(args.repair_invalid),
        "max_repair_attempts": int(args.max_repair_attempts),
        "resume": bool(args.resume),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    run_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Shard: {args.shard_id}/{args.num_shards}")
    print(f"[INFO] Writing fragments to: {fragments_dir}")
    print(f"[INFO] Quarantining invalid fragments to: {invalid_dir}")
    print(f"[INFO] Canonicalize names: {canonicalize}")
    print(f"[INFO] Validate Turtle: {validate}")
    print(f"[INFO] Repair invalid: {bool(args.repair_invalid)} (max attempts: {args.max_repair_attempts})")
    print(f"[INFO] Resume mode: {bool(args.resume)}")

    total_t0 = time.perf_counter()

    generated_count = 0
    skipped_count = 0
    processed_count = 0
    invalid_count = 0
    repaired_ok = 0
    repaired_fail = 0

    for global_idx, (chunk_id, doc_id, text) in enumerate(rows):
        if (global_idx % args.num_shards) != args.shard_id:
            continue

        processed_count += 1
        frag_path = fragment_path_db(fragments_dir, doc_id, chunk_id)
        bad_path = invalid_dir / frag_path.name

        if args.resume and not args.force:
            if frag_path.exists() or bad_path.exists():
                skipped_count += 1
                continue

        if frag_path.exists() and not args.force:
            skipped_count += 1
            continue

        if not text.strip():
            frag_path.write_text("", encoding="utf-8")
            skipped_count += 1
            continue

        user_msg = build_user_message_from_config(
            cfg,
            chunk=text,
            base_iri=args.base_iri,
            prefixes=args.prefixes,
        )
        prompt = format_prompt(tokenizer, system_prompt=system_prompt, user_msg=user_msg)

        print(f"[INFO] Generating: doc_id={doc_id} chunk_id={chunk_id} -> {frag_path.name}")

        out_text, gen_seconds = generate_chunk(
            tokenizer, model, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        cleaned = sanitize_turtle(out_text)

        if canonicalize and cleaned.strip():
            cleaned = canonicalize_fragment_names(cleaned)

        # Validate (and repair if requested)
        if validate:
            ok, err = validate_turtle_fragment(cleaned)
            if not ok:
                final_ok = False
                final_txt = cleaned
                final_err = err or "unknown parse error"

                if args.repair_invalid and args.max_repair_attempts > 0:
                    base_iri = args.base_iri or "http://example.org/hpc#"
                    for _attempt in range(int(args.max_repair_attempts)):
                        repair_user = build_repair_user_message(
                            base_iri=base_iri,
                            invalid_ttl=final_txt,
                            error_msg=final_err,
                        )
                        repair_prompt = format_prompt(tokenizer, system_prompt="", user_msg=repair_user)

                        repaired_raw, _ = generate_chunk(
                            tokenizer, model, repair_prompt,
                            max_new_tokens=min(300, int(args.max_new_tokens)),
                            temperature=0.0,
                            top_p=1.0,
                        )
                        repaired_clean = sanitize_turtle(repaired_raw)
                        if canonicalize and repaired_clean.strip():
                            repaired_clean = canonicalize_fragment_names(repaired_clean)

                        ok2, err2 = validate_turtle_fragment(repaired_clean)
                        if ok2:
                            final_ok = True
                            final_txt = repaired_clean
                            repaired_ok += 1
                            break
                        else:
                            final_txt = repaired_clean
                            final_err = err2 or final_err

                    if not final_ok:
                        repaired_fail += 1

                if not final_ok:
                    bad_path.write_text(final_txt + "\n", encoding="utf-8")
                    invalid_count += 1
                    append_jsonl(invalid_jsonl, {
                        "run_id": run_id,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "global_index": global_idx,
                        "fragment": bad_path.name,
                        "gen_seconds": gen_seconds,
                        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                        "error": final_err,
                    })
                    frag_path.write_text("", encoding="utf-8")
                    print(f"[WARN] Invalid Turtle -> quarantined: {bad_path.name}")
                    print(f"[WARN] Turtle parse error: {final_err}")
                    continue

                cleaned = final_txt  # repaired OK

        frag_path.write_text(cleaned + "\n", encoding="utf-8")
        generated_count += 1

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

    total_seconds = time.perf_counter() - total_t0
    n_frags = merge_fragments(fragments_dir, output_ttl)

    print("\n[INFO] Summary")
    print(f"[INFO] DB rows assigned to this shard (processed): {processed_count}")
    print(f"[INFO] Generated fragments this run: {generated_count}")
    print(f"[INFO] Skipped existing/empty fragments: {skipped_count}")
    print(f"[INFO] Invalid fragments quarantined: {invalid_count}")
    print(f"[INFO] Repair ok: {repaired_ok} ; repair failed: {repaired_fail}")
    print(f"[INFO] Total fragments present: {n_frags}")
    print(f"[INFO] Wrote merged ontology to: {output_ttl}")
    print(f"[INFO] Total time: {total_seconds:.2f}s ({total_seconds/60:.2f} min)")


if __name__ == "__main__":
    main()
