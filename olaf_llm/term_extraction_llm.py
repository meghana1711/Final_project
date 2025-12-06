import json
import re
from textwrap import shorten
from typing import List, Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = """\
You are an expert in High Performance Computing (HPC) and job schedulers like SLURM and IBM LSF.
Your task is STRICT TERM EXTRACTION.

Given a technical paragraph, extract the most important DOMAIN TERMS.
Focus on:
- scheduler names, job states, partitions, queues
- configuration parameters and options (e.g. Slurm options, LSF keywords)
- CLI commands and flags (sbatch, srun, bsub, --partition, -N, -n)
- resource types (nodes, GPUs, CPUs, memory, cores)
- log file names and config file names

Rules:
- Return ONLY valid JSON. No explanations, no prose.
- JSON format:
  {
    "terms": [
      {"term": "string", "reason": "very short reason"},
      ...
    ]
  }
- Use each unique term only once per paragraph.
- Ignore:
  - pure numbers or timestamps
  - generic English words (job, system, user) unless clearly technical in context
  - random punctuation or broken tokens.
"""

# ---------- MODEL LOADING ----------

def load_mistral(model_id: str = MODEL_ID) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    return tokenizer, model, device

# ---------- PROMPT + GENERATION ----------

def build_messages(text: str, max_terms: int = 30) -> List[Dict[str, str]]:
    user_prompt = (
        f"Extract up to {max_terms} domain terms from the following text.\n"
        f"Text:\n{text}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

def generate_completion(tokenizer, model, device, messages, max_new_tokens: int = 256) -> str:
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# ---------- PARSING ----------

_term_strip_re = re.compile(r"^[\s\.,;:\-\(\)\[\]\{\}]+|[\s\.,;:\-\(\)\[\]\{\}]+$")

def _normalize_term(term: str) -> str:
    term = _term_strip_re.sub("", term)
    term = re.sub(r"\s+", " ", term)
    return term.strip()

def _looks_like_number(term: str) -> bool:
    t = term.replace(".", "").replace(",", "")
    return t.isdigit()

def parse_terms(raw_output: str) -> List[Dict[str, Any]]:
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end == -1:
        return []

    json_str = raw_output[start : end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    items = data.get("terms", [])
    seen = set()
    cleaned = []

    for item in items:
        term = _normalize_term(str(item.get("term", "")))
        reason = str(item.get("reason", "")).strip()
        if not term:
            continue
        if len(term) < 2 or len(term) > 80:
            continue
        if _looks_like_number(term):
            continue

        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"term": term, "reason": reason})

    return cleaned

# ---------- PUBLIC FUNCTION YOU CALL ----------

def extract_terms_from_text(text: str, tokenizer, model, device: str, max_terms: int = 30):
    messages = build_messages(text, max_terms=max_terms)
    raw = generate_completion(tokenizer, model, device, messages)
    return parse_terms(raw)

# ---------- YOUR PIPELINE / MAIN CODE ----------

if __name__ == "__main__":
    # 1) Load model ONCE
    print("Loading Mistral...")
    tokenizer, model, device = load_mistral()

    # 2) Example text (here you can plug your corpus / pipeline)
    text = "Submit SLURM jobs with sbatch. Use the --partition=gpu queue to request GPU nodes."
    
    # 3) Call term extraction function FROM THE SAME FILE
    terms = extract_terms_from_text(text, tokenizer, model, device)

    print(f"\nFound {len(terms)} terms:\n")
    for t in terms:
        print(f"- {t['term']:<30} | {shorten(t['reason'], width=60)}")
