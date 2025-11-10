# make_fred_chunks.py
# Create 3–5 sentence context chunks from sentence-level JSON for FRED.

import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None  # token counting will fall back to whitespace split

def _count_tokens(text: str) -> int:
    if _NLP is not None:
        return len(_NLP.make_doc(text))
    # lightweight fallback if spaCy isn't available:
    return sum(len(p.split()) for p in text.splitlines())

def _is_header_like(s: str) -> bool:
    """Avoid starting/ending chunks on obvious headers."""
    s = s.strip()
    if not s:
        return True
    if s.endswith(":") and len(s.split()) <= 8:
        return True
    if s.lower().startswith(("figure", "table", "example", "section", "chapter")) and len(s.split()) <= 4:
        return True
    return False

def load_sentences(path: str) -> List[Dict[str, Any]]:
    """
    Expect items with (at least):
      - doc_id
      - sent_id
      - sentence
      - start_char (optional but used for ordering if present)
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # normalize keys
    for d in data:
        if "text" in d and "sentence" not in d:
            d["sentence"] = d["text"]
    return data

def group_by_doc(sentences: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_doc = defaultdict(list)
    for s in sentences:
        by_doc[s["doc_id"]].append(s)
    # stable order by start_char if present, else by sent_id numeric suffix if any
    for doc_id, arr in by_doc.items():
        def _order_key(x):
            if "start_char" in x:
                return (0, x["start_char"])
            # try parse trailing digits in sent_id
            sid = x.get("sent_id", "")
            digits = "".join(ch for ch in sid if ch.isdigit())
            return (1, int(digits) if digits.isdigit() else 10**9)
        arr.sort(key=_order_key)
    return by_doc

def make_chunks_for_doc(
    sents: List[Dict[str, Any]],
    doc_id: str,
    min_sents: int = 3,
    max_sents: int = 5,
    max_chars: int = 1400,
    max_tokens: int = 300,
) -> List[Dict[str, Any]]:
    """
    Build context chunks with soft boundaries:
    - aim for 3–5 sentences
    - keep under max_chars and max_tokens
    - avoid starting/ending on header-like fragments if possible
    """
    chunks = []
    i = 0
    chunk_idx = 0
    n = len(sents)

    while i < n:
        # skip header-like lines as chunk starters where possible
        while i < n and _is_header_like(sents[i]["sentence"]):
            i += 1
        if i >= n:
            break

        cur = []
        cur_text = ""
        cur_tokens = 0
        j = i

        while j < n and len(cur) < max_sents:
            candidate = sents[j]["sentence"].strip()
            # don't append empty
            if not candidate:
                j += 1
                continue

            # tentative addition
            tentative_text = (cur_text + (" " if cur_text else "") + candidate).strip()
            tentative_tokens = _count_tokens(tentative_text)
            if (len(tentative_text) > max_chars) or (tentative_tokens > max_tokens):
                # if we don't have minimum sentences yet, force include once
                if len(cur) < max_sents and len(cur) < max_sents and len(cur) < max(min_sents, 1):
                    # force add (last one), then break
                    cur.append(sents[j])
                    cur_text = tentative_text
                    cur_tokens = tentative_tokens
                break

            cur.append(sents[j])
            cur_text = tentative_text
            cur_tokens = tentative_tokens
            j += 1

            # stop early if we reached min_sents and next looks like a header or would blow limits
            if len(cur) >= min_sents:
                if j < n and _is_header_like(sents[j]["sentence"]):
                    break

        # if we somehow collected nothing (e.g., repeated headers), advance
        if not cur:
            i = max(i + 1, j)
            continue

        # avoid ending on header-like tail if we have wiggle room
        while len(cur) >= 1 and _is_header_like(cur[-1]["sentence"]) and len(cur) > min_sents:
            last = cur.pop()
            cur_text = " ".join(s["sentence"].strip() for s in cur)

        chunk_idx += 1
        chunk_id = f"{doc_id}_chunk_{chunk_idx:04d}"
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": cur_text,
            # provenance helps debugging and evaluation later:
            "sent_ids": [s["sent_id"] for s in cur],
            "char_len": len(cur_text),
            "token_len": cur_tokens
        })

        i = max(j, i + 1)

    return chunks

def build_chunks(
    sentences_path: str,
    output_path: str,
    min_sents: int = 3,
    max_sents: int = 5,
    max_chars: int = 1400,
    max_tokens: int = 300
):
    sentences = load_sentences(sentences_path)
    docs = group_by_doc(sentences)

    all_chunks: List[Dict[str, Any]] = []
    for doc_id, arr in docs.items():
        all_chunks.extend(
            make_chunks_for_doc(
                arr, doc_id,
                min_sents=min_sents,
                max_sents=max_sents,
                max_chars=max_chars,
                max_tokens=max_tokens
            )
        )

    # Keep only the fields FRED driver needs (plus provenance for you)
    fred_ready = [{
        "doc_id": c["doc_id"],
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        # keep provenance—harmless for FRED, useful for audits
        "sent_ids": c["sent_ids"]
    } for c in all_chunks]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(fred_ready, indent=2, ensure_ascii=False), encoding="utf-8")

    # quick console summary
    by_doc_counts = defaultdict(int)
    for c in fred_ready:
        by_doc_counts[c["doc_id"]] += 1

    print(f"✓ Wrote {len(fred_ready)} chunks to {output_path}")
    print(f"  Docs: {len(by_doc_counts)} | Avg chunks/doc: {sum(by_doc_counts.values())/max(1,len(by_doc_counts)):.2f}")
    if fred_ready:
        token_stats = [ _count_tokens(c['text']) for c in fred_ready ]
        print(f"  Token len (min/avg/max): {min(token_stats)}/{sum(token_stats)//len(token_stats)}/{max(token_stats)}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build context chunks for FRED from sentence JSON.")
    p.add_argument("--sentences", required=True, help="Path to sentence-level JSON (from your segmenter).")
    p.add_argument("--out", default="data/olaf_chunks.json", help="Output JSON path for FRED driver.")
    p.add_argument("--min_sents", type=int, default=3)
    p.add_argument("--max_sents", type=int, default=5)
    p.add_argument("--max_chars", type=int, default=1400)
    p.add_argument("--max_tokens", type=int, default=300)
    args = p.parse_args()

    build_chunks(
        sentences_path=args.sentences,
        output_path=args.out,
        min_sents=args.min_sents,
        max_sents=args.max_sents,
        max_chars=args.max_chars,
        max_tokens=args.max_tokens
    )
