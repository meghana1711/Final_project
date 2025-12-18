import json, re, statistics
from collections import defaultdict, Counter
from pathlib import Path

ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "mr.", "mrs.", "dr.", "prof.", "fig.", "no.", "approx.", "cf.",
    "al.", "et al.", "inc.", "ltd.", "u.s.", "u.k.", "p.m.", "a.m."
}

END_PUNCT = {".", "!", "?"}

def load_docs(docs_path):
    docs = json.load(open(docs_path, "r", encoding="utf-8"))
    by_id = {d["doc_id"]: d for d in docs}
    return by_id

def load_sents(sents_path):
    return json.load(open(sents_path, "r", encoding="utf-8"))

def normalize_ws(s):
    # collapse whitespace for reconstruction comparisons
    return re.sub(r"\s+", " ", s).strip()

def structural_checks(docs_by_id, sents):
    problems = []
    by_doc = defaultdict(list)
    for s in sents:
        by_doc[s["doc_id"]].append(s)

    for doc_id, items in by_doc.items():
        items = sorted(items, key=lambda x: (x["start_char"], x["end_char"]))
        text_len = len(docs_by_id[doc_id]["text"])
        last_end = -1
        for s in items:
            a, b = s["start_char"], s["end_char"]
            if not (0 <= a <= b <= text_len):
                problems.append((doc_id, "OUT_OF_BOUNDS", a, b, text_len))
            if a < last_end:
                problems.append((doc_id, "OVERLAP", last_end, a, b))
            last_end = max(last_end, b)
    return problems

def ending_capitalization_checks(sents, treat_headings_as_ok=False):
    proper_end, improper_end = 0, 0
    starts_cap, starts_not_cap = 0, 0
    very_short, very_long = [], []

    for s in sents:
        txt = s["sentence"].strip()
        if not txt:
            continue
        # Proper ending
        if txt[-1] in END_PUNCT:
            proper_end += 1
        else:
            if treat_headings_as_ok and (len(txt) < 60 and (txt.endswith(":") or txt.istitle())):
                proper_end += 1
            else:
                improper_end += 1
        # Capitalization
        first_alpha = re.search(r"[A-Za-z]", txt)
        if first_alpha and txt[first_alpha.start()].isupper():
            starts_cap += 1
        else:
            starts_not_cap += 1
        # Length buckets
        L = len(txt)
        if L < 10:
            very_short.append(s)
        if L > 600:
            very_long.append(s)

    return {
        "proper_end_pct": 100 * proper_end / max(1, proper_end + improper_end),
        "proper_end_counts": (proper_end, improper_end),
        "starts_cap_pct": 100 * starts_cap / max(1, starts_cap + starts_not_cap),
        "starts_cap_counts": (starts_cap, starts_not_cap),
        "very_short": very_short,
        "very_long": very_long,
    }

def abbreviation_split_checks(sents):
    # Look for sentences ending with a known abbreviation then a next sentence starting with a capital.
    issues = []
    for i in range(len(sents) - 1):
        a, b = sents[i], sents[i+1]
        at = a["sentence"].strip().lower()
        if any(at.endswith(abbr) for abbr in ABBREVIATIONS):
            nxt = b["sentence"].strip()
            if re.match(r"^[A-Z]", nxt):
                issues.append((a, b))
    return issues

def runon_candidates(sents):
    # Very long sentences with too few internal terminals → suspicious
    cands = []
    for s in sents:
        txt = s["sentence"]
        if len(txt) > 600:
            stops = len(re.findall(r"[.!?]", txt))
            if stops <= 1:
                cands.append(s)
    return cands

def reconstruction_check(docs_by_id, sents, sample_limit=5, whitespace_insensitive=True):
    # Sample up to N docs and verify that concatenating spans ≈ original
    from random import sample
    by_doc = defaultdict(list)
    for s in sents:
        by_doc[s["doc_id"]].append(s)
    doc_ids = list(by_doc.keys())
    picked = sample(doc_ids, min(sample_limit, len(doc_ids)))
    results = []

    for doc_id in picked:
        items = sorted(by_doc[doc_id], key=lambda x: x["start_char"])
        original = docs_by_id[doc_id]["text"]
        recon = "".join(original[s["start_char"]: s["end_char"]] for s in items)
        if whitespace_insensitive:
            ok = normalize_ws(original) == normalize_ws(recon)
        else:
            ok = original == recon
        results.append((doc_id, ok))
    return results

def distribution_stats(sents):
    lens = [len(s["sentence"]) for s in sents if s["sentence"]]
    words = [len(re.findall(r"\\w+", s["sentence"])) for s in sents]
    return {
        "n": len(sents),
        "char_avg": round(statistics.mean(lens), 1) if lens else 0,
        "char_min": min(lens) if lens else 0,
        "char_max": max(lens) if lens else 0,
        "word_avg": round(statistics.mean(words), 1) if words else 0,
        "word_p95": statistics.quantiles(words, n=20)[-1] if words else 0,
    }

def main(docs_path="documents_cleaned.json", sents_path="sentences_improved.json",
         treat_headings_as_ok=False):
    docs = load_docs(docs_path)
    sents = load_sents(sents_path)

    print("\\n=== STRUCTURAL CHECKS ===")
    probs = structural_checks(docs, sents)
    if not probs:
        print("OK: no overlaps or out-of-bounds spans found.")
    else:
        kinds = Counter(p[1] for p in probs)
        print("Problems:", kinds)
        for p in probs[:5]:
            print("  Example:", p)

    print("\\n=== SURFACE SIGNALS ===")
    endcap = ending_capitalization_checks(sents, treat_headings_as_ok)
    print(f"Proper endings: {endcap['proper_end_pct']:.1f}% "
          f"({endcap['proper_end_counts'][0]}/{sum(endcap['proper_end_counts'])})")
    print(f"Starts with capital: {endcap['starts_cap_pct']:.1f}% "
          f"({endcap['starts_cap_counts'][0]}/{sum(endcap['starts_cap_counts'])})")
    print(f"Very short (<10 chars): {len(endcap['very_short'])}")
    print(f"Very long (>600 chars): {len(endcap['very_long'])}")

    print("\\nExamples (very short):")
    for s in endcap["very_short"][:5]:
        print(" -", s["sent_id"], repr(s["sentence"]))
    print("Examples (very long):")
    for s in endcap["very_long"][:5]:
        print(" -", s["sent_id"], len(s["sentence"]))

    print("\\n=== ABBREVIATION SPLITS ===")
    ab = abbreviation_split_checks(sents)
    print("Potential abbreviation split pairs:", len(ab))
    for a, b in ab[:5]:
        print("  A:", a["sent_id"], repr(a["sentence"][-80:]))
        print("  B:", b["sent_id"], repr(b["sentence"][:80]))

    print("\\n=== RUN-ON CANDIDATES ===")
    ro = runon_candidates(sents)
    print("Very long & low-punctuation candidates:", len(ro))
    for s in ro[:5]:
        print("  ", s["sent_id"], len(s["sentence"]))

    print("\\n=== RECONSTRUCTION SAMPLE ===")
    recon = reconstruction_check(docs, sents, sample_limit=5, whitespace_insensitive=True)
    for doc_id, ok in recon:
        print(f"  {doc_id}: {'OK' if ok else 'MISMATCH'}")

    print("\\n=== DISTRIBUTION STATS ===")
    dist = distribution_stats(sents)
    for k, v in dist.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()

    
  