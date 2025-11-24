import json
import re
import sqlite3
import unicodedata
from datetime import datetime
from typing import List, Tuple, Dict, Set
import string

# Path to your stopwords file
STOP_WORDS_FILE = "other/stop_words.txt"

def load_stop_terms(path: str) -> Set[str]:
    """
    Load stop terms from a text file.

    Supports:
      - one word per line
      - comma-separated or whitespace-separated lists per line
        e.g. "a, able, about, above"
      - lines starting with '#' as comments
    All terms are lowercased.
    """
    terms: Set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # split on commas and/or whitespace
                for token in re.split(r"[,\s]+", line):
                    token = token.strip().lower()
                    if token:
                        terms.add(token)
    except FileNotFoundError:
        print(f"Warning: stopword file '{path}' not found; STOP_TERMS will be empty.")
    return terms


STOP_TERMS: Set[str] = load_stop_terms(STOP_WORDS_FILE)

# Keep only terms that appear in at least this many documents
MIN_DOC_FREQ = 1

# Maximum tokens per candidate term to avoid very long noisy phrases
MAX_TERM_TOKENS = 7

# Maximum characters in the lemma to avoid very long noisy phrases
MAX_TERM_CHARS = 60

# Optional: substrings that mark generic noisy phrases to drop patterns like "this section", "following example", etc.
GENERIC_NOISE_SUBSTRINGS: Set[str] = set()


# DB term extraction and term occurances

def init_term_tables(db_path: str) -> None:
    """Create term_candidates and term_occurrences tables if missing."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS term_candidates (
            term_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            term_text      TEXT NOT NULL,
            term_lemma     TEXT NOT NULL,
            length_tokens  INTEGER NOT NULL,
            freq_total     INTEGER NOT NULL DEFAULT 0,
            freq_docs      INTEGER NOT NULL DEFAULT 0,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            UNIQUE(term_lemma, length_tokens)
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS term_occurrences (
            term_id         INTEGER NOT NULL,
            doc_id          TEXT NOT NULL,
            sent_idx        INTEGER NOT NULL,
            token_start     INTEGER NOT NULL,
            token_end       INTEGER NOT NULL,
            cleaned_version INTEGER NOT NULL,
            PRIMARY KEY (term_id, doc_id, sent_idx, token_start, token_end, cleaned_version),
            FOREIGN KEY (term_id) REFERENCES term_candidates(term_id) ON DELETE CASCADE
        );
    """)

    conn.commit()
    conn.close()

# Candidate extraction
def _extract_candidates_from_sentence(
    tokens: List[str],
    lemmas: List[str],
    pos: List[str],
) -> List[Tuple[int, int, str, str, int]]:
    """
    Return list of spans as:
        (start, end, term_text, term_lemma, length_tokens)

    Patterns:
      (1) (ADJ|NOUN|PROPN)+ ending in (NOUN|PROPN)
      (2) (NOUN|PROPN) ADP (NOUN|PROPN)+
      (3) PROPN (PROPN|NOUN)+

    With additional filters to avoid:
      - pure/mostly numeric terms
      - key=value patterns
      - weird IDs like '-147101316', 'CPUs303030'
      - overly long noisy phrases
      - code-like / placeholder text (fopen(<logdir, <logdir>, etc.)
      - duplicate term_lemma within the same sentence

    And with:
      - Unicode normalization (ﬁ → fi etc.)
      - Simple singularization for NOUN/PROPN lemmas (jobs -> job, nodes -> node)
      - Pipe splitting: "cr_cpu_memory | cr_core | cr_core_memory" -> 3 terms
    """
    candidates: List[Tuple[int, int, str, str, int]] = []
    n = len(tokens)
    i = 0

    # Track lemmas we've already emitted in this sentence
    seen_lemmas: Set[str] = set()

    #  Unicode normalization 

    def _normalize_unicode(s: str) -> str:
        if not s:
            return s
        s = unicodedata.normalize("NFKC", s)
        # Replace NBSP-like spaces with regular spaces
        s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
        return s

    #  Text / lemma construction with singularization
    def _make_text_lemma(start: int, end: int) -> Tuple[str, str, int]:
        # Slice span
        span_tokens = tokens[start:end + 1]
        span_lemmas = lemmas[start:end + 1]
        span_pos = pos[start:end + 1]

        # Unicode normalization per token
        norm_tokens = [_normalize_unicode(t) for t in span_tokens]

        # Normalize + simple singularization for NOUN/PROPN lemmas
        norm_lemmas: List[str] = []
        for lemma_raw, tag in zip(span_lemmas, span_pos):
            lemma_norm = _normalize_unicode(lemma_raw)
            # Simple "make it singular" rule:
            # if POS is NOUN/PROPN and lemma ends with "s" (and is not too short),
            # drop the final "s". Conservative: ignore words like "ss".
            if tag in {"NOUN", "PROPN"}:
                low = lemma_norm.lower()
                if len(low) > 3 and low.endswith("s") and not low.endswith("ss"):
                    lemma_norm = lemma_norm[:-1]
            norm_lemmas.append(lemma_norm)

        term_text = " ".join(norm_tokens)
        # normalize hyphen/slash spacing
        term_text = term_text.replace(" - ", "-").replace(" / ", "/")

        term_lemma = " ".join(norm_lemmas).lower()
        term_lemma = term_lemma.replace(" - ", "-").replace(" / ", "/")

        length_tokens = len(norm_tokens)
        return term_text, term_lemma, length_tokens

    # Low-level emitter with all filters
    def _emit(term_text: str, term_lemma: str, length_tokens: int, start: int, end: int) -> None:
        nonlocal seen_lemmas, candidates

        # If normalization produced an empty or degenerate term, skip
        if not term_text or not term_lemma or length_tokens <= 0:
            return

        # ---- CODE / PLACEHOLDER FILTERS ----
        # Drop anything that looks like code or placeholder with brackets,
        # e.g. "fopen(<logdir", "<logdir>", "stat(/tmp)", etc.
        if any(ch in term_text for ch in "<>(){}[]"):
            return

        # ---- LENGTH / NOISE FILTERS ----

        # Reject very long multi-token candidates (likely noisy)
        if length_tokens > MAX_TERM_TOKENS:
            return

        # Reject extremely long lemmas (lots of boilerplate text)
        if len(term_lemma) > MAX_TERM_CHARS:
            return

        # Reject generic phrase-like junk by substring (domain configurable)
        for noise in GENERIC_NOISE_SUBSTRINGS:
            if noise in term_lemma:
                return

        # very short single tokens
        if length_tokens == 1 and len(term_lemma) < 3:
            return

        # Drop single-token stopwords using lemma (case-insensitive)
        if length_tokens == 1:
            lemma_token = term_lemma.strip().lower()
            if lemma_token in STOP_TERMS:
                return

        # reject pure numbers (including leading +/- and simple punctuation)
        stripped = term_lemma.strip(string.punctuation + " ")
        if stripped.isdigit():
            return

        # reject long digit sequences inside the term (IDs, CPUs303030, -147101316)
        if re.search(r"\d{3,}", term_lemma):
            return

        # reject key=value style terms (we'll treat these as config patterns later)
        if "=" in term_lemma:
            return

        # reject obvious garbage punctuation shapes
        if term_lemma.startswith("-") or term_lemma.endswith("?"):
            return

        # Strip trailing digits if there is a decent text base
        # e.g. "allocated cpus42" -> "allocated cpus"
        m = re.match(r"(.+?)(\d+)$", term_lemma)
        if m:
            base = m.group(1).strip()
            if len(base) >= 3 and not base.replace(" ", "").isdigit():
                term_lemma = base
                term_text = re.sub(r"\d+$", "", term_text).strip()
                # length_tokens stays the span length

        # Drop duplicates in term_lemma within the sentence 
        if term_lemma in seen_lemmas:
            return
        seen_lemmas.add(term_lemma)

        candidates.append((start, end, term_text, term_lemma, length_tokens))

    #  Candidate collection with filters + pipe splitting
    def _push(start: int, end: int) -> None:
        term_text, term_lemma, length_tokens = _make_text_lemma(start, end)

        # If normalization produced an empty or degenerate term, skip
        if not term_text or not term_lemma or length_tokens <= 0:
            return

        # Pipe splitting: "cr_cpu_memory | cr_core | cr_core_memory"
        # becomes separate terms, each emitted through _emit.
        if "|" in term_text:
            text_parts = [p.strip() for p in term_text.split("|") if p.strip()]
            lemma_parts = [p.strip() for p in term_lemma.split("|") if p.strip()]

            # Only split if we can align text and lemma parts sensibly
            if len(text_parts) == len(lemma_parts) and len(text_parts) > 1:
                for t_sub, l_sub in zip(text_parts, lemma_parts):
                    sub_len_tokens = len(t_sub.split())
                    _emit(t_sub, l_sub.lower(), sub_len_tokens, start, end)
                return  # we've handled this span via splitting

        # Default: emit the span as-is
        _emit(term_text, term_lemma, length_tokens, start, end)

    allowed_inside = {"ADJ", "NOUN", "PROPN"}
    allowed_end = {"NOUN", "PROPN"}

    while i < n:
        # ADJ/NOUN/PROPN+ ending in NOUN/PROPN
        if pos[i] in allowed_inside:
            start = i
            j = i + 1
            while j < n and pos[j] in allowed_inside:
                j += 1
            end = j - 1
            if pos[end] in allowed_end:
                _push(start, end)
            i = j
            continue

        # (NOUN|PROPN) ADP (NOUN|PROPN)+  e.g. "quality of service"
        if (
            i + 2 < n
            and pos[i] in {"NOUN", "PROPN"}
            and pos[i + 1] == "ADP"
            and pos[i + 2] in {"NOUN", "PROPN"}
        ):
            start = i
            j = i + 3
            while j < n and pos[j] in {"NOUN", "PROPN"}:
                j += 1
            end = j - 1
            _push(start, end)
            i = j
            continue

        # PROPN (PROPN|NOUN)+   e.g. "Slurm Controller"
        if pos[i] == "PROPN":
            start = i
            j = i + 1
            while j < n and pos[j] in {"PROPN", "NOUN"}:
                j += 1
            end = j - 1
            if end > start:
                _push(start, end)
                i = j
                continue

        i += 1

    return candidates

# Run extraction onto DB
def extract_term_candidates(db_path: str, cleaned_version: int) -> Tuple[int, int]:
    """
    Read from sentence_lemmatized and populate term_candidates + term_occurrences.

    Returns:
        (number_of_unique_terms_touched, number_of_occurrences_inserted)
    """
    init_term_tables(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute("""
        SELECT doc_id, sent_idx, tokens_json, lemmas_json, pos_tags_json
        FROM sentence_lemmatized
        WHERE cleaned_version = ?
        ORDER BY doc_id, sent_idx
    """, (cleaned_version,))
    rows = cur.fetchall()

    if not rows:
        print(f"No lemmatized sentences found for cleaned_version={cleaned_version}.")
        conn.close()
        return (0, 0)

    print(f"Building term candidates from {len(rows)} sentences (cleaned_version={cleaned_version})...")

    # key = (term_lemma, length_tokens)
    term_stats: Dict[Tuple[str, int], Dict] = {}
    occurrences = []

    for doc_id, sent_idx, tokens_json, lemmas_json, pos_json in rows:
        tokens = json.loads(tokens_json)
        lemmas = json.loads(lemmas_json)
        pos = json.loads(pos_json) if pos_json is not None else ["X"] * len(tokens)

        cands = _extract_candidates_from_sentence(tokens, lemmas, pos)
        for start, end, term_text, term_lemma, length_tokens in cands:
            key = (term_lemma, length_tokens)
            if key not in term_stats:
                term_stats[key] = {
                    "term_text": term_text,
                    "freq_total": 0,
                    "docs": set(),
                }
            term_stats[key]["freq_total"] += 1
            term_stats[key]["docs"].add(doc_id)

            occurrences.append((term_lemma, length_tokens, doc_id, sent_idx, start, end))

    if MIN_DOC_FREQ > 1:
        term_stats = {
            k: v for k, v in term_stats.items()
            if len(v["docs"]) >= MIN_DOC_FREQ
        }
        keep_keys = set(term_stats.keys())
        occurrences = [
            occ for occ in occurrences
            if (occ[0], occ[1]) in keep_keys
        ]

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    term_key_to_id: Dict[Tuple[str, int], int] = {}

    # Upsert term candidates
    for (term_lemma, length_tokens), info in term_stats.items():
        term_text = info["term_text"]
        freq_total = info["freq_total"]
        freq_docs = len(info["docs"])

        cur.execute("""
            INSERT INTO term_candidates
                (term_text, term_lemma, length_tokens, freq_total, freq_docs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term_lemma, length_tokens) DO UPDATE SET
                freq_total = term_candidates.freq_total + excluded.freq_total,
                freq_docs  = MAX(term_candidates.freq_docs, excluded.freq_docs),
                updated_at = excluded.updated_at
        """, (term_text, term_lemma, length_tokens, freq_total, freq_docs, now, now))

        cur.execute("""
            SELECT term_id
            FROM term_candidates
            WHERE term_lemma = ? AND length_tokens = ?
        """, (term_lemma, length_tokens))
        term_id = cur.fetchone()[0]
        term_key_to_id[(term_lemma, length_tokens)] = term_id

    # Insert term occurrences
    occ_count = 0
    for term_lemma, length_tokens, doc_id, sent_idx, start, end in occurrences:
        if (term_lemma, length_tokens) not in term_key_to_id:
            continue
        term_id = term_key_to_id[(term_lemma, length_tokens)]
        cur.execute("""
            INSERT OR IGNORE INTO term_occurrences
                (term_id, doc_id, sent_idx, token_start, token_end, cleaned_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (term_id, doc_id, sent_idx, start, end, cleaned_version))
        occ_count += cur.rowcount

    conn.commit()
    conn.close()

    print(f"Inserted/updated {len(term_stats)} term_candidates and {occ_count} term_occurrences.")
    return len(term_stats), occ_count


def main():
    DB_PATH = r"onto_db/ontology_sample_new.db"  
    CLEANED_VERSION = 1                          

    print("Running term extraction...")
    n_terms, n_occ = extract_term_candidates(DB_PATH, CLEANED_VERSION)
    print(f"Loaded {len(STOP_TERMS)} stop terms from {STOP_WORDS_FILE}")
    print("'way' in STOP_TERMS?", "way" in STOP_TERMS)
    print(f"Done: {n_terms} unique terms, {n_occ} occurrences.")


if __name__ == "__main__":
    main()
