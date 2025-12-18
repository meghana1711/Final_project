import json
import re
import sqlite3
import unicodedata
from datetime import datetime
from typing import List, Tuple, Dict, Set
import string
import math
import statistics

# Path to your stopwords file
STOP_WORDS_FILE = "stop_word/stop_words.txt"

# Only terms with length_tokens <= this value get TF-IDF
# (set to 3 for 1–3-gram terms, or increase if you want longer phrases scored)
MAX_TOKENS_FOR_TFIDF = 3


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

# Optional: substrings that mark generic noisy phrases to drop
# patterns like "this section", "following example", etc.
GENERIC_NOISE_SUBSTRINGS: Set[str] = set()


# -------------------------------------------------------------------
# DB term extraction and term occurrences
# -------------------------------------------------------------------

def init_term_tables(db_path: str) -> None:
    """Create term_candidates and term_occurrences tables if missing,
    and ensure term_candidates has idf / tf_idf columns.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # Create base term_candidates table if needed
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS term_candidates (
            term_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            term_text      TEXT NOT NULL,
            term_lemma     TEXT NOT NULL,
            length_tokens  INTEGER NOT NULL,
            freq_total     INTEGER NOT NULL DEFAULT 0,
            freq_docs      INTEGER NOT NULL DEFAULT 0,
            idf            REAL    NOT NULL DEFAULT 0.0,
            tf_idf         REAL    NOT NULL DEFAULT 0.0,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL,
            UNIQUE(term_lemma, length_tokens)
        );
        """
    )

    # If table already existed before we added idf/tf_idf, add columns
    cur.execute("PRAGMA table_info(term_candidates);")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "idf" not in existing_cols:
        cur.execute(
            "ALTER TABLE term_candidates ADD COLUMN idf REAL NOT NULL DEFAULT 0.0;"
        )
    if "tf_idf" not in existing_cols:
        cur.execute(
            "ALTER TABLE term_candidates ADD COLUMN tf_idf REAL NOT NULL DEFAULT 0.0;"
        )

    # term_occurrences as before
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS term_occurrences (
            term_id         INTEGER NOT NULL,
            doc_id          TEXT NOT NULL,
            sent_idx        INTEGER NOT NULL,
            token_start     INTEGER NOT NULL,
            token_end       INTEGER NOT NULL,
            cleaned_version INTEGER NOT NULL,
            PRIMARY KEY (term_id, doc_id, sent_idx, token_start, token_end, cleaned_version),
            FOREIGN KEY (term_id) REFERENCES term_candidates(term_id) 
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


# -------------------------------------------------------------------
# Candidate extraction
# -------------------------------------------------------------------

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

    With many filters (numbers, junk, code, table noise, etc.) and:
      - Unicode normalization
      - simple singularization for NOUN/PROPN
      - pipe splitting
      - edge stopword stripping
    """
    candidates: List[Tuple[int, int, str, str, int]] = []
    n = len(tokens)
    i = 0

    seen_lemmas: Set[str] = set()

    EDGE_STRIP_CHARS = "".join(ch for ch in string.punctuation if ch not in "_-")

    def _normalize_unicode(s: str) -> str:
        if not s:
            return s
        s = unicodedata.normalize("NFKC", s)
        s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
        return s

    def _make_text_lemma(start: int, end: int) -> Tuple[str, str, int]:
        span_tokens = tokens[start : end + 1]
        span_lemmas = lemmas[start : end + 1]
        span_pos = pos[start : end + 1]

        norm_tokens = [_normalize_unicode(t) for t in span_tokens]

        norm_lemmas: List[str] = []
        for lemma_raw, tag in zip(span_lemmas, span_pos):
            lemma_norm = _normalize_unicode(lemma_raw)
            if tag in {"NOUN", "PROPN"}:
                low = lemma_norm.lower()
                if len(low) > 3 and low.endswith("s") and not low.endswith("ss"):
                    lemma_norm = lemma_norm[:-1]
            norm_lemmas.append(lemma_norm)

        term_text = " ".join(norm_tokens)
        term_text = term_text.replace(" - ", "-").replace(" / ", "/")

        term_lemma = " ".join(norm_lemmas).lower()
        term_lemma = term_lemma.replace(" - ", "-").replace(" / ", "/")

        length_tokens = len(norm_tokens)
        return term_text, term_lemma, length_tokens

    def _strip_edge_specials(term_text: str, term_lemma: str) -> Tuple[str, str, int]:
        text_tokens = term_text.split()
        lemma_tokens = term_lemma.split()
        if not text_tokens or not lemma_tokens:
            return "", "", 0

        def clean_tok_list(tok_list: List[str]) -> List[str]:
            if not tok_list:
                return []
            tok_list[0] = tok_list[0].strip(EDGE_STRIP_CHARS)
            tok_list[-1] = tok_list[-1].strip(EDGE_STRIP_CHARS)
            return [t for t in tok_list if t]

        text_tokens = clean_tok_list(text_tokens)
        lemma_tokens = clean_tok_list(lemma_tokens)

        if not text_tokens or not lemma_tokens:
            return "", "", 0

        min_len = min(len(text_tokens), len(lemma_tokens))
        text_tokens = text_tokens[:min_len]
        lemma_tokens = lemma_tokens[:min_len]

        text_clean = " ".join(text_tokens)
        lemma_clean = " ".join(lemma_tokens).lower()
        return text_clean, lemma_clean, len(text_tokens)

    def _emit(term_text: str, term_lemma: str, length_tokens: int, start: int, end: int) -> None:
        nonlocal seen_lemmas, candidates

        if not term_text or not term_lemma or length_tokens <= 0:
            return

        term_text, term_lemma, length_tokens = _strip_edge_specials(term_text, term_lemma)
        if not term_text or not term_lemma or length_tokens <= 0:
            return

        text_tokens = term_text.split()
        lemma_tokens = term_lemma.split()

        if len(text_tokens) > 1 and len(lemma_tokens) == len(text_tokens):
            # strip leading stopwords
            while len(text_tokens) > 1 and lemma_tokens[0].lower() in STOP_TERMS:
                text_tokens.pop(0)
                lemma_tokens.pop(0)
            # strip trailing stopwords
            while len(text_tokens) > 1 and lemma_tokens[-1].lower() in STOP_TERMS:
                text_tokens.pop()
                lemma_tokens.pop()

            if not text_tokens or not lemma_tokens:
                return

            term_text = " ".join(text_tokens)
            term_lemma = " ".join(lemma_tokens)
            length_tokens = len(text_tokens)

        # ----- junk filters for pipes / punctuation -----
        if "|" in term_text:
            text_parts = [p.strip() for p in term_text.split("|") if p.strip()]
            lemma_parts = [p.strip() for p in term_lemma.split("|") if p.strip()]
            if not text_parts:
                return
            if len(text_parts) != len(lemma_parts):
                lemma_parts = text_parts
            term_text = " ".join(text_parts)
            term_lemma = " ".join(lemma_parts).lower()
            length_tokens = len(term_text.split())
            if not term_text or not term_lemma or length_tokens <= 0:
                return

        compact = term_text.replace(" ", "")
        if not compact:
            return
        if all(ch in "|/\\-_=+*~" for ch in compact):
            return

        alpha_chars = sum(ch.isalpha() for ch in compact)
        if alpha_chars == 0:
            return
        if alpha_chars / len(compact) < 0.4:
            return

        # code / brackets
        if any(ch in term_text for ch in "<>(){}[]"):
            return

        if length_tokens > MAX_TERM_TOKENS:
            return
        if len(term_lemma) > MAX_TERM_CHARS:
            return

        for noise in GENERIC_NOISE_SUBSTRINGS:
            if noise in term_lemma:
                return

        if length_tokens == 1 and len(term_lemma) < 3:
            return

        if length_tokens == 1:
            lemma_token = term_lemma.strip().lower()
            if lemma_token in STOP_TERMS:
                return

        stripped = term_lemma.strip(string.punctuation + " ")
        if stripped.isdigit():
            return

        if re.search(r"\d{3,}", term_lemma):
            return
        if "=" in term_lemma:
            return
        if term_lemma.startswith("-") or term_lemma.endswith("?"):
            return

        m = re.match(r"(.+?)(\d+)$", term_lemma)
        if m:
            base = m.group(1).strip()
            if len(base) >= 3 and not base.replace(" ", "").isdigit():
                term_lemma = base
                term_text = re.sub(r"\d+$", "", term_text).strip()
                length_tokens = len(term_text.split())

        if term_lemma in seen_lemmas:
            return
        seen_lemmas.add(term_lemma)

        candidates.append((start, end, term_text, term_lemma, length_tokens))

    def _push(start: int, end: int) -> None:
        term_text, term_lemma, length_tokens = _make_text_lemma(start, end)
        if not term_text or not term_lemma or length_tokens <= 0:
            return

        if "|" in term_text:
            text_parts = [p.strip() for p in term_text.split("|") if p.strip()]
            lemma_parts = [p.strip() for p in term_lemma.split("|") if p.strip()]
            if len(text_parts) == len(lemma_parts) and len(text_parts) > 1:
                for t_sub, l_sub in zip(text_parts, lemma_parts):
                    sub_len_tokens = len(t_sub.split())
                    _emit(t_sub, l_sub.lower(), sub_len_tokens, start, end)
                return

        _emit(term_text, term_lemma, length_tokens, start, end)

    allowed_inside = {"ADJ", "NOUN", "PROPN"}
    allowed_end = {"NOUN", "PROPN"}

    while i < n:
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


# -------------------------------------------------------------------
# Run extraction onto DB
# -------------------------------------------------------------------

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

    cur.execute(
        """
        SELECT doc_id, sent_idx, tokens_json, lemmas_json, pos_tags_json
        FROM sentence_lemmatized
        WHERE cleaned_version = ?
        ORDER BY doc_id, sent_idx
        """,
        (cleaned_version,),
    )
    rows = cur.fetchall()

    if not rows:
        print(f"No lemmatized sentences found for cleaned_version={cleaned_version}.")
        conn.close()
        return (0, 0)

    print(f"Building term candidates from {len(rows)} sentences (cleaned_version={cleaned_version})...")

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

    for (term_lemma, length_tokens), info in term_stats.items():
        term_text = info["term_text"]
        freq_total = info["freq_total"]
        freq_docs = len(info["docs"])

        cur.execute(
            """
            INSERT INTO term_candidates
                (term_text, term_lemma, length_tokens, freq_total, freq_docs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term_lemma, length_tokens) DO UPDATE SET
                freq_total = term_candidates.freq_total + excluded.freq_total,
                freq_docs  = MAX(term_candidates.freq_docs, excluded.freq_docs),
                updated_at = excluded.updated_at
            """,
            (term_text, term_lemma, length_tokens, freq_total, freq_docs, now, now),
        )

        cur.execute(
            """
            SELECT term_id
            FROM term_candidates
            WHERE term_lemma = ? AND length_tokens = ?
            """,
            (term_lemma, length_tokens),
        )
        term_id = cur.fetchone()[0]
        term_key_to_id[(term_lemma, length_tokens)] = term_id

    occ_count = 0
    for term_lemma, length_tokens, doc_id, sent_idx, start, end in occurrences:
        if (term_lemma, length_tokens) not in term_key_to_id:
            continue
        term_id = term_key_to_id[(term_lemma, length_tokens)]
        cur.execute(
            """
            INSERT OR IGNORE INTO term_occurrences
                (term_id, doc_id, sent_idx, token_start, token_end, cleaned_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (term_id, doc_id, sent_idx, start, end, cleaned_version),
        )
        occ_count += cur.rowcount

    conn.commit()
    conn.close()

    print(f"Inserted/updated {len(term_stats)} term_candidates and {occ_count} term_occurrences.")
    return len(term_stats), occ_count


# -------------------------------------------------------------------
# TF-IDF inline on term_candidates
# -------------------------------------------------------------------

def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def compute_tf_idf_for_terms(db_path: str) -> None:
    """
    Compute TF-IDF scores and write them into term_candidates.idf and term_candidates.tf_idf.

    Uses:
      tf = freq_total
      df = freq_docs
      N  = number of distinct docs in term_occurrences

    Only updates terms with length_tokens <= MAX_TOKENS_FOR_TFIDF.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # make sure table exists + has columns
    init_term_tables(db_path)

    # count docs
    cur.execute("SELECT COUNT(DISTINCT doc_id) FROM term_occurrences;")
    row = cur.fetchone()
    total_docs = row[0] if row and row[0] is not None else 0
    if total_docs == 0:
        print("No documents in term_occurrences; skipping TF-IDF.")
        conn.close()
        return

    print(f"Computing TF-IDF inside term_candidates (N={total_docs})...")

    cur.execute(
        """
        SELECT term_id, length_tokens, freq_total, freq_docs
        FROM term_candidates
        """
    )
    rows = cur.fetchall()
    if not rows:
        print("No term_candidates rows; nothing to score.")
        conn.close()
        return

    tfidf_values: List[float] = []

    for term_id, length_tokens, freq_total, freq_docs in rows:
        length_tokens = int(length_tokens)
        tf = float(freq_total) if freq_total is not None else 0.0
        df = int(freq_docs) if freq_docs is not None else 0

        if df <= 0:
            df = 1

        # terms longer than MAX_TOKENS_FOR_TFIDF: leave idf / tf_idf as 0
        if length_tokens > MAX_TOKENS_FOR_TFIDF:
            idf = 0.0
            tf_idf = 0.0
        else:
            idf = math.log((total_docs + 1) / (df + 1)) + 1.0
            tf_idf = tf * idf
            tfidf_values.append(tf_idf)

        cur.execute(
            "UPDATE term_candidates SET idf = ?, tf_idf = ? WHERE term_id = ?",
            (idf, tf_idf, term_id),
        )

    conn.commit()
    conn.close()

    if tfidf_values:
        tfidf_values_sorted = sorted(tfidf_values)
        n = len(tfidf_values_sorted)
        tfidf_min = tfidf_values_sorted[0]
        tfidf_max = tfidf_values_sorted[-1]
        tfidf_mean = statistics.mean(tfidf_values_sorted)
        tfidf_median = statistics.median(tfidf_values_sorted)

        p25 = _percentile(tfidf_values_sorted, 25)
        p50 = _percentile(tfidf_values_sorted, 50)
        p75 = _percentile(tfidf_values_sorted, 75)
        p90 = _percentile(tfidf_values_sorted, 90)
        p95 = _percentile(tfidf_values_sorted, 95)
        p99 = _percentile(tfidf_values_sorted, 99)

        print("\n=== TF-IDF statistics (stored in term_candidates) ===")
        print(f"Total scored terms (len <= {MAX_TOKENS_FOR_TFIDF}): {n}")
        print(f"Min tf_idf: {tfidf_min:.4f}")
        print(f"25th pct : {p25:.4f}")
        print(f"50th pct : {p50:.4f}")
        print(f"75th pct : {p75:.4f}")
        print(f"90th pct : {p90:.4f}")
        print(f"95th pct : {p95:.4f}")
        print(f"99th pct : {p99:.4f}")
        print(f"Max tf_idf: {tfidf_max:.4f}")
        print(f"Mean tf_idf: {tfidf_mean:.4f}")
        print(f"Median tf_idf: {tfidf_median:.4f}")

        bins = [5, 10, 20, 40, 80, 160]
        counts = [0] * (len(bins) + 1)
        for v in tfidf_values_sorted:
            placed = False
            for i, b in enumerate(bins):
                if v < b:
                    counts[i] += 1
                    placed = True
                    break
            if not placed:
                counts[-1] += 1

        print("\nTF-IDF buckets (counts):")
        prev = 0.0
        for i, b in enumerate(bins):
            print(f"  [{prev:>5.1f}, {b:>5.1f}) : {counts[i]}")
            prev = b
        print(f"  [ {bins[-1]:>5.1f}, +inf) : {counts[-1]}")
        print("=========================================\n")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():
    DB_PATH = r"onto_db/onto_new.db"
    CLEANED_VERSION = 1

    print("Running term extraction...")
    n_terms, n_occ = extract_term_candidates(DB_PATH, CLEANED_VERSION)
    print(f"Loaded {len(STOP_TERMS)} stop terms from {STOP_WORDS_FILE}")
    print("'way' in STOP_TERMS?", "way" in STOP_TERMS)
    print(f"Done: {n_terms} unique terms, {n_occ} occurrences.")

    # Now compute TF-IDF directly into term_candidates
    compute_tf_idf_for_terms(DB_PATH)


if __name__ == "__main__":
    main()
