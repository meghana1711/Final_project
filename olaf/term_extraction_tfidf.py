import argparse
import json
import re
import sqlite3
import unicodedata
from datetime import datetime
from typing import List, Tuple, Dict, Set, Optional
import string
import math
import statistics


# -----------------------------
# Stopwords
# -----------------------------

def load_stop_terms(path: str) -> Set[str]:
    terms: Set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for token in re.split(r"[,\s]+", line):
                    token = token.strip().lower()
                    if token:
                        terms.add(token)
    except FileNotFoundError:
        print(f"Warning: stopword file '{path}' not found; STOP_TERMS will be empty.")
    return terms


# -----------------------------
# Defaults (can be overridden by CLI)
# -----------------------------
DEFAULT_STOP_WORDS_FILE = "stop_word/stop_words.txt"
DEFAULT_MIN_DOC_FREQ = 1
DEFAULT_MAX_TERM_TOKENS = 7
DEFAULT_MAX_TERM_CHARS = 60
DEFAULT_MAX_TOKENS_FOR_TFIDF = 3


# -----------------------------
# DB schema
# -----------------------------

def init_term_tables(
    db_path: str,
    term_candidates_table: str,
    term_occurrences_table: str,
) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {term_candidates_table} (
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

    cur.execute(f"PRAGMA table_info({term_candidates_table});")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "idf" not in existing_cols:
        cur.execute(
            f"ALTER TABLE {term_candidates_table} ADD COLUMN idf REAL NOT NULL DEFAULT 0.0;"
        )
    if "tf_idf" not in existing_cols:
        cur.execute(
            f"ALTER TABLE {term_candidates_table} ADD COLUMN tf_idf REAL NOT NULL DEFAULT 0.0;"
        )

    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {term_occurrences_table} (
            term_id         INTEGER NOT NULL,
            doc_id          TEXT NOT NULL,
            sent_idx        INTEGER NOT NULL,
            token_start     INTEGER NOT NULL,
            token_end       INTEGER NOT NULL,
            cleaned_version INTEGER NOT NULL,
            PRIMARY KEY (term_id, doc_id, sent_idx, token_start, token_end, cleaned_version),
            FOREIGN KEY (term_id) REFERENCES {term_candidates_table}(term_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    conn.close()


# -----------------------------
# Candidate extraction
# -----------------------------

def _is_cli_flag(term: str) -> bool:
    """
    Accept common HPC/CLI flags:
      --nodes
      -N
      --gres=gpu:1
      --wrap="..."
      --time=01:00:00
    """
    if not term or " " in term:
        return False
    if not term.startswith("-"):
        return False

    # Must contain at least one alphabetic character (reject '--' or '-')
    if not any(ch.isalpha() for ch in term):
        return False

    # Allow: -X, -X=..., --flag, --flag=..., --flag:..., --flag=value:value
    # Disallow spaces, brackets are handled elsewhere.
    return re.match(r"^-{1,2}[A-Za-z][A-Za-z0-9_-]*(?:[=:][^\s]+)?$", term) is not None


def _extract_candidates_from_sentence(
    tokens: List[str],
    lemmas: List[str],
    pos: List[str],
    stop_terms: Set[str],
    min_doc_freq: int,
    max_term_tokens: int,
    max_term_chars: int,
    generic_noise_substrings: Set[str],
) -> List[Tuple[int, int, str, str, int]]:
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

        term_text = " ".join(norm_tokens).replace(" - ", "-").replace(" / ", "/")
        term_lemma = " ".join(norm_lemmas).lower().replace(" - ", "-").replace(" / ", "/")
        length_tokens = len(norm_tokens)
        return term_text, term_lemma, length_tokens

    def _strip_edge_specials(term_text: str, term_lemma: str) -> Tuple[str, str, int]:
        text_tokens = term_text.split()
        lemma_tokens = term_lemma.split()
        if not text_tokens or not lemma_tokens:
            return "", "", 0

        def clean_tok_list(tok_list: List[str]) -> List[str]:
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

        return " ".join(text_tokens), " ".join(lemma_tokens).lower(), len(text_tokens)

    def _emit(term_text: str, term_lemma: str, length_tokens: int, start: int, end: int) -> None:
        nonlocal seen_lemmas, candidates

        if not term_text or not term_lemma or length_tokens <= 0:
            return

        term_text, term_lemma, length_tokens = _strip_edge_specials(term_text, term_lemma)
        if not term_text or not term_lemma or length_tokens <= 0:
            return

        # strip leading/trailing stopwords for multiword terms
        text_tokens = term_text.split()
        lemma_tokens = term_lemma.split()

        if len(text_tokens) > 1 and len(lemma_tokens) == len(text_tokens):
            while len(text_tokens) > 1 and lemma_tokens[0].lower() in stop_terms:
                text_tokens.pop(0)
                lemma_tokens.pop(0)
            while len(text_tokens) > 1 and lemma_tokens[-1].lower() in stop_terms:
                text_tokens.pop()
                lemma_tokens.pop()
            if not text_tokens:
                return
            term_text = " ".join(text_tokens)
            term_lemma = " ".join(lemma_tokens).lower()
            length_tokens = len(text_tokens)

        # pipe splitting cleanup
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

        # code / brackets (keep strict)
        if any(ch in term_text for ch in "<>(){}[]"):
            return

        if length_tokens > max_term_tokens:
            return
        if len(term_lemma) > max_term_chars:
            return

        for noise in generic_noise_substrings:
            if noise in term_lemma:
                return

        # single token stopword
        if length_tokens == 1:
            if term_lemma.strip().lower() in stop_terms:
                return
            if len(term_lemma) < 3 and not _is_cli_flag(term_text):
                return

        stripped = term_lemma.strip(string.punctuation + " ")
        if stripped.isdigit():
            return
        if re.search(r"\d{3,}", term_lemma):
            return
        if "=" in term_lemma and not _is_cli_flag(term_text):
            # allow '=' only for CLI flags like --gres=gpu:1
            return

        # Previously: drop anything starting with '-' (kills --nodes, --gres, -N)
        # Now: allow if it's a CLI flag; otherwise drop dash-start junk.
        if term_lemma.startswith("-") and not _is_cli_flag(term_text):
            return

        if term_lemma.endswith("?"):
            return

        # normalize trailing digits like foo123 -> foo (but keep flags)
        if not _is_cli_flag(term_text):
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

        # If it's a single-token CLI flag, emit directly (even if POS patterns would be weird)
        if length_tokens == 1 and _is_cli_flag(term_text):
            _emit(term_text, term_lemma, length_tokens, start, end)
            return

        if "|" in term_text:
            text_parts = [p.strip() for p in term_text.split("|") if p.strip()]
            lemma_parts = [p.strip() for p in term_lemma.split("|") if p.strip()]
            if len(text_parts) == len(lemma_parts) and len(text_parts) > 1:
                for t_sub, l_sub in zip(text_parts, lemma_parts):
                    _emit(t_sub, l_sub.lower(), len(t_sub.split()), start, end)
                return

        _emit(term_text, term_lemma, length_tokens, start, end)

    allowed_inside = {"ADJ", "NOUN", "PROPN"}
    allowed_end = {"NOUN", "PROPN"}

    while i < n:
        # Pattern 1: (ADJ|NOUN|PROPN)+ ending in (NOUN|PROPN)
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

        # Pattern 2: NOUN/PROPN ADP NOUN/PROPN+
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

        # Pattern 3: PROPN (PROPN|NOUN)+
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

        # Also: if token itself looks like a CLI flag, try emitting it
        if _is_cli_flag(tokens[i]):
            _push(i, i)

        i += 1

    return candidates


# -----------------------------
# Extraction + occurrences
# -----------------------------

def extract_term_candidates(
    db_path: str,
    cleaned_version: int,
    sentence_table: str,
    term_candidates_table: str,
    term_occurrences_table: str,
    stop_terms: Set[str],
    min_doc_freq: int,
    max_term_tokens: int,
    max_term_chars: int,
    generic_noise_substrings: Set[str],
) -> Tuple[int, int]:
    init_term_tables(db_path, term_candidates_table, term_occurrences_table)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        f"""
        SELECT doc_id, sent_idx, tokens_json, lemmas_json, pos_tags_json
        FROM {sentence_table}
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

        cands = _extract_candidates_from_sentence(
            tokens=tokens,
            lemmas=lemmas,
            pos=pos,
            stop_terms=stop_terms,
            min_doc_freq=min_doc_freq,
            max_term_tokens=max_term_tokens,
            max_term_chars=max_term_chars,
            generic_noise_substrings=generic_noise_substrings,
        )

        for start, end, term_text, term_lemma, length_tokens in cands:
            key = (term_lemma, length_tokens)
            if key not in term_stats:
                term_stats[key] = {"term_text": term_text, "freq_total": 0, "docs": set()}
            term_stats[key]["freq_total"] += 1
            term_stats[key]["docs"].add(doc_id)

            occurrences.append((term_lemma, length_tokens, doc_id, sent_idx, start, end))

    if min_doc_freq > 1:
        term_stats = {k: v for k, v in term_stats.items() if len(v["docs"]) >= min_doc_freq}
        keep_keys = set(term_stats.keys())
        occurrences = [occ for occ in occurrences if (occ[0], occ[1]) in keep_keys]

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    term_key_to_id: Dict[Tuple[str, int], int] = {}

    for (term_lemma, length_tokens), info in term_stats.items():
        term_text = info["term_text"]
        freq_total = info["freq_total"]
        freq_docs = len(info["docs"])

        cur.execute(
            f"""
            INSERT INTO {term_candidates_table}
                (term_text, term_lemma, length_tokens, freq_total, freq_docs, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(term_lemma, length_tokens) DO UPDATE SET
                freq_total = {term_candidates_table}.freq_total + excluded.freq_total,
                freq_docs  = MAX({term_candidates_table}.freq_docs, excluded.freq_docs),
                updated_at = excluded.updated_at
            """,
            (term_text, term_lemma, length_tokens, freq_total, freq_docs, now, now),
        )

        cur.execute(
            f"""
            SELECT term_id
            FROM {term_candidates_table}
            WHERE term_lemma = ? AND length_tokens = ?
            """,
            (term_lemma, length_tokens),
        )
        term_id = cur.fetchone()[0]
        term_key_to_id[(term_lemma, length_tokens)] = term_id

    occ_count = 0
    for term_lemma, length_tokens, doc_id, sent_idx, start, end in occurrences:
        term_id = term_key_to_id.get((term_lemma, length_tokens))
        if not term_id:
            continue
        cur.execute(
            f"""
            INSERT OR IGNORE INTO {term_occurrences_table}
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


# -----------------------------
# TF-IDF
# -----------------------------

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

def reset_term_tables(
    db_path: str,
    term_candidates_table: str,
    term_occurrences_table: str,
) -> None:
    """
    Delete all rows from occurrences first (FK dependency),
    then candidates. Keeps schema intact.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # Occurrences first because it references candidates
    cur.execute(f"DELETE FROM {term_occurrences_table};")
    cur.execute(f"DELETE FROM {term_candidates_table};")

    conn.commit()
    conn.close()
    print(f"[RESET] Cleared {term_occurrences_table} and {term_candidates_table}.")

def compute_tf_idf_for_terms(
    db_path: str,
    term_candidates_table: str,
    term_occurrences_table: str,
    max_tokens_for_tfidf: int,
) -> None:
    init_term_tables(db_path, term_candidates_table, term_occurrences_table)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(f"SELECT COUNT(DISTINCT doc_id) FROM {term_occurrences_table};")
    total_docs = cur.fetchone()[0] or 0
    if total_docs == 0:
        print("No documents in term_occurrences; skipping TF-IDF.")
        conn.close()
        return

    print(f"Computing TF-IDF inside {term_candidates_table} (N={total_docs})...")

    cur.execute(
        f"""
        SELECT term_id, length_tokens, freq_total, freq_docs
        FROM {term_candidates_table}
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
        tf = float(freq_total or 0.0)
        df = int(freq_docs or 1)

        if length_tokens > max_tokens_for_tfidf:
            idf = 0.0
            tf_idf = 0.0
        else:
            idf = math.log((total_docs + 1) / (df + 1)) + 1.0
            tf_idf = tf * idf
            tfidf_values.append(tf_idf)

        cur.execute(
            f"UPDATE {term_candidates_table} SET idf = ?, tf_idf = ? WHERE term_id = ?",
            (idf, tf_idf, term_id),
        )

    conn.commit()
    conn.close()

    if tfidf_values:
        tfidf_values_sorted = sorted(tfidf_values)
        n = len(tfidf_values_sorted)

        print("\n=== TF-IDF statistics (stored in term_candidates) ===")
        print(f"Total scored terms (len <= {max_tokens_for_tfidf}): {n}")
        print(f"Min tf_idf: {tfidf_values_sorted[0]:.4f}")
        print(f"Median:     {statistics.median(tfidf_values_sorted):.4f}")
        print(f"Mean:       {statistics.mean(tfidf_values_sorted):.4f}")
        print(f"90th pct:   {_percentile(tfidf_values_sorted, 90):.4f}")
        print(f"95th pct:   {_percentile(tfidf_values_sorted, 95):.4f}")
        print(f"Max tf_idf: {tfidf_values_sorted[-1]:.4f}")
        print("=========================================\n")


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--db", required=True)
    ap.add_argument("--cleaned_version", type=int, default=1)

    ap.add_argument("--sentence_table", default="sentence_lemmatized")
    ap.add_argument("--term_candidates_table", default="term_candidates")
    ap.add_argument("--term_occurrences_table", default="term_occurrences")

    ap.add_argument("--stopwords", default=DEFAULT_STOP_WORDS_FILE)
    ap.add_argument("--reset_terms", action="store_true", help="Clear term tables before extraction")
  
    ap.add_argument("--min_doc_freq", type=int, default=DEFAULT_MIN_DOC_FREQ)
    ap.add_argument("--max_term_tokens", type=int, default=DEFAULT_MAX_TERM_TOKENS)
    ap.add_argument("--max_term_chars", type=int, default=DEFAULT_MAX_TERM_CHARS)
    ap.add_argument("--max_tfidf_tokens", type=int, default=DEFAULT_MAX_TOKENS_FOR_TFIDF)
    args = ap.parse_args()

    stop_terms = load_stop_terms(args.stopwords)

    print("Running term extraction (TF-IDF)...")
    n_terms, n_occ = extract_term_candidates(
        db_path=args.db,
        cleaned_version=args.cleaned_version,
        sentence_table=args.sentence_table,
        term_candidates_table=args.term_candidates_table,
        term_occurrences_table=args.term_occurrences_table,
        stop_terms=stop_terms,
        min_doc_freq=args.min_doc_freq,
        max_term_tokens=args.max_term_tokens,
        max_term_chars=args.max_term_chars,
        generic_noise_substrings=set(),
    )

    print(f"Loaded {len(stop_terms)} stop terms from {args.stopwords}")
    print(f"Done extraction: {n_terms} unique terms, {n_occ} occurrences.")

    compute_tf_idf_for_terms(
        db_path=args.db,
        term_candidates_table=args.term_candidates_table,
        term_occurrences_table=args.term_occurrences_table,
        max_tokens_for_tfidf=args.max_tfidf_tokens,
    )


if __name__ == "__main__":
    main()
