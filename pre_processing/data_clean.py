import argparse
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from typing import Dict, List, Set, Tuple

from .db_utils import init_db, utc_now
from . import patterns as pat


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\ufeff", "")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s


class TechnicalDocumentCleaner:
    def __init__(self, min_words: int = 4):
        self.min_words = min_words
        self.boilerplate_patterns = pat.BOILERPLATE_PATTERNS
        self.preserve_patterns = pat.PRESERVE_PATTERNS

        self.compiled_boilerplate = [re.compile(p, re.IGNORECASE) for p in self.boilerplate_patterns]
        self.compiled_preserve = [re.compile(p, re.IGNORECASE) for p in self.preserve_patterns]
        self.footer_re = re.compile(
            r'(?:^|\s)(last modified|last updated|modified:|updated:|date:)\b.*'
            r'|^©\s*copyright\s+ibm\s+corp\.?\s*\d{4}\.?$'
            r'|^(?:ibm\s+spectrum\s+lsf\s+\d+|\d+\s+ibm\s+spectrum\s+lsf)\s*$',
            re.IGNORECASE
        )

    def is_technical_content(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        norm = _norm(stripped)
        for pattern in self.compiled_preserve:
            if pattern.search(norm):
                return True
        if re.search(r'\b(download|documentation|about|help|home|back)\b', norm):
            return False
        if any(tok in stripped for tok in ['(', ')', '{', '}', '[', ']', '=', ';', '->', '::', '*', '&', '|', '^', '~', '<<', '>>', '==', '!=']):
            return True
        return False

    def is_structural_header(self, line: str) -> bool:
        norm = _norm(line)
        if not norm:
            return False
        structural_headers = pat.STRUCTURAL_HEADERS
        if norm in structural_headers or (norm.endswith(':') and norm[:-1] in structural_headers):
            return True
        for h in structural_headers:
            if norm.startswith(h + ':'):
                return True
        return False

    def is_boilerplate_line(self, line: str) -> bool:
        norm = _norm(line)
        if not norm:
            return True
        if self.is_technical_content(line) or self.is_structural_header(line):
            return False
        for pattern in self.compiled_boilerplate:
            if pattern.search(norm):
                return True
        word_count = len(norm.split())
        if 0 < word_count < self.min_words:
            if self.is_structural_header(line) or self.is_technical_content(line):
                return False
            alpha_ratio = sum(c.isalpha() for c in norm) / max(1, len(norm))
            if alpha_ratio < 0.6:
                return True
            if norm in {'home', 'back', 'next', 'previous', 'menu', 'contents', 'index', 'search', 'login', 'logout', 'help'}:
                return True
        return False

    def find_repeated_headers(self, lines: List[str], threshold: int = 3) -> Set[str]:
        line_counts = Counter(_norm(line) for line in lines if _norm(line))
        repeated: Set[str] = set()
        for norm_line, count in line_counts.items():
            if count >= threshold and len(norm_line.split()) <= 15:
                if not self.is_structural_header(norm_line) and not any(p.search(norm_line) for p in self.compiled_preserve):
                    repeated.add(norm_line)
        return repeated

    def clean_document(self, text: str) -> Tuple[str, Dict]:
        text = text.lstrip('\ufeff')
        lines = text.split('\n')

        stats = {
            'lines_total': len(lines),
            'lines_removed_boilerplate': 0,
            'lines_removed_repeated': 0,
            'lines_removed_leading': 0,
            'lines_technical_preserved': 0,
            'lines_structural_preserved': 0,
            'lines_kept': 0
        }

        first_proper_line_idx = 0
        for i, line in enumerate(lines):
            norm = _norm(line)
            if norm and len(norm.split()) > 5:
                first_proper_line_idx = i
                break
        stats['lines_removed_leading'] = first_proper_line_idx
        lines = lines[first_proper_line_idx:]

        n = len(lines)
        repeat_threshold = 2 if n < 50 else 3 if n < 200 else 4
        repeated_headers = self.find_repeated_headers(lines, threshold=repeat_threshold)

        cleaned_lines: List[str] = []
        for line in lines:
            norm = _norm(line)

            if norm in repeated_headers:
                stats['lines_removed_repeated'] += 1
                continue
            if re.search(r'^©\s*copyright\s+ibm\s+corp\.?\s*\d{4}\.?$', norm, re.IGNORECASE):
                stats['lines_removed_boilerplate'] += 1
                continue
            if re.search(r'^(?:ibm\s+spectrum\s+lsf\s+\d+|\d+\s+ibm\s+spectrum\s+lsf)\s*$', norm, re.IGNORECASE):
                stats['lines_removed_boilerplate'] += 1
                continue

            if self.is_technical_content(line):
                stats['lines_technical_preserved'] += 1
                cleaned_lines.append(line.replace("\ufeff", ""))
                stats['lines_kept'] += 1
                continue

            if self.is_structural_header(line):
                stats['lines_structural_preserved'] += 1
                out = line.replace("\ufeff", "")
                if not out.strip().endswith(':'):
                    out = out.rstrip() + ':'
                cleaned_lines.append(out)
                stats['lines_kept'] += 1
                continue

            if self.is_boilerplate_line(line):
                stats['lines_removed_boilerplate'] += 1
                continue

            cleaned_lines.append(line.replace("\ufeff", ""))
            stats['lines_kept'] += 1

        while cleaned_lines and self.footer_re.search(_norm(cleaned_lines[-1])):
            cleaned_lines.pop()
            stats['lines_removed_boilerplate'] += 1

        if not any(s.strip() for s in cleaned_lines) and any(s.strip() for s in lines):
            cleaned_lines = lines
            stats['lines_kept'] = len(lines)

        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()
        return cleaned_text, stats


def clean_into_db(
    db_path: str,
    raw_table: str,
    cleaned_table: str,
    raw_version: int,
    cleaned_version: int,
    min_words: int = 4,
) -> int:
    init_db(db_path, raw_table=raw_table, cleaned_table=cleaned_table)
    cleaner = TechnicalDocumentCleaner(min_words=min_words)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.execute(
        f"""
        SELECT d.doc_id, d.title, d.text, d.version
        FROM {raw_table} d
        WHERE d.version = ?
          AND NOT EXISTS (
              SELECT 1 FROM {cleaned_table} c
              WHERE c.doc_id = d.doc_id
                AND c.cleaned_version = ?
          )
        """,
        (raw_version, cleaned_version),
    )
    rows = cur.fetchall()

    if not rows:
        print("Nothing to clean.")
        conn.close()
        return 0

    now = utc_now()
    total = 0

    for doc_id, title, raw_text, r_version in rows:
        cleaned_text, stats = cleaner.clean_document(raw_text)
        cur.execute(
            f"""
            INSERT INTO {cleaned_table}
                (doc_id, title, cleaned_text, raw_version,
                 cleaned_version, created_at, stats_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, title, cleaned_text, r_version, cleaned_version, now, json.dumps(stats, ensure_ascii=False)),
        )

        before_len = len(raw_text)
        after_len = len(cleaned_text)
        reduction_pct = 100 * (1 - after_len / before_len) if before_len else 0
        print(f"{doc_id}: {before_len:,} → {after_len:,} chars ({reduction_pct:.1f}% reduction)")
        total += 1

    conn.commit()
    conn.close()
    print(f"Updated {total} doc(s) into {cleaned_table} (cleaned_version={cleaned_version})")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--raw_table", default="raw_documents")
    ap.add_argument("--cleaned_table", default="cleaned_documents")
    ap.add_argument("--raw_version", type=int, default=1)
    ap.add_argument("--cleaned_version", type=int, default=1)
    ap.add_argument("--min_words", type=int, default=4)
    args = ap.parse_args()

    clean_into_db(
        db_path=args.db,
        raw_table=args.raw_table,
        cleaned_table=args.cleaned_table,
        raw_version=args.raw_version,
        cleaned_version=args.cleaned_version,
        min_words=args.min_words,
    )


if __name__ == "__main__":
    main()
