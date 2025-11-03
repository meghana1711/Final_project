import os
import json
import re
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
import unicodedata
from typing import List, Dict, Set, Tuple
from collections import Counter


# Create ontology_workspace.db
def init_db(db_path: str) -> None:
    """
    Create ontology_workspace.db (if missing) and required tables.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    # raw_documents: source of truth from ingest
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_documents (
            doc_id        TEXT PRIMARY KEY,
            title         TEXT,
            text          TEXT,
            content_hash  TEXT UNIQUE,
            version       INTEGER,
            created_at    TEXT
        )
    """)

    # cleaned_documents: 1 cleaned row per doc_id (latest), FK to raw
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cleaned_documents (
            doc_id           TEXT PRIMARY KEY,
            title            TEXT,
            cleaned_text     TEXT,
            raw_version      INTEGER,
            cleaned_version  INTEGER,
            created_at       TEXT,
            stats_json       TEXT,
            FOREIGN KEY (doc_id) REFERENCES raw_documents(doc_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()


# Basic whitespace normalizer
def normalize_whitespace(s: str) -> str:
    """
    Basic whitespace normalizer used at ingest time before hashing.
    """
    if s is None:
        return ""
    s = s.replace("\u2028", "\n").replace("\u2029", "\n")  # Unicode line/para sep -> newline
    s = s.replace("\u00A0", " ")  # NBSP -> space
    s = re.sub(r"[^\S\n\r\t]", " ", s)  # odd whitespace -> space (except \n\r\t)
    s = re.sub(r"[ \t]+", " ", s)      # collapse spaces/tabs
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s


def hash_text(text: str) -> str:
    """
    Stable fingerprint of cleaned text (for dedupe & ID).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Read .txt files from folder_path
def read_data(folder_path: str, db_path: str, version: int = 1) -> List[Dict[str, str]]:
    """
    Read .txt files from folder_path, normalize, hash, insert into SQLite (raw_documents).
    Returns a list of newly inserted rows' metadata.
    """
    init_db(db_path)

    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        print(f"Warning: No .txt files found in {folder_path}")
        return []

    print(f"Found {len(txt_files)} .txt files in {folder_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    inserted_docs: List[Dict[str, str]] = []
    count_new = 0

    for file_path in txt_files:
        try:
            raw_text = Path(file_path).read_text(encoding="utf-8")
            if not raw_text.strip():
                print(f"Skipping empty file: {file_path.name}")
                continue

            # normalize BEFORE hashing & storing
            cleaned_for_hash = normalize_whitespace(raw_text).strip()
            if not cleaned_for_hash:
                print(f"Skipping empty-after-normalize file: {file_path.name}")
                continue

            h = hash_text(cleaned_for_hash)
            doc_id = f"doc_{h[:12]}"  # stable, content-based ID
            title = file_path.stem
            created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"

            before = conn.total_changes
            cur.execute("""
                INSERT OR IGNORE INTO raw_documents
                    (doc_id, title, text, content_hash, version, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_id, title, cleaned_for_hash, h, version, created_at))
            inserted = (conn.total_changes > before)

            if inserted:
                count_new += 1
                inserted_docs.append({
                    "doc_id": doc_id,
                    "title": title,
                    "version": version,
                    "created_at": created_at
                })
                print(f"[NEW] {file_path.name} -> {doc_id} ({len(cleaned_for_hash)} chars)")
            else:
                print(f"[SKIP exists] {file_path.name} (duplicate content)")

        except Exception as e:
            print(f"[ERROR] {file_path.name}: {e}")
            continue

    conn.commit()
    conn.close()

    print(f"\nDone. Inserted {count_new} new document(s) into {db_path} with version={version}.")
    return inserted_docs



# Normalize a line
def _norm(s: str) -> str:
    """Normalize a line for robust matching while keeping the original for output."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)  # normalize width/quotes/dashes
    s = s.replace("\ufeff", "")           # strip BOM
    s = s.replace("\xa0", " ")            # NBSP -> space
    s = re.sub(r"\s+", " ", s.strip())    # collapse whitespace
    return s



# Cleaning documents
class TechnicalDocumentCleaner:
    def __init__(self, min_words: int = 4):
        self.min_words = min_words
        self.boilerplate_patterns = [
            r'^\s*(navigation|menu|home|back to top|skip to content)\s*$',
            r'^\s*version\s+[\d.]+\s*$',
            r'^\s*v\d+\.\d+\s*$',
            r'^\s*release\s+\d+\s*$',
            r'^\s*(last modified|last updated|modified:|updated:|date:)\b.*$',
            r'^\s*\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\s*$',
            r'^\s*\d{4}-\d{2}-\d{2}\s*$',
            r'^\s*page\s+\d+\s*$',
            r'^\s*chapter\s+\d+\s*$',
            r'^\s*\d+\s*$',
            r'^\s*(contents|table of contents)\s*$',
            r'^\s*(references|bibliography)\s*$',
            r'^\s*appendix\s+[a-z]\s*$',
            r'^\s*(figure|table)\s+\d+\s*$',
            r'^\s*©\s*copyright\s+ibm\s+corp\.?\s*\d{4}\.?\s*$',
            r'^\s*(?:ibm\s+spectrum\s+lsf\s+\d+|\d+\s+ibm\s+spectrum\s+lsf)\s*$',
            r'copyright\s+©',
            r'©\s*\d{4}',
            r'\ball rights reserved\b',
            r'^\s*\[?\d+\]?\s*$',
            r'^\s*(see also:?|retrieved from|available at:|more information)\b.*$',
            r'^\s*[\*_\-=#{3,}]+\s*$',
            r'^\s*(about|overview|using|installing|get(ting)? help|get(ting)? started|documentation|mailing lists?|support( and training)?|training|troubleshooting|faq|faqs|publications?|downloads?|installation guide|release notes|changelog|related software)\s*$',
            r'^\s*slurm workload manager\s*$',
            r'^\s*schedmd\s*$',
        ]
        self.preserve_patterns = [
            r'^\s*(?:int|void|char|const|static|extern|uint\d+_t|bool|float|double|struct|enum|typedef|union|long|short|unsigned|signed)\b',
            r'^\s*[a-z_][a-z0-9_]*\s*\([^)]*\)\s*;?$',
            r'\b[a-z_][a-z0-9_]*\s*\([^)]*\)\s*;?$',
            r'^\s*[A-Z_][A-Z0-9_]+\b',
            r'^\s*#\s*define\b',
            r'^\s*#\s*include\b',
            r'^\s*(arguments?|returns?|description|parameters?|example|note|warning|syntax|usage|input|output|overview):\s*$',
            r'^\s*(api\s+(?:functions|methods|calls)|function\s+(?:reference|list)|method\s+(?:reference|list))\s*$',
            r'\b(?:SLURM_SUCCESS|SLURM_ERROR|SUCCESS|ERROR|FAILURE|OK)\b',
            r'\([^)]*(?:input|output|in|out|inout)[^)]*\)',
            r'^\s*//',
            r'^\s*/\*',
            r'\*/\s*$',
        ]
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
        structural_headers = {
            'api functions', 'api', 'functions', 'methods',
            'description', 'arguments', 'returns', 'parameters',
            'examples', 'example', 'syntax', 'usage',
            'notes', 'note', 'warnings', 'warning',
            'input', 'output', 'configuration', 'options', 'specifications'
        }
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

        # leading trim until >5-word line
        first_proper_line_idx = 0
        for i, line in enumerate(lines):
            norm = _norm(line)
            if norm and len(norm.split()) > 5:
                first_proper_line_idx = i
                break
        stats['lines_removed_leading'] = first_proper_line_idx
        lines = lines[first_proper_line_idx:]

        # adaptive threshold
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

    # FIXED: added self; uses self.clean_document
    def clean_into_db(self, db_path: str, raw_version: int, cleaned_version: int):
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        # fetch docs of given raw_version that don't yet have this cleaned_version
        cur.execute("""
            SELECT d.doc_id, d.title, d.text, d.version
            FROM raw_documents d
            WHERE d.version = ?
              AND NOT EXISTS (
                  SELECT 1 FROM cleaned_documents c
                  WHERE c.doc_id = d.doc_id
                    AND c.cleaned_version = ?
              )
        """, (raw_version, cleaned_version))
        rows = cur.fetchall()

        if not rows:
            print("Nothing to clean.")
            conn.close()
            return []

        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cleaned_batch = []

        for doc_id, title, raw_text, r_version in rows:
            cleaned_text, stats = self.clean_document(raw_text)

            cur.execute("""
                INSERT OR REPLACE INTO cleaned_documents
                    (doc_id, title, cleaned_text, raw_version, cleaned_version, created_at, stats_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                title,
                cleaned_text,
                r_version,
                cleaned_version,
                now,
                json.dumps(stats, ensure_ascii=False)
            ))

            before_len = len(raw_text)
            after_len = len(cleaned_text)
            reduction_pct = 100 * (1 - after_len / before_len) if before_len else 0
            print(f"{doc_id}: {before_len:,} → {after_len:,} chars ({reduction_pct:.1f}% reduction)")

            cleaned_batch.append({
                "doc_id": doc_id,
                "title": title,
                "raw_version": r_version,
                "cleaned_version": cleaned_version,
                "created_at": now
            })

        conn.commit()
        conn.close()
        print(f"✓ Wrote {len(cleaned_batch)} cleaned docs into cleaned_documents (cleaned_version={cleaned_version})")
        return cleaned_batch

