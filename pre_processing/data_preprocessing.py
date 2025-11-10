import json
import re
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime
import unicodedata
from typing import List, Dict, Set, Tuple
from collections import Counter
from . import patterns as pat
import spacy
from spacy.language import Language

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

    # leaned row per cleaned_id (latest), FK is doc_id
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cleaned_documents (
            cleaned_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id          TEXT NOT NULL,
            title           TEXT,
            cleaned_text    TEXT,
            raw_version     INTEGER,
            cleaned_version INTEGER,
            created_at      TEXT,
            stats_json      TEXT,
            FOREIGN KEY (doc_id) REFERENCES raw_documents(doc_id)
                ON UPDATE CASCADE
                ON DELETE CASCADE
        )
    """)

      # Sentence Segmentation table per sentence_id is PK, doc_id is FK
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentence_segmented (
        sentence_id     TEXT PRIMARY KEY,   
        doc_id          TEXT NOT NULL,
        sent_idx        INTEGER NOT NULL,   
        sentence        TEXT NOT NULL,
        start_char      INTEGER,
        end_char        INTEGER,
        length          INTEGER,
        cleaned_version INTEGER,           
        created_at      TEXT,
        FOREIGN KEY (doc_id) REFERENCES raw_documents(doc_id)
            ON UPDATE CASCADE
            ON DELETE CASCADE
        )
    """)

    # Sentence Lemmatizing table per sentenclemma_id is PK, doc_id is FK
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sentence_lemmatized (
        lemma_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id                      TEXT NOT NULL,
        sent_idx                    INTEGER NOT NULL,
        sentence                    TEXT NOT NULL,
        tokens_json                 TEXT,  
        lemmas_json                 TEXT,   
        lemmas_with_case_json       TEXT,  
        lemmatized_text             TEXT,   
        lemmatized_text_with_case   TEXT,  
        pos_tags_json               TEXT,   
        cleaned_version             INTEGER,
        created_at                  TEXT,
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
            doc_id = f"doc_{h[:4]}" 
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



# Normalize a line with containing width/quotes/dashes
def _norm(s: str) -> str:
    """Normalize a line for robust matching while keeping the original for output."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)  # 
    s = s.replace("\ufeff", "")           # strip BOM
    s = s.replace("\xa0", " ")            # NBSP -> space
    s = re.sub(r"\s+", " ", s.strip())    # collapse whitespace
    return s


# Cleaning documents
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

    # uses self.clean_document
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
                INSERT INTO cleaned_documents
                    (doc_id, title, cleaned_text, raw_version,
                    cleaned_version, created_at, stats_json)
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
        print(f"Updated {len(cleaned_batch)} pre_processed docs into cleaned_documents (cleaned_version={cleaned_version})")
        return cleaned_batch

#Sentence segmentation 
class ImprovedSentenceSegmenter:
    def __init__(self, model_name, max_length):
        try:
            self.nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}")
        except OSError:
            print(f"Model '{model_name}' not found. Run: python -m spacy download {model_name}")
            raise

        self.nlp.max_length = max_length
        disable_pipes = ['ner', 'lemmatizer', 'textcat']
        for pipe in disable_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipes(pipe)

        self._add_custom_sentencizer_rules()
        print(f"Active pipes: {self.nlp.pipe_names}")

    def _add_custom_sentencizer_rules(self):
        """Register and add custom sentence-boundary logic to spaCy."""
        self.abbreviations = pat.ABBREVIATIONS
        self.non_boundary_patterns = pat.NON_BOUNDARY

        abbreviations = self.abbreviations
        non_boundary_patterns = self.non_boundary_patterns

        def custom_sentencizer(doc):
            for i, token in enumerate(doc[:-1]):
                if token.text in '.!?':
                    prev_text = doc[max(0, i - 5):i + 1].text.lower()
                    next_token = doc[i + 1]

                    is_abbrev = any(abbrev in prev_text for abbrev in abbreviations)
                    context = doc[max(0, i - 2):min(len(doc), i + 3)].text
                    is_non_boundary = any(
                        re.search(pattern, context, re.IGNORECASE)
                        for pattern in non_boundary_patterns
                    )

                    if not is_abbrev and not is_non_boundary and next_token.is_alpha:
                        next_token.is_sent_start = next_token.text[0].isupper()
                    else:
                        next_token.is_sent_start = False
            return doc

        if not Language.has_factory("custom_sentencizer"):
            Language.component("custom_sentencizer", func=custom_sentencizer)

        if "custom_sentencizer" not in self.nlp.pipe_names:
            try:
                self.nlp.add_pipe("custom_sentencizer", before="parser")
            except ValueError:
                self.nlp.add_pipe("custom_sentencizer", first=True)

    def _is_valid_sentence(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False

        if len(text) < 10:
            if not (text[0].isupper() and text[-1] in '.!?'):
                return False

        if len(text) > 800:
            return False

        fragment_patterns = pat.FRAGMENT_PATTERNS
        for pattern in fragment_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False

        if not re.search(r'[a-zA-Z]', text):
            return False

        words = text.split()
        if len(words) == 1 and len(text) < 20:
            if not (text[-1] in '.!?' or len(text) > 5):
                return False

        return True

    def _split_long_sentence(self, text: str, doc_id: str, sent_counter: int, start_offset: int) -> List[Dict]:
        if len(text) <= 800:
            return [{
                'doc_id': doc_id,
                'sent_idx': sent_counter,
                'sentence': text,
                'start_char': start_offset,
                'end_char': start_offset + len(text),
                'length': len(text)
            }]

        sentences = []
        sub_counter = 0

        split_patterns = pat.SPLIT_PATTERNS
        remaining_text = text

        for pattern in split_patterns:
            if len(remaining_text) <= 800:
                break

            parts = re.split(pattern, remaining_text)
            if len(parts) > 1:
                new_parts = []
                current = ""

                for part in parts:
                    if current and len(current + part) > 600:
                        new_parts.append(current.strip())
                        current = part
                    else:
                        current += part

                if current:
                    new_parts.append(current.strip())

                offset = start_offset
                for part in new_parts:
                    if self._is_valid_sentence(part):
                        sub_counter += 1
                        sentences.append({
                            'doc_id': doc_id,
                            'sent_idx': sent_counter,  # keep same index; you could encode sub-counter if you like
                            'sentence': part,
                            'start_char': offset,
                            'end_char': offset + len(part),
                            'length': len(part)
                        })
                    offset += len(part)

                return sentences or [{
                    'doc_id': doc_id,
                    'sent_idx': sent_counter,
                    'sentence': text,
                    'start_char': start_offset,
                    'end_char': start_offset + len(text),
                    'length': len(text)
                }]

        return [{
            'doc_id': doc_id,
            'sent_idx': sent_counter,
            'sentence': text,
            'start_char': start_offset,
            'end_char': start_offset + len(text),
            'length': len(text)
        }]

    def segment_text(self, text: str, doc_id: str) -> List[Dict]:
        if len(text) > self.nlp.max_length:
            return self._segment_large_text(text, doc_id)

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        doc = self.nlp(text)

        sentences = []
        sent_counter = 0

        for sent in doc.sents:
            sentence_text = sent.text.strip()
            if not sentence_text:
                continue
            if sentence_text.endswith(':') and len(sentence_text.split()) < 10:
                continue
            if not self._is_valid_sentence(sentence_text):
                continue

            sent_counter += 1

            if len(sentence_text) > 800:
                split_sentences = self._split_long_sentence(
                    sentence_text, doc_id, sent_counter, sent.start_char
                )
                sentences.extend(split_sentences)
            else:
                sentences.append({
                    'doc_id': doc_id,
                    'sent_idx': sent_counter,
                    'sentence': sentence_text,
                    'start_char': sent.start_char,
                    'end_char': sent.end_char,
                    'length': len(sentence_text)
                })

        return sentences

    def _segment_large_text(self, text: str, doc_id: str) -> List[Dict]:
        chunk_size = int(self.nlp.max_length * 0.8)
        overlap_size = 1000
        all_sentences = []
        sentence_counter = 0

        i = 0
        while i < len(text):
            end = min(i + chunk_size, len(text))

            if end < len(text):
                last_para = text.rfind('\n\n', i, end)
                if last_para > i + chunk_size // 2:
                    end = last_para + 2
                else:
                    for punct in ['. ', '! ', '? ']:
                        last_sent = text.rfind(punct, i + chunk_size // 2, end)
                        if last_sent > i:
                            end = last_sent + 2
                            break

            chunk = text[i:end]
            doc = self.nlp(chunk)

            for sent in doc.sents:
                sentence_text = sent.text.strip()
                if not sentence_text:
                    continue
                if not self._is_valid_sentence(sentence_text):
                    continue

                abs_start = i + sent.start_char
                abs_end = i + sent.end_char

                is_duplicate = False
                for existing in all_sentences[-5:]:
                    if abs(existing['start_char'] - abs_start) < 50:
                        is_duplicate = True
                        break
                if is_duplicate:
                    continue

                sentence_counter += 1

                if len(sentence_text) > 800:
                    split_sentences = self._split_long_sentence(
                        sentence_text, doc_id, sentence_counter, abs_start
                    )
                    all_sentences.extend(split_sentences)
                else:
                    all_sentences.append({
                        'doc_id': doc_id,
                        'sent_idx': sentence_counter,
                        'sentence': sentence_text,
                        'start_char': abs_start,
                        'end_char': abs_end,
                        'length': len(sentence_text)
                    })

            next_start = end - overlap_size if end < len(text) else end
            i = max(next_start, i + chunk_size // 2)

        return all_sentences

    def segment_cleaned_to_db(self, db_path: str, cleaned_version: int):
        """
        Read cleaned documents from cleaned_documents
        and write sentence rows into sentence_segmented table.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        cur.execute("""
            SELECT cd.doc_id, cd.cleaned_text
            FROM cleaned_documents cd
            WHERE cd.cleaned_version = ?
              AND NOT EXISTS (
                  SELECT 1 FROM sentence_segmented s
                  WHERE s.doc_id = cd.doc_id
                    AND s.cleaned_version = cd.cleaned_version
              )
        """, (cleaned_version,))
        rows = cur.fetchall()

        if not rows:
            print(f"No cleaned docs to segment for cleaned_version={cleaned_version}.")
            conn.close()
            return []

        print(f"Segmenting {len(rows)} cleaned document(s) from DB...")
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        inserted_total = 0

        for doc_id, cleaned_text in rows:
            sentences = self.segment_text(cleaned_text, doc_id)
            print(f"  {doc_id}: {len(sentences)} sentences")

            for s in sentences:
                cur.execute("""
                    INSERT INTO sentence_segmented
                        (doc_id, sent_idx, sentence, start_char, end_char,
                         length, cleaned_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    s['doc_id'],
                    s['sent_idx'],
                    s['sentence'],
                    s['start_char'],
                    s['end_char'],
                    s['length'],
                    cleaned_version,
                    now
                ))
            inserted_total += len(sentences)

        conn.commit()
        conn.close()
        print(f"Inserted {inserted_total} sentence rows into sentence_segmented (cleaned_version={cleaned_version})")
        return inserted_total


#Sentence lemmatizaing and POS tagging
class SentenceLemmatizer:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy lemmatizer.
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}")
        except OSError:
            print(f"Model '{model_name}' not found.")
            print(f"Run: python -m spacy download {model_name}")
            raise

        # We need tokenizer + tagger + lemmatizer; disable NER & parser
        disable_pipes = ['ner', 'parser']
        for pipe in disable_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipes(pipe)

        print(f"Active pipes: {self.nlp.pipe_names}")

    def preserve_original_case(self, original_token: str, lemma: str) -> str:
        """
        Preserve the capitalization pattern of the original token in the lemma.
        """
        if not original_token or not lemma:
            return lemma

        if original_token.isupper():
            return lemma.upper()
        elif original_token[0].isupper():
            return lemma.capitalize()
        elif any(c.isupper() for c in original_token[1:]):
            if len(original_token) == len(lemma):
                result = ""
                for i, char in enumerate(lemma):
                    if i < len(original_token) and original_token[i].isupper():
                        result += char.upper()
                    else:
                        result += char.lower()
                return result
            else:
                return lemma.capitalize() if original_token[0].isupper() else lemma

        return lemma

    def lemmatize_sentence(
        self,
        sentence: str,
        keep_pos: bool = True,
        remove_stopwords: bool = False,
        remove_punct: bool = False,
    ) -> Dict:
        """
        Lemmatize a sentence and optionally filter tokens.
        """
        doc = self.nlp(sentence)

        tokens = []
        lemmas = []
        lemmas_with_case = []
        pos_tags = []

        for token in doc:
            if remove_stopwords and token.is_stop:
                continue
            if remove_punct and token.is_punct:
                continue

            tokens.append(token.text)
            lemmas.append(token.lemma_)
            case_preserved_lemma = self.preserve_original_case(token.text, token.lemma_)
            lemmas_with_case.append(case_preserved_lemma)

            if keep_pos:
                pos_tags.append(token.pos_)

        result = {
            "tokens": tokens,
            "lemmas": lemmas,
            "lemmas_with_case": lemmas_with_case,
            "lemmatized_text": " ".join(lemmas),
            "lemmatized_text_with_case": " ".join(lemmas_with_case),
        }

        if keep_pos:
            result["pos_tags"] = pos_tags

        return result

    def process_sentences_db(
        self,
        db_path: str,
        cleaned_version: int,
        keep_pos: bool = True,
        remove_stopwords: bool = False,
        remove_punct: bool = False,
        batch_size: int = 200,
    ) -> int:
        """
        Lemmatize sentences from sentence_segmented table and write into
        sentence_lemmatized table.

        Only processes sentences for given cleaned_version that don't
        already have lemma rows.
        """
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")

        # Fetch sentences to lemmatize
        cur.execute(
            """
            SELECT s.doc_id, s.sent_idx, s.sentence
            FROM sentence_segmented s
            WHERE s.cleaned_version = ?
              AND NOT EXISTS (
                  SELECT 1 FROM sentence_lemmatized l
                  WHERE l.doc_id = s.doc_id
                    AND l.sent_idx = s.sent_idx
                    AND l.cleaned_version = s.cleaned_version
              )
            ORDER BY s.doc_id, s.sent_idx
        """,
            (cleaned_version,),
        )
        rows = cur.fetchall()

        if not rows:
            print(f"No sentences to lemmatize for cleaned_version={cleaned_version}.")
            conn.close()
            return 0

        print(
            f"\nLemmatizing {len(rows)} sentences from DB "
            f"(cleaned_version={cleaned_version})..."
        )
        print(
            f" Settings → keep_pos={keep_pos}, remove_stopwords={remove_stopwords}, "
            f"remove_punct={remove_punct}"
        )

        total = 0
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]

            for doc_id, sent_idx, sentence_text in batch:
                lemma_result = self.lemmatize_sentence(
                    sentence_text,
                    keep_pos=keep_pos,
                    remove_stopwords=remove_stopwords,
                    remove_punct=remove_punct,
                )

                cur.execute(
                    """
                    INSERT OR REPLACE INTO sentence_lemmatized
                        (doc_id, sent_idx, sentence,
                         tokens_json, lemmas_json, lemmas_with_case_json,
                         lemmatized_text, lemmatized_text_with_case,
                         pos_tags_json, cleaned_version, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        doc_id,
                        sent_idx,
                        sentence_text,
                        json.dumps(lemma_result["tokens"], ensure_ascii=False),
                        json.dumps(lemma_result["lemmas"], ensure_ascii=False),
                        json.dumps(
                            lemma_result["lemmas_with_case"], ensure_ascii=False
                        ),
                        lemma_result["lemmatized_text"],
                        lemma_result["lemmatized_text_with_case"],
                        json.dumps(
                            lemma_result.get("pos_tags"), ensure_ascii=False
                        )
                        if keep_pos
                        else None,
                        cleaned_version,
                        now,
                    ),
                )
                total += 1

            conn.commit()
            processed = min(i + batch_size, len(rows))
            print(f"  Processed {processed}/{len(rows)} sentences...")

        conn.close()
        print(
            f"\n Updated {total} rows into sentence_lemmatized "
            f"(cleaned_version={cleaned_version})"
        )
        return total

