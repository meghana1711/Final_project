import json
import re
import unicodedata
from typing import List, Dict, Set, Tuple
from collections import Counter


# >>> ADDED: robust normalizer used by all match checks
def _norm(s: str) -> str:
    """Normalize a line for robust matching while keeping the original for output."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)  # normalize width/quotes/dashes
    s = s.replace("\ufeff", "")           # strip BOM
    s = s.replace("\xa0", " ")            # NBSP -> space
    s = re.sub(r"\s+", " ", s.strip())    # collapse whitespace
    return s


class TechnicalDocumentCleaner:
    def __init__(self, min_words: int = 4):
        """
        Enhanced cleaner for technical/API documentation.
        Preserves code, function signatures, and structured technical content.

        Args:
            min_words: Minimum words for non-technical content lines (default: 4)
        """
        self.min_words = min_words

        # Boilerplate patterns - navigation, metadata, UI elements
        self.boilerplate_patterns = [
            r'^\s*(navigation|menu|home|back to top|skip to content)\s*$',
            r'^\s*version\s+[\d.]+\s*$',
            r'^\s*v\d+\.\d+\s*$',
            r'^\s*release\s+\d+\s*$',
            r'^\s*(last modified|last updated|modified:|updated:|date:)\b.*$',  # case-insensitive via compile
            r'^\s*\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\s*$',
            r'^\s*\d{4}-\d{2}-\d{2}\s*$',
            r'^\s*page\s+\d+\s*$',
            r'^\s*chapter\s+\d+\s*$',
            r'^\s*\d+\s*$',
            r'^\s*(contents|table of contents)\s*$',  # include single-word Contents
            r'^\s*(references|bibliography)\s*$',
            r'^\s*appendix\s+[a-z]\s*$',
            r'^\s*(figure|table)\s+\d+\s*$',

            # --- IBM specific boilerplate we're trying to kill ---
            # >>> ADDED: match lines like "© Copyright IBM Corp. 2023."
            r'^\s*©\s*copyright\s+ibm\s+corp\.?\s*\d{4}\.?\s*$',

            # >>> ADDED: match footer-like page stamps
            # "IBM Spectrum LSF 157" OR "158 IBM Spectrum LSF"
            r'^\s*(?:ibm\s+spectrum\s+lsf\s+\d+|\d+\s+ibm\s+spectrum\s+lsf)\s*$',

            # generic copyright-y stuff, keep these too
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

        # PRESERVE these patterns - critical technical content
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

        # >>> UPDATED: footer regex
        # We now match:
        #   - "last modified ...", "updated: ..."
        #   - "© Copyright IBM Corp. 2023."
        #   - "IBM Spectrum LSF 157" / "158 IBM Spectrum LSF"
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

        norm = _norm(stripped)  # >>> ADDED: normalized matching

        for pattern in self.compiled_preserve:
            if pattern.search(norm):
                return True

        # exclude obvious nav words
        if re.search(r'\b(download|documentation|about|help|home|back)\b', norm):
            return False

        # code-like delimiters/operators
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

        # direct boilerplate matches first
        for pattern in self.compiled_boilerplate:
            if pattern.search(norm):
                return True

        # short junky lines with almost no words
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
        # >>> ADDED: count on normalized lines
        line_counts = Counter(_norm(line) for line in lines if _norm(line))
        repeated: Set[str] = set()
        for norm_line, count in line_counts.items():
            if count >= threshold and len(norm_line.split()) <= 15:
                # Don’t mark repeated if technical/structural
                if not self.is_structural_header(norm_line) and not any(p.search(norm_line) for p in self.compiled_preserve):
                    repeated.add(norm_line)
        return repeated

    def clean_document(self, text: str) -> Tuple[str, Dict]:
        # Remove BOM if present
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

        # STEP 1: Drop ALL leading lines until we hit a line with >5 words
        first_proper_line_idx = 0
        for i, line in enumerate(lines):
            norm = _norm(line)
            if norm and len(norm.split()) > 5:
                first_proper_line_idx = i
                break
        stats['lines_removed_leading'] = first_proper_line_idx
        lines = lines[first_proper_line_idx:]

        # Adaptive threshold for repeated content
        n = len(lines)
        if n < 50:
            repeat_threshold = 2
        elif n < 200:
            repeat_threshold = 3
        else:
            repeat_threshold = 4

        repeated_headers = self.find_repeated_headers(lines, threshold=repeat_threshold)

        cleaned_lines: List[str] = []
        for line in lines:
            norm = _norm(line)

            # Skip repeated nav/footer elements
            if norm in repeated_headers:
                stats['lines_removed_repeated'] += 1
                continue

            # >>> NEW: explicitly skip IBM copyright + IBM Spectrum LSF footer lines here too
            # This is redundant with boilerplate_patterns, but it's cheap and defensive.
            if re.search(r'^©\s*copyright\s+ibm\s+corp\.?\s*\d{4}\.?$', norm, re.IGNORECASE):
                stats['lines_removed_boilerplate'] += 1
                continue
            if re.search(r'^(?:ibm\s+spectrum\s+lsf\s+\d+|\d+\s+ibm\s+spectrum\s+lsf)\s*$', norm, re.IGNORECASE):
                stats['lines_removed_boilerplate'] += 1
                continue

            if self.is_technical_content(line):
                stats['lines_technical_preserved'] += 1
                cleaned_lines.append(line.replace("\ufeff", ""))  # ensure no BOM remains
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

        # ADDED / UPDATED: strip trailing junk footers from the END of the doc:
        # "last modified ...", "© Copyright IBM Corp. 2023.", "IBM Spectrum LSF 157"
        while cleaned_lines and self.footer_re.search(_norm(cleaned_lines[-1])):
            cleaned_lines.pop()
            stats['lines_removed_boilerplate'] += 1

        # Fallback: if we removed everything, keep original post-leading-trim
        if not any(s.strip() for s in cleaned_lines) and any(s.strip() for s in lines):
            cleaned_lines = lines
            stats['lines_kept'] = len(lines)

        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()

        return cleaned_text, stats

    def process_documents(self, input_file: str, output_file: str) -> List[Dict]:
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)

        print(f"Processing {len(documents)} documents (min_words={self.min_words})...\n")

        cleaned_docs = []
        overall_stats = {
            'total_chars_before': 0,
            'total_chars_after': 0,
            'total_leading_removed': 0,
            'docs_processed': 0
        }

        for doc in documents:
            original_text = doc.get('text', '')
            cleaned_text, doc_stats = self.clean_document(original_text)

            overall_stats['total_chars_before'] += len(original_text)
            overall_stats['total_chars_after'] += len(cleaned_text)
            overall_stats['total_leading_removed'] += doc_stats['lines_removed_leading']
            overall_stats['docs_processed'] += 1

            char_reduction = 100 * (1 - len(cleaned_text) / len(original_text)) if original_text else 0

            print(f"{doc.get('doc_id', '(no id)')}: {len(original_text):,} → {len(cleaned_text):,} chars "
                  f"({char_reduction:.1f}% reduction, {doc_stats['lines_removed_leading']} leading lines removed)")

            cleaned_doc = {
                'doc_id': doc.get('doc_id'),
                'title': doc.get('title'),
                'text': cleaned_text,
            }
            cleaned_docs.append(cleaned_doc)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_docs, f, indent=2, ensure_ascii=False)

        total_char_reduction = 100 * (1 - overall_stats['total_chars_after'] / max(1, overall_stats['total_chars_before']))

        print(f"\n{'='*70}")
        print(f"✓ Saved {len(cleaned_docs)} documents to {output_file}")
        print(f"  Overall reduction: {total_char_reduction:.1f}%")
        print(f"  Total leading short lines removed: {overall_stats['total_leading_removed']:,}")
        print(f"{'='*70}")

        return cleaned_docs


if __name__ == "__main__":
    INPUT_FILE = "documents_new.json"
    OUTPUT_FILE = "cleaned_new.json"

    cleaner = TechnicalDocumentCleaner(min_words=4)
    cleaned_docs = cleaner.process_documents(INPUT_FILE, OUTPUT_FILE)



