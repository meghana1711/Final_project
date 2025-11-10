import json
import re
from typing import List, Dict, Set
from collections import Counter

class BoilerplateRemover:
    def __init__(self, custom_prefixes: List[str] = None):
        # Custom prefixes to remove from beginning of sentences
        self.custom_prefixes = custom_prefixes or []
        
        # Common boilerplate patterns
        self.boilerplate_patterns = [
            r'^page\s+\d+',
            r'^chapter\s+\d+',
            r'^\d+\s*$',  # standalone numbers
            r'^table of contents',
            r'^references\s*$',
            r'^bibliography\s*$',
            r'^appendix\s+[a-z]',
            r'^figure\s+\d+',
            r'^table\s+\d+',
            r'copyright\s+©',
            r'all rights reserved',
            r'^\[?\d+\]?\s*$',  # citation numbers
            r'^see also:?',
            r'^retrieved from',
            r'^available at:',
            r'^\s*\*\s*\*\s*\*\s*$',  # separator lines
            r'^_{3,}',  # underscores
            r'^-{3,}',  # dashes
            r'^={3,}',  # equals signs
            r'^\.{3,}',  # dots
        ]
        
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.boilerplate_patterns]
    
    def is_complete_sentence(self, line: str) -> bool:
        """Check if line looks like a complete sentence."""
        stripped = line.strip()
        
        if not stripped:
            return False
    
        
        # Should contain mostly alphabetic characters
        alpha_chars = sum(c.isalpha() for c in stripped)
        if alpha_chars < 10:  # At least 10 letters
            return False
        
        # Check for sentence-ending punctuation or capital letter start
        # (indicating it's part of proper prose)
        has_punctuation = any(stripped.endswith(p) for p in '.!?;:')
        starts_with_capital = stripped[0].isupper()
        
        # Either ends with punctuation OR starts with capital (for sentence fragments)
        if not (has_punctuation and starts_with_capital):
            return False
        
        return True
    
    def is_boilerplate_line(self, line: str) -> bool:
        """Check if a line is likely boilerplate or fragment."""
        stripped = line.strip()
        
        # Empty lines
        if not stripped:
            return True
        
        # Match specific boilerplate patterns first
        for pattern in self.compiled_patterns:
            if pattern.search(stripped):
                return True
        
        # Check for URL-like content
        if re.search(r'https?://|www\.', stripped, re.IGNORECASE):
            return True
        
        # Check for lines that are mostly numbers or special characters
        if len(stripped) > 0:
            alpha_ratio = sum(c.isalpha() for c in stripped) / len(stripped)
            if alpha_ratio < 0.4:  # Less than 40% alphabetic
                return True
        
        # Check if it's a complete sentence
        if not self.is_complete_sentence(line):
            return True
        
        return False
    
    def find_repeated_headers(self, lines: List[str], threshold: int = 3) -> Set[str]:
        """Find lines that repeat frequently (likely headers/footers)."""
        line_counts = Counter(line.strip() for line in lines if line.strip())
        repeated = {line for line, count in line_counts.items() 
                   if count >= threshold and len(line.split()) <= 10}
        return repeated
    
    def clean_text_blocks(self, text: str) -> str:
        """Group text into paragraphs and keep only substantial blocks."""
        # Split into paragraphs (separated by blank lines)
        paragraphs = re.split(r'\n\s*\n', text)
        
        cleaned_paragraphs = []
        for para in paragraphs:
            # Combine lines in paragraph
            para_text = ' '.join(line.strip() for line in para.split('\n') if line.strip())
            
            # Keep paragraph only if it's substantial
            # At least 50 characters and 5 words
            words = para_text.split()
            if len(para_text) >= 20 or len(words) > 5:
                # Check that it's mostly readable text
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in para_text) / len(para_text)
                if alpha_ratio > 0.7:  # At least 70% letters and spaces
                    cleaned_paragraphs.append(para_text)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    def remove_custom_prefixes(self, text: str) -> str:
        """Remove custom prefixes from the beginning of text and lines."""
        if not self.custom_prefixes:
            return text
        
        # Remove from beginning of entire text
        for prefix in self.custom_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
        
        # Remove from beginning of each line
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = line
            for prefix in self.custom_prefixes:
                if cleaned_line.strip().startswith(prefix):
                    cleaned_line = cleaned_line.strip()[len(prefix):].lstrip()
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def remove_boilerplate(self, text: str) -> str:
        """Remove boilerplate from text."""
        # First remove custom prefixes
        text = self.remove_custom_prefixes(text)
        
        lines = text.split('\n')
        
        # Find repeated headers/footers
        repeated_headers = self.find_repeated_headers(lines)
        
        # Filter lines
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            
            # Skip if it's a repeated header
            if stripped in repeated_headers:
                continue
            
            # Skip if it's boilerplate or incomplete
            if self.is_boilerplate_line(line):
                continue
            
            cleaned_lines.append(line)
        
        # Join lines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Additional pass: group into paragraphs and filter
        cleaned_text = self.clean_text_blocks(cleaned_text)
        
        # Remove excessive whitespace
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
        
        return cleaned_text.strip()
    
    def process_documents(self, input_file: str, output_file: str) -> List[Dict[str, str]]:
        """Process all documents and remove boilerplate."""
        # Load documents
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"Processing {len(documents)} documents...")
        
        cleaned_docs = []
        stats = {
            'total_chars_before': 0,
            'total_chars_after': 0,
            'docs_processed': 0
        }
        
        for doc in documents:
            original_text = doc['text']
            cleaned_text = self.remove_boilerplate(original_text)
            
            # Update stats
            stats['total_chars_before'] += len(original_text)
            stats['total_chars_after'] += len(cleaned_text)
            stats['docs_processed'] += 1
            
            # Calculate reduction
            reduction = 100 * (1 - len(cleaned_text) / len(original_text)) if original_text else 0
            
            print(f"  {doc['doc_id']}: {len(original_text)} → {len(cleaned_text)} chars ({reduction:.1f}% reduction)")
            
            # Create cleaned document
            cleaned_doc = {
                'doc_id': doc['doc_id'],
                'title': doc['title'],
                'text': cleaned_text
            }
            
            cleaned_docs.append(cleaned_doc)
        
        # Save cleaned documents
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_docs, f, indent=2, ensure_ascii=False)
        
        # Print summary
        if stats['total_chars_before'] > 0:
            total_reduction = 100 * (1 - stats['total_chars_after'] / stats['total_chars_before'])
        else:
            total_reduction = 0
            
        print(f"\n✓ Saved {len(cleaned_docs)} cleaned documents to {output_file}")
        print(f"\nSummary:")
        print(f"  Total chars before: {stats['total_chars_before']:,}")
        print(f"  Total chars after: {stats['total_chars_after']:,}")
        print(f"  Overall reduction: {total_reduction:.1f}%")
        
        return cleaned_docs


def preview_comparison(input_file: str, output_file: str, doc_index: int = 0):
    """Preview before/after comparison of a document."""
    with open(input_file, 'r', encoding='utf-8') as f:
        original_docs = json.load(f)
    
    with open(output_file, 'r', encoding='utf-8') as f:
        cleaned_docs = json.load(f)
    
    if doc_index >= len(original_docs):
        print(f"Document index {doc_index} not found")
        return
    
    orig = original_docs[doc_index]
    clean = cleaned_docs[doc_index]
    
    print(f"\n{'='*60}")
    print(f"Document: {orig['doc_id']} - {orig['title']}")
    print(f"{'='*60}")
    
    print(f"\nORIGINAL (first 500 chars):")
    print(f"{'-'*60}")
    print(orig['text'][:500])
    
    print(f"\n\nCLEANED (first 500 chars):")
    print(f"{'-'*60}")
    print(clean['text'][:500])
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    INPUT_FILE = "documents.json"  # Output from step 1
    OUTPUT_FILE = "documents_cleaned.json"
    
    # Define custom prefixes to remove
    CUSTOM_PREFIXES = [
        "Slurm Workload Manager Support and Training",
        # Add more prefixes here as needed
    ]
    
    # Create remover and process documents
    remover = BoilerplateRemover(custom_prefixes=CUSTOM_PREFIXES)
    cleaned_docs = remover.process_documents(INPUT_FILE, OUTPUT_FILE)
    
    # Show before/after comparison
    if cleaned_docs:
        preview_comparison(INPUT_FILE, OUTPUT_FILE, doc_index=0)