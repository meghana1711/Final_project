import json
import spacy
import re
from typing import List, Dict, Tuple
from spacy.lang.en import English

class ImprovedSentenceSegmenter:
    def __init__(self, model_name: str = "en_core_web_sm", max_length: int = 1500000):
        """
        Initialize improved spaCy sentence segmenter with custom rules.
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
            max_length: Maximum text length to process (default: 1.5MB)
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"✓ Loaded spaCy model: {model_name}")
        except OSError:
            print(f"Model '{model_name}' not found. Installing...")
            print(f"Run: python -m spacy download {model_name}")
            raise
        
        # Increase max_length to handle large documents
        self.nlp.max_length = max_length
        print(f"Set max_length to {max_length:,} characters")
        
        # Disable unnecessary pipeline components for speed
        disable_pipes = ['ner', 'lemmatizer', 'textcat']
        for pipe in disable_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipes(pipe)
        
        # Add custom sentence boundary rules
        self._add_custom_sentencizer_rules()
        
        print(f"Active pipes: {self.nlp.pipe_names}")
    
    def _add_custom_sentencizer_rules(self):
        """Add custom rules for better sentence boundary detection."""
        
        # Common abbreviations that shouldn't end sentences
        self.abbreviations = {
            'e.g.', 'i.e.', 'etc.', 'vs.', 'cf.', 'inc.', 'ltd.', 'corp.',
            'fig.', 'figs.', 'eq.', 'eqs.', 'sec.', 'secs.', 'ch.', 'chs.',
            'vol.', 'vols.', 'no.', 'nos.', 'p.', 'pp.', 'ref.', 'refs.',
            'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'rev.', 'gen.', 'col.',
            'maj.', 'capt.', 'lt.', 'sgt.', 'pvt.', 'jr.', 'sr.',
            'min.', 'max.', 'avg.', 'std.', 'var.', 'cpu.', 'gpu.',
            'api.', 'url.', 'uri.', 'sql.', 'xml.', 'json.', 'csv.',
            'kb.', 'mb.', 'gb.', 'tb.', 'hz.', 'mhz.', 'ghz.'
        }
        
        # Patterns that indicate NOT a sentence boundary
        self.non_boundary_patterns = [
            r'\b[A-Z][a-z]*\.\s+[a-z]',  # "Inc. and"
            r'\b\d+\.\s*\d',             # "3.14" or "3. 5"
            r'\b[A-Z]\.\s*[A-Z]\.',      # "U.S.A."
            r'Fig\.\s*\d+',              # "Fig. 1"
            r'Table\s*\d+\.',            # "Table 1."
            r'Section\s*\d+\.',          # "Section 1."
            r'Chapter\s*\d+\.',          # "Chapter 1."
        ]
        
        # Create custom sentencizer function
        def custom_sentencizer(doc):
            """Custom sentence boundary detection."""
            for i, token in enumerate(doc[:-1]):
                # Check if current token could end a sentence
                if token.text in '.!?':
                    # Get the context around this potential boundary
                    prev_text = doc[max(0, i-5):i+1].text.lower()
                    next_token = doc[i+1]
                    
                    # Don't split on abbreviations
                    is_abbrev = any(abbrev in prev_text for abbrev in self.abbreviations)
                    
                    # Don't split on patterns like "Fig. 1"
                    context = doc[max(0, i-2):min(len(doc), i+3)].text
                    is_non_boundary = any(re.search(pattern, context, re.IGNORECASE) 
                                        for pattern in self.non_boundary_patterns)
                    
                    # Set sentence boundary
                    if not is_abbrev and not is_non_boundary and next_token.is_alpha:
                        if next_token.text[0].isupper():
                            next_token.is_sent_start = True
                        else:
                            next_token.is_sent_start = False
                    else:
                        next_token.is_sent_start = False
            
            return doc
        
        # Try to add the component - handle different spaCy versions
        try:
            # For newer spaCy versions (3.4+)
            if "custom_sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe(custom_sentencizer, name="custom_sentencizer", before="parser")
        except ValueError:
            # For older spaCy versions, try without 'before' parameter
            try:
                if "custom_sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe(custom_sentencizer, name="custom_sentencizer", first=True)
            except Exception as e:
                print(f"Warning: Could not add custom sentencizer: {e}")
                print("Proceeding with default sentence segmentation...")
    
    def _is_valid_sentence(self, text: str) -> bool:
        """Check if a text segment is a valid sentence."""
        text = text.strip()
        
        # Minimum length check
        if len(text) < 10:
            # Allow only if it's a complete sentence with proper structure
            if not (text[0].isupper() and text[-1] in '.!?'):
                return False
        
        # Maximum length check - split very long sentences
        if len(text) > 800:
            return False
        
        # Filter out fragments that are clearly not sentences
        fragment_patterns = [
            r'^\w{1,3}$',                    # Single short words
            r'^[A-Z]+$',                     # All caps (likely acronyms)
            r'^Figure\s*\d*\.?$',            # "Figure" or "Figure 1."
            r'^Table\s*\d*\.?$',             # "Table" or "Table 1."
            r'^Example\s*\d*\.?$',           # "Example" or "Example 1."
            r'^Section\s*\d*\.?$',           # "Section" or "Section 1."
            r'^Chapter\s*\d*\.?$',           # "Chapter" or "Chapter 1."
            r'^\d+\.?\s*$',                  # Just numbers
            r'^[a-z]$',                      # Single lowercase letters
            r'^\([^)]*\)$',                  # Just parentheses content
            r'^[\[\]{}().,;:!?-]+$',         # Just punctuation
            r'^none$',                       # Common non-sentences
            r'^yes$',
            r'^no$',
        ]
        
        for pattern in fragment_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        # Must have some alphabetic content
        if not re.search(r'[a-zA-Z]', text):
            return False
        
        # Check for reasonable word count (not just one word unless special cases)
        words = text.split()
        if len(words) == 1 and len(text) < 20:
            # Single word must be substantial or end with punctuation
            if not (text[-1] in '.!?' or len(text) > 5):
                return False
        
        return True
    
    def _split_long_sentence(self, text: str, doc_id: str, sent_counter: int, start_offset: int) -> List[Dict]:
        """Split very long sentences at logical boundaries."""
        if len(text) <= 800:
            return [{
                'doc_id': doc_id,
                'sent_id': f"{doc_id}_sent_{sent_counter:04d}",
                'sentence': text,
                'start_char': start_offset,
                'end_char': start_offset + len(text),
                'length': len(text)
            }]
        
        sentences = []
        current_pos = 0
        sub_counter = 0
        
        # Try to split at natural boundaries
        split_patterns = [
            r'([.!?]\s+)(?=[A-Z])',          # Sentence endings
            r'(;\s+)(?=[A-Z])',              # Semicolons before capitals
            r'(:\s+)(?=[A-Z][a-z])',         # Colons before sentences
            r'(\n\n+)',                      # Paragraph breaks
            r'(\.\s+)(?=\d+\.)',             # Before numbered items
        ]
        
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
                
                # Create sentence objects
                offset = start_offset
                for part in new_parts:
                    if self._is_valid_sentence(part):
                        sub_counter += 1
                        sentences.append({
                            'doc_id': doc_id,
                            'sent_id': f"{doc_id}_sent_{sent_counter:04d}_{sub_counter}",
                            'sentence': part,
                            'start_char': offset,
                            'end_char': offset + len(part),
                            'length': len(part)
                        })
                    offset += len(part)
                
                return sentences if sentences else [{
                    'doc_id': doc_id,
                    'sent_id': f"{doc_id}_sent_{sent_counter:04d}",
                    'sentence': text,
                    'start_char': start_offset,
                    'end_char': start_offset + len(text),
                    'length': len(text)
                }]
        
        # If no good split found, return as single sentence
        return [{
            'doc_id': doc_id,
            'sent_id': f"{doc_id}_sent_{sent_counter:04d}",
            'sentence': text,
            'start_char': start_offset,
            'end_char': start_offset + len(text),
            'length': len(text)
        }]
    
    def segment_text(self, text: str, doc_id: str) -> List[Dict]:
        """
        Segment text into sentences with improved boundary detection.
        
        Args:
            text: Input text to segment
            doc_id: Document identifier
        
        Returns:
            List of sentence dictionaries with text and offsets
        """
        # Check if document is too large
        if len(text) > self.nlp.max_length:
            print(f"    WARNING: {doc_id} is {len(text):,} chars, splitting into chunks...")
            return self._segment_large_text(text, doc_id)
        
        # Clean text slightly
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive newlines
        text = re.sub(r' {2,}', ' ', text)      # Reduce excessive spaces
        
        doc = self.nlp(text)
        
        sentences = []
        sent_counter = 0
        
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            # Skip empty sentences
            if not sentence_text:
                continue
            
            # Skip pure colon endings (likely headers)
            if sentence_text.endswith(':') and len(sentence_text.split()) < 10:
                continue
            
            # Validate sentence
            if not self._is_valid_sentence(sentence_text):
                continue
            
            sent_counter += 1
            
            # Handle very long sentences by splitting them
            if len(sentence_text) > 800:
                split_sentences = self._split_long_sentence(
                    sentence_text, doc_id, sent_counter, sent.start_char
                )
                sentences.extend(split_sentences)
            else:
                sentence_dict = {
                    'doc_id': doc_id,
                    'sent_id': f"{doc_id}_sent_{sent_counter:04d}",
                    'sentence': sentence_text,
                    'start_char': sent.start_char,
                    'end_char': sent.end_char,
                    'length': len(sentence_text)
                }
                sentences.append(sentence_dict)
        
        return sentences
    
    def _segment_large_text(self, text: str, doc_id: str) -> List[Dict]:
        """
        Segment very large texts by processing in chunks with overlap.
        
        Args:
            text: Input text to segment
            doc_id: Document identifier
        
        Returns:
            List of sentence dictionaries
        """
        chunk_size = int(self.nlp.max_length * 0.8)  # More conservative
        overlap_size = 1000  # Overlap to avoid cutting sentences
        all_sentences = []
        sentence_counter = 0
        
        i = 0
        while i < len(text):
            # Get chunk with overlap
            end = min(i + chunk_size, len(text))
            
            # If not at end, try to break at good boundaries
            if end < len(text):
                # Look for paragraph breaks first
                last_para = text.rfind('\n\n', i, end)
                if last_para > i + chunk_size // 2:
                    end = last_para + 2
                else:
                    # Look for sentence boundaries
                    for punct in ['. ', '! ', '? ']:
                        last_sent = text.rfind(punct, i + chunk_size // 2, end)
                        if last_sent > i:
                            end = last_sent + 2
                            break
            
            chunk = text[i:end]
            
            # Process chunk
            doc = self.nlp(chunk)
            
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                
                if not sentence_text:
                    continue
                    
                if not self._is_valid_sentence(sentence_text):
                    continue
                
                # Avoid duplicates from overlap
                abs_start = i + sent.start_char
                abs_end = i + sent.end_char
                
                # Check if this sentence overlaps with already processed ones
                is_duplicate = False
                for existing in all_sentences[-5:]:  # Check last 5 sentences
                    if abs(existing['start_char'] - abs_start) < 50:
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                sentence_counter += 1
                
                # Handle long sentences
                if len(sentence_text) > 800:
                    split_sentences = self._split_long_sentence(
                        sentence_text, doc_id, sentence_counter, abs_start
                    )
                    all_sentences.extend(split_sentences)
                else:
                    sentence_dict = {
                        'doc_id': doc_id,
                        'sent_id': f"{doc_id}_sent_{sentence_counter:04d}",
                        'sentence': sentence_text,
                        'start_char': abs_start,
                        'end_char': abs_end,
                        'length': len(sentence_text)
                    }
                    all_sentences.append(sentence_dict)
            
            # Move to next chunk with overlap consideration
            next_start = end - overlap_size if end < len(text) else end
            i = max(next_start, i + chunk_size // 2)  # Ensure progress
        
        return all_sentences
    
    def process_documents(self, input_file: str, output_file: str) -> List[Dict]:
        """
        Process all documents and segment into sentences with improved quality.
        
        Args:
            input_file: JSON file with cleaned documents
            output_file: Output JSON file with segmented sentences
        
        Returns:
            List of all sentences from all documents
        """
        # Load cleaned documents
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"\nProcessing {len(documents)} documents with improved segmentation...")
        
        all_sentences = []
        stats = {
            'total_docs': len(documents),
            'total_sentences': 0,
            'total_chars': 0,
            'avg_sent_length': 0,
            'filtered_fragments': 0,
            'split_long_sentences': 0
        }
        
        for idx, doc in enumerate(documents, 1):
            doc_id = doc['doc_id']
            text = doc['text']
            title = doc.get('title', 'Untitled')
            
            if not text.strip():
                print(f"  [{idx}/{len(documents)}] {doc_id}: Skipping (empty)")
                continue
            
            # Track initial sentence count for this doc
            initial_count = len(all_sentences)
            
            # Segment document
            sentences = self.segment_text(text, doc_id)
            
            # Update stats
            stats['total_sentences'] += len(sentences)
            stats['total_chars'] += sum(s['length'] for s in sentences)
            
            # Count split sentences (those with sub-IDs)
            split_count = sum(1 for s in sentences if '_' in s['sent_id'].split('_')[-1])
            stats['split_long_sentences'] += split_count
            
            all_sentences.extend(sentences)
            
            print(f"  [{idx}/{len(documents)}] {doc_id} ({title[:50]}...): {len(sentences)} sentences")
        
        # Calculate average
        if stats['total_sentences'] > 0:
            stats['avg_sent_length'] = stats['total_chars'] / stats['total_sentences']
        
        # Save segmented sentences
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_sentences, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n✓ Saved {stats['total_sentences']} sentences to {output_file}")
        print(f"\nImproved Segmentation Summary:")
        print(f"  Total documents: {stats['total_docs']}")
        print(f"  Total sentences: {stats['total_sentences']}")
        print(f"  Average sentences/doc: {stats['total_sentences'] / stats['total_docs']:.1f}")
        print(f"  Average sentence length: {stats['avg_sent_length']:.1f} chars")
        print(f"  Long sentences split: {stats['split_long_sentences']}")
        
        return all_sentences
    
    def create_document_view(self, sentences: List[Dict], output_file: str):
        """
        Create an alternative view grouped by document.
        
        Args:
            sentences: List of sentence dictionaries
            output_file: Output JSON file
        """
        # Group sentences by document
        docs = {}
        for sent in sentences:
            doc_id = sent['doc_id']
            if doc_id not in docs:
                docs[doc_id] = {
                    'doc_id': doc_id,
                    'sentences': []
                }
            docs[doc_id]['sentences'].append({
                'sent_id': sent['sent_id'],
                'sentence': sent['sentence'],
                'start_char': sent['start_char'],
                'end_char': sent['end_char']
            })
        
        # Convert to list
        doc_list = list(docs.values())
        
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(doc_list, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created document-grouped view: {output_file}")
    
    def validate_quality(self, sentences: List[Dict]) -> Dict:
        """
        Validate the quality of sentence segmentation.
        
        Returns:
            Dictionary with quality metrics
        """
        if not sentences:
            return {}
        
        total = len(sentences)
        proper_endings = sum(1 for s in sentences if s['sentence'].rstrip()[-1] in '.!?')
        proper_starts = sum(1 for s in sentences if s['sentence'][0].isupper())
        short_sentences = sum(1 for s in sentences if s['length'] < 10)
        long_sentences = sum(1 for s in sentences if s['length'] > 600)
        
        # Check for common abbreviation splits
        abbrev_issues = 0
        for i, sent in enumerate(sentences[:-1]):
            if sent['sentence'].rstrip().endswith(('.', 'etc.')):
                next_sent = sentences[i + 1]
                if next_sent['sentence'][0].islower():
                    abbrev_issues += 1
        
        quality_metrics = {
            'total_sentences': total,
            'proper_endings_pct': (proper_endings / total) * 100,
            'proper_starts_pct': (proper_starts / total) * 100,
            'short_sentences': short_sentences,
            'long_sentences': long_sentences,
            'potential_abbrev_splits': abbrev_issues,
            'avg_length': sum(s['length'] for s in sentences) / total
        }
        
        return quality_metrics


def preview_sentences(json_file: str, num_sentences: int = 10):
    """Preview first few sentences from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    
    print(f"\nPreview of {min(num_sentences, len(sentences))} sentences:")
    print("=" * 80)
    
    for sent in sentences[:num_sentences]:
        print(f"\nSent ID: {sent['sent_id']}")
        print(f"Length: {sent['length']} chars")
        print(f"Offsets: [{sent['start_char']}, {sent['end_char']}]")
        print(f"Text: {sent['sentence']}")
        print("-" * 80)


def validate_segmentation(input_file: str, sentences_file: str, doc_id: str):
    """
    Validate segmentation by checking reconstruction accuracy.
    
    Args:
        input_file: Original cleaned documents file
        sentences_file: Segmented sentences file
        doc_id: Document ID to validate
    """
    # Load original
    with open(input_file, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    
    original = next((d for d in docs if d['doc_id'] == doc_id), None)
    if not original:
        print(f"Document {doc_id} not found")
        return
    
    # Load sentences
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    
    doc_sentences = [s for s in sentences if s['doc_id'] == doc_id]
    doc_sentences.sort(key=lambda x: x['start_char'])
    
    print(f"\nValidation for {doc_id}:")
    print(f"  Original length: {len(original['text'])} chars")
    print(f"  Number of sentences: {len(doc_sentences)}")
    print(f"  Coverage: {doc_sentences[-1]['end_char'] if doc_sentences else 0}/{len(original['text'])}")
    
    # Check for gaps
    gaps = []
    for i in range(len(doc_sentences) - 1):
        if doc_sentences[i]['end_char'] < doc_sentences[i + 1]['start_char']:
            gaps.append((doc_sentences[i]['end_char'], doc_sentences[i + 1]['start_char']))
    
    if gaps:
        print(f"  Found {len(gaps)} gaps in coverage")
    else:
        print("  ✓ No gaps in sentence coverage")
    
    print(f"\nFirst 3 sentences:")
    for i, sent in enumerate(doc_sentences[:3], 1):
        print(f"  {i}. [{sent['start_char']}:{sent['end_char']}] {sent['sentence'][:100]}...")


if __name__ == "__main__":
    INPUT_FILE = "documents_new.json"  # Output from step 2
    OUTPUT_FILE = "sentences_new.json"
    OUTPUT_BY_DOC = "sentences_by_document_new.json"
    
    # Create improved segmenter
    segmenter = ImprovedSentenceSegmenter(
        model_name="en_core_web_sm",
        max_length=1500000
    )
    
    # Process documents
    sentences = segmenter.process_documents(INPUT_FILE, OUTPUT_FILE)
    
    # Create document-grouped view
    if sentences:
        segmenter.create_document_view(sentences, OUTPUT_BY_DOC)
        
        # Show quality metrics
        quality = segmenter.validate_quality(sentences)
        print(f"\n=== QUALITY METRICS ===")
        print(f"Total sentences: {quality['total_sentences']}")
        print(f"Proper endings: {quality['proper_endings_pct']:.1f}%")
        print(f"Proper starts: {quality['proper_starts_pct']:.1f}%")
        print(f"Short sentences (<10 chars): {quality['short_sentences']}")
        print(f"Long sentences (>600 chars): {quality['long_sentences']}")
        print(f"Potential abbreviation splits: {quality['potential_abbrev_splits']}")
        print(f"Average length: {quality['avg_length']:.1f} chars")
        
        # Preview results
        preview_sentences(OUTPUT_FILE, num_sentences=5)
        
        # Validate first document
        if sentences:
            first_doc_id = sentences[0]['doc_id']
            validate_segmentation(INPUT_FILE, OUTPUT_FILE, first_doc_id)

