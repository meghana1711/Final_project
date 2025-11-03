import json
import spacy
from typing import List, Dict, Tuple

class SentenceSegmenter:
    def __init__(self, model_name: str = "en_core_web_sm", max_length: int = 1500000):
        """
        Initialize spaCy sentence segmenter.
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
            max_length: Maximum text length to process (default: 2MB)5
        
        Note: Install model first with: python -m spacy download en_core_web_sm
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
        # We only need sentence segmentation
        disable_pipes = ['ner', 'lemmatizer', 'textcat']
        for pipe in disable_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipes(pipe)
        
        print(f"Active pipes: {self.nlp.pipe_names}")
    
    def segment_text(self, text: str, doc_id: str) -> List[Dict]:
        """
        Segment text into sentences with offsets.
        Handles large documents by chunking if necessary.
        
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
        
        doc = self.nlp(text)
        
        sentences = []
        for i, sent in enumerate(doc.sents):
            sentence_text = sent.text.strip()
            
            # Skip empty sentences
            if not sentence_text:
                continue
            
            # Skip sentences ending with ':'
            if sentence_text.endswith(':'):
                continue
            
            sentence_dict = {
                'doc_id': doc_id,
                'sent_id': f"{doc_id}_sent_{i+1:04d}",
                'sentence': sentence_text,
                'start_char': sent.start_char,
                'end_char': sent.end_char,
                'length': len(sentence_text)
            }
            
            sentences.append(sentence_dict)
        
        return sentences
    
    def _segment_large_text(self, text: str, doc_id: str) -> List[Dict]:
        """
        Segment very large texts by processing in chunks.
        
        Args:
            text: Input text to segment
            doc_id: Document identifier
        
        Returns:
            List of sentence dictionaries
        """
        # Split into manageable chunks (90% of max_length for safety)
        chunk_size = int(self.nlp.max_length * 0.9)
        all_sentences = []
        sentence_counter = 0
        
        # Process text in overlapping chunks to avoid cutting sentences
        i = 0
        while i < len(text):
            # Get chunk
            end = min(i + chunk_size, len(text))
            
            # If not at end, try to break at paragraph or sentence boundary
            if end < len(text):
                # Look for paragraph break
                last_para = text.rfind('\n\n', i, end)
                if last_para > i:
                    end = last_para + 2
                else:
                    # Look for sentence break
                    for punct in ['. ', '! ', '? ']:
                        last_sent = text.rfind(punct, i, end)
                        if last_sent > i:
                            end = last_sent + 2
                            break
            
            chunk = text[i:end]
            
            # Process chunk
            doc = self.nlp(chunk)
            
            for sent in doc.sents:
                sentence_text = sent.text.strip()
                
                if not sentence_text or sentence_text.endswith(':'):
                    continue
                
                sentence_counter += 1
                sentence_dict = {
                    'doc_id': doc_id,
                    'sent_id': f"{doc_id}_sent_{sentence_counter:04d}",
                    'sentence': sentence_text,
                    'start_char': i + sent.start_char,  # Adjust offset
                    'end_char': i + sent.end_char,      # Adjust offset
                    'length': len(sentence_text)
                }
                
                all_sentences.append(sentence_dict)
            
            i = end
        
        return all_sentences
    
    def process_documents(self, input_file: str, output_file: str) -> List[Dict]:
        """
        Process all documents and segment into sentences.
        
        Args:
            input_file: JSON file with cleaned documents
            output_file: Output JSON file with segmented sentences
        
        Returns:
            List of all sentences from all documents
        """
        # Load cleaned documents
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        print(f"\nProcessing {len(documents)} documents...")
        
        all_sentences = []
        stats = {
            'total_docs': len(documents),
            'total_sentences': 0,
            'total_chars': 0,
            'avg_sent_length': 0
        }
        
        for idx, doc in enumerate(documents, 1):
            doc_id = doc['doc_id']
            text = doc['text']
            title = doc['title']
            
            if not text.strip():
                print(f"  [{idx}/{len(documents)}] {doc_id}: Skipping (empty)")
                continue
            
            # Segment document
            sentences = self.segment_text(text, doc_id)
            
            # Update stats
            stats['total_sentences'] += len(sentences)
            stats['total_chars'] += sum(s['length'] for s in sentences)
            
            all_sentences.extend(sentences)
            
            print(f"  [{idx}/{len(documents)}] {doc_id} ({title}): {len(sentences)} sentences")
        
        # Calculate average
        if stats['total_sentences'] > 0:
            stats['avg_sent_length'] = stats['total_chars'] / stats['total_sentences']
        
        # Save segmented sentences
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_sentences, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n✓ Saved {stats['total_sentences']} sentences to {output_file}")
        print(f"\nSummary:")
        print(f"  Total documents: {stats['total_docs']}")
        print(f"  Total sentences: {stats['total_sentences']}")
        print(f"  Average sentences/doc: {stats['total_sentences'] / stats['total_docs']:.1f}")
        print(f"  Average sentence length: {stats['avg_sent_length']:.1f} chars")
        
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


def preview_sentences(json_file: str, num_sentences: int = 10):
    """Preview first few sentences from JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    
    print(f"\nPreview of {min(num_sentences, len(sentences))} sentences:")
    print("=" * 80)
    
    for sent in sentences[:num_sentences]:
        print(f"\nSent ID: {sent['sent_id']}")
        print(f"Offsets: [{sent['start_char']}, {sent['end_char']}]")
        print(f"Text: {sent['sentence']}")
        print("-" * 80)


def validate_segmentation(input_file: str, sentences_file: str, doc_id: str):
    """
    Validate segmentation by reconstructing original text.
    
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
    
    print(f"\nValidation for {doc_id}:")
    print(f"  Original length: {len(original['text'])} chars")
    print(f"  Number of sentences: {len(doc_sentences)}")
    print(f"\nFirst 3 sentences:")
    for i, sent in enumerate(doc_sentences[:3], 1):
        print(f"  {i}. [{sent['start_char']}:{sent['end_char']}] {sent['sentence'][:100]}...")


if __name__ == "__main__":
    INPUT_FILE = "documents_cleaned.json"  # Output from step 2
    OUTPUT_FILE = "sentences.json"
    OUTPUT_BY_DOC = "sentences_by_document.json"
    
    # Create segmenter with increased max_length for large documents
    segmenter = SentenceSegmenter(
        model_name="en_core_web_sm",  # or "en_core_web_lg" if you have it
        max_length=1500000  # 5MB limit (increase if needed)
    )
    
    # Process documents
    sentences = segmenter.process_documents(INPUT_FILE, OUTPUT_FILE)
    
    # Create document-grouped view (optional)
    if sentences:
        segmenter.create_document_view(sentences, OUTPUT_BY_DOC)
        
        # Preview results
        preview_sentences(OUTPUT_FILE, num_sentences=5)
        
        # Validate first document
        if sentences:
            first_doc_id = sentences[0]['doc_id']
            validate_segmentation(INPUT_FILE, OUTPUT_FILE, first_doc_id)