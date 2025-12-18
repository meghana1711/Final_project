import json
from typing import List, Dict, Optional
from collections import defaultdict

class SentenceChunker:
    def __init__(self, 
                 min_sentences: int = 3,
                 max_sentences: int = 5,
                 min_tokens: int = 400,
                 max_tokens: int = 800,
                 overlap_sentences: int = 1):
        """
        Initialize sentence chunker with context stabilization.
        
        Args:
            min_sentences: Minimum sentences per chunk
            max_sentences: Maximum sentences per chunk
            min_tokens: Minimum tokens per chunk (soft limit)
            max_tokens: Maximum tokens per chunk (hard limit)
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        
        print(f"Chunker initialized:")
        print(f"  Sentences per chunk: {min_sentences}-{max_sentences}")
        print(f"  Token range: {min_tokens}-{max_tokens}")
        print(f"  Overlap: {overlap_sentences} sentence(s)")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: words * 1.3)."""
        words = len(text.split())
        return int(words * 1.3)
    
    def create_chunk(self, sentences: List[Dict], chunk_id: str, 
                    doc_id: str, section: Optional[str] = None) -> Dict:
        """
        Create a chunk from a list of sentences.
        
        Args:
            sentences: List of sentence dictionaries
            chunk_id: Unique chunk identifier
            doc_id: Document identifier
            section: Optional section name
        
        Returns:
            Chunk dictionary with metadata
        """
        # Combine sentences
        combined_text = ' '.join(s['sentence'] for s in sentences)
        
        # Get sentence IDs
        sent_ids = [s['sent_id'] for s in sentences]
        
        # Calculate offsets (from first to last sentence)
        start_char = sentences[0]['start_char']
        end_char = sentences[-1]['end_char']
        
        # Estimate tokens
        token_count = self.estimate_tokens(combined_text)
        
        chunk = {
            'chunk_id': chunk_id,
            'doc_id': doc_id,
            'section': section or 'main',
            'text': combined_text,
            'sentence_ids': sent_ids,
            'num_sentences': len(sentences),
            'start_char': start_char,
            'end_char': end_char,
            'estimated_tokens': token_count
        }
        
        return chunk
    
    def chunk_document_sentences(self, sentences: List[Dict], doc_id: str) -> List[Dict]:
        """
        Chunk sentences from a single document.
        
        Args:
            sentences: List of sentences from one document
            doc_id: Document identifier
        
        Returns:
            List of chunks
        """
        if not sentences:
            return []
        
        chunks = []
        i = 0
        chunk_num = 1
        max_iterations = len(sentences) * 2  # Safety limit
        iterations = 0
        
        while i < len(sentences):
            iterations += 1
            if iterations > max_iterations:
                print(f"  WARNING: {doc_id} - Breaking infinite loop at sentence {i}/{len(sentences)}")
                break
            
            # Start building a chunk
            current_chunk_sents = []
            current_tokens = 0
            start_i = i  # Track starting position
            
            # Add sentences until we hit limits
            while i < len(sentences) and len(current_chunk_sents) < self.max_sentences:
                sent = sentences[i]
                
                # Validate sentence structure
                if 'sentence' not in sent:
                    print(f"  WARNING: {doc_id} - Sentence at index {i} missing 'sentence' field, skipping")
                    i += 1
                    continue
                
                sent_tokens = self.estimate_tokens(sent['sentence'])
                
                # Check if adding this sentence would exceed max tokens
                if current_tokens + sent_tokens > self.max_tokens and current_chunk_sents:
                    break
                
                current_chunk_sents.append(sent)
                current_tokens += sent_tokens
                i += 1
                
                # Check if we have enough sentences and tokens
                if (len(current_chunk_sents) >= self.min_sentences and 
                    current_tokens >= self.min_tokens):
                    # We have a good chunk, but can we add more?
                    if len(current_chunk_sents) >= self.max_sentences:
                        break
                    # Peek ahead - if next sentence is small, include it
                    if i < len(sentences):
                        next_tokens = self.estimate_tokens(sentences[i]['sentence'])
                        if current_tokens + next_tokens > self.max_tokens:
                            break
            
            # Create chunk if we have sentences
            if current_chunk_sents:
                chunk_id = f"{doc_id}_chunk_{chunk_num:04d}"
                chunk = self.create_chunk(current_chunk_sents, chunk_id, doc_id)
                chunks.append(chunk)
                chunk_num += 1
                
                # Move back for overlap - with safety check
                if self.overlap_sentences > 0 and i < len(sentences):
                    overlap = min(self.overlap_sentences, len(current_chunk_sents))
                    new_i = i - overlap
                    # Prevent infinite loop - must move forward
                    if new_i <= start_i:
                        # Don't overlap if it would cause us to go backwards
                        pass
                    else:
                        i = new_i
            else:
                # No sentences added - move forward to prevent infinite loop
                if i == start_i:
                    print(f"  WARNING: {doc_id} - No progress at sentence {i}, forcing skip")
                    i += 1
        
        return chunks
    
    def process_sentences(self, input_file: str, output_file: str) -> List[Dict]:
        """
        Process all sentences and create chunks.
        
        Args:
            input_file: JSON file with sentences (lemmatized or regular)
            output_file: Output JSON file with chunks
        
        Returns:
            List of all chunks
        """
        # Load sentences
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        
        print(f"\nProcessing {len(sentences)} sentences...")
        
        # Group sentences by document
        docs = defaultdict(list)
        for sent in sentences:
            docs[sent['doc_id']].append(sent)
        
        print(f"Found {len(docs)} documents")
        
        # Process each document
        all_chunks = []
        stats = {
            'total_chunks': 0,
            'total_tokens': 0,
            'total_sentences': 0,
            'chunks_per_doc': []
        }
        
        for idx, doc_id in enumerate(sorted(docs.keys()), 1):
            doc_sentences = docs[doc_id]
            
            print(f"  [{idx}/{len(docs)}] Processing {doc_id}: {len(doc_sentences)} sentences...", end=' ')
            
            try:
                doc_chunks = self.chunk_document_sentences(doc_sentences, doc_id)
                
                all_chunks.extend(doc_chunks)
                
                # Update stats
                stats['total_chunks'] += len(doc_chunks)
                stats['chunks_per_doc'].append(len(doc_chunks))
                
                for chunk in doc_chunks:
                    stats['total_tokens'] += chunk['estimated_tokens']
                    stats['total_sentences'] += chunk['num_sentences']
                
                print(f"→ {len(doc_chunks)} chunks")
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
                print(f"  Skipping document {doc_id}")
                continue
        
        # Save chunks
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        # Calculate averages
        avg_tokens = stats['total_tokens'] / stats['total_chunks'] if stats['total_chunks'] else 0
        avg_sents = stats['total_sentences'] / stats['total_chunks'] if stats['total_chunks'] else 0
        avg_chunks_per_doc = sum(stats['chunks_per_doc']) / len(stats['chunks_per_doc']) if stats['chunks_per_doc'] else 0
        
        print(f"\n✓ Saved {stats['total_chunks']} chunks to {output_file}")
        print(f"\nSummary:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Total documents: {len(docs)}")
        print(f"  Average chunks/doc: {avg_chunks_per_doc:.1f}")
        print(f"  Average sentences/chunk: {avg_sents:.1f}")
        print(f"  Average tokens/chunk: {avg_tokens:.1f}")
        
        return all_chunks
    
    def create_document_view(self, chunks: List[Dict], output_file: str):
        """
        Create a view of chunks grouped by document.
        
        Args:
            chunks: List of chunk dictionaries
            output_file: Output JSON file
        """
        docs = defaultdict(list)
        for chunk in chunks:
            docs[chunk['doc_id']].append({
                'chunk_id': chunk['chunk_id'],
                'section': chunk['section'],
                'num_sentences': chunk['num_sentences'],
                'estimated_tokens': chunk['estimated_tokens'],
                'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text']
            })
        
        doc_list = [{'doc_id': doc_id, 'chunks': chunks_list} 
                   for doc_id, chunks_list in sorted(docs.items())]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(doc_list, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created document-grouped view: {output_file}")


def preview_chunks(json_file: str, num_chunks: int = 3):
    """Preview first few chunks."""
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"\nPreview of {min(num_chunks, len(chunks))} chunks:")
    print("=" * 80)
    
    for chunk in chunks[:num_chunks]:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Doc ID: {chunk['doc_id']}")
        print(f"Section: {chunk['section']}")
        print(f"Sentences: {chunk['num_sentences']}")
        print(f"Tokens: {chunk['estimated_tokens']}")
        print(f"Sentence IDs: {', '.join(chunk['sentence_ids'])}")
        print(f"\nText preview:")
        preview_text = chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text']
        print(f"{preview_text}")
        print("-" * 80)


def analyze_chunk_distribution(json_file: str):
    """Analyze token and sentence distribution across chunks."""
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    token_counts = [c['estimated_tokens'] for c in chunks]
    sent_counts = [c['num_sentences'] for c in chunks]
    
    print("\nChunk Distribution Analysis:")
    print("=" * 80)
    print(f"\nToken Distribution:")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")
    print(f"  Mean: {sum(token_counts) / len(token_counts):.1f}")
    print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
    
    print(f"\nSentence Distribution:")
    print(f"  Min: {min(sent_counts)}")
    print(f"  Max: {max(sent_counts)}")
    print(f"  Mean: {sum(sent_counts) / len(sent_counts):.1f}")
    print(f"  Median: {sorted(sent_counts)[len(sent_counts)//2]}")
    
    # Distribution buckets
    print(f"\nToken Range Distribution:")
    ranges = [(0, 400), (400, 600), (600, 800), (800, 10000)]
    for low, high in ranges:
        count = sum(1 for t in token_counts if low <= t < high)
        pct = 100 * count / len(token_counts)
        print(f"  {low}-{high}: {count} chunks ({pct:.1f}%)")


if __name__ == "__main__":
    # Can work with either regular sentences or lemmatized sentences
    INPUT_FILE = "sentences_lemmatized_case_preserved.json"  # or "sentences.json"
    OUTPUT_FILE = "chunks.json"
    OUTPUT_BY_DOC = "chunks_by_document.json"
    
    # Create chunker with optimal settings for OLAF
    chunker = SentenceChunker(
        min_sentences=3,      # Minimum context
        max_sentences=5,      # Maximum for coherence
        min_tokens=400,       # Soft minimum
        max_tokens=800,       # Hard maximum
        overlap_sentences=1   # 1 sentence overlap for context continuity
    )
    
    # Process sentences into chunks
    chunks = chunker.process_sentences(INPUT_FILE, OUTPUT_FILE)
    
    if chunks:
        # Create document-grouped view
        chunker.create_document_view(chunks, OUTPUT_BY_DOC)
        
        # Preview results
        preview_chunks(OUTPUT_FILE, num_chunks=3)
        
        # Analyze distribution
        analyze_chunk_distribution(OUTPUT_FILE)
        
        print("\n" + "="*80)
        print("Output files:")
        print(f"  1. {OUTPUT_FILE} - All chunks with full metadata")
        print(f"  2. {OUTPUT_BY_DOC} - Chunks grouped by document")
        
        print(f"✓ Created document-grouped view: {OUTPUT_FILE}")


def preview_chunks(json_file: str, num_chunks: int = 3):
    """Preview first few chunks."""
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"\nPreview of {min(num_chunks, len(chunks))} chunks:")
    print("=" * 80)
    
    for chunk in chunks[:num_chunks]:
        print(f"\nChunk ID: {chunk['chunk_id']}")
        print(f"Doc ID: {chunk['doc_id']}")
        print(f"Section: {chunk['section']}")
        print(f"Sentences: {chunk['num_sentences']}")
        print(f"Tokens: {chunk['estimated_tokens']}")
        print(f"Sentence IDs: {', '.join(chunk['sentence_ids'])}")
        print(f"\nText preview:")
        preview_text = chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text']
        print(f"{preview_text}")
        print("-" * 80)


def analyze_chunk_distribution(json_file: str):
    """Analyze token and sentence distribution across chunks."""
    with open(json_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    token_counts = [c['estimated_tokens'] for c in chunks]
    sent_counts = [c['num_sentences'] for c in chunks]
    
    print("\nChunk Distribution Analysis:")
    print("=" * 80)
    print(f"\nToken Distribution:")
    print(f"  Min: {min(token_counts)}")
    print(f"  Max: {max(token_counts)}")
    print(f"  Mean: {sum(token_counts) / len(token_counts):.1f}")
    print(f"  Median: {sorted(token_counts)[len(token_counts)//2]}")
    
    print(f"\nSentence Distribution:")
    print(f"  Min: {min(sent_counts)}")
    print(f"  Max: {max(sent_counts)}")
    print(f"  Mean: {sum(sent_counts) / len(sent_counts):.1f}")
    print(f"  Median: {sorted(sent_counts)[len(sent_counts)//2]}")
    
    # Distribution buckets
    print(f"\nToken Range Distribution:")
    ranges = [(0, 400), (400, 600), (600, 800), (800, 10000)]
    for low, high in ranges:
        count = sum(1 for t in token_counts if low <= t < high)
        pct = 100 * count / len(token_counts)
        print(f"  {low}-{high}: {count} chunks ({pct:.1f}%)")


if __name__ == "__main__":
    # Can work with either regular sentences or lemmatized sentences
    INPUT_FILE = "sentences_lemmatized.json"  # or "sentences.json"
    OUTPUT_FILE = "chunks.json"
    OUTPUT_BY_DOC = "chunks_by_document.json"
    
    # Create chunker with optimal settings for OLAF
    chunker = SentenceChunker(
        min_sentences=3,      # Minimum context
        max_sentences=5,      # Maximum for coherence
        min_tokens=400,       # Soft minimum
        max_tokens=800,       # Hard maximum
        overlap_sentences=1   # 1 sentence overlap for context continuity
    )
    
    # Process sentences into chunks
    chunks = chunker.process_sentences(INPUT_FILE, OUTPUT_FILE)
    
    if chunks:
        # Create document-grouped view
        chunker.create_document_view(chunks, OUTPUT_BY_DOC)
        
        # Preview results
        preview_chunks(OUTPUT_FILE, num_chunks=3)
        
        # Analyze distribution
        analyze_chunk_distribution(OUTPUT_FILE)
        
        print("\n" + "="*80)
        print("Output files:")
        print(f"  1. {OUTPUT_FILE} - All chunks with full metadata")
        print(f"  2. {OUTPUT_BY_DOC} - Chunks grouped by document")