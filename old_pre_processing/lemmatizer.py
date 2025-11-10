import json
import spacy
from typing import List, Dict

class SentenceLemmatizer:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy lemmatizer.
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        try:
            self.nlp = spacy.load(model_name)
            print(f"✓ Loaded spaCy model: {model_name}")
        except OSError:
            print(f"Model '{model_name}' not found.")
            print(f"Run: python -m spacy download {model_name}")
            raise

        # Keep only necessary components
        # We need: tokenizer, tagger, lemmatizer
        disable_pipes = ['ner', 'parser']
        for pipe in disable_pipes:
            if pipe in self.nlp.pipe_names:
                self.nlp.disable_pipes(pipe)
        
        print(f"Active pipes: {self.nlp.pipe_names}")

    def preserve_original_case(self, original_token: str, lemma: str) -> str:
        """
        Preserve the capitalization pattern of the original token in the lemma.
        
        Args:
            original_token: Original token with case
            lemma: Lemmatized form (usually lowercase)
            
        Returns:
            Lemma with preserved case pattern
        """
        if not original_token or not lemma:
            return lemma
            
        # If original is all uppercase, make lemma uppercase
        if original_token.isupper():
            return lemma.upper()
        
        # If original starts with uppercase, capitalize lemma
        elif original_token[0].isupper():
            return lemma.capitalize()
        
        # If original has mixed case, try to preserve pattern
        elif any(c.isupper() for c in original_token[1:]):
            # For complex patterns, keep original case if lemma is same length
            if len(original_token) == len(lemma):
                result = ""
                for i, char in enumerate(lemma):
                    if i < len(original_token):
                        if original_token[i].isupper():
                            result += char.upper()
                        else:
                            result += char.lower()
                    else:
                        result += char
                return result
            else:
                # Fallback: just capitalize first letter if original was capitalized
                return lemma.capitalize() if original_token[0].isupper() else lemma
        
        # Default: return lemma as-is (lowercase)
        return lemma

    def lemmatize_sentence(self, sentence: str, 
                          keep_pos: bool = True, 
                          remove_stopwords: bool = False, 
                          remove_punct: bool = False) -> Dict:  # Changed default to False
        """
        Lemmatize a sentence and optionally filter tokens.
        
        Args:
            sentence: Input sentence text
            keep_pos: Keep POS tags for each token
            remove_stopwords: Remove stopwords (the, a, is, etc.)
            remove_punct: Remove punctuation tokens
            
        Returns:
            Dictionary with lemmatized tokens and metadata
        """
        doc = self.nlp(sentence)
        
        tokens = []
        lemmas = []
        lemmas_with_case = []  # New: case-preserved lemmas
        pos_tags = []
        
        for token in doc:
            # Skip based on filters
            if remove_stopwords and token.is_stop:
                continue
            if remove_punct and token.is_punct:
                continue
            
            # Store original token
            tokens.append(token.text)
            
            # Store standard lemma (lowercase)
            lemmas.append(token.lemma_)
            
            # Store case-preserved lemma
            case_preserved_lemma = self.preserve_original_case(token.text, token.lemma_)
            lemmas_with_case.append(case_preserved_lemma)
            
            if keep_pos:
                pos_tags.append(token.pos_)
        
        result = {
            'tokens': tokens,
            'lemmas': lemmas,
            'lemmas_with_case': lemmas_with_case,  # New field
            'lemmatized_text': ' '.join(lemmas),
            'lemmatized_text_with_case': ' '.join(lemmas_with_case)  # New field
        }
        
        if keep_pos:
            result['pos_tags'] = pos_tags
            
        return result

    def process_sentences(self, input_file: str, output_file: str, 
                         keep_pos: bool = True, 
                         remove_stopwords: bool = False, 
                         remove_punct: bool = False,  # Changed default to False
                         batch_size: int = 100) -> List[Dict]:
        """
        Process all sentences and add lemmatization.
        
        Args:
            input_file: JSON file with segmented sentences
            output_file: Output JSON file with lemmatized sentences
            keep_pos: Include POS tags in output
            remove_stopwords: Remove stopwords from lemmatized text
            remove_punct: Remove punctuation from lemmatized text
            batch_size: Number of sentences to process at once
            
        Returns:
            List of sentences with lemmatization
        """
        # Load sentences
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        
        print(f"\nProcessing {len(sentences)} sentences...")
        print(f"Settings:")
        print(f" Keep POS tags: {keep_pos}")
        print(f" Remove stopwords: {remove_stopwords}")
        print(f" Remove punctuation: {remove_punct}")
        print(f" Preserve capitalization: True")  # Always true now
        
        lemmatized_sentences = []
        
        # Process in batches for better performance
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            
            for sent in batch:
                # Lemmatize
                lemma_result = self.lemmatize_sentence(
                    sent['sentence'],
                    keep_pos=keep_pos,
                    remove_stopwords=remove_stopwords,
                    remove_punct=remove_punct
                )
                
                # Create enriched sentence object
                lemmatized_sent = {
                    'doc_id': sent['doc_id'],
                    'sent_id': sent['sent_id'],
                    'sentence': sent['sentence'],  # Original sentence
                    'start_char': sent['start_char'],
                    'end_char': sent['end_char'],
                    'tokens': lemma_result['tokens'],
                    'lemmas': lemma_result['lemmas'],
                    'lemmas_with_case': lemma_result['lemmas_with_case'],  # New field
                    'lemmatized_text': lemma_result['lemmatized_text'],
                    'lemmatized_text_with_case': lemma_result['lemmatized_text_with_case']  # New field
                }
                
                if keep_pos:
                    lemmatized_sent['pos_tags'] = lemma_result['pos_tags']
                
                lemmatized_sentences.append(lemmatized_sent)
            
            # Progress indicator
            processed = min(i + batch_size, len(sentences))
            if processed % 500 == 0 or processed == len(sentences):
                print(f" Processed {processed}/{len(sentences)} sentences...")
        
        # Save lemmatized sentences
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(lemmatized_sentences, f, indent=2, ensure_ascii=False)
        
        #print(f"\n✓ Saved {len(lemmatized_sentences)} lemmatized sentences to {output_file}")
        return lemmatized_sentences

    def create_lemma_only_view(self, lemmatized_file: str, output_file: str):
        """
        Create a simplified view with just lemmatized text for quick analysis.
        
        Args:
            lemmatized_file: JSON file with full lemmatization data
            output_file: Simplified output file
        """
        with open(lemmatized_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        
        simplified = []
        for sent in sentences:
            simplified.append({
                'sent_id': sent['sent_id'],
                'original': sent['sentence'],
                'lemmatized': sent['lemmatized_text'],
                'lemmatized_with_case': sent['lemmatized_text_with_case']  # Include both versions
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created simplified view: {output_file}")

    def create_case_preserved_view(self, lemmatized_file: str, output_file: str):
        """
        Create a view with only case-preserved lemmatized text.
        
        Args:
            lemmatized_file: JSON file with full lemmatization data
            output_file: Case-preserved output file
        """
        with open(lemmatized_file, 'r', encoding='utf-8') as f:
            sentences = json.load(f)
        
        case_preserved = []
        for sent in sentences:
            case_preserved.append({
                'sent_id': sent['sent_id'],
                'original': sent['sentence'],
                'lemmatized_with_case': sent['lemmatized_text_with_case']
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(case_preserved, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Created case-preserved view: {output_file}")


def preview_lemmatization(json_file: str, num_sentences: int = 5):
    """Preview lemmatized sentences."""
    with open(json_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    
    print(f"\nPreview of {min(num_sentences, len(sentences))} lemmatized sentences:")
    print("=" * 80)
    
    for sent in sentences[:num_sentences]:
        print(f"\nSent ID: {sent['sent_id']}")
        print(f"Original:                {sent['sentence']}")
        print(f"Standard Lemmatized:     {sent['lemmatized_text']}")
        print(f"Case-Preserved Lemmas:   {sent['lemmatized_text_with_case']}")
        if 'pos_tags' in sent:
            print(f"POS Tags:                {' '.join(sent['pos_tags'])}")
        print("-" * 80)


def compare_lemmatization_settings(input_file: str):
    """
    Compare different lemmatization settings on sample sentences.
    """
    print("\nComparing lemmatization settings...")
    print("=" * 80)
    
    lemmatizer = SentenceLemmatizer()
    
    # Load a few sample sentences
    with open(input_file, 'r', encoding='utf-8') as f:
        sentences = json.load(f)
    
    sample = sentences[:3]
    
    settings = [
        {'remove_stopwords': False, 'remove_punct': False, 'label': 'Full (with punct & case)'},
        {'remove_stopwords': False, 'remove_punct': True, 'label': 'No Punct (with case)'},
        {'remove_stopwords': True, 'remove_punct': False, 'label': 'No Stop (with punct & case)'},
        {'remove_stopwords': True, 'remove_punct': True, 'label': 'No Stop/Punct (with case)'}
    ]
    
    for sent in sample:
        print(f"\nOriginal: {sent['sentence']}")
        print()
        
        for config in settings:
            result = lemmatizer.lemmatize_sentence(
                sent['sentence'],
                keep_pos=False,
                remove_stopwords=config['remove_stopwords'],
                remove_punct=config['remove_punct']
            )
            
            print(f" {config['label']:25s}: {result['lemmatized_text_with_case']}")
        print("-" * 80)


if __name__ == "__main__":
    INPUT_FILE = "sentences.json"  # Output from step 3
    OUTPUT_FILE = "sentences_lemmatized.json"
    OUTPUT_SIMPLE = "sentences_lemmatized_simple.json"
    OUTPUT_CASE = "sentences_lemmatized_case_preserved.json"
    
    # Settings for lemmatization - Updated defaults
    KEEP_POS = True         # Keep POS tags (useful for relation extraction)
    REMOVE_STOPWORDS = False  # Keep stopwords for now (OLAF might need them)
    REMOVE_PUNCT = False    # Changed: Keep punctuation
    
    # First, compare settings on sample data
    print("Step 1: Comparing lemmatization approaches...")
    try:
        compare_lemmatization_settings(INPUT_FILE)
    except FileNotFoundError:
        print(f"Warning: {INPUT_FILE} not found, skipping comparison")
    
    # Create lemmatizer and process
    print("\nStep 2: Processing all sentences...")
    lemmatizer = SentenceLemmatizer()
    
    lemmatized = lemmatizer.process_sentences(
        INPUT_FILE,
        OUTPUT_FILE,
        keep_pos=KEEP_POS,
        remove_stopwords=REMOVE_STOPWORDS,
        remove_punct=REMOVE_PUNCT
    )
    
    # Create multiple views
    if lemmatized:
        lemmatizer.create_lemma_only_view(OUTPUT_FILE, OUTPUT_SIMPLE)
        lemmatizer.create_case_preserved_view(OUTPUT_FILE, OUTPUT_CASE)
        
        # Preview results
        preview_lemmatization(OUTPUT_FILE, num_sentences=5)
        
        print("\n" + "="*80)
        print("Files created:")
        print(f" 1. {OUTPUT_FILE} - Full data with tokens, lemmas, POS tags, and both versions")
        print(f" 2. {OUTPUT_SIMPLE} - Simplified view with both original and lemmatized versions")
        print(f" 3. {OUTPUT_CASE} - Case-preserved lemmatized text only")
       