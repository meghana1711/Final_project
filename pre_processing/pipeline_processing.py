import os
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import logging

# Import your existing modules
import sys
sys.path.append('/mnt/user-data/uploads')

import data_preprocessing as dp

FOLDER_PATH = "C:/Users/20236193/Final_project/data/ibm_lsf/lsf_text"  
DB_PATH = "ontology_workspace.db"           
VERSION = 1      
MIN_WORDS=4                           

#Read text documents
docs = dp.read_data(FOLDER_PATH, DB_PATH, version=VERSION)

#Clean them from raw_version=1 and save to cleaned_documents as cleaned_version=1
cleaner = dp.TechnicalDocumentCleaner(min_words=MIN_WORDS)
cleaner.clean_into_db(db_path=DB_PATH, raw_version=VERSION, cleaned_version=VERSION)

"""
#INPUT_FILE = "documents_new.json"          # input JSON from your chapter loader
OUTPUT_FILE = "sentences_new.json"         # flat list of sentences
OUTPUT_BY_DOC = "sentences_by_document_new.json"  # sentences grouped by doc
    
segmenter = dp.ImprovedSentenceSegmenter(
        model_name="en_core_web_sm",
        max_length=1500000
    )
    
 # Segment all documents into sentences
sentences = segmenter.process_documents(dp.documents, OUTPUT_FILE)
# Step 2: also save view grouped by document
if sentences:
    segmenter.create_document_view(sentences, OUTPUT_BY_DOC)
    # Step 3: quality metrics
    quality = segmenter.validate_quality(sentences)
    print(f"\n=== QUALITY METRICS ===")
    print(f"Total sentences: {quality['total_sentences']}")
    print(f"Proper endings: {quality['proper_endings_pct']:.1f}%")
    print(f"Proper starts: {quality['proper_starts_pct']:.1f}%")
    print(f"Short sentences (<10 chars): {quality['short_sentences']}")
    print(f"Long sentences (>600 chars): {quality['long_sentences']}")
    print(f"Potential abbreviation splits: {quality['potential_abbrev_splits']}")
    print(f"Average length: {quality['avg_length']:.1f} chars")
        """
       

