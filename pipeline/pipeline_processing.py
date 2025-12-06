import os
import json
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
import logging
import pre_processing.data_preprocessing as dp


FOLDER_PATH = "data/slurm_doc"  
DB_PATH = "onto_db/olaf_trial.db"           
VERSION = 1      
MIN_WORDS=4   
MODEL_NAME = "en_core_web_sm"     
MAX_LENGTH = 1500000                 

#Read text documents
docs = dp.read_data(FOLDER_PATH, DB_PATH, version=VERSION)

#Clean the data, removing unwanted footers, headers, spaces, special charecters
cleaner = dp.TechnicalDocumentCleaner(min_words=MIN_WORDS)
cleaner.clean_into_db(db_path=DB_PATH, raw_version=VERSION, cleaned_version=VERSION)

#Sentence segmentation using Spacy, on cleaned data
sentences = dp.ImprovedSentenceSegmenter(model_name = MODEL_NAME , max_length = MAX_LENGTH)
sentences.segment_cleaned_to_db(db_path=DB_PATH, cleaned_version=VERSION)

#Lemmatizing and POS tagging the tokenized senetences
lemmatizer = dp.SentenceLemmatizer(model_name = MODEL_NAME)
lemmatizer.process_sentences_db(db_path=DB_PATH,cleaned_version=VERSION,keep_pos=True, remove_stopwords=False, remove_punct=False,)




