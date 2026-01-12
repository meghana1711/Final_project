# Project Overview
We use an OLAF framework to automatically generate, align, and refine LLM-based ontologies from HPC datasets. OLAF (Ontology Learning from Annotation Framework) is a hybrid ontology learning pipeline that combines classical NLP techniques, rule-based reasoning and LLMs to construct ontologies from unstructured domain text.

In this project, the OLAF pipeline operates in multiple sequential stages:

1. **Term Extraction**  
   Domain-relevant candidate terms are extracted using statistical and
   linguistic methods (e.g., TF-IDF, C-value, noun-phrase patterns).  
   This step prioritizes precision and grounding in the source text, reducing
   noise and hallucination.

2. **Term Enrichment**  
   Extracted terms are enriched with canonical forms, variants, and semantic
   context using lexical normalization, embeddings, and domain heuristics.
   Enrichment improves consistency across documents and supports downstream
   relation induction.

3. **Taxonomy Induction (IS-A Relations)**  
   Hierarchical relations are constructed by identifying parent/head terms and
   applying pattern-based and distributional rules (e.g., Hearst-style patterns,
   head-modifier analysis).  
   The output is a structured concept taxonomy capturing subclass relations.

4. **Non-Taxonomic Relation Induction**  
   Non-hierarchical relations (e.g., *uses*, *allocates*, *runs on*) are extracted
   using open-information-extraction–style patterns and filtered using frequency,
   diversity, and canonical alignment constraints.

5. **Axiom Generation**  
   The induced concepts and relations are translated into OWL axioms, enabling
   logical reasoning, validation, and downstream knowledge graph construction..


# Project Structure

Final_project/
├── data/                           # Raw input data (SLURM / IBM LSF documents)
│
├── pre_processing/                 # Text preprocessing + contextual chunking
│   ├── __init__.py
│   ├── data_preprocessing.py       # Cleaning, normalization, sentence handling
│   └── contextual_chunking.py      # Context-aware chunk generation
│
├── pipeline/                       # End-to-end pipeline orchestration
│   ├── __init__.py
│   └── pipeline_processing.py      # Runs full OLAF / OLAF_LLM pipeline
│
├── onto_db/                        # SQLite databases (intermediate + final artifacts)
│   ├── concept_taxonomy.db         # Concept-level taxonomy
│   ├── taxonomy_parent_candidates.db
│   ├── term_enrichment.db
│   ├── olaf_sample.db              # OLAF (rule/NLP-based) outputs
│   ├── olaf_sample_llm.db           # OLAF_LLM outputs
│   ├── olaf_trial.db                # Experimental runs
│   ├── onto_new.db                 # Main working ontology database
│   └── ontology_sample_new.db      # Sample / debug ontology
│
├── olaf/                           # OLAF: Hybrid ontology learning (NLP + statistics)
│   ├── __init__.py
│   ├── term_extraction_tfidf.py    # TF-IDF / C-value based term extraction
│   ├── term_enrichment.py          # Enrichment using lexical resources
│   ├── parent_terms.py             # Parent/head term identification
│   ├── taxonomy_induction.py       # IS-A hierarchy construction
│   ├── taxonomy.py                 # Taxonomy handling utilities
│   ├── taxonomy_old.py             # Legacy taxonomy logic
│   ├── relation_HiT.py             # Hearst-in-the-wild / pattern-based relations
│   ├── relation_induction.py       # Non-taxonomic relation induction
│   ├── non_taxonomy.py             # Filtering & storage of non-taxonomic relations
│   ├── embeddings.py               # Embedding utilities
│   └── axioms.py                   # OWL axiom generation (rule-based)
│
├── olaf_llm/                       # OLAF_LLM: LLM-assisted ontology learning
│   ├── __init__.py
│   ├── term_extraction_llm.py      # LLM-based term extraction
│   ├── term_enrichment_llm.py      # LLM-based enrichment
│   ├── taxonomy_llm.py             # LLM-driven taxonomy induction
│   ├── non_taxonomy_llm.py         # LLM-based non-taxonomic relations
│   └── axioms_llm.py               # LLM-generated OWL axioms
│
├── competency_question/            # Competency Questions (CQs)
│   ├── __init__.py
│   └── competency_question.py      # Documents grounded RAG based CQ generation
│
├── text2owl/                       # Text2OWL-style OWL serialization
│   ├── __init__.py
│   ├── text2owl.py                 # Converts extracted knowledge → OWL
│   └── owl1.ttl                    # Generated OWL/Turtle output
│
└── README.md

# Installation
git clone https://github.com/meghana1711/Final_project.git   
cd Final_project  
python -m venv venv  
.\venv\Scripts\Activate.ps1  

# Install PyTorch with CUDA
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124  
pip install -r requirements.txt  
python -m spacy download en_core_web_sm  

# Technologies Used
1. Programming Languages: Python
2. Libraries/Frameworks:
    • SpaCy – sentence segmentation, lemmatization, POS tagging

3. Databases:
    • SQLite – lightweight relational database used for storing and querying processed data

4. Tools & Platforms:
    • Jupyter Notebooks
    • Git and GitHub for version control

5. LLM Models used:
    • mistralai/Mistral-7B-Instruct-v0.3