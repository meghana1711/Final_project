# Project Overview
We use an OLAF framework to automatically generate, align, and refine LLM-based ontologies from HPC datasets.

# Installation
git clone https://github.com/meghana1711/Final_project.git
cd Final_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Data
For this study, we used the complete official documentation of the SLURM and IBM LSF workload managers as our primary data sources. The SLURM corpus was scraped from the SLURM website, while the IBM LSF documentation was obtained directly as a PDF file from the official IBM LSF website.

# Project Structure
data/             # input data
pre_processing/   # pre-processing
to run the code use -> python -m pre_processing.data_preprocessing
pipeline/         # pipline includes all the steps
to run the code use -> python -m pipeline.pipeline_processing

# Technologies Used
1. Programming Languages: Python
2. Libraries/Frameworks:
    • SpaCy – sentence segmentation, lemmatization, POS tagging

3. Databases:
    • SQLite – lightweight relational database used for storing and querying processed data

4. Tools & Platforms:
    • Jupyter Notebooks
    • Git and GitHub for version control