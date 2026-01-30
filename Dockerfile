FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set Hugging Face cache directories
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV TORCH_HOME=/app/.cache/torch

# Install PyTorch with CUDA support first (largest dependency)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Copy and install requirements
COPY requirements.txt .

# Fix the contourpy version issue before installing
RUN sed -i 's/contourpy==1\.3\.3/contourpy==1.3.2/g' requirements.txt || \
    (pip install 'contourpy<1.3.3' && grep -v "contourpy" requirements.txt > requirements_temp.txt && mv requirements_temp.txt requirements.txt)

RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy language model
RUN python -m spacy download en_core_web_sm

# (Optional) Pre-download common Hugging Face models to speed up first run
# Uncomment and add your specific models:
# RUN python -c "from sentence_transformers import SentenceTransformer; \
#     SentenceTransformer('all-MiniLM-L6-v2')"
# RUN python -c "from transformers import AutoModel, AutoTokenizer; \
#     AutoModel.from_pretrained('bert-base-uncased'); \
#     AutoTokenizer.from_pretrained('bert-base-uncased')"

# Copy project code
COPY . .

# Verify installations
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" && \
    python -c "import spacy, transformers, sentence_transformers; print('✓ All packages OK')"

# Default command (change to your actual script)
CMD ["python", "--version"]