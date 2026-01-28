FROM python:3.11-slim

WORKDIR /app

# System deps (add more if you need, e.g., git, build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Default command (change to your real entrypoint)
CMD ["python", "--version"]
