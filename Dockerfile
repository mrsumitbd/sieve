FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including cloc
RUN apt-get update && apt-get install -y \
    git \
    curl \
    cloc \
    && rm -rf /var/lib/apt/lists/*

# Copy everything first
COPY . .

# Install dependencies and the sieve package
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace Spaces runs as user 1000
RUN useradd -m -u 1000 user && \
    mkdir -p /home/user/.cache/huggingface && \
    mkdir -p /app/src/sieve/models/artifacts && \
    chown -R user:user /home/user && \
    chown -R user:user /app/src/sieve/models/artifacts

USER user

# Expose Streamlit port (HF Spaces requires 7860)
EXPOSE 7860

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers

CMD ["streamlit", "run", "src/sieve/ui/Home.py", "--server.port=7860", "--server.address=0.0.0.0"]