# Optimized for Render free tier - FIXED SYNTAX
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies efficiently
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Environment variables for memory optimization - FIXED SYNTAX
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with memory optimization
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Health check for Render
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start command optimized for Render
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1
