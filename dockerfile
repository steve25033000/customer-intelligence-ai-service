FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# FIXED: Install dependencies in stages to avoid conflicts
RUN pip install --no-cache-dir --upgrade pip

# Stage 1: Install core packages first
RUN pip install --no-cache-dir \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.20.0 \
    pydantic>=2.0.0

# Stage 2: Install AI packages
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    numpy>=1.21.0 \
    scikit-learn>=1.2.0

# Stage 3: Install transformers and related
RUN pip install --no-cache-dir \
    transformers>=4.30.0 \
    sentence-transformers>=2.2.0 \
    huggingface-hub>=0.15.0

# Stage 4: Install remaining utilities
RUN pip install --no-cache-dir \
    structlog>=21.0.0 \
    httpx>=0.24.0 \
    python-multipart>=0.0.5 \
    psutil>=5.9.0 \
    python-dateutil>=2.8.0

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Start command
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1






