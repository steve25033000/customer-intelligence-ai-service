# ==========================================
# Customer Intelligence AI Service - Railway Optimized Dockerfile
# Railway has 8GB memory - can use full features
# ==========================================

FROM python:3.10-slim

# Environment variables optimized for Railway
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Set working directory
WORKDIR /app

# Install system dependencies (Railway can handle more)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .

# Railway can handle all dependencies at once
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port (Railway assigns dynamically)
EXPOSE 8000

# Health check optimized for Railway
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Railway-optimized start command (2 workers since Railway has more memory)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --log-level info






