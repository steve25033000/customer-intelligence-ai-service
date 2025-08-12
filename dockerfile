FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
export PORT=${PORT:-10000}\n\
echo "Starting Customer Intelligence AI Service on port $PORT"\n\
exec uvicorn app.main:app \\\n\
    --host 0.0.0.0 \\\n\
    --port $PORT \\\n\
    --workers 2 \\\n\
    --worker-class uvicorn.workers.UvicornWorker \\\n\
    --access-log \\\n\
    --log-level info' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]




