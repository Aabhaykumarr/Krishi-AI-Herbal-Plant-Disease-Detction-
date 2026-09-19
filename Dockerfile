# Multi-stage Dockerfile for KrishiAI YOLO26 Inference Service (Section 51)
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install runtime OS dependencies for OpenCV / PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install ".[all]" --extra-index-url https://download.pytorch.org/whl/cpu || pip install -e .

# Copy application source & assets
COPY configs/ configs/
COPY static/ static/
COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/
COPY data/final/ data/final/

# Expose HTTP port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "krishiai.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
