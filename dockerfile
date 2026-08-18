# =============================================================================
# Telco Customer Churn — Production Multi-Stage Container Image
# =============================================================================

# --- Stage 1: Builder / Dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Runtime Image ---
FROM python:3.11-slim AS runner

WORKDIR /app

# Create non-root user for container security
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser

# Copy installed dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application source code and configurations
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup artifacts/ ./artifacts/

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/artifacts/model \
    PIPELINE_PATH=/app/artifacts/pipeline.pkl \
    PREDICTION_THRESHOLD=0.119 \
    MPLCONFIGDIR=/tmp/matplotlib \
    PORT=8000

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI serving server with Uvicorn
CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
