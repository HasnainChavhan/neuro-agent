# ════════════════════════════════════════════════════════════════════════════
# NeuroAgent — Multi-stage Dockerfile
# Stage 1: builder   (installs deps)
# Stage 2: runtime   (minimal image, Lambda-compatible)
# ════════════════════════════════════════════════════════════════════════════

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix directory (for clean copy)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="NeuroAgent"
LABEL version="1.0.0"
LABEL description="Autonomous AI Research Assistant"

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application source
COPY app/ ./app/
COPY .env.example .env

# Non-root user for security
RUN groupadd -r neuro && useradd -r -g neuro neuro
RUN chown -R neuro:neuro /app
USER neuro

# FastAPI port
EXPOSE 8000

# Health check (checks /api/v1/health)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Start Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
