# Nexus COO Strategic Intelligence Service
# Multi-stage build for production deployment

# =============================================================================
# Build stage
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies from the root project (includes distributed extras)
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[distributed]"

# =============================================================================
# Production stage
# =============================================================================
FROM python:3.11-slim as production

LABEL maintainer="GozerAI"
LABEL description="Nexus COO Strategic Intelligence Service"
LABEL version="0.1.0"

# Create non-root user
RUN groupadd -r nexus && useradd -r -g nexus nexus

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create directories
WORKDIR /app
RUN mkdir -p /app/data /app/logs && \
    chown -R nexus:nexus /app

# Copy application code
COPY --chown=nexus:nexus src/ ./src/
COPY --chown=nexus:nexus config/ ./config/

# Nexus package lives at src/nexus/src/nexus/ — include both paths so
# cross-service imports also resolve.
ENV PYTHONPATH=/app/src/nexus/src:/app/src
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check via HTTP endpoint
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER nexus

# Default command — start the federated service
CMD ["python", "-m", "nexus.service"]

# =============================================================================
# Development stage
# =============================================================================
FROM production as development

USER root

RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    ruff \
    mypy \
    ipython

USER nexus

CMD ["python", "-m", "nexus.service"]
