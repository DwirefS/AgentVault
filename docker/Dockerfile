# AgentVault™ Production Docker Image
# Multi-stage build for optimized production deployment
# Author: Dwiref Sharma
# Contact: DwirefS@SapientEdge.io

# Stage 1: Builder
FROM python:3.11-slim-bullseye AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .
COPY README.md .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn uvicorn[standard]

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Install the package
RUN pip install --no-cache-dir -e .

# Stage 2: Runtime
FROM python:3.11-slim-bullseye

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    zlib1g \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r agentvault && useradd -r -g agentvault -u 1000 agentvault

# Create necessary directories
RUN mkdir -p /app /var/log/agentvault /var/cache/agentvault /models \
    && chown -R agentvault:agentvault /app /var/log/agentvault /var/cache/agentvault /models

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application from builder
COPY --from=builder --chown=agentvault:agentvault /build/src /app/src
COPY --from=builder --chown=agentvault:agentvault /build/configs /app/configs

# Set working directory
WORKDIR /app

# Environment variables for production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    PORT=8080

# Azure AD authentication environment variables (set at runtime)
ENV AZURE_TENANT_ID="" \
    AZURE_CLIENT_ID="" \
    AZURE_CLIENT_SECRET="" \
    AZURE_SUBSCRIPTION_ID="" \
    AZURE_RESOURCE_GROUP="" \
    AZURE_KEY_VAULT_URL="" \
    AZURE_MONITOR_WORKSPACE_ID=""

# Health check script
COPY --chown=agentvault:agentvault docker/healthcheck.sh /usr/local/bin/healthcheck
RUN chmod +x /usr/local/bin/healthcheck

# Switch to non-root user
USER agentvault

# Expose ports
EXPOSE 8080 8081 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck || exit 1

# Use tini as entrypoint to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run the application
CMD ["python", "-m", "src.startup"]