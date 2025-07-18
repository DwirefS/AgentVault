# AgentVault™ Production Dockerfile
# Multi-stage build for optimal size and security

# Build stage
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    libxml2-dev \
    libxslt1-dev \
    libjpeg-dev \
    zlib1g-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install --user --no-cache-dir -e .

# Runtime stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libxml2 \
    libxslt1.1 \
    libjpeg62-turbo \
    curl \
    nfs-common \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash agentvault

# Copy Python packages from builder
COPY --from=builder /root/.local /home/agentvault/.local

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=agentvault:agentvault src/ ./src/
COPY --chown=agentvault:agentvault configs/ ./configs/
COPY --chown=agentvault:agentvault scripts/entrypoint.sh ./

# Create necessary directories
RUN mkdir -p /mnt/agentvault /var/log/agentvault /app/data \
    && chown -R agentvault:agentvault /mnt/agentvault /var/log/agentvault /app/data

# Switch to non-root user
USER agentvault

# Add local bin to PATH
ENV PATH=/home/agentvault/.local/bin:$PATH

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports
EXPOSE 8000 8001 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["python", "-m", "agentvault.api.rest_api"]