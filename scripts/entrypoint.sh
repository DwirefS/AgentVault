#!/bin/bash
# AgentVault™ Container Entrypoint Script
# Handles initialization, configuration, and startup

set -e

echo "Starting AgentVault™ Enterprise AI Agent Storage Platform..."

# Environment variable defaults
export AGENTVAULT_ENV=${AGENTVAULT_ENV:-production}
export AGENTVAULT_LOG_LEVEL=${AGENTVAULT_LOG_LEVEL:-INFO}
export AGENTVAULT_PORT=${AGENTVAULT_PORT:-8000}
export AGENTVAULT_WORKERS=${AGENTVAULT_WORKERS:-4}

# Wait for dependencies
echo "Checking dependencies..."

# Check Redis connection
if [ -n "$REDIS_URL" ]; then
    echo "Waiting for Redis..."
    until redis-cli -u "$REDIS_URL" ping &>/dev/null; do
        echo "Redis is unavailable - sleeping"
        sleep 2
    done
    echo "Redis is up!"
fi

# Check Azure NetApp Files mount
if [ -n "$ANF_MOUNT_PATH" ]; then
    echo "Checking ANF mount at $ANF_MOUNT_PATH..."
    if mountpoint -q "$ANF_MOUNT_PATH"; then
        echo "ANF is mounted!"
    else
        echo "WARNING: ANF mount not detected at $ANF_MOUNT_PATH"
    fi
fi

# Initialize configuration
if [ ! -f "/app/configs/config.yaml" ]; then
    echo "Generating configuration from environment..."
    python -m agentvault.utils.config_generator
fi

# Run database migrations if needed
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python -m agentvault.utils.migrate
fi

# Initialize system
echo "Initializing AgentVault™ system..."
python -m agentvault.cli init --config /app/configs/config.yaml

# Start the service based on AGENTVAULT_SERVICE
case "$AGENTVAULT_SERVICE" in
    "api")
        echo "Starting REST API service..."
        exec uvicorn agentvault.api.rest_api:app \
            --host 0.0.0.0 \
            --port $AGENTVAULT_PORT \
            --workers $AGENTVAULT_WORKERS \
            --log-level ${AGENTVAULT_LOG_LEVEL,,} \
            --access-log
        ;;
    "orchestrator")
        echo "Starting Storage Orchestrator service..."
        exec python -m agentvault.core.orchestrator_service
        ;;
    "ml-worker")
        echo "Starting ML Worker service..."
        exec python -m agentvault.ml.worker_service
        ;;
    "monitor")
        echo "Starting Monitoring service..."
        exec python -m agentvault.monitoring.prometheus_exporter
        ;;
    *)
        echo "Starting default API service..."
        exec "$@"
        ;;
esac