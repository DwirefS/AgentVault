#!/bin/bash
# AgentVault Health Check Script

# Check if the API is responding
curl -f http://localhost:8080/health || exit 1

# Check if the metrics endpoint is responding
curl -f http://localhost:8081/metrics || exit 1

# Check if the service is ready
curl -f http://localhost:8080/ready || exit 1

exit 0