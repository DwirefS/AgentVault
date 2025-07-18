"""
End-to-end integration tests for AgentVault™
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
import json
import os
from datetime import datetime
import httpx
from typing import Dict, Any

from src.core.storage_orchestrator import AgentVaultOrchestrator
from src.api.rest_api import app
from fastapi.testclient import TestClient


class TestEndToEndIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def api_client(self):
        """Create test API client"""
        return TestClient(app)
    
    @pytest.fixture
    async def test_environment(self):
        """Set up test environment"""
        # Set test environment variables
        os.environ["AGENTVAULT_ENV"] = "test"
        os.environ["AZURE_SUBSCRIPTION_ID"] = "test-subscription"
        os.environ["AZURE_TENANT_ID"] = "test-tenant"
        os.environ["AZURE_CLIENT_ID"] = "test-client"
        os.environ["AZURE_CLIENT_SECRET"] = "test-secret"
        os.environ["ANF_SUBNET_ID"] = "test-subnet"
        os.environ["REDIS_URL"] = "redis://localhost:6379/1"
        
        yield
        
        # Cleanup
        for key in list(os.environ.keys()):
            if key.startswith("AGENTVAULT_") or key.startswith("AZURE_"):
                del os.environ[key]
    
    def test_health_check(self, api_client):
        """Test health check endpoint"""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data
    
    def test_agent_lifecycle(self, api_client, test_environment):
        """Test complete agent lifecycle"""
        # 1. Register agent
        agent_config = {
            "agent_id": "test-agent-lifecycle",
            "name": "Lifecycle Test Agent",
            "framework": "langchain",
            "memory_size_gb": 5,
            "performance_tier": "premium",
            "features": {
                "vector_search": True,
                "memory_persistence": True
            }
        }
        
        response = api_client.post("/agents/register", json=agent_config)
        assert response.status_code == 200
        registration = response.json()
        assert registration["success"] is True
        assert registration["agent_id"] == agent_config["agent_id"]
        
        # 2. Store data
        storage_request = {
            "agent_id": agent_config["agent_id"],
            "operation": "write",
            "key": "memory/conversation/001",
            "data": {
                "messages": [
                    {"role": "user", "content": "Hello, AgentVault!"},
                    {"role": "assistant", "content": "Hello! How can I help you?"}
                ],
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": "test-session-001"
                }
            }
        }
        
        response = api_client.post("/storage/request", json=storage_request)
        assert response.status_code == 200
        write_result = response.json()
        assert write_result["success"] is True
        
        # 3. Read data back
        read_request = {
            "agent_id": agent_config["agent_id"],
            "operation": "read",
            "key": "memory/conversation/001"
        }
        
        response = api_client.post("/storage/request", json=read_request)
        assert response.status_code == 200
        read_result = response.json()
        assert read_result["success"] is True
        assert read_result["data"]["messages"][0]["content"] == "Hello, AgentVault!"
        
        # 4. List stored items
        list_request = {
            "agent_id": agent_config["agent_id"],
            "operation": "list",
            "key": "memory/*"
        }
        
        response = api_client.post("/storage/request", json=list_request)
        assert response.status_code == 200
        list_result = response.json()
        assert list_result["success"] is True
        assert len(list_result["keys"]) >= 1
        assert "memory/conversation/001" in list_result["keys"]
        
        # 5. Get agent stats
        response = api_client.get(f"/agents/{agent_config['agent_id']}/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["agent_id"] == agent_config["agent_id"]
        assert stats["total_operations"] >= 3
        
        # 6. Delete data
        delete_request = {
            "agent_id": agent_config["agent_id"],
            "operation": "delete",
            "key": "memory/conversation/001"
        }
        
        response = api_client.post("/storage/request", json=delete_request)
        assert response.status_code == 200
        delete_result = response.json()
        assert delete_result["success"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, api_client, test_environment):
        """Test concurrent storage operations"""
        agent_id = "test-concurrent-agent"
        
        # Register agent
        agent_config = {
            "agent_id": agent_id,
            "name": "Concurrent Test Agent",
            "framework": "autogen",
            "memory_size_gb": 10
        }
        
        response = api_client.post("/agents/register", json=agent_config)
        assert response.status_code == 200
        
        # Prepare concurrent requests
        async def write_data(session: httpx.AsyncClient, key: str, value: str):
            request = {
                "agent_id": agent_id,
                "operation": "write",
                "key": f"concurrent/{key}",
                "data": {"value": value, "timestamp": datetime.utcnow().isoformat()}
            }
            response = await session.post(
                "http://localhost:8000/storage/request",
                json=request
            )
            return response.json()
        
        # Execute concurrent writes
        async with httpx.AsyncClient() as client:
            tasks = [
                write_data(client, f"key-{i}", f"value-{i}")
                for i in range(50)
            ]
            results = await asyncio.gather(*tasks)
        
        # Verify all writes succeeded
        assert all(r["success"] for r in results)
        
        # Verify data integrity
        list_request = {
            "agent_id": agent_id,
            "operation": "list",
            "key": "concurrent/*"
        }
        
        response = api_client.post("/storage/request", json=list_request)
        assert response.status_code == 200
        list_result = response.json()
        assert len(list_result["keys"]) == 50
    
    def test_framework_integration(self, api_client, test_environment):
        """Test framework-specific integrations"""
        frameworks = ["langchain", "autogen", "crewai"]
        
        for framework in frameworks:
            # Register framework-specific agent
            agent_config = {
                "agent_id": f"test-{framework}-agent",
                "name": f"{framework.title()} Test Agent",
                "framework": framework,
                "memory_size_gb": 5,
                "features": {
                    "vector_search": framework == "langchain",
                    "multi_agent": framework in ["autogen", "crewai"],
                    "memory_persistence": True
                }
            }
            
            response = api_client.post("/agents/register", json=agent_config)
            assert response.status_code == 200
            
            # Test framework-specific storage patterns
            if framework == "langchain":
                # Test vector storage
                vector_request = {
                    "agent_id": agent_config["agent_id"],
                    "operation": "write",
                    "key": "vectors/embedding-001",
                    "data": {
                        "text": "Sample text for embedding",
                        "vector": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "metadata": {"source": "test"}
                    }
                }
                response = api_client.post("/storage/request", json=vector_request)
                assert response.status_code == 200
                
            elif framework == "autogen":
                # Test conversation cache
                cache_request = {
                    "agent_id": agent_config["agent_id"],
                    "operation": "write",
                    "key": "cache/conversation-001",
                    "data": {
                        "messages": [],
                        "agents": ["assistant", "user_proxy"],
                        "config": {"max_turns": 10}
                    }
                }
                response = api_client.post("/storage/request", json=cache_request)
                assert response.status_code == 200
    
    def test_performance_metrics(self, api_client, test_environment):
        """Test performance metrics collection"""
        agent_id = "test-metrics-agent"
        
        # Register agent
        agent_config = {
            "agent_id": agent_id,
            "name": "Metrics Test Agent",
            "framework": "langchain",
            "memory_size_gb": 1
        }
        
        response = api_client.post("/agents/register", json=agent_config)
        assert response.status_code == 200
        
        # Perform various operations
        operations = ["write", "read", "write", "read", "list"]
        latencies = []
        
        for i, op in enumerate(operations):
            request = {
                "agent_id": agent_id,
                "operation": op,
                "key": f"metrics/test-{i}" if op != "list" else "metrics/*"
            }
            
            if op == "write":
                request["data"] = {"test": f"data-{i}"}
            
            response = api_client.post("/storage/request", json=request)
            assert response.status_code == 200
            result = response.json()
            
            if "metrics" in result:
                latencies.append(result["metrics"]["latency_ms"])
        
        # Verify metrics were collected
        assert len(latencies) > 0
        assert all(lat > 0 for lat in latencies)
        
        # Get Prometheus metrics
        response = api_client.get("/metrics")
        assert response.status_code == 200
        metrics_text = response.text
        
        # Verify key metrics are present
        assert "agentvault_requests_total" in metrics_text
        assert "agentvault_request_duration_seconds" in metrics_text
        assert "agentvault_active_agents" in metrics_text