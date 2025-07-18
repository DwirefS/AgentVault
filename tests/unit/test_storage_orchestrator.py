"""
Unit tests for Storage Orchestrator
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from src.core.storage_orchestrator import (
    AgentVaultOrchestrator,
    StorageRequest,
    StorageResponse,
    OperationType
)
from src.storage.tier_manager import StorageTier


class TestStorageOrchestrator:
    """Test suite for AgentVaultOrchestrator"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance for testing"""
        with patch('src.core.storage_orchestrator.ANFManager'):
            with patch('src.core.storage_orchestrator.TierManager'):
                with patch('src.core.storage_orchestrator.RedisCache'):
                    orchestrator = AgentVaultOrchestrator()
                    await orchestrator.initialize()
                    yield orchestrator
                    await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_register_agent(self, orchestrator):
        """Test agent registration"""
        agent_config = {
            "agent_id": "test-agent-001",
            "name": "Test Agent",
            "framework": "langchain",
            "memory_size_gb": 10,
            "performance_tier": "premium"
        }
        
        response = await orchestrator.register_agent(agent_config)
        
        assert response["success"] is True
        assert response["agent_id"] == "test-agent-001"
        assert "volume_id" in response
        assert response["status"] == "registered"
    
    @pytest.mark.asyncio
    async def test_process_write_request(self, orchestrator):
        """Test write operation processing"""
        request = StorageRequest(
            agent_id="test-agent-001",
            operation=OperationType.WRITE,
            key="memory/conversation/123",
            data={"message": "Hello, AgentVault!"},
            metadata={"timestamp": datetime.utcnow().isoformat()}
        )
        
        # Mock dependencies
        orchestrator.dna_profiler.get_agent_profile = AsyncMock(return_value={
            "optimal_tier": StorageTier.PREMIUM,
            "compression_recommended": True
        })
        orchestrator.tier_manager.write_data = AsyncMock(return_value={
            "success": True,
            "location": "premium/test-agent-001/memory/conversation/123"
        })
        
        response = await orchestrator.process_storage_request(request)
        
        assert response["success"] is True
        assert response["operation"] == "write"
        assert "location" in response
        assert response["metrics"]["latency_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_process_read_request_with_cache(self, orchestrator):
        """Test read operation with cache hit"""
        request = StorageRequest(
            agent_id="test-agent-001",
            operation=OperationType.READ,
            key="memory/conversation/123"
        )
        
        # Mock cache hit
        cached_data = {"message": "Cached Hello!"}
        orchestrator.cache.get = AsyncMock(return_value=json.dumps(cached_data))
        
        response = await orchestrator.process_storage_request(request)
        
        assert response["success"] is True
        assert response["data"] == cached_data
        assert response["cache_hit"] is True
        assert response["metrics"]["latency_ms"] < 10  # Cache should be fast
    
    @pytest.mark.asyncio
    async def test_process_read_request_without_cache(self, orchestrator):
        """Test read operation with cache miss"""
        request = StorageRequest(
            agent_id="test-agent-001",
            operation=OperationType.READ,
            key="memory/conversation/456"
        )
        
        # Mock cache miss and storage read
        orchestrator.cache.get = AsyncMock(return_value=None)
        orchestrator.tier_manager.read_data = AsyncMock(return_value={
            "success": True,
            "data": {"message": "Hello from storage!"},
            "tier": "premium"
        })
        
        response = await orchestrator.process_storage_request(request)
        
        assert response["success"] is True
        assert response["data"]["message"] == "Hello from storage!"
        assert response["cache_hit"] is False
        
        # Verify cache was populated
        orchestrator.cache.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_operation(self, orchestrator):
        """Test delete operation"""
        request = StorageRequest(
            agent_id="test-agent-001",
            operation=OperationType.DELETE,
            key="memory/old_conversation"
        )
        
        orchestrator.tier_manager.delete_data = AsyncMock(return_value={
            "success": True,
            "deleted_count": 1
        })
        
        response = await orchestrator.process_storage_request(request)
        
        assert response["success"] is True
        assert response["operation"] == "delete"
        assert "deleted_count" in response
    
    @pytest.mark.asyncio
    async def test_list_operation(self, orchestrator):
        """Test list operation"""
        request = StorageRequest(
            agent_id="test-agent-001",
            operation=OperationType.LIST,
            key="memory/*"
        )
        
        orchestrator.tier_manager.list_data = AsyncMock(return_value={
            "success": True,
            "keys": ["memory/conversation/123", "memory/conversation/456"],
            "count": 2
        })
        
        response = await orchestrator.process_storage_request(request)
        
        assert response["success"] is True
        assert response["operation"] == "list"
        assert len(response["keys"]) == 2
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test error handling in orchestrator"""
        request = StorageRequest(
            agent_id="test-agent-001",
            operation=OperationType.WRITE,
            key="memory/test",
            data={"test": "data"}
        )
        
        # Mock an error
        orchestrator.tier_manager.write_data = AsyncMock(
            side_effect=Exception("Storage error")
        )
        
        response = await orchestrator.process_storage_request(request)
        
        assert response["success"] is False
        assert "error" in response
        assert "Storage error" in response["error"]
    
    @pytest.mark.asyncio
    async def test_request_validation(self, orchestrator):
        """Test request validation"""
        # Test invalid operation
        with pytest.raises(ValueError):
            request = StorageRequest(
                agent_id="test-agent",
                operation="INVALID",
                key="test"
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, orchestrator):
        """Test handling concurrent requests"""
        requests = [
            StorageRequest(
                agent_id=f"agent-{i}",
                operation=OperationType.WRITE,
                key=f"key-{i}",
                data={"value": i}
            )
            for i in range(10)
        ]
        
        orchestrator.tier_manager.write_data = AsyncMock(return_value={
            "success": True,
            "location": "test"
        })
        
        # Process requests concurrently
        responses = await asyncio.gather(*[
            orchestrator.process_storage_request(req) 
            for req in requests
        ])
        
        assert len(responses) == 10
        assert all(r["success"] for r in responses)
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, orchestrator):
        """Test metrics collection during operations"""
        request = StorageRequest(
            agent_id="test-agent-001",
            operation=OperationType.WRITE,
            key="metrics_test",
            data={"test": "data"}
        )
        
        orchestrator.tier_manager.write_data = AsyncMock(return_value={
            "success": True,
            "location": "test"
        })
        
        response = await orchestrator.process_storage_request(request)
        
        assert "metrics" in response
        assert "latency_ms" in response["metrics"]
        assert "data_size_bytes" in response["metrics"]
        assert response["metrics"]["latency_ms"] >= 0
        assert response["metrics"]["data_size_bytes"] > 0


class TestStorageRequest:
    """Test StorageRequest model"""
    
    def test_valid_request_creation(self):
        """Test creating valid storage request"""
        request = StorageRequest(
            agent_id="test-agent",
            operation=OperationType.WRITE,
            key="test-key",
            data={"test": "data"},
            metadata={"version": "1.0"}
        )
        
        assert request.agent_id == "test-agent"
        assert request.operation == OperationType.WRITE
        assert request.key == "test-key"
        assert request.data == {"test": "data"}
        assert request.metadata["version"] == "1.0"
    
    def test_request_serialization(self):
        """Test request serialization"""
        request = StorageRequest(
            agent_id="test-agent",
            operation=OperationType.READ,
            key="test-key"
        )
        
        serialized = request.to_dict()
        assert serialized["agent_id"] == "test-agent"
        assert serialized["operation"] == "read"
        assert serialized["key"] == "test-key"
    
    def test_request_validation(self):
        """Test request validation"""
        # Missing required fields should raise error
        with pytest.raises(TypeError):
            StorageRequest(
                operation=OperationType.WRITE,
                key="test"
            )