"""
AgentVault™ Storage Orchestrator Tests
Comprehensive test suite for core orchestration functionality

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.core.storage_orchestrator import (
    AgentVaultOrchestrator,
    StorageRequest,
    AgentStorageProfile,
    StorageTier,
    EncryptionLevel
)


@pytest.fixture
async def orchestrator():
    """Create a test orchestrator instance"""
    config = {
        "azure": {
            "subscription_id": "test-subscription",
            "resource_group": "test-rg",
            "location": "eastus2"
        },
        "anf": {
            "account_name": "test-anf",
            "subnet_id": "/subscriptions/test/subnets/test-subnet"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": "test-password"
        },
        "security": {
            "key_vault_url": "https://test-kv.vault.azure.net/",
            "default_encryption_level": "enhanced"
        }
    }
    
    orchestrator = AgentVaultOrchestrator(config)
    
    # Mock Azure clients
    orchestrator.netapp_client = Mock()
    orchestrator.resource_client = Mock()
    orchestrator.redis_client = AsyncMock()
    
    # Mock subsystems
    orchestrator.anf_manager = AsyncMock()
    orchestrator.tier_manager = AsyncMock()
    orchestrator.agent_dna_profiler = AsyncMock()
    orchestrator.cognitive_balancer = AsyncMock()
    orchestrator.encryption_manager = AsyncMock()
    orchestrator.telemetry = AsyncMock()
    
    # Initialize
    orchestrator.is_initialized = True
    
    yield orchestrator
    
    # Cleanup
    await orchestrator.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    """Test orchestrator initialization"""
    # Reset initialization
    orchestrator.is_initialized = False
    
    # Mock initialization methods
    orchestrator._load_agent_profiles = AsyncMock()
    
    # Initialize
    await orchestrator.initialize()
    
    # Verify initialization
    assert orchestrator.is_initialized
    orchestrator.anf_manager.initialize.assert_called_once()
    orchestrator.tier_manager.initialize.assert_called_once()
    orchestrator.agent_dna_profiler.initialize.assert_called_once()
    orchestrator.cognitive_balancer.initialize.assert_called_once()
    orchestrator.encryption_manager.initialize.assert_called_once()
    orchestrator.telemetry.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_register_agent(orchestrator):
    """Test agent registration"""
    agent_id = "test-agent-001"
    agent_type = "langchain"
    config = {
        "performance": {"latency_requirement": 0.1},
        "security": {"encryption_required": True}
    }
    
    # Mock DNA profile creation
    orchestrator.agent_dna_profiler.create_profile.return_value = {
        "dna_id": "dna-12345",
        "features": {},
        "confidence": 0.5
    }
    
    # Register agent
    profile = await orchestrator.register_agent(agent_id, agent_type, config)
    
    # Verify registration
    assert profile.agent_id == agent_id
    assert profile.agent_type == agent_type
    assert agent_id in orchestrator.agent_profiles
    orchestrator.agent_dna_profiler.create_profile.assert_called_once()


@pytest.mark.asyncio
async def test_process_storage_request_success(orchestrator):
    """Test successful storage request processing"""
    # Create test agent profile
    agent_id = "test-agent-001"
    profile = AgentStorageProfile(
        agent_id=agent_id,
        agent_type="langchain",
        created_at=datetime.utcnow()
    )
    orchestrator.agent_profiles[agent_id] = profile
    
    # Create storage request
    request = StorageRequest(
        agent_id=agent_id,
        operation="write",
        data_type="vector",
        data_size=1024,
        priority="high",
        latency_requirement=0.1,
        metadata={"data": [0.1, 0.2, 0.3]}
    )
    
    # Mock responses
    orchestrator._determine_optimal_tier = AsyncMock(return_value=StorageTier.ULTRA)
    orchestrator._determine_storage_location = AsyncMock(return_value="eastus2")
    orchestrator.encryption_manager.encrypt_request = AsyncMock(return_value=request)
    orchestrator.anf_manager.execute_operation.return_value = {
        "success": True,
        "result": "data-written"
    }
    
    # Process request
    result = await orchestrator.process_storage_request(request)
    
    # Verify result
    assert result["success"]
    assert result["request_id"] == request.request_id
    assert result["tier_used"] == StorageTier.ULTRA.value
    assert result["location"] == "eastus2"


@pytest.mark.asyncio
async def test_process_storage_request_encryption(orchestrator):
    """Test storage request with encryption"""
    agent_id = "secure-agent-001"
    profile = AgentStorageProfile(
        agent_id=agent_id,
        agent_type="langchain",
        created_at=datetime.utcnow()
    )
    orchestrator.agent_profiles[agent_id] = profile
    
    # Create request requiring encryption
    request = StorageRequest(
        agent_id=agent_id,
        operation="write",
        data_type="sensitive_data",
        data_size=2048,
        priority="critical",
        latency_requirement=0.05,
        encryption_required=True,
        compliance_tags=["HIPAA", "PHI"],
        metadata={"data": "sensitive information"}
    )
    
    # Mock encryption
    encrypted_request = StorageRequest(
        agent_id=agent_id,
        operation="write",
        data_type="sensitive_data",
        data_size=2048,
        priority="critical",
        latency_requirement=0.05,
        metadata={"encrypted_data": "encrypted_blob"}
    )
    orchestrator.encryption_manager.encrypt_request.return_value = encrypted_request
    
    # Mock other responses
    orchestrator._determine_optimal_tier = AsyncMock(return_value=StorageTier.ULTRA)
    orchestrator._determine_storage_location = AsyncMock(return_value="eastus2")
    orchestrator.anf_manager.execute_operation.return_value = {
        "success": True,
        "result": "encrypted-data-written"
    }
    
    # Process request
    result = await orchestrator.process_storage_request(request)
    
    # Verify encryption was applied
    orchestrator.encryption_manager.encrypt_request.assert_called_once_with(request)
    assert result["success"]


@pytest.mark.asyncio
async def test_determine_optimal_tier(orchestrator):
    """Test optimal tier determination"""
    agent_id = "test-agent-001"
    profile = AgentStorageProfile(
        agent_id=agent_id,
        agent_type="langchain",
        storage_dna={"preferred_tier": "premium"},
        created_at=datetime.utcnow()
    )
    
    # Test critical priority
    request = StorageRequest(
        agent_id=agent_id,
        operation="read",
        data_type="vector",
        priority="critical"
    )
    tier = await orchestrator._determine_optimal_tier(request, profile)
    assert tier == StorageTier.ULTRA
    
    # Test latency requirement
    request.priority = "normal"
    request.latency_requirement = 0.05
    tier = await orchestrator._determine_optimal_tier(request, profile)
    assert tier == StorageTier.ULTRA
    
    # Test data type specific
    request.latency_requirement = 10.0
    request.data_type = "chat_history"
    tier = await orchestrator._determine_optimal_tier(request, profile)
    assert tier == StorageTier.STANDARD
    
    # Test DNA preference
    request.data_type = "unknown"
    tier = await orchestrator._determine_optimal_tier(request, profile)
    assert tier == StorageTier.PREMIUM


@pytest.mark.asyncio
async def test_update_agent_access_patterns(orchestrator):
    """Test updating agent access patterns"""
    agent_id = "pattern-test-001"
    profile = AgentStorageProfile(
        agent_id=agent_id,
        agent_type="autogen",
        created_at=datetime.utcnow()
    )
    orchestrator.agent_profiles[agent_id] = profile
    
    # Create multiple requests
    requests = []
    for i in range(10):
        request = StorageRequest(
            agent_id=agent_id,
            operation="read" if i % 2 == 0 else "write",
            data_type="vector",
            data_size=1024 * (i + 1),
            priority="normal"
        )
        requests.append(request)
    
    # Update patterns
    for request in requests:
        await orchestrator._update_agent_access_patterns(request, profile)
    
    # Verify DNA profiler was called
    assert orchestrator.agent_dna_profiler.update_access_patterns.call_count == 10


@pytest.mark.asyncio
async def test_performance_metrics_collection(orchestrator):
    """Test performance metrics collection"""
    # Process multiple requests
    agent_id = "metrics-test-001"
    profile = AgentStorageProfile(
        agent_id=agent_id,
        agent_type="langchain",
        created_at=datetime.utcnow()
    )
    orchestrator.agent_profiles[agent_id] = profile
    
    # Mock responses
    orchestrator._determine_optimal_tier = AsyncMock(return_value=StorageTier.PREMIUM)
    orchestrator._determine_storage_location = AsyncMock(return_value="eastus2")
    orchestrator.encryption_manager.encrypt_request = AsyncMock(side_effect=lambda x: x)
    orchestrator.anf_manager.execute_operation.return_value = {
        "success": True,
        "result": "data"
    }
    
    # Process multiple requests
    for i in range(5):
        request = StorageRequest(
            agent_id=agent_id,
            operation="write",
            data_type="vector",
            data_size=1024
        )
        await orchestrator.process_storage_request(request)
    
    # Check metrics
    metrics = orchestrator.metrics
    assert metrics["requests_total"]._value._value > 0
    assert len(orchestrator.agent_profiles) == 1


@pytest.mark.asyncio
async def test_error_handling(orchestrator):
    """Test error handling in request processing"""
    agent_id = "error-test-001"
    profile = AgentStorageProfile(
        agent_id=agent_id,
        agent_type="langchain",
        created_at=datetime.utcnow()
    )
    orchestrator.agent_profiles[agent_id] = profile
    
    # Create request
    request = StorageRequest(
        agent_id=agent_id,
        operation="write",
        data_type="vector",
        data_size=1024
    )
    
    # Mock error
    orchestrator.anf_manager.execute_operation.side_effect = Exception("Storage error")
    orchestrator._determine_optimal_tier = AsyncMock(return_value=StorageTier.ULTRA)
    orchestrator._determine_storage_location = AsyncMock(return_value="eastus2")
    orchestrator.encryption_manager.encrypt_request = AsyncMock(return_value=request)
    
    # Process request
    result = await orchestrator.process_storage_request(request)
    
    # Verify error handling
    assert not result["success"]
    assert "error" in result
    assert "Storage error" in result["error"]


@pytest.mark.asyncio
async def test_concurrent_requests(orchestrator):
    """Test handling concurrent storage requests"""
    # Create multiple agents
    agents = []
    for i in range(10):
        agent_id = f"concurrent-agent-{i:03d}"
        profile = AgentStorageProfile(
            agent_id=agent_id,
            agent_type="langchain",
            created_at=datetime.utcnow()
        )
        orchestrator.agent_profiles[agent_id] = profile
        agents.append(agent_id)
    
    # Mock responses
    orchestrator._determine_optimal_tier = AsyncMock(return_value=StorageTier.PREMIUM)
    orchestrator._determine_storage_location = AsyncMock(return_value="eastus2")
    orchestrator.encryption_manager.encrypt_request = AsyncMock(side_effect=lambda x: x)
    orchestrator.anf_manager.execute_operation.return_value = {
        "success": True,
        "result": "data"
    }
    
    # Create concurrent requests
    tasks = []
    for agent_id in agents:
        request = StorageRequest(
            agent_id=agent_id,
            operation="write",
            data_type="vector",
            data_size=1024
        )
        task = orchestrator.process_storage_request(request)
        tasks.append(task)
    
    # Execute concurrently
    results = await asyncio.gather(*tasks)
    
    # Verify all succeeded
    assert len(results) == 10
    assert all(r["success"] for r in results)


@pytest.mark.asyncio
async def test_cache_behavior(orchestrator):
    """Test caching behavior"""
    agent_id = "cache-test-001"
    profile = AgentStorageProfile(
        agent_id=agent_id,
        agent_type="langchain",
        created_at=datetime.utcnow()
    )
    
    # Test profile caching
    orchestrator.agent_profiles[agent_id] = profile
    
    # First call - should be in memory
    cached_profile = await orchestrator._get_agent_profile(agent_id)
    assert cached_profile == profile
    
    # Test Redis cache
    orchestrator.agent_profiles.clear()
    orchestrator.redis_client.get.return_value = json.dumps(profile.__dict__, default=str)
    
    cached_profile = await orchestrator._get_agent_profile(agent_id)
    assert cached_profile.agent_id == agent_id
    orchestrator.redis_client.get.assert_called_once()


@pytest.mark.asyncio
async def test_background_tasks(orchestrator):
    """Test background optimization tasks"""
    # Mock methods called by background tasks
    orchestrator.tier_manager.optimize_tiers = AsyncMock()
    orchestrator.cognitive_balancer.rebalance_load = AsyncMock()
    orchestrator._update_storage_metrics = AsyncMock()
    orchestrator.anf_manager.get_performance_metrics = AsyncMock(return_value={})
    orchestrator._detect_performance_anomalies = AsyncMock()
    orchestrator.agent_dna_profiler.evolve_profile = AsyncMock(return_value={})
    
    # Manually trigger background task methods
    await orchestrator._optimize_storage_continuously()
    await orchestrator._monitor_performance()
    await orchestrator._update_agent_dna()
    
    # Verify they were called
    orchestrator.tier_manager.optimize_tiers.assert_called()
    orchestrator.cognitive_balancer.rebalance_load.assert_called()
    orchestrator.anf_manager.get_performance_metrics.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])