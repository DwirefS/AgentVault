"""
Pytest configuration and shared fixtures for AgentVault tests
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

# Set test environment
os.environ["AGENTVAULT_ENV"] = "test"
os.environ["TESTING"] = "true"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp(prefix="agentvault_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_azure_credentials():
    """Mock Azure credentials for testing"""
    with patch('azure.identity.DefaultAzureCredential') as mock_cred:
        mock_instance = Mock()
        mock_instance.get_token.return_value = Mock(token="test-token")
        mock_cred.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_kubernetes_client():
    """Mock Kubernetes client for testing"""
    with patch('kubernetes.client.AppsV1Api') as mock_apps:
        with patch('kubernetes.client.CoreV1Api') as mock_core:
            with patch('kubernetes.config.load_incluster_config'):
                yield {
                    'apps_v1': mock_apps.return_value,
                    'core_v1': mock_core.return_value
                }


@pytest.fixture
def mock_docker():
    """Mock Docker subprocess calls"""
    with patch('subprocess.run') as mock_subprocess:
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "mock-container-id"
        mock_subprocess.return_value.stderr = ""
        yield mock_subprocess


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing"""
    with patch('redis.Redis') as mock_redis_class:
        mock_client = Mock()
        mock_client.ping.return_value = True
        mock_client.get.return_value = None
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1
        mock_redis_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_database():
    """Mock database connections and operations"""
    with patch('asyncpg.connect') as mock_connect:
        mock_conn = AsyncMock()
        mock_connect.return_value.__aenter__.return_value = mock_conn
        
        # Mock common database operations
        mock_conn.fetch.return_value = []
        mock_conn.fetchrow.return_value = None
        mock_conn.execute.return_value = "INSERT 0 1"
        
        yield mock_conn


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration for testing"""
    return {
        "agent_id": "test-agent-123",
        "name": "Test Agent",
        "agent_type": "langchain",
        "configuration": {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2048,
            "runtime_type": "docker",
            "docker_image": "agentvault/langchain:latest",
            "cpu_cores": 2,
            "memory_gb": 4,
            "storage_gb": 20
        },
        "capabilities": ["chat", "analysis", "code_generation"],
        "tags": ["test", "development"]
    }


@pytest.fixture
def sample_storage_config():
    """Sample storage configuration for testing"""
    return {
        "tiers": {
            "ultra": {
                "mount_point": "/test/ultra",
                "latency_ms": 0.1,
                "iops": 1000000,
                "cost_per_gb": 1.0
            },
            "premium": {
                "mount_point": "/test/premium",
                "latency_ms": 1.0,
                "iops": 500000,
                "cost_per_gb": 0.5
            },
            "standard": {
                "mount_point": "/test/standard",
                "latency_ms": 5.0,
                "iops": 100000,
                "cost_per_gb": 0.1
            }
        }
    }


@pytest.fixture
def sample_monitoring_config():
    """Sample monitoring configuration for testing"""
    return {
        "prometheus_enabled": True,
        "alert_manager_url": "http://localhost:9093",
        "smtp_server": "smtp.test.com",
        "smtp_port": 587,
        "smtp_username": "test@example.com",
        "smtp_password": "test-password",
        "email_to": ["admin@example.com"],
        "pagerduty_key": "test-pd-key",
        "azure_subscription_id": "test-subscription",
        "azure_resource_group": "test-rg"
    }


@pytest.fixture
def sample_security_config():
    """Sample security configuration for testing"""
    return {
        "encryption_enabled": True,
        "quantum_resistant": False,
        "compliance_level": "FIPS_140_2_LEVEL_2",
        "key_rotation_days": 30,
        "hsm_enabled": False,
        "azure_keyvault_url": "https://test.vault.azure.net/",
        "audit_logging": True
    }


@pytest.fixture
async def mock_anf_manager():
    """Mock ANF (Azure NetApp Files) manager"""
    mock_manager = AsyncMock()
    mock_manager.create_volume.return_value = {
        "volume_id": "test-volume-123",
        "mount_point": "/test/mount",
        "size_gb": 100,
        "tier": "premium"
    }
    mock_manager.delete_volume.return_value = True
    mock_manager.get_volume_stats.return_value = {
        "used_gb": 50,
        "available_gb": 50,
        "iops": 1000,
        "latency_ms": 1.0
    }
    return mock_manager


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = AsyncMock()
    mock_store.add_vectors.return_value = {"status": "success", "count": 100}
    mock_store.search.return_value = [
        {"id": "vec1", "score": 0.95, "metadata": {"text": "test"}},
        {"id": "vec2", "score": 0.89, "metadata": {"text": "example"}}
    ]
    mock_store.delete_vectors.return_value = {"status": "success", "deleted": 50}
    return mock_store


@pytest.fixture
def mock_cache():
    """Mock distributed cache for testing"""
    mock_cache = AsyncMock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.exists.return_value = False
    return mock_cache


@pytest.fixture
def cleanup_test_files():
    """Cleanup test files after test completion"""
    created_files = []
    
    def track_file(filepath):
        created_files.append(filepath)
        return filepath
    
    yield track_file
    
    # Cleanup
    for filepath in created_files:
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        except (OSError, FileNotFoundError):
            pass


@pytest.fixture
def mock_smtp_server():
    """Mock SMTP server for email testing"""
    with patch('smtplib.SMTP') as mock_smtp:
        mock_server = Mock()
        mock_smtp.return_value = mock_server
        mock_server.starttls.return_value = None
        mock_server.login.return_value = None
        mock_server.sendmail.return_value = {}
        mock_server.quit.return_value = None
        yield mock_server


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API testing"""
    with patch('aiohttp.ClientSession') as mock_session:
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_response.text = AsyncMock(return_value="success")
        
        mock_session_instance = AsyncMock()
        mock_session_instance.get.return_value.__aenter__.return_value = mock_response
        mock_session_instance.post.return_value.__aenter__.return_value = mock_response
        mock_session_instance.put.return_value.__aenter__.return_value = mock_response
        mock_session_instance.delete.return_value.__aenter__.return_value = mock_response
        
        mock_session.return_value.__aenter__.return_value = mock_session_instance
        yield mock_session_instance


# Test data generators
def generate_test_agent_data(count=10):
    """Generate test agent data"""
    agents = []
    for i in range(count):
        agent_data = {
            "id": f"test-agent-{i:03d}",
            "name": f"Test Agent {i}",
            "agent_type": ["langchain", "autogen", "crewai"][i % 3],
            "status": "running",
            "created_at": datetime.utcnow() - timedelta(days=i),
            "configuration": {
                "model": "gpt-4",
                "temperature": 0.7 + (i * 0.01),
                "cpu_cores": 1 + (i % 4),
                "memory_gb": 2 + (i % 8)
            }
        }
        agents.append(agent_data)
    return agents


def generate_test_metrics_data(count=100):
    """Generate test metrics data"""
    metrics = []
    base_time = datetime.utcnow()
    
    for i in range(count):
        metric = {
            "timestamp": base_time - timedelta(minutes=i),
            "metric_name": "cpu_usage",
            "value": 50.0 + (i % 50),
            "labels": {
                "agent_id": f"agent-{i % 10}",
                "instance": f"instance-{i % 5}"
            }
        }
        metrics.append(metric)
    
    return metrics


# Pytest hooks for test organization
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location"""
    for item in items:
        # Add markers based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add slow marker for tests with specific patterns
        if "performance" in item.name or "load" in item.name:
            item.add_marker(pytest.mark.slow)


# Test utilities
class TestUtils:
    """Utility class for common test operations"""
    
    @staticmethod
    def create_temp_file(content="", suffix=".txt"):
        """Create a temporary file with content"""
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'w') as tmp:
                tmp.write(content)
        except:
            os.close(fd)
            raise
        return path
    
    @staticmethod
    def create_test_config(overrides=None):
        """Create test configuration with optional overrides"""
        base_config = {
            "environment": "test",
            "debug": True,
            "database_url": "postgresql://test:test@localhost/test",
            "redis_url": "redis://localhost:6379/1",
            "log_level": "DEBUG"
        }
        
        if overrides:
            base_config.update(overrides)
        
        return base_config
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout=10, interval=0.1):
        """Wait for a condition to become true"""
        start_time = datetime.utcnow()
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                return True
            await asyncio.sleep(interval)
        return False