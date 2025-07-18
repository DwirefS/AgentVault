"""
Unit tests for Agent Factory and Runtime implementations
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.agents.agent_factory import (
    AgentFactory,
    LangChainRuntime,
    AutoGenRuntime,
    CrewAIRuntime,
    CustomRuntime
)
from src.database.models import Agent


class TestAgentFactory:
    """Test suite for AgentFactory"""
    
    @pytest.fixture
    def mock_storage_manager(self):
        """Mock storage manager"""
        mock = AsyncMock()
        mock.update_agent = AsyncMock()
        return mock
    
    @pytest.fixture
    def agent_factory(self, mock_storage_manager):
        """Create agent factory instance"""
        return AgentFactory(storage_manager=mock_storage_manager)
    
    @pytest.fixture
    def sample_agent(self):
        """Create sample agent for testing"""
        return Agent(
            id="test-agent-123",
            name="Test Agent",
            agent_type="langchain",
            configuration={
                "model": "gpt-4",
                "temperature": 0.7,
                "docker_image": "test/langchain:latest"
            }
        )
    
    @pytest.mark.asyncio
    async def test_create_langchain_runtime(self, agent_factory, sample_agent):
        """Test LangChain runtime creation"""
        runtime = await agent_factory.create_runtime(sample_agent)
        assert isinstance(runtime, LangChainRuntime)
        assert runtime.agent == sample_agent
        assert runtime.factory == agent_factory
    
    @pytest.mark.asyncio
    async def test_create_autogen_runtime(self, agent_factory, sample_agent):
        """Test AutoGen runtime creation"""
        sample_agent.agent_type = "autogen"
        runtime = await agent_factory.create_runtime(sample_agent)
        assert isinstance(runtime, AutoGenRuntime)
    
    @pytest.mark.asyncio
    async def test_create_crewai_runtime(self, agent_factory, sample_agent):
        """Test CrewAI runtime creation"""
        sample_agent.agent_type = "crewai"
        runtime = await agent_factory.create_runtime(sample_agent)
        assert isinstance(runtime, CrewAIRuntime)
    
    @pytest.mark.asyncio
    async def test_create_custom_runtime(self, agent_factory, sample_agent):
        """Test custom runtime creation"""
        sample_agent.agent_type = "custom"
        runtime = await agent_factory.create_runtime(sample_agent)
        assert isinstance(runtime, CustomRuntime)
    
    @pytest.mark.asyncio
    async def test_unsupported_agent_type(self, agent_factory, sample_agent):
        """Test unsupported agent type raises error"""
        sample_agent.agent_type = "unsupported"
        with pytest.raises(ValueError, match="Unsupported agent type"):
            await agent_factory.create_runtime(sample_agent)


class TestLangChainRuntime:
    """Test suite for LangChain Runtime"""
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent"""
        agent = Mock()
        agent.id = "test-agent-123"
        agent.name = "Test Agent"
        agent.agent_type = "langchain"
        agent.configuration = {
            "model": "gpt-4",
            "docker_image": "test/langchain:latest"
        }
        return agent
    
    @pytest.fixture
    def mock_factory(self):
        """Mock factory"""
        factory = Mock()
        factory.storage_manager = AsyncMock()
        return factory
    
    @pytest.fixture
    def runtime(self, mock_agent, mock_factory):
        """Create runtime instance"""
        return LangChainRuntime(mock_agent, mock_factory)
    
    @pytest.mark.asyncio
    async def test_get_port_generation(self, runtime):
        """Test port generation is deterministic"""
        port1 = runtime._get_port()
        port2 = runtime._get_port()
        assert port1 == port2  # Should be deterministic
        assert 30000 <= port1 <= 40000  # Within expected range
    
    @pytest.mark.asyncio
    async def test_get_container_name(self, runtime):
        """Test container name generation"""
        container_name = runtime._get_container_name()
        expected = f"agentvault-{runtime.agent.agent_type}-{runtime.agent.id}"
        assert container_name == expected
    
    @pytest.mark.asyncio
    async def test_get_api_endpoint(self, runtime):
        """Test API endpoint generation"""
        endpoint = runtime._get_api_endpoint()
        expected_port = runtime._get_port()
        assert f"localhost:{expected_port}" in endpoint
    
    @pytest.mark.asyncio
    async def test_initialize_creates_container_config(self, runtime):
        """Test initialization creates proper container configuration"""
        config = {
            "openai_api_key": "test-key",
            "model": "gpt-4"
        }
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "container-id-123"
            
            await runtime.initialize(config)
            
            assert runtime._initialized
            assert mock_subprocess.called
    
    @pytest.mark.asyncio
    async def test_start_requires_initialization(self, runtime):
        """Test start requires initialization"""
        with pytest.raises(RuntimeError, match="Agent runtime not initialized"):
            await runtime.start()
    
    @pytest.mark.asyncio
    @patch('subprocess.run')
    async def test_get_status_returns_agent_info(self, mock_subprocess, runtime):
        """Test get_status returns proper agent information"""
        runtime._initialized = True
        runtime._running = True
        runtime.container_id = "test-container"
        
        status = await runtime.get_status()
        
        assert status['initialized'] is True
        assert status['running'] is True
        assert status['agent_id'] == str(runtime.agent.id)
        assert status['container_id'] == "test-container"


class TestAutoGenRuntime:
    """Test suite for AutoGen Runtime"""
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent"""
        agent = Mock()
        agent.id = "autogen-test-123"
        agent.name = "AutoGen Test Agent"
        agent.agent_type = "autogen"
        agent.configuration = {
            "openai_api_key": "test-key",
            "model": "gpt-4"
        }
        return agent
    
    @pytest.fixture
    def mock_factory(self):
        """Mock factory"""
        factory = Mock()
        factory.storage_manager = AsyncMock()
        return factory
    
    @pytest.fixture
    def runtime(self, mock_agent, mock_factory):
        """Create AutoGen runtime instance"""
        return AutoGenRuntime(mock_agent, mock_factory)
    
    @pytest.mark.asyncio
    async def test_initialize_creates_autogen_config(self, runtime):
        """Test AutoGen initialization creates proper config"""
        config = {
            "openai_api_key": "test-key",
            "model": "gpt-4",
            "system_message": "You are a helpful assistant"
        }
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "autogen-container-123"
            
            await runtime.initialize(config)
            
            assert runtime._initialized
            # Verify AutoGen-specific configuration was created
            assert mock_subprocess.called


class TestCrewAIRuntime:
    """Test suite for CrewAI Runtime"""
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent"""
        agent = Mock()
        agent.id = "crewai-test-123"
        agent.name = "CrewAI Test Agent"
        agent.agent_type = "crewai"
        agent.configuration = {
            "role": "Data Analyst",
            "goal": "Analyze data efficiently",
            "backstory": "Expert in data analysis"
        }
        return agent
    
    @pytest.fixture
    def mock_factory(self):
        """Mock factory"""
        factory = Mock()
        factory.storage_manager = AsyncMock()
        return factory
    
    @pytest.fixture
    def runtime(self, mock_agent, mock_factory):
        """Create CrewAI runtime instance"""
        return CrewAIRuntime(mock_agent, mock_factory)
    
    @pytest.mark.asyncio
    async def test_initialize_creates_crewai_config(self, runtime):
        """Test CrewAI initialization creates proper config"""
        config = {
            "role": "Data Analyst",
            "goal": "Analyze data efficiently",
            "backstory": "Expert in data analysis",
            "verbose": True
        }
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "crewai-container-123"
            
            await runtime.initialize(config)
            
            assert runtime._initialized
            assert runtime.container_id == "crewai-container-123"


class TestCustomRuntime:
    """Test suite for Custom Runtime"""
    
    @pytest.fixture
    def mock_agent(self):
        """Mock agent"""
        agent = Mock()
        agent.id = "custom-test-123"
        agent.name = "Custom Test Agent"
        agent.agent_type = "custom"
        agent.configuration = {
            "runtime_type": "python",
            "script_path": "/path/to/script.py"
        }
        return agent
    
    @pytest.fixture
    def mock_factory(self):
        """Mock factory"""
        factory = Mock()
        factory.storage_manager = AsyncMock()
        return factory
    
    @pytest.fixture
    def runtime(self, mock_agent, mock_factory):
        """Create custom runtime instance"""
        return CustomRuntime(mock_agent, mock_factory)
    
    @pytest.mark.asyncio
    async def test_initialize_python_runtime(self, runtime):
        """Test Python runtime initialization"""
        config = {
            "runtime_type": "python",
            "script_path": "/path/to/script.py"
        }
        
        await runtime.initialize(config)
        assert runtime._initialized
    
    @pytest.mark.asyncio
    async def test_initialize_unsupported_runtime(self, runtime):
        """Test unsupported runtime type raises error"""
        config = {
            "runtime_type": "unsupported"
        }
        
        with pytest.raises(ValueError, match="Unsupported custom runtime type"):
            await runtime.initialize(config)
    
    @pytest.mark.asyncio
    async def test_start_python_process(self, runtime):
        """Test starting Python process"""
        runtime._initialized = True
        runtime.agent.configuration["runtime_type"] = "python"
        runtime.agent.configuration["script_path"] = "/path/to/script.py"
        
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.returncode = None
            mock_subprocess.return_value = mock_process
            
            await runtime.start()
            
            assert runtime._running
            assert runtime.process == mock_process
            mock_subprocess.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_python_process(self, runtime):
        """Test stopping Python process"""
        runtime._running = True
        mock_process = AsyncMock()
        mock_process.terminate = Mock()
        mock_process.wait = AsyncMock()
        runtime.process = mock_process
        runtime.agent.configuration["runtime_type"] = "python"
        
        await runtime.stop()
        
        assert not runtime._running
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()


class TestRuntimeErrorHandling:
    """Test error handling across all runtimes"""
    
    @pytest.mark.asyncio
    async def test_container_creation_failure(self):
        """Test handling of container creation failure"""
        agent = Mock()
        agent.id = "test-agent"
        agent.agent_type = "langchain"
        agent.configuration = {}
        
        factory = Mock()
        factory.storage_manager = AsyncMock()
        
        runtime = LangChainRuntime(agent, factory)
        
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 1
            mock_subprocess.return_value.stderr = "Container creation failed"
            
            with pytest.raises(RuntimeError, match="Failed to create container"):
                await runtime.initialize({})
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self):
        """Test health check timeout handling"""
        agent = Mock()
        agent.id = "test-agent"
        agent.agent_type = "langchain"
        
        factory = Mock()
        runtime = LangChainRuntime(agent, factory)
        runtime.container_id = "test-container"
        
        with patch('subprocess.run') as mock_subprocess:
            # Mock container running but health check failing
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "true"
            
            with patch('aiohttp.ClientSession') as mock_session:
                mock_session.return_value.__aenter__.return_value.get.side_effect = Exception("Connection failed")
                
                with pytest.raises(TimeoutError, match="Agent failed to become ready"):
                    await runtime._wait_for_ready(timeout=1)  # Short timeout for testing