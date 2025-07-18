"""
AgentVault™ Agent Factory
Production-grade factory for creating and managing different agent frameworks
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Type, Protocol, List
from abc import ABC, abstractmethod
import json
import importlib
import sys
from datetime import datetime
import subprocess
import os

from ..database.models import Agent, AgentState
from ..storage.anf_advanced_manager import ANFAdvancedManager
from ..cache.distributed_cache import DistributedCache
from ..vectordb.vector_store import VectorStore

logger = logging.getLogger(__name__)


class AgentRuntime(Protocol):
    """Protocol for agent runtime implementations"""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the agent runtime"""
        ...
    
    async def start(self) -> None:
        """Start the agent"""
        ...
    
    async def stop(self, force: bool = False) -> None:
        """Stop the agent"""
        ...
    
    async def execute_command(self, command: str, parameters: Dict[str, Any]) -> Any:
        """Execute a command on the agent"""
        ...
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        ...
    
    async def update_config(self, updates: Dict[str, Any]) -> None:
        """Update agent configuration"""
        ...


class BaseAgentRuntime(ABC):
    """Base class for agent runtime implementations"""
    
    def __init__(self, agent: Agent, factory: 'AgentFactory'):
        self.agent = agent
        self.factory = factory
        self.process: Optional[asyncio.subprocess.Process] = None
        self.container_id: Optional[str] = None
        self.api_client: Optional[Any] = None
        self._initialized = False
        self._running = False
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the agent runtime"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the agent"""
        pass
    
    @abstractmethod
    async def stop(self, force: bool = False) -> None:
        """Stop the agent"""
        pass
    
    async def execute_command(self, command: str, parameters: Dict[str, Any]) -> Any:
        """Execute a command on the agent"""
        if not self._running:
            raise RuntimeError("Agent is not running")
        
        # Default implementation sends command via API
        if self.api_client:
            return await self.api_client.execute_command(command, parameters)
        else:
            raise NotImplementedError("Command execution not implemented")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'initialized': self._initialized,
            'running': self._running,
            'agent_id': str(self.agent.id),
            'agent_type': self.agent.agent_type,
            'container_id': self.container_id,
            'process_pid': self.process.pid if self.process else None
        }
    
    async def update_config(self, updates: Dict[str, Any]) -> None:
        """Update agent configuration"""
        # Merge updates
        self.agent.configuration.update(updates)
        
        # Apply updates if running
        if self._running and self.api_client:
            await self.api_client.update_config(updates)
    
    def _get_container_name(self) -> str:
        """Get container name for agent"""
        return f"agentvault-{self.agent.agent_type}-{self.agent.id}"
    
    def _get_api_endpoint(self) -> str:
        """Get API endpoint for agent"""
        if self.agent.internal_endpoint:
            return self.agent.internal_endpoint
        
        # Default endpoint based on container
        if self.container_id:
            return f"http://{self._get_container_name()}:8080"
        
        return f"http://localhost:{self._get_port()}"
    
    def _get_port(self) -> int:
        """Get port for agent API"""
        # Use a deterministic port based on agent ID
        # Port range: 30000-40000
        agent_hash = int(str(self.agent.id).replace('-', '')[:8], 16)
        return 30000 + (agent_hash % 10000)


class LangChainRuntime(BaseAgentRuntime):
    """Runtime implementation for LangChain agents"""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize LangChain agent runtime"""
        logger.info(f"Initializing LangChain runtime for agent {self.agent.id}")
        
        # Prepare environment
        env_vars = {
            'AGENT_ID': str(self.agent.id),
            'AGENT_NAME': self.agent.name,
            'AGENT_TYPE': 'langchain',
            'OPENAI_API_KEY': config.get('openai_api_key', ''),
            'LANGCHAIN_API_KEY': config.get('langchain_api_key', ''),
            'LANGCHAIN_PROJECT': config.get('langchain_project', f'agent-{self.agent.id}'),
            'VECTOR_STORE_URL': self.factory.vector_store.get_connection_url(),
            'CACHE_URL': self.factory.cache.get_connection_url(),
            'LOG_LEVEL': config.get('log_level', 'INFO')
        }
        
        # Create agent configuration file
        agent_config = {
            'agent': {
                'id': str(self.agent.id),
                'name': self.agent.name,
                'type': 'langchain',
                'capabilities': self.agent.capabilities
            },
            'langchain': {
                'model': config.get('model', 'gpt-4'),
                'temperature': config.get('temperature', 0.7),
                'max_tokens': config.get('max_tokens', 2000),
                'tools': config.get('tools', []),
                'memory': config.get('memory', {
                    'type': 'conversation_buffer_window',
                    'k': 10
                }),
                'chain_type': config.get('chain_type', 'conversational')
            },
            'storage': {
                'mount_points': {
                    '/data': f"/mnt/agentvault/{self.agent.id}/data",
                    '/cache': f"/mnt/agentvault/{self.agent.id}/cache"
                }
            }
        }
        
        # Save configuration
        config_path = f"/tmp/agent-{self.agent.id}-config.json"
        with open(config_path, 'w') as f:
            json.dump(agent_config, f, indent=2)
        
        # Build Docker image if needed
        image_name = f"agentvault/langchain:{config.get('version', 'latest')}"
        
        # Create container
        container_name = self._get_container_name()
        
        cmd = [
            'docker', 'run', '-d',
            '--name', container_name,
            '--restart', 'unless-stopped',
            '-e', f"CONFIG_PATH=/config/agent-config.json",
            '-v', f"{config_path}:/config/agent-config.json:ro"
        ]
        
        # Add environment variables
        for key, value in env_vars.items():
            cmd.extend(['-e', f"{key}={value}"])
        
        # Add resource limits
        cmd.extend([
            '--cpus', str(self.agent.cpu_cores),
            '--memory', f"{int(self.agent.memory_gb)}g"
        ])
        
        if self.agent.gpu_enabled:
            cmd.extend(['--gpus', f'device=0'])
        
        # Add volume mounts
        for mount_point, host_path in agent_config['storage']['mount_points'].items():
            cmd.extend(['-v', f"{host_path}:{mount_point}"])
        
        # Add networking
        cmd.extend([
            '--network', 'agentvault-network',
            '-p', f"0:8080"  # Random host port
        ])
        
        # Add image
        cmd.append(image_name)
        
        # Create container
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")
        
        self.container_id = result.stdout.strip()
        
        # Get container info
        inspect_result = subprocess.run(
            ['docker', 'inspect', self.container_id],
            capture_output=True,
            text=True
        )
        
        if inspect_result.returncode == 0:
            container_info = json.loads(inspect_result.stdout)[0]
            
            # Get exposed port
            port_mapping = container_info['NetworkSettings']['Ports'].get('8080/tcp', [])
            if port_mapping:
                host_port = port_mapping[0]['HostPort']
                self.agent.internal_endpoint = f"http://localhost:{host_port}"
        
        self._initialized = True
        logger.info(f"LangChain runtime initialized for agent {self.agent.id}")
    
    async def start(self) -> None:
        """Start LangChain agent"""
        if not self._initialized:
            raise RuntimeError("Agent runtime not initialized")
        
        logger.info(f"Starting LangChain agent {self.agent.id}")
        
        # Start container
        result = subprocess.run(
            ['docker', 'start', self.container_id],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start container: {result.stderr}")
        
        # Wait for agent to be ready
        await self._wait_for_ready()
        
        # Initialize API client
        from langchain.agents import AgentExecutor
        # This would be a custom client to communicate with the containerized agent
        # self.api_client = LangChainAPIClient(self._get_api_endpoint())
        
        self._running = True
        logger.info(f"LangChain agent {self.agent.id} started successfully")
    
    async def stop(self, force: bool = False) -> None:
        """Stop LangChain agent"""
        if not self._running:
            return
        
        logger.info(f"Stopping LangChain agent {self.agent.id}")
        
        # Stop container
        cmd = ['docker', 'stop']
        if force:
            cmd.extend(['-t', '0'])  # No grace period
        else:
            cmd.extend(['-t', '30'])  # 30 second grace period
        
        cmd.append(self.container_id)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to stop container: {result.stderr}")
            
            if force:
                # Force kill
                subprocess.run(['docker', 'kill', self.container_id])
        
        self._running = False
        logger.info(f"LangChain agent {self.agent.id} stopped")
    
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for agent to be ready"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            # Check if container is running
            result = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Running}}', self.container_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip() == 'true':
                # Check health endpoint
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self._get_api_endpoint()}/health") as response:
                            if response.status == 200:
                                return
                except:
                    pass
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Agent failed to become ready")


class AutoGenRuntime(BaseAgentRuntime):
    """Runtime implementation for AutoGen agents"""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize AutoGen agent runtime"""
        logger.info(f"Initializing AutoGen runtime for agent {self.agent.id}")
        
        # AutoGen specific initialization
        env_vars = {
            'AGENT_ID': str(self.agent.id),
            'AGENT_NAME': self.agent.name,
            'OPENAI_API_KEY': config.get('openai_api_key', ''),
            'AUTOGEN_USE_DOCKER': config.get('use_docker', 'True'),
            'AUTOGEN_CACHE_SEED': str(self.agent.id),
        }
        
        # Create AutoGen configuration
        autogen_config = {
            'agent': {
                'id': str(self.agent.id),
                'name': self.agent.name,
                'type': 'autogen'
            },
            'autogen': {
                'config_list': config.get('config_list', [
                    {
                        'model': config.get('model', 'gpt-4'),
                        'api_key': config.get('openai_api_key', '')
                    }
                ]),
                'system_message': config.get('system_message', ''),
                'human_input_mode': config.get('human_input_mode', 'NEVER'),
                'max_consecutive_auto_reply': config.get('max_consecutive_auto_reply', 10),
                'code_execution_config': {
                    'work_dir': '/workspace',
                    'use_docker': True,
                    'timeout': config.get('code_execution_timeout', 60)
                }
            }
        }
        
        # Similar container setup as LangChain
        # ... (implementation similar to LangChain but with AutoGen specifics)
        
        self._initialized = True
    
    async def start(self) -> None:
        """Start AutoGen agent"""
        if not self._initialized:
            raise RuntimeError("Agent must be initialized before starting")
        
        if self._running:
            logger.warning(f"AutoGen agent {self.agent.id} is already running")
            return
        
        logger.info(f"Starting AutoGen agent {self.agent.id}")
        
        # Start the container
        subprocess.run(
            ['docker', 'start', self.container_id],
            check=True
        )
        
        # Wait for agent to be ready
        await self._wait_for_ready()
        
        # Create API client for AutoGen
        try:
            from autogen import AssistantAgent
            self.api_client = AssistantAgent(
                name=self.agent.name,
                system_message=self.agent.configuration.get('system_message', ''),
                llm_config=self.agent.configuration.get('autogen', {}).get('config_list', [])
            )
        except ImportError:
            logger.warning("AutoGen not installed, using HTTP API")
            import aiohttp
            self.api_client = aiohttp.ClientSession()
        
        self._running = True
        self.agent.status = 'running'
        await self.factory.storage_manager.update_agent(self.agent)
        
        logger.info(f"AutoGen agent {self.agent.id} started successfully")
    
    async def stop(self, force: bool = False) -> None:
        """Stop AutoGen agent"""
        if not self._running:
            logger.warning(f"AutoGen agent {self.agent.id} is not running")
            return
        
        logger.info(f"Stopping AutoGen agent {self.agent.id}")
        
        # Clean up API client
        if hasattr(self.api_client, 'close'):
            await self.api_client.close()
        
        # Stop the container
        subprocess.run(
            ['docker', 'stop', self.container_id],
            check=True
        )
        
        if force:
            # Remove the container
            subprocess.run(
                ['docker', 'rm', '-f', self.container_id],
                check=True
            )
        
        self._running = False
        self.agent.status = 'stopped'
        await self.factory.storage_manager.update_agent(self.agent)
        
        logger.info(f"AutoGen agent {self.agent.id} stopped")
    
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for agent to be ready"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            # Check if container is running
            result = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Running}}', self.container_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip() == 'true':
                # Check health endpoint
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self._get_api_endpoint()}/health") as response:
                            if response.status == 200:
                                return
                except:
                    pass
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Agent failed to become ready")


class CrewAIRuntime(BaseAgentRuntime):
    """Runtime implementation for CrewAI agents"""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize CrewAI agent runtime"""
        logger.info(f"Initializing CrewAI runtime for agent {self.agent.id}")
        
        # CrewAI specific initialization
        crewai_config = {
            'agent': {
                'id': str(self.agent.id),
                'name': self.agent.name,
                'type': 'crewai'
            },
            'crewai': {
                'role': config.get('role', 'Assistant'),
                'goal': config.get('goal', ''),
                'backstory': config.get('backstory', ''),
                'verbose': config.get('verbose', True),
                'allow_delegation': config.get('allow_delegation', False),
                'tools': config.get('tools', []),
                'llm': config.get('llm', {
                    'model': 'gpt-4',
                    'temperature': 0.7
                })
            }
        }
        
        # Create container for CrewAI agent
        docker_image = config.get('docker_image', 'agentvault/crewai:latest')
        container_name = self._get_container_name()
        
        # Prepare environment variables
        env_vars = {
            'AGENT_ID': str(self.agent.id),
            'AGENT_NAME': self.agent.name,
            'AGENT_TYPE': 'crewai',
            'OPENAI_API_KEY': config.get('openai_api_key', ''),
            'CREWAI_CONFIG': json.dumps(crewai_config)
        }
        
        # Create and configure container
        docker_cmd = [
            'docker', 'create',
            '--name', container_name,
            '--network', 'agentvault-network',
            '-p', f'{self._get_port()}:8080',
            '--label', f'agent-id={self.agent.id}',
            '--label', 'managed-by=agentvault',
            '--restart', 'unless-stopped'
        ]
        
        # Add environment variables
        for key, value in env_vars.items():
            docker_cmd.extend(['-e', f'{key}={value}'])
        
        # Add volume mounts
        docker_cmd.extend([
            '-v', f'agentvault-{self.agent.id}:/data',
            '-v', '/var/run/docker.sock:/var/run/docker.sock:ro'
        ])
        
        # Add the image
        docker_cmd.append(docker_image)
        
        # Create the container
        result = subprocess.run(docker_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr}")
        
        self.container_id = result.stdout.strip()
        
        # Update agent with container info
        self.agent.internal_endpoint = f"http://{container_name}:8080"
        self.agent.configuration['container_id'] = self.container_id
        await self.factory.storage_manager.update_agent(self.agent)
        
        self._initialized = True
    
    async def start(self) -> None:
        """Start CrewAI agent"""
        if not self._initialized:
            raise RuntimeError("Agent must be initialized before starting")
        
        if self._running:
            logger.warning(f"CrewAI agent {self.agent.id} is already running")
            return
        
        logger.info(f"Starting CrewAI agent {self.agent.id}")
        
        # Start the container
        subprocess.run(
            ['docker', 'start', self.container_id],
            check=True
        )
        
        # Wait for agent to be ready
        await self._wait_for_ready()
        
        # Create API client for CrewAI
        try:
            from crewai import Agent
            self.api_client = Agent(
                role=self.agent.configuration.get('role', 'Assistant'),
                goal=self.agent.configuration.get('goal', ''),
                backstory=self.agent.configuration.get('backstory', ''),
                verbose=self.agent.configuration.get('verbose', True),
                allow_delegation=self.agent.configuration.get('allow_delegation', False)
            )
        except ImportError:
            logger.warning("CrewAI not installed, using HTTP API")
            import aiohttp
            self.api_client = aiohttp.ClientSession()
        
        self._running = True
        self.agent.status = 'running'
        await self.factory.storage_manager.update_agent(self.agent)
        
        logger.info(f"CrewAI agent {self.agent.id} started successfully")
    
    async def stop(self, force: bool = False) -> None:
        """Stop CrewAI agent"""
        if not self._running:
            logger.warning(f"CrewAI agent {self.agent.id} is not running")
            return
        
        logger.info(f"Stopping CrewAI agent {self.agent.id}")
        
        # Clean up API client
        if hasattr(self.api_client, 'close'):
            await self.api_client.close()
        
        # Stop the container
        subprocess.run(
            ['docker', 'stop', self.container_id],
            check=True
        )
        
        if force:
            # Remove the container
            subprocess.run(
                ['docker', 'rm', '-f', self.container_id],
                check=True
            )
        
        self._running = False
        self.agent.status = 'stopped'
        await self.factory.storage_manager.update_agent(self.agent)
        
        logger.info(f"CrewAI agent {self.agent.id} stopped")
    
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for agent to be ready"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            # Check if container is running
            result = subprocess.run(
                ['docker', 'inspect', '-f', '{{.State.Running}}', self.container_id],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip() == 'true':
                # Check health endpoint
                try:
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(f"{self._get_api_endpoint()}/health") as response:
                            if response.status == 200:
                                return
                except:
                    pass
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Agent failed to become ready")


class CustomRuntime(BaseAgentRuntime):
    """Runtime implementation for custom agents"""
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize custom agent runtime"""
        logger.info(f"Initializing custom runtime for agent {self.agent.id}")
        
        # Get custom implementation details
        runtime_type = config.get('runtime_type', 'python')
        
        if runtime_type == 'python':
            await self._initialize_python_runtime(config)
        elif runtime_type == 'docker':
            await self._initialize_docker_runtime(config)
        elif runtime_type == 'kubernetes':
            await self._initialize_kubernetes_runtime(config)
        else:
            raise ValueError(f"Unsupported custom runtime type: {runtime_type}")
        
        self._initialized = True
    
    async def _initialize_python_runtime(self, config: Dict[str, Any]):
        """Initialize Python-based custom runtime"""
        # Create virtual environment
        venv_path = f"/opt/agentvault/agents/{self.agent.id}/venv"
        
        # Install dependencies
        requirements = config.get('requirements', [])
        
        # Create startup script
        startup_script = config.get('startup_script', '')
        
        # Implementation details...
    
    async def _initialize_docker_runtime(self, config: Dict[str, Any]):
        """Initialize Docker-based custom runtime"""
        # Use provided Docker image
        image = config.get('docker_image')
        if not image:
            raise ValueError("docker_image is required for Docker runtime")
        
        # Create container with custom configuration
        # Implementation...
    
    async def _initialize_kubernetes_runtime(self, config: Dict[str, Any]):
        """Initialize Kubernetes-based custom runtime"""
        # Create Kubernetes deployment
        from kubernetes import client, config as k8s_config
        k8s_config.load_incluster_config()
        apps_v1 = client.AppsV1Api()
        
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(
                name=f"agent-{self.agent.id}",
                labels={"agent-id": str(self.agent.id)}
            ),
            spec=client.V1DeploymentSpec(
                replicas=0,  # Start with 0, will scale up on start
                selector=client.V1LabelSelector(
                    match_labels={"agent-id": str(self.agent.id)}
                ),
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"agent-id": str(self.agent.id)}
                    ),
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="agent",
                                image=config.get('kubernetes_image', 'agentvault/custom-agent:latest'),
                                env=[
                                    client.V1EnvVar(name="AGENT_ID", value=str(self.agent.id)),
                                    client.V1EnvVar(name="AGENT_NAME", value=self.agent.name),
                                ],
                                ports=[client.V1ContainerPort(container_port=8080)]
                            )
                        ]
                    )
                )
            )
        )
        
        namespace = config.get('namespace', 'default')
        apps_v1.create_namespaced_deployment(
            namespace=namespace,
            body=deployment
        )
    
    async def start(self) -> None:
        """Start custom agent"""
        if not self._initialized:
            raise RuntimeError("Agent must be initialized before starting")
        
        if self._running:
            logger.warning(f"Custom agent {self.agent.id} is already running")
            return
        
        logger.info(f"Starting custom agent {self.agent.id}")
        
        runtime_type = self.agent.configuration.get('runtime_type', 'python')
        
        if runtime_type == 'python':
            # Start Python process
            script_path = self.agent.configuration.get('script_path')
            if not script_path:
                raise ValueError("script_path is required for Python runtime")
            
            self.process = await asyncio.create_subprocess_exec(
                'python', script_path,
                env={
                    **os.environ,
                    'AGENT_ID': str(self.agent.id),
                    'AGENT_NAME': self.agent.name,
                    'AGENTVAULT_API': self._get_api_endpoint()
                },
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
        elif runtime_type == 'docker':
            # Start Docker container
            subprocess.run(
                ['docker', 'start', self.container_id],
                check=True
            )
            await self._wait_for_ready()
            
        elif runtime_type == 'kubernetes':
            # Update deployment to start pods
            from kubernetes import client, config as k8s_config
            k8s_config.load_incluster_config()
            apps_v1 = client.AppsV1Api()
            
            deployment_name = f"agent-{self.agent.id}"
            namespace = self.agent.configuration.get('namespace', 'default')
            
            # Scale deployment to 1
            apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body={'spec': {'replicas': 1}}
            )
        
        self._running = True
        self.agent.status = 'running'
        await self.factory.storage_manager.update_agent(self.agent)
        
        logger.info(f"Custom agent {self.agent.id} started successfully")
    
    async def stop(self, force: bool = False) -> None:
        """Stop custom agent"""
        if not self._running:
            logger.warning(f"Custom agent {self.agent.id} is not running")
            return
        
        logger.info(f"Stopping custom agent {self.agent.id}")
        
        runtime_type = self.agent.configuration.get('runtime_type', 'python')
        
        if runtime_type == 'python':
            # Stop Python process
            if self.process:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
                    
        elif runtime_type == 'docker':
            # Stop Docker container
            subprocess.run(
                ['docker', 'stop', self.container_id],
                check=True
            )
            if force:
                subprocess.run(
                    ['docker', 'rm', '-f', self.container_id],
                    check=True
                )
                
        elif runtime_type == 'kubernetes':
            # Scale deployment to 0
            from kubernetes import client, config as k8s_config
            k8s_config.load_incluster_config()
            apps_v1 = client.AppsV1Api()
            
            deployment_name = f"agent-{self.agent.id}"
            namespace = self.agent.configuration.get('namespace', 'default')
            
            # Scale to 0
            apps_v1.patch_namespaced_deployment_scale(
                name=deployment_name,
                namespace=namespace,
                body={'spec': {'replicas': 0}}
            )
            
            if force:
                # Delete deployment
                apps_v1.delete_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )
        
        self._running = False
        self.agent.status = 'stopped'
        await self.factory.storage_manager.update_agent(self.agent)
        
        logger.info(f"Custom agent {self.agent.id} stopped")
    
    async def _wait_for_ready(self, timeout: int = 60):
        """Wait for agent to be ready"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            runtime_type = self.agent.configuration.get('runtime_type', 'python')
            
            if runtime_type == 'docker':
                # Check if container is running
                result = subprocess.run(
                    ['docker', 'inspect', '-f', '{{.State.Running}}', self.container_id],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout.strip() == 'true':
                    # Check health endpoint
                    try:
                        import aiohttp
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{self._get_api_endpoint()}/health") as response:
                                if response.status == 200:
                                    return
                    except:
                        pass
            elif runtime_type == 'python':
                # For Python processes, check if process is running
                if self.process and self.process.returncode is None:
                    return
            elif runtime_type == 'kubernetes':
                # Check Kubernetes deployment status
                from kubernetes import client, config as k8s_config
                k8s_config.load_incluster_config()
                apps_v1 = client.AppsV1Api()
                
                deployment_name = f"agent-{self.agent.id}"
                namespace = self.agent.configuration.get('namespace', 'default')
                
                deployment = apps_v1.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )
                
                if deployment.status.ready_replicas and deployment.status.ready_replicas > 0:
                    return
            
            await asyncio.sleep(2)
        
        raise TimeoutError("Agent failed to become ready")


class AgentFactory:
    """
    Factory for creating and managing agent runtimes
    """
    
    def __init__(
        self,
        anf_manager: ANFAdvancedManager,
        cache: DistributedCache,
        vector_store: VectorStore
    ):
        self.anf_manager = anf_manager
        self.cache = cache
        self.vector_store = vector_store
        
        # Runtime registry
        self._runtime_classes: Dict[str, Type[BaseAgentRuntime]] = {
            'langchain': LangChainRuntime,
            'autogen': AutoGenRuntime,
            'crewai': CrewAIRuntime,
            'custom': CustomRuntime
        }
        
        # Active runtimes
        self._active_runtimes: Dict[str, BaseAgentRuntime] = {}
    
    def register_runtime(self, agent_type: str, runtime_class: Type[BaseAgentRuntime]):
        """Register a custom runtime implementation"""
        self._runtime_classes[agent_type] = runtime_class
    
    async def create_runtime(self, agent: Agent) -> BaseAgentRuntime:
        """Create runtime for an agent"""
        runtime_class = self._runtime_classes.get(agent.agent_type)
        if not runtime_class:
            raise ValueError(f"Unsupported agent type: {agent.agent_type}")
        
        runtime = runtime_class(agent, self)
        self._active_runtimes[str(agent.id)] = runtime
        
        return runtime
    
    async def initialize_agent_runtime(self, agent: Agent) -> None:
        """Initialize runtime for an agent"""
        runtime = await self.create_runtime(agent)
        await runtime.initialize(agent.configuration)
    
    async def start_agent_runtime(self, agent: Agent) -> None:
        """Start agent runtime"""
        runtime = self._active_runtimes.get(str(agent.id))
        if not runtime:
            runtime = await self.create_runtime(agent)
            await runtime.initialize(agent.configuration)
        
        await runtime.start()
    
    async def stop_agent_runtime(self, agent: Agent, force: bool = False) -> None:
        """Stop agent runtime"""
        runtime = self._active_runtimes.get(str(agent.id))
        if runtime:
            await runtime.stop(force=force)
            
            if force:
                # Remove from active runtimes
                del self._active_runtimes[str(agent.id)]
    
    async def execute_agent_command(
        self,
        agent: Agent,
        command: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute command on agent runtime"""
        runtime = self._active_runtimes.get(str(agent.id))
        if not runtime:
            raise RuntimeError(f"No active runtime for agent {agent.id}")
        
        return await runtime.execute_command(command, parameters)
    
    async def update_agent_runtime(
        self,
        agent: Agent,
        updates: Dict[str, Any]
    ) -> None:
        """Update agent runtime configuration"""
        runtime = self._active_runtimes.get(str(agent.id))
        if runtime:
            await runtime.update_config(updates)
    
    def get_runtime_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get runtime status for an agent"""
        runtime = self._active_runtimes.get(agent_id)
        if runtime:
            return asyncio.run(runtime.get_status())
        return None
    
    async def cleanup_inactive_runtimes(self) -> int:
        """Clean up inactive runtimes"""
        cleaned = 0
        
        for agent_id, runtime in list(self._active_runtimes.items()):
            status = await runtime.get_status()
            if not status.get('running'):
                # Remove inactive runtime
                del self._active_runtimes[agent_id]
                cleaned += 1
        
        return cleaned


class AgentBuilder:
    """
    Builder pattern for constructing agents with validation
    """
    
    def __init__(self):
        self._config = {
            'name': None,
            'agent_type': None,
            'tenant_id': None,
            'owner_id': None,
            'configuration': {},
            'capabilities': [],
            'resources': {},
            'tags': []
        }
    
    def with_name(self, name: str) -> 'AgentBuilder':
        """Set agent name"""
        self._config['name'] = name
        return self
    
    def with_type(self, agent_type: str) -> 'AgentBuilder':
        """Set agent type"""
        if agent_type not in ['langchain', 'autogen', 'crewai', 'custom']:
            raise ValueError(f"Invalid agent type: {agent_type}")
        self._config['agent_type'] = agent_type
        return self
    
    def with_tenant(self, tenant_id: str) -> 'AgentBuilder':
        """Set tenant ID"""
        self._config['tenant_id'] = tenant_id
        return self
    
    def with_owner(self, owner_id: str) -> 'AgentBuilder':
        """Set owner ID"""
        self._config['owner_id'] = owner_id
        return self
    
    def with_configuration(self, config: Dict[str, Any]) -> 'AgentBuilder':
        """Set agent configuration"""
        self._config['configuration'].update(config)
        return self
    
    def with_capability(self, capability: str) -> 'AgentBuilder':
        """Add a capability"""
        if capability not in self._config['capabilities']:
            self._config['capabilities'].append(capability)
        return self
    
    def with_resources(
        self,
        cpu_cores: float = 1.0,
        memory_gb: float = 4.0,
        storage_gb: float = 10.0,
        gpu_enabled: bool = False,
        gpu_memory_gb: float = 0.0
    ) -> 'AgentBuilder':
        """Set resource requirements"""
        self._config['resources'] = {
            'cpu_cores': cpu_cores,
            'memory_gb': memory_gb,
            'storage_gb': storage_gb,
            'gpu_enabled': gpu_enabled,
            'gpu_memory_gb': gpu_memory_gb
        }
        return self
    
    def with_tag(self, tag: str) -> 'AgentBuilder':
        """Add a tag"""
        if tag not in self._config['tags']:
            self._config['tags'].append(tag)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and validate agent configuration"""
        # Validate required fields
        required = ['name', 'agent_type', 'tenant_id', 'owner_id']
        for field in required:
            if not self._config.get(field):
                raise ValueError(f"Missing required field: {field}")
        
        # Set defaults
        if not self._config['resources']:
            # Default resources based on type
            defaults = {
                'langchain': {'cpu_cores': 2.0, 'memory_gb': 8.0, 'storage_gb': 20.0},
                'autogen': {'cpu_cores': 4.0, 'memory_gb': 16.0, 'storage_gb': 50.0, 'gpu_enabled': True},
                'crewai': {'cpu_cores': 2.0, 'memory_gb': 8.0, 'storage_gb': 30.0},
                'custom': {'cpu_cores': 1.0, 'memory_gb': 4.0, 'storage_gb': 10.0}
            }
            self._config['resources'] = defaults.get(
                self._config['agent_type'],
                defaults['custom']
            )
        
        return self._config