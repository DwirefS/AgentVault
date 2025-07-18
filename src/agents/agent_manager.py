"""
AgentVault™ Agent Manager
Comprehensive agent lifecycle management system
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from uuid import UUID
import json

from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import Agent, AgentState, AgentProfile, StorageVolume, StorageTier
from ..database.repositories import AgentRepository, StorageRepository, MetricsRepository
from ..storage.anf_advanced_manager import ANFAdvancedManager
from ..cache.distributed_cache import DistributedCache
from ..vectordb.vector_store import VectorStore
from ..ml.advanced_agent_dna import AdvancedAgentDNA
from ..security.advanced_encryption import AdvancedEncryptionManager
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem
from .state_machine import AgentStateMachine, StateValidator
from .agent_factory import AgentFactory
from .agent_health import AgentHealthMonitor
from .agent_metrics import AgentMetricsCollector

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Central manager for all agent operations
    Handles creation, lifecycle, resources, and monitoring
    """
    
    def __init__(
        self,
        db_session: Session,
        anf_manager: ANFAdvancedManager,
        cache: DistributedCache,
        vector_store: VectorStore,
        ml_engine: AdvancedAgentDNA,
        encryption_manager: AdvancedEncryptionManager,
        monitoring: AdvancedMonitoringSystem,
        config: Dict[str, Any]
    ):
        self.db_session = db_session
        self.anf_manager = anf_manager
        self.cache = cache
        self.vector_store = vector_store
        self.ml_engine = ml_engine
        self.encryption_manager = encryption_manager
        self.monitoring = monitoring
        self.config = config
        
        # Initialize components
        self.agent_repo = AgentRepository(db_session)
        self.storage_repo = StorageRepository(db_session)
        self.metrics_repo = MetricsRepository(db_session)
        
        self.state_machine = AgentStateMachine(monitoring)
        self.agent_factory = AgentFactory(self)
        self.health_monitor = AgentHealthMonitor(self)
        self.metrics_collector = AgentMetricsCollector(self)
        
        # Agent registry cache
        self._agent_cache: Dict[UUID, Agent] = {}
        self._cache_ttl = timedelta(minutes=5)
        self._cache_timestamps: Dict[UUID, datetime] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize agent manager and start background tasks"""
        logger.info("Initializing Agent Manager")
        
        # Start background tasks
        self._background_tasks.extend([
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._cache_cleanup_loop()),
            asyncio.create_task(self._resource_optimization_loop())
        ])
        
        logger.info("Agent Manager initialized successfully")
    
    async def shutdown(self):
        """Gracefully shutdown agent manager"""
        logger.info("Shutting down Agent Manager")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for background tasks
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clean up resources
        self._agent_cache.clear()
        
        logger.info("Agent Manager shutdown complete")
    
    async def create_agent(
        self,
        name: str,
        agent_type: str,
        tenant_id: UUID,
        owner_id: UUID,
        configuration: Dict[str, Any],
        capabilities: List[str],
        resources: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Agent:
        """
        Create a new agent with full lifecycle setup
        """
        logger.info(f"Creating agent: {name} of type {agent_type}")
        
        # Validate quota
        allowed, message = await self._check_resource_quota(
            tenant_id,
            resources or self._get_default_resources(agent_type)
        )
        if not allowed:
            raise ValueError(f"Resource quota exceeded: {message}")
        
        # Create agent in database
        agent = self.agent_repo.create_agent(
            name=name,
            agent_type=agent_type,
            tenant_id=tenant_id,
            owner_id=owner_id,
            configuration=configuration,
            capabilities=capabilities,
            tags=tags or [],
            cpu_cores=resources.get('cpu_cores', 1.0) if resources else 1.0,
            memory_gb=resources.get('memory_gb', 4.0) if resources else 4.0,
            gpu_enabled=resources.get('gpu_enabled', False) if resources else False,
            gpu_memory_gb=resources.get('gpu_memory_gb', 0.0) if resources else 0.0,
            storage_gb=resources.get('storage_gb', 10.0) if resources else 10.0
        )
        
        try:
            # Create storage volumes
            await self._create_agent_storage(agent)
            
            # Initialize vector store for agent
            await self._initialize_vector_store(agent)
            
            # Generate initial DNA profile
            await self._generate_initial_dna(agent)
            
            # Set up monitoring
            await self._setup_agent_monitoring(agent)
            
            # Initialize agent runtime (framework-specific)
            await self.agent_factory.initialize_agent_runtime(agent)
            
            # Transition to ready state
            await self.state_machine.transition(
                agent,
                AgentState.READY,
                self.agent_repo,
                metadata={"initialization_complete": True}
            )
            
            # Cache agent
            self._cache_agent(agent)
            
            # Record metrics
            self.monitoring.record_custom_metric(
                "agent_created",
                1,
                labels={
                    "agent_type": agent_type,
                    "tenant_id": str(tenant_id)
                }
            )
            
            logger.info(f"Agent {agent.id} created successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create agent: {str(e)}")
            # Cleanup on failure
            await self._cleanup_failed_agent(agent)
            raise
    
    async def get_agent(self, agent_id: UUID, use_cache: bool = True) -> Optional[Agent]:
        """Get agent by ID with caching"""
        # Check cache first
        if use_cache and agent_id in self._agent_cache:
            cache_time = self._cache_timestamps.get(agent_id)
            if cache_time and datetime.utcnow() - cache_time < self._cache_ttl:
                return self._agent_cache[agent_id]
        
        # Load from database
        agent = self.agent_repo.get(agent_id)
        if agent:
            self._cache_agent(agent)
        
        return agent
    
    async def list_agents(
        self,
        tenant_id: UUID,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Agent]:
        """List agents with filtering and pagination"""
        if filters:
            agents = self.agent_repo.search_agents(
                tenant_id=tenant_id,
                query=filters.get('query', ''),
                filters=filters,
                limit=limit
            )
        else:
            agents = self.agent_repo.get_active_agents(tenant_id, limit=limit)
        
        return agents
    
    async def start_agent(self, agent_id: UUID) -> bool:
        """Start an agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Validate transition
        if not self.state_machine.can_transition(agent, AgentState.RUNNING):
            raise ValueError(f"Cannot start agent in state {agent.state}")
        
        # Allocate resources
        await self._allocate_agent_resources(agent)
        
        # Start agent runtime
        await self.agent_factory.start_agent_runtime(agent)
        
        # Transition to running
        success = await self.state_machine.transition(
            agent,
            AgentState.RUNNING,
            self.agent_repo,
            metadata={"started_at": datetime.utcnow().isoformat()}
        )
        
        if success:
            # Start real-time monitoring
            await self.health_monitor.start_monitoring(agent)
            
            logger.info(f"Agent {agent_id} started successfully")
        
        return success
    
    async def stop_agent(self, agent_id: UUID, force: bool = False) -> bool:
        """Stop a running agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Determine target state
        target_state = AgentState.PAUSED
        if agent.state == AgentState.ERROR and not force:
            logger.warning(f"Agent {agent_id} is in error state")
            return False
        
        # Stop agent runtime
        await self.agent_factory.stop_agent_runtime(agent, force=force)
        
        # Transition to paused
        success = await self.state_machine.transition(
            agent,
            target_state,
            self.agent_repo,
            metadata={"stopped_at": datetime.utcnow().isoformat(), "forced": force}
        )
        
        if success:
            # Stop monitoring
            await self.health_monitor.stop_monitoring(agent)
            
            # Release resources
            await self._release_agent_resources(agent)
            
            logger.info(f"Agent {agent_id} stopped successfully")
        
        return success
    
    async def restart_agent(self, agent_id: UUID) -> bool:
        """Restart an agent"""
        # Stop first
        stop_success = await self.stop_agent(agent_id)
        if not stop_success:
            return False
        
        # Wait briefly
        await asyncio.sleep(2)
        
        # Start again
        return await self.start_agent(agent_id)
    
    async def update_agent(
        self,
        agent_id: UUID,
        updates: Dict[str, Any]
    ) -> Optional[Agent]:
        """Update agent configuration"""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Validate updates based on state
        if agent.state == AgentState.RUNNING:
            # Only allow certain updates while running
            allowed_keys = {'tags', 'display_name', 'description', 'capabilities'}
            if not set(updates.keys()).issubset(allowed_keys):
                raise ValueError("Cannot update agent configuration while running")
        
        # Apply updates
        updated_agent = self.agent_repo.update(agent_id, **updates)
        
        if updated_agent:
            # Invalidate cache
            self._invalidate_cache(agent_id)
            
            # Update runtime if needed
            if agent.state == AgentState.RUNNING:
                await self.agent_factory.update_agent_runtime(updated_agent, updates)
            
            # Record audit log
            self.monitoring.record_custom_metric(
                "agent_updated",
                1,
                labels={"agent_id": str(agent_id)}
            )
        
        return updated_agent
    
    async def delete_agent(self, agent_id: UUID, force: bool = False) -> bool:
        """Delete an agent (soft delete by default)"""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Check if agent can be deleted
        if agent.state in [AgentState.RUNNING, AgentState.MIGRATING] and not force:
            raise ValueError(f"Cannot delete agent in state {agent.state}")
        
        # Stop if running
        if agent.state == AgentState.RUNNING:
            await self.stop_agent(agent_id, force=True)
        
        # Transition to terminated
        await self.state_machine.transition(
            agent,
            AgentState.TERMINATED,
            self.agent_repo,
            metadata={"deletion_requested": True}
        )
        
        # Clean up resources
        await self._cleanup_agent_resources(agent)
        
        # Soft delete
        success = self.agent_repo.soft_delete(agent_id)
        
        if success:
            # Remove from cache
            self._invalidate_cache(agent_id)
            
            logger.info(f"Agent {agent_id} deleted successfully")
        
        return success
    
    async def migrate_agent(
        self,
        agent_id: UUID,
        target_node: str,
        target_region: Optional[str] = None
    ) -> bool:
        """Migrate agent to different node or region"""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Validate migration possibility
        if not self.state_machine.can_transition(agent, AgentState.MIGRATING):
            raise ValueError(f"Cannot migrate agent in state {agent.state}")
        
        # Start migration
        await self.state_machine.transition(
            agent,
            AgentState.MIGRATING,
            self.agent_repo,
            metadata={
                "migration_started": datetime.utcnow().isoformat(),
                "target_node": target_node,
                "target_region": target_region
            }
        )
        
        try:
            # Perform migration steps
            # 1. Snapshot current state
            snapshot = await self._create_agent_snapshot(agent)
            
            # 2. Prepare target resources
            await self._prepare_migration_target(agent, target_node, target_region)
            
            # 3. Sync data
            await self._sync_agent_data(agent, target_node)
            
            # 4. Switch over
            await self._perform_migration_switchover(agent, target_node)
            
            # 5. Verify migration
            verified = await self._verify_migration(agent, target_node)
            
            if verified:
                # Complete migration
                await self.state_machine.transition(
                    agent,
                    AgentState.RUNNING,
                    self.agent_repo,
                    metadata={
                        "migration_completed": datetime.utcnow().isoformat(),
                        "migration_duration_seconds": 0  # Calculate actual
                    }
                )
                
                logger.info(f"Agent {agent_id} migrated successfully")
                return True
            else:
                raise Exception("Migration verification failed")
                
        except Exception as e:
            logger.error(f"Agent migration failed: {str(e)}")
            
            # Rollback migration
            await self._rollback_migration(agent)
            
            # Transition to error state
            await self.state_machine.transition(
                agent,
                AgentState.ERROR,
                self.agent_repo,
                metadata={"migration_error": str(e)},
                force=True
            )
            
            return False
    
    async def get_agent_metrics(
        self,
        agent_id: UUID,
        metric_names: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get agent metrics"""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Default time range
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        # Collect metrics
        metrics = {}
        
        # Performance metrics
        if not metric_names or 'performance' in metric_names:
            metrics['performance'] = {
                'total_requests': agent.total_requests,
                'total_errors': agent.total_errors,
                'error_rate': (agent.total_errors / agent.total_requests * 100) if agent.total_requests > 0 else 0,
                'average_latency_ms': agent.average_latency_ms,
                'uptime_percentage': agent.uptime_percentage
            }
        
        # Resource metrics
        if not metric_names or 'resources' in metric_names:
            metrics['resources'] = await self.metrics_collector.get_resource_metrics(agent_id)
        
        # Time series metrics
        if not metric_names or 'timeseries' in metric_names:
            timeseries_metrics = {}
            
            for metric_name in ['cpu_usage', 'memory_usage', 'request_rate', 'error_rate']:
                timeseries_metrics[metric_name] = self.metrics_repo.get_metrics_timeseries(
                    metric_name=f"agent.{metric_name}",
                    start_time=start_time,
                    end_time=end_time,
                    agent_id=agent_id,
                    interval_seconds=300  # 5 minutes
                )
            
            metrics['timeseries'] = timeseries_metrics
        
        # Health metrics
        if not metric_names or 'health' in metric_names:
            metrics['health'] = await self.health_monitor.get_health_status(agent_id)
        
        return metrics
    
    async def get_agent_logs(
        self,
        agent_id: UUID,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get agent logs"""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Query logs from monitoring system
        logs = await self.monitoring.query_logs(
            resource_id=str(agent_id),
            resource_type='agent',
            start_time=start_time,
            end_time=end_time,
            level=level,
            limit=limit
        )
        
        return logs
    
    async def execute_agent_command(
        self,
        agent_id: UUID,
        command: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a command on an agent"""
        agent = await self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Validate agent state
        if agent.state != AgentState.RUNNING:
            raise ValueError(f"Agent must be running to execute commands")
        
        # Execute command through agent runtime
        result = await self.agent_factory.execute_agent_command(
            agent,
            command,
            parameters or {}
        )
        
        # Record command execution
        self.monitoring.record_custom_metric(
            "agent_command_executed",
            1,
            labels={
                "agent_id": str(agent_id),
                "command": command
            }
        )
        
        return result
    
    # Private helper methods
    
    def _cache_agent(self, agent: Agent):
        """Cache agent instance"""
        self._agent_cache[agent.id] = agent
        self._cache_timestamps[agent.id] = datetime.utcnow()
    
    def _invalidate_cache(self, agent_id: UUID):
        """Invalidate agent cache"""
        self._agent_cache.pop(agent_id, None)
        self._cache_timestamps.pop(agent_id, None)
    
    def _get_default_resources(self, agent_type: str) -> Dict[str, Any]:
        """Get default resource allocation for agent type"""
        defaults = {
            'langchain': {
                'cpu_cores': 2.0,
                'memory_gb': 8.0,
                'storage_gb': 20.0,
                'gpu_enabled': False
            },
            'autogen': {
                'cpu_cores': 4.0,
                'memory_gb': 16.0,
                'storage_gb': 50.0,
                'gpu_enabled': True,
                'gpu_memory_gb': 8.0
            },
            'crewai': {
                'cpu_cores': 2.0,
                'memory_gb': 8.0,
                'storage_gb': 30.0,
                'gpu_enabled': False
            },
            'custom': {
                'cpu_cores': 1.0,
                'memory_gb': 4.0,
                'storage_gb': 10.0,
                'gpu_enabled': False
            }
        }
        
        return defaults.get(agent_type, defaults['custom'])
    
    async def _check_resource_quota(
        self,
        tenant_id: UUID,
        resources: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Check if resources are within tenant quota"""
        # Get current usage
        current_usage = self.agent_repo.get_resource_usage(tenant_id)
        
        # Check each resource type against tenant limits
        tenant_limits = await self._get_tenant_limits(tenant_id)
        
        # Check CPU quota
        if current_usage.get('cpu_cores', 0) + cpu_cores > tenant_limits.get('max_cpu_cores', 100):
            return False, f"CPU quota exceeded. Current: {current_usage.get('cpu_cores', 0)}, Requested: {cpu_cores}, Limit: {tenant_limits.get('max_cpu_cores', 100)}"
        
        # Check memory quota
        if current_usage.get('memory_gb', 0) + memory_gb > tenant_limits.get('max_memory_gb', 1000):
            return False, f"Memory quota exceeded. Current: {current_usage.get('memory_gb', 0)}, Requested: {memory_gb}, Limit: {tenant_limits.get('max_memory_gb', 1000)}"
        
        # Check storage quota
        if current_usage.get('storage_gb', 0) + storage_gb > tenant_limits.get('max_storage_gb', 10000):
            return False, f"Storage quota exceeded. Current: {current_usage.get('storage_gb', 0)}, Requested: {storage_gb}, Limit: {tenant_limits.get('max_storage_gb', 10000)}"
        
        # Check agent count quota
        if current_usage.get('agent_count', 0) >= tenant_limits.get('max_agents', 50):
            return False, f"Agent count quota exceeded. Current: {current_usage.get('agent_count', 0)}, Limit: {tenant_limits.get('max_agents', 50)}"
        
        # Check GPU quota if requested
        if gpu_enabled and current_usage.get('gpu_count', 0) >= tenant_limits.get('max_gpus', 10):
            return False, f"GPU quota exceeded. Current: {current_usage.get('gpu_count', 0)}, Limit: {tenant_limits.get('max_gpus', 10)}"
        
        return True, None
    
    async def _create_agent_storage(self, agent: Agent):
        """Create storage volumes for agent"""
        # Create primary volume
        primary_volume = self.storage_repo.create_volume(
            agent_id=agent.id,
            volume_name=f"{agent.name}-primary",
            size_gb=agent.storage_gb,
            storage_tier=StorageTier.PREMIUM,
            mount_point="/data"
        )
        
        # Create cache volume
        cache_volume = self.storage_repo.create_volume(
            agent_id=agent.id,
            volume_name=f"{agent.name}-cache",
            size_gb=min(agent.storage_gb * 0.2, 100),  # 20% of primary or 100GB max
            storage_tier=StorageTier.PREMIUM,
            mount_point="/cache"
        )
        
        # Create volumes in ANF
        for volume in [primary_volume, cache_volume]:
            anf_volume = await self.anf_manager.create_volume(
                name=volume.volume_name,
                size_gb=int(volume.size_gb),
                service_level=volume.storage_tier
            )
            
            # Update volume with ANF details
            self.storage_repo.update(
                volume.id,
                anf_volume_id=anf_volume['id'],
                state='ready'
            )
    
    async def _initialize_vector_store(self, agent: Agent):
        """Initialize vector store for agent"""
        # Create dedicated index for agent
        index_name = f"agent_{agent.id}"
        await self.vector_store.create_index(
            index_name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric='cosine'
        )
        
        # Store index reference
        agent.configuration['vector_index'] = index_name
        self.agent_repo.update(agent.id, configuration=agent.configuration)
    
    async def _generate_initial_dna(self, agent: Agent):
        """Generate initial DNA profile for agent"""
        # Create initial profile
        dna_profile = self.ml_engine.create_dna_profile(
            agent_id=str(agent.id),
            agent_type=agent.agent_type,
            initial_config={
                'capabilities': agent.capabilities,
                'resources': {
                    'cpu_cores': agent.cpu_cores,
                    'memory_gb': agent.memory_gb,
                    'gpu_enabled': agent.gpu_enabled
                }
            }
        )
        
        # Update agent profile
        if agent.profile:
            agent.profile.dna_fingerprint = dna_profile
            agent.profile.last_dna_update = datetime.utcnow()
            self.db_session.commit()
    
    async def _setup_agent_monitoring(self, agent: Agent):
        """Set up monitoring for agent"""
        # Create monitoring namespace
        await self.monitoring.create_namespace(
            namespace=f"agent.{agent.id}",
            labels={
                'agent_id': str(agent.id),
                'agent_name': agent.name,
                'agent_type': agent.agent_type,
                'tenant_id': str(agent.tenant_id)
            }
        )
        
        # Set up default alerts
        alerts = [
            {
                'name': f'agent_{agent.id}_high_error_rate',
                'condition': 'error_rate > 5',
                'severity': 'warning',
                'duration': 300
            },
            {
                'name': f'agent_{agent.id}_down',
                'condition': 'up == 0',
                'severity': 'critical',
                'duration': 60
            },
            {
                'name': f'agent_{agent.id}_high_latency',
                'condition': 'latency_p95 > 1000',
                'severity': 'warning',
                'duration': 300
            }
        ]
        
        for alert in alerts:
            await self.monitoring.create_alert_rule(**alert)
    
    async def _allocate_agent_resources(self, agent: Agent):
        """Allocate compute resources for agent"""
        # Allocate through orchestrator
        # This would integrate with Kubernetes or other orchestrators
        logger.info(f"Allocating resources for agent {agent.id}")
        
        # Update resource allocation in monitoring
        self.monitoring.record_custom_metric(
            "agent_resources_allocated",
            1,
            labels={
                'agent_id': str(agent.id),
                'cpu_cores': str(agent.cpu_cores),
                'memory_gb': str(agent.memory_gb),
                'gpu_enabled': str(agent.gpu_enabled)
            }
        )
    
    async def _release_agent_resources(self, agent: Agent):
        """Release allocated resources"""
        logger.info(f"Releasing resources for agent {agent.id}")
        
        # Update metrics
        self.monitoring.record_custom_metric(
            "agent_resources_released",
            1,
            labels={'agent_id': str(agent.id)}
        )
    
    async def _cleanup_failed_agent(self, agent: Agent):
        """Clean up resources for failed agent creation"""
        logger.info(f"Cleaning up failed agent {agent.id}")
        
        try:
            # Delete storage volumes
            volumes = self.storage_repo.get_volumes_by_agent(agent.id)
            for volume in volumes:
                if volume.anf_volume_id:
                    await self.anf_manager.delete_volume(volume.anf_volume_id)
                self.storage_repo.delete(volume.id)
            
            # Delete vector index
            if 'vector_index' in agent.configuration:
                await self.vector_store.delete_index(agent.configuration['vector_index'])
            
            # Delete agent record
            self.agent_repo.delete(agent.id)
            
        except Exception as e:
            logger.error(f"Error cleaning up failed agent: {str(e)}")
    
    async def _cleanup_agent_resources(self, agent: Agent):
        """Clean up all agent resources"""
        logger.info(f"Cleaning up resources for agent {agent.id}")
        
        # Stop monitoring
        await self.health_monitor.stop_monitoring(agent)
        
        # Clean up storage (but keep data for terminated agents)
        if agent.state == AgentState.TERMINATED:
            # Just mark volumes as terminated
            volumes = self.storage_repo.get_volumes_by_agent(agent.id)
            for volume in volumes:
                self.storage_repo.update(volume.id, state='terminated')
        
        # Clear cache
        self._invalidate_cache(agent.id)
    
    async def _create_agent_snapshot(self, agent: Agent) -> str:
        """Create snapshot of agent state"""
        snapshot_id = f"snapshot_{agent.id}_{datetime.utcnow().timestamp()}"
        
        # Snapshot storage volumes
        volumes = self.storage_repo.get_volumes_by_agent(agent.id)
        for volume in volumes:
            self.storage_repo.create_snapshot(
                volume_id=volume.id,
                snapshot_name=f"{snapshot_id}_{volume.volume_name}",
                retention_days=7,
                tags={'migration_snapshot': 'true', 'snapshot_id': snapshot_id}
            )
        
        # Snapshot configuration and state
        snapshot_data = {
            'agent_id': str(agent.id),
            'state': agent.state,
            'configuration': agent.configuration,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await self.cache.set(
            f"agent_snapshot:{snapshot_id}",
            json.dumps(snapshot_data),
            ttl=86400  # 24 hours
        )
        
        return snapshot_id
    
    async def _prepare_migration_target(
        self,
        agent: Agent,
        target_node: str,
        target_region: Optional[str]
    ):
        """Prepare target for agent migration"""
        # Reserve resources at target
        # Create network connectivity
        # Prepare storage replication
        logger.info(f"Preparing migration target for agent {agent.id}")
    
    async def _sync_agent_data(self, agent: Agent, target_node: str):
        """Sync agent data to target"""
        # Sync storage volumes
        # Sync vector data
        # Sync cache
        logger.info(f"Syncing data for agent {agent.id} to {target_node}")
    
    async def _perform_migration_switchover(self, agent: Agent, target_node: str):
        """Perform migration switchover"""
        # Update routing
        # Switch storage
        # Update DNS/service discovery
        logger.info(f"Performing switchover for agent {agent.id}")
    
    async def _verify_migration(self, agent: Agent, target_node: str) -> bool:
        """Verify migration success"""
        # Check connectivity
        # Verify data integrity
        # Test functionality
        logger.info(f"Verifying migration for agent {agent.id}")
        return True
    
    async def _rollback_migration(self, agent: Agent):
        """Rollback failed migration"""
        logger.info(f"Rolling back migration for agent {agent.id}")
        # Restore original state
        # Clean up target resources
    
    # Background task loops
    
    async def _health_check_loop(self):
        """Background health check loop"""
        logger.info("Starting health check loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Get all running agents
                agents = self.agent_repo.get_active_agents(
                    tenant_id=UUID('00000000-0000-0000-0000-000000000000'),  # All tenants
                    limit=1000
                )
                
                # Check health of each agent
                for agent in agents:
                    if agent.state == AgentState.RUNNING:
                        try:
                            health = await self.health_monitor.check_agent_health(agent)
                            
                            # Update health status
                            if not health['healthy']:
                                logger.warning(f"Agent {agent.id} is unhealthy: {health}")
                                
                                # Trigger recovery if needed
                                if health.get('requires_restart'):
                                    await self.restart_agent(agent.id)
                                    
                        except Exception as e:
                            logger.error(f"Health check failed for agent {agent.id}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Health check loop error: {str(e)}")
            
            # Wait before next check
            await asyncio.sleep(30)  # 30 seconds
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        logger.info("Starting metrics collection loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics for all agents
                await self.metrics_collector.collect_all_metrics()
                
            except Exception as e:
                logger.error(f"Metrics collection error: {str(e)}")
            
            # Wait before next collection
            await asyncio.sleep(60)  # 1 minute
    
    async def _cache_cleanup_loop(self):
        """Background cache cleanup loop"""
        logger.info("Starting cache cleanup loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Clean expired cache entries
                now = datetime.utcnow()
                expired = []
                
                for agent_id, timestamp in self._cache_timestamps.items():
                    if now - timestamp > self._cache_ttl:
                        expired.append(agent_id)
                
                for agent_id in expired:
                    self._invalidate_cache(agent_id)
                
                if expired:
                    logger.debug(f"Cleaned {len(expired)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
            
            # Wait before next cleanup
            await asyncio.sleep(300)  # 5 minutes
    
    async def _resource_optimization_loop(self):
        """Background resource optimization loop"""
        logger.info("Starting resource optimization loop")
        
        while not self._shutdown_event.is_set():
            try:
                # Analyze resource usage
                agents = self.agent_repo.get_active_agents(
                    tenant_id=UUID('00000000-0000-0000-0000-000000000000'),
                    limit=1000
                )
                
                for agent in agents:
                    if agent.state == AgentState.RUNNING:
                        # Get resource metrics
                        metrics = await self.metrics_collector.get_resource_metrics(agent.id)
                        
                        # Check for optimization opportunities
                        if metrics['cpu_usage'] < 10 and metrics['memory_usage'] < 20:
                            # Agent is underutilized
                            logger.info(f"Agent {agent.id} is underutilized")
                            # Could scale down resources
                        
                        elif metrics['cpu_usage'] > 80 or metrics['memory_usage'] > 80:
                            # Agent needs more resources
                            logger.info(f"Agent {agent.id} needs more resources")
                            # Could scale up resources
                
            except Exception as e:
                logger.error(f"Resource optimization error: {str(e)}")
            
            # Wait before next optimization
            await asyncio.sleep(600)  # 10 minutes


class AgentLifecycleManager:
    """
    Manages agent lifecycle policies and automation
    """
    
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        self.policies: Dict[str, Dict[str, Any]] = {}
    
    def add_policy(self, name: str, policy: Dict[str, Any]):
        """Add a lifecycle policy"""
        self.policies[name] = policy
    
    async def apply_policies(self, agent: Agent):
        """Apply lifecycle policies to an agent"""
        for policy_name, policy in self.policies.items():
            if self._matches_policy(agent, policy):
                await self._execute_policy(agent, policy)
    
    def _matches_policy(self, agent: Agent, policy: Dict[str, Any]) -> bool:
        """Check if agent matches policy criteria"""
        criteria = policy.get('criteria', {})
        
        # Check agent type
        if 'agent_type' in criteria:
            if agent.agent_type not in criteria['agent_type']:
                return False
        
        # Check tags
        if 'tags' in criteria:
            required_tags = set(criteria['tags'])
            if not required_tags.issubset(set(agent.tags or [])):
                return False
        
        # Check state
        if 'state' in criteria:
            if agent.state not in criteria['state']:
                return False
        
        return True
    
    async def _execute_policy(self, agent: Agent, policy: Dict[str, Any]):
        """Execute a lifecycle policy"""
        actions = policy.get('actions', [])
        
        for action in actions:
            action_type = action['type']
            
            if action_type == 'auto_scale':
                await self._auto_scale_agent(agent, action)
            elif action_type == 'auto_restart':
                await self._auto_restart_agent(agent, action)
            elif action_type == 'auto_archive':
                await self._auto_archive_agent(agent, action)
            elif action_type == 'notification':
                await self._send_notification(agent, action)
    
    async def _auto_scale_agent(self, agent: Agent, action: Dict[str, Any]):
        """Auto-scale agent resources"""
        # Implement auto-scaling logic
        pass
    
    async def _auto_restart_agent(self, agent: Agent, action: Dict[str, Any]):
        """Auto-restart agent based on conditions"""
        # Implement auto-restart logic
        pass
    
    async def _auto_archive_agent(self, agent: Agent, action: Dict[str, Any]):
        """Auto-archive agent after inactivity"""
        # Implement auto-archive logic
        pass
    
    async def _send_notification(self, agent: Agent, action: Dict[str, Any]):
        """Send notification about agent state"""
        # Implement notification logic
        pass
    
    async def _get_tenant_limits(self, tenant_id: str) -> Dict[str, Any]:
        """Get resource limits for tenant"""
        # In production, this would query tenant configuration database
        # For now, return default limits based on tenant tier
        
        # Try to get from database first
        try:
            tenant_config = await self.agent_repo.get_tenant_config(tenant_id)
            if tenant_config:
                return tenant_config.get('resource_limits', {})
        except Exception:
            pass
        
        # Fallback to default limits based on tenant tier
        # These would be configurable in production
        default_limits = {
            'enterprise': {
                'max_cpu_cores': 1000,
                'max_memory_gb': 10000,
                'max_storage_gb': 100000,
                'max_agents': 500,
                'max_gpus': 100
            },
            'professional': {
                'max_cpu_cores': 200,
                'max_memory_gb': 2000,
                'max_storage_gb': 20000,
                'max_agents': 100,
                'max_gpus': 20
            },
            'starter': {
                'max_cpu_cores': 50,
                'max_memory_gb': 500,
                'max_storage_gb': 5000,
                'max_agents': 25,
                'max_gpus': 5
            }
        }
        
        # Default to starter tier
        tenant_tier = 'starter'
        
        # Try to determine tier from tenant_id or other metadata
        if 'enterprise' in tenant_id.lower():
            tenant_tier = 'enterprise'
        elif 'pro' in tenant_id.lower():
            tenant_tier = 'professional'
        
        return default_limits.get(tenant_tier, default_limits['starter'])