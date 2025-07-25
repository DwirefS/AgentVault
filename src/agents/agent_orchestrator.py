"""
AgentVault™ Agent Orchestrator
Coordinates agent deployment, scaling, and resource management
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from enum import Enum

from ..database.models import Agent, AgentState, Tenant
from ..database.repositories import AgentRepository, TenantRepository
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem

logger = logging.getLogger(__name__)


class DeploymentStrategy(str, Enum):
    """Deployment strategies for agents"""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class ScalingPolicy(str, Enum):
    """Scaling policies for agents"""
    MANUAL = "manual"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"


@dataclass
class DeploymentConfig:
    """Configuration for agent deployment"""
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    replicas: int = 1
    max_surge: int = 1
    max_unavailable: int = 0
    health_check_interval: int = 30
    rollback_on_failure: bool = True
    canary_percentage: int = 10
    canary_duration_minutes: int = 30
    resource_limits: Dict[str, Any] = None
    node_selector: Dict[str, str] = None
    tolerations: List[Dict[str, Any]] = None
    affinity_rules: Dict[str, Any] = None


@dataclass
class ScalingConfig:
    """Configuration for agent scaling"""
    policy: ScalingPolicy = ScalingPolicy.REACTIVE
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70
    target_memory_utilization: int = 80
    target_request_rate: int = 1000
    scale_up_rate: int = 2  # Max replicas to add at once
    scale_down_rate: int = 1  # Max replicas to remove at once
    scale_up_cooldown: int = 300  # 5 minutes
    scale_down_cooldown: int = 900  # 15 minutes
    predictive_model: Optional[str] = None
    schedule_rules: List[Dict[str, Any]] = None


@dataclass
class PlacementConstraints:
    """Constraints for agent placement"""
    required_labels: Dict[str, str] = None
    preferred_zones: List[str] = None
    avoid_zones: List[str] = None
    spread_across: str = "zone"  # zone, node, rack
    max_per_node: int = 2
    gpu_required: bool = False
    ssd_required: bool = False
    network_bandwidth_mbps: int = 1000
    memory_type: str = "standard"  # standard, high-performance


class AgentOrchestrator:
    """
    Orchestrates agent deployment, scaling, and placement
    """
    
    def __init__(
        self,
        agent_manager: Any,  # Avoid circular import
        monitoring: AdvancedMonitoringSystem,
        config: Dict[str, Any]
    ):
        self.agent_manager = agent_manager
        self.monitoring = monitoring
        self.config = config
        
        # Track deployments
        self._active_deployments: Dict[str, Dict[str, Any]] = {}
        self._scaling_decisions: Dict[str, List[Dict[str, Any]]] = {}
        
        # Cluster state
        self._cluster_nodes: Dict[str, Dict[str, Any]] = {}
        self._resource_availability: Dict[str, Dict[str, float]] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize orchestrator"""
        logger.info("Initializing Agent Orchestrator")
        
        # Start background tasks
        self._background_tasks.extend([
            asyncio.create_task(self._cluster_state_sync_loop()),
            asyncio.create_task(self._scaling_loop()),
            asyncio.create_task(self._placement_optimization_loop())
        ])
        
        # Initialize cluster state
        await self._sync_cluster_state()
        
        logger.info("Agent Orchestrator initialized")
    
    async def shutdown(self):
        """Shutdown orchestrator"""
        logger.info("Shutting down Agent Orchestrator")
        
        self._shutdown_event.set()
        
        # Wait for background tasks
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Agent Orchestrator shutdown complete")
    
    async def deploy_agent(
        self,
        agent: Agent,
        deployment_config: Optional[DeploymentConfig] = None
    ) -> Dict[str, Any]:
        """Deploy an agent with specified configuration"""
        if not deployment_config:
            deployment_config = DeploymentConfig()
        
        logger.info(f"Deploying agent {agent.id} with strategy {deployment_config.strategy}")
        
        deployment_id = f"deploy-{agent.id}-{datetime.utcnow().timestamp()}"
        
        # Create deployment record
        deployment = {
            'id': deployment_id,
            'agent_id': str(agent.id),
            'config': deployment_config,
            'status': 'pending',
            'started_at': datetime.utcnow(),
            'replicas': {
                'desired': deployment_config.replicas,
                'current': 0,
                'ready': 0,
                'updated': 0
            }
        }
        
        self._active_deployments[deployment_id] = deployment
        
        try:
            # Execute deployment based on strategy
            if deployment_config.strategy == DeploymentStrategy.ROLLING:
                result = await self._deploy_rolling(agent, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.BLUE_GREEN:
                result = await self._deploy_blue_green(agent, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.CANARY:
                result = await self._deploy_canary(agent, deployment_config)
            elif deployment_config.strategy == DeploymentStrategy.RECREATE:
                result = await self._deploy_recreate(agent, deployment_config)
            else:
                raise ValueError(f"Unknown deployment strategy: {deployment_config.strategy}")
            
            # Update deployment status
            deployment['status'] = 'completed'
            deployment['completed_at'] = datetime.utcnow()
            deployment['result'] = result
            
            # Record metrics
            self.monitoring.record_custom_metric(
                "agent_deployment",
                1,
                labels={
                    'agent_id': str(agent.id),
                    'strategy': deployment_config.strategy,
                    'status': 'success'
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Deployment failed for agent {agent.id}: {str(e)}")
            
            deployment['status'] = 'failed'
            deployment['error'] = str(e)
            deployment['failed_at'] = datetime.utcnow()
            
            # Rollback if configured
            if deployment_config.rollback_on_failure:
                await self._rollback_deployment(agent, deployment_id)
            
            # Record failure metrics
            self.monitoring.record_custom_metric(
                "agent_deployment",
                0,
                labels={
                    'agent_id': str(agent.id),
                    'strategy': deployment_config.strategy,
                    'status': 'failed'
                }
            )
            
            raise
    
    async def scale_agent(
        self,
        agent: Agent,
        target_replicas: int,
        scaling_config: Optional[ScalingConfig] = None
    ) -> Dict[str, Any]:
        """Scale agent to target replicas"""
        if not scaling_config:
            scaling_config = ScalingConfig()
        
        logger.info(f"Scaling agent {agent.id} to {target_replicas} replicas")
        
        # Validate scaling limits
        if target_replicas < scaling_config.min_replicas:
            target_replicas = scaling_config.min_replicas
        elif target_replicas > scaling_config.max_replicas:
            target_replicas = scaling_config.max_replicas
        
        # Get current replicas
        current_replicas = await self._get_agent_replicas(agent)
        
        # Check if scaling is needed
        if current_replicas == target_replicas:
            logger.info(f"Agent {agent.id} already at {target_replicas} replicas")
            return {'scaled': False, 'replicas': current_replicas}
        
        # Record scaling decision
        scaling_decision = {
            'timestamp': datetime.utcnow(),
            'from_replicas': current_replicas,
            'to_replicas': target_replicas,
            'reason': 'manual',
            'metrics': await self._get_scaling_metrics(agent)
        }
        
        if str(agent.id) not in self._scaling_decisions:
            self._scaling_decisions[str(agent.id)] = []
        self._scaling_decisions[str(agent.id)].append(scaling_decision)
        
        # Perform scaling
        if target_replicas > current_replicas:
            # Scale up
            await self._scale_up(agent, current_replicas, target_replicas, scaling_config)
        else:
            # Scale down
            await self._scale_down(agent, current_replicas, target_replicas, scaling_config)
        
        # Update agent metadata
        if not agent.state_metadata:
            agent.state_metadata = {}
        agent.state_metadata['replicas'] = target_replicas
        agent.state_metadata['last_scaled'] = datetime.utcnow().isoformat()
        
        # Record metrics
        self.monitoring.record_custom_metric(
            "agent_scaling",
            target_replicas - current_replicas,
            labels={
                'agent_id': str(agent.id),
                'direction': 'up' if target_replicas > current_replicas else 'down'
            }
        )
        
        return {
            'scaled': True,
            'previous_replicas': current_replicas,
            'current_replicas': target_replicas
        }
    
    async def update_agent_deployment(
        self,
        agent: Agent,
        updates: Dict[str, Any],
        deployment_config: Optional[DeploymentConfig] = None
    ) -> Dict[str, Any]:
        """Update agent deployment (rolling update)"""
        logger.info(f"Updating agent {agent.id} deployment")
        
        if not deployment_config:
            deployment_config = DeploymentConfig()
        
        # Use rolling update strategy
        return await self._deploy_rolling(agent, deployment_config, update=True)
    
    async def place_agent(
        self,
        agent: Agent,
        constraints: Optional[PlacementConstraints] = None
    ) -> str:
        """Determine optimal placement for agent"""
        if not constraints:
            constraints = PlacementConstraints()
        
        logger.info(f"Finding placement for agent {agent.id}")
        
        # Get eligible nodes
        eligible_nodes = await self._get_eligible_nodes(agent, constraints)
        
        if not eligible_nodes:
            raise ValueError("No eligible nodes found for agent placement")
        
        # Score nodes
        node_scores = {}
        for node_id, node_info in eligible_nodes.items():
            score = await self._score_node(node_id, node_info, agent, constraints)
            node_scores[node_id] = score
        
        # Select best node
        best_node = max(node_scores, key=node_scores.get)
        
        logger.info(f"Selected node {best_node} for agent {agent.id} (score: {node_scores[best_node]})")
        
        return best_node
    
    async def rebalance_agents(
        self,
        tenant_id: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Rebalance agent placement across cluster"""
        logger.info(f"Rebalancing agents (dry_run={dry_run})")
        
        # Get all agents to rebalance
        if tenant_id:
            agents = self.agent_manager.agent_repo.get_active_agents(
                tenant_id=tenant_id,
                limit=1000
            )
        else:
            # All agents
            agents = []  # Would query all agents
        
        # Calculate current distribution
        current_distribution = await self._calculate_distribution(agents)
        
        # Calculate optimal distribution
        optimal_distribution = await self._calculate_optimal_distribution(agents)
        
        # Generate migration plan
        migration_plan = self._generate_migration_plan(
            current_distribution,
            optimal_distribution
        )
        
        if dry_run:
            return {
                'current_distribution': current_distribution,
                'optimal_distribution': optimal_distribution,
                'migration_plan': migration_plan,
                'agents_to_migrate': len(migration_plan),
                'estimated_duration_minutes': len(migration_plan) * 5
            }
        
        # Execute migrations
        migrated = 0
        failed = 0
        
        for migration in migration_plan:
            try:
                await self.agent_manager.migrate_agent(
                    migration['agent_id'],
                    migration['target_node'],
                    migration.get('target_region')
                )
                migrated += 1
            except Exception as e:
                logger.error(f"Failed to migrate agent {migration['agent_id']}: {str(e)}")
                failed += 1
        
        return {
            'migrated': migrated,
            'failed': failed,
            'total': len(migration_plan)
        }
    
    # Deployment strategies implementation
    
    async def _deploy_rolling(
        self,
        agent: Agent,
        config: DeploymentConfig,
        update: bool = False
    ) -> Dict[str, Any]:
        """Rolling deployment strategy"""
        logger.info(f"Executing rolling deployment for agent {agent.id}")
        
        results = {
            'strategy': 'rolling',
            'started_at': datetime.utcnow(),
            'phases': []
        }
        
        # Calculate deployment waves
        current_replicas = await self._get_agent_replicas(agent) if update else 0
        target_replicas = config.replicas
        
        # Deploy in waves
        while current_replicas < target_replicas:
            wave_size = min(config.max_surge, target_replicas - current_replicas)
            
            phase_result = {
                'phase': len(results['phases']) + 1,
                'replicas': wave_size,
                'status': 'pending'
            }
            
            try:
                # Deploy replicas
                for i in range(wave_size):
                    replica_id = f"{agent.id}-{current_replicas + i}"
                    node = await self.place_agent(agent)
                    
                    # Start agent instance
                    await self._start_agent_replica(agent, replica_id, node)
                
                # Wait for health check
                await asyncio.sleep(config.health_check_interval)
                
                # Verify health
                if await self._verify_replica_health(agent, wave_size):
                    phase_result['status'] = 'completed'
                    current_replicas += wave_size
                    
                    # If updating, remove old replicas
                    if update and current_replicas > config.replicas:
                        await self._remove_old_replicas(agent, current_replicas - config.replicas)
                else:
                    phase_result['status'] = 'failed'
                    raise Exception("Health check failed for new replicas")
                
            except Exception as e:
                phase_result['status'] = 'failed'
                phase_result['error'] = str(e)
                results['phases'].append(phase_result)
                raise
            
            results['phases'].append(phase_result)
        
        results['completed_at'] = datetime.utcnow()
        results['success'] = True
        
        return results
    
    async def _deploy_blue_green(
        self,
        agent: Agent,
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Blue-green deployment strategy"""
        logger.info(f"Executing blue-green deployment for agent {agent.id}")
        
        # Deploy green (new) environment
        green_deployment = await self._create_deployment_environment(
            agent,
            config,
            label='green'
        )
        
        # Test green environment
        if not await self._test_deployment(green_deployment):
            await self._cleanup_deployment(green_deployment)
            raise Exception("Green deployment failed testing")
        
        # Switch traffic to green
        await self._switch_traffic(agent, 'green')
        
        # Remove blue (old) environment
        blue_deployment = await self._get_deployment_environment(agent, 'blue')
        if blue_deployment:
            await self._cleanup_deployment(blue_deployment)
        
        # Label green as new blue
        await self._label_deployment(green_deployment, 'blue')
        
        return {
            'strategy': 'blue_green',
            'success': True,
            'switched_at': datetime.utcnow()
        }
    
    async def _deploy_canary(
        self,
        agent: Agent,
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Canary deployment strategy"""
        logger.info(f"Executing canary deployment for agent {agent.id}")
        
        # Deploy canary instances
        canary_replicas = max(1, int(config.replicas * config.canary_percentage / 100))
        
        canary_deployment = await self._create_deployment_environment(
            agent,
            DeploymentConfig(
                replicas=canary_replicas,
                **{k: v for k, v in config.__dict__.items() if k != 'replicas'}
            ),
            label='canary'
        )
        
        # Route percentage of traffic to canary
        await self._configure_traffic_split(
            agent,
            {'stable': 100 - config.canary_percentage, 'canary': config.canary_percentage}
        )
        
        # Monitor canary
        start_time = datetime.utcnow()
        canary_healthy = True
        
        while (datetime.utcnow() - start_time).total_seconds() < config.canary_duration_minutes * 60:
            if not await self._monitor_canary_health(agent, canary_deployment):
                canary_healthy = False
                break
            await asyncio.sleep(60)  # Check every minute
        
        if canary_healthy:
            # Promote canary
            await self._promote_canary(agent, config)
            result = {'promoted': True}
        else:
            # Rollback canary
            await self._rollback_canary(agent, canary_deployment)
            result = {'promoted': False, 'reason': 'Canary health check failed'}
        
        return {
            'strategy': 'canary',
            'success': canary_healthy,
            'duration_minutes': (datetime.utcnow() - start_time).total_seconds() / 60,
            **result
        }
    
    async def _deploy_recreate(
        self,
        agent: Agent,
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """Recreate deployment strategy (downtime)"""
        logger.info(f"Executing recreate deployment for agent {agent.id}")
        
        # Stop all existing replicas
        await self._stop_all_replicas(agent)
        
        # Deploy new replicas
        for i in range(config.replicas):
            replica_id = f"{agent.id}-{i}"
            node = await self.place_agent(agent)
            await self._start_agent_replica(agent, replica_id, node)
        
        # Wait for all replicas to be ready
        await self._wait_for_replicas_ready(agent, config.replicas)
        
        return {
            'strategy': 'recreate',
            'success': True,
            'replicas': config.replicas
        }
    
    # Scaling implementation
    
    async def _scale_up(
        self,
        agent: Agent,
        current: int,
        target: int,
        config: ScalingConfig
    ):
        """Scale up agent replicas"""
        to_add = min(target - current, config.scale_up_rate)
        
        logger.info(f"Scaling up agent {agent.id} by {to_add} replicas")
        
        for i in range(to_add):
            replica_id = f"{agent.id}-{current + i}"
            node = await self.place_agent(agent)
            await self._start_agent_replica(agent, replica_id, node)
        
        # Update cooldown
        if not agent.state_metadata:
            agent.state_metadata = {}
        agent.state_metadata['last_scale_up'] = datetime.utcnow().isoformat()
    
    async def _scale_down(
        self,
        agent: Agent,
        current: int,
        target: int,
        config: ScalingConfig
    ):
        """Scale down agent replicas"""
        to_remove = min(current - target, config.scale_down_rate)
        
        logger.info(f"Scaling down agent {agent.id} by {to_remove} replicas")
        
        # Select replicas to remove (oldest first)
        replicas = await self._get_agent_replica_list(agent)
        replicas_to_remove = sorted(replicas, key=lambda r: r['created_at'])[:to_remove]
        
        for replica in replicas_to_remove:
            await self._stop_agent_replica(agent, replica['id'])
        
        # Update cooldown
        if not agent.state_metadata:
            agent.state_metadata = {}
        agent.state_metadata['last_scale_down'] = datetime.utcnow().isoformat()
    
    async def _auto_scale_agent(self, agent: Agent, config: ScalingConfig):
        """Auto-scale agent based on metrics"""
        # Get current metrics
        metrics = await self._get_scaling_metrics(agent)
        current_replicas = await self._get_agent_replicas(agent)
        
        # Determine scaling decision
        scale_decision = None
        
        if config.policy == ScalingPolicy.REACTIVE:
            # Check CPU utilization
            if metrics['cpu_usage'] > config.target_cpu_utilization:
                scale_decision = 'up'
            elif metrics['cpu_usage'] < config.target_cpu_utilization * 0.5:
                scale_decision = 'down'
            
            # Check memory utilization
            if metrics['memory_usage'] > config.target_memory_utilization:
                scale_decision = 'up'
            
            # Check request rate
            if metrics['request_rate'] > config.target_request_rate:
                scale_decision = 'up'
        
        elif config.policy == ScalingPolicy.PREDICTIVE:
            # Use ML model to predict future load
            prediction = await self._predict_load(agent, config.predictive_model)
            if prediction['future_load'] > current_replicas * config.target_request_rate:
                scale_decision = 'up'
            elif prediction['future_load'] < current_replicas * config.target_request_rate * 0.5:
                scale_decision = 'down'
        
        elif config.policy == ScalingPolicy.SCHEDULED:
            # Check schedule rules
            current_time = datetime.utcnow()
            for rule in config.schedule_rules or []:
                if self._matches_schedule(current_time, rule):
                    target_replicas = rule['replicas']
                    if target_replicas > current_replicas:
                        scale_decision = 'up'
                    elif target_replicas < current_replicas:
                        scale_decision = 'down'
                    break
        
        # Check cooldown periods
        if scale_decision == 'up':
            last_scale_up = agent.state_metadata.get('last_scale_up') if agent.state_metadata else None
            if last_scale_up:
                last_scale_up_time = datetime.fromisoformat(last_scale_up)
                if (datetime.utcnow() - last_scale_up_time).seconds < config.scale_up_cooldown:
                    scale_decision = None
        
        elif scale_decision == 'down':
            last_scale_down = agent.state_metadata.get('last_scale_down') if agent.state_metadata else None
            if last_scale_down:
                last_scale_down_time = datetime.fromisoformat(last_scale_down)
                if (datetime.utcnow() - last_scale_down_time).seconds < config.scale_down_cooldown:
                    scale_decision = None
        
        # Execute scaling decision
        if scale_decision == 'up':
            target = min(current_replicas + config.scale_up_rate, config.max_replicas)
            await self.scale_agent(agent, target, config)
        elif scale_decision == 'down':
            target = max(current_replicas - config.scale_down_rate, config.min_replicas)
            await self.scale_agent(agent, target, config)
    
    # Helper methods
    
    async def _get_agent_replicas(self, agent: Agent) -> int:
        """Get current number of replicas for agent"""
        # This would query the actual deployment system
        return agent.state_metadata.get('replicas', 1) if agent.state_metadata else 1
    
    async def _get_agent_replica_list(self, agent: Agent) -> List[Dict[str, Any]]:
        """Get list of agent replicas"""
        # This would query the actual deployment system
        replicas = []
        for i in range(await self._get_agent_replicas(agent)):
            replicas.append({
                'id': f"{agent.id}-{i}",
                'created_at': datetime.utcnow() - timedelta(hours=i)
            })
        return replicas
    
    async def _start_agent_replica(self, agent: Agent, replica_id: str, node: str):
        """Start an agent replica on specified node"""
        logger.info(f"Starting replica {replica_id} on node {node}")
        # This would integrate with container orchestration
    
    async def _stop_agent_replica(self, agent: Agent, replica_id: str):
        """Stop an agent replica"""
        logger.info(f"Stopping replica {replica_id}")
        # This would integrate with container orchestration
    
    async def _verify_replica_health(self, agent: Agent, count: int) -> bool:
        """Verify health of agent replicas"""
        # This would check actual replica health
        return True
    
    async def _get_scaling_metrics(self, agent: Agent) -> Dict[str, float]:
        """Get metrics for scaling decisions"""
        # This would query actual metrics
        return {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'request_rate': 100,
            'error_rate': 0.1,
            'latency_p99': 100
        }
    
    async def _get_eligible_nodes(
        self,
        agent: Agent,
        constraints: PlacementConstraints
    ) -> Dict[str, Dict[str, Any]]:
        """Get nodes eligible for agent placement"""
        eligible = {}
        
        for node_id, node_info in self._cluster_nodes.items():
            # Check required labels
            if constraints.required_labels:
                if not all(
                    node_info.get('labels', {}).get(k) == v
                    for k, v in constraints.required_labels.items()
                ):
                    continue
            
            # Check zone constraints
            node_zone = node_info.get('zone')
            if constraints.avoid_zones and node_zone in constraints.avoid_zones:
                continue
            
            # Check resource availability
            if not self._has_sufficient_resources(node_id, agent, constraints):
                continue
            
            eligible[node_id] = node_info
        
        return eligible
    
    def _has_sufficient_resources(
        self,
        node_id: str,
        agent: Agent,
        constraints: PlacementConstraints
    ) -> bool:
        """Check if node has sufficient resources"""
        node_resources = self._resource_availability.get(node_id, {})
        
        # Check CPU
        if node_resources.get('cpu_available', 0) < agent.cpu_cores:
            return False
        
        # Check memory
        if node_resources.get('memory_available', 0) < agent.memory_gb:
            return False
        
        # Check GPU if required
        if agent.gpu_enabled or constraints.gpu_required:
            if not node_resources.get('gpu_available', False):
                return False
        
        return True
    
    async def _score_node(
        self,
        node_id: str,
        node_info: Dict[str, Any],
        agent: Agent,
        constraints: PlacementConstraints
    ) -> float:
        """Score a node for agent placement"""
        score = 100.0
        
        # Resource availability score
        node_resources = self._resource_availability.get(node_id, {})
        cpu_available_pct = node_resources.get('cpu_available', 0) / node_resources.get('cpu_total', 1)
        memory_available_pct = node_resources.get('memory_available', 0) / node_resources.get('memory_total', 1)
        
        score *= (cpu_available_pct + memory_available_pct) / 2
        
        # Zone preference score
        if constraints.preferred_zones:
            if node_info.get('zone') in constraints.preferred_zones:
                score *= 1.2
        
        # Spread score
        current_agents_on_node = node_info.get('agent_count', 0)
        if current_agents_on_node >= constraints.max_per_node:
            score *= 0.1
        else:
            score *= (1 - current_agents_on_node / constraints.max_per_node)
        
        # Network bandwidth score
        if constraints.network_bandwidth_mbps > 0:
            available_bandwidth = node_info.get('network_bandwidth_available', 0)
            if available_bandwidth >= constraints.network_bandwidth_mbps:
                score *= 1.1
        
        return score
    
    async def _sync_cluster_state(self):
        """Sync cluster state from orchestration system"""
        # This would integrate with Kubernetes, Docker Swarm, etc.
        # For now, simulate some nodes
        self._cluster_nodes = {
            'node-1': {
                'zone': 'us-east-1a',
                'labels': {'type': 'compute', 'tier': 'premium'},
                'agent_count': 2
            },
            'node-2': {
                'zone': 'us-east-1b',
                'labels': {'type': 'compute', 'tier': 'standard'},
                'agent_count': 1
            },
            'node-3': {
                'zone': 'us-east-1c',
                'labels': {'type': 'gpu', 'tier': 'premium'},
                'agent_count': 0
            }
        }
        
        self._resource_availability = {
            'node-1': {
                'cpu_total': 16,
                'cpu_available': 8,
                'memory_total': 64,
                'memory_available': 32,
                'gpu_available': False
            },
            'node-2': {
                'cpu_total': 8,
                'cpu_available': 6,
                'memory_total': 32,
                'memory_available': 24,
                'gpu_available': False
            },
            'node-3': {
                'cpu_total': 32,
                'cpu_available': 30,
                'memory_total': 128,
                'memory_available': 120,
                'gpu_available': True
            }
        }
    
    async def _calculate_distribution(
        self,
        agents: List[Agent]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate current agent distribution"""
        distribution = {}
        
        for agent in agents:
            node = agent.state_metadata.get('node') if agent.state_metadata else 'unknown'
            zone = self._cluster_nodes.get(node, {}).get('zone', 'unknown')
            
            if zone not in distribution:
                distribution[zone] = {'agents': 0, 'cpu': 0, 'memory': 0}
            
            distribution[zone]['agents'] += 1
            distribution[zone]['cpu'] += agent.cpu_cores
            distribution[zone]['memory'] += agent.memory_gb
        
        return distribution
    
    async def _calculate_optimal_distribution(
        self,
        agents: List[Agent]
    ) -> Dict[str, Dict[str, int]]:
        """Calculate optimal agent distribution"""
        # This would implement bin packing or other optimization algorithms
        # For now, simple even distribution across zones
        zones = list(set(node['zone'] for node in self._cluster_nodes.values()))
        agents_per_zone = len(agents) // len(zones)
        
        distribution = {}
        for i, zone in enumerate(zones):
            distribution[zone] = {
                'agents': agents_per_zone + (1 if i < len(agents) % len(zones) else 0),
                'cpu': sum(a.cpu_cores for a in agents[i::len(zones)]),
                'memory': sum(a.memory_gb for a in agents[i::len(zones)])
            }
        
        return distribution
    
    def _generate_migration_plan(
        self,
        current: Dict[str, Dict[str, int]],
        optimal: Dict[str, Dict[str, int]]
    ) -> List[Dict[str, Any]]:
        """Generate migration plan to achieve optimal distribution"""
        migrations = []
        
        # Simple algorithm: move agents from over-provisioned to under-provisioned zones
        # In production, this would be more sophisticated
        
        return migrations
    
    def _matches_schedule(self, current_time: datetime, rule: Dict[str, Any]) -> bool:
        """Check if current time matches schedule rule"""
        # This would implement cron-like scheduling
        return False
    
    async def _predict_load(self, agent: Agent, model_name: str) -> Dict[str, float]:
        """Predict future load using ML model"""
        # This would integrate with ML models
        return {'future_load': 100.0}
    
    # Background loops
    
    async def _cluster_state_sync_loop(self):
        """Periodically sync cluster state"""
        while not self._shutdown_event.is_set():
            try:
                await self._sync_cluster_state()
            except Exception as e:
                logger.error(f"Cluster state sync error: {str(e)}")
            
            await asyncio.sleep(30)  # Every 30 seconds
    
    async def _scaling_loop(self):
        """Auto-scaling loop"""
        while not self._shutdown_event.is_set():
            try:
                # Get agents with auto-scaling enabled
                # This would query agents with scaling policies
                agents_to_scale = []
                
                for agent in agents_to_scale:
                    scaling_config = self._get_agent_scaling_config(agent)
                    if scaling_config.policy != ScalingPolicy.MANUAL:
                        await self._auto_scale_agent(agent, scaling_config)
                
            except Exception as e:
                logger.error(f"Scaling loop error: {str(e)}")
            
            await asyncio.sleep(60)  # Every minute
    
    async def _placement_optimization_loop(self):
        """Periodically optimize agent placement"""
        while not self._shutdown_event.is_set():
            try:
                # Run rebalancing in dry-run mode
                result = await self.rebalance_agents(dry_run=True)
                
                # If significant imbalance, log recommendation
                if result.get('agents_to_migrate', 0) > 10:
                    logger.info(
                        f"Cluster imbalance detected: {result['agents_to_migrate']} "
                        f"agents could be migrated for better distribution"
                    )
                
            except Exception as e:
                logger.error(f"Placement optimization error: {str(e)}")
            
            await asyncio.sleep(3600)  # Every hour
    
    def _get_agent_scaling_config(self, agent: Agent) -> ScalingConfig:
        """Get scaling configuration for agent"""
        # This would load from agent configuration
        return ScalingConfig()
    
    # Deployment helper methods (stubs for actual implementation)
    
    async def _create_deployment_environment(
        self,
        agent: Agent,
        config: DeploymentConfig,
        label: str
    ) -> Dict[str, Any]:
        """Create a deployment environment"""
        return {'id': f"deploy-{agent.id}-{label}", 'label': label}
    
    async def _test_deployment(self, deployment: Dict[str, Any]) -> bool:
        """Test a deployment"""
        return True
    
    async def _switch_traffic(self, agent: Agent, target: str):
        """Switch traffic to target deployment"""
        try:
            logger.info(f"Switching traffic for agent {agent.id} to {target}")
            
            # For Kubernetes deployments
            if agent.configuration.get('runtime_type') == 'kubernetes':
                from kubernetes import client, config as k8s_config
                k8s_config.load_incluster_config()
                v1 = client.CoreV1Api()
                
                # Update service selector to point to target deployment
                service_name = f"agent-{agent.id}-service"
                namespace = agent.configuration.get('namespace', 'default')
                
                # Get current service
                service = v1.read_namespaced_service(
                    name=service_name,
                    namespace=namespace
                )
                
                # Update selector to target deployment
                service.spec.selector['deployment'] = target
                
                v1.patch_namespaced_service(
                    name=service_name,
                    namespace=namespace,
                    body=service
                )
                
            # For Docker deployments using load balancer
            elif agent.configuration.get('runtime_type') == 'docker':
                # Update load balancer configuration
                lb_config = {
                    'target_container': f"agent-{agent.id}-{target}",
                    'port': agent.configuration.get('port', 8080)
                }
                
                # This would integrate with your load balancer (nginx, traefik, etc.)
                await self._update_load_balancer_config(agent.id, lb_config)
            
            logger.info(f"Traffic switched to {target} for agent {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to switch traffic for agent {agent.id}: {str(e)}")
            raise
    
    async def _cleanup_deployment(self, deployment: Dict[str, Any]):
        """Clean up a deployment"""
        try:
            deployment_id = deployment.get('id')
            agent_id = deployment.get('agent_id')
            deployment_type = deployment.get('type', 'kubernetes')
            
            logger.info(f"Cleaning up deployment {deployment_id} for agent {agent_id}")
            
            if deployment_type == 'kubernetes':
                from kubernetes import client, config as k8s_config
                k8s_config.load_incluster_config()
                apps_v1 = client.AppsV1Api()
                v1 = client.CoreV1Api()
                
                namespace = deployment.get('namespace', 'default')
                
                # Delete deployment
                try:
                    apps_v1.delete_namespaced_deployment(
                        name=deployment_id,
                        namespace=namespace
                    )
                except client.rest.ApiException as e:
                    if e.status != 404:  # Ignore if already deleted
                        raise
                
                # Delete associated services
                try:
                    v1.delete_namespaced_service(
                        name=f"{deployment_id}-service",
                        namespace=namespace
                    )
                except client.rest.ApiException as e:
                    if e.status != 404:
                        raise
                
                # Delete configmaps
                try:
                    v1.delete_namespaced_config_map(
                        name=f"{deployment_id}-config",
                        namespace=namespace
                    )
                except client.rest.ApiException as e:
                    if e.status != 404:
                        raise
                        
            elif deployment_type == 'docker':
                import subprocess
                container_name = deployment.get('container_name')
                
                if container_name:
                    # Stop and remove container
                    subprocess.run(
                        ['docker', 'stop', container_name],
                        check=False
                    )
                    subprocess.run(
                        ['docker', 'rm', '-f', container_name],
                        check=False
                    )
                    
                    # Remove associated volumes
                    volume_name = f"agentvault-{agent_id}-data"
                    subprocess.run(
                        ['docker', 'volume', 'rm', volume_name],
                        check=False
                    )
            
            logger.info(f"Deployment {deployment_id} cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup deployment {deployment.get('id')}: {str(e)}")
            raise
    
    async def _get_deployment_environment(
        self,
        agent: Agent,
        label: str
    ) -> Optional[Dict[str, Any]]:
        """Get deployment environment by label"""
        try:
            if agent.configuration.get('runtime_type') == 'kubernetes':
                from kubernetes import client, config as k8s_config
                k8s_config.load_incluster_config()
                apps_v1 = client.AppsV1Api()
                
                namespace = agent.configuration.get('namespace', 'default')
                
                # List deployments with the specified label
                deployments = apps_v1.list_namespaced_deployment(
                    namespace=namespace,
                    label_selector=f"agent-id={agent.id},deployment-stage={label}"
                )
                
                if deployments.items:
                    deployment = deployments.items[0]  # Get first matching deployment
                    return {
                        'id': deployment.metadata.name,
                        'type': 'kubernetes',
                        'namespace': namespace,
                        'labels': deployment.metadata.labels,
                        'replicas': deployment.spec.replicas,
                        'ready_replicas': deployment.status.ready_replicas or 0,
                        'created_at': deployment.metadata.creation_timestamp,
                        'agent_id': str(agent.id)
                    }
                    
            elif agent.configuration.get('runtime_type') == 'docker':
                import subprocess
                
                # Find containers with the specified label
                result = subprocess.run([
                    'docker', 'ps', '-a', '--format', '{{.ID}} {{.Names}} {{.Status}}',
                    '--filter', f"label=agent-id={agent.id}",
                    '--filter', f"label=deployment-stage={label}"
                ], capture_output=True, text=True)
                
                if result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    container_info = lines[0].split()
                    
                    return {
                        'id': f"agent-{agent.id}-{label}",
                        'type': 'docker',
                        'container_id': container_info[0],
                        'container_name': container_info[1],
                        'status': ' '.join(container_info[2:]),
                        'agent_id': str(agent.id),
                        'labels': {'deployment-stage': label}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get deployment environment {label} for agent {agent.id}: {str(e)}")
            return None
    
    async def _label_deployment(self, deployment: Dict[str, Any], label: str):
        """Label a deployment"""
        try:
            deployment_id = deployment.get('id')
            deployment_type = deployment.get('type', 'kubernetes')
            
            logger.info(f"Labeling deployment {deployment_id} with {label}")
            
            if deployment_type == 'kubernetes':
                from kubernetes import client, config as k8s_config
                k8s_config.load_incluster_config()
                apps_v1 = client.AppsV1Api()
                
                namespace = deployment.get('namespace', 'default')
                
                # Add label to deployment
                body = {
                    'metadata': {
                        'labels': {
                            'deployment-stage': label,
                            'last-updated': datetime.utcnow().strftime('%Y%m%d-%H%M%S')
                        }
                    }
                }
                
                apps_v1.patch_namespaced_deployment(
                    name=deployment_id,
                    namespace=namespace,
                    body=body
                )
                
            elif deployment_type == 'docker':
                import subprocess
                container_name = deployment.get('container_name')
                
                if container_name:
                    # Add label to container
                    subprocess.run([
                        'docker', 'update',
                        '--label', f"deployment-stage={label}",
                        '--label', f"last-updated={datetime.utcnow().isoformat()}",
                        container_name
                    ], check=False)
            
            # Update internal deployment tracking
            deployment['labels'] = deployment.get('labels', {})
            deployment['labels']['deployment-stage'] = label
            deployment['labels']['last-updated'] = datetime.utcnow().isoformat()
            
            logger.info(f"Deployment {deployment_id} labeled as {label}")
            
        except Exception as e:
            logger.error(f"Failed to label deployment {deployment.get('id')}: {str(e)}")
            raise
    
    async def _configure_traffic_split(
        self,
        agent: Agent,
        weights: Dict[str, int]
    ):
        """Configure traffic split between deployments"""
        try:
            logger.info(f"Configuring traffic split for agent {agent.id}: {weights}")
            
            # Normalize weights to percentages
            total_weight = sum(weights.values())
            if total_weight == 0:
                raise ValueError("Total weight cannot be zero")
                
            normalized_weights = {k: (v / total_weight) * 100 for k, v in weights.items()}
            
            if agent.configuration.get('runtime_type') == 'kubernetes':
                # Use Istio or similar service mesh for traffic splitting
                await self._configure_istio_traffic_split(agent, normalized_weights)
                
            elif agent.configuration.get('runtime_type') == 'docker':
                # Configure load balancer with weighted routing
                await self._configure_lb_traffic_split(agent, normalized_weights)
            
            # Store traffic split configuration
            agent.configuration['traffic_split'] = {
                'weights': normalized_weights,
                'configured_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Traffic split configured: {normalized_weights}")
            
        except Exception as e:
            logger.error(f"Failed to configure traffic split for agent {agent.id}: {str(e)}")
            raise
    
    async def _monitor_canary_health(
        self,
        agent: Agent,
        deployment: Dict[str, Any]
    ) -> bool:
        """Monitor canary deployment health"""
        return True
    
    async def _promote_canary(self, agent: Agent, config: DeploymentConfig):
        """Promote canary to stable"""
        try:
            logger.info(f"Promoting canary deployment for agent {agent.id}")
            
            # Get canary deployment
            canary_deployment = await self._get_deployment_environment(agent, 'canary')
            if not canary_deployment:
                raise ValueError("No canary deployment found")
            
            # Get current stable deployment
            stable_deployment = await self._get_deployment_environment(agent, 'stable')
            
            # Switch traffic completely to canary
            await self._switch_traffic(agent, 'canary')
            
            # Wait a bit to ensure traffic is switched
            await asyncio.sleep(5)
            
            # Label canary as stable
            await self._label_deployment(canary_deployment, 'stable')
            
            # Clean up old stable deployment if it exists
            if stable_deployment:
                await self._label_deployment(stable_deployment, 'deprecated')
                # Schedule cleanup after grace period
                asyncio.create_task(self._delayed_cleanup(stable_deployment, delay_minutes=10))
            
            # Update agent configuration
            agent.configuration['stable_deployment'] = canary_deployment['id']
            agent.configuration['promotion_time'] = datetime.utcnow().isoformat()
            
            # Remove canary designation
            if 'canary_deployment' in agent.configuration:
                del agent.configuration['canary_deployment']
            
            logger.info(f"Canary promoted to stable for agent {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to promote canary for agent {agent.id}: {str(e)}")
            raise
    
    async def _rollback_canary(self, agent: Agent, deployment: Dict[str, Any]):
        """Rollback canary deployment"""
        try:
            logger.info(f"Rolling back canary deployment for agent {agent.id}")
            
            # Switch traffic back to stable
            await self._switch_traffic(agent, 'stable')
            
            # Wait for traffic to stabilize
            await asyncio.sleep(5)
            
            # Clean up canary deployment
            await self._cleanup_deployment(deployment)
            
            # Remove canary configuration
            if 'canary_deployment' in agent.configuration:
                del agent.configuration['canary_deployment']
            
            if 'traffic_split' in agent.configuration:
                del agent.configuration['traffic_split']
            
            # Record rollback event
            agent.configuration['last_rollback'] = {
                'time': datetime.utcnow().isoformat(),
                'reason': 'canary_rollback',
                'deployment_id': deployment.get('id')
            }
            
            logger.info(f"Canary rollback completed for agent {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to rollback canary for agent {agent.id}: {str(e)}")
            raise
    
    async def _stop_all_replicas(self, agent: Agent):
        """Stop all agent replicas"""
        try:
            logger.info(f"Stopping all replicas for agent {agent.id}")
            
            if agent.configuration.get('runtime_type') == 'kubernetes':
                from kubernetes import client, config as k8s_config
                k8s_config.load_incluster_config()
                apps_v1 = client.AppsV1Api()
                
                namespace = agent.configuration.get('namespace', 'default')
                
                # Get all deployments for this agent
                deployments = apps_v1.list_namespaced_deployment(
                    namespace=namespace,
                    label_selector=f"agent-id={agent.id}"
                )
                
                # Scale all deployments to 0
                for deployment in deployments.items:
                    apps_v1.patch_namespaced_deployment_scale(
                        name=deployment.metadata.name,
                        namespace=namespace,
                        body={'spec': {'replicas': 0}}
                    )
                    logger.info(f"Scaled deployment {deployment.metadata.name} to 0 replicas")
                    
            elif agent.configuration.get('runtime_type') == 'docker':
                import subprocess
                
                # Find all containers for this agent
                result = subprocess.run([
                    'docker', 'ps', '-q', '--filter', f"label=agent-id={agent.id}"
                ], capture_output=True, text=True)
                
                container_ids = result.stdout.strip().split('\n')
                
                # Stop all containers
                for container_id in container_ids:
                    if container_id:
                        subprocess.run(['docker', 'stop', container_id], check=False)
                        logger.info(f"Stopped container {container_id}")
            
            # Update agent status
            agent.configuration['all_replicas_stopped'] = True
            agent.configuration['stop_time'] = datetime.utcnow().isoformat()
            
            logger.info(f"All replicas stopped for agent {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to stop replicas for agent {agent.id}: {str(e)}")
            raise
    
    async def _wait_for_replicas_ready(self, agent: Agent, count: int):
        """Wait for replicas to be ready"""
        await asyncio.sleep(10)
    
    async def _remove_old_replicas(self, agent: Agent, count: int):
        """Remove old replicas during update"""
        try:
            logger.info(f"Removing {count} old replicas for agent {agent.id}")
            
            if agent.configuration.get('runtime_type') == 'kubernetes':
                from kubernetes import client, config as k8s_config
                k8s_config.load_incluster_config()
                v1 = client.CoreV1Api()
                
                namespace = agent.configuration.get('namespace', 'default')
                
                # Get pods with old deployment labels
                pods = v1.list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"agent-id={agent.id},deployment-stage=deprecated"
                )
                
                # Sort by creation time and remove oldest first
                sorted_pods = sorted(
                    pods.items,
                    key=lambda p: p.metadata.creation_timestamp
                )
                
                removed = 0
                for pod in sorted_pods[:count]:
                    try:
                        v1.delete_namespaced_pod(
                            name=pod.metadata.name,
                            namespace=namespace
                        )
                        removed += 1
                        logger.info(f"Removed old pod {pod.metadata.name}")
                    except client.rest.ApiException as e:
                        if e.status != 404:  # Ignore if already deleted
                            logger.warning(f"Failed to delete pod {pod.metadata.name}: {e}")
                            
            elif agent.configuration.get('runtime_type') == 'docker':
                import subprocess
                
                # Get containers with old version labels
                result = subprocess.run([
                    'docker', 'ps', '-a', '-q',
                    '--filter', f"label=agent-id={agent.id}",
                    '--filter', 'label=deployment-stage=deprecated'
                ], capture_output=True, text=True)
                
                container_ids = result.stdout.strip().split('\n')[:count]
                
                removed = 0
                for container_id in container_ids:
                    if container_id:
                        subprocess.run(['docker', 'rm', '-f', container_id], check=False)
                        removed += 1
                        logger.info(f"Removed old container {container_id}")
            
            logger.info(f"Removed {removed} old replicas for agent {agent.id}")
            
        except Exception as e:
            logger.error(f"Failed to remove old replicas for agent {agent.id}: {str(e)}")
            raise
    
    async def _rollback_deployment(self, agent: Agent, deployment_id: str):
        """Rollback a failed deployment"""
        logger.info(f"Rolling back deployment {deployment_id}")
        # Implementation would restore previous state
    
    async def _update_load_balancer_config(self, agent_id: str, config: Dict[str, Any]) -> None:
        """Update load balancer configuration for Docker deployments"""
        try:
            # This would integrate with your load balancer configuration
            # For example, updating nginx upstream or traefik rules
            logger.info(f"Updating load balancer config for agent {agent_id}: {config}")
            
            # Example: Generate nginx upstream configuration
            upstream_config = f"""
            upstream agent_{agent_id} {{
                server {config['target_container']}:{config['port']};
            }}
            """
            
            # Write to nginx config file or API
            config_path = f"/etc/nginx/conf.d/agent_{agent_id}_upstream.conf"
            with open(config_path, 'w') as f:
                f.write(upstream_config)
            
            # Reload nginx (in production, use proper nginx reload)
            import subprocess
            subprocess.run(['nginx', '-s', 'reload'], check=False)
            
        except Exception as e:
            logger.error(f"Failed to update load balancer config: {str(e)}")
    
    async def _configure_istio_traffic_split(self, agent: Agent, weights: Dict[str, float]) -> None:
        """Configure Istio traffic splitting"""
        try:
            # This would create Istio VirtualService for traffic splitting
            virtual_service_config = {
                'apiVersion': 'networking.istio.io/v1alpha3',
                'kind': 'VirtualService',
                'metadata': {
                    'name': f'agent-{agent.id}-traffic-split',
                    'namespace': agent.configuration.get('namespace', 'default')
                },
                'spec': {
                    'hosts': [f'agent-{agent.id}-service'],
                    'http': [{
                        'match': [{'uri': {'prefix': '/'}}],
                        'route': [
                            {
                                'destination': {
                                    'host': f'agent-{agent.id}-service',
                                    'subset': deployment
                                },
                                'weight': int(weight)
                            }
                            for deployment, weight in weights.items()
                        ]
                    }]
                }
            }
            
            # Apply via kubectl or Istio API
            logger.info(f"Configured Istio traffic split: {weights}")
            
        except Exception as e:
            logger.error(f"Failed to configure Istio traffic split: {str(e)}")
    
    async def _configure_lb_traffic_split(self, agent: Agent, weights: Dict[str, float]) -> None:
        """Configure load balancer traffic splitting"""
        try:
            # Configure weighted round-robin or similar
            backend_configs = []
            
            for deployment, weight in weights.items():
                backend_configs.append({
                    'target': f'agent-{agent.id}-{deployment}',
                    'weight': weight,
                    'port': agent.configuration.get('port', 8080)
                })
            
            # Update load balancer configuration
            await self._update_load_balancer_config(str(agent.id), {
                'backends': backend_configs,
                'strategy': 'weighted'
            })
            
        except Exception as e:
            logger.error(f"Failed to configure LB traffic split: {str(e)}")
    
    async def _delayed_cleanup(self, deployment: Dict[str, Any], delay_minutes: int = 10) -> None:
        """Cleanup deployment after delay"""
        try:
            logger.info(f"Scheduled cleanup for deployment {deployment.get('id')} in {delay_minutes} minutes")
            await asyncio.sleep(delay_minutes * 60)  # Convert to seconds
            await self._cleanup_deployment(deployment)
            logger.info(f"Delayed cleanup completed for deployment {deployment.get('id')}")
        except Exception as e:
            logger.error(f"Delayed cleanup failed: {str(e)}")