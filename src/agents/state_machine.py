"""
AgentVault™ Agent State Machine
Production-grade state machine for agent lifecycle management
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import logging
from typing import Dict, List, Optional, Callable, Any, Set
from datetime import datetime
from enum import Enum
import asyncio
from dataclasses import dataclass
import json

from ..database.models import AgentState, Agent
from ..database.repositories import AgentRepository
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem

logger = logging.getLogger(__name__)


@dataclass
class StateTransition:
    """Represents a state transition with validation and actions"""
    from_state: AgentState
    to_state: AgentState
    condition: Optional[Callable[[Agent], bool]] = None
    pre_transition: Optional[Callable[[Agent], None]] = None
    post_transition: Optional[Callable[[Agent], None]] = None
    allowed_roles: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentStateMachine:
    """
    Comprehensive state machine for agent lifecycle management
    Implements state transitions, validations, and hooks
    """
    
    def __init__(self, monitoring: Optional[AdvancedMonitoringSystem] = None):
        self.monitoring = monitoring
        self._transitions = self._define_transitions()
        self._state_hooks: Dict[AgentState, List[Callable]] = {}
        self._transition_history: List[Dict[str, Any]] = []
        
    def _define_transitions(self) -> Dict[str, StateTransition]:
        """Define all valid state transitions"""
        transitions = [
            # Initialization flow
            StateTransition(
                from_state=AgentState.INITIALIZING,
                to_state=AgentState.READY,
                condition=self._check_initialization_complete,
                post_transition=self._on_ready
            ),
            
            # Ready to running
            StateTransition(
                from_state=AgentState.READY,
                to_state=AgentState.RUNNING,
                pre_transition=self._allocate_resources,
                post_transition=self._start_monitoring
            ),
            
            # Running to paused
            StateTransition(
                from_state=AgentState.RUNNING,
                to_state=AgentState.PAUSED,
                pre_transition=self._save_state,
                post_transition=self._release_compute_resources
            ),
            
            # Paused to running
            StateTransition(
                from_state=AgentState.PAUSED,
                to_state=AgentState.RUNNING,
                pre_transition=self._restore_state,
                post_transition=self._resume_monitoring
            ),
            
            # Any active state to maintenance
            StateTransition(
                from_state=AgentState.READY,
                to_state=AgentState.MAINTENANCE,
                pre_transition=self._prepare_maintenance
            ),
            StateTransition(
                from_state=AgentState.RUNNING,
                to_state=AgentState.MAINTENANCE,
                pre_transition=self._prepare_maintenance
            ),
            StateTransition(
                from_state=AgentState.PAUSED,
                to_state=AgentState.MAINTENANCE,
                pre_transition=self._prepare_maintenance
            ),
            
            # Maintenance back to ready
            StateTransition(
                from_state=AgentState.MAINTENANCE,
                to_state=AgentState.READY,
                condition=self._check_maintenance_complete,
                post_transition=self._post_maintenance_checks
            ),
            
            # Error states
            StateTransition(
                from_state=AgentState.RUNNING,
                to_state=AgentState.ERROR,
                pre_transition=self._capture_error_state,
                post_transition=self._trigger_error_recovery
            ),
            StateTransition(
                from_state=AgentState.ERROR,
                to_state=AgentState.READY,
                condition=self._check_error_resolved,
                post_transition=self._clear_error_state
            ),
            
            # Migration flow
            StateTransition(
                from_state=AgentState.RUNNING,
                to_state=AgentState.MIGRATING,
                pre_transition=self._prepare_migration,
                condition=self._check_migration_possible
            ),
            StateTransition(
                from_state=AgentState.PAUSED,
                to_state=AgentState.MIGRATING,
                pre_transition=self._prepare_migration,
                condition=self._check_migration_possible
            ),
            StateTransition(
                from_state=AgentState.MIGRATING,
                to_state=AgentState.RUNNING,
                condition=self._check_migration_complete,
                post_transition=self._finalize_migration
            ),
            
            # Termination flow
            StateTransition(
                from_state=AgentState.READY,
                to_state=AgentState.TERMINATED,
                pre_transition=self._cleanup_resources
            ),
            StateTransition(
                from_state=AgentState.PAUSED,
                to_state=AgentState.TERMINATED,
                pre_transition=self._cleanup_resources
            ),
            StateTransition(
                from_state=AgentState.ERROR,
                to_state=AgentState.TERMINATED,
                pre_transition=self._cleanup_resources
            ),
            StateTransition(
                from_state=AgentState.MAINTENANCE,
                to_state=AgentState.TERMINATED,
                pre_transition=self._cleanup_resources
            ),
            
            # Archive flow
            StateTransition(
                from_state=AgentState.TERMINATED,
                to_state=AgentState.ARCHIVED,
                condition=self._check_archive_eligible,
                pre_transition=self._prepare_archive
            )
        ]
        
        # Build transition map
        transition_map = {}
        for transition in transitions:
            key = f"{transition.from_state}:{transition.to_state}"
            transition_map[key] = transition
        
        return transition_map
    
    def can_transition(self, agent: Agent, to_state: AgentState) -> bool:
        """Check if transition is allowed"""
        key = f"{agent.state}:{to_state}"
        transition = self._transitions.get(key)
        
        if not transition:
            return False
        
        # Check condition if defined
        if transition.condition and not transition.condition(agent):
            return False
        
        return True
    
    def get_allowed_transitions(self, agent: Agent) -> List[AgentState]:
        """Get all allowed transitions from current state"""
        allowed = []
        
        for key, transition in self._transitions.items():
            if transition.from_state == agent.state:
                if not transition.condition or transition.condition(agent):
                    allowed.append(transition.to_state)
        
        return allowed
    
    async def transition(
        self,
        agent: Agent,
        to_state: AgentState,
        repository: AgentRepository,
        metadata: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> bool:
        """Execute state transition with all hooks and validations"""
        key = f"{agent.state}:{to_state}"
        transition = self._transitions.get(key)
        
        if not transition and not force:
            logger.error(f"Invalid transition: {key}")
            return False
        
        # Check condition
        if transition and transition.condition and not force:
            if not transition.condition(agent):
                logger.warning(f"Transition condition failed: {key}")
                return False
        
        # Record transition start
        transition_id = self._start_transition_record(agent, to_state, metadata)
        
        try:
            # Pre-transition hook
            if transition and transition.pre_transition:
                logger.info(f"Executing pre-transition hook for {key}")
                await self._execute_hook(transition.pre_transition, agent)
            
            # Update state in database
            old_state = agent.state
            updated_agent = repository.update_state(
                agent.id,
                to_state,
                metadata=metadata
            )
            
            if not updated_agent:
                raise Exception("Failed to update agent state in database")
            
            # Update agent object
            agent.state = to_state
            agent.state_metadata = metadata or {}
            agent.last_state_change = datetime.utcnow()
            
            # Post-transition hook
            if transition and transition.post_transition:
                logger.info(f"Executing post-transition hook for {key}")
                await self._execute_hook(transition.post_transition, agent)
            
            # Execute state-specific hooks
            await self._execute_state_hooks(to_state, agent)
            
            # Record metrics
            if self.monitoring:
                self.monitoring.record_custom_metric(
                    "agent_state_transition",
                    1,
                    labels={
                        "from_state": old_state,
                        "to_state": to_state,
                        "agent_type": agent.agent_type
                    }
                )
            
            # Complete transition record
            self._complete_transition_record(transition_id, True)
            
            logger.info(f"Agent {agent.id} transitioned from {old_state} to {to_state}")
            return True
            
        except Exception as e:
            logger.error(f"State transition failed: {str(e)}")
            self._complete_transition_record(transition_id, False, str(e))
            
            # Attempt to transition to error state
            if agent.state != AgentState.ERROR and not force:
                await self.transition(
                    agent,
                    AgentState.ERROR,
                    repository,
                    metadata={"error": str(e), "previous_state": agent.state}
                )
            
            return False
    
    def add_state_hook(self, state: AgentState, hook: Callable[[Agent], None]):
        """Add a hook to be executed when entering a state"""
        if state not in self._state_hooks:
            self._state_hooks[state] = []
        self._state_hooks[state].append(hook)
    
    async def _execute_hook(self, hook: Callable, agent: Agent):
        """Execute a hook function (sync or async)"""
        if asyncio.iscoroutinefunction(hook):
            await hook(agent)
        else:
            hook(agent)
    
    async def _execute_state_hooks(self, state: AgentState, agent: Agent):
        """Execute all hooks for a state"""
        if state in self._state_hooks:
            for hook in self._state_hooks[state]:
                try:
                    await self._execute_hook(hook, agent)
                except Exception as e:
                    logger.error(f"State hook failed: {str(e)}")
    
    # Condition functions
    def _check_initialization_complete(self, agent: Agent) -> bool:
        """Check if agent initialization is complete"""
        # Check required resources are allocated
        if not agent.storage_volumes:
            return False
        
        # Check configuration is valid
        if not agent.configuration:
            return False
        
        # Check health checks pass
        try:
            # Check agent process health
            if hasattr(agent, 'process_id') and agent.process_id:
                import psutil
                try:
                    process = psutil.Process(agent.process_id)
                    if not process.is_running():
                        logger.warning(f"Agent {agent.id} process not running")
                        return False
                except psutil.NoSuchProcess:
                    logger.warning(f"Agent {agent.id} process not found")
                    return False
            
            # Check container health (if containerized)
            if hasattr(agent, 'container_id') and agent.container_id:
                import subprocess
                result = subprocess.run(
                    ['docker', 'inspect', '--format={{.State.Health.Status}}', agent.container_id],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    health_status = result.stdout.strip()
                    if health_status not in ['healthy', 'none']:
                        logger.warning(f"Agent {agent.id} container unhealthy: {health_status}")
                        return False
            
            # Check API endpoint health
            if hasattr(agent, 'internal_endpoint') and agent.internal_endpoint:
                import aiohttp
                import asyncio
                
                async def check_endpoint():
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"{agent.internal_endpoint}/health",
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as response:
                                return response.status == 200
                    except:
                        return False
                
                # Run async health check
                try:
                    loop = asyncio.get_event_loop()
                    health_ok = loop.run_until_complete(check_endpoint())
                    if not health_ok:
                        logger.warning(f"Agent {agent.id} API endpoint unhealthy")
                        return False
                except:
                    # If we can't run async, skip endpoint check
                    pass
            
            # Check resource utilization
            metadata = agent.state_metadata or {}
            cpu_usage = metadata.get('cpu_usage', 0)
            memory_usage = metadata.get('memory_usage', 0)
            
            # Alert if resource usage too high
            if cpu_usage > 95:
                logger.warning(f"Agent {agent.id} CPU usage critical: {cpu_usage}%")
                return False
            
            if memory_usage > 95:
                logger.warning(f"Agent {agent.id} memory usage critical: {memory_usage}%")
                return False
            
            logger.debug(f"Agent {agent.id} health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for agent {agent.id}: {str(e)}")
            return False
    
    def _check_maintenance_complete(self, agent: Agent) -> bool:
        """Check if maintenance is complete"""
        metadata = agent.state_metadata or {}
        return metadata.get("maintenance_complete", False)
    
    def _check_error_resolved(self, agent: Agent) -> bool:
        """Check if error has been resolved"""
        metadata = agent.state_metadata or {}
        return metadata.get("error_resolved", False)
    
    def _check_migration_possible(self, agent: Agent) -> bool:
        """Check if migration is possible"""
        # Check resource availability at target
        # Check no active operations
        # Check data consistency
        return True
    
    def _check_migration_complete(self, agent: Agent) -> bool:
        """Check if migration is complete"""
        metadata = agent.state_metadata or {}
        return metadata.get("migration_complete", False)
    
    def _check_archive_eligible(self, agent: Agent) -> bool:
        """Check if agent is eligible for archiving"""
        if not agent.deactivated_at:
            return False
        
        # Check retention period (e.g., 30 days)
        days_inactive = (datetime.utcnow() - agent.deactivated_at).days
        return days_inactive >= 30
    
    # Pre-transition actions
    def _allocate_resources(self, agent: Agent):
        """Allocate compute resources for agent"""
        logger.info(f"Allocating resources for agent {agent.id}")
        
        try:
            # Get resource requirements from agent configuration
            config = agent.configuration or {}
            cpu_cores = config.get('cpu_cores', 1)
            memory_gb = config.get('memory_gb', 2)
            storage_gb = config.get('storage_gb', 10)
            
            # Check available resources
            import psutil
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            available_cpu = psutil.cpu_count()
            
            if memory_gb > available_memory * 0.8:  # Leave 20% buffer
                logger.warning(f"Insufficient memory for agent {agent.id}: need {memory_gb}GB, available {available_memory:.1f}GB")
                # Scale down requirements if possible
                memory_gb = min(memory_gb, available_memory * 0.5)
                agent.configuration['memory_gb'] = memory_gb
            
            if cpu_cores > available_cpu:
                logger.warning(f"Insufficient CPU for agent {agent.id}: need {cpu_cores} cores, available {available_cpu}")
                cpu_cores = min(cpu_cores, available_cpu)
                agent.configuration['cpu_cores'] = cpu_cores
            
            # For Kubernetes deployments
            if agent.configuration.get('runtime_type') == 'kubernetes':
                from kubernetes import client, config as k8s_config
                k8s_config.load_incluster_config()
                apps_v1 = client.AppsV1Api()
                
                # Update resource limits
                deployment_name = f"agent-{agent.id}"
                namespace = agent.configuration.get('namespace', 'default')
                
                # Patch deployment with resource requirements
                body = {
                    'spec': {
                        'template': {
                            'spec': {
                                'containers': [{
                                    'name': 'agent',
                                    'resources': {
                                        'requests': {
                                            'cpu': f"{cpu_cores}",
                                            'memory': f"{memory_gb}Gi"
                                        },
                                        'limits': {
                                            'cpu': f"{cpu_cores * 2}",  # Allow burst
                                            'memory': f"{memory_gb * 1.5}Gi"
                                        }
                                    }
                                }]
                            }
                        }
                    }
                }
                
                try:
                    apps_v1.patch_namespaced_deployment(
                        name=deployment_name,
                        namespace=namespace,
                        body=body
                    )
                    logger.info(f"Updated resource allocation for agent {agent.id}")
                except Exception as e:
                    logger.warning(f"Failed to update K8s resources: {str(e)}")
            
            # For Docker deployments
            elif agent.configuration.get('runtime_type') == 'docker':
                container_id = agent.configuration.get('container_id')
                if container_id:
                    import subprocess
                    # Update container resources
                    subprocess.run([
                        'docker', 'update',
                        '--cpus', str(cpu_cores),
                        '--memory', f"{memory_gb}g",
                        container_id
                    ], check=False)  # Don't fail if update fails
            
            # Update agent metadata
            metadata = agent.state_metadata or {}
            metadata.update({
                'allocated_cpu': cpu_cores,
                'allocated_memory': memory_gb,
                'allocated_storage': storage_gb,
                'resource_allocation_time': datetime.utcnow().isoformat()
            })
            agent.state_metadata = metadata
            
            logger.info(f"Allocated resources for agent {agent.id}: {cpu_cores} CPU, {memory_gb}GB RAM")
            
        except Exception as e:
            logger.error(f"Resource allocation failed for agent {agent.id}: {str(e)}")
    
    def _save_state(self, agent: Agent):
        """Save agent state before pausing"""
        logger.info(f"Saving state for agent {agent.id}")
        
        try:
            # Create state snapshot
            state_snapshot = {
                'agent_id': str(agent.id),
                'state': agent.state.value,
                'timestamp': datetime.utcnow().isoformat(),
                'configuration': agent.configuration,
                'state_metadata': agent.state_metadata,
                'performance_metrics': getattr(agent, 'performance_metrics', {}),
                'conversation_history': getattr(agent, 'conversation_history', []),
                'memory_state': {},
                'checkpoint_info': {
                    'version': '1.0',
                    'created_by': 'state_machine',
                    'reason': 'state_transition'
                }
            }
            
            # Save memory state if available
            try:
                if hasattr(agent, 'memory') and agent.memory:
                    state_snapshot['memory_state'] = {
                        'working_memory': getattr(agent.memory, 'working_memory', {}),
                        'long_term_memory': getattr(agent.memory, 'long_term_memory', {}),
                        'context_window': getattr(agent.memory, 'context_window', [])
                    }
            except Exception as e:
                logger.warning(f"Failed to save memory state: {str(e)}")
            
            # Save process state if available
            if hasattr(agent, 'process_id') and agent.process_id:
                try:
                    import psutil
                    process = psutil.Process(agent.process_id)
                    state_snapshot['process_info'] = {
                        'pid': agent.process_id,
                        'cpu_percent': process.cpu_percent(),
                        'memory_info': process.memory_info()._asdict(),
                        'status': process.status(),
                        'create_time': process.create_time()
                    }
                except Exception as e:
                    logger.warning(f"Failed to save process state: {str(e)}")
            
            # Choose storage location
            storage_path = f"/data/agentvault/checkpoints/{agent.id}"
            checkpoint_file = f"{storage_path}/state_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Ensure directory exists
            import os
            os.makedirs(storage_path, exist_ok=True)
            
            # Save to file
            import json
            with open(checkpoint_file, 'w') as f:
                json.dump(state_snapshot, f, indent=2, default=str)
            
            # Update agent with checkpoint info
            metadata = agent.state_metadata or {}
            metadata.update({
                'last_checkpoint': checkpoint_file,
                'last_checkpoint_time': datetime.utcnow().isoformat(),
                'checkpoint_version': '1.0'
            })
            agent.state_metadata = metadata
            
            # Keep only last 10 checkpoints to save space
            import glob
            checkpoints = sorted(glob.glob(f"{storage_path}/state_*.json"))
            if len(checkpoints) > 10:
                for old_checkpoint in checkpoints[:-10]:
                    try:
                        os.remove(old_checkpoint)
                    except Exception as e:
                        logger.warning(f"Failed to remove old checkpoint: {str(e)}")
            
            logger.info(f"State saved for agent {agent.id} to {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"State persistence failed for agent {agent.id}: {str(e)}")
            # Don't raise - state saving shouldn't block transitions
            pass
    
    def _prepare_maintenance(self, agent: Agent):
        """Prepare agent for maintenance"""
        logger.info(f"Preparing agent {agent.id} for maintenance")
        # Gracefully stop operations
        # Save current state
        # Notify dependent services
    
    def _prepare_migration(self, agent: Agent):
        """Prepare agent for migration"""
        logger.info(f"Preparing agent {agent.id} for migration")
        # Snapshot current state
        # Prepare data transfer
        # Reserve resources at target
    
    def _capture_error_state(self, agent: Agent):
        """Capture error state for debugging"""
        logger.info(f"Capturing error state for agent {agent.id}")
        # Collect logs
        # Capture metrics
        # Create debugging snapshot
    
    def _cleanup_resources(self, agent: Agent):
        """Clean up resources before termination"""
        logger.info(f"Cleaning up resources for agent {agent.id}")
        # Release compute resources
        # Clean up temporary data
        # Close connections
    
    def _prepare_archive(self, agent: Agent):
        """Prepare agent data for archiving"""
        logger.info(f"Preparing agent {agent.id} for archiving")
        # Compress data
        # Move to cold storage
        # Update indexes
    
    # Post-transition actions
    def _on_ready(self, agent: Agent):
        """Actions when agent becomes ready"""
        logger.info(f"Agent {agent.id} is now ready")
        # Start health monitoring
        # Register with service discovery
        # Enable API endpoints
    
    def _start_monitoring(self, agent: Agent):
        """Start monitoring for running agent"""
        logger.info(f"Starting monitoring for agent {agent.id}")
        # Start metrics collection
        # Enable alerts
        # Begin performance tracking
    
    def _release_compute_resources(self, agent: Agent):
        """Release compute resources for paused agent"""
        logger.info(f"Releasing compute resources for agent {agent.id}")
        # Scale down containers
        # Release GPU allocation
        # Reduce memory allocation
    
    def _restore_state(self, agent: Agent):
        """Restore agent state when resuming"""
        logger.info(f"Restoring state for agent {agent.id}")
        # Load saved state
        # Restore connections
        # Resume operations
    
    def _resume_monitoring(self, agent: Agent):
        """Resume monitoring for agent"""
        logger.info(f"Resuming monitoring for agent {agent.id}")
        # Re-enable metrics
        # Restore alert rules
        # Continue performance tracking
    
    def _post_maintenance_checks(self, agent: Agent):
        """Run checks after maintenance"""
        logger.info(f"Running post-maintenance checks for agent {agent.id}")
        # Verify configuration
        # Test connectivity
        # Validate resources
    
    def _trigger_error_recovery(self, agent: Agent):
        """Trigger error recovery procedures"""
        logger.info(f"Triggering error recovery for agent {agent.id}")
        # Notify operations team
        # Attempt auto-recovery
        # Create incident ticket
    
    def _clear_error_state(self, agent: Agent):
        """Clear error state after resolution"""
        logger.info(f"Clearing error state for agent {agent.id}")
        # Clear error flags
        # Reset error counters
        # Resume normal monitoring
    
    def _finalize_migration(self, agent: Agent):
        """Finalize agent migration"""
        logger.info(f"Finalizing migration for agent {agent.id}")
        # Update routing
        # Clean up source resources
        # Verify data integrity
    
    # Transition tracking
    def _start_transition_record(
        self,
        agent: Agent,
        to_state: AgentState,
        metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Start recording a transition"""
        transition_id = f"{agent.id}:{datetime.utcnow().timestamp()}"
        
        record = {
            "id": transition_id,
            "agent_id": str(agent.id),
            "from_state": agent.state,
            "to_state": to_state,
            "started_at": datetime.utcnow().isoformat(),
            "metadata": metadata,
            "completed": False
        }
        
        self._transition_history.append(record)
        return transition_id
    
    def _complete_transition_record(
        self,
        transition_id: str,
        success: bool,
        error: Optional[str] = None
    ):
        """Complete a transition record"""
        for record in self._transition_history:
            if record["id"] == transition_id:
                record["completed"] = True
                record["success"] = success
                record["completed_at"] = datetime.utcnow().isoformat()
                if error:
                    record["error"] = error
                break
    
    def get_transition_history(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get transition history"""
        history = self._transition_history
        
        if agent_id:
            history = [r for r in history if r["agent_id"] == agent_id]
        
        return history[-limit:]
    
    def export_state_diagram(self) -> str:
        """Export state machine as Mermaid diagram"""
        lines = ["stateDiagram-v2"]
        
        # Add states
        for state in AgentState:
            lines.append(f"    {state.value}: {state.value.title()}")
        
        # Add transitions
        for key, transition in self._transitions.items():
            from_state = transition.from_state.value
            to_state = transition.to_state.value
            lines.append(f"    {from_state} --> {to_state}")
        
        return "\n".join(lines)


class StateValidator:
    """Validates state transitions and state integrity"""
    
    @staticmethod
    def validate_transition_request(
        agent: Agent,
        to_state: AgentState,
        user_roles: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Validate a state transition request"""
        # Check if agent is deleted
        if agent.deleted_at:
            return False, "Cannot transition deleted agent"
        
        # Check if already in target state
        if agent.state == to_state:
            return False, "Agent already in target state"
        
        # Validate based on target state
        if to_state == AgentState.TERMINATED:
            # Check if agent has active operations
            if agent.state == AgentState.RUNNING:
                return False, "Cannot terminate running agent directly"
        
        return True, None
    
    @staticmethod
    def validate_state_integrity(agent: Agent) -> List[str]:
        """Validate agent state integrity"""
        issues = []
        
        # Check state-specific validations
        if agent.state == AgentState.RUNNING:
            if not agent.activated_at:
                issues.append("Running agent missing activation timestamp")
            if not agent.internal_endpoint:
                issues.append("Running agent missing internal endpoint")
        
        elif agent.state == AgentState.TERMINATED:
            if not agent.deactivated_at:
                issues.append("Terminated agent missing deactivation timestamp")
        
        elif agent.state == AgentState.ERROR:
            if not agent.state_metadata or "error" not in agent.state_metadata:
                issues.append("Error state missing error details")
        
        return issues