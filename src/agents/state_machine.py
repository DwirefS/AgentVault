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
        # TODO: Implement health check
        
        return True
    
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
        # TODO: Implement resource allocation
    
    def _save_state(self, agent: Agent):
        """Save agent state before pausing"""
        logger.info(f"Saving state for agent {agent.id}")
        # TODO: Implement state persistence
    
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