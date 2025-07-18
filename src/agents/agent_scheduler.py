"""
AgentVault™ Agent Scheduler
Task scheduling and workflow orchestration for agents
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import croniter
import uuid
from collections import defaultdict
import heapq

from ..database.models import Agent, AgentState
from ..cache.distributed_cache import DistributedCache
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRY = "retry"


class TaskPriority(int, Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


class ScheduleType(str, Enum):
    """Types of scheduling"""
    IMMEDIATE = "immediate"
    DELAYED = "delayed"
    CRON = "cron"
    INTERVAL = "interval"
    EVENT_TRIGGERED = "event_triggered"
    DEPENDENCY_BASED = "dependency_based"


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    agent_id: Optional[str] = None
    task_type: str = "generic"
    
    # Scheduling
    schedule_type: ScheduleType = ScheduleType.IMMEDIATE
    scheduled_time: Optional[datetime] = None
    cron_expression: Optional[str] = None
    interval_seconds: Optional[int] = None
    
    # Execution
    function: Optional[Callable] = None
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: int = 300  # 5 minutes
    max_retries: int = 3
    retry_delay_seconds: int = 60
    
    # State
    status: TaskStatus = TaskStatus.PENDING
    retry_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    next_run: Optional[datetime] = None
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Task IDs
    triggers: List[str] = field(default_factory=list)    # Event patterns
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """For priority queue comparison"""
        if not isinstance(other, ScheduledTask):
            return NotImplemented
        # Higher priority value = higher priority
        return self.priority.value < other.priority.value


@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """Represents a workflow of tasks"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Tasks in workflow
    tasks: List[ScheduledTask] = field(default_factory=list)
    task_graph: Dict[str, List[str]] = field(default_factory=dict)  # Dependencies
    
    # Configuration
    parallel_execution: bool = True
    continue_on_failure: bool = False
    timeout_seconds: int = 3600  # 1 hour
    
    # State
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    task_results: Dict[str, TaskResult] = field(default_factory=dict)


class AgentScheduler:
    """
    Manages task scheduling and execution for agents
    """
    
    def __init__(
        self,
        agent_manager: Any,  # Avoid circular import
        cache: DistributedCache,
        monitoring: Optional[AdvancedMonitoringSystem] = None
    ):
        self.agent_manager = agent_manager
        self.cache = cache
        self.monitoring = monitoring
        
        # Task storage
        self._tasks: Dict[str, ScheduledTask] = {}
        self._task_queue: List[ScheduledTask] = []  # Priority queue
        self._recurring_tasks: Dict[str, ScheduledTask] = {}
        
        # Workflow storage
        self._workflows: Dict[str, Workflow] = {}
        self._workflow_tasks: Dict[str, str] = {}  # task_id -> workflow_id
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Execution tracking
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, TaskResult] = {}
        
        # Background workers
        self._worker_tasks: List[asyncio.Task] = []
        self._num_workers = 4
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self):
        """Initialize scheduler"""
        logger.info("Initializing Agent Scheduler")
        
        # Start worker tasks
        for i in range(self._num_workers):
            self._worker_tasks.append(
                asyncio.create_task(self._worker_loop(i))
            )
        
        # Start scheduler loop
        self._worker_tasks.append(
            asyncio.create_task(self._scheduler_loop())
        )
        
        # Start cleanup loop
        self._worker_tasks.append(
            asyncio.create_task(self._cleanup_loop())
        )
        
        logger.info(f"Agent Scheduler initialized with {self._num_workers} workers")
    
    async def shutdown(self):
        """Shutdown scheduler"""
        logger.info("Shutting down Agent Scheduler")
        
        self._shutdown_event.set()
        
        # Cancel running tasks
        for task_id, task in self._running_tasks.items():
            task.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        logger.info("Agent Scheduler shutdown complete")
    
    async def schedule_task(self, task: ScheduledTask) -> str:
        """Schedule a task for execution"""
        logger.info(f"Scheduling task {task.id}: {task.name}")
        
        # Validate task
        if not task.name:
            task.name = f"task_{task.id}"
        
        # Calculate next run time
        if task.schedule_type == ScheduleType.IMMEDIATE:
            task.next_run = datetime.utcnow()
        elif task.schedule_type == ScheduleType.DELAYED:
            if not task.scheduled_time:
                raise ValueError("Delayed task requires scheduled_time")
            task.next_run = task.scheduled_time
        elif task.schedule_type == ScheduleType.CRON:
            if not task.cron_expression:
                raise ValueError("Cron task requires cron_expression")
            cron = croniter.croniter(task.cron_expression, datetime.utcnow())
            task.next_run = cron.get_next(datetime)
        elif task.schedule_type == ScheduleType.INTERVAL:
            if not task.interval_seconds:
                raise ValueError("Interval task requires interval_seconds")
            task.next_run = datetime.utcnow() + timedelta(seconds=task.interval_seconds)
        
        # Store task
        self._tasks[task.id] = task
        
        # Add to appropriate queue
        if task.schedule_type in [ScheduleType.CRON, ScheduleType.INTERVAL]:
            self._recurring_tasks[task.id] = task
        
        # Add to task queue if ready
        if task.next_run and task.next_run <= datetime.utcnow() + timedelta(seconds=1):
            heapq.heappush(self._task_queue, (-task.priority.value, task.id, task))
        
        # Persist task
        await self._persist_task(task)
        
        # Record metric
        if self.monitoring:
            self.monitoring.record_custom_metric(
                "task_scheduled",
                1,
                labels={
                    'task_type': task.task_type,
                    'schedule_type': task.schedule_type,
                    'agent_id': task.agent_id or 'system'
                }
            )
        
        return task.id
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        logger.info(f"Cancelling task {task_id}")
        
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        # Update status
        task.status = TaskStatus.CANCELLED
        
        # Cancel if running
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
        
        # Remove from queues
        self._recurring_tasks.pop(task_id, None)
        
        # Persist cancellation
        await self._persist_task(task)
        
        return True
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get task execution status"""
        # Check results first
        if task_id in self._task_results:
            return self._task_results[task_id]
        
        # Check running tasks
        task = self._tasks.get(task_id)
        if task:
            return TaskResult(
                task_id=task_id,
                status=task.status,
                started_at=task.started_at,
                metadata=task.metadata
            )
        
        return None
    
    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None
    ) -> Optional[TaskResult]:
        """Wait for task completion"""
        start_time = datetime.utcnow()
        
        while True:
            result = await self.get_task_status(task_id)
            
            if result and result.status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.TIMEOUT
            ]:
                return result
            
            # Check timeout
            if timeout:
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                if elapsed >= timeout:
                    return None
            
            await asyncio.sleep(1)
    
    async def create_workflow(self, workflow: Workflow) -> str:
        """Create a workflow of tasks"""
        logger.info(f"Creating workflow {workflow.id}: {workflow.name}")
        
        # Validate workflow
        self._validate_workflow(workflow)
        
        # Store workflow
        self._workflows[workflow.id] = workflow
        
        # Map tasks to workflow
        for task in workflow.tasks:
            self._workflow_tasks[task.id] = workflow.id
        
        # Persist workflow
        await self._persist_workflow(workflow)
        
        return workflow.id
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, TaskResult]:
        """Execute a workflow"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        logger.info(f"Executing workflow {workflow_id}")
        
        workflow.status = TaskStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        
        try:
            # Execute tasks based on dependencies
            results = await self._execute_workflow_tasks(workflow)
            
            # Update workflow status
            workflow.task_results = results
            workflow.status = TaskStatus.COMPLETED
            workflow.completed_at = datetime.utcnow()
            
            # Check if all tasks succeeded
            if not workflow.continue_on_failure:
                for result in results.values():
                    if result.status != TaskStatus.COMPLETED:
                        workflow.status = TaskStatus.FAILED
                        break
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            workflow.status = TaskStatus.FAILED
            workflow.completed_at = datetime.utcnow()
            raise
        
        # Persist final state
        await self._persist_workflow(workflow)
        
        return workflow.task_results
    
    async def schedule_recurring_task(
        self,
        name: str,
        function: Callable,
        cron_expression: Optional[str] = None,
        interval_seconds: Optional[int] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Schedule a recurring task"""
        if cron_expression:
            schedule_type = ScheduleType.CRON
        elif interval_seconds:
            schedule_type = ScheduleType.INTERVAL
        else:
            raise ValueError("Either cron_expression or interval_seconds required")
        
        task = ScheduledTask(
            name=name,
            agent_id=agent_id,
            schedule_type=schedule_type,
            cron_expression=cron_expression,
            interval_seconds=interval_seconds,
            function=function,
            kwargs=kwargs
        )
        
        return await self.schedule_task(task)
    
    def register_event_handler(
        self,
        event_pattern: str,
        handler: Callable[[Dict[str, Any]], Any]
    ):
        """Register handler for event-triggered tasks"""
        self._event_handlers[event_pattern].append(handler)
    
    async def trigger_event(self, event_name: str, data: Dict[str, Any]):
        """Trigger event-based tasks"""
        logger.debug(f"Triggering event: {event_name}")
        
        # Find matching handlers
        for pattern, handlers in self._event_handlers.items():
            if self._matches_event_pattern(event_name, pattern):
                for handler in handlers:
                    # Create task for handler
                    task = ScheduledTask(
                        name=f"event_{event_name}_{uuid.uuid4().hex[:8]}",
                        task_type='event_handler',
                        schedule_type=ScheduleType.EVENT_TRIGGERED,
                        function=handler,
                        args=[data],
                        metadata={'event': event_name}
                    )
                    
                    await self.schedule_task(task)
    
    # Private methods
    
    async def _worker_loop(self, worker_id: int):
        """Worker loop for executing tasks"""
        logger.info(f"Worker {worker_id} started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get next task from queue
                if self._task_queue:
                    _, task_id, task = heapq.heappop(self._task_queue)
                    
                    # Execute task
                    await self._execute_task(task)
                else:
                    # No tasks, wait briefly
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {str(e)}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop for managing task scheduling"""
        logger.info("Scheduler loop started")
        
        while not self._shutdown_event.is_set():
            try:
                now = datetime.utcnow()
                
                # Check scheduled tasks
                for task_id, task in list(self._tasks.items()):
                    if task.status == TaskStatus.PENDING and task.next_run:
                        if task.next_run <= now:
                            # Task is ready to run
                            heapq.heappush(
                                self._task_queue,
                                (-task.priority.value, task_id, task)
                            )
                            task.status = TaskStatus.SCHEDULED
                
                # Check recurring tasks
                for task_id, task in self._recurring_tasks.items():
                    if task.status == TaskStatus.COMPLETED:
                        # Calculate next run
                        if task.schedule_type == ScheduleType.CRON:
                            cron = croniter.croniter(task.cron_expression, task.completed_at or now)
                            task.next_run = cron.get_next(datetime)
                        elif task.schedule_type == ScheduleType.INTERVAL:
                            task.next_run = (task.completed_at or now) + timedelta(seconds=task.interval_seconds)
                        
                        # Reset for next run
                        task.status = TaskStatus.PENDING
                        task.retry_count = 0
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {str(e)}")
        
        logger.info("Scheduler loop stopped")
    
    async def _cleanup_loop(self):
        """Cleanup old tasks and results"""
        while not self._shutdown_event.is_set():
            try:
                cutoff = datetime.utcnow() - timedelta(hours=24)
                
                # Clean completed tasks
                to_remove = []
                for task_id, task in self._tasks.items():
                    if task.completed_at and task.completed_at < cutoff:
                        if task_id not in self._recurring_tasks:
                            to_remove.append(task_id)
                
                for task_id in to_remove:
                    del self._tasks[task_id]
                    self._task_results.pop(task_id, None)
                
                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old tasks")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
            
            await asyncio.sleep(3600)  # Clean every hour
    
    async def _execute_task(self, task: ScheduledTask):
        """Execute a single task"""
        logger.info(f"Executing task {task.id}: {task.name}")
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        
        # Create async task
        async_task = asyncio.create_task(
            self._run_task_with_timeout(task)
        )
        self._running_tasks[task.id] = async_task
        
        try:
            # Wait for completion
            result = await async_task
            
            # Update task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.result = result
            
            # Store result
            self._task_results[task.id] = TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                started_at=task.started_at,
                completed_at=task.completed_at,
                duration_seconds=(task.completed_at - task.started_at).total_seconds()
            )
            
            # Record metric
            if self.monitoring:
                self.monitoring.record_custom_metric(
                    "task_completed",
                    1,
                    labels={
                        'task_type': task.task_type,
                        'agent_id': task.agent_id or 'system'
                    }
                )
            
        except asyncio.TimeoutError:
            logger.error(f"Task {task.id} timed out")
            task.status = TaskStatus.TIMEOUT
            task.error = "Task execution timed out"
            
            # Handle retry
            if task.retry_count < task.max_retries:
                await self._retry_task(task)
            
        except Exception as e:
            logger.error(f"Task {task.id} failed: {str(e)}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            # Handle retry
            if task.retry_count < task.max_retries:
                await self._retry_task(task)
            
            # Store failed result
            self._task_results[task.id] = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                started_at=task.started_at,
                completed_at=datetime.utcnow()
            )
            
        finally:
            # Clean up
            self._running_tasks.pop(task.id, None)
            
            # Persist task state
            await self._persist_task(task)
            
            # Notify workflow if part of one
            if task.id in self._workflow_tasks:
                workflow_id = self._workflow_tasks[task.id]
                workflow = self._workflows.get(workflow_id)
                if workflow:
                    workflow.task_results[task.id] = self._task_results.get(task.id)
    
    async def _run_task_with_timeout(self, task: ScheduledTask) -> Any:
        """Run task with timeout"""
        if task.function:
            # Direct function execution
            if asyncio.iscoroutinefunction(task.function):
                return await asyncio.wait_for(
                    task.function(*task.args, **task.kwargs),
                    timeout=task.timeout_seconds
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        task.function,
                        *task.args,
                        **task.kwargs
                    ),
                    timeout=task.timeout_seconds
                )
        elif task.agent_id:
            # Execute on agent
            return await asyncio.wait_for(
                self.agent_manager.execute_agent_command(
                    task.agent_id,
                    task.task_type,
                    task.kwargs
                ),
                timeout=task.timeout_seconds
            )
        else:
            raise ValueError("Task must have either function or agent_id")
    
    async def _retry_task(self, task: ScheduledTask):
        """Retry a failed task"""
        task.retry_count += 1
        task.status = TaskStatus.RETRY
        
        # Schedule retry
        task.next_run = datetime.utcnow() + timedelta(seconds=task.retry_delay_seconds)
        
        logger.info(
            f"Scheduling retry {task.retry_count}/{task.max_retries} "
            f"for task {task.id} at {task.next_run}"
        )
    
    def _validate_workflow(self, workflow: Workflow):
        """Validate workflow structure"""
        # Check for cycles in task graph
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            for neighbor in workflow.task_graph.get(task_id, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(task_id)
            return False
        
        # Check each component
        for task in workflow.tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    raise ValueError("Workflow contains circular dependencies")
    
    async def _execute_workflow_tasks(self, workflow: Workflow) -> Dict[str, TaskResult]:
        """Execute workflow tasks respecting dependencies"""
        results = {}
        completed_tasks = set()
        
        async def can_execute(task: ScheduledTask) -> bool:
            """Check if task dependencies are met"""
            for dep_id in task.depends_on:
                if dep_id not in completed_tasks:
                    return False
                if not workflow.continue_on_failure:
                    if results[dep_id].status != TaskStatus.COMPLETED:
                        return False
            return True
        
        # Execute tasks
        while len(completed_tasks) < len(workflow.tasks):
            # Find tasks ready to execute
            ready_tasks = []
            for task in workflow.tasks:
                if task.id not in completed_tasks and await can_execute(task):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # No tasks ready - check for deadlock
                if len(completed_tasks) < len(workflow.tasks):
                    raise RuntimeError("Workflow deadlock detected")
                break
            
            # Execute ready tasks
            if workflow.parallel_execution:
                # Execute in parallel
                task_futures = []
                for task in ready_tasks:
                    task_id = await self.schedule_task(task)
                    task_futures.append(self.wait_for_task(task_id))
                
                task_results = await asyncio.gather(*task_futures)
                
                for task, result in zip(ready_tasks, task_results):
                    results[task.id] = result
                    completed_tasks.add(task.id)
            else:
                # Execute sequentially
                for task in ready_tasks:
                    task_id = await self.schedule_task(task)
                    result = await self.wait_for_task(task_id)
                    results[task.id] = result
                    completed_tasks.add(task.id)
                    
                    if not workflow.continue_on_failure and result.status != TaskStatus.COMPLETED:
                        break
        
        return results
    
    def _matches_event_pattern(self, event_name: str, pattern: str) -> bool:
        """Check if event name matches pattern"""
        # Simple wildcard matching
        if pattern == '*':
            return True
        
        pattern_parts = pattern.split('.')
        event_parts = event_name.split('.')
        
        if len(pattern_parts) != len(event_parts):
            return False
        
        for p, e in zip(pattern_parts, event_parts):
            if p != '*' and p != e:
                return False
        
        return True
    
    async def _persist_task(self, task: ScheduledTask):
        """Persist task to cache/storage"""
        await self.cache.set(
            f"task:{task.id}",
            task.__dict__,
            ttl=86400  # 24 hours
        )
    
    async def _persist_workflow(self, workflow: Workflow):
        """Persist workflow to cache/storage"""
        await self.cache.set(
            f"workflow:{workflow.id}",
            workflow.__dict__,
            ttl=86400 * 7  # 7 days
        )