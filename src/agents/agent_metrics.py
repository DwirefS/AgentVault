"""
AgentVault™ Agent Metrics Collector
Comprehensive metrics collection and analysis for agents
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

import psutil
import aiohttp
from prometheus_client import Counter, Gauge, Histogram, Summary

from ..database.models import Agent, PerformanceMetric
from ..database.repositories import MetricsRepository
from ..cache.distributed_cache import DistributedCache
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricCategory(str, Enum):
    """Categories of metrics"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    HEALTH = "health"
    RESOURCE = "resource"
    NETWORK = "network"
    STORAGE = "storage"
    CUSTOM = "custom"


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    type: MetricType
    category: MetricCategory
    description: str
    unit: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[List[float]] = None  # For summaries


@dataclass
class MetricSnapshot:
    """Point-in-time snapshot of metrics"""
    timestamp: datetime
    agent_id: str
    metrics: Dict[str, Any]
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0
    
    # Performance metrics
    request_rate: float = 0.0
    error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    
    # Agent-specific metrics
    active_tasks: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0


class AgentMetricsCollector:
    """
    Collects and manages metrics for agents
    """
    
    def __init__(
        self,
        agent_manager: Any,  # Avoid circular import
        metrics_repo: Optional[MetricsRepository] = None,
        cache: Optional[DistributedCache] = None,
        monitoring: Optional[AdvancedMonitoringSystem] = None
    ):
        self.agent_manager = agent_manager
        self.metrics_repo = metrics_repo
        self.cache = cache
        self.monitoring = monitoring
        
        # Metric definitions
        self._metric_definitions: Dict[str, MetricDefinition] = {}
        self._register_default_metrics()
        
        # Prometheus metrics
        self._prometheus_metrics: Dict[str, Any] = {}
        self._initialize_prometheus_metrics()
        
        # Collection state
        self._collection_intervals: Dict[str, int] = {
            MetricCategory.SYSTEM: 30,      # 30 seconds
            MetricCategory.PERFORMANCE: 60,  # 1 minute
            MetricCategory.BUSINESS: 300,    # 5 minutes
            MetricCategory.HEALTH: 60,       # 1 minute
            MetricCategory.RESOURCE: 120,    # 2 minutes
            MetricCategory.NETWORK: 60,      # 1 minute
            MetricCategory.STORAGE: 300,     # 5 minutes
            MetricCategory.CUSTOM: 60        # 1 minute
        }
        
        # Metric buffers for aggregation
        self._metric_buffers: Dict[str, List[Tuple[datetime, float]]] = {}
        self._buffer_size = 100
        
        # Background tasks
        self._collector_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
    
    def _register_default_metrics(self):
        """Register default metric definitions"""
        default_metrics = [
            # System metrics
            MetricDefinition(
                name="agent_cpu_usage",
                type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                description="CPU usage percentage",
                unit="percent",
                labels=["agent_id", "agent_type"]
            ),
            MetricDefinition(
                name="agent_memory_usage",
                type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                description="Memory usage percentage",
                unit="percent",
                labels=["agent_id", "agent_type"]
            ),
            MetricDefinition(
                name="agent_disk_usage",
                type=MetricType.GAUGE,
                category=MetricCategory.SYSTEM,
                description="Disk usage percentage",
                unit="percent",
                labels=["agent_id", "agent_type"]
            ),
            
            # Performance metrics
            MetricDefinition(
                name="agent_request_total",
                type=MetricType.COUNTER,
                category=MetricCategory.PERFORMANCE,
                description="Total number of requests",
                labels=["agent_id", "agent_type", "status"]
            ),
            MetricDefinition(
                name="agent_request_duration",
                type=MetricType.HISTOGRAM,
                category=MetricCategory.PERFORMANCE,
                description="Request duration in milliseconds",
                unit="milliseconds",
                labels=["agent_id", "agent_type", "operation"],
                buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
            ),
            MetricDefinition(
                name="agent_error_rate",
                type=MetricType.GAUGE,
                category=MetricCategory.PERFORMANCE,
                description="Error rate percentage",
                unit="percent",
                labels=["agent_id", "agent_type"]
            ),
            
            # Business metrics
            MetricDefinition(
                name="agent_task_completed",
                type=MetricType.COUNTER,
                category=MetricCategory.BUSINESS,
                description="Number of completed tasks",
                labels=["agent_id", "agent_type", "task_type"]
            ),
            MetricDefinition(
                name="agent_task_queue_length",
                type=MetricType.GAUGE,
                category=MetricCategory.BUSINESS,
                description="Current task queue length",
                labels=["agent_id", "agent_type"]
            ),
            
            # Health metrics
            MetricDefinition(
                name="agent_health_score",
                type=MetricType.GAUGE,
                category=MetricCategory.HEALTH,
                description="Overall health score (0-100)",
                labels=["agent_id", "agent_type"]
            ),
            MetricDefinition(
                name="agent_uptime_seconds",
                type=MetricType.COUNTER,
                category=MetricCategory.HEALTH,
                description="Agent uptime in seconds",
                labels=["agent_id", "agent_type"]
            ),
            
            # Resource metrics
            MetricDefinition(
                name="agent_resource_allocation",
                type=MetricType.GAUGE,
                category=MetricCategory.RESOURCE,
                description="Resource allocation",
                unit="cores",
                labels=["agent_id", "agent_type", "resource_type"]
            ),
            
            # Network metrics
            MetricDefinition(
                name="agent_network_bytes",
                type=MetricType.COUNTER,
                category=MetricCategory.NETWORK,
                description="Network bytes transferred",
                unit="bytes",
                labels=["agent_id", "agent_type", "direction"]
            ),
            MetricDefinition(
                name="agent_network_connections",
                type=MetricType.GAUGE,
                category=MetricCategory.NETWORK,
                description="Active network connections",
                labels=["agent_id", "agent_type", "state"]
            ),
            
            # Storage metrics
            MetricDefinition(
                name="agent_storage_used",
                type=MetricType.GAUGE,
                category=MetricCategory.STORAGE,
                description="Storage space used",
                unit="bytes",
                labels=["agent_id", "agent_type", "volume"]
            ),
            MetricDefinition(
                name="agent_storage_iops",
                type=MetricType.GAUGE,
                category=MetricCategory.STORAGE,
                description="Storage IOPS",
                labels=["agent_id", "agent_type", "operation"]
            )
        ]
        
        for metric in default_metrics:
            self.register_metric(metric)
    
    def _initialize_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        for name, definition in self._metric_definitions.items():
            if definition.type == MetricType.COUNTER:
                self._prometheus_metrics[name] = Counter(
                    name,
                    definition.description,
                    definition.labels
                )
            elif definition.type == MetricType.GAUGE:
                self._prometheus_metrics[name] = Gauge(
                    name,
                    definition.description,
                    definition.labels
                )
            elif definition.type == MetricType.HISTOGRAM:
                self._prometheus_metrics[name] = Histogram(
                    name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets or Histogram.DEFAULT_BUCKETS
                )
            elif definition.type == MetricType.SUMMARY:
                self._prometheus_metrics[name] = Summary(
                    name,
                    definition.description,
                    definition.labels
                )
    
    def register_metric(self, definition: MetricDefinition):
        """Register a metric definition"""
        self._metric_definitions[definition.name] = definition
        
        # Initialize Prometheus metric
        if definition.name not in self._prometheus_metrics:
            if definition.type == MetricType.COUNTER:
                metric = Counter(
                    definition.name,
                    definition.description,
                    definition.labels
                )
            elif definition.type == MetricType.GAUGE:
                metric = Gauge(
                    definition.name,
                    definition.description,
                    definition.labels
                )
            elif definition.type == MetricType.HISTOGRAM:
                metric = Histogram(
                    definition.name,
                    definition.description,
                    definition.labels,
                    buckets=definition.buckets or Histogram.DEFAULT_BUCKETS
                )
            elif definition.type == MetricType.SUMMARY:
                metric = Summary(
                    definition.name,
                    definition.description,
                    definition.labels
                )
            
            self._prometheus_metrics[definition.name] = metric
    
    async def start_collection(self, agent_id: str):
        """Start metric collection for an agent"""
        logger.info(f"Starting metric collection for agent {agent_id}")
        
        # Start collection tasks for each category
        for category in MetricCategory:
            if category not in self._collector_tasks:
                self._collector_tasks[category] = asyncio.create_task(
                    self._collection_loop(agent_id, category)
                )
    
    async def stop_collection(self, agent_id: str):
        """Stop metric collection for an agent"""
        logger.info(f"Stopping metric collection for agent {agent_id}")
        
        # Cancel collection tasks
        for task in self._collector_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._collector_tasks.values(), return_exceptions=True)
        
        self._collector_tasks.clear()
    
    async def collect_all_metrics(self):
        """Collect metrics for all active agents"""
        agents = self.agent_manager.agent_repo.get_active_agents(
            tenant_id=None,  # All tenants
            limit=1000
        )
        
        tasks = []
        for agent in agents:
            tasks.append(self.collect_agent_metrics(agent.id))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def collect_agent_metrics(self, agent_id: str) -> MetricSnapshot:
        """Collect current metrics for an agent"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        snapshot = MetricSnapshot(
            timestamp=datetime.utcnow(),
            agent_id=str(agent_id),
            metrics={}
        )
        
        # Collect system metrics
        system_metrics = await self._collect_system_metrics(agent)
        snapshot.cpu_usage = system_metrics.get('cpu_usage', 0)
        snapshot.memory_usage = system_metrics.get('memory_usage', 0)
        snapshot.disk_usage = system_metrics.get('disk_usage', 0)
        snapshot.metrics.update(system_metrics)
        
        # Collect performance metrics
        performance_metrics = await self._collect_performance_metrics(agent)
        snapshot.request_rate = performance_metrics.get('request_rate', 0)
        snapshot.error_rate = performance_metrics.get('error_rate', 0)
        snapshot.latency_p50 = performance_metrics.get('latency_p50', 0)
        snapshot.latency_p95 = performance_metrics.get('latency_p95', 0)
        snapshot.latency_p99 = performance_metrics.get('latency_p99', 0)
        snapshot.metrics.update(performance_metrics)
        
        # Collect business metrics
        business_metrics = await self._collect_business_metrics(agent)
        snapshot.active_tasks = business_metrics.get('active_tasks', 0)
        snapshot.queued_tasks = business_metrics.get('queued_tasks', 0)
        snapshot.completed_tasks = business_metrics.get('completed_tasks', 0)
        snapshot.failed_tasks = business_metrics.get('failed_tasks', 0)
        snapshot.metrics.update(business_metrics)
        
        # Store metrics
        if self.metrics_repo:
            for metric_name, value in snapshot.metrics.items():
                self.metrics_repo.record_metric(
                    metric_name=metric_name,
                    value=value,
                    metric_type='gauge',
                    agent_id=agent.id,
                    dimensions={
                        'agent_type': agent.agent_type,
                        'agent_name': agent.name
                    }
                )
        
        # Update Prometheus metrics
        self._update_prometheus_metrics(agent, snapshot)
        
        return snapshot
    
    async def get_resource_metrics(self, agent_id: str) -> Dict[str, float]:
        """Get current resource usage metrics for an agent"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return {}
        
        # Get from container/process if running
        if agent.container_id:
            return await self._get_container_metrics(agent.container_id)
        else:
            # Return allocated resources
            return {
                'cpu_cores': agent.cpu_cores,
                'memory_gb': agent.memory_gb,
                'storage_gb': agent.storage_gb,
                'cpu_usage': 0,
                'memory_usage': 0,
                'disk_usage': 0
            }
    
    async def get_metric_history(
        self,
        agent_id: str,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval: str = '5m'
    ) -> List[Dict[str, Any]]:
        """Get historical metrics for an agent"""
        if not self.metrics_repo:
            return []
        
        # Default time range
        if not end_time:
            end_time = datetime.utcnow()
        if not start_time:
            start_time = end_time - timedelta(hours=1)
        
        # Convert interval to seconds
        interval_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '1d': 86400
        }
        interval_seconds = interval_map.get(interval, 300)
        
        # Query metrics
        return self.metrics_repo.get_metrics_timeseries(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            agent_id=agent_id,
            interval_seconds=interval_seconds
        )
    
    async def calculate_sla_metrics(
        self,
        agent_id: str,
        period: timedelta = timedelta(days=30)
    ) -> Dict[str, float]:
        """Calculate SLA metrics for an agent"""
        end_time = datetime.utcnow()
        start_time = end_time - period
        
        # Get agent
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return {}
        
        # Calculate uptime
        total_seconds = period.total_seconds()
        
        # Get downtime from state history
        downtime_seconds = 0
        if agent.state_history:
            for transition in agent.state_history:
                if transition.get('to_state') in ['error', 'terminated']:
                    # Calculate downtime duration
                    # This is simplified - would need proper tracking
                    downtime_seconds += 3600  # Assume 1 hour per incident
        
        uptime_percentage = ((total_seconds - downtime_seconds) / total_seconds) * 100
        
        # Get performance metrics
        if self.metrics_repo:
            # Get error rate
            error_metrics = self.metrics_repo.get_metrics_timeseries(
                metric_name='agent.error_rate',
                start_time=start_time,
                end_time=end_time,
                agent_id=agent_id,
                aggregation='avg'
            )
            
            avg_error_rate = statistics.mean(
                [m['value'] for m in error_metrics]
            ) if error_metrics else 0
            
            # Get latency metrics
            latency_metrics = self.metrics_repo.get_metrics_timeseries(
                metric_name='agent.latency_ms',
                start_time=start_time,
                end_time=end_time,
                agent_id=agent_id,
                aggregation='avg'
            )
            
            avg_latency = statistics.mean(
                [m['value'] for m in latency_metrics]
            ) if latency_metrics else 0
            
            # Calculate percentiles
            latency_percentiles = self.metrics_repo.calculate_percentiles(
                metric_name='agent.latency_ms',
                start_time=start_time,
                end_time=end_time,
                agent_id=agent_id
            )
        else:
            avg_error_rate = 0
            avg_latency = agent.average_latency_ms
            latency_percentiles = {}
        
        return {
            'uptime_percentage': uptime_percentage,
            'availability_percentage': uptime_percentage * (1 - avg_error_rate / 100),
            'average_error_rate': avg_error_rate,
            'average_latency_ms': avg_latency,
            'latency_p50': latency_percentiles.get('p50', 0),
            'latency_p95': latency_percentiles.get('p95', 0),
            'latency_p99': latency_percentiles.get('p99', 0),
            'total_requests': agent.total_requests,
            'total_errors': agent.total_errors,
            'period_days': period.days
        }
    
    def record_custom_metric(
        self,
        agent_id: str,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a custom metric"""
        # Ensure metric is registered
        if metric_name not in self._metric_definitions:
            self.register_metric(MetricDefinition(
                name=metric_name,
                type=MetricType.GAUGE,
                category=MetricCategory.CUSTOM,
                description=f"Custom metric: {metric_name}",
                labels=list(labels.keys()) if labels else []
            ))
        
        # Update Prometheus metric
        if metric_name in self._prometheus_metrics:
            metric = self._prometheus_metrics[metric_name]
            
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
        
        # Store in buffer for aggregation
        if metric_name not in self._metric_buffers:
            self._metric_buffers[metric_name] = []
        
        self._metric_buffers[metric_name].append((datetime.utcnow(), value))
        
        # Trim buffer
        if len(self._metric_buffers[metric_name]) > self._buffer_size:
            self._metric_buffers[metric_name] = self._metric_buffers[metric_name][-self._buffer_size:]
    
    # Private collection methods
    
    async def _collection_loop(self, agent_id: str, category: MetricCategory):
        """Collection loop for a metric category"""
        interval = self._collection_intervals[category]
        
        while not self._shutdown_event.is_set():
            try:
                # Collect metrics for category
                if category == MetricCategory.SYSTEM:
                    await self._collect_and_store_system_metrics(agent_id)
                elif category == MetricCategory.PERFORMANCE:
                    await self._collect_and_store_performance_metrics(agent_id)
                elif category == MetricCategory.BUSINESS:
                    await self._collect_and_store_business_metrics(agent_id)
                elif category == MetricCategory.HEALTH:
                    await self._collect_and_store_health_metrics(agent_id)
                elif category == MetricCategory.RESOURCE:
                    await self._collect_and_store_resource_metrics(agent_id)
                elif category == MetricCategory.NETWORK:
                    await self._collect_and_store_network_metrics(agent_id)
                elif category == MetricCategory.STORAGE:
                    await self._collect_and_store_storage_metrics(agent_id)
                
            except Exception as e:
                logger.error(f"Error collecting {category} metrics for agent {agent_id}: {str(e)}")
            
            await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self, agent: Agent) -> Dict[str, float]:
        """Collect system-level metrics"""
        metrics = {}
        
        try:
            # Get container/process metrics
            if agent.container_id:
                container_metrics = await self._get_container_metrics(agent.container_id)
                metrics.update(container_metrics)
            else:
                # Use psutil for local process
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics['cpu_usage'] = cpu_percent
                metrics['memory_usage'] = memory.percent
                metrics['disk_usage'] = disk.percent
                metrics['memory_used_mb'] = memory.used / 1024 / 1024
                metrics['memory_available_mb'] = memory.available / 1024 / 1024
                
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
        
        return metrics
    
    async def _collect_performance_metrics(self, agent: Agent) -> Dict[str, float]:
        """Collect performance metrics"""
        metrics = {}
        
        try:
            # Calculate rates
            if hasattr(agent, '_last_request_count'):
                request_rate = (agent.total_requests - agent._last_request_count) / 60
                error_rate = (agent.total_errors - agent._last_error_count) / 60
            else:
                request_rate = 0
                error_rate = 0
            
            agent._last_request_count = agent.total_requests
            agent._last_error_count = agent.total_errors
            
            metrics['request_rate'] = request_rate
            metrics['error_rate'] = (error_rate / request_rate * 100) if request_rate > 0 else 0
            metrics['average_latency'] = agent.average_latency_ms
            
            # Get percentiles from recent metrics
            if self.metrics_repo:
                percentiles = self.metrics_repo.calculate_percentiles(
                    metric_name='agent.latency_ms',
                    start_time=datetime.utcnow() - timedelta(minutes=5),
                    end_time=datetime.utcnow(),
                    agent_id=agent.id
                )
                
                metrics['latency_p50'] = percentiles.get('p50', 0)
                metrics['latency_p95'] = percentiles.get('p95', 0)
                metrics['latency_p99'] = percentiles.get('p99', 0)
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {str(e)}")
        
        return metrics
    
    async def _collect_business_metrics(self, agent: Agent) -> Dict[str, float]:
        """Collect business-level metrics"""
        metrics = {}
        
        try:
            # Get task metrics from scheduler if available
            if hasattr(self.agent_manager, 'scheduler'):
                # This would query the scheduler for task stats
                metrics['active_tasks'] = 0
                metrics['queued_tasks'] = 0
                metrics['completed_tasks'] = 0
                metrics['failed_tasks'] = 0
            
            # Add custom business metrics based on agent type
            if agent.agent_type == 'langchain':
                metrics['chain_executions'] = 0
                metrics['tool_calls'] = 0
            elif agent.agent_type == 'autogen':
                metrics['code_executions'] = 0
                metrics['conversation_turns'] = 0
            
        except Exception as e:
            logger.error(f"Error collecting business metrics: {str(e)}")
        
        return metrics
    
    async def _collect_health_metrics(self, agent: Agent) -> Dict[str, float]:
        """Collect health metrics"""
        metrics = {}
        
        try:
            # Get health status
            if hasattr(self.agent_manager, 'health_monitor'):
                health_status = await self.agent_manager.health_monitor.get_health_status(str(agent.id))
                
                # Convert to numeric score
                health_score = {
                    'healthy': 100,
                    'degraded': 75,
                    'unhealthy': 50,
                    'critical': 25,
                    'unknown': 0
                }.get(health_status.get('status', 'unknown'), 0)
                
                metrics['health_score'] = health_score
            
            # Calculate uptime
            if agent.activated_at:
                uptime_seconds = (datetime.utcnow() - agent.activated_at).total_seconds()
                metrics['uptime_seconds'] = uptime_seconds
            
        except Exception as e:
            logger.error(f"Error collecting health metrics: {str(e)}")
        
        return metrics
    
    async def _collect_resource_metrics(self, agent: Agent) -> Dict[str, float]:
        """Collect resource utilization metrics"""
        metrics = {}
        
        try:
            # Resource allocation
            metrics['cpu_allocated'] = agent.cpu_cores
            metrics['memory_allocated_gb'] = agent.memory_gb
            metrics['storage_allocated_gb'] = agent.storage_gb
            
            if agent.gpu_enabled:
                metrics['gpu_allocated'] = 1
                metrics['gpu_memory_allocated_gb'] = agent.gpu_memory_gb
            
            # Resource utilization vs allocation
            current_usage = await self.get_resource_metrics(str(agent.id))
            
            if current_usage:
                metrics['cpu_utilization_percent'] = (
                    current_usage.get('cpu_usage', 0) / agent.cpu_cores * 100
                    if agent.cpu_cores > 0 else 0
                )
                metrics['memory_utilization_percent'] = (
                    current_usage.get('memory_usage', 0) / agent.memory_gb * 100
                    if agent.memory_gb > 0 else 0
                )
            
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {str(e)}")
        
        return metrics
    
    async def _collect_network_metrics(self, agent: Agent) -> Dict[str, float]:
        """Collect network metrics"""
        metrics = {}
        
        try:
            # Get network stats
            if agent.container_id:
                # Get from container
                pass
            else:
                # Use psutil
                net_io = psutil.net_io_counters()
                
                # Calculate rates
                if hasattr(agent, '_last_net_bytes_sent'):
                    bytes_sent_rate = (net_io.bytes_sent - agent._last_net_bytes_sent) / 60
                    bytes_recv_rate = (net_io.bytes_recv - agent._last_net_bytes_recv) / 60
                else:
                    bytes_sent_rate = 0
                    bytes_recv_rate = 0
                
                agent._last_net_bytes_sent = net_io.bytes_sent
                agent._last_net_bytes_recv = net_io.bytes_recv
                
                metrics['network_out_mbps'] = bytes_sent_rate / 1024 / 1024 * 8
                metrics['network_in_mbps'] = bytes_recv_rate / 1024 / 1024 * 8
                metrics['network_packets_sent'] = net_io.packets_sent
                metrics['network_packets_recv'] = net_io.packets_recv
                
                # Connection stats
                connections = psutil.net_connections()
                metrics['network_connections_active'] = len([
                    c for c in connections if c.status == 'ESTABLISHED'
                ])
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {str(e)}")
        
        return metrics
    
    async def _collect_storage_metrics(self, agent: Agent) -> Dict[str, float]:
        """Collect storage metrics"""
        metrics = {}
        
        try:
            # Get storage volumes
            volumes = self.agent_manager.storage_repo.get_volumes_by_agent(agent.id)
            
            total_size = 0
            total_used = 0
            
            for volume in volumes:
                total_size += volume.size_gb
                total_used += volume.used_gb
                
                # Volume-specific metrics
                metrics[f'storage_used_{volume.volume_name}'] = volume.used_gb
                metrics[f'storage_iops_read_{volume.volume_name}'] = volume.read_iops
                metrics[f'storage_iops_write_{volume.volume_name}'] = volume.write_iops
            
            metrics['storage_total_gb'] = total_size
            metrics['storage_used_gb'] = total_used
            metrics['storage_utilization_percent'] = (
                (total_used / total_size * 100) if total_size > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error collecting storage metrics: {str(e)}")
        
        return metrics
    
    async def _get_container_metrics(self, container_id: str) -> Dict[str, float]:
        """Get metrics from container runtime"""
        # This would integrate with Docker/containerd API
        # For now, return mock data
        return {
            'cpu_usage': 50.0,
            'memory_usage': 60.0,
            'disk_usage': 40.0,
            'network_in_mbps': 10.0,
            'network_out_mbps': 5.0
        }
    
    def _update_prometheus_metrics(self, agent: Agent, snapshot: MetricSnapshot):
        """Update Prometheus metrics from snapshot"""
        labels = {
            'agent_id': str(agent.id),
            'agent_type': agent.agent_type
        }
        
        # Update gauges
        if 'agent_cpu_usage' in self._prometheus_metrics:
            self._prometheus_metrics['agent_cpu_usage'].labels(**labels).set(snapshot.cpu_usage)
        
        if 'agent_memory_usage' in self._prometheus_metrics:
            self._prometheus_metrics['agent_memory_usage'].labels(**labels).set(snapshot.memory_usage)
        
        if 'agent_error_rate' in self._prometheus_metrics:
            self._prometheus_metrics['agent_error_rate'].labels(**labels).set(snapshot.error_rate)
        
        # Update counters
        if 'agent_request_total' in self._prometheus_metrics:
            self._prometheus_metrics['agent_request_total'].labels(
                **labels, status='success'
            )._value.set(agent.total_requests - agent.total_errors)
            
            self._prometheus_metrics['agent_request_total'].labels(
                **labels, status='error'
            )._value.set(agent.total_errors)
    
    async def _collect_and_store_system_metrics(self, agent_id: str):
        """Collect and store system metrics"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return
        
        metrics = await self._collect_system_metrics(agent)
        
        if self.metrics_repo:
            for name, value in metrics.items():
                self.metrics_repo.record_metric(
                    metric_name=f"agent.system.{name}",
                    value=value,
                    metric_type='gauge',
                    agent_id=agent.id
                )
    
    async def _collect_and_store_performance_metrics(self, agent_id: str):
        """Collect and store performance metrics"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return
        
        metrics = await self._collect_performance_metrics(agent)
        
        if self.metrics_repo:
            for name, value in metrics.items():
                self.metrics_repo.record_metric(
                    metric_name=f"agent.performance.{name}",
                    value=value,
                    metric_type='gauge',
                    agent_id=agent.id
                )
    
    async def _collect_and_store_business_metrics(self, agent_id: str):
        """Collect and store business metrics"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return
        
        metrics = await self._collect_business_metrics(agent)
        
        if self.metrics_repo:
            for name, value in metrics.items():
                self.metrics_repo.record_metric(
                    metric_name=f"agent.business.{name}",
                    value=value,
                    metric_type='gauge',
                    agent_id=agent.id
                )
    
    async def _collect_and_store_health_metrics(self, agent_id: str):
        """Collect and store health metrics"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return
        
        metrics = await self._collect_health_metrics(agent)
        
        if self.metrics_repo:
            for name, value in metrics.items():
                self.metrics_repo.record_metric(
                    metric_name=f"agent.health.{name}",
                    value=value,
                    metric_type='gauge',
                    agent_id=agent.id
                )
    
    async def _collect_and_store_resource_metrics(self, agent_id: str):
        """Collect and store resource metrics"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return
        
        metrics = await self._collect_resource_metrics(agent)
        
        if self.metrics_repo:
            for name, value in metrics.items():
                self.metrics_repo.record_metric(
                    metric_name=f"agent.resource.{name}",
                    value=value,
                    metric_type='gauge',
                    agent_id=agent.id
                )
    
    async def _collect_and_store_network_metrics(self, agent_id: str):
        """Collect and store network metrics"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return
        
        metrics = await self._collect_network_metrics(agent)
        
        if self.metrics_repo:
            for name, value in metrics.items():
                self.metrics_repo.record_metric(
                    metric_name=f"agent.network.{name}",
                    value=value,
                    metric_type='gauge',
                    agent_id=agent.id
                )
    
    async def _collect_and_store_storage_metrics(self, agent_id: str):
        """Collect and store storage metrics"""
        agent = await self.agent_manager.get_agent(agent_id)
        if not agent:
            return
        
        metrics = await self._collect_storage_metrics(agent)
        
        if self.metrics_repo:
            for name, value in metrics.items():
                self.metrics_repo.record_metric(
                    metric_name=f"agent.storage.{name}",
                    value=value,
                    metric_type='gauge',
                    agent_id=agent.id
                )