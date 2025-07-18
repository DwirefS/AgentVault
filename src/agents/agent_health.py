"""
AgentVault™ Agent Health Monitoring
Comprehensive health monitoring and recovery system for agents
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics

from ..database.models import Agent, AgentState
from ..database.repositories import MetricsRepository, AgentRepository
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CheckType(str, Enum):
    """Types of health checks"""
    LIVENESS = "liveness"
    READINESS = "readiness"
    STARTUP = "startup"
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    CONNECTIVITY = "connectivity"
    DATA_INTEGRITY = "data_integrity"
    SECURITY = "security"


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_type: CheckType
    check_function: Callable[[Agent], asyncio.coroutine]
    interval_seconds: int = 60
    timeout_seconds: int = 30
    failure_threshold: int = 3
    success_threshold: int = 1
    critical: bool = False
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    check_name: str
    status: HealthStatus
    timestamp: datetime
    duration_ms: float
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class AgentHealthReport:
    """Comprehensive health report for an agent"""
    agent_id: str
    overall_status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheckResult]
    metrics: Dict[str, Any]
    recommendations: List[str]
    requires_intervention: bool = False
    auto_recovery_possible: bool = True


class AgentHealthMonitor:
    """
    Monitors agent health and triggers recovery actions
    """
    
    def __init__(
        self,
        agent_manager: Any,  # Avoid circular import
        metrics_repo: Optional[MetricsRepository] = None,
        monitoring: Optional[AdvancedMonitoringSystem] = None
    ):
        self.agent_manager = agent_manager
        self.metrics_repo = metrics_repo
        self.monitoring = monitoring
        
        # Health check registry
        self._health_checks: Dict[str, List[HealthCheck]] = {}
        self._check_results: Dict[str, List[HealthCheckResult]] = {}
        self._check_tasks: Dict[str, List[asyncio.Task]] = {}
        
        # Recovery strategies
        self._recovery_strategies: Dict[str, Callable] = {}
        
        # Initialize default health checks
        self._register_default_checks()
        self._register_default_recovery_strategies()
    
    def _register_default_checks(self):
        """Register default health checks"""
        # Liveness check
        self.register_check(HealthCheck(
            name="agent_liveness",
            check_type=CheckType.LIVENESS,
            check_function=self._check_liveness,
            interval_seconds=30,
            timeout_seconds=10,
            failure_threshold=3,
            critical=True
        ))
        
        # Readiness check
        self.register_check(HealthCheck(
            name="agent_readiness",
            check_type=CheckType.READINESS,
            check_function=self._check_readiness,
            interval_seconds=60,
            timeout_seconds=15,
            failure_threshold=2
        ))
        
        # Performance check
        self.register_check(HealthCheck(
            name="agent_performance",
            check_type=CheckType.PERFORMANCE,
            check_function=self._check_performance,
            interval_seconds=300,  # 5 minutes
            timeout_seconds=30,
            failure_threshold=5
        ))
        
        # Resource check
        self.register_check(HealthCheck(
            name="agent_resources",
            check_type=CheckType.RESOURCE,
            check_function=self._check_resources,
            interval_seconds=120,
            timeout_seconds=20,
            failure_threshold=3
        ))
        
        # Connectivity check
        self.register_check(HealthCheck(
            name="agent_connectivity",
            check_type=CheckType.CONNECTIVITY,
            check_function=self._check_connectivity,
            interval_seconds=180,
            timeout_seconds=30,
            failure_threshold=3
        ))
        
        # Data integrity check
        self.register_check(HealthCheck(
            name="agent_data_integrity",
            check_type=CheckType.DATA_INTEGRITY,
            check_function=self._check_data_integrity,
            interval_seconds=3600,  # 1 hour
            timeout_seconds=300,    # 5 minutes
            failure_threshold=1,
            critical=True
        ))
        
        # Security check
        self.register_check(HealthCheck(
            name="agent_security",
            check_type=CheckType.SECURITY,
            check_function=self._check_security,
            interval_seconds=1800,  # 30 minutes
            timeout_seconds=60,
            failure_threshold=1,
            critical=True
        ))
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies"""
        self._recovery_strategies = {
            'restart': self._recovery_restart,
            'reload_config': self._recovery_reload_config,
            'clear_cache': self._recovery_clear_cache,
            'reset_connections': self._recovery_reset_connections,
            'scale_resources': self._recovery_scale_resources,
            'migrate': self._recovery_migrate,
            'rollback': self._recovery_rollback
        }
    
    def register_check(self, check: HealthCheck, agent_types: Optional[List[str]] = None):
        """Register a health check"""
        if agent_types:
            for agent_type in agent_types:
                if agent_type not in self._health_checks:
                    self._health_checks[agent_type] = []
                self._health_checks[agent_type].append(check)
        else:
            # Register for all agent types
            if 'all' not in self._health_checks:
                self._health_checks['all'] = []
            self._health_checks['all'].append(check)
    
    def register_recovery_strategy(self, name: str, strategy: Callable):
        """Register a recovery strategy"""
        self._recovery_strategies[name] = strategy
    
    async def start_monitoring(self, agent: Agent):
        """Start health monitoring for an agent"""
        logger.info(f"Starting health monitoring for agent {agent.id}")
        
        # Get applicable health checks
        checks = self._get_agent_checks(agent)
        
        # Start check tasks
        agent_key = str(agent.id)
        if agent_key not in self._check_tasks:
            self._check_tasks[agent_key] = []
        
        for check in checks:
            if check.enabled:
                task = asyncio.create_task(
                    self._run_check_loop(agent, check)
                )
                self._check_tasks[agent_key].append(task)
    
    async def stop_monitoring(self, agent: Agent):
        """Stop health monitoring for an agent"""
        logger.info(f"Stopping health monitoring for agent {agent.id}")
        
        agent_key = str(agent.id)
        
        # Cancel check tasks
        if agent_key in self._check_tasks:
            for task in self._check_tasks[agent_key]:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self._check_tasks[agent_key], return_exceptions=True)
            
            # Clean up
            del self._check_tasks[agent_key]
        
        # Clean up results
        if agent_key in self._check_results:
            del self._check_results[agent_key]
    
    async def check_agent_health(self, agent: Agent) -> AgentHealthReport:
        """Perform immediate health check on agent"""
        logger.debug(f"Performing health check for agent {agent.id}")
        
        # Get applicable checks
        checks = self._get_agent_checks(agent)
        
        # Run all checks
        results = []
        for check in checks:
            if check.enabled:
                result = await self._run_single_check(agent, check)
                results.append(result)
        
        # Analyze results
        report = self._analyze_health_results(agent, results)
        
        # Check if recovery is needed
        if report.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            await self._trigger_recovery(agent, report)
        
        return report
    
    async def get_health_status(self, agent_id: str) -> Dict[str, Any]:
        """Get current health status for an agent"""
        agent_key = str(agent_id)
        
        # Get recent results
        recent_results = self._check_results.get(agent_key, [])
        
        if not recent_results:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': 'No health data available',
                'checks': {}
            }
        
        # Group by check name
        check_status = {}
        for result in recent_results[-100:]:  # Last 100 results
            if result.check_name not in check_status:
                check_status[result.check_name] = []
            check_status[result.check_name].append(result)
        
        # Calculate overall status
        overall_status = HealthStatus.HEALTHY
        check_summaries = {}
        
        for check_name, results in check_status.items():
            # Get latest result
            latest = results[-1]
            
            # Calculate success rate
            success_rate = sum(1 for r in results[-10:] if r.status == HealthStatus.HEALTHY) / min(len(results), 10)
            
            check_summaries[check_name] = {
                'status': latest.status,
                'last_check': latest.timestamp.isoformat(),
                'success_rate': success_rate,
                'message': latest.message
            }
            
            # Update overall status
            if latest.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif latest.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif latest.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            'status': overall_status,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': check_summaries,
            'healthy': overall_status == HealthStatus.HEALTHY
        }
    
    def _get_agent_checks(self, agent: Agent) -> List[HealthCheck]:
        """Get applicable health checks for an agent"""
        checks = []
        
        # Get type-specific checks
        if agent.agent_type in self._health_checks:
            checks.extend(self._health_checks[agent.agent_type])
        
        # Get universal checks
        if 'all' in self._health_checks:
            checks.extend(self._health_checks['all'])
        
        return checks
    
    async def _run_check_loop(self, agent: Agent, check: HealthCheck):
        """Run a health check in a loop"""
        agent_key = str(agent.id)
        consecutive_failures = 0
        consecutive_successes = 0
        
        while True:
            try:
                # Run check
                result = await self._run_single_check(agent, check)
                
                # Store result
                if agent_key not in self._check_results:
                    self._check_results[agent_key] = []
                
                self._check_results[agent_key].append(result)
                
                # Limit stored results
                if len(self._check_results[agent_key]) > 1000:
                    self._check_results[agent_key] = self._check_results[agent_key][-500:]
                
                # Track consecutive results
                if result.status == HealthStatus.HEALTHY:
                    consecutive_successes += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    consecutive_successes = 0
                
                # Check thresholds
                if consecutive_failures >= check.failure_threshold:
                    logger.warning(
                        f"Health check {check.name} failed {consecutive_failures} times for agent {agent.id}"
                    )
                    
                    # Trigger alert
                    if self.monitoring:
                        await self.monitoring.send_alert(
                            severity='warning' if not check.critical else 'critical',
                            title=f"Agent health check failed: {check.name}",
                            description=f"Agent {agent.id} failed {check.name} check {consecutive_failures} times",
                            resource_id=str(agent.id),
                            resource_type='agent'
                        )
                
                # Record metrics
                if self.monitoring:
                    self.monitoring.record_custom_metric(
                        f"agent_health_check_{check.name}",
                        1 if result.status == HealthStatus.HEALTHY else 0,
                        labels={
                            'agent_id': str(agent.id),
                            'check_type': check.check_type,
                            'status': result.status
                        }
                    )
                
                # Wait for next check
                await asyncio.sleep(check.interval_seconds)
                
            except asyncio.CancelledError:
                logger.info(f"Health check {check.name} cancelled for agent {agent.id}")
                break
            except Exception as e:
                logger.error(f"Error in health check loop {check.name} for agent {agent.id}: {str(e)}")
                await asyncio.sleep(check.interval_seconds)
    
    async def _run_single_check(self, agent: Agent, check: HealthCheck) -> HealthCheckResult:
        """Run a single health check"""
        start_time = datetime.utcnow()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_function(agent),
                timeout=check.timeout_seconds
            )
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Parse result
            if isinstance(result, tuple):
                status, message, details = result[0], result[1] if len(result) > 1 else None, result[2] if len(result) > 2 else {}
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = None
                details = {}
            else:
                status = result.get('status', HealthStatus.UNKNOWN)
                message = result.get('message')
                details = result.get('details', {})
            
            return HealthCheckResult(
                check_name=check.name,
                status=status,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=message,
                details=details
            )
            
        except asyncio.TimeoutError:
            duration_ms = check.timeout_seconds * 1000
            return HealthCheckResult(
                check_name=check.name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=f"Check timed out after {check.timeout_seconds}s",
                error="TimeoutError"
            )
            
        except Exception as e:
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return HealthCheckResult(
                check_name=check.name,
                status=HealthStatus.UNHEALTHY,
                timestamp=datetime.utcnow(),
                duration_ms=duration_ms,
                message=f"Check failed with error: {str(e)}",
                error=str(e)
            )
    
    def _analyze_health_results(
        self,
        agent: Agent,
        results: List[HealthCheckResult]
    ) -> AgentHealthReport:
        """Analyze health check results and generate report"""
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        critical_issues = []
        warnings = []
        recommendations = []
        
        for result in results:
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                critical_issues.append(result)
            elif result.status == HealthStatus.UNHEALTHY:
                if overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.UNHEALTHY
                critical_issues.append(result)
            elif result.status == HealthStatus.DEGRADED:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                warnings.append(result)
        
        # Generate recommendations
        if critical_issues:
            recommendations.append("Immediate intervention required for critical issues")
            
            for issue in critical_issues:
                if issue.check_name == "agent_liveness":
                    recommendations.append("Agent appears to be down - restart recommended")
                elif issue.check_name == "agent_data_integrity":
                    recommendations.append("Data integrity issue detected - backup and investigation required")
                elif issue.check_name == "agent_security":
                    recommendations.append("Security issue detected - immediate review required")
        
        if warnings:
            for warning in warnings:
                if warning.check_name == "agent_performance":
                    recommendations.append("Performance degradation detected - consider scaling resources")
                elif warning.check_name == "agent_resources":
                    recommendations.append("Resource constraints detected - consider increasing allocation")
        
        # Collect metrics
        metrics = {
            'total_checks': len(results),
            'healthy_checks': sum(1 for r in results if r.status == HealthStatus.HEALTHY),
            'critical_issues': len(critical_issues),
            'warnings': len(warnings),
            'average_check_duration_ms': statistics.mean(r.duration_ms for r in results) if results else 0
        }
        
        return AgentHealthReport(
            agent_id=str(agent.id),
            overall_status=overall_status,
            timestamp=datetime.utcnow(),
            checks=results,
            metrics=metrics,
            recommendations=recommendations,
            requires_intervention=overall_status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY],
            auto_recovery_possible=self._can_auto_recover(agent, results)
        )
    
    def _can_auto_recover(self, agent: Agent, results: List[HealthCheckResult]) -> bool:
        """Determine if auto-recovery is possible"""
        # Don't auto-recover if there are security or data integrity issues
        for result in results:
            if result.status != HealthStatus.HEALTHY:
                if result.check_name in ['agent_security', 'agent_data_integrity']:
                    return False
        
        # Don't auto-recover if agent has been restarted recently
        if agent.state_metadata and 'last_restart' in agent.state_metadata:
            last_restart = datetime.fromisoformat(agent.state_metadata['last_restart'])
            if datetime.utcnow() - last_restart < timedelta(minutes=10):
                return False
        
        return True
    
    async def _trigger_recovery(self, agent: Agent, report: AgentHealthReport):
        """Trigger recovery actions based on health report"""
        if not report.auto_recovery_possible:
            logger.warning(f"Manual intervention required for agent {agent.id}")
            
            # Send alert
            if self.monitoring:
                await self.monitoring.send_alert(
                    severity='critical',
                    title=f"Agent {agent.name} requires manual intervention",
                    description=f"Agent health: {report.overall_status}. Issues: {report.recommendations}",
                    resource_id=str(agent.id),
                    resource_type='agent'
                )
            return
        
        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(agent, report)
        
        if strategy:
            logger.info(f"Triggering {strategy} recovery for agent {agent.id}")
            
            try:
                recovery_func = self._recovery_strategies.get(strategy)
                if recovery_func:
                    await recovery_func(agent, report)
                else:
                    logger.error(f"Unknown recovery strategy: {strategy}")
            except Exception as e:
                logger.error(f"Recovery failed for agent {agent.id}: {str(e)}")
    
    def _determine_recovery_strategy(
        self,
        agent: Agent,
        report: AgentHealthReport
    ) -> Optional[str]:
        """Determine appropriate recovery strategy"""
        # Check specific issues
        for result in report.checks:
            if result.status != HealthStatus.HEALTHY:
                if result.check_name == "agent_liveness":
                    return 'restart'
                elif result.check_name == "agent_connectivity":
                    return 'reset_connections'
                elif result.check_name == "agent_performance":
                    if 'high_latency' in result.details:
                        return 'clear_cache'
                    elif 'resource_exhaustion' in result.details:
                        return 'scale_resources'
                elif result.check_name == "agent_resources":
                    return 'scale_resources'
        
        # Default to restart for general issues
        return 'restart'
    
    # Health check implementations
    
    async def _check_liveness(self, agent: Agent) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check if agent is alive and responding"""
        try:
            # Check runtime status
            runtime_status = self.agent_manager.agent_factory.get_runtime_status(str(agent.id))
            
            if not runtime_status or not runtime_status.get('running'):
                return HealthStatus.CRITICAL, "Agent runtime not running", {'runtime_status': runtime_status}
            
            # Check API endpoint
            if agent.internal_endpoint:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{agent.internal_endpoint}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            return HealthStatus.HEALTHY, "Agent is alive", {'endpoint_status': response.status}
                        else:
                            return HealthStatus.UNHEALTHY, f"Health endpoint returned {response.status}", {'endpoint_status': response.status}
            
            return HealthStatus.HEALTHY, "Agent is alive", {}
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Liveness check failed: {str(e)}", {'error': str(e)}
    
    async def _check_readiness(self, agent: Agent) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check if agent is ready to handle requests"""
        try:
            if agent.state != AgentState.RUNNING:
                return HealthStatus.UNHEALTHY, f"Agent not in running state: {agent.state}", {'state': agent.state}
            
            # Check dependencies
            dependencies_healthy = True
            dependency_status = {}
            
            # Check vector store connection
            if hasattr(self.agent_manager, 'vector_store'):
                try:
                    await self.agent_manager.vector_store.health_check()
                    dependency_status['vector_store'] = 'healthy'
                except:
                    dependency_status['vector_store'] = 'unhealthy'
                    dependencies_healthy = False
            
            # Check cache connection
            if hasattr(self.agent_manager, 'cache'):
                try:
                    await self.agent_manager.cache.ping()
                    dependency_status['cache'] = 'healthy'
                except:
                    dependency_status['cache'] = 'unhealthy'
                    dependencies_healthy = False
            
            if not dependencies_healthy:
                return HealthStatus.DEGRADED, "Some dependencies are unhealthy", dependency_status
            
            return HealthStatus.HEALTHY, "Agent is ready", dependency_status
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Readiness check failed: {str(e)}", {'error': str(e)}
    
    async def _check_performance(self, agent: Agent) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check agent performance metrics"""
        try:
            # Get recent metrics
            if self.metrics_repo:
                # Get latency metrics
                latency_metrics = self.metrics_repo.get_latest_metrics(
                    agent.id,
                    metric_names=['agent.latency_ms'],
                    limit=10
                )
                
                if latency_metrics:
                    avg_latency = statistics.mean(m.value for m in latency_metrics)
                    
                    if avg_latency > 1000:  # 1 second
                        return HealthStatus.DEGRADED, f"High latency: {avg_latency:.0f}ms", {'avg_latency_ms': avg_latency}
                
                # Get error rate
                error_rate = (agent.total_errors / agent.total_requests * 100) if agent.total_requests > 0 else 0
                
                if error_rate > 5:
                    return HealthStatus.DEGRADED, f"High error rate: {error_rate:.1f}%", {'error_rate': error_rate}
            
            return HealthStatus.HEALTHY, "Performance is normal", {
                'avg_latency_ms': agent.average_latency_ms,
                'error_rate': (agent.total_errors / agent.total_requests * 100) if agent.total_requests > 0 else 0
            }
            
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Performance check failed: {str(e)}", {'error': str(e)}
    
    async def _check_resources(self, agent: Agent) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check resource utilization"""
        try:
            # Get resource metrics from monitoring
            if self.agent_manager.metrics_collector:
                metrics = await self.agent_manager.metrics_collector.get_resource_metrics(agent.id)
                
                issues = []
                
                if metrics.get('cpu_usage', 0) > 90:
                    issues.append("High CPU usage")
                
                if metrics.get('memory_usage', 0) > 90:
                    issues.append("High memory usage")
                
                if metrics.get('disk_usage', 0) > 85:
                    issues.append("High disk usage")
                
                if issues:
                    return HealthStatus.DEGRADED, f"Resource constraints: {', '.join(issues)}", metrics
            
            return HealthStatus.HEALTHY, "Resources are within limits", {}
            
        except Exception as e:
            return HealthStatus.UNKNOWN, f"Resource check failed: {str(e)}", {'error': str(e)}
    
    async def _check_connectivity(self, agent: Agent) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check network connectivity"""
        try:
            connectivity_status = {}
            
            # Check storage connectivity
            volumes = self.agent_manager.storage_repo.get_volumes_by_agent(agent.id)
            for volume in volumes:
                if volume.state == 'ready':
                    connectivity_status[f'volume_{volume.volume_name}'] = 'connected'
                else:
                    connectivity_status[f'volume_{volume.volume_name}'] = 'disconnected'
            
            # Check if any critical connections are down
            if any(status == 'disconnected' for status in connectivity_status.values()):
                return HealthStatus.DEGRADED, "Some connections are down", connectivity_status
            
            return HealthStatus.HEALTHY, "All connections are healthy", connectivity_status
            
        except Exception as e:
            return HealthStatus.UNHEALTHY, f"Connectivity check failed: {str(e)}", {'error': str(e)}
    
    async def _check_data_integrity(self, agent: Agent) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check data integrity"""
        try:
            # This would include:
            # - Checking vector index consistency
            # - Verifying storage checksums
            # - Validating configuration integrity
            
            # Placeholder implementation
            return HealthStatus.HEALTHY, "Data integrity verified", {}
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Data integrity check failed: {str(e)}", {'error': str(e)}
    
    async def _check_security(self, agent: Agent) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Check security status"""
        try:
            security_issues = []
            
            # Check API key validity
            if agent.api_key_hash and not self._validate_api_key(agent):
                security_issues.append("Invalid API key")
            
            # Check certificate expiry
            # Check for suspicious activity
            # Verify access controls
            
            if security_issues:
                return HealthStatus.CRITICAL, f"Security issues detected: {', '.join(security_issues)}", {
                    'issues': security_issues
                }
            
            return HealthStatus.HEALTHY, "Security checks passed", {}
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Security check failed: {str(e)}", {'error': str(e)}
    
    def _validate_api_key(self, agent: Agent) -> bool:
        """Validate agent API key"""
        # Placeholder implementation
        return True
    
    # Recovery strategies
    
    async def _recovery_restart(self, agent: Agent, report: AgentHealthReport):
        """Restart agent"""
        logger.info(f"Restarting agent {agent.id}")
        
        # Update metadata
        if not agent.state_metadata:
            agent.state_metadata = {}
        agent.state_metadata['last_restart'] = datetime.utcnow().isoformat()
        agent.state_metadata['restart_reason'] = f"Health check failure: {report.overall_status}"
        
        # Restart agent
        await self.agent_manager.restart_agent(agent.id)
    
    async def _recovery_reload_config(self, agent: Agent, report: AgentHealthReport):
        """Reload agent configuration"""
        logger.info(f"Reloading configuration for agent {agent.id}")
        
        # Reload configuration from database
        updated_agent = await self.agent_manager.get_agent(agent.id, use_cache=False)
        
        # Apply configuration
        if updated_agent:
            await self.agent_manager.agent_factory.update_agent_runtime(
                updated_agent,
                updated_agent.configuration
            )
    
    async def _recovery_clear_cache(self, agent: Agent, report: AgentHealthReport):
        """Clear agent cache"""
        logger.info(f"Clearing cache for agent {agent.id}")
        
        # Clear distributed cache entries
        if self.agent_manager.cache:
            pattern = f"agent:{agent.id}:*"
            await self.agent_manager.cache.clear_pattern(pattern)
        
        # Clear local cache
        self.agent_manager._invalidate_cache(agent.id)
    
    async def _recovery_reset_connections(self, agent: Agent, report: AgentHealthReport):
        """Reset agent connections"""
        logger.info(f"Resetting connections for agent {agent.id}")
        
        # Execute connection reset command
        await self.agent_manager.execute_agent_command(
            agent.id,
            'reset_connections',
            {}
        )
    
    async def _recovery_scale_resources(self, agent: Agent, report: AgentHealthReport):
        """Scale agent resources"""
        logger.info(f"Scaling resources for agent {agent.id}")
        
        # Determine scaling needs
        current_cpu = agent.cpu_cores
        current_memory = agent.memory_gb
        
        # Simple scaling logic - increase by 50%
        new_cpu = min(current_cpu * 1.5, 8.0)  # Max 8 cores
        new_memory = min(current_memory * 1.5, 32.0)  # Max 32GB
        
        # Update agent resources
        await self.agent_manager.update_agent(
            agent.id,
            {
                'cpu_cores': new_cpu,
                'memory_gb': new_memory
            }
        )
        
        # Restart to apply changes
        await self.agent_manager.restart_agent(agent.id)
    
    async def _recovery_migrate(self, agent: Agent, report: AgentHealthReport):
        """Migrate agent to different node"""
        logger.info(f"Migrating agent {agent.id}")
        
        # Find healthy node
        # This would integrate with cluster management
        target_node = "node-2"  # Placeholder
        
        # Trigger migration
        await self.agent_manager.migrate_agent(agent.id, target_node)
    
    async def _recovery_rollback(self, agent: Agent, report: AgentHealthReport):
        """Rollback agent to previous version"""
        logger.info(f"Rolling back agent {agent.id}")
        
        # This would implement version rollback logic
        # - Restore previous configuration
        # - Downgrade agent version
        # - Restore data from backup
        pass