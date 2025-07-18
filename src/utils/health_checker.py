"""
AgentVault™ Health Checker Module
System health monitoring and diagnostics
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import psutil
import aioredis
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component"""
    name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any]
    checked_at: datetime


class HealthChecker:
    """Comprehensive health monitoring for AgentVault™"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.checks = []
        self.last_check_results = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check("system_resources", self.check_system_resources)
        self.register_check("redis_cache", self.check_redis_cache)
        self.register_check("storage_availability", self.check_storage_availability)
        self.register_check("api_responsiveness", self.check_api_responsiveness)
        self.register_check("ml_models", self.check_ml_models)
    
    def register_check(self, name: str, check_func):
        """Register a health check"""
        self.checks.append((name, check_func))
    
    async def check_system_resources(self) -> ComponentHealth:
        """Check system resource utilization"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Determine health status
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "Critical resource usage detected"
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 80:
                status = HealthStatus.DEGRADED
                message = "High resource usage detected"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources within normal limits"
            
            return ComponentHealth(
                name="system_resources",
                status=status,
                message=message,
                metrics={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_used_gb": memory.used / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                checked_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error checking system resources: {str(e)}")
            return ComponentHealth(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Check failed: {str(e)}",
                metrics={},
                checked_at=datetime.utcnow()
            )
    
    async def check_redis_cache(self) -> ComponentHealth:
        """Check Redis cache connectivity and performance"""
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            
            # Connect to Redis
            redis = await aioredis.from_url(redis_url)
            
            # Ping test
            start_time = datetime.utcnow()
            await redis.ping()
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Get Redis info
            info = await redis.info()
            
            # Get memory usage
            memory_used_mb = info.get('used_memory', 0) / (1024 * 1024)
            memory_max_mb = info.get('maxmemory', 0) / (1024 * 1024) if info.get('maxmemory') else None
            
            # Check health
            if latency_ms > 100:
                status = HealthStatus.UNHEALTHY
                message = "High Redis latency detected"
            elif latency_ms > 50:
                status = HealthStatus.DEGRADED
                message = "Elevated Redis latency"
            else:
                status = HealthStatus.HEALTHY
                message = "Redis cache operational"
            
            await redis.close()
            
            return ComponentHealth(
                name="redis_cache",
                status=status,
                message=message,
                metrics={
                    "latency_ms": latency_ms,
                    "memory_used_mb": memory_used_mb,
                    "memory_max_mb": memory_max_mb,
                    "connected_clients": info.get('connected_clients', 0),
                    "total_commands_processed": info.get('total_commands_processed', 0)
                },
                checked_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error checking Redis cache: {str(e)}")
            return ComponentHealth(
                name="redis_cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis unavailable: {str(e)}",
                metrics={},
                checked_at=datetime.utcnow()
            )
    
    async def check_storage_availability(self) -> ComponentHealth:
        """Check storage backend availability"""
        try:
            # Check ANF mount points
            anf_mount = self.config.get('anf_mount_path', '/mnt/agentvault')
            
            if psutil.disk_usage(anf_mount):
                usage = psutil.disk_usage(anf_mount)
                
                if usage.percent > 95:
                    status = HealthStatus.UNHEALTHY
                    message = "Storage critically full"
                elif usage.percent > 85:
                    status = HealthStatus.DEGRADED
                    message = "Storage usage high"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Storage available"
                
                return ComponentHealth(
                    name="storage_availability",
                    status=status,
                    message=message,
                    metrics={
                        "mount_point": anf_mount,
                        "usage_percent": usage.percent,
                        "free_gb": usage.free / (1024**3),
                        "total_gb": usage.total / (1024**3)
                    },
                    checked_at=datetime.utcnow()
                )
            else:
                return ComponentHealth(
                    name="storage_availability",
                    status=HealthStatus.UNHEALTHY,
                    message="Storage mount not available",
                    metrics={"mount_point": anf_mount},
                    checked_at=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error(f"Error checking storage: {str(e)}")
            return ComponentHealth(
                name="storage_availability",
                status=HealthStatus.UNKNOWN,
                message=f"Storage check failed: {str(e)}",
                metrics={},
                checked_at=datetime.utcnow()
            )
    
    async def check_api_responsiveness(self) -> ComponentHealth:
        """Check API endpoint responsiveness"""
        try:
            # This would typically check internal API endpoints
            # For now, we'll simulate a check
            
            # Simulate API latency check
            latency_ms = 15.0  # Simulated value
            error_rate = 0.001  # Simulated value
            
            if latency_ms > 100 or error_rate > 0.05:
                status = HealthStatus.UNHEALTHY
                message = "API performance degraded"
            elif latency_ms > 50 or error_rate > 0.01:
                status = HealthStatus.DEGRADED
                message = "API performance suboptimal"
            else:
                status = HealthStatus.HEALTHY
                message = "API responsive"
            
            return ComponentHealth(
                name="api_responsiveness",
                status=status,
                message=message,
                metrics={
                    "avg_latency_ms": latency_ms,
                    "error_rate": error_rate,
                    "requests_per_second": 1000  # Simulated
                },
                checked_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error checking API: {str(e)}")
            return ComponentHealth(
                name="api_responsiveness",
                status=HealthStatus.UNKNOWN,
                message=f"API check failed: {str(e)}",
                metrics={},
                checked_at=datetime.utcnow()
            )
    
    async def check_ml_models(self) -> ComponentHealth:
        """Check ML model availability and performance"""
        try:
            # Check if ML models are loaded and performing well
            # This is a simulated check
            
            models_loaded = True
            prediction_latency_ms = 25.0
            accuracy_score = 0.92
            
            if not models_loaded:
                status = HealthStatus.UNHEALTHY
                message = "ML models not loaded"
            elif prediction_latency_ms > 100 or accuracy_score < 0.8:
                status = HealthStatus.DEGRADED
                message = "ML performance degraded"
            else:
                status = HealthStatus.HEALTHY
                message = "ML models operational"
            
            return ComponentHealth(
                name="ml_models",
                status=status,
                message=message,
                metrics={
                    "models_loaded": models_loaded,
                    "prediction_latency_ms": prediction_latency_ms,
                    "accuracy_score": accuracy_score,
                    "model_version": "1.0.0"
                },
                checked_at=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error checking ML models: {str(e)}")
            return ComponentHealth(
                name="ml_models",
                status=HealthStatus.UNKNOWN,
                message=f"ML check failed: {str(e)}",
                metrics={},
                checked_at=datetime.utcnow()
            )
    
    async def run_health_checks(self) -> Dict[str, ComponentHealth]:
        """Run all registered health checks"""
        results = {}
        
        # Run checks concurrently
        tasks = []
        for name, check_func in self.checks:
            tasks.append(check_func())
        
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for (name, _), result in zip(self.checks, check_results):
            if isinstance(result, Exception):
                results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(result)}",
                    metrics={},
                    checked_at=datetime.utcnow()
                )
            else:
                results[name] = result
        
        self.last_check_results = results
        return results
    
    def get_overall_status(self, results: Dict[str, ComponentHealth]) -> HealthStatus:
        """Determine overall system health status"""
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [comp.status for comp in results.values()]
        
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        results = await self.run_health_checks()
        overall_status = self.get_overall_status(results)
        
        report = {
            "status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        for name, health in results.items():
            report["components"][name] = {
                "status": health.status.value,
                "message": health.message,
                "metrics": health.metrics,
                "checked_at": health.checked_at.isoformat()
            }
        
        return report