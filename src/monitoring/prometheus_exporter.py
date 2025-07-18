#!/usr/bin/env python3
"""
AgentVault™ Prometheus Metrics Exporter
Exports system metrics for monitoring
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry
)
from fastapi import FastAPI, Response
import psutil
import aioredis

logger = logging.getLogger(__name__)

# Create custom registry
registry = CollectorRegistry()

# Define metrics
request_counter = Counter(
    'agentvault_requests_total',
    'Total number of storage requests',
    ['agent_id', 'operation', 'status', 'tier'],
    registry=registry
)

request_duration = Histogram(
    'agentvault_request_duration_seconds',
    'Request duration in seconds',
    ['operation', 'tier'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=registry
)

active_agents = Gauge(
    'agentvault_active_agents',
    'Number of active agents',
    ['framework'],
    registry=registry
)

storage_usage = Gauge(
    'agentvault_storage_usage_bytes',
    'Storage usage in bytes',
    ['agent_id', 'tier'],
    registry=registry
)

cache_hit_rate = Gauge(
    'agentvault_cache_hit_rate',
    'Cache hit rate',
    ['agent_id'],
    registry=registry
)

tier_latency = Histogram(
    'agentvault_tier_latency_seconds',
    'Storage tier latency',
    ['tier'],
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
    registry=registry
)

compression_ratio = Gauge(
    'agentvault_compression_ratio',
    'Data compression ratio',
    ['agent_id'],
    registry=registry
)

ml_prediction_accuracy = Gauge(
    'agentvault_ml_prediction_accuracy',
    'ML model prediction accuracy',
    ['model_type'],
    registry=registry
)

system_info = Info(
    'agentvault_system',
    'System information',
    registry=registry
)

# System resource metrics
cpu_usage = Gauge(
    'agentvault_cpu_usage_percent',
    'CPU usage percentage',
    registry=registry
)

memory_usage = Gauge(
    'agentvault_memory_usage_bytes',
    'Memory usage in bytes',
    ['type'],
    registry=registry
)

disk_usage = Gauge(
    'agentvault_disk_usage_bytes',
    'Disk usage in bytes',
    ['mount_point', 'type'],
    registry=registry
)

network_traffic = Counter(
    'agentvault_network_bytes_total',
    'Network traffic in bytes',
    ['direction', 'interface'],
    registry=registry
)


class PrometheusExporter:
    """Prometheus metrics exporter for AgentVault™"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.update_interval = self.config.get('update_interval', 30)
        self._running = False
        self._task = None
        
        # Initialize system info
        system_info.info({
            'version': '1.0.0',
            'environment': self.config.get('environment', 'production'),
            'region': self.config.get('region', 'eastus')
        })
    
    async def start(self) -> None:
        """Start metrics collection"""
        self._running = True
        self._task = asyncio.create_task(self._collect_metrics_loop())
        logger.info("Prometheus exporter started")
    
    async def stop(self) -> None:
        """Stop metrics collection"""
        self._running = False
        if self._task:
            await self._task
        logger.info("Prometheus exporter stopped")
    
    async def _collect_metrics_loop(self) -> None:
        """Main metrics collection loop"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system resource metrics"""
        # CPU usage
        cpu_usage.set(psutil.cpu_percent(interval=1))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage.labels(type='used').set(memory.used)
        memory_usage.labels(type='available').set(memory.available)
        memory_usage.labels(type='cached').set(memory.cached)
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage.labels(
                    mount_point=partition.mountpoint,
                    type='used'
                ).set(usage.used)
                disk_usage.labels(
                    mount_point=partition.mountpoint,
                    type='free'
                ).set(usage.free)
            except PermissionError:
                continue
        
        # Network traffic
        net_io = psutil.net_io_counters(pernic=True)
        for interface, stats in net_io.items():
            network_traffic.labels(
                direction='sent',
                interface=interface
            )._value._value = stats.bytes_sent
            network_traffic.labels(
                direction='received',
                interface=interface
            )._value._value = stats.bytes_recv
    
    async def _collect_application_metrics(self) -> None:
        """Collect application-specific metrics"""
        # This would integrate with the storage orchestrator
        # Example implementation:
        try:
            # Get active agents count by framework
            # In real implementation, this would query the orchestrator
            active_agents.labels(framework='langchain').set(10)
            active_agents.labels(framework='autogen').set(5)
            active_agents.labels(framework='crewai').set(3)
            
            # ML model accuracy (example values)
            ml_prediction_accuracy.labels(model_type='access_prediction').set(0.92)
            ml_prediction_accuracy.labels(model_type='tier_optimization').set(0.88)
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {str(e)}")
    
    def record_request(self, agent_id: str, operation: str, status: str,
                      tier: str, duration: float) -> None:
        """Record a storage request metric"""
        request_counter.labels(
            agent_id=agent_id,
            operation=operation,
            status=status,
            tier=tier
        ).inc()
        
        request_duration.labels(
            operation=operation,
            tier=tier
        ).observe(duration)
    
    def record_cache_hit(self, agent_id: str, hit: bool) -> None:
        """Record cache hit/miss"""
        # Update cache hit rate metrics
        cache_status = "hit" if hit else "miss"
        cache_hits.labels(
            agent_id=agent_id,
            cache_type="agent_cache",
            status=cache_status
        ).inc()
        
        # Calculate and update hit rate
        # This would be done more efficiently in production with time windows
        try:
            hit_count = cache_hits.labels(agent_id=agent_id, cache_type="agent_cache", status="hit")._value._value
            miss_count = cache_hits.labels(agent_id=agent_id, cache_type="agent_cache", status="miss")._value._value
            
            total = hit_count + miss_count
            if total > 0:
                hit_rate = (hit_count / total) * 100
                cache_hit_rate.labels(agent_id=agent_id).set(hit_rate)
        except Exception:
            # Fallback if metric access fails
            pass
    
    def record_storage_usage(self, agent_id: str, tier: str, bytes_used: int) -> None:
        """Record storage usage"""
        storage_usage.labels(
            agent_id=agent_id,
            tier=tier
        ).set(bytes_used)
    
    def record_tier_latency(self, tier: str, latency: float) -> None:
        """Record storage tier latency"""
        tier_latency.labels(tier=tier).observe(latency)
    
    def record_compression(self, agent_id: str, ratio: float) -> None:
        """Record compression ratio"""
        compression_ratio.labels(agent_id=agent_id).set(ratio)
    
    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format"""
        return generate_latest(registry)


# FastAPI integration
def create_metrics_app() -> FastAPI:
    """Create FastAPI app for metrics endpoint"""
    app = FastAPI(title="AgentVault™ Metrics")
    exporter = PrometheusExporter()
    
    @app.on_event("startup")
    async def startup():
        await exporter.start()
    
    @app.on_event("shutdown")
    async def shutdown():
        await exporter.stop()
    
    @app.get("/metrics")
    async def metrics():
        return Response(
            content=exporter.get_metrics(),
            media_type=CONTENT_TYPE_LATEST
        )
    
    @app.get("/health")
    async def health():
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    
    return app


# CLI entry point
def main():
    """Run standalone metrics server"""
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="AgentVault™ Prometheus Exporter")
    parser.add_argument("--port", type=int, default=9090, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    
    args = parser.parse_args()
    
    app = create_metrics_app()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()