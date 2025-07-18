"""
AgentVault™ Storage Tier Manager
Intelligent multi-tier storage management for optimal performance and cost

This module provides:
- Automatic data tiering based on access patterns
- Policy-based data movement between tiers
- Cost optimization through intelligent placement
- Performance monitoring and optimization
- Lifecycle management for data aging

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from collections import defaultdict

from ..core.storage_orchestrator import StorageTier, AgentStorageProfile


class TierOptimizationPolicy(Enum):
    """Storage tier optimization policies"""
    PERFORMANCE_FIRST = "performance_first"     # Prioritize lowest latency
    COST_OPTIMIZED = "cost_optimized"          # Minimize storage costs
    BALANCED = "balanced"                      # Balance performance and cost
    COMPLIANCE_DRIVEN = "compliance_driven"    # Compliance requirements first
    CUSTOM = "custom"                          # Custom policy rules


@dataclass
class TierTransition:
    """Represents a data movement between storage tiers"""
    transition_id: str
    source_tier: StorageTier
    target_tier: StorageTier
    data_size_bytes: int
    reason: str
    policy: TierOptimizationPolicy
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    error_message: Optional[str] = None


@dataclass
class TierMetrics:
    """Performance and usage metrics for a storage tier"""
    tier: StorageTier
    total_capacity_bytes: int
    used_capacity_bytes: int
    utilization_percent: float
    read_iops: int
    write_iops: int
    read_throughput_mbps: float
    write_throughput_mbps: float
    average_latency_ms: float
    object_count: int
    hot_data_percent: float  # Frequently accessed data
    cold_data_percent: float  # Rarely accessed data
    cost_per_gb_month: float
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DataAccessPattern:
    """Access pattern tracking for intelligent tiering"""
    data_id: str
    current_tier: StorageTier
    size_bytes: int
    created_at: datetime
    last_accessed_at: datetime
    access_count_24h: int = 0
    access_count_7d: int = 0
    access_count_30d: int = 0
    read_write_ratio: float = 0.5  # 0 = write-only, 1 = read-only
    average_access_size: int = 0
    access_pattern_type: str = "unknown"  # sequential, random, burst
    predicted_next_access: Optional[datetime] = None
    tier_score: float = 0.0  # Score for tier placement decision


class StorageTierManager:
    """
    Intelligent Storage Tier Manager for AgentVault™
    
    Manages data placement across multiple storage tiers to optimize for:
    - Performance requirements (latency, throughput)
    - Cost efficiency (storage costs vs access costs)
    - Compliance requirements (data residency, retention)
    - Access patterns (hot, warm, cold data)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Tier configuration
        self.tier_configs = self._initialize_tier_configs()
        self.tier_metrics: Dict[StorageTier, TierMetrics] = {}
        
        # Access pattern tracking
        self.access_patterns: Dict[str, DataAccessPattern] = {}
        self.access_history: Dict[str, List[datetime]] = defaultdict(list)
        
        # Tier transitions
        self.active_transitions: Dict[str, TierTransition] = {}
        self.transition_history: List[TierTransition] = []
        
        # Optimization settings
        self.optimization_policy = TierOptimizationPolicy(
            config.get('optimization_policy', 'balanced')
        )
        self.optimization_interval = timedelta(
            minutes=config.get('optimization_interval_minutes', 30)
        )
        self.min_data_age_for_tiering = timedelta(
            hours=config.get('min_data_age_hours', 24)
        )
        
        # Performance tracking
        self.tier_operations = defaultdict(int)
        self.optimization_runs = 0
        self.data_moved_bytes = 0
        self.cost_savings = 0.0
        
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the storage tier manager"""
        try:
            self.logger.info("Initializing Storage Tier Manager...")
            
            # Initialize tier metrics
            await self._initialize_tier_metrics()
            
            # Load existing access patterns
            await self._load_access_patterns()
            
            # Start background optimization task
            asyncio.create_task(self._run_optimization_loop())
            
            # Start metrics collection
            asyncio.create_task(self._collect_tier_metrics())
            
            self.is_initialized = True
            self.logger.info("Storage Tier Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Storage Tier Manager: {e}")
            raise
    
    async def track_data_access(self, data_id: str, tier: StorageTier,
                               size_bytes: int, operation: str) -> None:
        """Track data access for intelligent tiering decisions"""
        try:
            now = datetime.utcnow()
            
            # Update or create access pattern
            if data_id in self.access_patterns:
                pattern = self.access_patterns[data_id]
                pattern.last_accessed_at = now
                
                # Update access counts
                pattern.access_count_24h += 1
                
                # Update read/write ratio
                if operation == "read":
                    pattern.read_write_ratio = (
                        (pattern.read_write_ratio * pattern.access_count_30d + 1) /
                        (pattern.access_count_30d + 1)
                    )
                else:  # write
                    pattern.read_write_ratio = (
                        (pattern.read_write_ratio * pattern.access_count_30d) /
                        (pattern.access_count_30d + 1)
                    )
            else:
                # Create new access pattern
                pattern = DataAccessPattern(
                    data_id=data_id,
                    current_tier=tier,
                    size_bytes=size_bytes,
                    created_at=now,
                    last_accessed_at=now,
                    access_count_24h=1,
                    read_write_ratio=1.0 if operation == "read" else 0.0
                )
                self.access_patterns[data_id] = pattern
            
            # Add to access history
            self.access_history[data_id].append(now)
            
            # Predict next access time
            pattern.predicted_next_access = await self._predict_next_access(data_id)
            
            # Calculate tier score
            pattern.tier_score = await self._calculate_tier_score(pattern)
            
            # Check if immediate tier change is needed
            if await self._needs_immediate_tiering(pattern):
                await self._initiate_tier_transition(pattern)
            
        except Exception as e:
            self.logger.error(f"Failed to track data access: {e}")
    
    async def optimize_tiers(self) -> Dict[str, Any]:
        """Run tier optimization based on current policy"""
        try:
            self.logger.info("Running tier optimization...")
            start_time = datetime.utcnow()
            
            optimization_results = {
                "start_time": start_time.isoformat(),
                "policy": self.optimization_policy.value,
                "transitions_initiated": 0,
                "data_moved_bytes": 0,
                "estimated_cost_savings": 0.0,
                "errors": []
            }
            
            # Analyze all tracked data patterns
            candidates = await self._identify_tier_candidates()
            
            # Apply optimization policy
            transitions = await self._apply_optimization_policy(candidates)
            
            # Execute tier transitions
            for transition in transitions:
                try:
                    await self._execute_tier_transition(transition)
                    optimization_results["transitions_initiated"] += 1
                    optimization_results["data_moved_bytes"] += transition.data_size_bytes
                except Exception as e:
                    self.logger.error(f"Failed to execute transition: {e}")
                    optimization_results["errors"].append(str(e))
            
            # Calculate cost savings
            optimization_results["estimated_cost_savings"] = await self._calculate_cost_savings(transitions)
            
            # Update metrics
            self.optimization_runs += 1
            self.data_moved_bytes += optimization_results["data_moved_bytes"]
            self.cost_savings += optimization_results["estimated_cost_savings"]
            
            optimization_results["end_time"] = datetime.utcnow().isoformat()
            optimization_results["duration_seconds"] = (
                datetime.utcnow() - start_time
            ).total_seconds()
            
            self.logger.info(f"Tier optimization completed: {optimization_results['transitions_initiated']} transitions")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Tier optimization failed: {e}")
            return {"error": str(e)}
    
    async def get_tier_recommendations(self, data_id: str,
                                     requirements: Dict[str, Any]) -> StorageTier:
        """Get optimal tier recommendation for specific data"""
        try:
            # Check if we have access pattern history
            if data_id in self.access_patterns:
                pattern = self.access_patterns[data_id]
                
                # Use ML-based recommendation
                return await self._ml_tier_recommendation(pattern, requirements)
            else:
                # New data - use requirements-based recommendation
                return await self._requirements_based_recommendation(requirements)
            
        except Exception as e:
            self.logger.error(f"Failed to get tier recommendation: {e}")
            return StorageTier.STANDARD  # Safe default
    
    async def get_tier_metrics(self) -> Dict[StorageTier, TierMetrics]:
        """Get current metrics for all storage tiers"""
        return self.tier_metrics.copy()
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive tier optimization report"""
        try:
            # Calculate tier distribution
            tier_distribution = defaultdict(lambda: {"count": 0, "size_bytes": 0})
            
            for pattern in self.access_patterns.values():
                tier_distribution[pattern.current_tier]["count"] += 1
                tier_distribution[pattern.current_tier]["size_bytes"] += pattern.size_bytes
            
            # Calculate access pattern statistics
            hot_data_count = sum(
                1 for p in self.access_patterns.values()
                if p.access_count_24h > 10
            )
            
            cold_data_count = sum(
                1 for p in self.access_patterns.values()
                if p.access_count_30d < 5
            )
            
            # Generate report
            report = {
                "generation_time": datetime.utcnow().isoformat(),
                "optimization_policy": self.optimization_policy.value,
                "total_tracked_objects": len(self.access_patterns),
                "optimization_runs": self.optimization_runs,
                "total_data_moved_gb": self.data_moved_bytes / (1024**3),
                "estimated_cost_savings_usd": self.cost_savings,
                "tier_distribution": dict(tier_distribution),
                "access_patterns": {
                    "hot_data_count": hot_data_count,
                    "cold_data_count": cold_data_count,
                    "hot_data_percent": (hot_data_count / len(self.access_patterns) * 100
                                       if self.access_patterns else 0)
                },
                "active_transitions": len(self.active_transitions),
                "recent_transitions": [
                    {
                        "transition_id": t.transition_id,
                        "source_tier": t.source_tier.value,
                        "target_tier": t.target_tier.value,
                        "size_gb": t.data_size_bytes / (1024**3),
                        "reason": t.reason,
                        "status": t.status
                    }
                    for t in self.transition_history[-10:]  # Last 10 transitions
                ],
                "tier_metrics": {
                    tier.value: {
                        "utilization_percent": metrics.utilization_percent,
                        "average_latency_ms": metrics.average_latency_ms,
                        "cost_per_gb_month": metrics.cost_per_gb_month,
                        "hot_data_percent": metrics.hot_data_percent
                    }
                    for tier, metrics in self.tier_metrics.items()
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization report: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    def _initialize_tier_configs(self) -> Dict[StorageTier, Dict[str, Any]]:
        """Initialize configuration for each storage tier"""
        return {
            StorageTier.ULTRA: {
                "latency_ms": 0.1,
                "iops": 450000,
                "throughput_mbps": 12800,
                "cost_per_gb_month": 0.50,
                "min_size_gb": 100,
                "max_size_gb": 100000
            },
            StorageTier.PREMIUM: {
                "latency_ms": 1.0,
                "iops": 64000,
                "throughput_mbps": 1024,
                "cost_per_gb_month": 0.20,
                "min_size_gb": 100,
                "max_size_gb": 500000
            },
            StorageTier.STANDARD: {
                "latency_ms": 10.0,
                "iops": 16000,
                "throughput_mbps": 250,
                "cost_per_gb_month": 0.10,
                "min_size_gb": 100,
                "max_size_gb": 1000000
            },
            StorageTier.COOL: {
                "latency_ms": 60000.0,  # 1 minute
                "iops": 1000,
                "throughput_mbps": 50,
                "cost_per_gb_month": 0.02,
                "min_size_gb": 1000,
                "max_size_gb": 10000000
            },
            StorageTier.ARCHIVE: {
                "latency_ms": 3600000.0,  # 1 hour
                "iops": 100,
                "throughput_mbps": 10,
                "cost_per_gb_month": 0.004,
                "min_size_gb": 10000,
                "max_size_gb": 100000000
            }
        }
    
    async def _initialize_tier_metrics(self) -> None:
        """Initialize metrics for all storage tiers"""
        for tier in StorageTier:
            self.tier_metrics[tier] = TierMetrics(
                tier=tier,
                total_capacity_bytes=self.tier_configs[tier]["max_size_gb"] * 1024**3,
                used_capacity_bytes=0,
                utilization_percent=0.0,
                read_iops=0,
                write_iops=0,
                read_throughput_mbps=0.0,
                write_throughput_mbps=0.0,
                average_latency_ms=self.tier_configs[tier]["latency_ms"],
                object_count=0,
                hot_data_percent=0.0,
                cold_data_percent=0.0,
                cost_per_gb_month=self.tier_configs[tier]["cost_per_gb_month"]
            )
    
    async def _identify_tier_candidates(self) -> List[DataAccessPattern]:
        """Identify data that should be moved to different tiers"""
        candidates = []
        
        for pattern in self.access_patterns.values():
            # Skip recently created data
            data_age = datetime.utcnow() - pattern.created_at
            if data_age < self.min_data_age_for_tiering:
                continue
            
            # Calculate optimal tier based on access pattern
            optimal_tier = await self._calculate_optimal_tier(pattern)
            
            # Add to candidates if tier change would be beneficial
            if optimal_tier != pattern.current_tier:
                candidates.append(pattern)
        
        return candidates
    
    async def _calculate_optimal_tier(self, pattern: DataAccessPattern) -> StorageTier:
        """Calculate optimal storage tier for data based on access patterns"""
        # Simple heuristic - can be replaced with ML model
        
        # Ultra tier for very hot data
        if pattern.access_count_24h > 100:
            return StorageTier.ULTRA
        
        # Premium tier for hot data
        elif pattern.access_count_24h > 20:
            return StorageTier.PREMIUM
        
        # Standard tier for warm data
        elif pattern.access_count_7d > 10:
            return StorageTier.STANDARD
        
        # Cool tier for cold data
        elif pattern.access_count_30d > 1:
            return StorageTier.COOL
        
        # Archive tier for very cold data
        else:
            return StorageTier.ARCHIVE
    
    async def _apply_optimization_policy(self, 
                                       candidates: List[DataAccessPattern]) -> List[TierTransition]:
        """Apply optimization policy to determine which transitions to make"""
        transitions = []
        
        if self.optimization_policy == TierOptimizationPolicy.PERFORMANCE_FIRST:
            # Move all frequently accessed data to faster tiers
            for pattern in candidates:
                if pattern.access_count_24h > 5:
                    optimal_tier = await self._calculate_optimal_tier(pattern)
                    if optimal_tier.value < pattern.current_tier.value:  # Move to faster tier
                        transitions.append(self._create_transition(
                            pattern, optimal_tier, "Performance optimization"
                        ))
        
        elif self.optimization_policy == TierOptimizationPolicy.COST_OPTIMIZED:
            # Move infrequently accessed data to cheaper tiers
            for pattern in candidates:
                if pattern.access_count_30d < 5:
                    optimal_tier = await self._calculate_optimal_tier(pattern)
                    if optimal_tier.value > pattern.current_tier.value:  # Move to cheaper tier
                        transitions.append(self._create_transition(
                            pattern, optimal_tier, "Cost optimization"
                        ))
        
        elif self.optimization_policy == TierOptimizationPolicy.BALANCED:
            # Balance between performance and cost
            for pattern in candidates:
                optimal_tier = await self._calculate_optimal_tier(pattern)
                cost_benefit = await self._calculate_cost_benefit(
                    pattern, pattern.current_tier, optimal_tier
                )
                
                if cost_benefit > 0.2:  # 20% improvement threshold
                    transitions.append(self._create_transition(
                        pattern, optimal_tier, f"Balanced optimization (benefit: {cost_benefit:.1%})"
                    ))
        
        return transitions
    
    def _create_transition(self, pattern: DataAccessPattern,
                          target_tier: StorageTier, reason: str) -> TierTransition:
        """Create a tier transition object"""
        return TierTransition(
            transition_id=f"trans-{pattern.data_id}-{datetime.utcnow().timestamp()}",
            source_tier=pattern.current_tier,
            target_tier=target_tier,
            data_size_bytes=pattern.size_bytes,
            reason=reason,
            policy=self.optimization_policy
        )
    
    async def _calculate_cost_benefit(self, pattern: DataAccessPattern,
                                    current_tier: StorageTier,
                                    target_tier: StorageTier) -> float:
        """Calculate cost-benefit ratio of tier transition"""
        # Current monthly cost
        current_cost = (
            pattern.size_bytes / (1024**3) *
            self.tier_configs[current_tier]["cost_per_gb_month"]
        )
        
        # Target monthly cost
        target_cost = (
            pattern.size_bytes / (1024**3) *
            self.tier_configs[target_tier]["cost_per_gb_month"]
        )
        
        # Performance impact (simplified)
        latency_impact = (
            self.tier_configs[current_tier]["latency_ms"] -
            self.tier_configs[target_tier]["latency_ms"]
        ) / self.tier_configs[current_tier]["latency_ms"]
        
        # Combined benefit score
        cost_benefit = (current_cost - target_cost) / current_cost
        performance_benefit = latency_impact * pattern.access_count_24h / 100
        
        return cost_benefit + performance_benefit
    
    async def _predict_next_access(self, data_id: str) -> Optional[datetime]:
        """Predict when data will be accessed next based on history"""
        if data_id not in self.access_history or len(self.access_history[data_id]) < 2:
            return None
        
        # Simple prediction based on average interval
        access_times = sorted(self.access_history[data_id][-10:])  # Last 10 accesses
        
        if len(access_times) < 2:
            return None
        
        # Calculate average interval
        intervals = []
        for i in range(1, len(access_times)):
            interval = (access_times[i] - access_times[i-1]).total_seconds()
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Predict next access
        last_access = access_times[-1]
        predicted_time = last_access + timedelta(seconds=avg_interval)
        
        return predicted_time
    
    async def _calculate_tier_score(self, pattern: DataAccessPattern) -> float:
        """Calculate a score for tier placement decision"""
        # Factors for scoring
        recency_score = 1.0 / (1 + (datetime.utcnow() - pattern.last_accessed_at).days)
        frequency_score = min(pattern.access_count_24h / 100, 1.0)
        size_score = 1.0 / (1 + pattern.size_bytes / (1024**3))  # Favor smaller objects
        
        # Weighted combination
        score = (
            recency_score * 0.4 +
            frequency_score * 0.4 +
            size_score * 0.2
        )
        
        return score
    
    async def _needs_immediate_tiering(self, pattern: DataAccessPattern) -> bool:
        """Check if data needs immediate tier change"""
        # Hot data in slow tier
        if pattern.access_count_24h > 50 and pattern.current_tier in [StorageTier.COOL, StorageTier.ARCHIVE]:
            return True
        
        # Very cold data in expensive tier
        if pattern.access_count_30d == 0 and pattern.current_tier in [StorageTier.ULTRA, StorageTier.PREMIUM]:
            data_age = datetime.utcnow() - pattern.created_at
            if data_age > timedelta(days=7):
                return True
        
        return False
    
    async def _execute_tier_transition(self, transition: TierTransition) -> None:
        """Execute a tier transition"""
        transition.status = "in_progress"
        self.active_transitions[transition.transition_id] = transition
        
        try:
            # Log transition start
            self.logger.info(f"Starting tier transition: {transition.transition_id}")
            
            # TODO: Actual data movement implementation would go here
            # This would integrate with ANFStorageManager to move data
            
            # Simulate transition (in production, this would be actual data movement)
            await asyncio.sleep(0.1)
            
            # Update pattern
            pattern = self.access_patterns.get(
                transition.transition_id.split('-')[1]  # Extract data_id
            )
            if pattern:
                pattern.current_tier = transition.target_tier
            
            # Mark transition complete
            transition.status = "completed"
            transition.completed_at = datetime.utcnow()
            
            # Update metrics
            self.tier_operations[f"{transition.source_tier.value}_to_{transition.target_tier.value}"] += 1
            
            self.logger.info(f"Tier transition completed: {transition.transition_id}")
            
        except Exception as e:
            transition.status = "failed"
            transition.error_message = str(e)
            self.logger.error(f"Tier transition failed: {e}")
            raise
        
        finally:
            # Move to history
            self.transition_history.append(transition)
            del self.active_transitions[transition.transition_id]
    
    async def _run_optimization_loop(self) -> None:
        """Background task for continuous tier optimization"""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval.total_seconds())
                
                # Run optimization
                await self.optimize_tiers()
                
                # Clean up old access history
                await self._cleanup_old_access_history()
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
    
    async def _collect_tier_metrics(self) -> None:
        """Background task for collecting tier metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect every minute
                
                # Update tier metrics
                for tier in StorageTier:
                    metrics = self.tier_metrics[tier]
                    
                    # Calculate utilization from tracked patterns
                    tier_data = [
                        p for p in self.access_patterns.values()
                        if p.current_tier == tier
                    ]
                    
                    metrics.object_count = len(tier_data)
                    metrics.used_capacity_bytes = sum(p.size_bytes for p in tier_data)
                    metrics.utilization_percent = (
                        metrics.used_capacity_bytes / metrics.total_capacity_bytes * 100
                        if metrics.total_capacity_bytes > 0 else 0
                    )
                    
                    # Calculate hot/cold data percentages
                    hot_count = sum(1 for p in tier_data if p.access_count_24h > 10)
                    cold_count = sum(1 for p in tier_data if p.access_count_30d < 5)
                    
                    metrics.hot_data_percent = (
                        hot_count / len(tier_data) * 100 if tier_data else 0
                    )
                    metrics.cold_data_percent = (
                        cold_count / len(tier_data) * 100 if tier_data else 0
                    )
                    
                    metrics.last_updated = datetime.utcnow()
                
            except Exception as e:
                self.logger.error(f"Error collecting tier metrics: {e}")
    
    async def _cleanup_old_access_history(self) -> None:
        """Clean up old access history to prevent memory bloat"""
        cutoff_time = datetime.utcnow() - timedelta(days=90)
        
        for data_id, access_times in self.access_history.items():
            # Keep only recent accesses
            self.access_history[data_id] = [
                t for t in access_times if t > cutoff_time
            ]
            
            # Remove empty entries
            if not self.access_history[data_id]:
                del self.access_history[data_id]
                
                # Also remove from patterns if very old
                if data_id in self.access_patterns:
                    pattern = self.access_patterns[data_id]
                    if pattern.last_accessed_at < cutoff_time:
                        del self.access_patterns[data_id]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("agentvault.tier_manager")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def shutdown(self) -> None:
        """Shutdown the tier manager"""
        self.logger.info("Shutting down Storage Tier Manager...")
        
        # Complete any active transitions
        for transition in self.active_transitions.values():
            if transition.status == "in_progress":
                transition.status = "cancelled"
                self.transition_history.append(transition)
        
        self.logger.info("Storage Tier Manager shutdown complete")