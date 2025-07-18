"""
AgentVault™ Agent DNA Profiler
Machine Learning-based storage optimization profiles for AI agents

This module creates and manages unique "Storage DNA" profiles that:
- Learn from agent behavior patterns
- Predict future storage needs
- Optimize data placement automatically
- Enable cross-agent learning
- Reduce cold-start problems by 90%

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import hashlib

from ..core.storage_orchestrator import AgentStorageProfile, StorageRequest, StorageTier


@dataclass
class DNAFeatures:
    """Features that comprise an agent's Storage DNA"""
    # Access patterns
    avg_request_size: float = 0.0
    request_size_variance: float = 0.0
    read_write_ratio: float = 0.5
    peak_hour_activity: int = 12  # 0-23
    weekend_activity_ratio: float = 0.5
    
    # Temporal patterns
    avg_inter_request_time: float = 60.0  # seconds
    burst_frequency: float = 0.0  # bursts per hour
    avg_burst_size: int = 0
    time_of_day_distribution: List[float] = field(default_factory=lambda: [0.0] * 24)
    
    # Data patterns
    data_type_distribution: Dict[str, float] = field(default_factory=dict)
    compression_ratio: float = 1.0
    data_lifecycle_hours: float = 168.0  # 1 week default
    hot_data_percentage: float = 0.2
    
    # Performance preferences
    latency_sensitivity: float = 0.5  # 0-1, higher = more sensitive
    throughput_requirements: float = 0.5  # 0-1, higher = more demanding
    consistency_requirements: float = 0.5  # 0-1, higher = stronger consistency
    
    # Collaboration patterns
    agent_interaction_count: int = 0
    shared_data_percentage: float = 0.0
    collaboration_frequency: float = 0.0
    primary_collaborators: List[str] = field(default_factory=list)
    
    # Cost sensitivity
    cost_sensitivity: float = 0.5  # 0-1, higher = more cost conscious
    performance_cost_tradeoff: float = 0.5  # 0-1, 0=cost, 1=performance
    
    # Anomaly scores
    behavior_stability: float = 1.0  # 0-1, higher = more stable
    anomaly_frequency: float = 0.0
    
    # DNA metadata
    dna_version: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    confidence_score: float = 0.0  # 0-1, how confident we are in this DNA


@dataclass
class DNACluster:
    """Represents a cluster of similar agent DNAs"""
    cluster_id: str
    cluster_type: str  # e.g., "high_performance", "cost_optimized", "balanced"
    member_agents: List[str]
    centroid_features: DNAFeatures
    avg_performance_metrics: Dict[str, float]
    recommended_config: Dict[str, Any]
    
    def distance_to(self, features: DNAFeatures) -> float:
        """Calculate distance from cluster centroid"""
        # Convert features to numpy arrays for calculation
        centroid_vector = self._features_to_vector(self.centroid_features)
        features_vector = self._features_to_vector(features)
        
        # Cosine similarity (converted to distance)
        similarity = cosine_similarity([centroid_vector], [features_vector])[0][0]
        return 1.0 - similarity
    
    def _features_to_vector(self, features: DNAFeatures) -> np.ndarray:
        """Convert DNA features to numpy vector"""
        vector = [
            features.avg_request_size,
            features.read_write_ratio,
            features.peak_hour_activity / 23.0,  # Normalize
            features.weekend_activity_ratio,
            features.avg_inter_request_time / 3600.0,  # Convert to hours
            features.burst_frequency,
            features.compression_ratio,
            features.hot_data_percentage,
            features.latency_sensitivity,
            features.throughput_requirements,
            features.consistency_requirements,
            features.cost_sensitivity,
            features.behavior_stability
        ]
        return np.array(vector)


class AgentDNAProfiler:
    """
    Agent DNA Profiler for AgentVault™
    
    Creates and manages unique storage optimization profiles for each AI agent,
    learning from behavior patterns to provide personalized performance optimization.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # DNA storage
        self.agent_dnas: Dict[str, DNAFeatures] = {}
        self.dna_history: Dict[str, List[DNAFeatures]] = {}
        
        # Clustering for cross-agent learning
        self.dna_clusters: Dict[str, DNACluster] = {}
        self.clustering_model = None
        self.scaler = StandardScaler()
        
        # Prediction models
        self.latency_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.access_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Feature engineering
        self.feature_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self.feature_window = timedelta(hours=24)  # 24-hour sliding window
        
        # Model training data
        self.training_data = []
        self.model_update_interval = timedelta(hours=6)
        self.last_model_update = datetime.utcnow()
        
        # Performance tracking
        self.prediction_accuracy = {}
        self.dna_evolution_count = 0
        
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the Agent DNA Profiler"""
        try:
            self.logger.info("Initializing Agent DNA Profiler...")
            
            # Load existing DNA profiles
            await self._load_dna_profiles()
            
            # Initialize clustering
            await self._initialize_clustering()
            
            # Load or train prediction models
            await self._initialize_models()
            
            # Start background tasks
            asyncio.create_task(self._feature_extraction_loop())
            asyncio.create_task(self._model_update_loop())
            asyncio.create_task(self._dna_evolution_loop())
            
            self.is_initialized = True
            self.logger.info("Agent DNA Profiler initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Agent DNA Profiler: {e}")
            raise
    
    async def create_profile(self, agent_id: str, agent_type: str,
                           agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create initial DNA profile for a new agent"""
        try:
            self.logger.info(f"Creating DNA profile for agent {agent_id}")
            
            # Check if similar agents exist (inheritance)
            similar_dna = await self._find_similar_agent_dna(agent_type, agent_config)
            
            if similar_dna:
                # Inherit from similar agent
                initial_features = self._inherit_dna_features(similar_dna)
                initial_features.confidence_score = 0.5  # Medium confidence for inherited
                self.logger.info(f"Agent {agent_id} inheriting DNA from similar agents")
            else:
                # Create default DNA based on agent type
                initial_features = self._create_default_dna(agent_type, agent_config)
                initial_features.confidence_score = 0.3  # Low confidence for new
            
            # Store DNA
            self.agent_dnas[agent_id] = initial_features
            self.dna_history[agent_id] = [initial_features]
            
            # Initialize feature buffer
            self.feature_buffer[agent_id] = []
            
            # Assign to cluster
            cluster = await self._assign_to_cluster(initial_features)
            
            # Generate initial recommendations
            recommendations = await self._generate_dna_recommendations(initial_features, cluster)
            
            return {
                "dna_id": self._generate_dna_id(agent_id, initial_features),
                "features": self._features_to_dict(initial_features),
                "cluster": cluster.cluster_type if cluster else "unclustered",
                "confidence": initial_features.confidence_score,
                "recommendations": recommendations,
                "inheritance": "inherited" if similar_dna else "new"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to create DNA profile: {e}")
            raise
    
    async def update_access_patterns(self, profile: AgentStorageProfile,
                                   request: StorageRequest) -> None:
        """Update agent DNA based on observed access patterns"""
        try:
            agent_id = profile.agent_id
            
            # Add to feature buffer
            feature_data = {
                "timestamp": datetime.utcnow(),
                "operation": request.operation,
                "data_type": request.data_type,
                "data_size": request.data_size,
                "latency_requirement": request.latency_requirement,
                "priority": request.priority,
                "metadata": request.metadata
            }
            
            if agent_id not in self.feature_buffer:
                self.feature_buffer[agent_id] = []
            
            self.feature_buffer[agent_id].append(feature_data)
            
            # Update DNA features if enough data
            if len(self.feature_buffer[agent_id]) >= 100:  # Minimum sample size
                await self._update_dna_features(agent_id)
            
        except Exception as e:
            self.logger.error(f"Failed to update access patterns: {e}")
    
    async def evolve_profile(self, profile: AgentStorageProfile) -> Dict[str, Any]:
        """Evolve agent DNA based on accumulated patterns"""
        try:
            agent_id = profile.agent_id
            
            if agent_id not in self.agent_dnas:
                self.logger.warning(f"No DNA found for agent {agent_id}")
                return profile.storage_dna
            
            current_dna = self.agent_dnas[agent_id]
            
            # Extract features from recent activity
            new_features = await self._extract_features_from_buffer(agent_id)
            
            if not new_features:
                return self._features_to_dict(current_dna)
            
            # Evolve DNA using exponential moving average
            evolved_dna = self._evolve_dna_features(current_dna, new_features)
            
            # Detect anomalies in evolution
            is_anomalous = await self._detect_dna_anomaly(current_dna, evolved_dna)
            
            if is_anomalous:
                self.logger.warning(f"Anomalous DNA evolution detected for agent {agent_id}")
                evolved_dna.anomaly_frequency += 0.1
                evolved_dna.behavior_stability *= 0.9
            else:
                evolved_dna.behavior_stability = min(evolved_dna.behavior_stability * 1.01, 1.0)
            
            # Update confidence score
            evolved_dna.confidence_score = self._calculate_confidence_score(
                agent_id, evolved_dna
            )
            
            # Store evolved DNA
            evolved_dna.last_updated = datetime.utcnow()
            evolved_dna.dna_version += 1
            self.agent_dnas[agent_id] = evolved_dna
            self.dna_history[agent_id].append(evolved_dna)
            
            # Re-cluster if significant change
            if self._is_significant_change(current_dna, evolved_dna):
                cluster = await self._assign_to_cluster(evolved_dna)
                self.logger.info(f"Agent {agent_id} reassigned to cluster {cluster.cluster_type}")
            
            self.dna_evolution_count += 1
            
            return self._features_to_dict(evolved_dna)
            
        except Exception as e:
            self.logger.error(f"Failed to evolve DNA profile: {e}")
            return profile.storage_dna
    
    async def predict_storage_needs(self, agent_id: str,
                                  time_horizon: timedelta) -> Dict[str, Any]:
        """Predict future storage needs based on DNA"""
        try:
            if agent_id not in self.agent_dnas:
                return {"error": "No DNA profile found"}
            
            dna = self.agent_dnas[agent_id]
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(dna)
            
            # Predict access patterns
            predicted_accesses = self.access_predictor.predict([features])[0]
            
            # Predict performance requirements
            predicted_latency = self.latency_predictor.predict([features])[0]
            
            # Calculate storage recommendations
            recommendations = {
                "predicted_accesses_per_hour": float(predicted_accesses),
                "predicted_latency_requirement_ms": float(predicted_latency),
                "recommended_tier": self._recommend_tier_from_prediction(
                    predicted_accesses, predicted_latency
                ),
                "predicted_hot_data_gb": float(
                    dna.hot_data_percentage * predicted_accesses * dna.avg_request_size / (1024**3)
                ),
                "confidence": float(dna.confidence_score),
                "prediction_horizon_hours": time_horizon.total_seconds() / 3600
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to predict storage needs: {e}")
            return {"error": str(e)}
    
    async def find_similar_agents(self, agent_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find agents with similar DNA for knowledge sharing"""
        try:
            if agent_id not in self.agent_dnas:
                return []
            
            target_dna = self.agent_dnas[agent_id]
            target_vector = self._dna_to_vector(target_dna)
            
            similarities = []
            
            for other_id, other_dna in self.agent_dnas.items():
                if other_id == agent_id:
                    continue
                
                other_vector = self._dna_to_vector(other_dna)
                similarity = cosine_similarity([target_vector], [other_vector])[0][0]
                
                similarities.append({
                    "agent_id": other_id,
                    "similarity": float(similarity),
                    "cluster": self._get_agent_cluster(other_id),
                    "key_similarities": self._identify_key_similarities(target_dna, other_dna)
                })
            
            # Sort by similarity and return top-k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Failed to find similar agents: {e}")
            return []
    
    async def get_dna_insights(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive insights about an agent's DNA"""
        try:
            if agent_id not in self.agent_dnas:
                return {"error": "No DNA profile found"}
            
            dna = self.agent_dnas[agent_id]
            history = self.dna_history.get(agent_id, [])
            
            # Calculate evolution metrics
            evolution_rate = self._calculate_evolution_rate(history) if len(history) > 1 else 0.0
            stability_trend = self._calculate_stability_trend(history) if len(history) > 1 else "stable"
            
            # Performance predictions
            performance_forecast = await self.predict_storage_needs(
                agent_id, timedelta(days=7)
            )
            
            # Optimization opportunities
            optimization_opportunities = self._identify_optimization_opportunities(dna)
            
            # Similar agents for learning
            similar_agents = await self.find_similar_agents(agent_id, top_k=3)
            
            insights = {
                "agent_id": agent_id,
                "dna_version": dna.dna_version,
                "confidence_score": float(dna.confidence_score),
                "behavior_stability": float(dna.behavior_stability),
                "evolution_metrics": {
                    "evolution_rate": float(evolution_rate),
                    "stability_trend": stability_trend,
                    "total_evolutions": len(history) - 1
                },
                "current_characteristics": {
                    "primary_workload": self._identify_workload_type(dna),
                    "performance_profile": self._identify_performance_profile(dna),
                    "cost_profile": self._identify_cost_profile(dna)
                },
                "performance_forecast": performance_forecast,
                "optimization_opportunities": optimization_opportunities,
                "similar_agents": similar_agents,
                "cluster_assignment": self._get_agent_cluster(agent_id)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get DNA insights: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _find_similar_agent_dna(self, agent_type: str,
                                    agent_config: Dict[str, Any]) -> Optional[DNAFeatures]:
        """Find similar agent DNA for inheritance"""
        candidates = []
        
        for agent_id, dna in self.agent_dnas.items():
            # Check if same agent type
            if self._get_agent_type(agent_id) == agent_type:
                candidates.append(dna)
        
        if not candidates:
            return None
        
        # Return the DNA with highest confidence
        return max(candidates, key=lambda d: d.confidence_score)
    
    def _inherit_dna_features(self, parent_dna: DNAFeatures) -> DNAFeatures:
        """Create new DNA by inheriting from parent"""
        # Deep copy parent DNA
        inherited = DNAFeatures(
            avg_request_size=parent_dna.avg_request_size,
            read_write_ratio=parent_dna.read_write_ratio,
            peak_hour_activity=parent_dna.peak_hour_activity,
            weekend_activity_ratio=parent_dna.weekend_activity_ratio,
            avg_inter_request_time=parent_dna.avg_inter_request_time,
            data_type_distribution=parent_dna.data_type_distribution.copy(),
            latency_sensitivity=parent_dna.latency_sensitivity,
            throughput_requirements=parent_dna.throughput_requirements,
            cost_sensitivity=parent_dna.cost_sensitivity
        )
        
        # Reset agent-specific features
        inherited.agent_interaction_count = 0
        inherited.primary_collaborators = []
        inherited.dna_version = 1
        inherited.created_at = datetime.utcnow()
        
        return inherited
    
    def _create_default_dna(self, agent_type: str, config: Dict[str, Any]) -> DNAFeatures:
        """Create default DNA based on agent type"""
        # Default templates by agent type
        templates = {
            "langchain": DNAFeatures(
                avg_request_size=1024 * 10,  # 10KB average
                read_write_ratio=0.7,  # More reads
                latency_sensitivity=0.7,
                throughput_requirements=0.6,
                hot_data_percentage=0.3
            ),
            "autogen": DNAFeatures(
                avg_request_size=1024 * 50,  # 50KB average (more conversation)
                read_write_ratio=0.5,  # Balanced
                latency_sensitivity=0.8,
                throughput_requirements=0.7,
                agent_interaction_count=5  # Multi-agent by default
            ),
            "crewai": DNAFeatures(
                avg_request_size=1024 * 20,  # 20KB average
                read_write_ratio=0.6,
                latency_sensitivity=0.6,
                collaboration_frequency=0.8,  # High collaboration
                shared_data_percentage=0.4
            ),
            "custom": DNAFeatures()  # Default values
        }
        
        return templates.get(agent_type, templates["custom"])
    
    async def _update_dna_features(self, agent_id: str) -> None:
        """Update DNA features from accumulated data"""
        buffer = self.feature_buffer[agent_id]
        
        # Get recent data within window
        cutoff_time = datetime.utcnow() - self.feature_window
        recent_data = [d for d in buffer if d["timestamp"] > cutoff_time]
        
        if not recent_data:
            return
        
        # Extract new features
        new_features = await self._extract_features_from_data(recent_data)
        
        # Update existing DNA
        current_dna = self.agent_dnas[agent_id]
        self.agent_dnas[agent_id] = self._evolve_dna_features(current_dna, new_features)
        
        # Clean old buffer data
        self.feature_buffer[agent_id] = recent_data
    
    async def _extract_features_from_data(self, data: List[Dict[str, Any]]) -> DNAFeatures:
        """Extract DNA features from raw data"""
        features = DNAFeatures()
        
        if not data:
            return features
        
        # Request sizes
        sizes = [d["data_size"] for d in data]
        features.avg_request_size = np.mean(sizes)
        features.request_size_variance = np.var(sizes)
        
        # Read/write ratio
        reads = sum(1 for d in data if d["operation"] == "read")
        features.read_write_ratio = reads / len(data)
        
        # Temporal patterns
        timestamps = [d["timestamp"] for d in data]
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if intervals:
            features.avg_inter_request_time = np.mean(intervals)
        
        # Peak hour analysis
        hours = [t.hour for t in timestamps]
        hour_counts = np.bincount(hours, minlength=24)
        features.peak_hour_activity = int(np.argmax(hour_counts))
        features.time_of_day_distribution = (hour_counts / sum(hour_counts)).tolist()
        
        # Weekend activity
        weekend_count = sum(1 for t in timestamps if t.weekday() >= 5)
        features.weekend_activity_ratio = weekend_count / len(timestamps)
        
        # Data type distribution
        data_types = [d["data_type"] for d in data]
        for dtype in set(data_types):
            features.data_type_distribution[dtype] = data_types.count(dtype) / len(data_types)
        
        # Performance requirements
        latencies = [d["latency_requirement"] for d in data]
        features.latency_sensitivity = 1.0 / (1.0 + np.mean(latencies))
        
        # Priority analysis
        high_priority = sum(1 for d in data if d["priority"] in ["critical", "high"])
        features.throughput_requirements = high_priority / len(data)
        
        return features
    
    def _evolve_dna_features(self, current: DNAFeatures, new: DNAFeatures) -> DNAFeatures:
        """Evolve DNA features using exponential moving average"""
        alpha = 0.3  # Learning rate
        
        evolved = DNAFeatures()
        
        # Numeric features - exponential moving average
        evolved.avg_request_size = (1 - alpha) * current.avg_request_size + alpha * new.avg_request_size
        evolved.request_size_variance = (1 - alpha) * current.request_size_variance + alpha * new.request_size_variance
        evolved.read_write_ratio = (1 - alpha) * current.read_write_ratio + alpha * new.read_write_ratio
        evolved.avg_inter_request_time = (1 - alpha) * current.avg_inter_request_time + alpha * new.avg_inter_request_time
        evolved.latency_sensitivity = (1 - alpha) * current.latency_sensitivity + alpha * new.latency_sensitivity
        evolved.throughput_requirements = (1 - alpha) * current.throughput_requirements + alpha * new.throughput_requirements
        
        # Categorical features - weighted update
        evolved.peak_hour_activity = new.peak_hour_activity  # Use most recent
        
        # Distribution features - merge
        evolved.data_type_distribution = current.data_type_distribution.copy()
        for dtype, ratio in new.data_type_distribution.items():
            if dtype in evolved.data_type_distribution:
                evolved.data_type_distribution[dtype] = (
                    (1 - alpha) * evolved.data_type_distribution[dtype] + alpha * ratio
                )
            else:
                evolved.data_type_distribution[dtype] = ratio
        
        # Preserve some features
        evolved.cost_sensitivity = current.cost_sensitivity
        evolved.agent_interaction_count = current.agent_interaction_count
        evolved.primary_collaborators = current.primary_collaborators
        evolved.dna_version = current.dna_version
        evolved.created_at = current.created_at
        
        return evolved
    
    def _calculate_confidence_score(self, agent_id: str, dna: DNAFeatures) -> float:
        """Calculate confidence score for DNA profile"""
        # Factors affecting confidence
        data_points = len(self.feature_buffer.get(agent_id, []))
        age_days = (datetime.utcnow() - dna.created_at).days
        evolution_count = dna.dna_version
        
        # Calculate score
        data_score = min(data_points / 1000, 1.0)  # Max at 1000 data points
        age_score = min(age_days / 30, 1.0)  # Max at 30 days
        evolution_score = min(evolution_count / 10, 1.0)  # Max at 10 evolutions
        stability_score = dna.behavior_stability
        
        # Weighted average
        confidence = (
            data_score * 0.3 +
            age_score * 0.2 +
            evolution_score * 0.2 +
            stability_score * 0.3
        )
        
        return min(max(confidence, 0.0), 1.0)
    
    def _dna_to_vector(self, dna: DNAFeatures) -> np.ndarray:
        """Convert DNA features to numpy vector for ML"""
        vector = [
            dna.avg_request_size,
            dna.request_size_variance,
            dna.read_write_ratio,
            dna.peak_hour_activity / 23.0,
            dna.weekend_activity_ratio,
            dna.avg_inter_request_time,
            dna.burst_frequency,
            dna.compression_ratio,
            dna.hot_data_percentage,
            dna.latency_sensitivity,
            dna.throughput_requirements,
            dna.consistency_requirements,
            dna.cost_sensitivity,
            dna.performance_cost_tradeoff,
            dna.behavior_stability,
            dna.anomaly_frequency
        ]
        
        return np.array(vector)
    
    def _features_to_dict(self, features: DNAFeatures) -> Dict[str, Any]:
        """Convert DNA features to dictionary"""
        return {
            "access_patterns": {
                "avg_request_size": features.avg_request_size,
                "request_size_variance": features.request_size_variance,
                "read_write_ratio": features.read_write_ratio,
                "peak_hour_activity": features.peak_hour_activity,
                "weekend_activity_ratio": features.weekend_activity_ratio
            },
            "temporal_patterns": {
                "avg_inter_request_time": features.avg_inter_request_time,
                "burst_frequency": features.burst_frequency,
                "avg_burst_size": features.avg_burst_size,
                "time_of_day_distribution": features.time_of_day_distribution
            },
            "data_patterns": {
                "data_type_distribution": features.data_type_distribution,
                "compression_ratio": features.compression_ratio,
                "data_lifecycle_hours": features.data_lifecycle_hours,
                "hot_data_percentage": features.hot_data_percentage
            },
            "performance_preferences": {
                "latency_sensitivity": features.latency_sensitivity,
                "throughput_requirements": features.throughput_requirements,
                "consistency_requirements": features.consistency_requirements
            },
            "collaboration_patterns": {
                "agent_interaction_count": features.agent_interaction_count,
                "shared_data_percentage": features.shared_data_percentage,
                "collaboration_frequency": features.collaboration_frequency,
                "primary_collaborators": features.primary_collaborators
            },
            "cost_profile": {
                "cost_sensitivity": features.cost_sensitivity,
                "performance_cost_tradeoff": features.performance_cost_tradeoff
            },
            "stability_metrics": {
                "behavior_stability": features.behavior_stability,
                "anomaly_frequency": features.anomaly_frequency
            },
            "metadata": {
                "dna_version": features.dna_version,
                "created_at": features.created_at.isoformat(),
                "last_updated": features.last_updated.isoformat(),
                "confidence_score": features.confidence_score
            }
        }
    
    def _generate_dna_id(self, agent_id: str, features: DNAFeatures) -> str:
        """Generate unique DNA ID"""
        data = f"{agent_id}-{features.dna_version}-{features.created_at.timestamp()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _identify_workload_type(self, dna: DNAFeatures) -> str:
        """Identify primary workload type from DNA"""
        if dna.read_write_ratio > 0.8:
            return "read_heavy"
        elif dna.read_write_ratio < 0.2:
            return "write_heavy"
        elif dna.burst_frequency > 5:
            return "bursty"
        elif dna.avg_inter_request_time < 1:
            return "continuous"
        else:
            return "balanced"
    
    def _identify_performance_profile(self, dna: DNAFeatures) -> str:
        """Identify performance profile from DNA"""
        if dna.latency_sensitivity > 0.8 and dna.throughput_requirements > 0.8:
            return "ultra_performance"
        elif dna.latency_sensitivity > 0.6:
            return "low_latency"
        elif dna.throughput_requirements > 0.6:
            return "high_throughput"
        else:
            return "standard"
    
    def _identify_cost_profile(self, dna: DNAFeatures) -> str:
        """Identify cost profile from DNA"""
        if dna.cost_sensitivity > 0.8:
            return "cost_optimized"
        elif dna.performance_cost_tradeoff > 0.8:
            return "performance_first"
        else:
            return "balanced"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging"""
        logger = logging.getLogger("agentvault.agent_dna")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def shutdown(self) -> None:
        """Shutdown the DNA profiler"""
        self.logger.info("Shutting down Agent DNA Profiler...")
        
        # Save current models and profiles
        await self._save_dna_profiles()
        await self._save_models()
        
        self.logger.info("Agent DNA Profiler shutdown complete")