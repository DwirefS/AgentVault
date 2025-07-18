"""
AgentVault™ Advanced Agent DNA Profiling
Production-ready ML models for intelligent storage optimization
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import joblib
import json
import logging
from collections import defaultdict, deque
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers, models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """ML model types for different predictions"""
    ACCESS_PREDICTION = "access_prediction"
    TIER_OPTIMIZATION = "tier_optimization"
    ANOMALY_DETECTION = "anomaly_detection"
    CAPACITY_FORECAST = "capacity_forecast"
    PERFORMANCE_PREDICTION = "performance_prediction"


class AgentBehavior(Enum):
    """Agent behavior patterns"""
    SEQUENTIAL = "sequential"  # Accesses data in order
    RANDOM = "random"  # Random access pattern
    TEMPORAL = "temporal"  # Time-based access
    SPATIAL = "spatial"  # Location-based access
    HYBRID = "hybrid"  # Mixed patterns


@dataclass
class EnhancedDNAFeatures:
    """Enhanced feature set for agent profiling"""
    # Basic features
    avg_request_size: float
    read_write_ratio: float
    peak_hour_activity: int
    latency_sensitivity: float
    access_frequency: float
    data_locality: float
    temporal_pattern: str
    compression_ratio: float
    cache_hit_rate: float
    concurrent_access: float
    
    # Advanced features
    request_size_variance: float
    access_pattern_entropy: float
    burst_intensity: float
    data_correlation: float
    query_complexity: float
    resource_utilization: float
    error_rate: float
    retry_pattern: float
    
    # Behavioral features
    behavior_type: AgentBehavior
    learning_rate: float  # How quickly agent adapts
    consistency_score: float  # How predictable the agent is
    
    # Performance features
    throughput_mbps: float
    iops_achieved: int
    queue_depth_avg: float
    
    # Cost features
    storage_cost_sensitivity: float
    compute_cost_sensitivity: float
    
    # Compliance features
    data_retention_days: int
    encryption_required: bool
    audit_frequency: int


@dataclass
class PredictionResult:
    """ML prediction result with confidence"""
    prediction: Any
    confidence: float
    reasoning: str
    feature_importance: Dict[str, float]
    alternatives: List[Tuple[Any, float]]  # Alternative predictions with confidence


@dataclass
class AgentCluster:
    """Cluster of similar agents"""
    cluster_id: int
    agents: List[str]
    centroid_features: EnhancedDNAFeatures
    variance: float
    representative_agent: str


class AdvancedAgentDNAProfiler:
    """
    Advanced ML-based agent profiling with:
    - Multiple specialized models for different predictions
    - Deep learning for complex pattern recognition
    - Online learning for continuous adaptation
    - Ensemble methods for robust predictions
    - Explainable AI for transparency
    - Transfer learning between similar agents
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Model configuration
        self.model_config = {
            ModelType.ACCESS_PREDICTION: {
                'type': 'lstm',
                'input_dim': 50,
                'hidden_dim': 128,
                'output_dim': 10,
                'learning_rate': 0.001
            },
            ModelType.TIER_OPTIMIZATION: {
                'type': 'gradient_boost',
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1
            },
            ModelType.ANOMALY_DETECTION: {
                'type': 'autoencoder',
                'encoding_dim': 32,
                'threshold': 0.05
            },
            ModelType.CAPACITY_FORECAST: {
                'type': 'prophet',
                'growth': 'linear',
                'seasonality_mode': 'multiplicative'
            },
            ModelType.PERFORMANCE_PREDICTION: {
                'type': 'neural_network',
                'layers': [128, 64, 32],
                'activation': 'relu'
            }
        }
        
        # Model storage
        self.models: Dict[ModelType, Any] = {}
        self.scalers: Dict[ModelType, StandardScaler] = {}
        self.model_versions: Dict[ModelType, int] = defaultdict(int)
        
        # Agent profiles
        self.agent_profiles: Dict[str, EnhancedDNAFeatures] = {}
        self.agent_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.agent_clusters: List[AgentCluster] = []
        
        # Feature engineering
        self.feature_extractors: Dict[str, Any] = {}
        self.feature_importance: Dict[ModelType, Dict[str, float]] = defaultdict(dict)
        
        # Online learning
        self.online_buffer: Dict[str, List[Any]] = defaultdict(list)
        self.update_frequency = config.get('update_frequency', 3600)  # 1 hour
        
        # Performance tracking
        self.prediction_accuracy: Dict[ModelType, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_metrics: Dict[ModelType, Dict[str, float]] = defaultdict(dict)
        
        # Background tasks
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> None:
        """Initialize ML models and load pre-trained weights"""
        logger.info("Initializing Advanced Agent DNA Profiler...")
        
        try:
            # Initialize models
            await self._initialize_models()
            
            # Load pre-trained models if available
            await self._load_pretrained_models()
            
            # Initialize feature extractors
            self._initialize_feature_extractors()
            
            # Start background tasks
            self._running = True
            self._start_background_tasks()
            
            logger.info("Advanced Agent DNA Profiler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize profiler: {str(e)}")
            raise
    
    async def _initialize_models(self) -> None:
        """Initialize all ML models"""
        # Access Prediction Model (LSTM)
        self.models[ModelType.ACCESS_PREDICTION] = self._build_lstm_model(
            self.model_config[ModelType.ACCESS_PREDICTION]
        )
        
        # Tier Optimization Model (Gradient Boosting)
        self.models[ModelType.TIER_OPTIMIZATION] = GradientBoostingClassifier(
            n_estimators=self.model_config[ModelType.TIER_OPTIMIZATION]['n_estimators'],
            max_depth=self.model_config[ModelType.TIER_OPTIMIZATION]['max_depth'],
            learning_rate=self.model_config[ModelType.TIER_OPTIMIZATION]['learning_rate']
        )
        
        # Anomaly Detection Model (Autoencoder)
        self.models[ModelType.ANOMALY_DETECTION] = self._build_autoencoder(
            self.model_config[ModelType.ANOMALY_DETECTION]
        )
        
        # Capacity Forecast Model (Custom implementation)
        self.models[ModelType.CAPACITY_FORECAST] = self._build_forecast_model(
            self.model_config[ModelType.CAPACITY_FORECAST]
        )
        
        # Performance Prediction Model (Neural Network)
        self.models[ModelType.PERFORMANCE_PREDICTION] = self._build_neural_network(
            self.model_config[ModelType.PERFORMANCE_PREDICTION]
        )
        
        # Initialize scalers
        for model_type in ModelType:
            self.scalers[model_type] = StandardScaler()
    
    def _build_lstm_model(self, config: Dict[str, Any]) -> tf.keras.Model:
        """Build LSTM model for access prediction"""
        model = models.Sequential([
            layers.LSTM(config['hidden_dim'], return_sequences=True, 
                       input_shape=(None, config['input_dim'])),
            layers.Dropout(0.2),
            layers.LSTM(config['hidden_dim'] // 2),
            layers.Dropout(0.2),
            layers.Dense(config['output_dim'], activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_autoencoder(self, config: Dict[str, Any]) -> tf.keras.Model:
        """Build autoencoder for anomaly detection"""
        input_dim = 50  # Feature dimension
        encoding_dim = config['encoding_dim']
        
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(encoding_dim * 2, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        autoencoder = models.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Also create encoder model for feature extraction
        self.encoder = models.Model(input_layer, encoded)
        
        return autoencoder
    
    def _build_neural_network(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Build PyTorch neural network for performance prediction"""
        
        class PerformanceNet(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim=1):
                super(PerformanceNet, self).__init__()
                
                layers_list = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers_list.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim
                
                layers_list.append(nn.Linear(prev_dim, output_dim))
                
                self.network = nn.Sequential(*layers_list)
            
            def forward(self, x):
                return self.network(x)
        
        return PerformanceNet(50, config['layers'])
    
    def _build_forecast_model(self, config: Dict[str, Any]) -> Any:
        """Build custom forecast model for capacity prediction"""
        # Simplified Prophet-like model
        class ForecastModel:
            def __init__(self, config):
                self.config = config
                self.trend_model = RandomForestRegressor(n_estimators=50)
                self.seasonality_model = RandomForestRegressor(n_estimators=30)
                
            def fit(self, X, y):
                # Extract trend and seasonality
                trend = self._extract_trend(X, y)
                seasonality = y - trend
                
                self.trend_model.fit(X, trend)
                self.seasonality_model.fit(X, seasonality)
                
            def predict(self, X):
                trend_pred = self.trend_model.predict(X)
                seasonality_pred = self.seasonality_model.predict(X)
                return trend_pred + seasonality_pred
                
            def _extract_trend(self, X, y):
                # Simple moving average for trend
                window = 7
                trend = pd.Series(y).rolling(window=window, center=True).mean()
                return trend.fillna(method='bfill').fillna(method='ffill').values
        
        return ForecastModel(config)
    
    def _initialize_feature_extractors(self) -> None:
        """Initialize feature extraction pipelines"""
        self.feature_extractors = {
            'time_features': self._extract_time_features,
            'access_pattern_features': self._extract_access_pattern_features,
            'performance_features': self._extract_performance_features,
            'cost_features': self._extract_cost_features,
            'behavioral_features': self._extract_behavioral_features
        }
    
    async def create_agent_profile(
        self,
        agent_id: str,
        framework: str,
        initial_data: Optional[Dict[str, Any]] = None
    ) -> EnhancedDNAFeatures:
        """Create enhanced agent profile"""
        
        # Default features based on framework
        framework_defaults = {
            'langchain': {
                'read_write_ratio': 0.8,
                'temporal_pattern': 'conversational',
                'behavior_type': AgentBehavior.SEQUENTIAL,
                'query_complexity': 0.7
            },
            'autogen': {
                'read_write_ratio': 0.6,
                'temporal_pattern': 'multi_agent',
                'behavior_type': AgentBehavior.HYBRID,
                'query_complexity': 0.8
            },
            'crewai': {
                'read_write_ratio': 0.5,
                'temporal_pattern': 'task_based',
                'behavior_type': AgentBehavior.SPATIAL,
                'query_complexity': 0.9
            }
        }
        
        defaults = framework_defaults.get(framework, {})
        
        # Create profile
        profile = EnhancedDNAFeatures(
            avg_request_size=initial_data.get('avg_request_size', 1024),
            read_write_ratio=defaults.get('read_write_ratio', 0.7),
            peak_hour_activity=initial_data.get('peak_hour', 14),
            latency_sensitivity=initial_data.get('latency_sensitivity', 0.8),
            access_frequency=initial_data.get('access_frequency', 100),
            data_locality=initial_data.get('data_locality', 0.7),
            temporal_pattern=defaults.get('temporal_pattern', 'random'),
            compression_ratio=initial_data.get('compression_ratio', 0.6),
            cache_hit_rate=initial_data.get('cache_hit_rate', 0.0),
            concurrent_access=initial_data.get('concurrent_access', 1),
            request_size_variance=initial_data.get('request_size_variance', 0.3),
            access_pattern_entropy=initial_data.get('pattern_entropy', 0.5),
            burst_intensity=initial_data.get('burst_intensity', 0.2),
            data_correlation=initial_data.get('data_correlation', 0.6),
            query_complexity=defaults.get('query_complexity', 0.5),
            resource_utilization=initial_data.get('resource_utilization', 0.4),
            error_rate=initial_data.get('error_rate', 0.01),
            retry_pattern=initial_data.get('retry_pattern', 0.05),
            behavior_type=defaults.get('behavior_type', AgentBehavior.RANDOM),
            learning_rate=initial_data.get('learning_rate', 0.1),
            consistency_score=initial_data.get('consistency_score', 0.7),
            throughput_mbps=initial_data.get('throughput_mbps', 10.0),
            iops_achieved=initial_data.get('iops_achieved', 1000),
            queue_depth_avg=initial_data.get('queue_depth_avg', 5.0),
            storage_cost_sensitivity=initial_data.get('storage_cost_sensitivity', 0.5),
            compute_cost_sensitivity=initial_data.get('compute_cost_sensitivity', 0.3),
            data_retention_days=initial_data.get('retention_days', 30),
            encryption_required=initial_data.get('encryption_required', True),
            audit_frequency=initial_data.get('audit_frequency', 24)
        )
        
        self.agent_profiles[agent_id] = profile
        
        # Find similar agents for transfer learning
        similar_agents = await self._find_similar_agents(agent_id, profile)
        if similar_agents:
            await self._transfer_learning(agent_id, similar_agents)
        
        return profile
    
    async def update_agent_profile(
        self,
        agent_id: str,
        access_patterns: List[Dict[str, Any]]
    ) -> None:
        """Update agent profile with new access patterns"""
        
        if agent_id not in self.agent_profiles:
            logger.warning(f"Agent {agent_id} not found, creating new profile")
            await self.create_agent_profile(agent_id, 'unknown')
        
        # Add to history
        self.agent_history[agent_id].extend(access_patterns)
        
        # Extract features from recent history
        features = await self._extract_features_from_history(
            list(self.agent_history[agent_id])[-1000:]  # Last 1000 patterns
        )
        
        # Update profile
        profile = self.agent_profiles[agent_id]
        
        # Use exponential moving average for smooth updates
        alpha = profile.learning_rate
        
        for feature_name, new_value in features.items():
            if hasattr(profile, feature_name) and isinstance(new_value, (int, float)):
                current_value = getattr(profile, feature_name)
                updated_value = alpha * new_value + (1 - alpha) * current_value
                setattr(profile, feature_name, updated_value)
        
        # Update behavioral classification
        profile.behavior_type = await self._classify_behavior(access_patterns)
        
        # Add to online learning buffer
        self.online_buffer[agent_id].extend(access_patterns)
    
    async def predict_next_access(
        self,
        agent_id: str,
        context: Dict[str, Any]
    ) -> PredictionResult:
        """Predict next access pattern using LSTM model"""
        
        try:
            # Get agent history
            history = list(self.agent_history[agent_id])[-100:]  # Last 100 accesses
            
            if len(history) < 10:
                return PredictionResult(
                    prediction={'key_pattern': 'unknown', 'time_delta': 60},
                    confidence=0.1,
                    reasoning="Insufficient history for prediction",
                    feature_importance={},
                    alternatives=[]
                )
            
            # Prepare features
            X = await self._prepare_sequence_features(history, context)
            
            # Scale features
            X_scaled = self.scalers[ModelType.ACCESS_PREDICTION].transform(X)
            
            # Predict
            model = self.models[ModelType.ACCESS_PREDICTION]
            predictions = model.predict(X_scaled.reshape(1, -1, X_scaled.shape[1]))
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            top_predictions = []
            
            for idx in top_indices:
                key_pattern = self._decode_key_pattern(idx)
                confidence = float(predictions[0][idx])
                top_predictions.append((key_pattern, confidence))
            
            # Estimate time delta
            time_delta = await self._predict_time_delta(agent_id, context)
            
            return PredictionResult(
                prediction={
                    'key_pattern': top_predictions[0][0],
                    'time_delta': time_delta,
                    'prefetch_keys': [p[0] for p in top_predictions[:3]]
                },
                confidence=top_predictions[0][1],
                reasoning=f"Based on {len(history)} historical accesses with {profile.behavior_type.value} behavior",
                feature_importance=await self._get_feature_importance(ModelType.ACCESS_PREDICTION),
                alternatives=top_predictions[1:]
            )
            
        except Exception as e:
            logger.error(f"Access prediction error: {str(e)}")
            return PredictionResult(
                prediction={'key_pattern': 'error', 'time_delta': 60},
                confidence=0.0,
                reasoning=f"Prediction failed: {str(e)}",
                feature_importance={},
                alternatives=[]
            )
    
    async def predict_optimal_tier(
        self,
        agent_id: str,
        data_characteristics: Dict[str, Any]
    ) -> PredictionResult:
        """Predict optimal storage tier using gradient boosting"""
        
        try:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                raise ValueError(f"No profile found for agent {agent_id}")
            
            # Prepare features
            features = await self._prepare_tier_features(profile, data_characteristics)
            
            # Scale features
            X = np.array(features).reshape(1, -1)
            X_scaled = self.scalers[ModelType.TIER_OPTIMIZATION].transform(X)
            
            # Predict
            model = self.models[ModelType.TIER_OPTIMIZATION]
            
            # Get prediction probabilities
            tier_probs = model.predict_proba(X_scaled)[0]
            tier_classes = model.classes_
            
            # Sort by probability
            tier_ranking = sorted(
                zip(tier_classes, tier_probs),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Select tier based on constraints
            selected_tier = tier_ranking[0][0]
            confidence = tier_ranking[0][1]
            
            # Consider cost sensitivity
            if profile.storage_cost_sensitivity > 0.7:
                # Prefer cheaper tiers if cost sensitive
                for tier, prob in tier_ranking:
                    if tier in ['Standard', 'Cool'] and prob > 0.3:
                        selected_tier = tier
                        confidence = prob
                        break
            
            return PredictionResult(
                prediction=selected_tier,
                confidence=confidence,
                reasoning=self._generate_tier_reasoning(profile, data_characteristics, selected_tier),
                feature_importance=await self._get_feature_importance(ModelType.TIER_OPTIMIZATION),
                alternatives=[(t, p) for t, p in tier_ranking[1:4]]
            )
            
        except Exception as e:
            logger.error(f"Tier prediction error: {str(e)}")
            return PredictionResult(
                prediction='Standard',  # Safe default
                confidence=0.5,
                reasoning=f"Prediction failed, using default: {str(e)}",
                feature_importance={},
                alternatives=[]
            )
    
    async def detect_anomalies(
        self,
        agent_id: str,
        recent_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect anomalous access patterns using autoencoder"""
        
        try:
            if not recent_patterns:
                return []
            
            # Prepare features
            X = await self._prepare_anomaly_features(recent_patterns)
            
            # Scale features
            X_scaled = self.scalers[ModelType.ANOMALY_DETECTION].transform(X)
            
            # Get reconstruction error
            model = self.models[ModelType.ANOMALY_DETECTION]
            reconstructed = model.predict(X_scaled)
            mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
            
            # Determine threshold
            threshold = self.model_config[ModelType.ANOMALY_DETECTION]['threshold']
            
            # Find anomalies
            anomalies = []
            for i, (pattern, error) in enumerate(zip(recent_patterns, mse)):
                if error > threshold:
                    anomaly_score = float(error / threshold)
                    
                    # Analyze anomaly type
                    anomaly_type = await self._classify_anomaly(pattern, X[i])
                    
                    anomalies.append({
                        'pattern': pattern,
                        'anomaly_score': anomaly_score,
                        'anomaly_type': anomaly_type,
                        'timestamp': pattern.get('timestamp', datetime.utcnow()),
                        'recommendation': self._get_anomaly_recommendation(anomaly_type)
                    })
            
            # Update model if too many anomalies (concept drift)
            if len(anomalies) > len(recent_patterns) * 0.2:
                logger.warning(f"High anomaly rate ({len(anomalies)}/{len(recent_patterns)}) for agent {agent_id}")
                self.online_buffer[f"{agent_id}_anomaly"].extend(recent_patterns)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection error: {str(e)}")
            return []
    
    async def forecast_capacity(
        self,
        agent_id: str,
        horizon_days: int = 30
    ) -> PredictionResult:
        """Forecast storage capacity needs"""
        
        try:
            # Get historical usage data
            history = await self._get_capacity_history(agent_id)
            
            if len(history) < 30:
                return PredictionResult(
                    prediction={'capacity_gb': 100, 'growth_rate': 0.1},
                    confidence=0.3,
                    reasoning="Limited historical data for accurate forecast",
                    feature_importance={},
                    alternatives=[]
                )
            
            # Prepare features
            X, y = await self._prepare_forecast_features(history)
            
            # Train/update model
            model = self.models[ModelType.CAPACITY_FORECAST]
            model.fit(X, y)
            
            # Generate future features
            future_X = await self._generate_future_features(X, horizon_days)
            
            # Predict
            predictions = model.predict(future_X)
            
            # Calculate metrics
            current_capacity = history[-1]['capacity_gb']
            predicted_capacity = float(predictions[-1])
            growth_rate = (predicted_capacity - current_capacity) / current_capacity
            
            # Generate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(predictions)
            
            return PredictionResult(
                prediction={
                    'capacity_gb': predicted_capacity,
                    'growth_rate': growth_rate,
                    'daily_forecast': predictions.tolist(),
                    'confidence_intervals': confidence_intervals
                },
                confidence=self._calculate_forecast_confidence(history, predictions),
                reasoning=f"Based on {len(history)} days of historical data with {growth_rate:.1%} projected growth",
                feature_importance={'trend': 0.6, 'seasonality': 0.3, 'events': 0.1},
                alternatives=[
                    ({'capacity_gb': ci['upper'], 'growth_rate': growth_rate * 1.5}, 0.8),
                    ({'capacity_gb': ci['lower'], 'growth_rate': growth_rate * 0.5}, 0.8)
                ]
            )
            
        except Exception as e:
            logger.error(f"Capacity forecast error: {str(e)}")
            return PredictionResult(
                prediction={'capacity_gb': 100, 'growth_rate': 0.1},
                confidence=0.0,
                reasoning=f"Forecast failed: {str(e)}",
                feature_importance={},
                alternatives=[]
            )
    
    async def predict_performance(
        self,
        agent_id: str,
        workload: Dict[str, Any]
    ) -> PredictionResult:
        """Predict performance metrics under given workload"""
        
        try:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                raise ValueError(f"No profile found for agent {agent_id}")
            
            # Prepare features
            X = await self._prepare_performance_features(profile, workload)
            X_tensor = torch.FloatTensor(X).unsqueeze(0)
            
            # Predict using neural network
            model = self.models[ModelType.PERFORMANCE_PREDICTION]
            model.eval()
            
            with torch.no_grad():
                predictions = model(X_tensor)
            
            # Extract predictions
            predicted_metrics = {
                'latency_ms': float(predictions[0][0]),
                'throughput_mbps': float(predictions[0][1]) if predictions.shape[1] > 1 else profile.throughput_mbps,
                'iops': int(predictions[0][2]) if predictions.shape[1] > 2 else profile.iops_achieved,
                'cpu_usage': float(predictions[0][3]) if predictions.shape[1] > 3 else 50.0,
                'memory_usage': float(predictions[0][4]) if predictions.shape[1] > 4 else 60.0
            }
            
            # Calculate confidence based on prediction uncertainty
            confidence = self._calculate_prediction_confidence(predictions)
            
            # Generate recommendations
            recommendations = await self._generate_performance_recommendations(
                predicted_metrics,
                workload
            )
            
            return PredictionResult(
                prediction=predicted_metrics,
                confidence=confidence,
                reasoning=f"Neural network prediction based on {len(X)} features",
                feature_importance=await self._get_feature_importance(ModelType.PERFORMANCE_PREDICTION),
                alternatives=recommendations
            )
            
        except Exception as e:
            logger.error(f"Performance prediction error: {str(e)}")
            return PredictionResult(
                prediction={'latency_ms': 10.0, 'throughput_mbps': 100.0},
                confidence=0.0,
                reasoning=f"Prediction failed: {str(e)}",
                feature_importance={},
                alternatives=[]
            )
    
    async def optimize_agent_placement(
        self,
        agent_id: str,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize agent data placement across tiers"""
        
        try:
            profile = self.agent_profiles.get(agent_id)
            if not profile:
                raise ValueError(f"No profile found for agent {agent_id}")
            
            # Get current data distribution
            current_distribution = await self._get_data_distribution(agent_id)
            
            # Predict access patterns
            access_prediction = await self.predict_next_access(agent_id, {})
            
            # Predict optimal tiers for different data segments
            recommendations = []
            
            for data_segment in current_distribution:
                tier_prediction = await self.predict_optimal_tier(
                    agent_id,
                    data_segment
                )
                
                if tier_prediction.confidence > 0.7:
                    recommendations.append({
                        'data_key': data_segment['key'],
                        'current_tier': data_segment['tier'],
                        'recommended_tier': tier_prediction.prediction,
                        'confidence': tier_prediction.confidence,
                        'estimated_savings': self._calculate_savings(
                            data_segment,
                            tier_prediction.prediction
                        )
                    })
            
            # Sort by potential impact
            recommendations.sort(
                key=lambda x: x['estimated_savings'] * x['confidence'],
                reverse=True
            )
            
            return {
                'agent_id': agent_id,
                'optimization_plan': recommendations[:10],  # Top 10 recommendations
                'total_potential_savings': sum(r['estimated_savings'] for r in recommendations[:10]),
                'implementation_priority': self._calculate_priority(recommendations),
                'risk_assessment': await self._assess_optimization_risk(recommendations)
            }
            
        except Exception as e:
            logger.error(f"Optimization error: {str(e)}")
            return {'error': str(e)}
    
    async def _extract_features_from_history(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Extract features from access history"""
        
        if not history:
            return {}
        
        features = {}
        
        # Basic statistics
        request_sizes = [h.get('data_size', 0) for h in history]
        features['avg_request_size'] = np.mean(request_sizes) if request_sizes else 0
        features['request_size_variance'] = np.var(request_sizes) if request_sizes else 0
        
        # Read/write ratio
        operations = [h.get('operation', 'read') for h in history]
        read_count = operations.count('read')
        write_count = operations.count('write')
        total_ops = read_count + write_count
        features['read_write_ratio'] = read_count / total_ops if total_ops > 0 else 0.5
        
        # Temporal patterns
        timestamps = [h.get('timestamp', datetime.utcnow()) for h in history]
        if timestamps:
            hours = [t.hour for t in timestamps if isinstance(t, datetime)]
            if hours:
                features['peak_hour_activity'] = max(set(hours), key=hours.count)
        
        # Access frequency
        time_diffs = []
        for i in range(1, len(timestamps)):
            if isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_diffs.append(diff)
        
        features['access_frequency'] = 1 / np.mean(time_diffs) if time_diffs else 0
        
        # Pattern entropy (measure of randomness)
        keys = [h.get('key', '') for h in history]
        key_counts = pd.Series(keys).value_counts()
        if len(key_counts) > 0:
            probs = key_counts / len(keys)
            features['access_pattern_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Performance metrics
        latencies = [h.get('latency_ms', 0) for h in history if h.get('latency_ms')]
        if latencies:
            features['latency_sensitivity'] = 1 / (1 + np.mean(latencies))
        
        # Cache performance
        cache_hits = [h.get('cache_hit', False) for h in history]
        features['cache_hit_rate'] = sum(cache_hits) / len(cache_hits) if cache_hits else 0
        
        # Error patterns
        errors = [h.get('error', False) for h in history]
        features['error_rate'] = sum(1 for e in errors if e) / len(errors) if errors else 0
        
        return features
    
    async def _classify_behavior(
        self,
        patterns: List[Dict[str, Any]]
    ) -> AgentBehavior:
        """Classify agent behavior type"""
        
        if len(patterns) < 10:
            return AgentBehavior.RANDOM
        
        # Extract key sequences
        keys = [p.get('key', '') for p in patterns]
        
        # Check for sequential access
        sequential_score = 0
        for i in range(1, len(keys)):
            if keys[i] > keys[i-1]:  # Simple sequential check
                sequential_score += 1
        
        if sequential_score / len(keys) > 0.7:
            return AgentBehavior.SEQUENTIAL
        
        # Check for temporal patterns
        timestamps = [p.get('timestamp', datetime.utcnow()) for p in patterns]
        time_diffs = []
        for i in range(1, len(timestamps)):
            if isinstance(timestamps[i], datetime) and isinstance(timestamps[i-1], datetime):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_diffs.append(diff)
        
        if time_diffs and np.std(time_diffs) < np.mean(time_diffs) * 0.3:
            return AgentBehavior.TEMPORAL
        
        # Check for spatial patterns (key similarity)
        key_prefixes = [k.split('/')[0] if '/' in k else k[:5] for k in keys]
        unique_prefixes = len(set(key_prefixes))
        
        if unique_prefixes < len(key_prefixes) * 0.3:
            return AgentBehavior.SPATIAL
        
        # Check for mixed patterns
        if sequential_score / len(keys) > 0.3 and unique_prefixes < len(key_prefixes) * 0.5:
            return AgentBehavior.HYBRID
        
        return AgentBehavior.RANDOM
    
    async def _find_similar_agents(
        self,
        agent_id: str,
        profile: EnhancedDNAFeatures,
        threshold: float = 0.8
    ) -> List[str]:
        """Find agents with similar profiles for transfer learning"""
        
        similar_agents = []
        
        # Convert profile to feature vector
        target_features = self._profile_to_vector(profile)
        
        for other_agent_id, other_profile in self.agent_profiles.items():
            if other_agent_id == agent_id:
                continue
            
            # Calculate similarity
            other_features = self._profile_to_vector(other_profile)
            similarity = self._calculate_similarity(target_features, other_features)
            
            if similarity > threshold:
                similar_agents.append(other_agent_id)
        
        return similar_agents[:5]  # Return top 5 similar agents
    
    def _profile_to_vector(self, profile: EnhancedDNAFeatures) -> np.ndarray:
        """Convert profile to numerical feature vector"""
        
        features = []
        
        # Numerical features
        numerical_attrs = [
            'avg_request_size', 'read_write_ratio', 'latency_sensitivity',
            'access_frequency', 'data_locality', 'compression_ratio',
            'cache_hit_rate', 'concurrent_access', 'request_size_variance',
            'access_pattern_entropy', 'burst_intensity', 'data_correlation',
            'query_complexity', 'resource_utilization', 'error_rate',
            'retry_pattern', 'learning_rate', 'consistency_score',
            'throughput_mbps', 'iops_achieved', 'queue_depth_avg',
            'storage_cost_sensitivity', 'compute_cost_sensitivity'
        ]
        
        for attr in numerical_attrs:
            value = getattr(profile, attr, 0)
            features.append(float(value))
        
        # Categorical features (one-hot encoding)
        features.append(1.0 if profile.behavior_type == AgentBehavior.SEQUENTIAL else 0.0)
        features.append(1.0 if profile.behavior_type == AgentBehavior.TEMPORAL else 0.0)
        features.append(1.0 if profile.behavior_type == AgentBehavior.SPATIAL else 0.0)
        features.append(1.0 if profile.encryption_required else 0.0)
        
        return np.array(features)
    
    def _calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between feature vectors"""
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _transfer_learning(
        self,
        target_agent: str,
        source_agents: List[str]
    ) -> None:
        """Transfer knowledge from similar agents"""
        
        logger.info(f"Transferring knowledge to {target_agent} from {source_agents}")
        
        # Get source agent histories
        combined_history = []
        for source_agent in source_agents:
            if source_agent in self.agent_history:
                # Weight by similarity
                similarity = 0.8  # Simplified
                weighted_history = [
                    {**h, 'weight': similarity}
                    for h in list(self.agent_history[source_agent])[-100:]
                ]
                combined_history.extend(weighted_history)
        
        if combined_history:
            # Pre-train models with weighted data
            # This is a simplified version - in production, use proper transfer learning
            self.online_buffer[f"{target_agent}_transfer"] = combined_history
    
    async def _prepare_sequence_features(
        self,
        history: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Prepare features for sequence prediction"""
        
        features = []
        
        for i, pattern in enumerate(history):
            pattern_features = []
            
            # Time features
            if 'timestamp' in pattern and isinstance(pattern['timestamp'], datetime):
                pattern_features.extend([
                    pattern['timestamp'].hour,
                    pattern['timestamp'].weekday(),
                    pattern['timestamp'].day
                ])
            else:
                pattern_features.extend([0, 0, 0])
            
            # Operation features
            op_encoding = {
                'read': [1, 0, 0, 0],
                'write': [0, 1, 0, 0],
                'delete': [0, 0, 1, 0],
                'list': [0, 0, 0, 1]
            }
            pattern_features.extend(op_encoding.get(pattern.get('operation', 'read'), [0, 0, 0, 0]))
            
            # Size features
            size = pattern.get('data_size', 0)
            pattern_features.extend([
                np.log1p(size),
                size / 1024,  # KB
                size / (1024 * 1024)  # MB
            ])
            
            # Performance features
            pattern_features.extend([
                pattern.get('latency_ms', 0),
                1.0 if pattern.get('cache_hit', False) else 0.0,
                pattern.get('tier', 2)  # Tier as numeric
            ])
            
            # Context features
            if context:
                pattern_features.extend([
                    context.get('current_load', 0.5),
                    context.get('time_of_day_factor', 1.0)
                ])
            else:
                pattern_features.extend([0.5, 1.0])
            
            features.append(pattern_features)
        
        return np.array(features)
    
    async def _predict_time_delta(
        self,
        agent_id: str,
        context: Dict[str, Any]
    ) -> float:
        """Predict time until next access"""
        
        history = list(self.agent_history[agent_id])[-100:]
        
        if len(history) < 2:
            return 60.0  # Default to 1 minute
        
        # Calculate historical time deltas
        time_deltas = []
        for i in range(1, len(history)):
            if 'timestamp' in history[i] and 'timestamp' in history[i-1]:
                t1 = history[i]['timestamp']
                t2 = history[i-1]['timestamp']
                if isinstance(t1, datetime) and isinstance(t2, datetime):
                    delta = (t1 - t2).total_seconds()
                    if 0 < delta < 3600:  # Reasonable range
                        time_deltas.append(delta)
        
        if not time_deltas:
            return 60.0
        
        # Use exponential weighted average
        weights = np.exp(np.linspace(-2, 0, len(time_deltas)))
        weights /= weights.sum()
        
        weighted_avg = np.average(time_deltas, weights=weights)
        
        # Adjust for context
        if context.get('peak_hours', False):
            weighted_avg *= 0.7  # Faster during peak
        
        return float(weighted_avg)
    
    def _decode_key_pattern(self, index: int) -> str:
        """Decode predicted key pattern index"""
        
        # Simplified pattern mapping
        patterns = [
            'conversation/*',
            'memory/short_term/*',
            'memory/long_term/*',
            'embeddings/*',
            'cache/*',
            'models/*',
            'config/*',
            'logs/*',
            'checkpoints/*',
            'tmp/*'
        ]
        
        if 0 <= index < len(patterns):
            return patterns[index]
        
        return 'unknown/*'
    
    async def _get_feature_importance(
        self,
        model_type: ModelType
    ) -> Dict[str, float]:
        """Get feature importance for model"""
        
        if model_type in self.feature_importance:
            return self.feature_importance[model_type]
        
        # Calculate or return defaults
        default_importance = {
            ModelType.ACCESS_PREDICTION: {
                'time_features': 0.3,
                'operation_history': 0.25,
                'size_patterns': 0.2,
                'performance_metrics': 0.15,
                'context': 0.1
            },
            ModelType.TIER_OPTIMIZATION: {
                'latency_sensitivity': 0.25,
                'access_frequency': 0.2,
                'cost_sensitivity': 0.2,
                'data_size': 0.15,
                'compliance': 0.1,
                'performance_history': 0.1
            }
        }
        
        return default_importance.get(model_type, {})
    
    async def _prepare_tier_features(
        self,
        profile: EnhancedDNAFeatures,
        data_characteristics: Dict[str, Any]
    ) -> List[float]:
        """Prepare features for tier optimization"""
        
        features = []
        
        # Profile features
        features.extend([
            profile.latency_sensitivity,
            profile.access_frequency,
            profile.storage_cost_sensitivity,
            profile.throughput_mbps,
            profile.cache_hit_rate,
            profile.compression_ratio
        ])
        
        # Data characteristics
        features.extend([
            np.log1p(data_characteristics.get('size_bytes', 0)),
            data_characteristics.get('access_recency_hours', 24),
            data_characteristics.get('access_count', 0),
            1.0 if data_characteristics.get('is_hot', False) else 0.0,
            1.0 if data_characteristics.get('is_compressed', False) else 0.0
        ])
        
        # Compliance requirements
        compliance_encoding = {
            'none': [0, 0, 0],
            'standard': [1, 0, 0],
            'high': [0, 1, 0],
            'critical': [0, 0, 1]
        }
        features.extend(
            compliance_encoding.get(
                data_characteristics.get('compliance_level', 'none'),
                [0, 0, 0]
            )
        )
        
        return features
    
    def _generate_tier_reasoning(
        self,
        profile: EnhancedDNAFeatures,
        data_characteristics: Dict[str, Any],
        selected_tier: str
    ) -> str:
        """Generate explanation for tier selection"""
        
        reasons = []
        
        if profile.latency_sensitivity > 0.8 and selected_tier in ['Ultra', 'Premium']:
            reasons.append("High latency sensitivity requires fast storage")
        
        if profile.storage_cost_sensitivity > 0.7 and selected_tier in ['Standard', 'Cool']:
            reasons.append("Cost optimization prioritized based on agent profile")
        
        if data_characteristics.get('access_count', 0) > 100:
            reasons.append(f"High access frequency ({data_characteristics['access_count']} accesses)")
        
        if data_characteristics.get('compliance_level') == 'critical':
            reasons.append("Critical compliance requirements limit tier options")
        
        if not reasons:
            reasons.append("Balanced selection based on multiple factors")
        
        return "; ".join(reasons)
    
    async def _prepare_anomaly_features(
        self,
        patterns: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Prepare features for anomaly detection"""
        
        features = []
        
        for pattern in patterns:
            pattern_features = []
            
            # Time-based features
            if 'timestamp' in pattern and isinstance(pattern['timestamp'], datetime):
                hour = pattern['timestamp'].hour
                day_of_week = pattern['timestamp'].weekday()
                
                # Encode cyclic features
                pattern_features.extend([
                    np.sin(2 * np.pi * hour / 24),
                    np.cos(2 * np.pi * hour / 24),
                    np.sin(2 * np.pi * day_of_week / 7),
                    np.cos(2 * np.pi * day_of_week / 7)
                ])
            else:
                pattern_features.extend([0, 0, 0, 0])
            
            # Access pattern features
            pattern_features.extend([
                np.log1p(pattern.get('data_size', 0)),
                pattern.get('latency_ms', 0) / 100,  # Normalize
                1.0 if pattern.get('cache_hit', False) else 0.0,
                pattern.get('retry_count', 0),
                1.0 if pattern.get('error', False) else 0.0
            ])
            
            # Key characteristics
            key = pattern.get('key', '')
            pattern_features.extend([
                len(key),
                key.count('/'),
                1.0 if key.startswith('temp') else 0.0,
                1.0 if 'backup' in key else 0.0
            ])
            
            features.append(pattern_features)
        
        return np.array(features)
    
    async def _classify_anomaly(
        self,
        pattern: Dict[str, Any],
        features: np.ndarray
    ) -> str:
        """Classify type of anomaly"""
        
        # Simple rule-based classification
        if pattern.get('latency_ms', 0) > 1000:
            return 'high_latency'
        elif pattern.get('retry_count', 0) > 3:
            return 'excessive_retries'
        elif pattern.get('data_size', 0) > 100 * 1024 * 1024:  # 100MB
            return 'unusually_large_request'
        elif pattern.get('error', False):
            return 'error_pattern'
        elif 'timestamp' in pattern and isinstance(pattern['timestamp'], datetime):
            hour = pattern['timestamp'].hour
            if 2 <= hour <= 5:  # Unusual hours
                return 'unusual_timing'
        
        return 'unknown_anomaly'
    
    def _get_anomaly_recommendation(self, anomaly_type: str) -> str:
        """Get recommendation for anomaly type"""
        
        recommendations = {
            'high_latency': "Consider moving data to faster tier or investigating network issues",
            'excessive_retries': "Check for intermittent failures or resource contention",
            'unusually_large_request': "Consider chunking large requests or using streaming",
            'error_pattern': "Investigate error logs and consider implementing circuit breaker",
            'unusual_timing': "Verify if this is expected batch processing or potential security issue",
            'unknown_anomaly': "Review access patterns manually for unexpected behavior"
        }
        
        return recommendations.get(anomaly_type, "Monitor pattern for recurrence")
    
    async def _get_capacity_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get historical capacity usage"""
        
        # In production, this would query actual metrics
        # For now, generate sample data
        history = []
        base_capacity = 50  # GB
        
        for i in range(90):  # 90 days of history
            date = datetime.utcnow() - timedelta(days=90-i)
            
            # Simulate growth with noise
            growth = 1.002 ** i  # 0.2% daily growth
            noise = np.random.normal(0, 2)
            capacity = base_capacity * growth + noise
            
            history.append({
                'date': date,
                'capacity_gb': max(0, capacity),
                'agent_id': agent_id
            })
        
        return history
    
    async def _prepare_forecast_features(
        self,
        history: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for capacity forecasting"""
        
        X = []
        y = []
        
        for i, record in enumerate(history):
            if i < 7:  # Need at least 7 days of history
                continue
            
            # Features
            features = []
            
            # Time features
            date = record['date']
            features.extend([
                i,  # Day index
                date.weekday(),
                date.day,
                date.month
            ])
            
            # Historical capacity (last 7 days)
            for j in range(7):
                features.append(history[i-j-1]['capacity_gb'])
            
            # Moving averages
            ma_3 = np.mean([history[i-k]['capacity_gb'] for k in range(1, 4)])
            ma_7 = np.mean([history[i-k]['capacity_gb'] for k in range(1, 8)])
            features.extend([ma_3, ma_7])
            
            X.append(features)
            y.append(record['capacity_gb'])
        
        return np.array(X), np.array(y)
    
    async def _generate_future_features(
        self,
        historical_X: np.ndarray,
        horizon_days: int
    ) -> np.ndarray:
        """Generate features for future predictions"""
        
        future_X = []
        last_features = historical_X[-1]
        
        for i in range(horizon_days):
            features = last_features.copy()
            
            # Update time features
            features[0] = len(historical_X) + i  # Day index
            
            # Shift historical values
            # This is simplified - in production, use predicted values
            features[4:11] = features[3:10]  # Shift 7-day history
            
            future_X.append(features)
        
        return np.array(future_X)
    
    def _calculate_confidence_intervals(
        self,
        predictions: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, List[float]]:
        """Calculate confidence intervals for predictions"""
        
        # Simplified confidence intervals
        # In production, use proper statistical methods
        std_dev = np.std(predictions) * 0.1  # Assume 10% standard deviation
        z_score = 1.96  # 95% confidence
        
        margin = z_score * std_dev
        
        return {
            'lower': (predictions - margin).tolist(),
            'upper': (predictions + margin).tolist(),
            'confidence_level': confidence_level
        }
    
    def _calculate_forecast_confidence(
        self,
        history: List[Dict[str, Any]],
        predictions: np.ndarray
    ) -> float:
        """Calculate confidence in forecast"""
        
        # Factors affecting confidence
        history_length = len(history)
        prediction_variance = np.var(predictions)
        
        # More history = higher confidence
        history_factor = min(1.0, history_length / 90)
        
        # Less variance = higher confidence
        variance_factor = 1.0 / (1.0 + prediction_variance / 100)
        
        return history_factor * variance_factor * 0.9  # Max 90% confidence
    
    async def _prepare_performance_features(
        self,
        profile: EnhancedDNAFeatures,
        workload: Dict[str, Any]
    ) -> List[float]:
        """Prepare features for performance prediction"""
        
        features = []
        
        # Profile features
        features.extend([
            profile.throughput_mbps,
            profile.iops_achieved,
            profile.queue_depth_avg,
            profile.latency_sensitivity,
            profile.resource_utilization
        ])
        
        # Workload features
        features.extend([
            workload.get('request_rate', 100),
            workload.get('avg_request_size', 1024),
            workload.get('read_ratio', 0.7),
            workload.get('concurrent_requests', 10),
            workload.get('cache_hit_rate', profile.cache_hit_rate)
        ])
        
        # System features
        features.extend([
            workload.get('cpu_available', 80),
            workload.get('memory_available', 70),
            workload.get('network_bandwidth', 1000),
            workload.get('storage_iops_limit', 10000)
        ])
        
        return features
    
    def _calculate_prediction_confidence(
        self,
        predictions: torch.Tensor
    ) -> float:
        """Calculate confidence based on prediction uncertainty"""
        
        # Use prediction variance as uncertainty measure
        if predictions.shape[0] > 1:
            variance = torch.var(predictions).item()
            confidence = 1.0 / (1.0 + variance)
        else:
            confidence = 0.8  # Default confidence
        
        return min(0.95, confidence)  # Cap at 95%
    
    async def _generate_performance_recommendations(
        self,
        predicted_metrics: Dict[str, float],
        workload: Dict[str, Any]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Generate performance optimization recommendations"""
        
        recommendations = []
        
        # Latency optimization
        if predicted_metrics['latency_ms'] > 10:
            recommendations.append((
                {
                    'action': 'increase_cache_size',
                    'expected_improvement': '30% latency reduction',
                    'metrics': {**predicted_metrics, 'latency_ms': predicted_metrics['latency_ms'] * 0.7}
                },
                0.8
            ))
        
        # Throughput optimization
        if predicted_metrics['throughput_mbps'] < workload.get('required_throughput', 100):
            recommendations.append((
                {
                    'action': 'enable_compression',
                    'expected_improvement': '40% throughput increase',
                    'metrics': {**predicted_metrics, 'throughput_mbps': predicted_metrics['throughput_mbps'] * 1.4}
                },
                0.7
            ))
        
        # Resource optimization
        if predicted_metrics.get('cpu_usage', 50) > 80:
            recommendations.append((
                {
                    'action': 'scale_horizontally',
                    'expected_improvement': 'Distribute load across instances',
                    'metrics': {**predicted_metrics, 'cpu_usage': predicted_metrics.get('cpu_usage', 50) / 2}
                },
                0.9
            ))
        
        return recommendations[:3]  # Top 3 recommendations
    
    async def _get_data_distribution(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get current data distribution across tiers"""
        
        # In production, query actual storage system
        # For now, return sample distribution
        return [
            {
                'key': f'data/segment_{i}',
                'tier': np.random.choice(['Ultra', 'Premium', 'Standard']),
                'size_gb': np.random.uniform(1, 10),
                'last_access': datetime.utcnow() - timedelta(days=np.random.randint(0, 30)),
                'access_count': np.random.randint(1, 1000)
            }
            for i in range(20)
        ]
    
    def _calculate_savings(
        self,
        data_segment: Dict[str, Any],
        recommended_tier: str
    ) -> float:
        """Calculate potential cost savings"""
        
        tier_costs = {
            'Ultra': 0.50,
            'Premium': 0.25,
            'Standard': 0.10,
            'Cool': 0.05,
            'Archive': 0.01
        }
        
        current_cost = data_segment['size_gb'] * tier_costs.get(data_segment['tier'], 0.10)
        new_cost = data_segment['size_gb'] * tier_costs.get(recommended_tier, 0.10)
        
        return max(0, current_cost - new_cost)
    
    def _calculate_priority(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate implementation priority"""
        
        priorities = []
        
        for rec in recommendations[:5]:
            priority_score = (
                rec['estimated_savings'] * 0.4 +
                rec['confidence'] * 100 * 0.3 +
                (100 if rec['current_tier'] != rec['recommended_tier'] else 0) * 0.3
            )
            
            priorities.append({
                'data_key': rec['data_key'],
                'priority_score': priority_score,
                'action': f"Move from {rec['current_tier']} to {rec['recommended_tier']}"
            })
        
        return sorted(priorities, key=lambda x: x['priority_score'], reverse=True)
    
    async def _assess_optimization_risk(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess risks of optimization plan"""
        
        risks = {
            'low': 0,
            'medium': 0,
            'high': 0
        }
        
        for rec in recommendations:
            # Moving to slower tier has higher risk
            if rec['recommended_tier'] in ['Cool', 'Archive'] and rec['current_tier'] in ['Ultra', 'Premium']:
                risks['high'] += 1
            elif rec['confidence'] < 0.6:
                risks['medium'] += 1
            else:
                risks['low'] += 1
        
        overall_risk = 'low'
        if risks['high'] > len(recommendations) * 0.2:
            overall_risk = 'high'
        elif risks['medium'] > len(recommendations) * 0.3:
            overall_risk = 'medium'
        
        return {
            'overall_risk': overall_risk,
            'risk_breakdown': risks,
            'mitigation': self._get_risk_mitigation(overall_risk)
        }
    
    def _get_risk_mitigation(self, risk_level: str) -> str:
        """Get risk mitigation strategies"""
        
        strategies = {
            'low': "Proceed with optimization as planned",
            'medium': "Implement changes gradually and monitor performance closely",
            'high': "Test changes in staging environment first and have rollback plan ready"
        }
        
        return strategies.get(risk_level, "Review optimization plan before proceeding")
    
    def _start_background_tasks(self) -> None:
        """Start background ML tasks"""
        self._background_tasks = [
            asyncio.create_task(self._online_learning_worker()),
            asyncio.create_task(self._model_evaluation_worker()),
            asyncio.create_task(self._cluster_update_worker())
        ]
    
    async def _online_learning_worker(self) -> None:
        """Online learning to update models"""
        while self._running:
            try:
                await asyncio.sleep(self.update_frequency)
                
                # Process buffered data for each agent
                for agent_id, buffer_data in self.online_buffer.items():
                    if len(buffer_data) > 100:  # Minimum batch size
                        await self._update_models(agent_id, buffer_data)
                        self.online_buffer[agent_id] = []  # Clear buffer
                
            except Exception as e:
                logger.error(f"Online learning error: {str(e)}")
    
    async def _model_evaluation_worker(self) -> None:
        """Evaluate model performance"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Hourly evaluation
                
                for model_type in ModelType:
                    if model_type in self.prediction_accuracy:
                        accuracy_data = list(self.prediction_accuracy[model_type])
                        if accuracy_data:
                            avg_accuracy = np.mean(accuracy_data)
                            self.model_metrics[model_type] = {
                                'average_accuracy': avg_accuracy,
                                'sample_count': len(accuracy_data),
                                'last_updated': datetime.utcnow()
                            }
                            
                            if avg_accuracy < 0.7:
                                logger.warning(
                                    f"Low accuracy for {model_type.value}: {avg_accuracy:.2%}"
                                )
                
            except Exception as e:
                logger.error(f"Model evaluation error: {str(e)}")
    
    async def _cluster_update_worker(self) -> None:
        """Update agent clusters periodically"""
        while self._running:
            try:
                await asyncio.sleep(86400)  # Daily clustering
                
                if len(self.agent_profiles) > 10:
                    await self._update_agent_clusters()
                
            except Exception as e:
                logger.error(f"Cluster update error: {str(e)}")
    
    async def _update_models(
        self,
        agent_id: str,
        new_data: List[Dict[str, Any]]
    ) -> None:
        """Update models with new data"""
        
        logger.info(f"Updating models for agent {agent_id} with {len(new_data)} samples")
        
        # Update different models based on data type
        # This is simplified - in production, use proper online learning algorithms
        
        # Update agent profile
        await self.update_agent_profile(agent_id, new_data)
        
        # Mark models for retraining
        for model_type in ModelType:
            self.model_versions[model_type] += 1
    
    async def _update_agent_clusters(self) -> None:
        """Update agent clusters using DBSCAN"""
        
        try:
            # Convert all profiles to feature vectors
            agent_ids = []
            feature_vectors = []
            
            for agent_id, profile in self.agent_profiles.items():
                agent_ids.append(agent_id)
                feature_vectors.append(self._profile_to_vector(profile))
            
            if len(feature_vectors) < 2:
                return
            
            # Normalize features
            X = np.array(feature_vectors)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=min(10, X.shape[1]))
            X_reduced = pca.fit_transform(X_scaled)
            
            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clustering.fit_predict(X_reduced)
            
            # Create cluster objects
            new_clusters = []
            unique_labels = set(cluster_labels) - {-1}  # Exclude noise
            
            for label in unique_labels:
                cluster_agents = [
                    agent_ids[i] for i in range(len(agent_ids))
                    if cluster_labels[i] == label
                ]
                
                if cluster_agents:
                    # Calculate centroid features
                    cluster_features = X[cluster_labels == label].mean(axis=0)
                    
                    # Find representative agent (closest to centroid)
                    distances = np.linalg.norm(
                        X[cluster_labels == label] - cluster_features,
                        axis=1
                    )
                    representative_idx = np.argmin(distances)
                    representative_agent = cluster_agents[representative_idx]
                    
                    new_clusters.append(AgentCluster(
                        cluster_id=int(label),
                        agents=cluster_agents,
                        centroid_features=self.agent_profiles[representative_agent],  # Simplified
                        variance=float(np.var(X[cluster_labels == label])),
                        representative_agent=representative_agent
                    ))
            
            self.agent_clusters = new_clusters
            logger.info(f"Updated {len(new_clusters)} agent clusters")
            
        except Exception as e:
            logger.error(f"Cluster update error: {str(e)}")
    
    async def _load_pretrained_models(self) -> None:
        """Load pre-trained models if available"""
        
        # In production, load from model registry
        logger.info("No pre-trained models found, using fresh initialization")
    
    async def shutdown(self) -> None:
        """Shutdown profiler and save models"""
        logger.info("Shutting down Advanced Agent DNA Profiler...")
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Save models
        await self._save_models()
        
        logger.info("Advanced Agent DNA Profiler shutdown complete")
    
    async def _save_models(self) -> None:
        """Save trained models"""
        
        # In production, save to model registry
        logger.info("Models saved (placeholder)")
    
    async def get_agent_profile(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent profile as dictionary"""
        
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            return None
        
        return {
            'features': {
                'avg_request_size': profile.avg_request_size,
                'read_write_ratio': profile.read_write_ratio,
                'latency_sensitivity': profile.latency_sensitivity,
                'cache_hit_rate': profile.cache_hit_rate,
                'behavior_type': profile.behavior_type.value,
                'learning_rate': profile.learning_rate,
                'consistency_score': profile.consistency_score
            },
            'performance': {
                'throughput_mbps': profile.throughput_mbps,
                'iops_achieved': profile.iops_achieved,
                'error_rate': profile.error_rate
            },
            'optimal_tier': await self._determine_optimal_tier(profile),
            'recommendations': await self._generate_recommendations(profile)
        }
    
    async def _determine_optimal_tier(
        self,
        profile: EnhancedDNAFeatures
    ) -> str:
        """Determine optimal tier based on profile"""
        
        if profile.latency_sensitivity > 0.9 and profile.access_frequency > 1000:
            return 'Ultra'
        elif profile.latency_sensitivity > 0.7 or profile.access_frequency > 500:
            return 'Premium'
        elif profile.storage_cost_sensitivity > 0.7 and profile.access_frequency < 100:
            return 'Cool'
        elif profile.data_retention_days > 365:
            return 'Archive'
        else:
            return 'Standard'
    
    async def _generate_recommendations(
        self,
        profile: EnhancedDNAFeatures
    ) -> Dict[str, Any]:
        """Generate storage recommendations based on profile"""
        
        recommendations = {
            'caching': {
                'enabled': profile.cache_hit_rate < 0.7,
                'ttl_seconds': int(3600 / max(1, profile.access_frequency)),
                'strategy': 'aggressive' if profile.latency_sensitivity > 0.8 else 'normal'
            },
            'compression': {
                'enabled': profile.compression_ratio < 0.7,
                'algorithm': 'zstd' if profile.latency_sensitivity < 0.5 else 'lz4'
            },
            'prefetch': {
                'enabled': profile.behavior_type in [AgentBehavior.SEQUENTIAL, AgentBehavior.TEMPORAL],
                'depth': min(10, int(profile.consistency_score * 10))
            },
            'replication': {
                'enabled': profile.data_retention_days > 30 or profile.encryption_required,
                'factor': 2 if profile.encryption_required else 1
            }
        }
        
        return recommendations
    
    async def predict_next_keys(
        self,
        agent_id: str,
        current_key: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Predict next likely keys for prefetching"""
        
        prediction = await self.predict_next_access(agent_id, {'current_key': current_key})
        
        if prediction.confidence > 0.5:
            return {
                'keys': prediction.prediction.get('prefetch_keys', [])[:limit],
                'confidence': prediction.confidence
            }
        
        return {'keys': [], 'confidence': 0.0}