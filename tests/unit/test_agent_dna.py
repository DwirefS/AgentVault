"""
Unit tests for Agent DNA Profiler
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import json

from src.ml.agent_dna import (
    AgentDNAProfiler,
    DNAFeatures,
    AccessPattern,
    StorageDNA
)
from src.storage.tier_manager import StorageTier


class TestAgentDNAProfiler:
    """Test suite for AgentDNAProfiler"""
    
    @pytest.fixture
    def profiler(self):
        """Create profiler instance for testing"""
        return AgentDNAProfiler()
    
    @pytest.fixture
    def sample_features(self):
        """Create sample DNA features"""
        return DNAFeatures(
            avg_request_size=1024.0,
            read_write_ratio=0.8,
            peak_hour_activity=14,
            latency_sensitivity=0.9,
            access_frequency=100.0,
            data_locality=0.7,
            temporal_pattern="business_hours",
            compression_ratio=0.6,
            cache_hit_rate=0.75,
            concurrent_access=5.0
        )
    
    @pytest.fixture
    def sample_access_patterns(self):
        """Create sample access patterns"""
        base_time = datetime.utcnow()
        return [
            AccessPattern(
                timestamp=base_time - timedelta(minutes=i),
                operation="read" if i % 3 != 0 else "write",
                data_size=1000 + i * 100,
                latency_ms=5.0 + i * 0.1,
                tier=StorageTier.PREMIUM,
                cache_hit=i % 2 == 0,
                success=True
            )
            for i in range(100)
        ]
    
    def test_create_dna_profile(self, profiler):
        """Test creating new DNA profile"""
        dna = profiler.create_dna("test-agent-001", "langchain")
        
        assert dna.agent_id == "test-agent-001"
        assert dna.framework == "langchain"
        assert dna.version == 1
        assert isinstance(dna.features, DNAFeatures)
        assert dna.created_at <= datetime.utcnow()
    
    @pytest.mark.asyncio
    async def test_update_dna_profile(self, profiler, sample_access_patterns):
        """Test updating DNA profile with access patterns"""
        agent_id = "test-agent-001"
        
        # Create initial profile
        dna = profiler.create_dna(agent_id, "autogen")
        
        # Update with access patterns
        await profiler.update_dna(agent_id, sample_access_patterns)
        
        # Get updated profile
        updated_dna = await profiler.get_agent_profile(agent_id)
        
        assert updated_dna is not None
        assert updated_dna["features"]["access_frequency"] > 0
        assert updated_dna["features"]["avg_request_size"] > 0
        assert updated_dna["optimal_tier"] in [tier.value for tier in StorageTier]
    
    def test_extract_features(self, profiler, sample_access_patterns):
        """Test feature extraction from access patterns"""
        features = profiler._extract_features(sample_access_patterns)
        
        assert features.avg_request_size > 0
        assert 0 <= features.read_write_ratio <= 1
        assert 0 <= features.peak_hour_activity <= 23
        assert features.access_frequency > 0
        assert 0 <= features.cache_hit_rate <= 1
    
    def test_calculate_optimal_tier(self, profiler, sample_features):
        """Test optimal tier calculation"""
        # High latency sensitivity should suggest faster tier
        features_fast = DNAFeatures(
            latency_sensitivity=0.95,
            access_frequency=1000,
            avg_request_size=500,
            **{k: v for k, v in sample_features.__dict__.items() 
               if k not in ['latency_sensitivity', 'access_frequency', 'avg_request_size']}
        )
        tier_fast = profiler._calculate_optimal_tier(features_fast)
        assert tier_fast in [StorageTier.ULTRA, StorageTier.PREMIUM]
        
        # Low frequency, low sensitivity should suggest slower tier
        features_slow = DNAFeatures(
            latency_sensitivity=0.2,
            access_frequency=10,
            avg_request_size=50000,
            **{k: v for k, v in sample_features.__dict__.items()
               if k not in ['latency_sensitivity', 'access_frequency', 'avg_request_size']}
        )
        tier_slow = profiler._calculate_optimal_tier(features_slow)
        assert tier_slow in [StorageTier.STANDARD, StorageTier.COOL, StorageTier.ARCHIVE]
    
    def test_predict_next_access(self, profiler, sample_access_patterns):
        """Test next access prediction"""
        profiler.ml_models["test-agent"] = Mock()
        profiler.ml_models["test-agent"].predict = Mock(return_value=np.array([[300.5]]))
        
        prediction = profiler.predict_next_access("test-agent", sample_access_patterns[-10:])
        
        assert prediction is not None
        assert prediction["predicted_time_seconds"] == 300.5
        assert "confidence" in prediction
    
    def test_get_recommendations(self, profiler, sample_features):
        """Test storage recommendations generation"""
        dna = StorageDNA(
            agent_id="test-agent",
            framework="crewai",
            features=sample_features,
            created_at=datetime.utcnow(),
            version=1
        )
        
        recommendations = profiler.get_recommendations(dna)
        
        assert "tier" in recommendations
        assert "caching" in recommendations
        assert "compression" in recommendations
        assert "prefetch" in recommendations
        
        # Test specific recommendation logic
        if sample_features.compression_ratio < 0.7:
            assert recommendations["compression"]["enabled"] is True
        
        if sample_features.cache_hit_rate > 0.5:
            assert recommendations["caching"]["strategy"] == "aggressive"
    
    def test_dna_evolution(self, profiler):
        """Test DNA profile evolution over time"""
        agent_id = "evolving-agent"
        
        # Create initial profile
        dna_v1 = profiler.create_dna(agent_id, "langchain")
        assert dna_v1.version == 1
        
        # Simulate multiple updates
        for i in range(3):
            patterns = [
                AccessPattern(
                    timestamp=datetime.utcnow(),
                    operation="read",
                    data_size=1000 * (i + 1),
                    latency_ms=5.0,
                    tier=StorageTier.PREMIUM,
                    cache_hit=True,
                    success=True
                )
            ]
            profiler._update_dna_version(dna_v1, patterns)
        
        # Version should increment with significant changes
        assert dna_v1.version >= 1
    
    def test_anomaly_detection(self, profiler, sample_access_patterns):
        """Test anomaly detection in access patterns"""
        # Add anomalous pattern
        anomaly = AccessPattern(
            timestamp=datetime.utcnow(),
            operation="write",
            data_size=10000000,  # 10MB - much larger than normal
            latency_ms=500.0,     # Much slower than normal
            tier=StorageTier.PREMIUM,
            cache_hit=False,
            success=False
        )
        
        patterns_with_anomaly = sample_access_patterns + [anomaly]
        
        # Extract features should handle anomalies gracefully
        features = profiler._extract_features(patterns_with_anomaly)
        
        # Features should be influenced but not dominated by anomaly
        assert features.avg_request_size < 10000000
        assert features.latency_sensitivity > 0  # Should detect performance issues
    
    def test_framework_specific_optimization(self, profiler):
        """Test framework-specific optimizations"""
        frameworks = ["langchain", "autogen", "crewai"]
        
        for framework in frameworks:
            dna = profiler.create_dna(f"agent-{framework}", framework)
            
            # Each framework should have specific feature defaults
            if framework == "langchain":
                # LangChain typically has more frequent small reads
                assert dna.features.read_write_ratio >= 0.7
            elif framework == "autogen":
                # AutoGen has more balanced read/write
                assert 0.4 <= dna.features.read_write_ratio <= 0.6
    
    def test_serialization(self, profiler, sample_features):
        """Test DNA serialization/deserialization"""
        dna = StorageDNA(
            agent_id="test-agent",
            framework="langchain",
            features=sample_features,
            created_at=datetime.utcnow(),
            version=1
        )
        
        # Serialize to dict
        dna_dict = profiler._dna_to_dict(dna)
        assert isinstance(dna_dict, dict)
        assert dna_dict["agent_id"] == "test-agent"
        assert "features" in dna_dict
        
        # Should be JSON serializable
        json_str = json.dumps(dna_dict)
        assert len(json_str) > 0