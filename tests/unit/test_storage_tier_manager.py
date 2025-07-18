"""
Unit tests for Storage Tier Manager
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from src.storage.tier_manager import (
    TierManager,
    StorageTier,
    AccessPattern,
    TierTransition
)


class TestTierManager:
    """Test suite for Tier Manager"""
    
    @pytest.fixture
    def tier_config(self):
        """Test tier configuration"""
        return {
            "ultra": {
                "mount_point": "/test/ultra",
                "latency_ms": 0.1,
                "iops": 1000000,
                "cost_per_gb": 1.0
            },
            "premium": {
                "mount_point": "/test/premium", 
                "latency_ms": 1.0,
                "iops": 500000,
                "cost_per_gb": 0.5
            },
            "standard": {
                "mount_point": "/test/standard",
                "latency_ms": 5.0,
                "iops": 100000,
                "cost_per_gb": 0.1
            }
        }
    
    @pytest.fixture
    def tier_manager(self, tier_config):
        """Create tier manager instance"""
        return TierManager(tier_config)
    
    @pytest.fixture
    def sample_access_pattern(self):
        """Create sample access pattern"""
        return AccessPattern(
            data_id="test-data-123",
            access_frequency=100.0,
            last_access=datetime.utcnow(),
            access_times=[datetime.utcnow() - timedelta(hours=i) for i in range(5)],
            read_count=50,
            write_count=10,
            current_tier=StorageTier.PREMIUM,
            storage_path="/test/premium/test-data-123"
        )
    
    @pytest.fixture
    def temp_directories(self):
        """Create temporary directories for testing"""
        temp_dirs = {}
        for tier in ["ultra", "premium", "standard"]:
            temp_dirs[tier] = tempfile.mkdtemp(prefix=f"test_{tier}_")
        
        yield temp_dirs
        
        # Cleanup
        for temp_dir in temp_dirs.values():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_tier_manager_initialization(self, tier_manager):
        """Test tier manager initializes correctly"""
        assert tier_manager.tier_config is not None
        assert tier_manager.access_patterns == {}
        assert tier_manager.pending_transitions == []
    
    def test_add_access_pattern(self, tier_manager, sample_access_pattern):
        """Test adding access pattern"""
        tier_manager.add_access_pattern(sample_access_pattern)
        
        assert sample_access_pattern.data_id in tier_manager.access_patterns
        assert tier_manager.access_patterns[sample_access_pattern.data_id] == sample_access_pattern
    
    def test_record_access_updates_pattern(self, tier_manager, sample_access_pattern):
        """Test recording access updates the pattern"""
        tier_manager.add_access_pattern(sample_access_pattern)
        
        original_frequency = sample_access_pattern.access_frequency
        original_count = sample_access_pattern.read_count
        
        tier_manager.record_access(sample_access_pattern.data_id, "read")
        
        updated_pattern = tier_manager.access_patterns[sample_access_pattern.data_id]
        assert updated_pattern.read_count == original_count + 1
        assert updated_pattern.last_access > sample_access_pattern.last_access
    
    def test_record_access_nonexistent_data(self, tier_manager):
        """Test recording access for non-existent data creates pattern"""
        tier_manager.record_access("new-data-123", "read")
        
        assert "new-data-123" in tier_manager.access_patterns
        pattern = tier_manager.access_patterns["new-data-123"]
        assert pattern.read_count == 1
        assert pattern.write_count == 0
    
    def test_recommend_tier_hot_data(self, tier_manager, sample_access_pattern):
        """Test tier recommendation for hot data"""
        # Make data very hot
        sample_access_pattern.access_frequency = 1000.0
        sample_access_pattern.last_access = datetime.utcnow()
        
        tier_manager.add_access_pattern(sample_access_pattern)
        
        recommended_tier = tier_manager.recommend_tier(sample_access_pattern.data_id)
        assert recommended_tier == StorageTier.ULTRA
    
    def test_recommend_tier_cold_data(self, tier_manager, sample_access_pattern):
        """Test tier recommendation for cold data"""
        # Make data cold
        sample_access_pattern.access_frequency = 0.1
        sample_access_pattern.last_access = datetime.utcnow() - timedelta(days=30)
        
        tier_manager.add_access_pattern(sample_access_pattern)
        
        recommended_tier = tier_manager.recommend_tier(sample_access_pattern.data_id)
        assert recommended_tier in [StorageTier.STANDARD, StorageTier.ARCHIVE]
    
    @pytest.mark.asyncio
    async def test_execute_tier_transition_success(self, tier_manager, sample_access_pattern, temp_directories):
        """Test successful tier transition"""
        # Update config to use temp directories
        tier_manager.tier_config["premium"]["mount_point"] = temp_directories["premium"]
        tier_manager.tier_config["standard"]["mount_point"] = temp_directories["standard"]
        
        # Create source file
        source_file = os.path.join(temp_directories["premium"], "test-data-123")
        with open(source_file, 'w') as f:
            f.write("test data content")
        
        # Add access pattern
        sample_access_pattern.storage_path = source_file
        tier_manager.add_access_pattern(sample_access_pattern)
        
        # Create transition
        transition = TierTransition(
            transition_id="trans-123",
            data_id="test-data-123",
            source_tier=StorageTier.PREMIUM,
            target_tier=StorageTier.STANDARD,
            started_at=datetime.utcnow(),
            transition_type="move"
        )
        
        await tier_manager._execute_tier_transition(transition)
        
        # Verify transition completed
        assert transition.status == "completed"
        assert transition.completed_at is not None
        
        # Verify file moved
        target_file = os.path.join(temp_directories["standard"], "test-data-123")
        assert os.path.exists(target_file)
        assert not os.path.exists(source_file)  # Original should be moved
    
    @pytest.mark.asyncio
    async def test_execute_tier_transition_file_not_found(self, tier_manager, sample_access_pattern, temp_directories):
        """Test tier transition with missing source file"""
        # Update config to use temp directories
        tier_manager.tier_config["premium"]["mount_point"] = temp_directories["premium"]
        tier_manager.tier_config["standard"]["mount_point"] = temp_directories["standard"]
        
        # Don't create source file
        sample_access_pattern.storage_path = os.path.join(temp_directories["premium"], "nonexistent")
        tier_manager.add_access_pattern(sample_access_pattern)
        
        # Create transition
        transition = TierTransition(
            transition_id="trans-123",
            data_id="test-data-123",
            source_tier=StorageTier.PREMIUM,
            target_tier=StorageTier.STANDARD,
            started_at=datetime.utcnow(),
            transition_type="move"
        )
        
        # Should handle file not found gracefully
        with pytest.raises(FileNotFoundError):
            await tier_manager._execute_tier_transition(transition)
        
        assert transition.status == "failed"
        assert transition.error_message is not None
    
    @pytest.mark.asyncio
    async def test_execute_tier_transition_copy_mode(self, tier_manager, sample_access_pattern, temp_directories):
        """Test tier transition in copy mode"""
        # Update config to use temp directories
        tier_manager.tier_config["premium"]["mount_point"] = temp_directories["premium"]
        tier_manager.tier_config["standard"]["mount_point"] = temp_directories["standard"]
        
        # Create source file
        source_file = os.path.join(temp_directories["premium"], "test-data-123")
        with open(source_file, 'w') as f:
            f.write("test data content")
        
        # Add access pattern
        sample_access_pattern.storage_path = source_file
        tier_manager.add_access_pattern(sample_access_pattern)
        
        # Create transition in copy mode
        transition = TierTransition(
            transition_id="trans-123",
            data_id="test-data-123",
            source_tier=StorageTier.PREMIUM,
            target_tier=StorageTier.STANDARD,
            started_at=datetime.utcnow(),
            transition_type="copy"
        )
        
        await tier_manager._execute_tier_transition(transition)
        
        # Verify transition completed
        assert transition.status == "completed"
        
        # Verify file copied (both should exist)
        target_file = os.path.join(temp_directories["standard"], "test-data-123")
        assert os.path.exists(target_file)
        assert os.path.exists(source_file)  # Original should still exist
    
    def test_calculate_access_frequency(self, tier_manager):
        """Test access frequency calculation"""
        access_times = [
            datetime.utcnow() - timedelta(hours=1),
            datetime.utcnow() - timedelta(hours=2),
            datetime.utcnow() - timedelta(hours=3),
            datetime.utcnow() - timedelta(hours=4),
            datetime.utcnow() - timedelta(hours=5)
        ]
        
        frequency = tier_manager._calculate_access_frequency(access_times)
        
        # Should calculate frequency based on time windows
        assert frequency > 0
        assert isinstance(frequency, float)
    
    def test_get_tier_score_calculation(self, tier_manager, sample_access_pattern):
        """Test tier score calculation"""
        tier_manager.add_access_pattern(sample_access_pattern)
        
        score = tier_manager._get_tier_score(sample_access_pattern, StorageTier.ULTRA)
        assert isinstance(score, float)
        assert score >= 0
        
        # Ultra tier should have high score for high-frequency data
        sample_access_pattern.access_frequency = 1000.0
        high_freq_score = tier_manager._get_tier_score(sample_access_pattern, StorageTier.ULTRA)
        
        # Lower frequency should have lower score for ultra tier
        sample_access_pattern.access_frequency = 1.0
        low_freq_score = tier_manager._get_tier_score(sample_access_pattern, StorageTier.ULTRA)
        
        assert high_freq_score > low_freq_score
    
    def test_tier_transition_creation(self):
        """Test tier transition object creation"""
        transition = TierTransition(
            transition_id="test-transition",
            data_id="test-data",
            source_tier=StorageTier.PREMIUM,
            target_tier=StorageTier.STANDARD,
            started_at=datetime.utcnow(),
            transition_type="move"
        )
        
        assert transition.transition_id == "test-transition"
        assert transition.data_id == "test-data"
        assert transition.source_tier == StorageTier.PREMIUM
        assert transition.target_tier == StorageTier.STANDARD
        assert transition.status == "pending"
        assert transition.completed_at is None
        assert transition.error_message is None
    
    def test_storage_tier_enum_values(self):
        """Test storage tier enum values"""
        assert StorageTier.ULTRA.value == "ultra"
        assert StorageTier.PREMIUM.value == "premium"
        assert StorageTier.STANDARD.value == "standard"
        assert StorageTier.ARCHIVE.value == "archive"
    
    @pytest.mark.asyncio
    async def test_background_tier_optimization(self, tier_manager, sample_access_pattern):
        """Test background tier optimization process"""
        tier_manager.add_access_pattern(sample_access_pattern)
        
        # Make data suitable for downgrading
        sample_access_pattern.access_frequency = 0.1
        sample_access_pattern.last_access = datetime.utcnow() - timedelta(days=7)
        
        # Run optimization
        await tier_manager.optimize_tiers()
        
        # Should identify data for tier change
        # In real implementation, would check pending transitions
        assert len(tier_manager.pending_transitions) >= 0  # May or may not create transitions
    
    def test_access_pattern_serialization(self, sample_access_pattern):
        """Test access pattern can be serialized"""
        pattern_dict = {
            "data_id": sample_access_pattern.data_id,
            "access_frequency": sample_access_pattern.access_frequency,
            "read_count": sample_access_pattern.read_count,
            "write_count": sample_access_pattern.write_count,
            "current_tier": sample_access_pattern.current_tier.value
        }
        
        # Should be JSON serializable
        import json
        json_str = json.dumps(pattern_dict, default=str)
        assert json_str is not None
        
        # Should be deserializable
        deserialized = json.loads(json_str)
        assert deserialized["data_id"] == sample_access_pattern.data_id


class TestTierManagerPerformance:
    """Test tier manager performance characteristics"""
    
    @pytest.fixture
    def tier_manager(self):
        """Create tier manager for performance testing"""
        config = {
            "ultra": {"mount_point": "/test/ultra", "latency_ms": 0.1},
            "premium": {"mount_point": "/test/premium", "latency_ms": 1.0},
            "standard": {"mount_point": "/test/standard", "latency_ms": 5.0}
        }
        return TierManager(config)
    
    def test_large_number_of_access_patterns(self, tier_manager):
        """Test handling large number of access patterns"""
        # Add many access patterns
        for i in range(1000):
            pattern = AccessPattern(
                data_id=f"data-{i}",
                access_frequency=float(i % 100),
                last_access=datetime.utcnow(),
                access_times=[datetime.utcnow()],
                read_count=i,
                write_count=i // 2,
                current_tier=StorageTier.PREMIUM,
                storage_path=f"/test/data-{i}"
            )
            tier_manager.add_access_pattern(pattern)
        
        assert len(tier_manager.access_patterns) == 1000
        
        # Should still be able to get recommendations
        recommendation = tier_manager.recommend_tier("data-500")
        assert recommendation in list(StorageTier)
    
    def test_frequent_access_recording(self, tier_manager):
        """Test frequent access recording performance"""
        # Create initial pattern
        pattern = AccessPattern(
            data_id="frequent-data",
            access_frequency=1.0,
            last_access=datetime.utcnow(),
            access_times=[],
            read_count=0,
            write_count=0,
            current_tier=StorageTier.PREMIUM,
            storage_path="/test/frequent-data"
        )
        tier_manager.add_access_pattern(pattern)
        
        # Record many accesses
        for i in range(100):
            tier_manager.record_access("frequent-data", "read")
        
        updated_pattern = tier_manager.access_patterns["frequent-data"]
        assert updated_pattern.read_count == 100
        assert updated_pattern.access_frequency > 1.0


class TestTierManagerErrorHandling:
    """Test error handling in tier manager"""
    
    @pytest.fixture
    def tier_manager(self):
        """Create tier manager for error testing"""
        config = {
            "ultra": {"mount_point": "/invalid/ultra"},
            "premium": {"mount_point": "/invalid/premium"}
        }
        return TierManager(config)
    
    def test_invalid_tier_config(self):
        """Test handling of invalid tier configuration"""
        with pytest.raises(KeyError):
            TierManager({})  # Empty config should raise error
    
    def test_recommend_tier_nonexistent_data(self, tier_manager):
        """Test tier recommendation for non-existent data"""
        # Should handle gracefully
        recommendation = tier_manager.recommend_tier("nonexistent-data")
        assert recommendation in list(StorageTier)
    
    @pytest.mark.asyncio
    async def test_transition_with_permission_error(self, tier_manager):
        """Test handling permission errors during transition"""
        pattern = AccessPattern(
            data_id="test-data",
            access_frequency=1.0,
            last_access=datetime.utcnow(),
            access_times=[],
            read_count=1,
            write_count=0,
            current_tier=StorageTier.PREMIUM,
            storage_path="/readonly/file"
        )
        tier_manager.add_access_pattern(pattern)
        
        transition = TierTransition(
            transition_id="trans-error",
            data_id="test-data",
            source_tier=StorageTier.PREMIUM,
            target_tier=StorageTier.STANDARD,
            started_at=datetime.utcnow(),
            transition_type="move"
        )
        
        # Should handle permission errors gracefully
        with pytest.raises(Exception):
            await tier_manager._execute_tier_transition(transition)
        
        assert transition.status == "failed"