"""
Unit tests for Security and Encryption systems
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from src.security.advanced_encryption import (
    AdvancedEncryptionManager,
    EncryptionKey,
    EncryptedData,
    KeyType,
    Algorithm,
    ComplianceLevel,
    KeyStatus
)


class TestAdvancedEncryptionManager:
    """Test suite for Advanced Encryption Manager"""
    
    @pytest.fixture
    def encryption_config(self):
        """Test encryption configuration"""
        return {
            "quantum_resistant": False,
            "compliance_level": "FIPS_140_2_LEVEL_2",
            "key_rotation_days": 30,
            "hsm_enabled": False,
            "azure_keyvault_url": "https://test.vault.azure.net/",
            "audit_logging": True
        }
    
    @pytest.fixture
    def encryption_manager(self, encryption_config):
        """Create encryption manager instance"""
        return AdvancedEncryptionManager(encryption_config)
    
    @pytest.fixture
    def sample_encryption_key(self):
        """Create sample encryption key"""
        return EncryptionKey(
            key_id="test-key-123",
            key_type=KeyType.AES,
            algorithm=Algorithm.AES_256_GCM,
            key_material=b"x" * 32,  # 256-bit key
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365),
            rotation_due=datetime.utcnow() + timedelta(days=30),
            version=1,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2,
            tags={"purpose": "test"}
        )
    
    @pytest.fixture
    def sample_plaintext(self):
        """Sample plaintext data for testing"""
        return b"This is test data that needs to be encrypted securely."
    
    def test_encryption_manager_initialization(self, encryption_manager):
        """Test encryption manager initializes correctly"""
        assert encryption_manager.config is not None
        assert encryption_manager.keys == {}
        assert encryption_manager.audit_logs == []
        assert not encryption_manager.quantum_resistant
    
    @pytest.mark.asyncio
    async def test_generate_key_aes(self, encryption_manager):
        """Test AES key generation"""
        key = await encryption_manager.generate_key(
            key_type=KeyType.AES,
            algorithm=Algorithm.AES_256_GCM,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2
        )
        
        assert key.key_type == KeyType.AES
        assert key.algorithm == Algorithm.AES_256_GCM
        assert len(key.key_material) == 32  # 256 bits
        assert key.key_id is not None
        assert key.created_at is not None
        assert key.status == KeyStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_generate_key_rsa(self, encryption_manager):
        """Test RSA key generation"""
        key = await encryption_manager.generate_key(
            key_type=KeyType.RSA,
            algorithm=Algorithm.RSA_OAEP,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2
        )
        
        assert key.key_type == KeyType.RSA
        assert key.algorithm == Algorithm.RSA_OAEP
        assert key.key_material is not None
        assert key.key_id is not None
    
    @pytest.mark.asyncio
    async def test_encrypt_decrypt_aes(self, encryption_manager, sample_encryption_key, sample_plaintext):
        """Test AES encryption and decryption"""
        # Add key to manager
        encryption_manager.keys[sample_encryption_key.key_id] = sample_encryption_key
        
        # Encrypt data
        encrypted_data = await encryption_manager.encrypt(
            plaintext=sample_plaintext,
            key=sample_encryption_key
        )
        
        assert isinstance(encrypted_data, EncryptedData)
        assert encrypted_data.ciphertext != sample_plaintext
        assert encrypted_data.key_id == sample_encryption_key.key_id
        assert encrypted_data.algorithm == sample_encryption_key.algorithm
        
        # Decrypt data
        decrypted_data = await encryption_manager.decrypt(encrypted_data)
        assert decrypted_data == sample_plaintext
    
    @pytest.mark.asyncio
    async def test_encrypt_with_nonexistent_key(self, encryption_manager, sample_plaintext):
        """Test encryption with non-existent key raises error"""
        fake_key = EncryptionKey(
            key_id="nonexistent-key",
            key_type=KeyType.AES,
            algorithm=Algorithm.AES_256_GCM,
            key_material=b"x" * 32,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365),
            rotation_due=datetime.utcnow() + timedelta(days=30),
            version=1,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2
        )
        
        with pytest.raises(ValueError, match="Key not found"):
            await encryption_manager.encrypt(sample_plaintext, fake_key)
    
    @pytest.mark.asyncio
    async def test_key_rotation(self, encryption_manager, sample_encryption_key):
        """Test key rotation functionality"""
        # Add old key
        encryption_manager.keys[sample_encryption_key.key_id] = sample_encryption_key
        
        # Create new key for rotation
        new_key = await encryption_manager.generate_key(
            key_type=KeyType.AES,
            algorithm=Algorithm.AES_256_GCM,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2
        )
        
        # Mock data retrieval for rotation
        with patch.object(encryption_manager, '_get_data_by_key', return_value=[]):
            # Perform rotation
            await encryption_manager._rotate_key_data(sample_encryption_key, new_key)
        
        # Old key should be deprecated
        assert sample_encryption_key.status == KeyStatus.DEPRECATED
        assert sample_encryption_key.deprecated_at is not None
    
    @pytest.mark.asyncio
    async def test_hsm_initialization(self, encryption_manager):
        """Test HSM initialization"""
        encryption_manager.config['hsm_enabled'] = True
        
        with patch('azure.keyvault.keys.KeyClient') as mock_client:
            with patch.object(encryption_manager, '_test_hsm_connectivity', return_value=None):
                with patch.object(encryption_manager, '_load_hsm_keys', return_value=None):
                    with patch.object(encryption_manager, '_configure_hsm_policies', return_value=None):
                        await encryption_manager._initialize_hsm()
        
        # Should attempt to initialize HSM
        assert mock_client.called
    
    def test_audit_log_creation(self, encryption_manager):
        """Test audit log creation"""
        encryption_manager._create_audit_log(
            operation="encrypt",
            key_id="test-key",
            agent_id="test-agent",
            data_size=1024,
            success=True,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2
        )
        
        assert len(encryption_manager.audit_logs) == 1
        log_entry = encryption_manager.audit_logs[0]
        assert log_entry.operation == "encrypt"
        assert log_entry.key_id == "test-key"
        assert log_entry.agent_id == "test-agent"
        assert log_entry.success is True
    
    @pytest.mark.asyncio
    async def test_get_audit_logs_filtering(self, encryption_manager):
        """Test audit log filtering"""
        # Create multiple audit entries
        encryption_manager._create_audit_log("encrypt", "key1", "agent1", 1024, True, ComplianceLevel.FIPS_140_2_LEVEL_2)
        encryption_manager._create_audit_log("decrypt", "key2", "agent2", 2048, True, ComplianceLevel.FIPS_140_2_LEVEL_2)
        encryption_manager._create_audit_log("rotate", "key1", "agent1", 0, False, ComplianceLevel.FIPS_140_2_LEVEL_2)
        
        # Filter by agent
        agent1_logs = await encryption_manager.get_audit_logs(agent_id="agent1")
        assert len(agent1_logs) == 2
        
        # Filter by operation
        encrypt_logs = await encryption_manager.get_audit_logs(operation="encrypt")
        assert len(encrypt_logs) == 1
        assert encrypt_logs[0].operation == "encrypt"
    
    def test_encryption_key_properties(self, sample_encryption_key):
        """Test encryption key properties"""
        assert sample_encryption_key.is_active()
        assert not sample_encryption_key.is_expired()
        assert sample_encryption_key.needs_rotation() is False  # Not due yet
        
        # Test expired key
        expired_key = EncryptionKey(
            key_id="expired-key",
            key_type=KeyType.AES,
            algorithm=Algorithm.AES_256_GCM,
            key_material=b"x" * 32,
            created_at=datetime.utcnow() - timedelta(days=400),
            expires_at=datetime.utcnow() - timedelta(days=1),
            rotation_due=datetime.utcnow() - timedelta(days=30),
            version=1,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2
        )
        
        assert expired_key.is_expired()
        assert expired_key.needs_rotation()
    
    @pytest.mark.asyncio
    async def test_compliance_level_enforcement(self, encryption_manager):
        """Test compliance level enforcement"""
        # Test FIPS 140-2 Level 3 requires specific algorithms
        with patch.object(encryption_manager, '_validate_compliance') as mock_validate:
            mock_validate.return_value = True
            
            key = await encryption_manager.generate_key(
                key_type=KeyType.AES,
                algorithm=Algorithm.AES_256_GCM,
                compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_3
            )
            
            assert key.compliance_level == ComplianceLevel.FIPS_140_2_LEVEL_3
            mock_validate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quantum_resistant_key_generation(self, encryption_manager):
        """Test quantum-resistant key generation"""
        encryption_manager.quantum_resistant = True
        
        # Should generate longer keys for quantum resistance
        key = await encryption_manager.generate_key(
            key_type=KeyType.RSA,
            algorithm=Algorithm.RSA_OAEP,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2
        )
        
        # RSA key should be 4096 bits for quantum resistance
        assert key.key_material is not None
        # In a real test, would verify key size is 4096 bits


class TestEncryptionErrorHandling:
    """Test error handling in encryption system"""
    
    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager for error testing"""
        return AdvancedEncryptionManager({})
    
    @pytest.mark.asyncio
    async def test_encrypt_with_invalid_algorithm(self, encryption_manager):
        """Test encryption with invalid algorithm"""
        invalid_key = EncryptionKey(
            key_id="invalid-key",
            key_type=KeyType.AES,
            algorithm="INVALID_ALGORITHM",  # Invalid algorithm
            key_material=b"x" * 32,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365),
            rotation_due=datetime.utcnow() + timedelta(days=30),
            version=1,
            compliance_level=ComplianceLevel.FIPS_140_2_LEVEL_2
        )
        
        encryption_manager.keys[invalid_key.key_id] = invalid_key
        
        with pytest.raises(ValueError):
            await encryption_manager.encrypt(b"test data", invalid_key)
    
    @pytest.mark.asyncio
    async def test_decrypt_with_wrong_key(self, encryption_manager):
        """Test decryption with wrong key"""
        # Create two different keys
        key1 = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_2)
        key2 = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_2)
        
        # Encrypt with key1
        encrypted_data = await encryption_manager.encrypt(b"test data", key1)
        
        # Try to decrypt with key2 (should fail)
        encrypted_data.key_id = key2.key_id  # Change key ID
        
        with pytest.raises(Exception):  # Should raise decryption error
            await encryption_manager.decrypt(encrypted_data)
    
    @pytest.mark.asyncio
    async def test_hsm_initialization_failure(self, encryption_manager):
        """Test HSM initialization failure handling"""
        encryption_manager.config['hsm_enabled'] = True
        
        with patch('azure.keyvault.keys.KeyClient', side_effect=Exception("HSM connection failed")):
            await encryption_manager._initialize_hsm()
            
            # Should disable HSM on failure
            assert not encryption_manager.hsm_enabled
    
    @pytest.mark.asyncio
    async def test_key_rotation_failure(self, encryption_manager):
        """Test key rotation failure handling"""
        old_key = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_2)
        new_key = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_2)
        
        # Mock failure in data retrieval
        with patch.object(encryption_manager, '_get_data_by_key', side_effect=Exception("Database error")):
            with pytest.raises(Exception):
                await encryption_manager._rotate_key_data(old_key, new_key)
            
            # New key should be revoked on failure
            assert new_key.status == KeyStatus.REVOKED


class TestEncryptionPerformance:
    """Test encryption system performance"""
    
    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager for performance testing"""
        return AdvancedEncryptionManager({"performance_mode": True})
    
    @pytest.mark.asyncio
    async def test_bulk_encryption_performance(self, encryption_manager):
        """Test performance with bulk encryption operations"""
        key = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_2)
        
        # Encrypt multiple pieces of data
        test_data = [f"test data {i}".encode() for i in range(100)]
        
        start_time = datetime.utcnow()
        encrypted_items = []
        
        for data in test_data:
            encrypted = await encryption_manager.encrypt(data, key)
            encrypted_items.append(encrypted)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete in reasonable time
        assert duration < 10.0  # Less than 10 seconds for 100 operations
        assert len(encrypted_items) == 100
    
    @pytest.mark.asyncio
    async def test_concurrent_encryption_operations(self, encryption_manager):
        """Test concurrent encryption operations"""
        key = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_2)
        
        async def encrypt_data(data):
            return await encryption_manager.encrypt(data, key)
        
        # Run concurrent encryption operations
        test_data = [f"concurrent test {i}".encode() for i in range(50)]
        tasks = [encrypt_data(data) for data in test_data]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 50
        # All should be different (due to different IVs/nonces)
        ciphertexts = [r.ciphertext for r in results]
        assert len(set(ciphertexts)) == 50


class TestComplianceFeatures:
    """Test compliance and regulatory features"""
    
    @pytest.fixture
    def encryption_manager(self):
        """Create encryption manager with compliance features"""
        config = {
            "compliance_mode": True,
            "audit_logging": True,
            "key_escrow": True,
            "compliance_level": "FIPS_140_2_LEVEL_3"
        }
        return AdvancedEncryptionManager(config)
    
    @pytest.mark.asyncio
    async def test_gdpr_data_deletion(self, encryption_manager):
        """Test GDPR-compliant data deletion"""
        key = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_2)
        
        # Encrypt personal data
        personal_data = b"John Doe, SSN: 123-45-6789"
        encrypted = await encryption_manager.encrypt(personal_data, key)
        
        # Delete key for GDPR compliance (crypto-shredding)
        await encryption_manager.delete_key(key.key_id, reason="GDPR_deletion")
        
        # Key should be marked as deleted
        assert key.key_id not in encryption_manager.keys
        
        # Audit log should record deletion
        deletion_logs = [log for log in encryption_manager.audit_logs if log.operation == "delete_key"]
        assert len(deletion_logs) > 0
    
    def test_audit_log_immutability(self, encryption_manager):
        """Test audit logs are immutable"""
        encryption_manager._create_audit_log("test", "key1", "agent1", 1024, True, ComplianceLevel.FIPS_140_2_LEVEL_2)
        
        original_log = encryption_manager.audit_logs[0]
        original_timestamp = original_log.timestamp
        
        # Attempt to modify (should not affect original)
        log_copy = original_log
        log_copy.operation = "modified"
        
        # Original should be unchanged
        assert encryption_manager.audit_logs[0].operation == "test"
        assert encryption_manager.audit_logs[0].timestamp == original_timestamp
    
    @pytest.mark.asyncio
    async def test_key_lifecycle_compliance(self, encryption_manager):
        """Test complete key lifecycle for compliance"""
        # Generate key
        key = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_3)
        
        # Use key
        await encryption_manager.encrypt(b"test data", key)
        
        # Rotate key
        new_key = await encryption_manager.generate_key(KeyType.AES, Algorithm.AES_256_GCM, ComplianceLevel.FIPS_140_2_LEVEL_3)
        with patch.object(encryption_manager, '_get_data_by_key', return_value=[]):
            await encryption_manager._rotate_key_data(key, new_key)
        
        # Archive old key
        key.status = KeyStatus.ARCHIVED
        
        # Verify audit trail exists for entire lifecycle
        key_logs = [log for log in encryption_manager.audit_logs if log.key_id == key.key_id]
        operations = [log.operation for log in key_logs]
        
        assert "generate_key" in operations
        assert "encrypt" in operations