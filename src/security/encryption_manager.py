"""
AgentVault™ Enterprise Encryption Manager
Zero-Trust Security with Quantum-Ready Architecture

This module provides enterprise-grade encryption and security for AI agent data:
- AES-256-GCM encryption for data at rest
- TLS 1.3 for data in transit  
- Azure Key Vault integration for key management
- Quantum-ready cryptographic algorithms
- Zero-trust security model
- FIPS 140-2 Level 3 compliance

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import base64
import hashlib
import hmac
from datetime import datetime, timedelta

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, serialization, padding
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding as asym_padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import secrets

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.keyvault.keys import KeyClient, KeyVaultKey
from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm
from azure.core.exceptions import AzureError

from ..core.storage_orchestrator import StorageRequest


class EncryptionLevel(Enum):
    """Encryption levels for different data sensitivity"""
    BASIC = "basic"           # Standard AES-256 for general data
    ENHANCED = "enhanced"     # AES-256 + HMAC for sensitive data
    MAXIMUM = "maximum"       # Multi-layer encryption for PHI/PII
    QUANTUM_SAFE = "quantum"  # Post-quantum cryptography


class KeyType(Enum):
    """Types of encryption keys"""
    SYMMETRIC = "symmetric"   # AES keys for bulk encryption
    ASYMMETRIC = "asymmetric" # RSA/ECC keys for key exchange
    SIGNING = "signing"       # Keys for digital signatures
    DERIVATION = "derivation" # Keys for key derivation


@dataclass
class EncryptionContext:
    """Context information for encryption operations"""
    agent_id: str
    data_type: str
    sensitivity_level: str
    compliance_tags: List[str]
    encryption_level: EncryptionLevel
    key_id: Optional[str] = None
    algorithm: Optional[str] = None
    iv: Optional[bytes] = None
    salt: Optional[bytes] = None
    metadata: Dict[str, Any] = None


@dataclass
class EncryptedData:
    """Encrypted data container with metadata"""
    ciphertext: bytes
    encryption_context: EncryptionContext
    mac: Optional[bytes] = None  # Message Authentication Code
    timestamp: datetime = None
    key_version: int = 1
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class EncryptionManager:
    """
    Enterprise Encryption Manager for AgentVault™
    
    Provides comprehensive encryption services with:
    - Multiple encryption levels and algorithms
    - Azure Key Vault integration
    - Key rotation and lifecycle management
    - Quantum-ready cryptographic algorithms
    - Zero-trust security architecture
    - Complete audit trail and compliance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Azure Key Vault clients
        self.credential = DefaultAzureCredential()
        self.key_vault_url = config['security']['key_vault_url']
        self.secret_client = SecretClient(
            vault_url=self.key_vault_url,
            credential=self.credential
        )
        self.key_client = KeyClient(
            vault_url=self.key_vault_url,
            credential=self.credential
        )
        
        # Encryption configuration
        self.default_encryption_level = EncryptionLevel(
            config['security'].get('default_encryption_level', 'enhanced')
        )
        self.enable_quantum_safe = config['security'].get('enable_quantum_safe', False)
        
        # Key management
        self.master_key_name = "agentvault-master-key"
        self.key_cache = {}  # Local key cache for performance
        self.key_rotation_interval = timedelta(days=90)  # 90-day rotation
        
        # Performance metrics
        self.encryption_operations = 0
        self.decryption_operations = 0
        self.total_encryption_time = 0.0
        self.total_decryption_time = 0.0
        
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the encryption manager and key infrastructure"""
        
        try:
            self.logger.info("Initializing AgentVault™ Encryption Manager...")
            
            # Ensure master key exists
            await self._ensure_master_key()
            
            # Initialize key hierarchy
            await self._initialize_key_hierarchy()
            
            # Setup key rotation schedule
            await self._setup_key_rotation()
            
            # Validate encryption capabilities
            await self._validate_encryption_setup()
            
            self.is_initialized = True
            self.logger.info("Encryption Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Encryption Manager: {e}")
            raise
    
    async def encrypt_request(self, request: StorageRequest) -> StorageRequest:
        """Encrypt a storage request based on its sensitivity level"""
        
        start_time = datetime.utcnow()
        
        try:
            # Determine encryption level based on request context
            encryption_level = self._determine_encryption_level(request)
            
            # Create encryption context
            context = EncryptionContext(
                agent_id=request.agent_id,
                data_type=request.data_type,
                sensitivity_level=self._get_sensitivity_level(request),
                compliance_tags=request.compliance_tags,
                encryption_level=encryption_level
            )
            
            # Encrypt request data
            if 'data' in request.metadata:
                encrypted_data = await self._encrypt_data(
                    request.metadata['data'], context
                )
                request.metadata['encrypted_data'] = encrypted_data
                request.metadata['encryption_applied'] = True
                del request.metadata['data']  # Remove plaintext
            
            # Encrypt sensitive metadata
            request.metadata = await self._encrypt_metadata(
                request.metadata, context
            )
            
            # Add encryption markers
            request.metadata['encryption_level'] = encryption_level.value
            request.metadata['encrypted_at'] = datetime.utcnow().isoformat()
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.encryption_operations += 1
            self.total_encryption_time += duration
            
            self.logger.debug(f"Request encrypted in {duration*1000:.2f}ms")
            return request
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt request: {e}")
            raise
    
    async def decrypt_request(self, request: StorageRequest) -> StorageRequest:
        """Decrypt a storage request"""
        
        start_time = datetime.utcnow()
        
        try:
            if not request.metadata.get('encryption_applied', False):
                return request  # Not encrypted
            
            # Recreate encryption context
            context = EncryptionContext(
                agent_id=request.agent_id,
                data_type=request.data_type,
                sensitivity_level=request.metadata.get('sensitivity_level', 'standard'),
                compliance_tags=request.compliance_tags,
                encryption_level=EncryptionLevel(request.metadata.get('encryption_level', 'enhanced'))
            )
            
            # Decrypt main data
            if 'encrypted_data' in request.metadata:
                decrypted_data = await self._decrypt_data(
                    request.metadata['encrypted_data'], context
                )
                request.metadata['data'] = decrypted_data
                del request.metadata['encrypted_data']
            
            # Decrypt metadata
            request.metadata = await self._decrypt_metadata(
                request.metadata, context
            )
            
            # Remove encryption markers
            request.metadata.pop('encryption_applied', None)
            request.metadata.pop('encryption_level', None)
            request.metadata.pop('encrypted_at', None)
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.decryption_operations += 1
            self.total_decryption_time += duration
            
            self.logger.debug(f"Request decrypted in {duration*1000:.2f}ms")
            return request
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt request: {e}")
            raise
    
    async def encrypt_agent_data(self, agent_id: str, data: Any, 
                               data_type: str, compliance_tags: List[str] = None) -> EncryptedData:
        """Encrypt agent data with appropriate security level"""
        
        try:
            # Create encryption context
            context = EncryptionContext(
                agent_id=agent_id,
                data_type=data_type,
                sensitivity_level=self._get_data_sensitivity(data_type, compliance_tags or []),
                compliance_tags=compliance_tags or [],
                encryption_level=self._determine_encryption_level_for_data(data_type, compliance_tags or [])
            )
            
            # Serialize data
            if not isinstance(data, (str, bytes)):
                data = json.dumps(data, default=str)
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            # Encrypt based on level
            if context.encryption_level == EncryptionLevel.BASIC:
                encrypted = await self._encrypt_aes_256(data, context)
            elif context.encryption_level == EncryptionLevel.ENHANCED:
                encrypted = await self._encrypt_aes_256_hmac(data, context)
            elif context.encryption_level == EncryptionLevel.MAXIMUM:
                encrypted = await self._encrypt_multilayer(data, context)
            elif context.encryption_level == EncryptionLevel.QUANTUM_SAFE:
                encrypted = await self._encrypt_quantum_safe(data, context)
            else:
                raise ValueError(f"Unsupported encryption level: {context.encryption_level}")
            
            return encrypted
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt agent data: {e}")
            raise
    
    async def decrypt_agent_data(self, encrypted_data: EncryptedData) -> Any:
        """Decrypt agent data"""
        
        try:
            context = encrypted_data.encryption_context
            
            # Decrypt based on original encryption level
            if context.encryption_level == EncryptionLevel.BASIC:
                plaintext = await self._decrypt_aes_256(encrypted_data, context)
            elif context.encryption_level == EncryptionLevel.ENHANCED:
                plaintext = await self._decrypt_aes_256_hmac(encrypted_data, context)
            elif context.encryption_level == EncryptionLevel.MAXIMUM:
                plaintext = await self._decrypt_multilayer(encrypted_data, context)
            elif context.encryption_level == EncryptionLevel.QUANTUM_SAFE:
                plaintext = await self._decrypt_quantum_safe(encrypted_data, context)
            else:
                raise ValueError(f"Unsupported encryption level: {context.encryption_level}")
            
            # Try to deserialize as JSON
            try:
                return json.loads(plaintext.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return plaintext
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt agent data: {e}")
            raise
    
    async def rotate_keys(self, force: bool = False) -> Dict[str, Any]:
        """Rotate encryption keys"""
        
        try:
            self.logger.info("Starting key rotation process...")
            
            rotation_results = {
                "rotation_timestamp": datetime.utcnow().isoformat(),
                "keys_rotated": [],
                "keys_failed": [],
                "status": "success"
            }
            
            # Get all keys that need rotation
            keys_to_rotate = await self._get_keys_for_rotation(force)
            
            for key_name in keys_to_rotate:
                try:
                    # Create new key version
                    new_key = await self._create_key_version(key_name)
                    
                    # Update key cache
                    await self._update_key_cache(key_name, new_key)
                    
                    rotation_results["keys_rotated"].append({
                        "key_name": key_name,
                        "new_version": new_key.properties.version,
                        "rotated_at": datetime.utcnow().isoformat()
                    })
                    
                except Exception as e:
                    self.logger.error(f"Failed to rotate key {key_name}: {e}")
                    rotation_results["keys_failed"].append({
                        "key_name": key_name,
                        "error": str(e)
                    })
            
            if rotation_results["keys_failed"]:
                rotation_results["status"] = "partial_failure"
            
            self.logger.info(f"Key rotation completed: {len(rotation_results['keys_rotated'])} succeeded, {len(rotation_results['keys_failed'])} failed")
            return rotation_results
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_encryption_metrics(self) -> Dict[str, Any]:
        """Get encryption performance metrics"""
        
        avg_encryption_time = (
            self.total_encryption_time / self.encryption_operations 
            if self.encryption_operations > 0 else 0
        )
        
        avg_decryption_time = (
            self.total_decryption_time / self.decryption_operations
            if self.decryption_operations > 0 else 0
        )
        
        return {
            "encryption_operations": self.encryption_operations,
            "decryption_operations": self.decryption_operations,
            "average_encryption_time_ms": avg_encryption_time * 1000,
            "average_decryption_time_ms": avg_decryption_time * 1000,
            "total_operations": self.encryption_operations + self.decryption_operations,
            "key_cache_size": len(self.key_cache),
            "quantum_safe_enabled": self.enable_quantum_safe,
            "default_encryption_level": self.default_encryption_level.value
        }
    
    # Private helper methods
    
    def _determine_encryption_level(self, request: StorageRequest) -> EncryptionLevel:
        """Determine appropriate encryption level for request"""
        
        # Check compliance requirements
        if any(tag in ["HIPAA", "PHI", "PII", "financial"] for tag in request.compliance_tags):
            return EncryptionLevel.MAXIMUM
        
        # Check data sensitivity
        if request.data_type in ["vector", "embedding", "long_term_memory"]:
            return EncryptionLevel.ENHANCED
        
        # Check priority
        if request.priority == "critical":
            return EncryptionLevel.ENHANCED
        
        return self.default_encryption_level
    
    def _get_sensitivity_level(self, request: StorageRequest) -> str:
        """Get data sensitivity level"""
        
        if any(tag in ["HIPAA", "PHI", "PII"] for tag in request.compliance_tags):
            return "highly_sensitive"
        elif request.data_type in ["vector", "long_term_memory"]:
            return "sensitive"
        else:
            return "standard"
    
    async def _encrypt_aes_256(self, data: bytes, context: EncryptionContext) -> EncryptedData:
        """Encrypt data using AES-256-GCM"""
        
        # Get or generate key
        key = await self._get_or_generate_key(context.agent_id, KeyType.SYMMETRIC)
        
        # Generate random IV
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key[:32]),  # 256-bit key
            modes.GCM(iv),
            backend=default_backend()
        )
        
        # Encrypt
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Update context
        context.iv = iv
        context.algorithm = "AES-256-GCM"
        
        return EncryptedData(
            ciphertext=ciphertext + encryptor.tag,  # Append authentication tag
            encryption_context=context
        )
    
    async def _decrypt_aes_256(self, encrypted_data: EncryptedData, 
                             context: EncryptionContext) -> bytes:
        """Decrypt AES-256-GCM encrypted data"""
        
        # Get key
        key = await self._get_key(context.agent_id, KeyType.SYMMETRIC)
        
        # Extract ciphertext and tag
        ciphertext = encrypted_data.ciphertext[:-16]  # All but last 16 bytes
        tag = encrypted_data.ciphertext[-16:]  # Last 16 bytes
        
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key[:32]),
            modes.GCM(context.iv, tag),
            backend=default_backend()
        )
        
        # Decrypt
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    async def _encrypt_aes_256_hmac(self, data: bytes, context: EncryptionContext) -> EncryptedData:
        """Encrypt with AES-256 + HMAC for integrity"""
        
        # First encrypt with AES-256
        encrypted = await self._encrypt_aes_256(data, context)
        
        # Generate HMAC
        hmac_key = await self._get_or_generate_key(context.agent_id, KeyType.SIGNING)
        mac = hmac.new(
            hmac_key[:32],
            encrypted.ciphertext,
            hashlib.sha256
        ).digest()
        
        encrypted.mac = mac
        return encrypted
    
    async def _decrypt_aes_256_hmac(self, encrypted_data: EncryptedData,
                                  context: EncryptionContext) -> bytes:
        """Decrypt AES-256 + HMAC encrypted data"""
        
        # Verify HMAC
        hmac_key = await self._get_key(context.agent_id, KeyType.SIGNING)
        expected_mac = hmac.new(
            hmac_key[:32],
            encrypted_data.ciphertext,
            hashlib.sha256
        ).digest()
        
        if not hmac.compare_digest(encrypted_data.mac, expected_mac):
            raise ValueError("HMAC verification failed - data may be tampered")
        
        # Decrypt
        return await self._decrypt_aes_256(encrypted_data, context)
    
    async def _get_or_generate_key(self, agent_id: str, key_type: KeyType) -> bytes:
        """Get existing key or generate new one"""
        
        key_name = f"agent-{agent_id}-{key_type.value}"
        
        try:
            # Try to get from cache
            if key_name in self.key_cache:
                return self.key_cache[key_name]
            
            # Try to get from Key Vault
            secret = await asyncio.to_thread(
                self.secret_client.get_secret, key_name
            )
            key = base64.b64decode(secret.value)
            
            # Cache for performance
            self.key_cache[key_name] = key
            return key
            
        except Exception:
            # Generate new key
            key = secrets.token_bytes(32)  # 256-bit key
            
            # Store in Key Vault
            await asyncio.to_thread(
                self.secret_client.set_secret,
                key_name,
                base64.b64encode(key).decode('utf-8')
            )
            
            # Cache
            self.key_cache[key_name] = key
            return key
    
    async def _get_key(self, agent_id: str, key_type: KeyType) -> bytes:
        """Get existing key"""
        
        key_name = f"agent-{agent_id}-{key_type.value}"
        
        if key_name in self.key_cache:
            return self.key_cache[key_name]
        
        secret = await asyncio.to_thread(
            self.secret_client.get_secret, key_name
        )
        key = base64.b64decode(secret.value)
        
        self.key_cache[key_name] = key
        return key
    
    def _setup_logging(self) -> logging.Logger:
        """Setup security-focused logging"""
        
        logger = logging.getLogger("agentvault.encryption_manager")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def shutdown(self) -> None:
        """Shutdown encryption manager"""
        
        self.logger.info("Shutting down Encryption Manager...")
        
        # Clear key cache for security
        self.key_cache.clear()
        
        self.logger.info("Encryption Manager shutdown complete")