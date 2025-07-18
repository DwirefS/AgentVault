"""
Azure NetApp Files Storage Manager for AgentVault™
Enterprise-grade storage management with intelligent tiering and optimization

This module provides direct integration with Azure NetApp Files, offering:
- Multi-tier storage management (Ultra/Premium/Standard/Cool/Archive)
- High-performance NFS/SMB connectivity with sub-millisecond latency
- Enterprise features: snapshots, backup, disaster recovery
- Intelligent data placement and migration
- Real-time performance monitoring and optimization

Author: Dwiref Sharma  
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import json
import os

from azure.identity import DefaultAzureCredential
from azure.mgmt.netapp import NetAppManagementClient
from azure.mgmt.netapp.models import (
    NetAppAccount, CapacityPool, Volume, VolumeProperties,
    ExportPolicyRule, VolumeBackupProperties, SnapshotPolicy
)
from azure.storage.filedatalake import DataLakeServiceClient, FileSystemClient
from azure.core.exceptions import AzureError
import aiofiles
import asyncio

from ..core.storage_orchestrator import StorageTier, AgentStorageProfile


class ANFVolumeType(Enum):
    """Azure NetApp Files volume types optimized for different AI workloads"""
    VECTOR_STORE = "vector_store"        # Ultra performance for vector operations
    MEMORY_CACHE = "memory_cache"        # Premium for active agent memory  
    KNOWLEDGE_BASE = "knowledge_base"    # Standard for RAG datastores
    ACTIVITY_LOG = "activity_log"        # Cool for audit and tracking
    ARCHIVE_STORE = "archive_store"      # Archive for long-term retention


@dataclass
class ANFVolume:
    """Represents an Azure NetApp Files volume with AI-specific metadata"""
    volume_id: str
    name: str
    capacity_pool: str
    service_level: str  # Ultra, Premium, Standard
    size_bytes: int
    mount_path: str
    volume_type: ANFVolumeType
    agent_associations: List[str]
    performance_tier: StorageTier
    encryption_enabled: bool = True
    snapshot_policy: Optional[str] = None
    backup_enabled: bool = True
    created_at: datetime = None
    last_accessed: datetime = None
    utilization_percent: float = 0.0
    iops_current: int = 0
    throughput_current: float = 0.0
    metadata: Dict[str, Any] = None


class ANFStorageManager:
    """
    Azure NetApp Files Storage Manager for AgentVault™
    
    Provides enterprise-grade storage management with intelligent tiering,
    real-time performance optimization, and AI-specific features for
    maximum efficiency and reliability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Azure credentials and clients
        self.credential = DefaultAzureCredential()
        self.netapp_client = NetAppManagementClient(
            self.credential,
            config['azure']['subscription_id']
        )
        
        # ANF configuration
        self.resource_group = config['azure']['resource_group']
        self.location = config['azure']['location']
        self.netapp_account = config['anf']['account_name']
        
        # Volume management
        self.volumes: Dict[str, ANFVolume] = {}
        self.capacity_pools: Dict[str, Any] = {}
        
        # Performance monitoring
        self.performance_metrics = {}
        self.optimization_history = []
        
        # Mount point base path
        self.mount_base = config['anf'].get('mount_base', '/mnt/agentvault')
        
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize ANF storage manager and discover existing resources"""
        try:
            self.logger.info("Initializing Azure NetApp Files Storage Manager...")
            
            # Ensure NetApp account exists
            await self._ensure_netapp_account()
            
            # Initialize capacity pools for different performance tiers
            await self._initialize_capacity_pools()
            
            # Discover existing volumes
            await self._discover_existing_volumes()
            
            # Setup monitoring
            await self._setup_performance_monitoring()
            
            self.is_initialized = True
            self.logger.info("ANF Storage Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ANF Storage Manager: {e}")
            raise
    
    async def create_volume_for_agent(self, agent_id: str, volume_type: ANFVolumeType,
                                    size_gb: int, performance_tier: StorageTier) -> ANFVolume:
        """Create a dedicated ANF volume for an AI agent's specific storage needs"""
        try:
            volume_name = f"agent-{agent_id}-{volume_type.value}"
            self.logger.info(f"Creating ANF volume {volume_name} for agent {agent_id}")
            
            # Determine capacity pool based on performance tier
            pool_name = self._get_capacity_pool_for_tier(performance_tier)
            
            # Create volume properties
            volume_props = self._create_volume_properties(
                volume_name, size_gb, performance_tier, volume_type
            )
            
            # Create the volume
            async_operation = self.netapp_client.volumes.begin_create_or_update(
                resource_group_name=self.resource_group,
                account_name=self.netapp_account,
                pool_name=pool_name,
                volume_name=volume_name,
                body=volume_props
            )
            
            # Wait for completion
            volume_result = await self._wait_for_operation(async_operation)
            
            # Create volume object
            anf_volume = ANFVolume(
                volume_id=volume_result.id,
                name=volume_name,
                capacity_pool=pool_name,
                service_level=performance_tier.value,
                size_bytes=size_gb * 1024**3,
                mount_path=f"{self.mount_base}/{volume_name}",
                volume_type=volume_type,
                agent_associations=[agent_id],
                performance_tier=performance_tier,
                created_at=datetime.utcnow(),
                metadata={'agent_id': agent_id}
            )
            
            # Cache volume
            self.volumes[volume_name] = anf_volume
            
            # Create mount point directory
            await self._create_mount_point(anf_volume.mount_path)
            
            # Apply security and encryption
            await self._apply_volume_security(anf_volume)
            
            # Setup snapshot policy
            await self._setup_snapshot_policy(anf_volume)
            
            self.logger.info(f"ANF volume {volume_name} created successfully")
            return anf_volume
            
        except Exception as e:
            self.logger.error(f"Failed to create volume for agent {agent_id}: {e}")
            raise
    
    async def execute_operation(self, operation: str, tier: StorageTier,
                              location: str, data: Any, agent_id: str) -> Dict[str, Any]:
        """Execute storage operation on appropriate ANF volume"""
        try:
            # Find optimal volume for the operation
            volume = await self._find_optimal_volume(agent_id, tier, operation)
            
            if not volume:
                # Create volume if none exists
                volume_type = self._determine_volume_type(operation, data)
                size_gb = self._calculate_volume_size(data)
                volume = await self.create_volume_for_agent(
                    agent_id, volume_type, size_gb, tier
                )
            
            # Execute the operation
            result = await self._execute_file_operation(volume, operation, data)
            
            # Update volume metrics
            await self._update_volume_metrics(volume, operation)
            
            return {
                'success': True,
                'volume_id': volume.volume_id,
                'mount_path': volume.mount_path,
                'operation': operation,
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to execute operation {operation}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def optimize_storage_placement(self, agent_id: str) -> Dict[str, Any]:
        """Optimize storage placement for an agent based on access patterns"""
        try:
            agent_volumes = [v for v in self.volumes.values() 
                           if agent_id in v.agent_associations]
            
            optimizations = []
            
            for volume in agent_volumes:
                # Analyze access patterns
                access_analysis = await self._analyze_access_patterns(volume)
                
                # Determine if tier migration is beneficial
                optimal_tier = await self._determine_optimal_tier(volume, access_analysis)
                
                if optimal_tier != volume.performance_tier:
                    # Migrate to optimal tier
                    migration_result = await self._migrate_volume_tier(volume, optimal_tier)
                    optimizations.append(migration_result)
            
            return {
                'agent_id': agent_id,
                'optimizations_applied': len(optimizations),
                'details': optimizations
            }
            
        except Exception as e:
            self.logger.error(f"Failed to optimize storage for agent {agent_id}: {e}")
            raise
    
    async def create_snapshot(self, volume_name: str, snapshot_name: str = None) -> str:
        """Create instant snapshot of ANF volume for backup/recovery"""
        try:
            if not snapshot_name:
                snapshot_name = f"{volume_name}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
            
            volume = self.volumes.get(volume_name)
            if not volume:
                raise ValueError(f"Volume {volume_name} not found")
            
            # Create snapshot
            async_operation = self.netapp_client.snapshots.begin_create(
                resource_group_name=self.resource_group,
                account_name=self.netapp_account,
                pool_name=volume.capacity_pool,
                volume_name=volume_name,
                snapshot_name=snapshot_name,
                body={}
            )
            
            snapshot = await self._wait_for_operation(async_operation)
            
            self.logger.info(f"Snapshot {snapshot_name} created for volume {volume_name}")
            return snapshot.id
            
        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {e}")
            raise
    
    async def restore_from_snapshot(self, volume_name: str, snapshot_name: str) -> bool:
        """Restore ANF volume from snapshot for disaster recovery"""
        try:
            # Implementation for snapshot restoration
            # This would use ANF's native snapshot restore capabilities
            
            self.logger.info(f"Restoring volume {volume_name} from snapshot {snapshot_name}")
            
            # ANF snapshot restore operation
            async_operation = self.netapp_client.volumes.begin_revert(
                resource_group_name=self.resource_group,
                account_name=self.netapp_account,
                pool_name=self.volumes[volume_name].capacity_pool,
                volume_name=volume_name,
                body={'snapshot_id': f"/snapshots/{snapshot_name}"}
            )
            
            await self._wait_for_operation(async_operation)
            
            self.logger.info(f"Volume {volume_name} restored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from snapshot: {e}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics for all ANF volumes"""
        try:
            metrics = {}
            
            for volume_name, volume in self.volumes.items():
                volume_metrics = await self._get_volume_performance(volume)
                metrics[volume_name] = {
                    'tier': volume.performance_tier.value,
                    'utilization_percent': volume.utilization_percent,
                    'iops': volume.iops_current,
                    'throughput_mbps': volume.throughput_current,
                    'latency_ms': volume_metrics.get('latency_ms', 0),
                    'agent_count': len(volume.agent_associations)
                }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {}
    
    async def get_storage_utilization(self) -> Dict[str, float]:
        """Get storage utilization by tier"""
        try:
            utilization = {}
            
            for tier in StorageTier:
                tier_volumes = [v for v in self.volumes.values() 
                              if v.performance_tier == tier]
                
                if tier_volumes:
                    total_size = sum(v.size_bytes for v in tier_volumes)
                    used_size = sum(v.size_bytes * v.utilization_percent / 100 
                                  for v in tier_volumes)
                    utilization[tier.value] = (used_size / total_size) * 100
                else:
                    utilization[tier.value] = 0.0
            
            return utilization
            
        except Exception as e:
            self.logger.error(f"Failed to get storage utilization: {e}")
            return {}
    
    async def store_agent_profile(self, profile: AgentStorageProfile) -> None:
        """Store agent profile in dedicated ANF volume"""
        try:
            # Find or create profile storage volume
            profile_volume = await self._get_profile_storage_volume()
            
            # Serialize profile
            profile_data = json.dumps(profile.__dict__, default=str, indent=2)
            
            # Store to ANF volume
            profile_path = f"{profile_volume.mount_path}/profiles/{profile.agent_id}.json"
            
            async with aiofiles.open(profile_path, 'w') as f:
                await f.write(profile_data)
            
            self.logger.debug(f"Agent profile stored for {profile.agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to store agent profile: {e}")
            raise
    
    async def load_all_agent_profiles(self) -> List[AgentStorageProfile]:
        """Load all agent profiles from ANF storage"""
        try:
            profiles = []
            profile_volume = await self._get_profile_storage_volume()
            profiles_dir = f"{profile_volume.mount_path}/profiles"
            
            if os.path.exists(profiles_dir):
                for filename in os.listdir(profiles_dir):
                    if filename.endswith('.json'):
                        profile_path = os.path.join(profiles_dir, filename)
                        
                        async with aiofiles.open(profile_path, 'r') as f:
                            profile_data = json.loads(await f.read())
                            profile = AgentStorageProfile(**profile_data)
                            profiles.append(profile)
            
            self.logger.info(f"Loaded {len(profiles)} agent profiles")
            return profiles
            
        except Exception as e:
            self.logger.error(f"Failed to load agent profiles: {e}")
            return []
    
    # Private helper methods
    
    async def _ensure_netapp_account(self) -> None:
        """Ensure NetApp account exists or create it"""
        try:
            account = self.netapp_client.accounts.get(
                self.resource_group, self.netapp_account
            )
            self.logger.info(f"Using existing NetApp account: {account.name}")
            
        except Exception:
            # Create new NetApp account
            self.logger.info(f"Creating NetApp account: {self.netapp_account}")
            
            account_body = NetAppAccount(
                location=self.location,
                tags={'project': 'agentvault', 'environment': 'production'}
            )
            
            async_operation = self.netapp_client.accounts.begin_create_or_update(
                self.resource_group, self.netapp_account, account_body
            )
            
            await self._wait_for_operation(async_operation)
    
    async def _initialize_capacity_pools(self) -> None:
        """Initialize capacity pools for different performance tiers"""
        
        pools_config = {
            'ultra-pool': {'service_level': 'Ultra', 'size': 4 * 1024**4},  # 4TB
            'premium-pool': {'service_level': 'Premium', 'size': 8 * 1024**4},  # 8TB  
            'standard-pool': {'service_level': 'Standard', 'size': 16 * 1024**4},  # 16TB
        }
        
        for pool_name, config in pools_config.items():
            try:
                pool = self.netapp_client.pools.get(
                    self.resource_group, self.netapp_account, pool_name
                )
                self.capacity_pools[pool_name] = pool
                
            except Exception:
                # Create capacity pool
                self.logger.info(f"Creating capacity pool: {pool_name}")
                
                pool_body = CapacityPool(
                    location=self.location,
                    service_level=config['service_level'],
                    size=config['size'],
                    tags={'tier': config['service_level'].lower()}
                )
                
                async_operation = self.netapp_client.pools.begin_create_or_update(
                    self.resource_group, self.netapp_account, pool_name, pool_body
                )
                
                pool = await self._wait_for_operation(async_operation)
                self.capacity_pools[pool_name] = pool
    
    def _get_capacity_pool_for_tier(self, tier: StorageTier) -> str:
        """Get appropriate capacity pool for storage tier"""
        
        tier_to_pool = {
            StorageTier.ULTRA: 'ultra-pool',
            StorageTier.PREMIUM: 'premium-pool', 
            StorageTier.STANDARD: 'standard-pool',
            StorageTier.COOL: 'standard-pool',
            StorageTier.ARCHIVE: 'standard-pool'
        }
        
        return tier_to_pool.get(tier, 'standard-pool')
    
    def _create_volume_properties(self, name: str, size_gb: int,
                                tier: StorageTier, volume_type: ANFVolumeType) -> Volume:
        """Create ANF volume properties for creation"""
        
        # Convert GB to bytes
        size_bytes = size_gb * 1024**3
        
        # Create export policy for NFS access
        export_rules = [
            ExportPolicyRule(
                rule_index=1,
                unix_read_only=False,
                unix_read_write=True,
                cifs=False,
                nfsv3=False,
                nfsv41=True,
                allowed_clients='0.0.0.0/0',  # Restrict in production
                kerberos5_read_only=False,
                kerberos5_read_write=False,
                kerberos5i_read_only=False,
                kerberos5i_read_write=False,
                kerberos5p_read_only=False,
                kerberos5p_read_write=False
            )
        ]
        
        return Volume(
            location=self.location,
            creation_token=name,
            service_level=tier.value.replace('_performance', '').capitalize(),
            usage_threshold=size_bytes,
            export_policy={'rules': export_rules},
            protocol_types=['NFSv4.1'],
            subnet_id=self.config['anf']['subnet_id'],
            tags={
                'agent_volume_type': volume_type.value,
                'performance_tier': tier.value,
                'project': 'agentvault'
            },
            encryption_key_source='Microsoft.NetApp',
            backup_policy_id=self.config['anf'].get('backup_policy_id'),
            snapshot_policy_id=self.config['anf'].get('snapshot_policy_id')
        )
    
    async def _wait_for_operation(self, async_operation) -> Any:
        """Wait for Azure async operation to complete"""
        
        while not async_operation.done():
            await asyncio.sleep(5)
            
        return async_operation.result()
    
    async def _create_mount_point(self, mount_path: str) -> None:
        """Create mount point directory"""
        
        os.makedirs(mount_path, exist_ok=True)
        self.logger.debug(f"Created mount point: {mount_path}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for ANF manager"""
        
        logger = logging.getLogger("agentvault.anf_manager")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def shutdown(self) -> None:
        """Shutdown ANF storage manager"""
        
        self.logger.info("Shutting down ANF Storage Manager...")
        # Cleanup operations would go here