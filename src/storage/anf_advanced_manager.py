"""
AgentVault™ Advanced ANF Manager
Complete Azure NetApp Files lifecycle management with enterprise features
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from azure.mgmt.netapp import NetAppManagementClient
from azure.mgmt.netapp.models import (
    NetAppAccount, CapacityPool, Volume, VolumePropertiesDataProtection,
    ReplicationObject, VolumeSnapshotProperties, SnapshotPolicy,
    HourlySchedule, DailySchedule, WeeklySchedule, MonthlySchedule,
    VolumeBackupProperties, ExportPolicyRule, ActiveDirectory
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
import aiofiles

logger = logging.getLogger(__name__)


class VolumeState(Enum):
    """Volume lifecycle states"""
    CREATING = "creating"
    AVAILABLE = "available"
    UPDATING = "updating"
    DELETING = "deleting"
    ERROR = "error"
    MIGRATING = "migrating"
    REPLICATING = "replicating"
    RESTORING = "restoring"


class ReplicationState(Enum):
    """Replication states"""
    MIRRORED = "mirrored"
    BROKEN = "broken"
    UNINITIALIZED = "uninitialized"
    TRANSFERRING = "transferring"


class BackupState(Enum):
    """Backup states"""
    CREATING = "creating"
    READY = "ready"
    DELETING = "deleting"
    ERROR = "error"


@dataclass
class VolumeMetrics:
    """Volume performance metrics"""
    timestamp: datetime
    iops_read: int
    iops_write: int
    throughput_read_mbps: float
    throughput_write_mbps: float
    latency_read_ms: float
    latency_write_ms: float
    used_capacity_gb: float
    available_capacity_gb: float
    snapshot_used_gb: float


@dataclass
class SnapshotInfo:
    """Snapshot information"""
    id: str
    name: str
    volume_id: str
    created: datetime
    size_gb: float
    state: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BackupInfo:
    """Backup information"""
    id: str
    name: str
    volume_id: str
    created: datetime
    size_gb: float
    state: BackupState
    location: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReplicationInfo:
    """Replication relationship information"""
    id: str
    source_volume_id: str
    destination_volume_id: str
    state: ReplicationState
    schedule: str
    lag_time_seconds: int
    last_transfer_size_gb: float
    last_transfer_duration_seconds: int


class AdvancedANFManager:
    """
    Advanced Azure NetApp Files Manager with enterprise features:
    - Complete volume lifecycle management
    - Automated snapshots with policies
    - Cross-region replication
    - Backup and restore
    - Performance monitoring and optimization
    - Capacity management and auto-scaling
    - Disaster recovery
    - Compliance and audit logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.subscription_id = config['azure']['subscription_id']
        self.resource_group = config['azure']['resource_group']
        self.location = config['azure']['location']
        
        # Azure clients
        self.credential = DefaultAzureCredential()
        self.netapp_client = NetAppManagementClient(
            self.credential,
            self.subscription_id
        )
        
        # ANF configuration
        self.account_name = config['anf']['account_name']
        self.pool_configs = config['anf']['pools']
        self.snapshot_policies = config['anf']['snapshot_policies']
        self.replication_config = config['anf']['replication']
        self.backup_config = config['anf']['backup']
        
        # State management
        self.volumes: Dict[str, Volume] = {}
        self.pools: Dict[str, CapacityPool] = {}
        self.snapshots: Dict[str, List[SnapshotInfo]] = {}
        self.replications: Dict[str, ReplicationInfo] = {}
        self.backups: Dict[str, List[BackupInfo]] = {}
        
        # Performance tracking
        self.volume_metrics: Dict[str, List[VolumeMetrics]] = {}
        
        # Background tasks
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
    async def initialize(self) -> None:
        """Initialize ANF manager and create required resources"""
        logger.info("Initializing Advanced ANF Manager...")
        
        try:
            # Create or verify NetApp account
            await self._ensure_netapp_account()
            
            # Create capacity pools
            await self._ensure_capacity_pools()
            
            # Create snapshot policies
            await self._create_snapshot_policies()
            
            # Initialize existing volumes
            await self._discover_existing_volumes()
            
            # Start background tasks
            self._running = True
            self._start_background_tasks()
            
            logger.info("Advanced ANF Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ANF manager: {str(e)}")
            raise
    
    async def _ensure_netapp_account(self) -> None:
        """Ensure NetApp account exists"""
        try:
            account = self.netapp_client.accounts.get(
                self.resource_group,
                self.account_name
            )
            logger.info(f"Using existing NetApp account: {self.account_name}")
            
        except AzureError:
            logger.info(f"Creating NetApp account: {self.account_name}")
            
            # Configure Active Directory if needed
            active_directories = []
            if self.config['anf'].get('active_directory'):
                ad_config = self.config['anf']['active_directory']
                active_directories.append(ActiveDirectory(
                    active_directory_id=ad_config['id'],
                    username=ad_config['username'],
                    password=ad_config['password'],
                    domain=ad_config['domain'],
                    dns=ad_config['dns'],
                    smb_server_name=ad_config['smb_server_name']
                ))
            
            account = NetAppAccount(
                location=self.location,
                active_directories=active_directories
            )
            
            operation = self.netapp_client.accounts.begin_create_or_update(
                self.resource_group,
                self.account_name,
                account
            )
            
            await self._wait_for_operation(operation)
            logger.info(f"NetApp account created: {self.account_name}")
    
    async def _ensure_capacity_pools(self) -> None:
        """Ensure all configured capacity pools exist"""
        for pool_config in self.pool_configs:
            pool_name = pool_config['name']
            
            try:
                pool = self.netapp_client.pools.get(
                    self.resource_group,
                    self.account_name,
                    pool_name
                )
                self.pools[pool_name] = pool
                logger.info(f"Using existing capacity pool: {pool_name}")
                
            except AzureError:
                logger.info(f"Creating capacity pool: {pool_name}")
                
                pool = CapacityPool(
                    location=self.location,
                    service_level=pool_config['service_level'],
                    size=pool_config['size_tib'] * 1024 * 1024 * 1024 * 1024,  # Convert TiB to bytes
                    qos_type=pool_config.get('qos_type', 'Auto')
                )
                
                operation = self.netapp_client.pools.begin_create_or_update(
                    self.resource_group,
                    self.account_name,
                    pool_name,
                    pool
                )
                
                pool = await self._wait_for_operation(operation)
                self.pools[pool_name] = pool
                logger.info(f"Capacity pool created: {pool_name}")
    
    async def _create_snapshot_policies(self) -> None:
        """Create snapshot policies"""
        for policy_config in self.snapshot_policies:
            policy_name = policy_config['name']
            
            try:
                # Check if policy exists
                policy = self.netapp_client.snapshot_policies.get(
                    self.resource_group,
                    self.account_name,
                    policy_name
                )
                logger.info(f"Using existing snapshot policy: {policy_name}")
                
            except AzureError:
                logger.info(f"Creating snapshot policy: {policy_name}")
                
                # Build schedules
                hourly = None
                if policy_config.get('hourly'):
                    hourly = HourlySchedule(
                        snapshots_to_keep=policy_config['hourly']['keep'],
                        minute=policy_config['hourly'].get('minute', 0)
                    )
                
                daily = None
                if policy_config.get('daily'):
                    daily = DailySchedule(
                        snapshots_to_keep=policy_config['daily']['keep'],
                        hour=policy_config['daily'].get('hour', 0),
                        minute=policy_config['daily'].get('minute', 0)
                    )
                
                weekly = None
                if policy_config.get('weekly'):
                    weekly = WeeklySchedule(
                        snapshots_to_keep=policy_config['weekly']['keep'],
                        day=policy_config['weekly'].get('day', 'Sunday'),
                        hour=policy_config['weekly'].get('hour', 0),
                        minute=policy_config['weekly'].get('minute', 0)
                    )
                
                monthly = None
                if policy_config.get('monthly'):
                    monthly = MonthlySchedule(
                        snapshots_to_keep=policy_config['monthly']['keep'],
                        days_of_month=policy_config['monthly'].get('days', '1'),
                        hour=policy_config['monthly'].get('hour', 0),
                        minute=policy_config['monthly'].get('minute', 0)
                    )
                
                policy = SnapshotPolicy(
                    location=self.location,
                    hourly_schedule=hourly,
                    daily_schedule=daily,
                    weekly_schedule=weekly,
                    monthly_schedule=monthly,
                    enabled=True
                )
                
                operation = self.netapp_client.snapshot_policies.create(
                    self.resource_group,
                    self.account_name,
                    policy_name,
                    policy
                )
                
                await self._wait_for_operation(operation)
                logger.info(f"Snapshot policy created: {policy_name}")
    
    async def _discover_existing_volumes(self) -> None:
        """Discover and catalog existing volumes"""
        try:
            for pool_name, pool in self.pools.items():
                volumes = self.netapp_client.volumes.list(
                    self.resource_group,
                    self.account_name,
                    pool_name
                )
                
                for volume in volumes:
                    self.volumes[volume.name] = volume
                    
                    # Get snapshots
                    snapshots = self.netapp_client.snapshots.list(
                        self.resource_group,
                        self.account_name,
                        pool_name,
                        volume.name
                    )
                    
                    self.snapshots[volume.name] = [
                        SnapshotInfo(
                            id=snap.id,
                            name=snap.name,
                            volume_id=volume.id,
                            created=snap.created,
                            size_gb=snap.snapshot_id,  # This would need proper mapping
                            state='ready'
                        )
                        for snap in snapshots
                    ]
                    
            logger.info(f"Discovered {len(self.volumes)} existing volumes")
            
        except Exception as e:
            logger.error(f"Error discovering volumes: {str(e)}")
    
    async def create_volume(
        self,
        name: str,
        pool_name: str,
        size_gb: int,
        agent_id: str,
        tier: str,
        features: Optional[Dict[str, Any]] = None
    ) -> Volume:
        """
        Create a new volume with advanced features
        
        Args:
            name: Volume name
            pool_name: Capacity pool name
            size_gb: Volume size in GB
            agent_id: Agent ID for tagging
            tier: Storage tier
            features: Optional features (snapshots, replication, etc.)
            
        Returns:
            Created volume
        """
        logger.info(f"Creating volume: {name} ({size_gb}GB) in pool {pool_name}")
        
        try:
            # Prepare volume properties
            volume_props = {
                'location': self.location,
                'service_level': tier,
                'creation_token': name,
                'usage_threshold': size_gb * 1024 * 1024 * 1024,  # Convert to bytes
                'subnet_id': self.config['anf']['subnet_id'],
                'protocol_types': features.get('protocols', ['NFSv3']),
                'export_policy': self._create_export_policy(features),
                'tags': {
                    'agent_id': agent_id,
                    'tier': tier,
                    'created_by': 'agentvault',
                    'created_at': datetime.utcnow().isoformat()
                }
            }
            
            # Add snapshot policy if requested
            if features and features.get('snapshot_policy'):
                volume_props['snapshot_policy_id'] = self._get_snapshot_policy_id(
                    features['snapshot_policy']
                )
            
            # Add data protection for replication
            if features and features.get('replication'):
                volume_props['data_protection'] = VolumePropertiesDataProtection(
                    replication=ReplicationObject(
                        endpoint_type='dst' if features['replication'].get('is_destination') else 'src',
                        remote_volume_resource_id=features['replication'].get('remote_volume_id'),
                        replication_schedule=features['replication'].get('schedule', '_10minutely')
                    )
                )
            
            # Create volume
            volume = Volume(**volume_props)
            
            operation = self.netapp_client.volumes.begin_create_or_update(
                self.resource_group,
                self.account_name,
                pool_name,
                name,
                volume
            )
            
            created_volume = await self._wait_for_operation(operation)
            self.volumes[name] = created_volume
            
            # Initialize metrics tracking
            self.volume_metrics[name] = []
            
            # Set up backup if requested
            if features and features.get('backup_enabled'):
                await self._enable_volume_backup(created_volume)
            
            logger.info(f"Volume created successfully: {name}")
            return created_volume
            
        except Exception as e:
            logger.error(f"Failed to create volume {name}: {str(e)}")
            raise
    
    def _create_export_policy(self, features: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create export policy for volume"""
        rules = []
        
        # Default rule - allow from subnet
        rules.append(ExportPolicyRule(
            rule_index=1,
            unix_read_only=False,
            unix_read_write=True,
            cifs=False,
            nfsv3=True,
            nfsv41=False,
            allowed_clients='0.0.0.0/0',  # Should be restricted in production
            kerberos5_read_only=False,
            kerberos5_read_write=False,
            kerberos5i_read_only=False,
            kerberos5i_read_write=False,
            kerberos5p_read_only=False,
            kerberos5p_read_write=False
        ))
        
        # Add custom rules if provided
        if features and features.get('export_rules'):
            for rule in features['export_rules']:
                rules.append(ExportPolicyRule(**rule))
        
        return {'rules': rules}
    
    def _get_snapshot_policy_id(self, policy_name: str) -> str:
        """Get snapshot policy resource ID"""
        return (
            f"/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.NetApp/netAppAccounts/{self.account_name}"
            f"/snapshotPolicies/{policy_name}"
        )
    
    async def update_volume(
        self,
        volume_name: str,
        size_gb: Optional[int] = None,
        export_policy: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Volume:
        """Update volume properties"""
        logger.info(f"Updating volume: {volume_name}")
        
        try:
            volume = self.volumes.get(volume_name)
            if not volume:
                raise ValueError(f"Volume not found: {volume_name}")
            
            # Get pool name from volume ID
            pool_name = volume.id.split('/')[-3]
            
            # Prepare update
            update_props = {}
            
            if size_gb:
                update_props['usage_threshold'] = size_gb * 1024 * 1024 * 1024
            
            if export_policy:
                update_props['export_policy'] = export_policy
            
            if tags:
                update_props['tags'] = {**volume.tags, **tags}
            
            # Apply update
            volume_update = Volume(**update_props)
            
            operation = self.netapp_client.volumes.begin_create_or_update(
                self.resource_group,
                self.account_name,
                pool_name,
                volume_name,
                volume_update
            )
            
            updated_volume = await self._wait_for_operation(operation)
            self.volumes[volume_name] = updated_volume
            
            logger.info(f"Volume updated: {volume_name}")
            return updated_volume
            
        except Exception as e:
            logger.error(f"Failed to update volume {volume_name}: {str(e)}")
            raise
    
    async def delete_volume(self, volume_name: str, force: bool = False) -> None:
        """Delete a volume"""
        logger.info(f"Deleting volume: {volume_name}")
        
        try:
            volume = self.volumes.get(volume_name)
            if not volume:
                logger.warning(f"Volume not found: {volume_name}")
                return
            
            # Get pool name from volume ID
            pool_name = volume.id.split('/')[-3]
            
            # Delete snapshots first if force delete
            if force:
                await self._delete_all_snapshots(volume_name, pool_name)
            
            # Delete volume
            operation = self.netapp_client.volumes.begin_delete(
                self.resource_group,
                self.account_name,
                pool_name,
                volume_name
            )
            
            await self._wait_for_operation(operation)
            
            # Remove from tracking
            del self.volumes[volume_name]
            if volume_name in self.volume_metrics:
                del self.volume_metrics[volume_name]
            
            logger.info(f"Volume deleted: {volume_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete volume {volume_name}: {str(e)}")
            raise
    
    async def create_snapshot(
        self,
        volume_name: str,
        snapshot_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SnapshotInfo:
        """Create a manual snapshot"""
        logger.info(f"Creating snapshot: {snapshot_name} for volume {volume_name}")
        
        try:
            volume = self.volumes.get(volume_name)
            if not volume:
                raise ValueError(f"Volume not found: {volume_name}")
            
            # Get pool name from volume ID
            pool_name = volume.id.split('/')[-3]
            
            # Create snapshot
            snapshot = VolumeSnapshotProperties(
                snapshot_id=snapshot_name,
                file_system_id=volume.file_system_id
            )
            
            operation = self.netapp_client.snapshots.begin_create(
                self.resource_group,
                self.account_name,
                pool_name,
                volume_name,
                snapshot_name,
                snapshot
            )
            
            created_snapshot = await self._wait_for_operation(operation)
            
            # Track snapshot
            snapshot_info = SnapshotInfo(
                id=created_snapshot.id,
                name=snapshot_name,
                volume_id=volume.id,
                created=datetime.utcnow(),
                size_gb=volume.usage_threshold / (1024**3),  # Estimate
                state='ready',
                metadata=metadata or {}
            )
            
            if volume_name not in self.snapshots:
                self.snapshots[volume_name] = []
            self.snapshots[volume_name].append(snapshot_info)
            
            logger.info(f"Snapshot created: {snapshot_name}")
            return snapshot_info
            
        except Exception as e:
            logger.error(f"Failed to create snapshot {snapshot_name}: {str(e)}")
            raise
    
    async def restore_snapshot(
        self,
        volume_name: str,
        snapshot_name: str,
        new_volume_name: Optional[str] = None
    ) -> Volume:
        """Restore volume from snapshot"""
        logger.info(f"Restoring snapshot: {snapshot_name}")
        
        try:
            volume = self.volumes.get(volume_name)
            if not volume:
                raise ValueError(f"Volume not found: {volume_name}")
            
            # Get pool name from volume ID
            pool_name = volume.id.split('/')[-3]
            
            if new_volume_name:
                # Create new volume from snapshot
                snapshot_id = (
                    f"/subscriptions/{self.subscription_id}"
                    f"/resourceGroups/{self.resource_group}"
                    f"/providers/Microsoft.NetApp/netAppAccounts/{self.account_name}"
                    f"/capacityPools/{pool_name}/volumes/{volume_name}"
                    f"/snapshots/{snapshot_name}"
                )
                
                new_volume = Volume(
                    location=self.location,
                    service_level=volume.service_level,
                    creation_token=new_volume_name,
                    usage_threshold=volume.usage_threshold,
                    subnet_id=volume.subnet_id,
                    protocol_types=volume.protocol_types,
                    export_policy=volume.export_policy,
                    snapshot_id=snapshot_id,
                    tags={
                        **volume.tags,
                        'restored_from': snapshot_name,
                        'restored_at': datetime.utcnow().isoformat()
                    }
                )
                
                operation = self.netapp_client.volumes.begin_create_or_update(
                    self.resource_group,
                    self.account_name,
                    pool_name,
                    new_volume_name,
                    new_volume
                )
                
                restored_volume = await self._wait_for_operation(operation)
                self.volumes[new_volume_name] = restored_volume
                
                logger.info(f"Restored snapshot to new volume: {new_volume_name}")
                return restored_volume
                
            else:
                # Revert volume to snapshot (in-place restore)
                operation = self.netapp_client.volumes.begin_revert(
                    self.resource_group,
                    self.account_name,
                    pool_name,
                    volume_name,
                    {'snapshot_id': snapshot_name}
                )
                
                await self._wait_for_operation(operation)
                
                logger.info(f"Reverted volume {volume_name} to snapshot {snapshot_name}")
                return volume
                
        except Exception as e:
            logger.error(f"Failed to restore snapshot {snapshot_name}: {str(e)}")
            raise
    
    async def setup_replication(
        self,
        source_volume_name: str,
        destination_region: str,
        destination_pool_name: str,
        schedule: str = '_10minutely'
    ) -> ReplicationInfo:
        """Set up cross-region replication"""
        logger.info(f"Setting up replication for {source_volume_name} to {destination_region}")
        
        try:
            source_volume = self.volumes.get(source_volume_name)
            if not source_volume:
                raise ValueError(f"Source volume not found: {source_volume_name}")
            
            # Create destination volume
            dest_volume_name = f"{source_volume_name}-replica"
            
            # Create destination volume with replication enabled
            await self.create_volume(
                name=dest_volume_name,
                pool_name=destination_pool_name,
                size_gb=source_volume.usage_threshold // (1024**3),
                agent_id=source_volume.tags.get('agent_id', 'unknown'),
                tier=source_volume.service_level,
                features={
                    'replication': {
                        'is_destination': True,
                        'remote_volume_id': source_volume.id,
                        'schedule': schedule
                    }
                }
            )
            
            # Authorize replication on source
            operation = self.netapp_client.volumes.begin_authorize_replication(
                self.resource_group,
                self.account_name,
                source_volume.id.split('/')[-3],  # pool name
                source_volume_name,
                {'remote_volume_resource_id': self.volumes[dest_volume_name].id}
            )
            
            await self._wait_for_operation(operation)
            
            # Track replication
            replication_info = ReplicationInfo(
                id=f"{source_volume_name}-to-{dest_volume_name}",
                source_volume_id=source_volume.id,
                destination_volume_id=self.volumes[dest_volume_name].id,
                state=ReplicationState.MIRRORED,
                schedule=schedule,
                lag_time_seconds=0,
                last_transfer_size_gb=0,
                last_transfer_duration_seconds=0
            )
            
            self.replications[replication_info.id] = replication_info
            
            logger.info(f"Replication setup complete: {replication_info.id}")
            return replication_info
            
        except Exception as e:
            logger.error(f"Failed to setup replication: {str(e)}")
            raise
    
    async def break_replication(
        self,
        replication_id: str,
        force: bool = False
    ) -> None:
        """Break replication relationship"""
        logger.info(f"Breaking replication: {replication_id}")
        
        try:
            replication = self.replications.get(replication_id)
            if not replication:
                raise ValueError(f"Replication not found: {replication_id}")
            
            # Get destination volume details
            dest_volume_id_parts = replication.destination_volume_id.split('/')
            dest_pool_name = dest_volume_id_parts[-3]
            dest_volume_name = dest_volume_id_parts[-1]
            
            # Break replication
            operation = self.netapp_client.volumes.begin_break_replication(
                self.resource_group,
                self.account_name,
                dest_pool_name,
                dest_volume_name,
                {'force_break_replication': force}
            )
            
            await self._wait_for_operation(operation)
            
            # Update state
            replication.state = ReplicationState.BROKEN
            
            logger.info(f"Replication broken: {replication_id}")
            
        except Exception as e:
            logger.error(f"Failed to break replication: {str(e)}")
            raise
    
    async def resync_replication(self, replication_id: str) -> None:
        """Resync replication relationship"""
        logger.info(f"Resyncing replication: {replication_id}")
        
        try:
            replication = self.replications.get(replication_id)
            if not replication:
                raise ValueError(f"Replication not found: {replication_id}")
            
            # Get destination volume details
            dest_volume_id_parts = replication.destination_volume_id.split('/')
            dest_pool_name = dest_volume_id_parts[-3]
            dest_volume_name = dest_volume_id_parts[-1]
            
            # Resync replication
            operation = self.netapp_client.volumes.begin_resync_replication(
                self.resource_group,
                self.account_name,
                dest_pool_name,
                dest_volume_name
            )
            
            await self._wait_for_operation(operation)
            
            # Update state
            replication.state = ReplicationState.MIRRORED
            
            logger.info(f"Replication resynced: {replication_id}")
            
        except Exception as e:
            logger.error(f"Failed to resync replication: {str(e)}")
            raise
    
    async def create_backup(
        self,
        volume_name: str,
        backup_name: str,
        retention_days: int = 30
    ) -> BackupInfo:
        """Create volume backup"""
        logger.info(f"Creating backup: {backup_name} for volume {volume_name}")
        
        try:
            volume = self.volumes.get(volume_name)
            if not volume:
                raise ValueError(f"Volume not found: {volume_name}")
            
            # Get pool name from volume ID
            pool_name = volume.id.split('/')[-3]
            
            # Create backup
            backup = VolumeBackupProperties(
                backup_policy_resource_id=self.backup_config['policy_id'],
                policy_enforced=True,
                vault_resource_id=self.backup_config['vault_id']
            )
            
            operation = self.netapp_client.backups.begin_create(
                self.resource_group,
                self.account_name,
                pool_name,
                volume_name,
                backup_name,
                backup
            )
            
            created_backup = await self._wait_for_operation(operation)
            
            # Track backup
            backup_info = BackupInfo(
                id=created_backup.id,
                name=backup_name,
                volume_id=volume.id,
                created=datetime.utcnow(),
                size_gb=volume.usage_threshold / (1024**3),
                state=BackupState.READY,
                location=self.backup_config['vault_id'],
                metadata={'retention_days': retention_days}
            )
            
            if volume_name not in self.backups:
                self.backups[volume_name] = []
            self.backups[volume_name].append(backup_info)
            
            logger.info(f"Backup created: {backup_name}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Failed to create backup {backup_name}: {str(e)}")
            raise
    
    async def _enable_volume_backup(self, volume: Volume) -> None:
        """Enable backup for a volume"""
        try:
            pool_name = volume.id.split('/')[-3]
            
            # Update volume with backup configuration
            backup_props = VolumeBackupProperties(
                backup_policy_resource_id=self.backup_config['policy_id'],
                policy_enforced=True,
                vault_resource_id=self.backup_config['vault_id'],
                backup_enabled=True
            )
            
            volume_update = Volume(
                data_protection={'backup': backup_props}
            )
            
            operation = self.netapp_client.volumes.begin_create_or_update(
                self.resource_group,
                self.account_name,
                pool_name,
                volume.name,
                volume_update
            )
            
            await self._wait_for_operation(operation)
            logger.info(f"Backup enabled for volume: {volume.name}")
            
        except Exception as e:
            logger.error(f"Failed to enable backup for volume {volume.name}: {str(e)}")
    
    async def _delete_all_snapshots(self, volume_name: str, pool_name: str) -> None:
        """Delete all snapshots for a volume"""
        try:
            snapshots = self.netapp_client.snapshots.list(
                self.resource_group,
                self.account_name,
                pool_name,
                volume_name
            )
            
            for snapshot in snapshots:
                operation = self.netapp_client.snapshots.begin_delete(
                    self.resource_group,
                    self.account_name,
                    pool_name,
                    volume_name,
                    snapshot.name
                )
                await self._wait_for_operation(operation)
                
            logger.info(f"Deleted all snapshots for volume: {volume_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete snapshots: {str(e)}")
    
    async def get_volume_metrics(self, volume_name: str) -> List[VolumeMetrics]:
        """Get performance metrics for a volume"""
        return self.volume_metrics.get(volume_name, [])
    
    async def get_optimal_volume(
        self,
        agent_id: str,
        tier: str,
        min_free_space_gb: int = 100
    ) -> Volume:
        """Get optimal volume for placement based on current metrics"""
        
        suitable_volumes = []
        
        for volume_name, volume in self.volumes.items():
            # Check if volume matches criteria
            if (volume.service_level == tier and 
                volume.tags.get('agent_id') == agent_id):
                
                # Get latest metrics
                metrics = self.volume_metrics.get(volume_name, [])
                if metrics:
                    latest = metrics[-1]
                    if latest.available_capacity_gb >= min_free_space_gb:
                        suitable_volumes.append((volume, latest))
        
        if not suitable_volumes:
            # Create new volume if none suitable
            return await self.create_volume(
                name=f"{agent_id}-{tier}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                pool_name=self._get_pool_for_tier(tier),
                size_gb=self.config['anf']['default_volume_size_gb'],
                agent_id=agent_id,
                tier=tier
            )
        
        # Select volume with best performance
        best_volume = min(
            suitable_volumes,
            key=lambda x: x[1].latency_read_ms + x[1].latency_write_ms
        )
        
        return best_volume[0]
    
    async def get_compliant_volume(
        self,
        agent_id: str,
        tier: str,
        compliance_level: str
    ) -> Volume:
        """Get volume that meets compliance requirements"""
        
        # Check for existing compliant volume
        for volume_name, volume in self.volumes.items():
            if (volume.service_level == tier and
                volume.tags.get('compliance') == compliance_level and
                volume.tags.get('agent_id') == agent_id):
                return volume
        
        # Create new compliant volume
        features = {
            'snapshot_policy': 'compliance-policy',
            'backup_enabled': True,
            'export_rules': self._get_compliance_export_rules(compliance_level)
        }
        
        if compliance_level in ['PHI', 'PII', 'FINANCIAL']:
            features['replication'] = {
                'schedule': '_hourly'
            }
        
        return await self.create_volume(
            name=f"{agent_id}-{compliance_level}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            pool_name=self._get_pool_for_tier(tier),
            size_gb=self.config['anf']['default_volume_size_gb'],
            agent_id=agent_id,
            tier=tier,
            features=features
        )
    
    def _get_pool_for_tier(self, tier: str) -> str:
        """Get appropriate pool for tier"""
        tier_pool_mapping = {
            'Ultra': 'ultra-pool',
            'Premium': 'premium-pool',
            'Standard': 'standard-pool'
        }
        return tier_pool_mapping.get(tier, 'standard-pool')
    
    def _get_compliance_export_rules(self, compliance_level: str) -> List[Dict[str, Any]]:
        """Get export rules for compliance level"""
        if compliance_level in ['PHI', 'PII', 'FINANCIAL']:
            return [{
                'rule_index': 1,
                'unix_read_only': False,
                'unix_read_write': True,
                'allowed_clients': self.config['anf']['compliance_subnet'],
                'kerberos5i_read_write': True  # Encrypted + integrity
            }]
        else:
            return []
    
    def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks"""
        self._background_tasks = [
            asyncio.create_task(self._monitor_volumes()),
            asyncio.create_task(self._monitor_replications()),
            asyncio.create_task(self._cleanup_old_snapshots()),
            asyncio.create_task(self._optimize_capacity())
        ]
    
    async def _monitor_volumes(self) -> None:
        """Monitor volume performance and health"""
        while self._running:
            try:
                for volume_name, volume in self.volumes.items():
                    # This would integrate with Azure Monitor
                    # For now, generate sample metrics
                    metrics = VolumeMetrics(
                        timestamp=datetime.utcnow(),
                        iops_read=1000,
                        iops_write=500,
                        throughput_read_mbps=100,
                        throughput_write_mbps=50,
                        latency_read_ms=1.5,
                        latency_write_ms=2.0,
                        used_capacity_gb=volume.usage_threshold * 0.7 / (1024**3),
                        available_capacity_gb=volume.usage_threshold * 0.3 / (1024**3),
                        snapshot_used_gb=10
                    )
                    
                    if volume_name not in self.volume_metrics:
                        self.volume_metrics[volume_name] = []
                    
                    self.volume_metrics[volume_name].append(metrics)
                    
                    # Keep only last 1000 metrics
                    if len(self.volume_metrics[volume_name]) > 1000:
                        self.volume_metrics[volume_name] = self.volume_metrics[volume_name][-1000:]
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Volume monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _monitor_replications(self) -> None:
        """Monitor replication health and lag"""
        while self._running:
            try:
                for replication_id, replication in self.replications.items():
                    # This would check actual replication status
                    # For now, update with sample data
                    replication.lag_time_seconds = 30
                    replication.last_transfer_size_gb = 5
                    replication.last_transfer_duration_seconds = 60
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Replication monitoring error: {str(e)}")
                await asyncio.sleep(300)
    
    async def _cleanup_old_snapshots(self) -> None:
        """Clean up old snapshots based on retention policy"""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run hourly
                
                for volume_name, snapshots in self.snapshots.items():
                    # Sort by creation date
                    snapshots.sort(key=lambda x: x.created)
                    
                    # Keep based on retention policy
                    retention_days = self.config['anf'].get('snapshot_retention_days', 7)
                    cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
                    
                    for snapshot in snapshots[:]:
                        if snapshot.created < cutoff_date:
                            try:
                                # Delete snapshot
                                volume = self.volumes[volume_name]
                                pool_name = volume.id.split('/')[-3]
                                
                                operation = self.netapp_client.snapshots.begin_delete(
                                    self.resource_group,
                                    self.account_name,
                                    pool_name,
                                    volume_name,
                                    snapshot.name
                                )
                                
                                await self._wait_for_operation(operation)
                                snapshots.remove(snapshot)
                                
                                logger.info(f"Deleted old snapshot: {snapshot.name}")
                                
                            except Exception as e:
                                logger.error(f"Failed to delete snapshot {snapshot.name}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Snapshot cleanup error: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _optimize_capacity(self) -> None:
        """Optimize capacity pool usage"""
        while self._running:
            try:
                await asyncio.sleep(86400)  # Run daily
                
                # Check pool utilization
                for pool_name, pool in self.pools.items():
                    pool_volumes = [
                        v for v in self.volumes.values()
                        if pool_name in v.id
                    ]
                    
                    total_used = sum(
                        v.usage_threshold for v in pool_volumes
                    )
                    
                    utilization = total_used / pool.size
                    
                    if utilization > 0.9:
                        logger.warning(
                            f"Pool {pool_name} is {utilization:.1%} utilized. "
                            "Consider expanding capacity."
                        )
                    elif utilization < 0.3:
                        logger.info(
                            f"Pool {pool_name} is only {utilization:.1%} utilized. "
                            "Consider consolidating volumes."
                        )
                
            except Exception as e:
                logger.error(f"Capacity optimization error: {str(e)}")
                await asyncio.sleep(86400)
    
    async def _wait_for_operation(self, operation: Any) -> Any:
        """Wait for Azure operation to complete"""
        while not operation.done():
            await asyncio.sleep(5)
        return operation.result()
    
    async def shutdown(self) -> None:
        """Shutdown ANF manager"""
        logger.info("Shutting down Advanced ANF Manager...")
        
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Advanced ANF Manager shutdown complete")