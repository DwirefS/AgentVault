"""
AgentVault™ Database Module
Enterprise-grade database layer with complete ORM models and schemas
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

from .models import (
    Agent,
    AgentProfile,
    AgentState,
    StorageVolume,
    StorageSnapshot,
    StorageReplication,
    VectorEmbedding,
    PerformanceMetric,
    AuditLog,
    SecurityEvent,
    CacheEntry,
    MLModel,
    BackupJob,
    AlertRule,
    User,
    Role,
    Permission,
    APIKey,
    Tenant,
    ResourceQuota
)
from .database import DatabaseManager, get_db
from .migrations import MigrationManager
from .repositories import (
    AgentRepository,
    StorageRepository,
    MetricsRepository,
    SecurityRepository,
    UserRepository
)

__all__ = [
    'Agent',
    'AgentProfile',
    'AgentState',
    'StorageVolume',
    'StorageSnapshot',
    'StorageReplication',
    'VectorEmbedding',
    'PerformanceMetric',
    'AuditLog',
    'SecurityEvent',
    'CacheEntry',
    'MLModel',
    'BackupJob',
    'AlertRule',
    'User',
    'Role',
    'Permission',
    'APIKey',
    'Tenant',
    'ResourceQuota',
    'DatabaseManager',
    'get_db',
    'MigrationManager',
    'AgentRepository',
    'StorageRepository',
    'MetricsRepository',
    'SecurityRepository',
    'UserRepository'
]