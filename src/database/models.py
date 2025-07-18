"""
AgentVault™ Database Models
Comprehensive SQLAlchemy ORM models for production use
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

from datetime import datetime
from typing import Optional, Dict, List, Any
from enum import Enum
import uuid

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, 
    ForeignKey, Table, BigInteger, LargeBinary, Numeric, Index,
    UniqueConstraint, CheckConstraint, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY, TSVECTOR
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import expression

Base = declarative_base()

# Association tables for many-to-many relationships
agent_roles = Table(
    'agent_roles',
    Base.metadata,
    Column('agent_id', UUID(as_uuid=True), ForeignKey('agents.id', ondelete='CASCADE')),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE')),
    Column('assigned_at', DateTime, default=datetime.utcnow),
    Column('assigned_by', UUID(as_uuid=True), ForeignKey('users.id')),
    UniqueConstraint('agent_id', 'role_id', name='uq_agent_role')
)

role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE')),
    Column('permission_id', UUID(as_uuid=True), ForeignKey('permissions.id', ondelete='CASCADE')),
    UniqueConstraint('role_id', 'permission_id', name='uq_role_permission')
)

user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE')),
    Column('role_id', UUID(as_uuid=True), ForeignKey('roles.id', ondelete='CASCADE')),
    Column('assigned_at', DateTime, default=datetime.utcnow),
    UniqueConstraint('user_id', 'role_id', name='uq_user_role')
)


class AgentState(str, Enum):
    """Agent lifecycle states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    TERMINATED = "terminated"
    MIGRATING = "migrating"
    ARCHIVED = "archived"


class StorageTier(str, Enum):
    """Storage tier types"""
    ULTRA = "ultra"
    PREMIUM = "premium"
    STANDARD = "standard"
    COOL = "cool"
    ARCHIVE = "archive"


class ReplicationState(str, Enum):
    """Replication states"""
    HEALTHY = "healthy"
    SYNCING = "syncing"
    BROKEN = "broken"
    PAUSED = "paused"
    INITIALIZING = "initializing"


class Agent(Base):
    """
    Core agent entity representing an AI agent instance
    Tracks complete lifecycle, configuration, and relationships
    """
    __tablename__ = 'agents'
    
    # Primary identification
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Multi-tenancy support
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Agent configuration
    agent_type = Column(String(100), nullable=False)  # langchain, autogen, crewai, custom
    framework_version = Column(String(50))
    configuration = Column(JSONB, nullable=False, default={})
    capabilities = Column(ARRAY(String), default=[])
    tags = Column(ARRAY(String), default=[])
    
    # State management
    state = Column(String(50), default=AgentState.INITIALIZING)
    state_metadata = Column(JSONB, default={})
    last_state_change = Column(DateTime, default=datetime.utcnow)
    state_history = Column(JSONB, default=[])
    
    # Resource allocation
    cpu_cores = Column(Float, default=1.0)
    memory_gb = Column(Float, default=4.0)
    gpu_enabled = Column(Boolean, default=False)
    gpu_memory_gb = Column(Float, default=0.0)
    storage_gb = Column(Float, default=10.0)
    
    # Networking
    internal_endpoint = Column(String(500))
    external_endpoint = Column(String(500))
    api_key_hash = Column(String(255))
    allowed_ips = Column(ARRAY(String), default=[])
    
    # Performance tracking
    total_requests = Column(BigInteger, default=0)
    total_errors = Column(BigInteger, default=0)
    average_latency_ms = Column(Float, default=0.0)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    # Lifecycle timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    activated_at = Column(DateTime)
    deactivated_at = Column(DateTime)
    deleted_at = Column(DateTime)  # Soft delete
    
    # Relationships
    tenant = relationship("Tenant", back_populates="agents")
    owner = relationship("User", back_populates="owned_agents", foreign_keys=[owner_id])
    profile = relationship("AgentProfile", back_populates="agent", uselist=False, cascade="all, delete-orphan")
    storage_volumes = relationship("StorageVolume", back_populates="agent", cascade="all, delete-orphan")
    metrics = relationship("PerformanceMetric", back_populates="agent", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="agent", cascade="all, delete-orphan")
    vector_embeddings = relationship("VectorEmbedding", back_populates="agent", cascade="all, delete-orphan")
    ml_models = relationship("MLModel", back_populates="agent", cascade="all, delete-orphan")
    roles = relationship("Role", secondary=agent_roles, back_populates="agents")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_agent_tenant', 'tenant_id'),
        Index('idx_agent_owner', 'owner_id'),
        Index('idx_agent_state', 'state'),
        Index('idx_agent_type', 'agent_type'),
        Index('idx_agent_name', 'name'),
        Index('idx_agent_created', 'created_at'),
        Index('idx_agent_active', 'last_active'),
        Index('idx_agent_deleted', 'deleted_at'),
        UniqueConstraint('tenant_id', 'name', name='uq_tenant_agent_name'),
    )
    
    @hybrid_property
    def is_active(self):
        """Check if agent is in active state"""
        return self.state in [AgentState.READY, AgentState.RUNNING]
    
    @hybrid_property
    def uptime_percentage(self):
        """Calculate uptime percentage"""
        if not self.activated_at:
            return 0.0
        total_time = (datetime.utcnow() - self.activated_at).total_seconds()
        if total_time == 0:
            return 0.0
        # Calculate from state history
        if hasattr(self, 'state_history') and self.state_history:
            # Calculate actual uptime from state transitions
            running_time = 0
            last_running_start = None
            
            for state_change in self.state_history:
                if state_change.get('state') == 'running':
                    last_running_start = datetime.fromisoformat(state_change.get('timestamp'))
                elif state_change.get('state') in ['stopped', 'error', 'failed'] and last_running_start:
                    running_time += (datetime.fromisoformat(state_change.get('timestamp')) - last_running_start).total_seconds()
                    last_running_start = None
            
            # Add current running time if agent is currently running
            if self.state == 'running' and last_running_start:
                running_time += (datetime.utcnow() - last_running_start).total_seconds()
            
            uptime_percentage = (running_time / total_time) * 100
            return min(100.0, max(0.0, uptime_percentage))
        
        # Fallback: estimate based on current state
        if self.state == 'running':
            return 98.0  # Assume high uptime for running agents
        elif self.state in ['stopped', 'paused']:
            return 85.0  # Lower uptime for stopped agents
        else:
            return 60.0  # Lower uptime for error states
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            'id': str(self.id),
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'agent_type': self.agent_type,
            'state': self.state,
            'capabilities': self.capabilities,
            'tags': self.tags,
            'resources': {
                'cpu_cores': self.cpu_cores,
                'memory_gb': self.memory_gb,
                'gpu_enabled': self.gpu_enabled,
                'gpu_memory_gb': self.gpu_memory_gb,
                'storage_gb': self.storage_gb
            },
            'performance': {
                'total_requests': self.total_requests,
                'total_errors': self.total_errors,
                'average_latency_ms': self.average_latency_ms,
                'uptime_percentage': self.uptime_percentage
            },
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_active': self.last_active.isoformat() if self.last_active else None
        }


class AgentProfile(Base):
    """
    Extended agent profile with ML-driven characteristics and behavior patterns
    """
    __tablename__ = 'agent_profiles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, unique=True)
    
    # Behavioral characteristics
    workload_pattern = Column(JSONB, default={})  # Time-series workload data
    io_characteristics = Column(JSONB, default={})  # Read/write patterns
    memory_usage_pattern = Column(JSONB, default={})
    network_pattern = Column(JSONB, default={})
    
    # ML-derived insights
    predicted_resource_needs = Column(JSONB, default={})
    anomaly_score = Column(Float, default=0.0)
    performance_score = Column(Float, default=100.0)
    reliability_score = Column(Float, default=100.0)
    
    # Optimization settings
    auto_scaling_enabled = Column(Boolean, default=True)
    preferred_storage_tier = Column(String(50), default=StorageTier.PREMIUM)
    cache_strategy = Column(String(50), default='adaptive')
    compression_enabled = Column(Boolean, default=True)
    
    # DNA fingerprint
    dna_fingerprint = Column(JSONB, default={})
    dna_version = Column(Integer, default=1)
    last_dna_update = Column(DateTime, default=datetime.utcnow)
    
    # Custom metadata
    custom_attributes = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="profile")
    
    __table_args__ = (
        Index('idx_profile_agent', 'agent_id'),
        Index('idx_profile_anomaly', 'anomaly_score'),
        Index('idx_profile_performance', 'performance_score'),
    )


class StorageVolume(Base):
    """
    Storage volume allocation for agents with complete lifecycle management
    """
    __tablename__ = 'storage_volumes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False)
    
    # Volume identification
    volume_name = Column(String(255), nullable=False)
    volume_path = Column(String(500), nullable=False)
    mount_point = Column(String(500))
    
    # Storage configuration
    storage_tier = Column(String(50), default=StorageTier.PREMIUM)
    size_gb = Column(Float, nullable=False)
    used_gb = Column(Float, default=0.0)
    iops_limit = Column(Integer)
    throughput_mbps = Column(Integer)
    
    # ANF specific
    anf_volume_id = Column(String(500))
    anf_capacity_pool_id = Column(String(500))
    anf_subnet_id = Column(String(500))
    service_level = Column(String(50))  # Ultra, Premium, Standard
    
    # Encryption and security
    encryption_enabled = Column(Boolean, default=True)
    encryption_key_id = Column(String(500))
    access_mode = Column(String(50), default='ReadWriteMany')
    
    # Performance metrics
    read_iops = Column(Float, default=0.0)
    write_iops = Column(Float, default=0.0)
    read_throughput_mbps = Column(Float, default=0.0)
    write_throughput_mbps = Column(Float, default=0.0)
    average_latency_ms = Column(Float, default=0.0)
    
    # Lifecycle
    state = Column(String(50), default='provisioning')
    health_status = Column(String(50), default='healthy')
    last_health_check = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime)
    
    # Relationships
    agent = relationship("Agent", back_populates="storage_volumes")
    snapshots = relationship("StorageSnapshot", back_populates="volume", cascade="all, delete-orphan")
    replications = relationship("StorageReplication", foreign_keys='StorageReplication.source_volume_id', 
                               back_populates="source_volume", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_volume_agent', 'agent_id'),
        Index('idx_volume_tier', 'storage_tier'),
        Index('idx_volume_state', 'state'),
        Index('idx_volume_health', 'health_status'),
        UniqueConstraint('agent_id', 'volume_name', name='uq_agent_volume_name'),
    )


class StorageSnapshot(Base):
    """
    Point-in-time snapshots for backup and recovery
    """
    __tablename__ = 'storage_snapshots'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    volume_id = Column(UUID(as_uuid=True), ForeignKey('storage_volumes.id', ondelete='CASCADE'), nullable=False)
    
    # Snapshot details
    snapshot_name = Column(String(255), nullable=False)
    snapshot_path = Column(String(500))
    anf_snapshot_id = Column(String(500))
    
    # Metadata
    size_gb = Column(Float, nullable=False)
    incremental = Column(Boolean, default=True)
    parent_snapshot_id = Column(UUID(as_uuid=True), ForeignKey('storage_snapshots.id'))
    
    # Lifecycle
    state = Column(String(50), default='creating')
    retention_days = Column(Integer, default=7)
    expires_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    deleted_at = Column(DateTime)
    
    # Tags and metadata
    tags = Column(JSONB, default={})
    metadata = Column(JSONB, default={})
    
    # Relationships
    volume = relationship("StorageVolume", back_populates="snapshots")
    parent_snapshot = relationship("StorageSnapshot", remote_side=[id])
    
    __table_args__ = (
        Index('idx_snapshot_volume', 'volume_id'),
        Index('idx_snapshot_created', 'created_at'),
        Index('idx_snapshot_expires', 'expires_at'),
        UniqueConstraint('volume_id', 'snapshot_name', name='uq_volume_snapshot_name'),
    )


class StorageReplication(Base):
    """
    Cross-region replication configuration and status
    """
    __tablename__ = 'storage_replications'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Replication pair
    source_volume_id = Column(UUID(as_uuid=True), ForeignKey('storage_volumes.id', ondelete='CASCADE'), nullable=False)
    target_volume_id = Column(UUID(as_uuid=True), ForeignKey('storage_volumes.id'), nullable=False)
    
    # Configuration
    replication_type = Column(String(50), default='async')  # async, sync, mirror
    schedule = Column(String(100))  # Cron expression
    bandwidth_limit_mbps = Column(Integer)
    
    # Status tracking
    state = Column(String(50), default=ReplicationState.INITIALIZING)
    last_sync_at = Column(DateTime)
    last_sync_duration_seconds = Column(Integer)
    lag_seconds = Column(Integer, default=0)
    bytes_transferred = Column(BigInteger, default=0)
    
    # Health metrics
    health_status = Column(String(50), default='healthy')
    error_count = Column(Integer, default=0)
    last_error = Column(Text)
    last_error_at = Column(DateTime)
    
    # Lifecycle
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    source_volume = relationship("StorageVolume", foreign_keys=[source_volume_id], back_populates="replications")
    target_volume = relationship("StorageVolume", foreign_keys=[target_volume_id])
    
    __table_args__ = (
        Index('idx_replication_source', 'source_volume_id'),
        Index('idx_replication_target', 'target_volume_id'),
        Index('idx_replication_state', 'state'),
        UniqueConstraint('source_volume_id', 'target_volume_id', name='uq_replication_pair'),
    )


class VectorEmbedding(Base):
    """
    Vector embeddings for RAG and semantic search
    """
    __tablename__ = 'vector_embeddings'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False)
    
    # Embedding details
    embedding_type = Column(String(100), nullable=False)  # text, code, image, etc.
    model_name = Column(String(255), nullable=False)  # openai/text-embedding-ada-002, etc.
    dimension = Column(Integer, nullable=False)
    
    # Vector data (stored in specialized vector DB, reference here)
    vector_db_id = Column(String(500))
    vector_index = Column(String(255))
    
    # Source content
    content_hash = Column(String(255), nullable=False)
    content_preview = Column(Text)
    content_metadata = Column(JSONB, default={})
    
    # Search optimization
    search_vector = Column(TSVECTOR)  # PostgreSQL full-text search
    tags = Column(ARRAY(String), default=[])
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    
    # Relationships
    agent = relationship("Agent", back_populates="vector_embeddings")
    
    __table_args__ = (
        Index('idx_embedding_agent', 'agent_id'),
        Index('idx_embedding_type', 'embedding_type'),
        Index('idx_embedding_hash', 'content_hash'),
        Index('idx_embedding_search', 'search_vector', postgresql_using='gin'),
        Index('idx_embedding_tags', 'tags', postgresql_using='gin'),
    )


class PerformanceMetric(Base):
    """
    Time-series performance metrics for agents and system
    """
    __tablename__ = 'performance_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id', ondelete='CASCADE'))
    
    # Metric identification
    metric_name = Column(String(255), nullable=False)
    metric_type = Column(String(50), nullable=False)  # gauge, counter, histogram
    unit = Column(String(50))  # bytes, ms, percentage, etc.
    
    # Values
    value = Column(Numeric(precision=20, scale=6), nullable=False)
    min_value = Column(Numeric(precision=20, scale=6))
    max_value = Column(Numeric(precision=20, scale=6))
    avg_value = Column(Numeric(precision=20, scale=6))
    p95_value = Column(Numeric(precision=20, scale=6))
    p99_value = Column(Numeric(precision=20, scale=6))
    
    # Dimensions/Labels
    dimensions = Column(JSONB, default={})
    
    # Time window
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    period_seconds = Column(Integer, default=60)
    
    # Relationships
    agent = relationship("Agent", back_populates="metrics")
    
    __table_args__ = (
        Index('idx_metric_agent_time', 'agent_id', 'timestamp'),
        Index('idx_metric_name_time', 'metric_name', 'timestamp'),
        Index('idx_metric_timestamp', 'timestamp'),
        Index('idx_metric_dimensions', 'dimensions', postgresql_using='gin'),
    )


class AuditLog(Base):
    """
    Comprehensive audit trail for compliance and security
    """
    __tablename__ = 'audit_logs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Actor information
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'))
    service_account = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Action details
    action = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(500))
    resource_name = Column(String(500))
    
    # Change tracking
    old_value = Column(JSONB)
    new_value = Column(JSONB)
    changes = Column(JSONB)
    
    # Context
    request_id = Column(String(255))
    correlation_id = Column(String(255))
    session_id = Column(String(255))
    
    # Result
    status = Column(String(50), nullable=False)  # success, failure, partial
    error_message = Column(Text)
    duration_ms = Column(Integer)
    
    # Compliance
    compliance_flags = Column(ARRAY(String), default=[])
    data_classification = Column(String(50))  # public, internal, confidential, restricted
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User")
    agent = relationship("Agent", back_populates="audit_logs")
    
    __table_args__ = (
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_user', 'user_id'),
        Index('idx_audit_agent', 'agent_id'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_status', 'status'),
        Index('idx_audit_correlation', 'correlation_id'),
    )


class SecurityEvent(Base):
    """
    Security events and incidents tracking
    """
    __tablename__ = 'security_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Event classification
    event_type = Column(String(100), nullable=False)  # auth_failure, anomaly, policy_violation
    severity = Column(String(20), nullable=False)  # critical, high, medium, low, info
    category = Column(String(100))  # authentication, authorization, data_access, network
    
    # Event details
    title = Column(String(500), nullable=False)
    description = Column(Text)
    details = Column(JSONB, default={})
    
    # Source information
    source_ip = Column(String(45))
    source_user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    source_agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'))
    source_service = Column(String(255))
    
    # Target information
    target_resource_type = Column(String(100))
    target_resource_id = Column(String(500))
    target_user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # Detection and response
    detection_method = Column(String(100))  # rule, ml, manual
    detection_confidence = Column(Float)
    response_actions = Column(JSONB, default=[])
    
    # Investigation
    investigation_status = Column(String(50), default='new')  # new, investigating, resolved, false_positive
    assigned_to = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resolution_notes = Column(Text)
    
    # Timestamps
    occurred_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime)
    
    # Relationships
    source_user = relationship("User", foreign_keys=[source_user_id])
    target_user = relationship("User", foreign_keys=[target_user_id])
    assigned_user = relationship("User", foreign_keys=[assigned_to])
    
    __table_args__ = (
        Index('idx_security_event_time', 'occurred_at'),
        Index('idx_security_event_type', 'event_type'),
        Index('idx_security_event_severity', 'severity'),
        Index('idx_security_event_status', 'investigation_status'),
        Index('idx_security_event_source', 'source_ip', 'source_user_id'),
    )


class CacheEntry(Base):
    """
    Distributed cache entries for tracking and management
    """
    __tablename__ = 'cache_entries'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Cache key and namespace
    cache_key = Column(String(500), nullable=False)
    namespace = Column(String(255), default='default')
    
    # Value reference (actual value in Redis)
    value_size_bytes = Column(BigInteger)
    value_hash = Column(String(255))
    compression_type = Column(String(50))
    
    # TTL and expiration
    ttl_seconds = Column(Integer)
    expires_at = Column(DateTime)
    
    # Access patterns
    access_count = Column(BigInteger, default=0)
    last_accessed_at = Column(DateTime, default=datetime.utcnow)
    hit_rate = Column(Float, default=0.0)
    
    # Cache metadata
    cache_tier = Column(String(50), default='memory')  # memory, disk, distributed
    node_id = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_cache_key', 'cache_key'),
        Index('idx_cache_namespace', 'namespace'),
        Index('idx_cache_expires', 'expires_at'),
        Index('idx_cache_accessed', 'last_accessed_at'),
        UniqueConstraint('namespace', 'cache_key', name='uq_namespace_cache_key'),
    )


class MLModel(Base):
    """
    Machine learning models used by agents
    """
    __tablename__ = 'ml_models'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agents.id'))
    
    # Model identification
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=False)  # classification, regression, nlp, vision
    framework = Column(String(100))  # tensorflow, pytorch, sklearn
    version = Column(String(50), nullable=False)
    
    # Model storage
    storage_path = Column(String(500))
    model_size_bytes = Column(BigInteger)
    checksum = Column(String(255))
    
    # Model metadata
    hyperparameters = Column(JSONB, default={})
    metrics = Column(JSONB, default={})  # accuracy, loss, etc.
    training_data_info = Column(JSONB, default={})
    
    # Performance characteristics
    inference_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    gpu_required = Column(Boolean, default=False)
    
    # Deployment status
    deployment_status = Column(String(50), default='inactive')
    deployed_at = Column(DateTime)
    deployment_endpoint = Column(String(500))
    
    # Lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deprecated_at = Column(DateTime)
    
    # Relationships
    agent = relationship("Agent", back_populates="ml_models")
    
    __table_args__ = (
        Index('idx_ml_model_agent', 'agent_id'),
        Index('idx_ml_model_type', 'model_type'),
        Index('idx_ml_model_status', 'deployment_status'),
        UniqueConstraint('agent_id', 'model_name', 'version', name='uq_agent_model_version'),
    )


class BackupJob(Base):
    """
    Backup and restore job tracking
    """
    __tablename__ = 'backup_jobs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Job details
    job_type = Column(String(50), nullable=False)  # backup, restore
    backup_type = Column(String(50), nullable=False)  # full, incremental, differential
    
    # Source and target
    source_type = Column(String(100))  # agent, volume, database
    source_id = Column(String(500))
    target_location = Column(String(500))
    
    # Job configuration
    schedule = Column(String(100))  # Cron expression
    retention_days = Column(Integer, default=30)
    compression_enabled = Column(Boolean, default=True)
    encryption_enabled = Column(Boolean, default=True)
    
    # Progress tracking
    status = Column(String(50), default='pending')
    progress_percentage = Column(Float, default=0.0)
    bytes_processed = Column(BigInteger, default=0)
    total_bytes = Column(BigInteger)
    
    # Timing
    scheduled_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Result
    success = Column(Boolean)
    error_message = Column(Text)
    backup_size_bytes = Column(BigInteger)
    backup_manifest = Column(JSONB, default={})
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_backup_job_status', 'status'),
        Index('idx_backup_job_type', 'job_type'),
        Index('idx_backup_job_scheduled', 'scheduled_at'),
        Index('idx_backup_job_source', 'source_type', 'source_id'),
    )


class AlertRule(Base):
    """
    Alerting rules for monitoring and incident response
    """
    __tablename__ = 'alert_rules'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Rule definition
    rule_name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))  # performance, security, availability, cost
    
    # Rule configuration
    metric_name = Column(String(255))
    condition = Column(String(50))  # gt, lt, eq, contains, regex
    threshold = Column(Numeric(precision=20, scale=6))
    duration_seconds = Column(Integer, default=300)
    
    # Advanced rules
    query = Column(Text)  # PromQL, SQL, or custom query
    aggregation = Column(String(50))  # avg, sum, max, min, count
    group_by = Column(ARRAY(String), default=[])
    
    # Alert configuration
    severity = Column(String(20), default='warning')  # critical, error, warning, info
    enabled = Column(Boolean, default=True)
    mute_until = Column(DateTime)
    
    # Notification settings
    notification_channels = Column(JSONB, default=[])  # email, slack, webhook, etc.
    notification_interval_minutes = Column(Integer, default=60)
    escalation_policy = Column(JSONB, default={})
    
    # State tracking
    last_evaluation = Column(DateTime)
    last_state = Column(String(50))  # ok, pending, firing
    firing_since = Column(DateTime)
    fire_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_alert_rule_enabled', 'enabled'),
        Index('idx_alert_rule_category', 'category'),
        Index('idx_alert_rule_severity', 'severity'),
        Index('idx_alert_rule_state', 'last_state'),
    )


class User(Base):
    """
    System users with authentication and authorization
    """
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Identity
    email = Column(String(255), nullable=False, unique=True)
    username = Column(String(100), unique=True)
    full_name = Column(String(255))
    
    # Authentication
    azure_ad_id = Column(String(255), unique=True)  # Azure AD object ID
    auth_provider = Column(String(50), default='azure_ad')  # azure_ad, local, saml
    password_hash = Column(String(255))  # For local auth only
    
    # Multi-factor authentication
    mfa_enabled = Column(Boolean, default=True)
    mfa_secret = Column(String(255))
    backup_codes = Column(JSONB, default=[])
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)
    
    # Profile
    department = Column(String(255))
    job_title = Column(String(255))
    phone_number = Column(String(50))
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='en')
    
    # Security
    last_login_at = Column(DateTime)
    last_login_ip = Column(String(45))
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime)
    
    # API access
    api_key_hash = Column(String(255))
    api_key_created_at = Column(DateTime)
    api_rate_limit = Column(Integer, default=1000)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime)
    
    # Relationships
    owned_agents = relationship("Agent", back_populates="owner", foreign_keys=[Agent.owner_id])
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_username', 'username'),
        Index('idx_user_azure_ad', 'azure_ad_id'),
        Index('idx_user_active', 'is_active'),
        Index('idx_user_deleted', 'deleted_at'),
    )


class Role(Base):
    """
    RBAC roles for fine-grained permissions
    """
    __tablename__ = 'roles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Role definition
    name = Column(String(100), nullable=False, unique=True)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Role type
    role_type = Column(String(50), default='custom')  # built-in, custom
    scope = Column(String(50), default='tenant')  # global, tenant, project
    
    # Configuration
    priority = Column(Integer, default=100)
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")
    users = relationship("User", secondary=user_roles, back_populates="roles")
    agents = relationship("Agent", secondary=agent_roles, back_populates="roles")
    
    __table_args__ = (
        Index('idx_role_name', 'name'),
        Index('idx_role_type', 'role_type'),
        Index('idx_role_active', 'is_active'),
    )


class Permission(Base):
    """
    Granular permissions for RBAC
    """
    __tablename__ = 'permissions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Permission definition
    name = Column(String(100), nullable=False, unique=True)
    resource = Column(String(100), nullable=False)  # agent, storage, user, etc.
    action = Column(String(100), nullable=False)  # create, read, update, delete, execute
    
    # Additional constraints
    conditions = Column(JSONB, default={})  # Additional conditions for permission
    
    # Metadata
    description = Column(Text)
    category = Column(String(100))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")
    
    __table_args__ = (
        Index('idx_permission_name', 'name'),
        Index('idx_permission_resource', 'resource'),
        Index('idx_permission_action', 'action'),
        UniqueConstraint('resource', 'action', name='uq_resource_action'),
    )


class APIKey(Base):
    """
    API keys for programmatic access
    """
    __tablename__ = 'api_keys'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Key details
    key_name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    key_prefix = Column(String(20))  # Visible prefix for identification
    
    # Permissions
    scopes = Column(ARRAY(String), default=[])
    allowed_ips = Column(ARRAY(String), default=[])
    
    # Usage tracking
    last_used_at = Column(DateTime)
    last_used_ip = Column(String(45))
    usage_count = Column(BigInteger, default=0)
    
    # Rate limiting
    rate_limit_per_hour = Column(Integer, default=1000)
    rate_limit_per_day = Column(Integer, default=10000)
    
    # Lifecycle
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    revoked_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        Index('idx_api_key_user', 'user_id'),
        Index('idx_api_key_hash', 'key_hash'),
        Index('idx_api_key_prefix', 'key_prefix'),
        Index('idx_api_key_active', 'is_active'),
        Index('idx_api_key_expires', 'expires_at'),
    )


class Tenant(Base):
    """
    Multi-tenancy support for enterprise isolation
    """
    __tablename__ = 'tenants'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Tenant information
    name = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(255))
    description = Column(Text)
    
    # Configuration
    azure_subscription_id = Column(String(255))
    azure_resource_group = Column(String(255))
    azure_ad_tenant_id = Column(String(255))
    
    # Resource limits
    max_agents = Column(Integer, default=100)
    max_storage_gb = Column(Integer, default=10000)
    max_users = Column(Integer, default=50)
    
    # Billing
    billing_plan = Column(String(50), default='standard')  # free, standard, premium, enterprise
    billing_contact_email = Column(String(255))
    trial_ends_at = Column(DateTime)
    
    # Features
    enabled_features = Column(JSONB, default=[])
    custom_config = Column(JSONB, default={})
    
    # Status
    is_active = Column(Boolean, default=True)
    is_trial = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime)
    
    # Relationships
    agents = relationship("Agent", back_populates="tenant", cascade="all, delete-orphan")
    resource_quotas = relationship("ResourceQuota", back_populates="tenant", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_tenant_name', 'name'),
        Index('idx_tenant_active', 'is_active'),
        Index('idx_tenant_plan', 'billing_plan'),
        Index('idx_tenant_deleted', 'deleted_at'),
    )


class ResourceQuota(Base):
    """
    Resource quotas and usage tracking per tenant
    """
    __tablename__ = 'resource_quotas'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenants.id', ondelete='CASCADE'), nullable=False)
    
    # Resource type
    resource_type = Column(String(100), nullable=False)  # cpu, memory, storage, agents, etc.
    resource_unit = Column(String(50))  # cores, GB, count, etc.
    
    # Limits
    soft_limit = Column(Numeric(precision=20, scale=6))
    hard_limit = Column(Numeric(precision=20, scale=6), nullable=False)
    
    # Current usage
    current_usage = Column(Numeric(precision=20, scale=6), default=0)
    peak_usage = Column(Numeric(precision=20, scale=6), default=0)
    peak_usage_at = Column(DateTime)
    
    # Alert thresholds
    warning_threshold_percent = Column(Integer, default=80)
    critical_threshold_percent = Column(Integer, default=90)
    
    # Enforcement
    enforcement_action = Column(String(50), default='block')  # block, throttle, alert_only
    is_enforced = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reset_at = Column(DateTime)  # For time-based quotas
    
    # Relationships
    tenant = relationship("Tenant", back_populates="resource_quotas")
    
    __table_args__ = (
        Index('idx_quota_tenant', 'tenant_id'),
        Index('idx_quota_type', 'resource_type'),
        Index('idx_quota_usage', 'current_usage'),
        UniqueConstraint('tenant_id', 'resource_type', name='uq_tenant_resource_type'),
    )