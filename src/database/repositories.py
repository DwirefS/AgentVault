"""
AgentVault™ Repository Pattern Implementation
Clean data access layer with business logic separation
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

from typing import List, Optional, Dict, Any, Tuple, Type, TypeVar, Generic
from datetime import datetime, timedelta
from uuid import UUID
import logging

from sqlalchemy import select, update, delete, and_, or_, func, text
from sqlalchemy.orm import Session, selectinload, joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

from .models import (
    Agent, AgentProfile, AgentState, StorageVolume, StorageTier,
    PerformanceMetric, AuditLog, User, Role, Permission,
    VectorEmbedding, MLModel, SecurityEvent, Tenant, ResourceQuota,
    StorageSnapshot, StorageReplication, BackupJob, AlertRule
)

logger = logging.getLogger(__name__)

# Type variable for generic repository
T = TypeVar('T')


class BaseRepository(Generic[T]):
    """Base repository with common CRUD operations"""
    
    def __init__(self, model: Type[T], session: Session):
        self.model = model
        self.session = session
    
    def create(self, **kwargs) -> T:
        """Create a new entity"""
        entity = self.model(**kwargs)
        self.session.add(entity)
        self.session.flush()
        return entity
    
    def get(self, id: UUID) -> Optional[T]:
        """Get entity by ID"""
        return self.session.get(self.model, id)
    
    def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination"""
        return self.session.query(self.model).limit(limit).offset(offset).all()
    
    def update(self, id: UUID, **kwargs) -> Optional[T]:
        """Update entity by ID"""
        entity = self.get(id)
        if entity:
            for key, value in kwargs.items():
                setattr(entity, key, value)
            self.session.flush()
        return entity
    
    def delete(self, id: UUID) -> bool:
        """Delete entity by ID"""
        entity = self.get(id)
        if entity:
            self.session.delete(entity)
            self.session.flush()
            return True
        return False
    
    def exists(self, id: UUID) -> bool:
        """Check if entity exists"""
        return self.session.query(
            self.session.query(self.model).filter_by(id=id).exists()
        ).scalar()
    
    def count(self, **filters) -> int:
        """Count entities with optional filters"""
        query = self.session.query(func.count(self.model.id))
        for key, value in filters.items():
            query = query.filter(getattr(self.model, key) == value)
        return query.scalar()


class AgentRepository(BaseRepository[Agent]):
    """Repository for Agent operations with business logic"""
    
    def __init__(self, session: Session):
        super().__init__(Agent, session)
    
    def create_agent(
        self,
        name: str,
        agent_type: str,
        tenant_id: UUID,
        owner_id: UUID,
        **kwargs
    ) -> Agent:
        """Create a new agent with profile"""
        # Create agent
        agent = self.create(
            name=name,
            agent_type=agent_type,
            tenant_id=tenant_id,
            owner_id=owner_id,
            state=AgentState.INITIALIZING,
            **kwargs
        )
        
        # Create associated profile
        profile = AgentProfile(
            agent_id=agent.id,
            workload_pattern={},
            predicted_resource_needs={
                'cpu_cores': agent.cpu_cores,
                'memory_gb': agent.memory_gb,
                'storage_gb': agent.storage_gb
            }
        )
        self.session.add(profile)
        
        # Create audit log
        audit_log = AuditLog(
            agent_id=agent.id,
            user_id=owner_id,
            action='agent.created',
            resource_type='agent',
            resource_id=str(agent.id),
            resource_name=agent.name,
            status='success',
            timestamp=datetime.utcnow()
        )
        self.session.add(audit_log)
        
        self.session.flush()
        return agent
    
    def get_by_name(self, name: str, tenant_id: UUID) -> Optional[Agent]:
        """Get agent by name within tenant"""
        return self.session.query(Agent).filter(
            and_(Agent.name == name, Agent.tenant_id == tenant_id)
        ).first()
    
    def get_active_agents(self, tenant_id: UUID, limit: int = 100) -> List[Agent]:
        """Get all active agents for a tenant"""
        return self.session.query(Agent).filter(
            and_(
                Agent.tenant_id == tenant_id,
                Agent.state.in_([AgentState.READY, AgentState.RUNNING]),
                Agent.deleted_at.is_(None)
            )
        ).options(
            selectinload(Agent.profile),
            selectinload(Agent.storage_volumes)
        ).limit(limit).all()
    
    def get_agents_by_type(self, agent_type: str, tenant_id: UUID) -> List[Agent]:
        """Get agents by type"""
        return self.session.query(Agent).filter(
            and_(
                Agent.agent_type == agent_type,
                Agent.tenant_id == tenant_id,
                Agent.deleted_at.is_(None)
            )
        ).all()
    
    def update_state(
        self,
        agent_id: UUID,
        new_state: AgentState,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Agent]:
        """Update agent state with history tracking"""
        agent = self.get(agent_id)
        if not agent:
            return None
        
        # Update state history
        if not agent.state_history:
            agent.state_history = []
        
        agent.state_history.append({
            'from_state': agent.state,
            'to_state': new_state,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        })
        
        # Update current state
        agent.state = new_state
        agent.state_metadata = metadata or {}
        agent.last_state_change = datetime.utcnow()
        
        # Update activation/deactivation timestamps
        if new_state == AgentState.RUNNING and not agent.activated_at:
            agent.activated_at = datetime.utcnow()
        elif new_state == AgentState.TERMINATED:
            agent.deactivated_at = datetime.utcnow()
        
        self.session.flush()
        return agent
    
    def update_metrics(
        self,
        agent_id: UUID,
        requests: int = 0,
        errors: int = 0,
        latency_ms: float = 0
    ) -> Optional[Agent]:
        """Update agent performance metrics"""
        agent = self.get(agent_id)
        if not agent:
            return None
        
        # Update counters
        agent.total_requests += requests
        agent.total_errors += errors
        
        # Update average latency (exponential moving average)
        if latency_ms > 0 and requests > 0:
            alpha = 0.1  # Smoothing factor
            agent.average_latency_ms = (
                alpha * latency_ms + (1 - alpha) * agent.average_latency_ms
            )
        
        agent.last_active = datetime.utcnow()
        self.session.flush()
        return agent
    
    def soft_delete(self, agent_id: UUID) -> bool:
        """Soft delete an agent"""
        agent = self.get(agent_id)
        if agent and not agent.deleted_at:
            agent.deleted_at = datetime.utcnow()
            agent.state = AgentState.TERMINATED
            self.session.flush()
            return True
        return False
    
    def get_resource_usage(self, tenant_id: UUID) -> Dict[str, float]:
        """Get total resource usage for a tenant"""
        result = self.session.query(
            func.sum(Agent.cpu_cores).label('total_cpu'),
            func.sum(Agent.memory_gb).label('total_memory'),
            func.sum(Agent.storage_gb).label('total_storage'),
            func.count(Agent.id).label('total_agents')
        ).filter(
            and_(
                Agent.tenant_id == tenant_id,
                Agent.deleted_at.is_(None),
                Agent.state != AgentState.TERMINATED
            )
        ).first()
        
        return {
            'cpu_cores': result.total_cpu or 0,
            'memory_gb': result.total_memory or 0,
            'storage_gb': result.total_storage or 0,
            'agent_count': result.total_agents or 0
        }
    
    def search_agents(
        self,
        tenant_id: UUID,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 50
    ) -> List[Agent]:
        """Search agents with full-text search and filters"""
        search_query = self.session.query(Agent).filter(
            and_(
                Agent.tenant_id == tenant_id,
                Agent.deleted_at.is_(None)
            )
        )
        
        # Text search on name and description
        if query:
            search_query = search_query.filter(
                or_(
                    Agent.name.ilike(f'%{query}%'),
                    Agent.display_name.ilike(f'%{query}%'),
                    Agent.description.ilike(f'%{query}%')
                )
            )
        
        # Apply filters
        if filters:
            for key, value in filters.items():
                if hasattr(Agent, key):
                    if isinstance(value, list):
                        search_query = search_query.filter(
                            getattr(Agent, key).in_(value)
                        )
                    else:
                        search_query = search_query.filter(
                            getattr(Agent, key) == value
                        )
        
        return search_query.limit(limit).all()


class StorageRepository(BaseRepository[StorageVolume]):
    """Repository for storage operations"""
    
    def __init__(self, session: Session):
        super().__init__(StorageVolume, session)
    
    def create_volume(
        self,
        agent_id: UUID,
        volume_name: str,
        size_gb: float,
        storage_tier: StorageTier = StorageTier.PREMIUM,
        **kwargs
    ) -> StorageVolume:
        """Create a new storage volume"""
        volume = self.create(
            agent_id=agent_id,
            volume_name=volume_name,
            size_gb=size_gb,
            storage_tier=storage_tier,
            volume_path=f"/mnt/agentvault/{agent_id}/{volume_name}",
            state='provisioning',
            **kwargs
        )
        
        # Create audit log
        agent = self.session.get(Agent, agent_id)
        if agent:
            audit_log = AuditLog(
                agent_id=agent_id,
                action='storage.volume.created',
                resource_type='storage_volume',
                resource_id=str(volume.id),
                resource_name=volume.volume_name,
                status='success',
                new_value={'size_gb': size_gb, 'tier': storage_tier},
                timestamp=datetime.utcnow()
            )
            self.session.add(audit_log)
        
        self.session.flush()
        return volume
    
    def get_volumes_by_agent(self, agent_id: UUID) -> List[StorageVolume]:
        """Get all volumes for an agent"""
        return self.session.query(StorageVolume).filter(
            and_(
                StorageVolume.agent_id == agent_id,
                StorageVolume.deleted_at.is_(None)
            )
        ).options(
            selectinload(StorageVolume.snapshots),
            selectinload(StorageVolume.replications)
        ).all()
    
    def get_volumes_by_tier(self, storage_tier: StorageTier) -> List[StorageVolume]:
        """Get all volumes in a specific tier"""
        return self.session.query(StorageVolume).filter(
            and_(
                StorageVolume.storage_tier == storage_tier,
                StorageVolume.deleted_at.is_(None)
            )
        ).all()
    
    def update_volume_metrics(
        self,
        volume_id: UUID,
        used_gb: float,
        read_iops: float = 0,
        write_iops: float = 0,
        read_throughput_mbps: float = 0,
        write_throughput_mbps: float = 0,
        latency_ms: float = 0
    ) -> Optional[StorageVolume]:
        """Update volume performance metrics"""
        volume = self.get(volume_id)
        if not volume:
            return None
        
        volume.used_gb = used_gb
        
        # Update performance metrics with exponential moving average
        alpha = 0.1
        volume.read_iops = alpha * read_iops + (1 - alpha) * volume.read_iops
        volume.write_iops = alpha * write_iops + (1 - alpha) * volume.write_iops
        volume.read_throughput_mbps = alpha * read_throughput_mbps + (1 - alpha) * volume.read_throughput_mbps
        volume.write_throughput_mbps = alpha * write_throughput_mbps + (1 - alpha) * volume.write_throughput_mbps
        volume.average_latency_ms = alpha * latency_ms + (1 - alpha) * volume.average_latency_ms
        
        volume.last_health_check = datetime.utcnow()
        self.session.flush()
        return volume
    
    def create_snapshot(
        self,
        volume_id: UUID,
        snapshot_name: str,
        retention_days: int = 7,
        incremental: bool = True,
        tags: Optional[Dict[str, str]] = None
    ) -> StorageSnapshot:
        """Create a volume snapshot"""
        volume = self.get(volume_id)
        if not volume:
            raise ValueError(f"Volume {volume_id} not found")
        
        # Get parent snapshot for incremental
        parent_snapshot_id = None
        if incremental:
            last_snapshot = self.session.query(StorageSnapshot).filter(
                and_(
                    StorageSnapshot.volume_id == volume_id,
                    StorageSnapshot.state == 'completed',
                    StorageSnapshot.deleted_at.is_(None)
                )
            ).order_by(StorageSnapshot.created_at.desc()).first()
            
            if last_snapshot:
                parent_snapshot_id = last_snapshot.id
        
        snapshot = StorageSnapshot(
            volume_id=volume_id,
            snapshot_name=snapshot_name,
            snapshot_path=f"{volume.volume_path}/snapshots/{snapshot_name}",
            size_gb=volume.used_gb,
            incremental=incremental,
            parent_snapshot_id=parent_snapshot_id,
            retention_days=retention_days,
            expires_at=datetime.utcnow() + timedelta(days=retention_days),
            tags=tags or {},
            state='creating'
        )
        
        self.session.add(snapshot)
        self.session.flush()
        return snapshot
    
    def setup_replication(
        self,
        source_volume_id: UUID,
        target_volume_id: UUID,
        replication_type: str = 'async',
        schedule: Optional[str] = None
    ) -> StorageReplication:
        """Setup replication between volumes"""
        replication = StorageReplication(
            source_volume_id=source_volume_id,
            target_volume_id=target_volume_id,
            replication_type=replication_type,
            schedule=schedule,
            state='initializing'
        )
        
        self.session.add(replication)
        self.session.flush()
        return replication
    
    def get_storage_usage_by_tier(self, tenant_id: UUID) -> Dict[str, Dict[str, float]]:
        """Get storage usage grouped by tier for a tenant"""
        results = self.session.query(
            StorageVolume.storage_tier,
            func.count(StorageVolume.id).label('volume_count'),
            func.sum(StorageVolume.size_gb).label('total_size_gb'),
            func.sum(StorageVolume.used_gb).label('total_used_gb')
        ).join(
            Agent, StorageVolume.agent_id == Agent.id
        ).filter(
            and_(
                Agent.tenant_id == tenant_id,
                StorageVolume.deleted_at.is_(None)
            )
        ).group_by(StorageVolume.storage_tier).all()
        
        usage = {}
        for result in results:
            usage[result.storage_tier] = {
                'volume_count': result.volume_count,
                'total_size_gb': float(result.total_size_gb or 0),
                'total_used_gb': float(result.total_used_gb or 0),
                'utilization_percent': (
                    (float(result.total_used_gb or 0) / float(result.total_size_gb or 1)) * 100
                    if result.total_size_gb else 0
                )
            }
        
        return usage


class MetricsRepository(BaseRepository[PerformanceMetric]):
    """Repository for performance metrics"""
    
    def __init__(self, session: Session):
        super().__init__(PerformanceMetric, session)
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        metric_type: str = 'gauge',
        agent_id: Optional[UUID] = None,
        dimensions: Optional[Dict[str, Any]] = None,
        unit: Optional[str] = None
    ) -> PerformanceMetric:
        """Record a performance metric"""
        metric = self.create(
            agent_id=agent_id,
            metric_name=metric_name,
            metric_type=metric_type,
            value=value,
            unit=unit,
            dimensions=dimensions or {},
            timestamp=datetime.utcnow()
        )
        return metric
    
    def get_metrics_timeseries(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        agent_id: Optional[UUID] = None,
        aggregation: str = 'avg',
        interval_seconds: int = 300  # 5 minutes
    ) -> List[Dict[str, Any]]:
        """Get metrics time series with aggregation"""
        # Build base query
        base_filters = [
            PerformanceMetric.metric_name == metric_name,
            PerformanceMetric.timestamp >= start_time,
            PerformanceMetric.timestamp <= end_time
        ]
        
        if agent_id:
            base_filters.append(PerformanceMetric.agent_id == agent_id)
        
        # Aggregation function
        agg_func = {
            'avg': func.avg,
            'sum': func.sum,
            'max': func.max,
            'min': func.min,
            'count': func.count
        }.get(aggregation, func.avg)
        
        # Time bucket query (PostgreSQL specific)
        time_bucket = func.date_trunc(
            'minute',
            func.date_trunc('epoch', PerformanceMetric.timestamp) +
            func.cast(
                func.floor(
                    func.extract('epoch', PerformanceMetric.timestamp) / interval_seconds
                ) * interval_seconds,
                text('interval')
            )
        )
        
        results = self.session.query(
            time_bucket.label('timestamp'),
            agg_func(PerformanceMetric.value).label('value'),
            func.count(PerformanceMetric.id).label('sample_count')
        ).filter(
            and_(*base_filters)
        ).group_by(
            time_bucket
        ).order_by(
            time_bucket
        ).all()
        
        return [
            {
                'timestamp': result.timestamp.isoformat(),
                'value': float(result.value),
                'sample_count': result.sample_count
            }
            for result in results
        ]
    
    def get_latest_metrics(
        self,
        agent_id: UUID,
        metric_names: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[PerformanceMetric]:
        """Get latest metrics for an agent"""
        query = self.session.query(PerformanceMetric).filter(
            PerformanceMetric.agent_id == agent_id
        )
        
        if metric_names:
            query = query.filter(PerformanceMetric.metric_name.in_(metric_names))
        
        return query.order_by(
            PerformanceMetric.timestamp.desc()
        ).limit(limit).all()
    
    def calculate_percentiles(
        self,
        metric_name: str,
        start_time: datetime,
        end_time: datetime,
        percentiles: List[float] = [0.5, 0.95, 0.99],
        agent_id: Optional[UUID] = None
    ) -> Dict[str, float]:
        """Calculate percentiles for a metric"""
        base_filters = [
            PerformanceMetric.metric_name == metric_name,
            PerformanceMetric.timestamp >= start_time,
            PerformanceMetric.timestamp <= end_time
        ]
        
        if agent_id:
            base_filters.append(PerformanceMetric.agent_id == agent_id)
        
        # Get all values
        values = self.session.query(PerformanceMetric.value).filter(
            and_(*base_filters)
        ).order_by(PerformanceMetric.value).all()
        
        if not values:
            return {}
        
        values_list = [float(v[0]) for v in values]
        n = len(values_list)
        
        result = {}
        for p in percentiles:
            index = int(p * (n - 1))
            result[f'p{int(p * 100)}'] = values_list[index]
        
        return result
    
    def cleanup_old_metrics(self, retention_days: int = 30) -> int:
        """Clean up old metrics beyond retention period"""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        deleted = self.session.query(PerformanceMetric).filter(
            PerformanceMetric.timestamp < cutoff_date
        ).delete()
        
        self.session.flush()
        logger.info(f"Cleaned up {deleted} old metrics")
        return deleted


class SecurityRepository(BaseRepository[SecurityEvent]):
    """Repository for security operations"""
    
    def __init__(self, session: Session):
        super().__init__(SecurityEvent, session)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        title: str,
        description: str,
        source_ip: Optional[str] = None,
        source_user_id: Optional[UUID] = None,
        source_agent_id: Optional[UUID] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> SecurityEvent:
        """Log a security event"""
        event = self.create(
            event_type=event_type,
            severity=severity,
            title=title,
            description=description,
            source_ip=source_ip,
            source_user_id=source_user_id,
            source_agent_id=source_agent_id,
            details=details or {},
            occurred_at=datetime.utcnow()
        )
        
        # Check if we need to trigger alerts
        if severity in ['critical', 'high']:
            # TODO: Integrate with alerting system
            logger.warning(f"High severity security event: {title}")
        
        return event
    
    def get_recent_events(
        self,
        hours: int = 24,
        severity_filter: Optional[List[str]] = None,
        event_type_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """Get recent security events"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = self.session.query(SecurityEvent).filter(
            SecurityEvent.occurred_at >= cutoff_time
        )
        
        if severity_filter:
            query = query.filter(SecurityEvent.severity.in_(severity_filter))
        
        if event_type_filter:
            query = query.filter(SecurityEvent.event_type.in_(event_type_filter))
        
        return query.order_by(
            SecurityEvent.occurred_at.desc()
        ).limit(limit).all()
    
    def get_events_by_source(
        self,
        source_ip: Optional[str] = None,
        source_user_id: Optional[UUID] = None,
        source_agent_id: Optional[UUID] = None,
        limit: int = 50
    ) -> List[SecurityEvent]:
        """Get security events by source"""
        filters = []
        
        if source_ip:
            filters.append(SecurityEvent.source_ip == source_ip)
        if source_user_id:
            filters.append(SecurityEvent.source_user_id == source_user_id)
        if source_agent_id:
            filters.append(SecurityEvent.source_agent_id == source_agent_id)
        
        if not filters:
            return []
        
        return self.session.query(SecurityEvent).filter(
            or_(*filters)
        ).order_by(
            SecurityEvent.occurred_at.desc()
        ).limit(limit).all()
    
    def update_investigation_status(
        self,
        event_id: UUID,
        status: str,
        assigned_to: Optional[UUID] = None,
        notes: Optional[str] = None
    ) -> Optional[SecurityEvent]:
        """Update security event investigation status"""
        event = self.get(event_id)
        if not event:
            return None
        
        event.investigation_status = status
        if assigned_to:
            event.assigned_to = assigned_to
        if notes:
            event.resolution_notes = notes
        if status == 'resolved':
            event.resolved_at = datetime.utcnow()
        
        self.session.flush()
        return event
    
    def get_security_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Get security events summary"""
        results = self.session.query(
            SecurityEvent.severity,
            SecurityEvent.event_type,
            func.count(SecurityEvent.id).label('count')
        ).filter(
            and_(
                SecurityEvent.occurred_at >= start_time,
                SecurityEvent.occurred_at <= end_time
            )
        ).group_by(
            SecurityEvent.severity,
            SecurityEvent.event_type
        ).all()
        
        summary = {
            'total_events': sum(r.count for r in results),
            'by_severity': {},
            'by_type': {},
            'critical_events': []
        }
        
        for result in results:
            # By severity
            if result.severity not in summary['by_severity']:
                summary['by_severity'][result.severity] = 0
            summary['by_severity'][result.severity] += result.count
            
            # By type
            if result.event_type not in summary['by_type']:
                summary['by_type'][result.event_type] = 0
            summary['by_type'][result.event_type] += result.count
        
        # Get critical events
        critical_events = self.session.query(SecurityEvent).filter(
            and_(
                SecurityEvent.severity == 'critical',
                SecurityEvent.occurred_at >= start_time,
                SecurityEvent.occurred_at <= end_time
            )
        ).limit(10).all()
        
        summary['critical_events'] = [
            {
                'id': str(event.id),
                'title': event.title,
                'occurred_at': event.occurred_at.isoformat()
            }
            for event in critical_events
        ]
        
        return summary


class UserRepository(BaseRepository[User]):
    """Repository for user operations"""
    
    def __init__(self, session: Session):
        super().__init__(User, session)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.session.query(User).filter(
            User.email == email
        ).first()
    
    def get_by_azure_ad_id(self, azure_ad_id: str) -> Optional[User]:
        """Get user by Azure AD ID"""
        return self.session.query(User).filter(
            User.azure_ad_id == azure_ad_id
        ).first()
    
    def create_user(
        self,
        email: str,
        full_name: str,
        azure_ad_id: Optional[str] = None,
        **kwargs
    ) -> User:
        """Create a new user"""
        user = self.create(
            email=email,
            full_name=full_name,
            azure_ad_id=azure_ad_id,
            username=kwargs.get('username', email.split('@')[0]),
            **kwargs
        )
        
        # Audit log
        audit_log = AuditLog(
            user_id=user.id,
            action='user.created',
            resource_type='user',
            resource_id=str(user.id),
            resource_name=user.email,
            status='success',
            timestamp=datetime.utcnow()
        )
        self.session.add(audit_log)
        
        self.session.flush()
        return user
    
    def update_last_login(self, user_id: UUID, ip_address: str) -> Optional[User]:
        """Update user's last login information"""
        user = self.get(user_id)
        if user:
            user.last_login_at = datetime.utcnow()
            user.last_login_ip = ip_address
            user.failed_login_attempts = 0
            self.session.flush()
        return user
    
    def increment_failed_login(self, email: str) -> int:
        """Increment failed login attempts"""
        user = self.get_by_email(email)
        if user:
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.utcnow() + timedelta(hours=1)
            
            self.session.flush()
            return user.failed_login_attempts
        return 0
    
    def assign_role(self, user_id: UUID, role_id: UUID) -> bool:
        """Assign a role to a user"""
        user = self.get(user_id)
        role = self.session.get(Role, role_id)
        
        if user and role and role not in user.roles:
            user.roles.append(role)
            
            # Audit log
            audit_log = AuditLog(
                user_id=user_id,
                action='user.role.assigned',
                resource_type='user',
                resource_id=str(user_id),
                new_value={'role_id': str(role_id), 'role_name': role.name},
                status='success',
                timestamp=datetime.utcnow()
            )
            self.session.add(audit_log)
            
            self.session.flush()
            return True
        return False
    
    def get_user_permissions(self, user_id: UUID) -> List[str]:
        """Get all permissions for a user through roles"""
        user = self.session.query(User).options(
            selectinload(User.roles).selectinload(Role.permissions)
        ).filter(User.id == user_id).first()
        
        if not user:
            return []
        
        permissions = set()
        for role in user.roles:
            for permission in role.permissions:
                permissions.add(f"{permission.resource}:{permission.action}")
        
        return list(permissions)
    
    def search_users(
        self,
        query: str,
        is_active: Optional[bool] = None,
        limit: int = 50
    ) -> List[User]:
        """Search users by email, username, or full name"""
        search_query = self.session.query(User)
        
        if query:
            search_query = search_query.filter(
                or_(
                    User.email.ilike(f'%{query}%'),
                    User.username.ilike(f'%{query}%'),
                    User.full_name.ilike(f'%{query}%')
                )
            )
        
        if is_active is not None:
            search_query = search_query.filter(User.is_active == is_active)
        
        return search_query.filter(
            User.deleted_at.is_(None)
        ).limit(limit).all()


class TenantRepository(BaseRepository[Tenant]):
    """Repository for tenant operations"""
    
    def __init__(self, session: Session):
        super().__init__(Tenant, session)
    
    def create_tenant(
        self,
        name: str,
        display_name: str,
        azure_subscription_id: str,
        **kwargs
    ) -> Tenant:
        """Create a new tenant with default quotas"""
        tenant = self.create(
            name=name,
            display_name=display_name,
            azure_subscription_id=azure_subscription_id,
            **kwargs
        )
        
        # Create default resource quotas
        default_quotas = [
            ('cpu', 'cores', 100, 200),
            ('memory', 'GB', 400, 800),
            ('storage', 'GB', 10000, 20000),
            ('agents', 'count', 50, 100),
            ('api_requests', 'per_hour', 10000, 50000)
        ]
        
        for resource_type, unit, soft_limit, hard_limit in default_quotas:
            quota = ResourceQuota(
                tenant_id=tenant.id,
                resource_type=resource_type,
                resource_unit=unit,
                soft_limit=soft_limit,
                hard_limit=hard_limit
            )
            self.session.add(quota)
        
        self.session.flush()
        return tenant
    
    def update_resource_usage(
        self,
        tenant_id: UUID,
        resource_type: str,
        usage_delta: float
    ) -> Optional[ResourceQuota]:
        """Update resource usage for a tenant"""
        quota = self.session.query(ResourceQuota).filter(
            and_(
                ResourceQuota.tenant_id == tenant_id,
                ResourceQuota.resource_type == resource_type
            )
        ).first()
        
        if quota:
            quota.current_usage += usage_delta
            
            # Update peak usage
            if quota.current_usage > quota.peak_usage:
                quota.peak_usage = quota.current_usage
                quota.peak_usage_at = datetime.utcnow()
            
            # Check thresholds
            usage_percent = (quota.current_usage / quota.hard_limit) * 100
            
            if usage_percent >= quota.critical_threshold_percent:
                logger.warning(
                    f"Critical resource usage for tenant {tenant_id}: "
                    f"{resource_type} at {usage_percent:.1f}%"
                )
            elif usage_percent >= quota.warning_threshold_percent:
                logger.info(
                    f"High resource usage for tenant {tenant_id}: "
                    f"{resource_type} at {usage_percent:.1f}%"
                )
            
            self.session.flush()
            return quota
        
        return None
    
    def check_resource_limit(
        self,
        tenant_id: UUID,
        resource_type: str,
        requested_amount: float
    ) -> Tuple[bool, Optional[str]]:
        """Check if resource request is within limits"""
        quota = self.session.query(ResourceQuota).filter(
            and_(
                ResourceQuota.tenant_id == tenant_id,
                ResourceQuota.resource_type == resource_type
            )
        ).first()
        
        if not quota:
            return False, f"No quota defined for resource type: {resource_type}"
        
        if not quota.is_enforced:
            return True, None
        
        new_usage = quota.current_usage + requested_amount
        
        if new_usage > quota.hard_limit:
            return False, (
                f"Resource limit exceeded: {resource_type} "
                f"(requested: {requested_amount}, "
                f"current: {quota.current_usage}, "
                f"limit: {quota.hard_limit})"
            )
        
        return True, None
    
    def get_tenant_summary(self, tenant_id: UUID) -> Dict[str, Any]:
        """Get comprehensive tenant summary"""
        tenant = self.get(tenant_id)
        if not tenant:
            return {}
        
        # Get resource usage
        quotas = self.session.query(ResourceQuota).filter(
            ResourceQuota.tenant_id == tenant_id
        ).all()
        
        # Get agent count by state
        agent_states = self.session.query(
            Agent.state,
            func.count(Agent.id).label('count')
        ).filter(
            and_(
                Agent.tenant_id == tenant_id,
                Agent.deleted_at.is_(None)
            )
        ).group_by(Agent.state).all()
        
        # Get user count
        user_count = self.session.query(func.count(User.id)).join(
            Agent, User.id == Agent.owner_id
        ).filter(
            Agent.tenant_id == tenant_id
        ).scalar()
        
        return {
            'tenant': {
                'id': str(tenant.id),
                'name': tenant.name,
                'display_name': tenant.display_name,
                'billing_plan': tenant.billing_plan,
                'is_active': tenant.is_active,
                'created_at': tenant.created_at.isoformat()
            },
            'resources': {
                quota.resource_type: {
                    'current_usage': float(quota.current_usage),
                    'soft_limit': float(quota.soft_limit),
                    'hard_limit': float(quota.hard_limit),
                    'usage_percent': (float(quota.current_usage) / float(quota.hard_limit)) * 100,
                    'unit': quota.resource_unit
                }
                for quota in quotas
            },
            'agents': {
                state: count
                for state, count in agent_states
            },
            'users': user_count
        }