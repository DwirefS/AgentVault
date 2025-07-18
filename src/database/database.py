"""
AgentVault™ Database Manager
Production-grade database connection and session management
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import os
import logging
from typing import Generator, Optional, Dict, Any
from contextlib import contextmanager
import asyncio
from datetime import datetime

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy.engine import Engine
import asyncpg

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Production database manager with connection pooling, retries, and monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database manager with configuration"""
        self.config = config or self._load_default_config()
        
        # Database URLs
        self.sync_url = self._build_database_url(async_driver=False)
        self.async_url = self._build_database_url(async_driver=True)
        
        # Engines
        self.sync_engine: Optional[Engine] = None
        self.async_engine: Optional[AsyncEngine] = None
        
        # Session factories
        self.sync_session_factory: Optional[sessionmaker] = None
        self.async_session_factory: Optional[sessionmaker] = None
        
        # Connection pool stats
        self.pool_stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'active_connections': 0,
            'pool_size': 0
        }
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load database configuration from environment variables"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'agentvault'),
            'username': os.getenv('DB_USER', 'agentvault'),
            'password': os.getenv('DB_PASSWORD', ''),
            'ssl_mode': os.getenv('DB_SSL_MODE', 'require'),
            'pool_size': int(os.getenv('DB_POOL_SIZE', '20')),
            'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '40')),
            'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
            'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600')),
            'echo': os.getenv('DB_ECHO', 'false').lower() == 'true',
            'statement_timeout': int(os.getenv('DB_STATEMENT_TIMEOUT', '30000')),  # 30 seconds
            'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', '10')),
            'command_timeout': int(os.getenv('DB_COMMAND_TIMEOUT', '10')),
            'application_name': os.getenv('APP_NAME', 'agentvault'),
            'jitter': os.getenv('DB_JITTER', 'true').lower() == 'true'
        }
    
    def _build_database_url(self, async_driver: bool = False) -> str:
        """Build database connection URL"""
        driver = 'asyncpg' if async_driver else 'psycopg2'
        protocol = 'postgresql+asyncpg' if async_driver else 'postgresql+psycopg2'
        
        # Build base URL
        url = f"{protocol}://{self.config['username']}:{self.config['password']}" \
              f"@{self.config['host']}:{self.config['port']}/{self.config['database']}"
        
        # Add connection parameters
        params = []
        
        # SSL configuration
        if self.config['ssl_mode'] != 'disable':
            params.append(f"sslmode={self.config['ssl_mode']}")
        
        # Timeouts
        params.append(f"connect_timeout={self.config['connect_timeout']}")
        
        if not async_driver:
            # psycopg2 specific
            params.append(f"options=-c statement_timeout={self.config['statement_timeout']}")
        
        # Application name for monitoring
        params.append(f"application_name={self.config['application_name']}")
        
        # Join parameters
        if params:
            url += '?' + '&'.join(params)
        
        return url
    
    def initialize_sync_engine(self) -> Engine:
        """Initialize synchronous database engine with production settings"""
        if self.sync_engine:
            return self.sync_engine
        
        logger.info("Initializing synchronous database engine")
        
        # Engine configuration
        engine_config = {
            'echo': self.config['echo'],
            'pool_size': self.config['pool_size'],
            'max_overflow': self.config['max_overflow'],
            'pool_timeout': self.config['pool_timeout'],
            'pool_recycle': self.config['pool_recycle'],
            'pool_pre_ping': True,  # Verify connections before use
            'poolclass': QueuePool,
            'connect_args': {
                'connect_timeout': self.config['connect_timeout'],
                'application_name': self.config['application_name'],
                'options': f"-c statement_timeout={self.config['statement_timeout']}"
            }
        }
        
        # Create engine
        self.sync_engine = create_engine(self.sync_url, **engine_config)
        
        # Add event listeners
        self._setup_engine_events(self.sync_engine)
        
        # Create session factory
        self.sync_session_factory = scoped_session(
            sessionmaker(
                bind=self.sync_engine,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
        )
        
        logger.info("Synchronous database engine initialized successfully")
        return self.sync_engine
    
    async def initialize_async_engine(self) -> AsyncEngine:
        """Initialize asynchronous database engine with production settings"""
        if self.async_engine:
            return self.async_engine
        
        logger.info("Initializing asynchronous database engine")
        
        # Engine configuration
        engine_config = {
            'echo': self.config['echo'],
            'pool_size': self.config['pool_size'],
            'max_overflow': self.config['max_overflow'],
            'pool_timeout': self.config['pool_timeout'],
            'pool_recycle': self.config['pool_recycle'],
            'pool_pre_ping': True,
            'poolclass': pool.AsyncAdaptedQueuePool,
            'connect_args': {
                'server_settings': {
                    'application_name': self.config['application_name'],
                    'jit': 'off' if self.config['jitter'] else 'on'
                },
                'timeout': self.config['connect_timeout'],
                'command_timeout': self.config['command_timeout']
            }
        }
        
        # Create engine
        self.async_engine = create_async_engine(self.async_url, **engine_config)
        
        # Create session factory
        self.async_session_factory = sessionmaker(
            self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False
        )
        
        logger.info("Asynchronous database engine initialized successfully")
        return self.async_engine
    
    def _setup_engine_events(self, engine: Engine):
        """Setup SQLAlchemy engine event listeners for monitoring and optimization"""
        
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            """Configure connection settings"""
            self.pool_stats['connections_created'] += 1
            self.pool_stats['active_connections'] += 1
            
            # Set PostgreSQL specific optimizations
            if hasattr(dbapi_conn, 'execute'):
                cursor = dbapi_conn.cursor()
                # Set work_mem for better sorting/hashing performance
                cursor.execute("SET work_mem = '256MB'")
                # Enable parallel queries
                cursor.execute("SET max_parallel_workers_per_gather = 4")
                # Set lock timeout to prevent long waits
                cursor.execute("SET lock_timeout = '10s'")
                cursor.close()
        
        @event.listens_for(engine, "close")
        def close_connection(dbapi_conn, connection_record):
            """Track connection closure"""
            self.pool_stats['connections_closed'] += 1
            self.pool_stats['active_connections'] -= 1
        
        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries and add query tags"""
            conn.info.setdefault('query_start_time', []).append(datetime.utcnow())
            if self.config['echo']:
                logger.debug(f"Executing query: {statement[:100]}...")
        
        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Track query execution time"""
            total = datetime.utcnow() - conn.info['query_start_time'].pop(-1)
            if total.total_seconds() > 1:  # Log slow queries
                logger.warning(f"Slow query detected ({total.total_seconds():.2f}s): {statement[:100]}...")
    
    def create_tables(self):
        """Create all database tables"""
        logger.info("Creating database tables")
        if not self.sync_engine:
            self.initialize_sync_engine()
        
        Base.metadata.create_all(self.sync_engine)
        logger.info("Database tables created successfully")
    
    async def create_tables_async(self):
        """Create all database tables asynchronously"""
        logger.info("Creating database tables asynchronously")
        if not self.async_engine:
            await self.initialize_async_engine()
        
        async with self.async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        logger.warning("Dropping all database tables")
        if not self.sync_engine:
            self.initialize_sync_engine()
        
        Base.metadata.drop_all(self.sync_engine)
        logger.info("Database tables dropped")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup"""
        if not self.sync_session_factory:
            self.initialize_sync_engine()
        
        session = self.sync_session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            session.close()
    
    async def get_async_session(self) -> AsyncSession:
        """Get an async database session"""
        if not self.async_session_factory:
            await self.initialize_async_engine()
        
        async with self.async_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Async database session error: {str(e)}")
                raise
            finally:
                await session.close()
    
    def check_connection(self) -> bool:
        """Check database connection health"""
        try:
            if not self.sync_engine:
                self.initialize_sync_engine()
            
            with self.sync_engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database connection check failed: {str(e)}")
            return False
    
    async def check_connection_async(self) -> bool:
        """Check database connection health asynchronously"""
        try:
            if not self.async_engine:
                await self.initialize_async_engine()
            
            async with self.async_engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Async database connection check failed: {str(e)}")
            return False
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status"""
        status = dict(self.pool_stats)
        
        if self.sync_engine:
            pool = self.sync_engine.pool
            status.update({
                'pool_size': pool.size(),
                'checked_in_connections': pool.checkedin(),
                'overflow': pool.overflow(),
                'total': pool.total_connections()
            })
        
        return status
    
    def optimize_tables(self):
        """Run VACUUM ANALYZE on all tables for PostgreSQL optimization"""
        logger.info("Running database optimization (VACUUM ANALYZE)")
        
        if not self.sync_engine:
            self.initialize_sync_engine()
        
        with self.sync_engine.connect() as conn:
            # Get all table names
            result = conn.execute(text(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            ))
            tables = [row[0] for row in result]
            
            # Run VACUUM ANALYZE on each table
            for table in tables:
                try:
                    conn.execute(text(f"VACUUM ANALYZE {table}"))
                    logger.info(f"Optimized table: {table}")
                except Exception as e:
                    logger.error(f"Failed to optimize table {table}: {str(e)}")
        
        logger.info("Database optimization completed")
    
    def close(self):
        """Close all database connections"""
        logger.info("Closing database connections")
        
        if self.sync_engine:
            self.sync_engine.dispose()
            self.sync_engine = None
        
        if self.sync_session_factory:
            self.sync_session_factory.remove()
            self.sync_session_factory = None
        
        logger.info("Database connections closed")
    
    async def close_async(self):
        """Close all async database connections"""
        logger.info("Closing async database connections")
        
        if self.async_engine:
            await self.async_engine.dispose()
            self.async_engine = None
        
        self.async_session_factory = None
        logger.info("Async database connections closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize_sync_engine()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize_async_engine()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_async()


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create global database manager instance"""
    global _db_manager
    if not _db_manager:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db_manager = get_db_manager()
    with db_manager.get_session() as session:
        yield session


async def get_async_db() -> AsyncSession:
    """
    FastAPI dependency for async database sessions
    
    Usage:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_async_db)):
            result = await db.execute(select(Item))
            return result.scalars().all()
    """
    db_manager = get_db_manager()
    async with db_manager.get_async_session() as session:
        yield session


# Health check functions
async def check_database_health() -> Dict[str, Any]:
    """Comprehensive database health check"""
    db_manager = get_db_manager()
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'connection': False,
        'pool_status': {},
        'latency_ms': 0,
        'errors': []
    }
    
    try:
        # Check connection
        start_time = datetime.utcnow()
        health_status['connection'] = await db_manager.check_connection_async()
        health_status['latency_ms'] = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Get pool status
        health_status['pool_status'] = db_manager.get_pool_status()
        
        # Overall status
        if not health_status['connection']:
            health_status['status'] = 'unhealthy'
            health_status['errors'].append('Database connection failed')
        elif health_status['latency_ms'] > 1000:
            health_status['status'] = 'degraded'
            health_status['errors'].append('High database latency')
        
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['errors'].append(str(e))
        logger.error(f"Database health check failed: {str(e)}")
    
    return health_status