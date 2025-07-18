"""
AgentVault™ Database Migration Manager
Production-grade schema migration with versioning and rollback support
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import os
import logging
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
import hashlib
import json

from sqlalchemy import text, Table, Column, String, DateTime, Text, Boolean, Integer
from sqlalchemy.orm import Session
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from alembic.migration import MigrationContext
from alembic.operations import Operations

from .database import DatabaseManager
from .models import Base

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Production database migration manager with versioning and rollback
    """
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.alembic_cfg = self._create_alembic_config()
        
        # Migration tracking table
        self.migration_history_table = Table(
            'migration_history',
            Base.metadata,
            Column('version', String(255), primary_key=True),
            Column('description', Text),
            Column('script', Text),
            Column('checksum', String(64)),
            Column('applied_at', DateTime, default=datetime.utcnow),
            Column('applied_by', String(255)),
            Column('execution_time_ms', Integer),
            Column('success', Boolean, default=True),
            Column('error_message', Text),
            Column('rollback_script', Text),
            Column('metadata', Text)  # JSON metadata
        )
    
    def _create_alembic_config(self) -> Config:
        """Create Alembic configuration"""
        # Get the migrations directory path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        migrations_dir = os.path.join(base_dir, '..', '..', 'migrations')
        
        # Create migrations directory if it doesn't exist
        os.makedirs(migrations_dir, exist_ok=True)
        
        # Create alembic.ini if it doesn't exist
        alembic_ini_path = os.path.join(migrations_dir, 'alembic.ini')
        if not os.path.exists(alembic_ini_path):
            self._create_alembic_ini(alembic_ini_path, migrations_dir)
        
        # Create Alembic config
        cfg = Config(alembic_ini_path)
        cfg.set_main_option('script_location', migrations_dir)
        cfg.set_main_option('sqlalchemy.url', self.db_manager.sync_url)
        
        return cfg
    
    def _create_alembic_ini(self, ini_path: str, migrations_dir: str):
        """Create default alembic.ini file"""
        content = f"""
# AgentVault Alembic Configuration

[alembic]
script_location = {migrations_dir}
prepend_sys_path = .
version_path_separator = os
sqlalchemy.url = driver://user:pass@localhost/dbname

[post_write_hooks]
hooks = black
black.type = console_scripts
black.entrypoint = black
black.options = -l 79 REVISION_SCRIPT_FILENAME

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""
        
        with open(ini_path, 'w') as f:
            f.write(content.strip())
        
        # Create versions directory
        versions_dir = os.path.join(migrations_dir, 'versions')
        os.makedirs(versions_dir, exist_ok=True)
        
        # Create env.py
        env_py_path = os.path.join(migrations_dir, 'env.py')
        if not os.path.exists(env_py_path):
            self._create_env_py(env_py_path)
        
        # Create script.py.mako
        mako_path = os.path.join(migrations_dir, 'script.py.mako')
        if not os.path.exists(mako_path):
            self._create_script_mako(mako_path)
    
    def _create_env_py(self, env_path: str):
        """Create Alembic env.py file"""
        content = '''
"""AgentVault Alembic Environment Configuration"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your models
from src.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
'''
        
        with open(env_path, 'w') as f:
            f.write(content.strip())
    
    def _create_script_mako(self, mako_path: str):
        """Create Alembic script template"""
        content = '''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
'''
        
        with open(mako_path, 'w') as f:
            f.write(content.strip())
    
    def initialize(self):
        """Initialize migration system"""
        logger.info("Initializing database migration system")
        
        # Ensure database exists
        if not self.db_manager.sync_engine:
            self.db_manager.initialize_sync_engine()
        
        # Create migration history table
        self.migration_history_table.create(self.db_manager.sync_engine, checkfirst=True)
        
        # Initialize Alembic
        try:
            command.init(self.alembic_cfg, os.path.dirname(self.alembic_cfg.get_main_option('script_location')))
            logger.info("Alembic initialized successfully")
        except Exception as e:
            logger.warning(f"Alembic already initialized or error: {str(e)}")
        
        # Stamp current database state
        self.stamp_current()
    
    def stamp_current(self):
        """Stamp current database state as head"""
        logger.info("Stamping current database state")
        command.stamp(self.alembic_cfg, "head")
    
    def create_migration(self, message: str, auto_generate: bool = True) -> str:
        """Create a new migration"""
        logger.info(f"Creating migration: {message}")
        
        if auto_generate:
            # Auto-generate migration based on model changes
            revision = command.revision(
                self.alembic_cfg,
                message=message,
                autogenerate=True
            )
        else:
            # Create empty migration
            revision = command.revision(
                self.alembic_cfg,
                message=message
            )
        
        logger.info(f"Created migration: {revision}")
        return revision
    
    def upgrade(self, revision: str = "head") -> bool:
        """Upgrade database to a revision"""
        logger.info(f"Upgrading database to revision: {revision}")
        
        start_time = datetime.utcnow()
        success = True
        error_message = None
        
        try:
            command.upgrade(self.alembic_cfg, revision)
            logger.info("Database upgrade completed successfully")
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Database upgrade failed: {error_message}")
            raise
        finally:
            # Record migration history
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._record_migration(
                version=revision,
                description=f"Upgrade to {revision}",
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message
            )
        
        return success
    
    def downgrade(self, revision: str = "-1") -> bool:
        """Downgrade database to a revision"""
        logger.info(f"Downgrading database to revision: {revision}")
        
        start_time = datetime.utcnow()
        success = True
        error_message = None
        
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info("Database downgrade completed successfully")
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Database downgrade failed: {error_message}")
            raise
        finally:
            # Record migration history
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._record_migration(
                version=revision,
                description=f"Downgrade to {revision}",
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message
            )
        
        return success
    
    def get_current_revision(self) -> Optional[str]:
        """Get current database revision"""
        with self.db_manager.sync_engine.connect() as connection:
            context = MigrationContext.configure(connection)
            return context.get_current_revision()
    
    def get_pending_migrations(self) -> List[str]:
        """Get list of pending migrations"""
        script_dir = ScriptDirectory.from_config(self.alembic_cfg)
        current_rev = self.get_current_revision()
        
        pending = []
        for revision in script_dir.walk_revisions():
            if revision.revision != current_rev:
                pending.append(revision.revision)
            else:
                break
        
        return pending
    
    def get_migration_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get migration history"""
        with self.db_manager.get_session() as session:
            result = session.execute(
                text(f"""
                    SELECT version, description, applied_at, applied_by, 
                           execution_time_ms, success, error_message
                    FROM migration_history
                    ORDER BY applied_at DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            
            return [
                {
                    'version': row.version,
                    'description': row.description,
                    'applied_at': row.applied_at.isoformat(),
                    'applied_by': row.applied_by,
                    'execution_time_ms': row.execution_time_ms,
                    'success': row.success,
                    'error_message': row.error_message
                }
                for row in result
            ]
    
    def _record_migration(
        self,
        version: str,
        description: str,
        execution_time_ms: int,
        success: bool,
        error_message: Optional[str] = None,
        script: Optional[str] = None,
        rollback_script: Optional[str] = None
    ):
        """Record migration in history"""
        with self.db_manager.get_session() as session:
            session.execute(
                text("""
                    INSERT INTO migration_history 
                    (version, description, script, checksum, applied_at, 
                     applied_by, execution_time_ms, success, error_message, 
                     rollback_script, metadata)
                    VALUES 
                    (:version, :description, :script, :checksum, :applied_at,
                     :applied_by, :execution_time_ms, :success, :error_message,
                     :rollback_script, :metadata)
                """),
                {
                    'version': version,
                    'description': description,
                    'script': script,
                    'checksum': hashlib.sha256(
                        (script or '').encode()
                    ).hexdigest() if script else None,
                    'applied_at': datetime.utcnow(),
                    'applied_by': os.getenv('USER', 'system'),
                    'execution_time_ms': execution_time_ms,
                    'success': success,
                    'error_message': error_message,
                    'rollback_script': rollback_script,
                    'metadata': json.dumps({
                        'hostname': os.uname().nodename,
                        'python_version': os.sys.version,
                        'alembic_version': '1.13.0'
                    })
                }
            )
    
    def validate_schema(self) -> Dict[str, Any]:
        """Validate current schema against models"""
        logger.info("Validating database schema")
        
        validation_result = {
            'valid': True,
            'missing_tables': [],
            'missing_columns': {},
            'missing_indexes': {},
            'extra_tables': [],
            'warnings': []
        }
        
        with self.db_manager.sync_engine.connect() as connection:
            inspector = connection.dialect.inspector(connection)
            
            # Get existing tables
            existing_tables = set(inspector.get_table_names())
            model_tables = set(Base.metadata.tables.keys())
            
            # Check for missing tables
            validation_result['missing_tables'] = list(model_tables - existing_tables)
            validation_result['extra_tables'] = list(existing_tables - model_tables - {'migration_history', 'alembic_version'})
            
            # Check columns and indexes for each table
            for table_name in model_tables.intersection(existing_tables):
                table = Base.metadata.tables[table_name]
                
                # Check columns
                existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
                model_columns = {col.name for col in table.columns}
                
                missing_columns = model_columns - existing_columns
                if missing_columns:
                    validation_result['missing_columns'][table_name] = list(missing_columns)
                
                # Check indexes
                existing_indexes = {idx['name'] for idx in inspector.get_indexes(table_name)}
                model_indexes = {idx.name for idx in table.indexes if idx.name}
                
                missing_indexes = model_indexes - existing_indexes
                if missing_indexes:
                    validation_result['missing_indexes'][table_name] = list(missing_indexes)
        
        # Update validity
        if (validation_result['missing_tables'] or 
            validation_result['missing_columns'] or 
            validation_result['missing_indexes']):
            validation_result['valid'] = False
        
        return validation_result
    
    def backup_schema(self, backup_path: Optional[str] = None) -> str:
        """Backup current database schema"""
        if not backup_path:
            backup_path = f"/tmp/agentvault_schema_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.sql"
        
        logger.info(f"Backing up database schema to: {backup_path}")
        
        # Use pg_dump for PostgreSQL
        import subprocess
        
        db_config = self.db_manager.config
        env = os.environ.copy()
        env['PGPASSWORD'] = db_config['password']
        
        cmd = [
            'pg_dump',
            '-h', db_config['host'],
            '-p', str(db_config['port']),
            '-U', db_config['username'],
            '-d', db_config['database'],
            '--schema-only',
            '-f', backup_path
        ]
        
        try:
            subprocess.run(cmd, env=env, check=True, capture_output=True)
            logger.info(f"Schema backup completed: {backup_path}")
            return backup_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Schema backup failed: {e.stderr.decode()}")
            raise
    
    def run_custom_migration(
        self,
        name: str,
        upgrade_func: Callable[[Operations], None],
        downgrade_func: Callable[[Operations], None],
        description: str = ""
    ) -> bool:
        """Run a custom migration function"""
        logger.info(f"Running custom migration: {name}")
        
        start_time = datetime.utcnow()
        success = True
        error_message = None
        
        try:
            with self.db_manager.sync_engine.connect() as connection:
                with connection.begin():
                    op = Operations(MigrationContext.configure(connection))
                    upgrade_func(op)
            
            logger.info(f"Custom migration '{name}' completed successfully")
        except Exception as e:
            success = False
            error_message = str(e)
            logger.error(f"Custom migration '{name}' failed: {error_message}")
            raise
        finally:
            # Record migration
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._record_migration(
                version=f"custom_{name}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                description=description or f"Custom migration: {name}",
                execution_time_ms=execution_time_ms,
                success=success,
                error_message=error_message
            )
        
        return success


# Production migrations
class ProductionMigrations:
    """Collection of production-ready migrations"""
    
    @staticmethod
    def add_performance_indexes(op: Operations):
        """Add performance-critical indexes"""
        # Agent performance indexes
        op.create_index(
            'idx_agent_tenant_state_active',
            'agents',
            ['tenant_id', 'state', 'last_active']
        )
        
        # Storage performance indexes
        op.create_index(
            'idx_volume_agent_state',
            'storage_volumes',
            ['agent_id', 'state', 'health_status']
        )
        
        # Metrics time-series index
        op.create_index(
            'idx_metrics_time_partition',
            'performance_metrics',
            ['metric_name', 'timestamp'],
            postgresql_using='brin'  # BRIN index for time-series data
        )
        
        # Security event composite index
        op.create_index(
            'idx_security_severity_time',
            'security_events',
            ['severity', 'occurred_at', 'investigation_status']
        )
    
    @staticmethod
    def add_json_indexes(op: Operations):
        """Add GIN indexes for JSONB columns"""
        # Agent configuration index
        op.execute(
            "CREATE INDEX idx_agent_config_gin ON agents USING gin(configuration)"
        )
        
        # Agent tags index
        op.execute(
            "CREATE INDEX idx_agent_tags_gin ON agents USING gin(tags)"
        )
        
        # Metric dimensions index
        op.execute(
            "CREATE INDEX idx_metric_dimensions_gin ON performance_metrics USING gin(dimensions)"
        )
    
    @staticmethod
    def add_partitioning(op: Operations):
        """Add table partitioning for large tables"""
        # Partition performance_metrics by month
        op.execute("""
            -- Create partitioned table
            CREATE TABLE performance_metrics_partitioned (
                LIKE performance_metrics INCLUDING ALL
            ) PARTITION BY RANGE (timestamp);
            
            -- Create partitions for next 12 months
            DO $$
            DECLARE
                start_date date := date_trunc('month', CURRENT_DATE);
                end_date date;
                partition_name text;
            BEGIN
                FOR i IN 0..11 LOOP
                    end_date := start_date + interval '1 month';
                    partition_name := 'performance_metrics_' || to_char(start_date, 'YYYY_MM');
                    
                    EXECUTE format(
                        'CREATE TABLE %I PARTITION OF performance_metrics_partitioned
                         FOR VALUES FROM (%L) TO (%L)',
                        partition_name, start_date, end_date
                    );
                    
                    start_date := end_date;
                END LOOP;
            END $$;
            
            -- Migrate data
            INSERT INTO performance_metrics_partitioned SELECT * FROM performance_metrics;
            
            -- Rename tables
            ALTER TABLE performance_metrics RENAME TO performance_metrics_old;
            ALTER TABLE performance_metrics_partitioned RENAME TO performance_metrics;
        """)
    
    @staticmethod
    def optimize_for_read_replicas(op: Operations):
        """Optimize schema for read replicas"""
        # Add covering indexes for common queries
        op.create_index(
            'idx_agent_list_covering',
            'agents',
            ['tenant_id', 'state', 'name', 'agent_type', 'created_at'],
            postgresql_include=['display_name', 'last_active']
        )
        
        # Add materialized view for agent statistics
        op.execute("""
            CREATE MATERIALIZED VIEW agent_statistics AS
            SELECT 
                a.tenant_id,
                a.agent_type,
                a.state,
                COUNT(*) as agent_count,
                AVG(a.cpu_cores) as avg_cpu,
                AVG(a.memory_gb) as avg_memory,
                SUM(a.total_requests) as total_requests,
                AVG(a.average_latency_ms) as avg_latency
            FROM agents a
            WHERE a.deleted_at IS NULL
            GROUP BY a.tenant_id, a.agent_type, a.state;
            
            CREATE INDEX idx_agent_stats_tenant ON agent_statistics(tenant_id);
        """)


# CLI commands for migration management
def create_migration_cli():
    """Create CLI commands for migration management"""
    import click
    
    @click.group()
    def migrate():
        """Database migration commands"""
        pass
    
    @migrate.command()
    @click.option('--message', '-m', required=True, help='Migration message')
    @click.option('--auto', is_flag=True, help='Auto-generate migration')
    def create(message: str, auto: bool):
        """Create a new migration"""
        db_manager = DatabaseManager()
        migration_manager = MigrationManager(db_manager)
        revision = migration_manager.create_migration(message, auto_generate=auto)
        click.echo(f"Created migration: {revision}")
    
    @migrate.command()
    @click.option('--revision', '-r', default='head', help='Target revision')
    def upgrade(revision: str):
        """Upgrade database schema"""
        db_manager = DatabaseManager()
        migration_manager = MigrationManager(db_manager)
        success = migration_manager.upgrade(revision)
        if success:
            click.echo("Database upgraded successfully")
        else:
            click.echo("Database upgrade failed", err=True)
    
    @migrate.command()
    @click.option('--revision', '-r', default='-1', help='Target revision')
    def downgrade(revision: str):
        """Downgrade database schema"""
        db_manager = DatabaseManager()
        migration_manager = MigrationManager(db_manager)
        success = migration_manager.downgrade(revision)
        if success:
            click.echo("Database downgraded successfully")
        else:
            click.echo("Database downgrade failed", err=True)
    
    @migrate.command()
    def validate():
        """Validate database schema"""
        db_manager = DatabaseManager()
        migration_manager = MigrationManager(db_manager)
        result = migration_manager.validate_schema()
        
        if result['valid']:
            click.echo("Database schema is valid")
        else:
            click.echo("Database schema validation failed:", err=True)
            click.echo(json.dumps(result, indent=2), err=True)
    
    @migrate.command()
    @click.option('--limit', '-l', default=50, help='Number of entries to show')
    def history(limit: int):
        """Show migration history"""
        db_manager = DatabaseManager()
        migration_manager = MigrationManager(db_manager)
        history = migration_manager.get_migration_history(limit)
        
        for entry in history:
            status = "✓" if entry['success'] else "✗"
            click.echo(
                f"{status} {entry['version']} - {entry['description']} "
                f"({entry['applied_at']}, {entry['execution_time_ms']}ms)"
            )
    
    return migrate


if __name__ == '__main__':
    # Run CLI if executed directly
    cli = create_migration_cli()
    cli()