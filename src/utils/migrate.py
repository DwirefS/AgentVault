#!/usr/bin/env python3
"""
AgentVault™ Migration Manager
Handles system migrations and upgrades
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    """Represents a system migration"""
    id: str
    name: str
    description: str
    version: str
    checksum: str
    applied_at: Optional[datetime] = None
    rollback_sql: Optional[str] = None


class MigrationManager:
    """Manages AgentVault™ system migrations"""
    
    def __init__(self, migrations_dir: str = "/app/migrations"):
        self.migrations_dir = Path(migrations_dir)
        self.state_file = Path("/app/data/.migration_state.json")
        self.migrations: List[Migration] = []
        self._load_state()
    
    def _load_state(self) -> None:
        """Load migration state from file"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.applied_migrations = set(state.get('applied', []))
        else:
            self.applied_migrations = set()
    
    def _save_state(self) -> None:
        """Save migration state to file"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'applied': list(self.applied_migrations),
            'last_updated': datetime.utcnow().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def discover_migrations(self) -> List[Migration]:
        """Discover available migrations"""
        migrations = []
        
        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return migrations
        
        for migration_file in sorted(self.migrations_dir.glob("*.py")):
            if migration_file.name.startswith("__"):
                continue
                
            # Parse migration file
            content = migration_file.read_text()
            checksum = hashlib.sha256(content.encode()).hexdigest()
            
            # Extract metadata from file
            migration_id = migration_file.stem
            migration = Migration(
                id=migration_id,
                name=migration_file.name,
                description=self._extract_description(content),
                version=self._extract_version(migration_id),
                checksum=checksum
            )
            
            migrations.append(migration)
        
        return migrations
    
    def _extract_description(self, content: str) -> str:
        """Extract description from migration file"""
        for line in content.split('\n'):
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                return line.strip().strip('"""').strip("'''")
        return "No description"
    
    def _extract_version(self, migration_id: str) -> str:
        """Extract version from migration ID"""
        # Expected format: 001_initial_setup
        parts = migration_id.split('_', 1)
        if parts and parts[0].isdigit():
            return parts[0]
        return "unknown"
    
    async def run_migrations(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """Run pending migrations"""
        migrations = self.discover_migrations()
        pending = [m for m in migrations if m.id not in self.applied_migrations]
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        results = {
            'total': len(migrations),
            'applied': len(self.applied_migrations),
            'pending': len(pending),
            'executed': [],
            'failed': []
        }
        
        for migration in pending:
            try:
                logger.info(f"Running migration: {migration.id}")
                await self._execute_migration(migration)
                
                self.applied_migrations.add(migration.id)
                results['executed'].append(migration.id)
                
                # Save state after each successful migration
                self._save_state()
                
            except Exception as e:
                logger.error(f"Migration {migration.id} failed: {str(e)}")
                results['failed'].append({
                    'id': migration.id,
                    'error': str(e)
                })
                
                # Stop on first failure
                break
        
        return results
    
    async def _execute_migration(self, migration: Migration) -> None:
        """Execute a single migration"""
        migration_file = self.migrations_dir / f"{migration.id}.py"
        
        # Import and execute migration
        import importlib.util
        spec = importlib.util.spec_from_file_location(migration.id, migration_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run migration
        if hasattr(module, 'up'):
            if asyncio.iscoroutinefunction(module.up):
                await module.up()
            else:
                module.up()
        else:
            raise ValueError(f"Migration {migration.id} missing 'up' function")
    
    async def rollback(self, target_version: str) -> Dict[str, Any]:
        """Rollback to a specific version"""
        migrations = self.discover_migrations()
        to_rollback = [
            m for m in migrations 
            if m.id in self.applied_migrations and m.version > target_version
        ]
        
        results = {
            'target_version': target_version,
            'rolled_back': [],
            'failed': []
        }
        
        for migration in reversed(to_rollback):
            try:
                logger.info(f"Rolling back migration: {migration.id}")
                await self._rollback_migration(migration)
                
                self.applied_migrations.remove(migration.id)
                results['rolled_back'].append(migration.id)
                
                # Save state after each successful rollback
                self._save_state()
                
            except Exception as e:
                logger.error(f"Rollback of {migration.id} failed: {str(e)}")
                results['failed'].append({
                    'id': migration.id,
                    'error': str(e)
                })
                
                # Stop on first failure
                break
        
        return results
    
    async def _rollback_migration(self, migration: Migration) -> None:
        """Rollback a single migration"""
        migration_file = self.migrations_dir / f"{migration.id}.py"
        
        # Import and execute rollback
        import importlib.util
        spec = importlib.util.spec_from_file_location(migration.id, migration_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Run rollback
        if hasattr(module, 'down'):
            if asyncio.iscoroutinefunction(module.down):
                await module.down()
            else:
                module.down()
        else:
            raise ValueError(f"Migration {migration.id} missing 'down' function")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        migrations = self.discover_migrations()
        
        return {
            'total_migrations': len(migrations),
            'applied_migrations': len(self.applied_migrations),
            'pending_migrations': len([m for m in migrations if m.id not in self.applied_migrations]),
            'current_version': max(self.applied_migrations) if self.applied_migrations else None,
            'migrations': [
                {
                    'id': m.id,
                    'name': m.name,
                    'version': m.version,
                    'applied': m.id in self.applied_migrations,
                    'checksum': m.checksum
                }
                for m in migrations
            ]
        }
    
    def validate_migrations(self) -> List[Dict[str, str]]:
        """Validate migration integrity"""
        issues = []
        migrations = self.discover_migrations()
        
        # Check for duplicate versions
        versions = [m.version for m in migrations]
        duplicates = [v for v in versions if versions.count(v) > 1]
        if duplicates:
            issues.append({
                'type': 'duplicate_version',
                'message': f"Duplicate versions found: {duplicates}"
            })
        
        # Check for gaps in version sequence
        numeric_versions = sorted([int(v) for v in versions if v.isdigit()])
        if numeric_versions:
            expected = list(range(1, max(numeric_versions) + 1))
            missing = set(expected) - set(numeric_versions)
            if missing:
                issues.append({
                    'type': 'missing_versions',
                    'message': f"Missing versions: {missing}"
                })
        
        # Check for checksum changes
        for migration in migrations:
            if migration.id in self.applied_migrations:
                # Compare with stored checksum
                # This would require storing checksums in state
                pass
        
        return issues


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage AgentVault™ migrations")
    parser.add_argument("command", choices=["migrate", "rollback", "status", "validate"])
    parser.add_argument("--version", help="Target version for migrate/rollback")
    parser.add_argument("--migrations-dir", help="Migrations directory path")
    
    args = parser.parse_args()
    
    manager = MigrationManager(
        migrations_dir=args.migrations_dir or "/app/migrations"
    )
    
    if args.command == "migrate":
        results = await manager.run_migrations(target_version=args.version)
        print(json.dumps(results, indent=2))
        
    elif args.command == "rollback":
        if not args.version:
            print("Error: --version required for rollback")
            return 1
        results = await manager.rollback(args.version)
        print(json.dumps(results, indent=2))
        
    elif args.command == "status":
        status = manager.get_status()
        print(json.dumps(status, indent=2))
        
    elif args.command == "validate":
        issues = manager.validate_migrations()
        if issues:
            print("Validation issues found:")
            for issue in issues:
                print(f"  - {issue['type']}: {issue['message']}")
            return 1
        else:
            print("All migrations are valid")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))