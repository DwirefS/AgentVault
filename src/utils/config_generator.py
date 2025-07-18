#!/usr/bin/env python3
"""
AgentVault™ Configuration Generator
Generates configuration from environment variables and templates
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from cryptography.fernet import Fernet
import base64

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """Generates and manages AgentVault™ configuration"""
    
    def __init__(self, template_path: Optional[str] = None):
        self.template_path = template_path or "/app/configs/config.template.yaml"
        self.output_path = "/app/configs/config.yaml"
        self.secrets_path = "/app/configs/secrets.json"
        
    def generate_from_environment(self) -> Dict[str, Any]:
        """Generate configuration from environment variables"""
        
        config = {
            "version": "1.0",
            "environment": os.getenv("AGENTVAULT_ENV", "production"),
            
            # Azure Configuration
            "azure": {
                "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
                "resource_group": os.getenv("AZURE_RESOURCE_GROUP", "agentvault-rg"),
                "location": os.getenv("AZURE_LOCATION", "eastus"),
                "tenant_id": os.getenv("AZURE_TENANT_ID"),
                "client_id": os.getenv("AZURE_CLIENT_ID"),
                "client_secret": self._encrypt_secret(os.getenv("AZURE_CLIENT_SECRET", "")),
            },
            
            # Azure NetApp Files Configuration
            "anf": {
                "account_name": os.getenv("ANF_ACCOUNT_NAME", "agentvault-anf"),
                "pool_name_prefix": os.getenv("ANF_POOL_PREFIX", "agentvault-pool"),
                "volume_name_prefix": os.getenv("ANF_VOLUME_PREFIX", "agent"),
                "subnet_id": os.getenv("ANF_SUBNET_ID"),
                "default_size_gb": int(os.getenv("ANF_DEFAULT_SIZE_GB", "100")),
                "mount_path": os.getenv("ANF_MOUNT_PATH", "/mnt/agentvault"),
                "service_levels": {
                    "ultra": {"enabled": os.getenv("ANF_ULTRA_ENABLED", "true").lower() == "true"},
                    "premium": {"enabled": os.getenv("ANF_PREMIUM_ENABLED", "true").lower() == "true"},
                    "standard": {"enabled": os.getenv("ANF_STANDARD_ENABLED", "true").lower() == "true"},
                }
            },
            
            # Redis Configuration
            "redis": {
                "url": os.getenv("REDIS_URL", "redis://localhost:6379"),
                "ssl_enabled": os.getenv("REDIS_SSL_ENABLED", "true").lower() == "true",
                "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "100")),
                "key_prefix": os.getenv("REDIS_KEY_PREFIX", "agentvault:"),
                "ttl_seconds": int(os.getenv("REDIS_TTL_SECONDS", "3600")),
            },
            
            # Storage Configuration
            "storage": {
                "default_tier": os.getenv("STORAGE_DEFAULT_TIER", "premium"),
                "compression_enabled": os.getenv("STORAGE_COMPRESSION_ENABLED", "true").lower() == "true",
                "encryption_enabled": os.getenv("STORAGE_ENCRYPTION_ENABLED", "true").lower() == "true",
                "deduplication_enabled": os.getenv("STORAGE_DEDUP_ENABLED", "true").lower() == "true",
                "auto_tiering_enabled": os.getenv("STORAGE_AUTO_TIERING", "true").lower() == "true",
                "retention_days": {
                    "ultra": int(os.getenv("RETENTION_ULTRA_DAYS", "7")),
                    "premium": int(os.getenv("RETENTION_PREMIUM_DAYS", "30")),
                    "standard": int(os.getenv("RETENTION_STANDARD_DAYS", "90")),
                    "cool": int(os.getenv("RETENTION_COOL_DAYS", "365")),
                    "archive": int(os.getenv("RETENTION_ARCHIVE_DAYS", "2555")),
                }
            },
            
            # ML Configuration
            "ml": {
                "dna_profiling_enabled": os.getenv("ML_DNA_ENABLED", "true").lower() == "true",
                "cognitive_balancing_enabled": os.getenv("ML_COGNITIVE_ENABLED", "true").lower() == "true",
                "model_update_interval": int(os.getenv("ML_UPDATE_INTERVAL", "3600")),
                "training_batch_size": int(os.getenv("ML_BATCH_SIZE", "1000")),
                "feature_dimensions": int(os.getenv("ML_FEATURE_DIMS", "128")),
            },
            
            # Security Configuration
            "security": {
                "encryption_algorithm": os.getenv("SECURITY_ENCRYPTION", "AES256-GCM"),
                "key_rotation_days": int(os.getenv("SECURITY_KEY_ROTATION", "90")),
                "audit_logging_enabled": os.getenv("SECURITY_AUDIT", "true").lower() == "true",
                "compliance_mode": os.getenv("SECURITY_COMPLIANCE", "SOC2,HIPAA,GDPR"),
                "zero_trust_enabled": os.getenv("SECURITY_ZERO_TRUST", "true").lower() == "true",
            },
            
            # API Configuration
            "api": {
                "host": os.getenv("API_HOST", "0.0.0.0"),
                "port": int(os.getenv("API_PORT", "8000")),
                "workers": int(os.getenv("API_WORKERS", "4")),
                "cors_enabled": os.getenv("API_CORS_ENABLED", "true").lower() == "true",
                "rate_limit_enabled": os.getenv("API_RATE_LIMIT", "true").lower() == "true",
                "max_requests_per_minute": int(os.getenv("API_MAX_REQUESTS", "1000")),
            },
            
            # Monitoring Configuration
            "monitoring": {
                "prometheus_enabled": os.getenv("MONITORING_PROMETHEUS", "true").lower() == "true",
                "prometheus_port": int(os.getenv("PROMETHEUS_PORT", "9090")),
                "metrics_interval": int(os.getenv("METRICS_INTERVAL", "30")),
                "alert_webhook": os.getenv("ALERT_WEBHOOK_URL"),
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
            },
            
            # Agent Framework Configuration
            "frameworks": {
                "langchain": {
                    "enabled": os.getenv("LANGCHAIN_ENABLED", "true").lower() == "true",
                    "memory_backend": "agentvault",
                    "vectorstore_backend": "agentvault",
                },
                "autogen": {
                    "enabled": os.getenv("AUTOGEN_ENABLED", "true").lower() == "true",
                    "cache_backend": "agentvault",
                    "conversation_backend": "agentvault",
                },
                "crewai": {
                    "enabled": os.getenv("CREWAI_ENABLED", "true").lower() == "true",
                    "memory_backend": "agentvault",
                }
            }
        }
        
        return config
    
    def _encrypt_secret(self, secret: str) -> str:
        """Encrypt sensitive configuration values"""
        if not secret:
            return ""
            
        # Generate or load encryption key
        key_file = Path("/app/configs/.encryption_key")
        if key_file.exists():
            key = key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            key_file.parent.mkdir(parents=True, exist_ok=True)
            key_file.write_bytes(key)
            key_file.chmod(0o600)
        
        fernet = Fernet(key)
        encrypted = fernet.encrypt(secret.encode())
        return base64.b64encode(encrypted).decode()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration completeness and correctness"""
        required_fields = [
            "azure.subscription_id",
            "azure.tenant_id",
            "azure.client_id",
            "anf.subnet_id",
            "redis.url"
        ]
        
        for field in required_fields:
            parts = field.split(".")
            value = config
            for part in parts:
                if part not in value:
                    logger.error(f"Missing required configuration: {field}")
                    return False
                value = value[part]
            
            if not value:
                logger.error(f"Empty required configuration: {field}")
                return False
        
        return True
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file"""
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Set appropriate permissions
        output_path.chmod(0o600)
        logger.info(f"Configuration saved to {output_path}")
    
    def generate_secrets_file(self) -> None:
        """Generate secrets file for sensitive data"""
        secrets = {
            "azure_client_secret": os.getenv("AZURE_CLIENT_SECRET"),
            "redis_password": os.getenv("REDIS_PASSWORD"),
            "encryption_key": os.getenv("ENCRYPTION_KEY"),
            "api_keys": {
                "openai": os.getenv("OPENAI_API_KEY"),
                "anthropic": os.getenv("ANTHROPIC_API_KEY"),
                "cohere": os.getenv("COHERE_API_KEY"),
            }
        }
        
        # Remove None values
        secrets = {k: v for k, v in secrets.items() if v is not None}
        
        secrets_path = Path(self.secrets_path)
        secrets_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(secrets_path, 'w') as f:
            json.dump(secrets, f, indent=2)
        
        # Set strict permissions
        secrets_path.chmod(0o600)
        logger.info(f"Secrets file saved to {secrets_path}")
    
    def run(self) -> None:
        """Main configuration generation process"""
        logger.info("Starting configuration generation...")
        
        # Generate configuration
        config = self.generate_from_environment()
        
        # Validate configuration
        if not self.validate_config(config):
            raise ValueError("Configuration validation failed")
        
        # Save configuration
        self.save_config(config)
        
        # Generate secrets file
        self.generate_secrets_file()
        
        logger.info("Configuration generation completed successfully")


def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate AgentVault™ configuration")
    parser.add_argument("--template", help="Configuration template path")
    parser.add_argument("--output", help="Output configuration path")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    
    args = parser.parse_args()
    
    generator = ConfigGenerator(template_path=args.template)
    
    if args.output:
        generator.output_path = args.output
    
    if args.validate_only:
        config = generator.generate_from_environment()
        if generator.validate_config(config):
            print("Configuration is valid")
            return 0
        else:
            print("Configuration validation failed")
            return 1
    else:
        generator.run()
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())