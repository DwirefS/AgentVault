"""
AgentVault™ Production Startup Script
Ensures all components are properly initialized with Azure AD authentication
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/agentvault/startup.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class AgentVaultStartup:
    """Production startup manager for AgentVault"""
    
    def __init__(self):
        self.config = self._load_configuration()
        self.components = {}
        
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        
        # Validate required environment variables
        required_vars = [
            'AZURE_TENANT_ID',
            'AZURE_CLIENT_ID',
            'AZURE_CLIENT_SECRET',
            'AZURE_SUBSCRIPTION_ID',
            'AZURE_RESOURCE_GROUP',
            'AZURE_KEY_VAULT_URL',
            'AZURE_ANF_ACCOUNT',
            'REDIS_URL'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            logger.error(f"Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Build configuration
        config = {
            'environment': os.getenv('ENVIRONMENT', 'production'),
            'azure': {
                'tenant_id': os.getenv('AZURE_TENANT_ID'),
                'client_id': os.getenv('AZURE_CLIENT_ID'),
                'client_secret': os.getenv('AZURE_CLIENT_SECRET'),
                'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
                'resource_group': os.getenv('AZURE_RESOURCE_GROUP'),
                'location': os.getenv('AZURE_LOCATION', 'eastus'),
                'anf_account': os.getenv('AZURE_ANF_ACCOUNT'),
                'anf_subnet_id': os.getenv('AZURE_ANF_SUBNET_ID')
            },
            'security': {
                'key_vault_url': os.getenv('AZURE_KEY_VAULT_URL'),
                'enable_encryption': True,
                'enable_audit': True,
                'default_algorithm': 'aes-256-gcm',
                'key_rotation_days': int(os.getenv('KEY_ROTATION_DAYS', '90')),
                'enable_hsm': os.getenv('ENABLE_HSM', 'true').lower() == 'true',
                'quantum_resistant': os.getenv('QUANTUM_RESISTANT', 'true').lower() == 'true'
            },
            'redis': {
                'redis_url': os.getenv('REDIS_URL'),
                'cluster_mode': os.getenv('REDIS_CLUSTER_MODE', 'true').lower() == 'true',
                'max_connections': int(os.getenv('REDIS_MAX_CONNECTIONS', '100')),
                'enable_pipelining': True,
                'enable_l1_cache': True,
                'compression_enabled': True
            },
            'ml': {
                'model_path': os.getenv('ML_MODEL_PATH', '/models'),
                'enable_training': os.getenv('ML_ENABLE_TRAINING', 'true').lower() == 'true',
                'update_interval': os.getenv('ML_UPDATE_INTERVAL', '1h'),
                'gpu_enabled': os.getenv('ML_GPU_ENABLED', 'true').lower() == 'true'
            },
            'monitoring': {
                'azure_monitor_workspace_id': os.getenv('AZURE_MONITOR_WORKSPACE_ID'),
                'data_collection_endpoint': os.getenv('DATA_COLLECTION_ENDPOINT'),
                'data_collection_rule_id': os.getenv('DATA_COLLECTION_RULE_ID'),
                'enable_custom_metrics': True,
                'anomaly_threshold': float(os.getenv('ANOMALY_THRESHOLD', '3.0'))
            },
            'api': {
                'host': os.getenv('API_HOST', '0.0.0.0'),
                'port': int(os.getenv('API_PORT', '8080')),
                'workers': int(os.getenv('API_WORKERS', '4')),
                'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
                'allowed_hosts': os.getenv('ALLOWED_HOSTS', '*').split(',')
            }
        }
        
        logger.info(f"Configuration loaded for environment: {config['environment']}")
        return config
    
    async def verify_azure_authentication(self) -> bool:
        """Verify Azure AD authentication is working"""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.resource import ResourceManagementClient
            
            logger.info("Verifying Azure AD authentication...")
            
            # Create credential
            credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credential=True,
                logging_enable=True
            )
            
            # Test authentication by listing resource groups
            resource_client = ResourceManagementClient(
                credential=credential,
                subscription_id=self.config['azure']['subscription_id']
            )
            
            # Try to get the resource group
            rg = resource_client.resource_groups.get(
                self.config['azure']['resource_group']
            )
            
            logger.info(f"Azure AD authentication successful. Resource group: {rg.name}")
            return True
            
        except Exception as e:
            logger.error(f"Azure AD authentication failed: {str(e)}")
            logger.error("Please check:")
            logger.error("1. Service principal credentials are correct")
            logger.error("2. Service principal has required permissions")
            logger.error("3. Workload identity is properly configured (if in AKS)")
            return False
    
    async def verify_key_vault_access(self) -> bool:
        """Verify Azure Key Vault access"""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            
            logger.info("Verifying Key Vault access...")
            
            credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credential=True
            )
            
            secret_client = SecretClient(
                vault_url=self.config['security']['key_vault_url'],
                credential=credential
            )
            
            # Try to list secrets (just one to verify access)
            secrets = secret_client.list_properties_of_secrets(max_page_size=1)
            for _ in secrets:
                break
            
            logger.info(f"Key Vault access verified: {self.config['security']['key_vault_url']}")
            return True
            
        except Exception as e:
            logger.error(f"Key Vault access failed: {str(e)}")
            logger.error("Required RBAC roles: Key Vault Secrets Officer or Key Vault Crypto Officer")
            return False
    
    async def verify_anf_access(self) -> bool:
        """Verify Azure NetApp Files access"""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.mgmt.netapp import NetAppManagementClient
            
            logger.info("Verifying Azure NetApp Files access...")
            
            credential = DefaultAzureCredential(
                exclude_interactive_browser_credential=True,
                exclude_visual_studio_code_credential=True
            )
            
            anf_client = NetAppManagementClient(
                credential=credential,
                subscription_id=self.config['azure']['subscription_id']
            )
            
            # Try to get the ANF account
            account = anf_client.accounts.get(
                resource_group_name=self.config['azure']['resource_group'],
                account_name=self.config['azure']['anf_account']
            )
            
            logger.info(f"ANF access verified. Account: {account.name}")
            return True
            
        except Exception as e:
            logger.error(f"ANF access failed: {str(e)}")
            logger.error("Required permissions: NetApp Account Reader or Contributor")
            return False
    
    async def initialize_components(self):
        """Initialize all AgentVault components"""
        logger.info("Initializing AgentVault components...")
        
        try:
            # Import components
            from .core.advanced_orchestrator import AdvancedOrchestrator
            from .storage.anf_advanced_manager import ANFAdvancedManager
            from .cache.distributed_cache import DistributedCache
            from .vectordb.vector_store import VectorStore
            from .ml.advanced_agent_dna import AdvancedAgentDNA
            from .security.advanced_encryption import AdvancedEncryptionManager
            from .monitoring.advanced_monitoring import AdvancedMonitoringSystem
            from .auth.azure_ad_auth import AzureADAuthProvider, load_azure_ad_config
            
            # Initialize authentication provider
            logger.info("Initializing Azure AD authentication provider...")
            ad_config = load_azure_ad_config()
            auth_provider = AzureADAuthProvider(ad_config)
            await auth_provider.initialize()
            self.components['auth'] = auth_provider
            
            # Initialize encryption manager first (needed by other components)
            logger.info("Initializing encryption manager...")
            encryption_manager = AdvancedEncryptionManager(self.config)
            await encryption_manager.initialize()
            self.components['encryption'] = encryption_manager
            
            # Initialize storage orchestrator
            logger.info("Initializing storage orchestrator...")
            orchestrator = AdvancedOrchestrator(self.config)
            await orchestrator.initialize()
            self.components['orchestrator'] = orchestrator
            
            # Initialize ANF manager
            logger.info("Initializing ANF manager...")
            anf_manager = ANFAdvancedManager(self.config['azure'])
            await anf_manager.initialize()
            self.components['anf'] = anf_manager
            
            # Initialize distributed cache
            logger.info("Initializing distributed cache...")
            cache = DistributedCache(self.config['redis'])
            await cache.initialize()
            self.components['cache'] = cache
            
            # Initialize vector store
            logger.info("Initializing vector store...")
            vector_store = VectorStore({
                'index_type': 'HNSW',
                'dimension': 1536,
                'metric': 'cosine',
                'gpu_enabled': self.config['ml']['gpu_enabled']
            })
            await vector_store.initialize()
            self.components['vector_store'] = vector_store
            
            # Initialize ML engine
            logger.info("Initializing ML engine...")
            ml_engine = AdvancedAgentDNA(self.config['ml'])
            await ml_engine.initialize()
            self.components['ml_engine'] = ml_engine
            
            # Initialize monitoring system
            logger.info("Initializing monitoring system...")
            monitoring = AdvancedMonitoringSystem(self.config['monitoring'])
            monitoring._start_background_tasks()
            self.components['monitoring'] = monitoring
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    async def perform_health_checks(self) -> Dict[str, bool]:
        """Perform health checks on all components"""
        logger.info("Performing health checks...")
        
        health_status = {}
        
        # Check each component
        for name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    health_status[name] = await component.health_check()
                else:
                    # Basic check - component exists
                    health_status[name] = component is not None
            except Exception as e:
                logger.error(f"Health check failed for {name}: {str(e)}")
                health_status[name] = False
        
        # Overall health
        health_status['overall'] = all(health_status.values())
        
        logger.info(f"Health check results: {health_status}")
        return health_status
    
    async def start_api_server(self):
        """Start the FastAPI server"""
        logger.info("Starting API server...")
        
        # Import and run the API server
        from .api.main import app
        import uvicorn
        
        # Pass components to the app
        app.state.components = self.components
        
        # Run server
        await uvicorn.Server(
            uvicorn.Config(
                app,
                host=self.config['api']['host'],
                port=self.config['api']['port'],
                workers=self.config['api']['workers'],
                log_level="info",
                access_log=True
            )
        ).serve()
    
    async def run(self):
        """Main startup sequence"""
        logger.info("=" * 60)
        logger.info("AgentVault™ Production Startup")
        logger.info("=" * 60)
        
        try:
            # Step 1: Verify Azure authentication
            if not await self.verify_azure_authentication():
                raise Exception("Azure AD authentication verification failed")
            
            # Step 2: Verify Key Vault access
            if not await self.verify_key_vault_access():
                raise Exception("Key Vault access verification failed")
            
            # Step 3: Verify ANF access
            if not await self.verify_anf_access():
                raise Exception("ANF access verification failed")
            
            # Step 4: Initialize all components
            await self.initialize_components()
            
            # Step 5: Perform health checks
            health_status = await self.perform_health_checks()
            if not health_status['overall']:
                raise Exception(f"Health checks failed: {health_status}")
            
            # Step 6: Start API server
            logger.info("All systems operational. Starting API server...")
            await self.start_api_server()
            
        except Exception as e:
            logger.error(f"Startup failed: {str(e)}")
            logger.error("Please check the logs and configuration")
            raise
        finally:
            # Cleanup on shutdown
            logger.info("Shutting down AgentVault...")
            for name, component in self.components.items():
                if hasattr(component, 'shutdown'):
                    try:
                        await component.shutdown()
                        logger.info(f"Shut down {name}")
                    except Exception as e:
                        logger.error(f"Error shutting down {name}: {str(e)}")


def main():
    """Main entry point"""
    startup = AgentVaultStartup()
    asyncio.run(startup.run())


if __name__ == "__main__":
    main()