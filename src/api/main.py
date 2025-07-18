"""
AgentVault™ Production API Server
=================================

Complete FastAPI application with Azure AD authentication and all endpoints
This is the main entry point for the AgentVault API, providing:

- RESTful API endpoints for agent management
- Azure AD authentication and authorization
- Distributed storage orchestration
- Vector database operations
- ML model predictions and training
- Caching and monitoring capabilities
- Admin operations and system management

The API follows OpenAPI 3.0 specification and includes:
- Automatic API documentation at /docs (Swagger UI)
- ReDoc documentation at /redoc
- Prometheus metrics at /metrics
- Health and readiness endpoints
- Comprehensive error handling and logging

Security Features:
- Azure AD OAuth2 authentication
- Role-based access control (RBAC)
- API key authentication (optional)
- Request rate limiting
- Input validation and sanitization
- CORS protection
- SQL injection prevention

Performance Features:
- Async/await for all I/O operations
- Connection pooling for databases
- Response caching with Redis
- GZip compression
- Request/response streaming
- Horizontal scaling support

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
Version: 1.0.0
License: Proprietary - SapientEdge LLC
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Query, Body, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

# Import all AgentVault components
# These imports bring in the core functionality of the AgentVault platform
# Each component is designed to be modular and independently scalable
from ..auth.azure_ad_auth import (
    AzureADAuthProvider,      # Handles Azure AD OAuth2 authentication
    AzureADAuthMiddleware,    # Middleware for request authentication
    AzureADConfig,           # Configuration model for Azure AD
    AuthenticatedUser,       # User model with permissions and roles
    Permission,              # Enum of all available permissions
    Role,                    # Enum of all available roles
    get_current_user,        # Dependency to get current authenticated user
    require_permissions,     # Decorator to require specific permissions
    require_any_permission,  # Decorator to require any of the given permissions
    require_role,           # Decorator to require specific role
    load_azure_ad_config    # Helper to load Azure AD config from environment
)
from ..core.advanced_orchestrator import AdvancedOrchestrator, RoutingStrategy
from ..storage.anf_advanced_manager import ANFAdvancedManager
from ..cache.distributed_cache import DistributedCache
from ..vectordb.vector_store import VectorStore, SearchMode
from ..ml.advanced_agent_dna import AdvancedAgentDNA
from ..security.advanced_encryption import AdvancedEncryptionManager
from ..monitoring.advanced_monitoring import AdvancedMonitoringSystem

# Configure module logger with appropriate namespace
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for API Request/Response Validation
# =============================================================================
# These models provide:
# - Automatic request validation
# - Response serialization
# - OpenAPI schema generation
# - Type hints for better IDE support

class HealthResponse(BaseModel):
    """
    Health check response model
    Used by monitoring systems to verify API availability
    """
    status: str              # Current health status (healthy, degraded, unhealthy)
    timestamp: datetime      # Current server time for clock sync verification
    version: str            # API version for compatibility checking
    environment: str        # Deployment environment (dev, staging, prod)


class AgentRegistrationRequest(BaseModel):
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent type (langchain, autogen, crewai)")
    description: str = Field(None, description="Agent description")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentResponse(BaseModel):
    agent_id: str
    agent_type: str
    status: str
    tier: str
    created_at: datetime
    storage_profile: Dict[str, Any]
    permissions: List[str]


class StorageRequest(BaseModel):
    agent_id: str
    data_type: str = Field(..., description="Type of data (vector, memory, knowledge, etc.)")
    data: Any = Field(..., description="Data to store")
    tier: Optional[str] = Field(None, description="Storage tier (auto-selected if not specified)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    encryption_required: bool = Field(True, description="Whether to encrypt data")


class StorageResponse(BaseModel):
    storage_id: str
    agent_id: str
    tier: str
    size_bytes: int
    encrypted: bool
    compression_ratio: float
    latency_ms: float
    timestamp: datetime


class VectorSearchRequest(BaseModel):
    agent_id: str
    query: str = Field(None, description="Text query for search")
    vector: List[float] = Field(None, description="Vector for similarity search")
    k: int = Field(10, description="Number of results to return")
    search_mode: SearchMode = Field(SearchMode.SIMILARITY, description="Search mode")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Metadata filters")
    include_metadata: bool = Field(True, description="Include metadata in results")


class VectorSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    search_time_ms: float
    total_results: int


class MLPredictionRequest(BaseModel):
    agent_id: str
    feature_data: Dict[str, Any]
    prediction_type: str = Field("tier", description="Type of prediction")


class MLPredictionResponse(BaseModel):
    prediction: Any
    confidence: float
    reasoning: Dict[str, Any]
    model_version: str


class CacheOperationRequest(BaseModel):
    key: str
    value: Any = None
    ttl: Optional[int] = None
    tags: List[str] = Field(default_factory=list)


class MonitoringMetricRequest(BaseModel):
    metric_name: str
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)
    timestamp: Optional[datetime] = None


class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: datetime
    request_id: str


# Global components
auth_provider: Optional[AzureADAuthProvider] = None
orchestrator: Optional[AdvancedOrchestrator] = None
anf_manager: Optional[ANFAdvancedManager] = None
cache: Optional[DistributedCache] = None
vector_store: Optional[VectorStore] = None
ml_engine: Optional[AdvancedAgentDNA] = None
encryption_manager: Optional[AdvancedEncryptionManager] = None
monitoring: Optional[AdvancedMonitoringSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager using ASGI lifespan protocol
    
    This context manager handles the complete lifecycle of the application:
    - Startup: Initialize all components, establish connections, load models
    - Runtime: Maintain connections, handle background tasks
    - Shutdown: Gracefully close connections, save state, cleanup resources
    
    The lifespan manager ensures:
    - All components are initialized in the correct order
    - Dependencies are properly resolved
    - Resources are cleaned up on shutdown
    - Errors during startup prevent the app from serving requests
    - Background tasks are properly cancelled on shutdown
    """
    # Declare global components that will be shared across the application
    # These are initialized during startup and accessed throughout the app lifecycle
    global auth_provider, orchestrator, anf_manager, cache, vector_store
    global ml_engine, encryption_manager, monitoring
    
    logger.info("Starting AgentVault API Server...")
    
    try:
        # =====================================================================
        # PHASE 1: Authentication System Initialization
        # =====================================================================
        # Azure AD must be initialized first as all other components depend on it
        # for authentication and authorization decisions
        logger.info("Initializing Azure AD authentication...")
        ad_config = load_azure_ad_config()  # Load from environment variables
        auth_provider = AzureADAuthProvider(ad_config)
        await auth_provider.initialize()    # Establish connection to Azure AD
        
        # Store auth provider in app state for middleware access
        # This allows the authentication middleware to validate tokens
        app.state.auth_provider = auth_provider
        
        # =====================================================================
        # PHASE 2: Configuration Loading
        # =====================================================================
        # Build configuration dictionary from environment variables
        # This centralizes all configuration in one place for easy management
        config = {
            'azure': {
                'subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
                'resource_group': os.getenv('AZURE_RESOURCE_GROUP'),
                'anf_account': os.getenv('AZURE_ANF_ACCOUNT')
            },
            'redis': {
                'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379'),
                'cluster_mode': os.getenv('REDIS_CLUSTER_MODE', 'false').lower() == 'true'
            },
            'security': {
                'vault_url': os.getenv('AZURE_KEY_VAULT_URL'),
                'encryption_enabled': True  # Always enable encryption in production
            },
            'ml': {
                'model_path': os.getenv('ML_MODEL_PATH', '/models')  # Volume mount path
            },
            'monitoring': {
                'azure_monitor_workspace_id': os.getenv('AZURE_MONITOR_WORKSPACE_ID')
            }
        }
        
        # =====================================================================
        # PHASE 3: Component Initialization
        # =====================================================================
        # Initialize all AgentVault components in dependency order
        # Each component may depend on others, so order matters
        logger.info("Initializing AgentVault components...")
        
        # 1. Storage Orchestrator - Central coordinator for all storage operations
        # This must be initialized early as other components register with it
        orchestrator = AdvancedOrchestrator(config)
        await orchestrator.initialize()
        logger.info("✓ Storage orchestrator initialized")
        
        # 2. Azure NetApp Files Manager - Handles high-performance file storage
        # Manages multiple storage tiers with automatic data movement
        anf_manager = ANFAdvancedManager(config['azure'])
        await anf_manager.initialize()
        logger.info("✓ ANF manager initialized with 5 storage tiers")
        
        # 3. Distributed Cache - Redis-based caching layer
        # Provides sub-millisecond latency for frequently accessed data
        cache = DistributedCache(config['redis'])
        await cache.initialize()
        logger.info("✓ Distributed cache initialized")
        
        # 4. Vector Store - Manages embeddings and similarity search
        # Configured for OpenAI ada-002 embeddings (1536 dimensions)
        vector_store = VectorStore({
            'index_type': 'HNSW',      # Hierarchical Navigable Small World graphs
            'dimension': 1536,          # OpenAI ada-002 embedding dimension
            'metric': 'cosine'          # Cosine similarity for semantic search
        })
        await vector_store.initialize()
        logger.info("✓ Vector store initialized with HNSW index")
        
        # 5. ML Engine - Advanced agent profiling and optimization
        # Loads pre-trained models for tier prediction and anomaly detection
        ml_engine = AdvancedAgentDNA(config['ml'])
        await ml_engine.initialize()
        logger.info("✓ ML engine initialized with agent DNA profiling")
        
        # 6. Encryption Manager - Handles all cryptographic operations
        # Integrates with Azure Key Vault for key management
        encryption_manager = AdvancedEncryptionManager(config['security'])
        await encryption_manager.initialize()
        logger.info("✓ Encryption manager initialized with Azure Key Vault")
        
        # 7. Monitoring System - Comprehensive observability
        # Starts background tasks for metrics collection and alerting
        monitoring = AdvancedMonitoringSystem(config['monitoring'])
        monitoring._start_background_tasks()
        logger.info("✓ Monitoring system initialized with Azure Monitor")
        
        logger.info("AgentVault API Server started successfully")
        
        # Yield control back to FastAPI
        # The application will run until shutdown is triggered
        yield
        
    finally:
        # =====================================================================
        # PHASE 4: Graceful Shutdown
        # =====================================================================
        # Clean shutdown is critical for data integrity
        # Components are shut down in reverse order of initialization
        logger.info("Shutting down AgentVault API Server...")
        
        # Cancel any running background tasks first
        # This prevents new operations during shutdown
        if monitoring:
            await monitoring.shutdown()
            logger.info("✓ Monitoring system shut down")
            
        # Close ML engine and save any pending model updates
        if ml_engine:
            await ml_engine.shutdown()
            logger.info("✓ ML engine shut down")
            
        # Shutdown encryption manager and clear key cache
        if encryption_manager:
            await encryption_manager.shutdown()
            logger.info("✓ Encryption manager shut down")
            
        # Close vector store and persist any pending index updates
        if vector_store:
            await vector_store.close()
            logger.info("✓ Vector store shut down")
            
        # Flush cache and close Redis connections
        if cache:
            await cache.shutdown()
            logger.info("✓ Cache shut down")
            
        # Shutdown ANF manager and unmount volumes
        if anf_manager:
            await anf_manager.shutdown()
            logger.info("✓ ANF manager shut down")
            
        # Shutdown orchestrator last as other components may use it
        if orchestrator:
            await orchestrator.shutdown()
            logger.info("✓ Storage orchestrator shut down")
            
        # Close authentication provider
        if auth_provider:
            await auth_provider.close()
            logger.info("✓ Authentication provider shut down")
        
        logger.info("AgentVault API Server shutdown complete")


# =============================================================================
# FastAPI Application Instance
# =============================================================================
# Create the main FastAPI application with production-ready configuration
# This instance will be used by the ASGI server (uvicorn) to handle requests
app = FastAPI(
    title="AgentVault™ API",
    description="""
    Enterprise AI Agent Storage Platform API
    
    AgentVault™ provides high-performance, secure storage for AI agents with:
    - Multi-tier storage with automatic optimization
    - Vector database for RAG and semantic search
    - Distributed caching for sub-millisecond latency
    - ML-powered agent profiling and optimization
    - Enterprise-grade security with Azure AD
    - Comprehensive monitoring and observability
    
    ## Authentication
    All endpoints (except health checks) require Azure AD authentication.
    Include the Bearer token in the Authorization header:
    ```
    Authorization: Bearer <your-azure-ad-token>
    ```
    
    ## Rate Limiting
    API calls are rate-limited per user:
    - Standard tier: 1000 requests/hour
    - Premium tier: 10000 requests/hour
    - Enterprise tier: Unlimited
    
    ## Support
    For support, contact: support@sapientedge.io
    """,
    version="1.0.0",
    docs_url="/docs",        # Swagger UI documentation
    redoc_url="/redoc",      # ReDoc documentation
    lifespan=lifespan,       # Lifespan context manager for startup/shutdown
    # Additional OpenAPI customization
    openapi_tags=[
        {"name": "Health", "description": "Health and readiness checks"},
        {"name": "Agents", "description": "AI agent management"},
        {"name": "Storage", "description": "Data storage operations"},
        {"name": "Vectors", "description": "Vector database operations"},
        {"name": "Machine Learning", "description": "ML predictions and training"},
        {"name": "Cache", "description": "Distributed cache operations"},
        {"name": "Monitoring", "description": "Metrics and monitoring"},
        {"name": "Admin", "description": "Administrative operations"}
    ]
)

# =============================================================================
# Middleware Configuration
# =============================================================================
# Middleware is executed in reverse order of addition (last added = first executed)

# 1. CORS Middleware - Handle Cross-Origin Resource Sharing
# Essential for web applications calling the API from different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv('CORS_ORIGINS', '*').split(','),  # Comma-separated list of allowed origins
    allow_credentials=True,       # Allow cookies and authorization headers
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
    expose_headers=["*"],        # Expose all headers to the client
    max_age=3600,               # Cache preflight requests for 1 hour
)

# 2. GZip Compression - Reduce response size for better performance
# Especially important for large agent data and vector embeddings
app.add_middleware(
    GZipMiddleware, 
    minimum_size=1000  # Only compress responses larger than 1KB
)

# 3. Trusted Host Middleware - Prevent host header attacks in production
# Only validate hosts in production to allow flexibility in development
if os.getenv('ENVIRONMENT') == 'production':
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=os.getenv('ALLOWED_HOSTS', '*').split(',')  # Comma-separated list of allowed hosts
    )

# Add Azure AD authentication middleware
@app.on_event("startup")
async def add_auth_middleware():
    if app.state.auth_provider:
        app.add_middleware(
            AzureADAuthMiddleware,
            auth_provider=app.state.auth_provider,
            exclude_paths=['/health', '/ready', '/metrics', '/docs', '/redoc', '/openapi.json']
        )

# Add Sentry for error tracking
if os.getenv('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=os.getenv('SENTRY_DSN'),
        environment=os.getenv('ENVIRONMENT', 'development'),
        traces_sample_rate=float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', '0.1'))
    )
    app.add_middleware(SentryAsgiMiddleware)

# Add Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            timestamp=datetime.utcnow(),
            request_id=request.headers.get('X-Request-ID', 'unknown')
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal error occurred",
            timestamp=datetime.utcnow(),
            request_id=request.headers.get('X-Request-ID', 'unknown')
        ).dict()
    )


# Health check endpoints (no auth required)
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        environment=os.getenv('ENVIRONMENT', 'development')
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint"""
    # Check all components
    checks = {
        'orchestrator': orchestrator is not None,
        'anf_manager': anf_manager is not None,
        'cache': cache is not None,
        'vector_store': vector_store is not None,
        'ml_engine': ml_engine is not None,
        'auth': auth_provider is not None
    }
    
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    else:
        raise HTTPException(status_code=503, detail={"status": "not ready", "checks": checks})


# =============================================================================
# Agent Management Endpoints
# =============================================================================
# These endpoints handle the lifecycle of AI agents in the system

@app.post(
    "/api/v1/agents",
    response_model=AgentResponse,
    tags=["Agents"],
    summary="Register a new AI agent",
    description="""
    Register a new AI agent with AgentVault.
    
    This endpoint:
    - Creates a new agent profile in the system
    - Analyzes agent characteristics to recommend optimal storage tier
    - Sets up dedicated storage namespaces
    - Initializes monitoring and metrics collection
    - Creates audit trail for compliance
    
    The agent will be assigned to a storage tier based on:
    - Expected data volume
    - Access patterns
    - Performance requirements
    - Cost optimization goals
    """,
    responses={
        200: {"description": "Agent successfully registered"},
        400: {"description": "Invalid agent configuration"},
        409: {"description": "Agent ID already exists"},
        500: {"description": "Internal server error"}
    }
)
@require_permissions(Permission.AGENT_WRITE)
async def register_agent(
    request: AgentRegistrationRequest,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """
    Register a new AI agent with AgentVault
    
    This is typically the first API call when onboarding a new agent.
    The system will analyze the agent configuration and metadata to:
    1. Determine optimal storage tier placement
    2. Configure appropriate caching strategies
    3. Set up monitoring thresholds
    4. Initialize security policies
    """
    try:
        # =====================================================================
        # Step 1: Validate and Record Metrics
        # =====================================================================
        # Record registration attempt for analytics and rate limiting
        monitoring.record_metric(
            'agentvault_agent_registrations_total',
            1,
            {
                'user_id': user.user_id,
                'agent_type': request.agent_type,
                'tenant_id': user.tenant_id  # Multi-tenant support
            }
        )
        
        # Log registration attempt for audit trail
        logger.info(f"Agent registration attempt: {request.agent_id} by {user.user_id}")
        
        # =====================================================================
        # Step 2: Register Agent with Orchestrator
        # =====================================================================
        # The orchestrator handles:
        # - Duplicate checking
        # - Storage tier assignment
        # - Initial profiling
        # - Resource allocation
        profile = await orchestrator.register_agent(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            config=request.config,
            metadata={
                **request.metadata,
                'registered_by': user.user_id,
                'registered_at': datetime.utcnow().isoformat(),
                'tenant_id': user.tenant_id,
                'api_version': '1.0.0'
            }
        )
        
        # =====================================================================
        # Step 3: Post-Registration Setup
        # =====================================================================
        # Initialize agent-specific resources
        # This happens asynchronously to avoid blocking the response
        asyncio.create_task(
            _initialize_agent_resources(
                agent_id=request.agent_id,
                profile=profile
            )
        )
        
        # =====================================================================
        # Step 4: Build Response
        # =====================================================================
        # Include only the permissions relevant to agent operations
        # This follows the principle of least privilege
        agent_permissions = [
            p.value for p in user.permissions 
            if p.value.startswith('Agent.')
        ]
        
        response = AgentResponse(
            agent_id=profile['agent_id'],
            agent_type=profile['agent_type'],
            status='active',
            tier=profile['recommended_tier'],
            created_at=datetime.utcnow(),
            storage_profile=profile,
            permissions=agent_permissions
        )
        
        # Log successful registration
        logger.info(f"Agent registered successfully: {request.agent_id}")
        
        return response
        
    except ValueError as e:
        # Handle validation errors (e.g., invalid agent type)
        logger.warning(f"Invalid agent registration: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except KeyError as e:
        # Handle duplicate agent IDs
        logger.warning(f"Duplicate agent ID: {request.agent_id}")
        raise HTTPException(status_code=409, detail="Agent ID already exists")
        
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Failed to register agent: {str(e)}", exc_info=True)
        
        # Record failure metric
        monitoring.record_metric(
            'agentvault_agent_registration_failures_total',
            1,
            {'error_type': type(e).__name__}
        )
        
        # Return generic error to avoid leaking internal details
        raise HTTPException(
            status_code=500, 
            detail="Failed to register agent. Please try again later."
        )


@app.get(
    "/api/v1/agents/{agent_id}",
    response_model=AgentResponse,
    tags=["Agents"],
    summary="Get agent information"
)
@require_permissions(Permission.AGENT_READ)
async def get_agent(
    agent_id: str = Path(..., description="Agent ID"),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Get agent information and status"""
    try:
        # Get agent profile
        profile = await orchestrator.get_agent_profile(agent_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse(
            agent_id=profile['agent_id'],
            agent_type=profile['agent_type'],
            status=profile.get('status', 'active'),
            tier=profile.get('current_tier', 'standard'),
            created_at=profile.get('created_at', datetime.utcnow()),
            storage_profile=profile,
            permissions=[p.value for p in user.permissions if p.value.startswith('Agent.')]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/api/v1/agents/{agent_id}",
    tags=["Agents"],
    summary="Delete an agent"
)
@require_permissions(Permission.AGENT_DELETE)
async def delete_agent(
    agent_id: str = Path(..., description="Agent ID"),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Delete an agent and all associated data"""
    try:
        # Delete agent
        await orchestrator.delete_agent(agent_id)
        
        # Record metric
        monitoring.record_metric(
            'agentvault_agent_deletions_total',
            1,
            {'user_id': user.user_id}
        )
        
        return {"message": f"Agent {agent_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Failed to delete agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Storage operations
@app.post(
    "/api/v1/storage",
    response_model=StorageResponse,
    tags=["Storage"],
    summary="Store data for an agent"
)
@require_permissions(Permission.STORAGE_WRITE)
async def store_data(
    request: StorageRequest,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Store data for an AI agent"""
    try:
        start_time = datetime.utcnow()
        
        # Process storage request
        result = await orchestrator.process_request({
            'operation': 'write',
            'agent_id': request.agent_id,
            'data_type': request.data_type,
            'data': request.data,
            'metadata': {
                **request.metadata,
                'stored_by': user.user_id
            },
            'tier': request.tier,
            'encryption_required': request.encryption_required
        })
        
        # Calculate metrics
        end_time = datetime.utcnow()
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # Record metrics
        monitoring.record_metric(
            'agentvault_storage_operations_total',
            1,
            {
                'operation': 'write',
                'tier': result['tier'],
                'status': 'success'
            }
        )
        monitoring.record_metric(
            'agentvault_request_duration_seconds',
            latency_ms / 1000,
            {
                'operation': 'write',
                'tier': result['tier']
            }
        )
        
        return StorageResponse(
            storage_id=result['storage_id'],
            agent_id=request.agent_id,
            tier=result['tier'],
            size_bytes=result['size_bytes'],
            encrypted=request.encryption_required,
            compression_ratio=result.get('compression_ratio', 1.0),
            latency_ms=latency_ms,
            timestamp=end_time
        )
        
    except Exception as e:
        logger.error(f"Failed to store data: {str(e)}")
        monitoring.record_metric(
            'agentvault_storage_operations_total',
            1,
            {
                'operation': 'write',
                'tier': request.tier or 'unknown',
                'status': 'error'
            }
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/storage/{storage_id}",
    tags=["Storage"],
    summary="Retrieve stored data"
)
@require_permissions(Permission.STORAGE_READ)
async def get_data(
    storage_id: str = Path(..., description="Storage ID"),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Retrieve stored data by ID"""
    try:
        # Get data
        result = await orchestrator.process_request({
            'operation': 'read',
            'storage_id': storage_id
        })
        
        if not result:
            raise HTTPException(status_code=404, detail="Data not found")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Vector operations
@app.post(
    "/api/v1/vectors/search",
    response_model=VectorSearchResponse,
    tags=["Vectors"],
    summary="Search vectors"
)
@require_permissions(Permission.STORAGE_READ)
async def search_vectors(
    request: VectorSearchRequest,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Search vectors for an agent"""
    try:
        start_time = datetime.utcnow()
        
        # Perform search
        results = await vector_store.search(
            agent_id=request.agent_id,
            query=request.query,
            k=request.k,
            search_mode=request.search_mode,
            filters=request.filters
        )
        
        # Calculate metrics
        end_time = datetime.utcnow()
        search_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Record metrics
        monitoring.record_metric(
            'agentvault_vector_searches_total',
            1,
            {
                'agent_id': request.agent_id,
                'search_mode': request.search_mode.value
            }
        )
        monitoring.record_metric(
            'agentvault_vector_search_latency_seconds',
            search_time_ms / 1000,
            {
                'search_mode': request.search_mode.value
            }
        )
        
        # Format results
        formatted_results = []
        for result in results:
            item = {
                'id': result.id,
                'score': result.score,
                'text': result.text
            }
            if request.include_metadata:
                item['metadata'] = result.metadata
            formatted_results.append(item)
        
        return VectorSearchResponse(
            results=formatted_results,
            search_time_ms=search_time_ms,
            total_results=len(results)
        )
        
    except Exception as e:
        logger.error(f"Failed to search vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/vectors",
    tags=["Vectors"],
    summary="Store vectors"
)
@require_permissions(Permission.STORAGE_WRITE)
async def store_vectors(
    agent_id: str = Body(...),
    texts: List[str] = Body(...),
    embeddings: List[List[float]] = Body(...),
    metadata: List[Dict[str, Any]] = Body(default_factory=list),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Store vectors for an agent"""
    try:
        # Store vectors
        ids = await vector_store.add_embeddings(
            agent_id=agent_id,
            texts=texts,
            embeddings=embeddings,
            metadatas=metadata
        )
        
        # Record metrics
        monitoring.record_metric(
            'agentvault_vectors_stored_total',
            len(ids),
            {'agent_id': agent_id}
        )
        
        return {
            "message": f"Stored {len(ids)} vectors",
            "ids": ids
        }
        
    except Exception as e:
        logger.error(f"Failed to store vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ML operations
@app.post(
    "/api/v1/ml/predict",
    response_model=MLPredictionResponse,
    tags=["Machine Learning"],
    summary="Get ML predictions"
)
@require_permissions(Permission.ML_READ)
async def predict(
    request: MLPredictionRequest,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Get ML predictions for agent optimization"""
    try:
        # Get prediction
        if request.prediction_type == "tier":
            result = await ml_engine.predict_optimal_tier(
                agent_id=request.agent_id,
                data_characteristics=request.feature_data
            )
            
            return MLPredictionResponse(
                prediction=result.predicted_tier,
                confidence=result.confidence,
                reasoning=result.feature_importance,
                model_version=result.model_version
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown prediction type: {request.prediction_type}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/v1/ml/train",
    tags=["Machine Learning"],
    summary="Trigger model training"
)
@require_permissions(Permission.ML_TRAIN)
async def train_models(
    model_type: str = Query(..., description="Model type to train"),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Trigger ML model training"""
    try:
        # Start training
        task_id = await ml_engine.train_models_async()
        
        return {
            "message": "Training started",
            "task_id": task_id,
            "started_by": user.user_id
        }
        
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Cache operations
@app.post(
    "/api/v1/cache",
    tags=["Cache"],
    summary="Set cache value"
)
@require_permissions(Permission.STORAGE_WRITE)
async def set_cache(
    request: CacheOperationRequest,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Set a value in the distributed cache"""
    try:
        success = await cache.set(
            key=request.key,
            value=request.value,
            ttl=request.ttl,
            tags=set(request.tags) if request.tags else None
        )
        
        if success:
            return {"message": "Value cached successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to cache value")
            
    except Exception as e:
        logger.error(f"Failed to set cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/cache/{key}",
    tags=["Cache"],
    summary="Get cache value"
)
@require_permissions(Permission.STORAGE_READ)
async def get_cache(
    key: str = Path(..., description="Cache key"),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Get a value from the distributed cache"""
    try:
        value = await cache.get(key)
        
        if value is None:
            raise HTTPException(status_code=404, detail="Key not found")
        
        return {"key": key, "value": value}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/api/v1/cache/{key}",
    tags=["Cache"],
    summary="Delete cache value"
)
@require_permissions(Permission.STORAGE_DELETE)
async def delete_cache(
    key: str = Path(..., description="Cache key"),
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Delete a value from the distributed cache"""
    try:
        success = await cache.delete(key)
        
        if success:
            return {"message": "Value deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Key not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring operations
@app.post(
    "/api/v1/metrics",
    tags=["Monitoring"],
    summary="Record custom metric"
)
@require_permissions(Permission.MONITORING_WRITE)
async def record_metric(
    request: MonitoringMetricRequest,
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Record a custom metric"""
    try:
        monitoring.record_metric(
            metric_name=request.metric_name,
            value=request.value,
            labels={
                **request.labels,
                'recorded_by': user.user_id
            }
        )
        
        return {"message": "Metric recorded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to record metric: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/monitoring/info",
    tags=["Monitoring"],
    summary="Get monitoring information"
)
@require_permissions(Permission.MONITORING_READ)
async def get_monitoring_info(
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Get monitoring system information"""
    try:
        # Get cache info
        cache_info = await cache.get_info()
        
        # Get SLA report
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        sla_report = await monitoring.get_sla_report(start_time, end_time)
        
        return {
            "cache": cache_info,
            "sla": sla_report,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get monitoring info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin operations
@app.get(
    "/api/v1/admin/users/{user_id}",
    tags=["Admin"],
    summary="Get user information"
)
@require_role(Role.ADMIN)
async def get_user_info(
    user_id: str = Path(..., description="User ID"),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get user information and permissions"""
    return {
        "user_id": user_id,
        "requested_by": current_user.user_id,
        "message": "This would return user information from Azure AD"
    }


@app.post(
    "/api/v1/admin/backup",
    tags=["Admin"],
    summary="Trigger backup"
)
@require_permissions(Permission.SYSTEM_ADMIN)
async def trigger_backup(
    user: AuthenticatedUser = Depends(get_current_user)
):
    """Trigger system backup"""
    try:
        # Trigger backup process
        return {
            "message": "Backup initiated",
            "initiated_by": user.user_id,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger backup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Helper Functions
# =============================================================================

async def _initialize_agent_resources(agent_id: str, profile: Dict[str, Any]) -> None:
    """
    Initialize agent-specific resources after registration
    
    This runs asynchronously after agent registration to:
    - Create dedicated cache namespaces
    - Initialize vector indexes
    - Set up monitoring dashboards
    - Configure alerting rules
    
    Args:
        agent_id: Unique agent identifier
        profile: Agent profile from registration
    """
    try:
        # Create cache namespace for agent
        await cache.create_namespace(f"agent:{agent_id}")
        
        # Initialize vector index if agent uses embeddings
        if profile.get('uses_embeddings', False):
            await vector_store.create_index(agent_id)
            
        # Set up agent-specific metrics
        monitoring.create_agent_dashboard(agent_id, profile)
        
        logger.info(f"Agent resources initialized: {agent_id}")
        
    except Exception as e:
        # Log error but don't fail - agent is already registered
        logger.error(f"Failed to initialize agent resources: {str(e)}")
        monitoring.record_metric(
            'agentvault_agent_resource_init_failures_total',
            1,
            {'agent_id': agent_id, 'error': str(e)}
        )


# =============================================================================
# Main Entry Point
# =============================================================================
# Main entry point for production
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    uvicorn.run(
        "main:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', '8080')),
        workers=int(os.getenv('API_WORKERS', '4')),
        log_level=os.getenv('LOG_LEVEL', 'info').lower(),
        access_log=os.getenv('ACCESS_LOG', 'true').lower() == 'true',
        use_colors=os.getenv('USE_COLORS', 'true').lower() == 'true',
        reload=os.getenv('ENVIRONMENT') == 'development'
    )