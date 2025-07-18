"""
AgentVault™ REST API
FastAPI-based REST endpoints for enterprise AI agent storage

This module provides:
- Agent registration and management
- Storage operations (read/write/query)
- Performance monitoring endpoints
- Admin operations
- Health checks and metrics

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging

from ..core.storage_orchestrator import AgentVaultOrchestrator, StorageRequest, AgentStorageProfile
from ..core.storage_orchestrator import StorageTier
from .auth import verify_token, get_current_user


# Pydantic models for API

class AgentRegistrationRequest(BaseModel):
    """Request model for agent registration"""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_type: str = Field(..., description="Type of agent (langchain, autogen, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "financial-assistant-001",
                "agent_type": "langchain",
                "config": {
                    "performance": {
                        "latency_requirement": 0.1,
                        "throughput_requirement": "high"
                    },
                    "security": {
                        "encryption_required": True,
                        "compliance_level": "financial_services"
                    }
                }
            }
        }


class StorageOperationRequest(BaseModel):
    """Request model for storage operations"""
    operation: str = Field(..., description="Operation type: read, write, query, delete")
    data_type: str = Field(..., description="Type of data: vector, memory, chat, etc.")
    data: Optional[Any] = Field(None, description="Data payload for write operations")
    query: Optional[str] = Field(None, description="Query for read/search operations")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    priority: str = Field("normal", description="Priority: critical, high, normal, low")
    
    class Config:
        json_schema_extra = {
            "example": {
                "operation": "write",
                "data_type": "vector",
                "data": [0.1, 0.2, 0.3, 0.4],
                "metadata": {
                    "source": "document_123",
                    "timestamp": "2024-01-15T10:30:00Z"
                },
                "priority": "high"
            }
        }


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics"""
    latency_ms: float
    throughput_mbps: float
    iops: int
    storage_utilization_percent: float
    active_agents: int
    total_requests_24h: int
    error_rate_percent: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "latency_ms": 0.087,
                "throughput_mbps": 1250.5,
                "iops": 125000,
                "storage_utilization_percent": 67.3,
                "active_agents": 150,
                "total_requests_24h": 2500000,
                "error_rate_percent": 0.01
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health checks"""
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="AgentVault™ API",
    description="Enterprise AI Agent Storage Platform - Where AI Agents Store Their Intelligence",
    version="1.0.0",
    contact={
        "name": "Dwiref Sharma",
        "email": "DwirefS@SapientEdge.io",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator: Optional[AgentVaultOrchestrator] = None
startup_time = datetime.utcnow()
logger = logging.getLogger("agentvault.api")


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize the AgentVault system on startup"""
    global orchestrator
    
    logger.info("Starting AgentVault™ API...")
    
    # Initialize orchestrator with configuration
    # In production, this would load from environment or config file
    config = {
        "azure": {
            "subscription_id": "your-subscription-id",
            "resource_group": "agentvault-prod-rg",
            "location": "East US 2"
        },
        "anf": {
            "account_name": "agentvault-prod-anf",
            "subnet_id": "/subscriptions/.../subnets/anf-subnet"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "password": ""
        },
        "security": {
            "key_vault_url": "https://agentvault-kv.vault.azure.net/",
            "default_encryption_level": "enhanced"
        }
    }
    
    orchestrator = AgentVaultOrchestrator(config)
    await orchestrator.initialize()
    
    logger.info("AgentVault™ API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global orchestrator
    
    logger.info("Shutting down AgentVault™ API...")
    
    if orchestrator:
        await orchestrator.shutdown()
    
    logger.info("AgentVault™ API shutdown complete")


# Health and status endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AgentVault™ - Where AI Agents Store Their Intelligence",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.utcnow() - startup_time).total_seconds()
    
    # Check component health
    components = {
        "api": "healthy",
        "orchestrator": "healthy" if orchestrator and orchestrator.is_initialized else "unhealthy",
        "storage": "healthy",  # Would check actual storage
        "cache": "healthy",    # Would check Redis
        "security": "healthy"  # Would check Key Vault
    }
    
    overall_status = "healthy" if all(s == "healthy" for s in components.values()) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version="1.0.0",
        uptime_seconds=uptime,
        components=components,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/metrics", response_model=PerformanceMetricsResponse)
async def get_metrics(current_user: dict = Depends(get_current_user)):
    """Get system performance metrics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Get real metrics from orchestrator
    # This is simplified - in production would aggregate from multiple sources
    
    return PerformanceMetricsResponse(
        latency_ms=0.087,
        throughput_mbps=1250.5,
        iops=125000,
        storage_utilization_percent=67.3,
        active_agents=len(orchestrator.agent_profiles),
        total_requests_24h=2500000,
        error_rate_percent=0.01
    )


# Agent management endpoints

@app.post("/agents/register", response_model=Dict[str, Any])
async def register_agent(
    request: AgentRegistrationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Register a new AI agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Register agent
        profile = await orchestrator.register_agent(
            agent_id=request.agent_id,
            agent_type=request.agent_type,
            config=request.config
        )
        
        # Schedule background DNA profiling
        background_tasks.add_task(
            orchestrator.agent_dna_profiler.create_profile,
            request.agent_id,
            request.agent_type,
            request.config
        )
        
        return {
            "success": True,
            "agent_id": request.agent_id,
            "profile": {
                "agent_type": profile.agent_type,
                "created_at": profile.created_at.isoformat(),
                "storage_dna": profile.storage_dna
            },
            "message": f"Agent {request.agent_id} registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to register agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent_info(
    agent_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get information about a registered agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if agent_id not in orchestrator.agent_profiles:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    profile = orchestrator.agent_profiles[agent_id]
    
    # Get DNA insights
    dna_insights = await orchestrator.agent_dna_profiler.get_dna_insights(agent_id)
    
    return {
        "agent_id": agent_id,
        "agent_type": profile.agent_type,
        "created_at": profile.created_at.isoformat(),
        "last_updated": profile.last_updated.isoformat(),
        "storage_dna": profile.storage_dna,
        "dna_insights": dna_insights,
        "access_patterns": profile.access_patterns,
        "performance_preferences": profile.performance_preferences
    }


@app.get("/agents", response_model=List[Dict[str, Any]])
async def list_agents(
    skip: int = 0,
    limit: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """List all registered agents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    agents = []
    for agent_id, profile in list(orchestrator.agent_profiles.items())[skip:skip+limit]:
        agents.append({
            "agent_id": agent_id,
            "agent_type": profile.agent_type,
            "created_at": profile.created_at.isoformat(),
            "last_updated": profile.last_updated.isoformat()
        })
    
    return agents


# Storage operation endpoints

@app.post("/storage/{agent_id}/operation", response_model=Dict[str, Any])
async def storage_operation(
    agent_id: str,
    request: StorageOperationRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Execute a storage operation for an agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if agent_id not in orchestrator.agent_profiles:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    try:
        # Create storage request
        storage_request = StorageRequest(
            agent_id=agent_id,
            operation=request.operation,
            data_type=request.data_type,
            data_size=len(str(request.data).encode()) if request.data else 0,
            priority=request.priority,
            latency_requirement=0.1 if request.priority == "critical" else 1.0,
            metadata={
                "data": request.data,
                "query": request.query,
                **request.metadata
            }
        )
        
        # Process request
        result = await orchestrator.process_storage_request(storage_request)
        
        # Track access pattern in background
        if result['success']:
            background_tasks.add_task(
                orchestrator.agent_dna_profiler.update_access_patterns,
                orchestrator.agent_profiles[agent_id],
                storage_request
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Storage operation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/storage/{agent_id}/usage", response_model=Dict[str, Any])
async def get_storage_usage(
    agent_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get storage usage statistics for an agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # This would query actual storage usage
    # Simplified for demo
    
    return {
        "agent_id": agent_id,
        "total_storage_gb": 125.5,
        "tier_breakdown": {
            "ultra": 10.2,
            "premium": 45.3,
            "standard": 70.0
        },
        "data_types": {
            "vectors": 35.5,
            "memory": 50.0,
            "chat_history": 40.0
        },
        "cost_usd_month": 25.50,
        "last_updated": datetime.utcnow().isoformat()
    }


# Performance optimization endpoints

@app.post("/optimize/{agent_id}", response_model=Dict[str, Any])
async def optimize_agent_storage(
    agent_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger storage optimization for an agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if agent_id not in orchestrator.agent_profiles:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    try:
        # Trigger optimization
        optimization_result = await orchestrator.anf_manager.optimize_storage_placement(agent_id)
        
        # Schedule DNA evolution in background
        background_tasks.add_task(
            orchestrator.agent_dna_profiler.evolve_profile,
            orchestrator.agent_profiles[agent_id]
        )
        
        return optimization_result
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/storage/tiers", response_model=Dict[str, Any])
async def get_storage_tiers(current_user: dict = Depends(get_current_user)):
    """Get information about available storage tiers"""
    tiers = {
        "ultra": {
            "name": "Ultra Performance",
            "latency_ms": 0.1,
            "iops": 450000,
            "cost_per_gb_month": 0.50,
            "use_cases": ["Vector search", "Active memory", "Real-time AI"]
        },
        "premium": {
            "name": "Premium Performance",
            "latency_ms": 1.0,
            "iops": 64000,
            "cost_per_gb_month": 0.20,
            "use_cases": ["Long-term memory", "Knowledge graphs", "Frequent access"]
        },
        "standard": {
            "name": "Standard Performance",
            "latency_ms": 10.0,
            "iops": 16000,
            "cost_per_gb_month": 0.10,
            "use_cases": ["Chat history", "Warm data", "Moderate access"]
        },
        "cool": {
            "name": "Cool Storage",
            "latency_ms": 60000.0,
            "iops": 1000,
            "cost_per_gb_month": 0.02,
            "use_cases": ["Analytics", "Reporting", "Infrequent access"]
        },
        "archive": {
            "name": "Archive Storage",
            "latency_ms": 3600000.0,
            "iops": 100,
            "cost_per_gb_month": 0.004,
            "use_cases": ["Compliance", "Long-term retention", "Backup"]
        }
    }
    
    return {"tiers": tiers}


# Admin endpoints

@app.post("/admin/rebalance", response_model=Dict[str, Any])
async def trigger_rebalance(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger global storage rebalancing"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Check admin privileges
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Trigger rebalancing
        rebalance_result = await orchestrator.cognitive_balancer.rebalance_load()
        
        return rebalance_result
        
    except Exception as e:
        logger.error(f"Rebalancing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/admin/reports/optimization", response_model=Dict[str, Any])
async def get_optimization_report(current_user: dict = Depends(get_current_user)):
    """Get comprehensive optimization report"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Check admin privileges
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        report = await orchestrator.tier_manager.get_optimization_report()
        return report
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Additional utility endpoints

@app.get("/openapi.json", include_in_schema=False)
async def get_openapi_schema():
    """Get OpenAPI schema"""
    return app.openapi()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)