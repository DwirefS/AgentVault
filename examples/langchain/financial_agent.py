"""
AgentVault™ + LangChain Integration Example
Financial Analysis Agent with Enterprise Storage

This example demonstrates how to build a sophisticated financial analysis
AI agent using LangChain and AgentVault™ for enterprise-grade storage.

Features:
- Real-time market data analysis
- Risk assessment and portfolio optimization
- Regulatory compliance (FINRA Rule 2210)
- Long-term memory for trading patterns
- Ultra-low latency vector search for market signals

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from langchain.agents import Agent, AgentType, initialize_agent
from langchain.llms import AzureOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from pydantic import BaseModel, Field

from agentvault import AgentVaultOrchestrator
from agentvault.core.storage_orchestrator import StorageRequest, StorageTier


class MarketAnalysisTool(BaseTool):
    """Custom tool for market analysis using AgentVault™ ultra-performance storage"""
    
    name = "market_analysis"
    description = "Analyze market data and trends using historical patterns"
    
    def __init__(self, orchestrator: AgentVaultOrchestrator):
        super().__init__()
        self.orchestrator = orchestrator
    
    def _run(self, query: str) -> str:
        """Run market analysis with ultra-fast data retrieval"""
        return asyncio.run(self._arun(query))
    
    async def _arun(self, query: str) -> str:
        """Asynchronous market analysis with AgentVault™"""
        
        # Create storage request for ultra-performance vector search
        storage_request = StorageRequest(
            agent_id="financial-analyst-001",
            operation="query",
            data_type="vector",
            priority="critical",
            latency_requirement=0.05,  # 50ms max latency
            metadata={
                "query": query,
                "query_type": "market_analysis",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Execute ultra-fast vector search
        result = await self.orchestrator.process_storage_request(storage_request)
        
        if result['success']:
            # Process market data from ultra-performance storage
            market_data = result['result']
            
            # Analyze trends and patterns
            analysis = self._analyze_market_patterns(market_data)
            
            return f"Market Analysis Results:\n{analysis}"
        else:
            return f"Analysis failed: {result.get('error', 'Unknown error')}"
    
    def _analyze_market_patterns(self, data: Dict[str, Any]) -> str:
        """Analyze market patterns from retrieved data"""
        
        # Simulated sophisticated market analysis
        # In production, this would use real ML models and market data
        
        analysis = {
            "trend": "bullish",
            "volatility": "moderate", 
            "risk_level": "medium",
            "recommendation": "hold",
            "confidence": 0.87,
            "key_factors": [
                "Strong earnings growth",
                "Favorable interest rate environment",
                "Geopolitical stability"
            ]
        }
        
        return json.dumps(analysis, indent=2)


class RiskAssessmentTool(BaseTool):
    """Risk assessment tool using AgentVault™ premium storage for portfolio data"""
    
    name = "risk_assessment"
    description = "Assess portfolio risk using historical performance data"
    
    def __init__(self, orchestrator: AgentVaultOrchestrator):
        super().__init__()
        self.orchestrator = orchestrator
    
    def _run(self, portfolio: str) -> str:
        """Run risk assessment"""
        return asyncio.run(self._arun(portfolio))
    
    async def _arun(self, portfolio: str) -> str:
        """Asynchronous risk assessment with premium storage"""
        
        # Create storage request for portfolio risk data
        storage_request = StorageRequest(
            agent_id="financial-analyst-001",
            operation="read",
            data_type="long_term_memory",
            priority="high",
            latency_requirement=0.5,  # 500ms acceptable for risk analysis
            metadata={
                "portfolio": portfolio,
                "analysis_type": "risk_assessment",
                "compliance_required": True
            }
        )
        
        # Retrieve portfolio data from premium storage
        result = await self.orchestrator.process_storage_request(storage_request)
        
        if result['success']:
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(result['result'])
            return f"Risk Assessment:\n{risk_metrics}"
        else:
            return f"Risk assessment failed: {result.get('error', 'Unknown error')}"
    
    def _calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> str:
        """Calculate comprehensive risk metrics"""
        
        risk_metrics = {
            "var_95": "2.3%",  # Value at Risk (95% confidence)
            "sharpe_ratio": 1.87,
            "beta": 1.12,
            "max_drawdown": "8.7%",
            "correlation_sp500": 0.73,
            "risk_rating": "Moderate",
            "recommendations": [
                "Consider diversification in emerging markets",
                "Reduce exposure to tech sector (overweight)",
                "Add defensive positions for volatility protection"
            ]
        }
        
        return json.dumps(risk_metrics, indent=2)


class FinancialAgent:
    """
    Sophisticated Financial Analysis Agent using AgentVault™
    
    This agent demonstrates enterprise-grade AI agent deployment with:
    - Ultra-performance storage for real-time market analysis
    - Premium storage for portfolio and risk data
    - Standard storage for chat history and compliance logs
    - Complete audit trail for regulatory compliance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize AgentVault™ orchestrator
        self.orchestrator = None
        self.agent_id = "financial-analyst-001"
        
        # LangChain components
        self.llm = None
        self.memory = None
        self.tools = []
        self.agent = None
        
        # Performance tracking
        self.query_count = 0
        self.total_latency = 0.0
    
    async def initialize(self) -> None:
        """Initialize the financial agent with AgentVault™ enterprise storage"""
        
        try:
            self.logger.info("Initializing Financial Analysis Agent with AgentVault™...")
            
            # Initialize AgentVault™ orchestrator
            self.orchestrator = AgentVaultOrchestrator(self.config)
            await self.orchestrator.initialize()
            
            # Register agent with specific requirements
            agent_profile = await self.orchestrator.register_agent(
                agent_id=self.agent_id,
                agent_type="langchain",
                config={
                    "performance": {
                        "latency_requirement": 0.1,  # 100ms for critical operations
                        "throughput_requirement": "high"
                    },
                    "security": {
                        "encryption_required": True,
                        "compliance_level": "financial_services",
                        "audit_enabled": True
                    },
                    "storage": {
                        "vector_storage": {"tier": "ultra", "size_gb": 500},
                        "memory_storage": {"tier": "premium", "size_gb": 1000},
                        "chat_storage": {"tier": "standard", "size_gb": 2000}
                    }
                }
            )
            
            # Initialize Azure OpenAI
            self.llm = AzureOpenAI(
                deployment_name=self.config['azure_openai']['deployment_name'],
                model_name="gpt-4",
                openai_api_base=self.config['azure_openai']['endpoint'],
                openai_api_key=self.config['azure_openai']['api_key'],
                openai_api_version="2023-12-01-preview",
                temperature=0.1  # Low temperature for financial analysis
            )
            
            # Initialize memory with AgentVault™ persistence
            self.memory = AgentVaultMemory(
                orchestrator=self.orchestrator,
                agent_id=self.agent_id,
                k=10  # Remember last 10 interactions
            )
            
            # Initialize tools
            self.tools = [
                MarketAnalysisTool(self.orchestrator),
                RiskAssessmentTool(self.orchestrator)
            ]
            
            # Create LangChain agent
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
            
            self.logger.info("Financial Analysis Agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            raise
    
    async def analyze_market(self, query: str) -> Dict[str, Any]:
        """Perform market analysis with ultra-low latency"""
        
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Performing market analysis: {query}")
            
            # Execute analysis through LangChain agent
            response = await self.agent.arun(input=query)
            
            # Calculate performance metrics
            latency = (datetime.utcnow() - start_time).total_seconds()
            self.query_count += 1
            self.total_latency += latency
            
            # Store analysis result for future reference
            await self._store_analysis_result(query, response, latency)
            
            return {
                "analysis": response,
                "latency_ms": latency * 1000,
                "timestamp": start_time.isoformat(),
                "agent_id": self.agent_id,
                "compliance_logged": True
            }
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            return {
                "error": str(e),
                "timestamp": start_time.isoformat(),
                "agent_id": self.agent_id
            }
    
    async def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio risk with comprehensive analysis"""
        
        start_time = datetime.utcnow()
        
        try:
            # Format portfolio data for analysis
            portfolio_summary = json.dumps(portfolio_data, indent=2)
            query = f"Assess the risk of this portfolio: {portfolio_summary}"
            
            # Execute risk assessment
            response = await self.agent.arun(input=query)
            
            # Calculate metrics
            latency = (datetime.utcnow() - start_time).total_seconds()
            
            # Store for compliance
            await self._store_risk_assessment(portfolio_data, response, latency)
            
            return {
                "risk_assessment": response,
                "latency_ms": latency * 1000,
                "timestamp": start_time.isoformat(),
                "portfolio_analyzed": len(portfolio_data.get('positions', [])),
                "compliance_logged": True
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment failed: {e}")
            return {"error": str(e)}
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        
        avg_latency = self.total_latency / self.query_count if self.query_count > 0 else 0
        
        return {
            "queries_processed": self.query_count,
            "average_latency_ms": avg_latency * 1000,
            "total_runtime_hours": self.total_latency / 3600,
            "agent_efficiency": "excellent" if avg_latency < 0.1 else "good",
            "storage_utilization": await self._get_storage_metrics()
        }
    
    async def _store_analysis_result(self, query: str, response: str, latency: float) -> None:
        """Store analysis result for compliance and future reference"""
        
        storage_request = StorageRequest(
            agent_id=self.agent_id,
            operation="write",
            data_type="activity_log",
            priority="normal",
            metadata={
                "activity_type": "market_analysis",
                "query": query,
                "response": response,
                "latency_ms": latency * 1000,
                "timestamp": datetime.utcnow().isoformat(),
                "compliance_tags": ["FINRA_Rule_2210", "SOC2", "audit_trail"]
            }
        )
        
        await self.orchestrator.process_storage_request(storage_request)
    
    async def _store_risk_assessment(self, portfolio: Dict[str, Any], 
                                   assessment: str, latency: float) -> None:
        """Store risk assessment for compliance"""
        
        storage_request = StorageRequest(
            agent_id=self.agent_id,
            operation="write",
            data_type="activity_log",
            priority="high",
            metadata={
                "activity_type": "risk_assessment",
                "portfolio_hash": hash(str(portfolio)),
                "assessment": assessment,
                "latency_ms": latency * 1000,
                "timestamp": datetime.utcnow().isoformat(),
                "compliance_tags": ["SEC_regulations", "risk_management", "audit_trail"]
            }
        )
        
        await self.orchestrator.process_storage_request(storage_request)
    
    async def _get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage utilization metrics"""
        
        try:
            metrics = await self.orchestrator.anf_manager.get_performance_metrics()
            return {
                "ultra_tier_utilization": metrics.get("vector-store", {}).get("utilization_percent", 0),
                "premium_tier_utilization": metrics.get("memory-cache", {}).get("utilization_percent", 0),
                "standard_tier_utilization": metrics.get("chat-history", {}).get("utilization_percent", 0)
            }
        except Exception as e:
            self.logger.warning(f"Could not retrieve storage metrics: {e}")
            return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for financial agent"""
        
        logger = logging.getLogger("agentvault.financial_agent")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger


class AgentVaultMemory(ConversationBufferWindowMemory):
    """Custom LangChain memory class integrated with AgentVault™"""
    
    def __init__(self, orchestrator: AgentVaultOrchestrator, agent_id: str, k: int = 5):
        super().__init__(k=k)
        self.orchestrator = orchestrator
        self.agent_id = agent_id
    
    async def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save conversation context to AgentVault™ premium storage"""
        
        # Save to LangChain memory
        super().save_context(inputs, outputs)
        
        # Also persist to AgentVault™ for enterprise features
        storage_request = StorageRequest(
            agent_id=self.agent_id,
            operation="write",
            data_type="chat_history",
            priority="normal",
            metadata={
                "conversation_turn": {
                    "inputs": inputs,
                    "outputs": outputs,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        )
        
        await self.orchestrator.process_storage_request(storage_request)


# Example usage
async def main():
    """Example usage of the Financial Analysis Agent"""
    
    # Configuration
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
        "azure_openai": {
            "endpoint": "https://your-openai.openai.azure.com/",
            "api_key": "your-api-key",
            "deployment_name": "gpt-4"
        },
        "redis": {
            "host": "agentvault-prod-redis.redis.cache.windows.net",
            "port": 6380,
            "password": "your-redis-password"
        }
    }
    
    # Initialize financial agent
    agent = FinancialAgent(config)
    await agent.initialize()
    
    # Perform market analysis
    analysis_result = await agent.analyze_market(
        "Analyze the current market trends for technology stocks and provide investment recommendations"
    )
    print("Market Analysis Result:")
    print(json.dumps(analysis_result, indent=2))
    
    # Assess portfolio risk
    portfolio = {
        "positions": [
            {"symbol": "AAPL", "shares": 100, "cost_basis": 150.00},
            {"symbol": "MSFT", "shares": 50, "cost_basis": 280.00},
            {"symbol": "GOOGL", "shares": 25, "cost_basis": 2500.00}
        ],
        "total_value": 210000,
        "cash": 15000
    }
    
    risk_result = await agent.assess_portfolio_risk(portfolio)
    print("\\nRisk Assessment Result:")
    print(json.dumps(risk_result, indent=2))
    
    # Get performance metrics
    performance = await agent.get_performance_metrics()
    print("\\nAgent Performance Metrics:")
    print(json.dumps(performance, indent=2))


if __name__ == "__main__":
    asyncio.run(main())