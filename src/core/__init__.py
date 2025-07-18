"""
AgentVault™ Core Module
Enterprise AI Agent Storage Platform on Azure NetApp Files

This module provides the core functionality for AgentVault™, including:
- Storage orchestration and intelligent routing
- Neural memory management and optimization
- Enterprise security and compliance
- High-performance caching and retrieval

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

from .storage_orchestrator import AgentVaultOrchestrator
from .neural_memory import NeuralMemoryManager
from .performance_optimizer import PerformanceOptimizer
from .security_gateway import SecurityGateway

__version__ = "1.0.0-alpha"
__author__ = "Dwiref Sharma"
__email__ = "DwirefS@SapientEdge.io"

__all__ = [
    "AgentVaultOrchestrator",
    "NeuralMemoryManager", 
    "PerformanceOptimizer",
    "SecurityGateway"
]