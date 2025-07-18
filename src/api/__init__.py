"""
AgentVault™ API Module
RESTful and GraphQL APIs for AI agent storage operations

This module provides:
- FastAPI-based REST endpoints
- GraphQL schema and resolvers
- WebSocket support for real-time updates
- OpenAPI documentation
- Authentication and authorization

Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

from .rest_api import app as rest_app
from .graphql_api import schema as graphql_schema
from .websocket import manager as ws_manager
from .auth import verify_token, create_access_token

__all__ = [
    "rest_app",
    "graphql_schema",
    "ws_manager",
    "verify_token",
    "create_access_token"
]