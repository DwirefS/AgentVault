"""
AgentVault™ Azure AD Authentication System
Production-ready authentication and authorization with Azure Active Directory
Author: Dwiref Sharma
Contact: DwirefS@SapientEdge.io
"""

import asyncio
import jwt
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from cachetools import TTLCache
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.core.credentials import AccessToken
from msal import ConfidentialClientApplication
import httpx
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class Permission(Enum):
    """AgentVault permissions aligned with Azure AD roles"""
    # Storage permissions
    STORAGE_READ = "Storage.Read"
    STORAGE_WRITE = "Storage.Write"
    STORAGE_DELETE = "Storage.Delete"
    STORAGE_ADMIN = "Storage.Admin"
    
    # Agent permissions
    AGENT_READ = "Agent.Read"
    AGENT_WRITE = "Agent.Write"
    AGENT_DELETE = "Agent.Delete"
    AGENT_ADMIN = "Agent.Admin"
    
    # ML permissions
    ML_READ = "ML.Read"
    ML_WRITE = "ML.Write"
    ML_TRAIN = "ML.Train"
    ML_ADMIN = "ML.Admin"
    
    # Security permissions
    SECURITY_READ = "Security.Read"
    SECURITY_WRITE = "Security.Write"
    SECURITY_ADMIN = "Security.Admin"
    
    # Monitoring permissions
    MONITORING_READ = "Monitoring.Read"
    MONITORING_WRITE = "Monitoring.Write"
    MONITORING_ADMIN = "Monitoring.Admin"
    
    # System permissions
    SYSTEM_ADMIN = "System.Admin"
    SYSTEM_DEBUG = "System.Debug"


class Role(Enum):
    """AgentVault roles mapped to Azure AD groups"""
    ADMIN = "AgentVault.Admin"
    DEVELOPER = "AgentVault.Developer"
    DATA_SCIENTIST = "AgentVault.DataScientist"
    OPERATOR = "AgentVault.Operator"
    VIEWER = "AgentVault.Viewer"
    SERVICE_ACCOUNT = "AgentVault.ServiceAccount"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.STORAGE_READ, Permission.STORAGE_WRITE, Permission.STORAGE_DELETE, Permission.STORAGE_ADMIN,
        Permission.AGENT_READ, Permission.AGENT_WRITE, Permission.AGENT_DELETE, Permission.AGENT_ADMIN,
        Permission.ML_READ, Permission.ML_WRITE, Permission.ML_TRAIN, Permission.ML_ADMIN,
        Permission.SECURITY_READ, Permission.SECURITY_WRITE, Permission.SECURITY_ADMIN,
        Permission.MONITORING_READ, Permission.MONITORING_WRITE, Permission.MONITORING_ADMIN,
        Permission.SYSTEM_ADMIN, Permission.SYSTEM_DEBUG
    },
    Role.DEVELOPER: {
        Permission.STORAGE_READ, Permission.STORAGE_WRITE,
        Permission.AGENT_READ, Permission.AGENT_WRITE,
        Permission.ML_READ, Permission.ML_WRITE, Permission.ML_TRAIN,
        Permission.MONITORING_READ, Permission.MONITORING_WRITE,
        Permission.SYSTEM_DEBUG
    },
    Role.DATA_SCIENTIST: {
        Permission.STORAGE_READ, Permission.STORAGE_WRITE,
        Permission.AGENT_READ, Permission.AGENT_WRITE,
        Permission.ML_READ, Permission.ML_WRITE, Permission.ML_TRAIN,
        Permission.MONITORING_READ
    },
    Role.OPERATOR: {
        Permission.STORAGE_READ,
        Permission.AGENT_READ,
        Permission.ML_READ,
        Permission.MONITORING_READ, Permission.MONITORING_WRITE,
        Permission.SYSTEM_DEBUG
    },
    Role.VIEWER: {
        Permission.STORAGE_READ,
        Permission.AGENT_READ,
        Permission.ML_READ,
        Permission.MONITORING_READ
    },
    Role.SERVICE_ACCOUNT: {
        Permission.STORAGE_READ, Permission.STORAGE_WRITE,
        Permission.AGENT_READ, Permission.AGENT_WRITE,
        Permission.ML_READ,
        Permission.MONITORING_WRITE
    }
}


@dataclass
class AzureADConfig:
    """Azure AD configuration"""
    tenant_id: str
    client_id: str
    client_secret: str
    authority: str
    api_scope: str = "api://agentvault/.default"
    graph_api_url: str = "https://graph.microsoft.com/v1.0"
    issuer: str = ""
    jwks_uri: str = ""
    validate_issuer: bool = True
    validate_audience: bool = True
    allowed_audiences: List[str] = field(default_factory=list)
    required_claims: Dict[str, Any] = field(default_factory=dict)
    cache_ttl: int = 3600  # 1 hour
    token_refresh_threshold: int = 300  # 5 minutes
    
    def __post_init__(self):
        if not self.authority:
            self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        if not self.issuer:
            self.issuer = f"https://sts.windows.net/{self.tenant_id}/"
        if not self.jwks_uri:
            self.jwks_uri = f"{self.authority}/discovery/v2.0/keys"
        if not self.allowed_audiences:
            self.allowed_audiences = [self.client_id, f"api://{self.client_id}"]


@dataclass
class AuthenticatedUser:
    """Authenticated user information"""
    user_id: str
    username: str
    email: str
    display_name: str
    roles: Set[Role]
    permissions: Set[Permission]
    groups: List[str]
    claims: Dict[str, Any]
    token: str
    expires_at: datetime
    tenant_id: str
    object_id: str
    
    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at
    
    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        return any(p in self.permissions for p in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        return all(p in self.permissions for p in permissions)
    
    def has_role(self, role: Role) -> bool:
        return role in self.roles


class AzureADAuthProvider:
    """
    Production-ready Azure AD authentication provider
    Handles token validation, user information retrieval, and permission management
    """
    
    def __init__(self, config: AzureADConfig):
        self.config = config
        
        # Initialize MSAL confidential client
        self.msal_app = ConfidentialClientApplication(
            client_id=config.client_id,
            client_credential=config.client_secret,
            authority=config.authority
        )
        
        # Initialize Azure credentials
        self.credential = ClientSecretCredential(
            tenant_id=config.tenant_id,
            client_id=config.client_id,
            client_secret=config.client_secret
        )
        
        # Caches
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_time: Optional[float] = None
        self._user_cache = TTLCache(maxsize=1000, ttl=config.cache_ttl)
        self._group_cache = TTLCache(maxsize=1000, ttl=config.cache_ttl)
        
        # HTTP client for API calls
        self._http_client: Optional[httpx.AsyncClient] = None
        
    async def initialize(self) -> None:
        """Initialize the auth provider"""
        self._http_client = httpx.AsyncClient(timeout=30.0)
        await self._refresh_jwks()
        logger.info("Azure AD Auth Provider initialized")
    
    async def close(self) -> None:
        """Cleanup resources"""
        if self._http_client:
            await self._http_client.aclose()
    
    async def authenticate_request(
        self, 
        authorization: Optional[str]
    ) -> AuthenticatedUser:
        """
        Authenticate a request with bearer token
        
        Args:
            authorization: Authorization header value
            
        Returns:
            AuthenticatedUser object
            
        Raises:
            HTTPException: If authentication fails
        """
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization header missing")
        
        # Extract bearer token
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization header format")
        
        token = parts[1]
        
        # Check cache first
        cache_key = self._get_token_hash(token)
        if cache_key in self._user_cache:
            user = self._user_cache[cache_key]
            if not user.is_expired:
                return user
        
        # Validate token
        try:
            payload = await self._validate_token(token)
            user = await self._create_authenticated_user(token, payload)
            
            # Cache the user
            self._user_cache[cache_key] = user
            
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(status_code=401, detail="Authentication failed")
    
    async def _validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token against Azure AD"""
        # Get JWKS
        jwks = await self._get_jwks()
        
        # Decode header without verification to get kid
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get('kid')
        
        if not kid:
            raise jwt.InvalidTokenError("Token missing kid header")
        
        # Find the key
        key = None
        for jwk in jwks.get('keys', []):
            if jwk.get('kid') == kid:
                key = jwk
                break
        
        if not key:
            raise jwt.InvalidTokenError(f"Unable to find key with kid: {kid}")
        
        # Convert JWK to public key
        public_key = self._jwk_to_pem(key)
        
        # Decode and validate token
        payload = jwt.decode(
            token,
            public_key,
            algorithms=['RS256'],
            audience=self.config.allowed_audiences if self.config.validate_audience else None,
            issuer=self.config.issuer if self.config.validate_issuer else None,
            options={
                'verify_signature': True,
                'verify_aud': self.config.validate_audience,
                'verify_iss': self.config.validate_issuer,
                'verify_exp': True,
                'verify_nbf': True,
                'verify_iat': True,
                'require_exp': True,
                'require_iat': True
            }
        )
        
        # Validate required claims
        for claim, expected_value in self.config.required_claims.items():
            if claim not in payload:
                raise jwt.InvalidTokenError(f"Required claim '{claim}' missing")
            if expected_value is not None and payload[claim] != expected_value:
                raise jwt.InvalidTokenError(f"Claim '{claim}' has invalid value")
        
        return payload
    
    async def _create_authenticated_user(
        self, 
        token: str, 
        payload: Dict[str, Any]
    ) -> AuthenticatedUser:
        """Create AuthenticatedUser from token payload"""
        # Extract user information
        user_id = payload.get('sub', '')
        email = payload.get('email', payload.get('preferred_username', ''))
        username = email.split('@')[0] if email else user_id
        display_name = payload.get('name', username)
        tenant_id = payload.get('tid', self.config.tenant_id)
        object_id = payload.get('oid', user_id)
        
        # Get user groups from token or Graph API
        groups = await self._get_user_groups(object_id, token)
        
        # Map groups to roles
        roles = self._map_groups_to_roles(groups)
        
        # Aggregate permissions from roles
        permissions = set()
        for role in roles:
            permissions.update(ROLE_PERMISSIONS.get(role, set()))
        
        # Add any custom permissions from claims
        if 'permissions' in payload:
            for perm_str in payload['permissions']:
                try:
                    permissions.add(Permission(perm_str))
                except ValueError:
                    logger.warning(f"Unknown permission in token: {perm_str}")
        
        # Calculate expiration
        exp_timestamp = payload.get('exp', 0)
        expires_at = datetime.fromtimestamp(exp_timestamp, timezone.utc)
        
        return AuthenticatedUser(
            user_id=user_id,
            username=username,
            email=email,
            display_name=display_name,
            roles=roles,
            permissions=permissions,
            groups=groups,
            claims=payload,
            token=token,
            expires_at=expires_at,
            tenant_id=tenant_id,
            object_id=object_id
        )
    
    async def _get_user_groups(
        self, 
        object_id: str, 
        token: str
    ) -> List[str]:
        """Get user's group memberships from Graph API"""
        cache_key = f"groups_{object_id}"
        
        # Check cache
        if cache_key in self._group_cache:
            return self._group_cache[cache_key]
        
        try:
            # Get groups from Microsoft Graph
            headers = {
                'Authorization': f'Bearer {token}',
                'Accept': 'application/json'
            }
            
            url = f"{self.config.graph_api_url}/users/{object_id}/memberOf"
            response = await self._http_client.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                groups = [
                    group.get('displayName', '') 
                    for group in data.get('value', [])
                    if group.get('@odata.type') == '#microsoft.graph.group'
                ]
                
                # Cache the result
                self._group_cache[cache_key] = groups
                
                return groups
            else:
                logger.warning(f"Failed to get user groups: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting user groups: {str(e)}")
            # Fall back to groups in token if available
            return []
    
    def _map_groups_to_roles(self, groups: List[str]) -> Set[Role]:
        """Map Azure AD groups to AgentVault roles"""
        roles = set()
        
        for group in groups:
            # Direct role mapping
            for role in Role:
                if group == role.value or group.lower() == role.value.lower():
                    roles.add(role)
                    break
            
            # Additional mappings for common group names
            group_lower = group.lower()
            if 'admin' in group_lower:
                roles.add(Role.ADMIN)
            elif 'developer' in group_lower or 'dev' in group_lower:
                roles.add(Role.DEVELOPER)
            elif 'data scientist' in group_lower or 'ml' in group_lower:
                roles.add(Role.DATA_SCIENTIST)
            elif 'operator' in group_lower or 'ops' in group_lower:
                roles.add(Role.OPERATOR)
            elif 'viewer' in group_lower or 'read' in group_lower:
                roles.add(Role.VIEWER)
        
        # Default to viewer if no roles mapped
        if not roles:
            roles.add(Role.VIEWER)
        
        return roles
    
    async def _get_jwks(self) -> Dict[str, Any]:
        """Get JSON Web Key Set from Azure AD"""
        # Check cache
        if (self._jwks_cache and self._jwks_cache_time and 
            time.time() - self._jwks_cache_time < 3600):  # Cache for 1 hour
            return self._jwks_cache
        
        # Refresh JWKS
        await self._refresh_jwks()
        return self._jwks_cache
    
    async def _refresh_jwks(self) -> None:
        """Refresh JWKS from Azure AD"""
        try:
            response = await self._http_client.get(self.config.jwks_uri)
            response.raise_for_status()
            
            self._jwks_cache = response.json()
            self._jwks_cache_time = time.time()
            
        except Exception as e:
            logger.error(f"Failed to refresh JWKS: {str(e)}")
            if not self._jwks_cache:
                raise
    
    def _jwk_to_pem(self, jwk: Dict[str, Any]) -> bytes:
        """Convert JWK to PEM format"""
        # Extract RSA components
        n = self._base64url_to_int(jwk['n'])
        e = self._base64url_to_int(jwk['e'])
        
        # Create RSA public key
        public_numbers = rsa.RSAPublicNumbers(e, n)
        public_key = public_numbers.public_key(default_backend())
        
        # Convert to PEM
        pem = public_key.public_key_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return pem
    
    @staticmethod
    def _base64url_to_int(value: str) -> int:
        """Convert base64url encoded value to integer"""
        # Add padding if needed
        padding = '=' * (4 - len(value) % 4)
        value += padding
        
        # Replace URL-safe characters
        value = value.replace('-', '+').replace('_', '/')
        
        # Decode and convert to integer
        return int.from_bytes(
            base64.b64decode(value.encode('ascii')), 
            byteorder='big'
        )
    
    @staticmethod
    def _get_token_hash(token: str) -> str:
        """Get hash of token for caching"""
        import hashlib
        return hashlib.sha256(token.encode()).hexdigest()[:16]
    
    async def get_service_token(self, scopes: Optional[List[str]] = None) -> str:
        """
        Get service-to-service token for Azure API calls
        
        Args:
            scopes: OAuth scopes (default: ["https://graph.microsoft.com/.default"])
            
        Returns:
            Access token
        """
        if not scopes:
            scopes = ["https://graph.microsoft.com/.default"]
        
        # Try to get token from MSAL cache
        result = self.msal_app.acquire_token_silent(scopes, account=None)
        
        if not result:
            # Get new token
            result = self.msal_app.acquire_token_for_client(scopes=scopes)
        
        if "access_token" in result:
            return result["access_token"]
        else:
            raise Exception(f"Failed to get service token: {result.get('error_description', 'Unknown error')}")
    
    async def validate_permissions(
        self,
        user: AuthenticatedUser,
        required_permissions: List[Permission],
        require_all: bool = True
    ) -> bool:
        """
        Validate user has required permissions
        
        Args:
            user: Authenticated user
            required_permissions: Required permissions
            require_all: Whether all permissions are required (True) or any (False)
            
        Returns:
            True if user has required permissions
        """
        if require_all:
            return user.has_all_permissions(required_permissions)
        else:
            return user.has_any_permission(required_permissions)


class AzureADAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for Azure AD authentication"""
    
    def __init__(self, app, auth_provider: AzureADAuthProvider, exclude_paths: List[str] = None):
        super().__init__(app)
        self.auth_provider = auth_provider
        self.exclude_paths = exclude_paths or ['/health', '/ready', '/metrics', '/docs', '/openapi.json']
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Get authorization header
        authorization = request.headers.get('Authorization')
        
        try:
            # Authenticate request
            user = await self.auth_provider.authenticate_request(authorization)
            
            # Add user to request state
            request.state.user = user
            
            # Add user info to headers for downstream services
            request.state.user_id = user.user_id
            request.state.user_roles = ','.join(role.value for role in user.roles)
            
            # Process request
            response = await call_next(request)
            
            # Add user info to response headers
            response.headers['X-User-ID'] = user.user_id
            
            return response
            
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail}
            )
        except Exception as e:
            logger.error(f"Authentication middleware error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )


# FastAPI dependency for authentication
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    request: Request = None
) -> AuthenticatedUser:
    """FastAPI dependency to get current authenticated user"""
    if hasattr(request.state, 'user'):
        return request.state.user
    
    # If not authenticated by middleware, authenticate here
    auth_provider = request.app.state.auth_provider
    return await auth_provider.authenticate_request(f"Bearer {credentials.credentials}")


def require_permissions(*permissions: Permission):
    """Decorator to require specific permissions"""
    def decorator(func):
        async def wrapper(
            request: Request,
            user: AuthenticatedUser = Depends(get_current_user),
            *args, 
            **kwargs
        ):
            if not user.has_all_permissions(list(permissions)):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required: {[p.value for p in permissions]}"
                )
            return await func(request, user, *args, **kwargs)
        return wrapper
    return decorator


def require_any_permission(*permissions: Permission):
    """Decorator to require any of the specified permissions"""
    def decorator(func):
        async def wrapper(
            request: Request,
            user: AuthenticatedUser = Depends(get_current_user),
            *args,
            **kwargs
        ):
            if not user.has_any_permission(list(permissions)):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required one of: {[p.value for p in permissions]}"
                )
            return await func(request, user, *args, **kwargs)
        return wrapper
    return decorator


def require_role(*roles: Role):
    """Decorator to require specific role"""
    def decorator(func):
        async def wrapper(
            request: Request,
            user: AuthenticatedUser = Depends(get_current_user),
            *args,
            **kwargs
        ):
            if not any(role in user.roles for role in roles):
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient role. Required one of: {[r.value for r in roles]}"
                )
            return await func(request, user, *args, **kwargs)
        return wrapper
    return decorator


# Production-ready configuration loader
def load_azure_ad_config() -> AzureADConfig:
    """Load Azure AD configuration from environment variables"""
    import os
    
    config = AzureADConfig(
        tenant_id=os.getenv('AZURE_AD_TENANT_ID', ''),
        client_id=os.getenv('AZURE_AD_CLIENT_ID', ''),
        client_secret=os.getenv('AZURE_AD_CLIENT_SECRET', ''),
        authority=os.getenv('AZURE_AD_AUTHORITY', ''),
        api_scope=os.getenv('AZURE_AD_API_SCOPE', 'api://agentvault/.default'),
        validate_issuer=os.getenv('AZURE_AD_VALIDATE_ISSUER', 'true').lower() == 'true',
        validate_audience=os.getenv('AZURE_AD_VALIDATE_AUDIENCE', 'true').lower() == 'true',
        cache_ttl=int(os.getenv('AZURE_AD_CACHE_TTL', '3600')),
        token_refresh_threshold=int(os.getenv('AZURE_AD_TOKEN_REFRESH_THRESHOLD', '300'))
    )
    
    # Validate required configuration
    if not all([config.tenant_id, config.client_id, config.client_secret]):
        raise ValueError("Azure AD configuration missing required values")
    
    return config