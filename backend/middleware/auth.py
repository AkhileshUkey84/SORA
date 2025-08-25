# JWT authentication
# middleware/auth.py
"""JWT authentication middleware"""

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import structlog
from services.auth_service import AuthService


logger = structlog.get_logger()
security = HTTPBearer()


class AuthMiddleware:
    """
    JWT authentication middleware.
    Validates tokens and enriches requests with user context.
    """
    
    def __init__(self, app=None):
        self.app = app
        self.auth_service = None
    
    async def __call__(self, scope, receive, send):
        """Process request with authentication"""
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        request = Request(scope, receive=receive, send=send)
        
        # Skip auth for health endpoints and docs
        if request.url.path in ["/healthz", "/metrics", "/", "/docs", "/redoc", "/openapi.json"]:
            await self.app(scope, receive, send)
            return
        
        # Initialize auth service if not already done
        if not self.auth_service:
            if hasattr(request.app.state, 'auth_service'):
                self.auth_service = request.app.state.auth_service
            else:
                # Create auth service instance
                self.auth_service = AuthService()
                logger.info("Created new auth service instance")
        
        # Extract token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            # For demo purposes, allow unauthenticated access to certain endpoints
            if request.url.path.startswith("/api/v1/upload") or request.url.path.startswith("/api/v1/query"):
                # Create a demo user for upload/query endpoints
                request.state.user = {
                    "id": "demo-user",
                    "email": "demo@example.com",
                    "role": "user",
                    "permissions": ["read", "write"]
                }
                logger.info("Using demo user for endpoint", path=request.url.path)
                await self.app(scope, receive, send)
                return
            
            # For other endpoints, continue without authentication
            await self.app(scope, receive, send)
            return
        
        token = auth_header.split(" ")[1]
        
        try:
            # Verify token
            user_data = await self.auth_service.verify_token(token)
            
            # Add to request state
            request.state.user = user_data
            
            # Log authenticated request
            logger.info("Authenticated request",
                       user_id=user_data["id"],
                       path=request.url.path)
            
            # Create a send wrapper to pass user data to downstream middleware
            async def send_wrapper(message):
                await send(message)
            
            # Call next middleware with authenticated request
            await self.app(scope, receive, send_wrapper)
            
        except HTTPException as http_ex:
            # Handle HTTP exceptions from auth service
            logger.warning("Authentication HTTP exception", 
                          status_code=http_ex.status_code, 
                          detail=http_ex.detail)
            
            # For demo endpoints, use demo user as fallback
            if request.url.path.startswith("/api/v1/upload") or request.url.path.startswith("/api/v1/query"):
                request.state.user = {
                    "id": "demo-user",
                    "email": "demo@example.com",
                    "role": "user",
                    "permissions": ["read", "write"]
                }
                logger.info("Using demo user after auth failure", path=request.url.path)
                await self.app(scope, receive, send)
                return
            
            raise
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            
            # For demo endpoints, use demo user as fallback
            if request.url.path.startswith("/api/v1/upload") or request.url.path.startswith("/api/v1/query"):
                request.state.user = {
                    "id": "demo-user",
                    "email": "demo@example.com",
                    "role": "user",
                    "permissions": ["read", "write"]
                }
                logger.info("Using demo user after auth error", path=request.url.path)
                await self.app(scope, receive, send)
                return
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Dependency to get current authenticated user.
    Used in route handlers.
    """
    
    # Get auth service from app state if available
    auth_service = None
    if hasattr(request.app.state, 'auth_service'):
        auth_service = request.app.state.auth_service
    else:
        # Initialize auth service as fallback
        auth_service = AuthService()
    
    try:
        # Verify token
        user_data = await auth_service.verify_token(credentials.credentials)
        return user_data
        
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


async def require_permission(permission: str):
    """
    Dependency to require specific permission.
    Used for role-based access control.
    """
    
    async def permission_checker(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ):
        auth_service = AuthService()
        
        has_permission = await auth_service.check_permission(
            current_user, permission
        )
        
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        
        return current_user
    
    return permission_checker