# Supabase JWT validation
# services/auth_service.py
"""
Authentication service integrating with Supabase JWT.
Provides user validation and permission checking.
"""

import jwt
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx
import structlog
from fastapi import HTTPException, status
from utils.config import settings


logger = structlog.get_logger()


class AuthService:
    """
    Handles JWT validation and user authentication.
    Integrates with Supabase for production, allows bypass for demos.
    """
    
    def __init__(self):
        self.jwt_secret = settings.jwt_secret
        self.jwt_algorithm = settings.jwt_algorithm
        self.supabase_url = settings.supabase_url
        self.supabase_anon_key = settings.supabase_anon_key
        self.logger = logger.bind(service="AuthService")
    
    async def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verifies JWT token and returns user data.
        Supports both Supabase and custom tokens.
        """
        
        # For demo/development purposes, allow a bypass token
        if token == "demo-token" and settings.debug:
            self.logger.info("Using demo token bypass")
            return {
                "id": "demo-user",
                "email": "demo@example.com",
                "role": "user",
                "permissions": ["read", "write"]
            }
        
        try:
            # Decode token
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={"verify_exp": True, "verify_aud": False}
            )
            
            # Extract user info
            user_data = {
                "id": payload.get("sub", payload.get("user_id")),
                "email": payload.get("email"),
                "role": payload.get("role", "user"),
                "permissions": payload.get("permissions", [])
            }
            
            # Ensure user ID is present
            if not user_data["id"]:
                self.logger.warning("Token missing user ID")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing user ID"
                )
            
            self.logger.info("Token verified successfully", user_id=user_data["id"])
            
            return user_data
            
        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            self.logger.warning("Invalid token", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        except Exception as e:
            self.logger.error("Token verification failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    async def verify_supabase_token(self, token: str) -> Dict[str, Any]:
        """
        Verifies token with Supabase service.
        Used when Supabase is configured.
        """
        
        if not self.supabase_url:
            # Fallback to local verification
            return await self.verify_token(token)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.supabase_url}/auth/v1/user",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "apikey": self.supabase_anon_key
                    }
                )
                
                if response.status_code == 200:
                    user_data = response.json()
                    return {
                        "id": user_data.get("id"),
                        "email": user_data.get("email"),
                        "role": user_data.get("role", "user"),
                        "permissions": []
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid token"
                    )
                    
        except httpx.RequestError as e:
            self.logger.error("Supabase verification failed", error=str(e))
            # Fallback to local verification
            return await self.verify_token(token)
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """
        Creates JWT token for testing/demo purposes.
        Not used in production with Supabase.
        """
        
        payload = {
            "sub": user_data["id"],
            "email": user_data.get("email"),
            "role": user_data.get("role", "user"),
            "permissions": user_data.get("permissions", []),
            "exp": datetime.utcnow() + timedelta(minutes=settings.jwt_expiry_minutes),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(
            payload,
            self.jwt_secret,
            algorithm=self.jwt_algorithm
        )
        
        return token
    
    async def has_permission(self, user_data: Dict[str, Any], required_permission: str) -> bool:
        """
        Checks if user has required permission.
        Implements role-based access control.
        """
        
        # Admin has all permissions
        if user_data.get("role") == "admin":
            return True
        
        # Check specific permissions
        permissions = user_data.get("permissions", [])
        return required_permission in permissions
    
    async def get_user_datasets(self, user_id: str) -> List[str]:
        """
        Gets list of datasets user has access to.
        In production, would query permissions database.
        """
        
        # For demo, return all datasets
        # In production, this would query a permissions table
        return ["*"]  # Wildcard for all datasets
    
    async def validate_dataset_access(
        self,
        user_id: str,
        dataset_id: str
    ) -> bool:
        """
        Validates user has access to specific dataset.
        Critical for multi-tenant security.
        """
        
        allowed_datasets = await self.get_user_datasets(user_id)
        
        # Check wildcard
        if "*" in allowed_datasets:
            return True
        
        return dataset_id in allowed_datasets