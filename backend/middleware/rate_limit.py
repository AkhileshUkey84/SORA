# Rate limiting
# middleware/rate_limit.py
"""Rate limiting middleware using Redis/in-memory fallback"""

from fastapi import Request, HTTPException, status
from typing import Optional, Dict, Any
import time
import structlog
from collections import defaultdict
from datetime import datetime, timedelta
from services.cache_service import CacheService
from utils.config import settings


logger = structlog.get_logger()


class RateLimiter:
    """
    Token bucket rate limiter implementation.
    Falls back to in-memory when Redis unavailable.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        self.cache = cache_service
        self.requests_per_window = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_window_seconds
        
        # In-memory fallback
        self.memory_buckets = defaultdict(lambda: {
            "tokens": self.requests_per_window,
            "last_update": time.time()
        })
    
    async def check_rate_limit(self, identifier: str) -> Dict[str, Any]:
        """
        Check if request is within rate limits.
        Returns limit info and remaining tokens.
        """
        
        if self.cache and self.cache.is_redis_available:
            return await self._check_redis_limit(identifier)
        else:
            return self._check_memory_limit(identifier)
    
    async def _check_redis_limit(self, identifier: str) -> Dict[str, Any]:
        """Check rate limit using Redis"""
        
        key = f"rate_limit:{identifier}"
        current_time = int(time.time())
        window_start = current_time - self.window_seconds
        
        try:
            # Remove old entries
            await self.cache.redis_client.zremrangebyscore(
                key, 0, window_start
            )
            
            # Count requests in window
            request_count = await self.cache.redis_client.zcard(key)
            
            if request_count >= self.requests_per_window:
                # Get oldest request time
                oldest = await self.cache.redis_client.zrange(
                    key, 0, 0, withscores=True
                )
                
                if oldest:
                    reset_time = int(oldest[0][1]) + self.window_seconds
                else:
                    reset_time = current_time + self.window_seconds
                
                return {
                    "allowed": False,
                    "limit": self.requests_per_window,
                    "remaining": 0,
                    "reset": reset_time
                }
            
            # Add current request
            await self.cache.redis_client.zadd(
                key, {str(current_time): current_time}
            )
            
            # Set expiry
            await self.cache.redis_client.expire(key, self.window_seconds)
            
            return {
                "allowed": True,
                "limit": self.requests_per_window,
                "remaining": self.requests_per_window - request_count - 1,
                "reset": current_time + self.window_seconds
            }
            
        except Exception as e:
            logger.warning("Redis rate limit check failed", error=str(e))
            # Fall back to memory
            return self._check_memory_limit(identifier)
    
    def _check_memory_limit(self, identifier: str) -> Dict[str, Any]:
        """Check rate limit using in-memory storage"""
        
        current_time = time.time()
        bucket = self.memory_buckets[identifier]
        
        # Refill tokens based on time passed
        time_passed = current_time - bucket["last_update"]
        tokens_to_add = (time_passed / self.window_seconds) * self.requests_per_window
        
        bucket["tokens"] = min(
            self.requests_per_window,
            bucket["tokens"] + tokens_to_add
        )
        bucket["last_update"] = current_time
        
        if bucket["tokens"] < 1:
            return {
                "allowed": False,
                "limit": self.requests_per_window,
                "remaining": 0,
                "reset": int(current_time + self.window_seconds)
            }
        
        # Consume a token
        bucket["tokens"] -= 1
        
        return {
            "allowed": True,
            "limit": self.requests_per_window,
            "remaining": int(bucket["tokens"]),
            "reset": int(current_time + self.window_seconds)
        }


class RateLimitMiddleware:
    """
    Rate limiting middleware for FastAPI.
    Protects API from abuse and ensures fair usage.
    """
    
    def __init__(self, app=None, cache_service: Optional[CacheService] = None):
        self.app = app
        self.limiter = RateLimiter(cache_service)
    
    async def __call__(self, scope, receive, send):
        """Process request with rate limiting"""
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        request = Request(scope, receive=receive, send=send)
        
        # Skip rate limiting for excluded paths
        if request.url.path in ["/healthz", "/metrics", "/", "/docs", "/redoc", "/openapi.json"]:
            await self.app(scope, receive, send)
            return
            
        # Get cache service from app state if not already set
        if not self.limiter.cache and hasattr(request.app.state, 'cache_service'):
            self.limiter.cache = request.app.state.cache_service
        
        # Get identifier (user ID or IP)
        identifier = self._get_identifier(request)
        
        # Check rate limit
        limit_info = await self.limiter.check_rate_limit(identifier)
        
        if not limit_info["allowed"]:
            logger.warning("Rate limit exceeded", identifier=identifier)
            
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(limit_info["reset"]),
                    "Retry-After": str(limit_info["reset"] - int(time.time()))
                }
            )
        
        # Process request
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Add rate limit headers
                headers = message.get("headers", [])
                headers.extend([
                    (b"X-RateLimit-Limit", str(limit_info["limit"]).encode()),
                    (b"X-RateLimit-Remaining", str(limit_info["remaining"]).encode()),
                    (b"X-RateLimit-Reset", str(limit_info["reset"]).encode())
                ])
                message["headers"] = headers
            await send(message)
            
        # Call the next middleware with the wrapped send function
        try:
            await self.app(scope, receive, send_wrapper)
        except TypeError as e:
            # Handle potential signature mismatch
            if "__call__() takes 3 positional arguments but 4 were given" in str(e):
                # Try alternative approach for middleware compatibility
                await self.app(scope)(receive, send_wrapper)
            else:
                raise
    
    def _get_identifier(self, request: Request) -> str:
        """Get unique identifier for rate limiting"""
        
        # Prefer authenticated user ID
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user['id']}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for proxy headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"


# Dependency for route-level rate limiting
async def rate_limit_check(request: Request):
    """
    Dependency to check rate limits.
    Can be used selectively on specific endpoints.
    """
    
    # Get cache service from app state if available
    cache_service = None
    if hasattr(request.app.state, 'cache_service'):
        cache_service = request.app.state.cache_service
    else:
        # Initialize services as fallback
        cache_service = CacheService()
        await cache_service.initialize()
    
    limiter = RateLimiter(cache_service)
    
    # Get identifier
    identifier = f"ip:{request.client.host}" if request.client else "unknown"
    
    if hasattr(request.state, "user") and request.state.user:
        identifier = f"user:{request.state.user['id']}"
    
    # Check limit
    limit_info = await limiter.check_rate_limit(identifier)
    
    if not limit_info["allowed"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(limit_info["reset"] - int(time.time()))
            }
        )