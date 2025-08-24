# Rate limiting middleware
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import time
from collections import defaultdict
from typing import Dict
from app.config import settings

class RateLimitMiddleware:
    """Rate limiting middleware"""
    
    def __init__(self, app):
        self.app = app
        self.requests: Dict[str, list] = defaultdict(list)
        self.max_requests = settings.MAX_QUERIES_PER_MINUTE
        
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Skip rate limiting for health checks
        if scope["path"].startswith("/api/health"):
            await self.app(scope, receive, send)
            return
        
        # Get client identifier (IP or user ID)
        client_id = scope.get("client", ("unknown", 0))[0] if scope.get("client") else "unknown"
        
        # Clean old requests
        current_time = time.time()
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.max_requests:
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": f'{{"detail": "Rate limit exceeded. Maximum {self.max_requests} requests per minute."}}'.encode()
            })
            return
        
        # Record request
        self.requests[client_id].append(current_time)
        
        # Process request
        await self.app(scope, receive, send)
