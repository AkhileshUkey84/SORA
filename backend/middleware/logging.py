# Structured logging
# middleware/logging.py
"""Structured logging middleware with request tracing"""

from fastapi import Request, Response
import time
import uuid
import structlog
from typing import Callable
import json


logger = structlog.get_logger()


class LoggingMiddleware:
    """
    Structured logging middleware for comprehensive request tracking.
    Essential for debugging and monitoring in production.
    """
    
    def __init__(self, app=None):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Log request details and response metrics"""
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        request = Request(scope, receive=receive, send=send)
        
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())
        
        # Add to request state
        request.state.request_id = request_id
        
        # Bind request context to logger
        request_logger = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None
        )
        
        # Log request start
        start_time = time.time()
        
        # Extract user if authenticated
        user_id = None
        if hasattr(request.state, "user") and request.state.user:
            user_id = request.state.user.get("id")
            request_logger = request_logger.bind(user_id=user_id)
        
        # Log request details
        request_logger.info(
            "Request started",
            query_params=dict(request.query_params),
            headers={
                k: v for k, v in request.headers.items()
                if k.lower() not in ["authorization", "cookie"]
            }
        )
        
        # Define a wrapper for the send function to capture the response
        response_status = {"status_code": 200, "headers": {}}
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_status["status_code"] = message["status"]
                response_status["headers"] = {k.decode(): v.decode() for k, v in message.get("headers", [])}
                
                # Add request ID to response headers
                headers = message.get("headers", [])
                headers.append((b"X-Request-ID", request_id.encode()))
                message["headers"] = headers
                
            await send(message)
        
        # Process request
        try:
            await self.app(scope, receive, send_wrapper)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log response
            request_logger.info(
                "Request completed",
                status_code=response_status["status_code"],
                duration_ms=round(duration_ms, 2),
                response_headers=response_status["headers"]
            )
            
        except Exception as e:
            # Log exception with enhanced details
            duration_ms = (time.time() - start_time) * 1000
            
            request_logger.error(
                "Request failed",
                exception=str(e),
                exception_type=type(e).__name__,
                duration_ms=round(duration_ms, 2),
                status_code=500,  # Default error status
                exc_info=True,
                traceback=True    # Include full traceback
            )
            
            # Re-raise exception for FastAPI's exception handlers
            raise


class StructuredLogger:
    """
    Structured logger configuration for the application.
    Ensures consistent logging format across all components.
    """
    
    @staticmethod
    def configure():
        """Configure structured logging"""
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                add_app_context,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


def add_app_context(logger, method_name, event_dict):
    """Add application context to all log entries"""
    
    # Add application info
    event_dict["app"] = "llm-data-analyst"
    event_dict["version"] = "1.0.0"
    event_dict["environment"] = "production"  # From env var
    
    return event_dict