# Prometheus instrumentation
# middleware/metrics.py
"""Prometheus metrics instrumentation middleware"""

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from typing import Callable
import structlog


logger = structlog.get_logger()


# Define metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    ["method", "endpoint"]
)

ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Active HTTP requests"
)

QUERY_COUNT = Counter(
    "nlp_queries_total",
    "Total NLP queries processed",
    ["dataset", "status"]
)

QUERY_DURATION = Histogram(
    "nlp_query_duration_seconds",
    "NLP query processing duration",
    ["dataset"]
)

LLM_REQUESTS = Counter(
    "llm_requests_total",
    "Total LLM API requests",
    ["model", "status"]
)

CACHE_OPERATIONS = Counter(
    "cache_operations_total",
    "Cache operations",
    ["operation", "status"]
)

APP_INFO = Info(
    "app_info",
    "Application information"
)

# Set app info
APP_INFO.info({
    "version": "1.0.0",
    "name": "llm-data-analyst"
})


class MetricsMiddleware:
    """
    Prometheus metrics collection middleware.
    Tracks key performance indicators for monitoring.
    """
    
    def __init__(self, app=None):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Collect metrics for each request"""
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
            
        request = Request(scope, receive=receive, send=send)
        
        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics":
            await self.app(scope, receive, send)
            return
        
        # Track active requests
        ACTIVE_REQUESTS.inc()
        
        # Start timer
        start_time = time.time()
        
        # Normalize endpoint for metrics
        endpoint = self._normalize_endpoint(request.url.path)
        
        # Define a wrapper for the send function to capture the response
        response_status = {"status_code": 200}  # Default value
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response_status["status_code"] = message["status"]
            await send(message)
        
        try:
            # Process request
            await self.app(scope, receive, send_wrapper)
            
            # Record metrics
            duration = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=response_status["status_code"]
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time
            
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=endpoint,
                status=500
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(duration)
            
            raise
            
        finally:
            # Decrement active requests
            ACTIVE_REQUESTS.dec()
    
    def _normalize_endpoint(self, path: str) -> str:
        """
        Normalize endpoint path for metrics.
        Replaces dynamic segments with placeholders.
        """
        
        # Replace UUIDs
        import re
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}',
            path
        )
        
        # Replace dataset IDs
        path = re.sub(r'/ds_[a-z0-9]+', '/{dataset_id}', path)
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path


class MetricsCollector:
    """
    Application-specific metrics collector.
    Tracks business metrics beyond HTTP.
    """
    
    @staticmethod
    def record_query(dataset_id: str, success: bool, duration: float):
        """Record NLP query metrics"""
        
        QUERY_COUNT.labels(
            dataset=dataset_id,
            status="success" if success else "failure"
        ).inc()
        
        if success:
            QUERY_DURATION.labels(dataset=dataset_id).observe(duration)
    
    @staticmethod
    def record_llm_request(model: str, success: bool):
        """Record LLM API request metrics"""
        
        LLM_REQUESTS.labels(
            model=model,
            status="success" if success else "failure"
        ).inc()
    
    @staticmethod
    def record_cache_operation(operation: str, hit: bool):
        """Record cache operation metrics"""
        
        CACHE_OPERATIONS.labels(
            operation=operation,
            status="hit" if hit else "miss"
        ).inc()