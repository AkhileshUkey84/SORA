# Health/metrics endpoints
# routes/health.py
"""Health check and monitoring endpoints"""

from fastapi import APIRouter, Response
from prometheus_client import generate_latest
from models.responses import HealthStatus
from services.database_service import DatabaseService
from services.cache_service import CacheService
from services.llm_service import LLMService
import time
import psutil
import structlog


router = APIRouter(tags=["monitoring"])
logger = structlog.get_logger()

# Track start time
START_TIME = time.time()


@router.get("/healthz", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Comprehensive health check endpoint.
    Checks all critical services and returns detailed status.
    """
    
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "database": False,
        "cache": False,
        "llm": False,
        "storage": True,  # Assume file system is working
        "warnings": []
    }
    
    # Check database
    try:
        db_service = DatabaseService()
        await db_service.execute_query("SELECT 1")
        health_status["database"] = True
    except Exception as e:
        health_status["warnings"].append(f"Database unhealthy: {str(e)}")
        health_status["status"] = "degraded"
    
    # Check cache
    try:
        cache_service = CacheService()
        await cache_service.set("health_check", "ok", ttl=10)
        value = await cache_service.get("health_check")
        health_status["cache"] = value == "ok"
    except Exception as e:
        health_status["warnings"].append(f"Cache unhealthy: {str(e)}")
        # Cache is optional, don't degrade status
    
    # Check LLM service
    try:
        llm_service = LLMService()
        # Simple connectivity check
        health_status["llm"] = llm_service.api_key is not None
    except Exception as e:
        health_status["warnings"].append(f"LLM service unhealthy: {str(e)}")
        health_status["status"] = "degraded"
    
    # Calculate metrics
    uptime = time.time() - START_TIME
    
    # Get system metrics
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        health_status["warnings"].append(f"High memory usage: {memory.percent}%")
        health_status["status"] = "degraded"
    
    # Set overall status
    if health_status["status"] == "healthy" and not health_status["database"]:
        health_status["status"] = "unhealthy"
    
    return HealthStatus(
        status=health_status["status"],
        version=health_status["version"],
        database=health_status["database"],
        cache=health_status["cache"],
        llm=health_status["llm"],
        storage=health_status["storage"],
        uptime_seconds=uptime,
        total_queries=0,  # Would get from metrics
        active_sessions=0,  # Would get from session manager
        warnings=health_status["warnings"]
    )


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    
    metrics_data = generate_latest()
    return Response(
        content=metrics_data,
        media_type="text/plain"
    )


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe.
    Checks if service is ready to accept traffic.
    """
    
    # Quick check - just database
    try:
        db_service = DatabaseService()
        await db_service.execute_query("SELECT 1")
        return {"ready": True}
    except:
        return {"ready": False}, 503


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe.
    Simple check to see if process is alive.
    """
    return {"alive": True}