# Health check routes
from fastapi import APIRouter
from app.config import settings
import redis
import duckdb

router = APIRouter()

@router.get("")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "LLM Data Analyst Assistant",
        "version": settings.API_VERSION
    }

@router.get("/ready")
async def readiness_check():
    """Check if all dependencies are ready"""
    checks = {
        "api": True,
        "redis": False,
        "duckdb": False,
        "gemini": False
    }
    
    # Check Redis
    try:
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        checks["redis"] = True
    except:
        pass
    
    # Check DuckDB
    try:
        conn = duckdb.connect(":memory:")
        conn.execute("SELECT 1").fetchone()
        conn.close()
        checks["duckdb"] = True
    except:
        pass
    
    # Check Gemini API
    try:
        # In production, make actual API call
        checks["gemini"] = bool(settings.GEMINI_API_KEY)
    except:
        pass
    
    all_ready = all(checks.values())
    
    return {
        "ready": all_ready,
        "checks": checks
    }