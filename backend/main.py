# FastAPI application entry
# main.py
"""
FastAPI application entry point with comprehensive middleware setup.
Optimized for demo reliability and monitoring.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time
import os
from datetime import datetime

# Import middleware
from middleware.logging import LoggingMiddleware, StructuredLogger
from middleware.metrics import MetricsMiddleware
from middleware.auth import AuthMiddleware
from middleware.rate_limit import RateLimitMiddleware

# Import routes
from routes import query_router, upload_router, health_router, session_router

# Import services
from services.database_service import DatabaseService
from services.cache_service import CacheService
from services.auth_service import AuthService
from services.agent_orchestrator import AgentOrchestrator

# Import configuration
from utils.config import settings

# Configure structured logging
StructuredLogger.configure()
logger = structlog.get_logger()

# Global instances for services
db_service = None
cache_service = None
auth_service = None
orchestrator = None


async def load_datasets(db_service: DatabaseService) -> int:
    """Load all available datasets from uploads directory into the database"""
    
    from pathlib import Path
    import pandas as pd
    
    uploads_dir = Path("./uploads")
    if not uploads_dir.exists():
        logger.warning("Uploads directory not found, no datasets to load")
        return 0
    
    parquet_files = list(uploads_dir.glob("*.parquet"))
    if not parquet_files:
        logger.warning("No dataset files found in uploads directory")
        return 0
    
    logger.info(f"Found {len(parquet_files)} dataset files to load")
    
    loaded_count = 0
    for file in parquet_files:
        dataset_id = file.stem
        
        try:
            # Check if dataset is already loaded (suppress warnings during startup)
            existing_schema = await db_service.get_table_schema(dataset_id, suppress_warnings=True)
            if existing_schema:
                logger.info(f"Dataset {dataset_id} already loaded, skipping")
                loaded_count += 1
                continue
            
            # Load the parquet file
            df = pd.read_parquet(file)
            
            # Create dataset view in database
            schema = await db_service.create_dataset_view(dataset_id, df)
            
            logger.info(f"Loaded dataset {dataset_id}: {len(df)} rows, {len(df.columns)} columns")
            loaded_count += 1
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_id}", error=str(e))
    
    logger.info(f"Dataset loading complete: {loaded_count}/{len(parquet_files)} datasets loaded")
    return loaded_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for proper startup/shutdown.
    Ensures all services are properly initialized and cleaned up.
    """
    global db_service, cache_service, auth_service, orchestrator
    
    logger.info("Starting LLM Data Analyst Backend", 
                version=settings.api_version,
                environment="production" if not settings.debug else "development")
    
    try:
        # Initialize core services
        logger.info("Initializing database service...")
        db_service = DatabaseService(settings.duckdb_path)
        await db_service.initialize()
        app.state.db_service = db_service
        
        # Load datasets from uploads directory into DuckDB
        logger.info("Loading available datasets...")
        loaded_count = await load_datasets(db_service)
        logger.info(f"Successfully loaded {loaded_count} datasets into DuckDB")
        
        logger.info("Initializing cache service...")
        cache_service = CacheService()
        await cache_service.initialize()
        app.state.cache_service = cache_service
        
        logger.info("Initializing auth service...")
        auth_service = AuthService()
        app.state.auth_service = auth_service
        
        logger.info("Initializing agent orchestrator...")
        orchestrator = AgentOrchestrator(db_service=db_service)
        # Don't need to initialize since we're passing the already initialized db_service
        app.state.orchestrator = orchestrator
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e), exc_info=True)
        raise
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down application...")
    
    try:
        if db_service:
            await db_service.close()
        
        if cache_service:
            await cache_service.close()
        
        logger.info("Cleanup completed successfully")
        
    except Exception as e:
        logger.error("Error during cleanup", error=str(e))


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="AI-powered natural language data analysis platform",
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else [
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware (order matters - executed in reverse)
app.add_middleware(MetricsMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(query_router)
app.include_router(upload_router)
app.include_router(health_router)
app.include_router(session_router)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.app_name,
        "version": settings.api_version,
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "query": "/api/v1/query",
            "upload": "/api/v1/upload",
            "datasets": "/api/v1/datasets",
            "sessions": "/api/v1/sessions",
            "health": "/healthz",
            "metrics": "/metrics",
            "docs": "/docs" if settings.debug else None
        },
        "features": {
            "natural_language_queries": True,
            "multi_turn_conversations": True,
            "automatic_insights": True,
            "security_validation": True,
            "audit_trails": True,
            "visualizations": True
        }
    }

# Metrics endpoint
@app.get("/metrics", tags=["monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled errors.
    Ensures consistent error responses.
    """
    
    # Log the error
    logger.error("Unhandled exception",
                 path=request.url.path,
                 method=request.method,
                 error=str(exc),
                 exc_info=True)
    
    # Don't expose internal errors in production
    if settings.debug:
        error_detail = str(exc)
    else:
        error_detail = "An internal error occurred. Please try again later."
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": error_detail,
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )

# Request size limit
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request body size to prevent abuse"""
    
    # Skip for file uploads
    if request.url.path.endswith("/upload"):
        return await call_next(request)
    
    # Check content length
    content_length = request.headers.get("content-length")
    if content_length:
        if int(content_length) > 1_000_000:  # 1MB limit for non-upload requests
            return JSONResponse(
                status_code=413,
                content={"error": "Request entity too large"}
            )
    
    return await call_next(request)

# Security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Add CSP for production
    if not settings.debug:
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'"
        )
    
    return response

# Custom 404 handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": f"The requested path '{request.url.path}' was not found",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )

# Startup message
@app.on_event("startup")
async def startup_message():
    """Display startup information"""
    
    logger.info("=" * 50)
    logger.info(f"🚀 {settings.app_name} v{settings.api_version}")
    logger.info(f"📊 Environment: {'Development' if settings.debug else 'Production'}")
    logger.info(f"🔧 Debug Mode: {settings.debug}")
    logger.info(f"🌐 API Documentation: {'/docs' if settings.debug else 'Disabled'}")
    logger.info(f"📈 Metrics: /metrics")
    logger.info(f"🏥 Health Check: /healthz")
    logger.info("=" * 50)

# Shutdown message
@app.on_event("shutdown")
async def shutdown_message():
    """Display shutdown message"""
    logger.info("Application shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=settings.debug,
        reload_dirs=["agents", "services", "routes", "middleware", "utils", "models"],
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                },
                "access": {
                    "()": "uvicorn.logging.AccessFormatter",
                    "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                }
            },
            "handlers": {
                "default": {
                    "formatter": "json" if not settings.debug else "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
            "loggers": {
                "uvicorn.error": {
                    "level": "INFO",
                    "handlers": ["default"],
                    "propagate": False
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["default"],
                    "propagate": False
                }
            }
        },
        # SSL configuration for production
        ssl_keyfile=os.getenv("SSL_KEYFILE") if not settings.debug else None,
        ssl_certfile=os.getenv("SSL_CERTFILE") if not settings.debug else None,
        
        # Performance settings
        workers=1,  # Single worker for demo, increase for production
        loop="asyncio",
        access_log=True,
        use_colors=settings.debug
    )