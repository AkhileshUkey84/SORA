from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from app.config import settings
from app.routes import datasets, queries, health
from app.middleware.auth import AuthMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.utils.database import init_db, close_db
from app.utils.cache import init_cache, close_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting application...")
    await init_db()
    await init_cache()
    yield
    # Shutdown
    logger.info("Shutting down application...")
    await close_db()
    await close_cache()

app = FastAPI(
    title="LLM Data Analyst Assistant",
    description="AI-powered natural language data analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(RateLimitMiddleware)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(queries.router, prefix="/api/queries", tags=["queries"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

@app.get("/")
async def root():
    return {
        "message": "LLM Data Analyst Assistant API",
        "docs": "/docs",
        "health": "/api/health"
    }