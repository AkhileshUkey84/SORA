#!/usr/bin/env python3
"""
Debug script to test service initialization issues
"""

import sys
import traceback
import asyncio
from typing import Optional

sys.path.append('.')

async def test_imports():
    """Test all imports individually"""
    print("Testing imports...")
    
    try:
        print("  ✓ Testing config...")
        from utils.config import settings
        print(f"  ✓ Config loaded: {settings.app_name}")
        
        print("  ✓ Testing middleware...")
        from middleware.logging import LoggingMiddleware, StructuredLogger
        from middleware.metrics import MetricsMiddleware
        from middleware.auth import AuthMiddleware
        from middleware.rate_limit import RateLimitMiddleware
        print("  ✓ Middleware imports OK")
        
        print("  ✓ Testing routes...")
        from routes import query_router, upload_router, health_router, session_router
        print("  ✓ Routes imports OK")
        
        print("  ✓ Testing services...")
        from services.database_service import DatabaseService
        from services.cache_service import CacheService
        from services.auth_service import AuthService
        from services.agent_orchestrator import AgentOrchestrator
        print("  ✓ Services imports OK")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        traceback.print_exc()
        return False

async def test_service_initialization():
    """Test service initialization"""
    print("\nTesting service initialization...")
    
    try:
        from utils.config import settings
        from services.database_service import DatabaseService
        from services.cache_service import CacheService
        from services.auth_service import AuthService
        from services.agent_orchestrator import AgentOrchestrator
        
        # Test Database Service
        print("  ✓ Testing DatabaseService initialization...")
        db_service = DatabaseService(settings.duckdb_path)
        await db_service.initialize()
        print("  ✓ DatabaseService OK")
        
        # Test Cache Service
        print("  ✓ Testing CacheService initialization...")
        cache_service = CacheService()
        await cache_service.initialize()
        print("  ✓ CacheService OK")
        
        # Test Auth Service
        print("  ✓ Testing AuthService initialization...")
        auth_service = AuthService()
        print("  ✓ AuthService OK")
        
        # Test Agent Orchestrator
        print("  ✓ Testing AgentOrchestrator initialization...")
        orchestrator = AgentOrchestrator()
        await orchestrator.initialize()
        print("  ✓ AgentOrchestrator OK")
        
        # Cleanup
        await db_service.close()
        await cache_service.close()
        
        return True
        
    except Exception as e:
        print(f"  ✗ Service initialization failed: {e}")
        traceback.print_exc()
        return False

async def test_app_creation():
    """Test FastAPI app creation with all middleware"""
    print("\nTesting FastAPI app creation...")
    
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        from utils.config import settings
        from middleware.logging import LoggingMiddleware
        from middleware.metrics import MetricsMiddleware
        from middleware.auth import AuthMiddleware
        from middleware.rate_limit import RateLimitMiddleware
        from routes import query_router, upload_router, health_router, session_router
        
        # Create app
        app = FastAPI(
            title=settings.app_name,
            description="AI-powered natural language data analysis platform",
            version=settings.api_version,
            docs_url="/docs" if settings.debug else None,
            redoc_url="/redoc" if settings.debug else None,
            openapi_url="/openapi.json" if settings.debug else None
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )
        
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        app.add_middleware(MetricsMiddleware)
        app.add_middleware(LoggingMiddleware)
        app.add_middleware(AuthMiddleware)
        app.add_middleware(RateLimitMiddleware)
        
        # Include routers
        app.include_router(query_router)
        app.include_router(upload_router)
        app.include_router(health_router)
        app.include_router(session_router)
        
        print("  ✓ FastAPI app created successfully")
        print(f"  ✓ Routes: {[route.path for route in app.routes]}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ App creation failed: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main debug function"""
    print("=== Service Debug Test ===\n")
    
    # Test imports
    if not await test_imports():
        print("\n❌ Import test failed")
        return
    
    # Test service initialization
    if not await test_service_initialization():
        print("\n❌ Service initialization test failed")
        return
    
    # Test app creation
    if not await test_app_creation():
        print("\n❌ App creation test failed")
        return
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())
