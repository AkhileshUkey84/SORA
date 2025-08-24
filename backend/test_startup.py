#!/usr/bin/env python3
"""
Minimal FastAPI app for debugging startup issues
"""

import os
import sys
import traceback
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add current directory to path
sys.path.append('.')

app = FastAPI(title="Debug Server", debug=True)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "status": "ok",
        "message": "Debug server running",
        "debug": True
    }

@app.get("/test")
async def test():
    """Test endpoint"""
    return {"test": "success"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    error_detail = f"{type(exc).__name__}: {str(exc)}"
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": error_detail,
            "path": request.url.path,
            "method": request.method
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("Starting debug server...")
    uvicorn.run(
        "test_startup:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="debug"
    )
