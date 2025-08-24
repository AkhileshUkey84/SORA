# Routes package
# routes/__init__.py
"""API route modules"""

from .query import router as query_router
from .upload import router as upload_router
from .health import router as health_router
from .session import router as session_router

# Export all routers
__all__ = [
    'query_router',
    'upload_router', 
    'health_router',
    'session_router'
]