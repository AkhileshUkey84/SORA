# Session management
# routes/session.py
"""Session management endpoints"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from models.requests import SessionRequest
from models.responses import SessionInfo
from services.cache_service import CacheService
from middleware.auth import get_current_user
import uuid
from datetime import datetime
import structlog


router = APIRouter(prefix="/api/v1", tags=["session"])
logger = structlog.get_logger()


@router.post("/sessions")
async def manage_session(
    request: SessionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create, retrieve, or end a session"""
    
    cache_service = CacheService()
    await cache_service.initialize()
    
    if request.action == "create":
        # Create new session
        session_id = str(uuid.uuid4())
        conversation_id = str(uuid.uuid4())
        
        session_data = {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "user_id": current_user["id"],
            "created_at": datetime.utcnow().isoformat(),
            "last_active": datetime.utcnow().isoformat(),
            "query_count": 0,
            "datasets_accessed": [],
            "metadata": request.metadata
        }
        
        # Store in cache
        await cache_service.set_json(
            f"session:{session_id}",
            session_data,
            ttl=3600  # 1 hour
        )
        
        return SessionInfo(**session_data)
    
    elif request.action == "retrieve":
        if not request.session_id:
            raise HTTPException(400, "Session ID required for retrieval")
        
        # Get session data
        session_data = await cache_service.get_json(f"session:{request.session_id}")
        
        if not session_data:
            raise HTTPException(404, "Session not found")
        
        # Verify ownership
        if session_data["user_id"] != current_user["id"]:
            raise HTTPException(403, "Access denied")
        
        return SessionInfo(**session_data)
    
    elif request.action == "end":
        if not request.session_id:
            raise HTTPException(400, "Session ID required to end session")
        
        # Delete session
        await cache_service.delete(f"session:{request.session_id}")
        
        return {"message": "Session ended successfully"}
    
    else:
        raise HTTPException(400, f"Invalid action: {request.action}")


@router.get("/sessions/{session_id}/report")
async def get_session_report(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Generate executive report for a session"""
    
    from agents.report_agent import ReportAgent
    from agents.base import AgentContext
    
    # Create context
    context = AgentContext(
        session_id=session_id,
        user_id=current_user["id"],
        conversation_id=session_id,  # Use session as conversation
        dataset_id="multiple",
        query="Generate session report"
    )
    
    # Initialize services
    cache_service = CacheService()
    await cache_service.initialize()
    
    # Generate report
    report_agent = ReportAgent(cache_service)
    result = await report_agent.execute(context)
    
    if result.success:
        return result.data
    else:
        raise HTTPException(500, "Failed to generate report")