# Main query endpoints
# routes/query.py
"""
Main query endpoint orchestrating the full agent pipeline.
Designed for optimal demo flow with rich responses.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional
from models.requests import QueryRequest
from models.responses import QueryResult
from agents.base import AgentContext
from services.agent_orchestrator import AgentOrchestrator
from middleware.auth import get_current_user
from middleware.rate_limit import rate_limit_check
import uuid
import structlog


logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResult)
async def analyze_query(
    request: QueryRequest,
    req: Request,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(rate_limit_check)
) -> QueryResult:
    """
    Main endpoint for natural language data analysis.
    Orchestrates the full agent pipeline from NL to insights.
    """
    
    # Create context
    context = AgentContext(
        session_id=str(uuid.uuid4()),
        user_id=current_user.get("id"),
        conversation_id=request.conversation_id,
        dataset_id=request.dataset_id,
        query=request.query
    )
    
    logger.info("Processing query", dataset_id=request.dataset_id, query=request.query[:100])
    
    # Handle invalid dataset IDs
    if request.dataset_id == "string" or not request.dataset_id:
        # Get available dataset IDs from uploads directory
        import os
        from pathlib import Path
        uploads_dir = Path("./uploads")
        if uploads_dir.exists():
            parquet_files = list(uploads_dir.glob("*.parquet"))
            if parquet_files:
                # Use the first available dataset
                dataset_id = parquet_files[0].stem
                logger.info(f"Using fallback dataset ID: {dataset_id}")
                context.dataset_id = dataset_id
            else:
                return QueryResult(
                    success=False,
                    error="No datasets available",
                    narrative="Please upload a dataset first before making queries."
                )
        else:
            return QueryResult(
                success=False,
                error="No datasets available",
                narrative="Please upload a dataset first before making queries."
            )
    
    # Get orchestrator from app state or create new one
    if hasattr(req.app.state, 'orchestrator'):
        orchestrator = req.app.state.orchestrator
    else:
        # Create new orchestrator with shared db_service
        db_service = getattr(req.app.state, 'db_service', None)
        orchestrator = AgentOrchestrator(db_service=db_service)
        
        # Initialize orchestrator only if db_service wasn't passed
        if not db_service:
            await orchestrator.initialize()
    
    try:
        # Run the full pipeline
        result = await orchestrator.process_query(
            context=context,
            enable_insights=request.enable_insights,
            enable_audit=request.enable_audit,
            enable_multimodal=request.enable_multimodal
        )
        
        return result
        
    except Exception as e:
        logger.error("Query processing failed", 
                    error=str(e),
                    session_id=context.session_id)
        return QueryResult(
            success=False,
            error=f"Query processing failed: {str(e)}",
            narrative="I encountered an issue processing your request. Please try rephrasing your question."
        )


@router.post("/report")
async def generate_report(
    request: Request,
    session_id: str,
    current_user: dict = Depends(get_current_user),
    _: None = Depends(rate_limit_check)
):
    """
    Generates executive summary report for a session.
    Uses ReportAgent to analyze session history and create insights.
    """
    
    try:
        # Get orchestrator from app state
        if hasattr(request.app.state, 'orchestrator'):
            orchestrator = request.app.state.orchestrator
        else:
            return {
                "success": False,
                "error": "Service not available"
            }
        
        # Initialize agents if needed
        if not orchestrator._agents_initialized:
            orchestrator._initialize_agents()
        
        # Generate report using ReportAgent
        report_result = await orchestrator.report_agent.generate_session_report(
            session_id=session_id,
            user_id=current_user.get("id")
        )
        
        if not report_result.success:
            return {
                "success": False,
                "error": report_result.error or "Failed to generate report"
            }
        
        return {
            "success": True,
            "session_id": session_id,
            "report": report_result.data
        }
        
    except Exception as e:
        logger.error("Report generation failed",
                    session_id=session_id,
                    error=str(e))
        return {
            "success": False,
            "error": "Failed to generate report"
        }
