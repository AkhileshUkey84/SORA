# Query routes

from fastapi import APIRouter, Depends, HTTPException
import logging
from app.models.query import QueryRequest, QueryResult
from app.services.query_service import QueryService
from app.middleware.auth import get_current_user
from datetime import datetime

router = APIRouter()
query_service = QueryService()
logger = logging.getLogger(__name__)

@router.post("/ask", response_model=QueryResult)
async def ask_question(
    request: QueryRequest,
    user_id: str = Depends(get_current_user)
):
    """Process a natural language question about a dataset"""
    
    # Log query for audit
    logger.info(f"User {user_id} querying dataset {request.dataset_id}: {request.question}")
    
    # Process query through pipeline
    result = await query_service.process_query(request, user_id)
    
    # Log result
    logger.info(f"Query completed in {result.execution_time_ms}ms with {result.row_count} rows")
    
    return result

@router.post("/clarify")
async def clarify_ambiguity(
    query_id: str,
    clarification: str,
    user_id: str = Depends(get_current_user)
):
    """Provide clarification for an ambiguous query"""
    # In production, update query context and reprocess
    return {
        "message": "Clarification received",
        "query_id": query_id,
        "clarification": clarification
    }

@router.get("/history")
async def get_query_history(
    limit: int = 10,
    user_id: str = Depends(get_current_user)
):
    """Get user's query history"""
    # In production, fetch from database
    return {
        "queries": [
            {
                "id": "query-1",
                "question": "What were last month's sales?",
                "timestamp": datetime.utcnow().isoformat(),
                "dataset_id": "dataset-1",
                "row_count": 100
            }
        ]
    }

@router.get("/{query_id}/reproduce")
async def reproduce_query(
    query_id: str,
    user_id: str = Depends(get_current_user)
):
    """Reproduce a previous query using provenance information"""
    # In production, fetch provenance and re-execute
    return {
        "message": "Query reproduced successfully",
        "query_id": query_id
    }