# API request models
# models/requests.py
"""Request models for API endpoints"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class QueryRequest(BaseModel):
    """Main query request model"""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    dataset_id: str = Field(..., description="ID of dataset to query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    
    # Feature flags
    enable_insights: bool = Field(True, description="Enable automatic insight discovery")
    enable_audit: bool = Field(True, description="Enable result validation")
    enable_multimodal: bool = Field(True, description="Enable chart generation")
    
    # Advanced options
    max_rows: Optional[int] = Field(None, description="Maximum rows to return (0 or None = use default)")
    timeout_seconds: Optional[int] = Field(None, description="Query timeout in seconds (0 or None = use default)")
    
    @validator('max_rows')
    def validate_max_rows(cls, v):
        """Validate max_rows - 0 means use default"""
        if v is not None and v < 0:
            raise ValueError("max_rows must be non-negative")
        if v is not None and v > 10000:
            raise ValueError("max_rows cannot exceed 10000")
        return v
    
    @validator('timeout_seconds')
    def validate_timeout(cls, v):
        """Validate timeout_seconds - 0 means use default"""
        if v is not None and v < 0:
            raise ValueError("timeout_seconds must be non-negative")
        if v is not None and v > 300:
            raise ValueError("timeout_seconds cannot exceed 300")
        return v
    
    @validator('query')
    def clean_query(cls, v):
        """Clean and validate query text"""
        # Remove excessive whitespace
        v = ' '.join(v.split())
        
        # Basic injection prevention
        forbidden = ['<script', 'javascript:', 'onclick']
        if any(f in v.lower() for f in forbidden):
            raise ValueError("Query contains forbidden content")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Show me total sales by category for last month",
                "dataset_id": "ds_abc123def456",
                "conversation_id": "conv_789xyz",
                "enable_insights": True,
                "enable_audit": True,
                "enable_multimodal": True
            }
        }


class UploadRequest(BaseModel):
    """Dataset upload request model"""
    
    name: Optional[str] = Field(None, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, max_length=1000, description="Dataset description")
    tags: List[str] = Field(default_factory=list, description="Dataset tags")
    is_public: bool = Field(False, description="Make dataset publicly accessible")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Q4 Sales Data",
                "description": "Sales transactions for Q4 2024",
                "tags": ["sales", "q4", "2024"],
                "is_public": False
            }
        }


class SessionRequest(BaseModel):
    """Session management request"""
    
    action: str = Field(..., pattern="^(create|retrieve|end)$", description="Session action")
    session_id: Optional[str] = Field(None, description="Existing session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "action": "create",
                "metadata": {
                    "client": "web",
                    "version": "1.0"
                }
            }
        }


class FeedbackRequest(BaseModel):
    """User feedback request"""
    
    query_id: str = Field(..., description="Query ID to provide feedback for")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback_type: str = Field(..., pattern="^(accuracy|speed|relevance|clarity)$")
    comment: Optional[str] = Field(None, max_length=1000, description="Additional comments")
    
    class Config:
        schema_extra = {
            "example": {
                "query_id": "qry_123abc",
                "rating": 4,
                "feedback_type": "accuracy",
                "comment": "Results were accurate but could be more detailed"
            }
        }