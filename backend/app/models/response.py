from pydantic import BaseModel
from typing import Optional, Any, List

class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Any] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Any] = None

class InsightResponse(BaseModel):
    type: str  # trend, anomaly, correlation, summary
    description: str
    evidence: List[Dict[str, Any]]
    confidence: float
    visualization_hint: Optional[str] = None