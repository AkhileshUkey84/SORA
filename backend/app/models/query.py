from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID
from enum import Enum

class QueryIntent(str, Enum):
    AGGREGATE = "aggregate"
    FILTER = "filter"
    JOIN = "join"
    TIME_SERIES = "time_series"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"

class AmbiguityType(str, Enum):
    TEMPORAL = "temporal"
    ENTITY = "entity"
    METRIC = "metric"
    SCOPE = "scope"

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    dataset_id: str  # Changed from UUID to str for flexibility
    context: Optional[Dict[str, Any]] = None
    enable_counterfactual: bool = True
    enable_explanation: bool = True

class ClarificationRequest(BaseModel):
    ambiguity_type: AmbiguityType
    message: str
    options: List[str]
    original_question: str

class QueryPlan(BaseModel):
    intent: QueryIntent
    complexity: str  # simple, moderate, complex
    requires_joins: bool
    estimated_rows: Optional[int]
    clarifications_needed: Optional[List[ClarificationRequest]]

class GuardrailResult(BaseModel):
    passed: bool
    violations: List[str]
    modified_query: Optional[str]
    explanation: str

class QueryResult(BaseModel):
    question: str
    sql_query: str
    results: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    explanation: Optional[str]
    counterfactual: Optional[Dict[str, Any]] = None
    guardrail_report: Optional[GuardrailResult] = None
    confidence_score: float
    provenance: Dict[str, Any]
