# API response models
# models/responses.py
"""Response models for API endpoints"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ChartType(str, Enum):
    """Supported chart types"""
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BOX = "box"


class ChartConfig(BaseModel):
    """Chart configuration for visualization"""
    
    type: ChartType
    title: str
    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    
    # Advanced options
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    legend_position: Optional[str] = "right"
    color_scheme: Optional[str] = "default"
    
    # Data hints
    sort: Optional[str] = None  # "ascending" or "descending"
    limit: Optional[int] = None
    aggregation: Optional[str] = None
    
    # Insights to highlight
    insights: List[str] = Field(default_factory=list)


class Insight(BaseModel):
    """Individual insight from analysis"""
    
    type: str  # "trend", "outlier", "correlation", etc.
    title: str
    description: str
    importance: float = Field(ge=0, le=1)
    visual_hint: Optional[str] = None
    follow_up_query: Optional[str] = None
    supporting_data: Dict[str, Any] = Field(default_factory=dict)


class AuditReport(BaseModel):
    """Audit validation report"""
    
    confidence_score: float = Field(ge=0, le=1)
    audit_status: str
    audit_checks: List[str]
    report: str
    recommendations: List[str]


class DataProvenance(BaseModel):
    """Data lineage information"""
    
    lineage_id: str
    data_sources: List[Dict[str, Any]]
    transformations: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    timestamp: datetime
    reproducible: bool


class QueryResult(BaseModel):
    """Comprehensive query result response"""
    
    success: bool
    
    # Core results
    sql_query: Optional[str] = None
    results: Optional[Dict[str, Any]] = None  # columns, data, row_count
    
    # Enhancements
    narrative: Optional[str] = None
    chart_config: Optional[ChartConfig] = None
    insights: List[Insight] = Field(default_factory=list)
    
    # Suggestions
    suggested_actions: List[str] = Field(default_factory=list)
    drill_down_queries: List[str] = Field(default_factory=list)
    
    # Validation
    audit_report: Optional[AuditReport] = None
    security_explanation: Optional[str] = None
    provenance: Optional[DataProvenance] = None
    
    # Metadata
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
    # Error handling
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # Clarification
    needs_clarification: bool = False
    clarification_questions: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "sql_query": "SELECT category, SUM(sales) as total FROM dataset GROUP BY category",
                "results": {
                    "columns": ["category", "total"],
                    "data": [
                        {"category": "Electronics", "total": 50000},
                        {"category": "Clothing", "total": 30000}
                    ],
                    "row_count": 2
                },
                "narrative": "Electronics leads with $50,000 in sales...",
                "insights": [{
                    "type": "comparison",
                    "title": "Electronics Dominates",
                    "description": "Electronics accounts for 62.5% of total sales",
                    "importance": 0.8
                }],
                "suggested_actions": ["Analyze electronics subcategories"],
                "execution_time_ms": 125.5
            }
        }


class UploadResult(BaseModel):
    """Dataset upload result"""
    
    success: bool
    dataset_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    preview: Optional[List[Dict[str, Any]]] = None
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "dataset_id": "ds_abc123def456",
                "metadata": {
                    "name": "Sales Data",
                    "row_count": 10000,
                    "column_count": 15,
                    "columns": ["date", "product", "sales", "region"]
                },
                "preview": [{"date": "2024-01-01", "product": "Widget", "sales": 100}]
            }
        }


class HealthStatus(BaseModel):
    """Health check response"""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    
    # Component health
    database: bool
    cache: bool
    llm: bool
    storage: bool
    
    # Metrics
    uptime_seconds: float
    total_queries: int
    active_sessions: int
    
    # Warnings
    warnings: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "database": True,
                "cache": True,
                "llm": True,
                "storage": True,
                "uptime_seconds": 3600.5,
                "total_queries": 1523,
                "active_sessions": 12
            }
        }


class SessionInfo(BaseModel):
    """Session information response"""
    
    session_id: str
    conversation_id: Optional[str] = None
    created_at: datetime
    last_active: datetime
    query_count: int
    datasets_accessed: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)