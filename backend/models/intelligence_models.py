# models/intelligence_models.py
"""
Response models for intelligence dashboard and enhanced AI features.
These models support the hackathon differentiators.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class BusinessDomainType(str, Enum):
    """Business domain types"""
    SALES = "sales"
    FINANCE = "finance"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    GENERAL = "general"


class InsightCategory(str, Enum):
    """Categories of insights"""
    TREND = "trend"
    OUTLIER = "outlier"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    COMPARISON = "comparison"
    ANOMALY = "anomaly"
    RISK_ANALYSIS = "risk_analysis"
    PROFITABILITY_ANALYSIS = "profitability_analysis"
    CAMPAIGN_OPTIMIZATION = "campaign_optimization"
    DATA_QUALITY = "data_quality"


class ConfidenceLevel(str, Enum):
    """AI confidence levels"""
    VERY_HIGH = "very_high"    # 90-100%
    HIGH = "high"             # 75-89%
    MEDIUM = "medium"         # 50-74%
    LOW = "low"              # 25-49%
    VERY_LOW = "very_low"     # 0-24%


class BusinessInsightResponse(BaseModel):
    """Business insight with full context"""
    insight_type: str
    title: str
    description: str
    business_impact: str
    recommended_actions: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: Dict[str, Any] = Field(default_factory=dict)
    next_questions: List[str] = Field(default_factory=list)
    domain_relevance: Optional[str] = None
    priority_score: float = Field(ge=0.0, le=1.0, default=0.5)


class SuggestionResponse(BaseModel):
    """Smart query suggestion response"""
    suggestion_id: str
    suggested_query: str
    explanation: str
    suggestion_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    priority: int = Field(ge=1, le=5)
    business_value: str
    expected_insight: str
    domain_relevance: Optional[str] = None


class IntelligenceDashboardResponse(BaseModel):
    """Real-time intelligence dashboard data"""
    active_sessions: List[Dict[str, Any]] = Field(default_factory=list)
    global_metrics: Dict[str, Any] = Field(default_factory=dict)
    system_health: Dict[str, Any] = Field(default_factory=dict)
    recent_insights: List[str] = Field(default_factory=list)
    confidence_trends: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReasoningStepResponse(BaseModel):
    """Individual reasoning step in AI process"""
    stage: str
    timestamp: str
    description: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float
    agent_name: str
    decision_factors: List[str] = Field(default_factory=list)
    alternatives_considered: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)


class SessionMetricsResponse(BaseModel):
    """Comprehensive metrics for a reasoning session"""
    total_processing_time_ms: float
    query_complexity_score: float = Field(ge=0.0, le=1.0)
    confidence_progression: List[tuple] = Field(default_factory=list)
    stage_processing_times: Dict[str, float] = Field(default_factory=dict)
    cost_estimate: float
    steps_completed: int
    average_confidence: float = Field(ge=0.0, le=1.0)
    final_confidence: float = Field(ge=0.0, le=1.0)
    cache_utilization: float = Field(ge=0.0, le=1.0, default=0.0)
    validation_score: float = Field(ge=0.0, le=1.0, default=0.0)


class SecurityPolicyViolationResponse(BaseModel):
    """Security policy violation with educational context"""
    policy_id: str
    policy_name: str
    violation_type: str
    severity: str
    message: str
    educational_explanation: str
    remediation_steps: List[str] = Field(default_factory=list)
    auto_fix_available: bool = False
    auto_fix_description: Optional[str] = None
    learning_opportunity: str = ""
    learning_resources: List[str] = Field(default_factory=list)


class BusinessContextResponse(BaseModel):
    """Business context and domain intelligence"""
    detected_domain: BusinessDomainType
    domain_confidence: float = Field(ge=0.0, le=1.0)
    key_metrics: List[str] = Field(default_factory=list)
    common_dimensions: List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)
    business_rules: Dict[str, str] = Field(default_factory=dict)
    domain_insights: List[BusinessInsightResponse] = Field(default_factory=list)


class CounterfactualTestResponse(BaseModel):
    """Results of counterfactual validation testing"""
    test_type: str
    description: str
    original_hypothesis: str
    counterfactual_hypothesis: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    row_citations: List[str] = Field(default_factory=list)


class EvidenceTrackingResponse(BaseModel):
    """Evidence tracking and citations"""
    insight: str
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    data_provenance: Dict[str, Any] = Field(default_factory=dict)
    statistical_significance: Optional[float] = None
    business_impact: Optional[str] = None
    citations: List[str] = Field(default_factory=list)


class EnhancedQueryResponse(BaseModel):
    """Enhanced query response with all intelligence features"""
    # Core response data
    success: bool
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Intelligence enhancements
    insights: List[BusinessInsightResponse] = Field(default_factory=list)
    suggestions: List[SuggestionResponse] = Field(default_factory=list)
    business_context: Optional[BusinessContextResponse] = None
    
    # AI reasoning transparency
    reasoning_explanation: Optional[str] = None
    reasoning_steps: List[ReasoningStepResponse] = Field(default_factory=list)
    session_metrics: Optional[SessionMetricsResponse] = None
    
    # Security and validation
    security_assessment: Dict[str, Any] = Field(default_factory=dict)
    policy_violations: List[SecurityPolicyViolationResponse] = Field(default_factory=list)
    
    # Audit and evidence
    counterfactual_tests: List[CounterfactualTestResponse] = Field(default_factory=list)
    evidence_tracking: List[EvidenceTrackingResponse] = Field(default_factory=list)
    
    # Overall confidence and quality
    overall_confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    query_complexity: float = Field(ge=0.0, le=1.0, default=0.5)
    processing_time_ms: float = 0.0
    
    # Conversation continuity
    conversation_id: Optional[str] = None
    turn_number: int = 1
    
    # Next steps
    recommended_follow_ups: List[str] = Field(default_factory=list)
    learning_opportunities: List[str] = Field(default_factory=list)


class PerformanceAnalyticsResponse(BaseModel):
    """Performance analytics and trends"""
    period: str
    total_sessions: int
    success_rate: float = Field(ge=0.0, le=1.0)
    average_processing_time_ms: float
    average_confidence: float = Field(ge=0.0, le=1.0)
    stage_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    query_complexity_distribution: Dict[str, int] = Field(default_factory=dict)
    confidence_distribution: Dict[str, int] = Field(default_factory=dict)
    top_insights: List[str] = Field(default_factory=list)


class SecurityInsightsResponse(BaseModel):
    """Security insights and recommendations"""
    security_score: int = Field(ge=0, le=100)
    common_issues: List[Dict[str, str]] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    security_trends: Dict[str, List[str]] = Field(default_factory=dict)
    next_steps: List[str] = Field(default_factory=list)
    policy_summary: Dict[str, Any] = Field(default_factory=dict)


class LearningPathResponse(BaseModel):
    """Personalized learning path based on user interactions"""
    title: str
    total_topics: int
    estimated_time: str
    topics: List[Dict[str, Any]] = Field(default_factory=list)
    current_level: str = "beginner"
    completion_percentage: float = Field(ge=0.0, le=100.0, default=0.0)


class SystemHealthResponse(BaseModel):
    """System health and status indicators"""
    active_session_count: int
    total_sessions_today: int
    average_query_complexity: float = Field(ge=0.0, le=1.0)
    system_load: str = "normal"
    cache_hit_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    error_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    avg_response_time_ms: float = 0.0


class SemanticLayerResponse(BaseModel):
    """Semantic layer translation and suggestions"""
    original_query: str
    found_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    found_dimensions: List[Dict[str, Any]] = Field(default_factory=list)
    suggested_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    has_business_terms: bool = False
    translations: Dict[str, str] = Field(default_factory=dict)
    metric_library: List[Dict[str, Any]] = Field(default_factory=list)


# Utility response models for API endpoints
class SimpleResponse(BaseModel):
    """Simple success/error response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel):
    """Paginated response for large datasets"""
    success: bool
    data: List[Dict[str, Any]] = Field(default_factory=list)
    total_count: int
    page: int = 1
    page_size: int = 50
    has_more: bool = False


class ValidationResponse(BaseModel):
    """Query validation response"""
    is_valid: bool
    violations: List[SecurityPolicyViolationResponse] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    auto_fixes_available: bool = False
    security_score: float = Field(ge=0.0, le=1.0, default=1.0)
