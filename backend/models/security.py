# Security policy models
# models/security.py
"""Security and validation models"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class SecurityLevel(str, Enum):
    """Security severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(str, Enum):
    """Types of security threats"""
    SQL_INJECTION = "sql_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PII_EXPOSURE = "pii_exposure"


class SecurityPolicy(BaseModel):
    """Security policy configuration"""
    
    name: str
    description: str
    enabled: bool = True
    level: SecurityLevel
    
    # Rules
    max_rows: int = 10000
    max_execution_time: int = 30
    allowed_operations: List[str] = ["SELECT"]
    forbidden_keywords: List[str] = Field(default_factory=list)
    require_row_limit: bool = True
    
    # PII handling
    pii_detection_enabled: bool = True
    pii_masking_enabled: bool = True
    pii_patterns: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """SQL validation result"""
    
    is_valid: bool
    original_sql: str
    validated_sql: Optional[str] = None
    
    # Issues found
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    threats_detected: List[ThreatType] = Field(default_factory=list)
    
    # Modifications made
    modifications: List[str] = Field(default_factory=list)
    
    # Risk assessment
    risk_level: SecurityLevel
    confidence: float = Field(ge=0, le=1)
    
    # Human-readable explanation
    explanation: str


class PIIDetectionResult(BaseModel):
    """PII detection result"""
    
    has_pii: bool
    pii_fields: List[str] = Field(default_factory=list)
    pii_types: Dict[str, str] = Field(default_factory=dict)  # field -> type
    masked_count: int = 0
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)


class AccessControlResult(BaseModel):
    """Access control check result"""
    
    granted: bool
    user_id: str
    resource_id: str
    action: str
    
    # Denial reason if applicable
    denial_reason: Optional[str] = None
    
    # Applicable policies
    policies_checked: List[str] = Field(default_factory=list)
    policies_passed: List[str] = Field(default_factory=list)
    policies_failed: List[str] = Field(default_factory=list)