# Agent data models
# models/agents.py
"""Agent-specific data models"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class AgentContext(BaseModel):
    """
    Shared context passed between agents for maintaining state.
    Central to the agent pipeline architecture.
    """
    session_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    dataset_id: str
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trace: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_trace(self, agent_name: str, action: str, details: Dict[str, Any]):
        """Add audit trail entry"""
        self.trace.append({
            "agent": agent_name,
            "action": action,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })


class AgentResult(BaseModel):
    """
    Standard result format for agent communication.
    Ensures consistent error handling and metadata passing.
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(None, ge=0, le=1)


class AgentType(str, Enum):
    """Types of agents in the pipeline"""
    MEMORY = "memory"
    ROUTER = "router"
    PLANNER = "planner"
    SQL_GENERATOR = "sql_generator"
    VALIDATOR = "validator"
    EXECUTOR = "executor"
    EXPLAINER = "explainer"
    INSIGHT = "insight"
    AUDIT = "audit"
    LINEAGE = "lineage"
    REPORT = "report"


class AgentMetrics(BaseModel):
    """Performance metrics for an agent"""
    agent_type: AgentType
    execution_time_ms: float
    success_rate: float
    error_count: int
    avg_confidence: float
    last_execution: datetime