# Base agent interface
# agents/base.py
"""
Base agent interface establishing the contract for all pipeline agents.
Designed for maximum extensibility and consistent error handling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field
import structlog
from datetime import datetime


logger = structlog.get_logger()


class AgentContext(BaseModel):
    """
    Shared context passed between agents for maintaining state.
    Enables rich traceability and debugging during demos.
    """
    session_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    dataset_id: str
    query: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}
    trace: List[Dict[str, Any]] = []
    
    def add_trace(self, agent_name: str, action: str, details: Dict[str, Any]):
        """Add audit trail entry for complete reproducibility"""
        self.trace.append({
            "agent": agent_name,
            "action": action,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })


class AgentResult(BaseModel):
    """
    Standard result format ensuring consistent agent communication.
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    confidence: Optional[float] = None


class BaseAgent(ABC):
    """
    Foundation for all agents with built-in observability and error handling.
    Each agent focuses on a single responsibility for maintainability.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logger.bind(agent=name)
    
    @abstractmethod
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Core processing logic - must be implemented by each agent.
        Async by default for optimal performance under load.
        """
        pass
    
    async def execute(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Wrapper providing consistent logging and error handling.
        Critical for demo reliability - graceful degradation on failures.
        """
        start_time = datetime.utcnow()
        
        try:
            self.logger.info(f"Starting {self.name} processing", 
                           session_id=context.session_id,
                           query=context.query[:100])  # Log truncated query
            
            result = await self.process(context, **kwargs)
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.logger.info(f"Completed {self.name} processing",
                           session_id=context.session_id,
                           success=result.success,
                           duration_ms=duration_ms)
            
            # Add to trace for full auditability
            context.add_trace(
                self.name,
                "completed",
                {
                    "success": result.success,
                    "duration_ms": duration_ms,
                    "confidence": result.confidence
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {self.name}",
                            session_id=context.session_id,
                            error=str(e),
                            exc_info=True)
            
            context.add_trace(
                self.name,
                "error",
                {"error": str(e)}
            )
            
            return AgentResult(
                success=False,
                error=f"Agent {self.name} encountered an error: {str(e)}"
            )