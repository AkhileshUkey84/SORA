# services/intelligence_dashboard_service.py
"""
Intelligence Dashboard Service - Real-time AI reasoning transparency.
Key hackathon differentiator showing live AI decision-making process.
"""

from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import structlog
from dataclasses import dataclass, asdict

logger = structlog.get_logger()


class ReasoningStage(str, Enum):
    """Stages of AI reasoning process"""
    QUERY_ANALYSIS = "query_analysis"
    INTENT_DETECTION = "intent_detection"
    CONTEXT_BUILDING = "context_building"
    AMBIGUITY_RESOLUTION = "ambiguity_resolution"
    SQL_GENERATION = "sql_generation"
    VALIDATION = "validation"
    EXECUTION = "execution"
    INSIGHT_DISCOVERY = "insight_discovery"
    RESULT_SYNTHESIS = "result_synthesis"


class ConfidenceLevel(str, Enum):
    """AI confidence levels"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MEDIUM = "medium"       # 50-74%
    LOW = "low"            # 25-49%
    VERY_LOW = "very_low"   # 0-24%


@dataclass
class ReasoningStep:
    """Individual step in the AI reasoning process"""
    stage: ReasoningStage
    timestamp: datetime
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence_score: float
    processing_time_ms: float
    agent_name: str
    decision_factors: List[str] = None
    alternatives_considered: List[str] = None
    evidence: List[str] = None


@dataclass
class IntelligenceMetrics:
    """Real-time intelligence metrics"""
    total_processing_time_ms: float
    query_complexity_score: float
    confidence_progression: List[Tuple[str, float]]  # stage -> confidence
    cost_estimate: float
    token_usage: Dict[str, int]
    cache_hit_rate: float
    validation_passes: int
    counterfactual_tests: int


class ReasoningSession(BaseModel):
    """Complete reasoning session with all steps"""
    session_id: str
    query: str
    user_context: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Optional[Dict[str, Any]] = None
    final_confidence: float = 0.0
    status: str = "in_progress"  # in_progress, completed, failed
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    insights_discovered: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class IntelligenceDashboardService:
    """
    Provides real-time visibility into AI reasoning process.
    Makes AI decision-making transparent and explainable for judges/users.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="IntelligenceDashboardService")
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self.session_history: List[ReasoningSession] = []
        self.max_history = 100
        
        # Real-time subscribers for live updates
        self.subscribers: Dict[str, List] = {}  # session_id -> [callback_functions]
        
        # Performance metrics
        self.global_metrics = {
            "total_queries_processed": 0,
            "average_processing_time": 0.0,
            "average_confidence": 0.0,
            "success_rate": 0.0
        }
    
    def start_reasoning_session(self, session_id: str, query: str, 
                              user_context: Dict[str, Any] = None) -> ReasoningSession:
        """Start a new reasoning session for tracking"""
        session = ReasoningSession(
            session_id=session_id,
            query=query,
            user_context=user_context or {}
        )
        
        self.active_sessions[session_id] = session
        self.logger.info(f"Started reasoning session {session_id}", query=query)
        
        # Notify subscribers
        self._notify_subscribers(session_id, "session_started", session)
        
        return session
    
    async def add_reasoning_step(self, session_id: str, step: ReasoningStep) -> None:
        """Add a reasoning step to the session"""
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} not found")
            return
        
        session = self.active_sessions[session_id]
        step_dict = asdict(step)
        step_dict['timestamp'] = step.timestamp.isoformat()
        session.steps.append(step_dict)
        
        self.logger.debug(f"Added reasoning step to {session_id}", 
                         stage=step.stage, confidence=step.confidence_score)
        
        # Notify subscribers in real-time
        await self._notify_subscribers_async(session_id, "step_added", step_dict)
    
    async def update_session_confidence(self, session_id: str, 
                                      stage: ReasoningStage, confidence: float) -> None:
        """Update confidence score for a specific stage"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session.final_confidence = confidence
        
        # Track confidence progression
        if not hasattr(session, '_confidence_history'):
            session._confidence_history = []
        session._confidence_history.append((stage.value, confidence))
        
        await self._notify_subscribers_async(session_id, "confidence_updated", {
            "stage": stage.value,
            "confidence": confidence
        })
    
    async def complete_reasoning_session(self, session_id: str, 
                                       final_result: Dict[str, Any],
                                       insights: List[str] = None,
                                       recommendations: List[str] = None) -> ReasoningSession:
        """Complete a reasoning session"""
        if session_id not in self.active_sessions:
            self.logger.warning(f"Session {session_id} not found")
            return None
        
        session = self.active_sessions[session_id]
        session.end_time = datetime.utcnow()
        session.status = "completed"
        session.insights_discovered = insights or []
        session.recommendations = recommendations or []
        
        # Calculate metrics
        session.metrics = self._calculate_session_metrics(session)
        
        # Move to history
        self.session_history.append(session)
        if len(self.session_history) > self.max_history:
            self.session_history.pop(0)
        
        del self.active_sessions[session_id]
        
        # Update global metrics
        self._update_global_metrics(session)
        
        # Final notification
        await self._notify_subscribers_async(session_id, "session_completed", {
            "final_confidence": session.final_confidence,
            "metrics": session.metrics,
            "insights": session.insights_discovered
        })
        
        self.logger.info(f"Completed reasoning session {session_id}", 
                        confidence=session.final_confidence)
        
        return session
    
    def _calculate_session_metrics(self, session: ReasoningSession) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a session"""
        if not session.steps:
            return {}
        
        steps = session.steps
        start_time = datetime.fromisoformat(steps[0]['timestamp'])
        end_time = session.end_time or datetime.utcnow()
        total_time = (end_time - start_time).total_seconds() * 1000
        
        # Confidence progression
        confidence_history = []
        for step in steps:
            confidence_history.append((step['stage'], step['confidence_score']))
        
        # Processing time by stage
        stage_times = {}
        for step in steps:
            stage = step['stage']
            time_ms = step.get('processing_time_ms', 0)
            if stage in stage_times:
                stage_times[stage] += time_ms
            else:
                stage_times[stage] = time_ms
        
        # Query complexity analysis
        complexity_score = self._calculate_complexity_score(session.query, steps)
        
        # Cost estimation (simplified)
        cost_estimate = self._estimate_processing_cost(steps)
        
        return {
            "total_processing_time_ms": total_time,
            "query_complexity_score": complexity_score,
            "confidence_progression": confidence_history,
            "stage_processing_times": stage_times,
            "cost_estimate": cost_estimate,
            "steps_completed": len(steps),
            "average_confidence": sum(s['confidence_score'] for s in steps) / len(steps),
            "final_confidence": session.final_confidence,
            "cache_utilization": self._calculate_cache_utilization(steps),
            "validation_score": self._calculate_validation_score(steps)
        }
    
    def _calculate_complexity_score(self, query: str, steps: List[Dict]) -> float:
        """Calculate query complexity score (0-1)"""
        complexity = 0.0
        
        # Base complexity from query length and keywords
        query_words = len(query.split())
        complexity += min(query_words / 20, 0.3)  # Max 0.3 for length
        
        # Complexity from processing steps
        complexity += min(len(steps) / 15, 0.4)  # Max 0.4 for steps
        
        # Complexity from ambiguity resolution
        ambiguity_steps = [s for s in steps if s['stage'] == 'ambiguity_resolution']
        complexity += min(len(ambiguity_steps) * 0.1, 0.3)  # Max 0.3 for ambiguities
        
        return min(complexity, 1.0)
    
    def _estimate_processing_cost(self, steps: List[Dict]) -> float:
        """Estimate processing cost in arbitrary units"""
        cost = 0.0
        
        cost_per_stage = {
            "query_analysis": 0.1,
            "intent_detection": 0.2,
            "context_building": 0.3,
            "ambiguity_resolution": 0.4,
            "sql_generation": 0.5,
            "validation": 0.3,
            "execution": 0.2,
            "insight_discovery": 0.6,
            "result_synthesis": 0.4
        }
        
        for step in steps:
            stage = step['stage']
            base_cost = cost_per_stage.get(stage, 0.1)
            time_multiplier = step.get('processing_time_ms', 100) / 1000
            cost += base_cost * time_multiplier
        
        return round(cost, 3)
    
    def _calculate_cache_utilization(self, steps: List[Dict]) -> float:
        """Calculate cache hit rate (simplified)"""
        # This would integrate with actual cache metrics
        return 0.75  # Placeholder
    
    def _calculate_validation_score(self, steps: List[Dict]) -> float:
        """Calculate validation thoroughness score"""
        validation_steps = [s for s in steps if s['stage'] == 'validation']
        if not validation_steps:
            return 0.5
        
        return sum(s['confidence_score'] for s in validation_steps) / len(validation_steps)
    
    def _update_global_metrics(self, session: ReasoningSession) -> None:
        """Update global performance metrics"""
        self.global_metrics["total_queries_processed"] += 1
        
        # Update averages
        total_queries = self.global_metrics["total_queries_processed"]
        
        if session.metrics:
            current_avg_time = self.global_metrics["average_processing_time"]
            new_time = session.metrics["total_processing_time_ms"]
            self.global_metrics["average_processing_time"] = (
                (current_avg_time * (total_queries - 1) + new_time) / total_queries
            )
        
        # Update confidence average
        current_avg_conf = self.global_metrics["average_confidence"]
        new_conf = session.final_confidence
        self.global_metrics["average_confidence"] = (
            (current_avg_conf * (total_queries - 1) + new_conf) / total_queries
        )
        
        # Update success rate
        if session.status == "completed" and session.final_confidence > 0.7:
            successful_queries = self.global_metrics["success_rate"] * (total_queries - 1) + 1
            self.global_metrics["success_rate"] = successful_queries / total_queries
    
    async def _notify_subscribers_async(self, session_id: str, event_type: str, data: Dict[str, Any]) -> None:
        """Notify subscribers asynchronously"""
        if session_id in self.subscribers:
            tasks = []
            for callback in self.subscribers[session_id]:
                tasks.append(callback(event_type, data))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def _notify_subscribers(self, session_id: str, event_type: str, data: Any) -> None:
        """Notify subscribers synchronously"""
        if session_id in self.subscribers:
            for callback in self.subscribers[session_id]:
                try:
                    callback(event_type, data)
                except Exception as e:
                    self.logger.warning(f"Subscriber notification failed: {e}")
    
    def subscribe_to_session(self, session_id: str, callback) -> None:
        """Subscribe to real-time updates for a session"""
        if session_id not in self.subscribers:
            self.subscribers[session_id] = []
        self.subscribers[session_id].append(callback)
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a reasoning session"""
        # Check active sessions first
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
        else:
            # Check history
            session = next((s for s in self.session_history if s.session_id == session_id), None)
        
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "query": session.query,
            "status": session.status,
            "confidence": session.final_confidence,
            "steps_count": len(session.steps),
            "insights_count": len(session.insights_discovered),
            "start_time": session.start_time.isoformat(),
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "metrics": session.metrics
        }
    
    def get_live_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for UI"""
        active_sessions_data = []
        for session_id, session in self.active_sessions.items():
            active_sessions_data.append({
                "session_id": session_id,
                "query": session.query[:100] + "..." if len(session.query) > 100 else session.query,
                "current_stage": session.steps[-1]['stage'] if session.steps else "starting",
                "confidence": session.final_confidence,
                "steps_completed": len(session.steps),
                "elapsed_time": (datetime.utcnow() - session.start_time).total_seconds()
            })
        
        return {
            "active_sessions": active_sessions_data,
            "global_metrics": self.global_metrics,
            "system_health": {
                "active_session_count": len(self.active_sessions),
                "total_sessions_today": len([s for s in self.session_history 
                                           if s.start_time.date() == datetime.utcnow().date()]),
                "average_query_complexity": self._get_average_complexity(),
                "system_load": "normal"  # This would be real system metrics
            },
            "recent_insights": self._get_recent_insights(),
            "confidence_trends": self._get_confidence_trends()
        }
    
    def _get_average_complexity(self) -> float:
        """Get average query complexity from recent sessions"""
        recent_sessions = [s for s in self.session_history[-20:] if s.metrics]
        if not recent_sessions:
            return 0.5
        
        return sum(s.metrics.get("query_complexity_score", 0.5) for s in recent_sessions) / len(recent_sessions)
    
    def _get_recent_insights(self) -> List[str]:
        """Get recent insights discovered across sessions"""
        insights = []
        for session in self.session_history[-10:]:  # Last 10 sessions
            insights.extend(session.insights_discovered[:2])  # Top 2 insights per session
        
        return insights[-5:]  # Return 5 most recent
    
    def _get_confidence_trends(self) -> List[Dict[str, Any]]:
        """Get confidence trends over time"""
        trends = []
        for session in self.session_history[-20:]:  # Last 20 sessions
            if session.end_time:
                trends.append({
                    "timestamp": session.end_time.isoformat(),
                    "confidence": session.final_confidence,
                    "complexity": session.metrics.get("query_complexity_score", 0.5) if session.metrics else 0.5
                })
        
        return trends
    
    def generate_reasoning_explanation(self, session_id: str) -> str:
        """Generate human-readable explanation of AI reasoning"""
        session = self.active_sessions.get(session_id) or next(
            (s for s in self.session_history if s.session_id == session_id), None
        )
        
        if not session:
            return "Session not found"
        
        explanation = [f"🤖 **AI Reasoning Process for:** \"{session.query}\"\n"]
        
        if session.steps:
            explanation.append("**Processing Steps:**")
            
            stage_descriptions = {
                "query_analysis": "🔍 Analyzed your question to understand what you're asking",
                "intent_detection": "🎯 Determined your intent and what type of analysis you need",
                "context_building": "🧠 Built context using conversation history and domain knowledge",
                "ambiguity_resolution": "❓ Resolved ambiguous terms and clarified requirements",
                "sql_generation": "💾 Generated optimized SQL query to get your data",
                "validation": "✅ Validated query for security, performance, and accuracy", 
                "execution": "⚡ Executed query and retrieved results",
                "insight_discovery": "💡 Analyzed results to discover key insights and patterns",
                "result_synthesis": "📊 Synthesized findings into actionable intelligence"
            }
            
            for i, step in enumerate(session.steps, 1):
                stage = step['stage']
                confidence = step['confidence_score']
                time_ms = step.get('processing_time_ms', 0)
                
                desc = stage_descriptions.get(stage, f"Processed {stage}")
                confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"
                
                explanation.append(f"{i}. {desc} {confidence_emoji}")
                explanation.append(f"   *Confidence: {confidence:.1%}, Time: {time_ms:.0f}ms*")
                
                # Add decision factors if available
                if step.get('decision_factors'):
                    factors = step['decision_factors'][:2]  # Top 2 factors
                    explanation.append(f"   *Key factors: {', '.join(factors)}*")
        
        # Add final assessment
        final_conf = session.final_confidence
        if final_conf > 0.9:
            assessment = "🟢 **Very High Confidence** - Results are highly reliable"
        elif final_conf > 0.7:
            assessment = "🟡 **High Confidence** - Results are reliable with minor uncertainties"
        elif final_conf > 0.5:
            assessment = "🟠 **Moderate Confidence** - Results are reasonably accurate"
        else:
            assessment = "🔴 **Lower Confidence** - Results may need additional verification"
        
        explanation.append(f"\n**Final Assessment:** {assessment}")
        
        # Add insights if any
        if session.insights_discovered:
            explanation.append("\n**Key Insights Discovered:**")
            for insight in session.insights_discovered[:3]:  # Top 3
                explanation.append(f"• {insight}")
        
        # Add recommendations if any
        if session.recommendations:
            explanation.append("\n**Recommendations:**")
            for rec in session.recommendations[:2]:  # Top 2
                explanation.append(f"• {rec}")
        
        return "\n".join(explanation)
    
    def get_performance_analytics(self, time_period: str = "24h") -> Dict[str, Any]:
        """Get detailed performance analytics"""
        # Filter sessions based on time period
        cutoff_time = datetime.utcnow()
        if time_period == "24h":
            cutoff_time -= timedelta(hours=24)
        elif time_period == "7d":
            cutoff_time -= timedelta(days=7)
        elif time_period == "30d":
            cutoff_time -= timedelta(days=30)
        
        relevant_sessions = [s for s in self.session_history if s.start_time >= cutoff_time]
        
        if not relevant_sessions:
            return {"message": "No data available for the specified period"}
        
        # Calculate analytics
        total_sessions = len(relevant_sessions)
        successful_sessions = len([s for s in relevant_sessions if s.final_confidence > 0.7])
        
        avg_processing_time = sum(
            s.metrics.get("total_processing_time_ms", 0) for s in relevant_sessions if s.metrics
        ) / len([s for s in relevant_sessions if s.metrics])
        
        avg_confidence = sum(s.final_confidence for s in relevant_sessions) / total_sessions
        
        # Stage performance
        stage_performance = {}
        for session in relevant_sessions:
            for step in session.steps:
                stage = step['stage']
                if stage not in stage_performance:
                    stage_performance[stage] = {"total_time": 0, "count": 0, "avg_confidence": 0}
                
                stage_performance[stage]["total_time"] += step.get("processing_time_ms", 0)
                stage_performance[stage]["count"] += 1
                stage_performance[stage]["avg_confidence"] += step["confidence_score"]
        
        # Calculate averages
        for stage, data in stage_performance.items():
            if data["count"] > 0:
                data["avg_time"] = data["total_time"] / data["count"]
                data["avg_confidence"] = data["avg_confidence"] / data["count"]
        
        return {
            "period": time_period,
            "total_sessions": total_sessions,
            "success_rate": successful_sessions / total_sessions,
            "average_processing_time_ms": avg_processing_time,
            "average_confidence": avg_confidence,
            "stage_performance": stage_performance,
            "query_complexity_distribution": self._get_complexity_distribution(relevant_sessions),
            "confidence_distribution": self._get_confidence_distribution(relevant_sessions),
            "top_insights": self._get_top_insights(relevant_sessions)
        }
    
    def _get_complexity_distribution(self, sessions: List[ReasoningSession]) -> Dict[str, int]:
        """Get distribution of query complexities"""
        distribution = {"low": 0, "medium": 0, "high": 0}
        
        for session in sessions:
            if session.metrics:
                complexity = session.metrics.get("query_complexity_score", 0.5)
                if complexity < 0.3:
                    distribution["low"] += 1
                elif complexity < 0.7:
                    distribution["medium"] += 1
                else:
                    distribution["high"] += 1
        
        return distribution
    
    def _get_confidence_distribution(self, sessions: List[ReasoningSession]) -> Dict[str, int]:
        """Get distribution of confidence levels"""
        distribution = {"very_low": 0, "low": 0, "medium": 0, "high": 0, "very_high": 0}
        
        for session in sessions:
            confidence = session.final_confidence
            if confidence < 0.25:
                distribution["very_low"] += 1
            elif confidence < 0.5:
                distribution["low"] += 1
            elif confidence < 0.75:
                distribution["medium"] += 1
            elif confidence < 0.9:
                distribution["high"] += 1
            else:
                distribution["very_high"] += 1
        
        return distribution
    
    def _get_top_insights(self, sessions: List[ReasoningSession]) -> List[str]:
        """Get most frequently discovered insights"""
        insight_counts = {}
        
        for session in sessions:
            for insight in session.insights_discovered:
                # Normalize insight (simple keyword extraction)
                key_words = set(insight.lower().split())
                
                # Find existing similar insights
                found_similar = False
                for existing_insight in insight_counts:
                    existing_words = set(existing_insight.lower().split())
                    if len(key_words & existing_words) > 2:  # 2+ common words
                        insight_counts[existing_insight] += 1
                        found_similar = True
                        break
                
                if not found_similar:
                    insight_counts[insight] = 1
        
        # Return top insights
        sorted_insights = sorted(insight_counts.items(), key=lambda x: x[1], reverse=True)
        return [insight for insight, count in sorted_insights[:5]]
