# Executive summaries
# agents/report_agent.py
"""
Executive report generation agent that creates session summaries.
Provides high-level insights from query history.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from agents.base import BaseAgent, AgentContext, AgentResult
from services.cache_service import CacheService


class ReportAgent(BaseAgent):
    """
    Generates executive summaries from user session history.
    Synthesizes insights across multiple queries.
    """
    
    def __init__(self, cache_service: CacheService):
        super().__init__("ReportAgent")
        self.cache = cache_service
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Generates comprehensive session report.
        Aggregates insights from conversation history.
        """
        
        # Get conversation history
        conversation_history = await self._get_conversation_history(context)
        
        if not conversation_history:
            return AgentResult(
                success=True,
                data={
                    "report_type": "empty",
                    "summary": "No conversation history available for reporting."
                }
            )
        
        # Analyze conversation patterns
        analysis = self._analyze_conversation(conversation_history)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(analysis)
        
        # Key insights across queries
        cross_query_insights = self._extract_cross_query_insights(analysis)
        
        # Recommendations
        recommendations = self._generate_recommendations(analysis)
        
        # Create structured report
        report = {
            "report_id": f"report_{context.session_id[:8]}",
            "generated_at": datetime.utcnow().isoformat(),
            "conversation_id": context.conversation_id,
            "summary": executive_summary,
            "statistics": {
                "total_queries": len(conversation_history),
                "unique_datasets": len(set(q.get("dataset_id") for q in conversation_history)),
                "time_span": self._calculate_time_span(conversation_history),
                "domains_explored": analysis["domains"],
                "total_insights": analysis["total_insights"]
            },
            "key_findings": cross_query_insights,
            "query_progression": self._analyze_query_progression(conversation_history),
            "recommendations": recommendations,
            "next_steps": self._suggest_next_steps(analysis)
        }
        
        return AgentResult(
            success=True,
            data=report,
            confidence=0.9
        )
    
    async def _get_conversation_history(self, context: AgentContext) -> List[Dict[str, Any]]:
        """Retrieves full conversation history"""
        if not context.conversation_id:
            return []
        
        # Get from cache
        try:
            cached = await self.cache.get(f"conversation:{context.conversation_id}")
            if cached:
                conversation = json.loads(cached)
                return conversation.get("turns", [])
        except Exception as e:
            self.logger.warning("Failed to retrieve conversation history", error=str(e))
        
        return []
    
    def _analyze_conversation(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes conversation patterns and extracts metrics"""
        
        analysis = {
            "domains": set(),
            "query_types": [],
            "metrics_explored": set(),
            "tables_accessed": set(),
            "total_insights": 0,
            "key_topics": [],
            "complexity_progression": []
        }
        
        for turn in history:
            # Extract metadata
            metadata = turn.get("metadata", {})
            
            # Domains
            if metadata.get("domain"):
                analysis["domains"].add(metadata["domain"])
            
            # Query types
            if metadata.get("intent"):
                analysis["query_types"].append(metadata["intent"])
            
            # Metrics
            if metadata.get("metrics"):
                analysis["metrics_explored"].update(metadata["metrics"])
            
            # Insights count
            if metadata.get("insights_found"):
                analysis["total_insights"] += metadata["insights_found"]
            
            # Complexity
            if metadata.get("complexity_score"):
                analysis["complexity_progression"].append(metadata["complexity_score"])
        
        # Convert sets to lists for JSON serialization
        analysis["domains"] = list(analysis["domains"])
        analysis["metrics_explored"] = list(analysis["metrics_explored"])
        analysis["tables_accessed"] = list(analysis["tables_accessed"])
        
        # Identify key topics
        analysis["key_topics"] = self._extract_key_topics(history)
        
        return analysis
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> str:
        """Generates high-level executive summary"""
        
        parts = []
        
        # Opening
        query_count = len(analysis.get("query_types", []))
        domains = analysis.get("domains", [])
        
        if query_count == 1:
            parts.append("This session consisted of a single analytical query")
        else:
            parts.append(f"This session involved {query_count} analytical queries")
        
        if domains:
            parts.append(f"focusing on {', '.join(domains)} data")
        
        # Key achievements
        insights = analysis.get("total_insights", 0)
        if insights > 0:
            parts.append(f". The analysis uncovered {insights} actionable insights")
        
        # Query pattern
        query_types = analysis.get("query_types", [])
        if query_types:
            most_common = max(set(query_types), key=query_types.count)
            parts.append(f", with a primary focus on {most_common} analysis")
        
        # Complexity trend
        complexity_progression = analysis.get("complexity_progression", [])
        if len(complexity_progression) > 1:
            if complexity_progression[-1] > complexity_progression[0]:
                parts.append(". The analysis progressed from simple to more complex queries")
            else:
                parts.append(". The analysis maintained consistent complexity")
        
        parts.append(".")
        
        return "".join(parts)
    
    def _extract_cross_query_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Identifies insights that span multiple queries"""
        
        insights = []
        
        # Domain focus
        domains = analysis.get("domains", [])
        if len(domains) > 1:
            insights.append(f"Cross-functional analysis spanning {', '.join(domains)} domains")
        elif domains:
            insights.append(f"Deep dive into {domains[0]} domain")
        
        # Metric patterns
        metrics = analysis.get("metrics_explored", [])
        if len(metrics) > 3:
            insights.append(f"Comprehensive analysis covering {len(metrics)} different metrics")
        
        # Query progression
        query_types = analysis.get("query_types", [])
        if len(set(query_types)) > 2:
            insights.append("Multi-faceted analysis using various analytical approaches")
        
        # Complexity evolution
        complexity = analysis.get("complexity_progression", [])
        if complexity and max(complexity) > 0.7:
            insights.append("Advanced analytical techniques employed for deeper insights")
        
        return insights[:4]  # Limit to top 4
    
    def _analyze_query_progression(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes how queries evolved during the session"""
        
        if len(history) < 2:
            return {"type": "single_query", "description": "Single query session"}
        
        # Look for patterns
        progression_type = "exploratory"
        
        # Check if queries build on each other
        has_refinement = any("filter" in q.get("query", "").lower() for q in history[1:])
        has_drill_down = any("group by" in q.get("sql", "").lower() for q in history)
        
        if has_refinement and has_drill_down:
            progression_type = "iterative_refinement"
            description = "Queries progressively refined the analysis"
        elif has_drill_down:
            progression_type = "drill_down"
            description = "Analysis drilled down into specific segments"
        elif len(set(q.get("dataset_id") for q in history)) > 1:
            progression_type = "multi_dataset"
            description = "Analysis spanned multiple datasets"
        else:
            description = "Exploratory analysis across different aspects"
        
        return {
            "type": progression_type,
            "description": description,
            "query_count": len(history)
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generates actionable recommendations based on session analysis"""
        
        recommendations = []
        
        # Based on domains explored
        domains = analysis.get("domains", [])
        if "sales" in domains and "marketing" not in domains:
            recommendations.append("Consider analyzing marketing data to understand sales drivers")
        
        if "finance" in domains and analysis.get("total_insights", 0) < 3:
            recommendations.append("Explore financial trends over time for budget optimization")
        
        # Based on query types
        query_types = analysis.get("query_types", [])
        if "aggregation" in query_types and "trend" not in query_types:
            recommendations.append("Analyze temporal trends to understand data evolution")
        
        if "comparison" not in query_types and len(analysis.get("metrics_explored", [])) > 2:
            recommendations.append("Compare metrics across segments to identify opportunities")
        
        # Based on complexity
        complexity = analysis.get("complexity_progression", [])
        if complexity and max(complexity) < 0.5:
            recommendations.append("Consider more advanced analyses like correlations or predictions")
        
        # Generic valuable recommendations
        if not recommendations:
            recommendations = [
                "Set up automated alerts for key metrics",
                "Create a dashboard for ongoing monitoring",
                "Schedule regular analysis reviews"
            ]
        
        return recommendations[:3]
    
    def _suggest_next_steps(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggests concrete next steps for the user"""
        
        next_steps = []
        
        # Based on insights found
        if analysis.get("total_insights", 0) > 5:
            next_steps.append("Prioritize top insights for action planning")
            next_steps.append("Share findings with relevant stakeholders")
        
        # Based on topics
        topics = analysis.get("key_topics", [])
        if topics:
            next_steps.append(f"Deep dive into {topics[0]} for detailed analysis")
        
        # Based on query progression
        if len(analysis.get("query_types", [])) > 3:
            next_steps.append("Consolidate findings into an executive presentation")
        
        # Always suggest
        next_steps.append("Export data for further analysis")
        
        return next_steps[:3]
    
    def _extract_key_topics(self, history: List[Dict[str, Any]]) -> List[str]:
        """Extracts key topics from query history"""
        
        # Simple keyword extraction
        topics = []
        all_queries = " ".join(q.get("query", "") for q in history).lower()
        
        # Business topics
        topic_keywords = {
            "revenue": ["revenue", "sales", "income"],
            "costs": ["cost", "expense", "spend"],
            "customers": ["customer", "client", "user"],
            "performance": ["performance", "efficiency", "productivity"],
            "growth": ["growth", "increase", "trend"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_queries for keyword in keywords):
                topics.append(topic)
        
        return topics[:3]
    
    def _calculate_time_span(self, history: List[Dict[str, Any]]) -> str:
        """Calculates session time span"""
        
        if not history:
            return "0 minutes"
        
        timestamps = []
        for turn in history:
            if turn.get("timestamp"):
                try:
                    timestamps.append(datetime.fromisoformat(turn["timestamp"]))
                except:
                    continue
        
        if len(timestamps) < 2:
            return "Single query"
        
        duration = max(timestamps) - min(timestamps)
        
        if duration.total_seconds() < 60:
            return f"{int(duration.total_seconds())} seconds"
        elif duration.total_seconds() < 3600:
            return f"{int(duration.total_seconds() / 60)} minutes"
        else:
            return f"{duration.total_seconds() / 3600:.1f} hours"