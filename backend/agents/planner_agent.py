# Query intent analysis
# agents/planner_agent.py
"""
Query planning agent that analyzes intent and identifies ambiguities.
Ensures queries are well-formed before SQL generation.
"""

from typing import Dict, Any, List, Optional, Tuple
from agents.base import BaseAgent, AgentContext, AgentResult
from enum import Enum
import re


class QueryIntent(str, Enum):
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    COMPARISON = "comparison"
    TREND = "trend"
    DETAIL = "detail"
    EXPLORATION = "exploration"


class PlannerAgent(BaseAgent):
    """
    Analyzes query intent and identifies clarification needs.
    Improves query precision through proactive disambiguation.
    """
    
    def __init__(self):
        super().__init__("PlannerAgent")
        self.ambiguity_patterns = self._initialize_ambiguity_patterns()
        self.intent_patterns = self._initialize_intent_patterns()
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Plans query execution by identifying intent and ambiguities.
        Returns clarification questions if query is unclear.
        """
        
        # Detect query intent
        intent = self._detect_intent(context.query)
        
        # Check for ambiguities
        ambiguities = self._detect_ambiguities(context.query, context.metadata)
        
        # Generate clarification questions if needed
        clarifications = []
        if ambiguities:
            clarifications = self._generate_clarifications(ambiguities, context)
        
        # Extract query components
        components = self._extract_query_components(context.query)
        
        # Build execution plan
        plan = self._build_execution_plan(intent, components, context.metadata)
        
        return AgentResult(
            success=True,
            data={
                "intent": intent,
                "needs_clarification": len(clarifications) > 0,
                "clarification_questions": clarifications,
                "components": components,
                "execution_plan": plan,
                "complexity_score": self._calculate_complexity(components)
            },
            confidence=0.9 if not clarifications else 0.6
        )
    
    def _initialize_ambiguity_patterns(self) -> List[Dict[str, Any]]:
        """Patterns that indicate ambiguous queries"""
        return [
            {
                "pattern": re.compile(r'\b(recent|latest|new)\b', re.I),
                "type": "temporal",
                "clarification": "What time period do you consider 'recent'?"
            },
            {
                "pattern": re.compile(r'\b(top|best|worst|bottom)\b(?!\s+\d+)', re.I),
                "type": "ranking",
                "clarification": "How many top results would you like to see?"
            },
            {
                "pattern": re.compile(r'\b(large|small|high|low|big)\b', re.I),
                "type": "threshold",
                "clarification": "What threshold defines '{match}' in your context?"
            },
            {
                "pattern": re.compile(r'\b(performance|efficiency|quality)\b', re.I),
                "type": "metric",
                "clarification": "Which specific metric should I use to measure {match}?"
            },
            {
                "pattern": re.compile(r'\b(compare|versus|vs|difference)\b', re.I),
                "type": "comparison",
                "clarification": "What specific aspects would you like to compare?"
            }
        ]
    
    def _initialize_intent_patterns(self) -> Dict[QueryIntent, List[re.Pattern]]:
        """Patterns for detecting query intent"""
        return {
            QueryIntent.AGGREGATION: [
                re.compile(r'\b(total|sum|count|average|avg|mean|max|min)\b', re.I),
                re.compile(r'\b(how many|how much)\b', re.I),
                re.compile(r'\b(aggregate|group by)\b', re.I)
            ],
            QueryIntent.FILTERING: [
                re.compile(r'\b(where|filter|only|exclude|specific)\b', re.I),
                re.compile(r'\b(with|without|having)\b', re.I),
                re.compile(r'=|!=|>|<|>=|<=', re.I)
            ],
            QueryIntent.COMPARISON: [
                re.compile(r'\b(compare|versus|vs|difference|between)\b', re.I),
                re.compile(r'\b(higher|lower|better|worse|more|less)\b', re.I)
            ],
            QueryIntent.TREND: [
                re.compile(r'\b(trend|over time|growth|change|evolution)\b', re.I),
                re.compile(r'\b(by month|by year|by quarter|daily|weekly)\b', re.I),
                re.compile(r'\b(historical|timeline|progression)\b', re.I)
            ],
            QueryIntent.DETAIL: [
                re.compile(r'\b(show|list|display|get|find)\b', re.I),
                re.compile(r'\b(details|records|rows|data)\b', re.I),
                re.compile(r'\b(specific|individual|each)\b', re.I)
            ]
        }
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Determines the primary intent of the query"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern.search(query))
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        
        return QueryIntent.EXPLORATION
    
    def _detect_ambiguities(self, query: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifies ambiguous elements in the query"""
        ambiguities = []
        
        for ambiguity in self.ambiguity_patterns:
            match = ambiguity["pattern"].search(query)
            if match:
                # Check if context resolves the ambiguity
                if not self._is_resolved_by_context(ambiguity["type"], metadata):
                    ambiguities.append({
                        "type": ambiguity["type"],
                        "match": match.group(),
                        "clarification": ambiguity["clarification"].format(match=match.group())
                    })
        
        # Check for missing time context in trend queries
        if "trend" in query.lower() and not re.search(r'\d{4}|\d{1,2}/\d{1,2}', query):
            ambiguities.append({
                "type": "temporal",
                "match": "time period",
                "clarification": "What time period should I analyze for the trend?"
            })
        
        return ambiguities
    
    def _is_resolved_by_context(self, ambiguity_type: str, metadata: Dict[str, Any]) -> bool:
        """Checks if conversation context resolves the ambiguity"""
        context = metadata.get("conversation_context", {})
        
        if ambiguity_type == "temporal":
            # Check if previous queries established time context
            return bool(context.get("previous_filters", []))
        
        if ambiguity_type == "ranking":
            # Check if default limits are acceptable
            return metadata.get("default_limit_ok", False)
        
        return False
    
    def _generate_clarifications(self, ambiguities: List[Dict[str, Any]], 
                               context: AgentContext) -> List[str]:
        """Generates user-friendly clarification questions"""
        questions = []
        
        # Group similar ambiguities
        by_type = {}
        for amb in ambiguities:
            by_type.setdefault(amb["type"], []).append(amb)
        
        # Generate questions by type
        for amb_type, items in by_type.items():
            if amb_type == "temporal":
                questions.append(
                    "Please specify the time period (e.g., 'last 30 days', '2024', 'January to March')"
                )
            elif amb_type == "ranking":
                questions.append(
                    "How many results would you like to see? (e.g., 'top 10', 'all results')"
                )
            elif amb_type == "threshold":
                values = [item["match"] for item in items]
                questions.append(
                    f"What values do you consider {' and '.join(values)}? Please provide specific thresholds."
                )
            elif amb_type == "metric":
                questions.append(
                    "Which specific metric should I use? (e.g., 'revenue', 'count', 'conversion rate')"
                )
        
        # Add dataset-specific clarifications
        if not context.metadata.get("schema"):
            questions.append(
                "Which columns should I focus on for this analysis?"
            )
        
        return questions[:3]  # Limit to 3 questions
    
    def _extract_query_components(self, query: str) -> Dict[str, List[str]]:
        """Extracts key components from the query"""
        components = {
            "metrics": [],
            "dimensions": [],
            "filters": [],
            "time_references": [],
            "sort_orders": [],
            "limits": []
        }
        
        # Extract metrics (aggregations)
        metric_patterns = [
            (r'\b(sum|total)\s+(?:of\s+)?(\w+)', 'sum'),
            (r'\b(count|number)\s+(?:of\s+)?(\w+)', 'count'),
            (r'\b(average|avg|mean)\s+(?:of\s+)?(\w+)', 'avg'),
            (r'\b(max|maximum)\s+(?:of\s+)?(\w+)', 'max'),
            (r'\b(min|minimum)\s+(?:of\s+)?(\w+)', 'min')
        ]
        
        for pattern, func in metric_patterns:
            matches = re.finditer(pattern, query, re.I)
            for match in matches:
                components["metrics"].append(f"{func}({match.group(2)})")
        
        # Extract time references
        time_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',
            r'\b(?:last|next|this)\s+(?:day|week|month|quarter|year)s?\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        ]
        
        for pattern in time_patterns:
            matches = re.finditer(pattern, query, re.I)
            components["time_references"].extend(match.group() for match in matches)
        
        # Extract sort orders
        if re.search(r'\b(top|highest|largest|most)\b', query, re.I):
            components["sort_orders"].append("DESC")
        elif re.search(r'\b(bottom|lowest|smallest|least)\b', query, re.I):
            components["sort_orders"].append("ASC")
        
        # Extract limits
        limit_match = re.search(r'\b(?:top|bottom|first|last)\s+(\d+)\b', query, re.I)
        if limit_match:
            components["limits"].append(limit_match.group(1))
        
        return components
    
    def _build_execution_plan(self, intent: QueryIntent, components: Dict[str, List[str]], 
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Builds a structured execution plan for the query"""
        plan = {
            "intent": intent,
            "steps": [],
            "optimizations": [],
            "warnings": []
        }
        
        # Define steps based on intent
        if intent == QueryIntent.AGGREGATION:
            plan["steps"] = [
                "Identify grouping dimensions",
                "Calculate aggregated metrics",
                "Apply filters if specified",
                "Sort by metric values"
            ]
            if not components["dimensions"]:
                plan["warnings"].append("No grouping dimension specified - will show overall totals")
        
        elif intent == QueryIntent.TREND:
            plan["steps"] = [
                "Extract time series data",
                "Aggregate by time period",
                "Calculate period-over-period changes",
                "Identify trend direction"
            ]
            if not components["time_references"]:
                plan["warnings"].append("No specific time period mentioned - will use all available data")
        
        elif intent == QueryIntent.COMPARISON:
            plan["steps"] = [
                "Identify comparison dimensions",
                "Calculate metrics for each segment",
                "Compute differences/ratios",
                "Rank by difference magnitude"
            ]
        
        # Add optimizations
        if len(components["metrics"]) > 3:
            plan["optimizations"].append("Multiple metrics - consider using subqueries for efficiency")
        
        if metadata.get("domain") == "sales" and "revenue" in str(components["metrics"]):
            plan["optimizations"].append("Sales domain detected - will include relevant business metrics")
        
        return plan
    
    def _calculate_complexity(self, components: Dict[str, List[str]]) -> float:
        """Calculates query complexity score (0-1)"""
        complexity = 0.0
        
        # Factor in different components
        complexity += len(components["metrics"]) * 0.15
        complexity += len(components["dimensions"]) * 0.10
        complexity += len(components["filters"]) * 0.10
        complexity += len(components["time_references"]) * 0.15
        
        # Cap at 1.0
        return min(complexity, 1.0)