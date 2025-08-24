import logging
from typing import Dict, Any, List, Optional
from app.models.query import QueryPlan, QueryIntent, ClarificationRequest, AmbiguityType

logger = logging.getLogger(__name__)

class QueryPlanner:
    """Plans query execution and detects ambiguities"""
    
    def __init__(self):
        self.temporal_keywords = {
            "recent": ["last 7 days", "last 30 days", "last quarter"],
            "this year": ["year to date", "calendar year", "fiscal year"],
            "last period": ["last month", "last quarter", "last year"]
        }
        
        self.metric_ambiguities = {
            "sales": ["revenue", "units sold", "transactions"],
            "performance": ["revenue", "profit", "efficiency"],
            "growth": ["absolute growth", "percentage growth", "year-over-year"]
        }
    
    async def plan(
        self, 
        question: str, 
        schema: Dict[str, Any],
        routing: Dict[str, Any]
    ) -> QueryPlan:
        """Create a query execution plan"""
        
        # Detect ambiguities
        clarifications = await self._detect_ambiguities(question, schema)
        
        # Estimate complexity
        complexity = self._estimate_complexity(question, routing["intent"])
        
        # Check if joins are needed
        requires_joins = self._check_join_requirements(question, schema)
        
        # Estimate result size
        estimated_rows = self._estimate_result_size(question, routing["intent"])
        
        plan = QueryPlan(
            intent=routing["intent"],
            complexity=complexity,
            requires_joins=requires_joins,
            estimated_rows=estimated_rows,
            clarifications_needed=clarifications if clarifications else None
        )
        
        logger.info(f"Query plan created: {plan}")
        return plan
    
    async def _detect_ambiguities(
        self, 
        question: str, 
        schema: Dict[str, Any]
    ) -> Optional[List[ClarificationRequest]]:
        """Detect ambiguities in the question"""
        clarifications = []
        question_lower = question.lower()
        
        # Check temporal ambiguities
        for ambiguous_term, options in self.temporal_keywords.items():
            if ambiguous_term in question_lower:
                clarifications.append(ClarificationRequest(
                    ambiguity_type=AmbiguityType.TEMPORAL,
                    message=f"When you say '{ambiguous_term}', which time period do you mean?",
                    options=options,
                    original_question=question
                ))
        
        # Check metric ambiguities
        for ambiguous_term, options in self.metric_ambiguities.items():
            if ambiguous_term in question_lower:
                clarifications.append(ClarificationRequest(
                    ambiguity_type=AmbiguityType.METRIC,
                    message=f"Which specific metric for '{ambiguous_term}'?",
                    options=options,
                    original_question=question
                ))
        
        # Check for entity ambiguities (e.g., "customers" could mean count or list)
        if "customers" in question_lower and not any(
            word in question_lower for word in ["count", "number", "how many", "list"]
        ):
            clarifications.append(ClarificationRequest(
                ambiguity_type=AmbiguityType.ENTITY,
                message="Do you want to count customers or see a list?",
                options=["Count of customers", "List of customers", "Customer details"],
                original_question=question
            ))
        
        return clarifications if clarifications else None
    
    def _estimate_complexity(self, question: str, intent: QueryIntent) -> str:
        """Estimate query complexity"""
        question_lower = question.lower()
        
        # Complex indicators
        complex_indicators = ["compare", "correlation", "trend analysis", "forecast"]
        if any(indicator in question_lower for indicator in complex_indicators):
            return "complex"
        
        # Moderate indicators
        moderate_indicators = ["group by", "over time", "by category", "breakdown"]
        if any(indicator in question_lower for indicator in moderate_indicators):
            return "moderate"
        
        # Simple queries
        if intent in [QueryIntent.FILTER, QueryIntent.AGGREGATE]:
            return "simple"
        
        return "moderate"
    
    def _check_join_requirements(self, question: str, schema: Dict[str, Any]) -> bool:
        """Check if the query requires joins"""
        # Simple heuristic: if question mentions multiple entities
        entities = ["customer", "product", "order", "transaction", "category"]
        mentioned_entities = sum(1 for entity in entities if entity in question.lower())
        return mentioned_entities > 1
    
    def _estimate_result_size(self, question: str, intent: QueryIntent) -> Optional[int]:
        """Estimate the number of rows in the result"""
        question_lower = question.lower()
        
        if "count" in question_lower or "how many" in question_lower:
            return 1  # Single aggregation result
        
        if "top" in question_lower:
            # Extract number if specified
            import re
            match = re.search(r"top (\d+)", question_lower)
            if match:
                return int(match.group(1))
            return 10  # Default top N
        
        if intent == QueryIntent.AGGREGATE:
            return 10  # Grouped results
        
        return 100  # Default estimate