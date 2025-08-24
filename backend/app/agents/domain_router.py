import logging
from typing import Dict, Any, Optional
from app.models.query import QueryIntent

logger = logging.getLogger(__name__)

class DomainRouter:
    """Routes queries to appropriate domain-specific handlers"""
    
    def __init__(self):
        self.domains = {
            "sales": ["revenue", "sales", "orders", "transactions", "customers"],
            "finance": ["profit", "cost", "expense", "budget", "margin"],
            "marketing": ["campaign", "conversion", "acquisition", "retention"],
            "operations": ["inventory", "supply", "logistics", "efficiency"],
            "hr": ["employee", "salary", "headcount", "turnover", "hiring"]
        }
    
    async def route(self, question: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the domain and routing strategy for a query"""
        question_lower = question.lower()
        
        # Detect domain
        detected_domain = "general"
        domain_confidence = 0.5
        
        for domain, keywords in self.domains.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            if matches > 0:
                confidence = matches / len(keywords)
                if confidence > domain_confidence:
                    detected_domain = domain
                    domain_confidence = confidence
        
        # Determine query intent
        intent = self._detect_intent(question_lower)
        
        # Check for special handling needs
        special_handling = []
        if "compare" in question_lower or "vs" in question_lower:
            special_handling.append("comparison")
        if any(word in question_lower for word in ["trend", "over time", "monthly", "daily"]):
            special_handling.append("time_series")
        if "top" in question_lower or "bottom" in question_lower:
            special_handling.append("ranking")
        
        routing_result = {
            "domain": detected_domain,
            "confidence": domain_confidence,
            "intent": intent,
            "special_handling": special_handling,
            "recommended_agent": self._get_recommended_agent(detected_domain, intent),
            "context_requirements": self._get_context_requirements(detected_domain)
        }
        
        logger.info(f"Domain routing result: {routing_result}")
        return routing_result
    
    def _detect_intent(self, question: str) -> QueryIntent:
        """Detect the primary intent of the query"""
        if any(word in question for word in ["sum", "total", "average", "count", "max", "min"]):
            return QueryIntent.AGGREGATE
        elif any(word in question for word in ["where", "filter", "only", "exclude"]):
            return QueryIntent.FILTER
        elif any(word in question for word in ["compare", "vs", "versus", "difference"]):
            return QueryIntent.COMPARISON
        elif any(word in question for word in ["trend", "over time", "by month", "by day"]):
            return QueryIntent.TIME_SERIES
        else:
            return QueryIntent.UNKNOWN
    
    def _get_recommended_agent(self, domain: str, intent: QueryIntent) -> str:
        """Get the recommended processing agent based on domain and intent"""
        if intent == QueryIntent.TIME_SERIES:
            return "time_series_analyzer"
        elif intent == QueryIntent.COMPARISON:
            return "comparison_analyzer"
        elif domain in ["finance", "sales"]:
            return "financial_analyzer"
        else:
            return "general_analyzer"
    
    def _get_context_requirements(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific context requirements"""
        context = {
            "sales": {
                "key_metrics": ["revenue", "units_sold", "average_order_value"],
                "time_dimensions": ["order_date", "ship_date"],
                "key_dimensions": ["product", "customer", "region"]
            },
            "finance": {
                "key_metrics": ["profit", "cost", "margin_percentage"],
                "time_dimensions": ["transaction_date", "period"],
                "key_dimensions": ["account", "cost_center", "department"]
            }
        }
        return context.get(domain, {})