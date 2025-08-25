# Query intent analysis
# agents/planner_agent.py
"""
Query planning agent that analyzes intent and identifies ambiguities.
Ensures queries are well-formed before SQL generation.
"""

from typing import Dict, Any, List, Optional, Tuple
from agents.base import BaseAgent, AgentContext, AgentResult
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime
import re


class QueryIntent(str, Enum):
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    COMPARISON = "comparison"
    TREND = "trend"
    DETAIL = "detail"
    EXPLORATION = "exploration"


class BusinessContext(BaseModel):
    """Business context for domain-aware query planning"""
    domain: Optional[str] = None  # e.g., "sales", "finance", "marketing"
    key_metrics: List[str] = []  # Primary metrics for this domain
    common_dimensions: List[str] = []  # Typical grouping columns
    time_periods: List[str] = []  # Common time ranges
    business_rules: Dict[str, Any] = {}  # Domain-specific rules


class AmbiguityResolution(BaseModel):
    """Represents a resolved ambiguity with confidence scoring"""
    ambiguity_type: str
    original_term: str
    resolved_term: str
    confidence: float
    reasoning: str
    business_context: Optional[Dict[str, Any]] = None


class IntentConfidence(BaseModel):
    """Confidence scoring for detected intent"""
    intent: QueryIntent
    confidence: float
    evidence: List[str] = []  # What led to this intent classification
    alternative_intents: List[Tuple[QueryIntent, float]] = []  # Other possible intents


class PlannerAgent(BaseAgent):
    """
    Analyzes query intent and identifies clarification needs.
    Improves query precision through proactive disambiguation.
    """
    
    def __init__(self):
        super().__init__("PlannerAgent")
        self.ambiguity_patterns = self._initialize_ambiguity_patterns()
        self.intent_patterns = self._initialize_intent_patterns()
        # Add semantic layer support
        from services.semantic_layer_service import SemanticLayerService
        self.semantic_layer = SemanticLayerService()
        
        # Enhanced intelligence features
        self.business_contexts = self._initialize_business_contexts()
        self.confidence_threshold = 0.7  # Minimum confidence for automatic resolution
        self.context_weight = 0.3  # How much conversation context influences decisions
    
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
        """Identifies ambiguous elements in the query with enhanced detection"""
        ambiguities = []
        
        for ambiguity in self.ambiguity_patterns:
            match = ambiguity["pattern"].search(query)
            if match:
                # Check if context resolves the ambiguity
                if not self._is_resolved_by_context(ambiguity["type"], metadata):
                    ambiguities.append({
                        "type": ambiguity["type"],
                        "match": match.group(),
                        "clarification": ambiguity["clarification"].format(match=match.group()),
                        "confidence": 0.8
                    })
        
        # Enhanced temporal ambiguity detection
        if any(word in query.lower() for word in ["trend", "over time", "change", "growth"]) and not re.search(r'\d{4}|\d{1,2}/\d{1,2}|last \d+|past \d+', query):
            ambiguities.append({
                "type": "temporal",
                "match": "time period",
                "clarification": "What time period should I analyze for the trend?",
                "confidence": 0.9,
                "suggestions": ["last 30 days", "2024 only", "year over year", "quarterly"]
            })
        
        # Check for vague aggregation requests
        if re.search(r'\b(total|sum|average)\b', query, re.I) and not re.search(r'\bby\s+\w+|group\s+by|per\s+\w+', query, re.I):
            ambiguities.append({
                "type": "aggregation",
                "match": "aggregation scope",
                "clarification": "Should I group this by a specific category or show the overall total?",
                "confidence": 0.7,
                "suggestions": ["by category", "by month", "by region", "overall total"]
            })
        
        # Detect comparison ambiguities
        if re.search(r'\b(compare|vs|versus|difference)\b', query, re.I) and query.count(' ') < 6:
            ambiguities.append({
                "type": "comparison",
                "match": "comparison dimensions",
                "clarification": "What specific aspects or groups should I compare?",
                "confidence": 0.8
            })
        
        # Check for missing measurement context
        measurement_words = ['performance', 'success', 'quality', 'efficiency']
        if any(word in query.lower() for word in measurement_words):
            # Look for specific metrics
            if not re.search(r'\b(revenue|sales|profit|count|rate|score|percentage)\b', query, re.I):
                ambiguities.append({
                    "type": "metric",
                    "match": "measurement metric",
                    "clarification": "Which specific metric should I use to measure this?",
                    "confidence": 0.75,
                    "suggestions": ["revenue", "count", "percentage", "growth rate"]
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
        """Generates user-friendly clarification questions with examples"""
        questions = []
        
        # Sort ambiguities by confidence (highest first)
        ambiguities.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
        
        # Group similar ambiguities
        by_type = {}
        for amb in ambiguities:
            by_type.setdefault(amb["type"], []).append(amb)
        
        # Generate questions by type with better context
        for amb_type, items in by_type.items():
            if amb_type == "temporal":
                # Include suggestions if available
                suggestions = items[0].get("suggestions", ["last 30 days", "2024 only", "quarterly"])
                questions.append(
                    f"📅 Please specify the time period. Try: {', '.join(suggestions[:3])}"
                )
                
            elif amb_type == "ranking":
                questions.append(
                    "🔢 How many results would you like? (e.g., 'top 10', 'show all', 'bottom 5')"
                )
                
            elif amb_type == "threshold":
                values = [item["match"] for item in items]
                questions.append(
                    f"📊 What defines '{' and '.join(values)}' in your context? Please provide specific numbers or ranges."
                )
                
            elif amb_type == "metric":
                # Look for available suggestions
                suggestions = items[0].get("suggestions", ["revenue", "count", "percentage"])
                questions.append(
                    f"📈 Which metric should I use? Options: {', '.join(suggestions)}"
                )
                
            elif amb_type == "aggregation":
                suggestions = items[0].get("suggestions", ["by category", "by month", "overall total"])
                questions.append(
                    f"📋 Should I break this down? Options: {', '.join(suggestions)}"
                )
                
            elif amb_type == "comparison":
                questions.append(
                    "⚖️ What specific aspects should I compare? (e.g., 'this year vs last year', 'by region', 'by product')"
                )
        
        # Add contextual suggestions based on query content
        query_lower = context.query.lower()
        
        # If no specific ambiguities but query is very short/vague
        if not questions and len(context.query.split()) < 4:
            questions.append(
                "🤔 Could you be more specific? For example: 'Show me sales by region' or 'Compare revenue this year vs last year'"
            )
        
        # Add helpful context about available data if needed
        if not context.metadata.get("schema") and len(questions) < 2:
            questions.append(
                "💡 Tip: You can ask me about trends, comparisons, totals, or specific data points in your dataset."
            )
        
        return questions[:3]  # Limit to 3 questions for better UX
    
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
    
    def _initialize_business_contexts(self) -> Dict[str, BusinessContext]:
        """Initialize domain-specific business contexts for better clarifications"""
        return {
            "sales": BusinessContext(
                domain="sales",
                key_metrics=["revenue", "units_sold", "conversion_rate", "average_order_value"],
                common_dimensions=["region", "product_category", "sales_rep", "channel"],
                time_periods=["monthly", "quarterly", "year_over_year", "season"],
                business_rules={
                    "recent": "last_30_days",
                    "performance": "revenue_growth_rate",
                    "top": "by_revenue_desc"
                }
            ),
            "finance": BusinessContext(
                domain="finance",
                key_metrics=["profit", "margin", "cost", "roi", "burn_rate"],
                common_dimensions=["department", "cost_center", "category", "account"],
                time_periods=["monthly", "quarterly", "fiscal_year"],
                business_rules={
                    "recent": "current_quarter",
                    "performance": "profit_margin",
                    "efficiency": "cost_per_unit"
                }
            ),
            "marketing": BusinessContext(
                domain="marketing",
                key_metrics=["impressions", "clicks", "ctr", "cac", "ltv"],
                common_dimensions=["campaign", "channel", "audience", "geo"],
                time_periods=["daily", "weekly", "campaign_period"],
                business_rules={
                    "recent": "last_7_days",
                    "performance": "conversion_rate",
                    "success": "roi_positive"
                }
            )
        }
    
    def _detect_intent_with_confidence(self, query: str, metadata: Dict[str, Any]) -> IntentConfidence:
        """
        Enhanced intent detection with confidence scoring and evidence tracking.
        """
        intent_scores = {}
        evidence = []
        
        for intent, patterns in self.intent_patterns.items():
            matches = []
            for pattern in patterns:
                match = pattern.search(query)
                if match:
                    matches.append(match.group())
            
            if matches:
                intent_scores[intent] = len(matches)
                evidence.extend([f"Found '{match}' indicating {intent.value}" for match in matches])
        
        # Factor in business context
        domain = self._detect_business_domain(query, metadata)
        if domain and domain in self.business_contexts:
            business_context = self.business_contexts[domain]
            
            # Boost confidence for domain-appropriate intents
            if QueryIntent.AGGREGATION in intent_scores and any(
                metric in query.lower() for metric in business_context.key_metrics
            ):
                intent_scores[QueryIntent.AGGREGATION] *= 1.2
                evidence.append(f"Domain '{domain}' context supports aggregation intent")
        
        if not intent_scores:
            return IntentConfidence(
                intent=QueryIntent.EXPLORATION,
                confidence=0.5,
                evidence=["No clear intent patterns found - defaulting to exploration"]
            )
        
        # Determine primary intent and confidence
        primary_intent = max(intent_scores, key=intent_scores.get)
        max_score = intent_scores[primary_intent]
        total_score = sum(intent_scores.values())
        
        confidence = max_score / max(total_score, 1)
        
        # Build alternative intents
        alternatives = [
            (intent, score / max(total_score, 1)) 
            for intent, score in intent_scores.items() 
            if intent != primary_intent and score > 0
        ]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return IntentConfidence(
            intent=primary_intent,
            confidence=min(confidence, 1.0),
            evidence=evidence,
            alternative_intents=alternatives[:2]  # Top 2 alternatives
        )
    
    def _detect_business_domain(self, query: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Detect business domain from query content and metadata.
        """
        query_lower = query.lower()
        
        # Check for explicit domain indicators
        domain_keywords = {
            "sales": ["revenue", "sales", "orders", "customers", "conversion"],
            "finance": ["profit", "cost", "budget", "margin", "expense", "roi"],
            "marketing": ["campaign", "clicks", "impressions", "ctr", "ads", "leads"]
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        # Check metadata for domain hints
        if "domain" in metadata:
            explicit_domain = metadata["domain"]
            if explicit_domain in domain_scores:
                domain_scores[explicit_domain] += 2  # Boost explicit domain
            else:
                domain_scores[explicit_domain] = 2
        
        # Check conversation context
        conversation_context = metadata.get("conversation_context", {})
        if "domain" in conversation_context:
            context_domain = conversation_context["domain"]
            domain_scores[context_domain] = domain_scores.get(context_domain, 0) + 1
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return None
    
    def _enhanced_generate_clarifications(
        self, 
        ambiguities: List[Dict[str, Any]], 
        context: AgentContext
    ) -> List[str]:
        """
        Enhanced clarification generation with business-aware suggestions.
        """
        questions = []
        detected_domain = self._detect_business_domain(context.query, context.metadata)
        business_context = self.business_contexts.get(detected_domain) if detected_domain else None
        
        # Sort ambiguities by confidence (highest first)
        ambiguities.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
        
        for ambiguity in ambiguities[:3]:  # Limit to top 3 ambiguities
            amb_type = ambiguity["type"]
            
            if amb_type == "temporal" and business_context:
                # Use business-specific time periods
                suggestions = business_context.time_periods
                questions.append(
                    f"📅 What time period should I analyze? "
                    f"For {detected_domain} data, common options are: {', '.join(suggestions[:3])}"
                )
                
            elif amb_type == "metric" and business_context:
                # Suggest domain-specific metrics
                suggestions = business_context.key_metrics
                questions.append(
                    f"📈 Which {detected_domain} metric would you like to focus on? "
                    f"Options: {', '.join(suggestions[:4])}"
                )
                
            elif amb_type == "aggregation" and business_context:
                # Suggest domain-specific dimensions
                suggestions = business_context.common_dimensions
                questions.append(
                    f"📊 How should I break down the {detected_domain} data? "
                    f"Common groupings: {', '.join(suggestions[:3])}"
                )
                
            else:
                # Fallback to generic clarification
                questions.append(ambiguity.get("clarification", "Could you clarify this part of your query?"))
        
        # Add proactive business insights
        if business_context and not questions:
            questions.append(
                f"💡 I detected you're asking about {detected_domain} data. "
                f"You might want to explore: {', '.join(business_context.key_metrics[:3])}"
            )
        
        return questions
    
    def _resolve_ambiguity_with_context(
        self, 
        ambiguity: Dict[str, Any], 
        context: AgentContext
    ) -> Optional[AmbiguityResolution]:
        """
        Attempt to resolve ambiguity using business context and conversation history.
        """
        amb_type = ambiguity["type"]
        original_term = ambiguity["match"]
        
        detected_domain = self._detect_business_domain(context.query, context.metadata)
        business_context = self.business_contexts.get(detected_domain) if detected_domain else None
        
        if not business_context:
            return None
        
        # Try to resolve using business rules
        business_rules = business_context.business_rules
        
        if amb_type == "temporal" and "recent" in original_term.lower():
            resolved_term = business_rules.get("recent")
            if resolved_term:
                return AmbiguityResolution(
                    ambiguity_type=amb_type,
                    original_term=original_term,
                    resolved_term=resolved_term,
                    confidence=0.8,
                    reasoning=f"Using {detected_domain} domain default for 'recent'",
                    business_context={"domain": detected_domain}
                )
        
        elif amb_type == "metric" and any(word in original_term.lower() for word in ["performance", "efficiency"]):
            resolved_term = business_rules.get(original_term.lower())
            if resolved_term:
                return AmbiguityResolution(
                    ambiguity_type=amb_type,
                    original_term=original_term,
                    resolved_term=resolved_term,
                    confidence=0.75,
                    reasoning=f"Using {detected_domain} domain interpretation for '{original_term}'",
                    business_context={"domain": detected_domain}
                )
        
        elif amb_type == "ranking" and "top" in original_term.lower():
            # Use business context to suggest appropriate limit
            if detected_domain == "sales":
                resolved_term = "top_10_by_revenue"
                confidence = 0.7
            elif detected_domain == "marketing":
                resolved_term = "top_5_by_performance"
                confidence = 0.7
            else:
                resolved_term = "top_10"
                confidence = 0.6
            
            return AmbiguityResolution(
                ambiguity_type=amb_type,
                original_term=original_term,
                resolved_term=resolved_term,
                confidence=confidence,
                reasoning=f"Using {detected_domain or 'general'} domain default for ranking",
                business_context={"domain": detected_domain}
            )
        
        return None
    
    def _calculate_clarification_confidence(self, ambiguities: List[Dict[str, Any]]) -> float:
        """
        Calculate overall confidence in the need for clarification.
        """
        if not ambiguities:
            return 1.0  # No ambiguities = high confidence
        
        # Weight ambiguities by their confidence scores
        total_confidence = sum(amb.get("confidence", 0.5) for amb in ambiguities)
        avg_confidence = total_confidence / len(ambiguities)
        
        # Lower confidence means more clarification needed
        clarification_confidence = 1.0 - avg_confidence
        
        return max(0.1, min(clarification_confidence, 0.9))
