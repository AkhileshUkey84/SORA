# agents/suggestion_agent.py
"""
Suggestion Agent for proactive next-question prediction.
Key differentiator - anticipates user's follow-up questions and provides intelligent suggestions.
"""

from typing import Dict, Any, List, Optional, Tuple
from agents.base import BaseAgent, AgentContext, AgentResult
from pydantic import BaseModel, Field
from datetime import datetime
import re
import pandas as pd
from enum import Enum


class SuggestionType(str, Enum):
    """Types of suggestions"""
    DRILL_DOWN = "drill_down"
    TREND_ANALYSIS = "trend_analysis" 
    COMPARISON = "comparison"
    ROOT_CAUSE = "root_cause"
    FORECAST = "forecast"
    SEGMENTATION = "segmentation"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"


class SuggestionContext(str, Enum):
    """Context that triggered the suggestion"""
    OUTLIER_DETECTED = "outlier_detected"
    TREND_IDENTIFIED = "trend_identified"
    CORRELATION_FOUND = "correlation_found"
    THRESHOLD_BREACH = "threshold_breach"
    PATTERN_ANOMALY = "pattern_anomaly"
    DATA_GAP = "data_gap"
    PERFORMANCE_ISSUE = "performance_issue"


class SmartSuggestion(BaseModel):
    """Intelligent suggestion with context"""
    suggestion_id: str
    suggested_query: str
    explanation: str
    suggestion_type: SuggestionType
    context: SuggestionContext
    confidence: float = Field(ge=0.0, le=1.0)
    priority: int = Field(ge=1, le=5)  # 1 = highest priority
    business_value: str
    expected_insight: str
    sql_template: Optional[str] = None
    domain_relevance: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuggestionAgent(BaseAgent):
    """
    Proactively suggests follow-up questions based on current results.
    Uses pattern recognition and business logic to anticipate user needs.
    """
    
    def __init__(self):
        super().__init__("SuggestionAgent")
        self.suggestion_patterns = self._initialize_patterns()
        self.business_contexts = self._initialize_business_contexts()
        
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Generate intelligent suggestions based on query results and context.
        """
        results = kwargs.get("results", {})
        sql = kwargs.get("sql", "")
        domain = kwargs.get("domain", "general")
        insights = kwargs.get("insights", [])
        
        if not results or not results.get("data"):
            return AgentResult(
                success=True,
                data={"suggestions": [], "message": "No data available for suggestions"}
            )
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results["data"])
        
        # Generate suggestions from multiple sources
        suggestions = []
        
        # 1. Pattern-based suggestions
        pattern_suggestions = await self._generate_pattern_suggestions(df, sql, context)
        suggestions.extend(pattern_suggestions)
        
        # 2. Insight-driven suggestions
        insight_suggestions = await self._generate_insight_suggestions(insights, df, context)
        suggestions.extend(insight_suggestions)
        
        # 3. Domain-specific suggestions
        domain_suggestions = await self._generate_domain_suggestions(domain, df, context)
        suggestions.extend(domain_suggestions)
        
        # 4. Data-driven suggestions
        data_suggestions = await self._generate_data_driven_suggestions(df, sql, context)
        suggestions.extend(data_suggestions)
        
        # 5. Conversational context suggestions
        context_suggestions = await self._generate_context_suggestions(context)
        suggestions.extend(context_suggestions)
        
        # Rank and filter suggestions
        ranked_suggestions = self._rank_suggestions(suggestions, context)
        top_suggestions = ranked_suggestions[:5]  # Top 5 suggestions
        
        return AgentResult(
            success=True,
            data={
                "suggestions": [s.dict() for s in top_suggestions],
                "total_generated": len(suggestions),
                "suggestion_summary": self._generate_suggestion_summary(top_suggestions)
            },
            confidence=0.85 if suggestions else 0.3
        )
    
    async def _generate_pattern_suggestions(self, df: pd.DataFrame, sql: str, 
                                         context: AgentContext) -> List[SmartSuggestion]:
        """Generate suggestions based on recognized patterns"""
        suggestions = []
        
        # Detect current query pattern
        query_pattern = self._detect_query_pattern(sql)
        
        # Generate complementary patterns
        if query_pattern == "aggregation" and self._has_time_dimension(df):
            suggestions.append(SmartSuggestion(
                suggestion_id="trend_analysis",
                suggested_query="Show me this metric trend over time",
                explanation="Since you're looking at aggregated data, seeing trends over time would reveal patterns",
                suggestion_type=SuggestionType.TREND_ANALYSIS,
                context=SuggestionContext.PATTERN_ANOMALY,
                confidence=0.85,
                priority=2,
                business_value="Identify growth patterns, seasonality, and performance trends",
                expected_insight="Time-based patterns that explain current performance"
            ))
        
        # If grouping by category, suggest drilling down
        if self._has_categorical_grouping(df):
            category_col = self._get_categorical_column(df)
            if category_col:
                suggestions.append(SmartSuggestion(
                    suggestion_id="drill_down_category",
                    suggested_query=f"Break down the top {category_col} results further",
                    explanation=f"Drilling deeper into top-performing {category_col} can reveal specific drivers",
                    suggestion_type=SuggestionType.DRILL_DOWN,
                    context=SuggestionContext.PERFORMANCE_ISSUE,
                    confidence=0.8,
                    priority=1,
                    business_value="Identify specific factors driving performance",
                    expected_insight=f"Detailed breakdown of what makes certain {category_col} successful"
                ))
        
        # If showing totals, suggest comparisons
        if self._is_total_query(sql):
            suggestions.append(SmartSuggestion(
                suggestion_id="comparative_analysis",
                suggested_query="Compare this metric across different segments",
                explanation="Total numbers are more meaningful when compared across categories or time periods",
                suggestion_type=SuggestionType.COMPARISON,
                context=SuggestionContext.PATTERN_ANOMALY,
                confidence=0.75,
                priority=3,
                business_value="Understand relative performance and identify gaps",
                expected_insight="Which segments over/under-perform relative to average"
            ))
        
        return suggestions
    
    async def _generate_insight_suggestions(self, insights: List, df: pd.DataFrame,
                                          context: AgentContext) -> List[SmartSuggestion]:
        """Generate suggestions based on discovered insights"""
        suggestions = []
        
        if not insights:
            return suggestions
        
        for insight_dict in insights:
            insight_type = insight_dict.get("type")
            insight_title = insight_dict.get("title", "")
            
            if insight_type == "outlier":
                suggestions.append(SmartSuggestion(
                    suggestion_id="investigate_outliers",
                    suggested_query="What factors contribute to these outlier values?",
                    explanation="Outliers often reveal important exceptions or opportunities worth investigating",
                    suggestion_type=SuggestionType.ROOT_CAUSE,
                    context=SuggestionContext.OUTLIER_DETECTED,
                    confidence=0.9,
                    priority=1,
                    business_value="Identify root causes of exceptional performance",
                    expected_insight="Specific factors that create outlier conditions"
                ))
            
            elif insight_type == "correlation":
                suggestions.append(SmartSuggestion(
                    suggestion_id="causal_analysis",
                    suggested_query="Is this correlation consistent across different segments?",
                    explanation="Strong correlations should be validated across different contexts",
                    suggestion_type=SuggestionType.VALIDATION,
                    context=SuggestionContext.CORRELATION_FOUND,
                    confidence=0.85,
                    priority=2,
                    business_value="Validate relationships and identify when they break down",
                    expected_insight="Conditions where the correlation holds or fails"
                ))
            
            elif insight_type == "trend":
                suggestions.append(SmartSuggestion(
                    suggestion_id="trend_projection",
                    suggested_query="What would happen if this trend continues?",
                    explanation="Understanding trend implications helps with planning and decision-making",
                    suggestion_type=SuggestionType.FORECAST,
                    context=SuggestionContext.TREND_IDENTIFIED,
                    confidence=0.8,
                    priority=2,
                    business_value="Project future outcomes and identify intervention points",
                    expected_insight="Future trajectory and potential impact"
                ))
            
            elif "concentration" in insight_title.lower():
                suggestions.append(SmartSuggestion(
                    suggestion_id="risk_mitigation",
                    suggested_query="How can we reduce this concentration risk?",
                    explanation="High concentration creates vulnerability - exploring diversification options is valuable",
                    suggestion_type=SuggestionType.OPTIMIZATION,
                    context=SuggestionContext.THRESHOLD_BREACH,
                    confidence=0.85,
                    priority=1,
                    business_value="Develop risk mitigation strategies",
                    expected_insight="Specific actions to reduce concentration risk"
                ))
        
        return suggestions
    
    async def _generate_domain_suggestions(self, domain: str, df: pd.DataFrame,
                                         context: AgentContext) -> List[SmartSuggestion]:
        """Generate domain-specific suggestions"""
        suggestions = []
        
        domain_patterns = self.business_contexts.get(domain, {})
        
        if domain == "sales":
            if "revenue" in df.columns:
                suggestions.append(SmartSuggestion(
                    suggestion_id="customer_segmentation",
                    suggested_query="Analyze customer lifetime value by segment",
                    explanation="Revenue analysis is more actionable when segmented by customer value",
                    suggestion_type=SuggestionType.SEGMENTATION,
                    context=SuggestionContext.PERFORMANCE_ISSUE,
                    confidence=0.8,
                    priority=2,
                    business_value="Identify highest-value customer segments for targeting",
                    expected_insight="Customer segments with highest ROI potential",
                    domain_relevance="sales"
                ))
                
                suggestions.append(SmartSuggestion(
                    suggestion_id="conversion_funnel",
                    suggested_query="Show me the conversion funnel breakdown",
                    explanation="Revenue insights are enhanced by understanding conversion bottlenecks",
                    suggestion_type=SuggestionType.ROOT_CAUSE,
                    context=SuggestionContext.PERFORMANCE_ISSUE,
                    confidence=0.85,
                    priority=1,
                    business_value="Identify conversion optimization opportunities",
                    expected_insight="Specific funnel stages with improvement potential",
                    domain_relevance="sales"
                ))
        
        elif domain == "marketing":
            if any(col in df.columns for col in ["campaign", "channel", "impressions"]):
                suggestions.append(SmartSuggestion(
                    suggestion_id="attribution_analysis",
                    suggested_query="What's the attribution breakdown across touchpoints?",
                    explanation="Marketing performance requires understanding multi-touch attribution",
                    suggestion_type=SuggestionType.ROOT_CAUSE,
                    context=SuggestionContext.PERFORMANCE_ISSUE,
                    confidence=0.8,
                    priority=1,
                    business_value="Optimize marketing mix and budget allocation",
                    expected_insight="True contribution of each marketing channel",
                    domain_relevance="marketing"
                ))
                
                suggestions.append(SmartSuggestion(
                    suggestion_id="cohort_analysis",
                    suggested_query="Analyze user behavior by acquisition cohort",
                    explanation="Campaign effectiveness is best measured through cohort retention",
                    suggestion_type=SuggestionType.SEGMENTATION,
                    context=SuggestionContext.PATTERN_ANOMALY,
                    confidence=0.75,
                    priority=2,
                    business_value="Understand long-term campaign effectiveness",
                    expected_insight="Which campaigns drive lasting engagement",
                    domain_relevance="marketing"
                ))
        
        elif domain == "finance":
            if any(col in df.columns for col in ["cost", "profit", "margin"]):
                suggestions.append(SmartSuggestion(
                    suggestion_id="variance_analysis",
                    suggested_query="Compare actual vs budget performance",
                    explanation="Financial analysis requires comparing against planned targets",
                    suggestion_type=SuggestionType.COMPARISON,
                    context=SuggestionContext.PERFORMANCE_ISSUE,
                    confidence=0.9,
                    priority=1,
                    business_value="Identify budget variances and their causes",
                    expected_insight="Areas of over/under-spending and their drivers",
                    domain_relevance="finance"
                ))
        
        return suggestions
    
    async def _generate_data_driven_suggestions(self, df: pd.DataFrame, sql: str,
                                              context: AgentContext) -> List[SmartSuggestion]:
        """Generate suggestions based on data characteristics"""
        suggestions = []
        
        # Check for data quality issues
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if null_percentage > 0.1:  # >10% null values
            suggestions.append(SmartSuggestion(
                suggestion_id="data_quality_investigation",
                suggested_query="Investigate data completeness by source/time period",
                explanation="High missing data rates may indicate collection issues affecting accuracy",
                suggestion_type=SuggestionType.VALIDATION,
                context=SuggestionContext.DATA_GAP,
                confidence=0.85,
                priority=1,
                business_value="Ensure analysis reliability and identify data improvement needs",
                expected_insight="Root causes of data gaps and improvement opportunities"
            ))
        
        # Check for small sample size
        if len(df) < 50:
            suggestions.append(SmartSuggestion(
                suggestion_id="expand_sample",
                suggested_query="Expand the time period or filters to get more data",
                explanation="Small sample sizes may not be representative - more data increases reliability",
                suggestion_type=SuggestionType.VALIDATION,
                context=SuggestionContext.DATA_GAP,
                confidence=0.75,
                priority=3,
                business_value="Improve statistical reliability of insights",
                expected_insight="More robust patterns with larger sample size"
            ))
        
        # Check for extreme skewness in numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 10:
                skewness = df[col].skew()
                if abs(skewness) > 2:  # Highly skewed
                    suggestions.append(SmartSuggestion(
                        suggestion_id=f"investigate_{col}_distribution",
                        suggested_query=f"What drives the extreme values in {col}?",
                        explanation=f"{col} has a highly skewed distribution - extreme values may reveal important insights",
                        suggestion_type=SuggestionType.ROOT_CAUSE,
                        context=SuggestionContext.PATTERN_ANOMALY,
                        confidence=0.7,
                        priority=3,
                        business_value="Understand factors creating extreme performance",
                        expected_insight=f"Specific conditions that create high/low {col} values"
                    ))
                    break  # Only suggest for first skewed column
        
        return suggestions
    
    async def _generate_context_suggestions(self, context: AgentContext) -> List[SmartSuggestion]:
        """Generate suggestions based on conversational context"""
        suggestions = []
        
        # Check conversation history for patterns
        conversation_context = context.metadata.get("conversation_context", {})
        previous_queries = conversation_context.get("previous_queries", [])
        
        if len(previous_queries) > 1:
            # Suggest synthesis if multiple analyses done
            suggestions.append(SmartSuggestion(
                suggestion_id="synthesize_findings",
                suggested_query="Summarize key insights from our analysis so far",
                explanation="You've explored multiple aspects - synthesizing findings reveals the bigger picture",
                suggestion_type=SuggestionType.OPTIMIZATION,
                context=SuggestionContext.PATTERN_ANOMALY,
                confidence=0.75,
                priority=4,
                business_value="Connect insights into actionable strategy",
                expected_insight="Holistic understanding and prioritized action items"
            ))
        
        # Suggest exploring unexplored dimensions
        if context.dataset_id:
            # This would check available columns vs. used columns
            suggestions.append(SmartSuggestion(
                suggestion_id="explore_new_dimensions",
                suggested_query="What other factors might influence these results?",
                explanation="Exploring additional data dimensions often reveals hidden drivers",
                suggestion_type=SuggestionType.ROOT_CAUSE,
                context=SuggestionContext.DATA_GAP,
                confidence=0.6,
                priority=4,
                business_value="Discover previously unknown influence factors",
                expected_insight="New variables that explain performance variation"
            ))
        
        return suggestions
    
    def _rank_suggestions(self, suggestions: List[SmartSuggestion], 
                         context: AgentContext) -> List[SmartSuggestion]:
        """Rank suggestions by relevance and value"""
        
        # Calculate composite scores
        for suggestion in suggestions:
            score = 0.0
            
            # Base score from confidence and priority
            score += suggestion.confidence * 0.4
            score += (6 - suggestion.priority) / 5 * 0.3  # Invert priority (1=high, 5=low)
            
            # Boost for high-value contexts
            context_boosts = {
                SuggestionContext.OUTLIER_DETECTED: 0.2,
                SuggestionContext.THRESHOLD_BREACH: 0.15,
                SuggestionContext.CORRELATION_FOUND: 0.1,
                SuggestionContext.TREND_IDENTIFIED: 0.1
            }
            score += context_boosts.get(suggestion.context, 0)
            
            # Boost for high-value suggestion types
            type_boosts = {
                SuggestionType.ROOT_CAUSE: 0.15,
                SuggestionType.OPTIMIZATION: 0.1,
                SuggestionType.DRILL_DOWN: 0.1
            }
            score += type_boosts.get(suggestion.suggestion_type, 0)
            
            # Store score for sorting
            suggestion.composite_score = score
        
        # Sort by composite score (descending)
        return sorted(suggestions, key=lambda s: getattr(s, 'composite_score', 0), reverse=True)
    
    def _generate_suggestion_summary(self, suggestions: List[SmartSuggestion]) -> str:
        """Generate human-readable summary of suggestions"""
        if not suggestions:
            return "No specific suggestions available based on current results."
        
        summary_parts = [f"🎯 **{len(suggestions)} Smart Suggestions** to deepen your analysis:\n"]
        
        # Group by type
        by_type = {}
        for suggestion in suggestions:
            by_type.setdefault(suggestion.suggestion_type, []).append(suggestion)
        
        type_descriptions = {
            SuggestionType.DRILL_DOWN: "🔍 **Drill Deeper**",
            SuggestionType.ROOT_CAUSE: "🎯 **Investigate Causes**", 
            SuggestionType.TREND_ANALYSIS: "📈 **Trend Analysis**",
            SuggestionType.COMPARISON: "⚖️ **Comparisons**",
            SuggestionType.SEGMENTATION: "🎨 **Segmentation**",
            SuggestionType.VALIDATION: "✅ **Validation**",
            SuggestionType.OPTIMIZATION: "🚀 **Optimization**",
            SuggestionType.FORECAST: "🔮 **Forecasting**"
        }
        
        for suggestion_type, items in by_type.items():
            desc = type_descriptions.get(suggestion_type, f"**{suggestion_type.title()}**")
            summary_parts.append(f"{desc} ({len(items)} suggestions)")
        
        # Add top suggestion detail
        if suggestions:
            top = suggestions[0]
            priority_emoji = "🔥" if top.priority == 1 else "⭐" if top.priority == 2 else "💡"
            summary_parts.append(f"\n{priority_emoji} **Top suggestion:** {top.explanation}")
        
        return "\n".join(summary_parts)
    
    def _detect_query_pattern(self, sql: str) -> str:
        """Detect the pattern of the current query"""
        sql_upper = sql.upper()
        
        if "GROUP BY" in sql_upper:
            return "aggregation"
        elif "ORDER BY" in sql_upper and "LIMIT" in sql_upper:
            return "ranking"
        elif "WHERE" in sql_upper:
            return "filtering"
        elif any(func in sql_upper for func in ["SUM", "COUNT", "AVG", "MAX", "MIN"]):
            return "summary"
        else:
            return "exploration"
    
    def _has_time_dimension(self, df: pd.DataFrame) -> bool:
        """Check if data has time-based columns"""
        time_indicators = ["date", "time", "created", "month", "year", "quarter"]
        return any(
            any(indicator in col.lower() for indicator in time_indicators)
            for col in df.columns
        )
    
    def _has_categorical_grouping(self, df: pd.DataFrame) -> bool:
        """Check if data appears to be grouped by categories"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        return len(categorical_cols) > 0 and len(df) > 2
    
    def _get_categorical_column(self, df: pd.DataFrame) -> Optional[str]:
        """Get the main categorical column for grouping"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            return categorical_cols[0]
        return None
    
    def _is_total_query(self, sql: str) -> bool:
        """Check if query shows total/summary values"""
        sql_upper = sql.upper()
        return any(func in sql_upper for func in ["SUM", "COUNT", "TOTAL"]) and "GROUP BY" not in sql_upper
    
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize suggestion patterns"""
        return {
            "follow_up_patterns": {
                "after_outlier": ["root_cause", "validation", "drill_down"],
                "after_trend": ["forecast", "comparison", "root_cause"],
                "after_summary": ["breakdown", "trend", "comparison"],
                "after_comparison": ["drill_down", "root_cause", "trend"]
            },
            "domain_patterns": {
                "sales": ["customer_analysis", "funnel_analysis", "cohort_analysis"],
                "marketing": ["attribution", "campaign_optimization", "audience_analysis"],
                "finance": ["variance_analysis", "profitability", "cost_analysis"]
            }
        }
    
    def _initialize_business_contexts(self) -> Dict[str, Any]:
        """Initialize business domain contexts"""
        return {
            "sales": {
                "key_metrics": ["revenue", "conversion_rate", "customer_lifetime_value"],
                "common_analyses": ["funnel", "cohort", "segmentation"],
                "follow_up_triggers": ["performance_gaps", "seasonal_patterns"]
            },
            "marketing": {
                "key_metrics": ["roas", "cac", "ltv", "attribution"],
                "common_analyses": ["attribution", "campaign_performance", "audience_analysis"],
                "follow_up_triggers": ["channel_performance", "creative_optimization"]
            },
            "finance": {
                "key_metrics": ["profit_margin", "cash_flow", "roi"],
                "common_analyses": ["variance", "profitability", "cost_breakdown"],
                "follow_up_triggers": ["budget_variance", "cost_anomalies"]
            }
        }
