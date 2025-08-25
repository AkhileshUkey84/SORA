# services/business_context_service.py
"""
Business Context Service providing domain-aware intelligence and insights.
Key differentiator for hackathon - makes AI understand business context.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import re
import structlog
from dataclasses import dataclass

logger = structlog.get_logger()


class BusinessDomain(str, Enum):
    """Supported business domains"""
    SALES = "sales"
    FINANCE = "finance" 
    MARKETING = "marketing"
    OPERATIONS = "operations"
    HR = "hr"
    CUSTOMER_SERVICE = "customer_service"
    GENERAL = "general"


class InsightPattern(BaseModel):
    """Template for business insight patterns"""
    pattern_id: str
    name: str
    description: str
    trigger_keywords: List[str] = []
    required_columns: List[str] = []
    sql_template: Optional[str] = None
    insight_template: str
    importance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    domain: BusinessDomain = BusinessDomain.GENERAL


class BusinessRule(BaseModel):
    """Business rules for domain-specific logic"""
    rule_id: str
    domain: BusinessDomain
    rule_type: str  # "threshold", "calculation", "validation", "interpretation"
    condition: str
    action: str
    explanation: str
    confidence_boost: float = 0.0


class DomainContext(BaseModel):
    """Context information for a business domain"""
    domain: BusinessDomain
    key_metrics: List[str] = []
    common_dimensions: List[str] = []
    typical_time_periods: List[str] = []
    kpi_thresholds: Dict[str, Dict[str, float]] = {}  # metric -> {good, warning, critical}
    business_rules: List[BusinessRule] = []
    terminology: Dict[str, str] = {}  # business term -> technical meaning
    seasonal_patterns: Dict[str, List[str]] = {}  # metric -> seasonal indicators


@dataclass
class BusinessInsight:
    """Business-aware insight with context"""
    insight_type: str
    title: str
    description: str
    business_impact: str
    recommended_actions: List[str] = None
    confidence: float = 0.5
    supporting_evidence: Dict[str, Any] = None
    next_questions: List[str] = None


class BusinessContextService:
    """
    Provides business-aware intelligence and domain-specific insights.
    Makes the AI system understand business context, not just data patterns.
    """
    
    def __init__(self):
        self.logger = logger.bind(component="BusinessContextService")
        self.domain_contexts: Dict[BusinessDomain, DomainContext] = {}
        self.insight_patterns: Dict[str, InsightPattern] = {}
        self._initialize_domains()
        self._initialize_insight_patterns()
    
    def _initialize_domains(self):
        """Initialize business domain contexts"""
        
        # Sales Domain
        sales_context = DomainContext(
            domain=BusinessDomain.SALES,
            key_metrics=[
                "revenue", "units_sold", "conversion_rate", "average_order_value", 
                "customer_acquisition_cost", "lifetime_value", "churn_rate"
            ],
            common_dimensions=[
                "sales_rep", "region", "product_category", "customer_segment", 
                "channel", "campaign"
            ],
            typical_time_periods=[
                "daily", "weekly", "monthly", "quarterly", "year_over_year", 
                "month_over_month", "seasonal"
            ],
            kpi_thresholds={
                "conversion_rate": {"good": 0.05, "warning": 0.02, "critical": 0.01},
                "churn_rate": {"good": 0.05, "warning": 0.10, "critical": 0.20},
                "revenue_growth": {"good": 0.10, "warning": 0.05, "critical": 0.00}
            },
            terminology={
                "performance": "revenue and conversion metrics",
                "efficiency": "cost per acquisition and ROI",
                "recent": "last 30 days",
                "success": "meeting or exceeding quota"
            },
            seasonal_patterns={
                "revenue": ["holiday_season", "end_of_quarter", "back_to_school"],
                "conversion_rate": ["seasonal_demand", "promotional_periods"]
            }
        )
        
        # Finance Domain
        finance_context = DomainContext(
            domain=BusinessDomain.FINANCE,
            key_metrics=[
                "profit", "margin", "cash_flow", "roi", "burn_rate", 
                "operating_expenses", "gross_margin", "net_income"
            ],
            common_dimensions=[
                "cost_center", "department", "budget_category", "account_type",
                "fiscal_period", "business_unit"
            ],
            typical_time_periods=[
                "monthly", "quarterly", "fiscal_year", "budget_period",
                "year_to_date", "quarter_to_date"
            ],
            kpi_thresholds={
                "gross_margin": {"good": 0.40, "warning": 0.25, "critical": 0.15},
                "operating_margin": {"good": 0.15, "warning": 0.05, "critical": 0.00},
                "cash_flow": {"good": 1000000, "warning": 100000, "critical": 0}
            },
            terminology={
                "performance": "profitability and efficiency ratios",
                "efficiency": "cost management and margin optimization",
                "recent": "current quarter",
                "health": "liquidity and profitability indicators"
            }
        )
        
        # Marketing Domain
        marketing_context = DomainContext(
            domain=BusinessDomain.MARKETING,
            key_metrics=[
                "cac", "ltv", "roas", "ctr", "cpm", "impressions", 
                "clicks", "leads", "mql", "sql", "attribution"
            ],
            common_dimensions=[
                "campaign", "channel", "audience_segment", "creative", 
                "geography", "device_type", "attribution_model"
            ],
            typical_time_periods=[
                "daily", "weekly", "campaign_duration", "monthly",
                "seasonal", "cohort_analysis"
            ],
            kpi_thresholds={
                "ctr": {"good": 0.02, "warning": 0.01, "critical": 0.005},
                "roas": {"good": 4.0, "warning": 2.0, "critical": 1.0},
                "cac_ltv_ratio": {"good": 0.3, "warning": 0.5, "critical": 0.8}
            },
            terminology={
                "performance": "ROAS and conversion metrics",
                "efficiency": "cost per acquisition and attribution",
                "recent": "last 7 days",
                "success": "positive ROI and growing MQLs"
            }
        )
        
        self.domain_contexts[BusinessDomain.SALES] = sales_context
        self.domain_contexts[BusinessDomain.FINANCE] = finance_context
        self.domain_contexts[BusinessDomain.MARKETING] = marketing_context
    
    def _initialize_insight_patterns(self):
        """Initialize business insight patterns"""
        
        # Sales patterns
        self.insight_patterns["sales_concentration"] = InsightPattern(
            pattern_id="sales_concentration",
            name="Revenue Concentration Analysis",
            description="Identifies concentration of revenue in top customers/products",
            trigger_keywords=["revenue", "sales", "top", "concentration"],
            required_columns=["customer", "revenue"],
            insight_template="Top {top_n} {dimension} account for {percentage}% of revenue - {risk_level} concentration risk",
            importance_score=0.8,
            domain=BusinessDomain.SALES
        )
        
        self.insight_patterns["seasonal_trend"] = InsightPattern(
            pattern_id="seasonal_trend",
            name="Seasonal Pattern Detection",
            description="Identifies seasonal patterns in business metrics",
            trigger_keywords=["seasonal", "monthly", "quarterly", "trend"],
            required_columns=["date", "metric"],
            insight_template="{metric} shows {pattern_type} seasonal pattern with {peak_period} peak",
            importance_score=0.7,
            domain=BusinessDomain.GENERAL
        )
        
        self.insight_patterns["performance_gaps"] = InsightPattern(
            pattern_id="performance_gaps",
            name="Performance Gap Analysis",
            description="Identifies underperforming segments or categories",
            trigger_keywords=["performance", "gap", "underperform", "comparison"],
            required_columns=["category", "performance_metric"],
            insight_template="{category} underperforms by {gap_percentage}% - potential for {improvement_opportunity}",
            importance_score=0.9,
            domain=BusinessDomain.GENERAL
        )
    
    def detect_domain(self, query: str, metadata: Dict[str, Any] = None) -> Tuple[BusinessDomain, float]:
        """
        Detect business domain from query and metadata.
        Returns domain and confidence score.
        """
        query_lower = query.lower()
        domain_scores = {}
        
        # Score each domain based on keyword matches
        for domain, context in self.domain_contexts.items():
            score = 0
            
            # Check key metrics
            for metric in context.key_metrics:
                if metric in query_lower:
                    score += 3
            
            # Check dimensions
            for dimension in context.common_dimensions:
                if dimension in query_lower:
                    score += 2
            
            # Check terminology
            for term, meaning in context.terminology.items():
                if term in query_lower:
                    score += 2
            
            # Check time periods
            for period in context.typical_time_periods:
                if period.replace("_", " ") in query_lower:
                    score += 1
            
            if score > 0:
                domain_scores[domain] = score
        
        # Check metadata for domain hints
        if metadata:
            dataset_name = metadata.get("dataset_name", "").lower()
            schema_info = metadata.get("schema", {})
            
            # Domain-specific table/column patterns
            domain_indicators = {
                BusinessDomain.SALES: ["sales", "revenue", "customer", "order", "product"],
                BusinessDomain.FINANCE: ["finance", "budget", "cost", "expense", "profit"],
                BusinessDomain.MARKETING: ["campaign", "ad", "marketing", "lead", "conversion"]
            }
            
            for domain, indicators in domain_indicators.items():
                for indicator in indicators:
                    if indicator in dataset_name:
                        domain_scores[domain] = domain_scores.get(domain, 0) + 2
                    
                    # Check column names
                    if schema_info and "columns" in schema_info:
                        for col in schema_info["columns"]:
                            col_name = col.get("name", "").lower()
                            if indicator in col_name:
                                domain_scores[domain] = domain_scores.get(domain, 0) + 1
        
        if not domain_scores:
            return BusinessDomain.GENERAL, 0.5
        
        # Return highest scoring domain
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        
        # Normalize confidence (max possible score is roughly 20)
        confidence = min(max_score / 20.0, 1.0)
        
        return best_domain, confidence
    
    def generate_domain_suggestions(self, domain: BusinessDomain, query: str) -> List[str]:
        """Generate domain-specific query suggestions"""
        if domain not in self.domain_contexts:
            return []
        
        context = self.domain_contexts[domain]
        suggestions = []
        
        # Time-based suggestions
        suggestions.extend([
            f"Show me {context.key_metrics[0]} trends over {context.typical_time_periods[0]}",
            f"Compare {context.key_metrics[1]} {context.typical_time_periods[2]} vs {context.typical_time_periods[1]}",
        ])
        
        # Dimensional analysis suggestions
        if context.common_dimensions:
            suggestions.extend([
                f"Break down {context.key_metrics[0]} by {context.common_dimensions[0]}",
                f"Which {context.common_dimensions[1]} has the highest {context.key_metrics[1]}?",
            ])
        
        # Domain-specific insights
        if domain == BusinessDomain.SALES:
            suggestions.extend([
                "Show me customer lifetime value analysis",
                "What's our customer churn rate by segment?",
                "Identify our top performing sales regions"
            ])
        elif domain == BusinessDomain.FINANCE:
            suggestions.extend([
                "Analyze profit margin trends by department",
                "Show me budget vs actual variance analysis", 
                "What are our biggest cost drivers?"
            ])
        elif domain == BusinessDomain.MARKETING:
            suggestions.extend([
                "Compare campaign performance by channel",
                "Show me customer acquisition funnel metrics",
                "What's our return on ad spend by audience?"
            ])
        
        return suggestions[:5]  # Return top 5
    
    def interpret_metric_threshold(self, domain: BusinessDomain, metric: str, value: float) -> Dict[str, Any]:
        """Interpret metric value against domain-specific thresholds"""
        if domain not in self.domain_contexts:
            return {"status": "unknown", "interpretation": "No domain context available"}
        
        context = self.domain_contexts[domain]
        thresholds = context.kpi_thresholds.get(metric, {})
        
        if not thresholds:
            return {"status": "unknown", "interpretation": f"No threshold defined for {metric}"}
        
        # Determine status
        if value >= thresholds.get("good", float('inf')):
            status = "excellent"
            interpretation = f"{metric} is performing excellently at {value:.2f}"
        elif value >= thresholds.get("warning", float('inf')):
            status = "good"
            interpretation = f"{metric} is within acceptable range at {value:.2f}"
        elif value >= thresholds.get("critical", float('inf')):
            status = "warning"
            interpretation = f"{metric} is below target at {value:.2f} - attention needed"
        else:
            status = "critical"
            interpretation = f"{metric} is critically low at {value:.2f} - immediate action required"
        
        return {
            "status": status,
            "interpretation": interpretation,
            "thresholds": thresholds,
            "recommendations": self._get_metric_recommendations(domain, metric, status)
        }
    
    def _get_metric_recommendations(self, domain: BusinessDomain, metric: str, status: str) -> List[str]:
        """Get domain-specific recommendations for metric performance"""
        if status in ["excellent", "good"]:
            return [f"Continue current strategies for {metric}", "Monitor for consistency"]
        
        # Domain-specific recommendations
        domain_recs = {
            BusinessDomain.SALES: {
                "conversion_rate": [
                    "Review and optimize sales funnel",
                    "Analyze top performing sales reps' tactics",
                    "A/B test pricing and product positioning"
                ],
                "churn_rate": [
                    "Implement customer success program",
                    "Analyze churn reasons through exit surveys", 
                    "Improve onboarding and support processes"
                ]
            },
            BusinessDomain.FINANCE: {
                "gross_margin": [
                    "Review supplier contracts and negotiate better rates",
                    "Optimize product mix towards higher margin items",
                    "Implement cost reduction initiatives"
                ],
                "cash_flow": [
                    "Accelerate collections from customers",
                    "Review and optimize payment terms",
                    "Consider invoice factoring or credit line"
                ]
            },
            BusinessDomain.MARKETING: {
                "ctr": [
                    "A/B test ad creative and messaging",
                    "Refine audience targeting",
                    "Review ad placement and timing"
                ],
                "roas": [
                    "Pause underperforming campaigns",
                    "Increase budget for high-ROAS channels",
                    "Optimize landing pages for conversion"
                ]
            }
        }
        
        return domain_recs.get(domain, {}).get(metric, [f"Review {metric} performance drivers"])
    
    def generate_business_insights(self, data: List[Dict[str, Any]], domain: BusinessDomain, 
                                 query_context: Dict[str, Any] = None) -> List[BusinessInsight]:
        """Generate business-aware insights from data"""
        insights = []
        
        if not data:
            return insights
        
        # Convert to more analyzable format
        import pandas as pd
        df = pd.DataFrame(data)
        
        if len(df) < 3:  # Need minimum data for insights
            return insights
        
        # Generate domain-specific insights
        if domain == BusinessDomain.SALES:
            insights.extend(self._generate_sales_insights(df))
        elif domain == BusinessDomain.FINANCE:
            insights.extend(self._generate_finance_insights(df))
        elif domain == BusinessDomain.MARKETING:
            insights.extend(self._generate_marketing_insights(df))
        
        # General business insights
        insights.extend(self._generate_general_insights(df, domain))
        
        # Sort by confidence and return top insights
        insights.sort(key=lambda x: x.confidence, reverse=True)
        return insights[:5]
    
    def _generate_sales_insights(self, df: pd.DataFrame) -> List[BusinessInsight]:
        """Generate sales-specific insights"""
        insights = []
        
        # Revenue concentration insight
        if "revenue" in df.columns and len(df) > 5:
            total_revenue = df["revenue"].sum()
            # Check if we have customer or product dimension
            dimension_col = None
            for col in ["customer", "product", "category", "region"]:
                if col in df.columns:
                    dimension_col = col
                    break
            
            if dimension_col:
                top_contributors = df.nlargest(3, "revenue")
                top_3_revenue = top_contributors["revenue"].sum()
                concentration_pct = (top_3_revenue / total_revenue) * 100
                
                if concentration_pct > 60:
                    risk_level = "high"
                    impact = "High revenue concentration creates customer dependency risk"
                    actions = [
                        "Diversify customer base to reduce concentration risk",
                        "Develop retention strategies for top customers",
                        "Expand into new market segments"
                    ]
                elif concentration_pct > 40:
                    risk_level = "moderate"
                    impact = "Moderate revenue concentration - monitor key accounts"
                    actions = [
                        "Monitor top customer health metrics",
                        "Create backup plans for key account loss"
                    ]
                else:
                    risk_level = "low"
                    impact = "Well-diversified revenue base"
                    actions = ["Continue diversification strategy"]
                
                insights.append(BusinessInsight(
                    insight_type="risk_analysis",
                    title=f"{risk_level.title()} Revenue Concentration",
                    description=f"Top 3 {dimension_col}s account for {concentration_pct:.1f}% of total revenue",
                    business_impact=impact,
                    recommended_actions=actions,
                    confidence=0.85,
                    supporting_evidence={
                        "concentration_percentage": concentration_pct,
                        "top_contributors": top_contributors[dimension_col].tolist(),
                        "risk_level": risk_level
                    },
                    next_questions=[
                        f"Show me revenue trends for top {dimension_col}s",
                        f"What's the churn risk for our top {dimension_col}s?",
                        f"How has {dimension_col} concentration changed over time?"
                    ]
                ))
        
        return insights
    
    def _generate_finance_insights(self, df: pd.DataFrame) -> List[BusinessInsight]:
        """Generate finance-specific insights"""
        insights = []
        
        # Margin analysis
        if "revenue" in df.columns and "cost" in df.columns:
            df["margin"] = (df["revenue"] - df["cost"]) / df["revenue"]
            avg_margin = df["margin"].mean()
            
            # Check for low margin items
            low_margin_threshold = 0.15  # 15%
            low_margin_items = df[df["margin"] < low_margin_threshold]
            
            if len(low_margin_items) > 0:
                low_margin_pct = len(low_margin_items) / len(df) * 100
                
                insights.append(BusinessInsight(
                    insight_type="profitability_analysis", 
                    title="Low Margin Products Identified",
                    description=f"{len(low_margin_items)} items ({low_margin_pct:.1f}%) have margins below {low_margin_threshold*100}%",
                    business_impact=f"Low margin products are reducing overall profitability (avg margin: {avg_margin*100:.1f}%)",
                    recommended_actions=[
                        "Review pricing strategy for low margin products",
                        "Negotiate better supplier terms",
                        "Consider discontinuing unprofitable lines",
                        "Focus marketing on higher margin products"
                    ],
                    confidence=0.9,
                    supporting_evidence={
                        "low_margin_count": len(low_margin_items),
                        "average_margin": avg_margin,
                        "threshold": low_margin_threshold
                    },
                    next_questions=[
                        "Show me profit margin trends over time",
                        "Which products have the highest margins?", 
                        "How do our margins compare by category?"
                    ]
                ))
        
        return insights
    
    def _generate_marketing_insights(self, df: pd.DataFrame) -> List[BusinessInsight]:
        """Generate marketing-specific insights"""
        insights = []
        
        # Campaign performance analysis
        if "campaign" in df.columns and "roas" in df.columns:
            avg_roas = df["roas"].mean()
            underperforming = df[df["roas"] < 2.0]  # ROAS below 2:1
            
            if len(underperforming) > 0:
                insights.append(BusinessInsight(
                    insight_type="campaign_optimization",
                    title="Underperforming Campaigns Detected",
                    description=f"{len(underperforming)} campaigns have ROAS below 2:1 threshold",
                    business_impact=f"Poor performing campaigns are dragging down overall ROAS ({avg_roas:.1f}:1)",
                    recommended_actions=[
                        "Pause campaigns with ROAS < 1:1 immediately",
                        "A/B test creative for campaigns with 1:1 < ROAS < 2:1",
                        "Reallocate budget to high-performing campaigns",
                        "Review targeting and messaging for underperformers"
                    ],
                    confidence=0.8,
                    supporting_evidence={
                        "underperforming_count": len(underperforming),
                        "avg_roas": avg_roas,
                        "threshold": 2.0
                    }
                ))
        
        return insights
    
    def _generate_general_insights(self, df: pd.DataFrame, domain: BusinessDomain) -> List[BusinessInsight]:
        """Generate general business insights applicable across domains"""
        insights = []
        
        # Data quality insight
        null_counts = df.isnull().sum()
        high_null_columns = null_counts[null_counts > len(df) * 0.2]  # > 20% null
        
        if len(high_null_columns) > 0:
            insights.append(BusinessInsight(
                insight_type="data_quality",
                title="Data Quality Issues Detected", 
                description=f"{len(high_null_columns)} columns have >20% missing values",
                business_impact="Missing data may lead to inaccurate analysis and decisions",
                recommended_actions=[
                    "Investigate data collection processes",
                    "Implement data validation at source",
                    "Consider data imputation strategies"
                ],
                confidence=0.9,
                supporting_evidence={
                    "columns_with_nulls": high_null_columns.index.tolist(),
                    "null_percentages": (high_null_columns / len(df) * 100).to_dict()
                }
            ))
        
        return insights
    
    def get_domain_context(self, domain: BusinessDomain) -> Optional[DomainContext]:
        """Get full context information for a domain"""
        return self.domain_contexts.get(domain)
    
    def translate_business_term(self, term: str, domain: BusinessDomain) -> Optional[str]:
        """Translate business term to technical meaning within domain context"""
        if domain not in self.domain_contexts:
            return None
        
        context = self.domain_contexts[domain]
        return context.terminology.get(term.lower())
    
    def get_recommended_metrics(self, domain: BusinessDomain, query_intent: str = None) -> List[str]:
        """Get recommended metrics for a domain and query intent"""
        if domain not in self.domain_contexts:
            return []
        
        context = self.domain_contexts[domain]
        
        if not query_intent:
            return context.key_metrics[:5]
        
        # Filter metrics based on intent
        intent_lower = query_intent.lower()
        relevant_metrics = []
        
        for metric in context.key_metrics:
            if any(word in metric for word in intent_lower.split()):
                relevant_metrics.append(metric)
        
        return relevant_metrics[:3] if relevant_metrics else context.key_metrics[:3]
