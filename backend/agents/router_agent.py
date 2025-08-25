# Domain detection and routing
# agents/router_agent.py
"""
Domain routing agent that detects dataset context and routes to appropriate handlers.
Enables domain-specific optimizations and schema understanding.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from agents.base import BaseAgent, AgentContext, AgentResult
import re


class DomainType(str, Enum):
    SALES = "sales"
    FINANCE = "finance"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    HR = "hr"
    GENERAL = "general"


class RouterAgent(BaseAgent):
    """
    Routes queries to domain-specific handlers based on content analysis.
    Improves SQL generation accuracy through domain awareness.
    """
    
    def __init__(self):
        super().__init__("RouterAgent")
        self.domain_keywords = self._initialize_domain_keywords()
        self.column_patterns = self._initialize_column_patterns()
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Analyzes query and dataset to determine optimal routing.
        Adds domain-specific context for downstream agents.
        """
        
        # Detect domain from query
        query_domain = self._detect_query_domain(context.query)
        
        # Get schema hints if available
        schema_domain = None
        if context.metadata.get("schema"):
            schema_domain = self._detect_schema_domain(context.metadata["schema"])
        
        # Reconcile domains
        final_domain = self._reconcile_domains(query_domain, schema_domain)
        
        # Add domain-specific hints
        domain_hints = self._get_domain_hints(final_domain)
        
        # Update context with routing information
        context.metadata["domain"] = final_domain
        context.metadata["domain_hints"] = domain_hints
        
        self.logger.info(f"Routed to domain: {final_domain}",
                        query_domain=query_domain,
                        schema_domain=schema_domain)
        
        return AgentResult(
            success=True,
            data={
                "domain": final_domain,
                "confidence": 0.9 if final_domain != DomainType.GENERAL else 0.7,
                "hints": domain_hints,
                "suggested_metrics": self._get_domain_metrics(final_domain)
            }
        )
    
    def _initialize_domain_keywords(self) -> Dict[DomainType, List[str]]:
        """Domain-specific keyword mappings for classification"""
        return {
            DomainType.SALES: [
                "sales", "revenue", "customer", "order", "product", "deal",
                "pipeline", "quota", "commission", "forecast", "opportunity"
            ],
            DomainType.FINANCE: [
                "expense", "budget", "cost", "profit", "margin", "invoice",
                "payment", "accounting", "balance", "cash flow", "asset"
            ],
            DomainType.MARKETING: [
                "campaign", "lead", "conversion", "impression", "click",
                "engagement", "audience", "channel", "attribution", "roi"
            ],
            DomainType.OPERATIONS: [
                "inventory", "supply", "logistics", "warehouse", "shipment",
                "production", "efficiency", "utilization", "capacity"
            ],
            DomainType.HR: [
                "employee", "salary", "hire", "turnover", "performance",
                "attendance", "leave", "benefit", "training", "headcount"
            ]
        }
    
    def _initialize_column_patterns(self) -> Dict[DomainType, List[re.Pattern]]:
        """Column name patterns that indicate specific domains"""
        return {
            DomainType.SALES: [
                re.compile(r'.*sales.*', re.I),
                re.compile(r'.*revenue.*', re.I),
                re.compile(r'.*customer.*', re.I),
                re.compile(r'.*order.*', re.I)
            ],
            DomainType.FINANCE: [
                re.compile(r'.*cost.*', re.I),
                re.compile(r'.*expense.*', re.I),
                re.compile(r'.*budget.*', re.I),
                re.compile(r'.*profit.*', re.I)
            ],
            DomainType.MARKETING: [
                re.compile(r'.*campaign.*', re.I),
                re.compile(r'.*lead.*', re.I),
                re.compile(r'.*conversion.*', re.I),
                re.compile(r'.*click.*', re.I)
            ],
            DomainType.OPERATIONS: [
                re.compile(r'.*inventory.*', re.I),
                re.compile(r'.*warehouse.*', re.I),
                re.compile(r'.*shipment.*', re.I),
                re.compile(r'.*supply.*', re.I)
            ],
            DomainType.HR: [
                re.compile(r'.*employee.*', re.I),
                re.compile(r'.*salary.*', re.I),
                re.compile(r'.*hire.*', re.I),
                re.compile(r'.*performance.*', re.I)
            ]
        }
    
    def _detect_query_domain(self, query: str) -> DomainType:
        """Detects domain from query text using keyword matching"""
        query_lower = query.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return DomainType.GENERAL
    
    def _detect_schema_domain(self, schema: Dict[str, Any]) -> Optional[DomainType]:
        """Detects domain from schema column names"""
        columns = schema.get("columns", [])
        domain_scores = {}
        
        for domain, patterns in self.column_patterns.items():
            score = 0
            for col in columns:
                col_name = col.get("name", "")
                for pattern in patterns:
                    if pattern.match(col_name):
                        score += 1
                        break
            
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return None
    
    def _reconcile_domains(self, query_domain: DomainType, 
                         schema_domain: Optional[DomainType]) -> DomainType:
        """Reconciles domain detection from multiple sources"""
        if query_domain == schema_domain:
            return query_domain
        
        if schema_domain and query_domain == DomainType.GENERAL:
            return schema_domain
        
        return query_domain
    
    def _get_domain_hints(self, domain: DomainType) -> Dict[str, Any]:
        """Provides domain-specific hints for SQL generation"""
        hints = {
            DomainType.SALES: {
                "common_aggregations": ["SUM(revenue)", "COUNT(DISTINCT customer_id)", "AVG(order_value)"],
                "common_groupings": ["product_category", "sales_rep", "region", "month"],
                "important_metrics": ["conversion_rate", "average_deal_size", "sales_velocity"],
                "typical_filters": ["date_range", "product_line", "customer_segment"]
            },
            DomainType.FINANCE: {
                "common_aggregations": ["SUM(amount)", "AVG(expense)", "COUNT(*)"],
                "common_groupings": ["department", "category", "month", "cost_center"],
                "important_metrics": ["burn_rate", "gross_margin", "expense_ratio"],
                "typical_filters": ["fiscal_period", "account_type", "vendor"]
            },
            DomainType.MARKETING: {
                "common_aggregations": ["SUM(clicks)", "COUNT(DISTINCT leads)", "AVG(ctr)"],
                "common_groupings": ["campaign", "channel", "source", "audience"],
                "important_metrics": ["cac", "ltv", "roas", "conversion_rate"],
                "typical_filters": ["campaign_type", "date_range", "channel"]
            },
            DomainType.OPERATIONS: {
                "common_aggregations": ["SUM(quantity)", "AVG(lead_time)", "COUNT(*)"],
                "common_groupings": ["warehouse", "product", "supplier", "status"],
                "important_metrics": ["inventory_turnover", "fill_rate", "cycle_time"],
                "typical_filters": ["location", "product_category", "date_range"]
            },
            DomainType.HR: {
                "common_aggregations": ["COUNT(*)", "AVG(salary)", "SUM(hours)"],
                "common_groupings": ["department", "role", "location", "tenure"],
                "important_metrics": ["turnover_rate", "time_to_hire", "employee_satisfaction"],
                "typical_filters": ["employment_status", "department", "hire_date"]
            },
            DomainType.GENERAL: {
                "common_aggregations": ["COUNT(*)", "SUM()", "AVG()"],
                "common_groupings": ["category", "date", "status"],
                "important_metrics": [],
                "typical_filters": ["date_range"]
            }
        }
        
        return hints.get(domain, hints[DomainType.GENERAL])
    
    def _get_domain_metrics(self, domain: DomainType) -> List[str]:
        """Returns suggested metrics for the domain"""
        metrics = {
            DomainType.SALES: [
                "Total Revenue",
                "Number of Customers",
                "Average Order Value",
                "Sales Growth Rate"
            ],
            DomainType.FINANCE: [
                "Total Expenses",
                "Budget Variance",
                "Gross Margin",
                "Operating Cash Flow"
            ],
            DomainType.MARKETING: [
                "Lead Generation",
                "Conversion Rate",
                "Customer Acquisition Cost",
                "Return on Ad Spend"
            ],
            DomainType.OPERATIONS: [
                "Inventory Levels",
                "Order Fulfillment Rate",
                "Average Lead Time",
                "Capacity Utilization"
            ],
            DomainType.HR: [
                "Headcount",
                "Turnover Rate",
                "Average Tenure",
                "Training Hours"
            ],
            DomainType.GENERAL: []
        }
        
        return metrics.get(domain, [])