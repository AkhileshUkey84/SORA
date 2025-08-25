# services/semantic_layer_service.py
"""
Semantic Layer Service that manages business metrics and terminology mappings.
Provides a consistent interface for business users to define and use metrics.
"""

from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
from datetime import datetime
import json
import structlog
from pathlib import Path

logger = structlog.get_logger()


class MetricDefinition(BaseModel):
    """Defines a business metric with its calculation logic"""
    id: str
    name: str
    description: str
    category: str  # sales, finance, operations, etc.
    formula: str  # SQL expression
    aggregation: str = "SUM"  # SUM, AVG, COUNT, etc.
    required_columns: List[str] = Field(default_factory=list)
    filters: Optional[Dict[str, Any]] = None
    unit: Optional[str] = None  # $, %, units, etc.
    precision: int = 2
    synonyms: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DimensionDefinition(BaseModel):
    """Defines a business dimension for grouping"""
    id: str
    name: str
    description: str
    column_mapping: str  # Actual column name
    hierarchy: Optional[List[str]] = None  # e.g., ["region", "country", "city"]
    synonyms: List[str] = Field(default_factory=list)
    default_values: Optional[List[str]] = None


class SemanticLayerService:
    """
    Manages the semantic layer providing business-friendly abstractions
    over technical database schemas. This is a key differentiator.
    """
    
    def __init__(self, config_path: str = "./semantic_config.json"):
        self.config_path = Path(config_path)
        self.metrics: Dict[str, MetricDefinition] = {}
        self.dimensions: Dict[str, DimensionDefinition] = {}
        self.term_mappings: Dict[str, str] = {}  # business term -> technical term
        self.dataset_metrics: Dict[str, Set[str]] = {}  # dataset_id -> applicable metrics
        self.logger = logger.bind(component="SemanticLayerService")
        
        # Initialize with default metrics
        self._initialize_default_metrics()
        
        # Load custom configuration if exists
        self._load_configuration()
    
    def _initialize_default_metrics(self):
        """Initialize common business metrics"""
        
        # Sales metrics
        self.add_metric(MetricDefinition(
            id="total_revenue",
            name="Total Revenue",
            description="Sum of all sales revenue",
            category="sales",
            formula="SUM(CASE WHEN {revenue_column} IS NOT NULL THEN {revenue_column} ELSE 0 END)",
            aggregation="SUM",
            required_columns=["revenue_column"],
            unit="$",
            synonyms=["revenue", "sales", "income", "total sales"],
            examples=["What's our total revenue?", "Show me sales numbers"]
        ))
        
        self.add_metric(MetricDefinition(
            id="average_order_value",
            name="Average Order Value",
            description="Average revenue per order",
            category="sales",
            formula="AVG({revenue_column})",
            aggregation="AVG",
            required_columns=["revenue_column"],
            unit="$",
            synonyms=["AOV", "avg order", "average sale"],
            examples=["What's our average order value?", "Show me AOV"]
        ))
        
        self.add_metric(MetricDefinition(
            id="customer_count",
            name="Customer Count",
            description="Number of unique customers",
            category="sales",
            formula="COUNT(DISTINCT {customer_column})",
            aggregation="COUNT",
            required_columns=["customer_column"],
            synonyms=["customers", "unique customers", "client count"],
            examples=["How many customers do we have?"]
        ))
        
        # Finance metrics
        self.add_metric(MetricDefinition(
            id="profit_margin",
            name="Profit Margin",
            description="Profit as percentage of revenue",
            category="finance",
            formula="(SUM({revenue_column}) - SUM({cost_column})) / SUM({revenue_column}) * 100",
            aggregation="CUSTOM",
            required_columns=["revenue_column", "cost_column"],
            unit="%",
            precision=1,
            synonyms=["margin", "profitability", "profit %"],
            examples=["What's our profit margin?", "Show profitability"]
        ))
        
        # Operations metrics
        self.add_metric(MetricDefinition(
            id="inventory_turnover",
            name="Inventory Turnover",
            description="How quickly inventory is sold and replaced",
            category="operations",
            formula="SUM({sales_column}) / AVG({inventory_column})",
            aggregation="CUSTOM",
            required_columns=["sales_column", "inventory_column"],
            synonyms=["turnover", "inventory rotation"],
            examples=["What's our inventory turnover rate?"]
        ))
        
        # Common dimensions
        self.add_dimension(DimensionDefinition(
            id="time_period",
            name="Time Period",
            description="Temporal grouping dimension",
            column_mapping="date",
            hierarchy=["year", "quarter", "month", "week", "day"],
            synonyms=["period", "date", "time", "when"]
        ))
        
        self.add_dimension(DimensionDefinition(
            id="product_category",
            name="Product Category",
            description="Product classification",
            column_mapping="category",
            synonyms=["category", "product type", "item category"]
        ))
        
        self.add_dimension(DimensionDefinition(
            id="geography",
            name="Geography",
            description="Geographic location",
            column_mapping="region",
            hierarchy=["region", "country", "state", "city"],
            synonyms=["location", "region", "area", "geography"]
        ))
    
    def add_metric(self, metric: MetricDefinition) -> None:
        """Add or update a metric definition"""
        self.metrics[metric.id] = metric
        
        # Update term mappings
        self.term_mappings[metric.name.lower()] = metric.id
        for synonym in metric.synonyms:
            self.term_mappings[synonym.lower()] = metric.id
        
        self.logger.info(f"Added metric: {metric.name}")
    
    def add_dimension(self, dimension: DimensionDefinition) -> None:
        """Add or update a dimension definition"""
        self.dimensions[dimension.id] = dimension
        
        # Update term mappings
        self.term_mappings[dimension.name.lower()] = dimension.id
        for synonym in dimension.synonyms:
            self.term_mappings[synonym.lower()] = dimension.id
        
        self.logger.info(f"Added dimension: {dimension.name}")
    
    def map_schema_to_metrics(self, schema: Dict[str, Any], dataset_id: str) -> List[str]:
        """
        Maps a dataset schema to applicable metrics.
        Returns list of metric IDs that can be calculated on this dataset.
        """
        columns = {col["name"].lower() for col in schema.get("columns", [])}
        applicable_metrics = []
        
        for metric_id, metric in self.metrics.items():
            # Check if required columns exist (with fuzzy matching)
            can_apply = True
            column_mappings = {}
            
            for required_col in metric.required_columns:
                # Try exact match first
                if required_col in columns:
                    column_mappings[required_col] = required_col
                    continue
                
                # Try fuzzy matching
                matched = False
                for col in columns:
                    if self._fuzzy_column_match(required_col, col):
                        column_mappings[required_col] = col
                        matched = True
                        break
                
                if not matched:
                    can_apply = False
                    break
            
            if can_apply:
                applicable_metrics.append(metric_id)
        
        # Store applicable metrics for this dataset
        self.dataset_metrics[dataset_id] = set(applicable_metrics)
        
        return applicable_metrics
    
    def _fuzzy_column_match(self, required: str, actual: str) -> bool:
        """Fuzzy matching for column names"""
        required_lower = required.lower()
        actual_lower = actual.lower()
        
        # Exact match
        if required_lower == actual_lower:
            return True
        
        # Common mappings
        mappings = {
            "revenue_column": ["revenue", "sales", "amount", "total", "price"],
            "cost_column": ["cost", "expense", "cogs"],
            "customer_column": ["customer", "client", "customer_id", "client_id"],
            "date_column": ["date", "order_date", "transaction_date", "created_at"],
            "quantity_column": ["quantity", "qty", "amount", "units"],
            "inventory_column": ["inventory", "stock", "on_hand"]
        }
        
        if required in mappings:
            return any(term in actual_lower for term in mappings[required])
        
        # Partial match
        return required_lower in actual_lower or actual_lower in required_lower
    
    def translate_business_terms(self, query: str) -> Dict[str, Any]:
        """
        Translates business terms in a query to technical terms.
        Returns enhanced query with metric definitions.
        """
        query_lower = query.lower()
        found_metrics = []
        found_dimensions = []
        suggested_metrics = []
        
        # Look for metrics
        for term, metric_id in self.term_mappings.items():
            if term in query_lower:
                if metric_id in self.metrics:
                    found_metrics.append(self.metrics[metric_id])
                elif metric_id in self.dimensions:
                    found_dimensions.append(self.dimensions[metric_id])
        
        # Suggest related metrics based on keywords
        keywords = query_lower.split()
        for keyword in keywords:
            for metric in self.metrics.values():
                if keyword in metric.category or any(keyword in ex.lower() for ex in metric.examples):
                    if metric not in found_metrics:
                        suggested_metrics.append(metric)
        
        return {
            "original_query": query,
            "found_metrics": found_metrics,
            "found_dimensions": found_dimensions,
            "suggested_metrics": suggested_metrics[:3],  # Top 3 suggestions
            "has_business_terms": len(found_metrics) > 0 or len(found_dimensions) > 0
        }
    
    def get_metric_sql(self, metric_id: str, dataset_id: str, 
                      column_mappings: Optional[Dict[str, str]] = None) -> Optional[str]:
        """
        Generate SQL for a specific metric on a dataset.
        Returns the SQL expression with proper column mappings.
        """
        if metric_id not in self.metrics:
            return None
        
        metric = self.metrics[metric_id]
        
        # Use provided mappings or try to infer
        if not column_mappings:
            # This would need actual schema inspection
            # For now, return the template
            return metric.formula
        
        # Replace placeholders with actual columns
        sql = metric.formula
        for placeholder, actual_column in column_mappings.items():
            sql = sql.replace(f"{{{placeholder}}}", actual_column)
        
        return sql
    
    def explain_metric(self, metric_id: str) -> Dict[str, Any]:
        """Get detailed explanation of a metric"""
        if metric_id not in self.metrics:
            return {"error": "Metric not found"}
        
        metric = self.metrics[metric_id]
        return {
            "name": metric.name,
            "description": metric.description,
            "category": metric.category,
            "unit": metric.unit,
            "how_calculated": f"This metric is calculated using {metric.aggregation} aggregation",
            "examples": metric.examples,
            "also_known_as": metric.synonyms
        }
    
    def suggest_metrics_for_query(self, query: str, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Suggest relevant metrics based on query intent and available data.
        This helps users discover what metrics they can calculate.
        """
        suggestions = []
        
        # Get applicable metrics for this dataset
        applicable = self.dataset_metrics.get(dataset_id, set())
        
        # Score metrics based on query relevance
        query_lower = query.lower()
        
        for metric_id in applicable:
            metric = self.metrics[metric_id]
            score = 0
            
            # Check name match
            if metric.name.lower() in query_lower:
                score += 3
            
            # Check synonym matches
            for synonym in metric.synonyms:
                if synonym.lower() in query_lower:
                    score += 2
                    break
            
            # Check example matches
            for example in metric.examples:
                if any(word in query_lower for word in example.lower().split()):
                    score += 1
            
            if score > 0:
                suggestions.append({
                    "metric": metric,
                    "score": score,
                    "reason": self._get_suggestion_reason(metric, query_lower)
                })
        
        # Sort by score and return top suggestions
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:5]
    
    def _get_suggestion_reason(self, metric: MetricDefinition, query: str) -> str:
        """Generate explanation for why a metric was suggested"""
        if metric.name.lower() in query:
            return f"You mentioned '{metric.name}'"
        
        for synonym in metric.synonyms:
            if synonym.lower() in query:
                return f"'{synonym}' relates to {metric.name}"
        
        return f"This metric is commonly used for {metric.category} analysis"
    
    def _load_configuration(self):
        """Load custom metric definitions from file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load custom metrics
                for metric_data in config.get("metrics", []):
                    metric = MetricDefinition(**metric_data)
                    self.add_metric(metric)
                
                # Load custom dimensions
                for dim_data in config.get("dimensions", []):
                    dimension = DimensionDefinition(**dim_data)
                    self.add_dimension(dimension)
                
                self.logger.info(f"Loaded configuration from {self.config_path}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load configuration: {e}")
    
    def save_configuration(self):
        """Save current metrics and dimensions to file"""
        config = {
            "metrics": [metric.dict() for metric in self.metrics.values()],
            "dimensions": [dim.dict() for dim in self.dimensions.values()],
            "last_updated": datetime.utcnow().isoformat()
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        self.logger.info(f"Saved configuration to {self.config_path}")
    
    def get_metric_library(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all available metrics, optionally filtered by category.
        Useful for UI to show metric catalog.
        """
        metrics = list(self.metrics.values())
        
        if category:
            metrics = [m for m in metrics if m.category == category]
        
        return [
            {
                "id": m.id,
                "name": m.name,
                "description": m.description,
                "category": m.category,
                "unit": m.unit,
                "examples": m.examples[:2]  # Show first 2 examples
            }
            for m in metrics
        ]
