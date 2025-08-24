import logging
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime
from app.agents import (
    DomainRouter, QueryPlanner, SQLGenerator, 
    Validator, Executor, Explainer
)
from app.services.llm_service import LLMService
from app.services.dataset_service import DatasetService
from app.utils.cache import cache_get, cache_set
from app.utils.table_utils import sanitize_table_name
from app.models.query import QueryRequest, QueryResult
from app.config import settings

logger = logging.getLogger(__name__)

class QueryService:
    """Orchestrates the entire query processing pipeline"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.dataset_service = DatasetService()
        self.domain_router = DomainRouter()
        self.query_planner = QueryPlanner()
        self.sql_generator = SQLGenerator(self.llm_service)
        self.validator = Validator()
        self.executor = Executor()
        self.explainer = Explainer(self.llm_service)
    
    async def process_query(
        self,
        request: QueryRequest,
        user_id: str
    ) -> QueryResult:
        """Process a natural language query through the entire pipeline"""
        
        # Check cache
        cache_key = self._generate_cache_key(request)
        cached_result = await cache_get(cache_key)
        if cached_result:
            logger.info("Returning cached result")
            return QueryResult(**cached_result)
        
        try:
            # Get dataset schema
            # In production, load from database
            schema = await self._get_dataset_schema(request.dataset_id)
            
            # Check if this is a schema information request
            if self._is_schema_info_request(request.question):
                return await self._handle_schema_info_request(request, schema)
            
            # Route domain
            routing = await self.domain_router.route(request.question, schema)
            
            # Plan query
            plan = await self.query_planner.plan(
                request.question,
                schema,
                routing
            )
            
            # Check for clarifications needed
            if plan.clarifications_needed:
                return QueryResult(
                    question=request.question,
                    sql_query="",
                    results=[],
                    row_count=0,
                    execution_time_ms=0,
                    explanation="Clarification needed",
                    clarifications=plan.clarifications_needed,
                    confidence_score=0,
                    provenance={}
                )
            
            # Generate SQL
            sql_result = await self.sql_generator.generate(
                request.question,
                schema,
                plan,
                f"dataset_{request.dataset_id}"
            )
            
            # Validate SQL
            guardrail_result = await self.validator.validate(
                sql_result["sql"],
                schema,
                request.question
            )
            
            if not guardrail_result.passed:
                return QueryResult(
                    question=request.question,
                    sql_query=sql_result["sql"],
                    results=[],
                    row_count=0,
                    execution_time_ms=0,
                    explanation="Query blocked by safety guardrails",
                    guardrail_report=guardrail_result,
                    confidence_score=0,
                    provenance={}
                )
            
            # Use modified query if needed
            final_sql = guardrail_result.modified_query or sql_result["sql"]
            
            # Execute query
            execution_result = await self.executor.execute(
                final_sql,
                f"dataset_{request.dataset_id}"
            )
            
            if not execution_result["success"]:
                return QueryResult(
                    question=request.question,
                    sql_query=final_sql,
                    results=[],
                    row_count=0,
                    execution_time_ms=execution_result["execution_time_ms"],
                    explanation=f"Query execution failed: {execution_result['error']}",
                    guardrail_report=guardrail_result,
                    confidence_score=sql_result["confidence"],
                    provenance={}
                )
            
            # Generate counterfactual if requested
            counterfactual = None
            if request.enable_counterfactual and execution_result["results"]:
                counterfactual = await self._generate_counterfactual(
                    final_sql,
                    request.question,
                    f"dataset_{request.dataset_id}"
                )
            
            # Generate explanation
            explanation_result = None
            if request.enable_explanation:
                explanation_result = await self.explainer.explain(
                    request.question,
                    final_sql,
                    execution_result["results"],
                    counterfactual
                )
            
            # Create result
            result = QueryResult(
                question=request.question,
                sql_query=final_sql,
                results=execution_result["results"],
                row_count=execution_result["row_count"],
                execution_time_ms=execution_result["execution_time_ms"],
                explanation=explanation_result["narrative"] if explanation_result else None,
                counterfactual=counterfactual,
                guardrail_report=guardrail_result,
                confidence_score=sql_result["confidence"],
                provenance=self._generate_provenance(request, final_sql)
            )
            
            # Cache result
            await cache_set(cache_key, result.dict(), ttl=settings.CACHE_TTL)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            return QueryResult(
                question=request.question,
                sql_query="",
                results=[],
                row_count=0,
                execution_time_ms=0,
                explanation=f"Error processing query: {str(e)}",
                confidence_score=0,
                provenance={}
            )
    
    def _generate_cache_key(self, request: QueryRequest) -> str:
        """Generate cache key for query"""
        key_data = f"{request.question}:{request.dataset_id}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _get_dataset_schema(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset schema"""
        # In production, load from database
        # For now, return mock schema
        return {
            "order_id": {"type": "integer", "sample_values": [1, 2, 3]},
            "customer_name": {"type": "string", "sample_values": ["John", "Jane"], "has_pii": True, "pii_type": "name"},
            "product_name": {"type": "string", "sample_values": ["Widget A", "Widget B"]},
            "quantity": {"type": "integer", "sample_values": [1, 5, 10]},
            "price": {"type": "float", "sample_values": [9.99, 19.99, 29.99]},
            "order_date": {"type": "date", "sample_values": ["2024-01-01", "2024-01-02"]},
            "region": {"type": "string", "sample_values": ["North", "South", "East", "West"]}
        }
    
    async def _generate_counterfactual(
        self,
        sql: str,
        question: str,
        dataset_name: str
    ) -> Optional[Dict[str, Any]]:
        """Generate counterfactual analysis"""
        # Identify key condition to remove
        # Simplified: remove first WHERE condition
        if "WHERE" in sql.upper():
            # Extract first condition (simplified)
            condition = "promotion = true"  # Example
            return await self.executor.create_counterfactual(
                sql,
                condition,
                dataset_name
            )
        return None
    
    def _generate_provenance(
        self,
        request: QueryRequest,
        sql: str
    ) -> Dict[str, Any]:
        """Generate provenance information for reproducibility"""
        return {
            "query_hash": hashlib.md5(sql.encode()).hexdigest(),
            "dataset_id": str(request.dataset_id),
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": settings.GEMINI_MODEL,
            "context": request.context
        }
    
    def _is_schema_info_request(self, question: str) -> bool:
        """Check if the question is asking for schema information"""
        question_lower = question.lower()
        
        schema_keywords = [
            'how many column', 'how many field', 'column', 'schema', 'structure',
            'what column', 'list column', 'show column', 'describe', 'field',
            'name them', 'table structure', 'what field'
        ]
        
        return any(keyword in question_lower for keyword in schema_keywords)
    
    async def _handle_schema_info_request(self, request: QueryRequest, schema: Dict[str, Any]) -> QueryResult:
        """Handle schema information requests directly"""
        
        # Create schema information results
        columns = []
        for col_name, col_info in schema.items():
            columns.append({
                "column_name": col_name,
                "data_type": col_info.get("type", "unknown"),
                "has_pii": col_info.get("has_pii", False),
                "sample_values": col_info.get("sample_values", [])
            })
        
        # Create summary result
        result_data = [{
            "total_columns": len(schema),
            "column_names": ", ".join(schema.keys()),
            "column_details": columns
        }]
        
        # Generate explanation
        explanation = f"""Dataset Schema Information:
        
**Total Columns:** {len(schema)}
        
**Column Names:**
{chr(10).join([f"- {col_name} ({col_info.get('type', 'unknown')})" + (" [PII]" if col_info.get('has_pii') else "") for col_name, col_info in schema.items()])}
        
**Column Details:**
{chr(10).join([f"• {col_name}: {col_info.get('type', 'unknown')} type" + (f", sample values: {col_info.get('sample_values', [])[:3]}" if col_info.get('sample_values') else "") for col_name, col_info in schema.items()])}
        """
        
        return QueryResult(
            question=request.question,
            sql_query="-- Schema information query (no SQL execution needed)",
            results=result_data,
            row_count=1,
            execution_time_ms=1.0,
            explanation=explanation,
            counterfactual=None,
            guardrail_report=None,
            confidence_score=1.0,
            provenance={
                "query_type": "schema_info",
                "dataset_id": str(request.dataset_id),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
