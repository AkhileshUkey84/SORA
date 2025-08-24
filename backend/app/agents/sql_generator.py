import logging
import json
from typing import Dict, Any, Optional, List
from app.services.llm_service import LLMService
from app.models.query import QueryPlan
from app.utils.table_utils import sanitize_table_name

logger = logging.getLogger(__name__)

class SQLGenerator:
    """Generates SQL queries from natural language"""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        
    async def generate(
        self,
        question: str,
        schema: Dict[str, Any],
        plan: QueryPlan,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Generate SQL query from natural language question"""
        
        # Build context-aware prompt
        prompt = self._build_prompt(question, schema, plan, dataset_name)
        
        # Generate SQL using LLM
        sql_response = await self.llm_service.generate_sql(prompt)
        
        # Post-process and validate syntax
        sql_query = self._post_process_sql(sql_response, dataset_name)
        
        # Generate confidence score
        confidence = self._calculate_confidence(sql_query, plan)
        
        result = {
            "sql": sql_query,
            "confidence": confidence,
            "reasoning": sql_response.get("reasoning", ""),
            "alternatives": sql_response.get("alternatives", [])
        }
        
        logger.info(f"Generated SQL: {sql_query[:100]}...")
        return result
    
    def _build_prompt(
        self,
        question: str,
        schema: Dict[str, Any],
        plan: QueryPlan,
        dataset_name: str
    ) -> str:
        """Build a comprehensive prompt for SQL generation"""
        
        # Format schema information
        schema_str = self._format_schema(schema)
        
        prompt = f"""You are a SQL expert. Generate a SQL query for the following question.

Dataset: {dataset_name}
Question: {question}
Query Intent: {plan.intent}
Complexity: {plan.complexity}

Schema:
{schema_str}

Important Rules:
1. Use the exact table name: {dataset_name}
2. Use only columns that exist in the schema
3. Always add LIMIT 10000 to prevent large results
4. For time-based queries, use appropriate date functions
5. For aggregations, include GROUP BY when needed
6. Use clear column aliases for readability

Generate a SQL query that answers the question. Also provide:
1. Your reasoning for the query structure
2. Any assumptions you made
3. Alternative query approaches if applicable

Return as JSON:
{{
    "sql": "your SQL query",
    "reasoning": "explanation of your approach",
    "assumptions": ["list of assumptions"],
    "alternatives": ["alternative SQL approaches"]
}}
"""
        return prompt
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema for prompt"""
        lines = []
        for col_name, col_info in schema.items():
            lines.append(f"- {col_name}: {col_info.get('type', 'unknown')}")
            if col_info.get('sample_values'):
                samples = col_info['sample_values'][:3]
                lines.append(f"  Sample values: {samples}")
        return "\n".join(lines)
    
    def _post_process_sql(self, sql_response: Dict[str, Any], dataset_name: str) -> str:
        """Post-process and clean the generated SQL"""
        sql = sql_response.get("sql", "")
        
        # Sanitize table name (replace hyphens with underscores)
        sanitized_table_name = sanitize_table_name(dataset_name)
        
        # Ensure proper table name
        sql = sql.replace("FROM table", f"FROM {sanitized_table_name}")
        sql = sql.replace("from table", f"FROM {sanitized_table_name}")
        sql = sql.replace(dataset_name, sanitized_table_name)
        
        # Add LIMIT if not present
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + " LIMIT 10000;"
        
        # Clean up formatting
        sql = sql.strip()
        if not sql.endswith(";"):
            sql += ";"
        
        return sql
    
    def _calculate_confidence(self, sql: str, plan: QueryPlan) -> float:
        """Calculate confidence score for the generated SQL"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for simpler queries
        if plan.complexity == "simple":
            confidence += 0.3
        elif plan.complexity == "moderate":
            confidence += 0.2
        
        # Check for common SQL patterns
        if "SELECT" in sql and "FROM" in sql:
            confidence += 0.1
        
        # Reduce confidence for complex joins
        if plan.requires_joins:
            confidence -= 0.1
        
        # Cap confidence
        return min(max(confidence, 0.1), 0.95)