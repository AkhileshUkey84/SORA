# NL to SQL translation
# agents/sql_generator.py
"""
Core NL-to-SQL translation agent using Gemini with schema awareness.
Implements few-shot learning and safety constraints for accurate generation.
"""

from typing import Dict, Any, List, Optional
from agents.base import BaseAgent, AgentContext, AgentResult
from services.llm_service import LLMService
from services.database_service import DatabaseService
import json


class SQLGeneratorAgent(BaseAgent):
    """
    Translates natural language to SQL using LLM with schema context.
    Prioritizes accuracy and safety over query complexity.
    """
    
    def __init__(self, llm_service: LLMService, db_service: DatabaseService):
        super().__init__("SQLGeneratorAgent")
        self.llm = llm_service
        self.db = db_service
        
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Generates SQL from natural language with schema awareness.
        Uses few-shot prompting for consistent, safe SQL generation.
        """
        
        # Get table schema for the dataset
        schema = await self.db.get_table_schema(context.dataset_id)
        if not schema:
            return AgentResult(
                success=False,
                error=f"Dataset {context.dataset_id} not found"
            )
        
        # Get conversation context if available
        conv_context = context.metadata.get("conversation_context", {})
        
        # Build the prompt with schema and examples
        prompt = self._build_generation_prompt(
            query=context.query,
            schema=schema,
            conversation_context=conv_context
        )
        
        # Generate SQL using LLM
        self.logger.info(f"Starting SQL generation for query: {context.query[:100]}")
        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,  # Low temperature for consistency
                max_tokens=512,
                schema_info=schema,
                user_query=context.query
            )
            
            # Parse SQL from response
            sql = self._extract_sql_from_response(response)
            
            if not sql:
                return AgentResult(
                    success=False,
                    error="Failed to generate valid SQL"
                )
            
            # Basic validation
            if not self._is_select_only(sql):
                return AgentResult(
                    success=False,
                    error="Only SELECT queries are allowed"
                )
            
            return AgentResult(
                success=True,
                data={"sql": sql, "raw_response": response},
                confidence=0.9 if conv_context else 0.95
            )
            
        except Exception as e:
            self.logger.error("SQL generation failed", error=str(e))
            return AgentResult(
                success=False,
                error=f"SQL generation error: {str(e)}"
            )
    
    def _build_generation_prompt(self, query: str, schema: Dict[str, Any],
                               conversation_context: Dict[str, Any]) -> str:
        """
        Constructs a comprehensive prompt with schema context and examples.
        Designed to maximize accuracy and minimize hallucination.
        """
        
        # Format schema for readability
        schema_str = self._format_schema(schema)
        
        # Include conversation context if available
        context_str = ""
        if conversation_context.get("previous_tables"):
            context_str += f"\nPrevious tables used: {', '.join(conversation_context['previous_tables'])}"
        if conversation_context.get("previous_filters"):
            context_str += f"\nPrevious filters: {'; '.join(conversation_context['previous_filters'][:2])}"
        
        # Get the actual table name for examples
        table_name = schema['table_name']
        
        self.logger.info(f"SQL Generator using table name: {table_name}")
        self.logger.info(f"Schema table_name field: {schema.get('table_name')}")
        
        prompt = f"""You are a SQL expert. Generate a DuckDB SQL query for the following request.

DATABASE SCHEMA:
{schema_str}

CONVERSATION CONTEXT:{context_str if context_str else " None"}

RULES:
1. Generate ONLY a SELECT statement
2. Use ONLY the columns that exist in the schema
3. Be precise with column names (case-sensitive)
4. Include appropriate JOINs if multiple tables exist
5. Add LIMIT clause if not specified (default LIMIT 100)
6. Use DuckDB SQL syntax
7. Return ONLY the SQL query, no explanations
8. Use the exact table name: {table_name}
9. Do NOT use generic table names like "dataset" - always use the full table name

EXAMPLES:
Request: "Show me total sales by category"
SQL: SELECT category, SUM(sales_amount) as total_sales FROM {table_name} GROUP BY category ORDER BY total_sales DESC LIMIT 100;

Request: "What are the top 5 products?"
SQL: SELECT product_name, COUNT(*) as count FROM {table_name} GROUP BY product_name ORDER BY count DESC LIMIT 5;

Request: "Filter by date in 2024"
SQL: SELECT * FROM {table_name} WHERE date >= '2024-01-01' AND date < '2025-01-01' LIMIT 100;

USER REQUEST: {query}

SQL:"""
        
        return prompt
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Formats schema information for LLM consumption"""
        lines = [f"Table: {schema['table_name']}"]
        lines.append("Columns:")
        
        for col in schema['columns']:
            nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
            lines.append(f"  - {col['name']} ({col['type']}) {nullable}")
            
            # Add sample values if available
            if col.get('sample_values'):
                samples = ', '.join(str(v) for v in col['sample_values'][:3])
                lines.append(f"    Sample values: {samples}")
        
        if schema.get('row_count'):
            lines.append(f"\nTotal rows: {schema['row_count']:,}")
        
        return '\n'.join(lines)
    
    def _extract_sql_from_response(self, response: str) -> Optional[str]:
        """
        Extracts clean SQL from LLM response.
        Handles various response formats robustly.
        """
        
        # Remove markdown code blocks if present
        if "```sql" in response:
            start = response.find("```sql") + 6
            end = response.find("```", start)
            if end > start:
                response = response[start:end]
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end]
        
        # Clean up the SQL
        sql = response.strip()
        
        # Remove any trailing semicolon (DuckDB doesn't require it)
        sql = sql.rstrip(';')
        
        # Basic validation
        if not sql.upper().startswith('SELECT'):
            return None
        
        # Ensure it has a LIMIT clause
        if 'LIMIT' not in sql.upper():
            sql += ' LIMIT 100'
        
        return sql
    
    def _is_select_only(self, sql: str) -> bool:
        """Validates that SQL is read-only SELECT statement"""
        
        # Forbidden keywords
        forbidden = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
            'ALTER', 'TRUNCATE', 'EXEC', 'EXECUTE'
        ]
        
        sql_upper = sql.upper()
        
        # Check it starts with SELECT
        if not sql_upper.strip().startswith('SELECT'):
            return False
        
        # Check for forbidden operations
        for keyword in forbidden:
            if keyword in sql_upper:
                return False
        
        return True