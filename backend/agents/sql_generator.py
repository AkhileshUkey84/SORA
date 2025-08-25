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
        Generates SQL from natural language with strict schema adherence.
        Implements pre-flight validation to prevent type mismatches.
        """
        
        # Get table schema for the dataset
        schema = await self.db.get_table_schema(context.dataset_id)
        if not schema:
            return AgentResult(
                success=False,
                error=f"Dataset {context.dataset_id} not found"
            )
        
        # STEP 1: Pre-flight analysis - detect potential type mismatches
        pre_flight_check = self._analyze_query_requirements(context.query, schema)
        if not pre_flight_check['valid']:
            return AgentResult(
                success=False,
                error=pre_flight_check['user_friendly_error'],
                data={"suggested_alternative": pre_flight_check.get('suggestion')}
            )
        
        # Get conversation context if available
        conv_context = context.metadata.get("conversation_context", {})
        
        # Build the prompt with enhanced schema and examples
        prompt = self._build_generation_prompt(
            query=context.query,
            schema=schema,
            conversation_context=conv_context,
            pre_flight_hints=pre_flight_check.get('hints', [])
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
            
            # STEP 1: Strict schema validation
            schema_validation = self._validate_sql_against_schema(sql, schema)
            if not schema_validation['valid']:
                return AgentResult(
                    success=False,
                    error=schema_validation['user_friendly_error'],
                    data={"suggested_fix": schema_validation.get('suggestion')}
                )
            
            # Basic security validation
            if not self._is_select_only(sql):
                return AgentResult(
                    success=False,
                    error="Only SELECT queries are allowed"
                )
            
            return AgentResult(
                success=True,
                data={
                    "sql": sql, 
                    "raw_response": response,
                    "schema_validation": schema_validation
                },
                confidence=0.9 if conv_context else 0.95
            )
            
        except Exception as e:
            self.logger.error("SQL generation failed", error=str(e))
            return AgentResult(
                success=False,
                error=f"SQL generation error: {str(e)}"
            )
    
    def _build_generation_prompt(self, query: str, schema: Dict[str, Any],
                               conversation_context: Dict[str, Any], 
                               pre_flight_hints: List[str] = None) -> str:
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
            col_line = f"  - {col['name']} ({col['type']}) {nullable}"
            
            # Add semantic type info if available
            if col.get('semantic_type'):
                semantic_type = col['semantic_type']
                if semantic_type == 'numeric' and col.get('aggregable'):
                    col_line += f" [NUMERIC - can use SUM, AVG, etc.]"
                elif semantic_type == 'categorical':
                    col_line += f" [CATEGORICAL - good for GROUP BY]"
                elif col['name'].endswith('_notes'):
                    col_line += f" [NOTES - contains non-numeric values like 'C' for confidential data]"
            
            lines.append(col_line)
            
            # Add sample values if available
            if col.get('sample_values'):
                samples = ', '.join(str(v) for v in col['sample_values'][:3])
                lines.append(f"    Sample values: {samples}")
            elif col.get('unique_values') and len(col['unique_values']) <= 5:
                values = ', '.join(str(v) for v in col['unique_values'][:5])
                lines.append(f"    Possible values: {values}")
        
        if schema.get('row_count'):
            lines.append(f"\nTotal rows: {schema['row_count']:,}")
        
        # Add note about data cleaning
        notes_cols = [col['name'] for col in schema['columns'] if col['name'].endswith('_notes')]
        if notes_cols:
            lines.append(f"\nNOTE: Columns ending with '_notes' contain original non-numeric values that were cleaned for analysis.")
            lines.append(f"Use the corresponding main column for numeric operations.")
        
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
    
    def _analyze_query_requirements(self, query: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        STEP 1: Pre-flight analysis to detect type mismatches before SQL generation.
        Prevents generating invalid queries by analyzing intent vs. schema.
        """
        query_lower = query.lower()
        columns = {col['name'].lower(): col for col in schema['columns']}
        table_name = schema['table_name']
        
        # Detect aggregation requests
        agg_functions = ['sum', 'average', 'avg', 'total', 'mean']
        wants_aggregation = any(func in query_lower for func in agg_functions)
        
        if wants_aggregation:
            # Find numeric columns that can be aggregated
            numeric_cols = [col for col in schema['columns'] 
                          if col.get('semantic_type') == 'numeric' or 
                             col['type'].lower() in ['integer', 'bigint', 'double', 'float', 'decimal', 'numeric']]
            
            # Check if query mentions specific column names that might not be numeric
            mentioned_cols = []
            for col_name in columns.keys():
                if col_name in query_lower or col_name.replace('_', ' ') in query_lower:
                    mentioned_cols.append(columns[col_name])
            
            # Validate mentioned columns for aggregation
            for col in mentioned_cols:
                if col['type'].lower() in ['varchar', 'text', 'string'] and not col.get('aggregable', False):
                    return {
                        'valid': False,
                        'user_friendly_error': f"I can't calculate a {agg_functions[0] if agg_functions[0] in query_lower else 'sum'} for the '{col['name']}' column because it contains text, not numbers. Would you like me to count the occurrences instead?",
                        'suggestion': f"Try asking: 'How many different {col['name']} values are there?' or 'Count by {col['name']}'"
                    }
        
        # Detect year-over-year or time comparison requests
        if any(pattern in query_lower for pattern in ['year over year', 'yoy', 'compare', 'vs previous', 'trend']):
            date_cols = [col for col in schema['columns'] if 'date' in col['name'].lower() or 'time' in col['name'].lower()]
            if not date_cols:
                return {
                    'valid': False,
                    'user_friendly_error': "I can't perform time-based comparisons because this dataset doesn't have date columns. Would you like to see a different type of analysis?",
                    'suggestion': "Try: 'Show me the data breakdown by category' or 'What are the top values?'"
                }
        
        return {
            'valid': True,
            'hints': []
        }
    
    def _validate_sql_against_schema(self, sql: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        STEP 1: Strict post-generation schema validation.
        Catches any SQL that references non-existent columns or invalid operations.
        """
        import re
        
        sql_upper = sql.upper()
        columns = {col['name']: col for col in schema['columns']}
        column_names = set(columns.keys())
        table_name = schema['table_name']
        
        # Check table name usage
        if table_name.lower() not in sql.lower():
            return {
                'valid': False,
                'user_friendly_error': f"The generated query doesn't use the correct table name. Expected '{table_name}' but it may be missing or incorrect.",
                'suggestion': f"The query should reference table '{table_name}'"
            }
        
        # Extract column references from SQL
        # This regex finds column names after SELECT, WHERE, GROUP BY, ORDER BY
        col_pattern = r'(?:SELECT|FROM|WHERE|GROUP BY|ORDER BY|HAVING)\s+(?:[^\s,()]+\.)?([a-zA-Z_][a-zA-Z0-9_]*)'  
        referenced_columns = set(re.findall(col_pattern, sql, re.IGNORECASE))
        
        # Also check for columns in function calls like SUM(column_name)
        func_pattern = r'(?:SUM|AVG|COUNT|MAX|MIN)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)'
        function_columns = set(re.findall(func_pattern, sql, re.IGNORECASE))
        referenced_columns.update(function_columns)
        
        # Remove common SQL keywords that might be captured
        sql_keywords = {'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'BY', 'AS', 'AND', 'OR', 'COUNT', 'SUM', 'AVG', 'LIMIT'}
        referenced_columns = {col for col in referenced_columns if col.upper() not in sql_keywords}
        
        # Check for non-existent columns
        invalid_columns = referenced_columns - column_names
        if invalid_columns:
            invalid_col = list(invalid_columns)[0]
            # Suggest similar column names
            suggestions = [col for col in column_names if col.lower().startswith(invalid_col.lower()[:3])]
            suggestion_text = f"Did you mean: {', '.join(suggestions[:3])}?" if suggestions else "Check the available column names."
            
            return {
                'valid': False,
                'user_friendly_error': f"The column '{invalid_col}' doesn't exist in the dataset. {suggestion_text}",
                'suggestion': f"Available columns include: {', '.join(list(column_names)[:5])}..."
            }
        
        # Validate aggregation functions on appropriate column types
        agg_matches = re.findall(r'(SUM|AVG)\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)', sql, re.IGNORECASE)
        for func, col_name in agg_matches:
            if col_name in columns:
                col = columns[col_name]
                if col['type'].lower() in ['varchar', 'text', 'string'] and not col.get('aggregable', False):
                    return {
                        'valid': False,
                        'user_friendly_error': f"I can't calculate {func.lower()} on '{col_name}' because it contains text values, not numbers. Would you like me to count them instead?",
                        'suggestion': f"Try: 'COUNT({col_name})' or 'COUNT(*)' to count occurrences"
                    }
        
        return {
            'valid': True,
            'validated_columns': list(referenced_columns),
            'table_name': table_name
        }
