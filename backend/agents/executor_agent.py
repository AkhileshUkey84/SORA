# Safe query execution
# agents/executor_agent.py
"""
Query execution agent that safely runs SQL against DuckDB.
Handles timeouts, resource limits, and error recovery.
"""

from typing import Dict, Any, List, Optional
import asyncio
import time
import re
from datetime import datetime
from agents.base import BaseAgent, AgentContext, AgentResult
from services.database_service import DatabaseService
from utils.config import settings
import numpy as np


class ExecutorAgent(BaseAgent):
    """
    Executes validated SQL queries with comprehensive safety measures.
    Ensures performance and prevents resource exhaustion.
    """
    
    def __init__(self, db_service: DatabaseService):
        super().__init__("ExecutorAgent")
        self.db = db_service
        self.timeout = settings.query_timeout_seconds
        self.max_rows = settings.max_query_rows
    
    async def process(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Executes SQL query with timeout and resource management.
        Returns structured results with performance metrics.
        """
        
        sql = kwargs.get("sql", "")
        if not sql:
            return AgentResult(
                success=False,
                error="No SQL query provided for execution"
            )
        
        # Add execution context
        context.metadata["execution_start"] = datetime.utcnow()
        
        try:
            # Execute with timeout
            start_time = time.time()
            
            results = await asyncio.wait_for(
                self._execute_query(sql, context),
                timeout=self.timeout
            )
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            # Process results
            processed_results = self._process_results(results)
            
            # Add performance metrics
            processed_results["execution_time_ms"] = execution_time
            processed_results["query_complexity"] = self._estimate_complexity(sql)
            
            # Log successful execution
            self.logger.info("Query executed successfully",
                           rows_returned=len(processed_results.get("data", [])),
                           execution_time_ms=execution_time)
            
            return AgentResult(
                success=True,
                data=processed_results,
                confidence=0.95
            )
            
        except asyncio.TimeoutError:
            return AgentResult(
                success=False,
                error=f"Query exceeded timeout limit of {self.timeout} seconds. Please simplify your query."
            )
        
        except Exception as e:
            self.logger.error("Query execution failed", error=str(e), sql=sql[:200])
            
            # Provide user-friendly error messages
            error_message = self._format_error_message(str(e))
            
            return AgentResult(
                success=False,
                error=error_message,
                data={"original_error": str(e)}
            )
    
    async def _execute_query(self, sql: str, context: AgentContext) -> Any:
        """
        Executes query against DuckDB with safety measures.
        Uses parameterized queries when applicable.
        """
        
        # Extract parameters if query is parameterized
        params = context.metadata.get("query_params", {})
        
        # Create a view for the specific dataset
        dataset_view = f"dataset_{context.dataset_id}"
        
        # Log the original SQL for debugging
        self.logger.info(f"Original SQL: {sql}")
        self.logger.info(f"Dataset view: {dataset_view}")
        
        # Ensure we're querying the right dataset
        # Check if the SQL already contains the full table name or just a generic "dataset"
        if f"FROM dataset_{context.dataset_id}" in sql:
            # SQL already has the correct table name, no need to replace
            self.logger.info("SQL already contains correct table name, no replacement needed")
        elif "FROM dataset" in sql:
            # Replace generic "dataset" with the specific table name
            sql = sql.replace("FROM dataset", f"FROM {dataset_view}")
            self.logger.info(f"Replaced generic table name with: {dataset_view}")
        else:
            # Check if there's any FROM clause with a different table name
            import re
            from_match = re.search(r'FROM\s+(\w+)', sql, re.IGNORECASE)
            if from_match:
                current_table = from_match.group(1)
                if current_table != dataset_view and not current_table.startswith(f"dataset_{context.dataset_id}"):
                    # Replace the incorrect table name with the correct one
                    sql = re.sub(r'FROM\s+\w+', f'FROM {dataset_view}', sql, flags=re.IGNORECASE)
                    self.logger.info(f"Replaced incorrect table name '{current_table}' with '{dataset_view}'")
                else:
                    self.logger.info(f"Table name '{current_table}' is correct")
            else:
                # If no FROM clause found, this might be an error
                self.logger.warning(f"No FROM clause found in SQL: {sql}")
        
        # Log the final SQL for debugging
        self.logger.info(f"Final SQL: {sql}")
        
        # Final safety check: ensure no duplicated table names
        sql = self._clean_table_names(sql, dataset_view)
        self.logger.info(f"After cleaning: {sql}")
        
        # Execute query
        return await self.db.execute_query(
            sql=sql,
            params=params,
            dataset_id=context.dataset_id
        )
    
    def _process_results(self, raw_results: Any) -> Dict[str, Any]:
        """
        Processes raw query results into structured format.
        Handles different result types and ensures consistency.
        """
        
        if not raw_results:
            return {
                "columns": [],
                "data": [],
                "row_count": 0,
                "truncated": False
            }
        
        # Extract columns and rows
        if hasattr(raw_results, 'description'):
            columns = [desc[0] for desc in raw_results.description]
            rows = raw_results.fetchall()
        else:
            # Handle different result formats
            columns = list(raw_results[0].keys()) if raw_results else []
            rows = [list(row.values()) for row in raw_results]
        
        # Convert to list of dicts for easier processing
        data = []
        for row in rows[:self.max_rows]:  # Apply row limit
            row_dict = dict(zip(columns, row))
            # Convert numpy types to Python native types for JSON serialization
            row_dict = self._convert_numpy_types(row_dict)
            data.append(row_dict)
        
        # Check if results were truncated
        truncated = len(rows) > self.max_rows
        
        # Add data type information
        column_types = self._infer_column_types(data) if data else {}
        
        return {
            "columns": columns,
            "column_types": column_types,
            "data": data,
            "row_count": len(data),
            "total_row_count": len(rows),
            "truncated": truncated
        }
    
    def _infer_column_types(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Infers data types for each column"""
        if not data:
            return {}
        
        column_types = {}
        
        for column in data[0].keys():
            # Sample first non-null value
            for row in data:
                value = row.get(column)
                if value is not None:
                    if isinstance(value, bool):
                        column_types[column] = "boolean"
                    elif isinstance(value, int):
                        column_types[column] = "integer"
                    elif isinstance(value, float):
                        column_types[column] = "float"
                    elif isinstance(value, str):
                        # Check if it's a date
                        if self._is_date_string(value):
                            column_types[column] = "date"
                        else:
                            column_types[column] = "string"
                    else:
                        column_types[column] = "unknown"
                    break
            else:
                column_types[column] = "null"
        
        return column_types
    
    def _convert_numpy_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert numpy types to Python native types for JSON serialization"""
        converted = {}
        for key, value in data.items():
            if isinstance(value, np.integer):
                converted[key] = int(value)
            elif isinstance(value, np.floating):
                converted[key] = float(value)
            elif isinstance(value, np.bool_):
                converted[key] = bool(value)
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            else:
                converted[key] = value
        return converted
    
    def _is_date_string(self, value: str) -> bool:
        """Checks if a string value represents a date"""
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}',  # ISO date
            r'^\d{1,2}/\d{1,2}/\d{2,4}',  # US date
            r'^\d{1,2}-\d{1,2}-\d{2,4}'   # Other date
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value):
                return True
        
        return False
    
    def _estimate_complexity(self, sql: str) -> str:
        """Estimates query complexity for performance insights"""
        complexity_score = 0
        
        # Check for complex operations
        if re.search(r'\bJOIN\b', sql, re.I):
            complexity_score += 2
        
        if re.search(r'\bGROUP BY\b', sql, re.I):
            complexity_score += 1
        
        if re.search(r'\bHAVING\b', sql, re.I):
            complexity_score += 1
        
        if re.search(r'\bUNION\b', sql, re.I):
            complexity_score += 3
        
        if re.search(r'\bSUBQUERY\b|KATEX_INLINE_OPEN\s*SELECT\b', sql, re.I):
            complexity_score += 2
        
        # Map to complexity level
        if complexity_score == 0:
            return "simple"
        elif complexity_score <= 2:
            return "moderate"
        else:
            return "complex"
    
    def _clean_table_names(self, sql: str, correct_table: str) -> str:
        """Cleans up any malformed or duplicated table names in SQL"""
        import re
        
        # Check for duplicated table names like "dataset_ds_123_ds_123"
        pattern = r'FROM\s+(\w+)'
        match = re.search(pattern, sql, re.IGNORECASE)
        
        if match:
            current_table = match.group(1)
            
            # If the table name contains the correct table name multiple times, clean it up
            if current_table.count(correct_table) > 1:
                # Find the first occurrence and replace the entire table name
                sql = re.sub(pattern, f'FROM {correct_table}', sql, flags=re.IGNORECASE)
                self.logger.warning(f"Cleaned up duplicated table name: {current_table} -> {correct_table}")
            
            # Also check for any table name that doesn't match the expected pattern
            elif not current_table.startswith('dataset_'):
                sql = re.sub(pattern, f'FROM {correct_table}', sql, flags=re.IGNORECASE)
                self.logger.warning(f"Replaced invalid table name: {current_table} -> {correct_table}")
        
        return sql
    
    def _format_error_message(self, error: str) -> str:
        """Formats technical errors into user-friendly messages"""
        
        error_lower = error.lower()
        
        # Common error patterns
        if "column" in error_lower and "not found" in error_lower:
            return "The query references a column that doesn't exist in the dataset. Please check column names."
        
        elif "syntax error" in error_lower:
            return "There's a syntax error in the generated SQL. Please try rephrasing your question."
        
        elif "permission" in error_lower or "access" in error_lower:
            return "You don't have permission to access this data."
        
        elif "timeout" in error_lower:
            return "The query took too long to execute. Try limiting the data range or simplifying the analysis."
        
        elif "memory" in error_lower or "resource" in error_lower:
            return "The query requires too many resources. Please try a smaller data range."
        
        else:
            # Generic error
            return f"Query execution failed: {error[:200]}"