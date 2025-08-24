import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
import duckdb
from app.config import settings

logger = logging.getLogger(__name__)

class Executor:
    """Executes SQL queries safely against the database"""
    
    def __init__(self):
        self.connection = None
        
    async def execute(
        self,
        sql: str,
        dataset_name: str,
        timeout: int = settings.QUERY_TIMEOUT
    ) -> Dict[str, Any]:
        """Execute SQL query with timeout and safety measures"""
        
        start_time = time.time()
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_query(sql, dataset_name),
                timeout=timeout
            )
            
            execution_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "success": True,
                "results": result["rows"],
                "columns": result["columns"],
                "row_count": len(result["rows"]),
                "execution_time_ms": execution_time,
                "truncated": result.get("truncated", False)
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Query timeout after {timeout} seconds")
            return {
                "success": False,
                "error": f"Query exceeded timeout of {timeout} seconds",
                "execution_time_ms": timeout * 1000
            }
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _execute_query(self, sql: str, dataset_name: str) -> Dict[str, Any]:
        """Execute the actual query"""
        conn = duckdb.connect(":memory:")
        
        try:
            # Create sample data for testing
            self._create_sample_data(conn, dataset_name)
            
            # Execute the query
            result = conn.execute(sql).fetchall()
            columns = [desc[0] for desc in conn.description] if conn.description else []
            
            # Convert to list of dicts
            rows = []
            for row in result[:settings.MAX_QUERY_ROWS]:
                rows.append(dict(zip(columns, row)))
            
            truncated = len(result) > settings.MAX_QUERY_ROWS
            
            return {
                "rows": rows,
                "columns": columns,
                "truncated": truncated
            }
            
        finally:
            conn.close()
    
    def _create_sample_data(self, conn: duckdb.DuckDBPyConnection, dataset_name: str) -> None:
        """Create sample data for testing purposes"""
        # Sanitize table name
        from app.utils.table_utils import sanitize_table_name
        safe_table_name = sanitize_table_name(dataset_name)
        
        # Create a sample table with realistic e-commerce data
        create_table_sql = f"""
        CREATE TABLE {safe_table_name} (
            order_id INTEGER,
            customer_name VARCHAR,
            product_name VARCHAR,
            quantity INTEGER,
            price DECIMAL(10,2),
            order_date DATE,
            region VARCHAR
        )
        """
        
        # Sample data
        insert_data_sql = f"""
        INSERT INTO {safe_table_name} VALUES
            (1, 'John Smith', 'Widget A', 2, 19.99, '2024-01-01', 'North'),
            (2, 'Jane Doe', 'Widget B', 1, 29.99, '2024-01-02', 'South'),
            (3, 'Bob Wilson', 'Widget A', 3, 19.99, '2024-01-03', 'East'),
            (4, 'Alice Johnson', 'Widget C', 1, 39.99, '2024-01-04', 'West'),
            (5, 'Charlie Brown', 'Widget B', 2, 29.99, '2024-01-05', 'North'),
            (6, 'Diana Prince', 'Widget A', 1, 19.99, '2024-01-06', 'South'),
            (7, 'Eve Adams', 'Widget C', 2, 39.99, '2024-01-07', 'East'),
            (8, 'Frank Miller', 'Widget B', 3, 29.99, '2024-01-08', 'West'),
            (9, 'Grace Lee', 'Widget A', 1, 19.99, '2024-01-09', 'North'),
            (10, 'Henry Ford', 'Widget C', 1, 39.99, '2024-01-10', 'South')
        """
        
        try:
            conn.execute(create_table_sql)
            conn.execute(insert_data_sql)
            logger.info(f"Created sample table {safe_table_name} with test data")
        except Exception as e:
            logger.warning(f"Could not create sample data: {e}")
    
    async def create_counterfactual(
        self,
        original_sql: str,
        counterfactual_condition: str,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Execute a counterfactual query"""
        
        # Modify SQL to add/remove condition
        if "WHERE" in original_sql.upper():
            # Add additional condition
            counterfactual_sql = original_sql.replace(
                "WHERE",
                f"WHERE NOT ({counterfactual_condition}) AND"
            )
        else:
            # Add WHERE clause
            counterfactual_sql = original_sql.replace(
                "FROM " + dataset_name,
                f"FROM {dataset_name} WHERE NOT ({counterfactual_condition})"
            )
        
        # Execute counterfactual
        result = await self.execute(counterfactual_sql, dataset_name)
        
        if result["success"]:
            return {
                "original_sql": original_sql,
                "counterfactual_sql": counterfactual_sql,
                "condition_removed": counterfactual_condition,
                "original_rows": result.get("row_count", 0),
                "results": result["results"]
            }
        else:
            return {
                "error": "Failed to generate counterfactual",
                "details": result.get("error")
            }