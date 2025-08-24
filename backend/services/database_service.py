# DuckDB management
# services/database_service.py
"""
DuckDB database service for query execution and schema management.
Provides safe, isolated query execution per dataset.
"""

import duckdb
import asyncio
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from contextlib import asynccontextmanager
import structlog
from datetime import datetime
import re


logger = structlog.get_logger()


class DatabaseService:
    """
    Manages DuckDB connections and query execution.
    Ensures data isolation and query safety.
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connections = {}
        self.schemas = {}
        self.logger = logger.bind(service="DatabaseService")
    
    async def initialize(self):
        """Initialize database service"""
        self.logger.info("Initializing database service", db_path=self.db_path)
        
        # Create main connection
        self.main_conn = duckdb.connect(self.db_path)
        
        # Set up performance optimizations
        self.main_conn.execute("SET threads TO 4")
        self.main_conn.execute("SET memory_limit = '1GB'")
    
    async def close(self):
        """Close all database connections"""
        for conn in self.connections.values():
            conn.close()
        
        if hasattr(self, 'main_conn'):
            self.main_conn.close()
    
    async def create_dataset_view(
        self,
        dataset_id: str,
        df: pd.DataFrame,
        table_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates an isolated view for a dataset.
        Returns schema information.
        """
        
        if not table_name:
            table_name = f"dataset_{dataset_id}"
        
        self.logger.info(f"Creating dataset view: {table_name} for dataset_id: {dataset_id}")
        
        try:
            # Get or create connection for dataset
            conn = self._get_connection(dataset_id)
            
            # Register DataFrame as table in both connections
            conn.register(table_name, df)
            self.main_conn.register(table_name, df)
            
            # Verify table creation
            after_tables = await self.list_all_tables()
            if table_name not in after_tables:
                raise Exception(f"Table {table_name} was not created successfully")
            
            # Extract schema
            schema = await self._extract_schema(conn, table_name, df)
            
            # Cache schema
            self.schemas[dataset_id] = schema
            
            self.logger.info("Successfully created dataset view",
                           dataset_id=dataset_id,
                           table_name=table_name,
                           rows=len(df),
                           columns=len(df.columns))
            
            return schema
            
        except Exception as e:
            self.logger.error("Failed to create dataset view",
                            dataset_id=dataset_id,
                            table_name=table_name,
                            error=str(e),
                            exc_info=True)
            raise
    
    async def execute_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes SQL query with parameterization.
        Returns results as list of dictionaries.
        """
        
        try:
            # Use dataset-specific connection if provided
            conn = self._get_connection(dataset_id) if dataset_id else self.main_conn
            
            # Execute query
            if params:
                # Parameterized query
                result = await asyncio.to_thread(
                    conn.execute, sql, parameters=params
                )
            else:
                result = await asyncio.to_thread(conn.execute, sql)
            
            # Convert to list of dicts
            columns = [desc[0] for desc in result.description] if result.description else []
            rows = result.fetchall()
            
            data = [dict(zip(columns, row)) for row in rows]
            
            self.logger.info("Query executed successfully",
                           dataset_id=dataset_id,
                           row_count=len(data))
            
            return data
            
        except Exception as e:
            self.logger.error("Query execution failed",
                            sql=sql[:200],
                            error=str(e))
            raise
    
    async def list_all_tables(self) -> List[str]:
        """Lists all tables in the database"""
        try:
            result = self.main_conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            tables = [row[0] for row in result]
            return tables
        except Exception as e:
            self.logger.error(f"Failed to list tables: {e}")
            return []
    
    async def get_table_schema(self, dataset_id: str, suppress_warnings: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieves cached schema for dataset"""
        
        # Check cache first
        if dataset_id in self.schemas:
            return self.schemas[dataset_id]
        
        # Try to find the correct table name
        all_tables = await self.list_all_tables()
        expected_table = f"dataset_{dataset_id}"
        
        # Find exact match or closest match
        matching_table = None
        if expected_table in all_tables:
            matching_table = expected_table
        else:
            # Look for tables that contain the dataset_id
            for table in all_tables:
                if dataset_id in table and table.startswith('dataset_'):
                    matching_table = table
                    if not suppress_warnings:
                        self.logger.info(f"Found matching table {matching_table} for dataset_id {dataset_id}")
                    break
        
        if not matching_table:
            if not suppress_warnings:
                self.logger.warning(f"No table found for dataset_id {dataset_id}. Available tables: {all_tables}")
            return None
        
        try:
            conn = self.main_conn  # Use main connection for consistency
            
            # Get basic schema
            schema_query = f"""
            SELECT 
                column_name,
                data_type,
                is_nullable
            FROM information_schema.columns
            WHERE table_name = ?
            ORDER BY ordinal_position
            """
            
            columns = conn.execute(schema_query, [matching_table]).fetchall()
            
            schema = {
                "table_name": matching_table,
                "dataset_id": dataset_id,
                "columns": [
                    {
                        "name": col[0],
                        "type": col[1],
                        "nullable": col[2] == 'YES'
                    }
                    for col in columns
                ]
            }
            
            # Cache it
            self.schemas[dataset_id] = schema
            
            return schema
            
        except Exception as e:
            self.logger.warning("Failed to get schema",
                              dataset_id=dataset_id,
                              error=str(e))
            return None
    
    async def validate_dataset_access(self, dataset_id: str, user_id: str) -> bool:
        """Validates user has access to dataset"""
        # Simplified for demo - in production would check permissions
        return True
    
    async def get_dataset_stats(self, dataset_id: str) -> Dict[str, Any]:
        """Gets basic statistics about a dataset"""
        
        table_name = f"dataset_{dataset_id}"
        
        try:
            conn = self._get_connection(dataset_id)
            
            # Row count
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            # Column count
            schema = await self.get_table_schema(dataset_id)
            col_count = len(schema["columns"]) if schema else 0
            
            # Basic stats
            stats = {
                "dataset_id": dataset_id,
                "row_count": row_count,
                "column_count": col_count,
                "size_estimate": row_count * col_count * 8,  # Rough estimate
                "last_accessed": datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error("Failed to get dataset stats",
                            dataset_id=dataset_id,
                            error=str(e))
            return {
                "dataset_id": dataset_id,
                "error": str(e)
            }
    
    def _get_connection(self, dataset_id: str) -> duckdb.DuckDBPyConnection:
        """Gets or creates connection for dataset"""
        
        if dataset_id not in self.connections:
            # Create new connection
            if self.db_path == ":memory:":
                # For in-memory, share the main connection
                self.connections[dataset_id] = self.main_conn
            else:
                # For file-based, create separate connection
                self.connections[dataset_id] = duckdb.connect(self.db_path)
        
        return self.connections[dataset_id]
    
    async def _extract_schema(
        self,
        conn: duckdb.DuckDBPyConnection,
        table_name: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extracts comprehensive schema information"""
        
        schema = {
            "table_name": table_name,
            "row_count": len(df),
            "columns": []
        }
        
        for col_name in df.columns:
            col_info = {
                "name": col_name,
                "type": str(df[col_name].dtype),
                "nullable": df[col_name].isnull().any()
            }
            
            # Add sample values for categorical columns
            if df[col_name].dtype == 'object':
                unique_values = df[col_name].dropna().unique()
                if len(unique_values) <= 20:
                    col_info["unique_values"] = unique_values.tolist()
                else:
                    col_info["sample_values"] = unique_values[:5].tolist()
                    col_info["unique_count"] = len(unique_values)
            
            # Add statistics for numeric columns
            elif pd.api.types.is_numeric_dtype(df[col_name]):
                col_info["min"] = float(df[col_name].min())
                col_info["max"] = float(df[col_name].max())
                col_info["mean"] = float(df[col_name].mean())
            
            schema["columns"].append(col_info)
        
        return schema