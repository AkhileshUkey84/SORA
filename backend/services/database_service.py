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
            
            # Clean and prepare the DataFrame for DuckDB
            df_clean = self._prepare_dataframe_for_duckdb(df.copy())
            
            # Register DataFrame as table in both connections
            conn.register(table_name, df_clean)
            self.main_conn.register(table_name, df_clean)
            
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
    
    def _prepare_dataframe_for_duckdb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares DataFrame for DuckDB by handling mixed data types and problematic values.
        This fixes the 'C' to DECIMAL conversion error.
        """
        
        df_clean = df.copy()
        
        for col_name in df_clean.columns:
            col = df_clean[col_name]
            
            # Handle object columns that might contain mixed numeric/text data
            if col.dtype == 'object':
                # Check if column contains mostly numeric data
                if self._is_numeric_with_exceptions(col):
                    self.logger.info(f"Converting mixed numeric/text column '{col_name}' for DuckDB compatibility")
                    
                    # Clean the column for numeric conversion
                    df_clean[col_name] = self._clean_numeric_column(col)
                    
                    # Add a companion column to preserve original non-numeric values
                    non_numeric_col = f"{col_name}_notes"
                    df_clean[non_numeric_col] = col.where(~self._can_convert_to_numeric(col), None)
                    
                    self.logger.info(f"Created companion column '{non_numeric_col}' to preserve non-numeric values")
        
        return df_clean
    
    def _is_numeric_with_exceptions(self, series: pd.Series) -> bool:
        """
        Checks if a series contains mostly numeric data but has some non-numeric exceptions.
        Returns True if > 50% of values can be converted to numeric.
        """
        if series.empty:
            return False
        
        # Sample up to 1000 values for performance
        sample = series.dropna().head(1000)
        if len(sample) == 0:
            return False
        
        # Count how many can be converted to numeric
        numeric_count = 0
        for val in sample:
            if self._can_convert_to_numeric_value(val):
                numeric_count += 1
        
        return numeric_count / len(sample) > 0.5
    
    def _can_convert_to_numeric(self, series: pd.Series) -> pd.Series:
        """
        Returns a boolean series indicating which values can be converted to numeric.
        """
        return series.apply(self._can_convert_to_numeric_value)
    
    def _can_convert_to_numeric_value(self, value) -> bool:
        """
        Checks if a single value can be converted to numeric.
        """
        if pd.isna(value):
            return True  # NaN is numeric
        
        try:
            # Handle string values
            if isinstance(value, str):
                # Remove common thousand separators
                clean_val = value.replace(',', '').replace(' ', '').strip()
                
                # Skip obviously non-numeric values
                if clean_val.upper() in ['C', 'CONFIDENTIAL', 'SUPPRESSED', 'N/A', 'NULL', '']:
                    return False
                
                # Try to convert
                float(clean_val)
                return True
            
            # Try direct conversion for non-string types
            float(value)
            return True
            
        except (ValueError, TypeError):
            return False
    
    def _clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """
        Cleans a column with mixed numeric/text data for DuckDB.
        Converts numeric values and replaces non-numeric with NULL.
        """
        
        def convert_value(val):
            if pd.isna(val):
                return None
            
            try:
                if isinstance(val, str):
                    # Remove thousand separators
                    clean_val = val.replace(',', '').replace(' ', '').strip()
                    
                    # Handle special cases
                    if clean_val.upper() in ['C', 'CONFIDENTIAL', 'SUPPRESSED', 'N/A', 'NULL', '']:
                        return None
                    
                    # Try conversion
                    return float(clean_val)
                
                return float(val)
                
            except (ValueError, TypeError):
                return None
        
        return series.apply(convert_value)
    
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
        """Extracts comprehensive schema information with intelligent data type detection"""
        
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
            
            # Intelligent data type classification for query planning
            semantic_type, aggregable, suggested_ops = self._classify_column_semantics(df[col_name])
            col_info["semantic_type"] = semantic_type  # 'numeric', 'categorical', 'text', 'date', 'boolean'
            col_info["aggregable"] = aggregable  # Can use SUM, AVG, etc.
            col_info["suggested_operations"] = suggested_ops  # Valid operations for this column
            
            # Add detailed analysis based on semantic type
            self._add_column_details(col_info, df[col_name], semantic_type)
            
            schema["columns"].append(col_info)
        
        return schema
    
    def _classify_column_semantics(self, series: pd.Series) -> tuple[str, bool, List[str]]:
        """Intelligently classify column semantics for query planning"""
        
        # Check for numeric types first
        if pd.api.types.is_numeric_dtype(series):
            # Check if it's actually a categorical ID (high cardinality integers)
            if pd.api.types.is_integer_dtype(series):
                unique_ratio = series.nunique() / len(series)
                if unique_ratio > 0.9:  # Likely an ID or code
                    return 'categorical', False, ['COUNT', 'DISTINCT', 'FILTER', 'GROUP BY']
            
            # True numeric - can aggregate
            return 'numeric', True, ['SUM', 'AVG', 'MIN', 'MAX', 'COUNT', 'GROUP BY', 'ORDER BY', 'FILTER']
        
        # Check for boolean
        elif pd.api.types.is_bool_dtype(series):
            return 'boolean', False, ['COUNT', 'GROUP BY', 'FILTER']
        
        # Check for datetime
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'date', False, ['MIN', 'MAX', 'COUNT', 'GROUP BY', 'ORDER BY', 'FILTER', 'DATE_FUNCTIONS']
        
        # String/object types - need deeper analysis
        elif series.dtype == 'object':
            unique_count = series.nunique()
            total_count = len(series)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            # Check if it looks like a date string
            if self._is_date_column(series):
                return 'date', False, ['MIN', 'MAX', 'COUNT', 'GROUP BY', 'ORDER BY', 'FILTER']
            
            # Low cardinality = categorical (good for grouping)
            elif unique_ratio < 0.1 or unique_count < 50:
                return 'categorical', False, ['COUNT', 'DISTINCT', 'GROUP BY', 'FILTER']
            
            # High cardinality = likely text/names/descriptions
            else:
                return 'text', False, ['COUNT', 'DISTINCT', 'FILTER', 'SEARCH']
        
        # Fallback
        else:
            return 'unknown', False, ['COUNT', 'FILTER']
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a string column contains dates"""
        if series.dtype != 'object':
            return False
        
        # Sample a few non-null values
        sample = series.dropna().head(10)
        if len(sample) == 0:
            return False
        
        # Common date patterns
        date_patterns = [
            r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YY or MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YY or MM-DD-YYYY
            r'\d{4}\d{2}\d{2}',  # YYYYMMDD
            r'\w{3}\s+\d{1,2},?\s+\d{4}',  # Jan 1, 2024
        ]
        
        matches = 0
        for value in sample:
            if pd.isna(value):
                continue
            value_str = str(value).strip()
            for pattern in date_patterns:
                if re.match(pattern, value_str):
                    matches += 1
                    break
        
        # If more than 70% match date patterns, likely a date column
        return matches / len(sample) > 0.7
    
    def _add_column_details(self, col_info: Dict[str, Any], series: pd.Series, semantic_type: str):
        """Add detailed analysis based on semantic type"""
        
        # Add sample values for categorical/text columns
        if semantic_type in ['categorical', 'text']:
            unique_values = series.dropna().unique()
            if len(unique_values) <= 20:
                col_info["unique_values"] = [str(v) for v in unique_values.tolist()]
                col_info["cardinality"] = "low"  # Categorical
            else:
                col_info["sample_values"] = [str(v) for v in unique_values[:5].tolist()]
                col_info["unique_count"] = len(unique_values)
                col_info["cardinality"] = "high" if len(unique_values) > len(series) * 0.8 else "medium"
        
        # Add statistics for numeric columns
        elif semantic_type == 'numeric':
            try:
                col_info["min"] = float(series.min())
                col_info["max"] = float(series.max())
                col_info["mean"] = float(series.mean())
                col_info["std"] = float(series.std())
                
                # Detect if it's likely an ID or code (high cardinality integers)
                if pd.api.types.is_integer_dtype(series):
                    unique_ratio = series.nunique() / len(series)
                    if unique_ratio > 0.9:  # Likely an ID
                        col_info["likely_identifier"] = True
                        col_info["suggested_operations"] = ["COUNT", "DISTINCT", "FILTER"]
            except (ValueError, TypeError):
                # Handle edge cases with non-numeric values in numeric columns
                pass
        
        # Add date range for date columns
        elif semantic_type == 'date':
            try:
                col_info["date_range"] = {
                    "min": str(series.min()),
                    "max": str(series.max())
                }
            except (ValueError, TypeError):
                # Handle edge cases with date parsing
                pass
        
        # Add basic stats for boolean columns
        elif semantic_type == 'boolean':
            try:
                value_counts = series.value_counts()
                col_info["value_distribution"] = {
                    str(k): int(v) for k, v in value_counts.items()
                }
            except (ValueError, TypeError):
                pass
