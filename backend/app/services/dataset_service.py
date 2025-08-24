import logging
import pandas as pd
import duckdb
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
from app.config import settings
from app.models.dataset import DatasetStatus, ColumnInfo

logger = logging.getLogger(__name__)

class DatasetService:
    """Service for managing datasets"""
    
    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        
    async def process_upload(
        self,
        file_path: Path,
        dataset_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Process uploaded dataset file"""
        try:
            # Read file based on extension
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Analyze schema
            schema_info = await self._analyze_schema(df)
            
            # Create DuckDB view
            conn = duckdb.connect(":memory:")
            conn.register('dataset', df)
            
            # Get basic statistics
            row_count = len(df)
            column_count = len(df.columns)
            
            return {
                "status": DatasetStatus.READY,
                "row_count": row_count,
                "column_count": column_count,
                "schema_info": schema_info,
                "dataframe": df  # Store for later use
            }
            
        except Exception as e:
            logger.error(f"Dataset processing error: {e}")
            return {
                "status": DatasetStatus.ERROR,
                "error_message": str(e)
            }
    
    async def _analyze_schema(self, df: pd.DataFrame) -> Dict[str, ColumnInfo]:
        """Analyze dataset schema and detect PII"""
        schema_info = {}
        
        for column in df.columns:
            col_data = df[column]
            
            # Basic column info
            col_info = ColumnInfo(
                name=column,
                type=str(col_data.dtype),
                nullable=col_data.isnull().any(),
                unique_values=col_data.nunique(),
                sample_values=col_data.dropna().head(5).tolist() if not col_data.empty else []
            )
            
            # Check for PII
            col_info.has_pii, col_info.pii_type = self._check_pii(column, col_data)
            
            schema_info[column] = col_info.dict()
        
        return schema_info
    
    def _check_pii(self, column_name: str, data: pd.Series) -> tuple[bool, Optional[str]]:
        """Check if column contains PII"""
        column_lower = column_name.lower()
        
        # Check column name for PII indicators
        pii_indicators = {
            "email": ["email", "mail", "e-mail"],
            "phone": ["phone", "mobile", "cell", "telephone"],
            "ssn": ["ssn", "social_security", "social security"],
            "name": ["first_name", "last_name", "full_name", "name"],
            "address": ["address", "street", "city", "zip", "postal"],
            "credit_card": ["card", "credit", "payment"]
        }
        
        for pii_type, indicators in pii_indicators.items():
            if any(indicator in column_lower for indicator in indicators):
                return True, pii_type
        
        # Check data patterns (sample first 100 non-null values)
        sample_data = data.dropna().head(100).astype(str)
        
        for pii_type, pattern in settings.PII_PATTERNS.items():
            if any(re.match(pattern, str(value)) for value in sample_data):
                return True, pii_type
        
        return False, None
    
    async def get_preview(
        self,
        dataset_id: str,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get preview of dataset"""
        # In production, load from persistent storage
        # For now, return mock data
        return {
            "columns": ["column1", "column2", "column3"],
            "rows": [
                {"column1": "value1", "column2": 123, "column3": "2024-01-01"},
                {"column1": "value2", "column2": 456, "column3": "2024-01-02"}
            ],
            "total_rows": 1000,
            "preview_rows": 2
        }