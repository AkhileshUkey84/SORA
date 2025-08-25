# Dataset upload handling
# services/storage_service.py
"""
Dataset storage service for handling file uploads.
Manages CSV/Excel processing and validation.
"""

import os
import uuid
import re
from typing import Dict, Any, Optional, BinaryIO, List
import pandas as pd
import aiofiles
from fastapi import UploadFile, HTTPException
import structlog
from datetime import datetime
import hashlib
from pathlib import Path


logger = structlog.get_logger()


class StorageService:
    """
    Handles dataset uploads and storage.
    Validates and processes CSV/Excel files.
    """
    
    def __init__(self, storage_path: str = "./uploads"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.logger = logger.bind(service="StorageService")
        
        # File constraints
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {'.csv', '.xlsx', '.xls'}
        self.max_rows = 1_000_000
        self.max_columns = 100
    
    async def upload_dataset(
        self,
        file: UploadFile,
        user_id: str,
        dataset_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes and stores uploaded dataset.
        Returns dataset metadata and ID.
        """
        
        # Validate file
        validation = await self._validate_file(file)
        if not validation["valid"]:
            raise HTTPException(400, validation["error"])
        
        # Generate dataset ID
        dataset_id = self._generate_dataset_id()
        
        # Save file temporarily
        temp_path = self.storage_path / f"temp_{dataset_id}"
        
        try:
            # Save uploaded file
            async with aiofiles.open(temp_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Process file into DataFrame
            df = await self._process_file(temp_path, file.filename)
            
            # Validate data
            data_validation = self._validate_dataframe(df)
            if not data_validation["valid"]:
                raise HTTPException(400, data_validation["error"])
            
            # Clean and prepare data
            df = self._clean_dataframe(df)
            
            # Generate metadata
            metadata = {
                "dataset_id": dataset_id,
                "name": dataset_name or file.filename,
                "original_filename": file.filename,
                "user_id": user_id,
                "uploaded_at": datetime.utcnow().isoformat(),
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "columns": list(df.columns),
                "size_bytes": os.path.getsize(temp_path),
                "file_hash": self._calculate_file_hash(content),
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                "preview": self._serialize_preview(df.head(5))
            }
            
            # Save processed data
            processed_path = self.storage_path / f"{dataset_id}.parquet"
            df.to_parquet(processed_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            self.logger.info("Dataset uploaded successfully",
                           dataset_id=dataset_id,
                           rows=len(df),
                           columns=len(df.columns))
            
            return {
                "dataset_id": dataset_id,
                "metadata": metadata,
                "dataframe": df
            }
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            self.logger.error("Dataset upload failed",
                            error=str(e),
                            user_id=user_id)
            raise
    
    async def get_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Retrieves dataset by ID.
        Returns DataFrame or None if not found.
        """
        
        file_path = self.storage_path / f"{dataset_id}.parquet"
        
        if not file_path.exists():
            return None
        
        try:
            df = pd.read_parquet(file_path)
            self.logger.info("Dataset retrieved", dataset_id=dataset_id)
            return df
        except Exception as e:
            self.logger.error("Failed to read dataset",
                            dataset_id=dataset_id,
                            error=str(e))
            return None
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """Deletes dataset file"""
        
        file_path = self.storage_path / f"{dataset_id}.parquet"
        
        if file_path.exists():
            try:
                os.remove(file_path)
                self.logger.info("Dataset deleted", dataset_id=dataset_id)
                return True
            except Exception as e:
                self.logger.error("Failed to delete dataset",
                                dataset_id=dataset_id,
                                error=str(e))
                return False
        
        return False
    
    async def _validate_file(self, file: UploadFile) -> Dict[str, Any]:
        """Validates uploaded file"""
        
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            return {
                "valid": False,
                "error": f"File type {file_ext} not allowed. Use: {', '.join(self.allowed_extensions)}"
            }
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset
        
        if size > self.max_file_size:
            return {
                "valid": False,
                "error": f"File too large. Maximum size: {self.max_file_size / 1024 / 1024:.0f}MB"
            }
        
        if size == 0:
            return {
                "valid": False,
                "error": "File is empty"
            }
        
        return {"valid": True}
    
    async def _process_file(self, file_path: Path, filename: str) -> pd.DataFrame:
        """Processes file into DataFrame"""
        
        file_ext = Path(filename).suffix.lower()
        
        try:
            if file_ext == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Unable to decode CSV file")
            
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to read file: {str(e)}")
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates DataFrame constraints"""
        
        # Check size limits
        if len(df) > self.max_rows:
            return {
                "valid": False,
                "error": f"Dataset too large: {len(df):,} rows (max: {self.max_rows:,})"
            }
        
        if len(df.columns) > self.max_columns:
            return {
                "valid": False,
                "error": f"Too many columns: {len(df.columns)} (max: {self.max_columns})"
            }
        
        if len(df) == 0:
            return {
                "valid": False,
                "error": "Dataset is empty"
            }
        
        # Check for completely empty columns
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            return {
                "valid": False,
                "error": f"Empty columns found: {', '.join(empty_cols[:5])}"
            }
        
        return {"valid": True}
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares DataFrame"""
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean column names
        df.columns = [self._clean_column_name(col) for col in df.columns]
        
        # Infer better data types
        df = df.infer_objects()
        
        # Convert date columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to convert to datetime with format inference
                try:
                    # First try common date formats
                    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                    if sample and isinstance(sample, str):
                        # Try to infer if it looks like a date
                        if any(char.isdigit() for char in sample) and any(sep in sample for sep in ['-', '/', ' ', ':']):
                            # Use format='mixed' for flexible parsing in newer pandas versions
                            df[col] = pd.to_datetime(df[col], errors='ignore', format='mixed')
                except:
                    pass
        
        return df
    
    def _clean_column_name(self, name: str) -> str:
        """Cleans column name for SQL compatibility"""
        
        # Convert to string
        name = str(name)
        
        # Replace special characters
        name = re.sub(r'[^\w\s]', '_', name)
        
        # Replace spaces
        name = name.replace(' ', '_')
        
        # Remove consecutive underscores
        name = re.sub(r'_+', '_', name)
        
        # Ensure doesn't start with number
        if name and name[0].isdigit():
            name = f"col_{name}"
        
        # Truncate if too long
        if len(name) > 64:
            name = name[:64]
        
        return name.lower()
    
    def _generate_dataset_id(self) -> str:
        """Generates unique dataset ID"""
        return f"ds_{uuid.uuid4().hex[:12]}"
    
    def _serialize_preview(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Serializes DataFrame preview to JSON-compatible format"""
        import numpy as np
        
        records = df.to_dict(orient='records')
        
        # Convert numpy types to native Python types
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif isinstance(value, (np.integer, np.int64, np.int32)):
                    record[key] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    record[key] = float(value)
                elif isinstance(value, np.bool_):
                    record[key] = bool(value)
                elif isinstance(value, (np.datetime64, pd.Timestamp)):
                    record[key] = value.isoformat() if hasattr(value, 'isoformat') else str(value)
                elif isinstance(value, np.ndarray):
                    record[key] = value.tolist()
                else:
                    # Convert other types to string as fallback
                    record[key] = str(value)
        
        return records
    
    def _calculate_file_hash(self, content: bytes) -> str:
        """Calculates SHA256 hash of file content"""
        return hashlib.sha256(content).hexdigest()
