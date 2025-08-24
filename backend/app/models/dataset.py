from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID
from enum import Enum

class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool
    unique_values: Optional[int] = None
    sample_values: Optional[List[Any]] = None
    has_pii: bool = False
    pii_type: Optional[str] = None

class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None

class DatasetResponse(BaseModel):
    id: str  # Changed from UUID to str for flexibility
    name: str
    description: Optional[str]
    file_type: str
    file_size: int
    file_path: str
    status: DatasetStatus
    row_count: Optional[int]
    column_count: Optional[int]
    schema_info: Optional[Dict[str, ColumnInfo]]
    error_message: Optional[str]
    owner_id: str  # Changed from UUID to str to accept email addresses
    created_at: datetime
    updated_at: datetime

class DatasetPreview(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int
    preview_rows: int