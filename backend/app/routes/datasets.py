from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from typing import Optional
import uuid
from datetime import datetime
from pathlib import Path
from app.models.dataset import DatasetCreate, DatasetResponse, DatasetPreview
from app.services.dataset_service import DatasetService
from app.middleware.auth import get_current_user
from app.config import settings

router = APIRouter()
dataset_service = DatasetService()

@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: Optional[str] = None,
    description: Optional[str] = None,
    user_id: str = Depends(get_current_user)
):
    """Upload a dataset file for analysis"""
    
    # Validate file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_extension} not supported. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Validate file size
    if file.size > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum of {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    # Generate unique dataset ID
    dataset_id = str(uuid.uuid4())
    
    # Save file
    file_path = settings.UPLOAD_DIR / f"{dataset_id}{file_extension}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create initial response
    dataset = DatasetResponse(
        id=dataset_id,
        name=name or file.filename,
        description=description,
        file_type=file_extension,
        file_size=file.size,
        file_path=str(file_path),
        status="processing",
        row_count=None,
        column_count=None,
        schema_info=None,
        error_message=None,
        owner_id=user_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # Process in background
    background_tasks.add_task(
        dataset_service.process_upload,
        file_path,
        dataset_id,
        user_id
    )
    
    return dataset

@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get dataset details"""
    # In production, fetch from database
    # For now, return mock data
    return DatasetResponse(
        id=dataset_id,  # Now accepts string
        name="Sample Dataset",
        description="Mock dataset for testing",
        file_type=".csv",
        file_size=1024,
        file_path="/uploads/sample.csv",
        status="ready",
        row_count=1000,
        column_count=10,
        schema_info={},
        error_message=None,
        owner_id=user_id,  # Now accepts email string
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

@router.get("/{dataset_id}/schema")
async def get_dataset_schema(
    dataset_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get dataset schema information"""
    # In production, fetch from database
    schema = await dataset_service._get_dataset_schema(dataset_id)
    return {"dataset_id": dataset_id, "schema": schema}

@router.get("/{dataset_id}/preview", response_model=DatasetPreview)
async def preview_dataset(
    dataset_id: str,
    limit: int = 100,
    user_id: str = Depends(get_current_user)
):
    """Get a preview of dataset rows"""
    preview = await dataset_service.get_preview(dataset_id, limit)
    return preview

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    user_id: str = Depends(get_current_user)
):
    """Delete a dataset"""
    # In production, verify ownership and delete from storage
    return {"message": f"Dataset {dataset_id} deleted successfully"}