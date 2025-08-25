# Dataset upload endpoints
# routes/upload.py
"""Dataset upload endpoints"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Request
from typing import Optional
from models.requests import UploadRequest
from models.responses import UploadResult
from services.storage_service import StorageService
from services.database_service import DatabaseService
from services.auth_service import AuthService
from middleware.auth import get_current_user
import structlog


router = APIRouter(prefix="/api/v1", tags=["upload"])
logger = structlog.get_logger()


@router.post("/upload", response_model=UploadResult)
async def upload_dataset(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: Optional[str] = None,
    description: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
) -> UploadResult:
    """
    Upload a CSV or Excel dataset for analysis.
    Processes file and creates queryable database view.
    """
    
    try:
        # Get shared services from app state
        db_service = request.app.state.db_service
        orchestrator = request.app.state.orchestrator
        
        # Process file upload using storage service directly
        storage_service = StorageService()
        
        # Upload and process file
        upload_result = await storage_service.upload_dataset(
            file=file,
            user_id=current_user["id"],
            dataset_name=name
        )
        
        dataset_id = upload_result["dataset_id"]
        df = upload_result["dataframe"]
        
        # Create database view using the SHARED database service
        logger.info(f"Creating DuckDB view for dataset {dataset_id}")
        schema = await db_service.create_dataset_view(dataset_id, df)
        
        # Verify the view was created successfully
        verification_schema = await db_service.get_table_schema(dataset_id)
        if not verification_schema:
            logger.error(f"Failed to verify DuckDB view creation for dataset {dataset_id}")
            return UploadResult(
                success=False,
                error="Failed to register dataset in database"
            )
        
        logger.info(f"Successfully registered DuckDB view for dataset {dataset_id}")
        
        result = {
            "success": True,
            "dataset_id": dataset_id,
            "metadata": upload_result["metadata"],
            "preview": upload_result["metadata"]["preview"],
            "warnings": []
        }
        
        # Add to background processing if successful
        if result["success"]:
            # Initialize services for background task
            storage_service = StorageService()
            
            # Get the dataset
            dataset = await storage_service.get_dataset(result["dataset_id"])
            
            # Add background task
            if dataset is not None:
                background_tasks.add_task(
                    process_dataset_insights,
                    result["dataset_id"],
                    dataset
                )
        
        # Return result as UploadResult model
        return UploadResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Dataset upload failed",
                    user_id=current_user["id"],
                    error=str(e))
        
        return UploadResult(
            success=False,
            error=f"Upload failed: {str(e)}"
        )


@router.get("/datasets/{dataset_id}")
async def get_dataset_info(
    request: Request,
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get dataset metadata and schema"""
    
    # Check access
    auth_service = request.app.state.auth_service
    has_access = await auth_service.validate_dataset_access(
        current_user["id"],
        dataset_id
    )
    
    if not has_access:
        raise HTTPException(403, "Access denied to dataset")
    
    # Get dataset info using shared database service
    db_service = request.app.state.db_service
    schema = await db_service.get_table_schema(dataset_id)
    
    if not schema:
        raise HTTPException(404, "Dataset not found")
    
    stats = await db_service.get_dataset_stats(dataset_id)
    
    return {
        "dataset_id": dataset_id,
        "schema": schema,
        "statistics": stats
    }


@router.get("/datasets")
async def list_datasets(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """List all datasets for current user"""
    
    try:
        storage_service = StorageService()
        db_service = request.app.state.db_service
        
        # Get all datasets for user (simplified for demo)
        datasets = await storage_service.list_user_datasets(current_user["id"])
        
        # Enrich with schema info from database
        enriched_datasets = []
        for dataset in datasets:
            dataset_info = dataset.copy()
            
            # Try to get schema info
            schema = await db_service.get_table_schema(dataset["dataset_id"], suppress_warnings=True)
            if schema:
                dataset_info["schema_available"] = True
                dataset_info["column_count"] = len(schema["columns"])
                dataset_info["row_count"] = schema.get("row_count", 0)
            else:
                dataset_info["schema_available"] = False
                
            enriched_datasets.append(dataset_info)
        
        return {
            "success": True,
            "datasets": enriched_datasets,
            "total_count": len(enriched_datasets)
        }
        
    except Exception as e:
        logger.error("Failed to list datasets",
                    user_id=current_user["id"],
                    error=str(e))
        raise HTTPException(500, "Failed to retrieve datasets")


@router.get("/datasets/{dataset_id}/preview")
async def preview_dataset(
    request: Request,
    dataset_id: str,
    current_user: dict = Depends(get_current_user),
    limit: int = 10
):
    """Returns first N rows and column list for preview"""
    
    try:
        # Check access
        auth_service = request.app.state.auth_service
        has_access = await auth_service.validate_dataset_access(
            current_user["id"],
            dataset_id
        )
        
        if not has_access:
            raise HTTPException(403, "Access denied to dataset")
        
        # Get database service
        db_service = request.app.state.db_service
        
        # Get schema first
        schema = await db_service.get_table_schema(dataset_id)
        if not schema:
            raise HTTPException(404, "Dataset not found")
        
        # Get preview data
        table_name = schema["table_name"]
        preview_sql = f"SELECT * FROM {table_name} LIMIT ?"
        
        preview_data = await db_service.execute_query(
            preview_sql,
            params=[limit],
            dataset_id=dataset_id
        )
        
        return {
            "success": True,
            "dataset_id": dataset_id,
            "columns": [col["name"] for col in schema["columns"]],
            "column_info": schema["columns"],
            "preview_data": preview_data,
            "preview_row_count": len(preview_data),
            "total_rows": schema.get("row_count", 0),
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to preview dataset",
                    dataset_id=dataset_id,
                    error=str(e))
        raise HTTPException(500, "Failed to preview dataset")


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Delete a dataset"""
    
    # Only owner or admin can delete
    # Simplified for demo
    
    storage_service = StorageService()
    success = await storage_service.delete_dataset(dataset_id)
    
    if not success:
        raise HTTPException(404, "Dataset not found")
    
    return {"message": "Dataset deleted successfully"}


async def process_dataset_insights(dataset_id: str, df):
    """Background task to process dataset insights"""
    
    try:
        # Calculate basic statistics
        numeric_columns = df.select_dtypes(include=['number']).columns
        
        insights = {
            "numeric_summary": {},
            "categorical_summary": {},
            "quality_metrics": {}
        }
        
        # Numeric summaries
        for col in numeric_columns:
            insights["numeric_summary"][col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
        
        # Data quality
        insights["quality_metrics"] = {
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage": int(df.memory_usage(deep=True).sum())
        }
        
        # Store insights (would save to database in production)
        logger.info("Dataset insights processed",
                   dataset_id=dataset_id,
                   insights=insights)
        
    except Exception as e:
        logger.error("Failed to process dataset insights",
                    dataset_id=dataset_id,
                    error=str(e))