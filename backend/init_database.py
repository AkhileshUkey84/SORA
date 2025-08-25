#!/usr/bin/env python3
"""
Initialize persistent database with all available datasets
"""

import sys
import os
import asyncio
import pandas as pd
from pathlib import Path

# Add current directory to path
sys.path.append('.')

from services.database_service import DatabaseService
from utils.config import settings

async def init_database():
    """Initialize persistent database with all datasets"""
    
    print("🚀 Initializing Persistent Database")
    print("=" * 50)
    
    # Initialize database service with persistent path
    print(f"📦 Database path: {settings.duckdb_path}")
    db_service = DatabaseService(settings.duckdb_path)
    await db_service.initialize()
    
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Check uploaded files
    uploads_dir = Path("./uploads")
    if not uploads_dir.exists():
        print("❌ Uploads directory doesn't exist")
        return
    
    parquet_files = list(uploads_dir.glob("*.parquet"))
    print(f"📄 Loading {len(parquet_files)} datasets into persistent database...")
    print()
    
    loaded_count = 0
    
    for file in parquet_files:
        dataset_id = file.stem
        print(f"🔄 Loading dataset: {dataset_id}")
        
        try:
            # Read the parquet file
            df = pd.read_parquet(file)
            print(f"  📊 Data: {len(df)} rows, {len(df.columns)} columns")
            
            # Create dataset view in database
            schema = await db_service.create_dataset_view(dataset_id, df)
            
            print(f"  ✅ Loaded successfully as table: {schema['table_name']}")
            print(f"     Columns: {[col['name'] for col in schema['columns'][:5]]}{'...' if len(schema['columns']) > 5 else ''}")
            
            loaded_count += 1
            
        except Exception as e:
            print(f"  ❌ Failed to load: {e}")
        
        print()
    
    print("=" * 50)
    print(f"✅ Database initialization complete!")
    print(f"📊 Loaded {loaded_count}/{len(parquet_files)} datasets")
    print(f"💾 Database saved to: {settings.duckdb_path}")
    print()
    
    # List available datasets
    print("📋 Available Dataset IDs:")
    for file in parquet_files:
        dataset_id = file.stem
        schema = await db_service.get_table_schema(dataset_id)
        if schema:
            print(f"  ✅ {dataset_id}")
        else:
            print(f"  ❌ {dataset_id} (failed to load)")
    
    print()
    print("💡 Usage:")
    print("  • Start your FastAPI server: python main.py")
    print("  • Use any of the dataset IDs above in your query requests")
    print("  • The database will persist between restarts")
    
    await db_service.close()

if __name__ == "__main__":
    asyncio.run(init_database())
