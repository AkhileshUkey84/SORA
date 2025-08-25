#!/usr/bin/env python3
"""
Debug script to check dataset availability and database status
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

async def debug_datasets():
    """Check what datasets are available and their status"""
    
    print("🔍 Debugging Dataset Availability")
    print("=" * 50)
    
    # Initialize database service
    print("📊 Initializing database service...")
    db_service = DatabaseService(settings.duckdb_path)
    await db_service.initialize()
    
    print(f"📦 Database path: {settings.duckdb_path}")
    print()
    
    # Check uploaded files
    print("📁 Checking uploads directory...")
    uploads_dir = Path("./uploads")
    
    if not uploads_dir.exists():
        print("❌ Uploads directory doesn't exist")
        return
    
    parquet_files = list(uploads_dir.glob("*.parquet"))
    print(f"📄 Found {len(parquet_files)} parquet files:")
    
    for file in parquet_files:
        dataset_id = file.stem  # Get filename without extension
        print(f"  - {dataset_id} ({file.stat().st_size} bytes)")
        
        # Try to load and check the file
        try:
            df = pd.read_parquet(file)
            print(f"    ✅ File readable: {len(df)} rows, {len(df.columns)} columns")
            
            # Try to get schema from database
            schema = await db_service.get_table_schema(dataset_id)
            if schema:
                print(f"    ✅ Schema available in database")
            else:
                print(f"    ❌ No schema in database - need to load dataset")
                
                # Try to create the dataset view
                try:
                    print(f"    🔄 Attempting to load dataset into database...")
                    result = await db_service.create_dataset_view(dataset_id, df)
                    print(f"    ✅ Dataset loaded successfully")
                    print(f"       Table name: {result.get('table_name')}")
                    print(f"       Columns: {len(result.get('columns', []))}")
                except Exception as e:
                    print(f"    ❌ Failed to load dataset: {e}")
            
        except Exception as e:
            print(f"    ❌ File error: {e}")
        
        print()
    
    # Check what tables are in the database
    print("🗄️  Checking database tables...")
    try:
        # Get all tables
        result = await db_service.execute_query(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        )
        
        if result:
            print(f"📊 Found {len(result)} tables in database:")
            for row in result:
                table_name = row['table_name']
                print(f"  - {table_name}")
                
                # Get row count
                try:
                    count_result = await db_service.execute_query(f"SELECT COUNT(*) as count FROM {table_name}")
                    row_count = count_result[0]['count'] if count_result else 0
                    print(f"    📈 Rows: {row_count:,}")
                except Exception as e:
                    print(f"    ❌ Error getting count: {e}")
        else:
            print("❌ No tables found in database")
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")
    
    print()
    print("💡 Usage Tips:")
    print("  • Use one of the dataset IDs listed above in your query request")
    print("  • Make sure the dataset_id in your request matches exactly")
    print("  • If a dataset shows 'No schema in database', it was loaded above")
    
    await db_service.close()

if __name__ == "__main__":
    asyncio.run(debug_datasets())
