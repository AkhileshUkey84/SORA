#!/usr/bin/env python3
"""
Quick debug script to check what tables exist in the database
and verify dataset loading is working correctly.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.database_service import DatabaseService
from utils.config import settings

async def main():
    print("=== Database Debug Info ===")
    
    # Initialize database service
    db_service = DatabaseService(settings.duckdb_path)
    await db_service.initialize()
    
    try:
        # List all tables
        tables = await db_service.list_all_tables()
        print(f"\nFound {len(tables)} tables:")
        for i, table in enumerate(tables, 1):
            print(f"  {i}. {table}")
        
        # Check schemas for each table
        if tables:
            print(f"\nTable schemas:")
            for table in tables:
                # Extract dataset_id from table name
                if table.startswith('dataset_'):
                    dataset_id = table[8:]  # Remove 'dataset_' prefix
                    schema = await db_service.get_table_schema(dataset_id)
                    if schema:
                        print(f"\n{table}:")
                        print(f"  Dataset ID: {dataset_id}")
                        print(f"  Columns: {len(schema['columns'])}")
                        for col in schema['columns'][:3]:  # Show first 3 columns
                            print(f"    - {col['name']} ({col['type']})")
                        if len(schema['columns']) > 3:
                            print(f"    ... and {len(schema['columns']) - 3} more columns")
        else:
            print("\nNo tables found!")
        
    finally:
        await db_service.close()

if __name__ == "__main__":
    asyncio.run(main())
