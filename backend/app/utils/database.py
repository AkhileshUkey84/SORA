# Database utilities
import duckdb
import logging
from pathlib import Path
from app.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages DuckDB connections and operations"""
    
    def __init__(self):
        self.conn = None
        
    async def init_db(self):
        """Initialize database connection"""
        try:
            # Create database directory if needed
            db_path = Path(settings.DATABASE_PATH)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Connect to DuckDB
            self.conn = duckdb.connect(str(db_path))
            
            # Set memory limit
            self.conn.execute(f"SET memory_limit='{settings.DUCKDB_MEMORY_LIMIT}'")
            
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def create_dataset_view(self, dataset_id: str, df):
        """Create a DuckDB view for a dataset"""
        if self.conn:
            view_name = f"dataset_{dataset_id}"
            self.conn.register(view_name, df)
            return view_name
        raise RuntimeError("Database not initialized")
    
    def execute_query(self, sql: str):
        """Execute a SQL query"""
        if self.conn:
            return self.conn.execute(sql).fetchall()
        raise RuntimeError("Database not initialized")

# Global instance
db_manager = DatabaseManager()

async def init_db():
    await db_manager.init_db()

async def close_db():
    await db_manager.close_db()