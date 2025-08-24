# Cache utilities
import redis.asyncio as redis
import json
import logging
from typing import Optional, Any
from app.config import settings

logger = logging.getLogger(__name__)

class CacheManager:
    """Manages Redis caching"""
    
    def __init__(self):
        self.redis_client = None
    
    async def init_cache(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Cache initialized successfully")
        except Exception as e:
            logger.error(f"Cache initialization failed: {e}")
            # Continue without cache
            self.redis_client = None
    
    async def close_cache(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Cache connection closed")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = settings.CACHE_TTL):
        """Set value in cache"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.set(
                key,
                json.dumps(value),
                ex=ttl
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")

# Global instance
cache_manager = CacheManager()

async def init_cache():
    await cache_manager.init_cache()

async def close_cache():
    await cache_manager.close_cache()

async def cache_get(key: str) -> Optional[Any]:
    return await cache_manager.get(key)

async def cache_set(key: str, value: Any, ttl: int = settings.CACHE_TTL):
    await cache_manager.set(key, value, ttl)