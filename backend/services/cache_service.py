# Redis session store
# services/cache_service.py
"""
Redis cache service with fallback to in-memory caching.
Ensures demo resilience when Redis is unavailable.
"""

import json
import asyncio
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
import redis.asyncio as redis
from collections import OrderedDict
import structlog
from utils.config import settings


logger = structlog.get_logger()


class InMemoryCache:
    """Fallback in-memory cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if key in self.cache:
            # Move to end (LRU)
            self.cache.move_to_end(key)
            value, expiry = self.cache[key]
            
            # Check expiry
            if expiry and datetime.utcnow() > expiry:
                del self.cache[key]
                return None
            
            return value
        return None
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        
        # Enforce size limit
        if len(self.cache) >= self.max_size:
            # Remove oldest
            self.cache.popitem(last=False)
        
        expiry = None
        if ttl:
            expiry = datetime.utcnow() + timedelta(seconds=ttl)
        
        self.cache[key] = (value, expiry)
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        return key in self.cache


class CacheService:
    """
    Cache service with Redis primary and in-memory fallback.
    Ensures system functions even without Redis.
    """
    
    def __init__(self):
        self.redis_url = settings.redis_url
        self.ttl = settings.cache_ttl_seconds
        self.enable_fallback = settings.enable_cache_fallback
        self.redis_client = None
        self.fallback_cache = InMemoryCache()
        self.is_redis_available = False
        self.logger = logger.bind(service="CacheService")
    
    async def initialize(self):
        """Initialize cache connections"""
        if self.redis_url:
            try:
                self.redis_client = await redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
                
                # Test connection
                await self.redis_client.ping()
                self.is_redis_available = True
                self.logger.info("Redis cache initialized successfully")
                
            except Exception as e:
                self.logger.warning("Redis connection failed, using in-memory cache",
                                  error=str(e))
                self.is_redis_available = False
        else:
            self.logger.info("No Redis URL configured, using in-memory cache")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        
        try:
            if self.is_redis_available and self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    self.logger.debug("Cache hit", key=key)
                return value
        except Exception as e:
            self.logger.warning("Redis get failed", key=key, error=str(e))
            
            if not self.enable_fallback:
                return None
        
        # Fallback to in-memory
        return await self.fallback_cache.get(key)
    
    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL"""
        
        if ttl is None:
            ttl = self.ttl
        
        try:
            if self.is_redis_available and self.redis_client:
                await self.redis_client.set(key, value, ex=ttl)
                self.logger.debug("Cache set", key=key, ttl=ttl)
                return True
        except Exception as e:
            self.logger.warning("Redis set failed", key=key, error=str(e))
            
            if not self.enable_fallback:
                return False
        
        # Fallback to in-memory
        return await self.fallback_cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        
        try:
            if self.is_redis_available and self.redis_client:
                result = await self.redis_client.delete(key)
                return result > 0
        except Exception as e:
            self.logger.warning("Redis delete failed", key=key, error=str(e))
            
            if not self.enable_fallback:
                return False
        
        # Fallback to in-memory
        return await self.fallback_cache.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        
        try:
            if self.is_redis_available and self.redis_client:
                return await self.redis_client.exists(key) > 0
        except Exception as e:
            self.logger.warning("Redis exists check failed", key=key, error=str(e))
            
            if not self.enable_fallback:
                return False
        
        # Fallback to in-memory
        return await self.fallback_cache.exists(key)
    
    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value from cache"""
        
        value = await self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                self.logger.warning("Failed to decode JSON from cache", key=key)
        return None
    
    async def set_json(
        self,
        key: str,
        value: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Set JSON value in cache"""
        
        try:
            json_str = json.dumps(value)
            return await self.set(key, json_str, ttl)
        except Exception as e:
            self.logger.error("Failed to encode JSON for cache", key=key, error=str(e))
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        
        try:
            if self.is_redis_available and self.redis_client:
                return await self.redis_client.incr(key, amount)
        except Exception as e:
            self.logger.warning("Redis increment failed", key=key, error=str(e))
        
        # Simple fallback - get, increment, set
        current = await self.get(key)
        new_value = int(current or 0) + amount
        await self.set(key, str(new_value))
        return new_value
    
    async def close(self):
        """Close cache connections"""
        if self.redis_client:
            await self.redis_client.close()