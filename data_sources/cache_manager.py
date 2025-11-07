"""Cache Manager - Handles Redis caching operations."""

from typing import Dict, Any, Optional
import json
import asyncio

from core.database import db_manager
from agents.router import DataSource
import structlog

logger = structlog.get_logger()


class CacheManager:
    """Manages Redis caching operations."""
    
    def __init__(self):
        self.redis_client = None
        self.default_ttl = 3600  # 1 hour
    
    async def initialize(self):
        """Initialize Redis connection."""
        self.redis_client = db_manager.redis_client
        logger.info("Cache manager initialized")
    
    async def query(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """Check cache for existing results."""
        try:
            cache_key = query_params.get("cache_key", "")
            
            cached_result = await self.redis_client.get(cache_key)
            
            if cached_result:
                result = json.loads(cached_result)
                logger.info("Cache hit", cache_key=cache_key)
                return {
                    "source": DataSource.CACHE.value,
                    "results": [result],
                    "cache_hit": True
                }
            else:
                logger.info("Cache miss", cache_key=cache_key)
                return {
                    "source": DataSource.CACHE.value,
                    "results": [],
                    "cache_hit": False
                }
                
        except Exception as e:
            logger.error("Cache query failed", error=str(e))
            return {"source": DataSource.CACHE.value, "results": [], "error": str(e)}
    
    async def set_cache(self, cache_key: str, data: Any, ttl: Optional[int] = None):
        """Store data in cache."""
        try:
            ttl = ttl or self.default_ttl
            serialized_data = json.dumps(data, default=str)
            
            await self.redis_client.setex(cache_key, ttl, serialized_data)
            logger.info("Data cached", cache_key=cache_key, ttl=ttl)
        except Exception as e:
            logger.error("Failed to cache data", error=str(e))
    
    async def invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
                logger.info("Cache invalidated", pattern=pattern, keys_deleted=len(keys))
        except Exception as e:
            logger.error("Failed to invalidate cache", error=str(e))


cache_manager = CacheManager()
