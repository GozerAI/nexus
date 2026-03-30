"""
Redis cache backend.
"""

import json
import logging
from typing import Optional, Any

from nexus.core.cache.cache_manager import CacheBackend

logger = logging.getLogger(__name__)


class RedisBackend(CacheBackend):
    """
    Redis cache backend.
    
    Provides persistent, distributed caching.
    Requires Redis server to be running.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize Redis backend.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        try:
            import redis
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Redis backend initialized ({host}:{port}/{db})")
        except ImportError:
            logger.error("Redis package not installed. Run: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting key {key} from Redis: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in Redis with TTL."""
        try:
            serialized = json.dumps(value)
            self.redis_client.setex(key, ttl, serialized)
            logger.debug(f"Cached key in Redis: {key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Error setting key {key} in Redis: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        try:
            result = self.redis_client.delete(key)
            logger.debug(f"Deleted Redis key: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Error deleting key {key} from Redis: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking key {key} in Redis: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all keys in current Redis database."""
        try:
            self.redis_client.flushdb()
            logger.info("Cleared all Redis cache entries")
            return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
