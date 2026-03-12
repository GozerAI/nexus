"""
In-memory cache backend.
"""

import time
import logging
from typing import Optional, Any, Dict
from dataclasses import dataclass

from nexus.core.cache.cache_manager import CacheBackend

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    value: Any
    expires_at: float


class MemoryBackend(CacheBackend):
    """
    In-memory cache backend using dictionary.
    
    Simple and fast but not persistent or distributed.
    Good for development and testing.
    """
    
    def __init__(self):
        """Initialize memory cache."""
        self.cache: Dict[str, CacheEntry] = {}
        logger.info("MemoryBackend initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if time.time() > entry.expires_at:
            del self.cache[key]
            logger.debug(f"Cache entry expired: {key}")
            return None
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in memory cache."""
        try:
            expires_at = time.time() + ttl
            self.cache[key] = CacheEntry(value=value, expires_at=expires_at)
            logger.debug(f"Cached key: {key} (TTL={ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Error caching key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Deleted cache key: {key}")
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return True
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry.expires_at
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
