"""
Cache manager for response caching.
"""

import hashlib
import logging
from typing import Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass


class CacheManager:
    """
    Manages response caching for ensemble inference.
    
    Features:
    - Multiple backend support (Redis, Memory)
    - Configurable TTL
    - Cache key generation
    - Hit/miss tracking
    """
    
    def __init__(self, backend: CacheBackend, default_ttl: int = 3600):
        """
        Initialize cache manager.
        
        Args:
            backend: Cache backend implementation
            default_ttl: Default TTL in seconds
        """
        self.backend = backend
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        logger.info(f"CacheManager initialized with {backend.__class__.__name__}")
    
    def get_response(self, prompt: str, model_config: Optional[str] = None) -> Optional[dict]:
        """
        Get cached response for a prompt.
        
        Args:
            prompt: Input prompt
            model_config: Optional model configuration identifier
            
        Returns:
            Cached response dict or None
        """
        cache_key = self._generate_key(prompt, model_config)
        
        cached = self.backend.get(cache_key)
        
        if cached:
            self.hits += 1
            logger.debug(f"Cache HIT for prompt: {prompt[:50]}...")
            return cached
        
        self.misses += 1
        logger.debug(f"Cache MISS for prompt: {prompt[:50]}...")
        return None
    
    def set_response(
        self,
        prompt: str,
        response: dict,
        model_config: Optional[str] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache a response.
        
        Args:
            prompt: Input prompt
            response: Response to cache
            model_config: Optional model configuration identifier
            ttl: Optional TTL override
            
        Returns:
            True if cached successfully
        """
        cache_key = self._generate_key(prompt, model_config)
        ttl = ttl or self.default_ttl
        
        success = self.backend.set(cache_key, response, ttl)
        
        if success:
            logger.debug(f"Cached response for prompt: {prompt[:50]}... (TTL={ttl}s)")
        else:
            logger.warning(f"Failed to cache response for prompt: {prompt[:50]}...")
        
        return success
    
    def invalidate(self, prompt: str, model_config: Optional[str] = None) -> bool:
        """
        Invalidate cached response.
        
        Args:
            prompt: Input prompt
            model_config: Optional model configuration identifier
            
        Returns:
            True if deleted successfully
        """
        cache_key = self._generate_key(prompt, model_config)
        return self.backend.delete(cache_key)
    
    def clear_all(self) -> bool:
        """
        Clear all cached responses.
        
        Returns:
            True if cleared successfully
        """
        logger.info("Clearing all cached responses")
        return self.backend.clear()
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "backend": self.backend.__class__.__name__,
        }
    
    def _generate_key(self, prompt: str, model_config: Optional[str] = None) -> str:
        """
        Generate cache key from prompt and config.
        
        Args:
            prompt: Input prompt
            model_config: Optional model configuration
            
        Returns:
            Cache key string
        """
        # Normalize prompt
        normalized_prompt = prompt.strip().lower()
        
        # Include model config in key if provided
        if model_config:
            key_data = f"{normalized_prompt}:{model_config}"
        else:
            key_data = normalized_prompt
        
        # Hash for consistent key length
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()
        
        return f"thenexus:response:{key_hash}"
