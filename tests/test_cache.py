"""Tests for caching system."""

import os
import sys
import pytest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.cache.cache_manager import CacheManager
from nexus.core.cache.memory_backend import MemoryBackend


class TestMemoryBackend:
    """Tests for MemoryBackend."""
    
    def test_backend_initialization(self):
        """Test backend initialization."""
        backend = MemoryBackend()
        assert backend is not None
        assert len(backend.cache) == 0
    
    def test_set_and_get(self):
        """Test setting and getting values."""
        backend = MemoryBackend()
        
        backend.set("key1", "value1", ttl=10)
        assert backend.get("key1") == "value1"
    
    def test_get_nonexistent(self):
        """Test getting nonexistent key."""
        backend = MemoryBackend()
        assert backend.get("nonexistent") is None
    
    def test_expiration(self):
        """Test value expiration."""
        backend = MemoryBackend()
        
        backend.set("key1", "value1", ttl=1)
        assert backend.get("key1") == "value1"
        
        time.sleep(1.1)
        assert backend.get("key1") is None
    
    def test_delete(self):
        """Test deleting values."""
        backend = MemoryBackend()
        
        backend.set("key1", "value1")
        assert backend.exists("key1")
        
        backend.delete("key1")
        assert not backend.exists("key1")
    
    def test_clear(self):
        """Test clearing all values."""
        backend = MemoryBackend()
        
        backend.set("key1", "value1")
        backend.set("key2", "value2")
        
        backend.clear()
        assert len(backend.cache) == 0
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        backend = MemoryBackend()
        
        backend.set("key1", "value1", ttl=1)
        backend.set("key2", "value2", ttl=10)
        
        time.sleep(1.1)
        
        removed = backend.cleanup_expired()
        assert removed == 1
        assert backend.get("key2") == "value2"


class TestCacheManager:
    """Tests for CacheManager."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        backend = MemoryBackend()
        manager = CacheManager(backend, default_ttl=3600)
        
        assert manager.backend == backend
        assert manager.default_ttl == 3600
        assert manager.hits == 0
        assert manager.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss."""
        backend = MemoryBackend()
        manager = CacheManager(backend)
        
        result = manager.get_response("test prompt")
        assert result is None
        assert manager.misses == 1
        assert manager.hits == 0
    
    def test_cache_hit(self):
        """Test cache hit."""
        backend = MemoryBackend()
        manager = CacheManager(backend)
        
        # Set response
        response = {"content": "test response"}
        manager.set_response("test prompt", response)
        
        # Get response
        cached = manager.get_response("test prompt")
        assert cached == response
        assert manager.hits == 1
        assert manager.misses == 0
    
    def test_cache_with_model_config(self):
        """Test caching with model configuration."""
        backend = MemoryBackend()
        manager = CacheManager(backend)
        
        # Set responses with different configs
        manager.set_response("prompt", {"result": "A"}, model_config="config1")
        manager.set_response("prompt", {"result": "B"}, model_config="config2")
        
        # Get responses
        result_a = manager.get_response("prompt", model_config="config1")
        result_b = manager.get_response("prompt", model_config="config2")
        
        assert result_a == {"result": "A"}
        assert result_b == {"result": "B"}
    
    def test_invalidate(self):
        """Test cache invalidation."""
        backend = MemoryBackend()
        manager = CacheManager(backend)
        
        manager.set_response("prompt", {"result": "test"})
        assert manager.get_response("prompt") is not None
        
        manager.invalidate("prompt")
        assert manager.get_response("prompt") is None
    
    def test_cache_stats(self):
        """Test getting cache statistics."""
        backend = MemoryBackend()
        manager = CacheManager(backend)
        
        # Generate some hits and misses
        manager.get_response("prompt1")  # miss
        manager.set_response("prompt1", {"result": "1"})
        manager.get_response("prompt1")  # hit
        manager.get_response("prompt1")  # hit
        manager.get_response("prompt2")  # miss
        
        stats = manager.get_stats()
        
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["total_requests"] == 4
        assert stats["hit_rate_percent"] == 50.0
        assert stats["backend"] == "MemoryBackend"
    
    def test_clear_all(self):
        """Test clearing all cache."""
        backend = MemoryBackend()
        manager = CacheManager(backend)
        
        manager.set_response("prompt1", {"result": "1"})
        manager.set_response("prompt2", {"result": "2"})
        
        manager.clear_all()
        
        assert manager.get_response("prompt1") is None
        assert manager.get_response("prompt2") is None
