"""
KV-cache reuse for transformer inference.

Manages key-value cache state across inference requests with shared
prefixes (system prompts, conversation history). When multiple requests
share the same prefix tokens, the KV-cache for those tokens is computed
once and reused, dramatically reducing latency for subsequent requests.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class KVCacheEntry:
    """A cached KV state for a token prefix."""
    prefix_hash: str
    prefix_text: str
    token_count: int
    cache_state: Any  # Opaque KV tensor data (model-specific)
    model_name: str
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    hit_count: int = 0
    size_bytes: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class KVCacheStats:
    """Statistics for the KV-cache manager."""
    total_entries: int = 0
    total_size_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    prefix_reuse_count: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class KVCacheManager:
    """
    Manages KV-cache state reuse across inference requests.

    Features:
    - Prefix-based cache keying (system prompt + conversation history)
    - Longest prefix match for partial reuse
    - LRU eviction with configurable memory limit
    - Per-model cache isolation
    - Thread-safe operations
    """

    DEFAULT_MAX_SIZE_MB = 512
    DEFAULT_MAX_ENTRIES = 256

    def __init__(
        self,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = 3600.0,
    ):
        """
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cached prefixes
            ttl_seconds: Time-to-live for cache entries
        """
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._cache: Dict[str, KVCacheEntry] = {}
        self._lock = threading.Lock()
        self._stats = KVCacheStats()

    @staticmethod
    def _compute_prefix_hash(prefix: str, model_name: str) -> str:
        """Compute a hash key for a prefix-model pair."""
        content = f"{model_name}:{prefix}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def get(self, prefix: str, model_name: str) -> Optional[KVCacheEntry]:
        """
        Look up cached KV state for a prefix.

        Args:
            prefix: The token prefix (system prompt + history)
            model_name: Model identifier

        Returns:
            KVCacheEntry if found and valid, None otherwise
        """
        key = self._compute_prefix_hash(prefix, model_name)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats.misses += 1
                return None

            # Check TTL
            if entry.age_seconds > self._ttl_seconds:
                del self._cache[key]
                self._stats.total_entries -= 1
                self._stats.total_size_bytes -= entry.size_bytes
                self._stats.misses += 1
                return None

            entry.hit_count += 1
            entry.last_used_at = time.time()
            self._stats.hits += 1
            self._stats.prefix_reuse_count += 1
            return entry

    def get_longest_prefix(
        self, text: str, model_name: str, min_prefix_ratio: float = 0.3
    ) -> Optional[KVCacheEntry]:
        """
        Find the longest cached prefix that matches the beginning of ``text``.

        Uses incremental hashing to find the longest prefix substring
        that has a cached KV state.

        Args:
            text: Full input text
            model_name: Model identifier
            min_prefix_ratio: Minimum fraction of text that must be a prefix match

        Returns:
            Best matching KVCacheEntry, or None
        """
        min_length = int(len(text) * min_prefix_ratio)
        best: Optional[KVCacheEntry] = None

        with self._lock:
            for entry in self._cache.values():
                if entry.model_name != model_name:
                    continue
                if entry.age_seconds > self._ttl_seconds:
                    continue
                ptext = entry.prefix_text
                if len(ptext) < min_length:
                    continue
                if text.startswith(ptext):
                    if best is None or len(ptext) > len(best.prefix_text):
                        best = entry

            if best:
                best.hit_count += 1
                best.last_used_at = time.time()
                self._stats.hits += 1
                self._stats.prefix_reuse_count += 1
            else:
                self._stats.misses += 1

        return best

    def put(
        self,
        prefix: str,
        model_name: str,
        cache_state: Any,
        token_count: int,
        size_bytes: int = 0,
    ) -> KVCacheEntry:
        """
        Store a KV-cache entry.

        Args:
            prefix: The token prefix
            model_name: Model identifier
            cache_state: Opaque KV tensor data
            token_count: Number of tokens in the prefix
            size_bytes: Approximate size of the cache state

        Returns:
            The stored KVCacheEntry
        """
        key = self._compute_prefix_hash(prefix, model_name)
        entry = KVCacheEntry(
            prefix_hash=key,
            prefix_text=prefix,
            token_count=token_count,
            cache_state=cache_state,
            model_name=model_name,
            size_bytes=size_bytes,
        )

        with self._lock:
            # Evict if over limits
            self._maybe_evict(size_bytes)

            if key in self._cache:
                old = self._cache[key]
                self._stats.total_size_bytes -= old.size_bytes
            else:
                self._stats.total_entries += 1

            self._cache[key] = entry
            self._stats.total_size_bytes += size_bytes

        logger.debug(
            "Cached KV state for %s (%d tokens, %d bytes)",
            model_name, token_count, size_bytes,
        )
        return entry

    def _maybe_evict(self, incoming_bytes: int) -> None:
        """Evict entries if adding incoming_bytes would exceed limits."""
        # Evict by entry count
        while len(self._cache) >= self._max_entries:
            self._evict_lru()

        # Evict by size
        while (
            self._stats.total_size_bytes + incoming_bytes > self._max_size_bytes
            and self._cache
        ):
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return
        lru_key = min(self._cache, key=lambda k: self._cache[k].last_used_at)
        evicted = self._cache.pop(lru_key)
        self._stats.total_entries -= 1
        self._stats.total_size_bytes -= evicted.size_bytes
        self._stats.evictions += 1
        logger.debug("Evicted KV cache entry for %s", evicted.model_name)

    def invalidate(self, model_name: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            model_name: If provided, only invalidate entries for this model.
                       If None, invalidate all entries.

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if model_name is None:
                count = len(self._cache)
                self._cache.clear()
                self._stats.total_entries = 0
                self._stats.total_size_bytes = 0
                return count

            to_remove = [
                k for k, v in self._cache.items() if v.model_name == model_name
            ]
            for k in to_remove:
                entry = self._cache.pop(k)
                self._stats.total_entries -= 1
                self._stats.total_size_bytes -= entry.size_bytes
            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_entries": self._stats.total_entries,
                "total_size_bytes": self._stats.total_size_bytes,
                "total_size_mb": self._stats.total_size_bytes / (1024 * 1024),
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "hit_rate": self._stats.hit_rate,
                "evictions": self._stats.evictions,
                "prefix_reuse_count": self._stats.prefix_reuse_count,
            }
