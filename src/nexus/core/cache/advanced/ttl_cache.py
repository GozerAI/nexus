"""
Query result caching with TTL-based invalidation.

Provides a general-purpose TTL cache with namespace support,
tag-based invalidation, and automatic cleanup.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    """Internal cache entry with metadata."""
    value: Any
    expires_at: float
    created_at: float
    tags: Set[str] = field(default_factory=set)
    hit_count: int = 0
    size_bytes: int = 0


class TTLCache:
    """
    Thread-safe TTL cache with tag-based invalidation.

    Features:
    - Configurable TTL per entry or per namespace
    - Tag-based bulk invalidation
    - LRU eviction when max size is reached
    - Background cleanup of expired entries
    - Hit/miss statistics per namespace
    """

    def __init__(
        self,
        default_ttl: int = 300,
        max_entries: int = 10000,
        max_memory_bytes: int = 100 * 1024 * 1024,
        cleanup_interval: int = 60,
    ):
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self.max_memory_bytes = max_memory_bytes
        self.cleanup_interval = cleanup_interval

        self._store: Dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()
        self._stats: Dict[str, Dict[str, int]] = {}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of keys
        self._namespace_ttls: Dict[str, int] = {}
        self._total_bytes = 0

        self._last_cleanup = time.time()

        logger.info(
            "TTLCache initialized (max_entries=%d, default_ttl=%ds)",
            max_entries,
            default_ttl,
        )

    def set_namespace_ttl(self, namespace: str, ttl: int) -> None:
        """Set a custom TTL for a namespace."""
        self._namespace_ttls[namespace] = ttl

    def _make_key(self, namespace: str, key: str) -> str:
        return f"{namespace}:{key}"

    def _estimate_size(self, value: Any) -> int:
        """Rough estimate of object memory size."""
        try:
            import sys
            return sys.getsizeof(value)
        except Exception:
            return 256

    def _maybe_cleanup(self) -> None:
        """Run cleanup if interval has elapsed."""
        now = time.time()
        if now - self._last_cleanup > self.cleanup_interval:
            self._cleanup_expired()
            self._last_cleanup = now

    def _cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [k for k, e in self._store.items() if now > e.expires_at]
        for k in expired:
            self._remove_entry(k)
        if expired:
            logger.debug("TTLCache cleanup: removed %d expired entries", len(expired))
        return len(expired)

    def _remove_entry(self, full_key: str) -> None:
        """Remove a single entry and update indexes."""
        entry = self._store.pop(full_key, None)
        if entry:
            self._total_bytes -= entry.size_bytes
            for tag in entry.tags:
                if tag in self._tag_index:
                    self._tag_index[tag].discard(full_key)
                    if not self._tag_index[tag]:
                        del self._tag_index[tag]

    def _evict_lru(self) -> None:
        """Evict least-recently-used entries until under limits."""
        while (
            len(self._store) > self.max_entries
            or self._total_bytes > self.max_memory_bytes
        ) and self._store:
            # Evict entry with lowest hit_count (approximates LRU)
            victim_key = min(self._store, key=lambda k: self._store[k].hit_count)
            self._remove_entry(victim_key)

    def _update_stats(self, namespace: str, hit: bool) -> None:
        if namespace not in self._stats:
            self._stats[namespace] = {"hits": 0, "misses": 0}
        if hit:
            self._stats[namespace]["hits"] += 1
        else:
            self._stats[namespace]["misses"] += 1

    def get(self, namespace: str, key: str) -> Optional[Any]:
        """
        Get a cached value.

        Args:
            namespace: Cache namespace
            key: Cache key

        Returns:
            Cached value or None if not found / expired
        """
        full_key = self._make_key(namespace, key)
        with self._lock:
            self._maybe_cleanup()
            entry = self._store.get(full_key)
            if entry is None:
                self._update_stats(namespace, False)
                return None
            if time.time() > entry.expires_at:
                self._remove_entry(full_key)
                self._update_stats(namespace, False)
                return None
            entry.hit_count += 1
            self._update_stats(namespace, True)
            return entry.value

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Set a cached value.

        Args:
            namespace: Cache namespace
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses namespace or default TTL if None)
            tags: Optional tags for bulk invalidation

        Returns:
            True if cached successfully
        """
        effective_ttl = ttl or self._namespace_ttls.get(namespace, self.default_ttl)
        full_key = self._make_key(namespace, key)
        tag_set = set(tags) if tags else set()
        size = self._estimate_size(value)
        now = time.time()

        with self._lock:
            # Remove existing entry if present
            if full_key in self._store:
                self._remove_entry(full_key)

            self._store[full_key] = _CacheEntry(
                value=value,
                expires_at=now + effective_ttl,
                created_at=now,
                tags=tag_set,
                size_bytes=size,
            )
            self._total_bytes += size

            # Update tag index
            for tag in tag_set:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(full_key)

            self._evict_lru()

        return True

    def invalidate(self, namespace: str, key: str) -> bool:
        """Invalidate a specific cache entry."""
        full_key = self._make_key(namespace, key)
        with self._lock:
            if full_key in self._store:
                self._remove_entry(full_key)
                return True
        return False

    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a given tag. Returns count invalidated."""
        with self._lock:
            keys = list(self._tag_index.get(tag, []))
            for k in keys:
                self._remove_entry(k)
            return len(keys)

    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all entries in a namespace."""
        prefix = f"{namespace}:"
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                self._remove_entry(k)
            return len(keys)

    def clear(self) -> int:
        """Clear all entries."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._tag_index.clear()
            self._total_bytes = 0
            return count

    def get_stats(self, namespace: Optional[str] = None) -> dict:
        """Get cache statistics."""
        with self._lock:
            if namespace:
                ns_stats = self._stats.get(namespace, {"hits": 0, "misses": 0})
                total = ns_stats["hits"] + ns_stats["misses"]
                return {
                    "namespace": namespace,
                    **ns_stats,
                    "hit_rate": ns_stats["hits"] / total if total > 0 else 0,
                    "entries": sum(
                        1 for k in self._store if k.startswith(f"{namespace}:")
                    ),
                }

            total_hits = sum(s["hits"] for s in self._stats.values())
            total_misses = sum(s["misses"] for s in self._stats.values())
            total = total_hits + total_misses
            return {
                "total_entries": len(self._store),
                "total_bytes": self._total_bytes,
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": total_hits / total if total > 0 else 0,
                "namespaces": list(self._stats.keys()),
                "tag_count": len(self._tag_index),
            }

    def get_or_set(
        self,
        namespace: str,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> Any:
        """
        Get a cached value or compute and cache it.

        Args:
            namespace: Cache namespace
            key: Cache key
            factory: Callable to produce the value if not cached
            ttl: Optional TTL override
            tags: Optional tags

        Returns:
            Cached or freshly computed value
        """
        value = self.get(namespace, key)
        if value is not None:
            return value
        value = factory()
        self.set(namespace, key, value, ttl=ttl, tags=tags)
        return value
