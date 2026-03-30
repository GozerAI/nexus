"""
Embedding vector cache for repeated queries.

Caches computed embedding vectors keyed by their input text hash,
avoiding redundant calls to embedding models.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _EmbeddingEntry:
    """Cached embedding vector."""
    vector: List[float]
    model_name: str
    dimension: int
    created_at: float
    expires_at: float
    hit_count: int = 0


class EmbeddingVectorCache:
    """
    Cache for embedding vectors to avoid redundant model calls.

    Features:
    - Per-model caching (different models produce different embeddings)
    - TTL-based expiration
    - Memory-aware eviction
    - Batch lookup and store

    Usage::

        cache = EmbeddingVectorCache(max_entries=50000, default_ttl=86400)

        # Check cache first
        vec = cache.get("hello world", model_name="all-MiniLM-L6-v2")
        if vec is None:
            vec = embedding_model.embed("hello world")
            cache.put("hello world", vec, model_name="all-MiniLM-L6-v2", dimension=384)
    """

    def __init__(
        self,
        max_entries: int = 50000,
        default_ttl: int = 86400,
        max_memory_mb: int = 500,
    ):
        self._store: Dict[str, _EmbeddingEntry] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._max_memory_bytes = max_memory_mb * 1024 * 1024

        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "stores": 0}

        logger.info(
            "EmbeddingVectorCache initialized (max_entries=%d, ttl=%ds, max_mb=%d)",
            max_entries,
            default_ttl,
            max_memory_mb,
        )

    def _make_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _estimate_entry_bytes(self, dimension: int) -> int:
        """Estimate memory for one entry: dimension * 8 bytes (float64)."""
        return dimension * 8 + 128  # vector + overhead

    def _total_memory(self) -> int:
        return sum(
            self._estimate_entry_bytes(e.dimension)
            for e in self._store.values()
        )

    def _evict_if_needed(self) -> None:
        """Evict entries if over limits."""
        while len(self._store) > self._max_entries or self._total_memory() > self._max_memory_bytes:
            if not self._store:
                break
            # Evict entry with lowest hit_count
            victim = min(self._store, key=lambda k: self._store[k].hit_count)
            del self._store[victim]
            self._stats["evictions"] += 1

    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get cached embedding vector.

        Args:
            text: Input text
            model_name: Embedding model name

        Returns:
            Cached vector or None
        """
        key = self._make_key(text, model_name)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if time.time() > entry.expires_at:
                del self._store[key]
                self._stats["misses"] += 1
                return None
            entry.hit_count += 1
            self._stats["hits"] += 1
            return entry.vector

    def put(
        self,
        text: str,
        vector: List[float],
        model_name: str,
        dimension: Optional[int] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache an embedding vector.

        Args:
            text: Input text
            vector: Embedding vector
            model_name: Embedding model name
            dimension: Vector dimension (auto-detected if None)
            ttl: TTL in seconds
        """
        key = self._make_key(text, model_name)
        dim = dimension or len(vector)
        effective_ttl = ttl or self._default_ttl
        now = time.time()

        with self._lock:
            self._store[key] = _EmbeddingEntry(
                vector=vector,
                model_name=model_name,
                dimension=dim,
                created_at=now,
                expires_at=now + effective_ttl,
            )
            self._stats["stores"] += 1
            self._evict_if_needed()

    def get_batch(
        self, texts: List[str], model_name: str
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """
        Batch lookup of embeddings.

        Args:
            texts: List of input texts
            model_name: Embedding model name

        Returns:
            Tuple of (found: {index: vector}, missing_indices: [index])
        """
        found = {}
        missing = []
        for i, text in enumerate(texts):
            vec = self.get(text, model_name)
            if vec is not None:
                found[i] = vec
            else:
                missing.append(i)
        return found, missing

    def put_batch(
        self,
        texts: List[str],
        vectors: List[List[float]],
        model_name: str,
        dimension: Optional[int] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """Store a batch of embeddings."""
        for text, vec in zip(texts, vectors):
            self.put(text, vec, model_name, dimension, ttl)

    def clear(self) -> int:
        """Clear all cached embeddings."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def get_stats(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        with self._lock:
            return {
                **self._stats,
                "entries": len(self._store),
                "memory_bytes": self._total_memory(),
                "hit_rate": self._stats["hits"] / total if total > 0 else 0,
            }
