"""
Model response cache with semantic similarity matching.

Unlike exact-match caching, this cache can return a cached response
for a query that is semantically similar (but not identical) to a
previously cached query. This dramatically improves cache hit rates
for paraphrased or slightly modified queries.
"""

import hashlib
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class _SemanticEntry:
    """A semantic cache entry."""
    query: str
    query_embedding: List[float]
    response: Dict[str, Any]
    model_config: str
    created_at: float
    expires_at: float
    hit_count: int = 0
    exact_hash: str = ""


class SemanticResponseCache:
    """
    Response cache that matches on semantic similarity.

    When a new query arrives, we compute its embedding and compare
    it against cached query embeddings. If the cosine similarity
    exceeds a threshold, we return the cached response.

    For deterministic prompts (temperature=0), exact hash matching
    is also used as a fast path.

    Usage::

        cache = SemanticResponseCache(embed_fn=my_embed_fn, threshold=0.95)

        # Check cache
        hit = cache.get("What is the capital of France?", model_config="gpt-4")
        if hit:
            return hit["response"]

        # Compute and cache
        response = model.generate(query)
        cache.put(query, response, model_config="gpt-4")
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.95,
        max_entries: int = 5000,
        default_ttl: int = 3600,
    ):
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._default_ttl = default_ttl

        self._store: Dict[str, _SemanticEntry] = {}
        self._exact_index: Dict[str, str] = {}  # hash -> store key
        self._lock = threading.RLock()
        self._stats = {
            "exact_hits": 0,
            "semantic_hits": 0,
            "misses": 0,
            "stores": 0,
        }

        logger.info(
            "SemanticResponseCache initialized (threshold=%.2f, max=%d, ttl=%ds)",
            similarity_threshold,
            max_entries,
            default_ttl,
        )

    def _exact_hash(self, query: str, model_config: str) -> str:
        content = f"{model_config}:{query.strip().lower()}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _make_key(self) -> str:
        import uuid
        return uuid.uuid4().hex[:16]

    def _simple_embed(self, text: str) -> List[float]:
        """Very basic character-frequency embedding as fallback."""
        vec = [0.0] * 128
        for ch in text.lower():
            idx = ord(ch) % 128
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def _get_embedding(self, text: str) -> List[float]:
        if self._embed_fn:
            return self._embed_fn(text)
        return self._simple_embed(text)

    def _evict_if_needed(self) -> None:
        while len(self._store) > self._max_entries and self._store:
            victim = min(self._store, key=lambda k: self._store[k].hit_count)
            entry = self._store.pop(victim)
            self._exact_index.pop(entry.exact_hash, None)

    def get(
        self,
        query: str,
        model_config: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Get a cached response for a query.

        Tries exact match first, then semantic similarity.

        Args:
            query: Query text
            model_config: Model configuration identifier

        Returns:
            Cached response dict or None
        """
        exact = self._exact_hash(query, model_config)

        with self._lock:
            # Fast path: exact match
            if exact in self._exact_index:
                store_key = self._exact_index[exact]
                entry = self._store.get(store_key)
                if entry and time.time() <= entry.expires_at:
                    entry.hit_count += 1
                    self._stats["exact_hits"] += 1
                    return {
                        "response": entry.response,
                        "match_type": "exact",
                        "similarity": 1.0,
                        "original_query": entry.query,
                    }
                elif entry:
                    # Expired
                    del self._store[store_key]
                    del self._exact_index[exact]

            # Slow path: semantic similarity
            query_embedding = self._get_embedding(query)
            best_key = None
            best_sim = 0.0

            now = time.time()
            expired_keys = []

            for key, entry in self._store.items():
                if now > entry.expires_at:
                    expired_keys.append(key)
                    continue
                if model_config and entry.model_config != model_config:
                    continue

                sim = _cosine_similarity(query_embedding, entry.query_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            # Cleanup expired
            for k in expired_keys:
                e = self._store.pop(k, None)
                if e:
                    self._exact_index.pop(e.exact_hash, None)

            if best_key and best_sim >= self._threshold:
                entry = self._store[best_key]
                entry.hit_count += 1
                self._stats["semantic_hits"] += 1
                return {
                    "response": entry.response,
                    "match_type": "semantic",
                    "similarity": best_sim,
                    "original_query": entry.query,
                }

            self._stats["misses"] += 1
            return None

    def put(
        self,
        query: str,
        response: Dict[str, Any],
        model_config: str = "",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache a response.

        Args:
            query: Query text
            response: Response to cache
            model_config: Model configuration identifier
            ttl: TTL in seconds
        """
        effective_ttl = ttl or self._default_ttl
        now = time.time()
        exact = self._exact_hash(query, model_config)
        embedding = self._get_embedding(query)
        store_key = self._make_key()

        with self._lock:
            # Remove existing exact match
            if exact in self._exact_index:
                old_key = self._exact_index.pop(exact)
                self._store.pop(old_key, None)

            self._store[store_key] = _SemanticEntry(
                query=query,
                query_embedding=embedding,
                response=response,
                model_config=model_config,
                created_at=now,
                expires_at=now + effective_ttl,
                exact_hash=exact,
            )
            self._exact_index[exact] = store_key
            self._stats["stores"] += 1
            self._evict_if_needed()

    def invalidate(self, query: str, model_config: str = "") -> bool:
        """Invalidate a specific cached response."""
        exact = self._exact_hash(query, model_config)
        with self._lock:
            if exact in self._exact_index:
                store_key = self._exact_index.pop(exact)
                self._store.pop(store_key, None)
                return True
        return False

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._exact_index.clear()
            return count

    def get_stats(self) -> dict:
        total = (
            self._stats["exact_hits"]
            + self._stats["semantic_hits"]
            + self._stats["misses"]
        )
        with self._lock:
            return {
                **self._stats,
                "entries": len(self._store),
                "total_lookups": total,
                "hit_rate": (
                    (self._stats["exact_hits"] + self._stats["semantic_hits"]) / total
                    if total > 0
                    else 0
                ),
                "threshold": self._threshold,
            }
