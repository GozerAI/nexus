"""
RAG retrieval cache with embedding fingerprints.

Caches RAG retrieval results keyed by an embedding fingerprint
of the query. When a query's embedding is close enough to a cached
query's fingerprint, the cached retrieval results are returned.
"""

import hashlib
import logging
import math
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _embedding_fingerprint(embedding: List[float], bits: int = 64) -> str:
    """
    Create a compact fingerprint from an embedding vector.

    Uses locality-sensitive hashing: quantizes each dimension
    to a single bit (positive/negative) then hashes the bit string.
    """
    if not embedding:
        return "empty"

    # Sign-based quantization
    bit_string = "".join("1" if x >= 0 else "0" for x in embedding[:bits])
    return hashlib.md5(bit_string.encode()).hexdigest()[:16]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class CachedRetrieval:
    """Cached RAG retrieval result."""
    query: str
    query_embedding: List[float]
    fingerprint: str
    documents: List[Dict[str, Any]]
    scores: List[float]
    collection_name: str
    top_k: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _CacheEntry:
    retrieval: CachedRetrieval
    created_at: float
    expires_at: float
    hit_count: int = 0


class RAGRetrievalCache:
    """
    Caches RAG retrieval results using embedding fingerprints.

    Two-level lookup:
    1. Fast path: fingerprint exact match
    2. Slow path: cosine similarity of query embeddings

    This avoids re-running vector search when the same (or very similar)
    query has already been processed.

    Usage::

        cache = RAGRetrievalCache(similarity_threshold=0.92)

        # Before running retrieval
        hit = cache.get(query_embedding, collection_name="docs", top_k=10)
        if hit:
            return hit.documents

        # After retrieval
        docs, scores = vector_search(query_embedding, top_k=10)
        cache.put(
            query="What is X?",
            query_embedding=query_embedding,
            documents=docs,
            scores=scores,
            collection_name="docs",
            top_k=10,
        )
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 5000,
        default_ttl: int = 1800,
    ):
        self._threshold = similarity_threshold
        self._max_entries = max_entries
        self._default_ttl = default_ttl

        self._store: Dict[str, _CacheEntry] = {}
        self._fingerprint_index: Dict[str, List[str]] = {}  # fp -> list of store keys
        self._lock = threading.RLock()
        self._stats = {
            "fingerprint_hits": 0,
            "similarity_hits": 0,
            "misses": 0,
            "stores": 0,
        }

        logger.info(
            "RAGRetrievalCache initialized (threshold=%.2f, max=%d, ttl=%ds)",
            similarity_threshold,
            max_entries,
            default_ttl,
        )

    def _make_key(self) -> str:
        import uuid
        return uuid.uuid4().hex[:16]

    def _evict_if_needed(self) -> None:
        while len(self._store) > self._max_entries and self._store:
            victim = min(self._store, key=lambda k: self._store[k].hit_count)
            entry = self._store.pop(victim)
            fp = entry.retrieval.fingerprint
            if fp in self._fingerprint_index:
                self._fingerprint_index[fp] = [
                    k for k in self._fingerprint_index[fp] if k != victim
                ]
                if not self._fingerprint_index[fp]:
                    del self._fingerprint_index[fp]

    def get(
        self,
        query_embedding: List[float],
        collection_name: str = "",
        top_k: int = 10,
    ) -> Optional[CachedRetrieval]:
        """
        Look up cached retrieval results.

        Args:
            query_embedding: Query embedding vector
            collection_name: Vector collection name
            top_k: Number of results requested

        Returns:
            CachedRetrieval or None
        """
        fp = _embedding_fingerprint(query_embedding)
        now = time.time()

        with self._lock:
            # Fast path: fingerprint match
            candidate_keys = self._fingerprint_index.get(fp, [])
            for key in candidate_keys:
                entry = self._store.get(key)
                if entry is None:
                    continue
                if now > entry.expires_at:
                    continue
                r = entry.retrieval
                if r.collection_name == collection_name and r.top_k >= top_k:
                    entry.hit_count += 1
                    self._stats["fingerprint_hits"] += 1
                    return r

            # Slow path: cosine similarity
            best_key = None
            best_sim = 0.0

            expired = []
            for key, entry in self._store.items():
                if now > entry.expires_at:
                    expired.append(key)
                    continue
                r = entry.retrieval
                if r.collection_name != collection_name or r.top_k < top_k:
                    continue
                sim = _cosine_sim(query_embedding, r.query_embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_key = key

            for k in expired:
                e = self._store.pop(k, None)
                if e:
                    efp = e.retrieval.fingerprint
                    if efp in self._fingerprint_index:
                        self._fingerprint_index[efp] = [
                            x for x in self._fingerprint_index[efp] if x != k
                        ]

            if best_key and best_sim >= self._threshold:
                entry = self._store[best_key]
                entry.hit_count += 1
                self._stats["similarity_hits"] += 1
                return entry.retrieval

            self._stats["misses"] += 1
            return None

    def put(
        self,
        query: str,
        query_embedding: List[float],
        documents: List[Dict[str, Any]],
        scores: List[float],
        collection_name: str = "",
        top_k: int = 10,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Cache RAG retrieval results.

        Args:
            query: Original query text
            query_embedding: Query embedding vector
            documents: Retrieved documents
            scores: Retrieval scores
            collection_name: Vector collection name
            top_k: Number of results
            ttl: TTL in seconds
            metadata: Additional metadata
        """
        effective_ttl = ttl or self._default_ttl
        now = time.time()
        fp = _embedding_fingerprint(query_embedding)
        store_key = self._make_key()

        retrieval = CachedRetrieval(
            query=query,
            query_embedding=query_embedding,
            fingerprint=fp,
            documents=documents,
            scores=scores,
            collection_name=collection_name,
            top_k=top_k,
            metadata=metadata or {},
        )

        with self._lock:
            self._store[store_key] = _CacheEntry(
                retrieval=retrieval,
                created_at=now,
                expires_at=now + effective_ttl,
            )
            if fp not in self._fingerprint_index:
                self._fingerprint_index[fp] = []
            self._fingerprint_index[fp].append(store_key)
            self._stats["stores"] += 1
            self._evict_if_needed()

    def invalidate_collection(self, collection_name: str) -> int:
        """Invalidate all cached results for a collection."""
        with self._lock:
            keys = [
                k for k, v in self._store.items()
                if v.retrieval.collection_name == collection_name
            ]
            for k in keys:
                entry = self._store.pop(k)
                fp = entry.retrieval.fingerprint
                if fp in self._fingerprint_index:
                    self._fingerprint_index[fp] = [
                        x for x in self._fingerprint_index[fp] if x != k
                    ]
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._fingerprint_index.clear()
            return count

    def get_stats(self) -> dict:
        total = (
            self._stats["fingerprint_hits"]
            + self._stats["similarity_hits"]
            + self._stats["misses"]
        )
        with self._lock:
            return {
                **self._stats,
                "entries": len(self._store),
                "fingerprints": len(self._fingerprint_index),
                "hit_rate": (
                    (self._stats["fingerprint_hits"] + self._stats["similarity_hits"])
                    / total
                    if total > 0
                    else 0
                ),
                "threshold": self._threshold,
            }
