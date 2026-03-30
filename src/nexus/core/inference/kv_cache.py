"""
KV-cache reuse for conversation continuations.

Manages key-value caches from transformer inference, allowing
conversation continuations to skip recomputing previous tokens.
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
    """Cached key-value state for a conversation prefix."""
    conversation_id: str
    prefix_hash: str
    kv_state: Any
    token_count: int
    model_name: str
    created_at: float
    last_used: float
    hit_count: int = 0
    size_bytes: int = 0


class KVCacheManager:
    """
    Manages KV-cache entries for conversation continuations.

    When a conversation continues, the KV cache from the previous
    turn can be reused to skip recomputing attention for the prefix.
    """

    def __init__(self, max_entries=100, max_memory_mb=2048, default_ttl=1800):
        self._store: Dict[str, KVCacheEntry] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_mb * 1024 * 1024
        self._default_ttl = default_ttl
        self._total_bytes = 0
        self._stats = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0, "tokens_saved": 0}

    def _make_key(self, conversation_id: str, prefix_tokens: List[int], model_name: str) -> str:
        content = f"{model_name}:{conversation_id}:{str(prefix_tokens[-50:])}"
        return hashlib.sha256(content.encode()).hexdigest()[:24]

    def _prefix_hash(self, prefix_tokens: List[int]) -> str:
        return hashlib.md5(str(prefix_tokens).encode()).hexdigest()[:16]

    def get(self, conversation_id: str, prefix_tokens: List[int],
            model_name: str) -> Optional[Tuple[Any, int]]:
        """
        Look up cached KV state for a conversation prefix.

        Returns:
            Tuple of (kv_state, cached_token_count) or None
        """
        key = self._make_key(conversation_id, prefix_tokens, model_name)
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if time.time() - entry.created_at > self._default_ttl:
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None
            entry.hit_count += 1
            entry.last_used = time.time()
            self._stats["hits"] += 1
            self._stats["tokens_saved"] += entry.token_count
            return entry.kv_state, entry.token_count

    def put(self, conversation_id: str, prefix_tokens: List[int],
            model_name: str, kv_state: Any, token_count: int,
            size_bytes: int = 0):
        """Cache KV state for a conversation prefix."""
        key = self._make_key(conversation_id, prefix_tokens, model_name)
        ph = self._prefix_hash(prefix_tokens)
        now = time.time()

        with self._lock:
            if key in self._store:
                self._remove_entry(key)
            entry = KVCacheEntry(
                conversation_id=conversation_id,
                prefix_hash=ph,
                kv_state=kv_state,
                token_count=token_count,
                model_name=model_name,
                created_at=now,
                last_used=now,
                size_bytes=size_bytes,
            )
            self._store[key] = entry
            self._total_bytes += size_bytes
            self._stats["stores"] += 1
            self._evict_if_needed()

    def _remove_entry(self, key: str):
        entry = self._store.pop(key, None)
        if entry:
            self._total_bytes -= entry.size_bytes

    def _evict_if_needed(self):
        while (len(self._store) > self._max_entries or
               self._total_bytes > self._max_memory_bytes) and self._store:
            victim = min(self._store, key=lambda k: self._store[k].last_used)
            self._remove_entry(victim)
            self._stats["evictions"] += 1

    def invalidate_conversation(self, conversation_id: str) -> int:
        """Invalidate all KV cache entries for a conversation."""
        with self._lock:
            keys = [k for k, v in self._store.items() if v.conversation_id == conversation_id]
            for k in keys:
                self._remove_entry(k)
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._total_bytes = 0
            return count

    def get_stats(self) -> dict:
        total = self._stats["hits"] + self._stats["misses"]
        with self._lock:
            return {
                **self._stats,
                "entries": len(self._store),
                "total_bytes": self._total_bytes,
                "hit_rate": self._stats["hits"] / total if total > 0 else 0,
            }
