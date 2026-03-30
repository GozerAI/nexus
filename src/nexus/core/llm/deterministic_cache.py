"""
Model output caching for deterministic prompts.

Caches LLM responses for prompts that are known to produce deterministic
output (temperature=0, seed set, or explicitly marked). This avoids
redundant inference for repeated identical prompts.

Unlike semantic caching, this uses exact-match hashing on the full
request parameters (prompt + model + temperature + seed + system prompt).
"""

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """A cached model response."""
    cache_key: str
    model_name: str
    prompt_hash: str
    response: Any
    token_count: int
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    hit_count: int = 0
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeterministicOutputCache:
    """
    Exact-match cache for deterministic LLM outputs.

    Features:
    - Exact-match keying on (model, prompt, temperature, seed, system_prompt)
    - Configurable TTL per entry
    - LRU eviction with size limits
    - Statistics tracking
    - Automatic detection of deterministic parameters
    """

    DEFAULT_TTL = 3600  # 1 hour
    DEFAULT_MAX_ENTRIES = 10000
    DEFAULT_MAX_SIZE_MB = 256

    # Parameters that indicate deterministic output
    DETERMINISTIC_INDICATORS = {
        "temperature": 0.0,
        "top_p": 1.0,
    }

    def __init__(
        self,
        default_ttl: float = DEFAULT_TTL,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        auto_detect_deterministic: bool = True,
    ):
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._auto_detect = auto_detect_deterministic
        self._cache: Dict[str, CachedResponse] = {}
        self._lock = threading.Lock()
        self._total_size_bytes = 0
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "skipped_non_deterministic": 0,
        }

    @staticmethod
    def _make_key(
        model_name: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        seed: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a deterministic cache key from request parameters."""
        key_parts = {
            "model": model_name,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "seed": seed,
        }
        if extra_params:
            key_parts["extra"] = extra_params
        raw = json.dumps(key_parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def is_deterministic(self, temperature: float = 0.0, seed: Optional[int] = None, **kwargs: Any) -> bool:
        """Check if the given parameters indicate deterministic output."""
        if seed is not None:
            return True
        if temperature == 0.0:
            return True
        if not self._auto_detect:
            return False
        return False

    def get(
        self,
        model_name: str,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        seed: Optional[int] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[CachedResponse]:
        """
        Look up a cached response.

        Args:
            model_name: Model identifier
            prompt: The prompt text
            system_prompt: System prompt
            temperature: Sampling temperature
            seed: Random seed
            extra_params: Additional deterministic parameters

        Returns:
            CachedResponse if found and valid, None otherwise
        """
        key = self._make_key(model_name, prompt, system_prompt, temperature, seed, extra_params)

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.expires_at > 0 and time.time() > entry.expires_at:
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None

            entry.hit_count += 1
            self._stats["hits"] += 1
            return entry

    def put(
        self,
        model_name: str,
        prompt: str,
        response: Any,
        system_prompt: str = "",
        temperature: float = 0.0,
        seed: Optional[int] = None,
        token_count: int = 0,
        ttl: Optional[float] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Optional[CachedResponse]:
        """
        Store a response in the cache.

        Args:
            model_name: Model identifier
            prompt: The prompt text
            response: Model response to cache
            system_prompt: System prompt
            temperature: Sampling temperature
            seed: Random seed
            token_count: Number of tokens in response
            ttl: Time-to-live in seconds (None = default)
            extra_params: Additional parameters
            force: Store even if not detected as deterministic

        Returns:
            CachedResponse if stored, None if skipped
        """
        if not force and not self.is_deterministic(temperature, seed):
            self._stats["skipped_non_deterministic"] += 1
            return None

        key = self._make_key(model_name, prompt, system_prompt, temperature, seed, extra_params)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()

        # Estimate size
        size = len(json.dumps(response, default=str).encode("utf-8")) if response else 0

        entry = CachedResponse(
            cache_key=key,
            model_name=model_name,
            prompt_hash=hashlib.md5(prompt.encode()).hexdigest()[:16],
            response=response,
            token_count=token_count,
            created_at=now,
            expires_at=now + effective_ttl if effective_ttl > 0 else 0.0,
            size_bytes=size,
            metadata={"temperature": temperature, "seed": seed},
        )

        with self._lock:
            self._evict_if_needed(size)

            if key in self._cache:
                self._remove_entry(key)

            self._cache[key] = entry
            self._total_size_bytes += size
            self._stats["stores"] += 1

        return entry

    def _remove_entry(self, key: str) -> None:
        """Remove an entry (must be called under lock)."""
        if key in self._cache:
            self._total_size_bytes -= self._cache[key].size_bytes
            del self._cache[key]

    def _evict_if_needed(self, incoming_bytes: int) -> None:
        """Evict LRU entries if over limits."""
        while len(self._cache) >= self._max_entries:
            self._evict_lru()

        while self._total_size_bytes + incoming_bytes > self._max_size_bytes and self._cache:
            self._evict_lru()

    def _evict_lru(self) -> None:
        if not self._cache:
            return
        # Evict entry with oldest access (lowest hit_count as tiebreaker)
        lru_key = min(
            self._cache,
            key=lambda k: (self._cache[k].created_at, -self._cache[k].hit_count),
        )
        self._remove_entry(lru_key)
        self._stats["evictions"] += 1

    def invalidate(self, model_name: Optional[str] = None) -> int:
        """Invalidate cache entries, optionally filtered by model."""
        with self._lock:
            if model_name is None:
                count = len(self._cache)
                self._cache.clear()
                self._total_size_bytes = 0
                return count
            to_remove = [
                k for k, v in self._cache.items() if v.model_name == model_name
            ]
            for k in to_remove:
                self._remove_entry(k)
            return len(to_remove)

    def get_stats(self) -> Dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "entries": len(self._cache),
            "total_size_bytes": self._total_size_bytes,
            "hit_rate": self._stats["hits"] / total if total > 0 else 0.0,
        }
