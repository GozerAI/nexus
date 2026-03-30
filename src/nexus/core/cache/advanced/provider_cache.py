"""
Provider capability cache with health-based refresh.

Caches provider capabilities (available models, limits, supported features)
and refreshes them based on provider health status changes.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ProviderHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProviderCapabilities:
    """Cached provider capabilities."""
    provider_name: str
    models: List[str]
    capabilities: Set[str]
    rate_limits: Dict[str, int]
    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    health: ProviderHealth
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _CacheEntry:
    capabilities: ProviderCapabilities
    cached_at: float
    expires_at: float
    last_health: ProviderHealth
    refresh_count: int = 0


class ProviderCapabilityCache:
    """
    Caches provider capabilities with health-aware refresh.

    When a provider's health degrades, its cached capabilities are
    refreshed more aggressively to detect recovery or capability changes.

    Usage::

        cache = ProviderCapabilityCache()

        caps = cache.get("openai")
        if caps is None:
            caps = fetch_capabilities("openai")
            cache.put("openai", caps)

        # Health change triggers faster refresh
        cache.update_health("openai", ProviderHealth.DEGRADED)
    """

    def __init__(
        self,
        default_ttl: int = 300,
        degraded_ttl: int = 60,
        unhealthy_ttl: int = 15,
    ):
        self._store: Dict[str, _CacheEntry] = {}
        self._lock = threading.RLock()
        self._default_ttl = default_ttl
        self._degraded_ttl = degraded_ttl
        self._unhealthy_ttl = unhealthy_ttl
        self._stats = {
            "hits": 0,
            "misses": 0,
            "health_refreshes": 0,
            "stores": 0,
        }

        logger.info(
            "ProviderCapabilityCache initialized (ttl=%d/%d/%ds)",
            default_ttl,
            degraded_ttl,
            unhealthy_ttl,
        )

    def _ttl_for_health(self, health: ProviderHealth) -> int:
        if health == ProviderHealth.UNHEALTHY:
            return self._unhealthy_ttl
        if health == ProviderHealth.DEGRADED:
            return self._degraded_ttl
        return self._default_ttl

    def get(self, provider_name: str) -> Optional[ProviderCapabilities]:
        """
        Get cached capabilities for a provider.

        Returns None if not cached or expired.
        """
        with self._lock:
            entry = self._store.get(provider_name)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if time.time() > entry.expires_at:
                del self._store[provider_name]
                self._stats["misses"] += 1
                return None
            self._stats["hits"] += 1
            return entry.capabilities

    def put(
        self,
        provider_name: str,
        capabilities: ProviderCapabilities,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache provider capabilities.

        TTL is auto-adjusted based on health status unless explicitly provided.
        """
        health = capabilities.health
        effective_ttl = ttl or self._ttl_for_health(health)
        now = time.time()

        with self._lock:
            self._store[provider_name] = _CacheEntry(
                capabilities=capabilities,
                cached_at=now,
                expires_at=now + effective_ttl,
                last_health=health,
            )
            self._stats["stores"] += 1

    def update_health(
        self,
        provider_name: str,
        new_health: ProviderHealth,
    ) -> bool:
        """
        Update a provider's health status.

        If health has changed, the TTL is adjusted and the entry may
        expire sooner to force a capabilities refresh.

        Returns True if the entry was updated.
        """
        with self._lock:
            entry = self._store.get(provider_name)
            if entry is None:
                return False

            if entry.last_health != new_health:
                entry.capabilities.health = new_health
                entry.last_health = new_health
                entry.refresh_count += 1
                self._stats["health_refreshes"] += 1

                # Recalculate expiry based on new health
                new_ttl = self._ttl_for_health(new_health)
                entry.expires_at = time.time() + new_ttl

                logger.info(
                    "Provider %s health changed to %s, TTL adjusted to %ds",
                    provider_name,
                    new_health.value,
                    new_ttl,
                )
                return True
        return False

    def invalidate(self, provider_name: str) -> bool:
        with self._lock:
            if provider_name in self._store:
                del self._store[provider_name]
                return True
        return False

    def get_all_healthy(self) -> Dict[str, ProviderCapabilities]:
        """Get all cached capabilities for healthy providers."""
        result = {}
        now = time.time()
        with self._lock:
            for name, entry in self._store.items():
                if now <= entry.expires_at and entry.capabilities.health == ProviderHealth.HEALTHY:
                    result[name] = entry.capabilities
        return result

    def clear(self) -> int:
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
                "hit_rate": self._stats["hits"] / total if total > 0 else 0,
                "providers": list(self._store.keys()),
            }
