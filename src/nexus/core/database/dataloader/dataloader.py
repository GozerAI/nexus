"""
GraphQL-style DataLoader pattern for N+1 query prevention.

Inspired by Facebook's DataLoader: collects individual load requests
within a single "tick" and dispatches them as a single batch query.
"""

import asyncio
import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class DataLoader(Generic[K, V]):
    """
    Batches and caches individual load requests.

    The batch_fn receives a list of keys and must return a list of results
    in the same order.

    Usage::

        async def batch_users(keys):
            users = session.query(UserModel).filter(UserModel.user_id.in_(keys)).all()
            user_map = {u.user_id: u for u in users}
            return [user_map.get(k) for k in keys]

        loader = DataLoader(batch_fn=batch_users)

        # These will be batched into a single query
        user1 = await loader.load("user-1")
        user2 = await loader.load("user-2")

        # Or load many at once
        users = await loader.load_many(["user-3", "user-4"])
    """

    def __init__(
        self,
        batch_fn: Callable[[List[K]], Any],
        max_batch_size: int = 1000,
        cache_enabled: bool = True,
    ):
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size
        self._cache_enabled = cache_enabled

        self._cache: Dict[K, V] = {}
        self._queue: List[K] = []
        self._futures: Dict[K, List[asyncio.Future]] = defaultdict(list)
        self._lock = threading.Lock()
        self._dispatch_scheduled = False

        self._stats = {"loads": 0, "batches": 0, "cache_hits": 0}

    async def load(self, key: K) -> Optional[V]:
        """Load a single value by key."""
        self._stats["loads"] += 1

        if self._cache_enabled and key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[key]

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        with self._lock:
            self._queue.append(key)
            self._futures[key].append(future)

            if not self._dispatch_scheduled:
                self._dispatch_scheduled = True
                loop.call_soon(lambda: asyncio.ensure_future(self._dispatch()))

        return await future

    async def load_many(self, keys: List[K]) -> List[Optional[V]]:
        """Load multiple values by keys."""
        return await asyncio.gather(*[self.load(k) for k in keys])

    async def _dispatch(self) -> None:
        """Dispatch all queued loads as batch requests."""
        with self._lock:
            self._dispatch_scheduled = False
            if not self._queue:
                return
            queue = list(self._queue)
            self._queue.clear()
            pending_futures = dict(self._futures)
            self._futures.clear()

        # Deduplicate keys preserving order
        seen = set()
        unique_keys = []
        for k in queue:
            if k not in seen:
                seen.add(k)
                unique_keys.append(k)

        # Dispatch in chunks
        for i in range(0, len(unique_keys), self._max_batch_size):
            chunk = unique_keys[i : i + self._max_batch_size]
            self._stats["batches"] += 1

            try:
                result = self._batch_fn(chunk)
                if asyncio.iscoroutine(result):
                    result = await result

                if len(result) != len(chunk):
                    logger.warning(
                        "DataLoader batch_fn returned %d results for %d keys",
                        len(result),
                        len(chunk),
                    )

                for key, value in zip(chunk, result):
                    if self._cache_enabled:
                        self._cache[key] = value
                    for fut in pending_futures.get(key, []):
                        if not fut.done():
                            fut.set_result(value)

            except Exception as exc:
                for key in chunk:
                    for fut in pending_futures.get(key, []):
                        if not fut.done():
                            fut.set_exception(exc)

    def prime(self, key: K, value: V) -> None:
        """Prime the cache with a known value."""
        if self._cache_enabled:
            self._cache[key] = value

    def clear(self, key: Optional[K] = None) -> None:
        """Clear the cache (or a specific key)."""
        if key is not None:
            self._cache.pop(key, None)
        else:
            self._cache.clear()

    def get_stats(self) -> dict:
        """Get loader statistics."""
        total = self._stats["loads"]
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] / total if total > 0 else 0
            ),
        }


class DataLoaderRegistry:
    """
    Registry of named DataLoaders for use across a request lifecycle.

    Usage::

        registry = DataLoaderRegistry()
        registry.register("users", batch_fn=batch_load_users)
        registry.register("api_keys", batch_fn=batch_load_api_keys)

        user = await registry.get("users").load("user-1")
        keys = await registry.get("api_keys").load("user-1")
    """

    def __init__(self):
        self._loaders: Dict[str, DataLoader] = {}
        self._factories: Dict[str, Callable] = {}

    def register(
        self,
        name: str,
        batch_fn: Callable,
        max_batch_size: int = 1000,
        cache_enabled: bool = True,
    ) -> None:
        """Register a DataLoader by name."""
        self._factories[name] = lambda: DataLoader(
            batch_fn=batch_fn,
            max_batch_size=max_batch_size,
            cache_enabled=cache_enabled,
        )

    def get(self, name: str) -> DataLoader:
        """Get or create a DataLoader by name."""
        if name not in self._loaders:
            if name not in self._factories:
                raise KeyError(f"No DataLoader registered for '{name}'")
            self._loaders[name] = self._factories[name]()
        return self._loaders[name]

    def reset(self) -> None:
        """Reset all loaders (clear caches, start fresh)."""
        self._loaders.clear()

    def get_all_stats(self) -> Dict[str, dict]:
        """Get statistics for all active loaders."""
        return {name: loader.get_stats() for name, loader in self._loaders.items()}
