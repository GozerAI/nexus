"""
Blueprint execution plan caching.

Caches parsed and validated blueprint execution plans so repeated
executions of the same blueprint skip parsing and validation overhead.
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
class CachedExecutionPlan:
    """A cached blueprint execution plan."""
    blueprint_id: str
    plan_hash: str
    steps: List[Dict[str, Any]]
    dependency_graph: Dict[str, List[str]]
    estimated_duration_seconds: float
    created_at: float
    expires_at: float
    hit_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BlueprintExecutionPlanCache:
    """
    Caches blueprint execution plans.

    When a blueprint is compiled into an execution plan (DAG of steps),
    we cache the result keyed by blueprint content hash. Subsequent
    executions of the same blueprint reuse the cached plan.

    Features:
    - Content-based hashing (blueprint changes invalidate cache)
    - Version-aware (different versions have different plans)
    - TTL expiration
    - Manual invalidation

    Usage::

        cache = BlueprintExecutionPlanCache()

        plan = cache.get(blueprint_spec)
        if plan is None:
            plan = compile_blueprint(blueprint_spec)
            cache.put(blueprint_spec, plan)
    """

    def __init__(
        self,
        max_entries: int = 500,
        default_ttl: int = 3600,
    ):
        self._store: Dict[str, CachedExecutionPlan] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._stats = {"hits": 0, "misses": 0, "stores": 0, "invalidations": 0}

        logger.info(
            "BlueprintExecutionPlanCache initialized (max=%d, ttl=%ds)",
            max_entries,
            default_ttl,
        )

    def _hash_blueprint(self, blueprint_data: Any) -> str:
        """Generate content hash for a blueprint."""
        if hasattr(blueprint_data, "to_dict"):
            data = blueprint_data.to_dict()
        elif hasattr(blueprint_data, "__dict__"):
            data = blueprint_data.__dict__
        elif isinstance(blueprint_data, dict):
            data = blueprint_data
        else:
            data = str(blueprint_data)

        serialized = json.dumps(data, default=str, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:32]

    def _evict_oldest(self) -> None:
        while len(self._store) > self._max_entries:
            victim = min(
                self._store, key=lambda k: self._store[k].hit_count
            )
            del self._store[victim]

    def get(self, blueprint_data: Any) -> Optional[CachedExecutionPlan]:
        """
        Get a cached execution plan for a blueprint.

        Args:
            blueprint_data: Blueprint spec or dict

        Returns:
            Cached plan or None
        """
        plan_hash = self._hash_blueprint(blueprint_data)
        with self._lock:
            entry = self._store.get(plan_hash)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if time.time() > entry.expires_at:
                del self._store[plan_hash]
                self._stats["misses"] += 1
                return None
            entry.hit_count += 1
            self._stats["hits"] += 1
            return entry

    def put(
        self,
        blueprint_data: Any,
        steps: List[Dict[str, Any]],
        dependency_graph: Optional[Dict[str, List[str]]] = None,
        estimated_duration: float = 0.0,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Cache an execution plan.

        Args:
            blueprint_data: Blueprint spec or dict
            steps: Compiled execution steps
            dependency_graph: Step dependency graph
            estimated_duration: Estimated execution duration
            ttl: TTL in seconds
            metadata: Additional metadata

        Returns:
            Plan hash for reference
        """
        plan_hash = self._hash_blueprint(blueprint_data)
        effective_ttl = ttl or self._default_ttl
        now = time.time()

        blueprint_id = ""
        if hasattr(blueprint_data, "library_id"):
            blueprint_id = blueprint_data.library_id
        elif isinstance(blueprint_data, dict):
            blueprint_id = blueprint_data.get("library_id", "")

        with self._lock:
            self._store[plan_hash] = CachedExecutionPlan(
                blueprint_id=blueprint_id,
                plan_hash=plan_hash,
                steps=steps,
                dependency_graph=dependency_graph or {},
                estimated_duration_seconds=estimated_duration,
                created_at=now,
                expires_at=now + effective_ttl,
                metadata=metadata or {},
            )
            self._stats["stores"] += 1
            self._evict_oldest()

        return plan_hash

    def invalidate(self, blueprint_data: Any) -> bool:
        """Invalidate a cached plan."""
        plan_hash = self._hash_blueprint(blueprint_data)
        with self._lock:
            if plan_hash in self._store:
                del self._store[plan_hash]
                self._stats["invalidations"] += 1
                return True
        return False

    def invalidate_by_id(self, blueprint_id: str) -> int:
        """Invalidate all plans for a blueprint ID."""
        with self._lock:
            keys = [
                k
                for k, v in self._store.items()
                if v.blueprint_id == blueprint_id
            ]
            for k in keys:
                del self._store[k]
            self._stats["invalidations"] += len(keys)
            return len(keys)

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
            }
