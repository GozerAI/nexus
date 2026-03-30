"""
Offline-Capable Agent Execution (Item 752)

Provides an agent execution layer that transparently falls back to local
resources when remote/cloud services are unavailable. Agents can queue
tasks, use cached tools, and run local model inference offline.
"""

import logging
import time
import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectivityStatus(str, Enum):
    """Network connectivity status."""

    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    DEFERRED = "deferred"


@dataclass
class OfflineTask:
    """A task queued for execution (possibly offline)."""

    task_id: str
    task_type: str
    agent_name: str
    parameters: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    requires_online: bool = False
    retry_count: int = 0
    max_retries: int = 3
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, deferred
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    execute_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = field(
        default=None, repr=False, compare=False
    )


@dataclass
class CachedTool:
    """A cached tool result for offline use."""

    tool_name: str
    input_hash: str
    result: Any
    cached_at: float = field(default_factory=time.time)
    ttl_seconds: float = 3600.0
    hit_count: int = 0

    @property
    def is_expired(self) -> bool:
        return time.time() - self.cached_at > self.ttl_seconds


class OfflineAgentExecutor:
    """
    Executes agent tasks with transparent offline fallback.

    Capabilities:
    - Connectivity detection and status tracking
    - Task queuing with priority-based execution
    - Tool result caching for offline reuse
    - Automatic retry when connectivity is restored
    - Local fallback handlers for critical tasks
    """

    CONNECTIVITY_CHECK_INTERVAL = 30.0  # seconds

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_queue_size: int = 1000,
        max_cache_entries: int = 5000,
    ):
        self._connectivity = ConnectivityStatus.ONLINE
        self._last_connectivity_check = 0.0
        self._task_queue: deque = deque(maxlen=max_queue_size)
        self._completed_tasks: deque = deque(maxlen=max_queue_size)
        self._tool_cache: Dict[str, CachedTool] = {}
        self._max_cache = max_cache_entries
        self._local_handlers: Dict[str, Callable] = {}
        self._connectivity_checkers: List[Callable[[], bool]] = []
        self._cache_dir = Path(cache_dir) if cache_dir else None

        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Connectivity ────────────────────────────────────────────────

    def register_connectivity_checker(self, checker: Callable[[], bool]) -> None:
        """Register a function that returns True if online."""
        self._connectivity_checkers.append(checker)

    def check_connectivity(self) -> ConnectivityStatus:
        """Check current connectivity status."""
        now = time.time()
        if now - self._last_connectivity_check < self.CONNECTIVITY_CHECK_INTERVAL:
            return self._connectivity

        self._last_connectivity_check = now

        if not self._connectivity_checkers:
            return self._connectivity

        results = []
        for checker in self._connectivity_checkers:
            try:
                results.append(checker())
            except Exception:
                results.append(False)

        if not results:
            return self._connectivity

        online_count = sum(1 for r in results if r)
        total = len(results)

        if online_count == total:
            new_status = ConnectivityStatus.ONLINE
        elif online_count > 0:
            new_status = ConnectivityStatus.DEGRADED
        else:
            new_status = ConnectivityStatus.OFFLINE

        if new_status != self._connectivity:
            logger.info(
                "Connectivity changed: %s -> %s",
                self._connectivity.value,
                new_status.value,
            )
            old = self._connectivity
            self._connectivity = new_status

            # Re-process deferred tasks if we came back online
            if (
                old in (ConnectivityStatus.OFFLINE, ConnectivityStatus.DEGRADED)
                and new_status == ConnectivityStatus.ONLINE
            ):
                self._retry_deferred_tasks()

        return self._connectivity

    def set_connectivity(self, status: ConnectivityStatus) -> None:
        """Manually set connectivity status (useful for testing)."""
        self._connectivity = status

    @property
    def is_online(self) -> bool:
        return self._connectivity == ConnectivityStatus.ONLINE

    @property
    def is_offline(self) -> bool:
        return self._connectivity == ConnectivityStatus.OFFLINE

    # ── Task Execution ──────────────────────────────────────────────

    def register_local_handler(
        self, task_type: str, handler: Callable[..., Dict[str, Any]]
    ) -> None:
        """Register a local fallback handler for a task type."""
        self._local_handlers[task_type] = handler
        logger.info("Registered local handler for task type: %s", task_type)

    def submit_task(
        self,
        task_id: str,
        task_type: str,
        agent_name: str,
        parameters: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        requires_online: bool = False,
        execute_fn: Optional[Callable] = None,
    ) -> OfflineTask:
        """
        Submit a task for execution. If offline and the task requires
        online connectivity, it is deferred. Otherwise, it is executed
        using available resources.
        """
        task = OfflineTask(
            task_id=task_id,
            task_type=task_type,
            agent_name=agent_name,
            parameters=parameters,
            priority=priority,
            requires_online=requires_online,
            execute_fn=execute_fn,
        )

        connectivity = self.check_connectivity()

        if requires_online and connectivity == ConnectivityStatus.OFFLINE:
            task.status = "deferred"
            self._task_queue.append(task)
            logger.info("Task %s deferred (offline, requires_online)", task_id)
            return task

        # Try to execute
        result = self._execute_task(task, execute_fn)
        return result

    def _execute_task(
        self, task: OfflineTask, execute_fn: Optional[Callable] = None
    ) -> OfflineTask:
        """Execute a task with fallback chain."""
        task.status = "running"

        # Try primary execution function
        if execute_fn:
            try:
                result = execute_fn(task.parameters)
                task.result = result
                task.status = "completed"
                task.completed_at = time.time()
                self._completed_tasks.append(task)
                return task
            except Exception as e:
                logger.warning(
                    "Primary execution failed for %s: %s", task.task_id, e
                )

        # Try local handler
        handler = self._local_handlers.get(task.task_type)
        if handler:
            try:
                result = handler(task.parameters)
                task.result = result
                task.status = "completed"
                task.completed_at = time.time()
                self._completed_tasks.append(task)
                logger.info("Task %s completed via local handler", task.task_id)
                return task
            except Exception as e:
                logger.warning(
                    "Local handler failed for %s: %s", task.task_id, e
                )

        # Try cached result
        cache_key = self._make_cache_key(task.task_type, task.parameters)
        cached = self._tool_cache.get(cache_key)
        if cached and not cached.is_expired:
            cached.hit_count += 1
            task.result = {"cached": True, "data": cached.result}
            task.status = "completed"
            task.completed_at = time.time()
            self._completed_tasks.append(task)
            logger.info("Task %s completed from cache", task.task_id)
            return task

        # Defer the task
        task.status = "deferred"
        task.retry_count += 1
        if task.retry_count <= task.max_retries:
            self._task_queue.append(task)
        else:
            task.status = "failed"
            task.error = "Max retries exceeded"
            self._completed_tasks.append(task)

        return task

    def _retry_deferred_tasks(self) -> int:
        """Retry deferred tasks when connectivity is restored."""
        retried = 0
        queued_tasks = list(self._task_queue)
        self._task_queue.clear()
        remaining: List[OfflineTask] = []

        for task in queued_tasks:
            if task.status == "deferred":
                task.status = "pending"
                self._execute_task(task, task.execute_fn)
                retried += 1
            else:
                remaining.append(task)

        for t in remaining:
            self._task_queue.append(t)

        if retried:
            logger.info("Retried %d deferred tasks", retried)
        return retried

    # ── Tool Caching ────────────────────────────────────────────────

    def cache_tool_result(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        ttl_seconds: float = 3600.0,
    ) -> CachedTool:
        """Cache a tool result for offline reuse."""
        input_hash = self._make_cache_key(tool_name, parameters)

        # Evict expired entries
        self._evict_expired_cache()

        # Evict oldest if at capacity
        if len(self._tool_cache) >= self._max_cache:
            oldest_key = min(
                self._tool_cache, key=lambda k: self._tool_cache[k].cached_at
            )
            del self._tool_cache[oldest_key]

        cached = CachedTool(
            tool_name=tool_name,
            input_hash=input_hash,
            result=result,
            ttl_seconds=ttl_seconds,
        )
        self._tool_cache[input_hash] = cached

        # Persist to disk if configured
        if self._cache_dir:
            self._persist_cache_entry(cached)

        return cached

    def get_cached_result(
        self, tool_name: str, parameters: Dict[str, Any]
    ) -> Optional[Any]:
        """Retrieve a cached tool result."""
        key = self._make_cache_key(tool_name, parameters)
        cached = self._tool_cache.get(key)
        if cached and not cached.is_expired:
            cached.hit_count += 1
            return cached.result

        # Try disk cache
        if self._cache_dir:
            disk_result = self._load_cache_entry(key)
            if disk_result is not None:
                return disk_result

        return None

    def _evict_expired_cache(self) -> int:
        expired_keys = [k for k, v in self._tool_cache.items() if v.is_expired]
        for k in expired_keys:
            del self._tool_cache[k]
        return len(expired_keys)

    def _persist_cache_entry(self, cached: CachedTool) -> None:
        try:
            path = self._cache_dir / f"{cached.input_hash}.json"
            data = {
                "tool_name": cached.tool_name,
                "input_hash": cached.input_hash,
                "result": cached.result,
                "cached_at": cached.cached_at,
                "ttl_seconds": cached.ttl_seconds,
            }
            path.write_text(json.dumps(data, default=str))
        except Exception as e:
            logger.debug("Failed to persist cache entry: %s", e)

    def _load_cache_entry(self, key: str) -> Optional[Any]:
        try:
            path = self._cache_dir / f"{key}.json"
            if path.exists():
                data = json.loads(path.read_text())
                if time.time() - data["cached_at"] < data["ttl_seconds"]:
                    return data["result"]
                else:
                    path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    @staticmethod
    def _make_cache_key(name: str, params: Dict[str, Any]) -> str:
        import hashlib

        raw = json.dumps({"name": name, "params": params}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    # ── Reporting ───────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Get executor status."""
        pending = sum(1 for t in self._task_queue if t.status == "pending")
        deferred = sum(1 for t in self._task_queue if t.status == "deferred")
        completed = sum(1 for t in self._completed_tasks if t.status == "completed")
        failed = sum(1 for t in self._completed_tasks if t.status == "failed")

        return {
            "connectivity": self._connectivity.value,
            "queue_size": len(self._task_queue),
            "pending_tasks": pending,
            "deferred_tasks": deferred,
            "completed_tasks": completed,
            "failed_tasks": failed,
            "cache_entries": len(self._tool_cache),
            "local_handlers": list(self._local_handlers.keys()),
        }
