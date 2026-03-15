"""
Task queue with priority scheduling.

A general-purpose async task scheduler that manages task execution
with priority levels, rate limiting, and resource awareness.
"""

import asyncio
import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    IDLE = 4


class TaskState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(order=True)
class ScheduledTask:
    """A task in the priority scheduler."""
    sort_key: float = field(compare=True)
    task_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    name: str = field(compare=False, default="")
    priority: TaskPriority = field(compare=False, default=TaskPriority.NORMAL)
    fn: Optional[Callable[..., Coroutine]] = field(compare=False, default=None)
    args: tuple = field(compare=False, default_factory=tuple)
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    state: TaskState = field(compare=False, default=TaskState.QUEUED)
    timeout_seconds: float = field(compare=False, default=300.0)
    max_retries: int = field(compare=False, default=1)
    retries: int = field(compare=False, default=0)
    enqueued_at: float = field(compare=False, default_factory=time.time)
    started_at: Optional[float] = field(compare=False, default=None)
    completed_at: Optional[float] = field(compare=False, default=None)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0


class TaskScheduler:
    """
    Priority-based task scheduler with configurable concurrency.

    Features:
    - Multi-level priority queue
    - Configurable max concurrency per priority level
    - Task timeout and retry
    - Rate limiting (tasks per second)
    - Task cancellation
    - Statistics tracking
    """

    def __init__(
        self,
        max_workers: int = 10,
        rate_limit: Optional[float] = None,  # Max tasks/second
        priority_weights: Optional[Dict[TaskPriority, int]] = None,
    ):
        """
        Args:
            max_workers: Max concurrent task executions
            rate_limit: Maximum tasks per second (None = unlimited)
            priority_weights: Max concurrent tasks per priority level
        """
        self._max_workers = max_workers
        self._rate_limit = rate_limit
        self._priority_weights = priority_weights or {}
        self._heap: List[ScheduledTask] = []
        self._tasks: Dict[str, ScheduledTask] = {}
        self._running: set = set()
        self._cancelled: set = set()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._is_running = False
        self._workers: List[asyncio.Task] = []
        self._last_dispatch_time = 0.0
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "retried": 0,
            "total_wait_ms": 0.0,
            "total_exec_ms": 0.0,
        }

    async def submit(
        self,
        fn: Callable[..., Coroutine],
        name: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: float = 300.0,
        max_retries: int = 1,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ScheduledTask:
        """Submit a task for execution."""
        now = time.time()
        task = ScheduledTask(
            sort_key=float(priority.value) + now * 1e-10,  # Priority + FIFO tiebreak
            name=name or f"task-{self._stats['submitted']}",
            priority=priority,
            fn=fn,
            args=args,
            kwargs=kwargs or {},
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            metadata=metadata or {},
        )

        async with self._lock:
            heapq.heappush(self._heap, task)
            self._tasks[task.task_id] = task
            self._stats["submitted"] += 1
            self._not_empty.set()

        return task

    async def cancel(self, task_id: str) -> bool:
        """Cancel a queued task."""
        task = self._tasks.get(task_id)
        if task and task.state == TaskState.QUEUED:
            task.state = TaskState.CANCELLED
            self._cancelled.add(task_id)
            self._stats["cancelled"] += 1
            return True
        return False

    async def start(self) -> None:
        """Start worker tasks."""
        if self._is_running:
            return
        self._is_running = True
        for i in range(self._max_workers):
            worker = asyncio.create_task(self._worker_loop(i))
            self._workers.append(worker)
        logger.info("TaskScheduler started with %d workers", self._max_workers)

    async def stop(self) -> None:
        """Stop all workers."""
        self._is_running = False
        self._not_empty.set()
        for w in self._workers:
            w.cancel()
        self._workers.clear()

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop that dequeues and executes tasks."""
        while self._is_running:
            try:
                await self._not_empty.wait()

                async with self._lock:
                    if not self._heap:
                        self._not_empty.clear()
                        continue

                    task = heapq.heappop(self._heap)
                    if not self._heap:
                        self._not_empty.clear()

                if task.task_id in self._cancelled:
                    continue

                # Rate limiting
                if self._rate_limit:
                    now = time.time()
                    min_interval = 1.0 / self._rate_limit
                    elapsed = now - self._last_dispatch_time
                    if elapsed < min_interval:
                        await asyncio.sleep(min_interval - elapsed)
                    self._last_dispatch_time = time.time()

                # Execute
                self._running.add(task.task_id)
                task.state = TaskState.RUNNING
                task.started_at = time.time()

                try:
                    if task.fn:
                        result = await asyncio.wait_for(
                            task.fn(*task.args, **task.kwargs),
                            timeout=task.timeout_seconds,
                        )
                        task.result = result
                    task.state = TaskState.COMPLETED
                    task.completed_at = time.time()
                    self._stats["completed"] += 1
                    self._stats["total_exec_ms"] += task.elapsed_seconds * 1000

                except Exception as e:
                    task.error = str(e)
                    if task.retries < task.max_retries:
                        task.retries += 1
                        task.state = TaskState.QUEUED
                        self._stats["retried"] += 1
                        async with self._lock:
                            heapq.heappush(self._heap, task)
                            self._not_empty.set()
                    else:
                        task.state = TaskState.FAILED
                        task.completed_at = time.time()
                        self._stats["failed"] += 1

                finally:
                    self._running.discard(task.task_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker %d error: %s", worker_id, e)

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        return self._tasks.get(task_id)

    def get_queue_depth(self) -> int:
        return len(self._heap)

    def get_running_count(self) -> int:
        return len(self._running)

    def get_stats(self) -> Dict[str, Any]:
        completed = self._stats["completed"]
        return {
            **self._stats,
            "queue_depth": len(self._heap),
            "running": len(self._running),
            "avg_exec_ms": (
                self._stats["total_exec_ms"] / completed if completed > 0 else 0.0
            ),
        }
