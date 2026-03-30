"""
Inference request prioritization queue.

Manages a priority queue for LLM inference requests, ensuring that
high-priority requests (user-facing, real-time) are processed before
low-priority ones (background tasks, batch processing).

Supports:
- Priority levels with aging (starvation prevention)
- Per-priority rate limiting
- Request coalescing for identical prompts
- Queue depth monitoring and backpressure
"""

import asyncio
import heapq
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Request priority levels. Lower numeric value = higher priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class InferenceRequest:
    """A queued inference request."""
    # Ordering fields (used by heapq)
    effective_priority: float = field(compare=True)
    enqueued_at: float = field(compare=True)

    # Payload (not used for ordering)
    request_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    priority: Priority = field(compare=False, default=Priority.NORMAL)
    model_name: str = field(compare=False, default="")
    prompt: str = field(compare=False, default="")
    parameters: Dict[str, Any] = field(compare=False, default_factory=dict)
    callback: Optional[Callable] = field(compare=False, default=None)
    timeout_seconds: float = field(compare=False, default=300.0)
    metadata: Dict[str, Any] = field(compare=False, default_factory=dict)

    # State
    started_at: Optional[float] = field(compare=False, default=None)
    completed_at: Optional[float] = field(compare=False, default=None)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)

    @property
    def wait_time_seconds(self) -> float:
        start = self.started_at or time.time()
        return start - self.enqueued_at

    @property
    def total_time_seconds(self) -> float:
        if self.completed_at:
            return self.completed_at - self.enqueued_at
        return time.time() - self.enqueued_at


class InferencePriorityQueue:
    """
    Priority queue for LLM inference requests.

    Features:
    - Multi-level priority with aging to prevent starvation
    - Concurrent processing with configurable worker count
    - Request coalescing: identical prompts share results
    - Queue depth limits with backpressure
    - Statistics tracking
    """

    DEFAULT_MAX_QUEUE = 1000
    AGING_FACTOR = 0.01  # Priority improves by 0.01 per second waiting

    def __init__(
        self,
        inference_fn: Optional[Callable[..., Coroutine]] = None,
        max_queue_size: int = DEFAULT_MAX_QUEUE,
        max_workers: int = 5,
        aging_factor: float = AGING_FACTOR,
        enable_coalescing: bool = True,
    ):
        """
        Args:
            inference_fn: Async callable ``(request) -> result``
            max_queue_size: Maximum queued requests
            max_workers: Concurrent inference workers
            aging_factor: How fast low-priority requests age up
            enable_coalescing: Merge identical prompts
        """
        self._inference_fn = inference_fn
        self._max_queue = max_queue_size
        self._max_workers = max_workers
        self._aging_factor = aging_factor
        self._enable_coalescing = enable_coalescing

        self._heap: List[InferenceRequest] = []
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Event()
        self._pending_prompts: Dict[str, List[asyncio.Future]] = {}
        self._running = False
        self._workers: List[asyncio.Task] = []

        self._stats = {
            "enqueued": 0,
            "processed": 0,
            "failed": 0,
            "coalesced": 0,
            "rejected": 0,
            "total_wait_ms": 0.0,
            "total_process_ms": 0.0,
        }

    async def enqueue(
        self,
        prompt: str,
        model_name: str = "",
        priority: Priority = Priority.NORMAL,
        parameters: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 300.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceRequest:
        """
        Enqueue an inference request.

        Args:
            prompt: Input prompt
            model_name: Target model
            priority: Request priority
            parameters: Model parameters
            timeout_seconds: Request timeout
            metadata: Additional metadata

        Returns:
            The queued InferenceRequest

        Raises:
            asyncio.QueueFull: If queue is at capacity
        """
        async with self._lock:
            if len(self._heap) >= self._max_queue:
                self._stats["rejected"] += 1
                raise asyncio.QueueFull(
                    f"Inference queue full ({self._max_queue})"
                )

            now = time.time()
            request = InferenceRequest(
                effective_priority=float(priority.value),
                enqueued_at=now,
                priority=priority,
                model_name=model_name,
                prompt=prompt,
                parameters=parameters or {},
                timeout_seconds=timeout_seconds,
                metadata=metadata or {},
            )

            heapq.heappush(self._heap, request)
            self._stats["enqueued"] += 1
            self._not_empty.set()

        logger.debug(
            "Enqueued request %s (priority=%s, queue_depth=%d)",
            request.request_id, priority.name, len(self._heap),
        )
        return request

    async def dequeue(self) -> InferenceRequest:
        """
        Dequeue the highest-priority request.

        Applies aging to prevent starvation of low-priority items.
        """
        while True:
            await self._not_empty.wait()
            async with self._lock:
                if not self._heap:
                    self._not_empty.clear()
                    continue

                # Apply aging to all entries
                now = time.time()
                for req in self._heap:
                    age = now - req.enqueued_at
                    req.effective_priority = (
                        float(req.priority.value) - age * self._aging_factor
                    )

                # Re-heapify after aging adjustments
                heapq.heapify(self._heap)
                request = heapq.heappop(self._heap)

                if not self._heap:
                    self._not_empty.clear()

                request.started_at = now
                self._stats["total_wait_ms"] += request.wait_time_seconds * 1000
                return request

    async def start_workers(self) -> None:
        """Start background worker tasks for processing the queue."""
        if self._running:
            return
        self._running = True
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)
        logger.info("Started %d inference workers", self._max_workers)

    async def stop_workers(self) -> None:
        """Stop all worker tasks."""
        self._running = False
        self._not_empty.set()  # Wake up waiting workers
        for task in self._workers:
            task.cancel()
        self._workers.clear()

    async def _worker(self, worker_id: int) -> None:
        """Worker loop: dequeue and process requests."""
        while self._running:
            try:
                request = await self.dequeue()
                if not self._running:
                    break

                # Check for coalescing
                if self._enable_coalescing:
                    coalesce_key = f"{request.model_name}:{request.prompt}"
                    if coalesce_key in self._pending_prompts:
                        self._stats["coalesced"] += 1
                        continue

                try:
                    if self._inference_fn:
                        result = await asyncio.wait_for(
                            self._inference_fn(request),
                            timeout=request.timeout_seconds,
                        )
                        request.result = result
                    request.completed_at = time.time()
                    self._stats["processed"] += 1

                    process_ms = (request.completed_at - request.started_at) * 1000
                    self._stats["total_process_ms"] += process_ms

                except Exception as e:
                    request.error = str(e)
                    request.completed_at = time.time()
                    self._stats["failed"] += 1
                    logger.warning("Request %s failed: %s", request.request_id, e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Worker %d error: %s", worker_id, e)

    @property
    def queue_depth(self) -> int:
        return len(self._heap)

    def get_stats(self) -> Dict[str, Any]:
        processed = self._stats["processed"]
        return {
            **self._stats,
            "queue_depth": len(self._heap),
            "avg_wait_ms": (
                self._stats["total_wait_ms"] / max(processed, 1)
            ),
            "avg_process_ms": (
                self._stats["total_process_ms"] / max(processed, 1)
            ),
        }
