"""
Inference request prioritization queue.

Queues inference requests with priority levels, ensuring high-priority
requests (interactive, real-time) are processed before batch/background work.
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
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


@dataclass(order=True)
class InferenceRequest:
    """A prioritized inference request."""
    priority: int
    submitted_at: float = field(compare=True)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], compare=False)
    prompt: str = field(default="", compare=False)
    model_name: str = field(default="", compare=False)
    params: Dict[str, Any] = field(default_factory=dict, compare=False)
    callback: Optional[Callable] = field(default=None, compare=False)
    timeout_seconds: float = field(default=60.0, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class InferenceResult:
    """Result of a processed inference request."""
    request_id: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_seconds: float = 0.0
    queue_wait_seconds: float = 0.0


class InferencePriorityQueue:
    """
    Priority queue for inference requests with configurable concurrency.

    Ensures high-priority requests get processed first while limiting
    concurrent inference to prevent resource exhaustion.
    """

    def __init__(self, max_concurrent=4, max_queue_size=1000):
        self._queue: List[InferenceRequest] = []
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_queue_size = max_queue_size
        self._processing: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, InferenceResult] = {}
        self._inference_fn: Optional[Callable] = None
        self._stats = {
            "submitted": 0, "completed": 0, "failed": 0,
            "rejected": 0, "timed_out": 0,
        }
        self._stats_by_priority = {p.name: {"count": 0, "total_latency": 0.0} for p in Priority}

    def set_inference_fn(self, fn: Callable):
        """Set the inference function that processes requests."""
        self._inference_fn = fn

    async def submit(self, request: InferenceRequest) -> str:
        """Submit a request to the queue. Returns request_id."""
        async with self._lock:
            if len(self._queue) >= self._max_queue_size:
                self._stats["rejected"] += 1
                raise RuntimeError(f"Queue full ({self._max_queue_size})")
            heapq.heappush(self._queue, request)
            self._stats["submitted"] += 1

        asyncio.create_task(self._process_next())
        return request.request_id

    async def _process_next(self):
        async with self._semaphore:
            async with self._lock:
                if not self._queue:
                    return
                request = heapq.heappop(self._queue)

            await self._execute(request)

    async def _execute(self, request: InferenceRequest):
        start = time.time()
        queue_wait = start - request.submitted_at
        try:
            if self._inference_fn is None:
                raise RuntimeError("No inference function configured")
            result = self._inference_fn(request.prompt, request.model_name, **request.params)
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=request.timeout_seconds)
            latency = time.time() - start
            ir = InferenceResult(
                request_id=request.request_id, success=True,
                result=result, latency_seconds=latency, queue_wait_seconds=queue_wait,
            )
            self._results[request.request_id] = ir
            self._stats["completed"] += 1
            pname = Priority(request.priority).name
            self._stats_by_priority[pname]["count"] += 1
            self._stats_by_priority[pname]["total_latency"] += latency
            if request.callback:
                request.callback(ir)
        except asyncio.TimeoutError:
            latency = time.time() - start
            ir = InferenceResult(
                request_id=request.request_id, success=False,
                error="timeout", latency_seconds=latency, queue_wait_seconds=queue_wait,
            )
            self._results[request.request_id] = ir
            self._stats["timed_out"] += 1
        except Exception as exc:
            latency = time.time() - start
            ir = InferenceResult(
                request_id=request.request_id, success=False,
                error=str(exc), latency_seconds=latency, queue_wait_seconds=queue_wait,
            )
            self._results[request.request_id] = ir
            self._stats["failed"] += 1

    def get_result(self, request_id: str) -> Optional[InferenceResult]:
        return self._results.get(request_id)

    def queue_depth(self) -> int:
        return len(self._queue)

    def get_stats(self) -> dict:
        return {**self._stats, "queue_depth": len(self._queue),
                "by_priority": dict(self._stats_by_priority)}
