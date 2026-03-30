"""
Request batching for LLM providers.

Collects individual inference requests and groups them into batches
for providers that support batch inference (OpenAI batch API, Ollama
batch, vLLM batch). Reduces per-request overhead and improves throughput.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BatchedRequest:
    """A single request within a batch."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    model_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    future: Optional[asyncio.Future] = field(default=None, repr=False)
    submitted_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    """A group of requests to be sent together."""
    batch_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    requests: List[BatchedRequest] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    sent_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def size(self) -> int:
        return len(self.requests)


class RequestBatcher:
    """
    Batches individual requests for efficient provider calls.

    Features:
    - Configurable batch size and timeout
    - Per-model batching (requests grouped by model)
    - Automatic batch flushing on timeout or size threshold
    - Future-based result delivery to callers
    - Statistics tracking
    """

    DEFAULT_BATCH_SIZE = 10
    DEFAULT_BATCH_TIMEOUT = 0.1  # seconds

    def __init__(
        self,
        batch_fn: Optional[Callable[..., Coroutine]] = None,
        max_batch_size: int = DEFAULT_BATCH_SIZE,
        batch_timeout: float = DEFAULT_BATCH_TIMEOUT,
        per_model: bool = True,
    ):
        """
        Args:
            batch_fn: Async callable ``(requests: List[BatchedRequest]) -> List[result]``
            max_batch_size: Max requests per batch
            batch_timeout: Max seconds to wait before flushing
            per_model: If True, batch requests by model
        """
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size
        self._batch_timeout = batch_timeout
        self._per_model = per_model
        self._buffers: Dict[str, List[BatchedRequest]] = {}
        self._flush_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._stats = {
            "requests_submitted": 0,
            "batches_sent": 0,
            "requests_in_batches": 0,
            "avg_batch_size": 0.0,
            "total_batch_latency_ms": 0.0,
        }

    async def submit(
        self,
        prompt: str,
        model_name: str = "default",
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Submit a request for batched execution.

        Returns the result when the batch is processed.

        Args:
            prompt: Input prompt
            model_name: Target model
            parameters: Model parameters
            metadata: Additional metadata

        Returns:
            The inference result for this specific request
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = BatchedRequest(
            prompt=prompt,
            model_name=model_name,
            parameters=parameters or {},
            future=future,
            metadata=metadata or {},
        )

        batch_key = model_name if self._per_model else "default"

        async with self._lock:
            if batch_key not in self._buffers:
                self._buffers[batch_key] = []
            self._buffers[batch_key].append(request)
            self._stats["requests_submitted"] += 1

            buffer = self._buffers[batch_key]

            # Flush if batch is full
            if len(buffer) >= self._max_batch_size:
                batch = self._drain_buffer(batch_key)
                asyncio.ensure_future(self._process_batch(batch))
            elif batch_key not in self._flush_tasks or self._flush_tasks[batch_key].done():
                # Start timeout timer
                self._flush_tasks[batch_key] = asyncio.ensure_future(
                    self._flush_after_timeout(batch_key)
                )

        return await future

    async def _flush_after_timeout(self, batch_key: str) -> None:
        """Flush buffer after timeout."""
        await asyncio.sleep(self._batch_timeout)
        async with self._lock:
            if batch_key in self._buffers and self._buffers[batch_key]:
                batch = self._drain_buffer(batch_key)
                asyncio.ensure_future(self._process_batch(batch))

    def _drain_buffer(self, batch_key: str) -> Batch:
        """Drain the buffer for a key into a Batch."""
        requests = self._buffers.pop(batch_key, [])
        return Batch(requests=requests)

    async def _process_batch(self, batch: Batch) -> None:
        """Send batch to provider and distribute results."""
        if not batch.requests:
            return

        batch.sent_at = time.time()
        self._stats["batches_sent"] += 1
        self._stats["requests_in_batches"] += batch.size

        try:
            if self._batch_fn:
                results = await self._batch_fn(batch.requests)
            else:
                results = [None] * batch.size

            batch.completed_at = time.time()
            latency = (batch.completed_at - batch.sent_at) * 1000
            self._stats["total_batch_latency_ms"] += latency

            # Update avg batch size
            total = self._stats["batches_sent"]
            self._stats["avg_batch_size"] = (
                self._stats["requests_in_batches"] / total
            )

            # Deliver results to individual futures
            for request, result in zip(batch.requests, results):
                if request.future and not request.future.done():
                    request.future.set_result(result)

            # Handle case where fewer results than requests
            for request in batch.requests[len(results):]:
                if request.future and not request.future.done():
                    request.future.set_exception(
                        RuntimeError("No result returned for request")
                    )

        except Exception as e:
            logger.error("Batch processing failed: %s", e)
            for request in batch.requests:
                if request.future and not request.future.done():
                    request.future.set_exception(e)

    async def flush_all(self) -> None:
        """Flush all pending buffers immediately."""
        async with self._lock:
            batches = []
            for key in list(self._buffers.keys()):
                if self._buffers[key]:
                    batches.append(self._drain_buffer(key))

        for batch in batches:
            await self._process_batch(batch)

    @property
    def pending_count(self) -> int:
        return sum(len(buf) for buf in self._buffers.values())

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "pending_requests": self.pending_count,
            "active_buffers": len(self._buffers),
        }
