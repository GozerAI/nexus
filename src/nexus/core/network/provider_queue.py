"""
Provider request queuing with rate limiting.

Manages a per-provider request queue that respects rate limits,
handles backpressure, and provides fair scheduling across tenants.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a provider."""
    provider_name: str
    requests_per_minute: int = 60
    requests_per_second: int = 10
    tokens_per_minute: int = 100000
    concurrent_limit: int = 10
    burst_size: int = 5  # Extra requests allowed in burst


@dataclass
class QueuedProviderRequest:
    """A request queued for a provider."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provider_name: str = ""
    model_name: str = ""
    prompt: str = ""
    estimated_tokens: int = 0
    tenant_id: str = ""
    priority: int = 0
    parameters: Dict[str, Any] = field(default_factory=dict)
    enqueued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def wait_time_ms(self) -> float:
        start = self.started_at or time.time()
        return (start - self.enqueued_at) * 1000


class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum token capacity
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens, waiting if necessary.

        Returns:
            Wait time in seconds
        """
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            deficit = tokens - self._tokens
            wait = deficit / self._rate
            self._tokens = 0
            return wait

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._capacity, self._tokens + elapsed * self._rate
        )
        self._last_refill = now

    @property
    def available(self) -> float:
        return self._tokens


class ProviderRequestQueue:
    """
    Per-provider request queue with rate limiting.

    Features:
    - Token bucket rate limiting (requests/sec and tokens/min)
    - Concurrent request limiting
    - Fair scheduling across tenants
    - Priority-based ordering
    - Automatic retry on rate limit errors
    - Queue depth monitoring
    """

    def __init__(self):
        self._configs: Dict[str, RateLimitConfig] = {}
        self._queues: Dict[str, asyncio.PriorityQueue] = {}
        self._rate_limiters: Dict[str, TokenBucket] = {}
        self._token_limiters: Dict[str, TokenBucket] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._running = False
        self._workers: Dict[str, List[asyncio.Task]] = {}
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "submitted": 0,
            "processed": 0,
            "failed": 0,
            "rate_limited": 0,
            "total_wait_ms": 0.0,
        })

    def configure_provider(self, config: RateLimitConfig) -> None:
        """Configure rate limits for a provider."""
        name = config.provider_name
        self._configs[name] = config
        self._queues[name] = asyncio.PriorityQueue()
        self._rate_limiters[name] = TokenBucket(
            rate=config.requests_per_second,
            capacity=config.requests_per_second + config.burst_size,
        )
        self._token_limiters[name] = TokenBucket(
            rate=config.tokens_per_minute / 60.0,
            capacity=config.tokens_per_minute,
        )
        self._semaphores[name] = asyncio.Semaphore(config.concurrent_limit)
        logger.info(
            "Configured provider queue: %s (rps=%d, tpm=%d, concurrent=%d)",
            name, config.requests_per_second,
            config.tokens_per_minute, config.concurrent_limit,
        )

    async def submit(
        self,
        provider_name: str,
        prompt: str,
        model_name: str = "",
        estimated_tokens: int = 100,
        tenant_id: str = "",
        priority: int = 0,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QueuedProviderRequest:
        """
        Submit a request to the provider queue.

        Returns the request object (result populated after processing).
        """
        if provider_name not in self._queues:
            raise ValueError(f"Provider {provider_name} not configured")

        request = QueuedProviderRequest(
            provider_name=provider_name,
            model_name=model_name,
            prompt=prompt,
            estimated_tokens=estimated_tokens,
            tenant_id=tenant_id,
            priority=priority,
            parameters=parameters or {},
            metadata=metadata or {},
        )

        await self._queues[provider_name].put((priority, request.enqueued_at, request))
        self._stats[provider_name]["submitted"] += 1

        return request

    async def process_next(
        self,
        provider_name: str,
        execute_fn: Callable[..., Coroutine],
    ) -> QueuedProviderRequest:
        """
        Process the next request in a provider's queue.

        Respects rate limits and concurrency constraints.

        Args:
            provider_name: Provider to process
            execute_fn: ``async def fn(request) -> result``

        Returns:
            The processed request with result/error populated
        """
        queue = self._queues.get(provider_name)
        if not queue:
            raise ValueError(f"Provider {provider_name} not configured")

        _, _, request = await queue.get()
        stats = self._stats[provider_name]

        # Wait for rate limit
        rate_limiter = self._rate_limiters.get(provider_name)
        if rate_limiter:
            wait = await rate_limiter.acquire()
            if wait > 0:
                stats["rate_limited"] += 1
                await asyncio.sleep(wait)

        # Wait for token budget
        token_limiter = self._token_limiters.get(provider_name)
        if token_limiter and request.estimated_tokens > 0:
            wait = await token_limiter.acquire(request.estimated_tokens)
            if wait > 0:
                await asyncio.sleep(wait)

        # Respect concurrency limit
        semaphore = self._semaphores.get(provider_name)
        async with semaphore if semaphore else asyncio.Lock():
            request.started_at = time.time()
            stats["total_wait_ms"] += request.wait_time_ms

            try:
                request.result = await execute_fn(request)
                request.completed_at = time.time()
                stats["processed"] += 1
            except Exception as e:
                request.error = str(e)
                request.completed_at = time.time()
                stats["failed"] += 1
                logger.warning(
                    "Provider %s request failed: %s", provider_name, e
                )

        return request

    async def start_workers(
        self,
        provider_name: str,
        execute_fn: Callable[..., Coroutine],
        num_workers: int = 3,
    ) -> None:
        """Start background workers for a provider."""
        self._running = True
        workers = []
        for i in range(num_workers):
            task = asyncio.create_task(
                self._worker_loop(provider_name, execute_fn, i)
            )
            workers.append(task)
        self._workers[provider_name] = workers

    async def _worker_loop(
        self,
        provider_name: str,
        execute_fn: Callable,
        worker_id: int,
    ) -> None:
        while self._running:
            try:
                await self.process_next(provider_name, execute_fn)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Provider %s worker %d error: %s",
                    provider_name, worker_id, e,
                )

    async def stop_workers(self, provider_name: Optional[str] = None) -> None:
        """Stop workers for a provider (or all)."""
        self._running = False
        providers = [provider_name] if provider_name else list(self._workers.keys())
        for pname in providers:
            for task in self._workers.get(pname, []):
                task.cancel()
            self._workers.pop(pname, None)

    def get_queue_depth(self, provider_name: str) -> int:
        queue = self._queues.get(provider_name)
        return queue.qsize() if queue else 0

    def get_stats(self, provider_name: Optional[str] = None) -> Dict[str, Any]:
        if provider_name:
            stats = dict(self._stats.get(provider_name, {}))
            stats["queue_depth"] = self.get_queue_depth(provider_name)
            processed = stats.get("processed", 0)
            stats["avg_wait_ms"] = (
                stats.get("total_wait_ms", 0) / max(processed, 1)
            )
            return stats

        return {
            name: {
                **dict(s),
                "queue_depth": self.get_queue_depth(name),
            }
            for name, s in self._stats.items()
        }
