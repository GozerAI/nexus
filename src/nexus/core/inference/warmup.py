"""
Model warm-up with dummy inference on startup.

Pre-loads models and runs a dummy inference pass to populate caches,
JIT-compile code paths, and ensure models are ready for real requests.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WarmupResult:
    """Result of warming up a single model."""
    model_name: str
    success: bool
    latency_seconds: float
    error: Optional[str] = None


class ModelWarmup:
    """
    Warms up models by running dummy inference on startup.

    Ensures models are loaded into memory and code paths are hot
    before serving real requests.
    """

    def __init__(self, timeout_per_model: float = 30.0, max_concurrent: int = 3):
        self._timeout = timeout_per_model
        self._max_concurrent = max_concurrent
        self._models: Dict[str, Callable] = {}
        self._prompts: Dict[str, str] = {}
        self._results: Dict[str, WarmupResult] = {}

    def register(self, model_name: str, inference_fn: Callable,
                 warmup_prompt: str = "Hello, this is a warmup request."):
        """Register a model for warmup."""
        self._models[model_name] = inference_fn
        self._prompts[model_name] = warmup_prompt

    async def warmup_all(self) -> Dict[str, WarmupResult]:
        """Run warmup for all registered models concurrently."""
        semaphore = asyncio.Semaphore(self._max_concurrent)
        tasks = []
        for name in self._models:
            tasks.append(self._warmup_one(name, semaphore))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, WarmupResult):
                self._results[r.model_name] = r

        succeeded = sum(1 for r in self._results.values() if r.success)
        logger.info("Warmup complete: %d/%d models ready", succeeded, len(self._models))
        return dict(self._results)

    async def _warmup_one(self, model_name: str, semaphore: asyncio.Semaphore) -> WarmupResult:
        async with semaphore:
            fn = self._models[model_name]
            prompt = self._prompts[model_name]
            start = time.time()
            try:
                result = fn(prompt)
                if asyncio.iscoroutine(result):
                    result = await asyncio.wait_for(result, timeout=self._timeout)
                latency = time.time() - start
                logger.info("Warmup %s: %.2fs", model_name, latency)
                return WarmupResult(model_name=model_name, success=True, latency_seconds=latency)
            except Exception as exc:
                latency = time.time() - start
                logger.warning("Warmup %s failed: %s (%.2fs)", model_name, exc, latency)
                return WarmupResult(model_name=model_name, success=False,
                                    latency_seconds=latency, error=str(exc))

    def warmup_sync(self, model_name: str) -> WarmupResult:
        """Synchronous warmup for a single model."""
        fn = self._models.get(model_name)
        if not fn:
            return WarmupResult(model_name=model_name, success=False, latency_seconds=0,
                                error="Model not registered")
        prompt = self._prompts.get(model_name, "Hello")
        start = time.time()
        try:
            fn(prompt)
            latency = time.time() - start
            r = WarmupResult(model_name=model_name, success=True, latency_seconds=latency)
            self._results[model_name] = r
            return r
        except Exception as exc:
            latency = time.time() - start
            r = WarmupResult(model_name=model_name, success=False,
                             latency_seconds=latency, error=str(exc))
            self._results[model_name] = r
            return r

    def get_results(self) -> Dict[str, WarmupResult]:
        return dict(self._results)

    def is_ready(self, model_name: str) -> bool:
        r = self._results.get(model_name)
        return r is not None and r.success
