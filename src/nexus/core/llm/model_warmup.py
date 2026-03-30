"""
Model warm-up on startup.

Pre-loads models and runs dummy inference to warm up JIT compilation,
GPU memory allocation, and connection pools before the first real request.
This eliminates cold-start latency for the initial requests.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class WarmupStatus(str, Enum):
    PENDING = "pending"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WarmupResult:
    """Result of warming up a single model."""
    model_name: str
    status: WarmupStatus
    warmup_time_seconds: float = 0.0
    first_token_latency_ms: float = 0.0
    memory_allocated_mb: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WarmupConfig:
    """Configuration for a model warm-up."""
    model_name: str
    warmup_prompt: str = "Hello, this is a warm-up request."
    max_tokens: int = 10
    timeout_seconds: float = 60.0
    required: bool = False  # If True, startup fails if warmup fails
    priority: int = 0  # Higher = warmed up first


class ModelWarmupManager:
    """
    Manages model warm-up during application startup.

    Features:
    - Parallel warm-up of multiple models
    - Priority ordering (critical models first)
    - Configurable timeout per model
    - Health status tracking
    - Retry with exponential backoff
    """

    def __init__(
        self,
        inference_fn: Optional[Callable[..., Coroutine]] = None,
        max_parallel: int = 3,
        max_retries: int = 2,
    ):
        """
        Args:
            inference_fn: Async callable ``(model_name, prompt, max_tokens) -> response``
            max_parallel: Max models to warm up concurrently
            max_retries: Max retries per model
        """
        self._inference_fn = inference_fn
        self._max_parallel = max_parallel
        self._max_retries = max_retries
        self._configs: List[WarmupConfig] = []
        self._results: Dict[str, WarmupResult] = {}
        self._ready = False

    def register(self, config: WarmupConfig) -> None:
        """Register a model for warm-up."""
        self._configs.append(config)
        self._results[config.model_name] = WarmupResult(
            model_name=config.model_name, status=WarmupStatus.PENDING
        )

    def register_model(
        self,
        model_name: str,
        warmup_prompt: str = "Hello, this is a warm-up request.",
        max_tokens: int = 10,
        timeout_seconds: float = 60.0,
        required: bool = False,
        priority: int = 0,
    ) -> None:
        """Convenience method to register a model."""
        self.register(WarmupConfig(
            model_name=model_name,
            warmup_prompt=warmup_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            required=required,
            priority=priority,
        ))

    async def warmup_all(self) -> Dict[str, WarmupResult]:
        """
        Warm up all registered models.

        Returns:
            Dict mapping model names to WarmupResult

        Raises:
            RuntimeError: If a required model fails to warm up
        """
        if not self._configs:
            self._ready = True
            return {}

        # Sort by priority (higher first)
        sorted_configs = sorted(self._configs, key=lambda c: c.priority, reverse=True)

        semaphore = asyncio.Semaphore(self._max_parallel)
        tasks = [
            self._warmup_model(config, semaphore)
            for config in sorted_configs
        ]

        logger.info("Starting warm-up for %d models", len(tasks))
        start = time.time()
        await asyncio.gather(*tasks)
        total_time = time.time() - start

        # Check for required model failures
        failed_required = [
            r.model_name
            for r in self._results.values()
            if r.status == WarmupStatus.FAILED
            and any(c.required for c in self._configs if c.model_name == r.model_name)
        ]
        if failed_required:
            raise RuntimeError(
                f"Required models failed warm-up: {failed_required}"
            )

        ready_count = sum(
            1 for r in self._results.values() if r.status == WarmupStatus.READY
        )
        logger.info(
            "Warm-up complete: %d/%d ready in %.1fs",
            ready_count, len(self._configs), total_time,
        )
        self._ready = True
        return dict(self._results)

    async def _warmup_model(
        self, config: WarmupConfig, semaphore: asyncio.Semaphore
    ) -> None:
        """Warm up a single model with retries."""
        async with semaphore:
            result = self._results[config.model_name]
            result.status = WarmupStatus.WARMING

            for attempt in range(1, self._max_retries + 1):
                try:
                    start = time.time()
                    if self._inference_fn:
                        response = await asyncio.wait_for(
                            self._inference_fn(
                                config.model_name,
                                config.warmup_prompt,
                                config.max_tokens,
                            ),
                            timeout=config.timeout_seconds,
                        )
                        result.first_token_latency_ms = (time.time() - start) * 1000
                    else:
                        # No inference function — just mark as ready (dry run)
                        await asyncio.sleep(0.01)

                    result.warmup_time_seconds = time.time() - start
                    result.status = WarmupStatus.READY
                    logger.info(
                        "Model %s warmed up in %.1fs",
                        config.model_name, result.warmup_time_seconds,
                    )
                    return

                except asyncio.TimeoutError:
                    result.error = f"Timeout after {config.timeout_seconds}s (attempt {attempt})"
                    logger.warning(
                        "Warm-up timeout for %s (attempt %d/%d)",
                        config.model_name, attempt, self._max_retries,
                    )
                except Exception as e:
                    result.error = f"{type(e).__name__}: {e} (attempt {attempt})"
                    logger.warning(
                        "Warm-up failed for %s (attempt %d/%d): %s",
                        config.model_name, attempt, self._max_retries, e,
                    )

                if attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)

            result.status = WarmupStatus.FAILED
            logger.error("Model %s failed warm-up after %d attempts", config.model_name, self._max_retries)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_result(self, model_name: str) -> Optional[WarmupResult]:
        return self._results.get(model_name)

    def get_all_results(self) -> Dict[str, WarmupResult]:
        return dict(self._results)

    def get_ready_models(self) -> List[str]:
        return [
            name for name, r in self._results.items()
            if r.status == WarmupStatus.READY
        ]
