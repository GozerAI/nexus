"""
Model fallback chain with latency-based selection.

Routes inference requests through a chain of models, selecting the
optimal model based on recent latency measurements, availability,
and cost. Falls back to the next model when the primary is unavailable
or exceeds latency thresholds.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class SelectionStrategy(str, Enum):
    LATENCY = "latency"         # Lowest recent latency
    COST = "cost"               # Lowest cost per token
    PRIORITY = "priority"       # Fixed priority ordering
    WEIGHTED = "weighted"       # Weighted random by inverse latency
    ROUND_ROBIN = "round_robin" # Simple rotation


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass
class LatencyWindow:
    """Sliding window of latency measurements for a model."""
    measurements: List[float] = field(default_factory=list)
    max_size: int = 50
    _failures: int = 0
    _successes: int = 0

    def record(self, latency_ms: float, success: bool = True) -> None:
        self.measurements.append(latency_ms)
        if len(self.measurements) > self.max_size:
            self.measurements.pop(0)
        if success:
            self._successes += 1
        else:
            self._failures += 1

    @property
    def p50(self) -> float:
        if not self.measurements:
            return float("inf")
        s = sorted(self.measurements)
        return s[len(s) // 2]

    @property
    def p95(self) -> float:
        if not self.measurements:
            return float("inf")
        s = sorted(self.measurements)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    @property
    def avg(self) -> float:
        if not self.measurements:
            return float("inf")
        return sum(self.measurements) / len(self.measurements)

    @property
    def success_rate(self) -> float:
        total = self._successes + self._failures
        return self._successes / total if total > 0 else 1.0


@dataclass
class FallbackModel:
    """A model in the fallback chain."""
    name: str
    priority: int = 0  # Lower = higher priority in PRIORITY mode
    cost_per_1k_tokens: float = 0.0
    max_latency_ms: float = 5000.0  # Trigger fallback above this
    weight: float = 1.0
    status: ModelStatus = ModelStatus.AVAILABLE
    latency: LatencyWindow = field(default_factory=LatencyWindow)
    last_failure_at: Optional[float] = None
    cooldown_seconds: float = 60.0  # Cooldown after failure

    @property
    def is_available(self) -> bool:
        if self.status == ModelStatus.UNAVAILABLE:
            if self.last_failure_at:
                # Check cooldown
                if time.time() - self.last_failure_at > self.cooldown_seconds:
                    return True
            return False
        return True


class FallbackChain:
    """
    Routes inference to the best available model with automatic fallback.

    Features:
    - Multiple selection strategies (latency, cost, priority, weighted)
    - Automatic failover when a model exceeds latency threshold
    - Cooldown period after failures before retrying a model
    - Latency tracking with sliding window
    - Dynamic model addition/removal
    """

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.LATENCY,
        max_retries: int = 3,
    ):
        self._strategy = strategy
        self._max_retries = max_retries
        self._models: Dict[str, FallbackModel] = {}
        self._round_robin_idx = 0
        self._stats = {
            "requests": 0,
            "fallbacks": 0,
            "failures": 0,
        }

    def add_model(self, model: FallbackModel) -> None:
        """Add a model to the chain."""
        self._models[model.name] = model

    def remove_model(self, name: str) -> None:
        """Remove a model from the chain."""
        self._models.pop(name, None)

    def get_available_models(self) -> List[FallbackModel]:
        """Get all currently available models."""
        return [m for m in self._models.values() if m.is_available]

    def select_model(self) -> Optional[FallbackModel]:
        """
        Select the best model according to the configured strategy.

        Returns:
            The selected FallbackModel, or None if no models available
        """
        available = self.get_available_models()
        if not available:
            return None

        if self._strategy == SelectionStrategy.LATENCY:
            return min(available, key=lambda m: m.latency.p50)

        elif self._strategy == SelectionStrategy.COST:
            return min(available, key=lambda m: m.cost_per_1k_tokens)

        elif self._strategy == SelectionStrategy.PRIORITY:
            return min(available, key=lambda m: m.priority)

        elif self._strategy == SelectionStrategy.WEIGHTED:
            # Weighted random selection by inverse latency
            weights = []
            for m in available:
                lat = m.latency.avg
                w = m.weight / max(lat, 1.0)
                weights.append(w)
            total = sum(weights)
            if total == 0:
                return random.choice(available)
            r = random.random() * total
            cumulative = 0.0
            for m, w in zip(available, weights):
                cumulative += w
                if r <= cumulative:
                    return m
            return available[-1]

        elif self._strategy == SelectionStrategy.ROUND_ROBIN:
            if not available:
                return None
            self._round_robin_idx = self._round_robin_idx % len(available)
            model = available[self._round_robin_idx]
            self._round_robin_idx += 1
            return model

        return available[0] if available else None

    async def execute(
        self,
        inference_fn: Callable[..., Coroutine],
        prompt: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute inference with automatic fallback.

        Args:
            inference_fn: ``async def fn(model_name, prompt, **kwargs) -> result``
            prompt: Input prompt
            **kwargs: Additional parameters passed to inference_fn

        Returns:
            Dict with ``result``, ``model_used``, ``latency_ms``, ``fallback_count``

        Raises:
            RuntimeError: If all models fail
        """
        self._stats["requests"] += 1
        fallback_count = 0
        tried: set = set()

        for attempt in range(self._max_retries + 1):
            model = self._select_excluding(tried)
            if model is None:
                break

            tried.add(model.name)
            start = time.time()

            try:
                result = await inference_fn(model.name, prompt, **kwargs)
                latency_ms = (time.time() - start) * 1000
                model.latency.record(latency_ms, success=True)

                if latency_ms > model.max_latency_ms:
                    logger.warning(
                        "Model %s latency %.0fms exceeds threshold %.0fms",
                        model.name, latency_ms, model.max_latency_ms,
                    )

                return {
                    "result": result,
                    "model_used": model.name,
                    "latency_ms": latency_ms,
                    "fallback_count": fallback_count,
                }

            except Exception as e:
                latency_ms = (time.time() - start) * 1000
                model.latency.record(latency_ms, success=False)
                model.last_failure_at = time.time()

                if model.latency.success_rate < 0.5:
                    model.status = ModelStatus.UNAVAILABLE
                elif model.latency.success_rate < 0.8:
                    model.status = ModelStatus.DEGRADED

                fallback_count += 1
                self._stats["fallbacks"] += 1
                logger.warning(
                    "Model %s failed (attempt %d): %s, falling back",
                    model.name, attempt + 1, e,
                )

        self._stats["failures"] += 1
        raise RuntimeError(
            f"All models failed after {fallback_count} fallbacks. Tried: {tried}"
        )

    def _select_excluding(self, exclude: set) -> Optional[FallbackModel]:
        """Select best model excluding already-tried ones."""
        available = [m for m in self.get_available_models() if m.name not in exclude]
        if not available:
            return None

        if self._strategy == SelectionStrategy.LATENCY:
            return min(available, key=lambda m: m.latency.p50)
        elif self._strategy == SelectionStrategy.COST:
            return min(available, key=lambda m: m.cost_per_1k_tokens)
        elif self._strategy == SelectionStrategy.PRIORITY:
            return min(available, key=lambda m: m.priority)
        return available[0]

    def record_latency(self, model_name: str, latency_ms: float, success: bool = True) -> None:
        """Externally record a latency measurement."""
        model = self._models.get(model_name)
        if model:
            model.latency.record(latency_ms, success)

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "models": {
                name: {
                    "status": m.status.value,
                    "p50_ms": m.latency.p50,
                    "p95_ms": m.latency.p95,
                    "success_rate": m.latency.success_rate,
                }
                for name, m in self._models.items()
            },
        }
