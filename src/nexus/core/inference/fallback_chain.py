"""
Model fallback chain with latency-based selection.

Routes inference requests through a prioritized chain of models,
falling back to the next model if the current one fails or exceeds
latency thresholds. Dynamically re-ranks models based on observed latency.
"""

import asyncio
import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ModelEndpoint:
    """Configuration for a model in the fallback chain."""
    name: str
    inference_fn: Callable
    max_latency_seconds: float = 10.0
    weight: float = 1.0
    enabled: bool = True


@dataclass
class _LatencyWindow:
    """Rolling window of latency observations."""
    observations: deque = field(default_factory=lambda: deque(maxlen=100))
    failure_count: int = 0
    success_count: int = 0

    @property
    def avg_latency(self) -> float:
        if not self.observations:
            return float("inf")
        return sum(self.observations) / len(self.observations)

    @property
    def p95_latency(self) -> float:
        if not self.observations:
            return float("inf")
        sorted_obs = sorted(self.observations)
        idx = int(len(sorted_obs) * 0.95)
        return sorted_obs[min(idx, len(sorted_obs) - 1)]

    @property
    def failure_rate(self) -> float:
        total = self.failure_count + self.success_count
        return self.failure_count / total if total > 0 else 0


class ModelFallbackChain:
    """
    Routes requests through a chain of models with automatic fallback.

    Models are ranked by observed latency and reliability. If the
    primary model fails or is too slow, the next model is tried.
    """

    def __init__(self, latency_penalty_factor=1.5, failure_penalty_factor=2.0):
        self._models: Dict[str, ModelEndpoint] = {}
        self._chain_order: List[str] = []
        self._latency: Dict[str, _LatencyWindow] = {}
        self._lock = threading.RLock()
        self._latency_penalty = latency_penalty_factor
        self._failure_penalty = failure_penalty_factor
        self._stats = {"requests": 0, "fallbacks": 0, "failures": 0}

    def add_model(self, endpoint: ModelEndpoint):
        """Add a model to the fallback chain."""
        self._models[endpoint.name] = endpoint
        self._chain_order.append(endpoint.name)
        self._latency[endpoint.name] = _LatencyWindow()

    def _ranked_models(self) -> List[str]:
        """Get models ranked by effective score (lower is better)."""
        def score(name):
            ep = self._models[name]
            if not ep.enabled:
                return float("inf")
            lw = self._latency[name]
            latency_score = lw.avg_latency * self._latency_penalty
            failure_score = lw.failure_rate * self._failure_penalty * 100
            return (latency_score + failure_score) / ep.weight
        return sorted(self._models.keys(), key=score)

    async def infer(self, prompt: str, **kwargs) -> Tuple[Any, str]:
        """
        Run inference through the fallback chain.

        Returns:
            Tuple of (result, model_name) for the model that succeeded.

        Raises:
            RuntimeError: If all models in the chain fail.
        """
        self._stats["requests"] += 1
        ranked = self._ranked_models()
        errors = []

        for i, model_name in enumerate(ranked):
            ep = self._models[model_name]
            if not ep.enabled:
                continue

            if i > 0:
                self._stats["fallbacks"] += 1
                logger.info("Falling back to %s (attempt %d)", model_name, i + 1)

            start = time.time()
            try:
                result = ep.inference_fn(prompt, **kwargs)
                if asyncio.iscoroutine(result):
                    result = await asyncio.wait_for(result, timeout=ep.max_latency_seconds)
                latency = time.time() - start

                with self._lock:
                    self._latency[model_name].observations.append(latency)
                    self._latency[model_name].success_count += 1

                return result, model_name

            except (asyncio.TimeoutError, Exception) as exc:
                latency = time.time() - start
                with self._lock:
                    self._latency[model_name].observations.append(latency)
                    self._latency[model_name].failure_count += 1
                errors.append((model_name, str(exc)))
                logger.warning("Model %s failed (%.2fs): %s", model_name, latency, exc)

        self._stats["failures"] += 1
        raise RuntimeError(f"All models failed: {errors}")

    def infer_sync(self, prompt: str, **kwargs) -> Tuple[Any, str]:
        """Synchronous inference through the fallback chain."""
        self._stats["requests"] += 1
        ranked = self._ranked_models()
        errors = []

        for i, model_name in enumerate(ranked):
            ep = self._models[model_name]
            if not ep.enabled:
                continue
            if i > 0:
                self._stats["fallbacks"] += 1

            start = time.time()
            try:
                result = ep.inference_fn(prompt, **kwargs)
                latency = time.time() - start
                with self._lock:
                    self._latency[model_name].observations.append(latency)
                    self._latency[model_name].success_count += 1
                return result, model_name
            except Exception as exc:
                latency = time.time() - start
                with self._lock:
                    self._latency[model_name].observations.append(latency)
                    self._latency[model_name].failure_count += 1
                errors.append((model_name, str(exc)))

        self._stats["failures"] += 1
        raise RuntimeError(f"All models failed: {errors}")

    def disable_model(self, name: str):
        if name in self._models:
            self._models[name].enabled = False

    def enable_model(self, name: str):
        if name in self._models:
            self._models[name].enabled = True

    def get_rankings(self) -> List[dict]:
        ranked = self._ranked_models()
        return [
            {
                "name": n,
                "avg_latency": round(self._latency[n].avg_latency, 3),
                "p95_latency": round(self._latency[n].p95_latency, 3),
                "failure_rate": round(self._latency[n].failure_rate, 3),
                "enabled": self._models[n].enabled,
            }
            for n in ranked
        ]

    def get_stats(self) -> dict:
        return {**self._stats, "model_count": len(self._models),
                "rankings": self.get_rankings()}
