"""
Model Provider Health-Based Configuration Adjustment (Item 937)

Monitors provider health metrics (latency, error rates, availability)
and automatically adjusts routing configuration to prefer healthy
providers and deprioritize degraded ones.
"""

import logging
import time
import statistics
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Provider health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthSample:
    """A single health observation."""

    provider: str
    latency_ms: float
    success: bool
    error_type: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProviderHealth:
    """Aggregated health state for a provider."""

    provider: str
    status: HealthStatus = HealthStatus.UNKNOWN
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    success_rate: float = 1.0
    error_rate: float = 0.0
    total_samples: int = 0
    consecutive_failures: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    routing_weight: float = 1.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class HealthAdjustment:
    """Record of a health-based configuration adjustment."""

    adjustment_id: str
    provider: str
    old_status: HealthStatus
    new_status: HealthStatus
    old_weight: float
    new_weight: float
    reason: str
    timestamp: float = field(default_factory=time.time)


class ProviderHealthManager:
    """
    Monitors provider health and adjusts routing weights.

    The manager:
    1. Collects health samples (latency, success/failure) per provider
    2. Computes rolling health metrics (avg latency, success rate, p95)
    3. Classifies providers as healthy/degraded/unhealthy
    4. Adjusts routing weights based on health status
    5. Tracks health transitions and generates alerts
    """

    # Thresholds for health classification
    LATENCY_DEGRADED_MS = 2000.0
    LATENCY_UNHEALTHY_MS = 5000.0
    SUCCESS_RATE_DEGRADED = 0.9
    SUCCESS_RATE_UNHEALTHY = 0.7
    CONSECUTIVE_FAILURES_UNHEALTHY = 5

    # Weight adjustments
    WEIGHT_HEALTHY = 1.0
    WEIGHT_DEGRADED = 0.5
    WEIGHT_UNHEALTHY = 0.1

    # Rolling window size
    WINDOW_SIZE = 100

    def __init__(
        self,
        latency_degraded_ms: float = 2000.0,
        latency_unhealthy_ms: float = 5000.0,
        success_rate_degraded: float = 0.9,
        success_rate_unhealthy: float = 0.7,
        window_size: int = 100,
    ):
        self._latency_degraded = latency_degraded_ms
        self._latency_unhealthy = latency_unhealthy_ms
        self._sr_degraded = success_rate_degraded
        self._sr_unhealthy = success_rate_unhealthy
        self._window_size = window_size

        self._samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self._health: Dict[str, ProviderHealth] = {}
        self._adjustments: deque = deque(maxlen=5000)
        self._callbacks: List[Callable] = []
        self._lock = threading.RLock()

    # ── Sample Recording ────────────────────────────────────────────

    def record_sample(
        self,
        provider: str,
        latency_ms: float,
        success: bool,
        error_type: Optional[str] = None,
    ) -> Optional[HealthAdjustment]:
        """
        Record a health sample and recalculate provider health.
        Returns a HealthAdjustment if status changed.
        """
        sample = HealthSample(
            provider=provider,
            latency_ms=latency_ms,
            success=success,
            error_type=error_type,
        )

        with self._lock:
            self._samples[provider].append(sample)
            adjustment = self._recalculate_health(provider)

        return adjustment

    # ── Health Queries ──────────────────────────────────────────────

    def get_health(self, provider: str) -> ProviderHealth:
        """Get current health status for a provider."""
        with self._lock:
            if provider not in self._health:
                self._health[provider] = ProviderHealth(provider=provider)
            return self._health[provider]

    def get_all_health(self) -> Dict[str, ProviderHealth]:
        with self._lock:
            return dict(self._health)

    def get_routing_weights(self) -> Dict[str, float]:
        """Get current routing weights for all providers."""
        with self._lock:
            return {
                p: h.routing_weight for p, h in self._health.items()
            }

    def get_best_provider(
        self, candidates: Optional[List[str]] = None
    ) -> Optional[str]:
        """Select the best provider based on health metrics."""
        with self._lock:
            providers = candidates or list(self._health.keys())
            if not providers:
                return None

            best = None
            best_score = -1.0
            for p in providers:
                h = self._health.get(p)
                if h is None:
                    continue
                score = (
                    h.routing_weight * 0.4
                    + h.success_rate * 0.4
                    + (1.0 - min(h.avg_latency_ms / 5000.0, 1.0)) * 0.2
                )
                if score > best_score:
                    best_score = score
                    best = p

            return best

    def register_callback(self, callback: Callable[[HealthAdjustment], None]) -> None:
        """Register a callback for health status changes."""
        self._callbacks.append(callback)

    # ── Internal ────────────────────────────────────────────────────

    def _recalculate_health(self, provider: str) -> Optional[HealthAdjustment]:
        """Recalculate health metrics for a provider."""
        samples = list(self._samples[provider])
        if not samples:
            return None

        if provider not in self._health:
            self._health[provider] = ProviderHealth(provider=provider)

        health = self._health[provider]
        old_status = health.status
        old_weight = health.routing_weight

        # Calculate metrics
        latencies = [s.latency_ms for s in samples if s.success]
        successes = sum(1 for s in samples if s.success)
        total = len(samples)

        health.total_samples = total
        health.success_rate = successes / total if total > 0 else 0.0
        health.error_rate = 1.0 - health.success_rate

        if latencies:
            health.avg_latency_ms = statistics.mean(latencies)
            sorted_lat = sorted(latencies)
            p95_idx = int(len(sorted_lat) * 0.95)
            health.p95_latency_ms = sorted_lat[min(p95_idx, len(sorted_lat) - 1)]
        else:
            health.avg_latency_ms = 0.0
            health.p95_latency_ms = 0.0

        # Track consecutive failures
        if samples:
            last = samples[-1]
            if last.success:
                health.consecutive_failures = 0
                health.last_success = last.timestamp
            else:
                health.consecutive_failures += 1
                health.last_failure = last.timestamp

        # Classify health
        new_status = self._classify_health(health)
        health.status = new_status
        health.last_updated = time.time()

        # Adjust weight
        if new_status == HealthStatus.HEALTHY:
            health.routing_weight = self.WEIGHT_HEALTHY
        elif new_status == HealthStatus.DEGRADED:
            health.routing_weight = self.WEIGHT_DEGRADED
        elif new_status == HealthStatus.UNHEALTHY:
            health.routing_weight = self.WEIGHT_UNHEALTHY

        # Check for status change
        if new_status != old_status:
            adj = HealthAdjustment(
                adjustment_id=f"health_{provider}_{int(time.time())}",
                provider=provider,
                old_status=old_status,
                new_status=new_status,
                old_weight=old_weight,
                new_weight=health.routing_weight,
                reason=self._build_reason(health, old_status, new_status),
            )
            self._adjustments.append(adj)

            logger.info(
                "Provider %s health: %s -> %s (weight: %.2f -> %.2f)",
                provider,
                old_status.value,
                new_status.value,
                old_weight,
                health.routing_weight,
            )

            for cb in self._callbacks:
                try:
                    cb(adj)
                except Exception as e:
                    logger.debug("Health callback error: %s", e)

            return adj

        return None

    def _classify_health(self, health: ProviderHealth) -> HealthStatus:
        """Classify provider health status."""
        if health.consecutive_failures >= self.CONSECUTIVE_FAILURES_UNHEALTHY:
            return HealthStatus.UNHEALTHY

        if health.success_rate < self._sr_unhealthy:
            return HealthStatus.UNHEALTHY

        if health.avg_latency_ms > self._latency_unhealthy:
            return HealthStatus.UNHEALTHY

        if (
            health.success_rate < self._sr_degraded
            or health.avg_latency_ms > self._latency_degraded
        ):
            return HealthStatus.DEGRADED

        if health.total_samples < 3:
            return HealthStatus.UNKNOWN

        return HealthStatus.HEALTHY

    def _build_reason(
        self,
        health: ProviderHealth,
        old: HealthStatus,
        new: HealthStatus,
    ) -> str:
        parts = []
        if health.success_rate < self._sr_degraded:
            parts.append(f"success_rate={health.success_rate:.2f}")
        if health.avg_latency_ms > self._latency_degraded:
            parts.append(f"avg_latency={health.avg_latency_ms:.0f}ms")
        if health.consecutive_failures > 0:
            parts.append(f"consecutive_failures={health.consecutive_failures}")
        return f"{old.value} -> {new.value}: " + ", ".join(parts) if parts else f"{old.value} -> {new.value}"

    # ── Reporting ───────────────────────────────────────────────────

    def get_health_report(self) -> Dict[str, Any]:
        with self._lock:
            providers = {}
            for p, h in self._health.items():
                providers[p] = {
                    "status": h.status.value,
                    "routing_weight": round(h.routing_weight, 2),
                    "avg_latency_ms": round(h.avg_latency_ms, 1),
                    "p95_latency_ms": round(h.p95_latency_ms, 1),
                    "success_rate": round(h.success_rate, 3),
                    "total_samples": h.total_samples,
                    "consecutive_failures": h.consecutive_failures,
                }

            return {
                "providers": providers,
                "total_adjustments": len(self._adjustments),
                "recent_adjustments": [
                    {
                        "id": a.adjustment_id,
                        "provider": a.provider,
                        "transition": f"{a.old_status.value} -> {a.new_status.value}",
                        "reason": a.reason,
                    }
                    for a in list(self._adjustments)[-10:]
                ],
            }
