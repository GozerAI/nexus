"""
Provider Cost Comparison with Automatic Switching (Item 960)

Tracks per-provider cost metrics and automatically routes requests
to the most cost-effective provider that meets quality thresholds.
"""

import logging
import time
import statistics
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CostSample:
    """A single cost observation for a provider+model."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    quality_score: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ProviderCostProfile:
    """Aggregated cost profile for a provider."""

    provider: str
    avg_cost_per_1k_tokens: float = 0.0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    total_spend: float = 0.0
    total_tokens: int = 0
    sample_count: int = 0
    cost_efficiency: float = 0.0  # quality / cost
    last_updated: float = 0.0
    models: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SwitchEvent:
    """Record of a provider switch decision."""

    event_id: str
    from_provider: str
    to_provider: str
    reason: str
    cost_savings_estimate: float
    quality_impact: float
    timestamp: float = field(default_factory=time.time)


class ProviderCostSwitcher:
    """
    Tracks provider costs and automatically switches to the most
    cost-effective provider that meets quality thresholds.

    The switcher:
    1. Collects cost+quality samples per provider+model
    2. Computes cost-efficiency scores (quality / cost)
    3. Selects the optimal provider for each request
    4. Generates switch events when the preferred provider changes
    5. Supports budget constraints and quality floors
    """

    MIN_SAMPLES = 5
    SWITCH_COOLDOWN = 60.0  # seconds

    def __init__(
        self,
        quality_floor: float = 0.6,
        max_latency_ms: float = 10000.0,
        budget_limit_usd: Optional[float] = None,
        min_samples: int = 5,
        switch_cooldown: float = 60.0,
        max_history: int = 10000,
    ):
        self._quality_floor = quality_floor
        self._max_latency = max_latency_ms
        self._budget_limit = budget_limit_usd
        self._min_samples = min_samples
        self._switch_cooldown = switch_cooldown

        self._samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self._profiles: Dict[str, ProviderCostProfile] = {}
        self._switch_events: deque = deque(maxlen=5000)
        self._current_preferred: Optional[str] = None
        self._last_switch: float = 0.0
        self._total_spend: float = 0.0
        self._lock = threading.RLock()

    # ── Sample Recording ────────────────────────────────────────────

    def record_sample(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        quality_score: float,
        latency_ms: float = 0.0,
    ) -> Optional[SwitchEvent]:
        """
        Record a cost sample and recalculate provider rankings.
        Returns a SwitchEvent if the preferred provider changed.
        """
        sample = CostSample(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            quality_score=max(0.0, min(1.0, quality_score)),
            latency_ms=latency_ms,
        )

        with self._lock:
            self._samples[provider].append(sample)
            self._total_spend += cost_usd
            self._recalculate_profile(provider)

        return self._check_switch()

    # ── Provider Selection ──────────────────────────────────────────

    def select_provider(
        self,
        candidates: Optional[List[str]] = None,
        min_quality: Optional[float] = None,
        max_cost_per_1k: Optional[float] = None,
    ) -> Optional[str]:
        """
        Select the most cost-effective provider that meets constraints.
        """
        quality_threshold = min_quality or self._quality_floor
        profiles = candidates or list(self._profiles.keys())

        with self._lock:
            eligible = []
            for p in profiles:
                profile = self._profiles.get(p)
                if profile is None:
                    continue
                if profile.sample_count < self._min_samples:
                    continue
                if profile.avg_quality < quality_threshold:
                    continue
                if profile.avg_latency_ms > self._max_latency:
                    continue
                if max_cost_per_1k and profile.avg_cost_per_1k_tokens > max_cost_per_1k:
                    continue
                eligible.append(profile)

            if not eligible:
                # Return any provider with enough samples
                with_samples = [
                    self._profiles[p]
                    for p in profiles
                    if p in self._profiles and self._profiles[p].sample_count >= self._min_samples
                ]
                if with_samples:
                    return max(with_samples, key=lambda p: p.cost_efficiency).provider
                return candidates[0] if candidates else None

            # Sort by cost efficiency (higher = better)
            eligible.sort(key=lambda p: p.cost_efficiency, reverse=True)
            return eligible[0].provider

    def get_cost_comparison(self) -> List[Dict[str, Any]]:
        """Get a cost comparison across all providers."""
        with self._lock:
            comparison = []
            for p, profile in sorted(
                self._profiles.items(),
                key=lambda x: x[1].cost_efficiency,
                reverse=True,
            ):
                comparison.append(
                    {
                        "provider": p,
                        "avg_cost_per_1k_tokens": round(profile.avg_cost_per_1k_tokens, 6),
                        "avg_quality": round(profile.avg_quality, 3),
                        "avg_latency_ms": round(profile.avg_latency_ms, 1),
                        "cost_efficiency": round(profile.cost_efficiency, 3),
                        "total_spend": round(profile.total_spend, 4),
                        "total_tokens": profile.total_tokens,
                        "sample_count": profile.sample_count,
                        "models": dict(profile.models),
                    }
                )
            return comparison

    @property
    def preferred_provider(self) -> Optional[str]:
        return self._current_preferred

    @property
    def total_spend(self) -> float:
        return self._total_spend

    @property
    def is_over_budget(self) -> bool:
        if self._budget_limit is None:
            return False
        return self._total_spend >= self._budget_limit

    # ── Internal ────────────────────────────────────────────────────

    def _recalculate_profile(self, provider: str) -> None:
        """Recalculate the cost profile for a provider."""
        samples = list(self._samples[provider])
        if not samples:
            return

        if provider not in self._profiles:
            self._profiles[provider] = ProviderCostProfile(provider=provider)

        profile = self._profiles[provider]
        profile.sample_count = len(samples)
        profile.total_spend = sum(s.cost_usd for s in samples)
        profile.total_tokens = sum(s.input_tokens + s.output_tokens for s in samples)
        profile.avg_quality = statistics.mean(s.quality_score for s in samples)
        profile.avg_latency_ms = statistics.mean(s.latency_ms for s in samples)

        if profile.total_tokens > 0:
            profile.avg_cost_per_1k_tokens = (
                profile.total_spend / profile.total_tokens * 1000
            )
        else:
            profile.avg_cost_per_1k_tokens = 0.0

        # Cost efficiency = quality per dollar (normalized)
        if profile.avg_cost_per_1k_tokens > 0:
            profile.cost_efficiency = profile.avg_quality / profile.avg_cost_per_1k_tokens
        else:
            profile.cost_efficiency = profile.avg_quality * 1000.0  # free is very efficient

        # Per-model breakdown
        model_data: Dict[str, List[CostSample]] = defaultdict(list)
        for s in samples:
            model_data[s.model].append(s)

        profile.models = {}
        for model, model_samples in model_data.items():
            total_cost = sum(s.cost_usd for s in model_samples)
            total_toks = sum(s.input_tokens + s.output_tokens for s in model_samples)
            profile.models[model] = {
                "avg_cost_per_1k": total_cost / total_toks * 1000 if total_toks > 0 else 0.0,
                "avg_quality": statistics.mean(s.quality_score for s in model_samples),
                "samples": len(model_samples),
            }

        profile.last_updated = time.time()

    def _check_switch(self) -> Optional[SwitchEvent]:
        """Check if we should switch the preferred provider."""
        now = time.time()
        if now - self._last_switch < self._switch_cooldown:
            return None

        new_preferred = self.select_provider()
        if new_preferred is None:
            return None

        if new_preferred == self._current_preferred:
            return None

        old = self._current_preferred or "none"
        old_profile = self._profiles.get(old)
        new_profile = self._profiles.get(new_preferred)

        cost_savings = 0.0
        quality_impact = 0.0
        if old_profile and new_profile:
            cost_savings = old_profile.avg_cost_per_1k_tokens - new_profile.avg_cost_per_1k_tokens
            quality_impact = new_profile.avg_quality - old_profile.avg_quality

        event = SwitchEvent(
            event_id=f"switch_{int(now)}",
            from_provider=old,
            to_provider=new_preferred,
            reason=self._build_switch_reason(old, new_preferred),
            cost_savings_estimate=cost_savings,
            quality_impact=quality_impact,
        )

        self._current_preferred = new_preferred
        self._last_switch = now
        self._switch_events.append(event)

        logger.info(
            "Provider switch: %s -> %s (savings: $%.4f/1k, quality: %+.3f)",
            old,
            new_preferred,
            cost_savings,
            quality_impact,
        )
        return event

    def _build_switch_reason(self, old: str, new: str) -> str:
        old_p = self._profiles.get(old)
        new_p = self._profiles.get(new)

        parts = []
        if new_p:
            parts.append(f"{new} efficiency={new_p.cost_efficiency:.2f}")
        if old_p:
            parts.append(f"{old} efficiency={old_p.cost_efficiency:.2f}")
        if self._budget_limit and self._total_spend > self._budget_limit * 0.8:
            parts.append("approaching budget limit")

        return "; ".join(parts) if parts else "better cost efficiency"

    # ── Reporting ───────────────────────────────────────────────────

    def get_report(self) -> Dict[str, Any]:
        return {
            "preferred_provider": self._current_preferred,
            "total_spend": round(self._total_spend, 4),
            "budget_limit": self._budget_limit,
            "is_over_budget": self.is_over_budget,
            "providers": self.get_cost_comparison(),
            "total_switch_events": len(self._switch_events),
            "recent_switches": [
                {
                    "id": e.event_id,
                    "from": e.from_provider,
                    "to": e.to_provider,
                    "reason": e.reason,
                    "savings": round(e.cost_savings_estimate, 4),
                    "quality_impact": round(e.quality_impact, 3),
                }
                for e in list(self._switch_events)[-10:]
            ],
        }
