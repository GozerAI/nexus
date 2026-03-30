"""
Autonomous Reasoning Strategy Improvement (Item 703)

Analyses reasoning chain outcomes, identifies strategy weaknesses, and
autonomously adjusts strategy selection weights, step configurations,
and fallback orderings to improve overall reasoning quality.
"""

import logging
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Types of reasoning strategies."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    DECOMPOSITION = "decomposition"
    ANALOGICAL = "analogical"
    ABDUCTIVE = "abductive"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    META_REASONING = "meta_reasoning"
    PATTERN_MATCHING = "pattern_matching"
    ENSEMBLE = "ensemble"


@dataclass
class StrategyProfile:
    """Performance profile for a reasoning strategy."""

    strategy: StrategyType
    weight: float = 1.0
    success_rate: float = 0.5
    avg_quality: float = 0.5
    avg_latency_ms: float = 0.0
    total_uses: int = 0
    task_affinity: Dict[str, float] = field(default_factory=dict)
    step_config: Dict[str, Any] = field(default_factory=dict)
    last_adjusted: float = 0.0


@dataclass
class ReasoningOutcome:
    """Outcome of a reasoning attempt."""

    strategy: StrategyType
    task_type: str
    quality_score: float
    latency_ms: float
    step_count: int
    success: bool
    issues: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class StrategyAdjustment:
    """Record of a strategy adjustment."""

    adjustment_id: str
    strategy: StrategyType
    dimension: str  # weight, step_config, fallback_order
    old_value: Any
    new_value: Any
    reason: str
    expected_improvement: float
    actual_improvement: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


class AutonomousReasoningImprover:
    """
    Autonomously improves reasoning strategies by analyzing outcomes
    and adjusting strategy weights, configurations, and selection logic.

    The improver:
    1. Tracks reasoning outcomes per strategy+task_type
    2. Identifies underperforming strategies
    3. Adjusts strategy weights for the selection layer
    4. Tunes step configurations (depth, branching, verification)
    5. Reorders fallback chains based on actual performance
    """

    ADJUSTMENT_COOLDOWN = 120.0  # seconds between adjustments
    WEIGHT_DECAY = 0.95  # decay factor for unused strategies
    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 5.0
    MIN_OBSERVATIONS = 5

    def __init__(
        self,
        adjustment_cooldown: float = 120.0,
        min_observations: int = 5,
        max_history: int = 10000,
    ):
        self._cooldown = adjustment_cooldown
        self._min_obs = max(2, min_observations)
        self._profiles: Dict[StrategyType, StrategyProfile] = {}
        self._outcomes: deque = deque(maxlen=max_history)
        self._adjustments: deque = deque(maxlen=5000)
        self._fallback_order: List[StrategyType] = list(StrategyType)
        self._last_global_adjustment = 0.0

        # Initialize profiles with default weights
        for st in StrategyType:
            self._profiles[st] = StrategyProfile(
                strategy=st,
                step_config=self._default_step_config(st),
            )

    # ── Public API ──────────────────────────────────────────────────

    def record_outcome(
        self,
        strategy: StrategyType,
        task_type: str,
        quality_score: float,
        latency_ms: float = 0.0,
        step_count: int = 0,
        success: bool = True,
        issues: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[StrategyAdjustment]:
        """
        Record a reasoning outcome and potentially trigger an adjustment.
        """
        outcome = ReasoningOutcome(
            strategy=strategy,
            task_type=task_type,
            quality_score=max(0.0, min(1.0, quality_score)),
            latency_ms=latency_ms,
            step_count=step_count,
            success=success,
            issues=issues or [],
            context=context or {},
        )
        self._outcomes.append(outcome)

        # Update strategy profile
        profile = self._profiles[strategy]
        profile.total_uses += 1
        profile.avg_quality = (
            profile.avg_quality * (profile.total_uses - 1) + quality_score
        ) / profile.total_uses
        profile.avg_latency_ms = (
            profile.avg_latency_ms * (profile.total_uses - 1) + latency_ms
        ) / profile.total_uses
        if success:
            profile.success_rate = (
                profile.success_rate * (profile.total_uses - 1) + 1.0
            ) / profile.total_uses
        else:
            profile.success_rate = (
                profile.success_rate * (profile.total_uses - 1)
            ) / profile.total_uses

        # Update task affinity
        if task_type not in profile.task_affinity:
            profile.task_affinity[task_type] = quality_score
        else:
            # Exponential moving average
            alpha = 0.3
            profile.task_affinity[task_type] = (
                alpha * quality_score + (1 - alpha) * profile.task_affinity[task_type]
            )

        # Check if adjustment is warranted
        return self._maybe_adjust(strategy, task_type)

    def select_strategy(
        self,
        task_type: str,
        available: Optional[List[StrategyType]] = None,
    ) -> StrategyType:
        """
        Select the best reasoning strategy for a given task type,
        using performance-weighted selection.
        """
        candidates = available or list(StrategyType)

        best_strategy = candidates[0]
        best_score = -1.0

        for st in candidates:
            profile = self._profiles[st]
            affinity = profile.task_affinity.get(task_type, 0.5)
            score = (
                profile.weight * 0.3
                + affinity * 0.35
                + profile.success_rate * 0.25
                + profile.avg_quality * 0.1
            )
            if score > best_score:
                best_score = score
                best_strategy = st

        return best_strategy

    def get_fallback_order(self, task_type: Optional[str] = None) -> List[StrategyType]:
        """
        Get the current fallback order, optionally scoped to a task type.
        """
        if task_type:
            # Sort by task affinity for that task type
            return sorted(
                self._fallback_order,
                key=lambda st: self._profiles[st].task_affinity.get(task_type, 0.0),
                reverse=True,
            )
        return list(self._fallback_order)

    def get_strategy_config(self, strategy: StrategyType) -> Dict[str, Any]:
        """Get the current step configuration for a strategy."""
        return dict(self._profiles[strategy].step_config)

    def get_improvement_report(self) -> Dict[str, Any]:
        """Generate a comprehensive improvement report."""
        strategy_reports = {}
        for st, profile in self._profiles.items():
            recent_outcomes = [
                o for o in self._outcomes if o.strategy == st
            ][-50:]

            strategy_reports[st.value] = {
                "weight": round(profile.weight, 3),
                "success_rate": round(profile.success_rate, 3),
                "avg_quality": round(profile.avg_quality, 3),
                "avg_latency_ms": round(profile.avg_latency_ms, 1),
                "total_uses": profile.total_uses,
                "task_affinity": {
                    k: round(v, 3) for k, v in profile.task_affinity.items()
                },
                "recent_trend": self._compute_trend(
                    [o.quality_score for o in recent_outcomes]
                ),
            }

        adjustments = list(self._adjustments)[-20:]
        return {
            "strategy_profiles": strategy_reports,
            "fallback_order": [st.value for st in self._fallback_order],
            "total_outcomes": len(self._outcomes),
            "total_adjustments": len(self._adjustments),
            "recent_adjustments": [
                {
                    "id": a.adjustment_id,
                    "strategy": a.strategy.value,
                    "dimension": a.dimension,
                    "reason": a.reason,
                    "expected_improvement": a.expected_improvement,
                }
                for a in adjustments
            ],
        }

    # ── Internal Adjustment Logic ───────────────────────────────────

    def _maybe_adjust(
        self, strategy: StrategyType, task_type: str
    ) -> Optional[StrategyAdjustment]:
        profile = self._profiles[strategy]
        now = time.time()

        if now - profile.last_adjusted < self._cooldown:
            return None
        if profile.total_uses < self._min_obs:
            return None

        # Check for weight adjustment
        adjustment = self._adjust_weight(profile, task_type)
        if adjustment:
            return adjustment

        # Check for step config adjustment
        adjustment = self._adjust_step_config(profile, task_type)
        if adjustment:
            return adjustment

        # Check for fallback reorder
        if now - self._last_global_adjustment > self._cooldown * 2:
            self._reorder_fallbacks()
            self._last_global_adjustment = now

        return None

    def _adjust_weight(
        self, profile: StrategyProfile, task_type: str
    ) -> Optional[StrategyAdjustment]:
        """Adjust strategy weight based on recent performance."""
        recent = [
            o
            for o in self._outcomes
            if o.strategy == profile.strategy
        ][-self._min_obs:]

        if len(recent) < self._min_obs:
            return None

        recent_quality = statistics.mean(o.quality_score for o in recent)
        recent_success = sum(1 for o in recent if o.success) / len(recent)

        # Compute desired weight delta
        performance = recent_quality * 0.6 + recent_success * 0.4
        old_weight = profile.weight

        if performance > 0.8:
            new_weight = min(self.MAX_WEIGHT, old_weight * 1.1)
        elif performance < 0.4:
            new_weight = max(self.MIN_WEIGHT, old_weight * 0.85)
        else:
            return None  # No adjustment needed

        if abs(new_weight - old_weight) < 0.01:
            return None

        profile.weight = new_weight
        profile.last_adjusted = time.time()

        adj = StrategyAdjustment(
            adjustment_id=f"weight_{profile.strategy.value}_{int(time.time())}",
            strategy=profile.strategy,
            dimension="weight",
            old_value=round(old_weight, 3),
            new_value=round(new_weight, 3),
            reason=f"Performance {performance:.2f} triggered weight {'increase' if new_weight > old_weight else 'decrease'}",
            expected_improvement=abs(new_weight - old_weight) * 0.1,
        )
        self._adjustments.append(adj)
        logger.info(
            "Adjusted %s weight: %.3f -> %.3f (perf=%.2f)",
            profile.strategy.value,
            old_weight,
            new_weight,
            performance,
        )
        return adj

    def _adjust_step_config(
        self, profile: StrategyProfile, task_type: str
    ) -> Optional[StrategyAdjustment]:
        """Adjust step configuration based on outcome patterns."""
        recent = [
            o
            for o in self._outcomes
            if o.strategy == profile.strategy and o.task_type == task_type
        ][-20:]

        if len(recent) < self._min_obs:
            return None

        # Check for too-shallow reasoning
        avg_steps = statistics.mean(o.step_count for o in recent) if recent else 0
        avg_quality = statistics.mean(o.quality_score for o in recent)

        old_config = dict(profile.step_config)
        new_config = dict(old_config)
        reason = ""

        if avg_quality < 0.5 and avg_steps < 3:
            # Increase depth
            new_config["max_depth"] = min(
                old_config.get("max_depth", 5) + 1, 10
            )
            reason = f"Low quality ({avg_quality:.2f}) with shallow reasoning ({avg_steps:.1f} steps)"
        elif avg_quality > 0.85 and avg_steps > 6:
            # Can reduce depth for efficiency
            new_config["max_depth"] = max(
                old_config.get("max_depth", 5) - 1, 2
            )
            reason = f"High quality ({avg_quality:.2f}) allows reducing depth"
        else:
            return None

        if new_config == old_config:
            return None

        profile.step_config = new_config
        profile.last_adjusted = time.time()

        adj = StrategyAdjustment(
            adjustment_id=f"config_{profile.strategy.value}_{int(time.time())}",
            strategy=profile.strategy,
            dimension="step_config",
            old_value=old_config,
            new_value=new_config,
            reason=reason,
            expected_improvement=0.05,
        )
        self._adjustments.append(adj)
        logger.info(
            "Adjusted %s step_config: %s (reason: %s)",
            profile.strategy.value,
            new_config,
            reason,
        )
        return adj

    def _reorder_fallbacks(self) -> None:
        """Reorder the fallback chain based on overall performance."""
        self._fallback_order = sorted(
            list(StrategyType),
            key=lambda st: (
                self._profiles[st].success_rate * 0.4
                + self._profiles[st].avg_quality * 0.4
                + self._profiles[st].weight * 0.2
            ),
            reverse=True,
        )
        logger.debug(
            "Reordered fallbacks: %s",
            [st.value for st in self._fallback_order],
        )

    def _default_step_config(self, strategy: StrategyType) -> Dict[str, Any]:
        """Return default step configuration for a strategy type."""
        defaults = {
            StrategyType.CHAIN_OF_THOUGHT: {
                "max_depth": 5,
                "verification_enabled": True,
                "branching_factor": 1,
            },
            StrategyType.DECOMPOSITION: {
                "max_depth": 4,
                "max_sub_problems": 5,
                "merge_strategy": "weighted",
            },
            StrategyType.ANALOGICAL: {
                "max_depth": 3,
                "analogy_pool_size": 10,
                "similarity_threshold": 0.6,
            },
            StrategyType.META_REASONING: {
                "max_depth": 3,
                "self_reflection_steps": 2,
                "confidence_threshold": 0.7,
            },
            StrategyType.ENSEMBLE: {
                "max_depth": 4,
                "num_strategies": 3,
                "voting_method": "weighted",
            },
        }
        return defaults.get(strategy, {"max_depth": 4, "verification_enabled": True})

    @staticmethod
    def _compute_trend(scores: List[float]) -> float:
        if len(scores) < 2:
            return 0.0
        n = len(scores)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(scores)
        sum_xy = sum(x[i] * scores[i] for i in range(n))
        sum_xx = sum(xi * xi for xi in x)
        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom
