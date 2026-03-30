"""
Adaptive Strategy Selector

Automatically selects the best ensemble strategy based on:
- Request characteristics (complexity, latency requirements, budget)
- Historical performance data
- Real-time metrics
- Cost constraints

This enables intelligent, context-aware ensemble execution that optimizes
for the right balance of quality, cost, and latency.

Part of Phase 4: Advanced Features from TheNexus integration roadmap.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class RequestComplexity(str, Enum):
    """Request complexity classification."""
    TRIVIAL = "trivial"  # Simple factual questions
    SIMPLE = "simple"  # Basic reasoning
    MODERATE = "moderate"  # Multi-step reasoning
    COMPLEX = "complex"  # Advanced reasoning, creativity
    EXPERT = "expert"  # Domain expertise required


class LatencyRequirement(str, Enum):
    """Latency sensitivity classification."""
    REALTIME = "realtime"  # < 1s (interactive)
    FAST = "fast"  # < 3s (user-facing)
    NORMAL = "normal"  # < 10s (background)
    BATCH = "batch"  # No limit (offline)


@dataclass
class RequestProfile:
    """
    Profile of an ensemble request for strategy selection.

    Attributes:
        prompt: Input prompt text
        complexity: Estimated complexity level
        latency_requirement: Required response time
        budget_constraint: Maximum cost in USD (None = no limit)
        requires_consensus: Whether consensus is needed
        require_explanation: Whether explanation is needed
        model_count: Number of models to query
        metadata: Additional metadata
    """
    prompt: str
    complexity: RequestComplexity = RequestComplexity.MODERATE
    latency_requirement: LatencyRequirement = LatencyRequirement.NORMAL
    budget_constraint: Optional[float] = None
    requires_consensus: bool = False
    require_explanation: bool = False
    model_count: int = 3
    metadata: Dict[str, Any] = None


@dataclass
class StrategyPerformance:
    """Historical performance metrics for a strategy."""
    strategy_name: str
    request_count: int = 0
    avg_latency: float = 0.0
    avg_cost: float = 0.0
    avg_quality_score: float = 0.0
    success_rate: float = 1.0
    last_used: Optional[datetime] = None


class AdaptiveStrategySelector:
    """
    Intelligently selects ensemble strategy based on request characteristics.

    Example:
        >>> selector = AdaptiveStrategySelector()
        >>>
        >>> # Classify and select strategy
        >>> profile = RequestProfile(
        ...     prompt="What is the capital of France?",
        ...     complexity=RequestComplexity.TRIVIAL,
        ...     latency_requirement=LatencyRequirement.FAST
        ... )
        >>> strategy = selector.select_strategy(profile)
        >>> print(strategy)  # "cascading" (fast, cheap for simple queries)
        >>>
        >>> # Complex query with budget
        >>> profile = RequestProfile(
        ...     prompt="Analyze the economic implications...",
        ...     complexity=RequestComplexity.EXPERT,
        ...     budget_constraint=0.10
        ... )
        >>> strategy = selector.select_strategy(profile)
        >>> print(strategy)  # "cost_optimized" (respects budget)
    """

    def __init__(self):
        """Initialize adaptive strategy selector."""
        self.performance_history: Dict[str, StrategyPerformance] = {}
        self._initialize_default_performance()

        logger.info("🎯 AdaptiveStrategySelector initialized")

    def _initialize_default_performance(self):
        """Initialize default performance metrics for strategies."""
        strategies = [
            "weighted_voting",
            "cascading",
            "dynamic_weights",
            "majority_voting",
            "cost_optimized",
            "synthesized"
        ]

        for strategy in strategies:
            self.performance_history[strategy] = StrategyPerformance(
                strategy_name=strategy,
                avg_quality_score=0.75,  # Default baseline
                avg_latency=5.0,  # Default 5s
                avg_cost=0.05,  # Default $0.05
                success_rate=0.95  # Default 95%
            )

    def select_strategy(self, profile: RequestProfile) -> str:
        """
        Select best strategy for request profile.

        Args:
            profile: Request profile with characteristics

        Returns:
            Strategy name to use
        """
        # Rule-based selection with priority order

        # 1. Budget constraint (highest priority)
        if profile.budget_constraint is not None:
            logger.info(
                f"🎯 Selected cost_optimized (budget=${profile.budget_constraint})"
            )
            return "cost_optimized"

        # 2. Consensus requirement
        if profile.requires_consensus:
            logger.info("🎯 Selected majority_voting (consensus required)")
            return "majority_voting"

        # 3. Latency requirement
        if profile.latency_requirement == LatencyRequirement.REALTIME:
            # Use cascading for realtime (try cheap models first)
            logger.info("🎯 Selected cascading (realtime latency required)")
            return "cascading"

        # 4. Complexity-based selection
        if profile.complexity == RequestComplexity.TRIVIAL:
            # Simple queries don't need full ensemble
            logger.info("🎯 Selected cascading (trivial complexity)")
            return "cascading"

        elif profile.complexity == RequestComplexity.SIMPLE:
            # Use weighted voting for simple queries
            logger.info("🎯 Selected weighted_voting (simple complexity)")
            return "weighted_voting"

        elif profile.complexity in [RequestComplexity.COMPLEX, RequestComplexity.EXPERT]:
            # Complex queries benefit from dynamic learning
            logger.info(f"🎯 Selected dynamic_weights ({profile.complexity} complexity)")
            return "dynamic_weights"

        # 5. Default: weighted voting (good balance)
        logger.info("🎯 Selected weighted_voting (default)")
        return "weighted_voting"

    def select_with_learning(self, profile: RequestProfile) -> str:
        """
        Select strategy using historical performance data.

        This version considers past performance to make better decisions.

        Args:
            profile: Request profile

        Returns:
            Strategy name
        """
        # Apply rule-based filters first
        candidate_strategies = self._filter_by_constraints(profile)

        if not candidate_strategies:
            # Fallback to rule-based
            return self.select_strategy(profile)

        # Score each candidate strategy
        scores = {}
        for strategy in candidate_strategies:
            score = self._score_strategy(strategy, profile)
            scores[strategy] = score

        # Select best strategy
        best_strategy = max(scores, key=scores.get)

        logger.info(
            f"🎯 Selected {best_strategy} with learning "
            f"(score={scores[best_strategy]:.3f})"
        )

        return best_strategy

    def _filter_by_constraints(self, profile: RequestProfile) -> List[str]:
        """Filter strategies that meet hard constraints."""
        candidates = list(self.performance_history.keys())

        # Filter by budget
        if profile.budget_constraint is not None:
            candidates = [
                s for s in candidates
                if self.performance_history[s].avg_cost <= profile.budget_constraint
            ]

        # Filter by latency
        if profile.latency_requirement == LatencyRequirement.REALTIME:
            candidates = [
                s for s in candidates
                if self.performance_history[s].avg_latency < 1.0
            ]
        elif profile.latency_requirement == LatencyRequirement.FAST:
            candidates = [
                s for s in candidates
                if self.performance_history[s].avg_latency < 3.0
            ]

        return candidates

    def _score_strategy(self, strategy: str, profile: RequestProfile) -> float:
        """
        Score a strategy for given profile.

        Combines quality, cost, and latency into single score.

        Args:
            strategy: Strategy name
            profile: Request profile

        Returns:
            Score (higher is better)
        """
        perf = self.performance_history[strategy]

        # Quality weight (most important)
        quality_score = perf.avg_quality_score * 0.5

        # Cost efficiency (normalized, inverted - lower cost is better)
        cost_score = (1.0 - min(perf.avg_cost / 0.20, 1.0)) * 0.3

        # Latency efficiency (normalized, inverted - lower latency is better)
        latency_score = (1.0 - min(perf.avg_latency / 10.0, 1.0)) * 0.2

        # Success rate bonus
        success_bonus = perf.success_rate * 0.1

        total_score = quality_score + cost_score + latency_score + success_bonus

        return total_score

    def update_performance(
        self,
        strategy: str,
        latency: float,
        cost: float,
        quality_score: float,
        success: bool = True
    ):
        """
        Update performance metrics after strategy execution.

        Args:
            strategy: Strategy name
            latency: Request latency in seconds
            cost: Request cost in USD
            quality_score: Quality score (0.0-1.0)
            success: Whether request succeeded
        """
        if strategy not in self.performance_history:
            self.performance_history[strategy] = StrategyPerformance(
                strategy_name=strategy
            )

        perf = self.performance_history[strategy]

        # Update running averages (exponential moving average)
        alpha = 0.1  # Learning rate

        perf.avg_latency = (1 - alpha) * perf.avg_latency + alpha * latency
        perf.avg_cost = (1 - alpha) * perf.avg_cost + alpha * cost
        perf.avg_quality_score = (1 - alpha) * perf.avg_quality_score + alpha * quality_score

        # Update success rate
        perf.success_rate = (1 - alpha) * perf.success_rate + alpha * (1.0 if success else 0.0)

        # Update counters
        perf.request_count += 1
        perf.last_used = datetime.now(timezone.utc)

        logger.debug(
            f"📊 Updated {strategy} performance: "
            f"quality={perf.avg_quality_score:.3f}, "
            f"latency={perf.avg_latency:.2f}s, "
            f"cost=${perf.avg_cost:.4f}"
        )

    def get_performance_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance report for all strategies.

        Returns:
            Dictionary mapping strategy names to performance metrics
        """
        report = {}
        for strategy, perf in self.performance_history.items():
            report[strategy] = {
                "request_count": perf.request_count,
                "avg_latency": round(perf.avg_latency, 3),
                "avg_cost": round(perf.avg_cost, 4),
                "avg_quality": round(perf.avg_quality_score, 3),
                "success_rate": round(perf.success_rate, 3),
                "last_used": perf.last_used.isoformat() if perf.last_used else None
            }
        return report


def classify_complexity(prompt: str) -> RequestComplexity:
    """
    Automatically classify prompt complexity.

    Uses heuristics like length, keywords, and structure.

    Args:
        prompt: Input prompt

    Returns:
        Complexity classification
    """
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())

    # Trivial: Very short, factual questions
    if word_count < 10 and any(
        keyword in prompt_lower
        for keyword in ["what is", "who is", "when is", "where is", "define"]
    ):
        return RequestComplexity.TRIVIAL

    # Simple: Short queries, basic info
    if word_count < 25:
        return RequestComplexity.SIMPLE

    # Expert: Domain-specific keywords
    expert_keywords = [
        "analyze", "evaluate", "compare", "synthesize",
        "implications", "strategic", "optimize", "implement"
    ]
    if any(keyword in prompt_lower for keyword in expert_keywords):
        return RequestComplexity.EXPERT

    # Complex: Long, detailed queries
    if word_count > 100:
        return RequestComplexity.COMPLEX

    # Moderate: Default
    return RequestComplexity.MODERATE


def estimate_latency_requirement(
    context: Dict[str, Any]
) -> LatencyRequirement:
    """
    Estimate latency requirement from request context.

    Args:
        context: Request context with metadata

    Returns:
        Latency requirement classification
    """
    # Check explicit requirements
    if context.get("realtime"):
        return LatencyRequirement.REALTIME

    if context.get("interactive"):
        return LatencyRequirement.FAST

    if context.get("batch"):
        return LatencyRequirement.BATCH

    # Default to normal
    return LatencyRequirement.NORMAL
