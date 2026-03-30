"""
Ensemble Strategies Module for Unified Intelligence

Provides advanced strategies for combining model outputs:
- Weighted voting with configurable model weights
- Cascading inference (try cheap models first, escalate if needed)
- Dynamic weight adjustment based on performance history
- Majority voting for classification tasks
- Cost-optimized selection balancing quality and cost

Components:
- **EnsembleStrategy**: Base class for all strategies
- **WeightedVotingStrategy**: Combines weights with quality scores
- **CascadingStrategy**: Escalates from cheap to expensive models
- **DynamicWeightStrategy**: Learns from historical performance
- **MajorityVotingStrategy**: Selects most common response
- **CostOptimizedStrategy**: Optimizes quality/cost ratio
- **ModelPerformance**: Tracks historical model performance
- **EnsembleResult**: Contains ensemble inference results

Example:
    >>> from unified_intelligence.strategies import (
    ...     WeightedVotingStrategy, CascadingStrategy, EnsembleResult
    ... )
    >>>
    >>> # Weighted voting
    >>> strategy = WeightedVotingStrategy(
    ...     weights={"gpt-4": 1.5, "claude-3": 1.3, "llama": 1.0}
    ... )
    >>> result = await strategy.select_response(responses, prompt)
    >>> print(f"Selected: {result.model_name} (score: {result.score:.2f})")
    >>>
    >>> # Cascading (try cheap first)
    >>> cascade = CascadingStrategy(
    ...     confidence_threshold=0.7,
    ...     max_cascades=3
    ... )
    >>> result = await cascade.select_response(responses, prompt)
    >>> print(f"Models tried: {result.models_queried}, Cost: ${result.total_cost:.4f}")

Adapted from: TheNexus/src/thenexus/strategies.py
"""

from nexus.providers.strategies.ensemble_strategies import (
    EnsembleStrategy,
    ModelPerformance,
    EnsembleResult,
    WeightedVotingStrategy,
    CascadingStrategy,
    DynamicWeightStrategy,
    MajorityVotingStrategy,
    CostOptimizedStrategy,
)

__all__ = [
    # Base
    "EnsembleStrategy",
    "ModelPerformance",
    "EnsembleResult",
    # Strategies
    "WeightedVotingStrategy",
    "CascadingStrategy",
    "DynamicWeightStrategy",
    "MajorityVotingStrategy",
    "CostOptimizedStrategy",
]
