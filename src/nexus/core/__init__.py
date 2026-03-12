"""
TheNexus - Multi-model ensemble inference system for superintelligent AI.

This package provides core functionality for orchestrating multiple AI models
and cognitive processing.
"""

from nexus.core.ensemble_core import (
    ensemble_inference,
    load_model_ensemble,
    rank_responses,
    score_response,
    ModelStub,
)
from nexus.core.core_engine import (
    CognitiveCore,
    SymbolicReasoner,
    HolographicMemory,
    ConceptualMapper,
)
from nexus.core.strategies import (
    WeightedVotingStrategy,
    CascadingStrategy,
    DynamicWeightStrategy,
    MajorityVotingStrategy,
    CostOptimizedStrategy,
    EnsembleResult,
    ModelPerformance,
)

__version__ = "0.2.0"
__all__ = [
    "ensemble_inference",
    "load_model_ensemble",
    "rank_responses",
    "score_response",
    "ModelStub",
    "CognitiveCore",
    "SymbolicReasoner",
    "HolographicMemory",
    "ConceptualMapper",
    "WeightedVotingStrategy",
    "CascadingStrategy",
    "DynamicWeightStrategy",
    "MajorityVotingStrategy",
    "CostOptimizedStrategy",
    "EnsembleResult",
    "ModelPerformance",
]
