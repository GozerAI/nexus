"""Core Nexus exports with lazy loading.

Avoid importing heavyweight ensemble modules at package import time so direct
LLM paths can stay lightweight.
"""

from importlib import import_module

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

_EXPORTS = {
    "ensemble_inference": ("nexus.core.ensemble_core", "ensemble_inference"),
    "load_model_ensemble": ("nexus.core.ensemble_core", "load_model_ensemble"),
    "rank_responses": ("nexus.core.ensemble_core", "rank_responses"),
    "score_response": ("nexus.core.ensemble_core", "score_response"),
    "ModelStub": ("nexus.core.ensemble_core", "ModelStub"),
    "CognitiveCore": ("nexus.core.core_engine", "CognitiveCore"),
    "SymbolicReasoner": ("nexus.core.core_engine", "SymbolicReasoner"),
    "HolographicMemory": ("nexus.core.core_engine", "HolographicMemory"),
    "ConceptualMapper": ("nexus.core.core_engine", "ConceptualMapper"),
    "WeightedVotingStrategy": ("nexus.core.strategies", "WeightedVotingStrategy"),
    "CascadingStrategy": ("nexus.core.strategies", "CascadingStrategy"),
    "DynamicWeightStrategy": ("nexus.core.strategies", "DynamicWeightStrategy"),
    "MajorityVotingStrategy": ("nexus.core.strategies", "MajorityVotingStrategy"),
    "CostOptimizedStrategy": ("nexus.core.strategies", "CostOptimizedStrategy"),
    "EnsembleResult": ("nexus.core.strategies", "EnsembleResult"),
    "ModelPerformance": ("nexus.core.strategies", "ModelPerformance"),
}


def __getattr__(name: str):
    """Resolve public core exports on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
