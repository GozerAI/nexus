"""
Unified ensemble system integrating:
- TheNexus's 5 ensemble strategies
- combo1's response synthesis and confidence calibration
- fluffy-eureka's epistemic monitoring
"""

from nexus.providers.ensemble.core import UnifiedEnsemble
from nexus.providers.strategies.ensemble_strategies import (
    CascadingStrategy,
    CostOptimizedStrategy,
    DynamicWeightStrategy,
    MajorityVotingStrategy,
    SynthesizedStrategy,
    WeightedVotingStrategy,
)
from nexus.providers.ensemble.types import EnsembleRequest, EnsembleResponse, ModelResponse
from nexus.providers.ensemble.synthesis_advanced import (
    AdvancedResponseSynthesizer,
    SynthesisStrategy,
    SynthesizedResponse,
)
from nexus.providers.ensemble.calibration_advanced import (
    AdvancedCalibrator,
    CalibrationMetrics,
)

__all__ = [
    "UnifiedEnsemble",
    "EnsembleRequest",
    "EnsembleResponse",
    "ModelResponse",
    "WeightedVotingStrategy",
    "CascadingStrategy",
    "DynamicWeightStrategy",
    "MajorityVotingStrategy",
    "CostOptimizedStrategy",
    "SynthesizedStrategy",
    # Advanced synthesis (Phase 5)
    "AdvancedResponseSynthesizer",
    "SynthesisStrategy",
    "SynthesizedResponse",
    # Advanced calibration (Phase 5)
    "AdvancedCalibrator",
    "CalibrationMetrics",
]
