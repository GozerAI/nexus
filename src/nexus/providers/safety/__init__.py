"""
Safety and governance system for the Unified Intelligence System.

This module provides production-grade safety mechanisms including:
- Model quarantine: Automatically isolate misbehaving models
- Circuit breakers: Prevent cascading failures
- Rate limiting: Protect against overload
- Bias mitigation: Detect and prevent cognitive and systemic biases
- Virtue assessment: Ethical AI development with virtue-based evaluation
- Policy enforcement: Ensure safe operations
- Audit logging: Track all safety events

Based on psychic-bassoon's governance framework with enhancements from nexus-system.
"""

from nexus.providers.safety.quarantine import (
    ModelQuarantine,
    QuarantineReason,
    QuarantineStatus,
)
from nexus.providers.safety.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
)
from nexus.providers.safety.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
)
from nexus.providers.safety.bias_mitigation import (
    BiasMitigationSystem,
    BiasType,
    PerspectiveSource,
    VirtueCategory,
    PerspectiveInput,
    BiasDetectionResult,
    VirtueAssessment,
)
from nexus.providers.safety.production_safety import (
    ProductionSafetySystem,
    EthicsAssessment,
    SafetyMonitoringResult,
)
from nexus.providers.safety.safety_integration import (
    UnifiedSafetyPipeline,
    UnifiedSafetyResult,
    SafetyDecision,
)
from nexus.providers.safety.bias_mitigation_advanced import (
    AdvancedBiasMitigator,
    BiasDetection,
    BiasSeverity,
    VirtueAssessment as AdvancedVirtueAssessment,
    VirtueType,
)

__all__ = [
    # Quarantine
    "ModelQuarantine",
    "QuarantineReason",
    "QuarantineStatus",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    # Rate limiting
    "RateLimiter",
    "RateLimitExceeded",
    # Bias mitigation
    "BiasMitigationSystem",
    "BiasType",
    "PerspectiveSource",
    "VirtueCategory",
    "PerspectiveInput",
    "BiasDetectionResult",
    "VirtueAssessment",
    # Production safety
    "ProductionSafetySystem",
    "EthicsAssessment",
    "SafetyMonitoringResult",
    # Unified safety pipeline
    "UnifiedSafetyPipeline",
    "UnifiedSafetyResult",
    "SafetyDecision",
    # Advanced bias mitigation
    "AdvancedBiasMitigator",
    "BiasDetection",
    "BiasSeverity",
    "AdvancedVirtueAssessment",
    "VirtueType",
]
