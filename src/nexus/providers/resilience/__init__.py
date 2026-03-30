"""
Resilience patterns for fault tolerance and graceful degradation.
"""

from nexus.providers.resilience.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerError,
    CircuitBreakerRegistry,
    CircuitState,
    get_circuit_breaker_registry,
)

# Enhanced circuit breaker with rolling window and failure rate tracking
try:
    from nexus.providers.resilience.circuit_breaker_enhanced import (
        CircuitBreaker as EnhancedCircuitBreaker,
        CircuitBreakerManager,
        CircuitConfig,
        CircuitState as EnhancedCircuitState,
        circuit_breaker_manager,
    )
    __all_enhanced__ = [
        "EnhancedCircuitBreaker",
        "CircuitBreakerManager",
        "CircuitConfig",
        "circuit_breaker_manager",
    ]
except ImportError:
    __all_enhanced__ = []

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitBreakerRegistry",
    "CircuitState",
    "get_circuit_breaker_registry",
] + __all_enhanced__
