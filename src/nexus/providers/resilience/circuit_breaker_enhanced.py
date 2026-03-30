"""Enhanced circuit breaker pattern for model resilience.

Implements automatic failure detection and model quarantine to prevent
cascading failures and wasted resources on repeatedly calling broken models.

Features:
- Automatic failure rate tracking per model (rolling window)
- Circuit breaker states: CLOSED, OPEN, HALF_OPEN
- Auto-quarantine when failure threshold exceeded
- Automatic recovery after cooldown period
- Configurable thresholds and windows

Integrated from combo1 to provide enhanced resilience capabilities.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing, reject requests
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: float = 0.5  # 50% failure rate triggers open
    min_requests: int = 5            # Minimum requests before evaluating
    window_size: int = 20            # Rolling window size for tracking
    cooldown_seconds: float = 60.0   # Time before trying half-open
    half_open_max_requests: int = 3  # Max requests in half-open state
    auto_quarantine: bool = True     # Auto-quarantine on circuit open


class CircuitBreaker:
    """Circuit breaker for a single model.

    Tracks failure rates and automatically opens circuit when
    failures exceed threshold, preventing wasted calls to broken models.
    """

    def __init__(self, model_name: str, config: Optional[CircuitConfig] = None):
        """Initialize circuit breaker.

        Args:
            model_name: Name of the model this circuit protects
            config: Circuit breaker configuration
        """
        self.model_name = model_name
        self.config = config or CircuitConfig()
        self.state = CircuitState.CLOSED

        # Track recent requests (True = success, False = failure)
        self._recent_requests: deque[bool] = deque(maxlen=self.config.window_size)
        self._lock = threading.Lock()

        # State management
        self._opened_at: Optional[float] = None
        self._half_open_requests = 0

        logger.info(
            f"Circuit breaker initialized for model={model_name}, "
            f"failure_threshold={self.config.failure_threshold}"
        )

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            self._recent_requests.append(True)

            # If in half-open, successful requests close the circuit
            if self.state == CircuitState.HALF_OPEN:
                self._half_open_requests += 1

                if self._half_open_requests >= self.config.half_open_max_requests:
                    logger.info(
                        f"Circuit recovered for model={self.model_name}, "
                        f"half_open_successes={self._half_open_requests}"
                    )
                    self._transition_to_closed()

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._recent_requests.append(False)

            # Check if we should open the circuit
            if self.state == CircuitState.CLOSED:
                self._check_failure_threshold()

            # If in half-open and we get a failure, reopen circuit
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit half-open failed for model={self.model_name}")
                self._transition_to_open()

    def can_attempt(self) -> bool:
        """Check if a request can be attempted.

        Returns:
            True if request should be allowed, False otherwise
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            elif self.state == CircuitState.OPEN:
                # Check if cooldown period has passed
                if self._should_attempt_half_open():
                    self._transition_to_half_open()
                    return True
                return False

            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open
                return self._half_open_requests < self.config.half_open_max_requests

            return False

    def _check_failure_threshold(self) -> None:
        """Check if failure rate exceeds threshold."""
        if len(self._recent_requests) < self.config.min_requests:
            return  # Not enough data

        failures = sum(1 for result in self._recent_requests if not result)
        failure_rate = failures / len(self._recent_requests)

        if failure_rate >= self.config.failure_threshold:
            logger.warning(
                f"Circuit opening for model={self.model_name}, "
                f"failure_rate={failure_rate:.2f}, "
                f"threshold={self.config.failure_threshold}"
            )
            self._transition_to_open()

    def _should_attempt_half_open(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._opened_at is None:
            return False

        elapsed = time.time() - self._opened_at
        return elapsed >= self.config.cooldown_seconds

    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self._opened_at = None
        self._half_open_requests = 0

        logger.info(
            f"Circuit state change for model={self.model_name}, "
            f"from={old_state.value}, to=closed"
        )

    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self._opened_at = time.time()
        self._half_open_requests = 0

        logger.error(
            f"Circuit state change for model={self.model_name}, "
            f"from={old_state.value if old_state else 'none'}, to=open, "
            f"cooldown_seconds={self.config.cooldown_seconds}"
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self._half_open_requests = 0

        logger.info(
            f"Circuit state change for model={self.model_name}, "
            f"from={old_state.value}, to=half_open"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with state and metrics
        """
        with self._lock:
            if len(self._recent_requests) == 0:
                failure_rate = 0.0
            else:
                failures = sum(1 for result in self._recent_requests if not result)
                failure_rate = failures / len(self._recent_requests)

            return {
                "model": self.model_name,
                "state": self.state.value,
                "failure_rate": failure_rate,
                "total_requests": len(self._recent_requests),
                "opened_at": self._opened_at,
                "cooldown_remaining": (
                    max(0, self.config.cooldown_seconds - (time.time() - self._opened_at))
                    if self._opened_at else 0
                ),
            }


class CircuitBreakerManager:
    """Manages circuit breakers for all models.

    Provides centralized circuit breaker management with automatic
    quarantine integration.
    """

    def __init__(self, config: Optional[CircuitConfig] = None):
        """Initialize circuit breaker manager.

        Args:
            config: Default configuration for all circuit breakers
        """
        self.config = config or CircuitConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

        logger.info(f"Circuit breaker manager initialized with config={self.config}")

    def get_breaker(self, model_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a model.

        Args:
            model_name: Name of the model

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if model_name not in self._breakers:
                self._breakers[model_name] = CircuitBreaker(model_name, self.config)

            return self._breakers[model_name]

    def record_success(self, model_name: str) -> None:
        """Record successful request for a model."""
        breaker = self.get_breaker(model_name)
        breaker.record_success()

    def record_failure(self, model_name: str) -> None:
        """Record failed request for a model."""
        breaker = self.get_breaker(model_name)
        breaker.record_failure()

    def can_attempt(self, model_name: str) -> bool:
        """Check if request to model should be attempted.

        Args:
            model_name: Name of the model

        Returns:
            True if request should be allowed
        """
        breaker = self.get_breaker(model_name)
        can_attempt = breaker.can_attempt()

        if not can_attempt:
            logger.debug(
                f"Circuit breaker rejected request for model={model_name}, "
                f"state={breaker.state.value}"
            )

        return can_attempt

    def get_open_circuits(self) -> list[str]:
        """Get list of models with open circuits.

        Returns:
            List of model names with open or half-open circuits
        """
        with self._lock:
            return [
                name for name, breaker in self._breakers.items()
                if breaker.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)
            ]

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers.

        Returns:
            Dictionary mapping model names to their stats
        """
        with self._lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }

    def reset(self, model_name: Optional[str] = None) -> None:
        """Reset circuit breaker(s).

        Args:
            model_name: Specific model to reset, or None for all
        """
        with self._lock:
            if model_name is None:
                # Reset all
                for breaker in self._breakers.values():
                    breaker._transition_to_closed()
                logger.info("All circuit breakers reset")
            elif model_name in self._breakers:
                self._breakers[model_name]._transition_to_closed()
                logger.info(f"Circuit breaker reset for model={model_name}")


# Global circuit breaker manager
circuit_breaker_manager = CircuitBreakerManager()
