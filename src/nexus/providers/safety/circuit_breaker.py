"""
Circuit breaker pattern for preventing cascading failures.

This module implements the circuit breaker pattern to protect
the system from cascading failures when external services
(model APIs) become unavailable or slow.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Service is failing, requests are rejected immediately
- HALF_OPEN: Testing if service has recovered

The circuit breaker automatically transitions between states
based on failure rates and recovery testing.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional


class CircuitState(str, Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """
    Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures to trip circuit
        success_threshold: Number of successes to close circuit
        timeout_seconds: Seconds before attempting recovery
        half_open_max_calls: Max calls in half-open state
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    half_open_max_calls: int = 3


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    The circuit breaker wraps calls to external services (model APIs)
    and automatically trips when failures exceed a threshold, preventing
    further calls to the failing service.

    After a timeout period, the circuit enters HALF_OPEN state and
    allows a limited number of test calls. If they succeed, the circuit
    closes and normal operation resumes.

    Usage:
        breaker = CircuitBreaker(name="openai")

        try:
            with breaker:
                result = call_openai_api()
        except CircuitBreakerOpen:
            # Service is unavailable, use fallback
            result = use_fallback()

    Features:
    - Automatic state transitions
    - Configurable thresholds and timeouts
    - Success/failure tracking
    - State change callbacks
    - Per-service isolation
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
        on_state_change: Optional[Callable[[CircuitState], None]] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name of this circuit breaker (e.g., service name)
            config: Configuration
            on_state_change: Callback for state changes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        # Current state
        self._state = CircuitState.CLOSED

        # Counters
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

        # Timing
        self._last_failure_time: Optional[datetime] = None
        self._state_changed_at = datetime.now()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._trips = 0

    @property
    def state(self) -> CircuitState:
        """Get current state."""
        return self._state

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpen: If circuit is open
            Exception: Any exception from the function
        """
        self._total_calls += 1

        # Check if circuit is open
        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN"
                )

        # Check if in half-open and max calls reached
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached"
                )
            self._half_open_calls += 1

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise

    def __enter__(self):
        """Context manager entry."""
        self._total_calls += 1

        # Check if circuit is open
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is OPEN"
                )

        # Check half-open limit
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                raise CircuitBreakerOpen(
                    f"Circuit breaker '{self.name}' is HALF_OPEN with max calls reached"
                )
            self._half_open_calls += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            # Success
            self._record_success()
        else:
            # Failure (exception occurred)
            self._record_failure()

        # Don't suppress exceptions
        return False

    def _record_success(self):
        """Record a successful call."""
        self._total_successes += 1
        self._success_count += 1
        self._failure_count = 0  # Reset failure counter

        # Check if we should close the circuit
        if self._state == CircuitState.HALF_OPEN:
            if self._success_count >= self.config.success_threshold:
                self._transition_to_closed()

    def _record_failure(self):
        """Record a failed call."""
        self._total_failures += 1
        self._failure_count += 1
        self._success_count = 0  # Reset success counter
        self._last_failure_time = datetime.now()

        # Check if we should open the circuit
        if self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to_open()

        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open state opens the circuit
            self._transition_to_open()

    def _transition_to_open(self):
        """Transition to OPEN state."""
        old_state = self._state
        self._state = CircuitState.OPEN
        self._state_changed_at = datetime.now()
        self._trips += 1

        if self.on_state_change:
            self.on_state_change(self._state)

    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state."""
        old_state = self._state
        self._state = CircuitState.HALF_OPEN
        self._state_changed_at = datetime.now()
        self._half_open_calls = 0
        self._success_count = 0
        self._failure_count = 0

        if self.on_state_change:
            self.on_state_change(self._state)

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._state_changed_at = datetime.now()
        self._success_count = 0
        self._failure_count = 0
        self._half_open_calls = 0

        if self.on_state_change:
            self.on_state_change(self._state)

    def _should_attempt_reset(self) -> bool:
        """
        Check if we should attempt to reset (move to half-open).

        Returns:
            True if timeout has passed
        """
        if self._last_failure_time is None:
            return False

        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.config.timeout_seconds

    def reset(self):
        """Manually reset the circuit breaker to CLOSED state."""
        self._transition_to_closed()

    def force_open(self):
        """Manually open the circuit breaker."""
        self._transition_to_open()

    def get_stats(self) -> Dict:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with stats
        """
        time_in_state = (datetime.now() - self._state_changed_at).total_seconds()

        success_rate = (
            self._total_successes / self._total_calls
            if self._total_calls > 0
            else 0.0
        )

        return {
            "name": self.name,
            "state": self._state.value,
            "total_calls": self._total_calls,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
            "success_rate": success_rate,
            "current_failure_count": self._failure_count,
            "current_success_count": self._success_count,
            "trips": self._trips,
            "time_in_current_state": time_in_state,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
            },
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management of circuit breakers for
    different services.
    """

    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}

    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """
        Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            config: Optional configuration (used if creating new)

        Returns:
            Circuit breaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name=name, config=config)

        return self._breakers[name]

    def get_all_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all circuit breakers.

        Returns:
            Dictionary mapping breaker name to stats
        """
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()

    def get_open_breakers(self) -> List[str]:
        """
        Get names of all open circuit breakers.

        Returns:
            List of breaker names
        """
        return [
            name
            for name, breaker in self._breakers.items()
            if breaker.state == CircuitState.OPEN
        ]


# Import for type hint
from typing import List
