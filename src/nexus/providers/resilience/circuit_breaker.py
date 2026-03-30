"""Circuit breaker pattern implementation for resilience.

This module implements the circuit breaker pattern to prevent cascading failures
and provide graceful degradation when dependencies fail.

Adapted from psychic-bassoon for unified-intelligence system.
"""

import time
import logging
from enum import Enum
from threading import RLock
from typing import Callable, Optional, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failure detected, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is OPEN"):
        self.message = message
        super().__init__(self.message)


class CircuitBreaker:
    """Circuit breaker implementation for handling failures gracefully.

    The circuit breaker has three states:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures detected, requests are blocked
    - HALF_OPEN: Testing if the service has recovered

    State transitions:
    - CLOSED -> OPEN: When failure threshold is exceeded
    - OPEN -> HALF_OPEN: After timeout period
    - HALF_OPEN -> CLOSED: When test request succeeds
    - HALF_OPEN -> OPEN: When test request fails

    Example:
        >>> breaker = CircuitBreaker(
        ...     failure_threshold=5,
        ...     timeout=60.0,
        ...     expected_exception=ConnectionError
        ... )
        >>>
        >>> def risky_operation():
        ...     # Your code here
        ...     pass
        >>>
        >>> result = breaker.call(risky_operation)
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        name: str = "default",
    ):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes needed to close circuit from half-open
            timeout: Seconds to wait before attempting recovery (OPEN -> HALF_OPEN)
            expected_exception: Exception type that triggers the circuit breaker
            name: Name for this circuit breaker (for logging)
        """
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = RLock()

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"failure_threshold={failure_threshold}, "
            f"timeout={timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get the current circuit state."""
        with self._lock:
            return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (blocking requests)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self.state == CircuitState.HALF_OPEN

    def _check_timeout(self) -> None:
        """Check if timeout has passed and transition to HALF_OPEN if needed."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN (testing recovery)")

    def _on_success(self) -> None:
        """Handle successful request."""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit breaker '{self.name}' success in HALF_OPEN "
                    f"({self._success_count}/{self.success_threshold})"
                )

                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    logger.info(f"Circuit breaker '{self.name}' transitioned to CLOSED (recovered)")

    def _on_failure(self, exception: Exception) -> None:
        """Handle failed request.

        Args:
            exception: The exception that occurred
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Immediate transition back to OPEN on failure in HALF_OPEN
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' transitioned back to OPEN (recovery failed)")

            elif self._state == CircuitState.CLOSED:
                logger.debug(f"Circuit breaker '{self.name}' failure count: {self._failure_count}/{self.failure_threshold}")

                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.error(
                        f"Circuit breaker '{self.name}' transitioned to OPEN "
                        f"(threshold exceeded: {self._failure_count} failures)"
                    )

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker.

        Args:
            func: The function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The function's return value

        Raises:
            CircuitBreakerError: If the circuit is open
            Exception: Any exception raised by the function
        """
        with self._lock:
            self._check_timeout()

            if self._state == CircuitState.OPEN:
                retry_after = self.timeout - (time.time() - self._last_failure_time) if self._last_failure_time else self.timeout
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Retry after {retry_after:.1f}s"
                )

        # Execute the function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure(e)
            raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute an async function through the circuit breaker.

        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The function's return value

        Raises:
            CircuitBreakerError: If the circuit is open
            Exception: Any exception raised by the function
        """
        with self._lock:
            self._check_timeout()

            if self._state == CircuitState.OPEN:
                retry_after = self.timeout - (time.time() - self._last_failure_time) if self._last_failure_time else self.timeout
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Retry after {retry_after:.1f}s"
                )

        # Execute the async function
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure(e)
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")

    def stats(self) -> dict:
        """Get circuit breaker statistics.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            retry_after = None
            if self._state == CircuitState.OPEN and self._last_failure_time:
                retry_after = max(0, self.timeout - (time.time() - self._last_failure_time))

            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "retry_after_seconds": round(retry_after, 2) if retry_after is not None else None,
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        """Initialize the circuit breaker registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = RLock()
        logger.info("Circuit breaker registry initialized")

    def get_breaker(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        expected_exception: type = Exception,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker.

        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening
            success_threshold: Number of successes to close from half-open
            timeout: Timeout before attempting recovery
            expected_exception: Exception type to catch

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    success_threshold=success_threshold,
                    timeout=timeout,
                    expected_exception=expected_exception,
                    name=name,
                )
            return self._breakers[name]

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info(f"Reset all {len(self._breakers)} circuit breakers")

    def stats(self) -> dict:
        """Get statistics for all circuit breakers.

        Returns:
            Dictionary with statistics for each breaker
        """
        with self._lock:
            return {name: breaker.stats() for name, breaker in self._breakers.items()}


# Global circuit breaker registry
_global_registry: Optional[CircuitBreakerRegistry] = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry.

    Returns:
        CircuitBreakerRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry
