"""
Resilient model adapter wrapper with circuit breaker protection.

This module provides a wrapper for model adapters that adds circuit breaker
protection to prevent cascading failures when model APIs are unavailable.
"""

import logging
from typing import Any, Optional

from nexus.providers.resilience import CircuitBreaker, CircuitBreakerError, get_circuit_breaker_registry

logger = logging.getLogger(__name__)


class ResilientAdapterWrapper:
    """
    Wrapper that adds circuit breaker protection to any model adapter.

    Example:
        >>> from unified_intelligence.models import get_adapter_for_model
        >>> from nexus.providers.adapters.resilient_adapter import ResilientAdapterWrapper
        >>>
        >>> # Create adapter
        >>> adapter = get_adapter_for_model("gpt-4", api_key="...")
        >>>
        >>> # Wrap with circuit breaker
        >>> resilient_adapter = ResilientAdapterWrapper(
        ...     adapter,
        ...     failure_threshold=5,
        ...     timeout=60.0
        ... )
        >>>
        >>> # Use normally - circuit breaker protection is automatic
        >>> response = await resilient_adapter.generate(...)
    """

    def __init__(
        self,
        adapter: Any,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        name: Optional[str] = None,
    ):
        """
        Initialize resilient adapter wrapper.

        Args:
            adapter: The model adapter to wrap
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes to close from half-open
            timeout: Seconds before attempting recovery
            name: Circuit breaker name (defaults to adapter name)
        """
        self.adapter = adapter
        self.name = name or getattr(adapter, 'model_name', 'unknown')

        # Get or create circuit breaker
        registry = get_circuit_breaker_registry()
        self.circuit_breaker = registry.get_breaker(
            name=f"model_{self.name}",
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout=timeout,
            expected_exception=Exception,  # Catch all exceptions
        )

        logger.info(f"Created resilient wrapper for {self.name}")

    async def generate(self, *args, **kwargs):
        """
        Generate response through circuit breaker.

        Args:
            *args: Positional arguments for adapter.generate()
            **kwargs: Keyword arguments for adapter.generate()

        Returns:
            Response from the adapter

        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Any exception from the adapter
        """
        return await self.circuit_breaker.call_async(
            self.adapter.generate,
            *args,
            **kwargs
        )

    async def generate_stream(self, *args, **kwargs):
        """
        Generate streaming response through circuit breaker.

        Note: Streaming is allowed through circuit breaker, but failures
        will still be counted.

        Args:
            *args: Positional arguments for adapter.generate_stream()
            **kwargs: Keyword arguments for adapter.generate_stream()

        Yields:
            Streamed chunks from the adapter

        Raises:
            CircuitBreakerError: If circuit is open
        """
        # For streaming, we check circuit state but don't wrap the iterator
        if self.circuit_breaker.is_open:
            raise CircuitBreakerError(f"Circuit breaker for {self.name} is OPEN")

        try:
            async for chunk in self.adapter.generate_stream(*args, **kwargs):
                yield chunk
            # Success - reset failure count
            self.circuit_breaker._on_success()
        except Exception as e:
            # Failure - update circuit breaker
            self.circuit_breaker._on_failure(e)
            raise

    def get_stats(self) -> dict:
        """
        Get circuit breaker statistics.

        Returns:
            Dictionary with circuit breaker stats
        """
        return self.circuit_breaker.stats()

    def reset(self):
        """Reset the circuit breaker to CLOSED state."""
        self.circuit_breaker.reset()

    # Proxy other methods to underlying adapter
    def __getattr__(self, name):
        """Proxy attribute access to underlying adapter."""
        return getattr(self.adapter, name)


def wrap_adapter_with_circuit_breaker(
    adapter: Any,
    failure_threshold: int = 5,
    timeout: float = 60.0,
) -> ResilientAdapterWrapper:
    """
    Wrap a model adapter with circuit breaker protection.

    This is a convenience function for creating ResilientAdapterWrapper.

    Args:
        adapter: Model adapter to wrap
        failure_threshold: Number of failures before opening circuit
        timeout: Seconds before attempting recovery

    Returns:
        Resilient adapter wrapper

    Example:
        >>> adapter = OpenAIAdapter("gpt-4", api_key="...")
        >>> resilient = wrap_adapter_with_circuit_breaker(adapter)
        >>> response = await resilient.generate("Hello")
    """
    return ResilientAdapterWrapper(
        adapter,
        failure_threshold=failure_threshold,
        timeout=timeout,
    )


__all__ = [
    "ResilientAdapterWrapper",
    "wrap_adapter_with_circuit_breaker",
]
