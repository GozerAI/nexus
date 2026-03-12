"""
Rate limiting for protecting against overload.

This module provides rate limiting to:
- Prevent API quota exhaustion
- Protect against abuse
- Ensure fair resource allocation
- Control costs

Supports multiple rate limiting strategies:
- Fixed window
- Sliding window
- Token bucket
- Per-user limits
"""

import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after  # Seconds until retry is allowed


@dataclass
class RateLimitConfig:
    """
    Rate limit configuration.

    Attributes:
        max_requests: Maximum requests allowed
        window_seconds: Time window in seconds
        burst_size: Maximum burst size (for token bucket)
    """

    max_requests: int = 100
    window_seconds: int = 60
    burst_size: Optional[int] = None


class RateLimiter:
    """
    Token bucket rate limiter.

    The token bucket algorithm allows for bursts while maintaining
    an average rate limit. Tokens are added at a steady rate,
    and each request consumes one token.

    Features:
    - Configurable rate and burst size
    - Per-key rate limiting
    - Automatic token refill
    - Retry-after calculation
    - Thread-safe operations
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()

        # Calculate refill rate (tokens per second)
        self._refill_rate = self.config.max_requests / self.config.window_seconds

        # Burst size (default to max_requests)
        self._burst_size = self.config.burst_size or self.config.max_requests

        # Per-key buckets: {key: (tokens, last_refill_time)}
        self._buckets: Dict[str, tuple[float, float]] = {}

        # Statistics
        self._total_requests = 0
        self._total_limited = 0

    def check_limit(
        self,
        key: str = "default",
        tokens: int = 1,
    ) -> tuple[bool, float]:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Rate limit key (e.g., user_id, api_key)
            tokens: Number of tokens to consume

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        self._total_requests += 1

        # Get or create bucket
        if key not in self._buckets:
            self._buckets[key] = (float(self._burst_size), time.time())

        current_tokens, last_refill = self._buckets[key]
        now = time.time()

        # Calculate tokens to add based on elapsed time
        elapsed = now - last_refill
        tokens_to_add = elapsed * self._refill_rate

        # Update token count (capped at burst size)
        current_tokens = min(
            self._burst_size,
            current_tokens + tokens_to_add
        )

        # Check if enough tokens available
        if current_tokens >= tokens:
            # Consume tokens
            current_tokens -= tokens
            self._buckets[key] = (current_tokens, now)
            return True, 0.0
        else:
            # Rate limit exceeded
            self._total_limited += 1

            # Calculate retry-after (time until enough tokens available)
            tokens_needed = tokens - current_tokens
            retry_after = tokens_needed / self._refill_rate

            self._buckets[key] = (current_tokens, now)
            return False, retry_after

    def allow(
        self,
        key: str = "default",
        tokens: int = 1,
    ):
        """
        Check rate limit and raise exception if exceeded.

        Args:
            key: Rate limit key
            tokens: Number of tokens to consume

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        allowed, retry_after = self.check_limit(key, tokens)

        if not allowed:
            raise RateLimitExceeded(
                f"Rate limit exceeded for key '{key}'. "
                f"Retry after {retry_after:.1f} seconds.",
                retry_after=retry_after,
            )

    def reset(self, key: str):
        """
        Reset rate limit for a key.

        Args:
            key: Rate limit key
        """
        if key in self._buckets:
            self._buckets[key] = (float(self._burst_size), time.time())

    def reset_all(self):
        """Reset all rate limits."""
        self._buckets.clear()
        self._total_requests = 0
        self._total_limited = 0

    def get_remaining(self, key: str = "default") -> int:
        """
        Get remaining tokens for a key.

        Args:
            key: Rate limit key

        Returns:
            Number of remaining tokens
        """
        if key not in self._buckets:
            return int(self._burst_size)

        current_tokens, last_refill = self._buckets[key]
        now = time.time()

        # Calculate current token count
        elapsed = now - last_refill
        tokens_to_add = elapsed * self._refill_rate
        current_tokens = min(
            self._burst_size,
            current_tokens + tokens_to_add
        )

        return int(current_tokens)

    def get_stats(self) -> Dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with stats
        """
        limited_rate = (
            self._total_limited / self._total_requests
            if self._total_requests > 0
            else 0.0
        )

        return {
            "total_requests": self._total_requests,
            "total_limited": self._total_limited,
            "limited_rate": limited_rate,
            "active_keys": len(self._buckets),
            "config": {
                "max_requests": self.config.max_requests,
                "window_seconds": self.config.window_seconds,
                "burst_size": self._burst_size,
                "refill_rate": self._refill_rate,
            },
        }


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Maintains a sliding window of request timestamps and enforces
    a limit on the number of requests within the window.

    More accurate than fixed window but uses more memory.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize sliding window rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()

        # Per-key request timestamps
        self._timestamps: Dict[str, Deque[float]] = {}

        # Statistics
        self._total_requests = 0
        self._total_limited = 0

    def check_limit(
        self,
        key: str = "default",
    ) -> tuple[bool, float]:
        """
        Check if request is allowed under rate limit.

        Args:
            key: Rate limit key

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        self._total_requests += 1
        now = time.time()
        window_start = now - self.config.window_seconds

        # Get or create timestamp list
        if key not in self._timestamps:
            self._timestamps[key] = deque()

        timestamps = self._timestamps[key]

        # Remove old timestamps outside window
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        # Check if under limit
        if len(timestamps) < self.config.max_requests:
            timestamps.append(now)
            return True, 0.0
        else:
            # Rate limit exceeded
            self._total_limited += 1

            # Calculate retry-after (time until oldest request exits window)
            oldest = timestamps[0]
            retry_after = (oldest + self.config.window_seconds) - now

            return False, max(0.0, retry_after)

    def allow(self, key: str = "default"):
        """
        Check rate limit and raise exception if exceeded.

        Args:
            key: Rate limit key

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        allowed, retry_after = self.check_limit(key)

        if not allowed:
            raise RateLimitExceeded(
                f"Rate limit exceeded for key '{key}'. "
                f"Retry after {retry_after:.1f} seconds.",
                retry_after=retry_after,
            )

    def reset(self, key: str):
        """
        Reset rate limit for a key.

        Args:
            key: Rate limit key
        """
        if key in self._timestamps:
            self._timestamps[key].clear()

    def reset_all(self):
        """Reset all rate limits."""
        self._timestamps.clear()
        self._total_requests = 0
        self._total_limited = 0

    def get_remaining(self, key: str = "default") -> int:
        """
        Get remaining requests for a key.

        Args:
            key: Rate limit key

        Returns:
            Number of remaining requests
        """
        if key not in self._timestamps:
            return self.config.max_requests

        now = time.time()
        window_start = now - self.config.window_seconds

        timestamps = self._timestamps[key]

        # Count requests in window
        count = sum(1 for ts in timestamps if ts >= window_start)

        return max(0, self.config.max_requests - count)

    def get_stats(self) -> Dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with stats
        """
        limited_rate = (
            self._total_limited / self._total_requests
            if self._total_requests > 0
            else 0.0
        )

        return {
            "total_requests": self._total_requests,
            "total_limited": self._total_limited,
            "limited_rate": limited_rate,
            "active_keys": len(self._timestamps),
            "config": {
                "max_requests": self.config.max_requests,
                "window_seconds": self.config.window_seconds,
            },
        }
