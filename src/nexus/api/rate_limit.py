"""
In-memory rate limiter for the Nexus API.

Uses a token-bucket algorithm with per-key and per-IP tracking.
Thread-safe via threading.Lock.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from flask import Flask, Response, g, jsonify, request

logger = logging.getLogger(__name__)

# Defaults
DEFAULT_RATE = 60  # requests per window
DEFAULT_WINDOW = 60  # seconds
DEFAULT_BURST = 10  # extra burst allowance


@dataclass
class _Bucket:
    tokens: float
    last_refill: float
    rate: float  # tokens per second
    capacity: float


class RateLimiter:
    """Per-key token-bucket rate limiter."""

    def __init__(
        self,
        rate: int = DEFAULT_RATE,
        window: int = DEFAULT_WINDOW,
        burst: int = DEFAULT_BURST,
    ):
        self._rate_per_sec = rate / window
        self._capacity = rate + burst
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        # Evict stale buckets every N checks to prevent unbounded growth
        self._evict_after = window * 10
        self._last_eviction = time.monotonic()

    def _get_bucket(self, key: str) -> _Bucket:
        if key not in self._buckets:
            now = time.monotonic()
            self._buckets[key] = _Bucket(
                tokens=self._capacity,
                last_refill=now,
                rate=self._rate_per_sec,
                capacity=self._capacity,
            )
        return self._buckets[key]

    def _refill(self, bucket: _Bucket) -> None:
        now = time.monotonic()
        elapsed = now - bucket.last_refill
        bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.rate)
        bucket.last_refill = now

    def _maybe_evict(self) -> None:
        now = time.monotonic()
        if now - self._last_eviction < self._evict_after:
            return
        self._last_eviction = now
        stale_keys = [
            k
            for k, b in self._buckets.items()
            if now - b.last_refill > self._evict_after
        ]
        for k in stale_keys:
            del self._buckets[k]

    def allow(self, key: str) -> tuple[bool, dict[str, str]]:
        """Check whether *key* may proceed.  Returns (allowed, headers)."""
        with self._lock:
            self._maybe_evict()
            bucket = self._get_bucket(key)
            self._refill(bucket)

            headers = {
                "X-RateLimit-Limit": str(int(bucket.capacity)),
                "X-RateLimit-Remaining": str(max(0, int(bucket.tokens) - 1)),
            }

            if bucket.tokens >= 1:
                bucket.tokens -= 1
                return True, headers

            retry_after = (1 - bucket.tokens) / bucket.rate
            headers["Retry-After"] = str(int(retry_after) + 1)
            headers["X-RateLimit-Remaining"] = "0"
            return False, headers


# Module-level singleton
_limiter: Optional[RateLimiter] = None


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter


def reset_limiter() -> None:
    """Reset the singleton (for tests)."""
    global _limiter
    _limiter = None


def configure_limiter(
    rate: int = DEFAULT_RATE,
    window: int = DEFAULT_WINDOW,
    burst: int = DEFAULT_BURST,
) -> RateLimiter:
    """Create a fresh limiter with the given parameters."""
    global _limiter
    _limiter = RateLimiter(rate=rate, window=window, burst=burst)
    return _limiter


def init_rate_limiting(app: Flask) -> None:
    """Register before/after request hooks that enforce rate limits."""

    @app.before_request
    def _enforce_rate_limit():
        # Skip health endpoints
        if request.path in ("/health",) or request.path.endswith("/health"):
            return None

        # Identify caller: prefer API key, fall back to IP
        api_key = request.headers.get("X-API-Key") or request.headers.get(
            "Authorization", ""
        )
        key = api_key or request.remote_addr or "unknown"

        limiter = get_limiter()
        allowed, headers = limiter.allow(key)

        # Stash headers on g so after_request can attach them
        g._rate_limit_headers = headers

        if not allowed:
            logger.warning("Rate limit exceeded for key=%s path=%s", key[:12], request.path)
            resp = jsonify({"ok": False, "error": "Rate limit exceeded"})
            resp.status_code = 429
            for h, v in headers.items():
                resp.headers[h] = v
            return resp

        return None

    @app.after_request
    def _add_rate_limit_headers(response: Response) -> Response:
        headers = getattr(g, "_rate_limit_headers", None)
        if headers:
            for h, v in headers.items():
                response.headers[h] = v
        return response
