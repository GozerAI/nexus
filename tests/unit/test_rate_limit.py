"""Tests for the API rate limiter."""

import time

from nexus.api.rate_limit import RateLimiter, reset_limiter, configure_limiter, get_limiter


class TestRateLimiter:
    """Unit tests for the token-bucket rate limiter."""

    def test_allows_requests_under_limit(self):
        limiter = RateLimiter(rate=10, window=1, burst=0)
        for _ in range(10):
            allowed, _ = limiter.allow("key1")
            assert allowed

    def test_rejects_requests_over_limit(self):
        limiter = RateLimiter(rate=3, window=1, burst=0)
        for _ in range(3):
            limiter.allow("key1")
        allowed, headers = limiter.allow("key1")
        assert not allowed
        assert "Retry-After" in headers
        assert headers["X-RateLimit-Remaining"] == "0"

    def test_different_keys_are_independent(self):
        limiter = RateLimiter(rate=2, window=1, burst=0)
        limiter.allow("a")
        limiter.allow("a")
        allowed_a, _ = limiter.allow("a")
        allowed_b, _ = limiter.allow("b")
        assert not allowed_a
        assert allowed_b

    def test_burst_adds_extra_capacity(self):
        limiter = RateLimiter(rate=2, window=1, burst=3)
        # Should allow rate + burst = 5 requests
        results = [limiter.allow("k")[0] for _ in range(5)]
        assert all(results)
        allowed, _ = limiter.allow("k")
        assert not allowed

    def test_tokens_refill_over_time(self):
        limiter = RateLimiter(rate=10, window=1, burst=0)
        # Exhaust all tokens
        for _ in range(10):
            limiter.allow("k")
        allowed, _ = limiter.allow("k")
        assert not allowed
        # Wait for refill (tokens refill at 10/s, need 1 token → ~0.1s)
        time.sleep(0.15)
        allowed, _ = limiter.allow("k")
        assert allowed

    def test_headers_include_limit_and_remaining(self):
        limiter = RateLimiter(rate=5, window=1, burst=0)
        allowed, headers = limiter.allow("k")
        assert allowed
        assert headers["X-RateLimit-Limit"] == "5"
        assert int(headers["X-RateLimit-Remaining"]) >= 0

    def test_eviction_removes_stale_buckets(self):
        limiter = RateLimiter(rate=10, window=1, burst=0)
        limiter._evict_after = 0.01  # very short for testing
        limiter.allow("stale_key")
        time.sleep(0.02)
        limiter._last_eviction = 0  # force eviction check
        limiter.allow("fresh_key")
        assert "stale_key" not in limiter._buckets
        assert "fresh_key" in limiter._buckets


class TestRateLimiterSingleton:
    """Test module-level singleton helpers."""

    def setup_method(self):
        reset_limiter()

    def test_get_limiter_returns_same_instance(self):
        a = get_limiter()
        b = get_limiter()
        assert a is b

    def test_configure_limiter_replaces_singleton(self):
        a = get_limiter()
        b = configure_limiter(rate=100, window=10, burst=5)
        assert a is not b
        assert get_limiter() is b

    def test_reset_limiter_clears_singleton(self):
        a = get_limiter()
        reset_limiter()
        b = get_limiter()
        assert a is not b


class TestRateLimitIntegration:
    """Integration tests: rate limiting wired into Flask app."""

    def setup_method(self):
        reset_limiter()
        configure_limiter(rate=5, window=1, burst=0)

    def test_health_endpoint_not_rate_limited(self):
        from nexus.api.api import create_app
        app = create_app({"TESTING": True})
        with app.test_client() as client:
            for _ in range(20):
                resp = client.get("/health")
                assert resp.status_code == 200

    def test_think_endpoint_rate_limited(self):
        from nexus.api.api import create_app

        class StubEngine:
            def think(self, x):
                return x

        app = create_app({"TESTING": True}, engine_factory=StubEngine)
        with app.test_client() as client:
            for _ in range(5):
                resp = client.post("/think", json={"input": "hi"})
                assert resp.status_code == 200
            resp = client.post("/think", json={"input": "hi"})
            assert resp.status_code == 429
            body = resp.get_json()
            assert body["ok"] is False
            assert "Rate limit" in body["error"]

    def test_rate_limit_headers_present(self):
        from nexus.api.api import create_app

        class StubEngine:
            def think(self, x):
                return x

        app = create_app({"TESTING": True}, engine_factory=StubEngine)
        with app.test_client() as client:
            resp = client.post("/think", json={"input": "hi"})
            assert "X-RateLimit-Limit" in resp.headers
            assert "X-RateLimit-Remaining" in resp.headers
