"""Tests for CORS and security headers middleware."""

from nexus.api.api import create_app
from nexus.api.rate_limit import reset_limiter, configure_limiter


class TestSecurityHeaders:
    """Verify security headers are present on responses."""

    def setup_method(self):
        reset_limiter()
        configure_limiter(rate=1000, window=1, burst=0)

    def _app(self, **config):
        class StubEngine:
            def think(self, x):
                return x
        return create_app({"TESTING": True, **config}, engine_factory=StubEngine)

    def test_csp_header_present(self):
        app = self._app()
        with app.test_client() as c:
            resp = c.get("/health")
        assert "Content-Security-Policy" in resp.headers
        assert "default-src 'none'" in resp.headers["Content-Security-Policy"]

    def test_x_content_type_options(self):
        app = self._app()
        with app.test_client() as c:
            resp = c.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"

    def test_x_frame_options(self):
        app = self._app()
        with app.test_client() as c:
            resp = c.get("/health")
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_hsts_header(self):
        app = self._app()
        with app.test_client() as c:
            resp = c.get("/health")
        assert "Strict-Transport-Security" in resp.headers

    def test_cors_wildcard_by_default(self):
        app = self._app()
        with app.test_client() as c:
            resp = c.get("/health")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_cors_specific_origin_when_configured(self):
        origins = {"https://app.gozerai.com"}
        app = self._app(CORS_ORIGINS=origins)
        with app.test_client() as c:
            resp = c.get("/health", headers={"Origin": "https://app.gozerai.com"})
        assert resp.headers.get("Access-Control-Allow-Origin") == "https://app.gozerai.com"

    def test_cors_rejects_unlisted_origin(self):
        origins = {"https://app.gozerai.com"}
        app = self._app(CORS_ORIGINS=origins)
        with app.test_client() as c:
            resp = c.get("/health", headers={"Origin": "https://evil.com"})
        assert "Access-Control-Allow-Origin" not in resp.headers

    def test_options_preflight_returns_204(self):
        app = self._app()
        with app.test_client() as c:
            resp = c.options("/think")
        assert resp.status_code == 204

    def test_cors_allow_headers_includes_api_key(self):
        app = self._app()
        with app.test_client() as c:
            resp = c.get("/health")
        allow_headers = resp.headers.get("Access-Control-Allow-Headers", "")
        assert "X-API-Key" in allow_headers
        assert "Authorization" in allow_headers
