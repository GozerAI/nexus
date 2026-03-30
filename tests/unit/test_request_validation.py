"""Tests for request validation middleware."""

from nexus.api.api import create_app
from nexus.api.rate_limit import reset_limiter, configure_limiter


class TestRequestValidation:
    """Tests for request size limits and content-type enforcement."""

    def setup_method(self):
        reset_limiter()
        configure_limiter(rate=1000, window=1, burst=0)

    def _make_app(self, max_content_length=None):
        config = {"TESTING": True}
        if max_content_length is not None:
            config["MAX_CONTENT_LENGTH"] = max_content_length

        class StubEngine:
            def think(self, x):
                return x

        return create_app(config, engine_factory=StubEngine)

    def test_rejects_non_json_post(self):
        app = self._make_app()
        with app.test_client() as client:
            resp = client.post("/think", data="hello", content_type="text/plain")
        assert resp.status_code == 400
        body = resp.get_json()
        assert "application/json" in body["error"]

    def test_rejects_malformed_json(self):
        app = self._make_app()
        with app.test_client() as client:
            resp = client.post(
                "/think",
                data="{bad json",
                content_type="application/json",
            )
        assert resp.status_code == 400
        body = resp.get_json()
        assert "valid JSON" in body["error"]

    def test_accepts_valid_json_post(self):
        app = self._make_app()
        with app.test_client() as client:
            resp = client.post("/think", json={"input": "hello"})
        assert resp.status_code == 200

    def test_get_requests_not_validated(self):
        app = self._make_app()
        with app.test_client() as client:
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_payload_too_large_returns_413(self):
        app = self._make_app(max_content_length=50)
        with app.test_client() as client:
            # Send a raw body larger than 50 bytes
            big_body = b'{"input": "' + b"x" * 200 + b'"}'
            resp = client.post(
                "/think",
                data=big_body,
                content_type="application/json",
            )
        assert resp.status_code == 413
        body = resp.get_json()
        assert "too large" in body["error"]

    def test_health_endpoint_skips_validation(self):
        app = self._make_app()
        with app.test_client() as client:
            # GET to a health endpoint — no validation
            resp = client.get("/health")
        assert resp.status_code == 200
