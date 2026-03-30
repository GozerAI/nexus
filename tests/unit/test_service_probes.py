"""Tests for service liveness/readiness probes."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from nexus.service import ServiceRuntime, create_health_app, RUNTIME_APP_KEY


@pytest.fixture
def runtime():
    rt = ServiceRuntime()
    rt.infrastructure = None
    rt.bridge = None
    rt.coo = None
    return rt


async def _get_json(app: web.Application, path: str):
    """Helper: create a test client, make a GET, return (status, body)."""
    from aiohttp.test_utils import TestClient, TestServer
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        resp = await client.get(path)
        status = resp.status
        body = await resp.json()
    finally:
        await client.close()
    return status, body


class TestLivenessProbe:
    """GET /health — liveness: always 200 if process is alive."""

    @pytest.mark.asyncio
    async def test_returns_200_even_without_infrastructure(self, runtime):
        app = create_health_app(runtime)
        status, body = await _get_json(app, "/health")
        assert status == 200
        assert body["status"] == "alive"
        assert body["service"] == "nexus"

    @pytest.mark.asyncio
    async def test_returns_200_with_legacy_coo(self, runtime):
        runtime.bridge = MagicMock()
        runtime.coo = MagicMock()
        app = create_health_app(runtime)
        status, body = await _get_json(app, "/health")
        assert status == 200
        assert body["legacy_coo_enabled"] is True


class TestReadinessProbe:
    """GET /ready — readiness: 200 only when infrastructure is healthy."""

    @pytest.mark.asyncio
    async def test_returns_503_without_infrastructure(self, runtime):
        app = create_health_app(runtime)
        status, body = await _get_json(app, "/ready")
        assert status == 503
        assert body["ready"] is False

    @pytest.mark.asyncio
    async def test_returns_200_when_healthy(self, runtime):
        snapshot = MagicMock()
        snapshot.healthy = True
        snapshot.profile.to_dict.return_value = {"name": "test"}
        snapshot.services = {"llm": True}

        infra = AsyncMock()
        infra.snapshot.return_value = snapshot
        runtime.infrastructure = infra

        app = create_health_app(runtime)
        status, body = await _get_json(app, "/ready")
        assert status == 200
        assert body["ready"] is True

    @pytest.mark.asyncio
    async def test_returns_503_when_degraded(self, runtime):
        snapshot = MagicMock()
        snapshot.healthy = False
        snapshot.profile.to_dict.return_value = {}
        snapshot.services = {}

        infra = AsyncMock()
        infra.snapshot.return_value = snapshot
        runtime.infrastructure = infra

        app = create_health_app(runtime)
        status, body = await _get_json(app, "/ready")
        assert status == 503

    @pytest.mark.asyncio
    async def test_snapshot_timeout_returns_503(self, runtime):
        async def _slow_snapshot():
            await asyncio.sleep(60)

        infra = MagicMock()
        infra.snapshot = _slow_snapshot
        runtime.infrastructure = infra

        app = create_health_app(runtime)
        status, body = await _get_json(app, "/ready")
        assert status == 503
        assert body["ready"] is False

    @pytest.mark.asyncio
    async def test_snapshot_exception_returns_503(self, runtime):
        infra = AsyncMock()
        infra.snapshot.side_effect = RuntimeError("boom")
        runtime.infrastructure = infra

        app = create_health_app(runtime)
        status, body = await _get_json(app, "/ready")
        assert status == 503
