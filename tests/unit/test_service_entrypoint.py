"""Regression tests for the Nexus shared-infrastructure service entrypoint."""

from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nexus.service import ServiceRuntime, create_health_app, legacy_coo_enabled_from_env


def test_legacy_coo_enabled_from_env_recognizes_truthy_values():
    assert legacy_coo_enabled_from_env({"NEXUS_ENABLE_LEGACY_COO": "true"}) is True
    assert legacy_coo_enabled_from_env({"NEXUS_ENABLE_LEGACY_COO": "on"}) is True
    assert legacy_coo_enabled_from_env({"NEXUS_ENABLE_LEGACY_COO": "0"}) is False


@pytest.mark.asyncio
async def test_health_endpoint_reports_ready_runtime():
    profile = SimpleNamespace(to_dict=lambda: {"mode": "shared"})
    snapshot = SimpleNamespace(healthy=True, profile=profile, services={"llm": True})
    infrastructure = SimpleNamespace(snapshot=lambda: _awaitable(snapshot))
    runtime = ServiceRuntime(
        infrastructure=infrastructure,
        bridge=SimpleNamespace(is_connected=True, is_listening=True),
        coo=object(),
    )

    app = create_health_app(runtime)
    async with TestServer(app) as server, TestClient(server) as client:
        response = await client.get("/health")
        payload = await response.json()

    assert response.status == 200
    assert payload["ready"] is True
    assert payload["legacy_coo_enabled"] is True
    assert payload["bridge_connected"] is True


@pytest.mark.asyncio
async def test_ready_endpoint_reports_degraded_runtime():
    profile = SimpleNamespace(to_dict=lambda: {"mode": "shared"})
    snapshot = SimpleNamespace(healthy=False, profile=profile, services={"llm": False})
    infrastructure = SimpleNamespace(snapshot=lambda: _awaitable(snapshot))
    runtime = ServiceRuntime(infrastructure=infrastructure)

    app = create_health_app(runtime)
    async with TestServer(app) as server, TestClient(server) as client:
        response = await client.get("/ready")
        payload = await response.json()

    assert response.status == 503
    assert payload["ready"] is False
    assert payload["status"] == "degraded"


async def _awaitable(value):
    return value
