"""Nexus shared infrastructure service entry point.

By default, this service exposes Nexus as reusable infrastructure for an
external control plane. The previous AutonomousCOO runtime is still available
behind an explicit legacy toggle for compatibility with older integrations.

Usage:
    python -m nexus.service
"""

import asyncio
from dataclasses import dataclass
import json
import logging
import os
import signal
from typing import Optional
from aiohttp import web

logger = logging.getLogger(__name__)
RUNTIME_APP_KEY = web.AppKey("runtime", "ServiceRuntime")

_bridge = None
_coo = None
_infrastructure = None


@dataclass
class ServiceRuntime:
    """Mutable runtime state for the Nexus shared-infrastructure service."""

    infrastructure: object | None = None
    bridge: object | None = None
    coo: object | None = None


def legacy_coo_enabled_from_env(env: Optional[dict] = None) -> bool:
    """Return whether legacy COO compatibility mode is enabled."""
    env = env or os.environ
    return env.get("NEXUS_ENABLE_LEGACY_COO", "").lower() in {"1", "true", "yes", "on"}


async def _health_handler(request: web.Request) -> web.Response:
    """Return service health status."""
    runtime: ServiceRuntime = request.app[RUNTIME_APP_KEY]
    status, code = await _service_status(runtime)
    return web.json_response(status, status=code)


async def _ready_handler(request: web.Request) -> web.Response:
    """Return readiness status for load balancers and orchestrators."""
    runtime: ServiceRuntime = request.app[RUNTIME_APP_KEY]
    status, code = await _service_status(runtime)
    return web.json_response(status, status=code)


async def _service_status(runtime: ServiceRuntime) -> tuple[dict, int]:
    """Compute a machine-readable service health snapshot."""
    legacy_enabled = runtime.bridge is not None and runtime.coo is not None
    snapshot = await runtime.infrastructure.snapshot() if runtime.infrastructure else None
    healthy = snapshot.healthy if snapshot else False
    status = {
        "service": "nexus",
        "status": "healthy" if healthy else "degraded",
        "ready": healthy,
        "role": "shared_infrastructure",
        "profile": snapshot.profile.to_dict() if snapshot else None,
        "services": snapshot.services if snapshot else {},
        "legacy_coo_enabled": legacy_enabled,
        "bridge_connected": runtime.bridge.is_connected if runtime.bridge else False,
        "bridge_listening": runtime.bridge.is_listening if runtime.bridge else False,
    }
    code = 200 if healthy else 503
    return status, code


def create_health_app(runtime: ServiceRuntime) -> web.Application:
    """Create the HTTP app exposing health/readiness endpoints."""
    app = web.Application()
    app[RUNTIME_APP_KEY] = runtime
    app.router.add_get("/health", _health_handler)
    app.router.add_get("/ready", _ready_handler)
    return app


async def _start_health_server(port: int, runtime: ServiceRuntime) -> web.AppRunner:
    """Start a minimal HTTP server for health checks."""
    app = create_health_app(runtime)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Health endpoints listening on :%d/health and :%d/ready", port, port)
    return runner


async def main() -> None:
    """Start Nexus as shared infrastructure."""
    global _bridge, _coo, _infrastructure

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from nexus.infrastructure import NexusSharedInfrastructure

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    channel_prefix = os.environ.get("CHANNEL_PREFIX", "csuite:nexus")
    health_port = int(os.environ.get("HEALTH_PORT", "8080"))
    legacy_coo_enabled = legacy_coo_enabled_from_env()
    runtime = ServiceRuntime()

    _infrastructure = NexusSharedInfrastructure()
    runtime.infrastructure = _infrastructure
    service_status = await _infrastructure.initialize()

    logger.info(
        "Starting Nexus shared infrastructure service (legacy_coo=%s, prefix=%s)",
        legacy_coo_enabled,
        channel_prefix,
    )
    logger.info("Shared service status: %s", json.dumps(service_status))

    health_runner = await _start_health_server(health_port, runtime)

    if legacy_coo_enabled:
        from nexus.coo.core import AutonomousCOO, COOConfig, ExecutionMode
        from nexus.coo.csuite_bridge import CSuiteBridgeListener, CSuiteBridgeConfig

        mode_name = os.environ.get("COO_MODE", "supervised")
        mode = ExecutionMode(mode_name)

        _coo = AutonomousCOO(config=COOConfig(mode=mode))
        _bridge = CSuiteBridgeListener(
            coo=_coo,
            config=CSuiteBridgeConfig(
                redis_url=redis_url,
                channel_prefix=channel_prefix,
            ),
        )
        runtime.coo = _coo
        runtime.bridge = _bridge
        await _bridge.start_listening()
        await _coo.start()
        logger.info("Legacy AutonomousCOO compatibility mode enabled")

    # Keep running until SIGTERM/SIGINT
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    logger.info("Nexus service ready")
    await stop.wait()

    logger.info("Shutting down Nexus service")
    if _coo:
        await _coo.stop()
    if _bridge:
        await _bridge.disconnect()
    await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
