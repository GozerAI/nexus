"""Nexus federated service entry point.

Starts the AutonomousCOO with the csuite bridge listener so Nexus can
receive guidance requests, outcome reports, and health updates from the
CoS coordination layer via Redis pub/sub.

Usage:
    python -m nexus.service
"""

import asyncio
import json
import logging
import os
import signal
from aiohttp import web

logger = logging.getLogger(__name__)

_bridge = None
_coo = None


async def _health_handler(request: web.Request) -> web.Response:
    """Return service health status."""
    healthy = _bridge is not None and _bridge.is_connected
    status = {
        "service": "nexus",
        "status": "healthy" if healthy else "degraded",
        "bridge_connected": _bridge.is_connected if _bridge else False,
        "bridge_listening": _bridge.is_listening if _bridge else False,
    }
    code = 200 if healthy else 503
    return web.json_response(status, status=code)


async def _start_health_server(port: int) -> web.AppRunner:
    """Start a minimal HTTP server for health checks."""
    app = web.Application()
    app.router.add_get("/health", _health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info("Health endpoint listening on :%d/health", port)
    return runner


async def main() -> None:
    """Start Nexus as a federated service."""
    global _bridge, _coo

    logging.basicConfig(
        level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    channel_prefix = os.environ.get("CHANNEL_PREFIX", "csuite:nexus")
    health_port = int(os.environ.get("HEALTH_PORT", "8080"))

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

    logger.info("Starting Nexus service (mode=%s, prefix=%s)", mode_name, channel_prefix)

    health_runner = await _start_health_server(health_port)
    await _bridge.start_listening()
    await _coo.start()

    # Keep running until SIGTERM/SIGINT
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    logger.info("Nexus service ready")
    await stop.wait()

    logger.info("Shutting down Nexus service")
    await _coo.stop()
    await _bridge.disconnect()
    await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
