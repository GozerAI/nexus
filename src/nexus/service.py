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
_infrastructure = None


@dataclass
class ServiceRuntime:
    """Mutable runtime state for the Nexus shared-infrastructure service."""

    infrastructure: object | None = None
    bridge: object | None = None
    coo: object | None = None
    service_handler: object | None = None


def legacy_coo_enabled_from_env(env: Optional[dict] = None) -> bool:
    """Return whether legacy COO compatibility mode is enabled."""
    env = env or os.environ
    return env.get("NEXUS_ENABLE_LEGACY_COO", "").lower() in {"1", "true", "yes", "on"}


async def _health_handler(request: web.Request) -> web.Response:
    """Liveness probe — returns 200 if the process is alive.

    Kubernetes uses this to decide whether to *restart* the container.  It
    intentionally does NOT check subsystem health so that a temporarily
    degraded service is not killed during recovery.
    """
    runtime: ServiceRuntime = request.app[RUNTIME_APP_KEY]
    body = {
        "service": "nexus",
        "status": "alive",
        "role": "shared_infrastructure",
        "legacy_coo_enabled": runtime.bridge is not None,
    }
    if runtime.service_handler and hasattr(runtime.service_handler, "get_stats"):
        body["service_handler"] = runtime.service_handler.get_stats()
    return web.json_response(body, status=200)


async def _ready_handler(request: web.Request) -> web.Response:
    """Readiness probe — returns 200 only when the service can handle traffic.

    Kubernetes uses this to decide whether to *route traffic* to the pod.
    Checks infrastructure snapshot with a timeout so a hung subsystem
    doesn't block the probe indefinitely.
    """
    runtime: ServiceRuntime = request.app[RUNTIME_APP_KEY]
    status, code = await _service_status(runtime)
    return web.json_response(status, status=code)


_SNAPSHOT_TIMEOUT = 5  # seconds


async def _service_status(runtime: ServiceRuntime) -> tuple[dict, int]:
    """Compute a machine-readable service health snapshot (with timeout)."""
    legacy_enabled = runtime.bridge is not None

    snapshot = None
    if runtime.infrastructure:
        try:
            snapshot = await asyncio.wait_for(
                runtime.infrastructure.snapshot(),
                timeout=_SNAPSHOT_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning("Infrastructure snapshot timed out after %ds", _SNAPSHOT_TIMEOUT)
        except Exception as exc:
            logger.error("Infrastructure snapshot failed: %s: %s", type(exc).__name__, exc)

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


async def _metrics_handler(request: web.Request) -> web.Response:
    """Prometheus metrics endpoint."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return web.Response(
            body=generate_latest(),
            content_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return web.Response(text="# prometheus_client not installed\n", content_type="text/plain")


async def _knowledge_search_handler(request: web.Request) -> web.Response:
    """HTTP endpoint for knowledge base search (for non-Redis clients like Arclane)."""
    runtime: ServiceRuntime = request.app[RUNTIME_APP_KEY]
    platform = getattr(runtime.infrastructure, "platform", None) if runtime.infrastructure else None
    kb = getattr(platform, "_knowledge_base", None) if platform else None
    if not kb:
        return web.json_response({"items": [], "error": "knowledge base not available"})

    query = request.query.get("q", "")
    limit = min(int(request.query.get("limit", "10")), 50)

    results = kb.query_knowledge(query, max_results=limit)
    items = [
        {
            "id": item.id,
            "content": str(item.content)[:1000],
            "source": item.source,
            "type": item.knowledge_type.value,
            "confidence": item.confidence,
        }
        for item in results
    ]
    return web.json_response({"items": items, "count": len(items)})


async def _knowledge_add_handler(request: web.Request) -> web.Response:
    """HTTP endpoint for adding knowledge (for non-Redis clients like Arclane)."""
    runtime: ServiceRuntime = request.app[RUNTIME_APP_KEY]
    platform = getattr(runtime.infrastructure, "platform", None) if runtime.infrastructure else None
    kb = getattr(platform, "_knowledge_base", None) if platform else None
    if not kb:
        return web.json_response({"error": "knowledge base not available"}, status=503)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    from nexus.memory.knowledge_base import KnowledgeType
    content = data.get("content", "")
    ktype_str = data.get("knowledge_type", "factual").lower()
    ktype_map = {t.value: t for t in KnowledgeType}
    ktype = ktype_map.get(ktype_str, KnowledgeType.FACTUAL)

    kid = kb.add_knowledge(
        content=content,
        knowledge_type=ktype,
        source=data.get("source", "api"),
        confidence=data.get("confidence", 0.8),
        context_tags=data.get("context_tags", []),
    )
    return web.json_response({"id": kid, "success": True})


async def _generate_handler(request: web.Request) -> web.Response:
    """Shared LLM gateway — any product can route LLM calls through Nexus.

    Provides: multi-provider fallback, cost tracking, model selection,
    ensemble strategies, and a single point for LLM observability.

    POST /api/generate
    Body: {
        "prompt": "...",
        "model": "optional model name",
        "strategy": "simple_best|cascading|cost_optimized|...",
        "system_prompt": "optional system prompt",
        "max_tokens": 4096,
        "temperature": 0.7,
        "source": "arclane|trendscope|csuite|..."
    }
    """
    runtime: ServiceRuntime = request.app[RUNTIME_APP_KEY]
    platform = getattr(runtime.infrastructure, "platform", None) if runtime.infrastructure else None
    if not platform:
        return web.json_response({"error": "platform not initialized"}, status=503)

    # Try provider pool first (multi-instance Ollama + API routing)
    try:
        from nexus.providers.pool import get_provider_pool
        pool = get_provider_pool()
        if pool._instances:
            data_peek = await request.json()
            pool_result = await pool.generate(
                prompt=data_peek.get("prompt", ""),
                source=data_peek.get("source", "default"),
                system_prompt=data_peek.get("system_prompt"),
                model=data_peek.get("model"),
                max_tokens=data_peek.get("max_tokens", 4096),
            )
            if pool_result:
                pool_result["source"] = data_peek.get("source", "unknown")
                pool_result["routed_via"] = "pool"
                return web.json_response(pool_result)
            # Pool failed — fall through to platform
    except Exception as e:
        logger.debug("Provider pool routing skipped: %s", e)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    prompt = data.get("prompt", "")
    if not prompt:
        return web.json_response({"error": "prompt is required"}, status=400)

    import time as _time
    t0 = _time.perf_counter()

    try:
        # Use the platform's query method which handles ensemble, fallback, etc.
        result = await platform.query(
            prompt=prompt,
            strategy=data.get("strategy"),
            model=data.get("model"),
            system_prompt=data.get("system_prompt"),
            max_tokens=data.get("max_tokens", 4096),
        )

        latency_ms = (_time.perf_counter() - t0) * 1000

        # Track per-source metrics
        source = data.get("source", "unknown")
        try:
            if _PROM_ENABLED:
                from prometheus_client import Counter
                _llm_counter = Counter(
                    "nexus_llm_requests_total", "LLM requests via gateway",
                    ["source", "model"], registry=None,
                )
        except Exception:
            pass

        response = {
            "content": result.get("content", ""),
            "model": result.get("model_name", "unknown"),
            "provider": result.get("provider", "unknown"),
            "tokens_used": result.get("tokens_used", 0),
            "latency_ms": round(latency_ms, 1),
            "strategy_used": result.get("strategy_used", ""),
            "cached": result.get("cached", False),
            "source": source,
        }
        return web.json_response(response)

    except Exception as e:
        logger.error("LLM generate failed: %s: %s", type(e).__name__, e)
        return web.json_response({"error": str(e)}, status=500)


async def _list_models_handler(request: web.Request) -> web.Response:
    """List available models with their profiles."""
    runtime: ServiceRuntime = request.app[RUNTIME_APP_KEY]
    platform = getattr(runtime.infrastructure, "platform", None) if runtime.infrastructure else None

    models = []
    try:
        from nexus.providers.adapters.registry import MODEL_REGISTRY
        from nexus.providers.adapters.model_profiles import get_model_profile

        for name, info in MODEL_REGISTRY.items():
            entry = {
                "name": name,
                "display_name": info.display_name,
                "provider": info.provider.value if hasattr(info.provider, "value") else str(info.provider),
                "context_window": info.context_window,
                "cost_per_1k_input": info.cost_per_1k_input,
                "cost_per_1k_output": info.cost_per_1k_output,
                "quality_tier": info.quality_tier,
            }
            profile = get_model_profile(name)
            if profile:
                entry["strengths"] = profile.strengths[:3]
                entry["specializations"] = profile.specializations[:3]
            models.append(entry)
    except Exception as e:
        logger.debug("Model listing failed: %s", e)

    return web.json_response({
        "models": models[:100],
        "count": len(models),
    })


async def _pool_status_handler(request: web.Request) -> web.Response:
    """Get provider pool status — instances, health, load, workstream rules."""
    try:
        from nexus.providers.pool import get_provider_pool
        pool = get_provider_pool()
        return web.json_response(pool.get_pool_status())
    except ImportError:
        return web.json_response({"error": "pool not available"})


_PROM_ENABLED = False
try:
    from prometheus_client import Counter as _Counter
    _PROM_ENABLED = True
except ImportError:
    pass


def create_health_app(runtime: ServiceRuntime) -> web.Application:
    """Create the HTTP app exposing health/readiness/metrics/knowledge/LLM endpoints."""
    app = web.Application()
    app[RUNTIME_APP_KEY] = runtime
    app.router.add_get("/health", _health_handler)
    app.router.add_get("/ready", _ready_handler)
    app.router.add_get("/metrics", _metrics_handler)
    app.router.add_get("/api/knowledge/search", _knowledge_search_handler)
    app.router.add_post("/api/knowledge", _knowledge_add_handler)
    app.router.add_post("/api/generate", _generate_handler)
    app.router.add_get("/api/models", _list_models_handler)
    app.router.add_get("/api/pool/status", _pool_status_handler)
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


async def _periodic_discovery_refresh(platform, interval_seconds: int) -> None:
    """Periodically refresh model and resource discovery.

    Replaces the old AutonomousCOO polling loop with something useful:
    discovers new models, datasets, and resources at a configurable interval.
    """
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            logger.info("Running periodic discovery refresh...")
            model_count = await platform.discover_models()
            resource_counts = await platform.discover_resources()
            logger.info(
                "Discovery refresh complete: %d models, resources=%s",
                model_count, resource_counts,
            )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning("Discovery refresh failed (will retry): %s", e)


async def main() -> None:
    """Start Nexus as shared infrastructure."""
    global _bridge, _infrastructure

    log_format = os.environ.get("LOG_FORMAT", "text")
    log_level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO"))

    if log_format == "json":
        # Structured JSON logging for production
        import json as _json

        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                return _json.dumps({
                    "ts": self.formatTime(record),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                    "module": record.module,
                    "func": record.funcName,
                }, default=str)

        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logging.root.addHandler(handler)
        logging.root.setLevel(log_level)
    else:
        logging.basicConfig(
            level=log_level,
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

    platform = _infrastructure.platform

    # Initialize persistent knowledge base
    try:
        from nexus.memory.knowledge_base import KnowledgeBase
        kb_path = os.path.expanduser("~/.nexus/knowledge.db")
        kb = KnowledgeBase(db_path=kb_path)
        kb.initialize()
        platform._knowledge_base = kb
        logger.info("Knowledge base initialized: %d items (persistent at %s)", len(kb.knowledge_items), kb_path)
    except Exception as e:
        logger.warning("Knowledge base initialization failed: %s", e)

    # Wire intelligence sources (TS, KH, Production Line) into discovery refresh
    try:
        kb = getattr(platform, "_knowledge_base", None)
        rd = getattr(platform, "_resource_discovery", None)
        if kb and rd:
            from nexus.discovery.intelligence_sources import (
                TrendscopeDiscoverySource,
                KHDiscoverySource,
                ContentProductionDiscoverySource,
            )

            TrendscopeDiscoverySource(rd, kb)
            KHDiscoverySource(rd, kb)
            ContentProductionDiscoverySource(rd, kb)
            logger.info("Intelligence sources registered (Trendscope, KH, Content Production)")
        else:
            logger.warning("Intelligence sources skipped: kb=%s, rd=%s", bool(kb), bool(rd))
    except Exception as e:
        logger.warning("Intelligence source registration failed (non-fatal): %s", e)

    # Bootstrap model registry — register persisted models + discover new ones
    try:
        # First: register all previously discovered models from persistence
        if hasattr(platform, '_model_discovery') and platform._model_discovery:
            persisted = platform._model_discovery.register_persisted_models()
            logger.info("Registered %d persisted models into MODEL_REGISTRY", persisted)
        else:
            # Force discovery path initialization to get model_discovery
            await platform._ensure_model_discovery_components()
            if platform._model_discovery:
                persisted = platform._model_discovery.register_persisted_models()
                logger.info("Registered %d persisted models into MODEL_REGISTRY", persisted)

        # Then: discover any new models from APIs
        model_count = await platform.discover_models()
        logger.info("Model discovery: %d new models found", model_count)

        # Report total
        from nexus.providers.adapters.registry import MODEL_REGISTRY
        logger.info("MODEL_REGISTRY total: %d models available for routing", len(MODEL_REGISTRY))
    except Exception as e:
        logger.warning("Model bootstrap failed (non-fatal): %s", e)

    health_runner = await _start_health_server(health_port, runtime)

    # Start Nexus service handler — exposes discovery, ensemble, research,
    # knowledge services to C-Suite via Redis request/response
    _service_handler = None
    try:
        import redis.asyncio as aioredis
        from nexus.coo.service_handler import NexusServiceHandler

        service_redis = aioredis.from_url(redis_url)
        await service_redis.ping()
        _service_handler = NexusServiceHandler(
            platform=_infrastructure.platform,
            redis_client=service_redis,
            channel_prefix=channel_prefix,
        )
        await _service_handler.start()
        runtime.service_handler = _service_handler
        logger.info("Nexus service handler started (discovery, ensemble, research, knowledge)")
    except Exception as e:
        logger.warning("Service handler not started: %s", e)

    # Start C-Suite bridge listener (backward compatible)
    if legacy_coo_enabled:
        from nexus.coo.csuite_bridge import CSuiteBridgeListener, CSuiteBridgeConfig

        _bridge = CSuiteBridgeListener(
            coo=None,  # No COO needed — bridge handles messages directly
            config=CSuiteBridgeConfig(
                redis_url=redis_url,
                channel_prefix=channel_prefix,
            ),
        )
        runtime.bridge = _bridge
        await _bridge.start_listening()
        logger.info("C-Suite bridge listener started")

    # Start periodic discovery refresh (replaces the old COO polling loop)
    discovery_interval = int(os.environ.get("DISCOVERY_REFRESH_INTERVAL", "3600"))
    _refresh_task = asyncio.create_task(
        _periodic_discovery_refresh(_infrastructure.platform, discovery_interval)
    )

    # Keep running until SIGTERM/SIGINT
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    if os.name != "nt":
        # Unix: use proper signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
    else:
        # Windows: signal handlers not supported in asyncio, use thread
        import threading

        def _win_signal_handler(signum, frame):
            loop.call_soon_threadsafe(stop.set)

        signal.signal(signal.SIGTERM, _win_signal_handler)
        signal.signal(signal.SIGINT, _win_signal_handler)

    logger.info("Nexus service ready")
    await stop.wait()

    logger.info("Shutting down Nexus service")
    _refresh_task.cancel()
    shutdown_timeout = int(os.environ.get("SHUTDOWN_TIMEOUT", "10"))
    try:
        async with asyncio.timeout(shutdown_timeout):
            if _service_handler:
                await _service_handler.stop()
            if _bridge:
                await _bridge.disconnect()
    except TimeoutError:
        logger.warning("Graceful shutdown timed out after %ds, forcing exit", shutdown_timeout)
    await health_runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
