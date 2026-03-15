"""
Enhanced Flask API with authentication, caching, cost tracking, and monitoring.

This module preserves the historical import surface (`app`, `api_key_manager`,
`cache_manager`, `cost_tracker`, `metrics`) while moving expensive component
construction behind a lazy runtime. Importing the module no longer initializes
the database, trackers, or cognitive engine.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Tuple
import os
import sys
import time
import logging

from flask import Flask, Response, request, jsonify, g
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from nexus.core.core_engine import CognitiveCore
from nexus.core.auth import AuthMiddleware, UserRole
from nexus.core.auth.persistent_api_key_manager import PersistentAPIKeyManager
from nexus.core.cache import CacheManager, MemoryBackend
from nexus.core.tracking.persistent_cost_tracker import PersistentCostTracker
from nexus.core.tracking.persistent_usage_tracker import PersistentUsageTracker
from nexus.core.monitoring import MetricsCollector
from nexus.core.database import init_db, resolve_default_db_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 10_000


class EnhancedApiRuntime:
    """Lazy runtime container for the enhanced API dependencies."""

    def __init__(self) -> None:
        self._initialized = False
        self._cognitive_engine: CognitiveCore | None = None
        self._api_key_manager: PersistentAPIKeyManager | None = None
        self._auth_middleware: AuthMiddleware | None = None
        self._cache_manager: CacheManager | None = None
        self._cost_tracker: PersistentCostTracker | None = None
        self._usage_tracker: PersistentUsageTracker | None = None
        self._metrics: MetricsCollector | None = None
        self._model_ensemble: list[Any] | None = None
        self._strategic_ensemble: Any | None = None
        self._strategy_type: Any | None = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """Initialize persistence-backed API components once."""
        if self._initialized:
            return

        db_path = resolve_default_db_path()
        logger.info("Initializing enhanced API persistence layer at %s", db_path)
        init_db(db_path=db_path, echo=False)

        self._cognitive_engine = CognitiveCore()
        self._api_key_manager = PersistentAPIKeyManager()
        self._auth_middleware = AuthMiddleware(self._api_key_manager)
        self._cache_manager = CacheManager(MemoryBackend(), default_ttl=3600)
        self._cost_tracker = PersistentCostTracker(budget_limit_usd=100.0, alert_threshold=0.8)
        self._usage_tracker = PersistentUsageTracker()
        self._metrics = MetricsCollector()
        model_names = [model.name for model in self.model_ensemble]
        self._metrics.set_system_info(
            version="0.2.0",
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            models=model_names,
        )
        self._metrics.update_ensemble_size(len(model_names))
        self._initialized = True
        logger.info("Enhanced API runtime initialized")

    @property
    def cognitive_engine(self) -> CognitiveCore:
        self.initialize()
        assert self._cognitive_engine is not None
        return self._cognitive_engine

    @property
    def api_key_manager(self) -> PersistentAPIKeyManager:
        self.initialize()
        assert self._api_key_manager is not None
        return self._api_key_manager

    @property
    def auth_middleware(self) -> AuthMiddleware:
        self.initialize()
        assert self._auth_middleware is not None
        return self._auth_middleware

    @property
    def cache_manager(self) -> CacheManager:
        self.initialize()
        assert self._cache_manager is not None
        return self._cache_manager

    @property
    def cost_tracker(self) -> PersistentCostTracker:
        self.initialize()
        assert self._cost_tracker is not None
        return self._cost_tracker

    @property
    def usage_tracker(self) -> PersistentUsageTracker:
        self.initialize()
        assert self._usage_tracker is not None
        return self._usage_tracker

    @property
    def metrics(self) -> MetricsCollector:
        self.initialize()
        assert self._metrics is not None
        return self._metrics

    @property
    def model_ensemble(self) -> list[Any]:
        if self._model_ensemble is None:
            from nexus.core.ensemble_core_v2 import model_ensemble

            self._model_ensemble = model_ensemble
        return self._model_ensemble

    @property
    def strategic_ensemble(self) -> Any:
        if self._strategic_ensemble is None:
            from nexus.core.strategic_ensemble import strategic_ensemble

            self._strategic_ensemble = strategic_ensemble
        return self._strategic_ensemble

    @property
    def strategy_type(self) -> Any:
        if self._strategy_type is None:
            from nexus.core.strategic_ensemble import StrategyType

            self._strategy_type = StrategyType
        return self._strategy_type

    def configured_model_count(self) -> int:
        """Return configured model count without importing the ensemble runtime."""
        if self._model_ensemble is not None:
            return len(self._model_ensemble)
        try:
            import yaml

            with open("config/config.yaml", "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle) or {}
            return len(config.get("model_ensemble", {}).get("models", []))
        except Exception:
            return 0

    def status(self) -> dict[str, Any]:
        """Return machine-readable runtime status."""
        return {
            "initialized": self._initialized,
            "models_available": self.configured_model_count(),
            "components": {
                "cognitive_engine": self._cognitive_engine is not None,
                "api_key_manager": self._api_key_manager is not None,
                "auth_middleware": self._auth_middleware is not None,
                "cache_manager": self._cache_manager is not None,
                "cost_tracker": self._cost_tracker is not None,
                "usage_tracker": self._usage_tracker is not None,
                "metrics": self._metrics is not None,
                "model_ensemble": self._model_ensemble is not None,
                "strategic_ensemble": self._strategic_ensemble is not None,
            },
        }

    def reset_for_testing(self) -> None:
        """Dispose lazy state so tests can isolate runtime initialization."""
        try:
            from nexus.core.database.connection import get_db

            db = get_db()
            if db is not None:
                db.close()
        except Exception:
            pass

        self._initialized = False
        self._cognitive_engine = None
        self._api_key_manager = None
        self._auth_middleware = None
        self._cache_manager = None
        self._cost_tracker = None
        self._usage_tracker = None
        self._metrics = None
        self._model_ensemble = None
        self._strategic_ensemble = None
        self._strategy_type = None


class _LazyComponentProxy:
    """Expose lazy runtime-backed components through the legacy module globals."""

    def __init__(self, getter: Callable[[], Any]):
        self._getter = getter

    def __getattr__(self, name: str) -> Any:
        return getattr(self._getter(), name)


class _LazyAuthFacade:
    """Delay AuthMiddleware construction until an authenticated route is used."""

    def __init__(self, runtime: EnhancedApiRuntime):
        self._runtime = runtime

    def require_api_key(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            decorated = self._runtime.auth_middleware.require_api_key(func)
            return decorated(*args, **kwargs)

        return wrapper

    def require_role(self, required_role: UserRole) -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                decorated = self._runtime.auth_middleware.require_role(required_role)(func)
                return decorated(*args, **kwargs)

            return wrapper

        return decorator


def _ok(**data: Any) -> dict[str, Any]:
    return {"ok": True, **data}


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **extra}


def _require_json_body() -> tuple[dict[str, Any], int] | None:
    """Validate that the request contains a JSON body."""
    if not request.is_json:
        return _error("Content-Type must be application/json"), 400
    data = request.get_json(silent=True)
    if data is None:
        return _error("Request body must be valid JSON"), 400
    return None


def _validate_prompt_payload(data: dict[str, Any]) -> tuple[str | None, tuple[dict[str, Any], int] | None]:
    """Validate a request payload containing an 'input' field."""
    if "input" not in data:
        return None, (_error("Missing required field: 'input'"), 400)

    prompt = data.get("input")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, (_error("Input must be a non-empty string"), 400)

    if len(prompt) > MAX_INPUT_LENGTH:
        return None, (_error(f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters"), 400)

    return prompt, None


runtime = EnhancedApiRuntime()
auth_middleware = _LazyAuthFacade(runtime)
api_key_manager = _LazyComponentProxy(lambda: runtime.api_key_manager)
cache_manager = _LazyComponentProxy(lambda: runtime.cache_manager)
cost_tracker = _LazyComponentProxy(lambda: runtime.cost_tracker)
usage_tracker = _LazyComponentProxy(lambda: runtime.usage_tracker)
metrics = _LazyComponentProxy(lambda: runtime.metrics)
cognitive_engine = _LazyComponentProxy(lambda: runtime.cognitive_engine)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    """Health check endpoint."""
    return jsonify(_ok(
        status="healthy",
        service="Nexus Enhanced API",
        runtime=runtime.status(),
    )), 200


@app.route("/metrics", methods=["GET"])
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(runtime.metrics.registry), mimetype=CONTENT_TYPE_LATEST)


@app.route("/status", methods=["GET"])
def status() -> Tuple[Response, int]:
    """System status endpoint with stats."""
    cache_stats = runtime.cache_manager.get_stats()
    budget_status = runtime.cost_tracker.get_budget_status()

    return jsonify(_ok(
        status="operational",
        models_available=runtime.configured_model_count(),
        runtime=runtime.status(),
        cache=cache_stats,
        budget=budget_status,
    )), 200


@app.route("/auth/register", methods=["POST"])
def register_user() -> Tuple[Response, int]:
    """Register a new user."""
    try:
        invalid = _require_json_body()
        if invalid:
            payload, code = invalid
            return jsonify(payload), code

        data = request.get_json(silent=True) or {}
        username = data.get("username")
        email = data.get("email")
        role = data.get("role", "user")

        if not username or not email:
            return jsonify(_error("Username and email required")), 400

        user = runtime.api_key_manager.create_user(username, email, role)
        api_key, key_obj = runtime.api_key_manager.generate_key(
            user_id=user.user_id,
            name=f"{username}'s key",
            rate_limit=1000,
            expires_in_days=365,
        )

        logger.info("Registered user %s with API key", username)
        return jsonify(_ok(
            user_id=user.user_id,
            username=user.username,
            role=user.role.value,
            api_key=api_key,
            key_id=key_obj.key_id,
            rate_limit=key_obj.rate_limit,
            message="User registered successfully. Save your API key securely.",
        )), 201

    except Exception as exc:
        logger.error("Error registering user: %s", exc, exc_info=True)
        return jsonify(_error(str(exc))), 500


@app.route("/auth/keys", methods=["GET"])
@auth_middleware.require_api_key
def list_keys() -> Tuple[Response, int]:
    """List API keys for current user."""
    user_id = g.api_key.user_id
    keys = runtime.api_key_manager.list_keys(user_id)

    keys_data = [
        {
            "key_id": key.key_id,
            "name": key.name,
            "created_at": key.created_at.isoformat(),
            "last_used": key.last_used.isoformat() if key.last_used else None,
            "is_active": key.is_active,
            "usage_count": key.usage_count,
        }
        for key in keys
    ]

    return jsonify(_ok(keys=keys_data)), 200


@app.route("/think", methods=["POST"])
@auth_middleware.require_api_key
def think() -> Tuple[Response, int]:
    """Process input through the cognitive engine without ensemble routing."""
    start_time = time.time()

    try:
        invalid = _require_json_body()
        if invalid:
            payload, code = invalid
            return jsonify(payload), code

        data = request.get_json(silent=True) or {}
        input_data, error = _validate_prompt_payload(data)
        if error:
            payload, code = error
            return jsonify(payload), code

        result = runtime.cognitive_engine.think(input_data)
        duration = time.time() - start_time
        runtime.metrics.record_request("/think", "POST", 200, duration)

        return jsonify(_ok(
            response=result,
            status="success",
            processing_time_ms=round(duration * 1000, 2),
        )), 200

    except Exception as exc:
        logger.error("Error in /think: %s", exc, exc_info=True)
        duration = time.time() - start_time
        runtime.metrics.record_request("/think", "POST", 500, duration)
        return jsonify(_error(str(exc))), 500


@app.route("/ensemble", methods=["POST"])
@auth_middleware.require_api_key
def ensemble() -> Tuple[Response, int]:
    """Process input through strategic ensemble inference with caching and tracking."""
    start_time = time.time()

    try:
        invalid = _require_json_body()
        if invalid:
            payload, code = invalid
            return jsonify(payload), code

        data = request.get_json(silent=True) or {}
        prompt, error = _validate_prompt_payload(data)
        if error:
            payload, code = error
            return jsonify(payload), code

        use_cache = data.get("cache", True)
        strategy_name = data.get("strategy", "simple_best")

        try:
            strategy_type = runtime.strategy_type(strategy_name.lower())
        except ValueError:
            return jsonify(_error(
                f"Invalid strategy '{strategy_name}'",
                valid_strategies=runtime.strategic_ensemble.get_available_strategies(),
            )), 400

        cache_key = f"{prompt}:{strategy_name}"
        if use_cache:
            cached = runtime.cache_manager.get_response(cache_key)
            if cached:
                runtime.metrics.record_cache_hit()
                duration = time.time() - start_time
                runtime.metrics.record_request("/ensemble", "POST", 200, duration)
                runtime.usage_tracker.record_request(
                    endpoint="/ensemble",
                    user_id=g.api_key.user_id,
                    latency_ms=duration * 1000,
                    cached=True,
                    success=True,
                )
                cached["cached"] = True
                cached["processing_time_ms"] = round(duration * 1000, 2)
                return jsonify(cached), 200

        runtime.metrics.record_cache_miss()

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            ensemble_result = loop.run_until_complete(
                runtime.strategic_ensemble.execute_with_strategy(
                    runtime.model_ensemble, prompt, strategy_type
                )
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        runtime.cost_tracker.record_cost(
            model_name=ensemble_result.model_name,
            provider=ensemble_result.provider,
            tokens_used=0,
            cost_usd=ensemble_result.total_cost,
            user_id=g.api_key.user_id,
        )

        runtime.metrics.record_model_request(
            model_name=ensemble_result.model_name,
            provider=ensemble_result.provider,
            latency_ms=ensemble_result.total_latency_ms / ensemble_result.models_queried,
            tokens_used=0,
            cost_usd=ensemble_result.total_cost,
            success=True,
        )

        budget_status = runtime.cost_tracker.get_budget_status()
        runtime.metrics.update_budget_metrics(
            budget_limit=runtime.cost_tracker.budget_limit,
            current_spend=budget_status["current_spend"],
        )

        result = {
            "response": ensemble_result.content,
            "model": ensemble_result.model_name,
            "provider": ensemble_result.provider,
            "score": round(ensemble_result.score, 3),
            "confidence": round(ensemble_result.confidence, 3),
            "strategy_used": ensemble_result.strategy_used,
            "models_queried": ensemble_result.models_queried,
            "total_cost_usd": round(ensemble_result.total_cost, 4),
            "avg_latency_ms": round(ensemble_result.total_latency_ms / ensemble_result.models_queried, 2),
            "cached": False,
            "status": "success",
            "metadata": ensemble_result.metadata,
        }

        if use_cache:
            runtime.cache_manager.set_response(cache_key, result)

        duration = time.time() - start_time
        runtime.usage_tracker.record_request(
            endpoint="/ensemble",
            user_id=g.api_key.user_id,
            model_name=ensemble_result.model_name,
            latency_ms=duration * 1000,
            cached=False,
            success=True,
        )
        result["processing_time_ms"] = round(duration * 1000, 2)
        runtime.metrics.record_request("/ensemble", "POST", 200, duration)
        return jsonify(result), 200

    except Exception as exc:
        logger.error("Error in /ensemble: %s", exc, exc_info=True)
        duration = time.time() - start_time
        runtime.metrics.record_request("/ensemble", "POST", 500, duration)
        runtime.usage_tracker.record_request(
            endpoint="/ensemble",
            user_id=g.api_key.user_id if hasattr(g, "api_key") else None,
            latency_ms=duration * 1000,
            cached=False,
            success=False,
            error_type=type(exc).__name__,
        )
        return jsonify(_error(str(exc))), 500


@app.route("/costs/summary", methods=["GET"])
@auth_middleware.require_api_key
def cost_summary() -> Tuple[Response, int]:
    """Get cost summary for current user."""
    summary = runtime.cost_tracker.get_summary(user_id=g.api_key.user_id)
    return jsonify(_ok(
        total_cost=round(summary.total_cost, 4),
        total_requests=summary.total_requests,
        total_tokens=summary.total_tokens,
        cost_by_model={key: round(value, 4) for key, value in summary.cost_by_model.items()},
        period_start=summary.period_start.isoformat() if summary.period_start else None,
        period_end=summary.period_end.isoformat() if summary.period_end else None,
    )), 200


@app.route("/costs/budget", methods=["GET"])
@auth_middleware.require_api_key
@auth_middleware.require_role(UserRole.ADMIN)
def budget_status() -> Tuple[Response, int]:
    """Get budget status (admin only)."""
    return jsonify(runtime.cost_tracker.get_budget_status()), 200


@app.route("/strategies", methods=["GET"])
def list_strategies() -> Tuple[Response, int]:
    """List available ensemble strategies."""
    strategies = {
        "simple_best": {
            "name": "Simple Best",
            "description": "Select highest scoring model (default)",
            "use_case": "General purpose, fastest",
        },
        "weighted_voting": {
            "name": "Weighted Voting",
            "description": "Combine model trust weights with quality scores",
            "use_case": "When you have trusted models with different reliability",
        },
        "cascading": {
            "name": "Cascading",
            "description": "Try cheaper models first, escalate if needed",
            "use_case": "Cost optimization - can save up to 90% on simple queries",
        },
        "dynamic_weight": {
            "name": "Dynamic Weights",
            "description": "Adaptive learning from historical performance",
            "use_case": "Long-running systems that learn over time",
        },
        "majority_voting": {
            "name": "Majority Voting",
            "description": "Consensus-based selection",
            "use_case": "High-reliability scenarios requiring agreement",
        },
        "cost_optimized": {
            "name": "Cost Optimized",
            "description": "Balance quality and cost for best value",
            "use_case": "Budget-constrained applications",
        },
    }
    return jsonify(_ok(strategies=strategies, default="simple_best")), 200


@app.route("/analytics/usage", methods=["GET"])
@auth_middleware.require_api_key
@auth_middleware.require_role(UserRole.ADMIN)
def usage_analytics() -> Tuple[Response, int]:
    """Get usage analytics (admin only)."""
    hours = request.args.get("hours", 24, type=int)
    stats = runtime.usage_tracker.get_stats()
    hourly_stats = runtime.usage_tracker.get_hourly_stats(hours=hours)
    top_users = runtime.usage_tracker.get_top_users(limit=10)

    stats_dict = {
        "total_requests": stats.total_requests,
        "successful_requests": stats.successful_requests,
        "failed_requests": stats.failed_requests,
        "total_tokens": stats.total_tokens,
        "avg_latency_ms": round(stats.avg_latency_ms, 2),
        "cache_hit_rate": round(stats.cache_hit_rate, 2),
        "requests_by_endpoint": stats.requests_by_endpoint,
        "requests_by_model": stats.requests_by_model,
        "errors_by_type": stats.errors_by_type,
    }

    return jsonify(_ok(
        summary=stats_dict,
        hourly_stats=hourly_stats[:50],
        top_users=[{"user_id": uid, "request_count": count} for uid, count in top_users],
    )), 200


@app.errorhandler(404)
def not_found(error) -> Tuple[Response, int]:
    """Handle 404 errors."""
    return jsonify(_error("Endpoint not found")), 404


@app.errorhandler(500)
def internal_error(error) -> Tuple[Response, int]:
    """Handle 500 errors."""
    logger.error("Internal server error: %s", error)
    return jsonify(_error("Internal server error")), 500


if __name__ == "__main__":
    logger.info("Starting Nexus Enhanced API server on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
