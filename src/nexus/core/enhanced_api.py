"""
Enhanced Flask API with authentication, caching, cost tracking, and monitoring.
"""

from typing import Tuple
import time
import logging
from flask import Flask, request, jsonify, Response, g
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from nexus.core.core_engine import CognitiveCore
from nexus.core.ensemble_core_v2 import model_ensemble
from nexus.core.auth import AuthMiddleware, UserRole
from nexus.core.auth.persistent_api_key_manager import PersistentAPIKeyManager
from nexus.core.cache import CacheManager, MemoryBackend
from nexus.core.tracking.persistent_cost_tracker import PersistentCostTracker
from nexus.core.tracking.persistent_usage_tracker import PersistentUsageTracker
from nexus.core.monitoring import MetricsCollector
from nexus.core.strategic_ensemble import strategic_ensemble, StrategyType
from nexus.core.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize database (with persistence!)
# Database file will be created in data/thenexus.db
logger.info("Initializing database persistence layer...")
init_db(db_path="data/thenexus.db", echo=False)
logger.info("Database initialized successfully")

# Initialize components (now with persistence!)
cognitive_engine = CognitiveCore()
api_key_manager = PersistentAPIKeyManager()  # Database-backed
auth_middleware = AuthMiddleware(api_key_manager)
cache_manager = CacheManager(MemoryBackend(), default_ttl=3600)
cost_tracker = PersistentCostTracker(budget_limit_usd=100.0, alert_threshold=0.8)  # Database-backed
usage_tracker = PersistentUsageTracker()  # Database-backed
metrics = MetricsCollector()

# Set system info
import sys
metrics.set_system_info(
    version="0.2.0",
    python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    models=[m.name for m in model_ensemble]
)
metrics.update_ensemble_size(len(model_ensemble))


# Health and status endpoints
@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "TheNexus Enhanced API"}), 200


@app.route("/metrics", methods=["GET"])
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(metrics.registry), mimetype=CONTENT_TYPE_LATEST)


@app.route("/status", methods=["GET"])
def status() -> Tuple[Response, int]:
    """System status endpoint with stats."""
    cache_stats = cache_manager.get_stats()
    budget_status = cost_tracker.get_budget_status()
    
    return jsonify({
        "status": "operational",
        "models_available": len(model_ensemble),
        "cache": cache_stats,
        "budget": budget_status,
    }), 200


# Authentication endpoints
@app.route("/auth/register", methods=["POST"])
def register_user() -> Tuple[Response, int]:
    """Register a new user."""
    try:
        data = request.get_json()
        
        username = data.get("username")
        email = data.get("email")
        role = data.get("role", "user")
        
        if not username or not email:
            return jsonify({"error": "Username and email required"}), 400
        
        # Create user
        user = api_key_manager.create_user(username, email, role)
        
        # Generate API key
        api_key, key_obj = api_key_manager.generate_key(
            user_id=user.user_id,
            name=f"{username}'s key",
            rate_limit=1000,
            expires_in_days=365
        )
        
        logger.info(f"Registered user {username} with API key")
        
        return jsonify({
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "api_key": api_key,
            "key_id": key_obj.key_id,
            "rate_limit": key_obj.rate_limit,
            "message": "User registered successfully. Save your API key securely."
        }), 201
        
    except Exception as e:
        logger.error(f"Error registering user: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/auth/keys", methods=["GET"])
@auth_middleware.require_api_key
def list_keys() -> Tuple[Response, int]:
    """List API keys for current user."""
    user_id = g.api_key.user_id
    keys = api_key_manager.list_keys(user_id)
    
    keys_data = [
        {
            "key_id": k.key_id,
            "name": k.name,
            "created_at": k.created_at.isoformat(),
            "last_used": k.last_used.isoformat() if k.last_used else None,
            "is_active": k.is_active,
            "usage_count": k.usage_count,
        }
        for k in keys
    ]
    
    return jsonify({"keys": keys_data}), 200


# Core inference endpoints
@app.route("/think", methods=["POST"])
@auth_middleware.require_api_key
def think() -> Tuple[Response, int]:
    """
    Process input through cognitive engine (simple, no ensemble).
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or "input" not in data:
            return jsonify({"error": "Missing required field: 'input'"}), 400
        
        input_data = data.get("input")
        
        if not isinstance(input_data, str) or not input_data.strip():
            return jsonify({"error": "Input must be a non-empty string"}), 400
        
        if len(input_data) > 10000:
            return jsonify({"error": "Input exceeds maximum length"}), 400
        
        # Process through cognitive engine
        result = cognitive_engine.think(input_data)
        
        # Record metrics
        duration = time.time() - start_time
        metrics.record_request("/think", "POST", 200, duration)
        
        return jsonify({
            "response": result,
            "status": "success",
            "processing_time_ms": round(duration * 1000, 2)
        }), 200
        
    except Exception as e:
        logger.error(f"Error in /think: {e}", exc_info=True)
        duration = time.time() - start_time
        metrics.record_request("/think", "POST", 500, duration)
        return jsonify({"error": str(e)}), 500


@app.route("/ensemble", methods=["POST"])
@auth_middleware.require_api_key
def ensemble() -> Tuple[Response, int]:
    """
    Process input through strategic ensemble inference with caching and cost tracking.

    Accepts a "strategy" parameter to select ensemble selection strategy:
    - weighted_voting: Combine model weights with quality scores
    - cascading: Cost-optimized with early stopping
    - dynamic_weight: Adaptive learning from history
    - majority_voting: Consensus-based selection
    - cost_optimized: Quality-to-cost ratio optimization
    - simple_best: Default - just pick highest score
    """
    start_time = time.time()

    try:
        data = request.get_json()

        if not data or "input" not in data:
            return jsonify({"error": "Missing required field: 'input'"}), 400

        prompt = data.get("input")
        use_cache = data.get("cache", True)
        strategy_name = data.get("strategy", "simple_best")

        if not isinstance(prompt, str) or not prompt.strip():
            return jsonify({"error": "Input must be a non-empty string"}), 400

        if len(prompt) > 10000:
            return jsonify({"error": "Input exceeds maximum length"}), 400

        # Parse strategy
        try:
            strategy_type = StrategyType(strategy_name.lower())
        except ValueError:
            return jsonify({
                "error": f"Invalid strategy '{strategy_name}'",
                "valid_strategies": strategic_ensemble.get_available_strategies()
            }), 400

        # Check cache
        cache_key = f"{prompt}:{strategy_name}"
        if use_cache:
            cached = cache_manager.get_response(cache_key)
            if cached:
                metrics.record_cache_hit()
                duration = time.time() - start_time
                metrics.record_request("/ensemble", "POST", 200, duration)

                # Record usage
                usage_tracker.record_request(
                    endpoint="/ensemble",
                    user_id=g.api_key.user_id,
                    latency_ms=duration * 1000,
                    cached=True,
                    success=True
                )

                cached["cached"] = True
                cached["processing_time_ms"] = round(duration * 1000, 2)
                return jsonify(cached), 200

        metrics.record_cache_miss()

        # Run strategic ensemble inference
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        ensemble_result = loop.run_until_complete(
            strategic_ensemble.execute_with_strategy(
                model_ensemble, prompt, strategy_type
            )
        )

        # Record costs
        cost_tracker.record_cost(
            model_name=ensemble_result.model_name,
            provider=ensemble_result.provider,
            tokens_used=0,  # We don't have per-model token counts in ensemble_result
            cost_usd=ensemble_result.total_cost,
            user_id=g.api_key.user_id
        )

        # Record model metrics (for selected model)
        metrics.record_model_request(
            model_name=ensemble_result.model_name,
            provider=ensemble_result.provider,
            latency_ms=ensemble_result.total_latency_ms / ensemble_result.models_queried,
            tokens_used=0,
            cost_usd=ensemble_result.total_cost,
            success=True
        )

        # Update budget metrics
        budget_status = cost_tracker.get_budget_status()
        metrics.update_budget_metrics(
            budget_limit=cost_tracker.budget_limit,
            current_spend=budget_status["current_spend"]
        )

        # Prepare response
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
            "metadata": ensemble_result.metadata
        }

        # Cache the response
        if use_cache:
            cache_manager.set_response(cache_key, result)

        # Record usage
        duration = time.time() - start_time
        usage_tracker.record_request(
            endpoint="/ensemble",
            user_id=g.api_key.user_id,
            model_name=ensemble_result.model_name,
            latency_ms=duration * 1000,
            cached=False,
            success=True
        )

        # Record metrics
        result["processing_time_ms"] = round(duration * 1000, 2)
        metrics.record_request("/ensemble", "POST", 200, duration)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in /ensemble: {e}", exc_info=True)
        duration = time.time() - start_time
        metrics.record_request("/ensemble", "POST", 500, duration)

        # Record failed usage
        usage_tracker.record_request(
            endpoint="/ensemble",
            user_id=g.api_key.user_id if hasattr(g, 'api_key') else None,
            latency_ms=duration * 1000,
            cached=False,
            success=False,
            error_type=type(e).__name__
        )

        return jsonify({"error": str(e)}), 500


# Cost tracking endpoints
@app.route("/costs/summary", methods=["GET"])
@auth_middleware.require_api_key
def cost_summary() -> Tuple[Response, int]:
    """Get cost summary for current user."""
    user_id = g.api_key.user_id
    summary = cost_tracker.get_summary(user_id=user_id)
    
    return jsonify({
        "total_cost": round(summary.total_cost, 4),
        "total_requests": summary.total_requests,
        "total_tokens": summary.total_tokens,
        "cost_by_model": {k: round(v, 4) for k, v in summary.cost_by_model.items()},
        "period_start": summary.period_start.isoformat() if summary.period_start else None,
        "period_end": summary.period_end.isoformat() if summary.period_end else None,
    }), 200


@app.route("/costs/budget", methods=["GET"])
@auth_middleware.require_api_key
@auth_middleware.require_role(UserRole.ADMIN)
def budget_status() -> Tuple[Response, int]:
    """Get budget status (admin only)."""
    status = cost_tracker.get_budget_status()
    return jsonify(status), 200


# Strategy endpoints
@app.route("/strategies", methods=["GET"])
def list_strategies() -> Tuple[Response, int]:
    """List available ensemble strategies."""
    strategies = {
        "simple_best": {
            "name": "Simple Best",
            "description": "Select highest scoring model (default)",
            "use_case": "General purpose, fastest"
        },
        "weighted_voting": {
            "name": "Weighted Voting",
            "description": "Combine model trust weights with quality scores",
            "use_case": "When you have trusted models with different reliability"
        },
        "cascading": {
            "name": "Cascading",
            "description": "Try cheaper models first, escalate if needed",
            "use_case": "Cost optimization - can save up to 90% on simple queries"
        },
        "dynamic_weight": {
            "name": "Dynamic Weights",
            "description": "Adaptive learning from historical performance",
            "use_case": "Long-running systems that learn over time"
        },
        "majority_voting": {
            "name": "Majority Voting",
            "description": "Consensus-based selection",
            "use_case": "High-reliability scenarios requiring agreement"
        },
        "cost_optimized": {
            "name": "Cost Optimized",
            "description": "Balance quality and cost for best value",
            "use_case": "Budget-constrained applications"
        }
    }
    return jsonify({
        "strategies": strategies,
        "default": "simple_best"
    }), 200


# Analytics endpoints
@app.route("/analytics/usage", methods=["GET"])
@auth_middleware.require_api_key
@auth_middleware.require_role(UserRole.ADMIN)
def usage_analytics() -> Tuple[Response, int]:
    """Get usage analytics (admin only)."""
    hours = request.args.get("hours", 24, type=int)

    stats = usage_tracker.get_stats()
    hourly_stats = usage_tracker.get_hourly_stats(hours=hours)
    top_users = usage_tracker.get_top_users(limit=10)

    # Convert stats to dict for JSON serialization
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

    return jsonify({
        "summary": stats_dict,
        "hourly_stats": hourly_stats[:50],  # Limit response size
        "top_users": [{"user_id": uid, "request_count": count} for uid, count in top_users]
    }), 200


# Error handlers
@app.errorhandler(404)
def not_found(error) -> Tuple[Response, int]:
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error) -> Tuple[Response, int]:
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info("Starting TheNexus Enhanced API server on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
