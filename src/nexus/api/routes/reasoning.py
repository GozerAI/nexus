"""
Reasoning API Routes

Provides RESTful endpoints for Nexus's reasoning system.
"""

from flask import Blueprint, request, jsonify, current_app
from typing import Dict, Any, Optional
import logging

from nexus.licensing import license_gate
from nexus.reasoning import (
    MetaReasoner,
    ChainOfThought,
    PatternReasoner,
    DynamicLearner,
    ReasoningAnalytics,
)

logger = logging.getLogger(__name__)

_GATE = "nxs.reasoning.advanced"

reasoning_bp = Blueprint('reasoning', __name__, url_prefix='/api/v1/reasoning')


@reasoning_bp.before_request
def _require_auth():
    """Require API key for all non-health endpoints."""
    if request.endpoint and request.endpoint.endswith('.reasoning_health'):
        return None
    auth = getattr(current_app, 'auth_middleware', None)
    if auth is None:
        return None
    api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
    if not api_key:
        return jsonify({"error": "Missing API key", "message": "Provide API key in X-API-Key or Authorization header"}), 401
    if api_key.startswith("Bearer "):
        api_key = api_key[7:]
    api_key_obj = auth.api_key_manager.validate_key(api_key)
    if not api_key_obj:
        return jsonify({"error": "Invalid API key", "message": "The provided API key is invalid or expired"}), 401
    return None


# Global reasoning instances (initialized by app)
meta_reasoner: Optional[MetaReasoner] = None
chain_of_thought: Optional[ChainOfThought] = None
pattern_reasoner: Optional[PatternReasoner] = None
dynamic_learner: Optional[DynamicLearner] = None
reasoning_analytics: Optional[ReasoningAnalytics] = None


def initialize_reasoning_system(config: Dict[str, Any]):
    """Initialize reasoning system components"""
    global meta_reasoner, chain_of_thought, pattern_reasoner
    global dynamic_learner, reasoning_analytics

    logger.info("Initializing Nexus reasoning system...")

    # Initialize reasoning engines
    meta_reasoner = MetaReasoner()
    chain_of_thought = ChainOfThought()
    pattern_reasoner = PatternReasoner()
    dynamic_learner = DynamicLearner(db_path=config.get("dynamic_learner_db"))
    reasoning_analytics = ReasoningAnalytics()

    logger.info("Reasoning system initialized successfully")


# ===== Meta-Reasoning Endpoints =====

@reasoning_bp.route('/meta/reason', methods=['POST'])
def meta_reason():
    """Perform meta-reasoning for self-improvement"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        problem = data.get('problem')
        if not problem:
            return jsonify({"status": "error", "message": "Problem is required"}), 400

        result = meta_reasoner.reason(
            problem=problem,
            context=data.get('context', {}),
            constraints=data.get('constraints', [])
        )

        return jsonify({
            "status": "success",
            "result": result
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error in meta-reasoning: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


@reasoning_bp.route('/meta/improve', methods=['POST'])
def meta_improve():
    """Use meta-reasoning to improve strategies"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        strategy = data.get('strategy')
        performance_data = data.get('performance_data', {})

        improvements = meta_reasoner.suggest_improvements(
            strategy=strategy,
            performance_data=performance_data
        )

        return jsonify({
            "status": "success",
            "improvements": improvements
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error in meta-improvement: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Chain-of-Thought Endpoints =====

@reasoning_bp.route('/chain-of-thought', methods=['POST'])
def chain_of_thought_reasoning():
    """Perform chain-of-thought reasoning"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        query = data.get('query')
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400

        result = chain_of_thought.reason(
            query=query,
            steps=data.get('steps'),
            explain=data.get('explain', True)
        )

        return jsonify({
            "status": "success",
            "result": result
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error in chain-of-thought: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


@reasoning_bp.route('/chain-of-thought/steps', methods=['GET'])
def get_reasoning_steps():
    """Get reasoning steps for a completed chain-of-thought"""
    try:
        license_gate.gate(_GATE)
        reasoning_id = request.args.get('id')

        steps = chain_of_thought.get_steps(reasoning_id)

        return jsonify({
            "status": "success",
            "steps": steps,
            "count": len(steps) if steps else 0
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error getting reasoning steps: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Pattern-Based Reasoning Endpoints =====

@reasoning_bp.route('/patterns/reason', methods=['POST'])
def pattern_based_reasoning():
    """Perform pattern-based reasoning"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        input_data = data.get('data')
        if not input_data:
            return jsonify({"status": "error", "message": "Data is required"}), 400

        result = pattern_reasoner.reason(
            data=input_data,
            pattern_type=data.get('pattern_type'),
            confidence_threshold=data.get('confidence_threshold', 0.7)
        )

        return jsonify({
            "status": "success",
            "result": result
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error in pattern reasoning: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


@reasoning_bp.route('/patterns/discover', methods=['POST'])
def discover_patterns():
    """Discover new reasoning patterns"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        examples = data.get('examples', [])
        if not examples:
            return jsonify({"status": "error", "message": "Examples are required"}), 400

        patterns = pattern_reasoner.discover_patterns(
            examples=examples,
            min_support=data.get('min_support', 0.5)
        )

        return jsonify({
            "status": "success",
            "patterns": patterns,
            "count": len(patterns)
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error discovering patterns: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Dynamic Learning Endpoints =====

@reasoning_bp.route('/learn', methods=['POST'])
def dynamic_learn():
    """Perform dynamic adaptive learning"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        experience = data.get('experience')
        if not experience:
            return jsonify({"status": "error", "message": "Experience is required"}), 400

        result = dynamic_learner.learn(
            experience=experience,
            feedback=data.get('feedback'),
            update_strategy=data.get('update_strategy', 'incremental')
        )

        return jsonify({
            "status": "success",
            "result": result
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error in dynamic learning: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


@reasoning_bp.route('/learn/adapt', methods=['POST'])
def adapt_behavior():
    """Adapt behavior based on performance"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        performance_metrics = data.get('performance_metrics', {})

        adaptations = dynamic_learner.adapt(
            performance_metrics=performance_metrics,
            adaptation_rate=data.get('adaptation_rate', 0.1)
        )

        return jsonify({
            "status": "success",
            "adaptations": adaptations
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error adapting behavior: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Analytics Endpoints =====

@reasoning_bp.route('/analytics', methods=['GET'])
def get_reasoning_analytics():
    """Get reasoning system analytics"""
    try:
        license_gate.gate(_GATE)
        analytics = reasoning_analytics.get_analytics()

        return jsonify({
            "status": "success",
            "analytics": analytics
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


@reasoning_bp.route('/analytics/performance', methods=['GET'])
def get_performance_metrics():
    """Get reasoning performance metrics"""
    try:
        license_gate.gate(_GATE)
        time_period = request.args.get('period', '24h')
        engine = request.args.get('engine')  # meta, cot, pattern, dynamic

        metrics = reasoning_analytics.get_performance_metrics(
            time_period=time_period,
            engine=engine
        )

        return jsonify({
            "status": "success",
            "metrics": metrics
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Self-Improvement Endpoints =====

@reasoning_bp.route('/self-improve', methods=['POST'])
def trigger_self_improvement():
    """Trigger self-improvement cycle"""
    try:
        license_gate.gate(_GATE)
        data = request.json

        result = meta_reasoner.self_improve(
            target_metric=data.get('target_metric'),
            improvement_strategy=data.get('improvement_strategy', 'iterative')
        )

        return jsonify({
            "status": "success",
            "result": result
        })

    except PermissionError as e:
        return jsonify({"status": "error", "message": str(e)}), 403
    except Exception as e:
        logger.error(f"Error in self-improvement: {e}")
        return jsonify({"status": "error", "message": "An internal error occurred"}), 500


# ===== Health Check =====

@reasoning_bp.route('/health', methods=['GET'])
def reasoning_health():
    """Check reasoning system health"""
    try:
        health = {
            "status": "healthy",
            "components": {
                "meta_reasoner": meta_reasoner is not None,
                "chain_of_thought": chain_of_thought is not None,
                "pattern_reasoner": pattern_reasoner is not None,
                "dynamic_learner": dynamic_learner is not None,
                "reasoning_analytics": reasoning_analytics is not None,
            }
        }

        all_healthy = all(health["components"].values())
        if not all_healthy:
            health["status"] = "degraded"

        return jsonify(health)

    except Exception as e:
        logger.error(f"Error checking reasoning health: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500
