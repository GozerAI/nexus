"""
Flask API entry point for Nexus.

Provides a small, predictable REST surface and avoids eager engine startup on
module import so tests and service processes can control initialization.
"""

from typing import Any, Callable, Optional
import logging

from flask import Flask, Response, current_app, jsonify, request

from nexus.core.auth import AuthMiddleware
from nexus.core.core_engine import CognitiveCore
from nexus.api.routes.memory import memory_bp, initialize_memory_system
from nexus.api.routes.rag import rag_bp, initialize_rag_system
from nexus.api.routes.reasoning import reasoning_bp, initialize_reasoning_system
from nexus.api.routes.data import data_bp, initialize_data_system


logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 10_000


class _LazyAuthMiddleware:
    """Instantiate AuthMiddleware only when a route actually uses it."""

    def __init__(self, factory: Callable[[], AuthMiddleware]):
        self._factory = factory
        self._instance: Optional[AuthMiddleware] = None

    def _get_instance(self) -> AuthMiddleware:
        if self._instance is None:
            self._instance = self._factory()
            logger.info("Auth middleware initialized lazily")
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)


def _success_payload(**data: Any) -> dict[str, Any]:
    """Return a consistent success payload."""
    return {"ok": True, **data}


def _error_payload(message: str, **extra: Any) -> dict[str, Any]:
    """Return a consistent error payload."""
    return {"ok": False, "error": message, **extra}


def _register_blueprints(app: Flask) -> None:
    """Register API blueprints exactly once per app instance."""
    app.register_blueprint(memory_bp)
    app.register_blueprint(rag_bp)
    app.register_blueprint(reasoning_bp)
    app.register_blueprint(data_bp)


def _get_engine(app: Flask) -> CognitiveCore:
    """Create the cognitive engine lazily for the current app."""
    engine = app.extensions.get("nexus_engine")
    if engine is None:
        factory: Callable[[], CognitiveCore] = app.config["NEXUS_ENGINE_FACTORY"]
        engine = factory()
        app.extensions["nexus_engine"] = engine
        logger.info("Cognitive engine initialized lazily")
    return engine


def create_app(
    config: Optional[dict[str, Any]] = None,
    *,
    engine_factory: Optional[Callable[[], CognitiveCore]] = None,
    auth_factory: Optional[Callable[[], AuthMiddleware]] = None,
) -> Flask:
    """Create a configured Nexus Flask app."""
    app = Flask(__name__)
    app.config.update(
        {
            "NEXUS_ENGINE_FACTORY": engine_factory or CognitiveCore,
            "NEXUS_AUTH_FACTORY": auth_factory or AuthMiddleware,
            "NEXUS_SUBSYSTEMS_INITIALIZED": False,
        }
    )
    if config:
        app.config.update(config)

    _register_blueprints(app)
    app.auth_middleware = _LazyAuthMiddleware(app.config["NEXUS_AUTH_FACTORY"])

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        """Health check endpoint."""
        return (
            jsonify(
                _success_payload(
                    status="healthy",
                    service="nexus-api",
                    engine_initialized="nexus_engine" in current_app.extensions,
                    subsystems_initialized=current_app.config.get("NEXUS_SUBSYSTEMS_INITIALIZED", False),
                )
            ),
            200,
        )

    @app.route("/think", methods=["POST"])
    def think() -> tuple[Response, int]:
        """
        Process input through the cognitive engine.

        Expected JSON body:
        {
            "input": "string to process"
        }
        """
        try:
            if not request.is_json:
                logger.warning("Received non-JSON request")
                return jsonify(_error_payload("Content-Type must be application/json")), 400

            data = request.get_json(silent=True)
            if data is None:
                logger.warning("Invalid JSON body")
                return jsonify(_error_payload("Request body must be valid JSON")), 400

            if "input" not in data:
                logger.warning("Missing 'input' field in request")
                return jsonify(_error_payload("Missing required field: 'input'")), 400

            input_data = data.get("input")
            if not isinstance(input_data, str) or not input_data.strip():
                logger.warning("Invalid input type or empty: %s", type(input_data))
                return jsonify(_error_payload("Input must be a non-empty string")), 400

            if len(input_data) > MAX_INPUT_LENGTH:
                logger.warning("Input too long: %d characters", len(input_data))
                return (
                    jsonify(_error_payload(f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters")),
                    400,
                )

            logger.info("Processing input: %s...", input_data[:50])
            result = _get_engine(current_app).think(input_data)
            logger.info("Successfully processed request")
            return jsonify(_success_payload(status="success", response=result)), 200

        except Exception as exc:
            logger.error("Error processing request: %s", exc, exc_info=True)
            return jsonify(_error_payload("Internal server error", message="An internal error occurred")), 500

    @app.errorhandler(404)
    def not_found(error: Any) -> tuple[Response, int]:
        """Handle 404 errors."""
        return jsonify(_error_payload("Endpoint not found")), 404

    @app.errorhandler(500)
    def internal_error(error: Any) -> tuple[Response, int]:
        """Handle uncaught 500 errors."""
        logger.error("Internal server error: %s", error)
        return jsonify(_error_payload("Internal server error")), 500

    return app


def initialize_subsystems(app: Optional[Flask] = None, config: Optional[dict[str, Any]] = None) -> None:
    """Initialize optional API subsystems once for the given app."""
    target_app = app or current_app
    config = config or {}

    if target_app.config.get("NEXUS_SUBSYSTEMS_INITIALIZED", False):
        return

    initializers = (
        ("Memory", initialize_memory_system),
        ("RAG", initialize_rag_system),
        ("Reasoning", initialize_reasoning_system),
        ("Data", initialize_data_system),
    )

    for name, initializer in initializers:
        try:
            initializer(config)
            logger.info("%s subsystem initialized", name)
        except Exception as exc:
            logger.warning("%s subsystem initialization failed: %s", name, exc)

    target_app.config["NEXUS_SUBSYSTEMS_INITIALIZED"] = True


def main() -> None:
    """Entry point for the packaged nexus-api server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    app = create_app()
    initialize_subsystems(app)
    logger.info("Starting Nexus API server on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


app = create_app()


if __name__ == "__main__":
    main()
