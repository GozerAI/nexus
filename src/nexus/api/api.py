"""
Flask API for TheNexus cognitive engine.

Provides REST endpoints for interacting with the AI reasoning system.
"""

from typing import Tuple, Any
import logging
from flask import Flask, request, jsonify, Response
from nexus.core.core_engine import CognitiveCore
from nexus.api.routes.memory import memory_bp, initialize_memory_system
from nexus.api.routes.rag import rag_bp, initialize_rag_system
from nexus.api.routes.reasoning import reasoning_bp, initialize_reasoning_system
from nexus.api.routes.data import data_bp, initialize_data_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
engine = CognitiveCore()

# Register route blueprints
app.register_blueprint(memory_bp)
app.register_blueprint(rag_bp)
app.register_blueprint(reasoning_bp)
app.register_blueprint(data_bp)


@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "TheNexus"}), 200


@app.route("/think", methods=["POST"])
def think() -> Tuple[Response, int]:
    """
    Process input through the cognitive engine.

    Expected JSON body:
    {
        "input": "string to process"
    }

    Returns:
        JSON response with processed result or error message
    """
    try:
        # Validate request has JSON body
        if not request.is_json:
            logger.warning("Received non-JSON request")
            return jsonify({
                "error": "Content-Type must be application/json"
            }), 400

        # Get request data
        data = request.get_json()

        # Validate input field exists
        if data is None or "input" not in data:
            logger.warning("Missing 'input' field in request")
            return jsonify({
                "error": "Missing required field: 'input'"
            }), 400

        input_data = data.get("input")

        # Validate input is not empty
        if not input_data or not isinstance(input_data, str):
            logger.warning(f"Invalid input type or empty: {type(input_data)}")
            return jsonify({
                "error": "Input must be a non-empty string"
            }), 400

        # Validate input length
        if len(input_data) > 10000:
            logger.warning(f"Input too long: {len(input_data)} characters")
            return jsonify({
                "error": "Input exceeds maximum length of 10000 characters"
            }), 400

        logger.info(f"Processing input: {input_data[:50]}...")

        # Process through cognitive engine
        result = engine.think(input_data)

        logger.info("Successfully processed request")
        return jsonify({
            "response": result,
            "status": "success"
        }), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500


@app.errorhandler(404)
def not_found(error: Any) -> Tuple[Response, int]:
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error: Any) -> Tuple[Response, int]:
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500


def initialize_subsystems(config: dict = None):
    """Initialize all API subsystems with optional config."""
    config = config or {}
    try:
        initialize_memory_system(config)
        logger.info("Memory subsystem initialized")
    except Exception as e:
        logger.warning("Memory subsystem initialization failed: %s", e)
    try:
        initialize_rag_system(config)
        logger.info("RAG subsystem initialized")
    except Exception as e:
        logger.warning("RAG subsystem initialization failed: %s", e)
    try:
        initialize_reasoning_system(config)
        logger.info("Reasoning subsystem initialized")
    except Exception as e:
        logger.warning("Reasoning subsystem initialization failed: %s", e)
    try:
        initialize_data_system(config)
        logger.info("Data subsystem initialized")
    except Exception as e:
        logger.warning("Data subsystem initialization failed: %s", e)


def main():
    """Entry point for nexus-api."""
    initialize_subsystems()
    logger.info("Starting TheNexus API server on port 5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
