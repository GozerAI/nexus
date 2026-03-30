"""
Request validation middleware for the Nexus API.

Enforces request body size limits and content-type checks on all POST/PUT/PATCH
routes, returning 413 or 400 before the route handler runs.
"""

import logging

from flask import Flask, Response, jsonify, request

logger = logging.getLogger(__name__)

# 1 MB default max body size
DEFAULT_MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1 MB


def init_request_validation(
    app: Flask,
    max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
) -> None:
    """Register request validation hooks on *app*."""

    # Flask's built-in enforcement: rejects bodies larger than this with 413.
    # Respect any value already set in app config.
    if app.config.get("MAX_CONTENT_LENGTH") is None:
        app.config["MAX_CONTENT_LENGTH"] = max_content_length

    @app.before_request
    def _validate_request():
        # Only validate body on methods that send one
        if request.method not in ("POST", "PUT", "PATCH"):
            return None

        # Skip health endpoints
        if request.path.endswith("/health"):
            return None

        # Require JSON content-type
        if not request.is_json:
            return (
                jsonify({"ok": False, "error": "Content-Type must be application/json"}),
                400,
            )

        # Attempt to parse the body — catches malformed JSON
        data = request.get_json(silent=True)
        if data is None:
            return (
                jsonify({"ok": False, "error": "Request body must be valid JSON"}),
                400,
            )

        return None

    @app.errorhandler(413)
    def _payload_too_large(error):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"Request body too large (max {max_content_length} bytes)",
                }
            ),
            413,
        )
