"""
Security headers middleware for the Nexus API.

Adds CORS, CSP, and other defensive headers to every response.
"""

from flask import Flask, Response, request


# Default allowed origins — override via app.config["CORS_ORIGINS"]
_DEFAULT_ORIGINS = "http://localhost:3000"


def init_security_headers(app: Flask) -> None:
    """Register an after_request hook that adds security headers."""

    @app.after_request
    def _add_security_headers(response: Response) -> Response:
        # --- CORS ---
        allowed_origins = app.config.get("CORS_ORIGINS", _DEFAULT_ORIGINS)
        origin = request.headers.get("Origin", "")

        if origin and allowed_origins and origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"

        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-API-Key, X-Request-ID"
        )
        response.headers["Access-Control-Max-Age"] = "86400"

        # --- Security headers ---
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response

    # Handle CORS preflight
    @app.before_request
    def _handle_preflight():
        if request.method == "OPTIONS":
            response = Response("", status=204)
            return response
        return None
