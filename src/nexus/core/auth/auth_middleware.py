"""
Authentication middleware for Flask.
"""

import logging
from functools import wraps
from typing import Optional, Callable
from flask import request, jsonify, g

from nexus.core.auth.api_key_manager import APIKeyManager
from nexus.core.auth.models import UserRole

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """
    Authentication middleware for Flask applications.
    
    Provides:
    - API key validation
    - Rate limiting
    - Role-based access control
    """
    
    def __init__(self, api_key_manager: Optional[APIKeyManager] = None):
        """
        Initialize auth middleware.
        
        Args:
            api_key_manager: Optional APIKeyManager instance
        """
        self.api_key_manager = api_key_manager or APIKeyManager()
        logger.info("AuthMiddleware initialized")
    
    def require_api_key(self, func: Callable) -> Callable:
        """
        Decorator to require API key authentication.
        
        Usage:
            @app.route("/protected")
            @auth_middleware.require_api_key
            def protected_route():
                return {"message": "Success"}
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract API key from header
            api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
            
            if not api_key:
                logger.warning("Missing API key in request")
                return jsonify({
                    "error": "Missing API key",
                    "message": "Provide API key in X-API-Key or Authorization header"
                }), 401
            
            # Remove "Bearer " prefix if present
            if api_key.startswith("Bearer "):
                api_key = api_key[7:]
            
            # Validate API key
            api_key_obj = self.api_key_manager.validate_key(api_key)
            
            if not api_key_obj:
                logger.warning(f"Invalid API key attempted")
                return jsonify({
                    "error": "Invalid API key",
                    "message": "The provided API key is invalid or expired"
                }), 401
            
            # Check rate limit
            rate_limit_info = self.api_key_manager.check_rate_limit(api_key_obj.key_id)
            
            if rate_limit_info.remaining == 0:
                logger.warning(f"Rate limit exceeded for key {api_key_obj.key_id}")
                return jsonify({
                    "error": "Rate limit exceeded",
                    "message": f"Rate limit of {rate_limit_info.limit} requests/hour exceeded",
                    "retry_after": rate_limit_info.retry_after
                }), 429
            
            # Record the request
            self.api_key_manager.record_request(api_key_obj.key_id)
            
            # Store API key info in Flask g object
            g.api_key = api_key_obj
            g.rate_limit = rate_limit_info
            
            # Add rate limit headers to response
            response = func(*args, **kwargs)
            
            if isinstance(response, tuple):
                data, status_code = response[0], response[1]
            else:
                data, status_code = response, 200
            
            # Add rate limit headers
            if hasattr(response, 'headers'):
                response.headers['X-RateLimit-Limit'] = str(rate_limit_info.limit)
                response.headers['X-RateLimit-Remaining'] = str(rate_limit_info.remaining)
                response.headers['X-RateLimit-Reset'] = rate_limit_info.reset_at.isoformat()
            
            return data, status_code
        
        return wrapper
    
    def require_role(self, required_role: UserRole) -> Callable:
        """
        Decorator to require specific user role.
        
        Usage:
            @app.route("/admin")
            @auth_middleware.require_api_key
            @auth_middleware.require_role(UserRole.ADMIN)
            def admin_route():
                return {"message": "Admin only"}
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check if API key is present (should be set by require_api_key)
                if not hasattr(g, 'api_key'):
                    logger.error("require_role used without require_api_key")
                    return jsonify({
                        "error": "Authentication required",
                        "message": "This endpoint requires authentication"
                    }), 401
                
                # Get user from API key
                user = self.api_key_manager.users.get(g.api_key.user_id)
                
                if not user:
                    logger.error(f"User not found for key {g.api_key.key_id}")
                    return jsonify({
                        "error": "User not found",
                        "message": "Associated user account not found"
                    }), 403
                
                # Check role permission
                if not user.has_permission(required_role):
                    logger.warning(
                        f"Insufficient permissions for user {user.user_id} "
                        f"(has {user.role.value}, needs {required_role.value})"
                    )
                    return jsonify({
                        "error": "Insufficient permissions",
                        "message": f"This endpoint requires {required_role.value} role"
                    }), 403
                
                # Store user in g
                g.user = user
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


def get_current_api_key():
    """Get the current API key from Flask g object."""
    return getattr(g, 'api_key', None)


def get_current_user():
    """Get the current user from Flask g object."""
    return getattr(g, 'user', None)


def get_rate_limit_info():
    """Get the current rate limit info from Flask g object."""
    return getattr(g, 'rate_limit', None)
