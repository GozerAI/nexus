"""
API key management system.
"""

import secrets
import hashlib
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta, timezone

from nexus.core.auth.models import APIKey, User, RateLimitInfo

logger = logging.getLogger(__name__)


class APIKeyManager:
    """
    Manages API keys for authentication.
    
    Features:
    - Key generation with secure random tokens
    - Key validation and verification
    - Rate limiting per key
    - Usage tracking
    - Key expiration
    """
    
    def __init__(self):
        """Initialize the API key manager."""
        self.keys: Dict[str, APIKey] = {}
        self.users: Dict[str, User] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}
        logger.info("APIKeyManager initialized")
    
    def generate_key(
        self,
        user_id: str,
        name: str,
        rate_limit: int = 1000,
        expires_in_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Args:
            user_id: User ID who owns the key
            name: Descriptive name for the key
            rate_limit: Requests per hour limit
            expires_in_days: Optional expiration in days
            
        Returns:
            Tuple of (raw_key, APIKey object)
        """
        # Generate secure random key
        raw_key = f"sk_{secrets.token_urlsafe(32)}"
        
        # Hash the key for storage
        key_hash = self._hash_key(raw_key)
        
        # Generate key ID
        key_id = f"key_{secrets.token_urlsafe(8)}"
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            rate_limit=rate_limit,
            expires_at=expires_at,
        )
        
        # Store the key
        self.keys[key_id] = api_key
        
        logger.info(
            f"Generated API key '{name}' for user {user_id} "
            f"(rate_limit={rate_limit}/hr, expires={expires_at})"
        )
        
        return raw_key, api_key
    
    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key.
        
        Args:
            raw_key: Raw API key string
            
        Returns:
            APIKey object if valid, None otherwise
        """
        if not raw_key or not raw_key.startswith("sk_"):
            logger.warning("Invalid API key format")
            return None
        
        # Hash the provided key
        key_hash = self._hash_key(raw_key)
        
        # Find matching key
        for api_key in self.keys.values():
            if api_key.key_hash == key_hash:
                if api_key.is_valid():
                    logger.debug(f"API key {api_key.key_id} validated")
                    api_key.increment_usage()
                    return api_key
                else:
                    logger.warning(f"API key {api_key.key_id} is inactive or expired")
                    return None
        
        logger.warning("API key not found")
        return None
    
    def check_rate_limit(self, key_id: str) -> RateLimitInfo:
        """
        Check rate limit for an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            RateLimitInfo with current status
        """
        api_key = self.keys.get(key_id)
        if not api_key:
            return RateLimitInfo(
                limit=0,
                remaining=0,
                reset_at=datetime.now(timezone.utc),
                retry_after=3600
            )
        
        # Get request history
        if key_id not in self.rate_limits:
            self.rate_limits[key_id] = []
        
        # Clean old requests (older than 1 hour)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        self.rate_limits[key_id] = [
            ts for ts in self.rate_limits[key_id] if ts > cutoff
        ]
        
        current_count = len(self.rate_limits[key_id])
        remaining = max(0, api_key.rate_limit - current_count)
        
        # Calculate reset time (1 hour from oldest request)
        if self.rate_limits[key_id]:
            reset_at = min(self.rate_limits[key_id]) + timedelta(hours=1)
        else:
            reset_at = datetime.now(timezone.utc) + timedelta(hours=1)
        
        retry_after = None
        if remaining == 0:
            retry_after = int((reset_at - datetime.now(timezone.utc)).total_seconds())
        
        return RateLimitInfo(
            limit=api_key.rate_limit,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after
        )
    
    def record_request(self, key_id: str):
        """
        Record a request for rate limiting.
        
        Args:
            key_id: API key ID
        """
        if key_id not in self.rate_limits:
            self.rate_limits[key_id] = []
        
        self.rate_limits[key_id].append(datetime.now(timezone.utc))
        logger.debug(f"Recorded request for key {key_id}")
    
    def revoke_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            True if revoked, False if not found
        """
        if key_id in self.keys:
            self.keys[key_id].is_active = False
            logger.info(f"Revoked API key {key_id}")
            return True
        return False
    
    def list_keys(self, user_id: str) -> List[APIKey]:
        """
        List all API keys for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of APIKey objects
        """
        return [
            key for key in self.keys.values()
            if key.user_id == user_id
        ]
    
    def get_key_stats(self, key_id: str) -> Optional[dict]:
        """
        Get usage statistics for an API key.
        
        Args:
            key_id: API key ID
            
        Returns:
            Dictionary with statistics
        """
        api_key = self.keys.get(key_id)
        if not api_key:
            return None
        
        rate_limit_info = self.check_rate_limit(key_id)
        
        return {
            "key_id": key_id,
            "name": api_key.name,
            "created_at": api_key.created_at.isoformat(),
            "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
            "usage_count": api_key.usage_count,
            "is_active": api_key.is_active,
            "rate_limit": {
                "limit": rate_limit_info.limit,
                "remaining": rate_limit_info.remaining,
                "reset_at": rate_limit_info.reset_at.isoformat(),
            }
        }
    
    def _hash_key(self, raw_key: str) -> str:
        """
        Hash an API key for secure storage.
        
        Args:
            raw_key: Raw API key
            
        Returns:
            Hashed key
        """
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    def create_user(self, username: str, email: str, role: str = "user") -> User:
        """
        Create a new user.
        
        Args:
            username: Username
            email: Email address
            role: User role (admin, user, readonly)
            
        Returns:
            User object
        """
        from nexus.core.auth.models import UserRole
        
        user_id = f"user_{secrets.token_urlsafe(8)}"
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=UserRole(role.lower())
        )
        
        self.users[user_id] = user
        logger.info(f"Created user {username} with role {role}")
        
        return user
