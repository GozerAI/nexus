"""
Database-backed API key management system.

Provides the same interface as APIKeyManager but with SQLite persistence.
"""

import secrets
import hashlib
import logging
from typing import Optional, List
from datetime import datetime, timedelta, timezone

from nexus.core.auth.models import APIKey, User, UserRole
from nexus.core.database import get_db
from nexus.core.database.repositories import UserRepository, APIKeyRepository, RateLimitRepository


class _UserProxy:
    """Dict-like proxy for DB-backed user lookups (compatibility with in-memory manager)."""

    def __init__(self, manager):
        self._manager = manager

    def get(self, user_id, default=None):
        return self._manager.get_user(user_id) or default

logger = logging.getLogger(__name__)


class PersistentAPIKeyManager:
    """
    Database-backed API key manager.

    Features:
    - Key generation with secure random tokens
    - Key validation and verification
    - Rate limiting per key
    - Usage tracking
    - Key expiration
    - Persistent storage in SQLite
    """

    def __init__(self):
        """Initialize the API key manager."""
        self.db = get_db()
        self.users = _UserProxy(self)
        logger.info("PersistentAPIKeyManager initialized")

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def create_user(
        self,
        username: str,
        email: str,
        role: str = "user",
    ) -> User:
        """
        Create a new user.

        Args:
            username: Username
            email: Email address
            role: User role (user, admin, etc.)

        Returns:
            User object

        Raises:
            ValueError: If username or email already exists
        """
        with self.db.get_session() as session:
            user_repo = UserRepository(session)

            # Check if user already exists
            if user_repo.get_by_username(username):
                raise ValueError(f"Username '{username}' already exists")
            if user_repo.get_by_email(email):
                raise ValueError(f"Email '{email}' already exists")

            # Generate user ID
            user_id = f"user_{secrets.token_urlsafe(16)}"

            # Create user
            user_model = user_repo.create(
                user_id=user_id,
                username=username,
                email=email,
                role=role,
            )

            # Convert to dataclass
            return User(
                user_id=user_model.user_id,
                username=user_model.username,
                email=user_model.email,
                role=UserRole(user_model.role),
                created_at=user_model.created_at,
            )

    def get_user(self, user_id: str) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User ID

        Returns:
            User object or None
        """
        with self.db.get_session() as session:
            user_repo = UserRepository(session)
            user_model = user_repo.get_by_id(user_id)

            if user_model:
                return User(
                    user_id=user_model.user_id,
                    username=user_model.username,
                    email=user_model.email,
                    role=UserRole(user_model.role),
                    created_at=user_model.created_at,
                )
            return None

    def generate_key(
        self,
        user_id: str,
        name: str,
        rate_limit: int = 1000,
        expires_in_days: Optional[int] = None,
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

        # Store in database
        with self.db.get_session() as session:
            key_repo = APIKeyRepository(session)
            key_model = key_repo.create(
                key_id=key_id,
                user_id=user_id,
                key_hash=key_hash,
                name=name,
                rate_limit=rate_limit,
                expires_at=expires_at,
            )

            # Convert to dataclass
            api_key = APIKey(
                key_id=key_model.key_id,
                key_hash=key_model.key_hash,
                user_id=key_model.user_id,
                name=key_model.name,
                created_at=key_model.created_at,
                last_used=key_model.last_used,
                expires_at=key_model.expires_at,
                is_active=key_model.is_active,
                rate_limit=key_model.rate_limit,
                usage_count=key_model.usage_count,
            )

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

        # Hash the key
        key_hash = self._hash_key(raw_key)

        with self.db.get_session() as session:
            key_repo = APIKeyRepository(session)
            key_model = key_repo.get_by_hash(key_hash)

            if not key_model:
                logger.warning("API key not found")
                return None

            # Check if active
            if not key_model.is_active:
                logger.warning(f"API key {key_model.key_id} is inactive")
                return None

            # Check expiration (SQLite returns naive datetimes, normalize)
            expires_at = key_model.expires_at
            if expires_at and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at and datetime.now(timezone.utc) > expires_at:
                logger.warning(f"API key {key_model.key_id} has expired")
                return None

            # Update last used
            key_repo.update_last_used(key_model.key_id)

            # Convert to dataclass
            return APIKey(
                key_id=key_model.key_id,
                key_hash=key_model.key_hash,
                user_id=key_model.user_id,
                name=key_model.name,
                created_at=key_model.created_at,
                last_used=datetime.now(timezone.utc),  # Just updated
                expires_at=key_model.expires_at,
                is_active=key_model.is_active,
                rate_limit=key_model.rate_limit,
                usage_count=key_model.usage_count + 1,  # Just incremented
            )

    def check_rate_limit(self, key_id: str) -> "RateLimitInfo":
        """
        Check rate limit for an API key.

        Args:
            key_id: API key ID

        Returns:
            RateLimitInfo with current status
        """
        from nexus.core.auth.models import RateLimitInfo

        with self.db.get_session() as session:
            rate_limit_repo = RateLimitRepository(session)

            # Get count of requests in last hour
            count = rate_limit_repo.get_count_last_hour(key_id)

            # Look up the key to get its rate limit
            api_key_repo = APIKeyRepository(session)
            key_model = api_key_repo.get_by_id(key_id)
            limit = key_model.rate_limit if key_model else 100
            remaining = max(0, limit - count)

            reset_at = datetime.now(timezone.utc) + timedelta(hours=1)
            retry_after = None
            if remaining == 0:
                retry_after = 3600
                logger.warning(
                    f"Rate limit exceeded for key {key_id}: "
                    f"{count}/{limit}"
                )

            # Record this request
            rate_limit_repo.record(key_id)

            return RateLimitInfo(
                limit=limit,
                remaining=remaining,
                reset_at=reset_at,
                retry_after=retry_after,
            )

    def record_request(self, key_id: str):
        """Record a request for rate limiting (already done in check_rate_limit)."""
        pass

    def revoke_key(self, key_id: str):
        """
        Revoke an API key.

        Args:
            key_id: Key ID to revoke
        """
        with self.db.get_session() as session:
            key_repo = APIKeyRepository(session)
            key_repo.revoke(key_id)

    def list_keys(self, user_id: str) -> List[APIKey]:
        """
        List all API keys for a user.

        Args:
            user_id: User ID

        Returns:
            List of APIKey objects
        """
        with self.db.get_session() as session:
            key_repo = APIKeyRepository(session)
            key_models = key_repo.list_by_user(user_id)

            return [
                APIKey(
                    key_id=k.key_id,
                    key_hash=k.key_hash,
                    user_id=k.user_id,
                    name=k.name,
                    created_at=k.created_at,
                    last_used=k.last_used,
                    expires_at=k.expires_at,
                    is_active=k.is_active,
                    rate_limit=k.rate_limit,
                    usage_count=k.usage_count,
                )
                for k in key_models
            ]
