"""
Authentication data models.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List
from enum import Enum


class UserRole(Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"


@dataclass
class User:
    """User model."""
    user_id: str
    username: str
    email: str
    role: UserRole = UserRole.USER
    created_at: datetime = None
    is_active: bool = True
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
    
    def has_permission(self, required_role: UserRole) -> bool:
        """Check if user has required permission."""
        role_hierarchy = {
            UserRole.READONLY: 0,
            UserRole.USER: 1,
            UserRole.ADMIN: 2,
        }
        return role_hierarchy[self.role] >= role_hierarchy[required_role]


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    created_at: datetime = None
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit: int = 1000  # requests per hour
    usage_count: int = 0
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        if not self.is_active:
            return False
        
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        
        return True
    
    def increment_usage(self):
        """Increment usage counter."""
        self.usage_count += 1
        self.last_used = datetime.now(timezone.utc)


@dataclass
class RateLimitInfo:
    """Rate limit information."""
    limit: int
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None
