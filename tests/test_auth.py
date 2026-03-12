"""Tests for authentication system."""

import os
import sys
import pytest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.auth.models import User, APIKey, UserRole, RateLimitInfo
from nexus.core.auth.api_key_manager import APIKeyManager


class TestUserModel:
    """Tests for User model."""
    
    def test_user_creation(self):
        """Test user creation."""
        user = User(
            user_id="user_123",
            username="alice",
            email="alice@example.com",
            role=UserRole.USER
        )
        
        assert user.user_id == "user_123"
        assert user.username == "alice"
        assert user.role == UserRole.USER
        assert user.is_active
    
    def test_user_permissions(self):
        """Test user permission checking."""
        admin = User("1", "admin", "admin@test.com", UserRole.ADMIN)
        user = User("2", "user", "user@test.com", UserRole.USER)
        readonly = User("3", "readonly", "ro@test.com", UserRole.READONLY)
        
        # Admin has all permissions
        assert admin.has_permission(UserRole.ADMIN)
        assert admin.has_permission(UserRole.USER)
        assert admin.has_permission(UserRole.READONLY)
        
        # User has user and readonly permissions
        assert not user.has_permission(UserRole.ADMIN)
        assert user.has_permission(UserRole.USER)
        assert user.has_permission(UserRole.READONLY)
        
        # Readonly only has readonly permission
        assert not readonly.has_permission(UserRole.ADMIN)
        assert not readonly.has_permission(UserRole.USER)
        assert readonly.has_permission(UserRole.READONLY)


class TestAPIKeyModel:
    """Tests for APIKey model."""
    
    def test_api_key_creation(self):
        """Test API key creation."""
        api_key = APIKey(
            key_id="key_123",
            key_hash="hash123",
            user_id="user_123",
            name="Test Key",
            rate_limit=1000
        )
        
        assert api_key.key_id == "key_123"
        assert api_key.is_active
        assert api_key.rate_limit == 1000
    
    def test_api_key_validity(self):
        """Test API key validity checking."""
        # Active key
        active_key = APIKey("1", "hash", "user1", "Active")
        assert active_key.is_valid()
        
        # Inactive key
        inactive_key = APIKey("2", "hash", "user1", "Inactive", is_active=False)
        assert not inactive_key.is_valid()
        
        # Expired key
        past = datetime.now(timezone.utc) - timedelta(days=1)
        expired_key = APIKey("3", "hash", "user1", "Expired", expires_at=past)
        assert not expired_key.is_valid()
        
        # Future expiry
        future = datetime.now(timezone.utc) + timedelta(days=1)
        future_key = APIKey("4", "hash", "user1", "Future", expires_at=future)
        assert future_key.is_valid()
    
    def test_api_key_usage_increment(self):
        """Test incrementing usage counter."""
        api_key = APIKey("1", "hash", "user1", "Test")
        assert api_key.usage_count == 0
        assert api_key.last_used is None
        
        api_key.increment_usage()
        assert api_key.usage_count == 1
        assert api_key.last_used is not None


class TestAPIKeyManager:
    """Tests for APIKeyManager."""
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = APIKeyManager()
        assert manager is not None
        assert len(manager.keys) == 0
    
    def test_generate_key(self):
        """Test API key generation."""
        manager = APIKeyManager()
        
        raw_key, api_key = manager.generate_key(
            user_id="user_123",
            name="Test Key",
            rate_limit=500
        )
        
        assert raw_key.startswith("sk_")
        assert api_key.key_id.startswith("key_")
        assert api_key.user_id == "user_123"
        assert api_key.name == "Test Key"
        assert api_key.rate_limit == 500
        assert api_key.key_id in manager.keys
    
    def test_validate_key(self):
        """Test API key validation."""
        manager = APIKeyManager()
        
        # Generate key
        raw_key, api_key = manager.generate_key("user_123", "Test Key")
        
        # Validate with correct key
        validated = manager.validate_key(raw_key)
        assert validated is not None
        assert validated.key_id == api_key.key_id
        assert validated.usage_count == 1  # Incremented
        
        # Validate with incorrect key
        invalid = manager.validate_key("sk_invalid_key")
        assert invalid is None
    
    def test_rate_limiting(self):
        """Test rate limit checking."""
        manager = APIKeyManager()
        
        raw_key, api_key = manager.generate_key("user_123", "Test", rate_limit=10)
        
        # Check initial rate limit
        rate_info = manager.check_rate_limit(api_key.key_id)
        assert rate_info.limit == 10
        assert rate_info.remaining == 10
        
        # Record some requests
        for _ in range(5):
            manager.record_request(api_key.key_id)
        
        # Check updated rate limit
        rate_info = manager.check_rate_limit(api_key.key_id)
        assert rate_info.remaining == 5
        
        # Exhaust limit
        for _ in range(5):
            manager.record_request(api_key.key_id)
        
        rate_info = manager.check_rate_limit(api_key.key_id)
        assert rate_info.remaining == 0
        assert rate_info.retry_after is not None
    
    def test_revoke_key(self):
        """Test key revocation."""
        manager = APIKeyManager()
        
        raw_key, api_key = manager.generate_key("user_123", "Test")
        
        # Revoke key
        assert manager.revoke_key(api_key.key_id)
        assert not api_key.is_active
        
        # Validation should fail
        validated = manager.validate_key(raw_key)
        assert validated is None
    
    def test_list_keys(self):
        """Test listing user's keys."""
        manager = APIKeyManager()
        
        # Generate multiple keys
        manager.generate_key("user_1", "Key 1")
        manager.generate_key("user_1", "Key 2")
        manager.generate_key("user_2", "Key 3")
        
        # List keys for user_1
        user_1_keys = manager.list_keys("user_1")
        assert len(user_1_keys) == 2
        
        # List keys for user_2
        user_2_keys = manager.list_keys("user_2")
        assert len(user_2_keys) == 1
    
    def test_create_user(self):
        """Test user creation."""
        manager = APIKeyManager()
        
        user = manager.create_user("alice", "alice@example.com", "admin")
        
        assert user.username == "alice"
        assert user.email == "alice@example.com"
        assert user.role == UserRole.ADMIN
        assert user.user_id in manager.users
