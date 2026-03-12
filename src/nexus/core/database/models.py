"""
SQLAlchemy database models for persistence.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class UserModel(Base):
    """User database model."""

    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(50), nullable=False, default="user")
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, nullable=False, default=True)

    # Relationships
    api_keys = relationship("APIKeyModel", back_populates="user", cascade="all, delete-orphan")
    cost_entries = relationship("CostEntryModel", back_populates="user")
    usage_entries = relationship("UsageEntryModel", back_populates="user")


class APIKeyModel(Base):
    """API Key database model."""

    __tablename__ = "api_keys"

    key_id = Column(String(64), primary_key=True, index=True)
    user_id = Column(String(64), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    key_hash = Column(String(128), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    last_used = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)
    rate_limit = Column(Integer, nullable=False, default=1000)
    usage_count = Column(Integer, nullable=False, default=0)

    # Relationships
    user = relationship("UserModel", back_populates="api_keys")

    # Indexes
    __table_args__ = (
        Index("idx_api_key_user_active", "user_id", "is_active"),
        Index("idx_api_key_hash", "key_hash"),
    )


class CostEntryModel(Base):
    """Cost tracking entry database model."""

    __tablename__ = "cost_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=True, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    provider = Column(String(100), nullable=False, index=True)
    tokens_used = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False)
    request_id = Column(String(128), nullable=True, index=True)

    # Relationships
    user = relationship("UserModel", back_populates="cost_entries")

    # Indexes for common queries
    __table_args__ = (
        Index("idx_cost_timestamp_user", "timestamp", "user_id"),
        Index("idx_cost_model_provider", "model_name", "provider"),
    )


class UsageEntryModel(Base):
    """Usage analytics entry database model."""

    __tablename__ = "usage_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)
    user_id = Column(String(64), ForeignKey("users.user_id"), nullable=True, index=True)
    endpoint = Column(String(255), nullable=False, index=True)
    model_name = Column(String(255), nullable=True, index=True)
    tokens_used = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Float, nullable=False, default=0.0)
    cached = Column(Boolean, nullable=False, default=False)
    success = Column(Boolean, nullable=False, default=True)
    error_type = Column(String(255), nullable=True, index=True)

    # Relationships
    user = relationship("UserModel", back_populates="usage_entries")

    # Indexes for analytics queries
    __table_args__ = (
        Index("idx_usage_timestamp_endpoint", "timestamp", "endpoint"),
        Index("idx_usage_user_success", "user_id", "success"),
        Index("idx_usage_model", "model_name", "timestamp"),
    )


class StrategyPerformanceModel(Base):
    """Strategy performance history database model."""

    __tablename__ = "strategy_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    total_requests = Column(Integer, nullable=False, default=0)
    successful_requests = Column(Integer, nullable=False, default=0)
    average_score = Column(Float, nullable=False, default=0.0)
    average_latency_ms = Column(Float, nullable=False, default=0.0)
    average_cost = Column(Float, nullable=False, default=0.0)
    success_rate = Column(Float, nullable=False, default=1.0)
    last_updated = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Unique constraint on strategy + model
    __table_args__ = (
        Index("idx_strategy_model", "strategy_name", "model_name", unique=True),
    )


class RateLimitEntryModel(Base):
    """Rate limiting tracking database model."""

    __tablename__ = "rate_limit_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), index=True)

    # Index for efficient rate limit queries
    __table_args__ = (
        Index("idx_ratelimit_key_timestamp", "key_id", "timestamp"),
    )
