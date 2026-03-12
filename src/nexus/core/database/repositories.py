"""
Repository classes for database operations.

Provides clean interface for CRUD operations on database models.
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from nexus.core.database.models import (
    UserModel,
    APIKeyModel,
    CostEntryModel,
    UsageEntryModel,
    StrategyPerformanceModel,
    RateLimitEntryModel,
)

logger = logging.getLogger(__name__)


class UserRepository:
    """Repository for User operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def create(self, user_id: str, username: str, email: str, role: str = "user") -> UserModel:
        """Create a new user."""
        user = UserModel(
            user_id=user_id,
            username=username,
            email=email,
            role=role,
        )
        self.session.add(user)
        self.session.commit()
        self.session.refresh(user)
        logger.info(f"Created user: {username} ({user_id})")
        return user

    def get_by_id(self, user_id: str) -> Optional[UserModel]:
        """Get user by ID."""
        return self.session.query(UserModel).filter(UserModel.user_id == user_id).first()

    def get_by_username(self, username: str) -> Optional[UserModel]:
        """Get user by username."""
        return self.session.query(UserModel).filter(UserModel.username == username).first()

    def get_by_email(self, email: str) -> Optional[UserModel]:
        """Get user by email."""
        return self.session.query(UserModel).filter(UserModel.email == email).first()

    def list_all(self) -> List[UserModel]:
        """List all users."""
        return self.session.query(UserModel).all()

    def update(self, user: UserModel):
        """Update user."""
        self.session.commit()
        self.session.refresh(user)

    def delete(self, user_id: str):
        """Delete user (cascades to API keys)."""
        user = self.get_by_id(user_id)
        if user:
            self.session.delete(user)
            self.session.commit()
            logger.info(f"Deleted user: {user_id}")


class APIKeyRepository:
    """Repository for API Key operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def create(
        self,
        key_id: str,
        user_id: str,
        key_hash: str,
        name: str,
        rate_limit: int = 1000,
        expires_at: Optional[datetime] = None,
    ) -> APIKeyModel:
        """Create a new API key."""
        api_key = APIKeyModel(
            key_id=key_id,
            user_id=user_id,
            key_hash=key_hash,
            name=name,
            rate_limit=rate_limit,
            expires_at=expires_at,
        )
        self.session.add(api_key)
        self.session.commit()
        self.session.refresh(api_key)
        logger.info(f"Created API key: {name} ({key_id}) for user {user_id}")
        return api_key

    def get_by_id(self, key_id: str) -> Optional[APIKeyModel]:
        """Get API key by ID."""
        return self.session.query(APIKeyModel).filter(APIKeyModel.key_id == key_id).first()

    def get_by_hash(self, key_hash: str) -> Optional[APIKeyModel]:
        """Get API key by hash."""
        return self.session.query(APIKeyModel).filter(APIKeyModel.key_hash == key_hash).first()

    def list_by_user(self, user_id: str) -> List[APIKeyModel]:
        """List all API keys for a user."""
        return self.session.query(APIKeyModel).filter(APIKeyModel.user_id == user_id).all()

    def update_last_used(self, key_id: str):
        """Update last used timestamp."""
        api_key = self.get_by_id(key_id)
        if api_key:
            api_key.last_used = datetime.now(timezone.utc)
            api_key.usage_count += 1
            self.session.commit()

    def revoke(self, key_id: str):
        """Revoke an API key."""
        api_key = self.get_by_id(key_id)
        if api_key:
            api_key.is_active = False
            self.session.commit()
            logger.info(f"Revoked API key: {key_id}")

    def delete(self, key_id: str):
        """Delete an API key."""
        api_key = self.get_by_id(key_id)
        if api_key:
            self.session.delete(api_key)
            self.session.commit()
            logger.info(f"Deleted API key: {key_id}")


class CostEntryRepository:
    """Repository for Cost Entry operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def create(
        self,
        model_name: str,
        provider: str,
        tokens_used: int,
        cost_usd: float,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> CostEntryModel:
        """Create a new cost entry."""
        cost_entry = CostEntryModel(
            timestamp=timestamp or datetime.now(timezone.utc),
            model_name=model_name,
            provider=provider,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            user_id=user_id,
            request_id=request_id,
        )
        self.session.add(cost_entry)
        self.session.commit()
        return cost_entry

    def get_by_id(self, entry_id: int) -> Optional[CostEntryModel]:
        """Get cost entry by ID."""
        return self.session.query(CostEntryModel).filter(CostEntryModel.id == entry_id).first()

    def list_by_user(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[CostEntryModel]:
        """List cost entries for a user."""
        query = self.session.query(CostEntryModel).filter(CostEntryModel.user_id == user_id)

        if start_date:
            query = query.filter(CostEntryModel.timestamp >= start_date)
        if end_date:
            query = query.filter(CostEntryModel.timestamp <= end_date)

        return query.order_by(CostEntryModel.timestamp.desc()).all()

    def list_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
    ) -> List[CostEntryModel]:
        """List cost entries in date range."""
        query = self.session.query(CostEntryModel).filter(
            and_(
                CostEntryModel.timestamp >= start_date,
                CostEntryModel.timestamp <= end_date,
            )
        )

        if user_id:
            query = query.filter(CostEntryModel.user_id == user_id)

        return query.all()

    def get_total_cost(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ) -> float:
        """Get total cost for period."""
        query = self.session.query(func.sum(CostEntryModel.cost_usd))

        if start_date:
            query = query.filter(CostEntryModel.timestamp >= start_date)
        if end_date:
            query = query.filter(CostEntryModel.timestamp <= end_date)
        if user_id:
            query = query.filter(CostEntryModel.user_id == user_id)

        result = query.scalar()
        return result if result else 0.0


class UsageEntryRepository:
    """Repository for Usage Entry operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def create(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        cached: bool = False,
        success: bool = True,
        error_type: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> UsageEntryModel:
        """Create a new usage entry."""
        usage_entry = UsageEntryModel(
            timestamp=timestamp or datetime.now(timezone.utc),
            endpoint=endpoint,
            user_id=user_id,
            model_name=model_name,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cached=cached,
            success=success,
            error_type=error_type,
        )
        self.session.add(usage_entry)
        self.session.commit()
        return usage_entry

    def list_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> List[UsageEntryModel]:
        """List usage entries in date range."""
        query = self.session.query(UsageEntryModel).filter(
            and_(
                UsageEntryModel.timestamp >= start_date,
                UsageEntryModel.timestamp <= end_date,
            )
        )

        if user_id:
            query = query.filter(UsageEntryModel.user_id == user_id)
        if endpoint:
            query = query.filter(UsageEntryModel.endpoint == endpoint)

        return query.all()

    def get_count(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Get usage entry count."""
        query = self.session.query(func.count(UsageEntryModel.id))

        if start_date:
            query = query.filter(UsageEntryModel.timestamp >= start_date)
        if end_date:
            query = query.filter(UsageEntryModel.timestamp <= end_date)
        if user_id:
            query = query.filter(UsageEntryModel.user_id == user_id)

        return query.scalar() or 0


class StrategyPerformanceRepository:
    """Repository for Strategy Performance operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def get_or_create(
        self,
        strategy_name: str,
        model_name: str,
    ) -> StrategyPerformanceModel:
        """Get existing performance record or create new one."""
        perf = (
            self.session.query(StrategyPerformanceModel)
            .filter(
                and_(
                    StrategyPerformanceModel.strategy_name == strategy_name,
                    StrategyPerformanceModel.model_name == model_name,
                )
            )
            .first()
        )

        if not perf:
            perf = StrategyPerformanceModel(
                strategy_name=strategy_name,
                model_name=model_name,
            )
            self.session.add(perf)
            self.session.commit()
            self.session.refresh(perf)

        return perf

    def update(
        self,
        strategy_name: str,
        model_name: str,
        score: float,
        latency_ms: float,
        cost: float,
        success: bool,
    ):
        """Update performance metrics."""
        perf = self.get_or_create(strategy_name, model_name)

        perf.total_requests += 1
        if success:
            perf.successful_requests += 1

        # Update running averages
        perf.success_rate = perf.successful_requests / perf.total_requests
        perf.average_score = (
            perf.average_score * (perf.total_requests - 1) + score
        ) / perf.total_requests
        perf.average_latency_ms = (
            perf.average_latency_ms * (perf.total_requests - 1) + latency_ms
        ) / perf.total_requests
        perf.average_cost = (
            perf.average_cost * (perf.total_requests - 1) + cost
        ) / perf.total_requests
        perf.last_updated = datetime.now(timezone.utc)

        self.session.commit()

    def list_by_strategy(self, strategy_name: str) -> List[StrategyPerformanceModel]:
        """List performance records for a strategy."""
        return (
            self.session.query(StrategyPerformanceModel)
            .filter(StrategyPerformanceModel.strategy_name == strategy_name)
            .all()
        )


class RateLimitRepository:
    """Repository for Rate Limiting operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def record(self, key_id: str, timestamp: Optional[datetime] = None):
        """Record a rate limit entry."""
        entry = RateLimitEntryModel(
            key_id=key_id,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        self.session.add(entry)
        self.session.commit()

    def get_count_last_hour(self, key_id: str) -> int:
        """Get number of requests in last hour for a key."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        return (
            self.session.query(func.count(RateLimitEntryModel.id))
            .filter(
                and_(
                    RateLimitEntryModel.key_id == key_id,
                    RateLimitEntryModel.timestamp >= cutoff,
                )
            )
            .scalar()
            or 0
        )

    def cleanup_old_entries(self, days: int = 7):
        """Clean up old rate limit entries."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        self.session.query(RateLimitEntryModel).filter(
            RateLimitEntryModel.timestamp < cutoff
        ).delete()
        self.session.commit()
        logger.info(f"Cleaned up rate limit entries older than {days} days")
