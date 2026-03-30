"""
Database-backed cost tracking.

Provides the same interface as CostTracker but with SQLite persistence.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from nexus.core.tracking.cost_tracker import CostSummary
from nexus.core.database import get_db
from nexus.core.database.repositories import CostEntryRepository

logger = logging.getLogger(__name__)


class PersistentCostTracker:
    """
    Database-backed cost tracker.

    Features:
    - Per-model cost tracking
    - Per-user cost tracking
    - Budget alerts
    - Historical data (persisted)
    - Cost analytics
    """

    def __init__(self, budget_limit_usd: float = 100.0, alert_threshold: float = 0.8):
        """
        Initialize cost tracker.

        Args:
            budget_limit_usd: Monthly budget limit in USD
            alert_threshold: Alert when this percentage of budget is reached
        """
        self.budget_limit = budget_limit_usd
        self.alert_threshold = alert_threshold
        self.alerts_sent = set()  # Still in-memory, but alerts are transient
        self.db = get_db()
        logger.info(f"PersistentCostTracker initialized (budget=${budget_limit_usd}, threshold={alert_threshold*100}%)")

    def record_cost(
        self,
        model_name: str,
        provider: str,
        tokens_used: int,
        cost_usd: float,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """
        Record a cost entry.

        Args:
            model_name: Name of the model used
            provider: Provider name (openai, anthropic, etc.)
            tokens_used: Number of tokens used
            cost_usd: Cost in USD
            user_id: Optional user ID
            request_id: Optional request ID
        """
        with self.db.get_session() as session:
            cost_repo = CostEntryRepository(session)
            cost_repo.create(
                model_name=model_name,
                provider=provider,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                user_id=user_id,
                request_id=request_id,
            )

        logger.debug(
            f"Recorded cost: {model_name} ${cost_usd:.4f} "
            f"({tokens_used} tokens, user={user_id})"
        )

        # Check budget and send alerts if needed
        self._check_budget_alerts()

    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> CostSummary:
        """
        Get cost summary for a period.

        Args:
            start_date: Start of period (default: 30 days ago)
            end_date: End of period (default: now)
            user_id: Optional user filter

        Returns:
            CostSummary object
        """
        # Default to last 30 days
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        with self.db.get_session() as session:
            cost_repo = CostEntryRepository(session)

            # Get all entries for period
            entries = cost_repo.list_by_date_range(
                start_date=start_date,
                end_date=end_date,
                user_id=user_id,
            )

            # Calculate summary
            summary = CostSummary(
                period_start=start_date,
                period_end=end_date,
            )

            summary.total_requests = len(entries)
            summary.total_cost = sum(e.cost_usd for e in entries)
            summary.total_tokens = sum(e.tokens_used for e in entries)

            # Cost by model
            cost_by_model = {}
            for entry in entries:
                cost_by_model[entry.model_name] = (
                    cost_by_model.get(entry.model_name, 0.0) + entry.cost_usd
                )
            summary.cost_by_model = cost_by_model

            # Cost by provider
            cost_by_provider = {}
            for entry in entries:
                cost_by_provider[entry.provider] = (
                    cost_by_provider.get(entry.provider, 0.0) + entry.cost_usd
                )
            summary.cost_by_provider = cost_by_provider

            # Cost by user
            cost_by_user = {}
            for entry in entries:
                if entry.user_id:
                    cost_by_user[entry.user_id] = (
                        cost_by_user.get(entry.user_id, 0.0) + entry.cost_usd
                    )
            summary.cost_by_user = cost_by_user

            return summary

    def get_budget_status(self) -> dict:
        """
        Get current budget status.

        Returns:
            Dictionary with budget information
        """
        # Get current month's spend
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        with self.db.get_session() as session:
            cost_repo = CostEntryRepository(session)
            current_spend = cost_repo.get_total_cost(
                start_date=month_start,
                end_date=now,
            )

        remaining = self.budget_limit - current_spend
        percentage = (current_spend / self.budget_limit * 100) if self.budget_limit > 0 else 0

        return {
            "budget_limit": self.budget_limit,
            "current_spend": current_spend,
            "remaining": remaining,
            "percentage_used": percentage,
            "alert_threshold": self.alert_threshold * 100,
        }

    def _check_budget_alerts(self):
        """Check if budget alerts should be sent."""
        status = self.get_budget_status()
        percentage = status["percentage_used"] / 100.0

        # Check 80% threshold
        if percentage >= self.alert_threshold and "80%" not in self.alerts_sent:
            logger.warning(
                f"Budget alert: {percentage*100:.1f}% of monthly budget used "
                f"(${status['current_spend']:.2f} / ${self.budget_limit:.2f})"
            )
            self.alerts_sent.add("80%")

        # Check 100% threshold
        if percentage >= 1.0 and "100%" not in self.alerts_sent:
            logger.error(
                f"Budget exceeded: {percentage*100:.1f}% of monthly budget used "
                f"(${status['current_spend']:.2f} / ${self.budget_limit:.2f})"
            )
            self.alerts_sent.add("100%")

    def reset_alerts(self):
        """Reset alert flags (typically called monthly)."""
        self.alerts_sent.clear()
        logger.info("Budget alerts reset")
