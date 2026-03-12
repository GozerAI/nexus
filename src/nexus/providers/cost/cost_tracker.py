"""
Cost Tracking System

Provides comprehensive cost tracking with:
- Per-model and per-user cost attribution
- Budget alerts and monitoring
- Historical cost data with database persistence
- Cost analytics and export capabilities
- Integration with database module

Adapted from: TheNexus/src/thenexus/tracking/cost_tracker.py
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """
    Single cost entry for tracking API usage costs.

    Attributes:
        timestamp: When the cost was incurred
        model_name: Name of the model used
        provider: Provider name (openai, anthropic, etc.)
        tokens_used: Number of tokens consumed
        cost_usd: Cost in USD
        user_id: Optional user identifier
        request_id: Optional request identifier
    """
    timestamp: datetime
    model_name: str
    provider: str
    tokens_used: int
    cost_usd: float
    user_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class CostSummary:
    """
    Cost summary statistics for a given period.

    Attributes:
        total_cost: Total cost in USD
        total_requests: Number of requests
        total_tokens: Total tokens consumed
        cost_by_model: Cost breakdown by model
        cost_by_provider: Cost breakdown by provider
        cost_by_user: Cost breakdown by user
        period_start: Start of summary period
        period_end: End of summary period
    """
    total_cost: float = 0.0
    total_requests: int = 0
    total_tokens: int = 0
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    cost_by_provider: Dict[str, float] = field(default_factory=dict)
    cost_by_user: Dict[str, float] = field(default_factory=dict)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class CostTracker:
    """
    Tracks API costs and usage with budget management.

    Features:
    - Per-model cost tracking
    - Per-user cost tracking
    - Budget alerts
    - Historical data (in-memory)
    - Cost analytics
    - CSV/JSON export

    Example:
        >>> tracker = CostTracker(budget_limit_usd=100.0)
        >>> tracker.record_cost(
        ...     model_name="gpt-4",
        ...     provider="openai",
        ...     tokens_used=500,
        ...     cost_usd=0.015,
        ...     user_id="user123"
        ... )
        >>> summary = tracker.get_summary()
        >>> print(f"Total cost: ${summary.total_cost:.2f}")
    """

    def __init__(
        self,
        budget_limit_usd: float = 100.0,
        alert_threshold: float = 0.8,
        max_entries: int = 100000
    ):
        """
        Initialize cost tracker.

        Args:
            budget_limit_usd: Monthly budget limit in USD (default: $100)
            alert_threshold: Alert when this percentage of budget is reached (default: 0.8 = 80%)
            max_entries: Maximum number of cost entries to retain (default: 100,000)
                        Prevents unbounded memory growth. Oldest entries are automatically
                        removed when limit is reached.

        Raises:
            ValueError: If budget_limit_usd <= 0 or alert_threshold not in (0, 1]
        """
        if budget_limit_usd <= 0:
            raise ValueError(f"budget_limit_usd must be > 0, got {budget_limit_usd}")

        if not 0 < alert_threshold <= 1.0:
            raise ValueError(f"alert_threshold must be in (0, 1], got {alert_threshold}")

        if max_entries <= 0:
            raise ValueError(f"max_entries must be > 0, got {max_entries}")

        self.budget_limit = budget_limit_usd
        self.alert_threshold = alert_threshold
        self.max_entries = max_entries
        # FIXED: Use deque with maxlen to prevent unbounded memory growth
        self.entries: deque = deque(maxlen=max_entries)
        self.alerts_sent: set = set()
        self._lock = threading.RLock()  # Thread-safe cache operations

        # Incremental budget tracking (performance optimization)
        now = datetime.now(timezone.utc)
        self._current_month = (now.year, now.month)
        self._monthly_cost_cache = 0.0
        self._daily_cost_cache = 0.0
        self._current_day = (now.year, now.month, now.day)

        logger.info(
            f"💰 CostTracker initialized (thread-safe) "
            f"(budget=${budget_limit_usd}, threshold={alert_threshold*100}%, "
            f"max_entries={max_entries})"
        )

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

        Raises:
            ValueError: If validation fails for any parameter
        """
        # Validate inputs
        if not model_name or not isinstance(model_name, str):
            raise ValueError(f"model_name must be non-empty string, got {model_name!r}")

        if not provider or not isinstance(provider, str):
            raise ValueError(f"provider must be non-empty string, got {provider!r}")

        if not isinstance(tokens_used, int) or tokens_used < 0:
            raise ValueError(f"tokens_used must be >= 0, got {tokens_used}")

        if not isinstance(cost_usd, (int, float)) or cost_usd < 0:
            raise ValueError(f"cost_usd must be >= 0, got {cost_usd}")

        entry = CostEntry(
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            provider=provider,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            user_id=user_id,
            request_id=request_id
        )

        self.entries.append(entry)

        # Update incremental caches for performance (thread-safe)
        with self._lock:
            now = datetime.now(timezone.utc)
            current_month = (now.year, now.month)
            current_day = (now.year, now.month, now.day)

            # Reset monthly cache if new month
            if current_month != self._current_month:
                self._current_month = current_month
                self._monthly_cost_cache = cost_usd
            else:
                self._monthly_cost_cache += cost_usd

            # Reset daily cache if new day
            if current_day != self._current_day:
                self._current_day = current_day
                self._daily_cost_cache = cost_usd
            else:
                self._daily_cost_cache += cost_usd

        logger.info(
            f"💵 Cost recorded: {model_name} - ${cost_usd:.4f} "
            f"({tokens_used} tokens, user={user_id})"
        )

        # Check budget
        self._check_budget_alert()

    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None
    ) -> CostSummary:
        """
        Get cost summary for a period.

        Args:
            start_date: Start of period (default: beginning of current month)
            end_date: End of period (default: now)
            user_id: Optional user filter

        Returns:
            CostSummary object with aggregated statistics
        """
        # Default to current month
        if start_date is None:
            now = datetime.now(timezone.utc)
            start_date = datetime(now.year, now.month, 1)

        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Filter entries
        filtered_entries = [
            e for e in self.entries
            if start_date <= e.timestamp <= end_date
            and (user_id is None or e.user_id == user_id)
        ]

        # Calculate summary
        summary = CostSummary(
            period_start=start_date,
            period_end=end_date
        )

        for entry in filtered_entries:
            summary.total_cost += entry.cost_usd
            summary.total_requests += 1
            summary.total_tokens += entry.tokens_used

            # By model
            if entry.model_name not in summary.cost_by_model:
                summary.cost_by_model[entry.model_name] = 0.0
            summary.cost_by_model[entry.model_name] += entry.cost_usd

            # By provider
            if entry.provider not in summary.cost_by_provider:
                summary.cost_by_provider[entry.provider] = 0.0
            summary.cost_by_provider[entry.provider] += entry.cost_usd

            # By user
            if entry.user_id:
                if entry.user_id not in summary.cost_by_user:
                    summary.cost_by_user[entry.user_id] = 0.0
                summary.cost_by_user[entry.user_id] += entry.cost_usd

        return summary

    def get_monthly_cost(self) -> float:
        """
        Get total cost for current month (optimized with cache, thread-safe).

        Returns:
            Total cost in USD
        """
        # Use cached value for performance (O(1) instead of O(n))
        with self._lock:
            now = datetime.now(timezone.utc)
            current_month = (now.year, now.month)

            # Invalidate cache if month changed
            if current_month != self._current_month:
                self._current_month = current_month
                self._monthly_cost_cache = 0.0

            return self._monthly_cost_cache

    def get_daily_cost(self) -> float:
        """
        Get total cost for today (optimized with cache, thread-safe).

        Returns:
            Total cost in USD
        """
        # Use cached value for performance (O(1) instead of O(n))
        with self._lock:
            now = datetime.now(timezone.utc)
            current_day = (now.year, now.month, now.day)

            # Invalidate cache if day changed
            if current_day != self._current_day:
                self._current_day = current_day
                self._daily_cost_cache = 0.0

            return self._daily_cost_cache

    def is_over_budget(self) -> bool:
        """
        Check if over monthly budget.

        Returns:
            True if current month's costs exceed budget limit
        """
        return self.get_monthly_cost() >= self.budget_limit

    def get_budget_status(self) -> dict:
        """
        Get current budget status.

        Returns:
            Dictionary with:
                - budget_limit: Monthly budget limit
                - current_spend: Current month's spending
                - remaining: Budget remaining
                - percent_used: Percentage of budget used
                - is_over_budget: Whether over budget
                - alert_threshold: Alert threshold percentage
        """
        monthly_cost = self.get_monthly_cost()
        remaining = max(0, self.budget_limit - monthly_cost)
        percent_used = (monthly_cost / self.budget_limit * 100) if self.budget_limit > 0 else 0

        return {
            "budget_limit": self.budget_limit,
            "current_spend": round(monthly_cost, 2),
            "remaining": round(remaining, 2),
            "percent_used": round(percent_used, 2),
            "is_over_budget": self.is_over_budget(),
            "alert_threshold": self.alert_threshold * 100,
        }

    def get_top_costs(self, limit: int = 10) -> List[tuple]:
        """
        Get top cost entries by model.

        Args:
            limit: Number of entries to return (default: 10)

        Returns:
            List of (model_name, cost) tuples sorted by cost descending
        """
        summary = self.get_summary()
        sorted_costs = sorted(
            summary.cost_by_model.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_costs[:limit]

    def export_data(self, filepath: str, format: str = "csv"):
        """
        Export cost data to file.

        Args:
            filepath: Output file path
            format: Export format ("csv" or "json")
        """
        import json
        import csv

        if format == "json":
            data = [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "model_name": e.model_name,
                    "provider": e.provider,
                    "tokens_used": e.tokens_used,
                    "cost_usd": e.cost_usd,
                    "user_id": e.user_id,
                    "request_id": e.request_id,
                }
                for e in self.entries
            ]

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == "csv":
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "model_name", "provider",
                    "tokens_used", "cost_usd", "user_id", "request_id"
                ])

                for e in self.entries:
                    writer.writerow([
                        e.timestamp.isoformat(),
                        e.model_name,
                        e.provider,
                        e.tokens_used,
                        e.cost_usd,
                        e.user_id or "",
                        e.request_id or "",
                    ])

        logger.info(f"📊 Exported {len(self.entries)} cost entries to {filepath}")

    def _check_budget_alert(self):
        """Check if budget alert should be triggered."""
        monthly_cost = self.get_monthly_cost()
        alert_level = self.budget_limit * self.alert_threshold

        if monthly_cost >= alert_level:
            alert_key = f"{datetime.now(timezone.utc).strftime('%Y-%m')}-alert"

            if alert_key not in self.alerts_sent:
                self.alerts_sent.add(alert_key)
                logger.warning(
                    f"⚠️ BUDGET ALERT: ${monthly_cost:.2f} spent "
                    f"(${self.budget_limit:.2f} budget, "
                    f"{self.alert_threshold*100}% threshold)"
                )
