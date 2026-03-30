"""
Budget Management System

Provides multi-level budget management:
- Per-user budgets
- Per-team budgets
- Per-organization budgets
- Soft limits (warnings) and hard limits (blocking)
- Budget period management (daily, weekly, monthly, custom)
- Automatic budget reset
- Budget utilization tracking

Built on top of unified_intelligence.tracking.cost_tracker
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import RLock

logger = logging.getLogger(__name__)


class BudgetPeriod(Enum):
    """Budget period types."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class BudgetLimitType(Enum):
    """Budget limit enforcement types."""
    SOFT = "soft"  # Warning only
    HARD = "hard"  # Block requests


@dataclass
class BudgetLimit:
    """
    Budget limit configuration.

    Attributes:
        limit_usd: Budget limit in USD
        period: Budget period (daily, weekly, monthly, custom)
        limit_type: Soft (warning) or hard (blocking)
        alert_thresholds: List of alert thresholds (e.g., [0.5, 0.8, 0.9])
        period_start: Custom period start (for custom periods)
        period_end: Custom period end (for custom periods)
    """
    limit_usd: float
    period: BudgetPeriod
    limit_type: BudgetLimitType = BudgetLimitType.SOFT
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9, 1.0])
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


@dataclass
class BudgetStatus:
    """
    Current budget status.

    Attributes:
        entity_id: Budget entity ID (user_id, team_id, org_id)
        entity_type: Type of entity (user, team, organization)
        limit_usd: Budget limit in USD
        spent_usd: Amount spent in current period
        remaining_usd: Amount remaining
        percent_used: Percentage of budget used
        period: Budget period
        period_start: Current period start
        period_end: Current period end
        is_exceeded: Whether budget is exceeded
        is_blocked: Whether requests should be blocked
        alerts_triggered: List of triggered alert thresholds
    """
    entity_id: str
    entity_type: str
    limit_usd: float
    spent_usd: float
    remaining_usd: float
    percent_used: float
    period: BudgetPeriod
    period_start: datetime
    period_end: datetime
    is_exceeded: bool
    is_blocked: bool
    alerts_triggered: List[float] = field(default_factory=list)


class BudgetManager:
    """
    Multi-level budget management system.

    Features:
    - Hierarchical budgets (user → team → organization)
    - Multiple budget periods
    - Soft and hard limits
    - Automatic budget reset
    - Alert management
    - Budget utilization tracking

    Example:
        >>> from unified_intelligence.tracking import CostTracker, BudgetManager
        >>>
        >>> cost_tracker = CostTracker()
        >>> budget_manager = BudgetManager(cost_tracker)
        >>>
        >>> # Set user budget
        >>> budget_manager.set_budget(
        ...     entity_id="user123",
        ...     entity_type="user",
        ...     limit=BudgetLimit(
        ...         limit_usd=100.0,
        ...         period=BudgetPeriod.MONTHLY,
        ...         limit_type=BudgetLimitType.HARD,
        ...         alert_thresholds=[0.8, 0.9, 1.0]
        ...     )
        ... )
        >>>
        >>> # Check if request would exceed budget
        >>> if budget_manager.would_exceed_budget("user123", "user", 5.0):
        ...     raise BudgetExceededException("Budget exceeded")
        >>>
        >>> # Get budget status
        >>> status = budget_manager.get_budget_status("user123", "user")
        >>> print(f"Budget used: {status.percent_used:.1f}%")
    """

    def __init__(self, cost_tracker: 'CostTracker', enable_metrics: bool = True):
        """
        Initialize budget manager.

        Args:
            cost_tracker: CostTracker instance for cost data
            enable_metrics: Enable Prometheus metrics integration
        """
        self.cost_tracker = cost_tracker
        self.budgets: Dict[Tuple[str, str], BudgetLimit] = {}  # (entity_id, entity_type) -> BudgetLimit
        self.alerts_sent: Dict[Tuple[str, str, float], datetime] = {}  # Track sent alerts
        self._lock = RLock()
        self.enable_metrics = enable_metrics

        # Phase 1 Week 1: Prometheus metrics integration
        if self.enable_metrics:
            from nexus.providers.monitoring import get_metrics
            self.metrics = get_metrics()
        else:
            self.metrics = None

        logger.info("💰 BudgetManager initialized")

    def set_budget(
        self,
        entity_id: str,
        entity_type: str,
        limit: BudgetLimit
    ):
        """
        Set budget for an entity.

        Args:
            entity_id: Entity identifier (user_id, team_id, org_id)
            entity_type: Type of entity ("user", "team", "organization")
            limit: Budget limit configuration

        Raises:
            ValueError: If limit_usd <= 0 or invalid thresholds
        """
        if limit.limit_usd <= 0:
            raise ValueError(f"limit_usd must be > 0, got {limit.limit_usd}")

        for threshold in limit.alert_thresholds:
            if not 0 < threshold <= 1.0:
                raise ValueError(f"Alert thresholds must be in (0, 1], got {threshold}")

        with self._lock:
            self.budgets[(entity_id, entity_type)] = limit

        # Phase 1 Week 1: Update Prometheus metrics
        if self.metrics:
            self.metrics.budget_limit_usd.set(
                limit.limit_usd,
                entity_id=entity_id,
                entity_type=entity_type,
                period=limit.period.value
            )

        logger.info(
            f"💵 Budget set: {entity_type}={entity_id}, "
            f"${limit.limit_usd:.2f}/{limit.period.value}, "
            f"type={limit.limit_type.value}"
        )

    def remove_budget(self, entity_id: str, entity_type: str):
        """Remove budget for an entity."""
        with self._lock:
            key = (entity_id, entity_type)
            if key in self.budgets:
                del self.budgets[key]
                logger.info(f"Budget removed: {entity_type}={entity_id}")

    def get_budget_status(
        self,
        entity_id: str,
        entity_type: str
    ) -> Optional[BudgetStatus]:
        """
        Get current budget status for an entity.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity

        Returns:
            BudgetStatus object or None if no budget set
        """
        with self._lock:
            key = (entity_id, entity_type)
            if key not in self.budgets:
                return None

            limit = self.budgets[key]

        # Calculate period boundaries
        period_start, period_end = self._get_period_boundaries(limit)

        # Get cost for period
        if entity_type == "user":
            summary = self.cost_tracker.get_summary(
                start_date=period_start,
                end_date=period_end,
                user_id=entity_id
            )
        else:
            # For team/org, we'd need additional logic
            # For now, use total costs (could be enhanced with team/org tracking)
            summary = self.cost_tracker.get_summary(
                start_date=period_start,
                end_date=period_end
            )

        spent_usd = summary.total_cost
        remaining_usd = max(0, limit.limit_usd - spent_usd)
        percent_used = (spent_usd / limit.limit_usd) if limit.limit_usd > 0 else 0

        # Check triggered alerts
        alerts_triggered = [
            threshold for threshold in limit.alert_thresholds
            if percent_used >= threshold
        ]

        # Determine if exceeded and blocked
        is_exceeded = spent_usd >= limit.limit_usd
        is_blocked = is_exceeded and limit.limit_type == BudgetLimitType.HARD

        status = BudgetStatus(
            entity_id=entity_id,
            entity_type=entity_type,
            limit_usd=limit.limit_usd,
            spent_usd=spent_usd,
            remaining_usd=remaining_usd,
            percent_used=percent_used,
            period=limit.period,
            period_start=period_start,
            period_end=period_end,
            is_exceeded=is_exceeded,
            is_blocked=is_blocked,
            alerts_triggered=alerts_triggered
        )

        # Phase 1 Week 1: Update Prometheus metrics
        if self.metrics:
            self.metrics.budget_spent_usd.set(
                spent_usd,
                entity_id=entity_id,
                entity_type=entity_type,
                period=limit.period.value
            )

            self.metrics.budget_utilization.set(
                percent_used,
                entity_id=entity_id,
                entity_type=entity_type,
                period=limit.period.value
            )

            if is_exceeded:
                self.metrics.budget_exceeded_total.inc(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    limit_type=limit.limit_type.value
                )

        return status

    def would_exceed_budget(
        self,
        entity_id: str,
        entity_type: str,
        additional_cost_usd: float
    ) -> bool:
        """
        Check if additional cost would exceed budget.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity
            additional_cost_usd: Proposed additional cost

        Returns:
            True if would exceed budget and limit type is HARD
        """
        status = self.get_budget_status(entity_id, entity_type)

        if status is None:
            # No budget set, allow
            return False

        # Check if adding this cost would exceed limit
        projected_spend = status.spent_usd + additional_cost_usd

        if projected_spend > status.limit_usd:
            # Check if this is a hard limit
            with self._lock:
                limit = self.budgets.get((entity_id, entity_type))
                if limit and limit.limit_type == BudgetLimitType.HARD:
                    return True

        return False

    def check_and_alert(
        self,
        entity_id: str,
        entity_type: str
    ) -> List[float]:
        """
        Check budget and send alerts if thresholds crossed.

        Args:
            entity_id: Entity identifier
            entity_type: Type of entity

        Returns:
            List of newly triggered alert thresholds
        """
        status = self.get_budget_status(entity_id, entity_type)

        if status is None:
            return []

        new_alerts = []

        for threshold in status.alerts_triggered:
            alert_key = (entity_id, entity_type, threshold)

            # Check if we've already alerted for this threshold in this period
            with self._lock:
                last_alert = self.alerts_sent.get(alert_key)

                if last_alert is None or last_alert < status.period_start:
                    # New alert for this threshold in this period
                    self.alerts_sent[alert_key] = datetime.now(timezone.utc)
                    new_alerts.append(threshold)

                    # Phase 1 Week 1: Record alert in Prometheus
                    if self.metrics:
                        self.metrics.budget_alerts_total.inc(
                            entity_id=entity_id,
                            entity_type=entity_type,
                            threshold=str(int(threshold * 100))
                        )

                    logger.warning(
                        f"⚠️ BUDGET ALERT: {entity_type}={entity_id}, "
                        f"{threshold*100:.0f}% threshold reached "
                        f"(${status.spent_usd:.2f}/${status.limit_usd:.2f})"
                    )

        return new_alerts

    def get_all_budget_statuses(self) -> List[BudgetStatus]:
        """Get budget statuses for all entities."""
        statuses = []

        with self._lock:
            for (entity_id, entity_type) in self.budgets.keys():
                status = self.get_budget_status(entity_id, entity_type)
                if status:
                    statuses.append(status)

        return statuses

    def get_budgets_over_limit(self) -> List[BudgetStatus]:
        """Get all budgets that are over their limits."""
        return [
            status for status in self.get_all_budget_statuses()
            if status.is_exceeded
        ]

    def reset_budget(self, entity_id: str, entity_type: str):
        """
        Manually reset budget for an entity.

        Note: This doesn't clear cost data, just resets alert tracking.
        Budget will automatically reset at period boundaries.
        """
        with self._lock:
            # Clear alert tracking for this entity
            keys_to_remove = [
                key for key in self.alerts_sent.keys()
                if key[0] == entity_id and key[1] == entity_type
            ]

            for key in keys_to_remove:
                del self.alerts_sent[key]

        logger.info(f"Budget reset: {entity_type}={entity_id}")

    def _get_period_boundaries(
        self,
        limit: BudgetLimit
    ) -> Tuple[datetime, datetime]:
        """
        Calculate period start and end dates.

        Args:
            limit: Budget limit with period configuration

        Returns:
            Tuple of (period_start, period_end)
        """
        now = datetime.now(timezone.utc)

        if limit.period == BudgetPeriod.DAILY:
            start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            end = start + timedelta(days=1)

        elif limit.period == BudgetPeriod.WEEKLY:
            # Week starts on Monday
            days_since_monday = now.weekday()
            start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=days_since_monday)
            end = start + timedelta(weeks=1)

        elif limit.period == BudgetPeriod.MONTHLY:
            start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
            # Next month
            if now.month == 12:
                end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)

        elif limit.period == BudgetPeriod.CUSTOM:
            if limit.period_start is None or limit.period_end is None:
                raise ValueError("Custom period requires period_start and period_end")
            start = limit.period_start
            end = limit.period_end

        else:
            raise ValueError(f"Unknown budget period: {limit.period}")

        return start, end


class BudgetExceededException(Exception):
    """Exception raised when budget limit is exceeded."""
    pass
