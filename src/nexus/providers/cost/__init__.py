"""
Tracking Module for Unified Intelligence

Provides comprehensive tracking capabilities with:
- Cost tracking with budget alerts
- Multi-level budget management (user/team/org)
- Cost analytics and forecasting
- Usage analytics with performance metrics
- Historical data with export capabilities
- Per-user and per-model attribution

Components:
- **CostTracker**: Tracks API costs with budget management
- **BudgetManager**: Multi-level budget limits and alerts (Phase 1)
- **CostAnalytics**: Cost analysis, optimization, forecasting (Phase 1)
- **UsageTracker**: Tracks usage analytics and performance
- **CostEntry/UsageEntry**: Individual tracking records
- **CostSummary/UsageStats**: Aggregated statistics

Example:
    >>> from unified_intelligence.tracking import (
    ...     CostTracker, BudgetManager, CostAnalytics,
    ...     BudgetPeriod, BudgetLimitType, BudgetLimit
    ... )
    >>>
    >>> # Track costs
    >>> cost_tracker = CostTracker(budget_limit_usd=100.0)
    >>> cost_tracker.record_cost(
    ...     model_name="gpt-4",
    ...     provider="openai",
    ...     tokens_used=500,
    ...     cost_usd=0.015,
    ...     user_id="user123"
    ... )
    >>>
    >>> # Set user budget (Phase 1)
    >>> budget_manager = BudgetManager(cost_tracker)
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
    >>> # Check budget before request
    >>> if budget_manager.would_exceed_budget("user123", "user", 5.0):
    ...     raise Exception("Budget exceeded")
    >>>
    >>> # Get analytics (Phase 1)
    >>> analytics = CostAnalytics(cost_tracker)
    >>> efficiencies = analytics.analyze_model_efficiency()
    >>> recommendations = analytics.get_optimization_recommendations()
    >>> forecast = analytics.forecast_costs(days=30)

Adapted from: TheNexus/src/thenexus/tracking/
Enhanced: Phase 1 Week 1 (Budget Management & Cost Analytics)
"""

from nexus.providers.cost.cost_tracker import (
    CostTracker,
    CostEntry,
    CostSummary,
)

# Phase 1 Week 1: Budget Management
from nexus.providers.cost.budget_manager import (
    BudgetManager,
    BudgetLimit,
    BudgetStatus,
    BudgetPeriod,
    BudgetLimitType,
    BudgetExceededException,
)

# Phase 1 Week 1: Cost Analytics
from nexus.providers.cost.cost_analytics import (
    CostAnalytics,
    ModelEfficiency,
    CostAnomaly,
    CostOptimization,
    CostForecast,
    CostTier,
)

from nexus.providers.cost.usage_tracker import (
    UsageTracker,
    UsageEntry,
    UsageStats,
)

__all__ = [
    # Cost tracking
    "CostTracker",
    "CostEntry",
    "CostSummary",

    # Budget management (Phase 1 Week 1)
    "BudgetManager",
    "BudgetLimit",
    "BudgetStatus",
    "BudgetPeriod",
    "BudgetLimitType",
    "BudgetExceededException",

    # Cost analytics (Phase 1 Week 1)
    "CostAnalytics",
    "ModelEfficiency",
    "CostAnomaly",
    "CostOptimization",
    "CostForecast",
    "CostTier",

    # Usage tracking
    "UsageTracker",
    "UsageEntry",
    "UsageStats",
]
