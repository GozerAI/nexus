"""
Cost Analytics and Forecasting System

Provides advanced cost analysis and optimization:
- Cost per query analysis
- Model efficiency metrics (cost vs quality)
- Cost optimization recommendations
- Anomaly detection (unusual spending patterns)
- Cost trends and forecasting
- Cost savings tracking

Built on top of unified_intelligence.tracking.cost_tracker
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from statistics import mean, stdev
from enum import Enum

logger = logging.getLogger(__name__)


class CostTier(Enum):
    """Model cost tier classification."""
    ULTRA_LOW = "ultra_low"    # < $0.0001 per 1K tokens
    LOW = "low"                # $0.0001 - $0.001 per 1K tokens
    MEDIUM = "medium"          # $0.001 - $0.01 per 1K tokens
    HIGH = "high"              # $0.01 - $0.1 per 1K tokens
    PREMIUM = "premium"        # > $0.1 per 1K tokens


@dataclass
class ModelEfficiency:
    """
    Model efficiency metrics.

    Attributes:
        model_name: Model identifier
        provider: Provider name
        avg_cost_per_query: Average cost per query
        avg_tokens_per_query: Average tokens per query
        cost_per_1k_tokens: Cost per 1000 tokens
        total_queries: Total number of queries
        total_cost: Total cost incurred
        cost_tier: Model cost tier classification
        efficiency_score: 0-100 score (lower cost = higher score)
    """
    model_name: str
    provider: str
    avg_cost_per_query: float
    avg_tokens_per_query: float
    cost_per_1k_tokens: float
    total_queries: int
    total_cost: float
    cost_tier: CostTier
    efficiency_score: float


@dataclass
class CostAnomaly:
    """
    Detected cost anomaly.

    Attributes:
        timestamp: When anomaly occurred
        entity_id: User/team/org ID
        entity_type: Type of entity
        expected_cost: Expected cost based on historical data
        actual_cost: Actual cost observed
        deviation_percent: Percentage deviation from expected
        severity: Low/Medium/High/Critical
        description: Human-readable description
    """
    timestamp: datetime
    entity_id: str
    entity_type: str
    expected_cost: float
    actual_cost: float
    deviation_percent: float
    severity: str
    description: str


@dataclass
class CostOptimization:
    """
    Cost optimization recommendation.

    Attributes:
        recommendation_type: Type of recommendation
        current_model: Current model being used
        suggested_model: Suggested alternative model
        estimated_savings_usd: Estimated monthly savings
        estimated_savings_percent: Estimated percentage savings
        quality_impact: Expected quality impact (none/minimal/moderate)
        confidence: Confidence in recommendation (0-100)
        rationale: Explanation of recommendation
    """
    recommendation_type: str
    current_model: str
    suggested_model: str
    estimated_savings_usd: float
    estimated_savings_percent: float
    quality_impact: str
    confidence: float
    rationale: str


@dataclass
class CostForecast:
    """
    Cost forecast for future period.

    Attributes:
        forecast_period: Period being forecast (day/week/month)
        forecast_start: Start of forecast period
        forecast_end: End of forecast period
        predicted_cost: Predicted cost for period
        confidence_interval_low: Lower bound (95% CI)
        confidence_interval_high: Upper bound (95% CI)
        trend: Trend direction (increasing/decreasing/stable)
        trend_percent: Percentage change from previous period
        factors: Key factors influencing forecast
    """
    forecast_period: str
    forecast_start: datetime
    forecast_end: datetime
    predicted_cost: float
    confidence_interval_low: float
    confidence_interval_high: float
    trend: str
    trend_percent: float
    factors: List[str] = field(default_factory=list)


class CostAnalytics:
    """
    Advanced cost analytics and optimization system.

    Features:
    - Real-time cost per query analysis
    - Model efficiency comparison
    - Automated cost optimization recommendations
    - Anomaly detection with severity scoring
    - Time-series forecasting
    - Cost savings tracking

    Example:
        >>> from unified_intelligence.tracking import CostTracker, CostAnalytics
        >>>
        >>> cost_tracker = CostTracker()
        >>> analytics = CostAnalytics(cost_tracker)
        >>>
        >>> # Get model efficiency metrics
        >>> efficiencies = analytics.analyze_model_efficiency()
        >>> for eff in efficiencies:
        ...     print(f"{eff.model_name}: ${eff.cost_per_1k_tokens:.4f}/1K tokens")
        >>>
        >>> # Get optimization recommendations
        >>> recommendations = analytics.get_optimization_recommendations()
        >>> for rec in recommendations:
        ...     print(f"Switch {rec.current_model} -> {rec.suggested_model}")
        ...     print(f"Save ${rec.estimated_savings_usd:.2f}/month")
        >>>
        >>> # Detect anomalies
        >>> anomalies = analytics.detect_cost_anomalies()
        >>> for anomaly in anomalies:
        ...     if anomaly.severity == "Critical":
        ...         print(f"ALERT: {anomaly.description}")
        >>>
        >>> # Forecast costs
        >>> forecast = analytics.forecast_costs(days=30)
        >>> print(f"Predicted cost: ${forecast.predicted_cost:.2f}")
    """

    def __init__(self, cost_tracker: 'CostTracker'):
        """
        Initialize cost analytics.

        Args:
            cost_tracker: CostTracker instance for cost data
        """
        self.cost_tracker = cost_tracker
        self._anomaly_threshold = 2.0  # Standard deviations for anomaly detection

        logger.info("📊 CostAnalytics initialized")

    def analyze_model_efficiency(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ModelEfficiency]:
        """
        Analyze efficiency of different models.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            List of ModelEfficiency objects sorted by efficiency score
        """
        # Get cost data by model
        model_stats = defaultdict(lambda: {
            'total_cost': 0.0,
            'total_tokens': 0,
            'total_queries': 0
        })

        # Aggregate data from cost tracker
        entries = list(self.cost_tracker.cost_history)

        if start_date or end_date:
            entries = [
                e for e in entries
                if (start_date is None or e.timestamp >= start_date) and
                   (end_date is None or e.timestamp <= end_date)
            ]

        for entry in entries:
            key = (entry.model_name, entry.provider)
            model_stats[key]['total_cost'] += entry.cost_usd
            model_stats[key]['total_tokens'] += entry.tokens_used
            model_stats[key]['total_queries'] += 1

        # Calculate efficiency metrics
        efficiencies = []

        for (model_name, provider), stats in model_stats.items():
            if stats['total_queries'] == 0:
                continue

            avg_cost_per_query = stats['total_cost'] / stats['total_queries']
            avg_tokens_per_query = stats['total_tokens'] / stats['total_queries']

            cost_per_1k_tokens = (
                (stats['total_cost'] / stats['total_tokens']) * 1000
                if stats['total_tokens'] > 0 else 0
            )

            # Classify cost tier
            cost_tier = self._classify_cost_tier(cost_per_1k_tokens)

            # Calculate efficiency score (0-100, higher is more efficient)
            efficiency_score = self._calculate_efficiency_score(cost_per_1k_tokens)

            efficiencies.append(ModelEfficiency(
                model_name=model_name,
                provider=provider,
                avg_cost_per_query=avg_cost_per_query,
                avg_tokens_per_query=avg_tokens_per_query,
                cost_per_1k_tokens=cost_per_1k_tokens,
                total_queries=stats['total_queries'],
                total_cost=stats['total_cost'],
                cost_tier=cost_tier,
                efficiency_score=efficiency_score
            ))

        # Sort by efficiency score (highest first)
        efficiencies.sort(key=lambda x: x.efficiency_score, reverse=True)

        return efficiencies

    def get_optimization_recommendations(
        self,
        min_queries: int = 10,
        min_savings_usd: float = 10.0
    ) -> List[CostOptimization]:
        """
        Generate cost optimization recommendations.

        Args:
            min_queries: Minimum queries to consider for recommendations
            min_savings_usd: Minimum monthly savings to recommend

        Returns:
            List of CostOptimization recommendations
        """
        recommendations = []

        # Get model efficiencies
        efficiencies = self.analyze_model_efficiency()

        # Filter models with sufficient data
        high_volume_models = [e for e in efficiencies if e.total_queries >= min_queries]

        if not high_volume_models:
            return recommendations

        # Find most efficient model in each tier
        best_by_tier = {}
        for eff in efficiencies:
            if eff.cost_tier not in best_by_tier:
                best_by_tier[eff.cost_tier] = eff
            elif eff.efficiency_score > best_by_tier[eff.cost_tier].efficiency_score:
                best_by_tier[eff.cost_tier] = eff

        # Compare each high-volume model against more efficient alternatives
        for current in high_volume_models:
            # Find cheaper alternatives in same or lower tier
            for alternative in efficiencies:
                if alternative.model_name == current.model_name:
                    continue

                # Only recommend if alternative is significantly cheaper
                if alternative.cost_per_1k_tokens >= current.cost_per_1k_tokens * 0.8:
                    continue

                # Calculate potential savings
                savings_per_query = current.avg_cost_per_query - alternative.avg_cost_per_query

                # Estimate monthly savings (assume 30 days, same query rate)
                days_of_data = 30  # Could calculate from actual data
                queries_per_month = (current.total_queries / days_of_data) * 30
                estimated_savings = savings_per_query * queries_per_month

                if estimated_savings < min_savings_usd:
                    continue

                savings_percent = (savings_per_query / current.avg_cost_per_query) * 100

                # Assess quality impact based on cost tier change
                quality_impact = self._assess_quality_impact(
                    current.cost_tier,
                    alternative.cost_tier
                )

                # Calculate confidence based on data quality
                confidence = min(100, (current.total_queries / 100) * 100)

                rationale = (
                    f"Switching from {current.model_name} to {alternative.model_name} "
                    f"could save ${estimated_savings:.2f}/month "
                    f"({savings_percent:.1f}% reduction). "
                    f"Based on {current.total_queries} queries analyzed."
                )

                recommendations.append(CostOptimization(
                    recommendation_type="model_substitution",
                    current_model=f"{current.provider}/{current.model_name}",
                    suggested_model=f"{alternative.provider}/{alternative.model_name}",
                    estimated_savings_usd=estimated_savings,
                    estimated_savings_percent=savings_percent,
                    quality_impact=quality_impact,
                    confidence=confidence,
                    rationale=rationale
                ))

                # Only recommend one alternative per model
                break

        # Sort by estimated savings (highest first)
        recommendations.sort(key=lambda x: x.estimated_savings_usd, reverse=True)

        return recommendations

    def detect_cost_anomalies(
        self,
        lookback_days: int = 7,
        entity_id: Optional[str] = None,
        entity_type: str = "user"
    ) -> List[CostAnomaly]:
        """
        Detect unusual cost patterns.

        Args:
            lookback_days: Days of historical data to analyze
            entity_id: Specific entity to analyze (None for all)
            entity_type: Type of entity (user/team/organization)

        Returns:
            List of detected anomalies
        """
        anomalies = []

        now = datetime.now(timezone.utc)
        lookback_start = now - timedelta(days=lookback_days)

        # Get daily costs for the lookback period
        daily_costs = defaultdict(float)

        for entry in self.cost_tracker.cost_history:
            if entry.timestamp < lookback_start:
                continue

            if entity_id and entry.user_id != entity_id:
                continue

            day_key = entry.timestamp.date()
            daily_costs[day_key] += entry.cost_usd

        if len(daily_costs) < 3:
            # Not enough data for anomaly detection
            return anomalies

        costs = list(daily_costs.values())
        mean_cost = mean(costs)

        if len(costs) >= 2:
            std_cost = stdev(costs)
        else:
            std_cost = 0

        # Check most recent day for anomalies
        today = now.date()
        if today in daily_costs:
            today_cost = daily_costs[today]

            if std_cost > 0:
                z_score = (today_cost - mean_cost) / std_cost

                if abs(z_score) >= self._anomaly_threshold:
                    deviation_percent = ((today_cost - mean_cost) / mean_cost) * 100

                    # Determine severity
                    if abs(z_score) >= 4:
                        severity = "Critical"
                    elif abs(z_score) >= 3:
                        severity = "High"
                    elif abs(z_score) >= 2.5:
                        severity = "Medium"
                    else:
                        severity = "Low"

                    if today_cost > mean_cost:
                        description = (
                            f"Unusually high spending detected: "
                            f"${today_cost:.2f} vs ${mean_cost:.2f} average "
                            f"(+{deviation_percent:.1f}%)"
                        )
                    else:
                        description = (
                            f"Unusually low spending detected: "
                            f"${today_cost:.2f} vs ${mean_cost:.2f} average "
                            f"({deviation_percent:.1f}%)"
                        )

                    anomalies.append(CostAnomaly(
                        timestamp=now,
                        entity_id=entity_id or "all",
                        entity_type=entity_type,
                        expected_cost=mean_cost,
                        actual_cost=today_cost,
                        deviation_percent=deviation_percent,
                        severity=severity,
                        description=description
                    ))

        return anomalies

    def forecast_costs(
        self,
        days: int = 30,
        entity_id: Optional[str] = None
    ) -> CostForecast:
        """
        Forecast future costs using simple trend analysis.

        Args:
            days: Number of days to forecast
            entity_id: Specific entity to forecast (None for all)

        Returns:
            CostForecast object with predictions
        """
        now = datetime.now(timezone.utc)

        # Get historical daily costs (last 30 days)
        lookback_days = min(days * 2, 60)  # At least 2x forecast period
        lookback_start = now - timedelta(days=lookback_days)

        daily_costs = defaultdict(float)

        for entry in self.cost_tracker.cost_history:
            if entry.timestamp < lookback_start:
                continue

            if entity_id and entry.user_id != entity_id:
                continue

            day_key = entry.timestamp.date()
            daily_costs[day_key] += entry.cost_usd

        if not daily_costs:
            # No data, return zero forecast
            return CostForecast(
                forecast_period=f"{days} days",
                forecast_start=now,
                forecast_end=now + timedelta(days=days),
                predicted_cost=0.0,
                confidence_interval_low=0.0,
                confidence_interval_high=0.0,
                trend="unknown",
                trend_percent=0.0,
                factors=["Insufficient historical data"]
            )

        # Calculate daily average
        costs = list(daily_costs.values())
        avg_daily_cost = mean(costs)

        # Simple linear trend (compare first half vs second half)
        mid_point = len(costs) // 2
        if mid_point > 0:
            first_half_avg = mean(costs[:mid_point])
            second_half_avg = mean(costs[mid_point:])
            trend_factor = second_half_avg / first_half_avg if first_half_avg > 0 else 1.0
        else:
            trend_factor = 1.0

        # Predict future cost with trend
        predicted_daily_cost = avg_daily_cost * trend_factor
        predicted_cost = predicted_daily_cost * days

        # Calculate confidence interval (95% CI)
        if len(costs) >= 2:
            std_daily_cost = stdev(costs)
            margin_of_error = 1.96 * std_daily_cost  # 95% CI
            confidence_interval_low = max(0, (predicted_daily_cost - margin_of_error) * days)
            confidence_interval_high = (predicted_daily_cost + margin_of_error) * days
        else:
            confidence_interval_low = predicted_cost * 0.8
            confidence_interval_high = predicted_cost * 1.2

        # Determine trend
        trend_percent = ((trend_factor - 1.0) * 100)
        if trend_percent > 5:
            trend = "increasing"
        elif trend_percent < -5:
            trend = "decreasing"
        else:
            trend = "stable"

        # Identify key factors
        factors = []
        if trend == "increasing":
            factors.append(f"Upward trend detected: +{trend_percent:.1f}%")
        elif trend == "decreasing":
            factors.append(f"Downward trend detected: {trend_percent:.1f}%")
        else:
            factors.append("Stable usage pattern")

        factors.append(f"Based on {len(costs)} days of historical data")
        factors.append(f"Average daily cost: ${avg_daily_cost:.2f}")

        return CostForecast(
            forecast_period=f"{days} days",
            forecast_start=now,
            forecast_end=now + timedelta(days=days),
            predicted_cost=predicted_cost,
            confidence_interval_low=confidence_interval_low,
            confidence_interval_high=confidence_interval_high,
            trend=trend,
            trend_percent=trend_percent,
            factors=factors
        )

    def get_cost_breakdown(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        group_by: str = "model"
    ) -> Dict[str, float]:
        """
        Get cost breakdown by various dimensions.

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            group_by: Dimension to group by (model/provider/user)

        Returns:
            Dictionary mapping group key to total cost
        """
        breakdown = defaultdict(float)

        summary = self.cost_tracker.get_summary(
            start_date=start_date,
            end_date=end_date
        )

        if group_by == "model":
            return summary.cost_by_model
        elif group_by == "provider":
            return summary.cost_by_provider
        elif group_by == "user":
            return summary.cost_by_user
        else:
            raise ValueError(f"Invalid group_by: {group_by}")

    def _classify_cost_tier(self, cost_per_1k_tokens: float) -> CostTier:
        """Classify model into cost tier."""
        if cost_per_1k_tokens < 0.0001:
            return CostTier.ULTRA_LOW
        elif cost_per_1k_tokens < 0.001:
            return CostTier.LOW
        elif cost_per_1k_tokens < 0.01:
            return CostTier.MEDIUM
        elif cost_per_1k_tokens < 0.1:
            return CostTier.HIGH
        else:
            return CostTier.PREMIUM

    def _calculate_efficiency_score(self, cost_per_1k_tokens: float) -> float:
        """
        Calculate efficiency score (0-100).

        Lower cost = higher score using logarithmic scale.
        """
        if cost_per_1k_tokens <= 0:
            return 100.0

        # Use log scale: score = 100 - (log10(cost) + 4) * 20
        # This maps:
        #   $0.0001/1K -> 100
        #   $0.001/1K -> 80
        #   $0.01/1K -> 60
        #   $0.1/1K -> 40
        #   $1.0/1K -> 20
        import math
        score = 100 - (math.log10(cost_per_1k_tokens) + 4) * 20
        return max(0, min(100, score))

    def _assess_quality_impact(
        self,
        current_tier: CostTier,
        alternative_tier: CostTier
    ) -> str:
        """Assess expected quality impact of switching tiers."""
        tier_order = [
            CostTier.ULTRA_LOW,
            CostTier.LOW,
            CostTier.MEDIUM,
            CostTier.HIGH,
            CostTier.PREMIUM
        ]

        current_idx = tier_order.index(current_tier)
        alternative_idx = tier_order.index(alternative_tier)

        diff = current_idx - alternative_idx

        if diff == 0:
            return "none"
        elif diff == 1:
            return "minimal"
        elif diff >= 2:
            return "moderate"
        else:
            # Alternative is higher tier (should not happen in recommendations)
            return "improvement"
