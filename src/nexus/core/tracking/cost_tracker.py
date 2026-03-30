"""
Cost tracking for API usage.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CostEntry:
    """Single cost entry."""
    timestamp: datetime
    model_name: str
    provider: str
    tokens_used: int
    cost_usd: float
    user_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class CostSummary:
    """Cost summary statistics."""
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
    Tracks API costs and usage.
    
    Features:
    - Per-model cost tracking
    - Per-user cost tracking  
    - Budget alerts
    - Historical data
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
        self.entries: List[CostEntry] = []
        self.alerts_sent: set = set()
        logger.info(f"CostTracker initialized (budget=${budget_limit_usd}, threshold={alert_threshold*100}%)")
    
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
        
        logger.info(
            f"Cost recorded: {model_name} - ${cost_usd:.4f} "
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
            CostSummary object
        """
        # Default to current month
        if start_date is None:
            now = datetime.now(timezone.utc)
            start_date = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
        
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Normalize to tz-aware for comparison with tz-aware entries
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

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
        """Get total cost for current month."""
        summary = self.get_summary()
        return summary.total_cost
    
    def get_daily_cost(self) -> float:
        """Get total cost for today."""
        now = datetime.now(timezone.utc)
        start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        summary = self.get_summary(start_date=start_of_day)
        return summary.total_cost
    
    def is_over_budget(self) -> bool:
        """Check if over monthly budget."""
        return self.get_monthly_cost() >= self.budget_limit
    
    def get_budget_status(self) -> dict:
        """
        Get current budget status.
        
        Returns:
            Dictionary with budget information
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
        Get top cost entries.
        
        Args:
            limit: Number of entries to return
            
        Returns:
            List of (model_name, cost) tuples
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
            format: Export format (csv or json)
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
        
        logger.info(f"Exported {len(self.entries)} cost entries to {filepath}")
    
    def _check_budget_alert(self):
        """Check if budget alert should be triggered."""
        monthly_cost = self.get_monthly_cost()
        alert_level = self.budget_limit * self.alert_threshold
        
        if monthly_cost >= alert_level:
            alert_key = f"{datetime.now(timezone.utc).strftime('%Y-%m')}-alert"
            
            if alert_key not in self.alerts_sent:
                self.alerts_sent.add(alert_key)
                logger.warning(
                    f"BUDGET ALERT: ${monthly_cost:.2f} spent "
                    f"(${self.budget_limit:.2f} budget, "
                    f"{self.alert_threshold*100}% threshold)"
                )
