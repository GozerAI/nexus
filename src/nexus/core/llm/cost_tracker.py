"""
Inference cost tracking per request, model, and tenant.

Tracks token usage and computes costs in real-time for all LLM
inference requests. Supports per-model pricing, budget alerts,
and cost attribution to tenants/projects.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """A single inference cost record."""
    request_id: str
    model_name: str
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    tenant_id: Optional[str] = None
    project_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPricing:
    """Pricing configuration for a model."""
    model_name: str
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    cost_per_request: float = 0.0  # Fixed per-request cost (e.g., image gen)

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1000) * self.cost_per_1k_input
            + (output_tokens / 1000) * self.cost_per_1k_output
            + self.cost_per_request
        )


@dataclass
class BudgetAlert:
    """Budget alert configuration."""
    name: str
    threshold_usd: float
    tenant_id: Optional[str] = None  # None = global
    period: str = "monthly"  # "daily", "monthly", "total"
    callback: Optional[Callable[[str, float, float], None]] = None
    triggered: bool = False


class InferenceCostTracker:
    """
    Tracks inference costs across all models and tenants.

    Features:
    - Per-model pricing configuration
    - Real-time cost accumulation
    - Per-tenant/project cost attribution
    - Budget alerts with callbacks
    - Cost aggregation by model, provider, tenant, time period
    - Rolling window statistics
    """

    def __init__(
        self,
        max_records: int = 100000,
        alert_check_interval: int = 100,  # Check alerts every N records
    ):
        self._pricing: Dict[str, ModelPricing] = {}
        self._records: List[CostRecord] = []
        self._max_records = max_records
        self._alert_check_interval = alert_check_interval
        self._alerts: List[BudgetAlert] = []
        self._lock = threading.Lock()
        self._record_count = 0

        # Running totals for fast aggregation
        self._totals = {
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_requests": 0,
        }
        self._model_totals: Dict[str, Dict[str, float]] = {}
        self._tenant_totals: Dict[str, float] = {}

    def set_pricing(self, pricing: ModelPricing) -> None:
        """Set pricing for a model."""
        self._pricing[pricing.model_name] = pricing

    def set_pricing_bulk(self, pricing_list: List[ModelPricing]) -> None:
        """Set pricing for multiple models."""
        for p in pricing_list:
            self._pricing[p.model_name] = p

    def add_alert(self, alert: BudgetAlert) -> None:
        """Add a budget alert."""
        self._alerts.append(alert)

    def record(
        self,
        request_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0,
        provider: str = "",
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CostRecord:
        """
        Record an inference request and compute its cost.

        Args:
            request_id: Unique request identifier
            model_name: Model used
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Request latency
            provider: Provider name
            tenant_id: Tenant for cost attribution
            project_id: Project for cost attribution
            metadata: Additional metadata

        Returns:
            CostRecord with computed cost
        """
        pricing = self._pricing.get(model_name)
        cost = pricing.compute_cost(input_tokens, output_tokens) if pricing else 0.0

        record = CostRecord(
            request_id=request_id,
            model_name=model_name,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            tenant_id=tenant_id,
            project_id=project_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                self._records.pop(0)

            # Update running totals
            self._totals["total_cost_usd"] += cost
            self._totals["total_input_tokens"] += input_tokens
            self._totals["total_output_tokens"] += output_tokens
            self._totals["total_requests"] += 1

            # Per-model totals
            if model_name not in self._model_totals:
                self._model_totals[model_name] = {
                    "cost_usd": 0.0, "requests": 0, "tokens": 0
                }
            mt = self._model_totals[model_name]
            mt["cost_usd"] += cost
            mt["requests"] += 1
            mt["tokens"] += input_tokens + output_tokens

            # Per-tenant totals
            if tenant_id:
                self._tenant_totals[tenant_id] = (
                    self._tenant_totals.get(tenant_id, 0.0) + cost
                )

            self._record_count += 1

        # Periodic alert check
        if self._record_count % self._alert_check_interval == 0:
            self._check_alerts()

        return record

    def _check_alerts(self) -> None:
        """Check all budget alerts."""
        for alert in self._alerts:
            if alert.triggered:
                continue

            if alert.tenant_id:
                current = self._tenant_totals.get(alert.tenant_id, 0.0)
            else:
                current = self._totals["total_cost_usd"]

            if current >= alert.threshold_usd:
                alert.triggered = True
                if alert.callback:
                    try:
                        alert.callback(alert.name, current, alert.threshold_usd)
                    except Exception as e:
                        logger.error("Alert callback failed: %s", e)
                logger.warning(
                    "Budget alert '%s' triggered: $%.4f >= $%.4f",
                    alert.name, current, alert.threshold_usd,
                )

    def get_total_cost(self, tenant_id: Optional[str] = None) -> float:
        """Get total cost, optionally filtered by tenant."""
        if tenant_id:
            return self._tenant_totals.get(tenant_id, 0.0)
        return self._totals["total_cost_usd"]

    def get_cost_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get cost breakdown by model."""
        return dict(self._model_totals)

    def get_cost_by_tenant(self) -> Dict[str, float]:
        """Get cost breakdown by tenant."""
        return dict(self._tenant_totals)

    def get_recent_records(self, limit: int = 100) -> List[CostRecord]:
        """Get most recent cost records."""
        with self._lock:
            return list(self._records[-limit:])

    def get_cost_summary(
        self,
        since_timestamp: Optional[float] = None,
        model_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a cost summary with optional filters.

        Args:
            since_timestamp: Only include records after this time
            model_name: Filter by model
            tenant_id: Filter by tenant

        Returns:
            Summary dict with totals and breakdowns
        """
        with self._lock:
            filtered = self._records
            if since_timestamp:
                filtered = [r for r in filtered if r.timestamp >= since_timestamp]
            if model_name:
                filtered = [r for r in filtered if r.model_name == model_name]
            if tenant_id:
                filtered = [r for r in filtered if r.tenant_id == tenant_id]

        if not filtered:
            return {"total_cost_usd": 0.0, "total_requests": 0, "records": 0}

        total_cost = sum(r.cost_usd for r in filtered)
        total_input = sum(r.input_tokens for r in filtered)
        total_output = sum(r.output_tokens for r in filtered)

        return {
            "total_cost_usd": total_cost,
            "total_requests": len(filtered),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "avg_cost_per_request": total_cost / len(filtered),
            "avg_tokens_per_request": (total_input + total_output) / len(filtered),
            "records": len(filtered),
        }

    def get_stats(self) -> Dict[str, Any]:
        total = self._totals["total_requests"]
        return {
            **self._totals,
            "models_tracked": len(self._model_totals),
            "tenants_tracked": len(self._tenant_totals),
            "alerts_configured": len(self._alerts),
            "alerts_triggered": sum(1 for a in self._alerts if a.triggered),
            "avg_cost_per_request": (
                self._totals["total_cost_usd"] / total if total > 0 else 0.0
            ),
        }
