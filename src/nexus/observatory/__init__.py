"""
Observatory Module - Real-time monitoring and pattern detection

Provides visibility into:
- Pipeline execution metrics
- Expert performance tracking
- Cost accumulation
- Error rates and patterns
"""

from .collector import MetricsCollector, Metric, MetricType
from .patterns import PatternDetector, Pattern, Anomaly
from .alerts import AlertManager, Alert, AlertSeverity

__all__ = [
    "MetricsCollector",
    "Metric",
    "MetricType",
    "PatternDetector",
    "Pattern",
    "Anomaly",
    "AlertManager",
    "Alert",
    "AlertSeverity",
]
