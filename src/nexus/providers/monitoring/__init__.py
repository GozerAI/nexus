"""
Monitoring subsystem for the Nexus ensemble.

Provides drift detection, performance monitoring, and epistemic health tracking.
"""

from .drift_monitor import DriftMonitor, DriftAlert, DriftType
from .feedback_tracker import FeedbackTracker, FeedbackRecord

# Singleton metrics collector instance
_metrics_instance = None


def get_metrics():
    """Get or create the global MetricsCollector instance.

    Returns a MetricsCollector for Prometheus metrics, or None if
    prometheus_client is not available.
    """
    global _metrics_instance
    if _metrics_instance is None:
        try:
            from nexus.core.monitoring.metrics import MetricsCollector
            _metrics_instance = MetricsCollector()
        except ImportError:
            # prometheus_client not installed, return None
            return None
    return _metrics_instance


__all__ = [
    "DriftMonitor",
    "DriftAlert",
    "DriftType",
    "FeedbackTracker",
    "FeedbackRecord",
    "get_metrics",
]
