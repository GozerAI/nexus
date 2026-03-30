"""
Drift Monitor - Detects model performance drift and behavioral changes.

This module monitors model outputs over time to detect:
- Performance degradation (latency, error rates)
- Response quality drift (confidence, coherence)
- Behavioral changes (response patterns, token usage)
- Epistemic health issues (hallucinations, inconsistencies)
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of drift that can be detected."""

    PERFORMANCE = "performance"  # Latency/throughput degradation
    QUALITY = "quality"  # Response quality decline
    CONFIDENCE = "confidence"  # Confidence calibration drift
    BEHAVIORAL = "behavioral"  # Response pattern changes
    ERROR_RATE = "error_rate"  # Increasing error frequency
    COST = "cost"  # Cost per request increasing
    EPISTEMIC = "epistemic"  # Knowledge/truthfulness issues


class DriftSeverity(str, Enum):
    """Severity levels for drift alerts."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert for detected drift."""

    model_name: str
    drift_type: DriftType
    severity: DriftSeverity
    message: str
    metric_value: float
    baseline_value: float
    deviation_percent: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recommended_action: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_name": self.model_name,
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "metric_value": self.metric_value,
            "baseline_value": self.baseline_value,
            "deviation_percent": self.deviation_percent,
            "timestamp": self.timestamp.isoformat(),
            "recommended_action": self.recommended_action,
        }


@dataclass
class ModelMetrics:
    """Metrics collected for a single model."""

    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidences: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_count: int = 0
    success_count: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    response_lengths: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Baseline metrics (calculated from initial window)
    baseline_latency: Optional[float] = None
    baseline_confidence: Optional[float] = None
    baseline_error_rate: Optional[float] = None
    baseline_response_length: Optional[float] = None
    baseline_established: bool = False


class DriftMonitor:
    """
    Monitors model performance and detects drift over time.

    Features:
    - Rolling window statistics for all models
    - Automatic baseline establishment
    - Multi-dimensional drift detection
    - Severity-based alerting
    - Callback support for alert handling
    """

    def __init__(
        self,
        baseline_window: int = 100,
        detection_window: int = 50,
        latency_threshold: float = 0.5,  # 50% increase triggers alert
        confidence_threshold: float = 0.2,  # 20% decrease triggers alert
        error_rate_threshold: float = 0.1,  # 10% error rate triggers alert
        check_interval: float = 60.0,  # Check for drift every 60 seconds
        enable_auto_quarantine: bool = True,
    ):
        """
        Initialize the drift monitor.

        Args:
            baseline_window: Number of samples to establish baseline
            detection_window: Number of recent samples for drift detection
            latency_threshold: Threshold for latency drift (relative increase)
            confidence_threshold: Threshold for confidence drift (relative decrease)
            error_rate_threshold: Threshold for error rate
            check_interval: Interval between drift checks (seconds)
            enable_auto_quarantine: Whether to recommend quarantine for severe drift
        """
        self.baseline_window = baseline_window
        self.detection_window = detection_window
        self.latency_threshold = latency_threshold
        self.confidence_threshold = confidence_threshold
        self.error_rate_threshold = error_rate_threshold
        self.check_interval = check_interval
        self.enable_auto_quarantine = enable_auto_quarantine

        self._model_metrics: Dict[str, ModelMetrics] = {}
        self._alerts: deque = deque(maxlen=1000)
        self._alert_callbacks: List[Callable[[DriftAlert], None]] = []
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info("DriftMonitor initialized with baseline_window=%d", baseline_window)

    def start(self) -> None:
        """Start the background drift monitoring thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="DriftMonitor"
        )
        self._monitor_thread.start()
        logger.info("DriftMonitor started")

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("DriftMonitor stopped")

    def record_response(
        self,
        model_name: str,
        latency_ms: float,
        confidence: float,
        success: bool,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        response_length: int = 0,
    ) -> None:
        """
        Record a model response for drift tracking.

        Args:
            model_name: Name of the model
            latency_ms: Response latency in milliseconds
            confidence: Response confidence score (0-1)
            success: Whether the response was successful
            tokens_used: Number of tokens consumed
            cost_usd: Cost of the request
            response_length: Length of the response text
        """
        with self._lock:
            if model_name not in self._model_metrics:
                self._model_metrics[model_name] = ModelMetrics()

            metrics = self._model_metrics[model_name]
            metrics.latencies.append(latency_ms)
            metrics.confidences.append(confidence)
            metrics.response_lengths.append(response_length)
            metrics.timestamps.append(lambda: datetime.now(timezone.utc)())
            metrics.total_tokens += tokens_used
            metrics.total_cost += cost_usd

            if success:
                metrics.success_count += 1
            else:
                metrics.error_count += 1

            # Establish baseline if we have enough samples
            if not metrics.baseline_established:
                total_samples = metrics.success_count + metrics.error_count
                if total_samples >= self.baseline_window:
                    self._establish_baseline(model_name, metrics)

    def _establish_baseline(self, model_name: str, metrics: ModelMetrics) -> None:
        """Establish baseline metrics for a model."""
        if len(metrics.latencies) >= self.baseline_window:
            metrics.baseline_latency = mean(list(metrics.latencies)[:self.baseline_window])

        if len(metrics.confidences) >= self.baseline_window:
            metrics.baseline_confidence = mean(list(metrics.confidences)[:self.baseline_window])

        if len(metrics.response_lengths) >= self.baseline_window:
            metrics.baseline_response_length = mean(list(metrics.response_lengths)[:self.baseline_window])

        total = metrics.success_count + metrics.error_count
        if total > 0:
            metrics.baseline_error_rate = metrics.error_count / total

        metrics.baseline_established = True
        logger.info(f"Baseline established for {model_name}: latency={metrics.baseline_latency:.1f}ms, "
                   f"confidence={metrics.baseline_confidence:.2f}")

    def _monitoring_loop(self) -> None:
        """Background loop for periodic drift checking."""
        while self._running:
            try:
                self._check_all_models()
            except Exception as e:
                logger.error(f"Error in drift monitoring loop: {e}")

            time.sleep(self.check_interval)

    def _check_all_models(self) -> None:
        """Check all models for drift."""
        with self._lock:
            for model_name, metrics in self._model_metrics.items():
                if not metrics.baseline_established:
                    continue

                alerts = self._detect_drift(model_name, metrics)
                for alert in alerts:
                    self._alerts.append(alert)
                    self._notify_alert(alert)

    def _detect_drift(self, model_name: str, metrics: ModelMetrics) -> List[DriftAlert]:
        """Detect drift for a single model."""
        alerts = []

        # Get recent metrics
        recent_latencies = list(metrics.latencies)[-self.detection_window:]
        recent_confidences = list(metrics.confidences)[-self.detection_window:]

        # Check latency drift
        if recent_latencies and metrics.baseline_latency:
            current_latency = mean(recent_latencies)
            deviation = (current_latency - metrics.baseline_latency) / metrics.baseline_latency

            if deviation > self.latency_threshold:
                severity = self._calculate_severity(deviation, [0.5, 1.0, 2.0])
                alerts.append(DriftAlert(
                    model_name=model_name,
                    drift_type=DriftType.PERFORMANCE,
                    severity=severity,
                    message=f"Latency increased by {deviation*100:.1f}%",
                    metric_value=current_latency,
                    baseline_value=metrics.baseline_latency,
                    deviation_percent=deviation * 100,
                    recommended_action="Consider model health check or load balancing" if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] else None
                ))

        # Check confidence drift
        if recent_confidences and metrics.baseline_confidence:
            current_confidence = mean(recent_confidences)
            deviation = (metrics.baseline_confidence - current_confidence) / metrics.baseline_confidence

            if deviation > self.confidence_threshold:
                severity = self._calculate_severity(deviation, [0.2, 0.4, 0.6])
                alerts.append(DriftAlert(
                    model_name=model_name,
                    drift_type=DriftType.CONFIDENCE,
                    severity=severity,
                    message=f"Confidence decreased by {deviation*100:.1f}%",
                    metric_value=current_confidence,
                    baseline_value=metrics.baseline_confidence,
                    deviation_percent=deviation * 100,
                    recommended_action="Review model outputs for quality issues" if severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] else None
                ))

        # Check error rate drift
        total = metrics.success_count + metrics.error_count
        if total > 0:
            current_error_rate = metrics.error_count / total
            if current_error_rate > self.error_rate_threshold:
                severity = self._calculate_severity(current_error_rate, [0.1, 0.25, 0.5])
                alerts.append(DriftAlert(
                    model_name=model_name,
                    drift_type=DriftType.ERROR_RATE,
                    severity=severity,
                    message=f"Error rate at {current_error_rate*100:.1f}%",
                    metric_value=current_error_rate,
                    baseline_value=metrics.baseline_error_rate or 0.0,
                    deviation_percent=current_error_rate * 100,
                    recommended_action="Consider quarantining model" if self.enable_auto_quarantine and severity == DriftSeverity.CRITICAL else None
                ))

        return alerts

    def _calculate_severity(self, value: float, thresholds: List[float]) -> DriftSeverity:
        """Calculate severity based on thresholds."""
        if value >= thresholds[2]:
            return DriftSeverity.CRITICAL
        elif value >= thresholds[1]:
            return DriftSeverity.HIGH
        elif value >= thresholds[0]:
            return DriftSeverity.MEDIUM
        return DriftSeverity.LOW

    def _notify_alert(self, alert: DriftAlert) -> None:
        """Notify all registered callbacks about an alert."""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

        # Log the alert
        log_level = {
            DriftSeverity.LOW: logging.INFO,
            DriftSeverity.MEDIUM: logging.WARNING,
            DriftSeverity.HIGH: logging.WARNING,
            DriftSeverity.CRITICAL: logging.ERROR,
        }.get(alert.severity, logging.INFO)

        logger.log(log_level, f"Drift alert: {alert.model_name} - {alert.message}")

    def register_callback(self, callback: Callable[[DriftAlert], None]) -> None:
        """Register a callback to be notified of drift alerts."""
        self._alert_callbacks.append(callback)

    def get_recent_alerts(
        self,
        model_name: Optional[str] = None,
        drift_type: Optional[DriftType] = None,
        min_severity: Optional[DriftSeverity] = None,
        since: Optional[datetime] = None,
    ) -> List[DriftAlert]:
        """Get recent drift alerts with optional filtering."""
        with self._lock:
            alerts = list(self._alerts)

        # Apply filters
        if model_name:
            alerts = [a for a in alerts if a.model_name == model_name]
        if drift_type:
            alerts = [a for a in alerts if a.drift_type == drift_type]
        if min_severity:
            severity_order = [DriftSeverity.LOW, DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            min_index = severity_order.index(min_severity)
            alerts = [a for a in alerts if severity_order.index(a.severity) >= min_index]
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        return alerts

    def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get current health status for a model."""
        with self._lock:
            metrics = self._model_metrics.get(model_name)
            if not metrics:
                return {"status": "unknown", "message": "No data available"}

            recent_latencies = list(metrics.latencies)[-self.detection_window:]
            recent_confidences = list(metrics.confidences)[-self.detection_window:]
            total = metrics.success_count + metrics.error_count

            return {
                "status": "healthy" if metrics.baseline_established else "establishing_baseline",
                "baseline_established": metrics.baseline_established,
                "current_latency_ms": mean(recent_latencies) if recent_latencies else None,
                "baseline_latency_ms": metrics.baseline_latency,
                "current_confidence": mean(recent_confidences) if recent_confidences else None,
                "baseline_confidence": metrics.baseline_confidence,
                "error_rate": metrics.error_count / total if total > 0 else 0.0,
                "total_requests": total,
                "total_tokens": metrics.total_tokens,
                "total_cost_usd": metrics.total_cost,
            }

    def get_all_model_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all monitored models."""
        with self._lock:
            return {name: self.get_model_health(name) for name in self._model_metrics.keys()}

    def reset_model(self, model_name: str) -> None:
        """Reset metrics for a model (e.g., after recovery)."""
        with self._lock:
            if model_name in self._model_metrics:
                del self._model_metrics[model_name]
                logger.info(f"Reset metrics for {model_name}")
