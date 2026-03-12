"""
Alert Manager - Handles alerts and notifications
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading

from .collector import Metric
from .patterns import Anomaly, Pattern


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Status of an alert."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """An alert notification."""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    source: str
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Rule for generating alerts."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "spike", "drop", "anomaly"
    threshold: float
    severity: AlertSeverity
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[datetime] = None


class AlertManager:
    """
    Manages alert generation, routing, and lifecycle.
    
    Features:
    - Rule-based alert generation
    - Alert deduplication
    - Cooldown periods
    - Notification callbacks
    """
    
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._rules: Dict[str, AlertRule] = {}
        self._callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self._alert_counter = 0
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self._rules[rule.name] = rule
    
    def remove_rule(self, name: str):
        """Remove an alert rule."""
        self._rules.pop(name, None)
    
    def register_callback(self, callback: Callable[[Alert], None]):
        """Register callback for new alerts."""
        self._callbacks.append(callback)
    
    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        metadata: Dict[str, Any] = None
    ) -> Alert:
        """Create and store a new alert."""
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter:06d}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        self._alerts[alert_id] = alert
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception:
                pass
        
        return alert
    
    def check_rules(self, metric: Metric) -> List[Alert]:
        """Check all rules against a metric and generate alerts."""
        alerts = []
        
        for rule in self._rules.values():
            if not rule.enabled or rule.metric_name != metric.name:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    continue
            
            # Check condition
            triggered = False
            
            if rule.condition == "gt" and metric.value > rule.threshold:
                triggered = True
            elif rule.condition == "lt" and metric.value < rule.threshold:
                triggered = True
            elif rule.condition == "eq" and abs(metric.value - rule.threshold) < 0.001:
                triggered = True
            
            if triggered:
                rule.last_triggered = datetime.now()
                alert = self.create_alert(
                    severity=rule.severity,
                    title=f"Rule triggered: {rule.name}",
                    message=f"Metric {metric.name}={metric.value:.2f} {rule.condition} {rule.threshold}",
                    source=rule.name,
                    metadata={"rule": rule.name, "metric": metric.name, "value": metric.value}
                )
                alerts.append(alert)
        
        return alerts

    def alert_from_anomaly(self, anomaly: Anomaly) -> Alert:
        """Create alert from detected anomaly."""
        severity_map = {
            (0.0, 0.3): AlertSeverity.INFO,
            (0.3, 0.6): AlertSeverity.WARNING,
            (0.6, 0.8): AlertSeverity.ERROR,
            (0.8, 1.1): AlertSeverity.CRITICAL,
        }
        
        severity = AlertSeverity.WARNING
        for (low, high), sev in severity_map.items():
            if low <= anomaly.severity < high:
                severity = sev
                break
        
        return self.create_alert(
            severity=severity,
            title=f"Anomaly: {anomaly.anomaly_type.value}",
            message=anomaly.description,
            source=anomaly.metric_name,
            metadata={
                "anomaly_type": anomaly.anomaly_type.value,
                "value": anomaly.value,
                "expected_range": anomaly.expected_range,
            }
        )
    
    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.ACKNOWLEDGED
            self._alerts[alert_id].acknowledged_at = datetime.now()
            return True
        return False
    
    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self._alerts:
            self._alerts[alert_id].status = AlertStatus.RESOLVED
            self._alerts[alert_id].resolved_at = datetime.now()
            return True
        return False
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by severity."""
        alerts = [a for a in self._alerts.values() if a.status == AlertStatus.ACTIVE]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert."""
        return self._alerts.get(alert_id)
    
    def get_alert_counts(self) -> Dict[str, int]:
        """Get counts of alerts by severity and status."""
        counts = {
            "total": len(self._alerts),
            "active": 0,
            "acknowledged": 0,
            "resolved": 0,
            "info": 0,
            "warning": 0,
            "error": 0,
            "critical": 0,
        }
        
        for alert in self._alerts.values():
            counts[alert.status.value] += 1
            counts[alert.severity.value] += 1
        
        return counts
    
    def cleanup(self, max_age_hours: int = 24):
        """Remove old resolved alerts."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            to_remove = [
                aid for aid, alert in self._alerts.items()
                if alert.status == AlertStatus.RESOLVED and alert.resolved_at < cutoff
            ]
            for aid in to_remove:
                del self._alerts[aid]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard."""
        return {
            "counts": self.get_alert_counts(),
            "active": [
                {
                    "id": a.id,
                    "severity": a.severity.value,
                    "title": a.title,
                    "message": a.message,
                    "source": a.source,
                    "created": a.created_at.isoformat(),
                }
                for a in self.get_active_alerts()[:20]
            ],
            "rules": [
                {
                    "name": r.name,
                    "metric": r.metric_name,
                    "enabled": r.enabled,
                }
                for r in self._rules.values()
            ],
        }
