"""
Metrics Collector - Gathers and stores system metrics
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import statistics
import threading


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"          # Cumulative count
    GAUGE = "gauge"              # Point-in-time value
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"              # Duration measurements


@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    name: str
    count: int
    total: float
    mean: float
    min_val: float
    max_val: float
    std_dev: float
    last_value: float
    last_updated: datetime
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates system metrics.
    
    Features:
    - Thread-safe metric recording
    - Automatic aggregation
    - Time-windowed statistics
    - Tag-based filtering
    """
    
    def __init__(self, retention_hours: int = 24):
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._retention = timedelta(hours=retention_hours)
        self._callbacks: List[Callable[[Metric], None]] = []
    
    def record(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        tags: Dict[str, str] = None
    ) -> Metric:
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            tags=tags or {}
        )
        
        with self._lock:
            self._metrics[name].append(metric)
            
            if metric_type == MetricType.COUNTER:
                self._counters[name] += value
            elif metric_type == MetricType.GAUGE:
                self._gauges[name] = value
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(metric)
            except Exception:
                pass
        
        return metric
    
    def increment(self, name: str, amount: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.record(name, amount, MetricType.COUNTER, tags)
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        self.record(name, value, MetricType.GAUGE, tags)
    
    def timer(self, name: str, duration_seconds: float, tags: Dict[str, str] = None):
        """Record a timing metric."""
        self.record(name, duration_seconds, MetricType.TIMER, tags)

    def get_summary(self, name: str, window_minutes: int = 60) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        with self._lock:
            if name not in self._metrics:
                return None
            
            cutoff = datetime.now() - timedelta(minutes=window_minutes)
            values = [
                m.value for m in self._metrics[name]
                if m.timestamp >= cutoff
            ]
            
            if not values:
                return None
            
            return MetricSummary(
                name=name,
                count=len(values),
                total=sum(values),
                mean=statistics.mean(values),
                min_val=min(values),
                max_val=max(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0,
                last_value=values[-1],
                last_updated=self._metrics[name][-1].timestamp
            )
    
    def get_metrics(
        self,
        name: str = None,
        window_minutes: int = 60,
        tags: Dict[str, str] = None
    ) -> List[Metric]:
        """Get metrics matching criteria."""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            if name:
                metrics = self._metrics.get(name, [])
            else:
                metrics = [m for mlist in self._metrics.values() for m in mlist]
            
            # Filter by time
            metrics = [m for m in metrics if m.timestamp >= cutoff]
            
            # Filter by tags
            if tags:
                metrics = [
                    m for m in metrics
                    if all(m.tags.get(k) == v for k, v in tags.items())
                ]
            
            return metrics
    
    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self._counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get current gauge value."""
        return self._gauges.get(name)
    
    def get_all_names(self) -> List[str]:
        """Get all metric names."""
        with self._lock:
            return list(self._metrics.keys())
    
    def register_callback(self, callback: Callable[[Metric], None]):
        """Register callback for new metrics."""
        self._callbacks.append(callback)
    
    def cleanup(self):
        """Remove old metrics beyond retention period."""
        cutoff = datetime.now() - self._retention
        
        with self._lock:
            for name in self._metrics:
                self._metrics[name] = [
                    m for m in self._metrics[name]
                    if m.timestamp >= cutoff
                ]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard display."""
        summaries = {}
        for name in self.get_all_names():
            summary = self.get_summary(name)
            if summary:
                summaries[name] = {
                    "count": summary.count,
                    "mean": round(summary.mean, 3),
                    "min": round(summary.min_val, 3),
                    "max": round(summary.max_val, 3),
                    "last": round(summary.last_value, 3),
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "summaries": summaries,
        }


# Singleton instance
_collector: Optional[MetricsCollector] = None

def get_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector
