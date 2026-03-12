"""
Pattern Detector - Identifies trends, anomalies, and correlations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import statistics

from .collector import MetricsCollector, Metric, MetricSummary


class PatternType(Enum):
    """Types of patterns detected."""
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    SPIKE = "spike"
    DROP = "drop"
    PLATEAU = "plateau"
    CYCLIC = "cyclic"
    CORRELATION = "correlation"


class AnomalyType(Enum):
    """Types of anomalies."""
    OUTLIER = "outlier"
    SUDDEN_CHANGE = "sudden_change"
    MISSING_DATA = "missing_data"
    THRESHOLD_BREACH = "threshold_breach"
    UNUSUAL_PATTERN = "unusual_pattern"


@dataclass
class Pattern:
    """A detected pattern in metrics."""
    pattern_type: PatternType
    metric_name: str
    confidence: float
    description: str
    start_time: datetime
    end_time: datetime
    data_points: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """A detected anomaly."""
    anomaly_type: AnomalyType
    metric_name: str
    severity: float  # 0-1
    description: str
    detected_at: datetime
    value: float
    expected_range: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class PatternDetector:
    """
    Detects patterns and anomalies in metrics.
    
    Capabilities:
    - Trend detection (up/down/plateau)
    - Spike/drop detection
    - Anomaly identification
    - Cross-metric correlation
    """
    
    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self._thresholds: Dict[str, Tuple[float, float]] = {}  # metric -> (min, max)
        self._baselines: Dict[str, MetricSummary] = {}
    
    def set_threshold(self, metric_name: str, min_val: float, max_val: float):
        """Set threshold for anomaly detection."""
        self._thresholds[metric_name] = (min_val, max_val)
    
    def update_baseline(self, metric_name: str, window_minutes: int = 60):
        """Update baseline for a metric."""
        summary = self.collector.get_summary(metric_name, window_minutes)
        if summary:
            self._baselines[metric_name] = summary
    
    def detect_trend(self, metric_name: str, window_minutes: int = 30) -> Optional[Pattern]:
        """Detect trend in metric values."""
        metrics = self.collector.get_metrics(metric_name, window_minutes)
        
        if len(metrics) < 5:
            return None
        
        values = [m.value for m in metrics]
        
        # Simple linear regression slope
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return None
        
        slope = numerator / denominator
        
        # Normalize slope by mean
        normalized_slope = slope / y_mean if y_mean != 0 else 0
        
        # Determine pattern type
        if normalized_slope > 0.1:
            pattern_type = PatternType.TREND_UP
            confidence = min(abs(normalized_slope), 1.0)
        elif normalized_slope < -0.1:
            pattern_type = PatternType.TREND_DOWN
            confidence = min(abs(normalized_slope), 1.0)
        else:
            pattern_type = PatternType.PLATEAU
            confidence = 1.0 - abs(normalized_slope)
        
        return Pattern(
            pattern_type=pattern_type,
            metric_name=metric_name,
            confidence=confidence,
            description=f"{pattern_type.value}: slope={normalized_slope:.3f}",
            start_time=metrics[0].timestamp,
            end_time=metrics[-1].timestamp,
            data_points=len(metrics),
            metadata={"slope": slope, "normalized_slope": normalized_slope}
        )

    def detect_anomalies(self, metric_name: str, window_minutes: int = 60) -> List[Anomaly]:
        """Detect anomalies in recent metrics."""
        anomalies = []
        metrics = self.collector.get_metrics(metric_name, window_minutes)
        
        if len(metrics) < 3:
            return anomalies
        
        values = [m.value for m in metrics]
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        
        # Check for outliers (> 2 std from mean)
        for metric in metrics:
            if std > 0:
                z_score = abs(metric.value - mean) / std
                if z_score > 2:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.OUTLIER,
                        metric_name=metric_name,
                        severity=min(z_score / 4, 1.0),
                        description=f"Value {metric.value:.2f} is {z_score:.1f} std from mean",
                        detected_at=metric.timestamp,
                        value=metric.value,
                        expected_range=(mean - 2*std, mean + 2*std)
                    ))
        
        # Check threshold breaches
        if metric_name in self._thresholds:
            min_t, max_t = self._thresholds[metric_name]
            for metric in metrics:
                if metric.value < min_t or metric.value > max_t:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.THRESHOLD_BREACH,
                        metric_name=metric_name,
                        severity=0.8,
                        description=f"Value {metric.value:.2f} outside threshold [{min_t}, {max_t}]",
                        detected_at=metric.timestamp,
                        value=metric.value,
                        expected_range=(min_t, max_t)
                    ))
        
        return anomalies
    
    def detect_spike(self, metric_name: str, threshold_multiplier: float = 2.0) -> Optional[Anomaly]:
        """Detect sudden spike in latest value."""
        summary = self.collector.get_summary(metric_name, 60)
        if not summary or summary.count < 5:
            return None
        
        # Check if latest value is a spike
        if summary.last_value > summary.mean * threshold_multiplier:
            return Anomaly(
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                metric_name=metric_name,
                severity=min((summary.last_value / summary.mean) / threshold_multiplier, 1.0),
                description=f"Spike: {summary.last_value:.2f} vs mean {summary.mean:.2f}",
                detected_at=summary.last_updated,
                value=summary.last_value,
                expected_range=(summary.mean - summary.std_dev, summary.mean + summary.std_dev)
            )
        
        return None
    
    def detect_drop(self, metric_name: str, threshold_multiplier: float = 0.5) -> Optional[Anomaly]:
        """Detect sudden drop in latest value."""
        summary = self.collector.get_summary(metric_name, 60)
        if not summary or summary.count < 5:
            return None
        
        # Check if latest value is a drop
        if summary.last_value < summary.mean * threshold_multiplier:
            return Anomaly(
                anomaly_type=AnomalyType.SUDDEN_CHANGE,
                metric_name=metric_name,
                severity=min(summary.mean / summary.last_value if summary.last_value > 0 else 1.0, 1.0),
                description=f"Drop: {summary.last_value:.2f} vs mean {summary.mean:.2f}",
                detected_at=summary.last_updated,
                value=summary.last_value,
                expected_range=(summary.mean * threshold_multiplier, summary.mean)
            )
        
        return None
    
    def find_correlations(
        self,
        metric_names: List[str],
        window_minutes: int = 60
    ) -> List[Pattern]:
        """Find correlations between metrics."""
        correlations = []
        
        # Get aligned time series
        series = {}
        for name in metric_names:
            metrics = self.collector.get_metrics(name, window_minutes)
            if len(metrics) >= 5:
                series[name] = [m.value for m in metrics]
        
        # Pairwise correlation
        names = list(series.keys())
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                values1 = series[name1]
                values2 = series[name2]
                
                # Align lengths
                min_len = min(len(values1), len(values2))
                v1 = values1[:min_len]
                v2 = values2[:min_len]
                
                # Calculate correlation
                corr = self._pearson_correlation(v1, v2)
                
                if abs(corr) > 0.7:
                    correlations.append(Pattern(
                        pattern_type=PatternType.CORRELATION,
                        metric_name=f"{name1}:{name2}",
                        confidence=abs(corr),
                        description=f"{'Positive' if corr > 0 else 'Negative'} correlation: {corr:.2f}",
                        start_time=datetime.now() - timedelta(minutes=window_minutes),
                        end_time=datetime.now(),
                        data_points=min_len,
                        metadata={"correlation": corr, "metrics": [name1, name2]}
                    ))
        
        return correlations
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        n = len(x)
        if n < 3:
            return 0.0
        
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
        sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)
        
        denominator = (sum_sq_x * sum_sq_y) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def get_health_score(self, metric_names: List[str] = None) -> float:
        """Calculate overall system health score (0-1)."""
        if metric_names is None:
            metric_names = self.collector.get_all_names()
        
        if not metric_names:
            return 1.0
        
        penalties = 0.0
        checks = 0
        
        for name in metric_names:
            # Check for anomalies
            anomalies = self.detect_anomalies(name, 30)
            if anomalies:
                penalties += sum(a.severity for a in anomalies) / len(anomalies)
                checks += 1
            
            # Check trend
            trend = self.detect_trend(name, 30)
            if trend and trend.pattern_type == PatternType.TREND_DOWN:
                penalties += trend.confidence * 0.5
                checks += 1
        
        if checks == 0:
            return 1.0
        
        return max(0.0, 1.0 - (penalties / checks))
