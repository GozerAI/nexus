"""
Usage analytics and tracking.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


@dataclass
class UsageEntry:
    """Single usage entry."""
    timestamp: datetime
    user_id: Optional[str]
    endpoint: str
    model_name: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    success: bool = True
    error_type: Optional[str] = None


@dataclass
class UsageStats:
    """Usage statistics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    avg_latency_ms: float = 0.0
    cache_hit_rate: float = 0.0
    requests_by_endpoint: Dict[str, int] = field(default_factory=dict)
    requests_by_model: Dict[str, int] = field(default_factory=dict)
    requests_by_user: Dict[str, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class UsageTracker:
    """
    Tracks detailed usage analytics.
    
    Features:
    - Request tracking by endpoint, model, user
    - Performance metrics (latency, success rate)
    - Cache effectiveness
    - Error analytics
    - Time-series data
    - Top users/endpoints/models
    """
    
    def __init__(self):
        """Initialize usage tracker."""
        self.entries: List[UsageEntry] = []
        self.request_counter = 0
        logger.info("UsageTracker initialized")
    
    def record_request(
        self,
        endpoint: str,
        user_id: Optional[str] = None,
        model_name: Optional[str] = None,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
        cached: bool = False,
        success: bool = True,
        error_type: Optional[str] = None
    ):
        """
        Record a request.
        
        Args:
            endpoint: API endpoint
            user_id: Optional user ID
            model_name: Optional model name
            tokens_used: Number of tokens
            latency_ms: Request latency
            cached: Whether response was cached
            success: Whether request succeeded
            error_type: Type of error if failed
        """
        entry = UsageEntry(
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            endpoint=endpoint,
            model_name=model_name,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            cached=cached,
            success=success,
            error_type=error_type
        )
        
        self.entries.append(entry)
        self.request_counter += 1
        
        if self.request_counter % 100 == 0:
            logger.info(f"Usage tracker: {self.request_counter} requests recorded")
    
    def get_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None
    ) -> UsageStats:
        """
        Get usage statistics for a period.
        
        Args:
            start_date: Start of period
            end_date: End of period
            user_id: Optional user filter
            endpoint: Optional endpoint filter
            
        Returns:
            UsageStats object
        """
        # Default to last 24 hours
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(hours=24)
        
        # Filter entries
        filtered = [
            e for e in self.entries
            if start_date <= e.timestamp <= end_date
            and (user_id is None or e.user_id == user_id)
            and (endpoint is None or e.endpoint == endpoint)
        ]
        
        if not filtered:
            return UsageStats(
                period_start=start_date,
                period_end=end_date
            )
        
        # Calculate statistics
        stats = UsageStats(
            period_start=start_date,
            period_end=end_date
        )
        
        stats.total_requests = len(filtered)
        stats.successful_requests = sum(1 for e in filtered if e.success)
        stats.failed_requests = sum(1 for e in filtered if not e.success)
        stats.total_tokens = sum(e.tokens_used for e in filtered)
        
        # Average latency
        latencies = [e.latency_ms for e in filtered if e.latency_ms > 0]
        stats.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0
        
        # Cache hit rate
        cached_count = sum(1 for e in filtered if e.cached)
        stats.cache_hit_rate = (cached_count / stats.total_requests * 100) if stats.total_requests > 0 else 0.0
        
        # Requests by endpoint
        stats.requests_by_endpoint = dict(Counter(e.endpoint for e in filtered))
        
        # Requests by model
        model_counts = Counter(e.model_name for e in filtered if e.model_name)
        stats.requests_by_model = dict(model_counts)
        
        # Requests by user
        user_counts = Counter(e.user_id for e in filtered if e.user_id)
        stats.requests_by_user = dict(user_counts)
        
        # Errors by type
        error_counts = Counter(e.error_type for e in filtered if e.error_type)
        stats.errors_by_type = dict(error_counts)
        
        return stats
    
    def get_hourly_stats(self, hours: int = 24) -> List[Dict]:
        """
        Get hourly statistics for last N hours.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            List of hourly stat dictionaries
        """
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        hourly_stats = []
        
        for i in range(hours):
            hour_start = start_time + timedelta(hours=i)
            hour_end = hour_start + timedelta(hours=1)
            
            hour_entries = [
                e for e in self.entries
                if hour_start <= e.timestamp < hour_end
            ]
            
            hourly_stats.append({
                'hour': hour_start.isoformat(),
                'requests': len(hour_entries),
                'successful': sum(1 for e in hour_entries if e.success),
                'failed': sum(1 for e in hour_entries if not e.success),
                'tokens': sum(e.tokens_used for e in hour_entries),
                'avg_latency_ms': (
                    sum(e.latency_ms for e in hour_entries) / len(hour_entries)
                    if hour_entries else 0.0
                )
            })
        
        return hourly_stats
    
    def get_top_users(self, limit: int = 10, days: int = 7) -> List[tuple]:
        """
        Get top users by request count.
        
        Args:
            limit: Number of users to return
            days: Number of days to analyze
            
        Returns:
            List of (user_id, request_count) tuples
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_entries = [e for e in self.entries if e.timestamp >= cutoff and e.user_id]
        
        user_counts = Counter(e.user_id for e in recent_entries)
        return user_counts.most_common(limit)
    
    def get_top_endpoints(self, limit: int = 10) -> List[tuple]:
        """
        Get most used endpoints.
        
        Args:
            limit: Number of endpoints to return
            
        Returns:
            List of (endpoint, count) tuples
        """
        endpoint_counts = Counter(e.endpoint for e in self.entries)
        return endpoint_counts.most_common(limit)
    
    def get_error_summary(self, days: int = 7) -> Dict:
        """
        Get error summary for recent period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with error statistics
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_entries = [e for e in self.entries if e.timestamp >= cutoff]
        
        failed = [e for e in recent_entries if not e.success]
        
        return {
            'total_requests': len(recent_entries),
            'failed_requests': len(failed),
            'error_rate_percent': (len(failed) / len(recent_entries) * 100) if recent_entries else 0.0,
            'errors_by_type': dict(Counter(e.error_type for e in failed if e.error_type)),
            'errors_by_endpoint': dict(Counter(e.endpoint for e in failed)),
        }
    
    def export_data(self, filepath: str, start_date: Optional[datetime] = None):
        """
        Export usage data to CSV.
        
        Args:
            filepath: Output file path
            start_date: Optional start date filter
        """
        import csv
        
        # Filter by date if provided
        if start_date:
            entries = [e for e in self.entries if e.timestamp >= start_date]
        else:
            entries = self.entries
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'user_id', 'endpoint', 'model_name',
                'tokens_used', 'latency_ms', 'cached', 'success', 'error_type'
            ])
            
            for e in entries:
                writer.writerow([
                    e.timestamp.isoformat(),
                    e.user_id or '',
                    e.endpoint,
                    e.model_name or '',
                    e.tokens_used,
                    e.latency_ms,
                    e.cached,
                    e.success,
                    e.error_type or '',
                ])
        
        logger.info(f"Exported {len(entries)} usage entries to {filepath}")
