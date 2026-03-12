"""
Database-backed usage analytics.

Provides the same interface as UsageTracker but with SQLite persistence.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from collections import Counter

from nexus.core.tracking.usage_tracker import UsageStats
from nexus.core.database import get_db
from nexus.core.database.repositories import UsageEntryRepository

logger = logging.getLogger(__name__)


class PersistentUsageTracker:
    """
    Database-backed usage tracker.

    Features:
    - Request tracking by endpoint, model, user
    - Performance metrics (latency, success rate)
    - Cache effectiveness
    - Error analytics
    - Time-series data (persisted)
    - Top users/endpoints/models
    """

    def __init__(self):
        """Initialize usage tracker."""
        self.db = get_db()
        self.request_counter = 0
        logger.info("PersistentUsageTracker initialized")

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
        with self.db.get_session() as session:
            usage_repo = UsageEntryRepository(session)
            usage_repo.create(
                endpoint=endpoint,
                user_id=user_id,
                model_name=model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cached=cached,
                success=success,
                error_type=error_type,
            )

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

        with self.db.get_session() as session:
            usage_repo = UsageEntryRepository(session)

            # Get entries for period
            entries = usage_repo.list_by_date_range(
                start_date=start_date,
                end_date=end_date,
                user_id=user_id,
                endpoint=endpoint,
            )

            if not entries:
                return UsageStats(
                    period_start=start_date,
                    period_end=end_date
                )

            # Calculate statistics
            stats = UsageStats(
                period_start=start_date,
                period_end=end_date
            )

            stats.total_requests = len(entries)
            stats.successful_requests = sum(1 for e in entries if e.success)
            stats.failed_requests = sum(1 for e in entries if not e.success)
            stats.total_tokens = sum(e.tokens_used for e in entries)

            # Average latency
            latencies = [e.latency_ms for e in entries if e.latency_ms > 0]
            stats.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

            # Cache hit rate
            cached_count = sum(1 for e in entries if e.cached)
            stats.cache_hit_rate = (cached_count / stats.total_requests * 100) if stats.total_requests > 0 else 0.0

            # Requests by endpoint
            stats.requests_by_endpoint = dict(Counter(e.endpoint for e in entries))

            # Requests by model
            model_counts = Counter(e.model_name for e in entries if e.model_name)
            stats.requests_by_model = dict(model_counts)

            # Requests by user
            user_counts = Counter(e.user_id for e in entries if e.user_id)
            stats.requests_by_user = dict(user_counts)

            # Errors by type
            error_counts = Counter(e.error_type for e in entries if e.error_type)
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

        with self.db.get_session() as session:
            usage_repo = UsageEntryRepository(session)
            all_entries = usage_repo.list_by_date_range(
                start_date=start_time,
                end_date=end_time,
            )

        # Normalize entry timestamps to tz-aware (SQLite returns naive)
        for e in all_entries:
            if e.timestamp and e.timestamp.tzinfo is None:
                e.timestamp = e.timestamp.replace(tzinfo=timezone.utc)

        hourly_stats = []

        for i in range(hours):
            hour_start = start_time + timedelta(hours=i)
            hour_end = hour_start + timedelta(hours=1)

            hour_entries = [
                e for e in all_entries
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

        with self.db.get_session() as session:
            usage_repo = UsageEntryRepository(session)
            entries = usage_repo.list_by_date_range(
                start_date=cutoff,
                end_date=datetime.now(timezone.utc),
            )

        # Filter entries with user_id
        entries_with_user = [e for e in entries if e.user_id]

        # Count by user
        user_counts = Counter(e.user_id for e in entries_with_user)
        return user_counts.most_common(limit)

    def get_top_endpoints(self, limit: int = 10) -> List[tuple]:
        """
        Get most used endpoints.

        Args:
            limit: Number of endpoints to return

        Returns:
            List of (endpoint, count) tuples
        """
        # Get last 7 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)

        with self.db.get_session() as session:
            usage_repo = UsageEntryRepository(session)
            entries = usage_repo.list_by_date_range(
                start_date=cutoff,
                end_date=datetime.now(timezone.utc),
            )

        endpoint_counts = Counter(e.endpoint for e in entries)
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

        with self.db.get_session() as session:
            usage_repo = UsageEntryRepository(session)
            entries = usage_repo.list_by_date_range(
                start_date=cutoff,
                end_date=datetime.now(timezone.utc),
            )

        failed = [e for e in entries if not e.success]

        return {
            'total_requests': len(entries),
            'failed_requests': len(failed),
            'error_rate_percent': (len(failed) / len(entries) * 100) if entries else 0.0,
            'errors_by_type': dict(Counter(e.error_type for e in failed if e.error_type)),
            'errors_by_endpoint': dict(Counter(e.endpoint for e in failed)),
        }
