"""
Content Analytics and Quality Tracking for Content Library.

Provides:
- Interaction tracking
- Quality metric auto-updates
- Content performance analytics
- Trend analysis
- Recommendations for content improvement
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import statistics

from .models import (
    ContentItem,
    ContentInteraction,
    ContentQualityMetrics,
    InteractionType,
    ContentStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Analytics Report Structures
# =============================================================================

@dataclass
class ContentAnalyticsReport:
    """Detailed analytics report for a single content item."""
    content_id: str
    title: str

    # Engagement metrics
    total_views: int = 0
    unique_viewers: int = 0
    total_completions: int = 0
    completion_rate: float = 0.0
    avg_view_duration_seconds: float = 0.0
    bounce_rate: float = 0.0

    # Performance metrics
    avg_performance_score: float = 0.0
    pass_rate: float = 0.0
    avg_attempts: float = 0.0

    # Feedback metrics
    avg_rating: float = 0.0
    total_ratings: int = 0
    positive_feedback_rate: float = 0.0
    feedback_comments: List[str] = field(default_factory=list)

    # Trends
    views_trend: str = "stable"  # increasing, decreasing, stable
    completion_trend: str = "stable"

    # Time analysis
    peak_usage_hours: List[int] = field(default_factory=list)
    avg_time_to_complete_minutes: float = 0.0

    # Quality scores
    quality_score: float = 0.0
    engagement_score: float = 0.0
    effectiveness_score: float = 0.0

    # Recommendations
    improvement_suggestions: List[str] = field(default_factory=list)

    # Report metadata
    report_period_start: Optional[datetime] = None
    report_period_end: Optional[datetime] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TrendingContentReport:
    """Report on trending content."""
    period: str  # day, week, month
    top_viewed: List[Dict[str, Any]] = field(default_factory=list)
    top_completed: List[Dict[str, Any]] = field(default_factory=list)
    top_rated: List[Dict[str, Any]] = field(default_factory=list)
    fastest_growing: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UnderperformingContentReport:
    """Report on content that needs attention."""
    low_completion_rate: List[Dict[str, Any]] = field(default_factory=list)
    high_skip_rate: List[Dict[str, Any]] = field(default_factory=list)
    low_rating: List[Dict[str, Any]] = field(default_factory=list)
    stale_content: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Content Analytics Engine
# =============================================================================

class ContentAnalytics:
    """
    Track and analyze content performance.

    Records user interactions and generates analytics reports.
    """

    def __init__(self, content_library=None, storage_backend=None):
        """
        Initialize analytics engine.

        Args:
            content_library: Reference to content library for updates
            storage_backend: Optional storage backend (used if content_library not provided)
        """
        self.library = content_library
        self.storage = storage_backend

        # Interaction storage
        self.interactions: List[ContentInteraction] = []
        self.interactions_by_content: Dict[str, List[ContentInteraction]] = defaultdict(list)
        self.interactions_by_user: Dict[str, List[ContentInteraction]] = defaultdict(list)

        # Aggregated metrics
        self.viewer_sets: Dict[str, Set[str]] = defaultdict(set)  # content_id -> user_ids

        # Configuration
        self.retention_days = 90  # How long to keep interactions

        logger.info("ContentAnalytics initialized")

    # =========================================================================
    # Recording Interactions
    # =========================================================================

    def record_interaction(self, interaction: ContentInteraction):
        """
        Record a user interaction with content.

        Args:
            interaction: ContentInteraction to record
        """
        self.interactions.append(interaction)
        self.interactions_by_content[interaction.content_id].append(interaction)
        self.interactions_by_user[interaction.user_id].append(interaction)

        # Track unique viewers
        self.viewer_sets[interaction.content_id].add(interaction.user_id)

        # Update content quality metrics if library is available
        if self.library:
            self._update_content_metrics(interaction)

        logger.debug(f"Recorded {interaction.interaction_type.value} for content {interaction.content_id}")

    def record_view(
        self,
        content_id: str,
        user_id: str,
        duration_seconds: int,
        session_id: Optional[str] = None
    ):
        """Record a content view."""
        interaction = ContentInteraction(
            interaction_id="",
            content_id=content_id,
            user_id=user_id,
            interaction_type=InteractionType.VIEW,
            duration_seconds=duration_seconds,
            session_id=session_id
        )
        self.record_interaction(interaction)

    def record_completion(
        self,
        content_id: str,
        user_id: str,
        duration_seconds: int,
        performance_score: float,
        session_id: Optional[str] = None
    ):
        """Record a content completion."""
        interaction = ContentInteraction(
            interaction_id="",
            content_id=content_id,
            user_id=user_id,
            interaction_type=InteractionType.COMPLETE,
            duration_seconds=duration_seconds,
            performance_score=performance_score,
            session_id=session_id
        )
        self.record_interaction(interaction)

    def record_feedback(
        self,
        content_id: str,
        user_id: str,
        rating: float,
        feedback: Optional[str] = None
    ):
        """Record user feedback/rating."""
        interaction = ContentInteraction(
            interaction_id="",
            content_id=content_id,
            user_id=user_id,
            interaction_type=InteractionType.RATE,
            rating=rating,
            feedback=feedback
        )
        self.record_interaction(interaction)

    def record_skip(self, content_id: str, user_id: str, duration_seconds: int = 0):
        """Record when content is skipped."""
        interaction = ContentInteraction(
            interaction_id="",
            content_id=content_id,
            user_id=user_id,
            interaction_type=InteractionType.SKIP,
            duration_seconds=duration_seconds
        )
        self.record_interaction(interaction)

    def record_bookmark(self, content_id: str, user_id: str):
        """Record a content bookmark."""
        interaction = ContentInteraction(
            interaction_id="",
            content_id=content_id,
            user_id=user_id,
            interaction_type=InteractionType.BOOKMARK
        )
        self.record_interaction(interaction)

    # =========================================================================
    # Analytics Reports
    # =========================================================================

    def get_content_analytics(
        self,
        content_id: str,
        period_days: int = 30
    ) -> ContentAnalyticsReport:
        """
        Get detailed analytics for a specific content item.

        Args:
            content_id: Content ID to analyze
            period_days: Period to analyze (default 30 days)

        Returns:
            ContentAnalyticsReport with detailed metrics
        """
        # Get content info
        content = None
        title = content_id
        if self.library:
            content = self.library.get_content(content_id)
            if content:
                title = content.title

        # Filter interactions by period
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
        interactions = [
            i for i in self.interactions_by_content.get(content_id, [])
            if i.timestamp >= cutoff
        ]

        if not interactions:
            return ContentAnalyticsReport(
                content_id=content_id,
                title=title,
                report_period_start=cutoff,
                report_period_end=datetime.now(timezone.utc)
            )

        # Calculate metrics
        views = [i for i in interactions if i.interaction_type == InteractionType.VIEW]
        completions = [i for i in interactions if i.interaction_type == InteractionType.COMPLETE]
        skips = [i for i in interactions if i.interaction_type == InteractionType.SKIP]
        ratings = [i for i in interactions if i.interaction_type == InteractionType.RATE]

        unique_viewers = len(set(i.user_id for i in views))

        # Engagement metrics
        avg_duration = statistics.mean([i.duration_seconds for i in views]) if views else 0
        completion_rate = len(completions) / len(views) if views else 0
        bounce_rate = len([v for v in views if v.duration_seconds < 30]) / len(views) if views else 0

        # Performance metrics
        perf_scores = [c.performance_score for c in completions if c.performance_score is not None]
        avg_performance = statistics.mean(perf_scores) if perf_scores else 0
        pass_rate = len([p for p in perf_scores if p >= 0.7]) / len(perf_scores) if perf_scores else 0

        # Rating metrics
        rating_values = [r.rating for r in ratings if r.rating is not None]
        avg_rating = statistics.mean(rating_values) if rating_values else 0
        positive_rate = len([r for r in rating_values if r >= 4]) / len(rating_values) if rating_values else 0

        # Feedback comments
        comments = [r.feedback for r in ratings if r.feedback]

        # Time analysis
        hours = [i.timestamp.hour for i in interactions]
        hour_counts = defaultdict(int)
        for h in hours:
            hour_counts[h] += 1
        peak_hours = sorted(hour_counts.keys(), key=lambda h: hour_counts[h], reverse=True)[:3]

        # Time to complete
        completion_times = [c.duration_seconds / 60 for c in completions]
        avg_completion_time = statistics.mean(completion_times) if completion_times else 0

        # Trend analysis (compare first half to second half of period)
        mid_point = cutoff + timedelta(days=period_days // 2)
        first_half_views = len([v for v in views if v.timestamp < mid_point])
        second_half_views = len([v for v in views if v.timestamp >= mid_point])
        views_trend = "increasing" if second_half_views > first_half_views * 1.2 else \
                     "decreasing" if second_half_views < first_half_views * 0.8 else "stable"

        # Quality scores from content
        quality_score = content.quality_metrics.quality_score if content else 0
        engagement_score = content.quality_metrics.engagement_score if content else 0
        effectiveness_score = content.quality_metrics.effectiveness_score if content else 0

        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            completion_rate, avg_rating, bounce_rate, avg_performance
        )

        return ContentAnalyticsReport(
            content_id=content_id,
            title=title,
            total_views=len(views),
            unique_viewers=unique_viewers,
            total_completions=len(completions),
            completion_rate=completion_rate,
            avg_view_duration_seconds=avg_duration,
            bounce_rate=bounce_rate,
            avg_performance_score=avg_performance,
            pass_rate=pass_rate,
            avg_rating=avg_rating,
            total_ratings=len(ratings),
            positive_feedback_rate=positive_rate,
            feedback_comments=comments[:10],
            views_trend=views_trend,
            peak_usage_hours=peak_hours,
            avg_time_to_complete_minutes=avg_completion_time,
            quality_score=quality_score,
            engagement_score=engagement_score,
            effectiveness_score=effectiveness_score,
            improvement_suggestions=suggestions,
            report_period_start=cutoff,
            report_period_end=datetime.now(timezone.utc)
        )

    def get_trending_content(self, period: str = "week", limit: int = 10) -> TrendingContentReport:
        """
        Get trending content report.

        Args:
            period: Time period ("day", "week", "month")
            limit: Maximum items per category

        Returns:
            TrendingContentReport
        """
        # Determine cutoff
        period_days = {"day": 1, "week": 7, "month": 30}.get(period, 7)
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)

        # Filter recent interactions
        recent = [i for i in self.interactions if i.timestamp >= cutoff]

        # Count by content
        view_counts: Dict[str, int] = defaultdict(int)
        completion_counts: Dict[str, int] = defaultdict(int)
        rating_sums: Dict[str, List[float]] = defaultdict(list)

        for interaction in recent:
            if interaction.interaction_type == InteractionType.VIEW:
                view_counts[interaction.content_id] += 1
            elif interaction.interaction_type == InteractionType.COMPLETE:
                completion_counts[interaction.content_id] += 1
            elif interaction.interaction_type == InteractionType.RATE and interaction.rating:
                rating_sums[interaction.content_id].append(interaction.rating)

        # Get content titles
        def get_title(content_id: str) -> str:
            if self.library:
                content = self.library.get_content(content_id)
                if content:
                    return content.title
            return content_id

        # Top viewed
        top_viewed = sorted(view_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        top_viewed_list = [
            {"content_id": cid, "title": get_title(cid), "views": count}
            for cid, count in top_viewed
        ]

        # Top completed
        top_completed = sorted(completion_counts.items(), key=lambda x: x[1], reverse=True)[:limit]
        top_completed_list = [
            {"content_id": cid, "title": get_title(cid), "completions": count}
            for cid, count in top_completed
        ]

        # Top rated
        avg_ratings = {cid: statistics.mean(ratings) for cid, ratings in rating_sums.items() if ratings}
        top_rated = sorted(avg_ratings.items(), key=lambda x: x[1], reverse=True)[:limit]
        top_rated_list = [
            {"content_id": cid, "title": get_title(cid), "avg_rating": rating}
            for cid, rating in top_rated
        ]

        return TrendingContentReport(
            period=period,
            top_viewed=top_viewed_list,
            top_completed=top_completed_list,
            top_rated=top_rated_list
        )

    def get_underperforming_content(
        self,
        completion_threshold: float = 0.3,
        rating_threshold: float = 3.0,
        stale_days: int = 60
    ) -> UnderperformingContentReport:
        """
        Get report on content that needs improvement.

        Args:
            completion_threshold: Completion rate below this is flagged
            rating_threshold: Rating below this is flagged
            stale_days: Days without interaction to flag as stale

        Returns:
            UnderperformingContentReport
        """
        low_completion = []
        high_skip = []
        low_rating = []
        stale = []

        # Need library to get all content
        if not self.library:
            return UnderperformingContentReport()

        # Check each content
        for content in self.library.list():
            content_id = content.content_id
            interactions = self.interactions_by_content.get(content_id, [])

            if not interactions:
                # Check if stale
                if content.updated_at < datetime.now(timezone.utc) - timedelta(days=stale_days):
                    stale.append({
                        "content_id": content_id,
                        "title": content.title,
                        "last_updated": content.updated_at.isoformat()
                    })
                continue

            views = [i for i in interactions if i.interaction_type == InteractionType.VIEW]
            completions = [i for i in interactions if i.interaction_type == InteractionType.COMPLETE]
            skips = [i for i in interactions if i.interaction_type == InteractionType.SKIP]
            ratings = [i for i in interactions if i.interaction_type == InteractionType.RATE and i.rating]

            # Check completion rate
            if views and len(completions) / len(views) < completion_threshold:
                low_completion.append({
                    "content_id": content_id,
                    "title": content.title,
                    "completion_rate": len(completions) / len(views),
                    "total_views": len(views)
                })

            # Check skip rate
            if views and skips and len(skips) / len(views) > 0.5:
                high_skip.append({
                    "content_id": content_id,
                    "title": content.title,
                    "skip_rate": len(skips) / len(views)
                })

            # Check rating
            if ratings:
                avg_rating = statistics.mean([r.rating for r in ratings])
                if avg_rating < rating_threshold:
                    low_rating.append({
                        "content_id": content_id,
                        "title": content.title,
                        "avg_rating": avg_rating,
                        "total_ratings": len(ratings)
                    })

        return UnderperformingContentReport(
            low_completion_rate=sorted(low_completion, key=lambda x: x["completion_rate"])[:20],
            high_skip_rate=sorted(high_skip, key=lambda x: x["skip_rate"], reverse=True)[:20],
            low_rating=sorted(low_rating, key=lambda x: x["avg_rating"])[:20],
            stale_content=stale[:20]
        )

    def get_user_content_history(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[ContentInteraction]:
        """
        Get a user's content interaction history.

        Args:
            user_id: User identifier
            limit: Maximum interactions to return

        Returns:
            List of ContentInteractions, most recent first
        """
        interactions = self.interactions_by_user.get(user_id, [])
        return sorted(interactions, key=lambda i: i.timestamp, reverse=True)[:limit]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _update_content_metrics(self, interaction: ContentInteraction):
        """Update content quality metrics based on interaction."""
        if not self.library:
            return

        content = self.library.get_content(interaction.content_id)
        if not content:
            return

        metrics = content.quality_metrics

        if interaction.interaction_type == InteractionType.VIEW:
            metrics.update_from_view(interaction.duration_seconds)
        elif interaction.interaction_type == InteractionType.COMPLETE:
            metrics.update_from_completion(
                interaction.performance_score or 0.5,
                interaction.duration_seconds
            )
        elif interaction.interaction_type == InteractionType.RATE:
            metrics.update_from_rating(
                interaction.rating or 3.0,
                (interaction.rating or 3.0) >= 4.0
            )
        elif interaction.interaction_type == InteractionType.SKIP:
            metrics.update_from_skip()

        # Update content in library
        self.library.update_content(content.content_id, content)

    def _generate_improvement_suggestions(
        self,
        completion_rate: float,
        avg_rating: float,
        bounce_rate: float,
        avg_performance: float
    ) -> List[str]:
        """Generate content improvement suggestions based on metrics."""
        suggestions = []

        if completion_rate < 0.5:
            suggestions.append("Consider breaking content into smaller chunks")
            suggestions.append("Add progress indicators to encourage completion")

        if avg_rating < 3.5:
            suggestions.append("Review user feedback for specific improvement areas")
            suggestions.append("Consider updating content based on common complaints")

        if bounce_rate > 0.3:
            suggestions.append("Improve the introduction to hook learners")
            suggestions.append("Ensure content difficulty matches expectations")

        if avg_performance < 0.6:
            suggestions.append("Add more examples and practice exercises")
            suggestions.append("Consider adding prerequisite content")

        if not suggestions:
            suggestions.append("Content is performing well - maintain quality")

        return suggestions

    def cleanup_old_interactions(self):
        """Remove interactions older than retention period."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)

        self.interactions = [i for i in self.interactions if i.timestamp >= cutoff]

        for content_id in list(self.interactions_by_content.keys()):
            self.interactions_by_content[content_id] = [
                i for i in self.interactions_by_content[content_id]
                if i.timestamp >= cutoff
            ]

        for user_id in list(self.interactions_by_user.keys()):
            self.interactions_by_user[user_id] = [
                i for i in self.interactions_by_user[user_id]
                if i.timestamp >= cutoff
            ]

        logger.info(f"Cleaned up interactions older than {self.retention_days} days")

    def get_statistics(self) -> Dict[str, Any]:
        """Get analytics system statistics."""
        return {
            "total_interactions": len(self.interactions),
            "unique_content_items": len(self.interactions_by_content),
            "unique_users": len(self.interactions_by_user),
            "retention_days": self.retention_days,
            "interactions_by_type": {
                itype.value: len([i for i in self.interactions if i.interaction_type == itype])
                for itype in InteractionType
            }
        }


# =============================================================================
# Content Quality Manager
# =============================================================================

class ContentQualityManager:
    """
    Automatically manage content quality based on analytics.

    Provides:
    - Auto-update of quality scores
    - Quality-based actions (deprecation, review flagging)
    - Improvement recommendations
    """

    def __init__(
        self,
        content_library=None,
        storage_backend=None,
        analytics: ContentAnalytics = None,
        auto_update: bool = True
    ):
        """
        Initialize quality manager.

        Args:
            content_library: Content library reference
            storage_backend: Optional storage backend
            analytics: Analytics engine reference
            auto_update: Whether to auto-update quality on interactions
        """
        self.library = content_library
        self.storage = storage_backend
        self.analytics = analytics
        self.auto_update = auto_update

        # Thresholds
        self.deprecation_threshold = 0.3
        self.review_threshold = 0.5
        self.high_quality_threshold = 0.8

        logger.info("ContentQualityManager initialized")

    def update_all_quality_scores(self):
        """Recalculate quality scores for all content."""
        if not self.library:
            return

        updated = 0
        for content in self.library.list():
            self._update_quality_score(content)
            updated += 1

        logger.info(f"Updated quality scores for {updated} content items")

    def _update_quality_score(self, content: ContentItem):
        """Update quality score for a single content item."""
        report = self.analytics.get_content_analytics(content.content_id)

        # Update quality metrics
        content.quality_metrics.engagement_score = (
            report.completion_rate * 0.5 +
            (1 - report.bounce_rate) * 0.3 +
            (report.unique_viewers / max(report.total_views, 1)) * 0.2
        )

        content.quality_metrics.effectiveness_score = (
            report.avg_performance_score * 0.6 +
            report.pass_rate * 0.4
        )

        content.quality_metrics._recalculate_scores()

        self.library.update_content(content.content_id, content)

    def auto_deprecate_low_quality(self) -> List[str]:
        """
        Automatically deprecate content below quality threshold.

        Returns:
            List of deprecated content IDs
        """
        deprecated = []

        if not self.library:
            return deprecated

        for content in self.library.list():
            if (content.quality_metrics.quality_score < self.deprecation_threshold and
                content.quality_metrics.total_views >= 10 and  # Minimum sample size
                content.status == ContentStatus.PUBLISHED):

                content.deprecate()
                self.library.update_content(content.content_id, content)
                deprecated.append(content.content_id)
                logger.info(f"Auto-deprecated content: {content.content_id}")

        return deprecated

    def get_content_needing_review(self) -> List[ContentItem]:
        """Get content that should be reviewed for quality issues."""
        needs_review = []

        if not self.library:
            return needs_review

        for content in self.library.list():
            if (content.quality_metrics.quality_score < self.review_threshold and
                content.quality_metrics.total_views >= 5 and
                content.status == ContentStatus.PUBLISHED):
                needs_review.append(content)

        return sorted(needs_review, key=lambda c: c.quality_metrics.quality_score)

    def suggest_improvements(self, content_id: str) -> List[str]:
        """
        Generate specific improvement suggestions for content.

        Args:
            content_id: Content to analyze

        Returns:
            List of improvement suggestions
        """
        report = self.analytics.get_content_analytics(content_id)
        return report.improvement_suggestions

    def update_from_interaction(self, interaction: ContentInteraction):
        """
        Update content quality based on an interaction.

        Args:
            interaction: The interaction that occurred
        """
        if not self.auto_update:
            return

        # Get content from storage or library
        content = None
        if self.library:
            content = self.library.get_content(interaction.content_id)
        elif self.storage:
            content = self.storage.get(interaction.content_id)

        if not content:
            return

        # Update quality metrics based on interaction type
        metrics = content.quality_metrics

        if interaction.interaction_type == InteractionType.VIEW:
            metrics.update_from_view(interaction.duration_seconds)
        elif interaction.interaction_type == InteractionType.COMPLETE:
            metrics.update_from_completion(
                interaction.performance_score or 0.5,
                interaction.duration_seconds
            )
        elif interaction.interaction_type == InteractionType.RATE:
            metrics.update_from_rating(
                interaction.rating or 3.0,
                (interaction.rating or 3.0) >= 4.0
            )
        elif interaction.interaction_type == InteractionType.SKIP:
            metrics.update_from_skip()

        # Save updated content
        if self.storage:
            self.storage.update(interaction.content_id, content)
