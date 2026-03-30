"""
Feedback Tracker - Tracks user feedback for meta-learning.

This module records and analyzes user feedback on model responses
to support continuous improvement and model weight adjustment.
"""

import logging
import threading
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
from statistics import mean
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """A single feedback record."""

    request_id: UUID
    model_name: str
    feedback_score: float  # 0-1 scale
    feedback_text: Optional[str] = None
    query_type: Optional[str] = None
    response_latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": str(self.request_id),
            "model_name": self.model_name,
            "feedback_score": self.feedback_score,
            "feedback_text": self.feedback_text,
            "query_type": self.query_type,
            "response_latency_ms": self.response_latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ModelFeedbackStats:
    """Aggregated feedback statistics for a model."""

    total_feedback: int = 0
    average_score: float = 0.0
    positive_count: int = 0  # score >= 0.7
    negative_count: int = 0  # score < 0.3
    neutral_count: int = 0   # 0.3 <= score < 0.7
    recent_trend: float = 0.0  # Positive = improving, negative = declining
    query_type_scores: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackTracker:
    """
    Tracks and analyzes user feedback on model responses.

    Features:
    - Record feedback with context
    - Calculate per-model statistics
    - Detect feedback trends
    - Export feedback for analysis
    - Weight adjustment recommendations
    """

    def __init__(
        self,
        max_records: int = 10000,
        trend_window: int = 100,
        persistence_path: Optional[str] = None,
        auto_save_interval: int = 100,
    ):
        """
        Initialize the feedback tracker.

        Args:
            max_records: Maximum feedback records to keep in memory
            trend_window: Number of recent records for trend calculation
            persistence_path: Path to save feedback data (optional)
            auto_save_interval: Save to disk every N records
        """
        self.max_records = max_records
        self.trend_window = trend_window
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.auto_save_interval = auto_save_interval

        self._records: deque = deque(maxlen=max_records)
        self._model_records: Dict[str, deque] = {}
        self._model_stats: Dict[str, ModelFeedbackStats] = {}
        self._lock = threading.RLock()
        self._unsaved_count = 0

        # Load existing data if persistence is enabled
        if self.persistence_path and self.persistence_path.exists():
            self._load_from_disk()

        logger.info("FeedbackTracker initialized with max_records=%d", max_records)

    def record_feedback(
        self,
        request_id: UUID,
        model_name: str,
        feedback_score: float,
        feedback_text: Optional[str] = None,
        query_type: Optional[str] = None,
        response_latency_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record feedback for a model response.

        Args:
            request_id: Original request identifier
            model_name: Name of the model that generated the response
            feedback_score: Feedback score (0-1, where 1 is best)
            feedback_text: Optional textual feedback
            query_type: Type of query (factual, creative, etc.)
            response_latency_ms: Response latency for correlation analysis
            metadata: Additional metadata
        """
        # Validate score
        feedback_score = max(0.0, min(1.0, feedback_score))

        record = FeedbackRecord(
            request_id=request_id,
            model_name=model_name,
            feedback_score=feedback_score,
            feedback_text=feedback_text,
            query_type=query_type,
            response_latency_ms=response_latency_ms,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)

            # Track per-model records
            if model_name not in self._model_records:
                self._model_records[model_name] = deque(maxlen=self.max_records)
            self._model_records[model_name].append(record)

            # Update model statistics
            self._update_model_stats(model_name)

            # Auto-save if enabled
            self._unsaved_count += 1
            if self.persistence_path and self._unsaved_count >= self.auto_save_interval:
                self._save_to_disk()
                self._unsaved_count = 0

        logger.debug(f"Recorded feedback for {model_name}: score={feedback_score:.2f}")

    def _update_model_stats(self, model_name: str) -> None:
        """Update aggregated statistics for a model."""
        records = list(self._model_records.get(model_name, []))
        if not records:
            return

        # Calculate basic stats
        scores = [r.feedback_score for r in records]
        stats = ModelFeedbackStats(
            total_feedback=len(records),
            average_score=mean(scores),
            positive_count=sum(1 for s in scores if s >= 0.7),
            negative_count=sum(1 for s in scores if s < 0.3),
            neutral_count=sum(1 for s in scores if 0.3 <= s < 0.7),
            last_updated=lambda: datetime.now(timezone.utc)(),
        )

        # Calculate trend (comparing recent to older)
        if len(records) >= self.trend_window * 2:
            old_scores = scores[-self.trend_window*2:-self.trend_window]
            recent_scores = scores[-self.trend_window:]
            stats.recent_trend = mean(recent_scores) - mean(old_scores)
        elif len(records) >= self.trend_window:
            half = len(records) // 2
            old_scores = scores[:half]
            recent_scores = scores[half:]
            stats.recent_trend = mean(recent_scores) - mean(old_scores)

        # Calculate per-query-type scores
        query_type_scores: Dict[str, List[float]] = {}
        for record in records:
            if record.query_type:
                if record.query_type not in query_type_scores:
                    query_type_scores[record.query_type] = []
                query_type_scores[record.query_type].append(record.feedback_score)

        stats.query_type_scores = {
            qt: mean(scores) for qt, scores in query_type_scores.items()
        }

        self._model_stats[model_name] = stats

    def get_model_stats(self, model_name: str) -> Optional[ModelFeedbackStats]:
        """Get feedback statistics for a model."""
        with self._lock:
            return self._model_stats.get(model_name)

    def get_all_model_stats(self) -> Dict[str, ModelFeedbackStats]:
        """Get feedback statistics for all models."""
        with self._lock:
            return dict(self._model_stats)

    def get_weight_recommendations(self) -> Dict[str, float]:
        """
        Get recommended weight adjustments based on feedback.

        Returns:
            Dictionary of model_name -> weight_adjustment (-1 to +1)
        """
        with self._lock:
            if not self._model_stats:
                return {}

            recommendations = {}
            all_scores = [s.average_score for s in self._model_stats.values()]
            if not all_scores:
                return {}

            mean_score = mean(all_scores)

            for model_name, stats in self._model_stats.items():
                # Calculate adjustment based on performance vs average
                score_diff = stats.average_score - mean_score

                # Factor in trend
                trend_factor = stats.recent_trend * 0.5

                # Calculate total adjustment (capped at -0.5 to +0.5)
                adjustment = max(-0.5, min(0.5, score_diff + trend_factor))
                recommendations[model_name] = adjustment

            return recommendations

    def get_best_model_for_query_type(self, query_type: str) -> Optional[str]:
        """
        Get the best performing model for a specific query type.

        Args:
            query_type: The type of query

        Returns:
            Name of the best model, or None if no data
        """
        with self._lock:
            best_model = None
            best_score = -1.0

            for model_name, stats in self._model_stats.items():
                if query_type in stats.query_type_scores:
                    score = stats.query_type_scores[query_type]
                    if score > best_score:
                        best_score = score
                        best_model = model_name

            return best_model

    def get_query_type_rankings(self, query_type: str) -> List[Tuple[str, float]]:
        """
        Get model rankings for a specific query type.

        Args:
            query_type: The type of query

        Returns:
            List of (model_name, score) tuples sorted by score descending
        """
        with self._lock:
            rankings = []
            for model_name, stats in self._model_stats.items():
                if query_type in stats.query_type_scores:
                    rankings.append((model_name, stats.query_type_scores[query_type]))

            return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_recent_feedback(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[FeedbackRecord]:
        """Get recent feedback records."""
        with self._lock:
            if model_name:
                records = list(self._model_records.get(model_name, []))
            else:
                records = list(self._records)

            if since:
                records = [r for r in records if r.timestamp >= since]

            return records[-limit:]

    def export_feedback(self, filepath: str) -> int:
        """
        Export all feedback to a JSON file.

        Args:
            filepath: Path to export file

        Returns:
            Number of records exported
        """
        with self._lock:
            records = [r.to_dict() for r in self._records]

        with open(filepath, 'w') as f:
            json.dump(records, f, indent=2)

        logger.info(f"Exported {len(records)} feedback records to {filepath}")
        return len(records)

    def _save_to_disk(self) -> None:
        """Save feedback data to disk."""
        if not self.persistence_path:
            return

        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            records = [r.to_dict() for r in self._records]
            with open(self.persistence_path, 'w') as f:
                json.dump(records, f)
            logger.debug(f"Saved {len(records)} feedback records to disk")
        except Exception as e:
            logger.error(f"Failed to save feedback to disk: {e}")

    def _load_from_disk(self) -> None:
        """Load feedback data from disk."""
        if not self.persistence_path or not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, 'r') as f:
                data = json.load(f)

            for record_dict in data:
                # Reconstruct records (simplified - would need proper UUID parsing in production)
                self._records.append(FeedbackRecord(
                    request_id=UUID(record_dict["request_id"]),
                    model_name=record_dict["model_name"],
                    feedback_score=record_dict["feedback_score"],
                    feedback_text=record_dict.get("feedback_text"),
                    query_type=record_dict.get("query_type"),
                    response_latency_ms=record_dict.get("response_latency_ms"),
                    timestamp=datetime.fromisoformat(record_dict["timestamp"]),
                    metadata=record_dict.get("metadata", {}),
                ))

            logger.info(f"Loaded {len(self._records)} feedback records from disk")
        except Exception as e:
            logger.error(f"Failed to load feedback from disk: {e}")

    def clear(self) -> None:
        """Clear all feedback records."""
        with self._lock:
            self._records.clear()
            self._model_records.clear()
            self._model_stats.clear()
        logger.info("Cleared all feedback records")
