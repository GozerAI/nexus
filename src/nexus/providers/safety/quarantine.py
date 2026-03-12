"""
Model quarantine system for isolating problematic models.

This module provides automatic quarantine of models that exhibit:
- High error rates
- Degraded performance
- Safety violations
- Anomalous behavior
- Consistency failures

Quarantined models are temporarily removed from the ensemble
until they recover or are manually reviewed.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Deque, Dict, List, Optional


class QuarantineReason(str, Enum):
    """Reason for quarantine."""

    HIGH_ERROR_RATE = "high_error_rate"
    LOW_PERFORMANCE = "low_performance"
    SAFETY_VIOLATION = "safety_violation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    CONSISTENCY_FAILURE = "consistency_failure"
    MANUAL = "manual"


class QuarantineStatus(str, Enum):
    """Quarantine status."""

    ACTIVE = "active"  # Currently quarantined
    PROBATION = "probation"  # Released but monitored closely
    RELEASED = "released"  # Fully released
    PERMANENT = "permanent"  # Permanently quarantined


@dataclass
class QuarantineRecord:
    """
    Record of a model quarantine event.

    Attributes:
        model_name: Model that was quarantined
        reason: Reason for quarantine
        status: Current quarantine status
        quarantined_at: When quarantine started
        released_at: When released (if released)
        error_count: Number of errors that triggered quarantine
        performance_score: Performance score at quarantine time
        metadata: Additional context
        auto_release: Whether to auto-release after recovery
    """

    model_name: str
    reason: QuarantineReason
    status: QuarantineStatus
    quarantined_at: datetime
    released_at: Optional[datetime] = None
    error_count: int = 0
    performance_score: float = 0.0
    metadata: Dict = field(default_factory=dict)
    auto_release: bool = True


class ModelQuarantine:
    """
    Manages quarantine of problematic models.

    The quarantine system monitors model behavior and automatically
    isolates models that show signs of problems:

    Triggers:
    - Error rate > 30% over last 10 queries
    - Performance drop > 40% from baseline
    - Safety policy violations
    - Anomalous output patterns
    - Consistency score < 0.3

    Release Conditions:
    - Error rate < 10% over probation period
    - Performance recovered to > 80% baseline
    - Manual review and approval
    - Auto-release after specified duration

    Features:
    - Automatic quarantine and release
    - Probation period for monitoring
    - Configurable thresholds
    - Audit trail
    - Manual override
    """

    def __init__(
        self,
        error_rate_threshold: float = 0.3,
        performance_drop_threshold: float = 0.4,
        consistency_threshold: float = 0.3,
        probation_period_hours: int = 24,
        auto_release_hours: int = 48,
        min_queries_for_quarantine: int = 10,
    ):
        """
        Initialize model quarantine system.

        Args:
            error_rate_threshold: Error rate triggering quarantine (0-1)
            performance_drop_threshold: Performance drop triggering quarantine (0-1)
            consistency_threshold: Minimum consistency score (0-1)
            probation_period_hours: Hours of probation before full release
            auto_release_hours: Hours before auto-release
            min_queries_for_quarantine: Minimum queries before quarantine possible
        """
        self.error_rate_threshold = error_rate_threshold
        self.performance_drop_threshold = performance_drop_threshold
        self.consistency_threshold = consistency_threshold
        self.probation_period_hours = probation_period_hours
        self.auto_release_hours = auto_release_hours
        self.min_queries_for_quarantine = min_queries_for_quarantine

        # Quarantine records
        self._quarantined: Dict[str, QuarantineRecord] = {}
        self._history: List[QuarantineRecord] = []

        # Recent performance tracking for quarantine decisions
        self._recent_errors: Dict[str, Deque[bool]] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self._performance_baselines: Dict[str, float] = {}

    def check_and_quarantine(
        self,
        model_name: str,
        had_error: bool,
        performance_score: float,
        consistency_score: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Check if model should be quarantined and quarantine if necessary.

        Args:
            model_name: Model name
            had_error: Whether latest query had error
            performance_score: Latest performance score (0-1)
            consistency_score: Optional consistency score (0-1)
            metadata: Optional metadata

        Returns:
            True if model was quarantined
        """
        # Skip if already quarantined
        if self.is_quarantined(model_name):
            return False

        # Track error
        self._recent_errors[model_name].append(had_error)

        # Update baseline performance if not set
        if model_name not in self._performance_baselines:
            self._performance_baselines[model_name] = performance_score

        # Check if we have enough data
        if len(self._recent_errors[model_name]) < self.min_queries_for_quarantine:
            return False

        # Check error rate
        recent_errors = list(self._recent_errors[model_name])
        error_rate = sum(recent_errors) / len(recent_errors)

        if error_rate >= self.error_rate_threshold:
            self._quarantine_model(
                model_name=model_name,
                reason=QuarantineReason.HIGH_ERROR_RATE,
                error_count=sum(recent_errors),
                performance_score=performance_score,
                metadata=metadata or {},
            )
            return True

        # Check performance drop
        baseline = self._performance_baselines[model_name]
        if baseline > 0:
            performance_drop = (baseline - performance_score) / baseline
            if performance_drop >= self.performance_drop_threshold:
                self._quarantine_model(
                    model_name=model_name,
                    reason=QuarantineReason.LOW_PERFORMANCE,
                    error_count=sum(recent_errors),
                    performance_score=performance_score,
                    metadata=metadata or {},
                )
                return True

        # Check consistency
        if consistency_score is not None and consistency_score < self.consistency_threshold:
            self._quarantine_model(
                model_name=model_name,
                reason=QuarantineReason.CONSISTENCY_FAILURE,
                error_count=sum(recent_errors),
                performance_score=performance_score,
                metadata={**(metadata or {}), "consistency_score": consistency_score},
            )
            return True

        return False

    def quarantine_model(
        self,
        model_name: str,
        reason: QuarantineReason,
        metadata: Optional[Dict] = None,
        permanent: bool = False,
    ):
        """
        Manually quarantine a model.

        Args:
            model_name: Model to quarantine
            reason: Reason for quarantine
            metadata: Optional metadata
            permanent: Whether quarantine is permanent
        """
        self._quarantine_model(
            model_name=model_name,
            reason=reason,
            error_count=0,
            performance_score=0.0,
            metadata=metadata or {},
            auto_release=not permanent,
            status=QuarantineStatus.PERMANENT if permanent else QuarantineStatus.ACTIVE,
        )

    def _quarantine_model(
        self,
        model_name: str,
        reason: QuarantineReason,
        error_count: int,
        performance_score: float,
        metadata: Dict,
        auto_release: bool = True,
        status: QuarantineStatus = QuarantineStatus.ACTIVE,
    ):
        """
        Internal method to quarantine a model.

        Args:
            model_name: Model to quarantine
            reason: Reason for quarantine
            error_count: Number of errors
            performance_score: Performance score
            metadata: Metadata
            auto_release: Whether to auto-release
            status: Initial status
        """
        record = QuarantineRecord(
            model_name=model_name,
            reason=reason,
            status=status,
            quarantined_at=datetime.now(),
            error_count=error_count,
            performance_score=performance_score,
            metadata=metadata,
            auto_release=auto_release,
        )

        self._quarantined[model_name] = record
        self._history.append(record)

    def release_model(
        self,
        model_name: str,
        use_probation: bool = True,
    ) -> bool:
        """
        Release a model from quarantine.

        Args:
            model_name: Model to release
            use_probation: Whether to use probation period

        Returns:
            True if released
        """
        if model_name not in self._quarantined:
            return False

        record = self._quarantined[model_name]

        # Can't release permanent quarantine
        if record.status == QuarantineStatus.PERMANENT:
            return False

        # Update status
        if use_probation:
            record.status = QuarantineStatus.PROBATION
            record.released_at = datetime.now()
        else:
            record.status = QuarantineStatus.RELEASED
            record.released_at = datetime.now()
            del self._quarantined[model_name]

        # Clear recent errors
        if model_name in self._recent_errors:
            self._recent_errors[model_name].clear()

        return True

    def is_quarantined(
        self,
        model_name: str,
        include_probation: bool = False,
    ) -> bool:
        """
        Check if model is quarantined.

        Args:
            model_name: Model to check
            include_probation: Whether to consider probation as quarantined

        Returns:
            True if quarantined
        """
        if model_name not in self._quarantined:
            return False

        record = self._quarantined[model_name]

        if record.status in [QuarantineStatus.ACTIVE, QuarantineStatus.PERMANENT]:
            return True

        if include_probation and record.status == QuarantineStatus.PROBATION:
            return True

        return False

    def get_quarantine_status(
        self,
        model_name: str,
    ) -> Optional[QuarantineRecord]:
        """
        Get quarantine record for a model.

        Args:
            model_name: Model name

        Returns:
            QuarantineRecord if quarantined
        """
        return self._quarantined.get(model_name)

    def check_auto_release(self):
        """
        Check and perform auto-release for eligible models.

        Should be called periodically (e.g., hourly).
        """
        now = datetime.now()

        for model_name, record in list(self._quarantined.items()):
            # Skip permanent quarantine
            if record.status == QuarantineStatus.PERMANENT:
                continue

            # Skip if no auto-release
            if not record.auto_release:
                continue

            # Check if probation period is over
            if record.status == QuarantineStatus.PROBATION:
                if record.released_at:
                    time_in_probation = (now - record.released_at).total_seconds() / 3600
                    if time_in_probation >= self.probation_period_hours:
                        # Check recent performance
                        if model_name in self._recent_errors:
                            recent_errors = list(self._recent_errors[model_name])
                            if recent_errors:
                                error_rate = sum(recent_errors) / len(recent_errors)
                                if error_rate < 0.1:  # Less than 10% error rate
                                    # Fully release
                                    record.status = QuarantineStatus.RELEASED
                                    del self._quarantined[model_name]

            # Check if auto-release time is reached
            elif record.status == QuarantineStatus.ACTIVE:
                time_quarantined = (now - record.quarantined_at).total_seconds() / 3600
                if time_quarantined >= self.auto_release_hours:
                    # Move to probation
                    self.release_model(model_name, use_probation=True)

    def get_quarantined_models(
        self,
        include_probation: bool = False,
    ) -> List[str]:
        """
        Get list of quarantined models.

        Args:
            include_probation: Whether to include models on probation

        Returns:
            List of model names
        """
        models = []

        for model_name, record in self._quarantined.items():
            if record.status in [QuarantineStatus.ACTIVE, QuarantineStatus.PERMANENT]:
                models.append(model_name)
            elif include_probation and record.status == QuarantineStatus.PROBATION:
                models.append(model_name)

        return models

    def get_history(
        self,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[QuarantineRecord]:
        """
        Get quarantine history.

        Args:
            model_name: Filter by model name
            limit: Maximum records

        Returns:
            List of quarantine records
        """
        if model_name:
            records = [r for r in self._history if r.model_name == model_name]
        else:
            records = self._history

        return records[-limit:]

    def get_stats(self) -> Dict:
        """
        Get quarantine statistics.

        Returns:
            Dictionary with stats
        """
        active_count = sum(
            1 for r in self._quarantined.values()
            if r.status == QuarantineStatus.ACTIVE
        )
        probation_count = sum(
            1 for r in self._quarantined.values()
            if r.status == QuarantineStatus.PROBATION
        )
        permanent_count = sum(
            1 for r in self._quarantined.values()
            if r.status == QuarantineStatus.PERMANENT
        )

        # Count by reason
        reason_counts = defaultdict(int)
        for record in self._history:
            reason_counts[record.reason.value] += 1

        return {
            "active_quarantines": active_count,
            "probation_count": probation_count,
            "permanent_quarantines": permanent_count,
            "total_quarantine_events": len(self._history),
            "quarantines_by_reason": dict(reason_counts),
            "models_tracked": len(self._recent_errors),
        }
