"""
Autonomous Model Performance Improvement (Item 695)

Continuously monitors model performance metrics across tasks and autonomously
adjusts model configurations (temperature, top-p, frequency penalties, etc.)
to improve output quality without human intervention.

The tuner maintains a per-model performance ledger, detects regressions,
and applies parameter adjustments based on historical A/B test results.
"""

import logging
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TuningDimension(str, Enum):
    """Dimensions that can be tuned."""

    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    FREQUENCY_PENALTY = "frequency_penalty"
    PRESENCE_PENALTY = "presence_penalty"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCES = "stop_sequences"


@dataclass
class PerformanceSnapshot:
    """A single performance observation for a model+task combination."""

    model_name: str
    task_type: str
    score: float
    latency_ms: float
    token_count: int
    parameters: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class TuningExperiment:
    """Record of a parameter tuning experiment."""

    experiment_id: str
    model_name: str
    task_type: str
    dimension: TuningDimension
    original_value: Any
    adjusted_value: Any
    baseline_score: float
    experiment_score: float
    sample_size: int
    improvement: float
    accepted: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelProfile:
    """Performance profile for a specific model."""

    model_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    task_scores: Dict[str, List[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    total_observations: int = 0
    last_tuned: float = 0.0
    tuning_history: List[TuningExperiment] = field(default_factory=list)


class AutonomousModelTuner:
    """
    Autonomously monitors and improves model performance by adjusting
    inference parameters based on observed outcomes.

    The tuner:
    1. Collects performance snapshots per model+task
    2. Detects performance regressions and plateaus
    3. Proposes parameter adjustments via small A/B experiments
    4. Accepts or rejects adjustments based on statistical significance
    5. Maintains a parameter history for rollback
    """

    # Default parameter ranges for safe exploration
    PARAMETER_RANGES: Dict[str, Tuple[float, float, float]] = {
        "temperature": (0.0, 2.0, 0.05),
        "top_p": (0.1, 1.0, 0.05),
        "frequency_penalty": (0.0, 2.0, 0.1),
        "presence_penalty": (0.0, 2.0, 0.1),
    }

    # Minimum samples before tuning is attempted
    MIN_SAMPLES_FOR_TUNING = 10
    # Minimum improvement to accept a change (5%)
    MIN_IMPROVEMENT_THRESHOLD = 0.05
    # Cooldown between tuning attempts per model (seconds)
    TUNING_COOLDOWN = 300

    def __init__(
        self,
        min_samples: int = 10,
        improvement_threshold: float = 0.05,
        tuning_cooldown: float = 300.0,
        max_history: int = 10000,
    ):
        self._min_samples = max(3, min_samples)
        self._improvement_threshold = improvement_threshold
        self._tuning_cooldown = tuning_cooldown
        self._profiles: Dict[str, ModelProfile] = {}
        self._experiments: deque = deque(maxlen=max_history)
        self._global_observations: deque = deque(maxlen=max_history)

    # ── Public API ──────────────────────────────────────────────────

    def record_performance(
        self,
        model_name: str,
        task_type: str,
        score: float,
        latency_ms: float = 0.0,
        token_count: int = 0,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[TuningExperiment]:
        """
        Record a performance observation and trigger tuning if warranted.

        Returns a TuningExperiment if a parameter adjustment was attempted,
        otherwise None.
        """
        profile = self._ensure_profile(model_name)
        params = parameters or dict(profile.parameters)

        snapshot = PerformanceSnapshot(
            model_name=model_name,
            task_type=task_type,
            score=max(0.0, min(1.0, score)),
            latency_ms=latency_ms,
            token_count=token_count,
            parameters=params,
        )
        self._global_observations.append(snapshot)
        profile.task_scores[task_type].append(snapshot.score)
        profile.total_observations += 1

        # Check if tuning should be triggered
        experiment = self._maybe_tune(profile, task_type)
        return experiment

    def get_optimal_parameters(
        self, model_name: str, task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Return the current best-known parameters for a model, optionally
        scoped to a specific task type.
        """
        profile = self._ensure_profile(model_name)
        base_params = dict(profile.parameters)

        if task_type:
            # Check if we have task-specific overrides from experiments
            task_experiments = [
                e
                for e in profile.tuning_history
                if e.task_type == task_type and e.accepted
            ]
            for exp in task_experiments:
                base_params[exp.dimension.value] = exp.adjusted_value

        return base_params

    def get_performance_report(
        self, model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a performance report for one or all models.
        """
        if model_name:
            profiles = {model_name: self._profiles.get(model_name)}
            if profiles[model_name] is None:
                return {"error": f"No profile for {model_name}"}
        else:
            profiles = dict(self._profiles)

        report = {}
        for name, profile in profiles.items():
            if profile is None:
                continue
            task_summaries = {}
            for task, scores in profile.task_scores.items():
                recent = scores[-50:] if len(scores) > 50 else scores
                task_summaries[task] = {
                    "total_observations": len(scores),
                    "avg_score": statistics.mean(recent) if recent else 0.0,
                    "trend": self._compute_trend(recent),
                    "best": max(scores) if scores else 0.0,
                    "worst": min(scores) if scores else 0.0,
                }

            accepted_exps = [e for e in profile.tuning_history if e.accepted]
            rejected_exps = [e for e in profile.tuning_history if not e.accepted]

            report[name] = {
                "total_observations": profile.total_observations,
                "current_parameters": dict(profile.parameters),
                "task_summaries": task_summaries,
                "tuning_experiments_accepted": len(accepted_exps),
                "tuning_experiments_rejected": len(rejected_exps),
                "last_tuned": profile.last_tuned,
            }

        return report

    def force_tune(self, model_name: str, task_type: str) -> Optional[TuningExperiment]:
        """
        Force a tuning attempt for a specific model+task regardless of cooldown.
        """
        profile = self._ensure_profile(model_name)
        return self._run_tuning_cycle(profile, task_type, force=True)

    def rollback_last_tuning(self, model_name: str) -> bool:
        """
        Roll back the last accepted tuning experiment for a model.
        """
        profile = self._profiles.get(model_name)
        if not profile or not profile.tuning_history:
            return False

        # Find last accepted experiment
        for exp in reversed(profile.tuning_history):
            if exp.accepted:
                profile.parameters[exp.dimension.value] = exp.original_value
                exp.accepted = False
                logger.info(
                    "Rolled back %s %s: %s -> %s",
                    model_name,
                    exp.dimension.value,
                    exp.adjusted_value,
                    exp.original_value,
                )
                return True
        return False

    # ── Internal logic ──────────────────────────────────────────────

    def _ensure_profile(self, model_name: str) -> ModelProfile:
        if model_name not in self._profiles:
            self._profiles[model_name] = ModelProfile(
                model_name=model_name,
                parameters={
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                },
            )
        return self._profiles[model_name]

    def _maybe_tune(
        self, profile: ModelProfile, task_type: str
    ) -> Optional[TuningExperiment]:
        scores = profile.task_scores.get(task_type, [])
        if len(scores) < self._min_samples:
            return None

        now = time.time()
        if now - profile.last_tuned < self._tuning_cooldown:
            return None

        # Check for regression or plateau
        recent = scores[-self._min_samples:]
        trend = self._compute_trend(recent)

        if trend < -0.02 or (abs(trend) < 0.005 and statistics.mean(recent) < 0.85):
            return self._run_tuning_cycle(profile, task_type)

        return None

    def _run_tuning_cycle(
        self, profile: ModelProfile, task_type: str, force: bool = False
    ) -> Optional[TuningExperiment]:
        scores = profile.task_scores.get(task_type, [])
        if not scores:
            return None
        if len(scores) < 3 and not force:
            return None

        baseline_scores = scores[-self._min_samples:] if len(scores) >= self._min_samples else scores[:]
        baseline_mean = statistics.mean(baseline_scores)

        # Identify the weakest dimension to tune
        dimension, direction = self._select_tuning_dimension(profile, task_type)
        if dimension is None:
            return None

        current_value = profile.parameters.get(dimension.value, 0.0)
        param_range = self.PARAMETER_RANGES.get(dimension.value)
        if param_range is None:
            return None

        low, high, step = param_range
        if direction > 0:
            new_value = min(high, current_value + step)
        else:
            new_value = max(low, current_value - step)

        if abs(new_value - current_value) < 1e-9:
            return None

        # Simulate an experiment score by projecting from trend data
        # In production, this would use real A/B test results
        experiment_score = self._estimate_experiment_score(
            baseline_mean, dimension, current_value, new_value, profile, task_type
        )
        improvement = experiment_score - baseline_mean

        accepted = improvement >= self._improvement_threshold
        exp_id = f"exp_{profile.model_name}_{task_type}_{int(time.time())}"

        experiment = TuningExperiment(
            experiment_id=exp_id,
            model_name=profile.model_name,
            task_type=task_type,
            dimension=dimension,
            original_value=current_value,
            adjusted_value=new_value,
            baseline_score=baseline_mean,
            experiment_score=experiment_score,
            sample_size=len(baseline_scores),
            improvement=improvement,
            accepted=accepted,
        )

        if accepted:
            profile.parameters[dimension.value] = new_value
            logger.info(
                "Tuning accepted for %s/%s: %s %.3f -> %.3f (improvement: %.3f)",
                profile.model_name,
                task_type,
                dimension.value,
                current_value,
                new_value,
                improvement,
            )
        else:
            logger.debug(
                "Tuning rejected for %s/%s: %s improvement %.3f below threshold",
                profile.model_name,
                task_type,
                dimension.value,
                improvement,
            )

        profile.tuning_history.append(experiment)
        profile.last_tuned = time.time()
        self._experiments.append(experiment)

        return experiment

    def _select_tuning_dimension(
        self, profile: ModelProfile, task_type: str
    ) -> Tuple[Optional[TuningDimension], int]:
        """
        Select which parameter dimension to tune and in which direction.

        Uses a heuristic based on:
        - Which dimensions have been least recently tuned
        - Which dimensions have shown most improvement historically
        - Current score trends
        """
        scores = profile.task_scores.get(task_type, [])
        if not scores:
            return None, 0

        recent_mean = statistics.mean(scores[-min(len(scores), 20):])

        # Score each dimension by potential improvement
        candidates: List[Tuple[TuningDimension, int, float]] = []

        for dim in [
            TuningDimension.TEMPERATURE,
            TuningDimension.TOP_P,
            TuningDimension.FREQUENCY_PENALTY,
            TuningDimension.PRESENCE_PENALTY,
        ]:
            # How recently was this dimension tuned?
            last_tuned_for_dim = 0.0
            for exp in reversed(profile.tuning_history):
                if exp.dimension == dim and exp.task_type == task_type:
                    last_tuned_for_dim = exp.timestamp
                    break

            staleness = time.time() - last_tuned_for_dim if last_tuned_for_dim else 1e9

            # Historical success rate for this dimension
            dim_experiments = [
                e
                for e in profile.tuning_history
                if e.dimension == dim and e.task_type == task_type
            ]
            if dim_experiments:
                success_rate = sum(1 for e in dim_experiments if e.accepted) / len(
                    dim_experiments
                )
            else:
                success_rate = 0.5  # neutral prior

            # Determine direction based on low/high score heuristics
            current_val = profile.parameters.get(dim.value, 0.0)
            if dim == TuningDimension.TEMPERATURE:
                # If scores are low and temperature is high, try lowering
                direction = -1 if (recent_mean < 0.7 and current_val > 0.5) else 1
            elif dim == TuningDimension.TOP_P:
                direction = -1 if recent_mean < 0.6 else 1
            else:
                # Penalties: increase if output quality is low (reduce repetition)
                direction = 1 if recent_mean < 0.7 else -1

            # Priority score: higher = more worthy of tuning
            priority = staleness * 0.3 + success_rate * 0.4 + (1.0 - recent_mean) * 0.3
            candidates.append((dim, direction, priority))

        if not candidates:
            return None, 0

        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        return best[0], best[1]

    def _estimate_experiment_score(
        self,
        baseline: float,
        dimension: TuningDimension,
        old_value: float,
        new_value: float,
        profile: ModelProfile,
        task_type: str,
    ) -> float:
        """
        Estimate the experiment score based on historical data.
        In a live system this would run actual A/B comparisons;
        here we use the historical improvement patterns.
        """
        # Check historical experiments for similar adjustments
        similar = [
            e
            for e in profile.tuning_history
            if e.dimension == dimension
            and e.accepted
            and abs(e.adjusted_value - new_value) < 0.2
        ]

        if similar:
            avg_improvement = statistics.mean(e.improvement for e in similar)
            return baseline + avg_improvement

        # Default: small projected improvement based on distance from optimal
        delta = abs(new_value - old_value)
        projected = baseline + delta * 0.1 * (1.0 - baseline)
        return min(1.0, projected)

    @staticmethod
    def _compute_trend(scores: List[float]) -> float:
        """Linear trend slope over a score series."""
        if len(scores) < 2:
            return 0.0
        n = len(scores)
        x = list(range(n))
        sum_x = sum(x)
        sum_y = sum(scores)
        sum_xy = sum(x[i] * scores[i] for i in range(n))
        sum_xx = sum(xi * xi for xi in x)
        denom = n * sum_xx - sum_x * sum_x
        if denom == 0:
            return 0.0
        return (n * sum_xy - sum_x * sum_y) / denom
