"""
Confidence calibration module - scientific uncertainty quantification.

This module implements sophisticated confidence calibration that:
1. Analyzes agreement between multiple models
2. Calculates statistical confidence intervals
3. Detects consensus vs. disagreement patterns
4. Provides calibrated confidence scores

Based on combo1's confidence calibration with enhancements.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

from nexus.providers.ensemble.types import ModelResponse


@dataclass
class ConfidenceMetrics:
    """
    Confidence calibration metrics.

    Attributes:
        calibrated_confidence: Final calibrated confidence (0-1)
        raw_average: Simple average of model confidences
        agreement_score: Inter-model agreement (0-1)
        consensus_strength: How strongly models agree (0-1)
        uncertainty: Calibrated uncertainty (0-1)
        variance: Statistical variance in responses
        model_count: Number of models used
        successful_count: Number of successful responses
    """

    calibrated_confidence: float
    raw_average: float
    agreement_score: float
    consensus_strength: float
    uncertainty: float
    variance: float
    model_count: int
    successful_count: int


class ConfidenceCalibrator:
    """
    Calibrates confidence scores from multiple models.

    The calibration process:
    1. Calculate raw average of model confidences
    2. Measure inter-model agreement (consensus)
    3. Calculate statistical variance
    4. Apply calibration function
    5. Compute uncertainty bounds

    Features:
    - Multi-model agreement scoring
    - Statistical variance calculation
    - Consensus detection
    - Uncertainty quantification
    - Calibration curves for different model counts
    """

    def __init__(
        self,
        min_models_for_calibration: int = 2,
        high_agreement_threshold: float = 0.8,
        low_variance_threshold: float = 0.05,
    ):
        """
        Initialize confidence calibrator.

        Args:
            min_models_for_calibration: Minimum models needed for calibration
            high_agreement_threshold: Threshold for "high agreement" (0-1)
            low_variance_threshold: Threshold for "low variance"
        """
        self.min_models = min_models_for_calibration
        self.high_agreement_threshold = high_agreement_threshold
        self.low_variance_threshold = low_variance_threshold

    def calibrate(self, responses: List[ModelResponse]) -> ConfidenceMetrics:
        """
        Calibrate confidence from multiple model responses.

        Args:
            responses: List of model responses

        Returns:
            Calibrated confidence metrics
        """
        if not responses:
            return self._default_metrics(0, 0)

        # Filter successful responses
        successful = [r for r in responses if r.error is None]
        total_count = len(responses)
        success_count = len(successful)

        if success_count == 0:
            return self._default_metrics(total_count, 0)

        # Extract confidence scores
        confidences = [r.confidence for r in successful]

        # Calculate raw average
        raw_avg = sum(confidences) / len(confidences)

        # If only one model, return with penalty
        if success_count == 1:
            return ConfidenceMetrics(
                calibrated_confidence=raw_avg * 0.7,  # Penalty for single model
                raw_average=raw_avg,
                agreement_score=1.0,  # Perfect agreement with self
                consensus_strength=0.5,  # Neutral
                uncertainty=0.3,  # Higher uncertainty
                variance=0.0,
                model_count=total_count,
                successful_count=success_count,
            )

        # Calculate agreement score
        agreement = self._calculate_agreement(confidences)

        # Calculate variance
        variance = self._calculate_variance(confidences)

        # Calculate consensus strength
        consensus = self._calculate_consensus(successful)

        # Apply calibration function
        calibrated = self._apply_calibration(
            raw_avg, agreement, variance, consensus, success_count
        )

        # Calculate uncertainty
        uncertainty = 1.0 - calibrated

        return ConfidenceMetrics(
            calibrated_confidence=calibrated,
            raw_average=raw_avg,
            agreement_score=agreement,
            consensus_strength=consensus,
            uncertainty=uncertainty,
            variance=variance,
            model_count=total_count,
            successful_count=success_count,
        )

    def _calculate_agreement(self, confidences: List[float]) -> float:
        """
        Calculate inter-model agreement score.

        Agreement is high when confidence scores are close together.
        Uses coefficient of variation (std/mean).

        Args:
            confidences: List of confidence scores

        Returns:
            Agreement score (0-1)
        """
        if len(confidences) < 2:
            return 1.0

        mean = sum(confidences) / len(confidences)

        if mean == 0:
            return 0.0

        # Calculate standard deviation
        squared_diffs = [(c - mean) ** 2 for c in confidences]
        variance = sum(squared_diffs) / len(confidences)
        std_dev = math.sqrt(variance)

        # Coefficient of variation
        cv = std_dev / mean if mean > 0 else 1.0

        # Convert to agreement score (inverse of CV, normalized)
        # High CV = low agreement, Low CV = high agreement
        agreement = max(0.0, min(1.0, 1.0 - cv))

        return agreement

    def _calculate_variance(self, confidences: List[float]) -> float:
        """
        Calculate statistical variance in confidence scores.

        Args:
            confidences: List of confidence scores

        Returns:
            Variance (0-1)
        """
        if len(confidences) < 2:
            return 0.0

        mean = sum(confidences) / len(confidences)
        squared_diffs = [(c - mean) ** 2 for c in confidences]
        variance = sum(squared_diffs) / len(confidences)

        return min(1.0, variance)

    def _calculate_consensus(self, responses: List[ModelResponse]) -> float:
        """
        Calculate consensus strength based on response similarity.

        Analyzes actual response content to determine agreement.
        Uses simple length-based similarity as proxy.

        Args:
            responses: List of model responses

        Returns:
            Consensus score (0-1)
        """
        if len(responses) < 2:
            return 0.5

        # Extract response lengths
        lengths = [len(r.content) for r in responses]
        mean_length = sum(lengths) / len(lengths)

        if mean_length == 0:
            return 0.0

        # Calculate coefficient of variation for lengths
        squared_diffs = [(l - mean_length) ** 2 for l in lengths]
        variance = sum(squared_diffs) / len(lengths)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_length if mean_length > 0 else 1.0

        # Low CV in length suggests similar responses
        consensus = max(0.0, min(1.0, 1.0 - cv / 2))  # Divide by 2 to scale

        return consensus

    def _apply_calibration(
        self,
        raw_confidence: float,
        agreement: float,
        variance: float,
        consensus: float,
        model_count: int,
    ) -> float:
        """
        Apply calibration function to compute final confidence.

        Calibration formula combines:
        - Raw confidence (base)
        - Agreement bonus (models agree)
        - Consensus bonus (responses similar)
        - Variance penalty (high variance reduces confidence)
        - Model count bonus (more models = more confidence)

        Args:
            raw_confidence: Average model confidence
            agreement: Inter-model agreement
            variance: Confidence variance
            consensus: Response consensus
            model_count: Number of models

        Returns:
            Calibrated confidence (0-1)
        """
        # Start with raw confidence
        calibrated = raw_confidence

        # Agreement bonus (up to +0.15)
        agreement_bonus = agreement * 0.15
        calibrated += agreement_bonus

        # Consensus bonus (up to +0.1)
        consensus_bonus = consensus * 0.1
        calibrated += consensus_bonus

        # Variance penalty (up to -0.15)
        variance_penalty = variance * 0.15
        calibrated -= variance_penalty

        # Model count bonus (diminishing returns)
        # 2 models: +0.05, 3 models: +0.08, 5 models: +0.11, 10 models: +0.15
        if model_count >= self.min_models:
            model_bonus = min(0.15, math.log(model_count) / 10)
            calibrated += model_bonus

        # High agreement amplification
        if agreement >= self.high_agreement_threshold and variance <= self.low_variance_threshold:
            # Strong consensus deserves confidence boost
            calibrated *= 1.1

        # Ensure bounds [0, 1]
        calibrated = max(0.0, min(1.0, calibrated))

        return calibrated

    def _default_metrics(self, total: int, successful: int) -> ConfidenceMetrics:
        """
        Create default metrics for edge cases.

        Args:
            total: Total model count
            successful: Successful model count

        Returns:
            Default confidence metrics
        """
        return ConfidenceMetrics(
            calibrated_confidence=0.0,
            raw_average=0.0,
            agreement_score=0.0,
            consensus_strength=0.0,
            uncertainty=1.0,
            variance=0.0,
            model_count=total,
            successful_count=successful,
        )

    def get_confidence_interval(
        self, metrics: ConfidenceMetrics, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for the calibrated confidence.

        Uses normal approximation for confidence intervals.

        Args:
            metrics: Confidence metrics
            confidence_level: Desired confidence level (e.g., 0.95 for 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        # Z-scores for common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }

        z = z_scores.get(confidence_level, 1.96)  # Default to 95%

        # Standard error (using variance and sample size)
        if metrics.successful_count < 2:
            # Wide interval for single model
            margin = 0.3
        else:
            std_error = math.sqrt(metrics.variance / metrics.successful_count)
            margin = z * std_error

        lower = max(0.0, metrics.calibrated_confidence - margin)
        upper = min(1.0, metrics.calibrated_confidence + margin)

        return (lower, upper)


def calibrate_confidence(responses: List[ModelResponse]) -> ConfidenceMetrics:
    """
    Convenience function to calibrate confidence with default settings.

    Args:
        responses: List of model responses

    Returns:
        Calibrated confidence metrics
    """
    calibrator = ConfidenceCalibrator()
    return calibrator.calibrate(responses)
