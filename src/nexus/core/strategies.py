"""
Advanced ensemble strategies for model selection and orchestration.

This module provides sophisticated strategies for combining model outputs,
including weighted voting, cascading inference, dynamic weight adjustment,
and cost-optimized selection.
"""

import logging
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformance:
    """Track historical performance of a model."""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    average_score: float = 0.0
    average_latency_ms: float = 0.0
    average_cost: float = 0.0
    success_rate: float = 1.0


@dataclass
class EnsembleResult:
    """Result from ensemble inference."""
    content: str
    model_name: str
    provider: str
    score: float
    confidence: float
    strategy_used: str
    models_queried: int
    total_cost: float
    total_latency_ms: float
    metadata: Optional[Dict[str, Any]] = None


class EnsembleStrategy(ABC):
    """Base class for ensemble strategies."""

    def __init__(self, name: str):
        """
        Initialize strategy.

        Args:
            name: Strategy name
        """
        self.name = name
        self.performance_history: Dict[str, ModelPerformance] = {}
        logger.info(f"Initialized {name} strategy")

    @abstractmethod
    async def select_response(
        self,
        responses: List[Tuple[float, Any]],
        prompt: str
    ) -> EnsembleResult:
        """
        Select the best response from model outputs.

        Args:
            responses: List of (score, model_response) tuples
            prompt: Original prompt

        Returns:
            EnsembleResult with selected response
        """
        pass

    def update_performance(self, model_name: str, score: float, latency_ms: float, cost: float, success: bool):
        """
        Update performance history for a model.

        Args:
            model_name: Name of the model
            score: Response quality score
            latency_ms: Response latency in milliseconds
            cost: Cost in USD
            success: Whether request succeeded
        """
        if model_name not in self.performance_history:
            self.performance_history[model_name] = ModelPerformance(model_name=model_name)

        perf = self.performance_history[model_name]
        perf.total_requests += 1

        if success:
            perf.successful_requests += 1

        # Update running averages
        perf.success_rate = perf.successful_requests / perf.total_requests
        perf.average_score = (perf.average_score * (perf.total_requests - 1) + score) / perf.total_requests
        perf.average_latency_ms = (perf.average_latency_ms * (perf.total_requests - 1) + latency_ms) / perf.total_requests
        perf.average_cost = (perf.average_cost * (perf.total_requests - 1) + cost) / perf.total_requests


class WeightedVotingStrategy(EnsembleStrategy):
    """
    Weighted voting strategy using model weights and quality scores.

    Combines model weights with response quality scores to select the best response.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize weighted voting strategy.

        Args:
            weights: Optional dictionary of model_name -> weight
        """
        super().__init__("weighted_voting")
        self.weights = weights or {}

    async def select_response(
        self,
        responses: List[Tuple[float, Any]],
        prompt: str
    ) -> EnsembleResult:
        """Select response using weighted voting."""
        if not responses:
            raise ValueError("No responses to select from")

        weighted_scores = []
        total_cost = 0.0
        total_latency = 0.0

        for score, model_response in responses:
            model_name = model_response.model_name
            weight = self.weights.get(model_name, 1.0)

            # Weighted score = quality_score * model_weight
            weighted_score = score * weight

            weighted_scores.append((weighted_score, score, model_response))
            total_cost += model_response.cost
            total_latency += model_response.latency_ms

            logger.debug(
                f"{model_name}: score={score:.3f}, weight={weight:.2f}, "
                f"weighted_score={weighted_score:.3f}"
            )

        # Sort by weighted score
        weighted_scores.sort(reverse=True, key=lambda x: x[0])
        best_weighted_score, best_score, best_response = weighted_scores[0]

        logger.info(
            f"Selected {best_response.model_name} with weighted score {best_weighted_score:.3f}"
        )

        return EnsembleResult(
            content=best_response.content,
            model_name=best_response.model_name,
            provider=best_response.provider,
            score=best_score,
            confidence=best_weighted_score / max(w for w in self.weights.values() or [1.0]),
            strategy_used=self.name,
            models_queried=len(responses),
            total_cost=total_cost,
            total_latency_ms=total_latency,
            metadata={
                "weights": {r[1].model_name: self.weights.get(r[1].model_name, 1.0) for r in responses},
                "weighted_scores": {r[2].model_name: r[0] for r in weighted_scores}
            }
        )


class CascadingStrategy(EnsembleStrategy):
    """
    Cascading strategy that tries cheaper models first.

    Starts with low-cost models and escalates to expensive ones only if
    confidence is below threshold.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_cascades: int = 3
    ):
        """
        Initialize cascading strategy.

        Args:
            confidence_threshold: Minimum confidence to stop cascading
            max_cascades: Maximum number of models to try
        """
        super().__init__("cascading")
        self.confidence_threshold = confidence_threshold
        self.max_cascades = max_cascades

    async def select_response(
        self,
        responses: List[Tuple[float, Any]],
        prompt: str
    ) -> EnsembleResult:
        """Select response using cascading logic."""
        if not responses:
            raise ValueError("No responses to select from")

        # Sort by cost (ascending)
        sorted_by_cost = sorted(responses, key=lambda x: x[1].cost)

        total_cost = 0.0
        total_latency = 0.0
        models_tried = 0

        for score, model_response in sorted_by_cost[:self.max_cascades]:
            models_tried += 1
            total_cost += model_response.cost
            total_latency += model_response.latency_ms

            logger.debug(
                f"Cascade level {models_tried}: {model_response.model_name} "
                f"(score={score:.3f}, cost=${model_response.cost:.4f})"
            )

            # Check if confidence is high enough
            if score >= self.confidence_threshold or models_tried >= self.max_cascades:
                logger.info(
                    f"Cascading stopped at {model_response.model_name} "
                    f"(score={score:.3f}, models_tried={models_tried})"
                )

                return EnsembleResult(
                    content=model_response.content,
                    model_name=model_response.model_name,
                    provider=model_response.provider,
                    score=score,
                    confidence=score,
                    strategy_used=self.name,
                    models_queried=models_tried,
                    total_cost=total_cost,
                    total_latency_ms=total_latency,
                    metadata={
                        "cascade_level": models_tried,
                        "threshold": self.confidence_threshold
                    }
                )

        # If we get here, use the best score we found
        best_score, best_response = max(sorted_by_cost[:self.max_cascades], key=lambda x: x[0])

        return EnsembleResult(
            content=best_response.content,
            model_name=best_response.model_name,
            provider=best_response.provider,
            score=best_score,
            confidence=best_score,
            strategy_used=self.name,
            models_queried=models_tried,
            total_cost=total_cost,
            total_latency_ms=total_latency,
            metadata={
                "cascade_level": models_tried,
                "threshold": self.confidence_threshold,
                "max_reached": True
            }
        )


class DynamicWeightStrategy(EnsembleStrategy):
    """
    Dynamic weight adjustment based on historical performance.

    Adjusts model weights based on success rate, average scores, and latency.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        score_weight: float = 0.5,
        speed_weight: float = 0.3,
        cost_weight: float = 0.2
    ):
        """
        Initialize dynamic weight strategy.

        Args:
            learning_rate: How quickly weights adapt (0-1)
            score_weight: Weight for quality scores
            speed_weight: Weight for latency
            cost_weight: Weight for cost
        """
        super().__init__("dynamic_weight")
        self.learning_rate = learning_rate
        self.score_weight = score_weight
        self.speed_weight = speed_weight
        self.cost_weight = cost_weight
        self.model_weights: Dict[str, float] = {}

    def _calculate_dynamic_weight(self, perf: ModelPerformance) -> float:
        """Calculate dynamic weight based on performance history."""
        if perf.total_requests == 0:
            return 1.0

        # Normalize metrics (higher is better for all)
        score_metric = perf.average_score
        speed_metric = 1.0 / (perf.average_latency_ms / 1000.0 + 0.1)  # Inverse latency
        cost_metric = 1.0 / (perf.average_cost + 0.001)  # Inverse cost

        # Weighted combination
        weight = (
            self.score_weight * score_metric +
            self.speed_weight * speed_metric +
            self.cost_weight * cost_metric
        ) * perf.success_rate

        return max(0.1, min(10.0, weight))  # Clamp between 0.1 and 10.0

    async def select_response(
        self,
        responses: List[Tuple[float, Any]],
        prompt: str
    ) -> EnsembleResult:
        """Select response using dynamic weights."""
        if not responses:
            raise ValueError("No responses to select from")

        weighted_scores = []
        total_cost = 0.0
        total_latency = 0.0

        for score, model_response in responses:
            model_name = model_response.model_name

            # Get dynamic weight based on history
            if model_name in self.performance_history:
                perf = self.performance_history[model_name]
                weight = self._calculate_dynamic_weight(perf)
            else:
                weight = 1.0

            # Store current weight
            self.model_weights[model_name] = weight

            weighted_score = score * weight
            weighted_scores.append((weighted_score, score, model_response, weight))
            total_cost += model_response.cost
            total_latency += model_response.latency_ms

            logger.debug(
                f"{model_name}: score={score:.3f}, dynamic_weight={weight:.3f}, "
                f"weighted_score={weighted_score:.3f}"
            )

        # Sort by weighted score
        weighted_scores.sort(reverse=True, key=lambda x: x[0])
        best_weighted_score, best_score, best_response, best_weight = weighted_scores[0]

        logger.info(
            f"Selected {best_response.model_name} with dynamic weight {best_weight:.3f}"
        )

        return EnsembleResult(
            content=best_response.content,
            model_name=best_response.model_name,
            provider=best_response.provider,
            score=best_score,
            confidence=best_weighted_score / max(w[3] for w in weighted_scores),
            strategy_used=self.name,
            models_queried=len(responses),
            total_cost=total_cost,
            total_latency_ms=total_latency,
            metadata={
                "dynamic_weights": {r[2].model_name: r[3] for r in weighted_scores},
                "performance_history": {
                    name: {
                        "success_rate": perf.success_rate,
                        "avg_score": perf.average_score,
                        "requests": perf.total_requests
                    }
                    for name, perf in self.performance_history.items()
                }
            }
        )


class MajorityVotingStrategy(EnsembleStrategy):
    """
    Majority voting strategy for classification/simple tasks.

    Groups similar responses and selects the most common one.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize majority voting strategy.

        Args:
            similarity_threshold: Threshold for considering responses similar
        """
        super().__init__("majority_voting")
        self.similarity_threshold = similarity_threshold

    def _responses_similar(self, resp1: str, resp2: str) -> bool:
        """Check if two responses are similar."""
        # Simple similarity check - could be enhanced with semantic similarity
        resp1_normalized = resp1.lower().strip()
        resp2_normalized = resp2.lower().strip()

        if resp1_normalized == resp2_normalized:
            return True

        # Check overlap
        words1 = set(resp1_normalized.split())
        words2 = set(resp2_normalized.split())

        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        total = len(words1 | words2)

        return (overlap / total) >= self.similarity_threshold

    async def select_response(
        self,
        responses: List[Tuple[float, Any]],
        prompt: str
    ) -> EnsembleResult:
        """Select response using majority voting."""
        if not responses:
            raise ValueError("No responses to select from")

        # Group similar responses
        response_groups: List[List[Tuple[float, Any]]] = []

        for score, model_response in responses:
            added_to_group = False

            for group in response_groups:
                # Check if similar to any in group
                if self._responses_similar(model_response.content, group[0][1].content):
                    group.append((score, model_response))
                    added_to_group = True
                    break

            if not added_to_group:
                response_groups.append([(score, model_response)])

        # Find largest group (majority)
        largest_group = max(response_groups, key=len)
        group_size = len(largest_group)

        # Within the majority group, select highest score
        best_score, best_response = max(largest_group, key=lambda x: x[0])

        total_cost = sum(r[1].cost for r in responses)
        total_latency = sum(r[1].latency_ms for r in responses)

        logger.info(
            f"Majority voting: {group_size}/{len(responses)} models agreed. "
            f"Selected {best_response.model_name}"
        )

        return EnsembleResult(
            content=best_response.content,
            model_name=best_response.model_name,
            provider=best_response.provider,
            score=best_score,
            confidence=group_size / len(responses),
            strategy_used=self.name,
            models_queried=len(responses),
            total_cost=total_cost,
            total_latency_ms=total_latency,
            metadata={
                "majority_size": group_size,
                "total_groups": len(response_groups),
                "agreement_rate": group_size / len(responses)
            }
        )


class CostOptimizedStrategy(EnsembleStrategy):
    """
    Cost-optimized strategy balancing quality and cost.

    Selects the best quality-to-cost ratio within acceptable quality threshold.
    """

    def __init__(
        self,
        min_quality_threshold: float = 0.6,
        cost_weight: float = 0.4,
        quality_weight: float = 0.6
    ):
        """
        Initialize cost-optimized strategy.

        Args:
            min_quality_threshold: Minimum acceptable quality score
            cost_weight: Weight for cost in optimization
            quality_weight: Weight for quality in optimization
        """
        super().__init__("cost_optimized")
        self.min_quality_threshold = min_quality_threshold
        self.cost_weight = cost_weight
        self.quality_weight = quality_weight

    async def select_response(
        self,
        responses: List[Tuple[float, Any]],
        prompt: str
    ) -> EnsembleResult:
        """Select response optimizing for quality/cost ratio."""
        if not responses:
            raise ValueError("No responses to select from")

        # Filter by minimum quality
        qualified_responses = [
            (score, resp) for score, resp in responses
            if score >= self.min_quality_threshold
        ]

        if not qualified_responses:
            # If none meet threshold, just pick best quality
            logger.warning(
                f"No responses met quality threshold {self.min_quality_threshold}, "
                "selecting best available"
            )
            qualified_responses = responses

        # Calculate quality/cost ratio
        scored_responses = []
        total_cost = sum(r[1].cost for r in responses)
        total_latency = sum(r[1].latency_ms for r in responses)

        for score, model_response in qualified_responses:
            # Normalize quality (0-1) and cost (inverse, 0-1)
            quality_score = score
            cost_score = 1.0 / (model_response.cost + 0.001)  # Inverse cost

            # Combined score
            combined_score = (
                self.quality_weight * quality_score +
                self.cost_weight * cost_score
            )

            scored_responses.append((combined_score, score, model_response))

            logger.debug(
                f"{model_response.model_name}: quality={score:.3f}, "
                f"cost=${model_response.cost:.4f}, combined={combined_score:.3f}"
            )

        # Select best combined score
        best_combined, best_quality, best_response = max(scored_responses, key=lambda x: x[0])

        logger.info(
            f"Cost-optimized selection: {best_response.model_name} "
            f"(quality={best_quality:.3f}, cost=${best_response.cost:.4f})"
        )

        return EnsembleResult(
            content=best_response.content,
            model_name=best_response.model_name,
            provider=best_response.provider,
            score=best_quality,
            confidence=best_quality,
            strategy_used=self.name,
            models_queried=len(responses),
            total_cost=total_cost,
            total_latency_ms=total_latency,
            metadata={
                "quality_threshold": self.min_quality_threshold,
                "cost_saved": total_cost - best_response.cost,
                "quality_cost_ratio": best_quality / (best_response.cost + 0.001)
            }
        )
