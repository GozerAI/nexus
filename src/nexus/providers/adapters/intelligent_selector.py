"""
Intelligent Model Selector - Adaptive model selection from 1 to 100+ models.

Revolutionary model selection that scales based on task criticality:
- Simple task + casual = 1 model (fast, cheap)
- Complex task + critical = 30+ models (consensus, high quality)

This selector uses:
1. Task analysis (complexity, domain, criticality)
2. Real-world performance metrics (not just specs)
3. Cost-quality frontier optimization
4. Disagreement detection for quality assurance
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from nexus.providers.adapters.base import ModelCapability, ModelInfo, ModelSize
from nexus.providers.adapters.dynamic_registry import (
    DynamicModelRegistry,
    ModelPerformanceMetrics,
    TaskCriticality,
)

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Task complexity levels."""

    TRIVIAL = "trivial"  # One-word answers, simple lookups
    SIMPLE = "simple"  # Straightforward Q&A
    MODERATE = "moderate"  # Requires reasoning
    COMPLEX = "complex"  # Multi-step reasoning
    EXPERT = "expert"  # Specialized domain knowledge


class TaskDomain(str, Enum):
    """Task domain categories."""

    GENERAL = "general"
    CODE = "code"
    MATHEMATICS = "mathematics"
    REASONING = "reasoning"
    CREATIVE = "creative"
    VISION = "vision"
    MULTILINGUAL = "multilingual"
    MEDICAL = "medical"
    LEGAL = "legal"
    SCIENTIFIC = "scientific"


@dataclass
class TaskRequirements:
    """Requirements for a specific task."""

    # Core requirements
    primary_capability: ModelCapability
    secondary_capabilities: List[ModelCapability] = None

    # Task characteristics
    complexity: TaskComplexity = TaskComplexity.MODERATE
    criticality: TaskCriticality = TaskCriticality.STANDARD
    domain: TaskDomain = TaskDomain.GENERAL

    # Constraints
    max_latency_ms: Optional[float] = None  # None = no limit
    max_cost_usd: Optional[float] = None  # None = no limit
    min_confidence: float = 0.65

    # Context
    input_tokens: int = 0
    requires_streaming: bool = False

    def __post_init__(self):
        """Initialize defaults."""
        if self.secondary_capabilities is None:
            self.secondary_capabilities = []


@dataclass
class ModelSelection:
    """Selected models for a task."""

    primary_models: List[str]  # Core models for execution
    fallback_models: List[str]  # Backup if primary fails
    specialist_models: List[str]  # Domain specialists

    total_estimated_cost: float
    total_estimated_latency_ms: float
    expected_confidence: float

    selection_reasoning: str  # Why these models were chosen


class IntelligentModelSelector:
    """
    Revolutionary model selector that adapts from 1 to 100+ models.

    Selection Strategy:
    1. **Casual tasks**: 1-3 fast, cheap models
    2. **Standard tasks**: 3-5 balanced models
    3. **Important tasks**: 7-15 high-quality models
    4. **Critical tasks**: 15-30 diverse models with consensus
    5. **Research tasks**: 30-50+ models for maximum coverage

    Key Innovation: Scales model count based on task criticality, not just complexity.
    """

    def __init__(
        self,
        registry: DynamicModelRegistry,
        enable_disagreement_detection: bool = True,
        enable_specialist_routing: bool = True,
    ):
        """
        Initialize intelligent model selector.

        Args:
            registry: Dynamic model registry
            enable_disagreement_detection: Use disagreement as quality signal
            enable_specialist_routing: Route to domain specialists
        """
        self.registry = registry
        self.enable_disagreement_detection = enable_disagreement_detection
        self.enable_specialist_routing = enable_specialist_routing

        # Base model counts by criticality
        self._base_model_counts = {
            TaskCriticality.CASUAL: 1,
            TaskCriticality.STANDARD: 3,
            TaskCriticality.IMPORTANT: 7,
            TaskCriticality.CRITICAL: 15,
            TaskCriticality.RESEARCH: 30,
        }

        # Complexity multipliers
        self._complexity_multipliers = {
            TaskComplexity.TRIVIAL: 0.5,
            TaskComplexity.SIMPLE: 1.0,
            TaskComplexity.MODERATE: 1.3,
            TaskComplexity.COMPLEX: 1.6,
            TaskComplexity.EXPERT: 2.0,
        }

        logger.info("IntelligentModelSelector initialized")

    async def select_models(
        self,
        requirements: TaskRequirements,
        max_models: Optional[int] = None,
    ) -> ModelSelection:
        """
        Select optimal models for a task.

        Args:
            requirements: Task requirements
            max_models: Optional cap on number of models

        Returns:
            Model selection with primary, fallback, and specialist models
        """
        logger.info(
            f"Selecting models for {requirements.criticality} {requirements.complexity} task"
        )

        # Calculate optimal model count
        optimal_count = self._calculate_optimal_model_count(
            requirements, self.registry.get_total_model_count()
        )

        if max_models:
            optimal_count = min(optimal_count, max_models)

        logger.debug(f"Optimal model count: {optimal_count}")

        # Get candidate models
        candidates = await self._get_candidate_models(requirements)

        if not candidates:
            raise ValueError(
                f"No models available for capability {requirements.primary_capability}"
            )

        logger.debug(f"Found {len(candidates)} candidate models")

        # Rank candidates by suitability
        ranked = await self._rank_models(candidates, requirements)

        # Select models based on ranking
        primary_count = max(1, int(optimal_count * 0.7))  # 70% primary
        fallback_count = max(1, int(optimal_count * 0.2))  # 20% fallback
        specialist_count = int(optimal_count * 0.1)  # 10% specialists

        primary_models = [name for name, _ in ranked[:primary_count]]
        fallback_models = [
            name for name, _ in ranked[primary_count : primary_count + fallback_count]
        ]

        # Get domain specialists if enabled
        specialist_models = []
        if self.enable_specialist_routing and requirements.domain != TaskDomain.GENERAL:
            specialist_models = await self._get_domain_specialists(
                requirements.domain, specialist_count
            )

        # Estimate costs and latency
        total_cost = await self._estimate_total_cost(
            primary_models + fallback_models + specialist_models, requirements
        )
        total_latency = await self._estimate_total_latency(
            primary_models, requirements  # Only primary models run in parallel
        )

        # Estimate confidence
        expected_confidence = self._estimate_confidence(
            primary_models, requirements.criticality
        )

        # Generate reasoning
        reasoning = self._generate_selection_reasoning(
            requirements, len(primary_models), len(fallback_models), len(specialist_models)
        )

        selection = ModelSelection(
            primary_models=primary_models,
            fallback_models=fallback_models,
            specialist_models=specialist_models,
            total_estimated_cost=total_cost,
            total_estimated_latency_ms=total_latency,
            expected_confidence=expected_confidence,
            selection_reasoning=reasoning,
        )

        logger.info(
            f"Selected {len(primary_models)} primary + "
            f"{len(fallback_models)} fallback + "
            f"{len(specialist_models)} specialist models"
        )

        return selection

    def _calculate_optimal_model_count(
        self, requirements: TaskRequirements, available: int
    ) -> int:
        """
        Calculate optimal number of models based on criticality and complexity.

        Simple task + casual = 1 model
        Complex task + critical = 30+ models

        Args:
            requirements: Task requirements
            available: Total available models

        Returns:
            Optimal model count
        """
        # Base count from criticality
        base = self._base_model_counts[requirements.criticality]

        # Apply complexity multiplier
        multiplier = self._complexity_multipliers[requirements.complexity]

        # Calculate optimal count
        optimal = int(base * multiplier)

        # Cap at available models
        optimal = min(optimal, available)

        # Ensure at least 1 model
        optimal = max(1, optimal)

        return optimal

    async def _get_candidate_models(
        self, requirements: TaskRequirements
    ) -> List[str]:
        """
        Get candidate models that meet basic requirements.

        Args:
            requirements: Task requirements

        Returns:
            List of candidate model names
        """
        candidates = []

        # Get all available models
        all_models = self.registry.get_available_models(
            include_unhealthy=False, include_experimental=False
        )

        for model_name in all_models:
            model_info = self.registry.get_model_info(model_name)
            if not model_info:
                continue

            # Check primary capability
            if requirements.primary_capability not in model_info.capabilities:
                continue

            # Check secondary capabilities
            if requirements.secondary_capabilities:
                if not all(
                    cap in model_info.capabilities
                    for cap in requirements.secondary_capabilities
                ):
                    continue

            # Check cost constraint
            if requirements.max_cost_usd is not None:
                estimated_cost = model_info.calculate_cost(
                    requirements.input_tokens, 500  # Assume 500 output tokens
                )
                if estimated_cost > requirements.max_cost_usd:
                    continue

            # Check streaming requirement
            if requirements.requires_streaming and not model_info.supports_streaming:
                continue

            candidates.append(model_name)

        return candidates

    async def _rank_models(
        self, candidates: List[str], requirements: TaskRequirements
    ) -> List[Tuple[str, float]]:
        """
        Rank candidate models by suitability score.

        Scoring factors:
        1. Real-world performance (50%) - learned from usage
        2. Cost efficiency (20%)
        3. Latency (20%)
        4. Capability match (10%)

        Args:
            candidates: Candidate model names
            requirements: Task requirements

        Returns:
            List of (model_name, score) tuples, sorted by score descending
        """
        scored = []

        for model_name in candidates:
            model_info = self.registry.get_model_info(model_name)
            if not model_info:
                continue

            # Get performance metrics
            metrics = self.registry.get_performance_metrics(model_name)

            # Calculate score components
            performance_score = self._calculate_performance_score(metrics)
            cost_score = self._calculate_cost_score(model_info, requirements)
            latency_score = self._calculate_latency_score(metrics, requirements)
            capability_score = self._calculate_capability_score(model_info, requirements)

            # Weighted total score
            total_score = (
                performance_score * 0.5
                + cost_score * 0.2
                + latency_score * 0.2
                + capability_score * 0.1
            )

            scored.append((model_name, total_score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _calculate_performance_score(
        self, metrics: Optional[ModelPerformanceMetrics]
    ) -> float:
        """
        Calculate performance score from real-world metrics.

        Args:
            metrics: Performance metrics

        Returns:
            Score from 0-1
        """
        if not metrics or metrics.total_calls == 0:
            return 0.5  # Neutral score for unproven models

        # Combine multiple quality signals
        confidence_score = metrics.avg_confidence if metrics.avg_confidence > 0 else 0.5
        success_rate = (
            metrics.successful_calls / metrics.total_calls if metrics.total_calls > 0 else 0.0
        )
        user_rating_score = (
            (metrics.avg_user_rating + 1) / 2  # Convert -1 to 1 range to 0-1
            if metrics.avg_user_rating != 0
            else 0.5
        )

        # Weighted average
        score = confidence_score * 0.4 + success_rate * 0.4 + user_rating_score * 0.2

        return score

    def _calculate_cost_score(
        self, model_info: ModelInfo, requirements: TaskRequirements
    ) -> float:
        """
        Calculate cost efficiency score.

        Lower cost = higher score.

        Args:
            model_info: Model information
            requirements: Task requirements

        Returns:
            Score from 0-1
        """
        estimated_cost = model_info.calculate_cost(requirements.input_tokens, 500)

        if requirements.max_cost_usd is None:
            # No budget constraint - use relative scoring
            # Assume $1 as reference point
            return 1.0 - min(estimated_cost, 1.0)

        # Score based on budget utilization
        if estimated_cost >= requirements.max_cost_usd:
            return 0.0

        return 1.0 - (estimated_cost / requirements.max_cost_usd)

    def _calculate_latency_score(
        self,
        metrics: Optional[ModelPerformanceMetrics],
        requirements: TaskRequirements,
    ) -> float:
        """
        Calculate latency score.

        Lower latency = higher score.

        Args:
            metrics: Performance metrics
            requirements: Task requirements

        Returns:
            Score from 0-1
        """
        if not metrics or metrics.avg_latency_ms == 0:
            return 0.5  # Neutral for unknown latency

        latency = metrics.avg_latency_ms

        if requirements.max_latency_ms is None:
            # No latency constraint - use relative scoring
            # Assume 5000ms (5s) as reference point
            return 1.0 - min(latency / 5000, 1.0)

        # Score based on latency constraint
        if latency >= requirements.max_latency_ms:
            return 0.0

        return 1.0 - (latency / requirements.max_latency_ms)

    def _calculate_capability_score(
        self, model_info: ModelInfo, requirements: TaskRequirements
    ) -> float:
        """
        Calculate capability match score.

        Args:
            model_info: Model information
            requirements: Task requirements

        Returns:
            Score from 0-1
        """
        # Count capability matches
        required_caps = [requirements.primary_capability] + requirements.secondary_capabilities
        matched = sum(1 for cap in required_caps if cap in model_info.capabilities)

        if not required_caps:
            return 1.0

        return matched / len(required_caps)

    async def _get_domain_specialists(
        self, domain: TaskDomain, count: int
    ) -> List[str]:
        """
        Get specialist models for a specific domain.

        Args:
            domain: Task domain
            count: Number of specialists to retrieve

        Returns:
            List of specialist model names
        """
        # Domain-specific model mapping
        domain_specialists = {
            TaskDomain.CODE: ["deepseek-coder-33b", "phind-codellama-34b", "starcoder2-15b"],
            TaskDomain.MATHEMATICS: ["gpt-4-code-interpreter"],
            TaskDomain.MEDICAL: ["meditron-70b", "biomistral-7b"],
            TaskDomain.MULTILINGUAL: ["aya-101", "qwen-72b"],
        }

        specialists = domain_specialists.get(domain, [])

        # Filter to available models
        available = self.registry.get_available_models()
        specialists = [s for s in specialists if s in available]

        return specialists[:count]

    async def _estimate_total_cost(
        self, models: List[str], requirements: TaskRequirements
    ) -> float:
        """
        Estimate total cost for running all selected models.

        Args:
            models: Selected model names
            requirements: Task requirements

        Returns:
            Estimated total cost in USD
        """
        total = 0.0

        for model_name in models:
            model_info = self.registry.get_model_info(model_name)
            if model_info:
                cost = model_info.calculate_cost(requirements.input_tokens, 500)
                total += cost

        return total

    async def _estimate_total_latency(
        self, primary_models: List[str], requirements: TaskRequirements
    ) -> float:
        """
        Estimate total latency (assuming parallel execution).

        Args:
            primary_models: Primary models (run in parallel)
            requirements: Task requirements

        Returns:
            Estimated latency in milliseconds
        """
        # Parallel execution - take slowest model
        max_latency = 0.0

        for model_name in primary_models:
            metrics = self.registry.get_performance_metrics(model_name)
            if metrics and metrics.avg_latency_ms > 0:
                max_latency = max(max_latency, metrics.avg_latency_ms)

        # If no historical data, estimate based on model size
        if max_latency == 0.0:
            max_latency = 2000.0  # Default 2 seconds

        return max_latency

    def _estimate_confidence(
        self, models: List[str], criticality: TaskCriticality
    ) -> float:
        """
        Estimate expected confidence with selected models.

        More models = higher confidence (ensemble effect).

        Args:
            models: Selected model names
            criticality: Task criticality

        Returns:
            Expected confidence (0-1)
        """
        if not models:
            return 0.0

        # Base confidence from historical performance
        avg_confidence = 0.0
        count = 0

        for model_name in models:
            metrics = self.registry.get_performance_metrics(model_name)
            if metrics and metrics.avg_confidence > 0:
                avg_confidence += metrics.avg_confidence
                count += 1

        if count > 0:
            base_confidence = avg_confidence / count
        else:
            base_confidence = 0.7  # Default

        # Ensemble boost: more models = higher confidence
        ensemble_boost = min(len(models) / 10, 0.2)  # Up to +0.2

        total_confidence = min(base_confidence + ensemble_boost, 1.0)

        return total_confidence

    def _generate_selection_reasoning(
        self,
        requirements: TaskRequirements,
        primary_count: int,
        fallback_count: int,
        specialist_count: int,
    ) -> str:
        """
        Generate human-readable reasoning for model selection.

        Args:
            requirements: Task requirements
            primary_count: Number of primary models
            fallback_count: Number of fallback models
            specialist_count: Number of specialist models

        Returns:
            Reasoning string
        """
        reasoning_parts = []

        # Criticality reasoning
        reasoning_parts.append(
            f"Task criticality: {requirements.criticality.value} "
            f"({self._base_model_counts[requirements.criticality]} base models)"
        )

        # Complexity adjustment
        multiplier = self._complexity_multipliers[requirements.complexity]
        if multiplier != 1.0:
            reasoning_parts.append(
                f"Complexity: {requirements.complexity.value} "
                f"(x{multiplier} multiplier)"
            )

        # Model breakdown
        reasoning_parts.append(
            f"Selected: {primary_count} primary + "
            f"{fallback_count} fallback + "
            f"{specialist_count} specialist"
        )

        # Domain specialization
        if requirements.domain != TaskDomain.GENERAL:
            reasoning_parts.append(f"Domain: {requirements.domain.value} (specialists added)")

        return " | ".join(reasoning_parts)
