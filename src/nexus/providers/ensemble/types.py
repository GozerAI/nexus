"""
Core types for the unified ensemble system.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4


class QueryType(str, Enum):
    """Query classification types from combo1."""

    FACTUAL = "factual"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QA = "qa"
    FORECASTING = "forecasting"  # From 4cast


class ModelProvider(str, Enum):
    """AI model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    TOGETHER = "together"
    REPLICATE = "replicate"
    OLLAMA = "ollama"
    LOCAL = "local"


@dataclass
class EnsembleRequest:
    """
    Request to the unified ensemble system.

    Attributes:
        query: User query text
        request_id: Unique request identifier
        user_id: Optional user identifier
        conversation_id: Optional conversation identifier
        query_type: Classified query type (auto-detected if None)
        context: Optional conversation context
        max_models: Maximum number of models to query
        temperature: Model temperature setting
        max_tokens: Maximum tokens to generate
        stream: Enable streaming responses
        metadata: Additional request metadata
    """

    query: str
    request_id: UUID = field(default_factory=uuid4)
    user_id: Optional[str] = None
    conversation_id: Optional[UUID] = None
    query_type: Optional[QueryType] = None
    context: Optional[str] = None
    max_models: Optional[int] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    stream: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelResponse:
    """
    Response from a single model.

    Attributes:
        model_name: Name of the model
        provider: Model provider
        content: Generated response text
        confidence: Model's confidence score (0-1)
        latency_ms: Response generation time in milliseconds
        tokens_used: Number of tokens consumed
        cost_usd: Cost in USD
        error: Error message if generation failed
        metadata: Additional response metadata
    """

    model_name: str
    provider: ModelProvider
    content: str
    confidence: float = 0.5
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResponse:
    """
    Final synthesized response from the ensemble system.

    Attributes:
        request_id: Original request identifier
        content: Final synthesized response
        confidence: Overall confidence score (0-1)
        strategy_used: Name of strategy that selected this response
        model_responses: All individual model responses
        models_queried: Number of models queried
        total_latency_ms: Total time to generate response
        total_cost_usd: Total cost across all models
        epistemic_health: Epistemic health score from monitoring
        belief_score: Belief score if knowledge was stored
        metadata: Additional response metadata
        timestamp: Response generation timestamp
    """

    request_id: UUID
    content: str
    confidence: float
    strategy_used: str
    model_responses: List[ModelResponse]
    models_queried: int
    total_latency_ms: float
    total_cost_usd: float
    epistemic_health: Optional[float] = None
    belief_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def models_used(self) -> List[str]:
        """Get list of model names that were queried."""
        return [r.model_name for r in self.model_responses]

    @property
    def successful_models(self) -> List[ModelResponse]:
        """Get list of models that successfully generated responses."""
        return [r for r in self.model_responses if r.error is None]

    @property
    def failed_models(self) -> List[ModelResponse]:
        """Get list of models that failed to generate responses."""
        return [r for r in self.model_responses if r.error is not None]

    @property
    def average_confidence(self) -> float:
        """Calculate average confidence across successful models."""
        successful = self.successful_models
        if not successful:
            return 0.0
        return sum(r.confidence for r in successful) / len(successful)

    @property
    def consensus_score(self) -> float:
        """
        Calculate consensus score (agreement between models).
        Higher score means more agreement.
        """
        # Simplified consensus calculation - can be enhanced with semantic similarity
        successful = self.successful_models
        if len(successful) < 2:
            return 1.0

        # Calculate variance in response lengths (low variance = high consensus)
        lengths = [len(r.content) for r in successful]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
        normalized_variance = min(variance / (mean_length ** 2), 1.0) if mean_length > 0 else 0.0

        return 1.0 - normalized_variance


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for the ensemble system.

    Attributes:
        total_requests: Total number of requests processed
        successful_requests: Number of successful requests
        failed_requests: Number of failed requests
        average_latency_ms: Average response latency
        average_cost_usd: Average cost per request
        average_confidence: Average confidence score
        cache_hit_rate: Cache hit rate percentage
        model_performance: Performance metrics per model
    """

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    average_cost_usd: float = 0.0
    average_confidence: float = 0.0
    cache_hit_rate: float = 0.0
    model_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


@dataclass
class EpistemicHealth:
    """
    Epistemic health metrics from fluffy-eureka monitoring.

    Attributes:
        drift_score: Drift detection score (0-1, higher = more drift)
        consistency_score: Consistency across responses (0-1, higher = more consistent)
        belief_stability: Stability of beliefs over time (0-1)
        contradiction_count: Number of detected contradictions
        knowledge_gaps: Number of identified knowledge gaps
        last_check: Timestamp of last health check
    """

    drift_score: float = 0.0
    consistency_score: float = 1.0
    belief_stability: float = 1.0
    contradiction_count: int = 0
    knowledge_gaps: int = 0
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def overall_health(self) -> float:
        """
        Calculate overall epistemic health score (0-1).
        Higher is better.
        """
        # Weighted average of positive factors
        health = (
            (1.0 - self.drift_score) * 0.3
            + self.consistency_score * 0.3
            + self.belief_stability * 0.2
            + (1.0 - min(self.contradiction_count / 10, 1.0)) * 0.1
            + (1.0 - min(self.knowledge_gaps / 10, 1.0)) * 0.1
        )
        return max(0.0, min(1.0, health))

    @property
    def is_healthy(self) -> bool:
        """Check if system is epistemically healthy (> 0.7)."""
        return self.overall_health > 0.7
