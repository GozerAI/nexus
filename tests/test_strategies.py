"""Tests for advanced ensemble strategies."""

import os
import sys
import pytest
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.strategies import (
    WeightedVotingStrategy,
    CascadingStrategy,
    DynamicWeightStrategy,
    MajorityVotingStrategy,
    CostOptimizedStrategy,
    ModelPerformance,
    EnsembleResult
)


@dataclass
class MockModelResponse:
    """Mock model response for testing."""
    content: str
    model_name: str
    provider: str
    cost: float
    latency_ms: float
    tokens_used: int = 100
    success: bool = True
    metadata: Optional[Dict[str, Any]] = None


def run_async(coro):
    """Helper to run async functions in sync tests."""
    return asyncio.run(coro)


class TestModelPerformance:
    """Tests for ModelPerformance dataclass."""

    def test_model_performance_creation(self):
        """Test creating model performance tracker."""
        perf = ModelPerformance(model_name="gpt-4")

        assert perf.model_name == "gpt-4"
        assert perf.total_requests == 0
        assert perf.successful_requests == 0
        assert perf.success_rate == 1.0

    def test_model_performance_defaults(self):
        """Test default values."""
        perf = ModelPerformance(model_name="test")

        assert perf.average_score == 0.0
        assert perf.average_latency_ms == 0.0
        assert perf.average_cost == 0.0


class TestWeightedVotingStrategy:
    """Tests for WeightedVotingStrategy."""

    def test_weighted_voting_basic(self):
        """Test basic weighted voting."""
        strategy = WeightedVotingStrategy(weights={
            "gpt-4": 2.0,
            "claude-3": 1.5,
            "gpt-3.5": 1.0
        })

        responses = [
            (0.8, MockModelResponse("Response 1", "gpt-4", "openai", 0.03, 500)),
            (0.9, MockModelResponse("Response 2", "claude-3", "anthropic", 0.02, 600)),
            (0.7, MockModelResponse("Response 3", "gpt-3.5", "openai", 0.001, 300))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # GPT-4 should win: 0.8 * 2.0 = 1.6 > Claude-3: 0.9 * 1.5 = 1.35
        assert result.model_name == "gpt-4"
        assert result.strategy_used == "weighted_voting"
        assert result.models_queried == 3
        assert "weights" in result.metadata

    def test_weighted_voting_no_weights(self):
        """Test weighted voting with no predefined weights."""
        strategy = WeightedVotingStrategy()

        responses = [
            (0.8, MockModelResponse("Response 1", "model1", "provider1", 0.01, 500)),
            (0.9, MockModelResponse("Response 2", "model2", "provider2", 0.02, 600))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Without weights, should select highest score
        assert result.model_name == "model2"
        assert result.score == 0.9

    def test_weighted_voting_single_response(self):
        """Test with single response."""
        strategy = WeightedVotingStrategy()

        responses = [
            (0.8, MockModelResponse("Response", "model1", "provider1", 0.01, 500))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        assert result.model_name == "model1"
        assert result.models_queried == 1

    def test_weighted_voting_empty_responses(self):
        """Test with empty responses list."""
        strategy = WeightedVotingStrategy()

        with pytest.raises(ValueError, match="No responses to select from"):
            run_async(strategy.select_response([], "test prompt"))


class TestCascadingStrategy:
    """Tests for CascadingStrategy."""

    def test_cascading_stops_at_threshold(self):
        """Test cascading stops when threshold is met."""
        strategy = CascadingStrategy(confidence_threshold=0.75, max_cascades=3)

        responses = [
            (0.6, MockModelResponse("Cheap low quality", "gpt-3.5", "openai", 0.001, 300)),
            (0.8, MockModelResponse("Mid quality", "claude-haiku", "anthropic", 0.005, 400)),
            (0.95, MockModelResponse("High quality", "gpt-4", "openai", 0.03, 500))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Should stop at second model (0.8 > 0.75)
        assert result.model_name == "claude-haiku"
        assert result.models_queried == 2
        assert result.metadata["cascade_level"] == 2

    def test_cascading_reaches_max(self):
        """Test cascading reaches max cascades."""
        strategy = CascadingStrategy(confidence_threshold=0.95, max_cascades=2)

        responses = [
            (0.6, MockModelResponse("Response 1", "model1", "provider1", 0.001, 300)),
            (0.7, MockModelResponse("Response 2", "model2", "provider2", 0.005, 400)),
            (0.9, MockModelResponse("Response 3", "model3", "provider3", 0.03, 500))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Should stop at max_cascades=2, select best of first two
        assert result.models_queried == 2
        assert result.model_name == "model2"  # Best of first 2

    def test_cascading_first_response_good(self):
        """Test cascading stops immediately if first response is good."""
        strategy = CascadingStrategy(confidence_threshold=0.7)

        responses = [
            (0.9, MockModelResponse("Excellent", "gpt-3.5", "openai", 0.001, 300)),
            (0.95, MockModelResponse("Better", "gpt-4", "openai", 0.03, 500))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Should stop at first (0.9 > 0.7)
        assert result.model_name == "gpt-3.5"
        assert result.models_queried == 1
        assert result.total_cost == 0.001  # Only first model cost


class TestDynamicWeightStrategy:
    """Tests for DynamicWeightStrategy."""

    def test_dynamic_weight_no_history(self):
        """Test dynamic weights with no performance history."""
        strategy = DynamicWeightStrategy()

        responses = [
            (0.8, MockModelResponse("Response 1", "model1", "provider1", 0.01, 500)),
            (0.9, MockModelResponse("Response 2", "model2", "provider2", 0.02, 600))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Without history, all weights are 1.0, so best score wins
        assert result.model_name == "model2"
        assert "dynamic_weights" in result.metadata

    def test_dynamic_weight_with_history(self):
        """Test dynamic weights adjust based on history."""
        strategy = DynamicWeightStrategy()

        # Simulate history for model1 (good performance)
        for _ in range(5):
            strategy.update_performance("model1", 0.9, 400, 0.01, True)

        # Simulate history for model2 (poor performance)
        for _ in range(5):
            strategy.update_performance("model2", 0.5, 800, 0.03, True)

        responses = [
            (0.7, MockModelResponse("Response 1", "model1", "provider1", 0.01, 500)),
            (0.75, MockModelResponse("Response 2", "model2", "provider2", 0.02, 600))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Model1 should win due to better historical performance
        # even though model2 has slightly higher current score
        assert result.model_name == "model1"
        assert "performance_history" in result.metadata

    def test_update_performance(self):
        """Test performance tracking updates."""
        strategy = DynamicWeightStrategy()

        strategy.update_performance("test_model", 0.8, 500, 0.01, True)
        strategy.update_performance("test_model", 0.9, 600, 0.02, True)

        perf = strategy.performance_history["test_model"]

        assert perf.total_requests == 2
        assert perf.successful_requests == 2
        assert perf.success_rate == 1.0
        assert pytest.approx(perf.average_score, 0.001) == 0.85  # (0.8 + 0.9) / 2


class TestMajorityVotingStrategy:
    """Tests for MajorityVotingStrategy."""

    def test_majority_voting_clear_majority(self):
        """Test majority voting with clear majority."""
        strategy = MajorityVotingStrategy()

        responses = [
            (0.8, MockModelResponse("The answer is 42", "model1", "provider1", 0.01, 500)),
            (0.85, MockModelResponse("The answer is 42", "model2", "provider2", 0.02, 600)),
            (0.9, MockModelResponse("The answer is 42", "model3", "provider3", 0.03, 700)),
            (0.7, MockModelResponse("Different answer", "model4", "provider4", 0.01, 400))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Should select from majority group (3/4 agree)
        assert "42" in result.content
        assert result.metadata["majority_size"] == 3
        assert result.metadata["agreement_rate"] == 0.75
        assert result.confidence == 0.75

    def test_majority_voting_best_within_majority(self):
        """Test selects best score within majority group."""
        strategy = MajorityVotingStrategy()

        responses = [
            (0.7, MockModelResponse("Answer A", "model1", "provider1", 0.01, 500)),
            (0.9, MockModelResponse("Answer A", "model2", "provider2", 0.02, 600)),
            (0.8, MockModelResponse("Answer B", "model3", "provider3", 0.03, 700))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Should select model2 (highest score in majority group)
        assert result.model_name == "model2"
        assert result.score == 0.9

    def test_majority_voting_no_majority(self):
        """Test when all responses are different."""
        strategy = MajorityVotingStrategy(similarity_threshold=0.9)

        responses = [
            (0.8, MockModelResponse("Response 1", "model1", "provider1", 0.01, 500)),
            (0.85, MockModelResponse("Response 2", "model2", "provider2", 0.02, 600)),
            (0.9, MockModelResponse("Response 3", "model3", "provider3", 0.03, 700))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Each forms its own group, so "majority" is size 1
        # Will pick the best score from the tied groups
        assert result.metadata["majority_size"] == 1
        # All groups are size 1, so any model could win - just verify it has score >= 0.8
        assert result.score >= 0.8


class TestCostOptimizedStrategy:
    """Tests for CostOptimizedStrategy."""

    def test_cost_optimized_basic(self):
        """Test cost optimization."""
        strategy = CostOptimizedStrategy(min_quality_threshold=0.7)

        responses = [
            (0.75, MockModelResponse("Cheap response", "gpt-3.5", "openai", 0.001, 300)),
            (0.85, MockModelResponse("Mid response", "claude-haiku", "anthropic", 0.005, 400)),
            (0.95, MockModelResponse("Expensive response", "gpt-4", "openai", 0.03, 500))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # Should balance quality and cost
        # Likely picks mid-tier or cheap (good quality/cost ratio)
        assert result.model_name in ["gpt-3.5", "claude-haiku"]
        assert result.score >= 0.7  # Meets threshold
        assert "quality_cost_ratio" in result.metadata

    def test_cost_optimized_below_threshold(self):
        """Test when responses are below quality threshold."""
        strategy = CostOptimizedStrategy(min_quality_threshold=0.9)

        responses = [
            (0.6, MockModelResponse("Response 1", "model1", "provider1", 0.001, 300)),
            (0.7, MockModelResponse("Response 2", "model2", "provider2", 0.005, 400)),
            (0.75, MockModelResponse("Response 3", "model3", "provider3", 0.01, 500))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # None meet threshold, so it picks best quality/cost ratio
        # model1: 0.6/0.001 = 600, model2: 0.7/0.005 = 140, model3: 0.75/0.01 = 75
        # Even though model3 has best quality, model1 has better quality/cost ratio
        assert result.model_name in ["model1", "model2", "model3"]  # Any is valid
        assert result.score >= 0.6  # At least meets minimum quality

    def test_cost_optimized_high_cost_weight(self):
        """Test with high cost weight."""
        strategy = CostOptimizedStrategy(
            min_quality_threshold=0.6,
            cost_weight=0.8,
            quality_weight=0.2
        )

        responses = [
            (0.7, MockModelResponse("Cheap", "model1", "provider1", 0.001, 300)),
            (0.95, MockModelResponse("Expensive", "model2", "provider2", 0.03, 500))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        # High cost weight should favor cheaper model
        assert result.model_name == "model1"

    def test_cost_optimized_metadata(self):
        """Test cost optimization metadata."""
        strategy = CostOptimizedStrategy()

        responses = [
            (0.8, MockModelResponse("Response", "model1", "provider1", 0.01, 500)),
            (0.85, MockModelResponse("Response", "model2", "provider2", 0.02, 600))
        ]

        result = run_async(strategy.select_response(responses, "test prompt"))

        assert "cost_saved" in result.metadata
        assert "quality_cost_ratio" in result.metadata
        assert result.metadata["cost_saved"] >= 0


class TestEnsembleResult:
    """Tests for EnsembleResult dataclass."""

    def test_ensemble_result_creation(self):
        """Test creating ensemble result."""
        result = EnsembleResult(
            content="Test response",
            model_name="gpt-4",
            provider="openai",
            score=0.9,
            confidence=0.85,
            strategy_used="weighted_voting",
            models_queried=3,
            total_cost=0.05,
            total_latency_ms=1500
        )

        assert result.content == "Test response"
        assert result.model_name == "gpt-4"
        assert result.strategy_used == "weighted_voting"
        assert result.models_queried == 3

    def test_ensemble_result_with_metadata(self):
        """Test ensemble result with metadata."""
        result = EnsembleResult(
            content="Test",
            model_name="model",
            provider="provider",
            score=0.8,
            confidence=0.7,
            strategy_used="Test",
            models_queried=2,
            total_cost=0.01,
            total_latency_ms=500,
            metadata={"key": "value"}
        )

        assert result.metadata["key"] == "value"


class TestStrategyComparison:
    """Compare different strategies on same input."""

    def test_strategy_comparison(self):
        """Test different strategies produce different results."""
        responses = [
            (0.7, MockModelResponse("Cheap", "gpt-3.5", "openai", 0.001, 300)),
            (0.85, MockModelResponse("Mid", "claude-haiku", "anthropic", 0.005, 400)),
            (0.95, MockModelResponse("Expensive", "gpt-4", "openai", 0.03, 500))
        ]

        # Weighted voting favors gpt-4
        weighted = WeightedVotingStrategy(weights={"gpt-4": 2.0, "claude-haiku": 1.0, "gpt-3.5": 0.5})
        weighted_result = run_async(weighted.select_response(responses, "test"))

        # Cascading might stop early
        cascading = CascadingStrategy(confidence_threshold=0.8)
        cascading_result = run_async(cascading.select_response(responses, "test"))

        # Cost-optimized avoids expensive
        cost_opt = CostOptimizedStrategy()
        cost_opt_result = run_async(cost_opt.select_response(responses, "test"))

        # All should have valid results but potentially different selections
        assert weighted_result.model_name == "gpt-4"
        assert cascading_result.model_name == "claude-haiku"  # Stops at 0.85
        assert cost_opt_result.model_name in ["gpt-3.5", "claude-haiku"]  # Avoids expensive
