"""Tests for monitoring and metrics system."""

import os
import sys
import pytest
from prometheus_client import CollectorRegistry, REGISTRY

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.monitoring.metrics import MetricsCollector


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_collector_initialization(self):
        """Test metrics collector initialization."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        assert collector.registry == registry
        assert collector.request_count is not None
        assert collector.request_latency is not None
        assert collector.model_requests is not None
        assert collector.cache_hits is not None

    def test_collector_with_default_registry(self):
        """Test collector with default registry."""
        # Use custom registry to avoid conflicts with default REGISTRY
        collector = MetricsCollector(registry=CollectorRegistry())

        assert collector.registry is not None

    def test_record_request(self):
        """Test recording HTTP requests."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_request(
            endpoint="/ensemble",
            method="POST",
            status=200,
            duration=0.5
        )

        # Verify counter was incremented
        samples = list(collector.request_count.collect())[0].samples
        assert len(samples) > 0

        # Find our specific metric
        request_sample = next(
            (s for s in samples if s.labels.get('endpoint') == '/ensemble'),
            None
        )
        assert request_sample is not None
        assert request_sample.value == 1.0

    def test_record_multiple_requests(self):
        """Test recording multiple HTTP requests."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        # Record multiple requests to the same endpoint
        for _ in range(5):
            collector.record_request("/ensemble", "POST", 200, 0.5)

        samples = list(collector.request_count.collect())[0].samples
        request_sample = next(
            (s for s in samples if s.labels.get('endpoint') == '/ensemble'),
            None
        )
        assert request_sample.value == 5.0

    def test_record_request_latency(self):
        """Test recording request latency."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_request("/ensemble", "POST", 200, 0.5)

        # Check histogram was updated
        samples = list(collector.request_latency.collect())[0].samples

        # Histogram creates multiple samples (_count, _sum, _bucket)
        count_sample = next(
            (s for s in samples
             if s.name.endswith('_count') and s.labels.get('endpoint') == '/ensemble'),
            None
        )
        assert count_sample is not None
        assert count_sample.value == 1.0

    def test_record_different_endpoints(self):
        """Test recording requests to different endpoints."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_request("/ensemble", "POST", 200, 0.5)
        collector.record_request("/health", "GET", 200, 0.1)
        collector.record_request("/ensemble", "POST", 500, 1.0)

        samples = list(collector.request_count.collect())[0].samples

        # Should have separate counters for different endpoint/status combinations
        ensemble_200 = next(
            (s for s in samples
             if s.labels.get('endpoint') == '/ensemble' and s.labels.get('status') == '200'),
            None
        )
        ensemble_500 = next(
            (s for s in samples
             if s.labels.get('endpoint') == '/ensemble' and s.labels.get('status') == '500'),
            None
        )
        health_200 = next(
            (s for s in samples
             if s.labels.get('endpoint') == '/health' and s.labels.get('status') == '200'),
            None
        )

        assert ensemble_200.value == 1.0
        assert ensemble_500.value == 1.0
        assert health_200.value == 1.0

    def test_record_model_request_success(self):
        """Test recording successful model request."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_model_request(
            model_name="gpt-4",
            provider="openai",
            latency_ms=500.0,
            tokens_used=1000,
            cost_usd=0.03,
            success=True
        )

        # Check model_requests counter
        samples = list(collector.model_requests.collect())[0].samples
        model_sample = next(
            (s for s in samples
             if s.labels.get('model_name') == 'gpt-4' and s.labels.get('provider') == 'openai'),
            None
        )
        assert model_sample is not None
        assert model_sample.value == 1.0

    def test_record_model_request_tokens(self):
        """Test recording model token usage."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_model_request(
            model_name="gpt-4",
            provider="openai",
            latency_ms=500.0,
            tokens_used=1000,
            cost_usd=0.03,
            success=True
        )

        # Check tokens counter
        samples = list(collector.model_tokens.collect())[0].samples
        token_sample = next(
            (s for s in samples
             if s.labels.get('model_name') == 'gpt-4' and s.labels.get('provider') == 'openai'),
            None
        )
        assert token_sample is not None
        assert token_sample.value == 1000.0

    def test_record_model_request_cost(self):
        """Test recording model costs."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_model_request(
            model_name="gpt-4",
            provider="openai",
            latency_ms=500.0,
            tokens_used=1000,
            cost_usd=0.03,
            success=True
        )

        # Check cost counter
        samples = list(collector.total_cost.collect())[0].samples
        cost_sample = next(
            (s for s in samples
             if s.labels.get('model_name') == 'gpt-4' and s.labels.get('provider') == 'openai'),
            None
        )
        assert cost_sample is not None
        assert cost_sample.value == 0.03

    def test_record_model_request_latency(self):
        """Test recording model latency."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_model_request(
            model_name="gpt-4",
            provider="openai",
            latency_ms=500.0,
            tokens_used=1000,
            cost_usd=0.03,
            success=True
        )

        # Check latency histogram
        samples = list(collector.model_latency.collect())[0].samples
        count_sample = next(
            (s for s in samples
             if s.name.endswith('_count')
             and s.labels.get('model_name') == 'gpt-4'
             and s.labels.get('provider') == 'openai'),
            None
        )
        assert count_sample is not None
        assert count_sample.value == 1.0

    def test_record_model_error(self):
        """Test recording model errors."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_model_request(
            model_name="gpt-4",
            provider="openai",
            latency_ms=100.0,
            tokens_used=0,
            cost_usd=0.0,
            success=False,
            error_type="timeout"
        )

        # Check error counter
        samples = list(collector.model_errors.collect())[0].samples
        error_sample = next(
            (s for s in samples
             if s.labels.get('model_name') == 'gpt-4'
             and s.labels.get('provider') == 'openai'
             and s.labels.get('error_type') == 'timeout'),
            None
        )
        assert error_sample is not None
        assert error_sample.value == 1.0

    def test_record_multiple_model_errors(self):
        """Test recording multiple model errors."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        # Record different error types
        collector.record_model_request(
            "gpt-4", "openai", 100.0, 0, 0.0, False, "timeout"
        )
        collector.record_model_request(
            "gpt-4", "openai", 100.0, 0, 0.0, False, "rate_limit"
        )
        collector.record_model_request(
            "claude-3-opus", "anthropic", 100.0, 0, 0.0, False, "timeout"
        )

        samples = list(collector.model_errors.collect())[0].samples

        # Should have separate counters for different error types
        gpt4_timeout = next(
            (s for s in samples
             if s.labels.get('model_name') == 'gpt-4'
             and s.labels.get('error_type') == 'timeout'),
            None
        )
        gpt4_rate_limit = next(
            (s for s in samples
             if s.labels.get('model_name') == 'gpt-4'
             and s.labels.get('error_type') == 'rate_limit'),
            None
        )

        assert gpt4_timeout.value == 1.0
        assert gpt4_rate_limit.value == 1.0

    def test_record_cache_hit(self):
        """Test recording cache hits."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_cache_hit()
        collector.record_cache_hit()

        samples = list(collector.cache_hits.collect())[0].samples
        assert samples[0].value == 2.0

    def test_record_cache_miss(self):
        """Test recording cache misses."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_cache_miss()
        collector.record_cache_miss()
        collector.record_cache_miss()

        samples = list(collector.cache_misses.collect())[0].samples
        assert samples[0].value == 3.0

    def test_cache_hit_miss_tracking(self):
        """Test tracking both cache hits and misses."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        # Simulate some cache operations
        collector.record_cache_hit()
        collector.record_cache_miss()
        collector.record_cache_hit()
        collector.record_cache_hit()
        collector.record_cache_miss()

        hit_samples = list(collector.cache_hits.collect())[0].samples
        miss_samples = list(collector.cache_misses.collect())[0].samples

        assert hit_samples[0].value == 3.0
        assert miss_samples[0].value == 2.0

    def test_update_budget_metrics(self):
        """Test updating budget metrics."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.update_budget_metrics(budget_limit=100.0, current_spend=30.0)

        # Check monthly budget gauge
        budget_samples = list(collector.monthly_budget.collect())[0].samples
        assert budget_samples[0].value == 100.0

        # Check remaining budget gauge
        remaining_samples = list(collector.budget_remaining.collect())[0].samples
        assert remaining_samples[0].value == 70.0

    def test_update_budget_metrics_over_budget(self):
        """Test budget metrics when over budget."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.update_budget_metrics(budget_limit=100.0, current_spend=120.0)

        # Remaining should be 0 when over budget
        remaining_samples = list(collector.budget_remaining.collect())[0].samples
        assert remaining_samples[0].value == 0.0

    def test_update_budget_metrics_multiple_times(self):
        """Test updating budget metrics multiple times."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        # Initial budget state
        collector.update_budget_metrics(budget_limit=100.0, current_spend=20.0)

        # Update with more spending
        collector.update_budget_metrics(budget_limit=100.0, current_spend=50.0)

        # Should reflect latest values
        budget_samples = list(collector.monthly_budget.collect())[0].samples
        remaining_samples = list(collector.budget_remaining.collect())[0].samples

        assert budget_samples[0].value == 100.0
        assert remaining_samples[0].value == 50.0

    def test_update_ensemble_size(self):
        """Test updating ensemble model count."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.update_ensemble_size(5)

        samples = list(collector.ensemble_models_count.collect())[0].samples
        assert samples[0].value == 5.0

    def test_update_ensemble_size_multiple_times(self):
        """Test updating ensemble size multiple times."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.update_ensemble_size(3)
        collector.update_ensemble_size(5)
        collector.update_ensemble_size(4)

        # Should reflect latest value
        samples = list(collector.ensemble_models_count.collect())[0].samples
        assert samples[0].value == 4.0

    def test_record_ensemble_score(self):
        """Test recording ensemble scores."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_ensemble_score(model_name="gpt-4", score=0.95)

        samples = list(collector.ensemble_score.collect())[0].samples

        # Histogram creates multiple samples
        count_sample = next(
            (s for s in samples
             if s.name.endswith('_count') and s.labels.get('model_name') == 'gpt-4'),
            None
        )
        assert count_sample is not None
        assert count_sample.value == 1.0

    def test_record_multiple_ensemble_scores(self):
        """Test recording multiple ensemble scores."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.record_ensemble_score("gpt-4", 0.95)
        collector.record_ensemble_score("claude-3-opus", 0.88)
        collector.record_ensemble_score("gpt-4", 0.92)

        samples = list(collector.ensemble_score.collect())[0].samples

        # Should have separate histograms for different models
        gpt4_count = next(
            (s for s in samples
             if s.name.endswith('_count') and s.labels.get('model_name') == 'gpt-4'),
            None
        )
        claude_count = next(
            (s for s in samples
             if s.name.endswith('_count') and s.labels.get('model_name') == 'claude-3-opus'),
            None
        )

        assert gpt4_count.value == 2.0
        assert claude_count.value == 1.0

    def test_set_system_info(self):
        """Test setting system information."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        collector.set_system_info(
            version="1.0.0",
            python_version="3.11.14",
            models=["gpt-4", "claude-3-opus", "mistral"]
        )

        samples = list(collector.info.collect())[0].samples
        assert len(samples) > 0

        # Info metric should have our labels
        info_sample = samples[0]
        assert info_sample.labels.get('version') == '1.0.0'
        assert info_sample.labels.get('python_version') == '3.11.14'
        assert info_sample.labels.get('models') == 'gpt-4,claude-3-opus,mistral'

    def test_comprehensive_workflow(self):
        """Test a comprehensive metrics workflow."""
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)

        # Set system info
        collector.set_system_info("1.0.0", "3.11", ["gpt-4", "claude-3"])

        # Update ensemble
        collector.update_ensemble_size(2)

        # Record some requests
        collector.record_request("/ensemble", "POST", 200, 0.5)
        collector.record_request("/ensemble", "POST", 200, 0.6)

        # Record model requests
        collector.record_model_request("gpt-4", "openai", 500.0, 1000, 0.03, True)
        collector.record_model_request("claude-3-opus", "anthropic", 600.0, 800, 0.024, True)

        # Record cache operations
        collector.record_cache_hit()
        collector.record_cache_miss()

        # Update budget
        collector.update_budget_metrics(100.0, 0.054)

        # Record scores
        collector.record_ensemble_score("gpt-4", 0.95)
        collector.record_ensemble_score("claude-3-opus", 0.88)

        # Verify all metrics were recorded
        request_samples = list(collector.request_count.collect())[0].samples
        assert len([s for s in request_samples if s.value > 0]) > 0

        model_samples = list(collector.model_requests.collect())[0].samples
        assert len([s for s in model_samples if s.value > 0]) > 0

        cache_hit_samples = list(collector.cache_hits.collect())[0].samples
        assert cache_hit_samples[0].value == 1.0

        budget_samples = list(collector.monthly_budget.collect())[0].samples
        assert budget_samples[0].value == 100.0
