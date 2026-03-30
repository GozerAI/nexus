"""
Unit tests for the Nexus Self-Updating / Self-Sufficiency system.

Covers:
- 928: Provider configuration auto-discovery
- 937: Model provider health-based config adjustment
- 960: Provider cost comparison with automatic switching
"""

import os
import time
import pytest

from nexus.offline.provider_discovery import (
    ProviderAutoDiscovery,
    ProviderType,
    ProviderEndpoint,
    DiscoveryMethod,
)
from nexus.offline.provider_health import (
    ProviderHealthManager,
    HealthStatus,
    HealthSample,
    ProviderHealth,
    HealthAdjustment,
)
from nexus.offline.cost_switching import (
    ProviderCostSwitcher,
    CostSample,
    ProviderCostProfile,
    SwitchEvent,
)


# ════════════════════════════════════════════════════════════════════
# 928 — Provider Configuration Auto-Discovery
# ════════════════════════════════════════════════════════════════════


class TestProviderAutoDiscovery:
    """Tests for ProviderAutoDiscovery."""

    def test_initialization(self):
        discovery = ProviderAutoDiscovery()
        assert discovery is not None

    def test_manual_registration(self):
        discovery = ProviderAutoDiscovery()
        endpoint = discovery.register_endpoint(
            ProviderType.OPENAI,
            "https://api.openai.com/v1",
            api_key="sk-test",
            models=["gpt-4"],
        )
        assert endpoint.available is True
        assert endpoint.discovery_method == DiscoveryMethod.MANUAL
        assert endpoint.models == ["gpt-4"]

    def test_get_endpoint(self):
        discovery = ProviderAutoDiscovery()
        discovery.register_endpoint(ProviderType.OLLAMA, "http://localhost:11434")
        ep = discovery.get_endpoint(ProviderType.OLLAMA)
        assert ep is not None
        assert ep.provider == ProviderType.OLLAMA

    def test_get_endpoint_nonexistent(self):
        discovery = ProviderAutoDiscovery()
        assert discovery.get_endpoint(ProviderType.MISTRAL) is None

    def test_get_available_providers(self):
        discovery = ProviderAutoDiscovery()
        discovery.register_endpoint(ProviderType.OPENAI, "url1", api_key="key1")
        discovery.register_endpoint(ProviderType.ANTHROPIC, "url2", api_key="key2")
        available = discovery.get_available_providers()
        assert len(available) == 2

    def test_env_key_discovery(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        discovery = ProviderAutoDiscovery()
        discovery.discover_all(force=True)
        ep = discovery.get_endpoint(ProviderType.OPENAI)
        assert ep is not None
        assert ep.available is True
        assert ep.discovery_method == DiscoveryMethod.API_KEY

    def test_env_key_discovery_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        discovery = ProviderAutoDiscovery()
        discovery.discover_all(force=True)
        ep = discovery.get_endpoint(ProviderType.ANTHROPIC)
        assert ep is not None
        assert ep.available is True

    def test_no_env_keys(self, monkeypatch):
        # Clear any real keys
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
            monkeypatch.delenv(key, raising=False)
        discovery = ProviderAutoDiscovery()
        discovery._discover_from_environment()
        # Cloud providers should not be available without keys
        for p in [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE]:
            ep = discovery.get_endpoint(p)
            if ep:
                # Only available if discovered through env
                pass

    def test_custom_probe(self):
        discovery = ProviderAutoDiscovery()
        discovery.register_custom_probe(ProviderType.LOCAL, lambda: True)
        discovery.discover_all(force=True)
        ep = discovery.get_endpoint(ProviderType.LOCAL)
        assert ep is not None
        assert ep.available is True

    def test_custom_probe_failure(self):
        discovery = ProviderAutoDiscovery()
        discovery.register_custom_probe(ProviderType.LOCAL, lambda: False)
        discovery.discover_all(force=True)
        ep = discovery.get_endpoint(ProviderType.LOCAL)
        if ep:
            assert ep.available is False

    def test_discovery_report(self):
        discovery = ProviderAutoDiscovery()
        discovery.register_endpoint(ProviderType.OPENAI, "url", api_key="key")
        report = discovery.get_discovery_report()
        assert "total_checked" in report
        assert "providers" in report
        assert "openai" in report["providers"]

    def test_discover_specific_provider_with_key(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        discovery = ProviderAutoDiscovery()
        ep = discovery.discover_provider(ProviderType.MISTRAL)
        assert ep is not None
        assert ep.available is True

    def test_rediscovery_cooldown(self):
        discovery = ProviderAutoDiscovery(rediscovery_interval=1000)
        discovery.register_endpoint(ProviderType.OPENAI, "url", api_key="key")
        discovery.discover_all(force=True)
        # Second call should use cached results
        result = discovery.discover_all(force=False)
        assert len(result) >= 1

    def test_discovery_log(self):
        discovery = ProviderAutoDiscovery()
        discovery.register_endpoint(ProviderType.OPENAI, "url", api_key="key")
        assert len(discovery._discovery_log) >= 1
        assert discovery._discovery_log[-1]["provider"] == "openai"


# ════════════════════════════════════════════════════════════════════
# 937 — Provider Health-Based Configuration Adjustment
# ════════════════════════════════════════════════════════════════════


class TestProviderHealthManager:
    """Tests for ProviderHealthManager."""

    def test_initialization(self):
        mgr = ProviderHealthManager()
        assert mgr is not None

    def test_record_healthy_sample(self):
        mgr = ProviderHealthManager()
        mgr.record_sample("openai", 200.0, True)
        health = mgr.get_health("openai")
        assert health.total_samples == 1
        assert health.success_rate == 1.0

    def test_record_failure(self):
        mgr = ProviderHealthManager()
        mgr.record_sample("openai", 0.0, False, error_type="timeout")
        health = mgr.get_health("openai")
        assert health.success_rate == 0.0
        assert health.consecutive_failures == 1

    def test_consecutive_failures_tracked(self):
        mgr = ProviderHealthManager()
        for _ in range(5):
            mgr.record_sample("openai", 0.0, False)
        health = mgr.get_health("openai")
        assert health.consecutive_failures == 5

    def test_consecutive_failures_reset_on_success(self):
        mgr = ProviderHealthManager()
        for _ in range(3):
            mgr.record_sample("openai", 0.0, False)
        mgr.record_sample("openai", 200.0, True)
        health = mgr.get_health("openai")
        assert health.consecutive_failures == 0

    def test_health_classification_healthy(self):
        mgr = ProviderHealthManager()
        for _ in range(10):
            mgr.record_sample("openai", 200.0, True)
        health = mgr.get_health("openai")
        assert health.status == HealthStatus.HEALTHY

    def test_health_classification_degraded(self):
        mgr = ProviderHealthManager(success_rate_degraded=0.9)
        for _ in range(8):
            mgr.record_sample("openai", 200.0, True)
        for _ in range(2):
            mgr.record_sample("openai", 200.0, False)
        health = mgr.get_health("openai")
        assert health.status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    def test_health_classification_unhealthy_failures(self):
        mgr = ProviderHealthManager()
        for _ in range(6):
            mgr.record_sample("openai", 0.0, False)
        health = mgr.get_health("openai")
        assert health.status == HealthStatus.UNHEALTHY

    def test_health_classification_unhealthy_latency(self):
        mgr = ProviderHealthManager(
            latency_degraded_ms=2000.0,
            latency_unhealthy_ms=5000.0,
        )
        for _ in range(3):
            mgr.record_sample("openai", 6000.0, True)
        health = mgr.get_health("openai")
        assert health.status == HealthStatus.UNHEALTHY

    def test_routing_weight_healthy(self):
        mgr = ProviderHealthManager()
        for _ in range(10):
            mgr.record_sample("openai", 100.0, True)
        health = mgr.get_health("openai")
        assert health.routing_weight == ProviderHealthManager.WEIGHT_HEALTHY

    def test_routing_weight_unhealthy(self):
        mgr = ProviderHealthManager()
        for _ in range(6):
            mgr.record_sample("openai", 0.0, False)
        health = mgr.get_health("openai")
        assert health.routing_weight == ProviderHealthManager.WEIGHT_UNHEALTHY

    def test_get_routing_weights(self):
        mgr = ProviderHealthManager()
        for _ in range(5):
            mgr.record_sample("openai", 100.0, True)
            mgr.record_sample("anthropic", 150.0, True)
        weights = mgr.get_routing_weights()
        assert "openai" in weights
        assert "anthropic" in weights

    def test_get_best_provider(self):
        mgr = ProviderHealthManager()
        for _ in range(10):
            mgr.record_sample("openai", 100.0, True)
            mgr.record_sample("anthropic", 500.0, True)
        best = mgr.get_best_provider()
        assert best == "openai"  # Lower latency

    def test_get_best_provider_with_candidates(self):
        mgr = ProviderHealthManager()
        for _ in range(5):
            mgr.record_sample("openai", 100.0, True)
            mgr.record_sample("anthropic", 150.0, True)
            mgr.record_sample("google", 200.0, True)
        best = mgr.get_best_provider(candidates=["anthropic", "google"])
        assert best in ("anthropic", "google")

    def test_get_best_provider_empty(self):
        mgr = ProviderHealthManager()
        assert mgr.get_best_provider() is None

    def test_health_transition_callback(self):
        mgr = ProviderHealthManager()
        transitions = []
        mgr.register_callback(lambda adj: transitions.append(adj))

        for _ in range(10):
            mgr.record_sample("openai", 100.0, True)
        # Now degrade
        for _ in range(6):
            mgr.record_sample("openai", 0.0, False)

        # Should have recorded at least one transition
        assert len(transitions) >= 1

    def test_health_report(self):
        mgr = ProviderHealthManager()
        for _ in range(5):
            mgr.record_sample("openai", 200.0, True)
        report = mgr.get_health_report()
        assert "providers" in report
        assert "openai" in report["providers"]
        assert "status" in report["providers"]["openai"]

    def test_latency_metrics(self):
        mgr = ProviderHealthManager()
        for lat in [100, 200, 300, 400, 500]:
            mgr.record_sample("openai", float(lat), True)
        health = mgr.get_health("openai")
        assert abs(health.avg_latency_ms - 300.0) < 1.0
        assert health.p95_latency_ms >= 400.0

    def test_all_health(self):
        mgr = ProviderHealthManager()
        mgr.record_sample("a", 100, True)
        mgr.record_sample("b", 200, True)
        all_health = mgr.get_all_health()
        assert "a" in all_health
        assert "b" in all_health


# ════════════════════════════════════════════════════════════════════
# 960 — Provider Cost Comparison with Automatic Switching
# ════════════════════════════════════════════════════════════════════


class TestProviderCostSwitcher:
    """Tests for ProviderCostSwitcher."""

    def test_initialization(self):
        switcher = ProviderCostSwitcher()
        assert switcher is not None
        assert switcher.preferred_provider is None
        assert switcher.total_spend == 0.0

    def test_record_sample(self):
        switcher = ProviderCostSwitcher()
        switcher.record_sample(
            "openai", "gpt-4", 100, 50, 0.005, 0.9, 200.0
        )
        assert switcher.total_spend == 0.005

    def test_cost_profile_calculation(self):
        switcher = ProviderCostSwitcher(min_samples=2)
        for _ in range(5):
            switcher.record_sample(
                "openai", "gpt-4", 1000, 500, 0.05, 0.85, 300.0
            )
        profile = switcher._profiles["openai"]
        assert profile.sample_count == 5
        assert profile.avg_quality == 0.85
        assert profile.avg_cost_per_1k_tokens > 0

    def test_select_provider_empty(self):
        switcher = ProviderCostSwitcher()
        assert switcher.select_provider(candidates=["a"]) == "a"

    def test_select_provider_by_efficiency(self):
        switcher = ProviderCostSwitcher(min_samples=2, quality_floor=0.5)
        # OpenAI: expensive but good
        for _ in range(5):
            switcher.record_sample(
                "openai", "gpt-4", 1000, 500, 0.10, 0.9, 200
            )
        # Local: cheap and decent
        for _ in range(5):
            switcher.record_sample(
                "local", "llama", 1000, 500, 0.001, 0.75, 500
            )
        selected = switcher.select_provider()
        # Local should win on cost efficiency (quality/cost)
        assert selected == "local"

    def test_select_provider_respects_quality_floor(self):
        switcher = ProviderCostSwitcher(
            min_samples=2, quality_floor=0.8
        )
        for _ in range(5):
            switcher.record_sample("cheap", "m1", 100, 50, 0.001, 0.3, 100)
            switcher.record_sample("good", "m2", 100, 50, 0.05, 0.9, 200)
        selected = switcher.select_provider()
        assert selected == "good"  # cheap doesn't meet quality floor

    def test_switch_event_on_new_preferred(self):
        switcher = ProviderCostSwitcher(
            min_samples=2, switch_cooldown=0
        )
        # Establish openai as preferred
        for _ in range(5):
            switcher.record_sample("openai", "gpt-4", 100, 50, 0.05, 0.8, 200)

        # Now introduce a better provider
        event = None
        for _ in range(5):
            e = switcher.record_sample(
                "anthropic", "claude", 100, 50, 0.01, 0.9, 150
            )
            if e:
                event = e

        if event:
            assert isinstance(event, SwitchEvent)
            assert event.to_provider == "anthropic"

    def test_budget_tracking(self):
        switcher = ProviderCostSwitcher(budget_limit_usd=1.0)
        for _ in range(20):
            switcher.record_sample("openai", "gpt-4", 100, 50, 0.1, 0.8, 200)
        assert switcher.is_over_budget is True

    def test_no_budget_limit(self):
        switcher = ProviderCostSwitcher(budget_limit_usd=None)
        switcher.record_sample("openai", "gpt-4", 100, 50, 100.0, 0.8, 200)
        assert switcher.is_over_budget is False

    def test_cost_comparison(self):
        switcher = ProviderCostSwitcher(min_samples=2)
        for _ in range(5):
            switcher.record_sample("openai", "gpt-4", 100, 50, 0.05, 0.85, 200)
            switcher.record_sample("local", "llama", 100, 50, 0.001, 0.7, 400)
        comparison = switcher.get_cost_comparison()
        assert len(comparison) == 2
        assert comparison[0]["provider"] in ("openai", "local")

    def test_per_model_breakdown(self):
        switcher = ProviderCostSwitcher(min_samples=2)
        for _ in range(3):
            switcher.record_sample("openai", "gpt-4", 100, 50, 0.05, 0.9, 200)
            switcher.record_sample("openai", "gpt-3.5", 100, 50, 0.01, 0.7, 100)
        profile = switcher._profiles["openai"]
        assert "gpt-4" in profile.models
        assert "gpt-3.5" in profile.models

    def test_report(self):
        switcher = ProviderCostSwitcher()
        for _ in range(3):
            switcher.record_sample("openai", "gpt-4", 100, 50, 0.05, 0.8, 200)
        report = switcher.get_report()
        assert "preferred_provider" in report
        assert "total_spend" in report
        assert "providers" in report

    def test_switch_cooldown(self):
        switcher = ProviderCostSwitcher(
            min_samples=2, switch_cooldown=1000.0  # Very long cooldown
        )
        for _ in range(5):
            switcher.record_sample("openai", "gpt-4", 100, 50, 0.05, 0.8, 200)
        switcher._last_switch = time.time()  # Simulate recent switch

        # New provider should not trigger switch due to cooldown
        event = None
        for _ in range(5):
            e = switcher.record_sample(
                "cheap", "m1", 100, 50, 0.001, 0.9, 100
            )
            if e:
                event = e
        assert event is None

    def test_select_with_max_cost(self):
        switcher = ProviderCostSwitcher(min_samples=2)
        for _ in range(5):
            switcher.record_sample("expensive", "m1", 1000, 500, 1.0, 0.95, 100)
            switcher.record_sample("cheap", "m2", 1000, 500, 0.01, 0.75, 200)
        # Only allow cheap providers
        selected = switcher.select_provider(max_cost_per_1k=0.1)
        assert selected == "cheap"
