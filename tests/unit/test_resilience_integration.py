"""Integration tests for resilience patterns in Nexus data collectors.

Verifies that KHGraphCollector, TrendSignalCollector, and ArclaneCollector
degrade gracefully when circuit breakers are tripped.

Note: The collectors use module-level circuit breaker references created at
import time. After reset_all_breakers(), those references become stale.
We patch the module-level CB variables so the collectors see the tripped state.
"""

from unittest.mock import MagicMock

import pytest

from gozerai_telemetry.resilience import (
    CircuitState,
    get_circuit_breaker,
    reset_all_breakers,
)
from nexus.data import kh_graph_collector, trend_signal_collector, arclane_collector
from nexus.data.kh_graph_collector import KHGraphCollector
from nexus.data.trend_signal_collector import TrendSignalCollector
from nexus.data.arclane_collector import ArclaneCollector


def _trip_breaker(name: str, failure_threshold: int = 3) -> "CircuitBreaker":
    """Create and trip a named circuit breaker. Returns the tripped CB."""
    cb = get_circuit_breaker(name, failure_threshold=failure_threshold, recovery_timeout=120)
    for _ in range(failure_threshold):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN
    return cb


def _mock_knowledge_base():
    """Create a mock KnowledgeBase."""
    kb = MagicMock()
    kb.add_knowledge.return_value = "mock-kid-001"
    return kb


class TestCollectorCircuitBreakerResilience:
    """Verify collectors return empty results when their circuit breakers
    are OPEN, without making any network calls."""

    def setup_method(self):
        reset_all_breakers()
        # Re-assign module-level CB references so collectors use fresh breakers
        kh_graph_collector._kh_cb = get_circuit_breaker("kh", failure_threshold=3, recovery_timeout=120)
        trend_signal_collector._ts_cb = get_circuit_breaker("trendscope", failure_threshold=3, recovery_timeout=120)
        arclane_collector._arclane_cb = get_circuit_breaker("arclane", failure_threshold=3, recovery_timeout=120)

    def test_kh_graph_collector_fetch_json_returns_none_when_cb_open(self):
        """KHGraphCollector._fetch_json() returns None when the 'kh' breaker
        is OPEN."""
        cb = _trip_breaker("kh", failure_threshold=3)
        kh_graph_collector._kh_cb = cb

        collector = KHGraphCollector(
            knowledge_base=_mock_knowledge_base(),
            kh_base_url="http://localhost:59999",
        )
        result = collector._fetch_json("http://localhost:59999/api/discover/clusters")
        assert result is None

    def test_trend_signal_collector_fetch_json_returns_none_when_cb_open(self):
        """TrendSignalCollector._fetch_json() returns None when the
        'trendscope' breaker is OPEN."""
        cb = _trip_breaker("trendscope", failure_threshold=3)
        trend_signal_collector._ts_cb = cb

        collector = TrendSignalCollector(
            knowledge_base=_mock_knowledge_base(),
            ts_base_url="http://localhost:59999",
        )
        result = collector._fetch_json("http://localhost:59999/v1/trends/top")
        assert result is None

    def test_arclane_collector_fetch_json_returns_none_when_cb_open(self):
        """ArclaneCollector._fetch_json() returns None when the 'arclane'
        breaker is OPEN."""
        cb = _trip_breaker("arclane", failure_threshold=3)
        arclane_collector._arclane_cb = cb

        collector = ArclaneCollector(
            knowledge_base=_mock_knowledge_base(),
            arclane_base_url="http://localhost:59999",
        )
        result = collector._fetch_json("http://localhost:59999/api/v1/activity/recent")
        assert result is None

    def test_collectors_return_empty_lists_when_cb_open(self):
        """All collect_* methods return empty lists when their respective
        circuit breakers are OPEN."""
        kh_graph_collector._kh_cb = _trip_breaker("kh", failure_threshold=3)
        trend_signal_collector._ts_cb = _trip_breaker("trendscope", failure_threshold=3)
        arclane_collector._arclane_cb = _trip_breaker("arclane", failure_threshold=3)

        kb = _mock_knowledge_base()

        kh = KHGraphCollector(knowledge_base=kb, kh_base_url="http://localhost:59999")
        assert kh.collect_clusters() == []
        assert kh.collect_coverage_gaps() == []

        ts = TrendSignalCollector(knowledge_base=kb, ts_base_url="http://localhost:59999")
        assert ts.collect_top_trends() == []
        assert ts.collect_signals() == []
        assert ts.collect_lifecycle_distribution() == []

        arc = ArclaneCollector(knowledge_base=kb, arclane_base_url="http://localhost:59999")
        assert arc.collect_recent_activity() == []
        assert arc.collect_content_patterns() == []

    def test_breakers_independent_per_collector(self):
        """Tripping one collector's breaker does not affect others."""
        kh_graph_collector._kh_cb = _trip_breaker("kh", failure_threshold=3)

        ts_cb = get_circuit_breaker("trendscope", failure_threshold=3, recovery_timeout=120)
        arc_cb = get_circuit_breaker("arclane", failure_threshold=3, recovery_timeout=120)

        assert ts_cb.state == CircuitState.CLOSED
        assert arc_cb.state == CircuitState.CLOSED

    def test_reset_all_breakers_restores_collector_breakers(self):
        """After reset_all_breakers(), all collector breakers are fresh."""
        kh_graph_collector._kh_cb = _trip_breaker("kh", failure_threshold=3)
        trend_signal_collector._ts_cb = _trip_breaker("trendscope", failure_threshold=3)
        arclane_collector._arclane_cb = _trip_breaker("arclane", failure_threshold=3)

        reset_all_breakers()

        for name in ("kh", "trendscope", "arclane"):
            cb = get_circuit_breaker(name, failure_threshold=3, recovery_timeout=120)
            assert cb.state == CircuitState.CLOSED, f"{name} should be CLOSED after reset"
            assert cb.allow_request(), f"{name} should allow requests after reset"
