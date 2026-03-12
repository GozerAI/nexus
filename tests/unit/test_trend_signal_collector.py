"""Tests for TrendSignalCollector — trend, signal, and lifecycle knowledge collection."""

import json
import pytest
from unittest.mock import MagicMock, patch

from nexus.data.trend_signal_collector import TrendSignalCollector
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType


@pytest.fixture
def kb():
    kb = MagicMock(spec=KnowledgeBase)
    kb.add_knowledge.side_effect = lambda **kw: f"kid-{kb.add_knowledge.call_count}"
    return kb


@pytest.fixture
def collector(kb):
    return TrendSignalCollector(
        knowledge_base=kb,
        ts_base_url="http://ts-test:8002",
        bearer_token="test-token",
    )


# ─── collect_top_trends ───────────────────────────────────────


class TestCollectTopTrends:
    def test_creates_knowledge_items(self, collector, kb):
        data = [
            {"id": "t1", "name": "AI Agents", "score": 92, "velocity": 1.5,
             "category": "Technology", "status": "rising"},
            {"id": "t2", "name": "Green Energy", "score": 85, "velocity": 0.8,
             "category": "Energy", "status": "stable"},
        ]
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=data):
            ids = collector.collect_top_trends()
            assert len(ids) == 2
            assert kb.add_knowledge.call_count == 2

    def test_graceful_degradation(self, collector, kb):
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=None):
            ids = collector.collect_top_trends()
            assert ids == []
            assert kb.add_knowledge.call_count == 0

    def test_empty_list(self, collector, kb):
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=[]):
            ids = collector.collect_top_trends()
            assert ids == []

    def test_trend_knowledge_content(self, collector, kb):
        data = [
            {"id": "t1", "name": "Quantum Computing", "score": 78, "velocity": 2.1,
             "category": "Technology", "status": "emerging"},
        ]
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=data):
            collector.collect_top_trends()
            call_kwargs = kb.add_knowledge.call_args[1]
            assert "Quantum Computing" in call_kwargs["content"]
            assert "78" in call_kwargs["content"]
            assert "2.1" in call_kwargs["content"]
            assert "emerging" in call_kwargs["content"]
            assert call_kwargs["knowledge_type"] == KnowledgeType.FACTUAL
            assert call_kwargs["source"] == "trendscope:top_trend"
            assert "trendscope" in call_kwargs["context_tags"]
            assert "technology" in call_kwargs["context_tags"]


# ─── collect_signals ──────────────────────────────────────────


class TestCollectSignals:
    def test_processes_strong_buy_and_buy(self, collector, kb):
        data = {
            "strong_buy": [
                {"id": "s1", "name": "AI Agents", "score": 95, "velocity": 2.0,
                 "momentum": 1.8, "category": "Technology", "signal": "strong_buy"},
            ],
            "buy": [
                {"id": "s2", "name": "Edge Computing", "score": 80, "velocity": 1.2,
                 "momentum": 0.9, "category": "Infrastructure", "signal": "buy"},
            ],
            "hold": [
                {"id": "s3", "name": "Blockchain", "score": 50, "velocity": 0.1,
                 "momentum": -0.2, "category": "Fintech", "signal": "hold"},
            ],
            "sell": [],
            "strong_sell": [],
        }
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=data):
            ids = collector.collect_signals()
            assert len(ids) == 2
            assert kb.add_knowledge.call_count == 2

    def test_graceful_degradation(self, collector, kb):
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=None):
            ids = collector.collect_signals()
            assert ids == []

    def test_empty_signals(self, collector, kb):
        data = {"strong_buy": [], "buy": [], "hold": [], "sell": [], "strong_sell": []}
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=data):
            ids = collector.collect_signals()
            assert ids == []

    def test_ignores_hold_sell_strong_sell(self, collector, kb):
        data = {
            "strong_buy": [],
            "buy": [],
            "hold": [
                {"id": "s1", "name": "Stale Trend", "score": 40, "velocity": 0.0,
                 "momentum": -0.5, "category": "Misc", "signal": "hold"},
            ],
            "sell": [
                {"id": "s2", "name": "Dying Trend", "score": 20, "velocity": -1.0,
                 "momentum": -2.0, "category": "Legacy", "signal": "sell"},
            ],
            "strong_sell": [
                {"id": "s3", "name": "Dead Trend", "score": 5, "velocity": -3.0,
                 "momentum": -4.0, "category": "Obsolete", "signal": "strong_sell"},
            ],
        }
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=data):
            ids = collector.collect_signals()
            assert ids == []
            assert kb.add_knowledge.call_count == 0

    def test_signal_knowledge_content(self, collector, kb):
        data = {
            "strong_buy": [
                {"id": "s1", "name": "MLOps", "score": 90, "velocity": 1.7,
                 "momentum": 1.3, "category": "DevOps", "signal": "strong_buy"},
            ],
            "buy": [],
            "hold": [],
            "sell": [],
            "strong_sell": [],
        }
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=data):
            collector.collect_signals()
            call_kwargs = kb.add_knowledge.call_args[1]
            assert "MLOps" in call_kwargs["content"]
            assert "strong_buy" in call_kwargs["content"]
            assert "score=90" in call_kwargs["content"]
            assert call_kwargs["source"] == "trendscope:signal"
            assert "strong_buy" in call_kwargs["context_tags"]


# ─── collect_lifecycle_distribution ───────────────────────────


class TestCollectLifecycleDistribution:
    def test_creates_single_knowledge_item(self, collector, kb):
        data = {"emerging": 15, "growing": 30, "mature": 25, "declining": 10}
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=data):
            ids = collector.collect_lifecycle_distribution()
            assert len(ids) == 1
            call_kwargs = kb.add_knowledge.call_args[1]
            assert "emerging: 15" in call_kwargs["content"]
            assert "growing: 30" in call_kwargs["content"]
            assert call_kwargs["source"] == "trendscope:lifecycle"
            assert "lifecycle" in call_kwargs["context_tags"]

    def test_graceful_degradation(self, collector, kb):
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value=None):
            ids = collector.collect_lifecycle_distribution()
            assert ids == []

    def test_empty_dict(self, collector, kb):
        with patch("nexus.data.trend_signal_collector.resilient_fetch",
                    return_value={}):
            ids = collector.collect_lifecycle_distribution()
            assert ids == []


# ─── collect_all ──────────────────────────────────────────────


class TestCollectAll:
    def test_returns_combined_stats(self, collector, kb):
        trends_data = [
            {"id": "t1", "name": "AI", "score": 90, "velocity": 1.5,
             "category": "Tech", "status": "rising"},
        ]
        signals_data = {
            "strong_buy": [
                {"id": "s1", "name": "AI", "score": 95, "velocity": 2.0,
                 "momentum": 1.8, "category": "Tech", "signal": "strong_buy"},
            ],
            "buy": [],
            "hold": [],
            "sell": [],
            "strong_sell": [],
        }
        lifecycle_data = {"emerging": 5, "mature": 10}

        def side_effect(url, **kwargs):
            if "trends/top" in url:
                return trends_data
            if "signals" in url:
                return signals_data
            return lifecycle_data

        with patch("nexus.data.trend_signal_collector.resilient_fetch", side_effect=side_effect):
            stats = collector.collect_all()
            assert stats["trends_collected"] == 1
            assert stats["signals_collected"] == 1
            assert stats["lifecycle_collected"] == 1
            assert stats["total_knowledge_items"] == 3
            assert len(stats["knowledge_ids"]) == 3

    def test_partial_failure(self, collector, kb):
        """When trends endpoint fails but signals and lifecycle work."""
        signals_data = {
            "strong_buy": [],
            "buy": [
                {"id": "s1", "name": "Edge", "score": 80, "velocity": 1.0,
                 "momentum": 0.5, "category": "Infra", "signal": "buy"},
            ],
            "hold": [],
            "sell": [],
            "strong_sell": [],
        }
        lifecycle_data = {"growing": 20}

        def side_effect(url, **kwargs):
            if "trends/top" in url:
                return None
            if "signals" in url:
                return signals_data
            return lifecycle_data

        with patch("nexus.data.trend_signal_collector.resilient_fetch", side_effect=side_effect):
            stats = collector.collect_all()
            assert stats["trends_collected"] == 0
            assert stats["signals_collected"] == 1
            assert stats["lifecycle_collected"] == 1


# ─── Configuration ────────────────────────────────────────────


class TestConfiguration:
    def test_uses_provided_url(self, kb):
        c = TrendSignalCollector(knowledge_base=kb, ts_base_url="http://custom:9999")
        assert c.ts_base_url == "http://custom:9999"

    def test_uses_env_var(self, kb):
        with patch.dict("os.environ", {"TRENDSCOPE_BASE_URL": "http://env-ts:8002"}):
            c = TrendSignalCollector(knowledge_base=kb)
            assert c.ts_base_url == "http://env-ts:8002"

    def test_bearer_token_from_env(self, kb):
        with patch.dict("os.environ", {"TRENDSCOPE_API_TOKEN": "env-secret"}):
            c = TrendSignalCollector(knowledge_base=kb)
            assert c.bearer_token == "env-secret"

    def test_default_confidence(self, kb):
        c = TrendSignalCollector(knowledge_base=kb)
        assert c.confidence == 0.75
