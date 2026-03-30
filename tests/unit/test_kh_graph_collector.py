"""Tests for KHGraphCollector — cluster and coverage gap knowledge collection."""

import json
import pytest
from unittest.mock import MagicMock, patch

from nexus.data.kh_graph_collector import KHGraphCollector
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType


@pytest.fixture
def kb():
    kb = MagicMock(spec=KnowledgeBase)
    kb.add_knowledge.side_effect = lambda **kw: f"kid-{kb.add_knowledge.call_count}"
    return kb


@pytest.fixture
def collector(kb):
    return KHGraphCollector(
        knowledge_base=kb,
        kh_base_url="http://kh-test:8011",
    )


# ─── collect_clusters ──────────────────────────────────────────


class TestCollectClusters:
    def test_creates_knowledge_items(self, collector, kb):
        data = {
            "clusters": [
                {"name": "AI Tools", "size": 12, "categories": ["ai", "ml"]},
                {"name": "Web Dev", "size": 8, "categories": ["frontend", "backend"]},
            ]
        }
        with patch("nexus.data.kh_graph_collector.resilient_fetch",
                    return_value=data):
            ids = collector.collect_clusters()
            assert len(ids) == 2
            assert kb.add_knowledge.call_count == 2

    def test_graceful_degradation(self, collector, kb):
        with patch("nexus.data.kh_graph_collector.resilient_fetch",
                    return_value=None):
            ids = collector.collect_clusters()
            assert ids == []
            assert kb.add_knowledge.call_count == 0

    def test_empty_clusters(self, collector, kb):
        with patch("nexus.data.kh_graph_collector.resilient_fetch",
                    return_value={"clusters": []}):
            ids = collector.collect_clusters()
            assert ids == []

    def test_cluster_knowledge_content(self, collector, kb):
        data = {"clusters": [{"name": "DataOps", "size": 5, "categories": ["etl", "pipeline"]}]}
        with patch("nexus.data.kh_graph_collector.resilient_fetch",
                    return_value=data):
            collector.collect_clusters()
            call_kwargs = kb.add_knowledge.call_args[1]
            assert "DataOps" in call_kwargs["content"]
            assert "5 nodes" in call_kwargs["content"]
            assert call_kwargs["knowledge_type"] == KnowledgeType.FACTUAL


# ─── collect_coverage_gaps ─────────────────────────────────────


class TestCollectCoverageGaps:
    def test_creates_knowledge_items(self, collector, kb):
        data = {
            "gaps": [
                {"category": "security", "severity": "high", "artifact_count": 2},
            ]
        }
        with patch("nexus.data.kh_graph_collector.resilient_fetch",
                    return_value=data):
            ids = collector.collect_coverage_gaps()
            assert len(ids) == 1
            call_kwargs = kb.add_knowledge.call_args[1]
            assert "security" in call_kwargs["content"]
            assert "high" in call_kwargs["content"]

    def test_graceful_degradation(self, collector, kb):
        with patch("nexus.data.kh_graph_collector.resilient_fetch",
                    return_value=None):
            ids = collector.collect_coverage_gaps()
            assert ids == []

    def test_empty_gaps(self, collector, kb):
        with patch("nexus.data.kh_graph_collector.resilient_fetch",
                    return_value={"gaps": []}):
            ids = collector.collect_coverage_gaps()
            assert ids == []


# ─── collect_all ───────────────────────────────────────────────


class TestCollectAll:
    def test_returns_combined_stats(self, collector, kb):
        cluster_data = {"clusters": [{"name": "A", "size": 3, "categories": ["x"]}]}
        gap_data = {"gaps": [{"category": "y", "severity": "medium", "artifact_count": 1}]}

        def side_effect(url, **kwargs):
            if "clusters" in url:
                return cluster_data
            return gap_data

        with patch("nexus.data.kh_graph_collector.resilient_fetch", side_effect=side_effect):
            stats = collector.collect_all()
            assert stats["clusters_collected"] == 1
            assert stats["gaps_collected"] == 1
            assert stats["total_knowledge_items"] == 2
            assert len(stats["knowledge_ids"]) == 2

    def test_partial_failure(self, collector, kb):
        """When clusters endpoint fails but gaps works."""
        gap_data = {"gaps": [{"category": "z", "severity": "low", "artifact_count": 0}]}

        def side_effect(url, **kwargs):
            if "clusters" in url:
                return None
            return gap_data

        with patch("nexus.data.kh_graph_collector.resilient_fetch", side_effect=side_effect):
            stats = collector.collect_all()
            assert stats["clusters_collected"] == 0
            assert stats["gaps_collected"] == 1


# ─── Configuration ─────────────────────────────────────────────


class TestConfiguration:
    def test_uses_provided_url(self, kb):
        c = KHGraphCollector(knowledge_base=kb, kh_base_url="http://custom:9999")
        assert c.kh_base_url == "http://custom:9999"

    def test_uses_env_var(self, kb):
        with patch.dict("os.environ", {"KH_BASE_URL": "http://env-kh:8011"}):
            c = KHGraphCollector(knowledge_base=kb)
            assert c.kh_base_url == "http://env-kh:8011"

    def test_default_confidence(self, kb):
        c = KHGraphCollector(knowledge_base=kb)
        assert c.confidence == 0.75
