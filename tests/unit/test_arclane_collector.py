"""Tests for ArclaneCollector — business cycle knowledge collection."""

import json
import pytest
from unittest.mock import MagicMock, patch

from nexus.data.arclane_collector import ArclaneCollector
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType


@pytest.fixture
def kb():
    kb = MagicMock(spec=KnowledgeBase)
    kb.add_knowledge.side_effect = lambda **kw: f"kid-{kb.add_knowledge.call_count}"
    return kb


@pytest.fixture
def collector(kb):
    return ArclaneCollector(
        knowledge_base=kb,
        arclane_base_url="http://arclane-test:8012",
    )


def _mock_urlopen_response(data):
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(data).encode()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestCollectRecentActivity:
    def test_creates_knowledge_from_activities(self, collector, kb):
        data = {
            "activities": [
                {"business_name": "TechCo", "action": "Researching market", "detail": "Found 3 competitors in AI space"},
                {"business_name": "TechCo", "action": "Analyzing strategy", "detail": "Growth potential identified"},
            ]
        }
        with patch("nexus.data.arclane_collector.urlopen",
                    return_value=_mock_urlopen_response(data)):
            ids = collector.collect_recent_activity()
            assert len(ids) == 1  # grouped by business
            assert kb.add_knowledge.call_count == 1
            call_kwargs = kb.add_knowledge.call_args[1]
            assert "TechCo" in call_kwargs["content"]

    def test_graceful_degradation(self, collector, kb):
        with patch("nexus.data.arclane_collector.urlopen",
                    side_effect=ConnectionError("down")):
            ids = collector.collect_recent_activity()
            assert ids == []

    def test_empty_activities(self, collector, kb):
        with patch("nexus.data.arclane_collector.urlopen",
                    return_value=_mock_urlopen_response({"activities": []})):
            ids = collector.collect_recent_activity()
            assert ids == []

    def test_multiple_businesses(self, collector, kb):
        data = {
            "activities": [
                {"business_name": "BizA", "action": "Planning", "detail": "Details A"},
                {"business_name": "BizB", "action": "Building", "detail": "Details B"},
            ]
        }
        with patch("nexus.data.arclane_collector.urlopen",
                    return_value=_mock_urlopen_response(data)):
            ids = collector.collect_recent_activity()
            assert len(ids) == 2

    def test_handles_list_response(self, collector, kb):
        data = [
            {"business_name": "Biz", "action": "Act", "detail": "Det"},
        ]
        with patch("nexus.data.arclane_collector.urlopen",
                    return_value=_mock_urlopen_response(data)):
            ids = collector.collect_recent_activity()
            assert len(ids) == 1


class TestCollectContentPatterns:
    def test_creates_summary_knowledge(self, collector, kb):
        data = {
            "content": [
                {"content_type": "blog"},
                {"content_type": "blog"},
                {"content_type": "social"},
            ]
        }
        with patch("nexus.data.arclane_collector.urlopen",
                    return_value=_mock_urlopen_response(data)):
            ids = collector.collect_content_patterns()
            assert len(ids) == 1
            call_kwargs = kb.add_knowledge.call_args[1]
            assert "blog: 2" in call_kwargs["content"]
            assert "social: 1" in call_kwargs["content"]

    def test_graceful_degradation(self, collector, kb):
        from urllib.error import URLError
        with patch("nexus.data.arclane_collector.urlopen",
                    side_effect=URLError("timeout")):
            ids = collector.collect_content_patterns()
            assert ids == []

    def test_empty_content(self, collector, kb):
        with patch("nexus.data.arclane_collector.urlopen",
                    return_value=_mock_urlopen_response({"content": []})):
            ids = collector.collect_content_patterns()
            assert ids == []


class TestCollectAll:
    def test_returns_combined_stats(self, collector, kb):
        activity_data = {"activities": [{"business_name": "A", "action": "X", "detail": "Y"}]}
        content_data = {"content": [{"content_type": "blog"}]}

        def side_effect(req, timeout=5):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "activity" in url:
                return _mock_urlopen_response(activity_data)
            return _mock_urlopen_response(content_data)

        with patch("nexus.data.arclane_collector.urlopen", side_effect=side_effect):
            stats = collector.collect_all()
            assert stats["activities_collected"] == 1
            assert stats["content_patterns_collected"] == 1
            assert stats["total_knowledge_items"] == 2


class TestConfiguration:
    def test_uses_provided_url(self, kb):
        c = ArclaneCollector(knowledge_base=kb, arclane_base_url="http://custom:9999")
        assert c.arclane_base_url == "http://custom:9999"

    def test_uses_env_var(self, kb):
        with patch.dict("os.environ", {"ARCLANE_BASE_URL": "http://env:8012"}):
            c = ArclaneCollector(knowledge_base=kb)
            assert c.arclane_base_url == "http://env:8012"

    def test_default_confidence(self, kb):
        c = ArclaneCollector(knowledge_base=kb)
        assert c.confidence == 0.70
