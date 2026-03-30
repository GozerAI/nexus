"""
Tests for WikipediaCollector — fact extraction, tag building, article collection.
"""

import pytest
from unittest.mock import MagicMock, patch

from nexus.data.wikipedia_collector import (
    WikipediaCollector,
    extract_facts,
    strip_html,
    _build_tags,
    DEFAULT_TOPICS,
)
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType


class TestExtractFacts:
    """Test the pure extract_facts function."""

    def test_extracts_factual_sentences(self):
        text = "Python is a programming language. It was created by Guido van Rossum. Short."
        facts = extract_facts(text, min_length=20, max_length=200)
        assert len(facts) >= 1
        assert any("programming language" in f for f in facts)

    def test_filters_short_sentences(self):
        text = "AI is great. This is a much longer sentence that has enough words to pass the filter."
        facts = extract_facts(text, min_length=30, max_length=300)
        # "AI is great." is only 13 chars, should be filtered
        assert not any("AI is great" in f for f in facts)

    def test_filters_long_sentences(self):
        long = "X " * 200 + "is something."
        facts = extract_facts(long, min_length=10, max_length=50)
        assert len(facts) == 0

    def test_filters_hedging_language(self):
        text = "This might be useful for research. Python is a language."
        facts = extract_facts(text, min_length=10, max_length=300)
        assert not any("might" in f for f in facts)

    def test_requires_verb_indicator(self):
        text = "Neural networks for data processing. Neural networks are computational models."
        facts = extract_facts(text, min_length=20, max_length=300)
        # First sentence has no is/are/was — should be filtered
        # Second has "are" — should pass
        assert any("computational models" in f for f in facts)

    def test_empty_input(self):
        assert extract_facts("") == []
        assert extract_facts(None) == []

    def test_returns_list(self):
        result = extract_facts("Machine learning is a subset of AI.")
        assert isinstance(result, list)


class TestStripHtml:
    """Test HTML stripping."""

    def test_strips_tags(self):
        assert "hello world" in strip_html("<p>hello <b>world</b></p>")

    def test_handles_plain_text(self):
        assert strip_html("no tags here") == "no tags here"

    def test_handles_empty(self):
        assert strip_html("") == ""

    def test_collapses_whitespace(self):
        result = strip_html("<p>a</p>  <p>b</p>")
        assert "  " not in result


class TestBuildTags:
    """Test tag building."""

    def test_basic_tags(self):
        tags = _build_tags("Machine Learning")
        assert "wikipedia" in tags
        assert "machine_learning" in tags

    def test_with_topic(self):
        tags = _build_tags("Neural Network", topic="deep learning")
        assert "deep_learning" in tags

    def test_with_section(self):
        tags = _build_tags("AI", section="History")
        assert any("section:" in t for t in tags)

    def test_no_topic_or_section(self):
        tags = _build_tags("Test")
        assert len(tags) == 2  # "wikipedia" + title


class TestDefaultTopics:
    """Verify the default topic list."""

    def test_has_topics(self):
        assert len(DEFAULT_TOPICS) >= 10

    def test_includes_ai_topics(self):
        topic_str = " ".join(DEFAULT_TOPICS).lower()
        assert "artificial intelligence" in topic_str
        assert "machine learning" in topic_str
        assert "deep learning" in topic_str


class TestWikipediaCollectorInit:
    """Test collector initialization."""

    def test_init_with_defaults(self):
        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb)
        assert collector.knowledge_base is kb
        assert collector.confidence == 0.85
        assert collector.rate_limit_delay == 1.0

    def test_init_custom_params(self):
        kb = KnowledgeBase()
        collector = WikipediaCollector(
            knowledge_base=kb,
            rate_limit_delay=0.1,
            confidence=0.9,
            min_fact_length=50,
        )
        assert collector.rate_limit_delay == 0.1
        assert collector.confidence == 0.9
        assert collector.min_fact_length == 50


class TestWikipediaCollectorSearchArticles:
    """Test article search with mocked HTTP."""

    @patch("nexus.data.wikipedia_collector.requests.get")
    def test_search_returns_titles(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            "machine learning",
            ["Machine learning", "Machine learning in bioinformatics"],
            ["", ""],
            ["url1", "url2"],
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb, rate_limit_delay=0)
        titles = collector.search_articles("machine learning", max_results=5)
        assert "Machine learning" in titles

    @patch("nexus.data.wikipedia_collector.requests.get")
    def test_search_returns_empty_on_error(self, mock_get):
        mock_get.side_effect = Exception("network error")

        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb, rate_limit_delay=0)
        titles = collector.search_articles("test")
        assert titles == []


class TestWikipediaCollectorCollectArticle:
    """Test single article collection with mocked HTTP."""

    @patch("nexus.data.wikipedia_collector.requests.get")
    def test_collect_article_stores_facts(self, mock_get):
        summary_resp = MagicMock()
        summary_resp.status_code = 200
        summary_resp.json.return_value = {
            "type": "standard",
            "title": "Machine learning",
            "extract": (
                "Machine learning is a subset of artificial intelligence. "
                "It was developed from computational learning theory. "
                "ML algorithms are used in many applications."
            ),
        }

        sections_resp = MagicMock()
        sections_resp.status_code = 404  # Skip sections for simplicity

        mock_get.side_effect = [summary_resp, sections_resp]

        kb = KnowledgeBase()
        collector = WikipediaCollector(
            knowledge_base=kb, rate_limit_delay=0, min_fact_length=20
        )
        result = collector.collect_article("Machine learning", "AI")

        assert result is not None
        assert result["facts"] >= 1
        assert len(result["knowledge_ids"]) >= 1
        # Verify KB actually has the items
        assert len(kb.knowledge_items) >= 1

    @patch("nexus.data.wikipedia_collector.requests.get")
    def test_collect_article_skips_disambiguation(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"type": "disambiguation", "title": "AI"}
        mock_get.return_value = mock_resp

        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb, rate_limit_delay=0)
        result = collector.collect_article("AI")
        assert result is None

    @patch("nexus.data.wikipedia_collector.requests.get")
    def test_collect_article_returns_none_on_failure(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb, rate_limit_delay=0)
        result = collector.collect_article("Nonexistent Article XYZZY")
        assert result is None


class TestWikipediaCollectorCollectTopics:
    """Test bulk topic collection."""

    @patch.object(WikipediaCollector, "search_articles")
    @patch.object(WikipediaCollector, "collect_article")
    @patch.object(WikipediaCollector, "get_related_titles")
    def test_collect_topics_aggregates_stats(
        self, mock_related, mock_collect, mock_search
    ):
        mock_search.return_value = ["Article 1"]
        mock_collect.return_value = {"facts": 3, "knowledge_ids": ["k1", "k2", "k3"]}
        mock_related.return_value = []

        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb, rate_limit_delay=0)
        stats = collector.collect_topics(
            topics=["test topic"], max_articles_per_topic=2
        )

        assert stats["topics_processed"] == 1
        assert stats["articles_collected"] == 1
        assert stats["facts_extracted"] == 3
        assert len(stats["knowledge_ids"]) == 3

    @patch.object(WikipediaCollector, "search_articles")
    @patch.object(WikipediaCollector, "collect_article")
    @patch.object(WikipediaCollector, "get_related_titles")
    def test_collect_topics_handles_errors(
        self, mock_related, mock_collect, mock_search
    ):
        mock_search.side_effect = Exception("API error")

        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb, rate_limit_delay=0)
        stats = collector.collect_topics(topics=["broken topic"])

        assert stats["errors"] == 1
        assert stats["articles_collected"] == 0

    @patch.object(WikipediaCollector, "search_articles")
    @patch.object(WikipediaCollector, "collect_article")
    @patch.object(WikipediaCollector, "get_related_titles")
    def test_deduplicates_collected_titles(
        self, mock_related, mock_collect, mock_search
    ):
        # Return same title for two topics
        mock_search.return_value = ["Same Article"]
        mock_collect.return_value = {"facts": 1, "knowledge_ids": ["k1"]}
        mock_related.return_value = []

        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb, rate_limit_delay=0)
        stats = collector.collect_topics(topics=["topic1", "topic2"])

        # collect_article should only be called once for "Same Article"
        assert mock_collect.call_count == 1
        assert stats["articles_collected"] == 1


class TestWikipediaCollectorStats:
    """Test collection stats."""

    def test_empty_stats(self):
        kb = KnowledgeBase()
        collector = WikipediaCollector(knowledge_base=kb)
        stats = collector.get_collection_stats()
        assert stats["articles_collected"] == 0
        assert stats["collected_titles"] == []
