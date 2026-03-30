"""
Wikipedia Knowledge Collector for Nexus AI Platform.

Systematically harvests knowledge from Wikipedia articles,
extracting structured facts and storing them in the KnowledgeBase.

Unlike InternetKnowledgeRetriever._search_wikipedia (single query → summary),
this collector does bulk article harvesting with:
- Topic-based collection with related article traversal
- Full article section extraction (not just summaries)
- Category-based discovery of related articles
- Structured fact extraction with confidence scoring
- Deduplication against existing KB entries
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote_plus

import requests

from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType

logger = logging.getLogger(__name__)

# Default AI/ML topics to seed collection
DEFAULT_TOPICS = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "neural network",
    "transformer (deep learning architecture)",
    "large language model",
    "generative adversarial network",
    "convolutional neural network",
    "recurrent neural network",
    "attention (machine learning)",
    "transfer learning",
    "federated learning",
    "knowledge graph",
    "semantic web",
    "information retrieval",
    "data mining",
    "statistical classification",
]


class WikipediaCollector:
    """
    Systematic Wikipedia knowledge harvester.

    Collects articles by topic, extracts facts from sections,
    follows related links, and stores everything in KnowledgeBase.
    """

    SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
    SECTIONS_API = "https://en.wikipedia.org/api/rest_v1/page/mobile-sections"
    SEARCH_API = "https://en.wikipedia.org/w/api.php"

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        rate_limit_delay: float = 1.0,
        min_fact_length: int = 30,
        max_fact_length: int = 300,
        confidence: float = 0.85,
    ):
        self.knowledge_base = knowledge_base
        self.rate_limit_delay = rate_limit_delay
        self.min_fact_length = min_fact_length
        self.max_fact_length = max_fact_length
        self.confidence = confidence
        self._last_request_time = 0.0
        self._collected_titles: Set[str] = set()

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def collect_topics(
        self,
        topics: Optional[List[str]] = None,
        max_articles_per_topic: int = 5,
        follow_related: bool = True,
    ) -> Dict[str, Any]:
        """
        Collect knowledge from Wikipedia for a list of topics.

        Args:
            topics: List of topic strings to search. Defaults to AI/ML topics.
            max_articles_per_topic: Max articles to collect per topic.
            follow_related: Whether to also collect related articles.

        Returns:
            Collection statistics.
        """
        topics = topics or DEFAULT_TOPICS
        stats = {
            "topics_processed": 0,
            "articles_collected": 0,
            "facts_extracted": 0,
            "knowledge_ids": [],
            "errors": 0,
        }

        for topic in topics:
            try:
                topic_result = self._collect_topic(
                    topic, max_articles_per_topic, follow_related
                )
                stats["topics_processed"] += 1
                stats["articles_collected"] += topic_result["articles"]
                stats["facts_extracted"] += topic_result["facts"]
                stats["knowledge_ids"].extend(topic_result["knowledge_ids"])
            except Exception as e:
                logger.error(f"Error collecting topic '{topic}': {e}")
                stats["errors"] += 1

        logger.info(
            f"Wikipedia collection complete: {stats['articles_collected']} articles, "
            f"{stats['facts_extracted']} facts from {stats['topics_processed']} topics"
        )
        return stats

    def _collect_topic(
        self, topic: str, max_articles: int, follow_related: bool
    ) -> Dict[str, Any]:
        """Collect articles for a single topic."""
        result = {"articles": 0, "facts": 0, "knowledge_ids": []}

        # Search for articles matching the topic
        titles = self.search_articles(topic, max_results=max_articles)

        for title in titles:
            if title in self._collected_titles:
                continue

            article_result = self.collect_article(title, topic)
            if article_result:
                self._collected_titles.add(title)
                result["articles"] += 1
                result["facts"] += article_result["facts"]
                result["knowledge_ids"].extend(article_result["knowledge_ids"])

            # Optionally follow related articles (1 level deep)
            if follow_related and result["articles"] < max_articles:
                related = self.get_related_titles(title, max_results=3)
                for rel_title in related:
                    if rel_title in self._collected_titles:
                        continue
                    if result["articles"] >= max_articles:
                        break
                    rel_result = self.collect_article(rel_title, topic)
                    if rel_result:
                        self._collected_titles.add(rel_title)
                        result["articles"] += 1
                        result["facts"] += rel_result["facts"]
                        result["knowledge_ids"].extend(rel_result["knowledge_ids"])

        return result

    def search_articles(self, query: str, max_results: int = 5) -> List[str]:
        """
        Search Wikipedia for article titles matching a query.

        Uses the MediaWiki opensearch API for title suggestions.
        """
        self._rate_limit()
        try:
            response = requests.get(
                self.SEARCH_API,
                params={
                    "action": "opensearch",
                    "search": query,
                    "limit": max_results,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            # opensearch returns [query, [titles], [descriptions], [urls]]
            if len(data) >= 2 and isinstance(data[1], list):
                return data[1][:max_results]
        except Exception as e:
            logger.warning(f"Wikipedia search failed for '{query}': {e}")
        return []

    def get_article_summary(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch article summary from the REST API."""
        self._rate_limit()
        try:
            url = f"{self.SUMMARY_API}/{quote_plus(title)}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get summary for '{title}': {e}")
        return None

    def get_article_sections(self, title: str) -> Optional[Dict[str, Any]]:
        """Fetch full article sections from the mobile-sections API."""
        self._rate_limit()
        try:
            url = f"{self.SECTIONS_API}/{quote_plus(title)}"
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Failed to get sections for '{title}': {e}")
        return None

    def get_related_titles(self, title: str, max_results: int = 5) -> List[str]:
        """Get titles of related articles using the related pages API."""
        self._rate_limit()
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/related/{quote_plus(title)}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [
                    p["title"]
                    for p in data.get("pages", [])[:max_results]
                    if "title" in p
                ]
        except Exception as e:
            logger.debug(f"Failed to get related for '{title}': {e}")
        return []

    def collect_article(
        self, title: str, topic_context: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Collect a single Wikipedia article and store facts in the KB.

        Args:
            title: Article title.
            topic_context: Parent topic for context tagging.

        Returns:
            Dict with facts count and knowledge_ids, or None on failure.
        """
        summary = self.get_article_summary(title)
        if not summary or summary.get("type") == "disambiguation":
            return None

        result = {"facts": 0, "knowledge_ids": []}

        # Extract facts from the summary extract
        extract = summary.get("extract", "")
        if extract:
            facts = extract_facts(extract, self.min_fact_length, self.max_fact_length)
            for fact in facts:
                kid = self.knowledge_base.add_knowledge(
                    content=fact,
                    knowledge_type=KnowledgeType.FACTUAL,
                    source=f"wikipedia:{title}",
                    confidence=self.confidence,
                    context_tags=_build_tags(title, topic_context),
                )
                result["knowledge_ids"].append(kid)
                result["facts"] += 1

        # Try to get section content for deeper extraction
        sections_data = self.get_article_sections(title)
        if sections_data:
            remaining = sections_data.get("remaining", {}).get("sections", [])
            for section in remaining:
                section_text = strip_html(section.get("text", ""))
                if not section_text or len(section_text) < 50:
                    continue
                section_heading = section.get("line", "")
                # Skip non-content sections
                if section_heading.lower() in (
                    "references", "external links", "see also",
                    "further reading", "notes", "bibliography",
                ):
                    continue

                section_facts = extract_facts(
                    section_text, self.min_fact_length, self.max_fact_length
                )
                for fact in section_facts:
                    kid = self.knowledge_base.add_knowledge(
                        content=fact,
                        knowledge_type=KnowledgeType.FACTUAL,
                        source=f"wikipedia:{title}#{section_heading}",
                        confidence=self.confidence - 0.05,  # Slightly lower for sections
                        context_tags=_build_tags(title, topic_context, section_heading),
                    )
                    result["knowledge_ids"].append(kid)
                    result["facts"] += 1

        if result["facts"] > 0:
            logger.info(f"Collected {result['facts']} facts from '{title}'")

        return result if result["facts"] > 0 else None

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return statistics about collected articles."""
        return {
            "articles_collected": len(self._collected_titles),
            "collected_titles": sorted(self._collected_titles),
        }


def extract_facts(
    text: str, min_length: int = 30, max_length: int = 300
) -> List[str]:
    """
    Extract factual sentences from text.

    Splits by sentence boundaries, filters by length and quality.
    """
    if not text:
        return []

    # Split on sentence-ending punctuation followed by space or end
    sentences = []
    current = []
    for char in text:
        current.append(char)
        if char in ".!?" and len(current) > 1:
            sentence = "".join(current).strip()
            if sentence:
                sentences.append(sentence)
            current = []
    # Catch remainder
    if current:
        remainder = "".join(current).strip()
        if remainder:
            sentences.append(remainder)

    facts = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < min_length or len(sentence) > max_length:
            continue
        # Must contain a verb-like indicator (factual assertion)
        lower = sentence.lower()
        if not any(w in lower for w in ("is", "are", "was", "were", "has", "have", "can", "does")):
            continue
        # Skip hedging / uncertain language
        if any(w in lower for w in ("may be", "might", "possibly", "arguably")):
            continue
        facts.append(sentence)

    return facts


def strip_html(text: str) -> str:
    """Remove HTML tags from text, returning plain text."""
    import re
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def _build_tags(
    title: str, topic: str = "", section: str = ""
) -> List[str]:
    """Build context tags for a knowledge item."""
    tags = ["wikipedia", title.lower().replace(" ", "_")]
    if topic:
        tags.append(topic.lower().replace(" ", "_"))
    if section:
        tags.append(f"section:{section.lower().replace(' ', '_')}")
    return tags
