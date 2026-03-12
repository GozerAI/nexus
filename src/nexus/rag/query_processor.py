"""
Query Processing - Preprocess and expand queries for better retrieval.

Provides:
- Query cleaning and normalization
- Query expansion (synonyms, related terms)
- Multi-query generation for better recall
- Query decomposition for complex questions
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ProcessedQuery:
    """Result of query processing."""

    original: str
    cleaned: str
    expanded_queries: List[str]
    keywords: List[str]
    intent: str  # "factual", "conceptual", "procedural", "comparison"
    metadata: Dict[str, Any]


class QueryProcessor:
    """
    Preprocesses queries for better retrieval.

    Steps:
    1. Clean and normalize
    2. Extract keywords
    3. Detect intent
    4. Generate expansions
    """

    # Common words to filter
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "what", "which", "who",
        "when", "where", "why", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "such", "no", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "also",
    }

    # Intent indicators
    INTENT_PATTERNS = {
        "factual": [
            r"^what is\b", r"^who is\b", r"^when did\b", r"^where is\b",
            r"^define\b", r"^list\b", r"^name\b",
        ],
        "procedural": [
            r"^how to\b", r"^how do\b", r"^how can\b", r"^steps to\b",
            r"^guide\b", r"^tutorial\b", r"^explain how\b",
        ],
        "conceptual": [
            r"^why\b", r"^explain\b", r"^describe\b", r"^what are the\b",
            r"^understand\b", r"^concept\b",
        ],
        "comparison": [
            r"^compare\b", r"^difference between\b", r"^vs\b", r"versus",
            r"^which is better\b", r"^pros and cons\b",
        ],
    }

    def __init__(
        self,
        expand_queries: bool = True,
        max_expansions: int = 3,
        use_synonyms: bool = True,
    ):
        self.expand_queries = expand_queries
        self.max_expansions = max_expansions
        self.use_synonyms = use_synonyms

        # Simple synonym mapping (could be extended with WordNet, embeddings, etc.)
        self.synonyms = {
            "error": ["bug", "issue", "problem", "exception"],
            "create": ["make", "build", "generate", "construct"],
            "delete": ["remove", "drop", "eliminate", "clear"],
            "update": ["modify", "change", "edit", "revise"],
            "fast": ["quick", "rapid", "efficient", "performant"],
            "slow": ["sluggish", "laggy", "delayed"],
            "install": ["setup", "configure", "deploy"],
            "function": ["method", "procedure", "routine"],
            "api": ["interface", "endpoint", "service"],
            "database": ["db", "datastore", "storage"],
        }

    def process(self, query: str) -> ProcessedQuery:
        """
        Process a query for retrieval.

        Args:
            query: Raw user query

        Returns:
            ProcessedQuery with cleaned query, expansions, etc.
        """
        # Clean query
        cleaned = self._clean_query(query)

        # Extract keywords
        keywords = self._extract_keywords(cleaned)

        # Detect intent
        intent = self._detect_intent(query.lower())

        # Generate expansions
        expanded = []
        if self.expand_queries:
            expanded = self._expand_query(cleaned, keywords)

        return ProcessedQuery(
            original=query,
            cleaned=cleaned,
            expanded_queries=expanded,
            keywords=keywords,
            intent=intent,
            metadata={
                "word_count": len(cleaned.split()),
                "has_question_mark": "?" in query,
            },
        )

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Lowercase
        text = query.lower()

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r"[^\w\s\-]", " ", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        words = query.split()

        # Filter stop words and short words
        keywords = [
            word for word in words
            if word not in self.STOP_WORDS and len(word) > 2
        ]

        return keywords

    def _detect_intent(self, query: str) -> str:
        """Detect query intent from patterns."""
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent

        # Default to conceptual
        return "conceptual"

    def _expand_query(self, query: str, keywords: List[str]) -> List[str]:
        """Generate query expansions."""
        expansions = []

        # 1. Synonym expansion
        if self.use_synonyms:
            for keyword in keywords:
                if keyword in self.synonyms:
                    for synonym in self.synonyms[keyword][:2]:
                        expanded = query.replace(keyword, synonym)
                        if expanded not in expansions and expanded != query:
                            expansions.append(expanded)

        # 2. Keyword subset queries
        if len(keywords) > 3:
            # Create shorter queries from key terms
            expansions.append(" ".join(keywords[:3]))

        # Limit expansions
        return expansions[:self.max_expansions]


class MultiQueryGenerator:
    """
    Generate multiple query perspectives for better recall.

    Uses an LLM to reformulate the query from different angles.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def generate(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple query perspectives.

        Args:
            query: Original query
            num_queries: Number of alternative queries

        Returns:
            List of query variations
        """
        if not self.llm_client:
            # Fallback: simple variations
            return self._simple_variations(query, num_queries)

        prompt = f"""Generate {num_queries} different ways to ask the following question.
Each variation should approach the topic from a slightly different angle
to help find relevant information.

Original question: {query}

Return only the questions, one per line, without numbering."""

        try:
            response = self.llm_client.generate(prompt)
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            return queries[:num_queries]
        except Exception as e:
            logger.warning(f"Multi-query generation failed: {e}")
            return self._simple_variations(query, num_queries)

    def _simple_variations(self, query: str, num_queries: int) -> List[str]:
        """Generate simple variations without LLM."""
        variations = []

        # Remove question words and try different forms
        query_lower = query.lower()

        # "What is X" -> "X definition", "X explained"
        if query_lower.startswith("what is "):
            topic = query[8:].rstrip("?")
            variations.extend([
                f"{topic} definition",
                f"{topic} explained",
                f"explain {topic}",
            ])

        # "How to X" -> "X tutorial", "X guide"
        elif query_lower.startswith("how to "):
            topic = query[7:].rstrip("?")
            variations.extend([
                f"{topic} tutorial",
                f"{topic} guide",
                f"steps to {topic}",
            ])

        # "Why X" -> "reasons for X", "X causes"
        elif query_lower.startswith("why "):
            topic = query[4:].rstrip("?")
            variations.extend([
                f"reasons for {topic}",
                f"{topic} explanation",
            ])

        return variations[:num_queries]


class QueryDecomposer:
    """
    Decompose complex queries into sub-queries.

    For questions like "Compare X and Y, and explain how they relate to Z",
    breaks into:
    1. "What is X?"
    2. "What is Y?"
    3. "Comparison of X and Y"
    4. "How X relates to Z"
    5. "How Y relates to Z"
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def decompose(self, query: str) -> List[str]:
        """
        Decompose a complex query into sub-queries.

        Args:
            query: Complex query

        Returns:
            List of simpler sub-queries
        """
        if not self.llm_client:
            return self._simple_decompose(query)

        prompt = f"""Break down this complex question into simpler sub-questions
that can be answered individually. Each sub-question should be self-contained.

Question: {query}

Return only the sub-questions, one per line, without numbering.
If the question is already simple, return just the original question."""

        try:
            response = self.llm_client.generate(prompt)
            queries = [q.strip() for q in response.strip().split("\n") if q.strip()]
            return queries if queries else [query]
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return self._simple_decompose(query)

    def _simple_decompose(self, query: str) -> List[str]:
        """Simple decomposition without LLM."""
        sub_queries = []

        # Split on "and", "also", "additionally"
        parts = re.split(r"\s+and\s+|\s+also\s+|\s+additionally\s+", query, flags=re.IGNORECASE)

        if len(parts) > 1:
            for part in parts:
                part = part.strip().rstrip("?").strip()
                if len(part) > 10:  # Minimum length for valid query
                    sub_queries.append(part + "?")

        # If no decomposition possible, return original
        return sub_queries if sub_queries else [query]


class HyDEQueryExpander:
    """
    Hypothetical Document Embeddings (HyDE) query expansion.

    Generates a hypothetical answer to the query, then uses that
    for retrieval. Often improves recall significantly.
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def expand(self, query: str) -> str:
        """
        Generate hypothetical document for query.

        Args:
            query: User query

        Returns:
            Hypothetical document text
        """
        if not self.llm_client:
            # Without LLM, just return the query
            return query

        prompt = f"""Write a short paragraph that directly answers this question.
The answer should be factual and informative, as if from a reference document.

Question: {query}

Answer:"""

        try:
            response = self.llm_client.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"HyDE expansion failed: {e}")
            return query


def create_query_processor(
    expand: bool = True,
    multi_query: bool = False,
    decompose: bool = False,
    hyde: bool = False,
    llm_client=None,
) -> Dict[str, Any]:
    """
    Create query processing pipeline.

    Args:
        expand: Enable synonym/keyword expansion
        multi_query: Enable multi-query generation
        decompose: Enable query decomposition
        hyde: Enable HyDE expansion
        llm_client: LLM client for advanced features

    Returns:
        Dict with processor components
    """
    return {
        "processor": QueryProcessor(expand_queries=expand),
        "multi_query": MultiQueryGenerator(llm_client) if multi_query else None,
        "decomposer": QueryDecomposer(llm_client) if decompose else None,
        "hyde": HyDEQueryExpander(llm_client) if hyde else None,
    }
