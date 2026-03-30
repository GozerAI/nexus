"""
Query classification and understanding module.

This module analyzes user queries to determine:
1. Query type (factual, creative, technical, etc.)
2. Domain (science, math, business, etc.)
3. Intent (question, command, statement)
4. Complexity and required capabilities
5. Optimal model selection

Based on combo1's query classifier with enhancements.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Set

from nexus.providers.ensemble.types import QueryType
from nexus.providers.adapters.base import ModelCapability


class QueryIntent(str, Enum):
    """Query intent classification."""

    QUESTION = "question"
    COMMAND = "command"
    STATEMENT = "statement"
    CONVERSATION = "conversation"


class QueryDomain(str, Enum):
    """Domain classification."""

    GENERAL = "general"
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    MATHEMATICS = "mathematics"
    BUSINESS = "business"
    HEALTH = "health"
    ARTS = "arts"
    EDUCATION = "education"
    LEGAL = "legal"
    FINANCE = "finance"


class QueryComplexity(str, Enum):
    """Complexity level."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class QueryAnalysis:
    """
    Comprehensive query analysis results.

    Attributes:
        query_type: Classified query type
        intent: Query intent (question, command, etc.)
        domain: Primary domain
        complexity: Complexity level
        required_capabilities: Required model capabilities
        recommended_models: Suggested model types
        key_terms: Important keywords
        has_context: Whether query references context
        estimated_tokens: Estimated input tokens
        confidence: Classification confidence (0-1)
    """

    query_type: QueryType
    intent: QueryIntent
    domain: QueryDomain
    complexity: QueryComplexity
    required_capabilities: List[ModelCapability]
    recommended_models: List[str]
    key_terms: List[str]
    has_context: bool
    estimated_tokens: int
    confidence: float


class QueryClassifier:
    """
    Classifies and analyzes user queries.

    The classification process:
    1. Extract linguistic features (keywords, patterns)
    2. Classify query type (factual, creative, etc.)
    3. Determine intent (question vs. command)
    4. Identify domain and complexity
    5. Recommend capabilities and models

    Features:
    - Pattern-based classification
    - Keyword matching
    - NLP feature extraction
    - Multi-factor analysis
    - Confidence scoring
    """

    def __init__(self):
        """Initialize query classifier with pattern definitions."""
        # Define patterns for each query type
        self._patterns = self._initialize_patterns()

        # Define domain keywords
        self._domain_keywords = self._initialize_domain_keywords()

    def classify(self, query: str) -> QueryAnalysis:
        """
        Classify and analyze a query.

        Args:
            query: User query text

        Returns:
            Comprehensive query analysis
        """
        query_lower = query.lower().strip()

        # Extract features
        key_terms = self._extract_key_terms(query)
        intent = self._classify_intent(query_lower)
        complexity = self._estimate_complexity(query)
        domain = self._classify_domain(query_lower, key_terms)

        # Classify query type
        query_type, type_confidence = self._classify_type(query_lower, key_terms)

        # Determine required capabilities
        capabilities = self._determine_capabilities(query_type, domain, complexity)

        # Recommend models
        recommended = self._recommend_models(query_type, capabilities, complexity)

        # Check for context references
        has_context = self._has_context_reference(query_lower)

        # Estimate tokens
        estimated_tokens = len(query.split()) * 1.3  # Rough estimate

        return QueryAnalysis(
            query_type=query_type,
            intent=intent,
            domain=domain,
            complexity=complexity,
            required_capabilities=capabilities,
            recommended_models=recommended,
            key_terms=key_terms,
            has_context=has_context,
            estimated_tokens=int(estimated_tokens),
            confidence=type_confidence,
        )

    def _classify_type(self, query: str, key_terms: List[str]) -> tuple[QueryType, float]:
        """
        Classify query type with confidence.

        Args:
            query: Normalized query text
            key_terms: Extracted keywords

        Returns:
            Tuple of (query_type, confidence)
        """
        scores = {query_type: 0.0 for query_type in QueryType}

        # Score each type based on patterns
        for query_type, patterns in self._patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    scores[QueryType(query_type)] += 1.0

        # Keyword-based scoring
        for term in key_terms:
            if term in ['forecast', 'predict', 'future', 'trend']:
                scores[QueryType.FORECASTING] += 0.5
            elif term in ['code', 'function', 'program', 'debug']:
                scores[QueryType.TECHNICAL] += 0.5
            elif term in ['story', 'poem', 'creative', 'imagine']:
                scores[QueryType.CREATIVE] += 0.5
            elif term in ['analyze', 'compare', 'evaluate']:
                scores[QueryType.ANALYTICAL] += 0.5

        # Determine winner
        if max(scores.values()) == 0:
            # Default to conversational
            return QueryType.CONVERSATIONAL, 0.5

        best_type = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_type] / 3.0)  # Normalize

        return best_type, confidence

    def _classify_intent(self, query: str) -> QueryIntent:
        """
        Classify query intent.

        Args:
            query: Normalized query text

        Returns:
            Query intent
        """
        # Question indicators
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'should', 'would', 'could']
        if any(query.startswith(word) for word in question_words) or '?' in query:
            return QueryIntent.QUESTION

        # Command indicators
        command_words = ['create', 'generate', 'write', 'make', 'build', 'show', 'explain', 'describe', 'tell']
        if any(query.startswith(word) for word in command_words):
            return QueryIntent.COMMAND

        # Conversational indicators
        greeting_words = ['hello', 'hi', 'hey', 'thanks', 'thank you']
        if any(word in query for word in greeting_words):
            return QueryIntent.CONVERSATION

        # Default to statement
        return QueryIntent.STATEMENT

    def _classify_domain(self, query: str, key_terms: List[str]) -> QueryDomain:
        """
        Classify domain.

        Args:
            query: Normalized query text
            key_terms: Extracted keywords

        Returns:
            Primary domain
        """
        domain_scores = {domain: 0 for domain in QueryDomain}

        # Score by keyword matches
        for domain, keywords in self._domain_keywords.items():
            for keyword in keywords:
                if keyword in query or keyword in ' '.join(key_terms):
                    domain_scores[QueryDomain(domain)] += 1

        # Determine winner
        if max(domain_scores.values()) == 0:
            return QueryDomain.GENERAL

        return max(domain_scores, key=domain_scores.get)

    def _estimate_complexity(self, query: str) -> QueryComplexity:
        """
        Estimate query complexity.

        Factors:
        - Length (longer = more complex)
        - Technical terms
        - Multiple questions
        - Conditional logic

        Args:
            query: Query text

        Returns:
            Complexity level
        """
        score = 0

        # Length factor
        word_count = len(query.split())
        if word_count > 100:
            score += 3
        elif word_count > 50:
            score += 2
        elif word_count > 20:
            score += 1

        # Multiple questions
        question_marks = query.count('?')
        if question_marks > 1:
            score += 2

        # Technical indicators
        if re.search(r'[A-Z]{2,}', query):  # Acronyms
            score += 1
        if re.search(r'\d+', query):  # Numbers
            score += 1

        # Conditional logic
        conditional_words = ['if', 'when', 'unless', 'provided that', 'assuming']
        for word in conditional_words:
            if word in query.lower():
                score += 1

        # Map score to complexity
        if score >= 7:
            return QueryComplexity.VERY_COMPLEX
        elif score >= 4:
            return QueryComplexity.COMPLEX
        elif score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE

    def _determine_capabilities(
        self, query_type: QueryType, domain: QueryDomain, complexity: QueryComplexity
    ) -> List[ModelCapability]:
        """
        Determine required model capabilities.

        Args:
            query_type: Query type
            domain: Query domain
            complexity: Complexity level

        Returns:
            List of required capabilities
        """
        capabilities = [ModelCapability.TEXT_GENERATION]

        # Add based on query type
        if query_type == QueryType.TECHNICAL:
            capabilities.append(ModelCapability.CODE_GENERATION)
            capabilities.append(ModelCapability.REASONING)
        elif query_type == QueryType.CREATIVE:
            # No special capabilities needed
            pass
        elif query_type == QueryType.ANALYTICAL:
            capabilities.append(ModelCapability.REASONING)
            capabilities.append(ModelCapability.MATHEMATICS)
        elif query_type == QueryType.FORECASTING:
            capabilities.append(ModelCapability.REASONING)
            capabilities.append(ModelCapability.MATHEMATICS)

        # Add based on domain
        if domain in [QueryDomain.SCIENCE, QueryDomain.MATHEMATICS]:
            if ModelCapability.MATHEMATICS not in capabilities:
                capabilities.append(ModelCapability.MATHEMATICS)
            if ModelCapability.REASONING not in capabilities:
                capabilities.append(ModelCapability.REASONING)

        # Add based on complexity
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            if ModelCapability.REASONING not in capabilities:
                capabilities.append(ModelCapability.REASONING)
            if ModelCapability.LONG_CONTEXT not in capabilities:
                capabilities.append(ModelCapability.LONG_CONTEXT)

        return capabilities

    def _recommend_models(
        self, query_type: QueryType, capabilities: List[ModelCapability], complexity: QueryComplexity
    ) -> List[str]:
        """
        Recommend specific models for this query.

        Args:
            query_type: Query type
            capabilities: Required capabilities
            complexity: Complexity level

        Returns:
            List of recommended model names
        """
        recommended = []

        # For complex queries, recommend flagship models
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            recommended.extend(['gpt-4-turbo', 'claude-3-opus', 'gemini-ultra'])

        # For code tasks
        elif ModelCapability.CODE_GENERATION in capabilities:
            recommended.extend(['gpt-4-turbo', 'claude-3-opus', 'code-llama-34b', 'deepseek-coder-33b'])

        # For creative tasks
        elif query_type == QueryType.CREATIVE:
            recommended.extend(['claude-3-opus', 'gpt-4', 'dolphin-mixtral'])

        # For factual/simple queries
        elif query_type in [QueryType.FACTUAL, QueryType.QA]:
            recommended.extend(['gpt-3.5-turbo', 'claude-3-haiku', 'gemini-pro'])

        # For forecasting
        elif query_type == QueryType.FORECASTING:
            recommended.extend(['gpt-4-turbo', 'claude-3-opus'])

        # Default recommendations
        else:
            recommended.extend(['gpt-4-turbo', 'claude-3-sonnet', 'gpt-3.5-turbo'])

        return recommended[:5]  # Limit to top 5

    def _extract_key_terms(self, query: str) -> List[str]:
        """
        Extract important keywords from query.

        Args:
            query: Query text

        Returns:
            List of key terms
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
            'of', 'to', 'in', 'for', 'on', 'with', 'as', 'by', 'from', 'at'
        }

        words = query.lower().split()
        key_terms = [w for w in words if w not in stop_words and len(w) > 3]

        return key_terms[:10]  # Limit to top 10

    def _has_context_reference(self, query: str) -> bool:
        """
        Check if query references previous context.

        Args:
            query: Normalized query text

        Returns:
            True if has context reference
        """
        context_indicators = [
            'this', 'that', 'these', 'those', 'above', 'previous', 'earlier',
            'mentioned', 'said', 'discussed', 'it', 'they', 'them'
        ]

        return any(indicator in query for indicator in context_indicators)

    def _initialize_patterns(self) -> dict:
        """Initialize regex patterns for query types."""
        return {
            'factual': [
                r'^what is',
                r'^what are',
                r'^define',
                r'^who is',
                r'^when did',
                r'^where is',
                r'capital of',
                r'population of',
            ],
            'creative': [
                r'write a (story|poem|song)',
                r'create a',
                r'imagine',
                r'describe a fictional',
                r'generate a creative',
            ],
            'technical': [
                r'write code',
                r'program',
                r'function',
                r'implement',
                r'debug',
                r'fix (the )?(bug|error)',
                r'how do i code',
            ],
            'analytical': [
                r'analyze',
                r'compare',
                r'evaluate',
                r'assess',
                r'what are the (pros|cons)',
                r'advantages and disadvantages',
            ],
            'summarization': [
                r'summarize',
                r'tldr',
                r'in brief',
                r'key points',
                r'main ideas',
            ],
            'translation': [
                r'translate',
                r'in (spanish|french|german|chinese)',
                r'how do you say',
            ],
            'qa': [
                r'\?$',
                r'^can you',
                r'^could you',
                r'^would you',
            ],
            'forecasting': [
                r'forecast',
                r'predict',
                r'what will',
                r'future (of|trends)',
                r'estimate',
                r'projection',
            ],
        }

    def _initialize_domain_keywords(self) -> dict:
        """Initialize domain keywords."""
        return {
            'technology': ['computer', 'software', 'code', 'algorithm', 'data', 'ai', 'programming'],
            'science': ['experiment', 'hypothesis', 'theory', 'research', 'study', 'scientific'],
            'mathematics': ['equation', 'calculate', 'formula', 'theorem', 'proof', 'derivative'],
            'business': ['revenue', 'profit', 'strategy', 'market', 'sales', 'customer'],
            'health': ['medical', 'disease', 'treatment', 'health', 'symptom', 'diagnosis'],
            'finance': ['investment', 'stock', 'portfolio', 'trading', 'bonds', 'interest'],
            'legal': ['law', 'legal', 'contract', 'court', 'statute', 'regulation'],
            'education': ['learn', 'teach', 'study', 'course', 'lesson', 'education'],
            'arts': ['art', 'music', 'painting', 'literature', 'creative', 'artistic'],
        }


def classify_query(query: str) -> QueryAnalysis:
    """
    Convenience function to classify a query with default settings.

    Args:
        query: User query text

    Returns:
        Query analysis results
    """
    classifier = QueryClassifier()
    return classifier.classify(query)
