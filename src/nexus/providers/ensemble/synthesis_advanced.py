"""
Advanced Response Synthesis

Sophisticated multi-model response synthesis with:
- Sentence-level response combination
- Jaccard similarity for deduplication
- Quality-based sentence selection
- Coherence checking
- Transition generation
- Citation/source attribution
- Multiple synthesis strategies

Phase 5 Week 23

Example:
    >>> from unified_intelligence.ensemble import AdvancedResponseSynthesizer, SynthesisStrategy
    >>> from nexus.providers.ensemble.types import ModelResponse
    >>>
    >>> # Initialize synthesizer
    >>> synthesizer = AdvancedResponseSynthesizer()
    >>>
    >>> # Synthesize responses
    >>> responses = [response1, response2, response3]
    >>> synthesized = synthesizer.synthesize(
    ...     responses,
    ...     strategy=SynthesisStrategy.QUALITY_WEIGHTED
    ... )
"""
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import statistics

from nexus.providers.ensemble.types import ModelResponse

logger = logging.getLogger(__name__)


class SynthesisStrategy(str, Enum):
    """Response synthesis strategies."""
    BEST_SENTENCE = "best_sentence"  # Select best sentence from each position
    QUALITY_WEIGHTED = "quality_weighted"  # Weight by quality scores
    DIVERSITY_MAXIMIZING = "diversity_maximizing"  # Maximize information diversity
    CONSENSUS_BASED = "consensus_based"  # Prefer consensus sentences
    LENGTH_OPTIMIZED = "length_optimized"  # Optimize for target length


@dataclass
class SentenceScore:
    """
    Score for a sentence.

    Attributes:
        sentence: The sentence text
        quality_score: Quality score (0-1)
        confidence: Confidence in score
        source_model: Model that generated sentence
        position: Position in original response
        coherence_score: Coherence with context
        informativeness: Information content score
    """
    sentence: str
    quality_score: float
    confidence: float
    source_model: str
    position: int
    coherence_score: float = 0.0
    informativeness: float = 0.0


@dataclass
class SynthesizedResponse:
    """
    Synthesized response from multiple models.

    Attributes:
        content: Synthesized content
        source_models: Models that contributed
        sentence_sources: Source model for each sentence
        synthesis_strategy: Strategy used
        quality_score: Overall quality score
        coherence_score: Overall coherence
        citations: Source attributions
        metadata: Additional metadata
    """
    content: str
    source_models: List[str]
    sentence_sources: List[Tuple[str, str]]  # (sentence, model)
    synthesis_strategy: SynthesisStrategy
    quality_score: float
    coherence_score: float
    citations: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedResponseSynthesizer:
    """
    Advanced multi-model response synthesizer.

    Features:
    - Sentence-level combination from multiple responses
    - Jaccard similarity for deduplication
    - Quality-based sentence selection
    - Coherence checking between sentences
    - Automatic transition generation
    - Source attribution and citations
    - Multiple synthesis strategies

    Example:
        >>> synthesizer = AdvancedResponseSynthesizer(
        ...     similarity_threshold=0.7,
        ...     min_quality_score=0.6
        ... )
        >>> result = synthesizer.synthesize(responses)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        min_quality_score: float = 0.5,
        coherence_threshold: float = 0.6,
        max_length: Optional[int] = None,
        enable_citations: bool = True,
    ):
        """
        Initialize response synthesizer.

        Args:
            similarity_threshold: Threshold for considering sentences similar (deduplication)
            min_quality_score: Minimum quality score for sentence inclusion
            coherence_threshold: Minimum coherence score for sentence transitions
            max_length: Maximum synthesized response length
            enable_citations: Whether to include source citations
        """
        self.similarity_threshold = similarity_threshold
        self.min_quality_score = min_quality_score
        self.coherence_threshold = coherence_threshold
        self.max_length = max_length
        self.enable_citations = enable_citations

        logger.info("AdvancedResponseSynthesizer initialized")

    def synthesize(
        self,
        responses: List[ModelResponse],
        strategy: SynthesisStrategy = SynthesisStrategy.QUALITY_WEIGHTED,
        target_length: Optional[int] = None,
    ) -> SynthesizedResponse:
        """
        Synthesize multiple responses into one.

        Args:
            responses: List of model responses to synthesize
            strategy: Synthesis strategy to use
            target_length: Target length for synthesized response

        Returns:
            Synthesized response

        Example:
            >>> responses = [response1, response2, response3]
            >>> result = synthesizer.synthesize(
            ...     responses,
            ...     strategy=SynthesisStrategy.QUALITY_WEIGHTED,
            ...     target_length=500
            ... )
        """
        # Filter successful responses
        successful = [r for r in responses if not r.error and r.content]

        if not successful:
            return SynthesizedResponse(
                content="",
                source_models=[],
                sentence_sources=[],
                synthesis_strategy=strategy,
                quality_score=0.0,
                coherence_score=0.0,
            )

        # Extract and score sentences
        all_sentences = self._extract_sentences(successful)

        # Deduplicate similar sentences
        unique_sentences = self._deduplicate_sentences(all_sentences)

        # Apply synthesis strategy
        if strategy == SynthesisStrategy.BEST_SENTENCE:
            selected = self._synthesize_best_sentence(unique_sentences)
        elif strategy == SynthesisStrategy.QUALITY_WEIGHTED:
            selected = self._synthesize_quality_weighted(unique_sentences)
        elif strategy == SynthesisStrategy.DIVERSITY_MAXIMIZING:
            selected = self._synthesize_diversity_maximizing(unique_sentences)
        elif strategy == SynthesisStrategy.CONSENSUS_BASED:
            selected = self._synthesize_consensus(unique_sentences, all_sentences)
        elif strategy == SynthesisStrategy.LENGTH_OPTIMIZED:
            selected = self._synthesize_length_optimized(unique_sentences, target_length or self.max_length)
        else:
            selected = self._synthesize_quality_weighted(unique_sentences)

        # Check and improve coherence
        selected = self._ensure_coherence(selected)

        # Generate transitions if needed
        selected = self._add_transitions(selected)

        # Build synthesized content
        content = " ".join(s.sentence for s in selected)

        # Generate citations
        citations = self._generate_citations(selected) if self.enable_citations else {}

        # Calculate scores
        quality_score = statistics.mean([s.quality_score for s in selected]) if selected else 0.0
        coherence_score = statistics.mean([s.coherence_score for s in selected]) if selected else 0.0

        return SynthesizedResponse(
            content=content,
            source_models=list(set(r.model_name for r in successful)),
            sentence_sources=[(s.sentence, s.source_model) for s in selected],
            synthesis_strategy=strategy,
            quality_score=quality_score,
            coherence_score=coherence_score,
            citations=citations,
        )

    def _extract_sentences(self, responses: List[ModelResponse]) -> List[SentenceScore]:
        """Extract and score sentences from responses."""
        all_sentences = []

        for response in responses:
            # Split into sentences (simple split)
            sentences = re.split(r'[.!?]+\s+', response.content)

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence or len(sentence) < 10:
                    continue

                # Calculate quality score
                quality = self._calculate_sentence_quality(sentence, response)

                # Calculate informativeness
                informativeness = self._calculate_informativeness(sentence)

                all_sentences.append(
                    SentenceScore(
                        sentence=sentence,
                        quality_score=quality,
                        confidence=response.confidence,
                        source_model=response.model_name,
                        position=i,
                        informativeness=informativeness,
                    )
                )

        return all_sentences

    def _calculate_sentence_quality(
        self,
        sentence: str,
        response: ModelResponse,
    ) -> float:
        """Calculate quality score for a sentence."""
        score = 0.0

        # Base score from response confidence
        score += response.confidence * 0.4

        # Length score (prefer moderate length)
        length = len(sentence.split())
        if 10 <= length <= 30:
            score += 0.3
        elif 5 <= length < 10 or 30 < length <= 50:
            score += 0.15

        # Specificity score (has numbers, proper nouns)
        if re.search(r'\d', sentence):
            score += 0.15
        if re.search(r'[A-Z][a-z]+', sentence):
            score += 0.15

        return min(1.0, score)

    def _calculate_informativeness(self, sentence: str) -> float:
        """Calculate informativeness of a sentence."""
        # Simple heuristic: unique words vs total words
        words = sentence.lower().split()
        if not words:
            return 0.0

        unique_ratio = len(set(words)) / len(words)

        # Penalize very short or very long sentences
        length_penalty = 1.0
        if len(words) < 5:
            length_penalty = 0.5
        elif len(words) > 50:
            length_penalty = 0.7

        return unique_ratio * length_penalty

    def _deduplicate_sentences(
        self,
        sentences: List[SentenceScore],
    ) -> List[SentenceScore]:
        """Remove similar/duplicate sentences using Jaccard similarity."""
        if not sentences:
            return []

        unique = []
        seen_sentences = []

        for sentence in sorted(sentences, key=lambda s: s.quality_score, reverse=True):
            is_duplicate = False

            for seen in seen_sentences:
                similarity = self._jaccard_similarity(sentence.sentence, seen)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(sentence)
                seen_sentences.append(sentence.sentence)

        return unique

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _synthesize_best_sentence(
        self,
        sentences: List[SentenceScore],
    ) -> List[SentenceScore]:
        """Select best sentence from each position."""
        # Group by position
        by_position = defaultdict(list)
        for sentence in sentences:
            by_position[sentence.position].append(sentence)

        # Select best from each position
        selected = []
        for position in sorted(by_position.keys()):
            candidates = by_position[position]
            best = max(candidates, key=lambda s: s.quality_score)
            if best.quality_score >= self.min_quality_score:
                selected.append(best)

        return selected

    def _synthesize_quality_weighted(
        self,
        sentences: List[SentenceScore],
    ) -> List[SentenceScore]:
        """Select sentences weighted by quality."""
        # Filter by minimum quality
        qualified = [s for s in sentences if s.quality_score >= self.min_quality_score]

        # Sort by quality and take top sentences
        sorted_sentences = sorted(qualified, key=lambda s: s.quality_score, reverse=True)

        # Take sentences until max length or quality drops significantly
        selected = []
        total_length = 0

        for sentence in sorted_sentences:
            sent_length = len(sentence.sentence)
            if self.max_length and total_length + sent_length > self.max_length:
                break

            selected.append(sentence)
            total_length += sent_length

            # Stop if quality drops too much
            if selected and sentence.quality_score < selected[0].quality_score * 0.6:
                break

        return selected

    def _synthesize_diversity_maximizing(
        self,
        sentences: List[SentenceScore],
    ) -> List[SentenceScore]:
        """Maximize information diversity."""
        selected = []
        covered_words = set()

        # Sort by informativeness
        sorted_sentences = sorted(
            sentences,
            key=lambda s: s.informativeness,
            reverse=True,
        )

        for sentence in sorted_sentences:
            words = set(sentence.sentence.lower().split())

            # Calculate new information
            new_words = words - covered_words
            novelty = len(new_words) / len(words) if words else 0

            # Add if sufficiently novel
            if novelty >= 0.3 or not selected:
                selected.append(sentence)
                covered_words.update(words)

                if self.max_length and sum(len(s.sentence) for s in selected) >= self.max_length:
                    break

        return selected

    def _synthesize_consensus(
        self,
        unique_sentences: List[SentenceScore],
        all_sentences: List[SentenceScore],
    ) -> List[SentenceScore]:
        """Prefer sentences with consensus across models."""
        # Count how many models express similar sentences
        consensus_scores = {}

        for unique in unique_sentences:
            similar_count = sum(
                1 for s in all_sentences
                if self._jaccard_similarity(unique.sentence, s.sentence) >= 0.5
            )
            consensus_scores[unique.sentence] = similar_count

        # Sort by consensus and quality
        sorted_sentences = sorted(
            unique_sentences,
            key=lambda s: (consensus_scores[s.sentence], s.quality_score),
            reverse=True,
        )

        # Take high-consensus sentences
        return sorted_sentences[:10]  # Top 10

    def _synthesize_length_optimized(
        self,
        sentences: List[SentenceScore],
        target_length: Optional[int],
    ) -> List[SentenceScore]:
        """Optimize for target length."""
        if not target_length:
            return self._synthesize_quality_weighted(sentences)

        # Sort by quality/length ratio
        sorted_sentences = sorted(
            sentences,
            key=lambda s: s.quality_score / max(len(s.sentence), 1),
            reverse=True,
        )

        selected = []
        total_length = 0

        for sentence in sorted_sentences:
            sent_length = len(sentence.sentence)
            if total_length + sent_length <= target_length:
                selected.append(sentence)
                total_length += sent_length
            elif not selected:  # Always include at least one
                selected.append(sentence)
                break

        return selected

    def _ensure_coherence(
        self,
        sentences: List[SentenceScore],
    ) -> List[SentenceScore]:
        """Check and improve coherence between sentences."""
        if len(sentences) <= 1:
            return sentences

        # Calculate coherence scores
        for i in range(len(sentences) - 1):
            current = sentences[i]
            next_sent = sentences[i + 1]

            # Calculate coherence (simple: shared words)
            coherence = self._calculate_coherence(current.sentence, next_sent.sentence)
            current.coherence_score = coherence

        # Last sentence
        if sentences:
            sentences[-1].coherence_score = 1.0

        # Remove low-coherence sentences
        filtered = [
            s for s in sentences
            if s.coherence_score >= self.coherence_threshold or s == sentences[0]
        ]

        return filtered if filtered else sentences[:1]

    def _calculate_coherence(self, sent1: str, sent2: str) -> float:
        """Calculate coherence between two sentences."""
        # Shared words (normalized)
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        if not words1 or not words2:
            return 0.5

        shared = len(words1 & words2)
        total = max(len(words1), len(words2))

        return shared / total if total > 0 else 0.5

    def _add_transitions(
        self,
        sentences: List[SentenceScore],
    ) -> List[SentenceScore]:
        """Add transition words/phrases between sentences if needed."""
        # For now, return as-is
        # In production, would add "Furthermore," "However," etc. based on context
        return sentences

    def _generate_citations(
        self,
        sentences: List[SentenceScore],
    ) -> Dict[str, List[str]]:
        """Generate source citations for synthesized content."""
        citations = defaultdict(list)

        for sentence in sentences:
            citations[sentence.source_model].append(sentence.sentence[:100])

        return dict(citations)
