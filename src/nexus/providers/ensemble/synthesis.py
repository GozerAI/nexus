"""
Response synthesis module - combines best parts from multiple model responses.

This module implements sophisticated response synthesis that:
1. Extracts sentences from multiple model responses
2. Scores each sentence for quality
3. Detects and removes redundant content
4. Combines the best sentences into a coherent response

Based on combo1's response synthesis algorithm with enhancements.
"""

import re
from dataclasses import dataclass
from typing import List, Set, Tuple

from nexus.providers.ensemble.types import ModelResponse


@dataclass
class ScoredSentence:
    """
    A sentence with quality score and metadata.

    Attributes:
        text: Sentence text
        score: Quality score (0-1)
        model_name: Source model
        position: Position in original response (0-based)
        length: Character length
        word_count: Number of words
    """

    text: str
    score: float
    model_name: str
    position: int
    length: int
    word_count: int


class ResponseSynthesizer:
    """
    Synthesizes multiple model responses into a single high-quality response.

    The synthesis process:
    1. Extract sentences from each response
    2. Score sentences based on multiple quality factors
    3. Remove redundant/duplicate content
    4. Select best sentences
    5. Recombine into coherent response

    Features:
    - Quality scoring (completeness, specificity, coherence)
    - Redundancy detection (Jaccard similarity)
    - Position-aware selection (prefer diverse sources)
    - Length normalization
    - Model contribution tracking
    """

    def __init__(
        self,
        redundancy_threshold: float = 0.7,
        min_sentence_length: int = 10,
        max_sentences: int = 20,
        diversity_weight: float = 0.3,
    ):
        """
        Initialize response synthesizer.

        Args:
            redundancy_threshold: Similarity threshold for duplicate detection (0-1)
            min_sentence_length: Minimum sentence length to consider
            max_sentences: Maximum sentences in synthesized response
            diversity_weight: Weight for source diversity (0-1)
        """
        self.redundancy_threshold = redundancy_threshold
        self.min_sentence_length = min_sentence_length
        self.max_sentences = max_sentences
        self.diversity_weight = diversity_weight

    def synthesize(self, responses: List[ModelResponse]) -> Tuple[str, dict]:
        """
        Synthesize multiple responses into one.

        Args:
            responses: List of model responses to synthesize

        Returns:
            Tuple of (synthesized_text, metadata)
                metadata contains:
                - model_contributions: dict of model -> sentence count
                - total_sentences: total sentences in synthesis
                - redundancy_removed: number of duplicates removed
                - average_sentence_score: average quality score
        """
        if not responses:
            return "", {"error": "No responses to synthesize"}

        # Filter successful responses
        successful = [r for r in responses if r.error is None and r.content.strip()]

        if not successful:
            return "", {"error": "No successful responses to synthesize"}

        # If only one response, return it directly
        if len(successful) == 1:
            return successful[0].content, {
                "model_contributions": {successful[0].model_name: 1},
                "total_sentences": 1,
                "redundancy_removed": 0,
                "single_model": True,
            }

        # Step 1: Extract and score sentences from all responses
        all_sentences = []
        for response in successful:
            sentences = self._extract_sentences(response)
            all_sentences.extend(sentences)

        if not all_sentences:
            return successful[0].content, {"error": "No valid sentences extracted"}

        # Step 2: Remove redundant sentences
        unique_sentences, removed_count = self._remove_redundancy(all_sentences)

        # Step 3: Select best sentences
        selected_sentences = self._select_best_sentences(unique_sentences)

        # Step 4: Recombine into coherent text
        synthesized_text = self._recombine_sentences(selected_sentences)

        # Step 5: Calculate metadata
        model_contributions = {}
        for sent in selected_sentences:
            model_contributions[sent.model_name] = (
                model_contributions.get(sent.model_name, 0) + 1
            )

        avg_score = sum(s.score for s in selected_sentences) / len(selected_sentences)

        metadata = {
            "model_contributions": model_contributions,
            "total_sentences": len(selected_sentences),
            "redundancy_removed": removed_count,
            "average_sentence_score": avg_score,
            "synthesis_method": "quality_weighted",
        }

        return synthesized_text, metadata

    def _extract_sentences(self, response: ModelResponse) -> List[ScoredSentence]:
        """
        Extract sentences from a response and score them.

        Args:
            response: Model response to extract from

        Returns:
            List of scored sentences
        """
        content = response.content.strip()

        # Split into sentences using regex
        # Handles periods, question marks, exclamation points
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        raw_sentences = re.split(sentence_pattern, content)

        # Also handle newlines as sentence boundaries
        expanded = []
        for sent in raw_sentences:
            expanded.extend(sent.split('\n'))

        scored_sentences = []
        for i, text in enumerate(expanded):
            text = text.strip()

            # Skip too-short sentences
            if len(text) < self.min_sentence_length:
                continue

            # Calculate quality score
            score = self._score_sentence(text, response)

            # Create scored sentence
            words = text.split()
            scored_sent = ScoredSentence(
                text=text,
                score=score,
                model_name=response.model_name,
                position=i,
                length=len(text),
                word_count=len(words),
            )
            scored_sentences.append(scored_sent)

        return scored_sentences

    def _score_sentence(self, text: str, response: ModelResponse) -> float:
        """
        Score a sentence based on quality factors.

        Scoring factors:
        - Completeness (has proper punctuation)
        - Length (prefer 20-200 characters)
        - Specificity (has numbers, proper nouns, technical terms)
        - Coherence (grammatical structure)
        - Model confidence

        Args:
            text: Sentence text
            response: Parent response

        Returns:
            Quality score (0-1)
        """
        score = 0.0
        weights_sum = 0.0

        # Factor 1: Completeness (0.2 weight)
        # Check for proper ending punctuation
        if text and text[-1] in '.!?':
            score += 0.2
        elif text and text[-1] in ',;:':
            score += 0.1
        weights_sum += 0.2

        # Factor 2: Length (0.15 weight)
        # Prefer sentences between 20-200 characters
        length_score = 0.0
        if 20 <= len(text) <= 200:
            length_score = 1.0
        elif len(text) < 20:
            length_score = len(text) / 20
        else:  # > 200
            length_score = max(0.5, 1.0 - (len(text) - 200) / 300)
        score += length_score * 0.15
        weights_sum += 0.15

        # Factor 3: Specificity (0.25 weight)
        specificity_score = 0.0

        # Has numbers
        if re.search(r'\d', text):
            specificity_score += 0.3

        # Has capitalized words (proper nouns, acronyms)
        caps_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        if len(caps_words) > 0:
            specificity_score += min(0.3, len(caps_words) * 0.1)

        # Has technical indicators (parentheses, hyphens, colons)
        if re.search(r'[:()\-]', text):
            specificity_score += 0.2

        # Has specific keywords
        specific_keywords = ['specifically', 'example', 'such as', 'for instance', 'namely']
        for keyword in specific_keywords:
            if keyword.lower() in text.lower():
                specificity_score += 0.2
                break

        score += min(1.0, specificity_score) * 0.25
        weights_sum += 0.25

        # Factor 4: Coherence (0.2 weight)
        coherence_score = 0.0

        # Has subject-verb structure (very rough check)
        words = text.split()
        if len(words) >= 3:
            coherence_score += 0.4

        # Not all caps (indicates shouting/formatting)
        if not text.isupper():
            coherence_score += 0.3

        # Proper capitalization at start
        if text and text[0].isupper():
            coherence_score += 0.3

        score += coherence_score * 0.2
        weights_sum += 0.2

        # Factor 5: Model confidence (0.2 weight)
        score += response.confidence * 0.2
        weights_sum += 0.2

        # Normalize
        if weights_sum > 0:
            score = score / weights_sum

        return min(1.0, max(0.0, score))

    def _remove_redundancy(
        self, sentences: List[ScoredSentence]
    ) -> Tuple[List[ScoredSentence], int]:
        """
        Remove redundant/duplicate sentences.

        Uses Jaccard similarity to detect duplicates.
        Keeps the higher-scored version.

        Args:
            sentences: List of scored sentences

        Returns:
            Tuple of (unique_sentences, removed_count)
        """
        if not sentences:
            return [], 0

        # Sort by score (descending) so we keep best versions
        sorted_sentences = sorted(sentences, key=lambda s: s.score, reverse=True)

        unique = []
        removed = 0

        for sent in sorted_sentences:
            is_duplicate = False

            # Check against already-selected unique sentences
            for unique_sent in unique:
                similarity = self._jaccard_similarity(sent.text, unique_sent.text)

                if similarity >= self.redundancy_threshold:
                    is_duplicate = True
                    removed += 1
                    break

            if not is_duplicate:
                unique.append(sent)

        return unique, removed

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts.

        Jaccard = |intersection| / |union| of word sets

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Convert to lowercase word sets
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def _select_best_sentences(
        self, sentences: List[ScoredSentence]
    ) -> List[ScoredSentence]:
        """
        Select best sentences for final synthesis.

        Selection criteria:
        - Quality score (primary)
        - Source diversity (prefer sentences from different models)
        - Position balance (prefer sentences from beginning, middle, end)

        Args:
            sentences: Candidate sentences

        Returns:
            Selected sentences
        """
        if not sentences:
            return []

        # Track models used for diversity
        model_counts = {}

        # Apply diversity weighting to scores
        weighted_sentences = []
        for sent in sentences:
            # Base score
            final_score = sent.score

            # Penalty for over-representation of a model
            model_count = model_counts.get(sent.model_name, 0)
            diversity_penalty = model_count * self.diversity_weight * 0.1
            final_score -= diversity_penalty

            weighted_sentences.append((final_score, sent))

            # Update model count
            model_counts[sent.model_name] = model_count + 1

        # Sort by weighted score
        weighted_sentences.sort(key=lambda x: x[0], reverse=True)

        # Select top sentences up to max
        selected = [sent for score, sent in weighted_sentences[:self.max_sentences]]

        # Sort selected by original position for coherent flow
        selected.sort(key=lambda s: s.position)

        return selected

    def _recombine_sentences(self, sentences: List[ScoredSentence]) -> str:
        """
        Recombine selected sentences into coherent text.

        Args:
            sentences: Sentences to combine

        Returns:
            Combined text
        """
        if not sentences:
            return ""

        # Join sentences with spaces
        # Ensure proper spacing after periods
        text_parts = []
        for sent in sentences:
            text_parts.append(sent.text)

        # Join with space
        combined = " ".join(text_parts)

        # Clean up multiple spaces
        combined = re.sub(r'\s+', ' ', combined)

        # Ensure proper spacing after punctuation
        combined = re.sub(r'([.!?])([A-Z])', r'\1 \2', combined)

        return combined.strip()


def synthesize_responses(responses: List[ModelResponse]) -> Tuple[str, dict]:
    """
    Convenience function to synthesize responses with default settings.

    Args:
        responses: List of model responses

    Returns:
        Tuple of (synthesized_text, metadata)
    """
    synthesizer = ResponseSynthesizer()
    return synthesizer.synthesize(responses)
