"""Tests for response scoring."""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.scoring import ResponseScorer
from nexus.core.models.base import ModelResponse


class TestResponseScorer:
    """Tests for ResponseScorer class."""

    def test_scorer_init(self):
        """Test ResponseScorer initialization."""
        scorer = ResponseScorer()
        assert scorer is not None
        assert scorer.weights is not None

    def test_score_simple_response(self):
        """Test scoring a simple response."""
        scorer = ResponseScorer()
        
        response = ModelResponse(
            content="This is a test response with some content.",
            model_name="test-model",
            provider="stub",
            tokens_used=10,
            latency_ms=100.0,
        )
        
        score = scorer.score_response(response, "test prompt")
        assert 0.0 <= score <= 1.0

    def test_score_failed_response(self):
        """Test scoring a failed response."""
        scorer = ResponseScorer()
        
        response = ModelResponse(
            content="",
            model_name="test-model",
            provider="stub",
            error="Test error",
        )
        
        score = scorer.score_response(response, "test prompt")
        assert score == 0.0

    def test_score_length(self):
        """Test length scoring."""
        scorer = ResponseScorer()
        
        # Short response
        short_score = scorer._score_length("Short.")
        
        # Medium response (ideal)
        medium_text = " ".join(["word"] * 200)
        medium_score = scorer._score_length(medium_text)
        
        # Long response
        long_text = " ".join(["word"] * 1500)
        long_score = scorer._score_length(long_text)
        
        assert medium_score > short_score
        assert medium_score >= long_score

    def test_score_coherence(self):
        """Test coherence scoring."""
        scorer = ResponseScorer()
        
        # Coherent text with proper structure
        coherent_text = """
        This is a well-structured response. It has multiple sentences.
        
        Furthermore, it contains paragraphs. Therefore, it should score higher.
        Additionally, it uses transition words.
        """
        
        # Incoherent text
        incoherent_text = "word word word"
        
        coherent_score = scorer._score_coherence(coherent_text)
        incoherent_score = scorer._score_coherence(incoherent_text)
        
        assert coherent_score > incoherent_score

    def test_score_specificity(self):
        """Test specificity scoring."""
        scorer = ResponseScorer()
        
        # Specific text with examples and numbers
        specific_text = """
        For example, there are 42 different approaches. Specifically, we can use:
        - Method 1
        - Method 2
        This demonstrates the concept clearly.
        """
        
        # Generic text
        generic_text = "This is a general statement without details."
        
        specific_score = scorer._score_specificity(specific_text)
        generic_score = scorer._score_specificity(generic_text)
        
        assert specific_score > generic_score

    def test_score_confidence(self):
        """Test confidence scoring."""
        scorer = ResponseScorer()
        
        # Confident response (fast, no uncertainty)
        confident_response = ModelResponse(
            content="This is definitely the correct answer.",
            model_name="test-model",
            provider="stub",
            latency_ms=500,
        )
        
        # Uncertain response (slow, uncertain language)
        uncertain_response = ModelResponse(
            content="I think maybe this might be the answer, perhaps.",
            model_name="test-model",
            provider="stub",
            latency_ms=5000,
        )
        
        confident_score = scorer._score_confidence(confident_response)
        uncertain_score = scorer._score_confidence(uncertain_response)
        
        assert confident_score > uncertain_score
