"""
Response scoring algorithms for ensemble inference.
"""

import logging
from typing import Optional
import re

from nexus.core.models.base import ModelResponse

logger = logging.getLogger(__name__)


class ResponseScorer:
    """
    Scores model responses based on multiple criteria.
    """
    
    def __init__(self):
        """Initialize the response scorer."""
        self.weights = {
            "length": 0.2,
            "coherence": 0.3,
            "specificity": 0.2,
            "confidence": 0.3,
        }
    
    def score_response(self, response: ModelResponse, prompt: str) -> float:
        """
        Score a model response.
        
        Args:
            response: Model response to score
            prompt: Original prompt
            
        Returns:
            Score between 0 and 1
        """
        if not response.success:
            return 0.0
        
        scores = {
            "length": self._score_length(response.content),
            "coherence": self._score_coherence(response.content),
            "specificity": self._score_specificity(response.content),
            "confidence": self._score_confidence(response),
        }
        
        # Weighted average
        total_score = sum(
            scores[metric] * self.weights[metric]
            for metric in scores
        )
        
        logger.debug(
            f"Scoring {response.model_name}: "
            f"length={scores['length']:.2f}, "
            f"coherence={scores['coherence']:.2f}, "
            f"specificity={scores['specificity']:.2f}, "
            f"confidence={scores['confidence']:.2f}, "
            f"total={total_score:.2f}"
        )
        
        return total_score
    
    def _score_length(self, content: str) -> float:
        """
        Score based on response length.
        
        Ideal length is 100-500 words.
        
        Args:
            content: Response content
            
        Returns:
            Score between 0 and 1
        """
        word_count = len(content.split())
        
        if word_count < 10:
            return 0.1
        elif word_count < 50:
            return 0.5
        elif 100 <= word_count <= 500:
            return 1.0
        elif word_count <= 1000:
            return 0.8
        else:
            return 0.6
    
    def _score_coherence(self, content: str) -> float:
        """
        Score based on coherence indicators.
        
        Looks for:
        - Complete sentences
        - Proper punctuation
        - Paragraph structure
        
        Args:
            content: Response content
            
        Returns:
            Score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Check for proper sentence endings
        sentences = re.split(r'[.!?]+', content)
        valid_sentences = [s for s in sentences if len(s.strip()) > 10]
        
        if len(valid_sentences) >= 2:
            score += 0.2
        
        # Check for paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            score += 0.15
        
        # Check for transition words
        transitions = ['however', 'therefore', 'additionally', 'furthermore', 'moreover']
        if any(word in content.lower() for word in transitions):
            score += 0.15
        
        return min(score, 1.0)
    
    def _score_specificity(self, content: str) -> float:
        """
        Score based on specificity indicators.
        
        Looks for:
        - Numbers and data
        - Examples
        - Specific terms
        
        Args:
            content: Response content
            
        Returns:
            Score between 0 and 1
        """
        score = 0.3  # Base score
        
        # Check for numbers
        if re.search(r'\d+', content):
            score += 0.2
        
        # Check for examples
        if re.search(r'for example|such as|e\.g\.|i\.e\.', content, re.IGNORECASE):
            score += 0.2
        
        # Check for specific markers
        if any(marker in content.lower() for marker in ['specifically', 'particularly', 'namely']):
            score += 0.15
        
        # Check for lists
        if re.search(r'^\s*[-*•]\s', content, re.MULTILINE):
            score += 0.15
        
        return min(score, 1.0)
    
    def _score_confidence(self, response: ModelResponse) -> float:
        """
        Score based on model confidence indicators.
        
        Uses:
        - Latency (faster can indicate confidence)
        - Model weight
        - Error presence
        
        Args:
            response: Model response
            
        Returns:
            Score between 0 and 1
        """
        score = 0.5  # Base score
        
        # Faster responses can indicate model confidence
        if response.latency_ms < 1000:
            score += 0.2
        elif response.latency_ms < 3000:
            score += 0.1
        
        # Check for uncertainty markers
        uncertainty_markers = [
            'might', 'maybe', 'perhaps', 'possibly', 
            'i think', 'not sure', 'unclear'
        ]
        content_lower = response.content.lower()
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in content_lower)
        
        if uncertainty_count == 0:
            score += 0.3
        elif uncertainty_count <= 2:
            score += 0.15
        else:
            score -= 0.1
        
        return max(0.0, min(score, 1.0))
