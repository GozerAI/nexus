
"""
Knowledge Validation System using Ensemble Models.
Validates information retrieved from the internet before adding to knowledge base.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"
    CONFLICTING = "conflicting"

@dataclass
class ValidationResult:
    """Result of knowledge validation."""
    status: ValidationStatus
    confidence: float
    reasoning: str
    ensemble_scores: Dict[str, float]
    human_review_required: bool
    validation_timestamp: float

import logging
import time
from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status enumeration."""
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"
    CONFLICTING = "conflicting"

@dataclass
class ValidationResult:
    """Result of knowledge validation."""
    status: ValidationStatus
    confidence: float
    reasoning: str
    ensemble_scores: Dict[str, float]
    human_review_required: bool
    validation_timestamp: float

class KnowledgeValidator:
    """
    Validates knowledge using ensemble models and configurable criteria.
    """
    
    def __init__(self, ensemble_core=None, human_feedback_interface=None):
        self.ensemble_core = ensemble_core
        self.human_feedback_interface = human_feedback_interface
        
        # Validation thresholds
        self.auto_accept_threshold = 0.8
        self.auto_reject_threshold = 0.3
        self.consensus_threshold = 0.7
        
        # Validation criteria
        self.validation_prompts = {
            'factual_accuracy': "Is this statement factually accurate and verifiable: '{content}'?",
            'logical_consistency': "Is this statement logically consistent and coherent: '{content}'?",
            'source_reliability': "Based on the source '{source}', how reliable is this information: '{content}'?",
            'bias_detection': "Does this statement contain obvious bias or misleading information: '{content}'?",
            'temporal_relevance': "Is this information current and temporally relevant: '{content}'?"
        }
        
        self.validation_history = []
        self.validation_stats = {
            'total_validations': 0,
            'auto_accepted': 0,
            'auto_rejected': 0,
            'human_reviewed': 0,
            'conflicting_results': 0
        }
    
    def validate_knowledge(self, content: str, source: str, 
                          context: str = "", require_consensus: bool = True) -> ValidationResult:
        """Validate a piece of knowledge using multiple criteria."""
        
        if not self.ensemble_core:
            logger.warning("No ensemble core available for validation")
            return ValidationResult(
                status=ValidationStatus.PENDING_REVIEW,
                confidence=0.5,
                reasoning="No ensemble available for validation",
                ensemble_scores={},
                human_review_required=True,
                validation_timestamp=time.time()
            )
        
        # Run validation across multiple criteria
        validation_scores = {}
        ensemble_results = {}
        
        for criterion, prompt_template in self.validation_prompts.items():
            try:
                prompt = prompt_template.format(content=content, source=source)
                result = self.ensemble_core.predict(prompt)
                
                # Extract score from ensemble result
                score = self._extract_validation_score(result, criterion)
                validation_scores[criterion] = score
                ensemble_results[criterion] = {
                    'prediction': result.prediction,
                    'confidence': result.confidence,
                    'score': score
                }
                
            except Exception as e:
                logger.error(f"Error validating criterion {criterion}: {e}")
                validation_scores[criterion] = 0.5  # Neutral score on error
        
        # Calculate overall validation result
        overall_confidence = self._calculate_overall_confidence(validation_scores)
        consensus_score = self._calculate_consensus(list(validation_scores.values()))
        
        # Determine validation status
        status = self._determine_validation_status(
            overall_confidence, consensus_score, require_consensus
        )
        
        # Check if human review is required
        human_review_required = self._requires_human_review(
            status, overall_confidence, consensus_score
        )
        
        # Generate reasoning
        reasoning = self._generate_validation_reasoning(
            validation_scores, overall_confidence, consensus_score
        )
        
        result = ValidationResult(
            status=status,
            confidence=overall_confidence,
            reasoning=reasoning,
            ensemble_scores=validation_scores,
            human_review_required=human_review_required,
            validation_timestamp=time.time()
        )
        
        # Update statistics
        self._update_validation_stats(result)
        
        # Queue for human review if needed
        if human_review_required and self.human_feedback_interface:
            self._queue_for_human_review(content, source, result)
        
        self.validation_history.append({
            'content': content[:100],
            'source': source,
            'result': result,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        return result
    
    def batch_validate(self, knowledge_items: List[Dict[str, str]]) -> List[ValidationResult]:
        """Validate multiple knowledge items efficiently."""
        results = []
        
        for item in knowledge_items:
            content = item.get('content', '')
            source = item.get('source', 'unknown')
            context = item.get('context', '')
            
            result = self.validate_knowledge(content, source, context)
            results.append(result)
        
        return results
    
    def _extract_validation_score(self, ensemble_result, criterion: str) -> float:
        """Extract a validation score from ensemble result."""
        prediction = str(ensemble_result.prediction).lower()
        confidence = ensemble_result.confidence
        
        # Score based on prediction content and confidence
        if criterion == 'factual_accuracy':
            if any(word in prediction for word in ['true', 'accurate', 'correct', 'yes']):
                return confidence
            elif any(word in prediction for word in ['false', 'inaccurate', 'incorrect', 'no']):
                return 1.0 - confidence
            else:
                return 0.5
        
        elif criterion == 'bias_detection':
            if any(word in prediction for word in ['biased', 'misleading', 'propaganda']):
                return 1.0 - confidence  # High confidence in bias = low validation score
            elif any(word in prediction for word in ['unbiased', 'neutral', 'objective']):
                return confidence
            else:
                return 0.6  # Slightly positive default
        
        elif criterion in ['logical_consistency', 'source_reliability', 'temporal_relevance']:
            if any(word in prediction for word in ['yes', 'good', 'reliable', 'consistent', 'relevant']):
                return confidence
            elif any(word in prediction for word in ['no', 'poor', 'unreliable', 'inconsistent', 'outdated']):
                return 1.0 - confidence
            else:
                return 0.5
        
        return 0.5  # Default neutral score
    
    def _calculate_overall_confidence(self, validation_scores: Dict[str, float]) -> float:
        """Calculate overall confidence from individual validation scores."""
        if not validation_scores:
            return 0.0
        
        # Weighted average (factual accuracy gets higher weight)
        weights = {
            'factual_accuracy': 0.3,
            'logical_consistency': 0.25,
            'source_reliability': 0.2,
            'bias_detection': 0.15,
            'temporal_relevance': 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, score in validation_scores.items():
            weight = weights.get(criterion, 0.2)  # Default weight
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_consensus(self, scores: List[float]) -> float:
        """Calculate consensus among validation scores."""
        if len(scores) < 2:
            return 1.0
        
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        # Convert variance to consensus (lower variance = higher consensus)
        consensus = max(0.0, 1.0 - variance)
        return consensus
    
    def _determine_validation_status(self, confidence: float, consensus: float, 
                                   require_consensus: bool) -> ValidationStatus:
        """Determine validation status based on confidence and consensus."""
        
        if require_consensus and consensus < self.consensus_threshold:
            return ValidationStatus.CONFLICTING
        
        if confidence >= self.auto_accept_threshold:
            return ValidationStatus.ACCEPTED
        elif confidence <= self.auto_reject_threshold:
            return ValidationStatus.REJECTED
        else:
            return ValidationStatus.PENDING_REVIEW
    
    def _requires_human_review(self, status: ValidationStatus, 
                              confidence: float, consensus: float) -> bool:
        """Determine if human review is required."""
        
        return (status in [ValidationStatus.PENDING_REVIEW, ValidationStatus.CONFLICTING] or
                confidence < 0.6 or
                consensus < 0.5)
    
    def _generate_validation_reasoning(self, scores: Dict[str, float], 
                                     confidence: float, consensus: float) -> str:
        """Generate human-readable validation reasoning."""
        
        reasoning_parts = []
        
        # Overall assessment
        if confidence >= 0.8:
            reasoning_parts.append("High confidence validation")
        elif confidence >= 0.6:
            reasoning_parts.append("Moderate confidence validation")
        else:
            reasoning_parts.append("Low confidence validation")
        
        # Consensus assessment
        if consensus >= 0.8:
            reasoning_parts.append("strong consensus among criteria")
        elif consensus >= 0.6:
            reasoning_parts.append("moderate consensus among criteria")
        else:
            reasoning_parts.append("conflicting results among criteria")
        
        # Specific concerns
        concerns = []
        for criterion, score in scores.items():
            if score < 0.4:
                concerns.append(f"low {criterion.replace('_', ' ')} score ({score:.2f})")
        
        if concerns:
            reasoning_parts.append(f"Concerns: {', '.join(concerns)}")
        
        return "; ".join(reasoning_parts)
    
    def _update_validation_stats(self, result: ValidationResult):
        """Update validation statistics."""
        self.validation_stats['total_validations'] += 1
        
        if result.status == ValidationStatus.ACCEPTED:
            self.validation_stats['auto_accepted'] += 1
        elif result.status == ValidationStatus.REJECTED:
            self.validation_stats['auto_rejected'] += 1
        elif result.status == ValidationStatus.CONFLICTING:
            self.validation_stats['conflicting_results'] += 1
        
        if result.human_review_required:
            self.validation_stats['human_reviewed'] += 1
    
    def _queue_for_human_review(self, content: str, source: str, result: ValidationResult):
        """Queue knowledge for human review."""
        try:
            from core.human_loop.feedback_interface import FeedbackType
            
            self.human_feedback_interface.request_feedback(
                input_data=f"Knowledge validation for: {content}",
                current_output=f"Validation result: {result.status.value} (confidence: {result.confidence:.2f})",
                confidence=result.confidence,
                feedback_type=FeedbackType.VALIDATION,
                context={
                    'source': source,
                    'validation_reasoning': result.reasoning,
                    'ensemble_scores': result.ensemble_scores
                }
            )
        except Exception as e:
            logger.error(f"Error queuing for human review: {e}")
    
    def initialize(self):
        """Initialize the Knowledge Validator."""
        logger.info("Knowledge Validator initialized")

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation system statistics."""
        
        if self.validation_stats['total_validations'] > 0:
            acceptance_rate = self.validation_stats['auto_accepted'] / self.validation_stats['total_validations']
            rejection_rate = self.validation_stats['auto_rejected'] / self.validation_stats['total_validations']
            review_rate = self.validation_stats['human_reviewed'] / self.validation_stats['total_validations']
        else:
            acceptance_rate = rejection_rate = review_rate = 0.0
        
        recent_validations = [v for v in self.validation_history if time.time() - v['timestamp'] < 3600]
        
        return {
            'total_validations': self.validation_stats['total_validations'],
            'acceptance_rate': acceptance_rate,
            'rejection_rate': rejection_rate,
            'human_review_rate': review_rate,
            'recent_validations': len(recent_validations),
            'thresholds': {
                'auto_accept': self.auto_accept_threshold,
                'auto_reject': self.auto_reject_threshold,
                'consensus': self.consensus_threshold
            },
            'validation_criteria': list(self.validation_prompts.keys())
        }
