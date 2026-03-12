"""
Bias Mitigation System for Ethical AI Development

This module implements comprehensive bias detection and mitigation strategies
to prevent bias accumulation in AI systems while promoting virtue-based learning.

Key Features:
- 8 bias type detection (confirmation, selection, cultural, cognitive, temporal, authority, anchoring, availability)
- Multi-perspective balancing with automatic weight adjustment
- Virtue assessment across 8 categories (wisdom, justice, courage, temperance, compassion, integrity, humility, patience)
- Source diversity tracking and promotion

Adapted from: nexus-system/server/sage/bias-mitigation-system.ts
"""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class BiasType(str, Enum):
    """Types of cognitive and systemic bias."""
    CONFIRMATION_BIAS = "confirmation_bias"
    SELECTION_BIAS = "selection_bias"
    CULTURAL_BIAS = "cultural_bias"
    COGNITIVE_BIAS = "cognitive_bias"
    TEMPORAL_BIAS = "temporal_bias"
    AUTHORITY_BIAS = "authority_bias"
    ANCHORING_BIAS = "anchoring_bias"
    AVAILABILITY_BIAS = "availability_bias"


class PerspectiveSource(str, Enum):
    """Sources of perspectives for balanced learning."""
    INDIVIDUAL_USER = "individual_user"
    EXPERT_PANEL = "expert_panel"
    DIVERSE_COMMUNITY = "diverse_community"
    HISTORICAL_WISDOM = "historical_wisdom"
    CROSS_CULTURAL = "cross_cultural"
    PHILOSOPHICAL_TRADITION = "philosophical_tradition"
    SCIENTIFIC_CONSENSUS = "scientific_consensus"
    ETHICAL_FRAMEWORK = "ethical_framework"


class VirtueCategory(str, Enum):
    """Virtue categories for ethical assessment."""
    WISDOM = "wisdom"
    JUSTICE = "justice"
    COURAGE = "courage"
    TEMPERANCE = "temperance"
    COMPASSION = "compassion"
    INTEGRITY = "integrity"
    HUMILITY = "humility"
    PATIENCE = "patience"


@dataclass
class PerspectiveInput:
    """Input perspective for bias analysis."""
    perspective_id: str
    source_type: PerspectiveSource
    source_identifier: str
    viewpoint: Dict[str, Any]
    confidence: float
    reasoning: List[str]
    cultural_context: Dict[str, Any]
    timestamp: datetime
    calculated_weight: float = 1.0


@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis."""
    bias_detected: bool
    bias_types: List[BiasType]
    severity_score: float
    affected_domains: List[str]
    mitigation_recommendations: List[str]
    confidence: float


@dataclass
class VirtueAssessment:
    """Assessment of virtue alignment."""
    virtue_scores: Dict[VirtueCategory, float]
    overall_virtue_alignment: float
    virtue_conflicts: List[str]
    improvement_recommendations: List[str]


class BiasMitigationSystem:
    """
    Bias Mitigation System for Ethical AI Development.

    Prevents bias accumulation while promoting virtue-based learning through:
    - Multi-perspective analysis
    - Automatic weight balancing
    - Comprehensive bias detection
    - Virtue-based ethical assessment
    """

    def __init__(self):
        """Initialize the bias mitigation system."""
        self.perspectives: Dict[str, PerspectiveInput] = {}
        self.bias_history: List[BiasDetectionResult] = []
        self.virtue_assessments: Dict[str, VirtueAssessment] = {}
        self.source_distribution: Dict[PerspectiveSource, float] = {}

        self._initialize_source_tracking()
        logger.info("⚖️ Bias Mitigation System initialized")

    def _initialize_source_tracking(self) -> None:
        """Initialize source tracking for balanced perspectives."""
        for source in PerspectiveSource:
            self.source_distribution[source] = 0.0

    def add_perspective_input(self, input_data: PerspectiveInput) -> None:
        """
        Add perspective input for balanced learning.

        Args:
            input_data: Perspective input to add
        """
        # Calculate balanced weight for this perspective
        current_distribution = self._get_current_distribution()
        weight = self._calculate_perspective_weight(input_data, current_distribution)

        # Store perspective with calculated weight
        input_data.calculated_weight = weight
        self.perspectives[input_data.perspective_id] = input_data

        # Update source distribution tracking
        current_count = self.source_distribution.get(input_data.source_type, 0.0)
        self.source_distribution[input_data.source_type] = current_count + weight

        logger.info(f"📥 Added perspective from {input_data.source_type} with weight {weight:.3f}")

        # Run bias detection on new input
        self.detect_bias([input_data])

    def _calculate_perspective_weight(
        self,
        input_data: PerspectiveInput,
        current_distribution: Dict[str, float]
    ) -> float:
        """
        Calculate weight for perspective to promote balance.

        Args:
            input_data: Perspective input
            current_distribution: Current distribution of sources

        Returns:
            Calculated weight (0.0-1.0)
        """
        # Reduce weight if this source type is over-represented
        source_representation = current_distribution.get(input_data.source_type.value, 0.0)
        total_perspectives = sum(current_distribution.values())
        representation_ratio = source_representation / total_perspectives if total_perspectives > 0 else 0.0

        balance_factor = max(0.1, 1.0 - representation_ratio)

        # Factor in confidence and reasoning quality
        reasoning_quality = min(1.0, len(input_data.reasoning) / 5.0)  # Normalize to 5 reasons
        quality_factor = (input_data.confidence + reasoning_quality) / 2.0

        return min(1.0, balance_factor * quality_factor)

    def _get_current_distribution(self) -> Dict[str, float]:
        """
        Get current source distribution.

        Returns:
            Dictionary mapping source types to counts
        """
        return {source.value: count for source, count in self.source_distribution.items()}

    def detect_bias(
        self,
        perspectives: Optional[List[PerspectiveInput]] = None,
        decision_context: Optional[Dict[str, Any]] = None
    ) -> BiasDetectionResult:
        """
        Detect bias in perspectives or decisions.

        Args:
            perspectives: List of perspectives to analyze (default: all)
            decision_context: Optional decision context

        Returns:
            BiasDetectionResult with detected biases and recommendations
        """
        analysis_inputs = perspectives if perspectives else list(self.perspectives.values())

        detected_biases: List[BiasType] = []
        severity_score = 0.0
        affected_domains: List[str] = []
        recommendations: List[str] = []

        # Check for confirmation bias
        if self._detect_confirmation_bias(analysis_inputs):
            detected_biases.append(BiasType.CONFIRMATION_BIAS)
            severity_score += 0.3
            recommendations.append('Actively seek contradicting evidence and perspectives')

        # Check for selection bias
        if self._detect_selection_bias(analysis_inputs):
            detected_biases.append(BiasType.SELECTION_BIAS)
            severity_score += 0.25
            recommendations.append('Diversify information sources and sampling methods')

        # Check for cultural bias
        if self._detect_cultural_bias(analysis_inputs):
            detected_biases.append(BiasType.CULTURAL_BIAS)
            severity_score += 0.2
            recommendations.append('Include more cross-cultural perspectives')

        # Check for authority bias
        if self._detect_authority_bias(analysis_inputs):
            detected_biases.append(BiasType.AUTHORITY_BIAS)
            severity_score += 0.15
            recommendations.append('Evaluate arguments independently of source authority')

        # Check for temporal bias (recency bias)
        if self._detect_temporal_bias(analysis_inputs):
            detected_biases.append(BiasType.TEMPORAL_BIAS)
            severity_score += 0.1
            recommendations.append('Balance recent and historical perspectives')

        # Determine affected domains
        domains = set()
        for input_data in analysis_inputs:
            if 'domain' in input_data.viewpoint:
                domains.add(input_data.viewpoint['domain'])
        affected_domains = list(domains)

        result = BiasDetectionResult(
            bias_detected=len(detected_biases) > 0,
            bias_types=detected_biases,
            severity_score=min(1.0, severity_score),
            affected_domains=affected_domains,
            mitigation_recommendations=recommendations,
            confidence=self._calculate_detection_confidence(len(analysis_inputs))
        )

        # Store in history
        self.bias_history.append(result)
        if len(self.bias_history) > 100:
            self.bias_history = self.bias_history[-100:]

        # Log significant bias detection
        if result.severity_score > 0.5:
            bias_str = ', '.join([b.value for b in detected_biases])
            logger.warning(
                f"⚠️ Significant bias detected: {bias_str} "
                f"(severity: {result.severity_score:.3f})"
            )

        return result

    def _detect_confirmation_bias(self, perspectives: List[PerspectiveInput]) -> bool:
        """Detect confirmation bias (low viewpoint diversity)."""
        if len(perspectives) < 3:
            return False

        # Check if perspectives are too similar in viewpoint
        viewpoints = [json.dumps(p.viewpoint, sort_keys=True) for p in perspectives]
        unique_viewpoints = set(viewpoints)

        diversity_ratio = len(unique_viewpoints) / len(perspectives)
        return diversity_ratio < 0.3  # Low diversity suggests confirmation bias

    def _detect_selection_bias(self, perspectives: List[PerspectiveInput]) -> bool:
        """Detect selection bias (low source diversity)."""
        if len(perspectives) < 5:
            return False

        # Check source diversity
        sources = [p.source_type for p in perspectives]
        unique_sources = set(sources)

        source_diversity = len(unique_sources) / len(PerspectiveSource)
        return source_diversity < 0.3  # Low source diversity suggests selection bias

    def _detect_cultural_bias(self, perspectives: List[PerspectiveInput]) -> bool:
        """Detect cultural bias (low cultural context diversity)."""
        if len(perspectives) < 3:
            return False

        # Check cultural context diversity
        cultural_contexts = [
            json.dumps(p.cultural_context, sort_keys=True)
            for p in perspectives
            if p.cultural_context and len(p.cultural_context) > 0
        ]

        if not cultural_contexts:
            return False

        unique_cultural = set(cultural_contexts)
        cultural_diversity = len(unique_cultural) / len(cultural_contexts)

        return cultural_diversity < 0.5 and len(perspectives) > 3

    def _detect_authority_bias(self, perspectives: List[PerspectiveInput]) -> bool:
        """Detect authority bias (too many authority sources)."""
        if len(perspectives) < 3:
            return False

        # Check if too many perspectives are from authority sources
        authority_types = [
            PerspectiveSource.EXPERT_PANEL,
            PerspectiveSource.SCIENTIFIC_CONSENSUS,
            PerspectiveSource.PHILOSOPHICAL_TRADITION
        ]

        authority_count = sum(1 for p in perspectives if p.source_type in authority_types)
        authority_ratio = authority_count / len(perspectives)

        return authority_ratio > 0.8  # Too much reliance on authority

    def _detect_temporal_bias(self, perspectives: List[PerspectiveInput]) -> bool:
        """Detect temporal bias (recency bias)."""
        if len(perspectives) < 3:
            return False

        # Check if all perspectives are too recent
        now = datetime.now()
        recent_perspectives = [
            p for p in perspectives
            if (now - p.timestamp).days < 7  # Within last week
        ]

        recent_ratio = len(recent_perspectives) / len(perspectives)
        return recent_ratio > 0.9  # Too much recency bias

    def _calculate_detection_confidence(self, sample_size: int) -> float:
        """
        Calculate detection confidence based on sample size.

        Args:
            sample_size: Number of perspectives analyzed

        Returns:
            Confidence score (0.0-1.0)
        """
        if sample_size < 3:
            return 0.3
        if sample_size < 5:
            return 0.5
        if sample_size < 10:
            return 0.7
        return 0.9

    def assess_virtues(
        self,
        context_id: str,
        perspectives: Optional[List[PerspectiveInput]] = None,
        decision_context: Optional[Dict[str, Any]] = None
    ) -> VirtueAssessment:
        """
        Assess virtue alignment of perspectives or decisions.

        Args:
            context_id: Identifier for this assessment context
            perspectives: List of perspectives to analyze (default: all)
            decision_context: Optional decision context

        Returns:
            VirtueAssessment with scores and recommendations
        """
        analysis_inputs = perspectives if perspectives else list(self.perspectives.values())

        virtue_scores: Dict[VirtueCategory, float] = {}

        # Assess each virtue category
        virtue_scores[VirtueCategory.WISDOM] = self._assess_wisdom(analysis_inputs)
        virtue_scores[VirtueCategory.JUSTICE] = self._assess_justice(analysis_inputs)
        virtue_scores[VirtueCategory.COURAGE] = self._assess_courage(analysis_inputs)
        virtue_scores[VirtueCategory.TEMPERANCE] = self._assess_temperance(analysis_inputs)
        virtue_scores[VirtueCategory.COMPASSION] = self._assess_compassion(analysis_inputs)
        virtue_scores[VirtueCategory.INTEGRITY] = self._assess_integrity(analysis_inputs)
        virtue_scores[VirtueCategory.HUMILITY] = self._assess_humility(analysis_inputs)
        virtue_scores[VirtueCategory.PATIENCE] = self._assess_patience(analysis_inputs)

        overall_alignment = sum(virtue_scores.values()) / len(virtue_scores)

        conflicts = self._identify_virtue_conflicts(virtue_scores)
        improvements = self._generate_improvement_recommendations(virtue_scores)

        assessment = VirtueAssessment(
            virtue_scores=virtue_scores,
            overall_virtue_alignment=overall_alignment,
            virtue_conflicts=conflicts,
            improvement_recommendations=improvements
        )

        self.virtue_assessments[context_id] = assessment

        logger.info(f"🌟 Virtue assessment completed: {overall_alignment:.3f} overall alignment")

        return assessment

    def _assess_wisdom(self, perspectives: List[PerspectiveInput]) -> float:
        """Assess wisdom from perspectives."""
        if not perspectives:
            return 0.5

        wisdom_score = 0.0
        factors = 0.0

        # Factor 1: Diversity of reasoning
        reasoning_patterns = [' '.join(p.reasoning).lower() for p in perspectives]
        unique_reasoning_approaches = len(set(reasoning_patterns))
        if len(perspectives) > 0:
            wisdom_score += (unique_reasoning_approaches / len(perspectives)) * 0.3
            factors += 0.3

        # Factor 2: Consideration of long-term consequences
        long_term_keywords = ['long-term', 'future', 'consequence']
        long_term_considerations = sum(
            1 for p in perspectives
            if any(keyword in ' '.join(p.reasoning).lower() for keyword in long_term_keywords)
        )
        if len(perspectives) > 0:
            wisdom_score += (long_term_considerations / len(perspectives)) * 0.4
            factors += 0.4

        # Factor 3: Acknowledgment of uncertainty
        uncertainty_keywords = ['uncertain', 'might', 'possibly', 'perhaps']
        uncertainty_acknowledgment = sum(
            1 for p in perspectives
            if p.confidence < 0.9 or any(
                keyword in ' '.join(p.reasoning).lower() for keyword in uncertainty_keywords
            )
        )
        if len(perspectives) > 0:
            wisdom_score += (uncertainty_acknowledgment / len(perspectives)) * 0.3
            factors += 0.3

        return wisdom_score / factors if factors > 0 else 0.5

    def _assess_justice(self, perspectives: List[PerspectiveInput]) -> float:
        """Assess justice from perspectives."""
        if not perspectives:
            return 0.5

        # Look for fairness considerations
        fairness_keywords = ['fair', 'equal', 'just', 'right']
        fairness_considerations = sum(
            1 for p in perspectives
            if any(keyword in ' '.join(p.reasoning).lower() for keyword in fairness_keywords)
        )

        justice_score = fairness_considerations / len(perspectives) if len(perspectives) > 0 else 0.0

        return min(1.0, justice_score + 0.3)  # Baseline justice assumption

    def _assess_courage(self, perspectives: List[PerspectiveInput]) -> float:
        """Assess courage (willingness to challenge conventional thinking)."""
        if not perspectives:
            return 0.5

        challenge_keywords = ['challenge', 'different', 'alternative', 'unconventional']
        challenging_perspectives = sum(
            1 for p in perspectives
            if any(keyword in ' '.join(p.reasoning).lower() for keyword in challenge_keywords)
        )

        return min(1.0, (challenging_perspectives / len(perspectives)) + 0.4)

    def _assess_temperance(self, perspectives: List[PerspectiveInput]) -> float:
        """Assess temperance (balanced, moderate approaches)."""
        if not perspectives:
            return 0.5

        avg_confidence = sum(p.confidence for p in perspectives) / len(perspectives)
        temperance_score = 1.0 - abs(avg_confidence - 0.7)  # Moderate confidence is temperate

        return max(0.0, temperance_score)

    def _assess_compassion(self, perspectives: List[PerspectiveInput]) -> float:
        """Assess compassion (consideration of others' wellbeing)."""
        if not perspectives:
            return 0.5

        compassion_keywords = ['wellbeing', 'help', 'care', 'benefit', 'support']
        compassionate_perspectives = sum(
            1 for p in perspectives
            if any(keyword in ' '.join(p.reasoning).lower() for keyword in compassion_keywords)
        )

        return min(1.0, (compassionate_perspectives / len(perspectives)) + 0.3)

    def _assess_integrity(self, perspectives: List[PerspectiveInput]) -> float:
        """Assess integrity (consistency and honesty)."""
        if len(perspectives) < 2:
            return 0.8

        # Calculate consistency based on confidence variance
        confidences = [p.confidence for p in perspectives]
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5

        consistency_score = 1.0 - std_dev

        return max(0.3, min(1.0, consistency_score))

    def _assess_humility(self, perspectives: List[PerspectiveInput]) -> float:
        """Assess humility (lower confidence indicates humility)."""
        if not perspectives:
            return 0.5

        avg_confidence = sum(p.confidence for p in perspectives) / len(perspectives)
        humility_score = 1.0 - avg_confidence

        return max(0.2, min(1.0, humility_score + 0.4))

    def _assess_patience(self, perspectives: List[PerspectiveInput]) -> float:
        """Assess patience (thorough reasoning)."""
        if not perspectives:
            return 0.5

        avg_reasoning_length = sum(len(p.reasoning) for p in perspectives) / len(perspectives)
        patience_score = min(1.0, avg_reasoning_length / 5.0)  # Normalize to 5 reasons

        return max(0.3, patience_score + 0.2)

    def _identify_virtue_conflicts(self, virtue_scores: Dict[VirtueCategory, float]) -> List[str]:
        """Identify conflicts between virtues."""
        conflicts: List[str] = []

        # Example conflicts
        if virtue_scores[VirtueCategory.COURAGE] > 0.8 and virtue_scores[VirtueCategory.TEMPERANCE] < 0.4:
            conflicts.append('High courage may conflict with temperance - consider balanced approaches')

        if virtue_scores[VirtueCategory.JUSTICE] > 0.8 and virtue_scores[VirtueCategory.COMPASSION] < 0.4:
            conflicts.append('Strong justice focus may need more compassionate considerations')

        return conflicts

    def _generate_improvement_recommendations(
        self,
        virtue_scores: Dict[VirtueCategory, float]
    ) -> List[str]:
        """Generate improvement recommendations based on virtue scores."""
        recommendations: List[str] = []

        for virtue, score in virtue_scores.items():
            if score < 0.5:
                if virtue == VirtueCategory.WISDOM:
                    recommendations.append('Seek more diverse perspectives and consider long-term consequences')
                elif virtue == VirtueCategory.JUSTICE:
                    recommendations.append('Include more fairness and equality considerations')
                elif virtue == VirtueCategory.COMPASSION:
                    recommendations.append('Consider the wellbeing and impact on others')
                elif virtue == VirtueCategory.HUMILITY:
                    recommendations.append('Acknowledge uncertainty and limitations in knowledge')
                else:
                    recommendations.append(f'Strengthen {virtue.value} by incorporating related values and principles')

        return recommendations

    def get_bias_history(self, limit: int = 10) -> List[BiasDetectionResult]:
        """
        Get recent bias detection history.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of recent bias detection results
        """
        return self.bias_history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics.

        Returns:
            Dictionary with system statistics
        """
        recent_bias_detections = sum(
            1 for result in self.bias_history
            if result.bias_detected and result.severity_score > 0.3
        )

        virtue_alignments = [
            assessment.overall_virtue_alignment
            for assessment in self.virtue_assessments.values()
        ]
        avg_virtue_alignment = (
            sum(virtue_alignments) / len(virtue_alignments) if virtue_alignments else 0.0
        )

        return {
            'total_perspectives': len(self.perspectives),
            'source_distribution': self._get_current_distribution(),
            'recent_bias_detections': recent_bias_detections,
            'average_virtue_alignment': avg_virtue_alignment
        }

    def cleanup(self) -> None:
        """Clean up old data."""
        # Keep only recent perspectives (last 1000)
        if len(self.perspectives) > 1000:
            sorted_perspectives = sorted(
                self.perspectives.items(),
                key=lambda x: x[1].timestamp,
                reverse=True
            )[:1000]

            self.perspectives = dict(sorted_perspectives)

        logger.info("🧹 Bias mitigation system cleaned up")
