"""
Advanced Bias Mitigation and Virtue Assessment

Comprehensive bias detection and mitigation system with:
- 8 bias type detection with severity levels
- Automated mitigation recommendations
- Perspective balancing with auto-weighting
- 8 virtue assessments for ethical evaluation
- Integration with ethics frameworks

Phase 5 Week 21-22

Example:
    >>> from unified_intelligence.safety import AdvancedBiasMitigator, BiasType, VirtueType
    >>>
    >>> # Initialize mitigator
    >>> mitigator = AdvancedBiasMitigator()
    >>>
    >>> # Detect biases in content
    >>> biases = mitigator.detect_biases(
    ...     content="Recent studies show...",
    ...     context={"sources": ["source1"], "author_background": "..."}
    ... )
    >>>
    >>> # Assess virtues
    >>> virtues = mitigator.assess_virtues(
    ...     content="The decision prioritizes fairness and compassion...",
    ...     context={}
    ... )
    >>>
    >>> # Get mitigation recommendations
    >>> for bias in biases:
    ...     if bias.severity in [BiasSeverity.HIGH, BiasSeverity.CRITICAL]:
    ...         print(f"Mitigation: {bias.mitigation_recommendation}")
"""
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import Counter

logger = logging.getLogger(__name__)


class BiasType(str, Enum):
    """Types of cognitive and social biases."""
    CONFIRMATION = "confirmation"  # Favoring information that confirms beliefs
    SELECTION = "selection"  # Non-random data selection
    CULTURAL = "cultural"  # Cultural perspective bias
    COGNITIVE = "cognitive"  # General cognitive biases
    TEMPORAL = "temporal"  # Recency or historical bias
    AUTHORITY = "authority"  # Over-reliance on authority
    ANCHORING = "anchoring"  # Over-reliance on first information
    AVAILABILITY = "availability"  # Over-weighting easily recalled information


class BiasSeverity(str, Enum):
    """Severity levels for detected biases."""
    LOW = "low"  # Minor bias, minimal impact
    MEDIUM = "medium"  # Moderate bias, noticeable impact
    HIGH = "high"  # Significant bias, major impact
    CRITICAL = "critical"  # Severe bias, fundamental impact


class VirtueType(str, Enum):
    """Types of virtues for ethical assessment."""
    WISDOM = "wisdom"  # Practical wisdom and good judgment
    JUSTICE = "justice"  # Fairness and equity
    COURAGE = "courage"  # Moral courage and conviction
    TEMPERANCE = "temperance"  # Self-control and moderation
    COMPASSION = "compassion"  # Empathy and caring
    INTEGRITY = "integrity"  # Honesty and moral consistency
    HUMILITY = "humility"  # Intellectual humility and openness
    PATIENCE = "patience"  # Tolerance and long-term thinking


@dataclass
class BiasDetection:
    """
    Detected bias in content.

    Attributes:
        bias_type: Type of bias detected
        severity: Severity level
        confidence: Detection confidence (0-1)
        description: Description of the bias
        evidence: Evidence supporting detection
        mitigation_recommendation: Suggested mitigation
        affected_content: Content segments affected
        timestamp: When bias was detected
    """
    bias_type: BiasType
    severity: BiasSeverity
    confidence: float
    description: str
    evidence: List[str] = field(default_factory=list)
    mitigation_recommendation: str = ""
    affected_content: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class VirtueAssessment:
    """
    Assessment of virtue in content.

    Attributes:
        virtue_type: Type of virtue assessed
        score: Virtue score (0-1, higher = more virtuous)
        confidence: Assessment confidence (0-1)
        reasoning: Explanation of assessment
        examples: Example evidence
        recommendations: Recommendations for improvement
    """
    virtue_type: VirtueType
    score: float
    confidence: float
    reasoning: str
    examples: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerspectiveBalance:
    """
    Balance of perspectives in content.

    Attributes:
        perspectives: Detected perspectives
        weights: Weight of each perspective (0-1)
        balance_score: Overall balance (0-1, higher = more balanced)
        underrepresented: Perspectives that need more weight
        overrepresented: Perspectives with too much weight
        recommendations: Balance improvement recommendations
    """
    perspectives: List[str]
    weights: Dict[str, float]
    balance_score: float
    underrepresented: List[str] = field(default_factory=list)
    overrepresented: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class AdvancedBiasMitigator:
    """
    Advanced bias detection and mitigation system.

    Features:
    - Detects 8 types of cognitive and social biases
    - Assigns severity levels (low, medium, high, critical)
    - Provides mitigation recommendations
    - Balances perspectives with auto-weighting
    - Assesses 8 virtues for ethical evaluation
    - Integrates with ethics frameworks

    Example:
        >>> mitigator = AdvancedBiasMitigator()
        >>> biases = mitigator.detect_biases(content, context)
        >>> virtues = mitigator.assess_virtues(content, context)
        >>> balance = mitigator.balance_perspectives(content, context)
    """

    def __init__(
        self,
        bias_threshold: float = 0.6,
        virtue_threshold: float = 0.7,
        balance_threshold: float = 0.5,
    ):
        """
        Initialize bias mitigator.

        Args:
            bias_threshold: Minimum confidence for bias detection
            virtue_threshold: Minimum score for virtue presence
            balance_threshold: Minimum score for balanced perspectives
        """
        self.bias_threshold = bias_threshold
        self.virtue_threshold = virtue_threshold
        self.balance_threshold = balance_threshold

        # Bias detection patterns and indicators
        self._bias_detectors = self._initialize_bias_detectors()
        self._virtue_assessors = self._initialize_virtue_assessors()

        logger.info("AdvancedBiasMitigator initialized")

    def detect_biases(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[BiasDetection]:
        """
        Detect biases in content.

        Args:
            content: Content to analyze
            context: Additional context for detection

        Returns:
            List of detected biases

        Example:
            >>> biases = mitigator.detect_biases(
            ...     content="Studies consistently show X is true...",
            ...     context={"sources": ["study1"], "date": "2024"}
            ... )
        """
        context = context or {}
        detected_biases = []

        for bias_type in BiasType:
            detector = self._bias_detectors.get(bias_type)
            if detector:
                bias = detector(content, context)
                if bias and bias.confidence >= self.bias_threshold:
                    detected_biases.append(bias)

        logger.info(f"Detected {len(detected_biases)} biases in content")
        return detected_biases

    def assess_virtues(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[VirtueAssessment]:
        """
        Assess virtues in content.

        Args:
            content: Content to assess
            context: Additional context

        Returns:
            List of virtue assessments

        Example:
            >>> virtues = mitigator.assess_virtues(
            ...     content="We carefully considered all perspectives...",
            ...     context={}
            ... )
        """
        context = context or {}
        assessments = []

        for virtue_type in VirtueType:
            assessor = self._virtue_assessors.get(virtue_type)
            if assessor:
                assessment = assessor(content, context)
                assessments.append(assessment)

        logger.info(f"Assessed {len(assessments)} virtues in content")
        return assessments

    def balance_perspectives(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PerspectiveBalance:
        """
        Analyze and balance perspectives in content.

        Args:
            content: Content to analyze
            context: Additional context with known perspectives

        Returns:
            Perspective balance analysis

        Example:
            >>> balance = mitigator.balance_perspectives(
            ...     content="From a Western viewpoint...",
            ...     context={"known_perspectives": ["Western", "Eastern"]}
            ... )
        """
        context = context or {}

        # Detect perspectives mentioned
        perspectives = self._detect_perspectives(content, context)

        # Calculate weights
        weights = self._calculate_perspective_weights(content, perspectives)

        # Calculate balance score
        balance_score = self._calculate_balance_score(weights)

        # Identify under/over-represented
        mean_weight = 1.0 / len(perspectives) if perspectives else 0
        underrepresented = [p for p, w in weights.items() if w < mean_weight * 0.7]
        overrepresented = [p for p, w in weights.items() if w > mean_weight * 1.3]

        # Generate recommendations
        recommendations = []
        if balance_score < self.balance_threshold:
            if underrepresented:
                recommendations.append(
                    f"Include more content from: {', '.join(underrepresented)}"
                )
            if overrepresented:
                recommendations.append(
                    f"Reduce emphasis on: {', '.join(overrepresented)}"
                )

        return PerspectiveBalance(
            perspectives=perspectives,
            weights=weights,
            balance_score=balance_score,
            underrepresented=underrepresented,
            overrepresented=overrepresented,
            recommendations=recommendations,
        )

    def get_mitigation_summary(
        self,
        biases: List[BiasDetection],
        virtues: List[VirtueAssessment],
        balance: PerspectiveBalance,
    ) -> Dict[str, Any]:
        """
        Get comprehensive mitigation summary.

        Args:
            biases: Detected biases
            virtues: Virtue assessments
            balance: Perspective balance

        Returns:
            Summary with overall scores and recommendations
        """
        # Calculate overall bias score (0 = no bias, 1 = severe bias)
        if biases:
            severity_scores = {
                BiasSeverity.LOW: 0.25,
                BiasSeverity.MEDIUM: 0.5,
                BiasSeverity.HIGH: 0.75,
                BiasSeverity.CRITICAL: 1.0,
            }
            bias_score = sum(severity_scores[b.severity] * b.confidence for b in biases) / len(biases)
        else:
            bias_score = 0.0

        # Calculate overall virtue score
        virtue_score = sum(v.score for v in virtues) / len(virtues) if virtues else 0.0

        # Priority recommendations
        priority_recs = []

        # Critical biases first
        critical_biases = [b for b in biases if b.severity == BiasSeverity.CRITICAL]
        if critical_biases:
            priority_recs.append(
                f"URGENT: Address {len(critical_biases)} critical biases"
            )

        # Low virtues
        low_virtues = [v for v in virtues if v.score < 0.4]
        if low_virtues:
            priority_recs.append(
                f"Strengthen virtues: {', '.join(v.virtue_type.value for v in low_virtues)}"
            )

        # Poor balance
        if balance.balance_score < self.balance_threshold:
            priority_recs.append("Improve perspective balance")

        return {
            "overall_bias_score": bias_score,
            "overall_virtue_score": virtue_score,
            "perspective_balance_score": balance.balance_score,
            "bias_count": len(biases),
            "critical_bias_count": len(critical_biases),
            "virtue_count": len(virtues),
            "priority_recommendations": priority_recs,
            "needs_mitigation": bias_score > 0.5 or virtue_score < 0.5 or balance.balance_score < 0.5,
        }

    # Private methods

    def _initialize_bias_detectors(self) -> Dict[BiasType, callable]:
        """Initialize bias detection functions."""
        return {
            BiasType.CONFIRMATION: self._detect_confirmation_bias,
            BiasType.SELECTION: self._detect_selection_bias,
            BiasType.CULTURAL: self._detect_cultural_bias,
            BiasType.COGNITIVE: self._detect_cognitive_bias,
            BiasType.TEMPORAL: self._detect_temporal_bias,
            BiasType.AUTHORITY: self._detect_authority_bias,
            BiasType.ANCHORING: self._detect_anchoring_bias,
            BiasType.AVAILABILITY: self._detect_availability_bias,
        }

    def _initialize_virtue_assessors(self) -> Dict[VirtueType, callable]:
        """Initialize virtue assessment functions."""
        return {
            VirtueType.WISDOM: self._assess_wisdom,
            VirtueType.JUSTICE: self._assess_justice,
            VirtueType.COURAGE: self._assess_courage,
            VirtueType.TEMPERANCE: self._assess_temperance,
            VirtueType.COMPASSION: self._assess_compassion,
            VirtueType.INTEGRITY: self._assess_integrity,
            VirtueType.HUMILITY: self._assess_humility,
            VirtueType.PATIENCE: self._assess_patience,
        }

    def _detect_confirmation_bias(self, content: str, context: Dict) -> Optional[BiasDetection]:
        """Detect confirmation bias."""
        indicators = [
            r"studies consistently show",
            r"it is well known",
            r"everyone knows",
            r"obviously",
            r"clearly",
            r"undoubtedly",
        ]

        matches = sum(1 for pattern in indicators if re.search(pattern, content, re.I))
        confidence = min(1.0, matches * 0.3)

        if confidence >= self.bias_threshold:
            severity = (
                BiasSeverity.CRITICAL if confidence > 0.9
                else BiasSeverity.HIGH if confidence > 0.75
                else BiasSeverity.MEDIUM if confidence > 0.6
                else BiasSeverity.LOW
            )

            return BiasDetection(
                bias_type=BiasType.CONFIRMATION,
                severity=severity,
                confidence=confidence,
                description="Content may favor confirming information over contradicting evidence",
                evidence=[f"Found {matches} confirmation bias indicators"],
                mitigation_recommendation="Present counterarguments and alternative perspectives",
            )
        return None

    def _detect_selection_bias(self, content: str, context: Dict) -> Optional[BiasDetection]:
        """Detect selection bias."""
        sources = context.get("sources", [])
        source_diversity = len(set(sources)) / max(len(sources), 1) if sources else 0.5

        confidence = 1.0 - source_diversity

        if confidence >= self.bias_threshold:
            return BiasDetection(
                bias_type=BiasType.SELECTION,
                severity=BiasSeverity.MEDIUM,
                confidence=confidence,
                description="Limited source diversity may indicate selection bias",
                evidence=[f"Source diversity: {source_diversity:.2f}"],
                mitigation_recommendation="Include sources from diverse origins and perspectives",
            )
        return None

    def _detect_cultural_bias(self, content: str, context: Dict) -> Optional[BiasDetection]:
        """Detect cultural bias."""
        western_centric = [
            r"\bWestern\b",
            r"\bAmerican\b",
            r"\bEuropean\b",
        ]
        eastern_centric = [
            r"\bEastern\b",
            r"\bAsian\b",
        ]

        western_count = sum(len(re.findall(p, content, re.I)) for p in western_centric)
        eastern_count = sum(len(re.findall(p, content, re.I)) for p in eastern_centric)

        total = western_count + eastern_count
        if total > 0:
            imbalance = abs(western_count - eastern_count) / total
            confidence = min(1.0, imbalance * 1.5)

            if confidence >= self.bias_threshold:
                return BiasDetection(
                    bias_type=BiasType.CULTURAL,
                    severity=BiasSeverity.MEDIUM,
                    confidence=confidence,
                    description="Content shows cultural perspective imbalance",
                    evidence=[f"Cultural reference imbalance: {imbalance:.2f}"],
                    mitigation_recommendation="Include perspectives from diverse cultural backgrounds",
                )
        return None

    def _detect_cognitive_bias(self, content: str, context: Dict) -> Optional[BiasDetection]:
        """Detect general cognitive biases."""
        # Placeholder - would implement specific cognitive bias detection
        return None

    def _detect_temporal_bias(self, content: str, context: Dict) -> Optional[BiasDetection]:
        """Detect temporal bias (recency/historical)."""
        recency_indicators = [r"recent", r"latest", r"new", r"modern", r"today"]
        historical_indicators = [r"traditional", r"classical", r"historical", r"ancient"]

        recent_count = sum(len(re.findall(p, content, re.I)) for p in recency_indicators)
        historical_count = sum(len(re.findall(p, content, re.I)) for p in historical_indicators)

        total = recent_count + historical_count
        if total > 3:
            imbalance = abs(recent_count - historical_count) / total
            confidence = min(1.0, imbalance * 1.2)

            if confidence >= self.bias_threshold:
                return BiasDetection(
                    bias_type=BiasType.TEMPORAL,
                    severity=BiasSeverity.LOW,
                    confidence=confidence,
                    description="Content shows temporal perspective bias",
                    evidence=[f"Temporal imbalance: {imbalance:.2f}"],
                    mitigation_recommendation="Balance recent and historical perspectives",
                )
        return None

    def _detect_authority_bias(self, content: str, context: Dict) -> Optional[BiasDetection]:
        """Detect authority bias."""
        authority_markers = [
            r"expert",
            r"authority",
            r"professor",
            r"PhD",
            r"according to",
        ]

        matches = sum(len(re.findall(p, content, re.I)) for p in authority_markers)
        confidence = min(1.0, matches * 0.25)

        if confidence >= self.bias_threshold:
            return BiasDetection(
                bias_type=BiasType.AUTHORITY,
                severity=BiasSeverity.LOW,
                confidence=confidence,
                description="Over-reliance on authority citations",
                evidence=[f"Authority references: {matches}"],
                mitigation_recommendation="Support claims with evidence beyond authority",
            )
        return None

    def _detect_anchoring_bias(self, content: str, context: Dict) -> Optional[BiasDetection]:
        """Detect anchoring bias."""
        first_statement = context.get("first_statement", "")
        if first_statement and first_statement.lower() in content.lower():
            return BiasDetection(
                bias_type=BiasType.ANCHORING,
                severity=BiasSeverity.MEDIUM,
                confidence=0.7,
                description="Possible anchoring to initial information",
                mitigation_recommendation="Consider information independently of initial framing",
            )
        return None

    def _detect_availability_bias(self, content: str, context: Dict) -> Optional[BiasDetection]:
        """Detect availability bias."""
        easily_recalled = [r"remember", r"recall", r"familiar", r"well-known"]

        matches = sum(len(re.findall(p, content, re.I)) for p in easily_recalled)
        confidence = min(1.0, matches * 0.3)

        if confidence >= self.bias_threshold:
            return BiasDetection(
                bias_type=BiasType.AVAILABILITY,
                severity=BiasSeverity.LOW,
                confidence=confidence,
                description="May be overweighting easily recalled information",
                mitigation_recommendation="Seek systematic evidence beyond memorable examples",
            )
        return None

    def _assess_wisdom(self, content: str, context: Dict) -> VirtueAssessment:
        """Assess wisdom virtue."""
        wisdom_markers = [
            r"consider",
            r"reflect",
            r"thoughtful",
            r"balanced",
            r"nuanced",
            r"complex",
        ]

        matches = sum(len(re.findall(p, content, re.I)) for p in wisdom_markers)
        score = min(1.0, matches * 0.2)

        return VirtueAssessment(
            virtue_type=VirtueType.WISDOM,
            score=score,
            confidence=0.7,
            reasoning=f"Wisdom indicators: {matches}",
            recommendations=["Show more nuanced thinking"] if score < 0.5 else [],
        )

    def _assess_justice(self, content: str, context: Dict) -> VirtueAssessment:
        """Assess justice virtue."""
        justice_markers = [r"fair", r"equal", r"just", r"equity", r"rights"]

        matches = sum(len(re.findall(p, content, re.I)) for p in justice_markers)
        score = min(1.0, matches * 0.25)

        return VirtueAssessment(
            virtue_type=VirtueType.JUSTICE,
            score=score,
            confidence=0.7,
            reasoning=f"Justice indicators: {matches}",
        )

    def _assess_courage(self, content: str, context: Dict) -> VirtueAssessment:
        """Assess courage virtue."""
        courage_markers = [r"challenge", r"question", r"oppose", r"stand for"]

        matches = sum(len(re.findall(p, content, re.I)) for p in courage_markers)
        score = min(1.0, matches * 0.3)

        return VirtueAssessment(
            virtue_type=VirtueType.COURAGE,
            score=score,
            confidence=0.6,
            reasoning=f"Courage indicators: {matches}",
        )

    def _assess_temperance(self, content: str, context: Dict) -> VirtueAssessment:
        """Assess temperance virtue."""
        temperance_markers = [r"moderate", r"balanced", r"restrained", r"measured"]

        matches = sum(len(re.findall(p, content, re.I)) for p in temperance_markers)
        score = min(1.0, matches * 0.3)

        return VirtueAssessment(
            virtue_type=VirtueType.TEMPERANCE,
            score=score,
            confidence=0.7,
            reasoning=f"Temperance indicators: {matches}",
        )

    def _assess_compassion(self, content: str, context: Dict) -> VirtueAssessment:
        """Assess compassion virtue."""
        compassion_markers = [r"care", r"empathy", r"compassion", r"kindness", r"support"]

        matches = sum(len(re.findall(p, content, re.I)) for p in compassion_markers)
        score = min(1.0, matches * 0.25)

        return VirtueAssessment(
            virtue_type=VirtueType.COMPASSION,
            score=score,
            confidence=0.7,
            reasoning=f"Compassion indicators: {matches}",
        )

    def _assess_integrity(self, content: str, context: Dict) -> VirtueAssessment:
        """Assess integrity virtue."""
        integrity_markers = [r"honest", r"truth", r"transparent", r"consistent"]

        matches = sum(len(re.findall(p, content, re.I)) for p in integrity_markers)
        score = min(1.0, matches * 0.3)

        return VirtueAssessment(
            virtue_type=VirtueType.INTEGRITY,
            score=score,
            confidence=0.7,
            reasoning=f"Integrity indicators: {matches}",
        )

    def _assess_humility(self, content: str, context: Dict) -> VirtueAssessment:
        """Assess humility virtue."""
        humility_markers = [r"may be", r"might", r"possibly", r"uncertain", r"don't know"]

        matches = sum(len(re.findall(p, content, re.I)) for p in humility_markers)
        score = min(1.0, matches * 0.25)

        return VirtueAssessment(
            virtue_type=VirtueType.HUMILITY,
            score=score,
            confidence=0.6,
            reasoning=f"Humility indicators: {matches}",
        )

    def _assess_patience(self, content: str, context: Dict) -> VirtueAssessment:
        """Assess patience virtue."""
        patience_markers = [r"long-term", r"gradual", r"patient", r"sustained", r"over time"]

        matches = sum(len(re.findall(p, content, re.I)) for p in patience_markers)
        score = min(1.0, matches * 0.3)

        return VirtueAssessment(
            virtue_type=VirtueType.PATIENCE,
            score=score,
            confidence=0.6,
            reasoning=f"Patience indicators: {matches}",
        )

    def _detect_perspectives(self, content: str, context: Dict) -> List[str]:
        """Detect mentioned perspectives."""
        # Start with known perspectives from context
        perspectives = list(context.get("known_perspectives", []))

        # Detect common perspective markers
        perspective_patterns = {
            "scientific": [r"scientific", r"empirical", r"research"],
            "economic": [r"economic", r"financial", r"market"],
            "social": [r"social", r"community", r"society"],
            "environmental": [r"environmental", r"ecological", r"sustainability"],
            "ethical": [r"ethical", r"moral", r"values"],
            "technical": [r"technical", r"engineering", r"technological"],
        }

        for perspective, patterns in perspective_patterns.items():
            if any(re.search(p, content, re.I) for p in patterns):
                if perspective not in perspectives:
                    perspectives.append(perspective)

        return perspectives if perspectives else ["general"]

    def _calculate_perspective_weights(
        self,
        content: str,
        perspectives: List[str],
    ) -> Dict[str, float]:
        """Calculate weight of each perspective."""
        if not perspectives:
            return {}

        # Simple word count-based weighting
        weights = {}
        total_words = len(content.split())

        for perspective in perspectives:
            # Count mentions (simplified)
            mentions = len(re.findall(perspective, content, re.I))
            weight = mentions / max(total_words / 100, 1)  # Normalize
            weights[perspective] = weight

        # Normalize to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {p: w / total_weight for p, w in weights.items()}

        return weights

    def _calculate_balance_score(self, weights: Dict[str, float]) -> float:
        """Calculate balance score (0-1, higher = more balanced)."""
        if not weights:
            return 0.0

        # Perfect balance would be equal weights
        ideal_weight = 1.0 / len(weights)

        # Calculate deviation from ideal
        deviations = [abs(w - ideal_weight) for w in weights.values()]
        avg_deviation = sum(deviations) / len(deviations)

        # Convert to balance score (lower deviation = higher score)
        balance_score = 1.0 - (avg_deviation * 2)  # Scale appropriately

        return max(0.0, min(1.0, balance_score))
