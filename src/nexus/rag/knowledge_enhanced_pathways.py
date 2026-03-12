"""
Knowledge-Enhanced Adaptive Learning Pathways for Nexus AI Platform.

Integrates Adaptive Learning Pathways (ALP) with Knowledge-Augmented Generation (KAG)
to create a unified system that:

1. Knowledge-Grounded Learning - Uses KAG to verify and enrich learning content
2. Coherent Pathway Generation - Ensures factual accuracy in learning sequences
3. Adaptive Content Verification - Real-time verification of learning materials
4. Knowledge Gap-Aware Progression - Aligns learning gaps with knowledge gaps
5. Domain-Coherent Assessments - Factually grounded performance evaluation
6. Retention-Optimized Review - KAG-enhanced spaced repetition

This module bridges the personalized learning algorithms of ALP with the
domain-knowledge coherence guarantees of KAG.
"""

import logging
import asyncio
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class ContentVerificationLevel(Enum):
    """Verification levels for learning content."""
    NONE = "none"              # No verification
    BASIC = "basic"            # Simple fact check
    STANDARD = "standard"      # Full KAG verification
    STRICT = "strict"          # Comprehensive with contradiction prevention


class KnowledgeIntegrationMode(Enum):
    """Modes for KAG-ALP integration."""
    PASSIVE = "passive"           # KAG assists but doesn't modify pathways
    ACTIVE = "active"             # KAG actively shapes content selection
    COLLABORATIVE = "collaborative"  # ALP and KAG work together equally
    KAG_PRIORITY = "kag_priority"    # Knowledge accuracy prioritized over personalization


@dataclass
class VerifiedLearningContent:
    """Learning content verified by KAG."""
    content_id: str
    original_content: str
    verified_content: str
    verification_score: float
    coherence_level: str
    knowledge_additions: List[Dict[str, Any]]
    corrections_made: List[Dict[str, Any]]
    related_facts: List[Dict[str, Any]]
    knowledge_gaps_addressed: List[str]
    verification_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EnhancedStudySession:
    """Study session enhanced with KAG knowledge grounding."""
    session_id: str
    user_id: str
    items: List[Dict[str, Any]]
    knowledge_context: Dict[str, Any]  # KAG grounding for session
    domain_focus: str
    coherence_score: float
    scheduled_duration: int
    verification_level: ContentVerificationLevel
    knowledge_enrichments: List[Dict[str, Any]] = field(default_factory=list)
    pre_session_grounding: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeAlignedPathway:
    """Learning pathway aligned with verified knowledge."""
    pathway_id: str
    user_id: str
    goal: str
    modules: List[Dict[str, Any]]
    knowledge_coherence_score: float
    verified_topics: List[str]
    knowledge_gaps: List[str]
    domain_ontology: Dict[str, Any]
    kag_recommendations: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class IntegratedPerformancePrediction:
    """Performance prediction combining ALP analytics and KAG knowledge assessment."""
    topic: str
    predicted_score: float
    confidence: float
    alp_factors: Dict[str, float]  # From adaptive learning
    kag_factors: Dict[str, float]  # From knowledge verification
    combined_factors: Dict[str, float]
    knowledge_readiness: float
    prerequisite_knowledge_score: float
    recommendations: List[str]


class KnowledgeEnhancedPathways:
    """
    Unified system integrating Adaptive Learning Pathways with
    Knowledge-Augmented Generation for verified, coherent learning experiences.

    Key Integration Points:
    1. Content Verification - All learning content verified by KAG
    2. Knowledge-Grounded Sequences - Prerequisites verified against knowledge graph
    3. Gap Alignment - Learning gaps aligned with knowledge gaps
    4. Coherent Assessments - Performance evaluation with fact checking
    5. Enhanced Retention - KAG context for improved spaced repetition
    """

    def __init__(
        self,
        adaptive_pathways=None,
        kag_engine=None,
        knowledge_base=None,
        integration_mode: KnowledgeIntegrationMode = KnowledgeIntegrationMode.COLLABORATIVE
    ):
        """
        Initialize Knowledge-Enhanced Pathways.

        Args:
            adaptive_pathways: AdaptiveLearningPathways instance
            kag_engine: KnowledgeAugmentedGeneration instance
            knowledge_base: KnowledgeBase for additional queries
            integration_mode: How KAG and ALP should work together
        """
        self.adaptive_pathways = adaptive_pathways
        self.kag_engine = kag_engine
        self.knowledge_base = knowledge_base
        self.integration_mode = integration_mode

        # Verified content cache
        self.verified_content_cache: Dict[str, VerifiedLearningContent] = {}

        # Knowledge-aligned pathways
        self.enhanced_pathways: Dict[str, KnowledgeAlignedPathway] = {}

        # Session knowledge contexts
        self.session_contexts: Dict[str, Dict[str, Any]] = {}

        # Integration metrics
        self.integration_metrics = {
            "content_verifications": 0,
            "pathways_enhanced": 0,
            "sessions_enriched": 0,
            "knowledge_gaps_aligned": 0,
            "corrections_applied": 0,
            "avg_coherence_score": 0.0,
            "avg_verification_score": 0.0
        }

        # Configuration
        self.config = {
            "min_verification_score": 0.6,
            "min_coherence_threshold": 0.7,
            "max_corrections_per_content": 5,
            "enable_auto_enrichment": True,
            "verification_cache_ttl": 3600,  # 1 hour
            "grounding_depth": "standard",  # light, standard, comprehensive
        }

        self.initialized = False
        logger.info(f"KnowledgeEnhancedPathways created with mode: {integration_mode.value}")

    def initialize(self):
        """Initialize the integrated system."""
        if self.initialized:
            return

        logger.info("Initializing Knowledge-Enhanced Pathways...")

        # Initialize sub-components
        if self.adaptive_pathways and hasattr(self.adaptive_pathways, 'initialize'):
            self.adaptive_pathways.initialize()

        if self.kag_engine and hasattr(self.kag_engine, 'initialize'):
            self.kag_engine.initialize()

        self.initialized = True
        logger.info("Knowledge-Enhanced Pathways initialized")

    # =========================================================================
    # Content Verification & Enrichment
    # =========================================================================

    async def verify_learning_content(
        self,
        content_id: str,
        content: str,
        topic: str,
        verification_level: ContentVerificationLevel = ContentVerificationLevel.STANDARD
    ) -> VerifiedLearningContent:
        """
        Verify and enrich learning content using KAG.

        Args:
            content_id: Unique content identifier
            content: The learning content to verify
            topic: Topic of the content
            verification_level: Depth of verification

        Returns:
            VerifiedLearningContent with verification results
        """
        if not self.initialized:
            self.initialize()

        # Check cache first
        cache_key = f"{content_id}_{verification_level.value}"
        if cache_key in self.verified_content_cache:
            cached = self.verified_content_cache[cache_key]
            cache_age = (datetime.now(timezone.utc) - cached.verification_timestamp).seconds
            if cache_age < self.config["verification_cache_ttl"]:
                return cached

        verified = VerifiedLearningContent(
            content_id=content_id,
            original_content=content,
            verified_content=content,
            verification_score=0.5,
            coherence_level="acceptable",
            knowledge_additions=[],
            corrections_made=[],
            related_facts=[],
            knowledge_gaps_addressed=[]
        )

        if not self.kag_engine or verification_level == ContentVerificationLevel.NONE:
            return verified

        try:
            # Step 1: Ground the content in knowledge
            grounding = await self.kag_engine.augment_query(content, domain=topic)

            # Step 2: Find related facts
            related_facts = grounding.grounded_facts if grounding else []
            verified.related_facts = related_facts

            # Step 3: Verify the content
            if verification_level in [ContentVerificationLevel.STANDARD, ContentVerificationLevel.STRICT]:
                verification = await self.kag_engine.verify_response(
                    response=content,
                    query=f"Learning content about {topic}",
                    context=None
                )

                verified.verification_score = verification.confidence_score
                verified.coherence_level = verification.coherence_level.value
                verified.corrections_made = verification.corrections_made
                verified.verified_content = verification.verified_response

                # Track corrections
                self.integration_metrics["corrections_applied"] += len(verification.corrections_made)

            # Step 4: Augment with knowledge additions
            if self.config["enable_auto_enrichment"]:
                augmented = await self.kag_engine.augment_context(
                    context=content,
                    grounding=grounding,
                    query=topic
                )
                verified.knowledge_additions = augmented.knowledge_additions

            # Step 5: Detect knowledge gaps addressed
            if hasattr(self.kag_engine, 'detect_knowledge_gaps'):
                gaps = await self.kag_engine.detect_knowledge_gaps(topic, content)
                verified.knowledge_gaps_addressed = [g.topic for g in gaps if g.filled]

            # Update metrics
            self.integration_metrics["content_verifications"] += 1
            self._update_avg_metric("avg_verification_score", verified.verification_score)

        except Exception as e:
            logger.warning(f"Content verification failed: {e}")
            verified.verification_score = 0.5
            verified.coherence_level = "unverified"

        # Cache result
        self.verified_content_cache[cache_key] = verified

        return verified

    async def enrich_content_with_knowledge(
        self,
        content: str,
        topic: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Enrich learning content with personalized knowledge context.

        Combines:
        - KAG knowledge grounding
        - User's existing knowledge
        - Prerequisite relationships

        Args:
            content: The content to enrich
            topic: Topic area
            user_id: User for personalization

        Returns:
            Dict with enriched content and metadata
        """
        enrichment = {
            "original_content": content,
            "enriched_content": content,
            "knowledge_context": [],
            "user_context": {},
            "personalization_applied": False
        }

        # Get user profile from ALP
        user_profile = None
        if self.adaptive_pathways:
            user_profile = self.adaptive_pathways.get_user_profile(user_id)

        # Get knowledge grounding from KAG
        if self.kag_engine:
            try:
                grounding = await self.kag_engine.augment_query(content, domain=topic)

                # Add relevant entities
                if grounding.grounded_entities:
                    enrichment["knowledge_context"].extend([
                        {
                            "type": "entity",
                            "name": e["name"],
                            "entity_type": e["type"],
                            "confidence": e["confidence"]
                        }
                        for e in grounding.grounded_entities[:5]
                    ])

                # Add relevant facts
                if grounding.grounded_facts:
                    enrichment["knowledge_context"].extend([
                        {
                            "type": "fact",
                            "content": f["content"][:200],
                            "confidence": f["confidence"]
                        }
                        for f in grounding.grounded_facts[:3]
                    ])

            except Exception as e:
                logger.warning(f"Knowledge enrichment failed: {e}")

        # Add user context
        if user_profile:
            enrichment["user_context"] = {
                "mastery_level": user_profile.mastery_scores.get(topic, 0),
                "learning_style": user_profile.learning_style.value,
                "related_strengths": [
                    s for s in user_profile.strengths if topic.lower() in s.lower()
                ][:3],
                "related_gaps": [
                    g for g in user_profile.knowledge_gaps if topic.lower() in g.lower()
                ][:3]
            }
            enrichment["personalization_applied"] = True

        # Build enriched content
        if enrichment["knowledge_context"]:
            context_summary = self._build_context_summary(enrichment["knowledge_context"])
            enrichment["enriched_content"] = f"{context_summary}\n\n{content}"

        return enrichment

    # =========================================================================
    # Knowledge-Aligned Pathway Generation
    # =========================================================================

    async def generate_knowledge_aligned_pathway(
        self,
        user_id: str,
        learning_goal: str,
        duration_weeks: int = 4,
        target_topics: Optional[List[str]] = None
    ) -> KnowledgeAlignedPathway:
        """
        Generate a learning pathway verified against knowledge base.

        This method:
        1. Uses ALP to create initial pathway
        2. Verifies pathway topics against KAG
        3. Aligns knowledge gaps with learning gaps
        4. Ensures factual coherence of learning sequence

        Args:
            user_id: User identifier
            learning_goal: Description of learning objective
            duration_weeks: Target duration
            target_topics: Specific topics to cover

        Returns:
            KnowledgeAlignedPathway with verified learning sequence
        """
        if not self.initialized:
            self.initialize()

        pathway_id = f"kep_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Step 1: Generate base pathway from ALP
        base_pathway = None
        if self.adaptive_pathways:
            base_pathway = self.adaptive_pathways.generate_learning_pathway(
                user_id=user_id,
                learning_goal=learning_goal,
                duration_weeks=duration_weeks,
                target_topics=target_topics
            )

        # Step 2: Extract topics from pathway
        pathway_topics = []
        modules = []

        if base_pathway and "weekly_modules" in base_pathway:
            for module in base_pathway["weekly_modules"]:
                pathway_topics.extend(module.get("topics", []))
                modules.append(module)
        elif target_topics:
            pathway_topics = target_topics
        else:
            pathway_topics = self._extract_topics_from_goal(learning_goal)

        # Step 3: Verify topics against knowledge base
        verified_topics = []
        knowledge_gaps = []
        domain_ontology = {}

        if self.kag_engine:
            for topic in pathway_topics:
                try:
                    # Ground each topic
                    grounding = await self.kag_engine.augment_query(
                        f"Learn about {topic}",
                        domain=topic
                    )

                    if grounding.confidence_score >= self.config["min_verification_score"]:
                        verified_topics.append(topic)

                        # Build domain ontology
                        if grounding.domain_context:
                            domain_ontology[topic] = grounding.domain_context
                    else:
                        knowledge_gaps.append(topic)

                except Exception as e:
                    logger.warning(f"Topic verification failed for {topic}: {e}")
                    knowledge_gaps.append(topic)
        else:
            verified_topics = pathway_topics

        # Step 4: Align learning gaps with knowledge gaps
        aligned_gaps = await self._align_learning_knowledge_gaps(user_id, knowledge_gaps)
        self.integration_metrics["knowledge_gaps_aligned"] += len(aligned_gaps)

        # Step 5: Calculate coherence score
        coherence_score = 0.0
        if self.kag_engine:
            try:
                coherence_report = await self.kag_engine.ensure_coherence(
                    query=learning_goal,
                    response=f"Learning pathway covering: {', '.join(verified_topics)}",
                    domain=verified_topics[0] if verified_topics else None
                )
                coherence_score = coherence_report.overall_coherence
            except Exception as e:
                logger.warning(f"Coherence check failed: {e}")
                coherence_score = 0.5

        # Step 6: Generate KAG recommendations
        kag_recommendations = self._generate_kag_pathway_recommendations(
            verified_topics, knowledge_gaps, coherence_score
        )

        # Create enhanced pathway
        enhanced_pathway = KnowledgeAlignedPathway(
            pathway_id=pathway_id,
            user_id=user_id,
            goal=learning_goal,
            modules=modules,
            knowledge_coherence_score=coherence_score,
            verified_topics=verified_topics,
            knowledge_gaps=knowledge_gaps,
            domain_ontology=domain_ontology,
            kag_recommendations=kag_recommendations
        )

        # Store pathway
        self.enhanced_pathways[pathway_id] = enhanced_pathway
        self.integration_metrics["pathways_enhanced"] += 1
        self._update_avg_metric("avg_coherence_score", coherence_score)

        logger.info(f"Generated knowledge-aligned pathway {pathway_id}: "
                   f"{len(verified_topics)} verified topics, "
                   f"{len(knowledge_gaps)} gaps, "
                   f"coherence={coherence_score:.2f}")

        return enhanced_pathway

    # =========================================================================
    # Enhanced Study Sessions
    # =========================================================================

    async def generate_enhanced_study_session(
        self,
        user_id: str,
        duration_minutes: int = 25,
        focus_topics: Optional[List[str]] = None,
        verification_level: ContentVerificationLevel = ContentVerificationLevel.STANDARD
    ) -> EnhancedStudySession:
        """
        Generate a study session with KAG-verified content.

        This method:
        1. Gets base session from ALP
        2. Grounds session in domain knowledge
        3. Verifies all content items
        4. Adds knowledge enrichments

        Args:
            user_id: User identifier
            duration_minutes: Session duration
            focus_topics: Topics to focus on
            verification_level: Verification depth

        Returns:
            EnhancedStudySession with verified content
        """
        if not self.initialized:
            self.initialize()

        session_id = f"ess_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Get base session from ALP
        base_session = None
        items = []

        if self.adaptive_pathways:
            base_session = self.adaptive_pathways.generate_study_session(
                user_id=user_id,
                duration_minutes=duration_minutes,
                focus_topics=focus_topics
            )
            items = base_session.items if base_session else []

        # Determine domain focus
        domain_focus = "general"
        if focus_topics:
            domain_focus = focus_topics[0]
        elif items:
            # Infer from content
            domain_focus = self._infer_domain_from_items(items)

        # Get pre-session knowledge grounding
        pre_session_grounding = None
        if self.kag_engine:
            try:
                grounding = await self.kag_engine.augment_query(
                    f"Study session on {domain_focus}",
                    domain=domain_focus
                )
                pre_session_grounding = {
                    "entities": grounding.grounded_entities[:5] if grounding.grounded_entities else [],
                    "facts": grounding.grounded_facts[:3] if grounding.grounded_facts else [],
                    "domain_context": grounding.domain_context,
                    "confidence": grounding.confidence_score
                }
            except Exception as e:
                logger.warning(f"Pre-session grounding failed: {e}")

        # Verify and enrich session items
        verified_items = []
        knowledge_enrichments = []
        total_coherence = 0.0

        for item in items:
            content_id = item.get("content_id", "")
            content = item.get("title", "") or item.get("content", "")

            # Verify content
            if verification_level != ContentVerificationLevel.NONE:
                verified = await self.verify_learning_content(
                    content_id=content_id,
                    content=content,
                    topic=domain_focus,
                    verification_level=verification_level
                )

                item["verification_score"] = verified.verification_score
                item["coherence_level"] = verified.coherence_level
                item["corrections"] = len(verified.corrections_made)

                total_coherence += verified.verification_score

                # Add knowledge enrichments
                if verified.knowledge_additions:
                    knowledge_enrichments.append({
                        "content_id": content_id,
                        "additions": verified.knowledge_additions
                    })

            verified_items.append(item)

        # Calculate session coherence
        coherence_score = (
            total_coherence / len(verified_items)
            if verified_items else 0.5
        )

        # Build knowledge context for session
        knowledge_context = {
            "domain": domain_focus,
            "grounding": pre_session_grounding,
            "item_count": len(verified_items),
            "verified_count": sum(
                1 for i in verified_items
                if i.get("verification_score", 0) >= self.config["min_verification_score"]
            )
        }

        # Create enhanced session
        enhanced_session = EnhancedStudySession(
            session_id=session_id,
            user_id=user_id,
            items=verified_items,
            knowledge_context=knowledge_context,
            domain_focus=domain_focus,
            coherence_score=coherence_score,
            scheduled_duration=duration_minutes,
            verification_level=verification_level,
            knowledge_enrichments=knowledge_enrichments,
            pre_session_grounding=pre_session_grounding
        )

        # Store context
        self.session_contexts[session_id] = knowledge_context
        self.integration_metrics["sessions_enriched"] += 1

        return enhanced_session

    # =========================================================================
    # Integrated Performance Prediction
    # =========================================================================

    async def predict_performance_integrated(
        self,
        user_id: str,
        topic: str
    ) -> IntegratedPerformancePrediction:
        """
        Predict performance using both ALP analytics and KAG knowledge assessment.

        Combines:
        - ALP: Learning velocity, retention, mastery scores
        - KAG: Knowledge coverage, prerequisite verification, coherence

        Args:
            user_id: User identifier
            topic: Topic to predict performance for

        Returns:
            IntegratedPerformancePrediction with combined assessment
        """
        if not self.initialized:
            self.initialize()

        # Get ALP prediction
        alp_factors = {
            "current_mastery": 0.0,
            "prerequisite_strength": 0.5,
            "learning_velocity": 0.5,
            "retention_estimate": 0.5
        }
        alp_prediction = None

        if self.adaptive_pathways:
            alp_prediction = self.adaptive_pathways.predict_performance(user_id, topic)
            alp_factors = alp_prediction.factors.copy()

        # Get KAG knowledge assessment
        kag_factors = {
            "knowledge_coverage": 0.5,
            "fact_accuracy": 0.5,
            "domain_coherence": 0.5,
            "prerequisite_verification": 0.5
        }

        if self.kag_engine:
            try:
                # Check knowledge coverage for topic
                grounding = await self.kag_engine.augment_query(topic)
                kag_factors["knowledge_coverage"] = grounding.confidence_score

                # Check domain coherence
                user_profile = None
                if self.adaptive_pathways:
                    user_profile = self.adaptive_pathways.get_user_profile(user_id)

                # Get user's current understanding
                user_understanding = f"Current knowledge about {topic}"
                if user_profile:
                    mastery = user_profile.mastery_scores.get(topic, 0)
                    user_understanding = f"User has {mastery*100:.0f}% mastery of {topic}"

                coherence_report = await self.kag_engine.ensure_coherence(
                    query=topic,
                    response=user_understanding
                )
                kag_factors["domain_coherence"] = coherence_report.overall_coherence
                kag_factors["fact_accuracy"] = coherence_report.fact_accuracy

                # Verify prerequisites
                if self.adaptive_pathways:
                    prereqs_met, missing = self.adaptive_pathways.check_prerequisites_met(
                        topic, user_id
                    )
                    if not missing:
                        kag_factors["prerequisite_verification"] = 1.0
                    else:
                        # Verify missing prerequisites against knowledge base
                        verified_prereqs = 0
                        for prereq in missing:
                            prereq_grounding = await self.kag_engine.augment_query(prereq)
                            if prereq_grounding.confidence_score > 0.5:
                                verified_prereqs += 1
                        kag_factors["prerequisite_verification"] = (
                            verified_prereqs / len(missing) if missing else 1.0
                        )

            except Exception as e:
                logger.warning(f"KAG assessment failed: {e}")

        # Combine factors based on integration mode
        combined_factors = self._combine_prediction_factors(alp_factors, kag_factors)

        # Calculate predicted score
        if self.integration_mode == KnowledgeIntegrationMode.KAG_PRIORITY:
            # Weight KAG factors more heavily
            predicted_score = (
                sum(kag_factors.values()) / len(kag_factors) * 0.6 +
                sum(alp_factors.values()) / len(alp_factors) * 0.4
            )
        elif self.integration_mode == KnowledgeIntegrationMode.PASSIVE:
            # Use mostly ALP
            predicted_score = (
                sum(alp_factors.values()) / len(alp_factors) * 0.8 +
                sum(kag_factors.values()) / len(kag_factors) * 0.2
            )
        else:
            # Collaborative - equal weight
            predicted_score = sum(combined_factors.values()) / len(combined_factors)

        predicted_score = max(0.0, min(1.0, predicted_score))

        # Calculate knowledge readiness
        knowledge_readiness = (
            kag_factors["knowledge_coverage"] * 0.4 +
            kag_factors["prerequisite_verification"] * 0.4 +
            kag_factors["domain_coherence"] * 0.2
        )

        # Calculate prerequisite knowledge score
        prereq_score = (
            alp_factors.get("prerequisite_strength", 0.5) * 0.5 +
            kag_factors["prerequisite_verification"] * 0.5
        )

        # Calculate confidence
        data_points = 0
        if alp_prediction:
            data_points += 1
        if kag_factors["knowledge_coverage"] > 0.3:
            data_points += 1
        confidence = min(0.9, 0.3 + data_points * 0.25)

        # Generate recommendations
        recommendations = self._generate_integrated_recommendations(
            alp_factors, kag_factors, predicted_score
        )

        return IntegratedPerformancePrediction(
            topic=topic,
            predicted_score=predicted_score,
            confidence=confidence,
            alp_factors=alp_factors,
            kag_factors=kag_factors,
            combined_factors=combined_factors,
            knowledge_readiness=knowledge_readiness,
            prerequisite_knowledge_score=prereq_score,
            recommendations=recommendations
        )

    # =========================================================================
    # Knowledge Gap Alignment
    # =========================================================================

    async def align_and_fill_gaps(
        self,
        user_id: str,
        learning_topics: List[str]
    ) -> Dict[str, Any]:
        """
        Align learning gaps with knowledge gaps and create filling strategy.

        Args:
            user_id: User identifier
            learning_topics: Topics user is learning

        Returns:
            Dict with gap analysis and filling strategy
        """
        result = {
            "learning_gaps": [],
            "knowledge_gaps": [],
            "aligned_gaps": [],
            "filling_strategy": [],
            "priority_topics": []
        }

        # Get learning gaps from ALP
        if self.adaptive_pathways:
            user_profile = self.adaptive_pathways.get_user_profile(user_id)
            if user_profile:
                result["learning_gaps"] = list(user_profile.knowledge_gaps)

        # Get knowledge gaps from KAG
        if self.kag_engine:
            try:
                for topic in learning_topics:
                    gaps = await self.kag_engine.detect_knowledge_gaps(
                        topic, f"Learning about {topic}", None
                    )
                    result["knowledge_gaps"].extend([
                        {"topic": g.topic, "severity": g.severity, "fillable": g.auto_fillable}
                        for g in gaps
                    ])
            except Exception as e:
                logger.warning(f"Knowledge gap detection failed: {e}")

        # Find aligned gaps (appear in both)
        learning_gap_set = set(result["learning_gaps"])
        knowledge_gap_topics = {g["topic"] for g in result["knowledge_gaps"]}

        aligned = learning_gap_set & knowledge_gap_topics
        result["aligned_gaps"] = list(aligned)

        # Create filling strategy
        for topic in aligned:
            kg_info = next(
                (g for g in result["knowledge_gaps"] if g["topic"] == topic),
                {}
            )

            strategy = {
                "topic": topic,
                "priority": "high" if kg_info.get("severity") == "critical" else "medium",
                "approach": "auto_fill" if kg_info.get("fillable") else "manual_study",
                "estimated_effort": "moderate"
            }
            result["filling_strategy"].append(strategy)

        # Identify priority topics
        result["priority_topics"] = [
            s["topic"] for s in result["filling_strategy"]
            if s["priority"] == "high"
        ][:5]

        return result

    async def _align_learning_knowledge_gaps(
        self,
        user_id: str,
        knowledge_gaps: List[str]
    ) -> List[str]:
        """Align knowledge gaps with user's learning profile."""
        aligned = []

        if not self.adaptive_pathways:
            return knowledge_gaps

        user_profile = self.adaptive_pathways.get_user_profile(user_id)
        if not user_profile:
            return knowledge_gaps

        # Check which knowledge gaps align with user's learning gaps
        for gap in knowledge_gaps:
            # Check if gap is in user's knowledge gaps
            if gap in user_profile.knowledge_gaps:
                aligned.append(gap)
            # Check if gap is a topic user is working on
            elif gap in user_profile.topics_in_progress:
                aligned.append(gap)
                # Add to user's knowledge gaps
                user_profile.knowledge_gaps.append(gap)

        return aligned

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_context_summary(self, knowledge_context: List[Dict]) -> str:
        """Build a summary from knowledge context."""
        lines = ["[Knowledge Context]"]

        entities = [k for k in knowledge_context if k.get("type") == "entity"]
        facts = [k for k in knowledge_context if k.get("type") == "fact"]

        if entities:
            entity_names = [e["name"] for e in entities[:3]]
            lines.append(f"Related concepts: {', '.join(entity_names)}")

        if facts:
            for fact in facts[:2]:
                lines.append(f"- {fact['content'][:100]}")

        return "\n".join(lines)

    def _extract_topics_from_goal(self, goal: str) -> List[str]:
        """Extract topics from learning goal."""
        # Simple extraction - can be enhanced with NLP
        words = goal.lower().split()
        topics = [w for w in words if len(w) > 4 and w.isalpha()]
        return topics[:5] if topics else ["general"]

    def _infer_domain_from_items(self, items: List[Dict]) -> str:
        """Infer domain from session items."""
        if not items:
            return "general"

        # Look for topic/domain indicators
        for item in items:
            if "topic" in item:
                return item["topic"]
            if "domain" in item:
                return item["domain"]

        return "general"

    def _generate_kag_pathway_recommendations(
        self,
        verified_topics: List[str],
        knowledge_gaps: List[str],
        coherence_score: float
    ) -> List[str]:
        """Generate KAG-informed pathway recommendations."""
        recommendations = []

        if coherence_score < 0.6:
            recommendations.append("Consider simplifying pathway to improve coherence")

        if knowledge_gaps:
            recommendations.append(f"Address knowledge gaps: {', '.join(knowledge_gaps[:3])}")

        if len(verified_topics) < 3:
            recommendations.append("Limited verified content available - supplement with external resources")

        if coherence_score >= 0.8:
            recommendations.append("High coherence pathway - ready for advanced content")

        return recommendations

    def _combine_prediction_factors(
        self,
        alp_factors: Dict[str, float],
        kag_factors: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine ALP and KAG factors into unified prediction."""
        combined = {}

        # Map and combine related factors
        combined["mastery"] = alp_factors.get("current_mastery", 0.5)
        combined["prerequisites"] = (
            alp_factors.get("prerequisite_strength", 0.5) * 0.5 +
            kag_factors.get("prerequisite_verification", 0.5) * 0.5
        )
        combined["knowledge_coverage"] = kag_factors.get("knowledge_coverage", 0.5)
        combined["learning_velocity"] = alp_factors.get("learning_velocity", 0.5)
        combined["retention"] = alp_factors.get("retention_estimate", 0.5)
        combined["domain_coherence"] = kag_factors.get("domain_coherence", 0.5)

        return combined

    def _generate_integrated_recommendations(
        self,
        alp_factors: Dict[str, float],
        kag_factors: Dict[str, float],
        predicted_score: float
    ) -> List[str]:
        """Generate recommendations from integrated analysis."""
        recommendations = []

        # ALP-based recommendations
        if alp_factors.get("prerequisite_strength", 1.0) < 0.6:
            recommendations.append("Review prerequisite topics first")

        if alp_factors.get("retention_estimate", 1.0) < 0.5:
            recommendations.append("Schedule spaced repetition reviews")

        # KAG-based recommendations
        if kag_factors.get("knowledge_coverage", 1.0) < 0.5:
            recommendations.append("Topic has limited knowledge base coverage")

        if kag_factors.get("domain_coherence", 1.0) < 0.6:
            recommendations.append("Focus on building coherent domain understanding")

        # Combined recommendations
        if predicted_score < 0.5:
            recommendations.append("Consider breaking topic into smaller sub-topics")
        elif predicted_score > 0.8:
            recommendations.append("Ready for assessment or advanced content")

        return recommendations[:5]

    def _update_avg_metric(self, metric_name: str, new_value: float):
        """Update running average for a metric."""
        current = self.integration_metrics[metric_name]
        count = self.integration_metrics["content_verifications"] or 1
        self.integration_metrics[metric_name] = (
            current * (count - 1) + new_value
        ) / count

    # =========================================================================
    # Statistics & Analytics
    # =========================================================================

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        return {
            "integration_mode": self.integration_mode.value,
            "metrics": self.integration_metrics.copy(),
            "configuration": self.config.copy(),
            "components": {
                "alp_available": self.adaptive_pathways is not None,
                "kag_available": self.kag_engine is not None,
                "knowledge_base_available": self.knowledge_base is not None
            },
            "cache_status": {
                "verified_content_cached": len(self.verified_content_cache),
                "enhanced_pathways": len(self.enhanced_pathways),
                "session_contexts": len(self.session_contexts)
            }
        }

    def get_pathway(self, pathway_id: str) -> Optional[KnowledgeAlignedPathway]:
        """Get an enhanced pathway by ID."""
        return self.enhanced_pathways.get(pathway_id)

    def get_session_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get knowledge context for a session."""
        return self.session_contexts.get(session_id)

    def clear_cache(self):
        """Clear all caches."""
        self.verified_content_cache.clear()
        self.session_contexts.clear()
        logger.info("KnowledgeEnhancedPathways cache cleared")


# Factory function
def create_knowledge_enhanced_pathways(
    adaptive_pathways=None,
    kag_engine=None,
    knowledge_base=None,
    mode: str = "collaborative"
) -> KnowledgeEnhancedPathways:
    """
    Create a Knowledge-Enhanced Pathways instance.

    Args:
        adaptive_pathways: AdaptiveLearningPathways instance
        kag_engine: KnowledgeAugmentedGeneration instance
        knowledge_base: KnowledgeBase instance
        mode: Integration mode ('passive', 'active', 'collaborative', 'kag_priority')

    Returns:
        Configured KnowledgeEnhancedPathways instance
    """
    try:
        integration_mode = KnowledgeIntegrationMode(mode)
    except ValueError:
        integration_mode = KnowledgeIntegrationMode.COLLABORATIVE

    return KnowledgeEnhancedPathways(
        adaptive_pathways=adaptive_pathways,
        kag_engine=kag_engine,
        knowledge_base=knowledge_base,
        integration_mode=integration_mode
    )
