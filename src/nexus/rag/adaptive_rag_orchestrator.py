"""
Adaptive RAG Orchestrator that combines vectorized retrieval, pattern recognition,
context window management, learning pathways, and Knowledge-Augmented Generation (KAG)
for 150M context handling with enhanced domain-knowledge coherence.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import statistics

# Type checking imports for KAG and Knowledge-Enhanced Pathways
if TYPE_CHECKING:
    from nexus.rag.knowledge_augmented_generation import (
        KnowledgeAugmentedGeneration,
        KnowledgeGrounding,
        AugmentedContext,
        VerifiedResponse,
        DomainCoherenceReport,
    )
    from nexus.rag.knowledge_enhanced_pathways import (
        KnowledgeEnhancedPathways,
        KnowledgeAlignedPathway,
        EnhancedStudySession,
        IntegratedPerformancePrediction,
    )

logger = logging.getLogger(__name__)

class OrchestrationMode(Enum):
    REACTIVE = "reactive"  # Respond to queries
    PROACTIVE = "proactive"  # Anticipate needs
    ADAPTIVE = "adaptive"  # Learn and adapt
    COLLABORATIVE = "collaborative"  # Work with users
    FOCUSED = "focused" # Focused workflow
    EXPLORATORY = "exploratory" # Exploratory workflow
    SYNTHESIS = "synthesis" # Synthesis workflow

class LearningIntensity(Enum):
    PASSIVE = 1
    MODERATE = 3
    INTENSIVE = 5
    HYPER = 10

@dataclass
class OrchestrationRequest:
    """Request for RAG orchestration."""
    request_id: str
    query: str
    context: Dict[str, Any]
    user_profile: Optional[Dict[str, Any]]
    learning_goals: List[str]
    preferred_mode: OrchestrationMode
    max_context_length: int
    response_format: str
    timestamp: datetime

@dataclass
class OrchestrationResponse:
    """Response from RAG orchestration."""
    request_id: str
    generated_content: str
    retrieved_knowledge: List[Dict[str, Any]]
    learning_pathway_updates: Dict[str, Any]
    context_window_id: str
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    confidence_score: float
    processing_time: float
    # KAG-enhanced fields (optional)
    kag_grounding: Optional[Dict[str, Any]] = None
    kag_coherence: Optional[Dict[str, Any]] = None
    kag_verification: Optional[Dict[str, Any]] = None
    knowledge_gaps: Optional[List[str]] = None
    # Knowledge-Enhanced Pathways fields (optional)
    enhanced_pathway: Optional[Dict[str, Any]] = None
    enhanced_session: Optional[Dict[str, Any]] = None
    performance_prediction: Optional[Dict[str, Any]] = None

class AdaptiveRAGOrchestrator:
    """
    Advanced orchestrator that combines RAG vectorization, pattern recognition,
    context window management, adaptive learning, and Knowledge-Augmented Generation (KAG)
    for optimal knowledge delivery with enhanced domain-knowledge coherence.

    KAG Integration provides:
    - Pre-retrieval knowledge grounding
    - Post-retrieval context augmentation
    - Response verification against knowledge base
    - Domain coherence optimization
    - Knowledge gap detection and filling
    """

    def __init__(self, rag_engine, context_manager, pattern_engine,
                 adaptive_pathways, knowledge_base,
                 kag_engine: Optional['KnowledgeAugmentedGeneration'] = None,
                 enable_kag: bool = True,
                 enable_knowledge_enhanced_pathways: bool = True):
        """
        Initialize the Adaptive RAG Orchestrator.

        Args:
            rag_engine: RAG vector engine for retrieval
            context_manager: Context window manager
            pattern_engine: Pattern recognition engine
            adaptive_pathways: Adaptive learning pathways
            knowledge_base: Knowledge base for facts
            kag_engine: Optional KAG engine (will create one if not provided and enable_kag=True)
            enable_kag: Whether to enable KAG integration (default: True)
            enable_knowledge_enhanced_pathways: Whether to enable KEP integration (default: True)
        """
        self.rag_engine = rag_engine
        self.context_manager = context_manager
        self.pattern_engine = pattern_engine
        self.adaptive_pathways = adaptive_pathways
        self.knowledge_base = knowledge_base

        # KAG Integration
        self.enable_kag = enable_kag
        self._kag_engine = kag_engine
        self._kag_initialized = False

        # Knowledge-Enhanced Pathways Integration
        self.enable_kep = enable_knowledge_enhanced_pathways
        self._kep_engine: Optional['KnowledgeEnhancedPathways'] = None
        self._kep_initialized = False

        # Orchestration state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.learning_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_analytics: Dict[str, List[float]] = {
            "response_time": [],
            "user_satisfaction": [],
            "knowledge_accuracy": [],
            "learning_effectiveness": [],
            "kag_coherence_scores": [],  # KAG-specific metric
            "kag_verification_scores": []  # KAG-specific metric
        }

        # Configuration
        self.max_concurrent_sessions = 50
        self.context_optimization_interval = 300  # 5 minutes
        self.learning_update_threshold = 10  # Updates after 10 interactions

        # Adaptive parameters
        self.adaptation_rates = {
            "knowledge_weighting": 0.1,
            "pattern_recognition": 0.05,
            "context_prioritization": 0.15,
            "learning_pathway_adjustment": 0.2
        }

        # KAG configuration
        self.kag_config = {
            "min_coherence_threshold": 0.6,
            "verify_responses": True,
            "detect_knowledge_gaps": True,
            "augment_context": True,
            "ground_queries": True,
            "auto_fill_gaps": False
        }

        self.initialized = False
        self.workflows: Dict[str, callable] = {}  # Define workflows attribute

    @property
    def kag(self) -> Optional['KnowledgeAugmentedGeneration']:
        """Get the KAG engine, initializing if needed."""
        if not self.enable_kag:
            return None

        if self._kag_engine is None and not self._kag_initialized:
            self._initialize_kag()

        return self._kag_engine

    def _initialize_kag(self):
        """Initialize the KAG engine if not already provided."""
        if self._kag_initialized:
            return

        try:
            from nexus.rag.knowledge_augmented_generation import create_kag

            # Create KAG with available knowledge sources
            self._kag_engine = create_kag(
                knowledge_base=self.knowledge_base,
                mode='comprehensive'
            )
            self._kag_engine.initialize()
            self._kag_initialized = True
            logger.info("KAG engine initialized successfully")

        except ImportError as e:
            logger.warning(f"KAG module not available: {e}")
            self.enable_kag = False
            self._kag_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize KAG engine: {e}")
            self.enable_kag = False
            self._kag_initialized = True

    def configure_kag(self, **kwargs):
        """
        Configure KAG settings.

        Args:
            min_coherence_threshold: Minimum coherence score (0-1)
            verify_responses: Whether to verify responses against knowledge
            detect_knowledge_gaps: Whether to detect knowledge gaps
            augment_context: Whether to augment context with knowledge
            ground_queries: Whether to ground queries in facts
            auto_fill_gaps: Whether to auto-fill detected knowledge gaps
        """
        for key, value in kwargs.items():
            if key in self.kag_config:
                self.kag_config[key] = value
                logger.debug(f"KAG config updated: {key}={value}")

    # ==================== Knowledge-Enhanced Pathways Integration ====================

    @property
    def kep(self) -> Optional['KnowledgeEnhancedPathways']:
        """Get the Knowledge-Enhanced Pathways engine, initializing if needed."""
        if not self.enable_kep:
            return None

        if self._kep_engine is None and not self._kep_initialized:
            self._initialize_kep()

        return self._kep_engine

    def _initialize_kep(self):
        """Initialize the Knowledge-Enhanced Pathways engine."""
        if self._kep_initialized:
            return

        try:
            from nexus.rag.knowledge_enhanced_pathways import create_knowledge_enhanced_pathways

            # Create KEP with available components
            self._kep_engine = create_knowledge_enhanced_pathways(
                adaptive_pathways=self.adaptive_pathways,
                kag_engine=self.kag,
                knowledge_base=self.knowledge_base,
                mode='collaborative'
            )
            self._kep_engine.initialize()
            self._kep_initialized = True
            logger.info("Knowledge-Enhanced Pathways engine initialized successfully")

        except ImportError as e:
            logger.warning(f"KEP module not available: {e}")
            self.enable_kep = False
            self._kep_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize KEP engine: {e}")
            self.enable_kep = False
            self._kep_initialized = True

    def configure_kep(self, **kwargs):
        """
        Configure Knowledge-Enhanced Pathways settings.

        Args:
            min_verification_score: Minimum verification score (0-1)
            min_coherence_threshold: Minimum coherence threshold (0-1)
            enable_auto_enrichment: Whether to auto-enrich content
            verification_cache_ttl: Cache TTL in seconds
        """
        if self.kep:
            for key, value in kwargs.items():
                if key in self.kep.config:
                    self.kep.config[key] = value
                    logger.debug(f"KEP config updated: {key}={value}")

    async def generate_knowledge_aligned_pathway(
        self,
        user_id: str,
        learning_goal: str,
        duration_weeks: int = 4,
        target_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a knowledge-aligned learning pathway using KEP.

        This provides pathways verified against the knowledge base with:
        - Verified topic coverage
        - Knowledge gap alignment
        - Domain coherence scoring
        - KAG-informed recommendations

        Args:
            user_id: User identifier
            learning_goal: Learning goal description
            duration_weeks: Target duration in weeks
            target_topics: Optional specific topics

        Returns:
            Dict with pathway details and KAG verification
        """
        if not self.kep:
            # Fallback to standard pathway generation
            if self.adaptive_pathways:
                return self.adaptive_pathways.generate_learning_pathway(
                    user_id=user_id,
                    learning_goal=learning_goal,
                    duration_weeks=duration_weeks,
                    target_topics=target_topics
                )
            return {"error": "No pathway generation available"}

        try:
            pathway = await self.kep.generate_knowledge_aligned_pathway(
                user_id=user_id,
                learning_goal=learning_goal,
                duration_weeks=duration_weeks,
                target_topics=target_topics
            )

            return {
                "pathway_id": pathway.pathway_id,
                "user_id": pathway.user_id,
                "goal": pathway.goal,
                "modules": pathway.modules,
                "knowledge_coherence_score": pathway.knowledge_coherence_score,
                "verified_topics": pathway.verified_topics,
                "knowledge_gaps": pathway.knowledge_gaps,
                "kag_recommendations": pathway.kag_recommendations,
                "created_at": pathway.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate knowledge-aligned pathway: {e}")
            return {"error": str(e)}

    async def generate_enhanced_study_session(
        self,
        user_id: str,
        duration_minutes: int = 25,
        focus_topics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate an enhanced study session with KAG verification.

        Args:
            user_id: User identifier
            duration_minutes: Session duration
            focus_topics: Topics to focus on

        Returns:
            Dict with session details and verification
        """
        if not self.kep:
            # Fallback to standard session generation
            if self.adaptive_pathways:
                session = self.adaptive_pathways.generate_study_session(
                    user_id=user_id,
                    duration_minutes=duration_minutes,
                    focus_topics=focus_topics
                )
                return {
                    "session_id": session.session_id,
                    "items": session.items,
                    "scheduled_duration": session.scheduled_duration
                }
            return {"error": "No session generation available"}

        try:
            from nexus.rag.knowledge_enhanced_pathways import ContentVerificationLevel

            session = await self.kep.generate_enhanced_study_session(
                user_id=user_id,
                duration_minutes=duration_minutes,
                focus_topics=focus_topics,
                verification_level=ContentVerificationLevel.STANDARD
            )

            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "items": session.items,
                "knowledge_context": session.knowledge_context,
                "domain_focus": session.domain_focus,
                "coherence_score": session.coherence_score,
                "scheduled_duration": session.scheduled_duration,
                "knowledge_enrichments": session.knowledge_enrichments,
                "pre_session_grounding": session.pre_session_grounding
            }
        except Exception as e:
            logger.error(f"Failed to generate enhanced study session: {e}")
            return {"error": str(e)}

    async def predict_performance_with_knowledge(
        self,
        user_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Predict performance combining ALP and KAG analysis.

        Args:
            user_id: User identifier
            topic: Topic to predict performance for

        Returns:
            Dict with integrated prediction
        """
        if not self.kep:
            # Fallback to standard prediction
            if self.adaptive_pathways:
                prediction = self.adaptive_pathways.predict_performance(user_id, topic)
                return {
                    "topic": prediction.topic,
                    "predicted_score": prediction.predicted_score,
                    "confidence": prediction.confidence,
                    "factors": prediction.factors,
                    "recommendations": prediction.recommendations
                }
            return {"error": "No prediction available"}

        try:
            prediction = await self.kep.predict_performance_integrated(user_id, topic)

            return {
                "topic": prediction.topic,
                "predicted_score": prediction.predicted_score,
                "confidence": prediction.confidence,
                "alp_factors": prediction.alp_factors,
                "kag_factors": prediction.kag_factors,
                "combined_factors": prediction.combined_factors,
                "knowledge_readiness": prediction.knowledge_readiness,
                "prerequisite_knowledge_score": prediction.prerequisite_knowledge_score,
                "recommendations": prediction.recommendations
            }
        except Exception as e:
            logger.error(f"Failed to predict performance: {e}")
            return {"error": str(e)}

    async def align_learning_and_knowledge_gaps(
        self,
        user_id: str,
        learning_topics: List[str]
    ) -> Dict[str, Any]:
        """
        Align learning gaps with knowledge gaps for comprehensive gap analysis.

        Args:
            user_id: User identifier
            learning_topics: Topics user is learning

        Returns:
            Dict with aligned gaps and filling strategy
        """
        if not self.kep:
            return {"error": "KEP not available for gap alignment"}

        try:
            return await self.kep.align_and_fill_gaps(user_id, learning_topics)
        except Exception as e:
            logger.error(f"Failed to align gaps: {e}")
            return {"error": str(e)}

    def get_kep_statistics(self) -> Dict[str, Any]:
        """Get Knowledge-Enhanced Pathways statistics."""
        if not self.kep:
            return {"enabled": False}

        return {
            "enabled": True,
            **self.kep.get_integration_statistics()
        }

    def _setup_orchestration_workflows(self):
        """Set up orchestration workflows for different request types."""
        self.workflows = {
            OrchestrationMode.ADAPTIVE: self._adaptive_workflow,
            OrchestrationMode.FOCUSED: self._focused_workflow,
            OrchestrationMode.EXPLORATORY: self._exploratory_workflow,
            OrchestrationMode.SYNTHESIS: self._synthesis_workflow,
            OrchestrationMode.REACTIVE: self._reactive_workflow,
            OrchestrationMode.PROACTIVE: self._proactive_workflow,
            OrchestrationMode.COLLABORATIVE: self._collaborative_workflow,
        }

    def _adaptive_workflow(self, request: 'OrchestrationRequest') -> 'OrchestrationResponse':
        """
        Adaptive workflow that intelligently selects the best approach based on query characteristics.

        Analyzes the query to determine optimal workflow:
        - Specific/technical queries → Focused workflow
        - Broad/exploratory queries → Exploratory workflow
        - Multi-faceted queries → Synthesis workflow
        - Quick/simple queries → Reactive workflow
        - Context-dependent queries → Proactive workflow
        - Ambiguous queries → Collaborative workflow
        """
        import time
        start_time = time.time()

        # Analyze query to determine best workflow
        workflow_selection = self._select_optimal_workflow(request)
        selected_workflow = workflow_selection["workflow"]
        selection_confidence = workflow_selection["confidence"]

        # If high confidence in a specific workflow, delegate to it
        if selection_confidence >= 0.7 and selected_workflow != "adaptive":
            workflow_map = {
                "focused": self._focused_workflow,
                "exploratory": self._exploratory_workflow,
                "synthesis": self._synthesis_workflow,
                "reactive": self._reactive_workflow,
                "proactive": self._proactive_workflow,
                "collaborative": self._collaborative_workflow
            }

            if selected_workflow in workflow_map:
                delegated_response = workflow_map[selected_workflow](request)
                # Add metadata about adaptive delegation
                delegated_response.performance_metrics["adaptive_delegated_to"] = selected_workflow
                delegated_response.performance_metrics["delegation_confidence"] = selection_confidence
                return delegated_response

        # Hybrid adaptive approach: combine strategies
        knowledge_items = self.knowledge_base.query_knowledge(request.query, max_results=15)

        # Apply multi-strategy scoring
        scored_items = []
        for item in knowledge_items:
            # Calculate multiple relevance metrics
            base_confidence = item.confidence

            # Exact match bonus
            query_lower = request.query.lower()
            content_lower = str(item.content).lower()
            exact_match_bonus = 0.2 if query_lower in content_lower else 0.0

            # Keyword coverage
            query_words = set(query_lower.split())
            content_words = set(content_lower.split())
            coverage = len(query_words & content_words) / max(len(query_words), 1)

            # Combined adaptive score
            adaptive_score = (
                base_confidence * 0.4 +
                coverage * 0.3 +
                exact_match_bonus +
                0.1  # Base floor
            )

            scored_items.append({
                "content": str(item.content),
                "confidence": base_confidence,
                "adaptive_score": min(adaptive_score, 1.0),
                "exact_match": exact_match_bonus > 0,
                "keyword_coverage": coverage
            })

        # Sort by adaptive score
        scored_items.sort(key=lambda x: x["adaptive_score"], reverse=True)
        retrieved_knowledge = scored_items[:10]

        # Generate adaptive response
        if retrieved_knowledge:
            top_item = retrieved_knowledge[0]
            generated_content = f"Adaptive analysis: {top_item['content']}"

            # Add context from supporting items
            if len(retrieved_knowledge) > 1:
                supporting = retrieved_knowledge[1]["content"][:100]
                generated_content += f" Additional context: {supporting}"
        else:
            generated_content = f"Adaptive response for: {request.query}"

        # Calculate confidence
        avg_score = (
            statistics.mean([item["adaptive_score"] for item in retrieved_knowledge])
            if retrieved_knowledge else 0.0
        )

        processing_time = time.time() - start_time

        return OrchestrationResponse(
            request_id=request.request_id,
            generated_content=generated_content,
            retrieved_knowledge=retrieved_knowledge,
            learning_pathway_updates={
                "workflow_analysis": workflow_selection,
                "strategies_considered": list(workflow_selection.get("scores", {}).keys())
            },
            context_window_id=f"adaptive_{request.request_id}",
            performance_metrics={
                "retrieval_count": len(retrieved_knowledge),
                "avg_adaptive_score": avg_score,
                "workflow_selected": selected_workflow,
                "selection_confidence": selection_confidence,
                "hybrid_mode": True
            },
            recommendations=self._generate_adaptive_recommendations(
                workflow_selection, retrieved_knowledge
            ),
            confidence_score=avg_score * selection_confidence,
            processing_time=processing_time
        )

    def _select_optimal_workflow(self, request: 'OrchestrationRequest') -> Dict[str, Any]:
        """
        Analyze request to select the optimal workflow.

        Returns workflow name and confidence score.
        """
        query = request.query.lower()
        scores = {
            "focused": 0.0,
            "exploratory": 0.0,
            "synthesis": 0.0,
            "reactive": 0.0,
            "proactive": 0.0,
            "collaborative": 0.0
        }

        # Focused indicators: specific terms, technical queries
        focused_indicators = ["specific", "exactly", "precisely", "only", "just"]
        technical_patterns = ["how to", "function", "method", "api", "error", "bug"]
        scores["focused"] += sum(0.15 for ind in focused_indicators if ind in query)
        scores["focused"] += sum(0.1 for pat in technical_patterns if pat in query)

        # Exploratory indicators: broad terms, discovery intent
        exploratory_indicators = ["explore", "discover", "overview", "what are", "types of"]
        broad_patterns = ["options", "alternatives", "possibilities", "different"]
        scores["exploratory"] += sum(0.15 for ind in exploratory_indicators if ind in query)
        scores["exploratory"] += sum(0.1 for pat in broad_patterns if pat in query)

        # Synthesis indicators: comparison, combination intent
        synthesis_indicators = ["compare", "versus", "vs", "combine", "together"]
        multi_source_patterns = ["sources", "perspectives", "approaches", "both"]
        scores["synthesis"] += sum(0.15 for ind in synthesis_indicators if ind in query)
        scores["synthesis"] += sum(0.1 for pat in multi_source_patterns if pat in query)

        # Reactive indicators: quick, simple queries
        reactive_indicators = ["quick", "fast", "simple", "basic"]
        scores["reactive"] += sum(0.15 for ind in reactive_indicators if ind in query)
        if len(query.split()) < 5:
            scores["reactive"] += 0.2  # Short queries favor reactive

        # Proactive indicators: learning context, session-aware
        proactive_indicators = ["next", "then", "after", "continue", "follow up"]
        scores["proactive"] += sum(0.15 for ind in proactive_indicators if ind in query)
        if request.learning_goals:
            scores["proactive"] += 0.1 * min(len(request.learning_goals), 3)

        # Collaborative indicators: ambiguity, need for clarification
        collaborative_indicators = ["help me", "suggest", "recommend", "what should"]
        ambiguous_terms = ["thing", "stuff", "it", "this", "maybe"]
        scores["collaborative"] += sum(0.15 for ind in collaborative_indicators if ind in query)
        scores["collaborative"] += sum(0.1 for term in ambiguous_terms if term in query.split())

        # Normalize and select best
        max_score = max(scores.values()) if scores.values() else 0.0

        if max_score > 0:
            # Normalize to 0-1 range based on indicators found
            confidence = min(max_score, 1.0)
            selected = max(scores, key=scores.get)
        else:
            confidence = 0.3
            selected = "adaptive"

        return {
            "workflow": selected,
            "confidence": confidence,
            "scores": scores,
            "analysis": {
                "query_length": len(query.split()),
                "has_learning_goals": bool(request.learning_goals)
            }
        }

    def _generate_adaptive_recommendations(self, workflow_selection: Dict,
                                          results: List[Dict]) -> List[str]:
        """Generate recommendations based on adaptive analysis."""
        recommendations = []
        scores = workflow_selection.get("scores", {})

        # Suggest alternative workflows if scores are close
        sorted_workflows = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_workflows) >= 2:
            top_workflow, top_score = sorted_workflows[0]
            second_workflow, second_score = sorted_workflows[1]

            if top_score - second_score < 0.1 and second_score > 0.3:
                recommendations.append(f"Try {second_workflow} mode for different perspective")

        # Result-based recommendations
        if not results:
            recommendations.append("Try broadening your query")
        elif len(results) > 8:
            recommendations.append("Consider focused mode for precision")

        recommendations.append("Refine query for better results")
        return recommendations[:3]

    def _focused_workflow(self, request: 'OrchestrationRequest') -> 'OrchestrationResponse':
        """
        Focused workflow for specific, targeted queries.

        Uses deep, narrow retrieval strategy:
        - Prioritizes exact matches and high-confidence results
        - Applies strict relevance filtering
        - Focuses on single domain/topic depth
        - Minimal exploration, maximum precision
        """
        import time
        start_time = time.time()

        # Analyze query for domain focus
        query_analysis = self._analyze_query_focus(request.query)
        target_domain = query_analysis.get("primary_domain", "general")

        # Deep retrieval with strict filtering
        knowledge_items = self.knowledge_base.query_knowledge(
            request.query,
            max_results=20  # Retrieve more, filter strictly
        )

        # Apply strict relevance filtering for focused results
        filtered_items = []
        relevance_threshold = 0.7  # High threshold for focus

        for item in knowledge_items:
            relevance_score = self._calculate_item_relevance(
                item, request.query, target_domain
            )
            if relevance_score >= relevance_threshold:
                filtered_items.append({
                    "content": str(item.content),
                    "confidence": item.confidence,
                    "relevance": relevance_score,
                    "domain": target_domain
                })

        # Sort by combined confidence and relevance
        filtered_items.sort(
            key=lambda x: x["confidence"] * 0.4 + x["relevance"] * 0.6,
            reverse=True
        )

        # Take top focused results
        retrieved_knowledge = filtered_items[:5]

        # Generate focused response
        if retrieved_knowledge:
            # Synthesize from top results
            top_content = retrieved_knowledge[0]["content"]
            supporting_content = " ".join(
                item["content"][:100] for item in retrieved_knowledge[1:3]
            )
            generated_content = f"Focused analysis: {top_content}"
            if supporting_content:
                generated_content += f" Supporting context: {supporting_content}"
        else:
            generated_content = f"No highly relevant results found for focused query: {request.query}"

        # Calculate focused confidence (higher bar)
        avg_confidence = (
            statistics.mean([item["confidence"] for item in retrieved_knowledge])
            if retrieved_knowledge else 0.0
        )
        avg_relevance = (
            statistics.mean([item["relevance"] for item in retrieved_knowledge])
            if retrieved_knowledge else 0.0
        )
        confidence_score = avg_confidence * 0.5 + avg_relevance * 0.5

        processing_time = time.time() - start_time

        return OrchestrationResponse(
            request_id=request.request_id,
            generated_content=generated_content,
            retrieved_knowledge=retrieved_knowledge,
            learning_pathway_updates={"focus_domain": target_domain},
            context_window_id=f"focused_{request.request_id}",
            performance_metrics={
                "retrieval_count": len(retrieved_knowledge),
                "avg_relevance": avg_relevance,
                "focus_precision": len(filtered_items) / max(len(knowledge_items), 1),
                "target_domain": target_domain
            },
            recommendations=self._generate_focused_recommendations(
                request.query, target_domain, retrieved_knowledge
            ),
            confidence_score=confidence_score,
            processing_time=processing_time
        )

    def _exploratory_workflow(self, request: 'OrchestrationRequest') -> 'OrchestrationResponse':
        """
        Exploratory workflow for broad, discovery-oriented queries.

        Uses breadth-first discovery strategy:
        - Explores multiple domains and topics
        - Lower relevance threshold to capture diverse results
        - Identifies related concepts and connections
        - Prioritizes coverage over precision
        """
        import time
        start_time = time.time()

        # Expand query to related concepts
        expanded_queries = self._expand_query_for_exploration(request.query)

        # Collect results from multiple query variations
        all_results = []
        domain_coverage = {}

        for query_variant in expanded_queries:
            knowledge_items = self.knowledge_base.query_knowledge(
                query_variant,
                max_results=10
            )

            for item in knowledge_items:
                # Lower threshold for exploratory mode
                relevance_score = self._calculate_item_relevance(
                    item, request.query, "general"
                )

                if relevance_score >= 0.3:  # Lower threshold
                    domain = self._identify_item_domain(item)
                    domain_coverage[domain] = domain_coverage.get(domain, 0) + 1

                    all_results.append({
                        "content": str(item.content),
                        "confidence": item.confidence,
                        "relevance": relevance_score,
                        "domain": domain,
                        "query_variant": query_variant,
                        "exploration_depth": len(expanded_queries)
                    })

        # Deduplicate by content similarity
        unique_results = self._deduplicate_results(all_results)

        # Sort by diversity score (balance relevance with domain coverage)
        for result in unique_results:
            domain = result["domain"]
            # Boost underrepresented domains
            domain_count = domain_coverage.get(domain, 1)
            diversity_boost = 1.0 / (domain_count ** 0.5)
            result["diversity_score"] = (
                result["relevance"] * 0.5 +
                result["confidence"] * 0.3 +
                diversity_boost * 0.2
            )

        unique_results.sort(key=lambda x: x["diversity_score"], reverse=True)
        retrieved_knowledge = unique_results[:10]

        # Generate exploratory summary
        domains_explored = list(set(r["domain"] for r in retrieved_knowledge))
        generated_content = self._generate_exploration_summary(
            request.query, retrieved_knowledge, domains_explored
        )

        # Calculate exploration coverage metrics
        coverage_score = len(domains_explored) / max(len(domain_coverage), 1)
        avg_confidence = (
            statistics.mean([item["confidence"] for item in retrieved_knowledge])
            if retrieved_knowledge else 0.0
        )

        processing_time = time.time() - start_time

        return OrchestrationResponse(
            request_id=request.request_id,
            generated_content=generated_content,
            retrieved_knowledge=retrieved_knowledge,
            learning_pathway_updates={
                "explored_domains": domains_explored,
                "query_expansions": expanded_queries
            },
            context_window_id=f"exploratory_{request.request_id}",
            performance_metrics={
                "retrieval_count": len(retrieved_knowledge),
                "domains_explored": len(domains_explored),
                "coverage_score": coverage_score,
                "query_expansions": len(expanded_queries),
                "total_candidates": len(all_results)
            },
            recommendations=self._generate_exploration_recommendations(
                domains_explored, retrieved_knowledge
            ),
            confidence_score=avg_confidence * coverage_score,
            processing_time=processing_time
        )

    def _synthesis_workflow(self, request: 'OrchestrationRequest') -> 'OrchestrationResponse':
        """
        Synthesis workflow for combining multiple knowledge sources.

        Multi-source combination strategy:
        - Retrieves from diverse sources
        - Cross-references information for validation
        - Identifies agreements and contradictions
        - Produces synthesized, coherent output
        """
        import time
        start_time = time.time()

        # Retrieve from multiple retrieval strategies
        retrieval_sources = {}

        # Source 1: Direct knowledge base query
        direct_results = self.knowledge_base.query_knowledge(
            request.query, max_results=10
        )
        retrieval_sources["direct"] = [
            {"content": str(item.content), "confidence": item.confidence}
            for item in direct_results
        ]

        # Source 2: Pattern-based retrieval (if pattern engine available)
        if hasattr(self.pattern_engine, 'find_matching_patterns'):
            pattern_results = self.pattern_engine.find_matching_patterns(request.query)
            retrieval_sources["pattern"] = [
                {"content": str(p.get("pattern", "")), "confidence": p.get("confidence", 0.5)}
                for p in pattern_results[:5]
            ]
        else:
            retrieval_sources["pattern"] = []

        # Source 3: Context-based retrieval (if context manager available)
        if hasattr(self.context_manager, 'get_relevant_context'):
            context_results = self.context_manager.get_relevant_context(request.query)
            retrieval_sources["context"] = [
                {"content": str(c.get("content", "")), "confidence": c.get("relevance", 0.5)}
                for c in context_results[:5]
            ] if context_results else []
        else:
            retrieval_sources["context"] = []

        # Cross-reference and validate
        cross_referenced = self._cross_reference_sources(retrieval_sources)

        # Identify agreements and contradictions
        synthesis_analysis = self._analyze_source_agreement(cross_referenced)

        # Build synthesized knowledge
        synthesized_knowledge = []
        for item in cross_referenced:
            source_count = item.get("source_count", 1)
            agreement_score = item.get("agreement_score", 0.5)

            # Boost confidence for multi-source agreement
            adjusted_confidence = min(
                item["confidence"] * (1 + 0.2 * (source_count - 1)) * agreement_score,
                1.0
            )

            synthesized_knowledge.append({
                "content": item["content"],
                "confidence": adjusted_confidence,
                "sources": item.get("sources", ["unknown"]),
                "source_count": source_count,
                "agreement_score": agreement_score,
                "synthesis_type": "multi-source" if source_count > 1 else "single-source"
            })

        # Sort by adjusted confidence
        synthesized_knowledge.sort(key=lambda x: x["confidence"], reverse=True)
        retrieved_knowledge = synthesized_knowledge[:10]

        # Generate synthesized response
        generated_content = self._generate_synthesis_response(
            request.query, retrieved_knowledge, synthesis_analysis
        )

        # Calculate synthesis quality metrics
        multi_source_count = sum(
            1 for k in retrieved_knowledge if k["source_count"] > 1
        )
        avg_agreement = (
            statistics.mean([k["agreement_score"] for k in retrieved_knowledge])
            if retrieved_knowledge else 0.0
        )
        avg_confidence = (
            statistics.mean([k["confidence"] for k in retrieved_knowledge])
            if retrieved_knowledge else 0.0
        )

        processing_time = time.time() - start_time

        return OrchestrationResponse(
            request_id=request.request_id,
            generated_content=generated_content,
            retrieved_knowledge=retrieved_knowledge,
            learning_pathway_updates={
                "synthesis_analysis": synthesis_analysis,
                "source_distribution": {
                    k: len(v) for k, v in retrieval_sources.items()
                }
            },
            context_window_id=f"synthesis_{request.request_id}",
            performance_metrics={
                "retrieval_count": len(retrieved_knowledge),
                "multi_source_items": multi_source_count,
                "avg_agreement_score": avg_agreement,
                "source_types_used": len([k for k, v in retrieval_sources.items() if v]),
                "contradictions_found": synthesis_analysis.get("contradictions", 0)
            },
            recommendations=self._generate_synthesis_recommendations(
                synthesis_analysis, retrieved_knowledge
            ),
            confidence_score=avg_confidence * (1 + avg_agreement) / 2,
            processing_time=processing_time
        )

    def _reactive_workflow(self, request: 'OrchestrationRequest') -> 'OrchestrationResponse':
        """
        Reactive workflow for immediate response optimization.

        Optimized for speed and directness:
        - Minimal processing overhead
        - Quick retrieval with caching
        - Direct response generation
        - Prioritizes response time over depth
        """
        import time
        start_time = time.time()

        # Check cache first for quick response
        cache_key = self._generate_cache_key(request.query)
        cached_response = self._check_response_cache(cache_key)

        if cached_response:
            # Return cached response with minimal overhead
            cached_response["performance_metrics"]["cache_hit"] = True
            cached_response["processing_time"] = time.time() - start_time
            return OrchestrationResponse(**cached_response)

        # Quick retrieval with limited results
        knowledge_items = self.knowledge_base.query_knowledge(
            request.query,
            max_results=5  # Limited for speed
        )

        retrieved_knowledge = [
            {
                "content": str(item.content),
                "confidence": item.confidence,
                "retrieval_mode": "reactive"
            }
            for item in knowledge_items
        ]

        # Quick response generation
        if retrieved_knowledge:
            # Use top result directly
            generated_content = retrieved_knowledge[0]["content"]
            confidence_score = retrieved_knowledge[0]["confidence"]
        else:
            generated_content = f"Quick response for: {request.query}"
            confidence_score = 0.5

        processing_time = time.time() - start_time

        # Cache the response for future reactive calls
        response_data = {
            "request_id": request.request_id,
            "generated_content": generated_content,
            "retrieved_knowledge": retrieved_knowledge,
            "learning_pathway_updates": {},
            "context_window_id": f"reactive_{request.request_id}",
            "performance_metrics": {
                "retrieval_count": len(retrieved_knowledge),
                "cache_hit": False,
                "response_mode": "reactive"
            },
            "recommendations": ["Quick follow-up available"],
            "confidence_score": confidence_score,
            "processing_time": processing_time
        }

        self._cache_response(cache_key, response_data)

        return OrchestrationResponse(**response_data)

    def _proactive_workflow(self, request: 'OrchestrationRequest') -> 'OrchestrationResponse':
        """
        Proactive workflow for predictive, anticipatory approach.

        Anticipates user needs based on:
        - Session history and patterns
        - Learning goals and progress
        - Predicted next questions
        - Pre-fetched relevant knowledge
        """
        import time
        start_time = time.time()

        # Analyze session history for patterns
        session_id = f"session_{request.request_id.split('_')[0]}"
        session = self.active_sessions.get(session_id, {})
        session_history = session.get("context_history", [])

        # Predict likely follow-up topics
        predicted_topics = self._predict_next_topics(
            request.query, session_history, request.learning_goals
        )

        # Retrieve knowledge for current query
        current_results = self.knowledge_base.query_knowledge(
            request.query, max_results=8
        )

        # Pre-fetch knowledge for predicted topics
        anticipated_knowledge = []
        for topic in predicted_topics[:3]:  # Limit pre-fetching
            topic_results = self.knowledge_base.query_knowledge(
                topic, max_results=3
            )
            for item in topic_results:
                anticipated_knowledge.append({
                    "content": str(item.content),
                    "confidence": item.confidence * 0.8,  # Slightly lower for predictions
                    "anticipated_topic": topic,
                    "retrieval_type": "proactive"
                })

        # Combine current and anticipated results
        retrieved_knowledge = [
            {
                "content": str(item.content),
                "confidence": item.confidence,
                "retrieval_type": "current"
            }
            for item in current_results
        ] + anticipated_knowledge

        # Generate proactive response with anticipation
        generated_content = self._generate_proactive_response(
            request.query,
            retrieved_knowledge,
            predicted_topics
        )

        # Calculate proactive metrics
        avg_confidence = (
            statistics.mean([k["confidence"] for k in retrieved_knowledge])
            if retrieved_knowledge else 0.0
        )

        processing_time = time.time() - start_time

        return OrchestrationResponse(
            request_id=request.request_id,
            generated_content=generated_content,
            retrieved_knowledge=retrieved_knowledge,
            learning_pathway_updates={
                "predicted_topics": predicted_topics,
                "anticipation_depth": len(anticipated_knowledge)
            },
            context_window_id=f"proactive_{request.request_id}",
            performance_metrics={
                "retrieval_count": len(retrieved_knowledge),
                "anticipated_items": len(anticipated_knowledge),
                "predicted_topics": len(predicted_topics),
                "session_history_depth": len(session_history)
            },
            recommendations=self._generate_proactive_recommendations(
                predicted_topics, request.learning_goals
            ),
            confidence_score=avg_confidence,
            processing_time=processing_time
        )

    def _collaborative_workflow(self, request: 'OrchestrationRequest') -> 'OrchestrationResponse':
        """
        Collaborative workflow for interactive, user-guided exploration.

        Works with user input to:
        - Suggest refinements and clarifications
        - Offer multiple perspectives
        - Enable iterative exploration
        - Build shared understanding
        """
        import time
        start_time = time.time()

        # Analyze query for ambiguity and refinement opportunities
        query_analysis = self._analyze_query_for_collaboration(request.query)

        # Retrieve initial results
        knowledge_items = self.knowledge_base.query_knowledge(
            request.query, max_results=10
        )

        # Group results by perspective/approach
        perspectives = self._group_by_perspective(knowledge_items)

        retrieved_knowledge = []
        for perspective, items in perspectives.items():
            for item in items[:3]:  # Top 3 per perspective
                retrieved_knowledge.append({
                    "content": str(item.content),
                    "confidence": item.confidence,
                    "perspective": perspective,
                    "collaborative_type": "multi-perspective"
                })

        # Generate clarifying questions
        clarifying_questions = self._generate_clarifying_questions(
            request.query, query_analysis, perspectives
        )

        # Generate collaborative response
        generated_content = self._generate_collaborative_response(
            request.query,
            retrieved_knowledge,
            perspectives,
            clarifying_questions
        )

        # Calculate collaboration metrics
        avg_confidence = (
            statistics.mean([k["confidence"] for k in retrieved_knowledge])
            if retrieved_knowledge else 0.0
        )

        processing_time = time.time() - start_time

        return OrchestrationResponse(
            request_id=request.request_id,
            generated_content=generated_content,
            retrieved_knowledge=retrieved_knowledge,
            learning_pathway_updates={
                "perspectives_offered": list(perspectives.keys()),
                "clarifying_questions": clarifying_questions,
                "collaboration_state": "awaiting_input"
            },
            context_window_id=f"collaborative_{request.request_id}",
            performance_metrics={
                "retrieval_count": len(retrieved_knowledge),
                "perspectives_count": len(perspectives),
                "clarifying_questions": len(clarifying_questions),
                "ambiguity_score": query_analysis.get("ambiguity_score", 0)
            },
            recommendations=clarifying_questions[:3] if clarifying_questions else [
                "Explore a specific aspect",
                "Provide more context"
            ],
            confidence_score=avg_confidence,
            processing_time=processing_time
        )


    def initialize(self):
        """Initialize the orchestrator with all dependencies."""
        if self.initialized:
            return

        logger.info("Initializing Adaptive RAG Orchestrator...")

        try:
            # Set up orchestration workflows
            self._setup_orchestration_workflows()

            # Initialize RAG engine if it has an initialize method
            if hasattr(self.rag_engine, 'initialize') and not getattr(self.rag_engine, 'is_initialized', False):
                self.rag_engine.initialize()

            # Initialize context manager if it has an initialize method
            if hasattr(self.context_manager, 'initialize') and not getattr(self.context_manager, 'is_initialized', False):
                self.context_manager.initialize()

            # Initialize pattern engine if it has an initialize method
            if hasattr(self.pattern_engine, 'initialize') and not getattr(self.pattern_engine, 'is_initialized', False):
                self.pattern_engine.initialize()

            # Initialize adaptive pathways if it has an initialize method
            if hasattr(self.adaptive_pathways, 'initialize') and not getattr(self.adaptive_pathways, 'is_initialized', False):
                self.adaptive_pathways.initialize()

            # Initialize KAG if enabled
            if self.enable_kag:
                self._initialize_kag()

            # Initialize Knowledge-Enhanced Pathways if enabled
            if self.enable_kep:
                self._initialize_kep()

            self.initialized = True
            logger.info("Adaptive RAG Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Adaptive RAG Orchestrator: {e}")
            self.initialized = False
            raise

    # ==================== KAG-Enhanced Orchestration Methods ====================

    async def orchestrate_with_kag(self, request: OrchestrationRequest) -> OrchestrationResponse:
        """
        Orchestrate a complete learning session with full KAG integration.

        This method provides the most comprehensive knowledge augmentation:
        1. Query grounding in verified knowledge
        2. Context augmentation with entities/facts/relations
        3. Response verification against knowledge base
        4. Domain coherence optimization
        5. Knowledge gap detection

        Args:
            request: OrchestrationRequest with query and context

        Returns:
            OrchestrationResponse with KAG enhancements
        """
        import time
        start_time = time.time()

        # Initialize if needed
        if not self.initialized:
            self.initialize()

        # Check if KAG is available
        if not self.kag:
            logger.warning("KAG not available, falling back to standard orchestration")
            return await self.orchestrate_learning_session(request)

        session_id = f"session_{request.request_id}"
        session = await self._get_or_create_session(session_id, request)

        try:
            # Step 1: Ground query in knowledge
            kag_grounding = None
            if self.kag_config.get("ground_queries", True):
                kag_grounding = await self._kag_ground_query(request)

            # Step 2: Determine optimal workflow based on grounded query
            strategy = await self._determine_orchestration_strategy(request, session)

            # Step 3: Retrieve knowledge with KAG-guided relevance
            rag_results = await self._kag_enhanced_retrieval(request, strategy, kag_grounding)

            # Step 4: Augment context with knowledge
            augmented_context = None
            if self.kag_config.get("augment_context", True):
                augmented_context = await self._kag_augment_context(request, rag_results, kag_grounding)

            # Step 5: Apply pattern recognition
            pattern_insights = await self._apply_pattern_recognition(request, rag_results, session)

            # Step 6: Manage context window
            context_window_id = await self._manage_context_window(request, rag_results, session)

            # Step 7: Generate learning pathway updates
            learning_updates = await self._generate_learning_pathway_updates(
                request, rag_results, pattern_insights, session
            )

            # Step 8: Synthesize response
            generated_content = await self._kag_synthesize_response(
                request, rag_results, pattern_insights, learning_updates, augmented_context
            )

            # Step 9: Verify response against knowledge
            kag_verification = None
            knowledge_gaps = []
            if self.kag_config.get("verify_responses", True):
                kag_verification = await self._kag_verify_response(
                    generated_content, request.query, augmented_context
                )
                if kag_verification:
                    knowledge_gaps = kag_verification.get("knowledge_gaps", [])

            # Step 10: Check domain coherence
            kag_coherence = None
            if self.kag_config.get("verify_responses", True):
                kag_coherence = await self._kag_check_coherence(
                    request.query, generated_content, request.context.get("domain")
                )

            # Step 11: Detect and optionally fill knowledge gaps
            if self.kag_config.get("detect_knowledge_gaps", True):
                detected_gaps = await self._kag_detect_gaps(request.query, generated_content)
                knowledge_gaps.extend(detected_gaps)

                if self.kag_config.get("auto_fill_gaps", False) and detected_gaps:
                    await self._kag_fill_gaps(detected_gaps)

            # Calculate metrics
            processing_time = time.time() - start_time
            confidence_score = self._calculate_kag_confidence(
                rag_results, pattern_insights, kag_grounding, kag_verification, kag_coherence
            )

            # Track KAG-specific metrics
            if kag_coherence:
                self.performance_analytics["kag_coherence_scores"].append(
                    kag_coherence.get("overall_coherence", 0.0)
                )
            if kag_verification:
                self.performance_analytics["kag_verification_scores"].append(
                    kag_verification.get("confidence_score", 0.0)
                )

            # Generate recommendations (including KAG-specific ones)
            recommendations = await self._generate_kag_recommendations(
                request, rag_results, learning_updates, kag_grounding, kag_coherence, knowledge_gaps
            )

            # Create response
            response = OrchestrationResponse(
                request_id=request.request_id,
                generated_content=generated_content,
                retrieved_knowledge=rag_results,
                learning_pathway_updates=learning_updates,
                context_window_id=context_window_id,
                performance_metrics={
                    "processing_time": processing_time,
                    "confidence_score": confidence_score,
                    "retrieval_count": len(rag_results),
                    "pattern_matches": len(pattern_insights),
                    "kag_enabled": True,
                    "kag_grounding_confidence": kag_grounding.get("confidence", 0.0) if kag_grounding else 0.0,
                    "kag_coherence_score": kag_coherence.get("overall_coherence", 0.0) if kag_coherence else 0.0,
                    "knowledge_gaps_detected": len(knowledge_gaps)
                },
                recommendations=recommendations,
                confidence_score=confidence_score,
                processing_time=processing_time,
                kag_grounding=kag_grounding,
                kag_coherence=kag_coherence,
                kag_verification=kag_verification,
                knowledge_gaps=knowledge_gaps if knowledge_gaps else None
            )

            # Update session
            await self._update_session(session_id, request, response)

            return response

        except Exception as e:
            logger.error(f"Error in KAG-enhanced orchestration: {e}")
            return self._create_error_response(request, str(e))

    async def _kag_ground_query(self, request: OrchestrationRequest) -> Optional[Dict[str, Any]]:
        """Ground query in verified knowledge using KAG."""
        if not self.kag:
            return None

        try:
            domain = request.context.get("domain")
            grounding = await self.kag.augment_query(
                request.query,
                context=str(request.context),
                domain=domain
            )

            return {
                "entities": grounding.grounded_entities,
                "facts": grounding.grounded_facts,
                "relations": grounding.grounded_relations,
                "domain_context": grounding.domain_context,
                "confidence": grounding.confidence_score
            }
        except Exception as e:
            logger.warning(f"KAG query grounding failed: {e}")
            return None

    async def _kag_enhanced_retrieval(
        self,
        request: OrchestrationRequest,
        strategy: Dict[str, Any],
        kag_grounding: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform KAG-enhanced knowledge retrieval."""
        # Start with standard retrieval
        base_results = await self._orchestrated_knowledge_retrieval(request, strategy)

        if not kag_grounding or not self.kag:
            return base_results

        # Enhance results with KAG grounding information
        enhanced_results = []
        grounded_entities = {e.get("name", "").lower() for e in kag_grounding.get("entities", [])}
        grounded_facts = {f.get("content", "").lower()[:50] for f in kag_grounding.get("facts", [])}

        for result in base_results:
            content = result.get("content", "").lower()

            # Calculate KAG relevance boost
            entity_boost = sum(0.05 for entity in grounded_entities if entity in content)
            fact_overlap = sum(0.1 for fact in grounded_facts if fact[:20] in content)

            enhanced_result = {
                **result,
                "kag_entity_match": entity_boost > 0,
                "kag_fact_alignment": fact_overlap > 0,
                "kag_relevance_boost": min(entity_boost + fact_overlap, 0.3)
            }

            # Adjust confidence with KAG boost
            base_confidence = result.get("confidence", 0.5)
            enhanced_result["confidence"] = min(
                base_confidence + enhanced_result["kag_relevance_boost"],
                1.0
            )

            enhanced_results.append(enhanced_result)

        # Re-sort by enhanced confidence
        enhanced_results.sort(key=lambda x: x["confidence"], reverse=True)

        return enhanced_results

    async def _kag_augment_context(
        self,
        request: OrchestrationRequest,
        rag_results: List[Dict[str, Any]],
        kag_grounding: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Augment retrieval context with KAG knowledge."""
        if not self.kag:
            return None

        try:
            # Combine RAG results into context string
            context_text = "\n".join(
                result.get("content", "")[:500]
                for result in rag_results[:5]
            )

            augmented = await self.kag.augment_context(
                context_text,
                grounding=None,  # Already have grounding
                query=request.query
            )

            return {
                "original_context": context_text,
                "augmented_context": augmented.augmented_context,
                "knowledge_additions": augmented.knowledge_additions,
                "entity_annotations": augmented.entity_annotations,
                "fact_references": augmented.fact_references,
                "coherence_score": augmented.coherence_score
            }
        except Exception as e:
            logger.warning(f"KAG context augmentation failed: {e}")
            return None

    async def _kag_synthesize_response(
        self,
        request: OrchestrationRequest,
        rag_results: List[Dict[str, Any]],
        pattern_insights: List[Dict[str, Any]],
        learning_updates: Dict[str, Any],
        augmented_context: Optional[Dict[str, Any]]
    ) -> str:
        """Synthesize response with KAG-augmented context."""
        response_parts = []

        # Use augmented context if available
        if augmented_context and augmented_context.get("augmented_context"):
            # Extract knowledge additions
            for addition in augmented_context.get("knowledge_additions", [])[:3]:
                content = addition.get("content", "")
                if content:
                    response_parts.append(content)

        # Add RAG results
        if rag_results:
            top_result = rag_results[0].get("content", "")
            if top_result:
                response_parts.append(f"Based on retrieved knowledge: {top_result}")

        # Add pattern insights
        if pattern_insights:
            response_parts.append("Pattern-based insights applied.")

        # Add learning context
        if learning_updates:
            response_parts.append("Learning pathway updated.")

        if not response_parts:
            return f"Response for: {request.query}"

        return "\n\n".join(response_parts)

    async def _kag_verify_response(
        self,
        response: str,
        query: str,
        augmented_context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Verify response against knowledge using KAG."""
        if not self.kag:
            return None

        try:
            context = augmented_context.get("original_context") if augmented_context else None
            verified = await self.kag.verify_response(response, query, context)

            return {
                "original_response": verified.original_response,
                "verified_response": verified.verified_response,
                "verification_results": verified.verification_results,
                "corrections_made": verified.corrections_made,
                "confidence_score": verified.confidence_score,
                "coherence_level": verified.coherence_level.value,
                "knowledge_gaps": verified.knowledge_gaps
            }
        except Exception as e:
            logger.warning(f"KAG response verification failed: {e}")
            return None

    async def _kag_check_coherence(
        self,
        query: str,
        response: str,
        domain: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Check domain coherence using KAG."""
        if not self.kag:
            return None

        try:
            report = await self.kag.ensure_coherence(query, response, domain)

            return {
                "overall_coherence": report.overall_coherence,
                "coherence_level": report.coherence_level.value,
                "domain_alignment": report.domain_alignment,
                "entity_consistency": report.entity_consistency,
                "fact_accuracy": report.fact_accuracy,
                "relation_validity": report.relation_validity,
                "temporal_coherence": report.temporal_coherence,
                "recommendations": report.recommendations
            }
        except Exception as e:
            logger.warning(f"KAG coherence check failed: {e}")
            return None

    async def _kag_detect_gaps(self, query: str, response: str) -> List[str]:
        """Detect knowledge gaps using KAG."""
        if not self.kag:
            return []

        try:
            gaps = await self.kag.detect_knowledge_gaps(query, response)
            return [gap.description for gap in gaps]
        except Exception as e:
            logger.warning(f"KAG gap detection failed: {e}")
            return []

    async def _kag_fill_gaps(self, gaps: List[str]) -> int:
        """Attempt to fill detected knowledge gaps."""
        if not self.kag:
            return 0

        filled_count = 0
        for gap_desc in gaps:
            try:
                # Get the actual gap object from KAG
                kag_gaps = self.kag.get_knowledge_gaps(filled=False)
                for gap in kag_gaps:
                    if gap.description == gap_desc and gap.auto_fillable:
                        if await self.kag.fill_knowledge_gap(gap):
                            filled_count += 1
            except Exception as e:
                logger.debug(f"Failed to fill gap: {e}")

        return filled_count

    def _calculate_kag_confidence(
        self,
        rag_results: List[Dict],
        pattern_insights: List[Dict],
        kag_grounding: Optional[Dict],
        kag_verification: Optional[Dict],
        kag_coherence: Optional[Dict]
    ) -> float:
        """Calculate overall confidence with KAG factors."""
        # Base confidence from RAG and patterns
        base_confidence = self._calculate_confidence_score(rag_results, pattern_insights)

        # KAG boosts
        kag_boost = 0.0

        if kag_grounding:
            grounding_conf = kag_grounding.get("confidence", 0)
            kag_boost += grounding_conf * 0.1

        if kag_verification:
            verification_conf = kag_verification.get("confidence_score", 0)
            kag_boost += verification_conf * 0.15

        if kag_coherence:
            coherence_score = kag_coherence.get("overall_coherence", 0)
            kag_boost += coherence_score * 0.15

        return min(base_confidence + kag_boost, 1.0)

    async def _generate_kag_recommendations(
        self,
        request: OrchestrationRequest,
        rag_results: List[Dict],
        learning_updates: Dict,
        kag_grounding: Optional[Dict],
        kag_coherence: Optional[Dict],
        knowledge_gaps: List[str]
    ) -> List[str]:
        """Generate recommendations including KAG-specific insights."""
        recommendations = await self._generate_recommendations(
            request, rag_results, learning_updates,
            self.active_sessions.get(f"session_{request.request_id}", {})
        )

        # Add KAG-specific recommendations
        if kag_coherence:
            coherence_recs = kag_coherence.get("recommendations", [])
            recommendations.extend(coherence_recs[:2])

        if knowledge_gaps:
            recommendations.append(f"Knowledge gaps detected: {len(knowledge_gaps)} areas need more information")

        if kag_grounding and kag_grounding.get("entities"):
            entity_count = len(kag_grounding["entities"])
            recommendations.append(f"Query grounded in {entity_count} known entities")

        return recommendations[:5]

    def get_kag_statistics(self) -> Dict[str, Any]:
        """Get KAG-specific statistics."""
        stats = {
            "kag_enabled": self.enable_kag,
            "kag_initialized": self._kag_initialized,
            "kag_config": self.kag_config,
        }

        if self.kag:
            kag_stats = self.kag.get_statistics()
            stats.update({
                "kag_metrics": kag_stats.get("metrics", {}),
                "kag_domains": kag_stats.get("domains_registered", []),
                "kag_knowledge_sources": kag_stats.get("knowledge_sources", {})
            })

        # Add performance metrics
        coherence_scores = self.performance_analytics.get("kag_coherence_scores", [])
        verification_scores = self.performance_analytics.get("kag_verification_scores", [])

        if coherence_scores:
            stats["avg_coherence_score"] = statistics.mean(coherence_scores)
        if verification_scores:
            stats["avg_verification_score"] = statistics.mean(verification_scores)

        return stats

    async def orchestrate_learning_session(self, request: OrchestrationRequest) -> OrchestrationResponse:
        """Orchestrate a complete learning session with RAG, patterns, and adaptation."""
        start_time = datetime.now()
        session_id = f"session_{request.request_id}"

        try:
            # Create or retrieve user session
            session = await self._get_or_create_session(session_id, request)

            # Analyze request and determine optimal strategy
            strategy = await self._determine_orchestration_strategy(request, session)

            # Retrieve knowledge using RAG engine
            rag_results = await self._orchestrated_knowledge_retrieval(request, strategy)

            # Apply pattern recognition for enhanced understanding
            pattern_insights = await self._apply_pattern_recognition(request, rag_results, session)

            # Create or update context window
            context_window_id = await self._manage_context_window(request, rag_results, session)

            # Generate adaptive learning pathway updates
            learning_updates = await self._generate_learning_pathway_updates(
                request, rag_results, pattern_insights, session
            )

            # Synthesize final response
            generated_content = await self._synthesize_response(
                request, rag_results, pattern_insights, learning_updates
            )

            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = self._calculate_confidence_score(rag_results, pattern_insights)

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                request, rag_results, learning_updates, session
            )

            # Create response
            response = OrchestrationResponse(
                request_id=request.request_id,
                generated_content=generated_content,
                retrieved_knowledge=rag_results,
                learning_pathway_updates=learning_updates,
                context_window_id=context_window_id,
                performance_metrics={
                    "processing_time": processing_time,
                    "confidence_score": confidence_score,
                    "retrieval_count": len(rag_results),
                    "pattern_matches": len(pattern_insights)
                },
                recommendations=recommendations,
                confidence_score=confidence_score,
                processing_time=processing_time
            )

            # Update session and learning history
            await self._update_session(session_id, request, response)

            # Trigger adaptive learning if threshold reached
            await self._trigger_adaptive_learning_if_needed(session_id)

            return response

        except Exception as e:
            logger.error(f"Error in orchestration: {e}")
            return self._create_error_response(request, str(e))

    async def optimize_learning_pathway(self, user_id: str, learning_goals: List[str]) -> Dict[str, Any]:
        """Optimize learning pathways using all available components."""
        optimization_results = {
            "user_id": user_id,
            "optimized_pathways": {},
            "recommended_actions": [],
            "estimated_improvements": {},
            "implementation_plan": {}
        }

        # Get user's learning history and current state
        learning_history = self.learning_histories.get(user_id, [])
        current_session = self.active_sessions.get(f"session_{user_id}")

        # Create user profile from history and session data
        user_profile = self._build_comprehensive_user_profile(user_id, learning_history, current_session)

        # Use adaptive pathways to generate base pathways
        if hasattr(self.adaptive_pathways, 'generate_learning_pathway'):
            for goal in learning_goals:
                base_pathway = self.adaptive_pathways.generate_learning_pathway(
                    user_id, goal, duration_weeks=8
                )

                # Enhance with RAG vectorization
                vector_enhancements = await self._enhance_pathway_with_vectors(base_pathway, user_profile)

                # Apply pattern recognition for learning optimization
                pattern_optimizations = await self._optimize_pathway_with_patterns(
                    base_pathway, vector_enhancements, user_profile
                )

                # Create optimized context windows for learning content
                context_optimization = await self._optimize_learning_context_windows(
                    base_pathway, pattern_optimizations, user_profile
                )

                optimization_results["optimized_pathways"][goal] = {
                    "base_pathway": base_pathway,
                    "vector_enhancements": vector_enhancements,
                    "pattern_optimizations": pattern_optimizations,
                    "context_optimization": context_optimization,
                    "estimated_completion_time": self._estimate_completion_time(
                        base_pathway, user_profile
                    ),
                    "difficulty_progression": self._calculate_optimal_difficulty_progression(
                        base_pathway, user_profile
                    )
                }

        # Generate implementation plan
        optimization_results["implementation_plan"] = await self._create_implementation_plan(
            optimization_results["optimized_pathways"], user_profile
        )

        return optimization_results

    async def perform_adaptive_optimization(self) -> Dict[str, Any]:
        """Perform system-wide adaptive optimization."""
        optimization_results = {
            "rag_engine_optimizations": {},
            "context_window_optimizations": {},
            "pattern_recognition_improvements": {},
            "learning_pathway_adjustments": {},
            "overall_performance_gain": 0.0
        }

        # Optimize RAG engine
        if hasattr(self.rag_engine, 'optimize_context_windows'):
            rag_optimization = self.rag_engine.optimize_context_windows()
            optimization_results["rag_engine_optimizations"] = rag_optimization

        # Optimize context windows
        if hasattr(self.context_manager, 'optimize_all_windows'):
            context_optimization = self.context_manager.optimize_all_windows()
            optimization_results["context_window_optimizations"] = context_optimization

        # Update pattern recognition based on recent interactions
        pattern_improvements = await self._optimize_pattern_recognition()
        optimization_results["pattern_recognition_improvements"] = pattern_improvements

        # Adjust learning pathways based on performance data
        pathway_adjustments = await self._adjust_learning_pathways()
        optimization_results["learning_pathway_adjustments"] = pathway_adjustments

        # Calculate overall performance gain
        optimization_results["overall_performance_gain"] = self._calculate_overall_performance_gain(
            optimization_results
        )

        logger.info(f"Adaptive optimization completed with {optimization_results['overall_performance_gain']:.2%} improvement")
        return optimization_results

    def get_orchestration_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about orchestration performance."""
        analytics = {
            "session_analytics": {
                "active_sessions": len(self.active_sessions),
                "total_users": len(self.learning_histories),
                "avg_session_duration": self._calculate_avg_session_duration(),
                "session_success_rate": self._calculate_session_success_rate()
            },
            "performance_metrics": {
                "avg_response_time": statistics.mean(self.performance_analytics["response_time"])
                                   if self.performance_analytics["response_time"] else 0,
                "avg_user_satisfaction": statistics.mean(self.performance_analytics["user_satisfaction"])
                                      if self.performance_analytics["user_satisfaction"] else 0,
                "knowledge_accuracy": statistics.mean(self.performance_analytics["knowledge_accuracy"])
                                   if self.performance_analytics["knowledge_accuracy"] else 0,
                "learning_effectiveness": statistics.mean(self.performance_analytics["learning_effectiveness"])
                                       if self.performance_analytics["learning_effectiveness"] else 0
            },
            "component_utilization": {
                "rag_engine_calls": getattr(self.rag_engine, 'call_count', 0),
                "pattern_recognitions": len(getattr(self.pattern_engine, 'patterns', {})),
                "context_windows_created": len(getattr(self.context_manager, 'active_windows', {})),
                "learning_pathways_active": len(self.active_sessions)
            },
            "adaptive_learning_stats": {
                "adaptation_events": sum(len(history) for history in self.learning_histories.values()),
                "successful_adaptations": self._count_successful_adaptations(),
                "learning_improvements": self._calculate_learning_improvements()
            }
        }

        return analytics

    # Helper methods (implementations would be more detailed in practice)
    async def _get_or_create_session(self, session_id: str, request: OrchestrationRequest) -> Dict[str, Any]:
        """Get existing session or create new one."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": datetime.now(),
                "user_profile": request.user_profile,
                "learning_goals": request.learning_goals,
                "interaction_count": 0,
                "context_history": [],
                "adaptation_state": {}
            }
        return self.active_sessions[session_id]

    async def _orchestrated_knowledge_retrieval(self, request: OrchestrationRequest,
                                              strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform orchestrated knowledge retrieval using RAG engine."""
        if hasattr(self.rag_engine, 'retrieve_augmented_knowledge'):
            rag_result = self.rag_engine.retrieve_augmented_knowledge(
                query=request.query,
                context_length=min(request.max_context_length, 100000),  # Reasonable limit for processing
                max_results=strategy.get("max_results", 10)
            )
            return rag_result.get("augmented_results", [])

        # Fallback to knowledge base
        knowledge_items = self.knowledge_base.query_knowledge(request.query, max_results=10)
        return [{"content": str(item.content), "confidence": item.confidence} for item in knowledge_items]

    def _calculate_confidence_score(self, rag_results: List[Dict], pattern_insights: List[Dict]) -> float:
        """Calculate overall confidence score for the response."""
        if not rag_results and not pattern_insights:
            return 0.0

        rag_confidence = statistics.mean([result.get("confidence", 0.5) for result in rag_results]) if rag_results else 0.5
        pattern_confidence = 0.8 if pattern_insights else 0.5  # Pattern insights boost confidence

        return (rag_confidence + pattern_confidence) / 2

    def _create_error_response(self, request: OrchestrationRequest, error_message: str) -> OrchestrationResponse:
        """Create error response."""
        return OrchestrationResponse(
            request_id=request.request_id,
            generated_content=f"I apologize, but I encountered an error processing your request: {error_message}",
            retrieved_knowledge=[],
            learning_pathway_updates={},
            context_window_id="",
            performance_metrics={"error": True},
            recommendations=["Please try rephrasing your question", "Check if all required information is provided"],
            confidence_score=0.0,
            processing_time=0.0
        )

    # Placeholder methods for completeness (actual implementation would be complex)
    async def _determine_orchestration_strategy(self, request: OrchestrationRequest, session: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the best orchestration strategy based on request and session."""
        # In a real scenario, this would involve analyzing the query, user profile, and session history.
        # For this example, we'll just return a default strategy.
        return {"max_results": 10, "mode": request.preferred_mode}

    async def _apply_pattern_recognition(self, request: OrchestrationRequest, rag_results: List[Dict[str, Any]], session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply pattern recognition to RAG results."""
        # Placeholder: In a real system, this would involve the pattern engine.
        return []

    async def _manage_context_window(self, request: OrchestrationRequest, rag_results: List[Dict[str, Any]], session: Dict[str, Any]) -> str:
        """Manage and update the context window."""
        # Placeholder: In a real system, this would involve the context manager.
        return f"context_{request.request_id}"

    async def _generate_learning_pathway_updates(self, request: OrchestrationRequest, rag_results: List[Dict[str, Any]], pattern_insights: List[Dict[str, Any]], session: Dict[str, Any]) -> Dict[str, Any]:
        """Generate updates for the learning pathway."""
        # Placeholder: In a real system, this would involve the adaptive pathways component.
        return {}

    async def _synthesize_response(self, request: OrchestrationRequest, rag_results: List[Dict[str, Any]], pattern_insights: List[Dict[str, Any]], learning_updates: Dict[str, Any]) -> str:
        """Synthesize the final response content."""
        # Placeholder: Combine information from RAG, patterns, and learning updates.
        response_parts = [request.query]
        if rag_results:
            response_parts.append("Retrieved: " + rag_results[0].get("content", "No content"))
        if learning_updates:
            response_parts.append("Learning updates applied.")
        return " ".join(response_parts)

    async def _generate_recommendations(self, request: OrchestrationRequest, rag_results: List[Dict[str, Any]], learning_updates: Dict[str, Any], session: Dict[str, Any]) -> List[str]:
        """Generate relevant recommendations."""
        # Placeholder: Based on the interaction, suggest next steps.
        return ["Explore related topics", "Review recent learning"]

    async def _update_session(self, session_id: str, request: OrchestrationRequest, response: OrchestrationResponse):
        """Update the active session with new information."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["interaction_count"] += 1
            self.active_sessions[session_id]["context_history"].append({
                "query": request.query,
                "response": response.generated_content
            })
            # Update performance analytics
            self.performance_analytics["response_time"].append(response.processing_time)
            # Add placeholders for other metrics that would be captured from user feedback or system monitoring
            self.performance_analytics["user_satisfaction"].append(0.8) # Example satisfaction score
            self.performance_analytics["knowledge_accuracy"].append(0.7) # Example accuracy score
            self.performance_analytics["learning_effectiveness"].append(0.6) # Example effectiveness score

    async def _trigger_adaptive_learning_if_needed(self, session_id: str):
        """Trigger adaptive learning adjustments if thresholds are met."""
        if session_id in self.active_sessions:
            if self.active_sessions[session_id]["interaction_count"] >= self.learning_update_threshold:
                logger.info(f"Threshold reached for session {session_id}. Triggering adaptive learning.")
                # Placeholder for actual adaptive learning trigger logic
                await self._perform_adaptive_optimization() # Example: Trigger system-wide optimization

    def _build_comprehensive_user_profile(self, user_id: str, learning_history: List[Dict[str, Any]], current_session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Build a comprehensive user profile from history and current session."""
        # Placeholder: Combine information from learning history and current session data.
        profile = {"user_id": user_id}
        if current_session and current_session.get("user_profile"):
            profile.update(current_session["user_profile"])
        if learning_history:
            profile["learning_history_summary"] = f"Completed {len(learning_history)} interactions."
        return profile

    async def _enhance_pathway_with_vectors(self, base_pathway: Any, user_profile: Dict[str, Any]) -> Any:
        """Enhance learning pathway with vector embeddings and RAG."""
        # Placeholder: Use RAG engine to find relevant vector information.
        return base_pathway

    async def _optimize_pathway_with_patterns(self, base_pathway: Any, vector_enhancements: Any, user_profile: Dict[str, Any]) -> Any:
        """Optimize learning pathway using pattern recognition."""
        # Placeholder: Use pattern engine to refine the pathway.
        return vector_enhancements

    async def _optimize_learning_context_windows(self, base_pathway: Any, pattern_optimizations: Any, user_profile: Dict[str, Any]) -> Any:
        """Optimize context windows for learning content."""
        # Placeholder: Use context manager to create tailored learning contexts.
        return pattern_optimizations

    def _estimate_completion_time(self, base_pathway: Any, user_profile: Dict[str, Any]) -> str:
        """Estimate the time to complete the learning pathway."""
        # Placeholder: Calculate based on pathway complexity and user profile.
        return "8 weeks"

    def _calculate_optimal_difficulty_progression(self, base_pathway: Any, user_profile: Dict[str, Any]) -> List[str]:
        """Calculate the optimal difficulty progression for the pathway."""
        # Placeholder: Adjust difficulty based on user's past performance.
        return ["Beginner", "Intermediate", "Advanced"]

    async def _create_implementation_plan(self, optimized_pathways: Dict[str, Any], user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Create an implementation plan for the optimized pathways."""
        # Placeholder: Outline steps to enact the learning pathway changes.
        return {"steps": ["Review content", "Schedule practice sessions"]}

    async def _optimize_pattern_recognition(self) -> Dict[str, Any]:
        """Update pattern recognition models based on recent interactions."""
        # Placeholder: Retrain or fine-tune the pattern engine.
        return {"update_status": "completed"}

    async def _adjust_learning_pathways(self) -> Dict[str, Any]:
        """Adjust active learning pathways based on performance data."""
        # Placeholder: Modify existing pathways based on user progress and feedback.
        return {"adjustment_status": "applied"}

    def _calculate_overall_performance_gain(self, optimization_results: Dict[str, Any]) -> float:
        """Calculate the overall performance gain from optimizations."""
        # Placeholder: Aggregate improvements from different components.
        return 0.15 # Example 15% gain

    def _calculate_avg_session_duration(self) -> float:
        """Calculate the average duration of active sessions."""
        # Placeholder: Calculate from session start and end times.
        return 600.0 # Example 10 minutes

    def _calculate_session_success_rate(self) -> float:
        """Calculate the rate of successful sessions."""
        # Placeholder: Define success criteria and calculate the rate.
        return 0.9 # Example 90% success rate

    def _count_successful_adaptations(self) -> int:
        """Count the number of successful adaptive learning events."""
        # Placeholder: Track successful learning adjustments.
        return 50

    def _calculate_learning_improvements(self) -> float:
        """Calculate the average improvement attributed to learning adaptations."""
        # Placeholder: Quantify the impact of learning adjustments.
        return 0.08 # Example 8% improvement

    # ==================== Focused Workflow Helpers ====================

    def _analyze_query_focus(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine primary domain and focus areas.

        Returns dict with primary_domain, secondary_domains, specificity_score.
        """
        query_lower = query.lower()

        # Domain detection heuristics
        domain_keywords = {
            "technical": ["code", "programming", "api", "function", "algorithm", "software", "debug"],
            "conceptual": ["concept", "theory", "principle", "understand", "explain", "what is"],
            "procedural": ["how to", "steps", "process", "procedure", "implement", "create"],
            "analytical": ["analyze", "compare", "evaluate", "assess", "difference", "versus"],
            "troubleshooting": ["error", "issue", "problem", "fix", "not working", "failed"]
        }

        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                domain_scores[domain] = score

        # Determine primary domain
        if domain_scores:
            primary_domain = max(domain_scores, key=domain_scores.get)
            secondary_domains = [d for d in domain_scores if d != primary_domain]
        else:
            primary_domain = "general"
            secondary_domains = []

        # Calculate specificity (more specific terms = higher specificity)
        words = query.split()
        specific_word_count = sum(1 for w in words if len(w) > 6)
        specificity_score = min(specific_word_count / max(len(words), 1), 1.0)

        return {
            "primary_domain": primary_domain,
            "secondary_domains": secondary_domains,
            "specificity_score": specificity_score,
            "domain_scores": domain_scores
        }

    def _calculate_item_relevance(self, item: Any, query: str, target_domain: str) -> float:
        """
        Calculate relevance score of an item to the query and target domain.

        Uses word overlap and domain alignment.
        """
        item_content = str(item.content).lower() if hasattr(item, 'content') else str(item).lower()
        query_lower = query.lower()

        # Word overlap score
        query_words = set(query_lower.split())
        content_words = set(item_content.split())

        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with'}
        query_words -= stop_words
        content_words -= stop_words

        if not query_words:
            return 0.3  # Base relevance

        overlap = len(query_words & content_words)
        overlap_score = overlap / len(query_words)

        # Domain alignment bonus
        domain_keywords = {
            "technical": ["code", "function", "class", "method", "variable", "api"],
            "conceptual": ["concept", "theory", "principle", "definition"],
            "procedural": ["step", "process", "procedure", "instruction"],
            "analytical": ["analysis", "comparison", "evaluation", "assessment"],
            "troubleshooting": ["error", "fix", "solution", "debug", "issue"]
        }

        domain_bonus = 0.0
        if target_domain in domain_keywords:
            for keyword in domain_keywords[target_domain]:
                if keyword in item_content:
                    domain_bonus += 0.05

        domain_bonus = min(domain_bonus, 0.2)

        # Combine scores
        relevance = overlap_score * 0.8 + domain_bonus + 0.1  # Base floor
        return min(relevance, 1.0)

    def _generate_focused_recommendations(self, query: str, domain: str,
                                         results: List[Dict]) -> List[str]:
        """Generate recommendations for focused workflow follow-up."""
        recommendations = []

        if not results:
            recommendations.append(f"Try broadening your {domain} query")
            recommendations.append("Consider exploratory mode for discovery")
        else:
            recommendations.append(f"Dive deeper into {domain} specifics")
            if len(results) >= 3:
                recommendations.append("Compare top results for nuances")

        recommendations.append("Refine query for more precision")
        return recommendations[:3]

    # ==================== Exploratory Workflow Helpers ====================

    def _expand_query_for_exploration(self, query: str) -> List[str]:
        """
        Expand query into related variants for broad exploration.

        Returns list of query variations.
        """
        expanded = [query]  # Original query first

        words = query.split()

        # Synonym-based expansion (simple heuristics)
        synonym_map = {
            "how": ["what way", "method"],
            "create": ["build", "make", "develop"],
            "use": ["utilize", "apply", "employ"],
            "find": ["locate", "discover", "identify"],
            "understand": ["comprehend", "grasp", "learn"],
            "improve": ["enhance", "optimize", "better"],
            "problem": ["issue", "challenge", "difficulty"]
        }

        for word in words:
            word_lower = word.lower()
            if word_lower in synonym_map:
                for synonym in synonym_map[word_lower][:1]:  # Limit expansions
                    new_query = query.replace(word, synonym)
                    if new_query not in expanded:
                        expanded.append(new_query)

        # Related concept expansion
        # Add "related to [query]" variant
        expanded.append(f"related to {query}")

        # Add "examples of [query]" variant
        expanded.append(f"examples of {query}")

        return expanded[:5]  # Limit to 5 variants

    def _identify_item_domain(self, item: Any) -> str:
        """Identify the domain of a knowledge item."""
        content = str(item.content).lower() if hasattr(item, 'content') else str(item).lower()

        domain_indicators = {
            "technical": ["code", "api", "function", "class", "programming"],
            "conceptual": ["concept", "theory", "principle", "model"],
            "practical": ["example", "case", "scenario", "application"],
            "reference": ["documentation", "specification", "standard"],
            "tutorial": ["guide", "tutorial", "how-to", "step"]
        }

        for domain, indicators in domain_indicators.items():
            if any(ind in content for ind in indicators):
                return domain

        return "general"

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate or near-duplicate results based on content similarity."""
        if not results:
            return []

        unique = []
        seen_content = set()

        for result in results:
            # Create content signature (first 100 chars, normalized)
            content = result.get("content", "")
            signature = content[:100].lower().strip()

            if signature not in seen_content:
                seen_content.add(signature)
                unique.append(result)

        return unique

    def _generate_exploration_summary(self, query: str, results: List[Dict],
                                      domains: List[str]) -> str:
        """Generate summary of exploration results."""
        if not results:
            return f"Exploration for '{query}' found no results."

        summary_parts = [f"Exploration of '{query}' discovered {len(results)} relevant items"]

        if domains:
            summary_parts.append(f"across {len(domains)} domains: {', '.join(domains[:3])}")

        # Add top content preview
        if results:
            top_content = results[0].get("content", "")[:150]
            summary_parts.append(f". Key finding: {top_content}")

        return " ".join(summary_parts)

    def _generate_exploration_recommendations(self, domains: List[str],
                                             results: List[Dict]) -> List[str]:
        """Generate recommendations for continued exploration."""
        recommendations = []

        for domain in domains[:2]:
            recommendations.append(f"Deep dive into {domain} domain")

        if len(results) > 5:
            recommendations.append("Focus on specific aspect for deeper analysis")

        recommendations.append("Use synthesis mode to combine findings")
        return recommendations[:4]

    # ==================== Synthesis Workflow Helpers ====================

    def _cross_reference_sources(self, retrieval_sources: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Cross-reference results from multiple sources.

        Identifies items that appear in multiple sources and tracks agreement.
        """
        all_items = []
        content_to_sources = {}

        for source_name, items in retrieval_sources.items():
            for item in items:
                content = item.get("content", "")[:100].lower().strip()

                if content in content_to_sources:
                    content_to_sources[content]["sources"].append(source_name)
                    content_to_sources[content]["confidences"].append(item.get("confidence", 0.5))
                else:
                    content_to_sources[content] = {
                        "full_content": item.get("content", ""),
                        "sources": [source_name],
                        "confidences": [item.get("confidence", 0.5)]
                    }

        # Build cross-referenced items
        for content_sig, data in content_to_sources.items():
            source_count = len(data["sources"])
            avg_confidence = statistics.mean(data["confidences"])

            all_items.append({
                "content": data["full_content"],
                "confidence": avg_confidence,
                "sources": data["sources"],
                "source_count": source_count,
                "agreement_score": min(source_count / 3, 1.0)  # Normalize to 0-1
            })

        return all_items

    def _analyze_source_agreement(self, cross_referenced: List[Dict]) -> Dict[str, Any]:
        """Analyze agreement and contradictions among sources."""
        analysis = {
            "total_items": len(cross_referenced),
            "multi_source_items": 0,
            "single_source_items": 0,
            "avg_agreement": 0.0,
            "contradictions": 0,
            "source_coverage": {}
        }

        if not cross_referenced:
            return analysis

        agreement_scores = []
        source_counts = {}

        for item in cross_referenced:
            source_count = item.get("source_count", 1)
            if source_count > 1:
                analysis["multi_source_items"] += 1
            else:
                analysis["single_source_items"] += 1

            agreement_scores.append(item.get("agreement_score", 0.5))

            for source in item.get("sources", []):
                source_counts[source] = source_counts.get(source, 0) + 1

        analysis["avg_agreement"] = statistics.mean(agreement_scores)
        analysis["source_coverage"] = source_counts

        # Estimate contradictions (items with low agreement but from multiple sources)
        # This is a simplified heuristic
        analysis["contradictions"] = sum(
            1 for item in cross_referenced
            if item.get("source_count", 1) > 1 and item.get("agreement_score", 1) < 0.5
        )

        return analysis

    def _generate_synthesis_response(self, query: str, knowledge: List[Dict],
                                     analysis: Dict) -> str:
        """Generate synthesized response combining multiple sources."""
        if not knowledge:
            return f"Synthesis for '{query}' found no combinable sources."

        response_parts = [f"Synthesized analysis of '{query}':"]

        # Add top synthesized content
        for item in knowledge[:3]:
            source_info = f"[{item.get('source_count', 1)} source(s)]"
            content_preview = item.get("content", "")[:100]
            response_parts.append(f"{source_info} {content_preview}")

        # Add synthesis metadata
        if analysis.get("multi_source_items", 0) > 0:
            response_parts.append(
                f"\nCross-validated across {analysis['multi_source_items']} multi-source items."
            )

        if analysis.get("contradictions", 0) > 0:
            response_parts.append(
                f"Note: {analysis['contradictions']} potential contradictions identified."
            )

        return " ".join(response_parts)

    def _generate_synthesis_recommendations(self, analysis: Dict,
                                           knowledge: List[Dict]) -> List[str]:
        """Generate recommendations based on synthesis analysis."""
        recommendations = []

        if analysis.get("contradictions", 0) > 0:
            recommendations.append("Review contradicting sources for accuracy")

        if analysis.get("single_source_items", 0) > analysis.get("multi_source_items", 0):
            recommendations.append("Seek additional sources for validation")

        if analysis.get("avg_agreement", 0) > 0.7:
            recommendations.append("High agreement - results are well-validated")

        recommendations.append("Use focused mode for specific follow-up")
        return recommendations[:3]

    # ==================== Reactive Workflow Helpers ====================

    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for query."""
        import hashlib
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _check_response_cache(self, cache_key: str) -> Optional[Dict]:
        """Check if response is cached."""
        # Use active_sessions as a simple cache store
        cache_store = self.active_sessions.get("_response_cache", {})
        cached = cache_store.get(cache_key)

        if cached:
            # Check cache freshness (5 minute TTL)
            cache_time = cached.get("_cache_time")
            if cache_time:
                age = (datetime.now() - cache_time).total_seconds()
                if age < 300:  # 5 minutes
                    return cached.get("response")

        return None

    def _cache_response(self, cache_key: str, response_data: Dict) -> None:
        """Cache response for future reactive calls."""
        if "_response_cache" not in self.active_sessions:
            self.active_sessions["_response_cache"] = {}

        self.active_sessions["_response_cache"][cache_key] = {
            "response": response_data,
            "_cache_time": datetime.now()
        }

        # Limit cache size
        cache = self.active_sessions["_response_cache"]
        if len(cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                cache.keys(),
                key=lambda k: cache[k].get("_cache_time", datetime.min)
            )[:20]
            for key in oldest_keys:
                del cache[key]

    # ==================== Proactive Workflow Helpers ====================

    def _predict_next_topics(self, current_query: str, session_history: List[Dict],
                            learning_goals: List[str]) -> List[str]:
        """
        Predict likely next topics based on context.

        Uses session history patterns and learning goals.
        """
        predictions = []

        # Extract topics from current query
        query_words = [w for w in current_query.split() if len(w) > 4]

        # Pattern: "how to X" often followed by "examples of X" or "troubleshooting X"
        if "how to" in current_query.lower():
            subject = current_query.lower().replace("how to", "").strip()
            predictions.append(f"examples of {subject}")
            predictions.append(f"common issues with {subject}")

        # Pattern: Questions often followed by related concepts
        if current_query.endswith("?"):
            predictions.append(f"related concepts to {' '.join(query_words[:3])}")

        # Use learning goals
        for goal in learning_goals[:2]:
            if goal.lower() not in current_query.lower():
                predictions.append(f"connecting {current_query} to {goal}")

        # Use session history for pattern detection
        if len(session_history) >= 2:
            recent_topics = [
                h.get("query", "").split()[:3]
                for h in session_history[-3:]
            ]
            # Flatten and find common theme
            recent_words = [w for words in recent_topics for w in words if len(w) > 4]
            if recent_words:
                common = max(set(recent_words), key=recent_words.count)
                predictions.append(f"more about {common}")

        return predictions[:5]

    def _generate_proactive_response(self, query: str, knowledge: List[Dict],
                                     predicted_topics: List[str]) -> str:
        """Generate proactive response with anticipatory content."""
        response_parts = []

        # Current query response
        current_items = [k for k in knowledge if k.get("retrieval_type") == "current"]
        if current_items:
            response_parts.append(f"For your query: {current_items[0].get('content', '')[:150]}")

        # Anticipated content
        anticipated_items = [k for k in knowledge if k.get("retrieval_type") == "proactive"]
        if anticipated_items:
            response_parts.append("\n\nAnticipated follow-up information:")
            for item in anticipated_items[:2]:
                topic = item.get("anticipated_topic", "related topic")
                content = item.get("content", "")[:100]
                response_parts.append(f"- {topic}: {content}")

        # Predicted topics preview
        if predicted_topics:
            response_parts.append(f"\n\nYou might also want to explore: {', '.join(predicted_topics[:3])}")

        return " ".join(response_parts)

    def _generate_proactive_recommendations(self, predicted_topics: List[str],
                                           learning_goals: List[str]) -> List[str]:
        """Generate proactive recommendations."""
        recommendations = []

        for topic in predicted_topics[:2]:
            recommendations.append(f"Explore: {topic}")

        for goal in learning_goals[:1]:
            recommendations.append(f"Progress toward: {goal}")

        recommendations.append("Switch to focused mode for depth")
        return recommendations[:4]

    # ==================== Collaborative Workflow Helpers ====================

    def _analyze_query_for_collaboration(self, query: str) -> Dict[str, Any]:
        """Analyze query for collaborative refinement opportunities."""
        analysis = {
            "ambiguity_score": 0.0,
            "refinement_opportunities": [],
            "clarification_needed": []
        }

        query_lower = query.lower()

        # Detect ambiguous terms
        ambiguous_indicators = ["thing", "stuff", "something", "it", "this", "that"]
        ambiguity_count = sum(1 for ind in ambiguous_indicators if ind in query_lower.split())
        analysis["ambiguity_score"] = min(ambiguity_count / 3, 1.0)

        # Detect broad terms needing refinement
        broad_terms = ["best", "good", "better", "improve", "help", "make"]
        for term in broad_terms:
            if term in query_lower:
                analysis["refinement_opportunities"].append(f"Clarify what '{term}' means in this context")

        # Detect missing context
        if len(query.split()) < 5:
            analysis["clarification_needed"].append("Could you provide more context?")

        # Detect questions that could have multiple answers
        if "or" in query_lower:
            analysis["clarification_needed"].append("Which option would you prefer?")

        return analysis

    def _group_by_perspective(self, knowledge_items: List[Any]) -> Dict[str, List[Any]]:
        """Group knowledge items by perspective or approach."""
        perspectives = {
            "practical": [],
            "theoretical": [],
            "example-based": [],
            "general": []
        }

        for item in knowledge_items:
            content = str(item.content).lower() if hasattr(item, 'content') else str(item).lower()

            if any(kw in content for kw in ["example", "case", "scenario", "instance"]):
                perspectives["example-based"].append(item)
            elif any(kw in content for kw in ["how to", "step", "guide", "tutorial"]):
                perspectives["practical"].append(item)
            elif any(kw in content for kw in ["concept", "theory", "principle", "definition"]):
                perspectives["theoretical"].append(item)
            else:
                perspectives["general"].append(item)

        # Remove empty perspectives
        return {k: v for k, v in perspectives.items() if v}

    def _generate_clarifying_questions(self, query: str, analysis: Dict,
                                      perspectives: Dict) -> List[str]:
        """Generate clarifying questions for collaborative refinement."""
        questions = []

        # From analysis
        for clarification in analysis.get("clarification_needed", []):
            questions.append(clarification)

        for refinement in analysis.get("refinement_opportunities", []):
            questions.append(refinement)

        # From perspectives
        if len(perspectives) > 1:
            perspective_names = list(perspectives.keys())
            questions.append(
                f"Would you prefer a {perspective_names[0]} or {perspective_names[1]} approach?"
            )

        # Generic collaborative questions
        if not questions:
            questions.append("What aspect interests you most?")
            questions.append("Do you need examples or theory?")

        return questions[:4]

    def _generate_collaborative_response(self, query: str, knowledge: List[Dict],
                                        perspectives: Dict,
                                        questions: List[str]) -> str:
        """Generate collaborative response with multiple perspectives."""
        response_parts = [f"Collaborative analysis of '{query}':"]

        # Present perspectives
        for perspective, items in perspectives.items():
            if items:
                response_parts.append(f"\n{perspective.title()} perspective:")
                # Get first item from this perspective in retrieved knowledge
                for k in knowledge:
                    if k.get("perspective") == perspective:
                        response_parts.append(f"  - {k.get('content', '')[:100]}")
                        break

        # Present clarifying questions
        if questions:
            response_parts.append("\n\nTo better assist you, please consider:")
            for q in questions[:2]:
                response_parts.append(f"  - {q}")

        return " ".join(response_parts)

    async def _perform_adaptive_optimization(self) -> Dict[str, Any]:
        """Internal method for adaptive optimization triggered by thresholds."""
        return await self.perform_adaptive_optimization()

    # ==================== KAG Workflow Enhancement Helpers ====================

    def _enhance_workflow_results_with_kag(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Enhance workflow results with KAG grounding (synchronous helper).

        Can be called from any workflow to add KAG relevance scoring.
        """
        if not self.kag or not self.enable_kag:
            return results

        try:
            import asyncio

            # Run async grounding in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, return unenhanced
                return results

            grounding = loop.run_until_complete(
                self.kag.augment_query(query)
            )

            if not grounding:
                return results

            # Extract grounded entity names
            entity_names = {
                e.get("name", "").lower()
                for e in grounding.grounded_entities
            }

            # Enhance results
            enhanced = []
            for result in results:
                content = result.get("content", "").lower()

                # Calculate entity overlap
                entity_matches = sum(1 for name in entity_names if name in content)
                kag_boost = min(entity_matches * 0.05, 0.2)

                enhanced_result = {
                    **result,
                    "kag_entity_matches": entity_matches,
                    "kag_boost": kag_boost
                }

                # Adjust confidence if present
                if "confidence" in enhanced_result:
                    enhanced_result["confidence"] = min(
                        enhanced_result["confidence"] + kag_boost,
                        1.0
                    )

                enhanced.append(enhanced_result)

            return enhanced

        except Exception as e:
            logger.debug(f"KAG enhancement skipped: {e}")
            return results

    async def apply_kag_to_workflow_response(
        self,
        request: OrchestrationRequest,
        base_response: OrchestrationResponse
    ) -> OrchestrationResponse:
        """
        Apply KAG verification and coherence checking to a workflow response.

        Can be called after any workflow to add KAG enhancements.
        """
        if not self.kag or not self.enable_kag:
            return base_response

        try:
            # Verify response
            kag_verification = None
            if self.kag_config.get("verify_responses", True):
                kag_verification = await self._kag_verify_response(
                    base_response.generated_content,
                    request.query,
                    None
                )

            # Check coherence
            kag_coherence = None
            if self.kag_config.get("verify_responses", True):
                kag_coherence = await self._kag_check_coherence(
                    request.query,
                    base_response.generated_content,
                    request.context.get("domain")
                )

            # Detect gaps
            knowledge_gaps = []
            if self.kag_config.get("detect_knowledge_gaps", True):
                knowledge_gaps = await self._kag_detect_gaps(
                    request.query,
                    base_response.generated_content
                )

            # Update response with KAG data
            enhanced_metrics = {
                **base_response.performance_metrics,
                "kag_enhanced": True,
                "kag_coherence_score": kag_coherence.get("overall_coherence", 0.0) if kag_coherence else 0.0,
                "kag_verification_score": kag_verification.get("confidence_score", 0.0) if kag_verification else 0.0,
                "knowledge_gaps_detected": len(knowledge_gaps)
            }

            # Add KAG recommendations
            enhanced_recommendations = list(base_response.recommendations)
            if kag_coherence and kag_coherence.get("recommendations"):
                enhanced_recommendations.extend(kag_coherence["recommendations"][:2])

            return OrchestrationResponse(
                request_id=base_response.request_id,
                generated_content=base_response.generated_content,
                retrieved_knowledge=base_response.retrieved_knowledge,
                learning_pathway_updates=base_response.learning_pathway_updates,
                context_window_id=base_response.context_window_id,
                performance_metrics=enhanced_metrics,
                recommendations=enhanced_recommendations[:5],
                confidence_score=base_response.confidence_score,
                processing_time=base_response.processing_time,
                kag_grounding=None,
                kag_coherence=kag_coherence,
                kag_verification=kag_verification,
                knowledge_gaps=knowledge_gaps if knowledge_gaps else None
            )

        except Exception as e:
            logger.warning(f"Failed to apply KAG to workflow response: {e}")
            return base_response

    def register_kag_domain(self, domain: str, ontology: Dict[str, Any]):
        """
        Register a domain ontology with the KAG engine.

        Args:
            domain: Domain name (e.g., 'finance', 'technology')
            ontology: Ontology definition with concepts, relations, rules
        """
        if self.kag:
            self.kag.register_domain_ontology(domain, ontology)
            logger.info(f"Registered domain ontology '{domain}' with KAG")
        else:
            logger.warning("KAG not available, cannot register domain ontology")