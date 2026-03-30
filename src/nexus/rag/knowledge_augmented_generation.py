"""
Knowledge-Augmented Generation (KAG) System for Nexus AI Platform.

Integrates knowledge bases, knowledge graphs, and truth verification to boost
domain-knowledge coherence in RAG responses. KAG goes beyond traditional RAG by:

1. Pre-retrieval knowledge grounding - anchoring queries in factual knowledge
2. Cross-referencing with knowledge graphs for entity relationships
3. Real-time fact verification during generation
4. Knowledge gap detection and dynamic filling
5. Domain coherence scoring and optimization
6. Contradiction prevention through truth verification

Architecture:
    Query -> Knowledge Grounding -> Enriched Retrieval -> Verified Generation -> Coherent Response
"""

import logging
import time
import asyncio
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeAugmentationMode(Enum):
    """Modes for knowledge augmentation."""
    LIGHT = "light"              # Basic fact checking only
    STANDARD = "standard"        # Fact checking + entity linking
    COMPREHENSIVE = "comprehensive"  # Full knowledge integration
    DOMAIN_EXPERT = "domain_expert"  # Deep domain-specific augmentation
    VERIFICATION_STRICT = "verification_strict"  # Strict truth verification


class KnowledgeCoherenceLevel(Enum):
    """Coherence quality levels."""
    EXCELLENT = "excellent"    # 90%+ coherence
    GOOD = "good"              # 75-90% coherence
    ACCEPTABLE = "acceptable"  # 60-75% coherence
    NEEDS_IMPROVEMENT = "needs_improvement"  # 40-60% coherence
    POOR = "poor"              # Below 40% coherence


class KnowledgeSourceType(Enum):
    """Types of knowledge sources."""
    KNOWLEDGE_BASE = "knowledge_base"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    FACTUAL_MEMORY = "factual_memory"
    TRUTH_VERIFIER = "truth_verifier"
    DOMAIN_ONTOLOGY = "domain_ontology"
    EXTERNAL_REFERENCE = "external_reference"


@dataclass
class KnowledgeGrounding:
    """Represents grounded knowledge for a query."""
    query: str
    grounded_entities: List[Dict[str, Any]] = field(default_factory=list)
    grounded_facts: List[Dict[str, Any]] = field(default_factory=list)
    grounded_relations: List[Dict[str, Any]] = field(default_factory=list)
    domain_context: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AugmentedContext:
    """Context augmented with knowledge."""
    original_context: str
    augmented_context: str
    knowledge_additions: List[Dict[str, Any]] = field(default_factory=list)
    entity_annotations: List[Dict[str, Any]] = field(default_factory=list)
    fact_references: List[Dict[str, Any]] = field(default_factory=list)
    coherence_score: float = 0.0
    augmentation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifiedResponse:
    """A response verified against knowledge base."""
    original_response: str
    verified_response: str
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    corrections_made: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    coherence_level: KnowledgeCoherenceLevel = KnowledgeCoherenceLevel.ACCEPTABLE
    knowledge_gaps: List[str] = field(default_factory=list)


@dataclass
class KnowledgeGap:
    """Represents a detected gap in knowledge."""
    gap_id: str
    topic: str
    description: str
    detected_from: str
    severity: str  # critical, moderate, minor
    suggested_sources: List[str] = field(default_factory=list)
    auto_fillable: bool = False
    filled: bool = False
    fill_content: Optional[str] = None


@dataclass
class DomainCoherenceReport:
    """Report on domain coherence analysis."""
    query: str
    response: str
    overall_coherence: float
    coherence_level: KnowledgeCoherenceLevel
    domain_alignment: Dict[str, float] = field(default_factory=dict)
    entity_consistency: float = 0.0
    fact_accuracy: float = 0.0
    relation_validity: float = 0.0
    temporal_coherence: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class KAGConfig:
    """Configuration for Knowledge-Augmented Generation."""
    augmentation_mode: KnowledgeAugmentationMode = KnowledgeAugmentationMode.STANDARD
    min_coherence_threshold: float = 0.6
    max_knowledge_additions: int = 10
    enable_entity_linking: bool = True
    enable_fact_verification: bool = True
    enable_relation_inference: bool = True
    enable_gap_detection: bool = True
    enable_contradiction_prevention: bool = True
    auto_fill_gaps: bool = False
    strict_verification: bool = False
    domain_specific_rules: Dict[str, Any] = field(default_factory=dict)
    cache_grounding_results: bool = True
    grounding_cache_ttl: int = 300  # 5 minutes
    max_verification_depth: int = 3
    parallel_verification: bool = True


class KnowledgeAugmentedGeneration:
    """
    Main KAG system that integrates multiple knowledge sources to enhance
    RAG retrieval and generation with domain-knowledge coherence.

    This system provides:
    1. Knowledge Grounding - Anchoring queries in verified facts
    2. Entity Linking - Connecting mentions to knowledge graph entities
    3. Fact Verification - Validating retrieved and generated content
    4. Gap Detection - Identifying missing knowledge
    5. Coherence Optimization - Ensuring domain consistency
    """

    def __init__(
        self,
        knowledge_base=None,
        knowledge_graph=None,
        factual_memory=None,
        truth_verifier=None,
        config: Optional[KAGConfig] = None
    ):
        """
        Initialize KAG system with knowledge sources.

        Args:
            knowledge_base: KnowledgeBase instance for factual knowledge
            knowledge_graph: KnowledgeGraph instance for entity/relations
            factual_memory: FactualMemoryEngine for persistent facts
            truth_verifier: TruthVerifier for claim verification
            config: KAG configuration
        """
        self.knowledge_base = knowledge_base
        self.knowledge_graph = knowledge_graph
        self.factual_memory = factual_memory
        self.truth_verifier = truth_verifier
        self.config = config or KAGConfig()

        # Grounding cache
        self._grounding_cache: Dict[str, Tuple[KnowledgeGrounding, float]] = {}

        # Domain knowledge registries
        self._domain_ontologies: Dict[str, Dict[str, Any]] = {}
        self._domain_rules: Dict[str, List[Dict[str, Any]]] = {}
        self._entity_aliases: Dict[str, Set[str]] = defaultdict(set)

        # Performance tracking
        self._metrics = {
            'total_augmentations': 0,
            'successful_groundings': 0,
            'verifications_performed': 0,
            'contradictions_prevented': 0,
            'gaps_detected': 0,
            'gaps_filled': 0,
            'avg_coherence_score': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Knowledge gap tracker
        self._knowledge_gaps: Dict[str, KnowledgeGap] = {}

        # Entity extraction patterns for linking
        self._entity_patterns = {
            'technology': [
                r'\b(Python|JavaScript|TypeScript|Rust|Go|Java|C\+\+|Ruby|Kotlin|Swift)\b',
                r'\b(React|Vue|Angular|FastAPI|Django|Flask|Express|Spring)\b',
                r'\b(PostgreSQL|MySQL|MongoDB|Redis|SQLite|Elasticsearch)\b',
                r'\b(Docker|Kubernetes|AWS|GCP|Azure|Terraform)\b',
                r'\b(TensorFlow|PyTorch|scikit-learn|Keras|Pandas|NumPy)\b',
            ],
            'organization': [
                r'\b(Google|Microsoft|Amazon|Apple|Meta|OpenAI|Anthropic)\b',
                r'\b(Netflix|Uber|Airbnb|Spotify|Twitter|LinkedIn)\b',
            ],
            'concept': [
                r'\b(machine learning|deep learning|neural network|NLP|computer vision)\b',
                r'\b(microservices|serverless|containerization|CI/CD|DevOps)\b',
                r'\b(API|REST|GraphQL|gRPC|WebSocket)\b',
            ],
        }

        self.initialized = False
        logger.info("KAG system created with config: %s", self.config.augmentation_mode.value)

    def initialize(self):
        """Initialize the KAG system."""
        if self.initialized:
            return

        logger.info("Initializing Knowledge-Augmented Generation system...")

        # Initialize knowledge base if available
        if self.knowledge_base and hasattr(self.knowledge_base, 'initialize'):
            self.knowledge_base.initialize()

        # Load domain ontologies
        self._load_default_ontologies()

        # Build entity alias mappings
        self._build_entity_aliases()

        self.initialized = True
        logger.info("KAG system initialized successfully")

    # =========================================================================
    # Core KAG Pipeline Methods
    # =========================================================================

    async def augment_query(
        self,
        query: str,
        context: Optional[str] = None,
        domain: Optional[str] = None
    ) -> KnowledgeGrounding:
        """
        Ground a query in verified knowledge before retrieval.

        This is the first stage of KAG - anchoring the query in factual knowledge
        to improve retrieval relevance and accuracy.

        Args:
            query: The user query
            context: Optional conversation context
            domain: Optional domain hint

        Returns:
            KnowledgeGrounding with entities, facts, and relations
        """
        if not self.initialized:
            self.initialize()

        # Check cache first
        cache_key = self._compute_cache_key(query, domain)
        if self.config.cache_grounding_results and cache_key in self._grounding_cache:
            cached, timestamp = self._grounding_cache[cache_key]
            if time.time() - timestamp < self.config.grounding_cache_ttl:
                self._metrics['cache_hits'] += 1
                return cached

        self._metrics['cache_misses'] += 1

        grounding = KnowledgeGrounding(query=query)

        # Extract and link entities
        if self.config.enable_entity_linking:
            grounding.grounded_entities = await self._extract_and_link_entities(query, domain)

        # Find relevant facts
        grounding.grounded_facts = await self._find_relevant_facts(query, domain)

        # Discover relations
        if self.config.enable_relation_inference:
            grounding.grounded_relations = await self._infer_relations(
                grounding.grounded_entities,
                grounding.grounded_facts
            )

        # Build domain context
        grounding.domain_context = self._build_domain_context(
            query, grounding.grounded_entities, domain
        )

        # Calculate confidence
        grounding.confidence_score = self._calculate_grounding_confidence(grounding)

        self._metrics['successful_groundings'] += 1

        # Cache result
        if self.config.cache_grounding_results:
            self._grounding_cache[cache_key] = (grounding, time.time())

        return grounding

    async def augment_context(
        self,
        context: str,
        grounding: Optional[KnowledgeGrounding] = None,
        query: Optional[str] = None
    ) -> AugmentedContext:
        """
        Augment retrieved context with knowledge from multiple sources.

        This enriches the RAG context with:
        - Entity definitions and properties
        - Related facts and relationships
        - Domain-specific information

        Args:
            context: The retrieved context
            grounding: Pre-computed knowledge grounding
            query: Original query (used if grounding not provided)

        Returns:
            AugmentedContext with enriched information
        """
        if not self.initialized:
            self.initialize()

        # Get grounding if not provided
        if grounding is None and query:
            grounding = await self.augment_query(query)

        augmented = AugmentedContext(
            original_context=context,
            augmented_context=context
        )

        additions = []

        # Add entity information
        if grounding and grounding.grounded_entities:
            entity_info = self._generate_entity_information(grounding.grounded_entities)
            if entity_info:
                additions.append({
                    'type': 'entity_info',
                    'content': entity_info,
                    'source': KnowledgeSourceType.KNOWLEDGE_GRAPH.value
                })
                augmented.entity_annotations = grounding.grounded_entities

        # Add relevant facts
        if grounding and grounding.grounded_facts:
            fact_info = self._generate_fact_information(grounding.grounded_facts)
            if fact_info:
                additions.append({
                    'type': 'factual_context',
                    'content': fact_info,
                    'source': KnowledgeSourceType.KNOWLEDGE_BASE.value
                })
                augmented.fact_references = grounding.grounded_facts

        # Add relationship context
        if grounding and grounding.grounded_relations:
            relation_info = self._generate_relation_information(grounding.grounded_relations)
            if relation_info:
                additions.append({
                    'type': 'relationship_context',
                    'content': relation_info,
                    'source': KnowledgeSourceType.KNOWLEDGE_GRAPH.value
                })

        # Add domain-specific context
        if grounding and grounding.domain_context:
            domain_info = self._generate_domain_context(grounding.domain_context)
            if domain_info:
                additions.append({
                    'type': 'domain_context',
                    'content': domain_info,
                    'source': KnowledgeSourceType.DOMAIN_ONTOLOGY.value
                })

        # Apply additions up to limit
        additions = additions[:self.config.max_knowledge_additions]
        augmented.knowledge_additions = additions

        # Build augmented context
        augmented.augmented_context = self._build_augmented_context(context, additions)

        # Calculate coherence
        augmented.coherence_score = self._calculate_context_coherence(augmented)

        augmented.augmentation_metadata = {
            'grounding_confidence': grounding.confidence_score if grounding else 0.0,
            'additions_count': len(additions),
            'augmentation_mode': self.config.augmentation_mode.value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

        self._metrics['total_augmentations'] += 1

        return augmented

    async def verify_response(
        self,
        response: str,
        query: str,
        context: Optional[str] = None
    ) -> VerifiedResponse:
        """
        Verify a generated response against knowledge sources.

        This performs:
        - Claim extraction and verification
        - Contradiction detection
        - Coherence scoring
        - Knowledge gap identification

        Args:
            response: The generated response to verify
            query: The original query
            context: Optional context used in generation

        Returns:
            VerifiedResponse with verification results
        """
        if not self.initialized:
            self.initialize()

        verified = VerifiedResponse(
            original_response=response,
            verified_response=response
        )

        # Extract claims from response
        claims = self._extract_claims(response)

        verification_results = []
        corrections = []
        knowledge_gaps = []

        for claim in claims:
            # Verify each claim
            result = await self._verify_claim(claim, query)
            verification_results.append(result)

            if result.get('status') == 'contradicted':
                # Need correction
                correction = await self._generate_correction(claim, result)
                if correction:
                    corrections.append(correction)
                    self._metrics['contradictions_prevented'] += 1
            elif result.get('status') == 'unverified':
                # Knowledge gap
                gap = self._create_knowledge_gap(claim, query)
                knowledge_gaps.append(gap.description)
                self._knowledge_gaps[gap.gap_id] = gap
                self._metrics['gaps_detected'] += 1

        verified.verification_results = verification_results
        verified.corrections_made = corrections
        verified.knowledge_gaps = knowledge_gaps

        # Apply corrections to response
        if corrections:
            verified.verified_response = self._apply_corrections(response, corrections)

        # Calculate overall confidence
        verified.confidence_score = self._calculate_verification_confidence(verification_results)

        # Determine coherence level
        verified.coherence_level = self._determine_coherence_level(verified.confidence_score)

        self._metrics['verifications_performed'] += 1

        return verified

    async def ensure_coherence(
        self,
        query: str,
        response: str,
        domain: Optional[str] = None
    ) -> DomainCoherenceReport:
        """
        Ensure domain coherence between query and response.

        Analyzes multiple dimensions of coherence:
        - Domain alignment
        - Entity consistency
        - Fact accuracy
        - Relation validity
        - Temporal coherence

        Args:
            query: The original query
            response: The generated response
            domain: Optional domain context

        Returns:
            DomainCoherenceReport with detailed analysis
        """
        if not self.initialized:
            self.initialize()

        report = DomainCoherenceReport(
            query=query,
            response=response,
            overall_coherence=0.0,
            coherence_level=KnowledgeCoherenceLevel.ACCEPTABLE
        )

        # Analyze domain alignment
        report.domain_alignment = await self._analyze_domain_alignment(query, response, domain)

        # Check entity consistency
        report.entity_consistency = await self._check_entity_consistency(query, response)

        # Verify fact accuracy
        report.fact_accuracy = await self._verify_fact_accuracy(response)

        # Validate relations
        report.relation_validity = await self._validate_relations(response)

        # Check temporal coherence
        report.temporal_coherence = self._check_temporal_coherence(response)

        # Calculate overall coherence
        weights = {
            'domain_alignment': 0.25,
            'entity_consistency': 0.2,
            'fact_accuracy': 0.3,
            'relation_validity': 0.15,
            'temporal_coherence': 0.1
        }

        domain_score = sum(report.domain_alignment.values()) / max(len(report.domain_alignment), 1)

        report.overall_coherence = (
            weights['domain_alignment'] * domain_score +
            weights['entity_consistency'] * report.entity_consistency +
            weights['fact_accuracy'] * report.fact_accuracy +
            weights['relation_validity'] * report.relation_validity +
            weights['temporal_coherence'] * report.temporal_coherence
        )

        report.coherence_level = self._determine_coherence_level(report.overall_coherence)

        # Generate recommendations
        report.recommendations = self._generate_coherence_recommendations(report)

        # Update metrics
        self._update_coherence_metrics(report.overall_coherence)

        return report

    # =========================================================================
    # Knowledge Gap Management
    # =========================================================================

    async def detect_knowledge_gaps(
        self,
        query: str,
        response: str,
        context: Optional[str] = None
    ) -> List[KnowledgeGap]:
        """
        Detect gaps in knowledge base that affect response quality.

        Args:
            query: The original query
            response: The generated response
            context: Optional context

        Returns:
            List of detected knowledge gaps
        """
        if not self.config.enable_gap_detection:
            return []

        gaps = []

        # Extract topics and entities from query
        topics = self._extract_topics(query)
        entities = await self._extract_and_link_entities(query)

        # Check coverage for each topic
        for topic in topics:
            coverage = await self._check_topic_coverage(topic)
            if coverage < 0.5:
                gap = KnowledgeGap(
                    gap_id=f"gap_{hashlib.md5(topic.encode()).hexdigest()[:12]}",
                    topic=topic,
                    description=f"Insufficient knowledge about: {topic}",
                    detected_from=query,
                    severity='moderate' if coverage > 0.2 else 'critical',
                    suggested_sources=self._suggest_knowledge_sources(topic),
                    auto_fillable=self._is_auto_fillable(topic)
                )
                gaps.append(gap)
                self._knowledge_gaps[gap.gap_id] = gap

        # Check entity coverage
        for entity in entities:
            entity_name = entity.get('name', '')
            if not await self._has_entity_knowledge(entity_name):
                gap = KnowledgeGap(
                    gap_id=f"entity_gap_{hashlib.md5(entity_name.encode()).hexdigest()[:12]}",
                    topic=entity_name,
                    description=f"Missing entity information: {entity_name}",
                    detected_from=query,
                    severity='minor',
                    suggested_sources=['knowledge_graph', 'external_reference'],
                    auto_fillable=True
                )
                gaps.append(gap)
                self._knowledge_gaps[gap.gap_id] = gap

        self._metrics['gaps_detected'] += len(gaps)
        return gaps

    async def fill_knowledge_gap(
        self,
        gap: KnowledgeGap,
        content: Optional[str] = None,
        source: Optional[str] = None
    ) -> bool:
        """
        Fill a detected knowledge gap.

        Args:
            gap: The knowledge gap to fill
            content: Content to fill the gap (optional, will auto-generate if possible)
            source: Source of the fill content

        Returns:
            True if gap was successfully filled
        """
        if gap.filled:
            return True

        if content is None and gap.auto_fillable:
            # Try to auto-fill
            content = await self._auto_fill_gap(gap)

        if content is None:
            return False

        # Add to knowledge base
        if self.knowledge_base:
            try:
                from nexus.memory.knowledge_base import KnowledgeType
                self.knowledge_base.add_knowledge(
                    content=content,
                    knowledge_type=KnowledgeType.FACTUAL,
                    source=source or 'gap_filling',
                    confidence=0.7,
                    context_tags=[gap.topic]
                )
                gap.filled = True
                gap.fill_content = content
                self._metrics['gaps_filled'] += 1
                return True
            except Exception as e:
                logger.warning(f"Failed to fill knowledge gap: {e}")
                return False

        return False

    def get_knowledge_gaps(
        self,
        severity: Optional[str] = None,
        filled: Optional[bool] = None
    ) -> List[KnowledgeGap]:
        """Get tracked knowledge gaps with optional filtering."""
        gaps = list(self._knowledge_gaps.values())

        if severity:
            gaps = [g for g in gaps if g.severity == severity]

        if filled is not None:
            gaps = [g for g in gaps if g.filled == filled]

        return sorted(gaps, key=lambda g: {'critical': 0, 'moderate': 1, 'minor': 2}.get(g.severity, 3))

    # =========================================================================
    # Domain Knowledge Management
    # =========================================================================

    def register_domain_ontology(
        self,
        domain: str,
        ontology: Dict[str, Any]
    ):
        """
        Register a domain-specific ontology for enhanced coherence.

        Args:
            domain: Domain name (e.g., 'software_engineering', 'finance')
            ontology: Ontology definition with concepts, relations, rules
        """
        self._domain_ontologies[domain] = {
            'concepts': ontology.get('concepts', {}),
            'relations': ontology.get('relations', []),
            'rules': ontology.get('rules', []),
            'synonyms': ontology.get('synonyms', {}),
            'hierarchies': ontology.get('hierarchies', {}),
            'constraints': ontology.get('constraints', []),
            'registered_at': datetime.now(timezone.utc).isoformat()
        }

        # Update entity aliases from synonyms
        for canonical, aliases in ontology.get('synonyms', {}).items():
            self._entity_aliases[canonical.lower()].update(
                alias.lower() for alias in aliases
            )

        logger.info(f"Registered domain ontology: {domain}")

    def add_domain_rule(
        self,
        domain: str,
        rule: Dict[str, Any]
    ):
        """
        Add a domain-specific rule for coherence checking.

        Args:
            domain: Domain name
            rule: Rule definition with condition and action
        """
        if domain not in self._domain_rules:
            self._domain_rules[domain] = []

        self._domain_rules[domain].append({
            'rule_id': f"rule_{len(self._domain_rules[domain])}",
            'condition': rule.get('condition', ''),
            'action': rule.get('action', ''),
            'severity': rule.get('severity', 'warning'),
            'message': rule.get('message', ''),
            'created_at': datetime.now(timezone.utc).isoformat()
        })

    # =========================================================================
    # Internal Helper Methods
    # =========================================================================

    async def _extract_and_link_entities(
        self,
        text: str,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract entities from text and link to knowledge graph."""
        import re

        entities = []
        seen_names = set()

        # Extract using patterns
        for entity_type, patterns in self._entity_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    name = match.group(0)
                    if name.lower() not in seen_names:
                        seen_names.add(name.lower())

                        entity = {
                            'name': name,
                            'type': entity_type,
                            'source': 'pattern_extraction',
                            'confidence': 0.9
                        }

                        # Try to link to knowledge graph
                        if self.knowledge_graph:
                            try:
                                kg_entity = await self.knowledge_graph.find_entity(name)
                                if kg_entity:
                                    entity['kg_id'] = kg_entity.id
                                    entity['kg_properties'] = kg_entity.properties
                                    entity['confidence'] = 1.0
                            except Exception:
                                pass

                        entities.append(entity)

        # Check domain ontology for additional entities
        if domain and domain in self._domain_ontologies:
            ontology = self._domain_ontologies[domain]
            for concept in ontology.get('concepts', {}).keys():
                if concept.lower() in text.lower() and concept.lower() not in seen_names:
                    seen_names.add(concept.lower())
                    entities.append({
                        'name': concept,
                        'type': 'concept',
                        'source': 'domain_ontology',
                        'confidence': 0.85
                    })

        return entities

    async def _find_relevant_facts(
        self,
        query: str,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find facts relevant to the query."""
        facts = []

        # Query knowledge base
        if self.knowledge_base:
            try:
                kb_results = self.knowledge_base.query_knowledge(
                    query,
                    max_results=10,
                    min_confidence=0.5
                )
                for item in kb_results:
                    facts.append({
                        'id': item.id,
                        'content': str(item.content),
                        'type': item.knowledge_type.value if hasattr(item.knowledge_type, 'value') else str(item.knowledge_type),
                        'confidence': item.confidence,
                        'source': KnowledgeSourceType.KNOWLEDGE_BASE.value
                    })
            except Exception as e:
                logger.warning(f"Knowledge base query failed: {e}")

        # Search knowledge graph facts
        if self.knowledge_graph:
            try:
                kg_facts = await self.knowledge_graph.search_facts(query, topic=domain, limit=10)
                for fact in kg_facts:
                    facts.append({
                        'id': fact.id,
                        'content': fact.statement,
                        'topic': fact.topic,
                        'confidence': fact.confidence,
                        'source': KnowledgeSourceType.KNOWLEDGE_GRAPH.value
                    })
            except Exception as e:
                logger.warning(f"Knowledge graph query failed: {e}")

        # Get facts from factual memory
        if self.factual_memory:
            try:
                category = domain or 'general'
                fm_facts = self.factual_memory.get_facts_by_category(category, min_confidence=0.5)
                for fact in fm_facts[:5]:
                    facts.append({
                        'id': fact['fact_id'],
                        'content': str(fact['content']),
                        'confidence': fact['confidence'],
                        'source': KnowledgeSourceType.FACTUAL_MEMORY.value
                    })
            except Exception as e:
                logger.warning(f"Factual memory query failed: {e}")

        # Deduplicate and sort by confidence
        seen_content = set()
        unique_facts = []
        for fact in facts:
            content_hash = hashlib.md5(fact['content'].lower().encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_facts.append(fact)

        unique_facts.sort(key=lambda f: f['confidence'], reverse=True)
        return unique_facts[:self.config.max_knowledge_additions]

    async def _infer_relations(
        self,
        entities: List[Dict[str, Any]],
        facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Infer relations between entities."""
        relations = []

        if not self.knowledge_graph or len(entities) < 2:
            return relations

        # Get relations for entities from knowledge graph
        for entity in entities:
            if 'kg_id' in entity:
                try:
                    entity_relations = await self.knowledge_graph.get_relations(entity['kg_id'])
                    for relation, other_entity in entity_relations:
                        relations.append({
                            'source': entity['name'],
                            'target': other_entity.name,
                            'relation_type': relation.relation_type.value if hasattr(relation.relation_type, 'value') else str(relation.relation_type),
                            'confidence': relation.confidence,
                            'source_type': KnowledgeSourceType.KNOWLEDGE_GRAPH.value
                        })
                except Exception as e:
                    logger.debug(f"Failed to get relations for {entity['name']}: {e}")

        # Infer relations from co-occurrence in facts
        entity_names = {e['name'].lower() for e in entities}
        for fact in facts:
            content_lower = fact['content'].lower()
            co_occurring = [name for name in entity_names if name in content_lower]

            if len(co_occurring) >= 2:
                # Entities co-occur in a fact - infer relation
                for i, e1 in enumerate(co_occurring):
                    for e2 in co_occurring[i+1:]:
                        relations.append({
                            'source': e1,
                            'target': e2,
                            'relation_type': 'related_to',
                            'confidence': 0.6,
                            'source_type': 'fact_cooccurrence',
                            'evidence_fact': fact['id']
                        })

        return relations

    def _build_domain_context(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build domain-specific context for the query."""
        context = {
            'detected_domain': domain,
            'domain_concepts': [],
            'applicable_rules': [],
            'constraints': []
        }

        # Detect domain from entities if not specified
        if not domain:
            domain = self._detect_domain(entities)
            context['detected_domain'] = domain

        # Get domain ontology
        if domain and domain in self._domain_ontologies:
            ontology = self._domain_ontologies[domain]

            # Find matching concepts
            query_lower = query.lower()
            for concept, properties in ontology.get('concepts', {}).items():
                if concept.lower() in query_lower:
                    context['domain_concepts'].append({
                        'concept': concept,
                        'properties': properties
                    })

            # Get applicable rules
            for rule in self._domain_rules.get(domain, []):
                context['applicable_rules'].append(rule)

            # Get constraints
            context['constraints'] = ontology.get('constraints', [])

        return context

    def _calculate_grounding_confidence(self, grounding: KnowledgeGrounding) -> float:
        """Calculate overall confidence for knowledge grounding."""
        scores = []

        # Entity confidence
        if grounding.grounded_entities:
            entity_conf = sum(e['confidence'] for e in grounding.grounded_entities) / len(grounding.grounded_entities)
            scores.append(entity_conf * 0.3)

        # Fact confidence
        if grounding.grounded_facts:
            fact_conf = sum(f['confidence'] for f in grounding.grounded_facts) / len(grounding.grounded_facts)
            scores.append(fact_conf * 0.4)

        # Relation confidence
        if grounding.grounded_relations:
            rel_conf = sum(r['confidence'] for r in grounding.grounded_relations) / len(grounding.grounded_relations)
            scores.append(rel_conf * 0.2)

        # Domain context bonus
        if grounding.domain_context.get('domain_concepts'):
            scores.append(0.1)

        return sum(scores) if scores else 0.0

    def _generate_entity_information(self, entities: List[Dict[str, Any]]) -> str:
        """Generate textual information about entities."""
        if not entities:
            return ""

        lines = ["[Entity Context]"]
        for entity in entities[:5]:
            line = f"- {entity['name']}"
            if entity.get('type'):
                line += f" ({entity['type']})"
            if entity.get('kg_properties'):
                props = entity['kg_properties']
                if props:
                    line += f": {', '.join(f'{k}={v}' for k, v in list(props.items())[:3])}"
            lines.append(line)

        return "\n".join(lines)

    def _generate_fact_information(self, facts: List[Dict[str, Any]]) -> str:
        """Generate textual information about facts."""
        if not facts:
            return ""

        lines = ["[Relevant Facts]"]
        for fact in facts[:5]:
            content = fact['content']
            if len(content) > 150:
                content = content[:150] + "..."
            conf = fact.get('confidence', 0)
            lines.append(f"- {content} (confidence: {conf:.2f})")

        return "\n".join(lines)

    def _generate_relation_information(self, relations: List[Dict[str, Any]]) -> str:
        """Generate textual information about relations."""
        if not relations:
            return ""

        lines = ["[Entity Relationships]"]
        for rel in relations[:5]:
            lines.append(f"- {rel['source']} --[{rel['relation_type']}]--> {rel['target']}")

        return "\n".join(lines)

    def _generate_domain_context(self, domain_context: Dict[str, Any]) -> str:
        """Generate textual domain context."""
        if not domain_context or not domain_context.get('detected_domain'):
            return ""

        lines = [f"[Domain: {domain_context['detected_domain']}]"]

        if domain_context.get('domain_concepts'):
            concepts = [c['concept'] for c in domain_context['domain_concepts'][:3]]
            lines.append(f"Key concepts: {', '.join(concepts)}")

        if domain_context.get('constraints'):
            lines.append(f"Constraints: {len(domain_context['constraints'])} applicable")

        return "\n".join(lines)

    def _build_augmented_context(
        self,
        original: str,
        additions: List[Dict[str, Any]]
    ) -> str:
        """Build the final augmented context string."""
        if not additions:
            return original

        parts = []

        # Add knowledge augmentations first
        for addition in additions:
            if addition['content']:
                parts.append(addition['content'])

        # Add separator and original context
        if parts:
            parts.append("\n[Retrieved Context]")
        parts.append(original)

        return "\n\n".join(parts)

    def _calculate_context_coherence(self, augmented: AugmentedContext) -> float:
        """Calculate coherence score for augmented context."""
        score = 0.5  # Base score

        # Bonus for entity annotations
        if augmented.entity_annotations:
            score += min(0.15, len(augmented.entity_annotations) * 0.03)

        # Bonus for fact references
        if augmented.fact_references:
            avg_conf = sum(f['confidence'] for f in augmented.fact_references) / len(augmented.fact_references)
            score += 0.2 * avg_conf

        # Bonus for knowledge additions
        if augmented.knowledge_additions:
            score += min(0.15, len(augmented.knowledge_additions) * 0.05)

        return min(1.0, score)

    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text."""
        import re

        # Split into sentences
        sentences = re.split(r'[.!?]+', text)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter for meaningful claims
            if len(sentence) > 20:
                # Check for declarative patterns
                declarative_patterns = [
                    r'\bis\b',
                    r'\bare\b',
                    r'\bwas\b',
                    r'\bwere\b',
                    r'\bhas\b',
                    r'\bhave\b',
                    r'\bcan\b',
                    r'\bwill\b',
                ]
                if any(re.search(p, sentence.lower()) for p in declarative_patterns):
                    claims.append(sentence)

        return claims[:10]  # Limit to 10 claims

    async def _verify_claim(self, claim: str, query: str) -> Dict[str, Any]:
        """Verify a single claim against knowledge sources."""
        result = {
            'claim': claim,
            'status': 'unverified',
            'confidence': 0.0,
            'supporting_evidence': [],
            'contradictions': []
        }

        # Use truth verifier if available
        if self.truth_verifier:
            try:
                verification = await self.truth_verifier.verify_claim(
                    claim,
                    strict=self.config.strict_verification
                )

                result['confidence'] = verification.confidence_score
                result['supporting_evidence'] = [
                    {'statement': f.statement, 'confidence': f.confidence}
                    for f in verification.supporting_evidence
                ]
                result['contradictions'] = [
                    {'statement': f.statement, 'confidence': f.confidence}
                    for f in verification.contradictions
                ]

                if verification.contradictions:
                    result['status'] = 'contradicted'
                elif verification.supporting_evidence:
                    if verification.confidence_score > 0.7:
                        result['status'] = 'verified'
                    else:
                        result['status'] = 'partially_verified'

                return result
            except Exception as e:
                logger.warning(f"Truth verification failed: {e}")

        # Fallback: check against knowledge base
        if self.knowledge_base:
            try:
                kb_results = self.knowledge_base.query_knowledge(claim, max_results=5)
                if kb_results:
                    avg_conf = sum(r.confidence for r in kb_results) / len(kb_results)
                    result['confidence'] = avg_conf
                    result['status'] = 'partially_verified' if avg_conf > 0.5 else 'unverified'
                    result['supporting_evidence'] = [
                        {'statement': str(r.content), 'confidence': r.confidence}
                        for r in kb_results[:3]
                    ]
            except Exception as e:
                logger.warning(f"Knowledge base verification failed: {e}")

        return result

    async def _generate_correction(
        self,
        claim: str,
        verification_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a correction for a contradicted claim."""
        if not verification_result.get('contradictions'):
            return None

        # Get the most confident contradiction
        contradiction = max(
            verification_result['contradictions'],
            key=lambda c: c.get('confidence', 0)
        )

        return {
            'original_claim': claim,
            'contradicting_fact': contradiction['statement'],
            'suggested_correction': contradiction['statement'],
            'confidence': contradiction.get('confidence', 0.5)
        }

    def _apply_corrections(
        self,
        response: str,
        corrections: List[Dict[str, Any]]
    ) -> str:
        """Apply corrections to response."""
        corrected = response

        for correction in corrections:
            original = correction['original_claim']
            suggested = correction['suggested_correction']

            # Simple replacement (can be enhanced with more sophisticated NLP)
            if original in corrected:
                corrected = corrected.replace(
                    original,
                    f"{suggested} [corrected from: {original[:50]}...]"
                )

        return corrected

    def _calculate_verification_confidence(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall verification confidence."""
        if not results:
            return 0.5

        # Weight by verification status
        status_weights = {
            'verified': 1.0,
            'partially_verified': 0.7,
            'unverified': 0.3,
            'contradicted': 0.0
        }

        weighted_sum = 0.0
        for result in results:
            status_weight = status_weights.get(result['status'], 0.3)
            claim_conf = result.get('confidence', 0.5)
            weighted_sum += status_weight * claim_conf

        return weighted_sum / len(results)

    def _determine_coherence_level(self, score: float) -> KnowledgeCoherenceLevel:
        """Determine coherence level from score."""
        if score >= 0.9:
            return KnowledgeCoherenceLevel.EXCELLENT
        elif score >= 0.75:
            return KnowledgeCoherenceLevel.GOOD
        elif score >= 0.6:
            return KnowledgeCoherenceLevel.ACCEPTABLE
        elif score >= 0.4:
            return KnowledgeCoherenceLevel.NEEDS_IMPROVEMENT
        else:
            return KnowledgeCoherenceLevel.POOR

    def _create_knowledge_gap(self, claim: str, query: str) -> KnowledgeGap:
        """Create a knowledge gap record."""
        topic = self._extract_main_topic(claim)

        return KnowledgeGap(
            gap_id=f"gap_{hashlib.md5(claim.encode()).hexdigest()[:12]}",
            topic=topic,
            description=f"Unverifiable claim: {claim[:100]}...",
            detected_from=query,
            severity='moderate',
            suggested_sources=['knowledge_base', 'external_reference'],
            auto_fillable=False
        )

    async def _analyze_domain_alignment(
        self,
        query: str,
        response: str,
        domain: Optional[str] = None
    ) -> Dict[str, float]:
        """Analyze domain alignment between query and response."""
        alignment = {}

        # Detect domains in query and response
        query_entities = await self._extract_and_link_entities(query, domain)
        response_entities = await self._extract_and_link_entities(response, domain)

        query_types = {e['type'] for e in query_entities}
        response_types = {e['type'] for e in response_entities}

        # Calculate overlap
        if query_types:
            overlap = len(query_types & response_types) / len(query_types)
            alignment['entity_type_overlap'] = overlap

        # Check domain ontology alignment
        if domain and domain in self._domain_ontologies:
            ontology = self._domain_ontologies[domain]
            concepts = set(ontology.get('concepts', {}).keys())

            query_concepts = {c.lower() for c in concepts if c.lower() in query.lower()}
            response_concepts = {c.lower() for c in concepts if c.lower() in response.lower()}

            if query_concepts:
                concept_alignment = len(query_concepts & response_concepts) / len(query_concepts)
                alignment['concept_alignment'] = concept_alignment

        # Overall domain score
        if alignment:
            alignment['overall'] = sum(alignment.values()) / len(alignment)
        else:
            alignment['overall'] = 0.5  # Neutral score

        return alignment

    async def _check_entity_consistency(self, query: str, response: str) -> float:
        """Check consistency of entities between query and response."""
        query_entities = await self._extract_and_link_entities(query)
        response_entities = await self._extract_and_link_entities(response)

        if not query_entities:
            return 1.0  # No entities to check

        query_names = {e['name'].lower() for e in query_entities}
        response_names = {e['name'].lower() for e in response_entities}

        # Check if query entities are addressed in response
        addressed = len(query_names & response_names)
        coverage = addressed / len(query_names)

        return coverage

    async def _verify_fact_accuracy(self, response: str) -> float:
        """Verify factual accuracy of response."""
        claims = self._extract_claims(response)

        if not claims:
            return 0.5  # Neutral score

        verified_count = 0
        for claim in claims:
            result = await self._verify_claim(claim, response)
            if result['status'] in ['verified', 'partially_verified']:
                verified_count += 1

        return verified_count / len(claims)

    async def _validate_relations(self, response: str) -> float:
        """Validate relations mentioned in response."""
        # Extract entities
        entities = await self._extract_and_link_entities(response)

        if len(entities) < 2:
            return 1.0  # No relations to validate

        # Get actual relations from knowledge graph
        if not self.knowledge_graph:
            return 0.5  # Can't validate

        valid_relations = 0
        checked_relations = 0

        for i, e1 in enumerate(entities):
            if 'kg_id' not in e1:
                continue

            try:
                relations = await self.knowledge_graph.get_relations(e1['kg_id'])
                related_ids = {rel[1].name.lower() for rel in relations}

                for e2 in entities[i+1:]:
                    if e2['name'].lower() in related_ids:
                        valid_relations += 1
                    checked_relations += 1
            except Exception:
                pass

        if checked_relations == 0:
            return 0.5

        return valid_relations / checked_relations

    def _check_temporal_coherence(self, response: str) -> float:
        """Check temporal coherence of response."""
        import re

        # Extract temporal markers
        past_markers = len(re.findall(r'\bwas\b|\bwere\b|\bpreviously\b|\bformerly\b', response.lower()))
        present_markers = len(re.findall(r'\bis\b|\bare\b|\bcurrently\b|\bnow\b', response.lower()))
        future_markers = len(re.findall(r'\bwill\b|\bshall\b|\bgoing to\b', response.lower()))

        total_markers = past_markers + present_markers + future_markers

        if total_markers < 2:
            return 1.0  # Not enough markers to judge

        # Check for mixed tenses (potential inconsistency)
        marker_types = sum(1 for m in [past_markers, present_markers, future_markers] if m > 0)

        if marker_types == 1:
            return 1.0  # Consistent tense
        elif marker_types == 2:
            return 0.7  # Some variation (acceptable)
        else:
            return 0.5  # Mixed tenses (less coherent)

    def _generate_coherence_recommendations(
        self,
        report: DomainCoherenceReport
    ) -> List[str]:
        """Generate recommendations based on coherence report."""
        recommendations = []

        if report.entity_consistency < 0.6:
            recommendations.append(
                "Address more entities from the query in the response"
            )

        if report.fact_accuracy < 0.6:
            recommendations.append(
                "Verify factual claims against knowledge base"
            )

        if report.relation_validity < 0.6:
            recommendations.append(
                "Review entity relationships for accuracy"
            )

        if report.temporal_coherence < 0.7:
            recommendations.append(
                "Ensure consistent temporal references"
            )

        domain_score = report.domain_alignment.get('overall', 0.5)
        if domain_score < 0.6:
            recommendations.append(
                "Better align response with domain concepts"
            )

        return recommendations

    def _update_coherence_metrics(self, score: float):
        """Update running average of coherence scores."""
        total = self._metrics['total_augmentations'] + 1
        current_avg = self._metrics['avg_coherence_score']
        self._metrics['avg_coherence_score'] = (
            (current_avg * (total - 1) + score) / total
        )

    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from text."""
        import re

        # Simple noun phrase extraction
        words = text.lower().split()
        topics = []

        # Extract capitalized phrases
        capitalized = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        topics.extend(capitalized)

        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        topics.extend(quoted)

        # Extract technical terms (words with special characters)
        technical = re.findall(r'\b\w+[-_]\w+\b', text)
        topics.extend(technical)

        return list(set(topics))[:10]

    async def _check_topic_coverage(self, topic: str) -> float:
        """Check how well a topic is covered in knowledge base."""
        if not self.knowledge_base:
            return 0.5

        try:
            results = self.knowledge_base.query_knowledge(topic, max_results=10)
            if not results:
                return 0.0

            # Score based on number and confidence of results
            avg_confidence = sum(r.confidence for r in results) / len(results)
            coverage = min(1.0, len(results) / 5) * avg_confidence
            return coverage
        except Exception:
            return 0.5

    def _suggest_knowledge_sources(self, topic: str) -> List[str]:
        """Suggest sources to fill a knowledge gap."""
        suggestions = ['knowledge_base', 'external_reference']

        # Check if topic might be an entity
        if any(topic.lower().startswith(prefix) for prefix in ['python', 'java', 'react', 'aws']):
            suggestions.insert(0, 'knowledge_graph')

        return suggestions

    def _is_auto_fillable(self, topic: str) -> bool:
        """Check if a topic gap can be auto-filled."""
        # Topics that might be auto-fillable from web or existing resources
        auto_fillable_patterns = [
            r'^[A-Z][a-z]+$',  # Single capitalized word (likely a name/concept)
            r'\.(py|js|ts|java|go)$',  # File extensions
        ]

        import re
        return any(re.search(pattern, topic) for pattern in auto_fillable_patterns)

    async def _has_entity_knowledge(self, entity_name: str) -> bool:
        """Check if we have knowledge about an entity."""
        if self.knowledge_graph:
            try:
                entity = await self.knowledge_graph.find_entity(entity_name)
                return entity is not None
            except Exception:
                pass

        if self.knowledge_base:
            try:
                results = self.knowledge_base.query_knowledge(entity_name, max_results=1)
                return len(results) > 0
            except Exception:
                pass

        return False

    async def _auto_fill_gap(self, gap: KnowledgeGap) -> Optional[str]:
        """Attempt to auto-fill a knowledge gap."""
        # Try knowledge graph first
        if self.knowledge_graph and 'knowledge_graph' in gap.suggested_sources:
            try:
                entity = await self.knowledge_graph.find_entity(gap.topic)
                if entity:
                    return f"{gap.topic} is a {entity.entity_type.value}"
            except Exception:
                pass

        # Could integrate with external APIs here
        return None

    def _detect_domain(self, entities: List[Dict[str, Any]]) -> Optional[str]:
        """Detect domain from entities."""
        if not entities:
            return None

        type_counts = defaultdict(int)
        for entity in entities:
            type_counts[entity.get('type', 'unknown')] += 1

        if type_counts['technology'] > 0:
            return 'technology'
        elif type_counts['organization'] > 0:
            return 'business'
        elif type_counts['concept'] > 0:
            return 'general'

        return None

    def _extract_main_topic(self, text: str) -> str:
        """Extract main topic from text."""
        # Simple heuristic: first noun phrase
        words = text.split()[:5]
        return ' '.join(words)

    def _compute_cache_key(self, query: str, domain: Optional[str]) -> str:
        """Compute cache key for grounding."""
        content = f"{query}_{domain or 'none'}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_default_ontologies(self):
        """Load default domain ontologies."""
        # Software Engineering Domain
        self._domain_ontologies['technology'] = {
            'concepts': {
                'microservices': {'type': 'architecture', 'related': ['containers', 'api']},
                'API': {'type': 'interface', 'related': ['REST', 'GraphQL']},
                'database': {'type': 'storage', 'related': ['SQL', 'NoSQL']},
                'machine learning': {'type': 'ai', 'related': ['model', 'training']},
                'deployment': {'type': 'devops', 'related': ['CI/CD', 'containers']},
            },
            'relations': [
                ('microservices', 'uses', 'API'),
                ('microservices', 'uses', 'database'),
                ('machine learning', 'requires', 'data'),
            ],
            'synonyms': {
                'API': ['application programming interface', 'web service'],
                'database': ['DB', 'data store'],
                'machine learning': ['ML', 'artificial intelligence'],
            },
            'constraints': [
                'APIs should be versioned',
                'Databases require backup strategies',
            ]
        }

        # General Knowledge Domain
        self._domain_ontologies['general'] = {
            'concepts': {
                'fact': {'type': 'knowledge', 'verifiable': True},
                'opinion': {'type': 'knowledge', 'verifiable': False},
                'definition': {'type': 'knowledge', 'definitional': True},
            },
            'relations': [],
            'synonyms': {},
            'constraints': []
        }

    def _build_entity_aliases(self):
        """Build entity alias mappings from ontologies."""
        for domain, ontology in self._domain_ontologies.items():
            for canonical, aliases in ontology.get('synonyms', {}).items():
                self._entity_aliases[canonical.lower()].update(
                    alias.lower() for alias in aliases
                )

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get KAG performance metrics."""
        return {
            **self._metrics,
            'knowledge_gaps_count': len(self._knowledge_gaps),
            'unfilled_gaps_count': len([g for g in self._knowledge_gaps.values() if not g.filled]),
            'domain_ontologies_count': len(self._domain_ontologies),
            'cache_size': len(self._grounding_cache),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive KAG statistics."""
        return {
            'metrics': self.get_metrics(),
            'config': {
                'mode': self.config.augmentation_mode.value,
                'coherence_threshold': self.config.min_coherence_threshold,
                'verification_enabled': self.config.enable_fact_verification,
                'gap_detection_enabled': self.config.enable_gap_detection,
            },
            'knowledge_sources': {
                'knowledge_base': self.knowledge_base is not None,
                'knowledge_graph': self.knowledge_graph is not None,
                'factual_memory': self.factual_memory is not None,
                'truth_verifier': self.truth_verifier is not None,
            },
            'domains_registered': list(self._domain_ontologies.keys()),
        }

    def clear_cache(self):
        """Clear grounding cache."""
        self._grounding_cache.clear()
        self._metrics['cache_hits'] = 0
        self._metrics['cache_misses'] = 0


# Factory function
def create_kag(
    knowledge_base=None,
    knowledge_graph=None,
    factual_memory=None,
    truth_verifier=None,
    mode: str = 'standard'
) -> KnowledgeAugmentedGeneration:
    """
    Create a KAG instance with specified configuration.

    Args:
        knowledge_base: KnowledgeBase instance
        knowledge_graph: KnowledgeGraph instance
        factual_memory: FactualMemoryEngine instance
        truth_verifier: TruthVerifier instance
        mode: Augmentation mode ('light', 'standard', 'comprehensive', 'domain_expert')

    Returns:
        Configured KnowledgeAugmentedGeneration instance
    """
    mode_enum = KnowledgeAugmentationMode(mode)

    config = KAGConfig(augmentation_mode=mode_enum)

    # Adjust config based on mode
    if mode_enum == KnowledgeAugmentationMode.LIGHT:
        config.enable_relation_inference = False
        config.enable_gap_detection = False
        config.max_knowledge_additions = 5
    elif mode_enum == KnowledgeAugmentationMode.COMPREHENSIVE:
        config.max_knowledge_additions = 15
        config.auto_fill_gaps = True
        config.parallel_verification = True
    elif mode_enum == KnowledgeAugmentationMode.DOMAIN_EXPERT:
        config.max_knowledge_additions = 20
        config.auto_fill_gaps = True
        config.strict_verification = True
    elif mode_enum == KnowledgeAugmentationMode.VERIFICATION_STRICT:
        config.strict_verification = True
        config.enable_contradiction_prevention = True

    return KnowledgeAugmentedGeneration(
        knowledge_base=knowledge_base,
        knowledge_graph=knowledge_graph,
        factual_memory=factual_memory,
        truth_verifier=truth_verifier,
        config=config
    )
