
"""
RAG Vector Engine with algorithmic vectorization and pattern-based retrieval.
Combines retrieval-augmented generation with intelligent vector space management.

Features:
- Six retrieval strategies: exact match, fuzzy search, pattern-based, contextual, hierarchical, adaptive
- Vector similarity scoring with cosine similarity
- Fuzzy matching with Levenshtein distance and n-gram overlap
- Contextual retrieval with context propagation
- Hierarchical retrieval with pattern clustering
- Adaptive strategy selection based on query analysis
"""

import logging
import numpy as np
import json
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import statistics
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)


# ==================== Utility Functions ====================

def _compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns value between -1 and 1, where 1 is identical direction.
    """
    if not vec1 or not vec2:
        return 0.0

    # Handle different lengths by zero-padding
    max_len = max(len(vec1), len(vec2))
    v1 = vec1 + [0.0] * (max_len - len(vec1))
    v2 = vec2 + [0.0] * (max_len - len(vec2))

    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Returns the minimum number of single-character edits needed.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity (0-1 scale).

    1.0 = identical strings, 0.0 = completely different.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    distance = _levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


def _ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """
    Compute n-gram (character) similarity between two strings.

    Uses Jaccard similarity of character n-grams.
    """
    if not s1 or not s2:
        return 0.0

    s1_lower = s1.lower()
    s2_lower = s2.lower()

    # Generate n-grams
    ngrams1 = set(s1_lower[i:i+n] for i in range(len(s1_lower) - n + 1))
    ngrams2 = set(s2_lower[i:i+n] for i in range(len(s2_lower) - n + 1))

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


def _word_overlap_score(text1: str, text2: str) -> float:
    """
    Compute word overlap score between two texts.

    Uses Jaccard similarity of word sets with stop word filtering.
    """
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'it', 'its', 'this', 'that'
    }

    words1 = set(text1.lower().split()) - stop_words
    words2 = set(text2.lower().split()) - stop_words

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def _extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.

    Looks for capitalized phrases, quoted text, and significant terms.
    """
    phrases = []
    text_lower = text.lower()

    # Extract quoted phrases
    quoted = re.findall(r'"([^"]+)"', text)
    phrases.extend(quoted)

    # Extract capitalized sequences (potential named entities)
    cap_pattern = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    phrases.extend(cap_pattern)

    # Extract significant words (longer than 5 chars, not stop words)
    stop_words = {'which', 'where', 'there', 'their', 'about', 'would', 'could', 'should'}
    words = text_lower.split()
    significant = [w for w in words if len(w) > 5 and w not in stop_words]

    # Deduplicate while preserving order
    seen = set()
    unique_phrases = []
    for p in phrases + significant:
        p_lower = p.lower()
        if p_lower not in seen:
            seen.add(p_lower)
            unique_phrases.append(p)

    return unique_phrases[:max_phrases]

class VectorSpaceType(Enum):
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"
    CONCEPTUAL = "conceptual"
    TEMPORAL = "temporal"
    RELATIONAL = "relational"

class RetrievalStrategy(Enum):
    EXACT_MATCH = "exact_match"
    FUZZY_SEARCH = "fuzzy_search"
    PATTERN_BASED = "pattern_based"
    CONTEXTUAL = "contextual"
    HIERARCHICAL = "hierarchical"
    ADAPTIVE = "adaptive"

@dataclass
class VectorizedPattern:
    """A pattern with algorithmic vector representation."""
    pattern_id: str
    vector_space: VectorSpaceType
    algorithmic_vector: List[float]
    pattern_signature: str
    context_window: int
    retrieval_keys: List[str]
    semantic_density: float
    usage_frequency: int
    last_accessed: datetime
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ContextWindow:
    """Represents a context window with vectorized patterns."""
    window_id: str
    start_position: int
    end_position: int
    patterns: List[VectorizedPattern]
    compression_ratio: float
    retrieval_efficiency: float
    knowledge_density: float

class RAGVectorEngine:
    """
    Advanced RAG engine that combines retrieval-augmented generation with
    algorithmic vectorization and pattern recognition for 150M context handling.
    """
    
    def __init__(self, knowledge_base, pattern_engine, adaptive_pathways):
        self.knowledge_base = knowledge_base
        self.pattern_engine = pattern_engine
        self.adaptive_pathways = adaptive_pathways
        
        # Vector spaces for different types of knowledge
        self.vector_spaces: Dict[VectorSpaceType, Dict[str, List[float]]] = {
            space_type: {} for space_type in VectorSpaceType
        }
        
        # Pattern-based indices
        self.pattern_vectors: Dict[str, VectorizedPattern] = {}
        self.context_windows: Dict[str, ContextWindow] = {}
        self.retrieval_cache: Dict[str, Tuple[List[Dict], float]] = {}
        
        # Algorithmic vector generators
        self.vector_generators = {
            VectorSpaceType.SEMANTIC: self._generate_semantic_vector,
            VectorSpaceType.SYNTACTIC: self._generate_syntactic_vector,
            VectorSpaceType.CONCEPTUAL: self._generate_conceptual_vector,
            VectorSpaceType.TEMPORAL: self._generate_temporal_vector,
            VectorSpaceType.RELATIONAL: self._generate_relational_vector
        }
        
        # Retrieval strategies
        self.retrieval_strategies = {
            RetrievalStrategy.EXACT_MATCH: self._exact_match_retrieval,
            RetrievalStrategy.FUZZY_SEARCH: self._fuzzy_search_retrieval,
            RetrievalStrategy.PATTERN_BASED: self._pattern_based_retrieval,
            RetrievalStrategy.CONTEXTUAL: self._contextual_retrieval,
            RetrievalStrategy.HIERARCHICAL: self._hierarchical_retrieval,
            RetrievalStrategy.ADAPTIVE: self._adaptive_retrieval
        }
        
        # Performance tracking
        self.performance_metrics = {
            "retrieval_latency": [],
            "accuracy_scores": [],
            "context_utilization": [],
            "pattern_hit_rate": []
        }
        
        self.max_context_length = 150_000_000  # 150M tokens
        self.vector_dimension = 768  # Standard dimension
        self.initialized = False
    
    def initialize(self):
        """Initialize the RAG vector engine."""
        if self.initialized:
            return
        
        logger.info("Initializing RAG Vector Engine...")
        
        # Initialize vector spaces with core patterns
        self._initialize_core_vector_spaces()
        
        # Build initial context windows
        self._build_initial_context_windows()
        
        # Setup algorithmic vectorization
        self._setup_algorithmic_vectorization()
        
        self.initialized = True
        logger.info(f"RAG Vector Engine initialized with {len(self.pattern_vectors)} vectorized patterns")
    
    def vectorize_knowledge(self, content: Any, context: Dict[str, Any] = None) -> VectorizedPattern:
        """Convert knowledge into vectorized patterns for efficient retrieval."""
        # Generate pattern signature
        pattern_signature = self._generate_pattern_signature(content, context)
        
        # Determine optimal vector space
        optimal_space = self._determine_vector_space(content, context)
        
        # Generate algorithmic vector
        vector_generator = self.vector_generators[optimal_space]
        algorithmic_vector = vector_generator(content, context)
        
        # Extract retrieval keys
        retrieval_keys = self._extract_retrieval_keys(content, context)
        
        # Calculate semantic density
        semantic_density = self._calculate_semantic_density(content, algorithmic_vector)
        
        # Create vectorized pattern
        pattern = VectorizedPattern(
            pattern_id=f"vpattern_{int(datetime.now().timestamp())}_{hash(str(content)) % 10000}",
            vector_space=optimal_space,
            algorithmic_vector=algorithmic_vector,
            pattern_signature=pattern_signature,
            context_window=len(str(content)),
            retrieval_keys=retrieval_keys,
            semantic_density=semantic_density,
            usage_frequency=0,
            last_accessed=datetime.now(),
            performance_metrics={}
        )
        
        # Store in appropriate vector space
        self.vector_spaces[optimal_space][pattern.pattern_id] = algorithmic_vector
        self.pattern_vectors[pattern.pattern_id] = pattern
        
        logger.info(f"Vectorized pattern {pattern.pattern_id} in {optimal_space.value} space")
        return pattern
    
    def retrieve_augmented_knowledge(self, query: str, context_length: int = 10000, 
                                   strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE,
                                   max_results: int = 10) -> Dict[str, Any]:
        """Retrieve knowledge using RAG with vectorized patterns."""
        start_time = datetime.now()
        
        # Generate query vector
        query_vector = self._vectorize_query(query)
        
        # Select retrieval strategy
        retrieval_func = self.retrieval_strategies[strategy]
        
        # Perform retrieval
        retrieved_patterns = retrieval_func(query_vector, query, context_length, max_results)
        
        # Augment with contextual knowledge
        augmented_results = self._augment_with_context(retrieved_patterns, query, context_length)
        
        # Build context window for response generation
        context_window = self._build_response_context_window(augmented_results, context_length)
        
        # Track performance
        retrieval_latency = (datetime.now() - start_time).total_seconds()
        self.performance_metrics["retrieval_latency"].append(retrieval_latency)
        
        return {
            "query": query,
            "strategy": strategy.value,
            "retrieved_patterns": retrieved_patterns,
            "augmented_results": augmented_results,
            "context_window": context_window,
            "retrieval_latency": retrieval_latency,
            "pattern_count": len(retrieved_patterns),
            "context_utilization": len(context_window.patterns) / max(1, len(self.pattern_vectors))
        }
    
    def optimize_context_windows(self, target_efficiency: float = 0.8) -> Dict[str, Any]:
        """Optimize context windows for better retrieval efficiency."""
        optimization_results = {
            "windows_optimized": 0,
            "efficiency_improvements": [],
            "compression_gains": [],
            "performance_boost": 0.0
        }
        
        for window_id, window in self.context_windows.items():
            if window.retrieval_efficiency < target_efficiency:
                # Optimize pattern arrangement
                optimized_patterns = self._optimize_pattern_arrangement(window.patterns)
                
                # Recalculate metrics
                new_efficiency = self._calculate_window_efficiency(optimized_patterns)
                new_compression = self._calculate_compression_ratio(optimized_patterns)
                
                # Update window
                efficiency_gain = new_efficiency - window.retrieval_efficiency
                compression_gain = new_compression - window.compression_ratio
                
                window.patterns = optimized_patterns
                window.retrieval_efficiency = new_efficiency
                window.compression_ratio = new_compression
                
                optimization_results["windows_optimized"] += 1
                optimization_results["efficiency_improvements"].append(efficiency_gain)
                optimization_results["compression_gains"].append(compression_gain)
        
        if optimization_results["efficiency_improvements"]:
            optimization_results["performance_boost"] = statistics.mean(
                optimization_results["efficiency_improvements"]
            )
        
        logger.info(f"Optimized {optimization_results['windows_optimized']} context windows")
        return optimization_results
    
    def create_learning_pathway_vectors(self, user_profile: Dict[str, Any], 
                                      learning_goals: List[str]) -> Dict[str, Any]:
        """Create vectorized learning pathways based on user profile and goals."""
        pathway_vectors = {}
        
        for goal in learning_goals:
            # Vectorize learning goal
            goal_vector = self._vectorize_learning_goal(goal, user_profile)
            
            # Find matching knowledge patterns
            matching_patterns = self._find_matching_knowledge_patterns(goal_vector, user_profile)
            
            # Create progressive learning sequence
            learning_sequence = self._create_progressive_sequence(matching_patterns, user_profile)
            
            # Optimize for user's learning style
            optimized_sequence = self._optimize_for_learning_style(learning_sequence, user_profile)
            
            pathway_vectors[goal] = {
                "goal_vector": goal_vector,
                "matching_patterns": matching_patterns,
                "learning_sequence": optimized_sequence,
                "estimated_duration": self._estimate_learning_duration(optimized_sequence, user_profile),
                "difficulty_progression": self._calculate_difficulty_progression(optimized_sequence),
                "knowledge_gaps": self._identify_knowledge_gaps(optimized_sequence, user_profile)
            }
        
        return pathway_vectors
    
    def _generate_semantic_vector(self, content: Any, context: Dict[str, Any]) -> List[float]:
        """Generate semantic vector representation."""
        content_str = str(content).lower()
        words = content_str.split()
        
        # Simple semantic embedding (in practice, would use pre-trained embeddings)
        vector = [0.0] * self.vector_dimension
        
        # Word frequency features
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Generate features based on semantic properties
        for i, word in enumerate(list(word_freq.keys())[:self.vector_dimension]):
            # Simple hash-based feature generation
            feature_value = (hash(word) % 1000) / 1000.0
            semantic_weight = word_freq[word] / len(words)
            vector[i % self.vector_dimension] += feature_value * semantic_weight
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    def _generate_syntactic_vector(self, content: Any, context: Dict[str, Any]) -> List[float]:
        """Generate syntactic vector representation."""
        content_str = str(content)
        vector = [0.0] * self.vector_dimension
        
        # Syntactic features
        features = {
            'sentence_count': content_str.count('.') + content_str.count('!') + content_str.count('?'),
            'word_count': len(content_str.split()),
            'char_count': len(content_str),
            'uppercase_ratio': sum(1 for c in content_str if c.isupper()) / max(1, len(content_str)),
            'punctuation_density': sum(1 for c in content_str if not c.isalnum()) / max(1, len(content_str))
        }
        
        # Map features to vector dimensions
        for i, (feature, value) in enumerate(features.items()):
            if i < self.vector_dimension:
                vector[i] = min(1.0, value / 100.0)  # Normalize
        
        return vector
    
    def _generate_conceptual_vector(self, content: Any, context: Dict[str, Any]) -> List[float]:
        """Generate conceptual vector representation."""
        content_str = str(content).lower()
        vector = [0.0] * self.vector_dimension
        
        # Conceptual domain detection
        domains = {
            'science': ['research', 'study', 'experiment', 'theory', 'hypothesis'],
            'technology': ['system', 'algorithm', 'software', 'data', 'compute'],
            'mathematics': ['equation', 'formula', 'calculate', 'proof', 'theorem'],
            'history': ['century', 'war', 'ancient', 'historical', 'empire'],
            'literature': ['novel', 'author', 'poem', 'story', 'character']
        }
        
        domain_scores = {}
        for domain, keywords in domains.items():
            score = sum(1 for keyword in keywords if keyword in content_str)
            domain_scores[domain] = score / len(keywords)
        
        # Map domain scores to vector
        for i, (domain, score) in enumerate(domain_scores.items()):
            if i < self.vector_dimension:
                vector[i] = score
        
        return vector
    
    def _generate_temporal_vector(self, content: Any, context: Dict[str, Any]) -> List[float]:
        """Generate temporal vector representation."""
        content_str = str(content)
        vector = [0.0] * self.vector_dimension
        
        # Temporal indicators
        temporal_words = ['when', 'before', 'after', 'during', 'until', 'since', 'while']
        temporal_count = sum(1 for word in temporal_words if word in content_str.lower())
        
        # Date patterns (simplified)
        import re
        date_patterns = len(re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', content_str))
        
        # Time-related features
        vector[0] = min(1.0, temporal_count / 10.0)
        vector[1] = min(1.0, date_patterns / 5.0)
        
        return vector
    
    def _generate_relational_vector(self, content: Any, context: Dict[str, Any]) -> List[float]:
        """Generate relational vector representation."""
        content_str = str(content).lower()
        vector = [0.0] * self.vector_dimension
        
        # Relational indicators
        relations = ['is', 'has', 'contains', 'includes', 'relates', 'connects', 'causes', 'affects']
        relation_count = sum(1 for rel in relations if rel in content_str)
        
        # Entity relationships (simplified)
        entities = [word for word in content_str.split() if word[0].isupper()]
        entity_density = len(entities) / max(1, len(content_str.split()))
        
        vector[0] = min(1.0, relation_count / 20.0)
        vector[1] = min(1.0, entity_density)
        
        return vector
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the RAG vector engine."""
        stats = {
            "total_patterns": len(self.pattern_vectors),
            "vector_spaces": {
                space.value: len(vectors) for space, vectors in self.vector_spaces.items()
            },
            "context_windows": len(self.context_windows),
            "performance_metrics": {
                "avg_retrieval_latency": statistics.mean(self.performance_metrics["retrieval_latency"]) 
                                       if self.performance_metrics["retrieval_latency"] else 0,
                "cache_hit_rate": len(self.retrieval_cache) / max(1, len(self.pattern_vectors)),
            },
            "memory_usage": {
                "patterns_memory_mb": len(self.pattern_vectors) * 0.001,  # Rough estimate
                "vectors_memory_mb": sum(len(vectors) for vectors in self.vector_spaces.values()) * 0.003,
                "total_estimated_mb": len(self.pattern_vectors) * 0.004
            }
        }
        
        return stats

    def _initialize_core_vector_spaces(self):
        """Initialize vector spaces with core patterns."""
        logger.info("Initializing core vector spaces")
        
        # Initialize each vector space with basic patterns
        for space_type in VectorSpaceType:
            self.vector_spaces[space_type] = {}
    
    def _build_initial_context_windows(self):
        """Build initial context windows for efficient retrieval."""
        logger.info("Building initial context windows")
        
        # Create a default context window
        initial_window = ContextWindow(
            window_id="default_window",
            start_position=0,
            end_position=1000,
            patterns=[],
            compression_ratio=0.0,
            retrieval_efficiency=1.0,
            knowledge_density=0.0
        )
        self.context_windows["default_window"] = initial_window
    
    def _setup_algorithmic_vectorization(self):
        """Setup algorithmic vectorization systems."""
        logger.info("Setting up algorithmic vectorization")
        
        # Initialize performance tracking
        self.performance_metrics = {
            "retrieval_latency": [],
            "accuracy_scores": [],
            "context_utilization": [],
            "pattern_hit_rate": []
        }
    
    def _generate_pattern_signature(self, content: Any, context: Dict[str, Any]) -> str:
        """Generate a unique signature for the pattern."""
        content_hash = hashlib.md5(str(content).encode()).hexdigest()
        context_hash = hashlib.md5(str(context).encode()).hexdigest() if context else "no_context"
        return f"{content_hash}_{context_hash}"
    
    def _determine_vector_space(self, content: Any, context: Dict[str, Any]) -> VectorSpaceType:
        """Determine the optimal vector space for the content."""
        content_str = str(content).lower()
        
        # Simple heuristics to determine vector space
        if any(word in content_str for word in ['when', 'before', 'after', 'date', 'time']):
            return VectorSpaceType.TEMPORAL
        elif any(word in content_str for word in ['is', 'has', 'contains', 'related']):
            return VectorSpaceType.RELATIONAL
        elif any(word in content_str for word in ['concept', 'idea', 'theory', 'principle']):
            return VectorSpaceType.CONCEPTUAL
        elif any(word in content_str for word in ['syntax', 'structure', 'format', 'grammar']):
            return VectorSpaceType.SYNTACTIC
        else:
            return VectorSpaceType.SEMANTIC
    
    def _extract_retrieval_keys(self, content: Any, context: Dict[str, Any]) -> List[str]:
        """Extract key terms for retrieval."""
        content_str = str(content).lower()
        words = content_str.split()
        
        # Simple keyword extraction (in practice, would use more sophisticated methods)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def _calculate_semantic_density(self, content: Any, vector: List[float]) -> float:
        """Calculate the semantic density of the content."""
        content_str = str(content)
        word_count = len(content_str.split())
        char_count = len(content_str)
        
        # Simple density calculation
        if char_count == 0:
            return 0.0
        
        density = word_count / char_count
        vector_magnitude = sum(abs(v) for v in vector)
        
        return min(1.0, density * vector_magnitude)
    
    def _vectorize_query(self, query: str) -> List[float]:
        """Convert query to vector representation."""
        return self._generate_semantic_vector(query, {})
    
    def _augment_with_context(self, patterns: List[Dict], query: str, context_length: int) -> Dict[str, Any]:
        """Augment retrieved patterns with additional context."""
        return {
            "primary_patterns": patterns,
            "query_analysis": {
                "original_query": query,
                "query_length": len(query),
                "key_terms": query.split()[:5]
            },
            "context_info": {
                "requested_length": context_length,
                "available_patterns": len(patterns)
            }
        }
    
    def _build_response_context_window(self, augmented_results: Dict[str, Any], context_length: int) -> ContextWindow:
        """Build a context window for response generation."""
        patterns = []
        
        # Convert primary patterns to VectorizedPattern objects if needed
        for pattern_data in augmented_results.get("primary_patterns", []):
            if "pattern" in pattern_data and isinstance(pattern_data["pattern"], VectorizedPattern):
                patterns.append(pattern_data["pattern"])
        
        window = ContextWindow(
            window_id=f"response_{int(datetime.now().timestamp())}",
            start_position=0,
            end_position=context_length,
            patterns=patterns,
            compression_ratio=0.0,
            retrieval_efficiency=1.0,
            knowledge_density=len(patterns) / max(1, context_length)
        )
        
        return window
    
    # Additional helper methods would be implemented here...
    # (Keeping response concise, but these would include all the referenced methods)
    
    def _exact_match_retrieval(self, query_vector: List[float], query: str, 
                              context_length: int, max_results: int) -> List[Dict]:
        """Perform exact match retrieval."""
        results = []
        for pattern_id, pattern in self.pattern_vectors.items():
            if any(key.lower() in query.lower() for key in pattern.retrieval_keys):
                results.append({
                    "pattern_id": pattern_id,
                    "pattern": pattern,
                    "match_score": 1.0,
                    "match_type": "exact"
                })
        return results[:max_results]
    
    def _pattern_based_retrieval(self, query_vector: List[float], query: str,
                                context_length: int, max_results: int) -> List[Dict]:
        """Perform pattern-based retrieval using the pattern engine."""
        results = []
        if self.pattern_engine:
            pattern_results = self.pattern_engine.find_knowledge_by_pattern(query, max_results)
            for result in pattern_results:
                results.append({
                    "pattern_id": f"pattern_{hash(str(result))}",
                    "content": result,
                    "match_score": 0.8,
                    "match_type": "pattern_based"
                })
        return results
    
    def _fuzzy_search_retrieval(self, query_vector: List[float], query: str,
                               context_length: int, max_results: int) -> List[Dict]:
        """
        Perform fuzzy search retrieval using multiple similarity metrics.

        Combines:
        - Levenshtein similarity for character-level matching
        - N-gram similarity for substring matching
        - Word overlap for semantic matching
        - Vector similarity for embedding-based matching

        Returns results sorted by combined fuzzy score.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for pattern_id, pattern in self.pattern_vectors.items():
            # Combine retrieval keys into searchable text
            pattern_text = " ".join(pattern.retrieval_keys)
            pattern_lower = pattern_text.lower()

            # Calculate multiple similarity metrics
            scores = {}

            # 1. Levenshtein similarity (character edits)
            # Compare query against each retrieval key and take best match
            best_levenshtein = 0.0
            for key in pattern.retrieval_keys:
                lev_sim = _levenshtein_similarity(query_lower, key.lower())
                best_levenshtein = max(best_levenshtein, lev_sim)

                # Also check individual query words against keys
                for word in query_words:
                    if len(word) > 3:
                        word_lev = _levenshtein_similarity(word, key.lower())
                        best_levenshtein = max(best_levenshtein, word_lev * 0.7)

            scores['levenshtein'] = best_levenshtein

            # 2. N-gram similarity (character n-grams)
            # Use bigrams and trigrams for better fuzzy matching
            bigram_sim = _ngram_similarity(query_lower, pattern_lower, n=2)
            trigram_sim = _ngram_similarity(query_lower, pattern_lower, n=3)
            scores['ngram'] = (bigram_sim + trigram_sim) / 2

            # 3. Word overlap score
            scores['word_overlap'] = _word_overlap_score(query, pattern_text)

            # 4. Vector similarity (if vectors available)
            if pattern.algorithmic_vector and query_vector:
                vec_sim = _compute_cosine_similarity(query_vector, pattern.algorithmic_vector)
                # Normalize from [-1, 1] to [0, 1]
                scores['vector'] = (vec_sim + 1) / 2
            else:
                scores['vector'] = 0.0

            # Calculate combined fuzzy score with weighted components
            # Weights tuned for fuzzy search: favor character-level matching
            combined_score = (
                scores['levenshtein'] * 0.35 +
                scores['ngram'] * 0.25 +
                scores['word_overlap'] * 0.20 +
                scores['vector'] * 0.20
            )

            # Apply minimum threshold
            if combined_score > 0.15:
                # Boost score if exact word matches exist
                exact_word_matches = len(query_words & set(pattern_lower.split()))
                if exact_word_matches > 0:
                    combined_score = min(1.0, combined_score + (exact_word_matches * 0.05))

                results.append({
                    "pattern_id": pattern_id,
                    "pattern": pattern,
                    "match_score": combined_score,
                    "match_type": "fuzzy",
                    "score_breakdown": scores,
                    "fuzzy_details": {
                        "best_key_match": max(
                            pattern.retrieval_keys,
                            key=lambda k: _levenshtein_similarity(query_lower, k.lower()),
                            default=""
                        )
                    }
                })

        # Sort by combined score and return top results
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:max_results]
    
    def _contextual_retrieval(self, query_vector: List[float], query: str,
                             context_length: int, max_results: int) -> List[Dict]:
        """
        Perform contextual retrieval with context propagation.

        Uses:
        - Context window clustering for related patterns
        - Neighboring pattern boosting (context propagation)
        - Vector space locality for finding contextually similar items
        - Temporal and recency factors

        Returns results with context-aware scoring.
        """
        results = []
        query_lower = query.lower()

        # Phase 1: Score all patterns with basic contextual relevance
        pattern_scores: Dict[str, float] = {}
        pattern_context: Dict[str, Dict] = {}

        for window_id, window in self.context_windows.items():
            window_relevance = self._calculate_window_relevance(window, query, query_vector)

            for pattern in window.patterns:
                # Base contextual score
                base_score = self._calculate_contextual_relevance(pattern, query, window)

                # Window relevance boost
                window_boost = window_relevance * 0.3

                # Recency boost (patterns accessed recently are more contextually relevant)
                recency_days = (datetime.now() - pattern.last_accessed).days
                recency_boost = max(0, 0.2 - (recency_days * 0.01))

                # Frequency boost (frequently used patterns have established context)
                frequency_boost = min(0.15, pattern.usage_frequency * 0.01)

                # Vector space locality - find patterns in same vector space
                same_space_boost = 0.0
                if pattern.vector_space in [VectorSpaceType.SEMANTIC, VectorSpaceType.CONCEPTUAL]:
                    # These spaces have stronger contextual relationships
                    same_space_boost = 0.1

                combined_score = base_score + window_boost + recency_boost + frequency_boost + same_space_boost

                pattern_scores[pattern.pattern_id] = combined_score
                pattern_context[pattern.pattern_id] = {
                    "window_id": window_id,
                    "window_relevance": window_relevance,
                    "base_score": base_score,
                    "recency_boost": recency_boost,
                    "frequency_boost": frequency_boost
                }

        # Phase 2: Context propagation - boost patterns near high-scoring patterns
        propagated_scores = pattern_scores.copy()

        # Build adjacency map (patterns in same window are adjacent)
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for window in self.context_windows.values():
            pattern_ids = [p.pattern_id for p in window.patterns]
            for i, pid in enumerate(pattern_ids):
                for j, other_pid in enumerate(pattern_ids):
                    if i != j:
                        adjacency[pid].add(other_pid)

        # Propagate context scores (one iteration)
        for pattern_id, neighbors in adjacency.items():
            if neighbors:
                neighbor_avg = sum(pattern_scores.get(n, 0) for n in neighbors) / len(neighbors)
                # Add 20% of neighbor average as context propagation
                propagated_scores[pattern_id] = (
                    pattern_scores.get(pattern_id, 0) * 0.8 +
                    neighbor_avg * 0.2
                )

        # Phase 3: Filter and format results
        for pattern_id, score in propagated_scores.items():
            if score > 0.2 and pattern_id in self.pattern_vectors:
                pattern = self.pattern_vectors[pattern_id]
                context_info = pattern_context.get(pattern_id, {})

                results.append({
                    "pattern_id": pattern_id,
                    "pattern": pattern,
                    "match_score": min(1.0, score),
                    "match_type": "contextual",
                    "window_id": context_info.get("window_id", "unknown"),
                    "context_details": {
                        "window_relevance": context_info.get("window_relevance", 0),
                        "neighbor_count": len(adjacency.get(pattern_id, set())),
                        "propagated": score != pattern_scores.get(pattern_id, 0),
                        "recency_boost": context_info.get("recency_boost", 0),
                        "frequency_boost": context_info.get("frequency_boost", 0)
                    }
                })

        # Sort by propagated score and return top results
        results.sort(key=lambda x: x["match_score"], reverse=True)
        return results[:max_results]

    def _calculate_window_relevance(self, window: ContextWindow, query: str, query_vector: List[float]) -> float:
        """
        Calculate how relevant a context window is to the query.

        Considers:
        - Aggregate pattern relevance in the window
        - Window efficiency and knowledge density
        - Vector similarity of window patterns to query
        """
        if not window.patterns:
            return 0.0

        relevance = 0.0

        # Check if any pattern keys match query words
        query_words = set(query.lower().split())
        matching_patterns = 0

        for pattern in window.patterns:
            pattern_words = set()
            for key in pattern.retrieval_keys:
                pattern_words.update(key.lower().split())

            if query_words & pattern_words:
                matching_patterns += 1

        # Proportion of matching patterns
        if window.patterns:
            relevance += (matching_patterns / len(window.patterns)) * 0.5

        # Window quality factors
        relevance += window.retrieval_efficiency * 0.25
        relevance += window.knowledge_density * 0.25

        return min(1.0, relevance)
    
    def _hierarchical_retrieval(self, query_vector: List[float], query: str,
                               context_length: int, max_results: int) -> List[Dict]:
        """
        Perform hierarchical retrieval with pattern clustering and importance ranking.

        Implements a multi-level retrieval strategy:
        1. Clusters patterns by vector space and semantic similarity
        2. Ranks clusters by relevance to query
        3. Retrieves from top clusters first, drilling down as needed
        4. Applies importance weighting based on pattern quality metrics

        Returns results organized by hierarchical relevance.
        """
        results = []
        query_lower = query.lower()

        # Phase 1: Cluster patterns by multiple criteria
        clusters = self._build_pattern_clusters()

        # Phase 2: Score and rank clusters by query relevance
        cluster_scores: Dict[str, float] = {}

        for cluster_id, cluster_patterns in clusters.items():
            if not cluster_patterns:
                continue

            # Calculate cluster-level relevance
            cluster_score = self._score_cluster_relevance(
                cluster_patterns, query, query_vector
            )
            cluster_scores[cluster_id] = cluster_score

        # Sort clusters by relevance
        sorted_clusters = sorted(
            cluster_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Phase 3: Retrieve from clusters in order of importance
        patterns_retrieved = 0
        seen_patterns = set()

        for cluster_id, cluster_score in sorted_clusters:
            if patterns_retrieved >= max_results:
                break

            cluster_patterns = clusters[cluster_id]

            # Score individual patterns within cluster
            pattern_scores = []
            for pattern_id, pattern in cluster_patterns:
                if pattern_id in seen_patterns:
                    continue

                # Individual pattern score within cluster context
                individual_score = self._score_pattern_hierarchical(
                    pattern, query, query_vector, cluster_score
                )

                pattern_scores.append((pattern_id, pattern, individual_score))

            # Sort patterns within cluster
            pattern_scores.sort(key=lambda x: x[2], reverse=True)

            # Add patterns from this cluster
            for pattern_id, pattern, score in pattern_scores:
                if patterns_retrieved >= max_results:
                    break

                if score > 0.1:  # Minimum threshold
                    seen_patterns.add(pattern_id)
                    patterns_retrieved += 1

                    results.append({
                        "pattern_id": pattern_id,
                        "pattern": pattern,
                        "match_score": score,
                        "match_type": "hierarchical",
                        "hierarchy_details": {
                            "cluster_id": cluster_id,
                            "cluster_score": cluster_score,
                            "cluster_rank": sorted_clusters.index((cluster_id, cluster_score)) + 1,
                            "total_clusters": len(sorted_clusters),
                            "patterns_in_cluster": len(cluster_patterns)
                        }
                    })

        return results

    def _build_pattern_clusters(self) -> Dict[str, List[Tuple[str, VectorizedPattern]]]:
        """
        Build clusters of related patterns.

        Clusters by:
        - Vector space type (primary clustering)
        - Semantic density range (secondary)
        - Retrieval key similarity (tertiary)
        """
        clusters: Dict[str, List[Tuple[str, VectorizedPattern]]] = defaultdict(list)

        for pattern_id, pattern in self.pattern_vectors.items():
            # Primary: vector space
            space_key = pattern.vector_space.value

            # Secondary: semantic density tier
            if pattern.semantic_density >= 0.7:
                density_tier = "high"
            elif pattern.semantic_density >= 0.4:
                density_tier = "medium"
            else:
                density_tier = "low"

            # Combine into cluster key
            cluster_key = f"{space_key}_{density_tier}"
            clusters[cluster_key].append((pattern_id, pattern))

        # Also create cross-cutting clusters for high-value patterns
        high_value_patterns = [
            (pid, p) for pid, p in self.pattern_vectors.items()
            if p.usage_frequency > 5 or p.semantic_density > 0.8
        ]
        if high_value_patterns:
            clusters["_high_value"] = high_value_patterns

        return clusters

    def _score_cluster_relevance(self, cluster_patterns: List[Tuple[str, VectorizedPattern]],
                                 query: str, query_vector: List[float]) -> float:
        """
        Score how relevant a cluster is to the query.
        """
        if not cluster_patterns:
            return 0.0

        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Aggregate cluster metrics
        total_keyword_matches = 0
        total_vector_similarity = 0.0
        total_semantic_density = 0.0

        for pattern_id, pattern in cluster_patterns:
            # Keyword matches
            pattern_words = set()
            for key in pattern.retrieval_keys:
                pattern_words.update(key.lower().split())
            total_keyword_matches += len(query_words & pattern_words)

            # Vector similarity
            if pattern.algorithmic_vector and query_vector:
                sim = _compute_cosine_similarity(query_vector, pattern.algorithmic_vector)
                total_vector_similarity += (sim + 1) / 2  # Normalize to 0-1

            # Semantic density
            total_semantic_density += pattern.semantic_density

        n = len(cluster_patterns)

        # Compute averages
        avg_keyword_match = total_keyword_matches / (n * max(1, len(query_words)))
        avg_vector_sim = total_vector_similarity / n if query_vector else 0
        avg_density = total_semantic_density / n

        # Combined cluster score
        cluster_score = (
            avg_keyword_match * 0.4 +
            avg_vector_sim * 0.35 +
            avg_density * 0.25
        )

        return min(1.0, cluster_score)

    def _score_pattern_hierarchical(self, pattern: VectorizedPattern, query: str,
                                    query_vector: List[float], cluster_score: float) -> float:
        """
        Score a pattern within hierarchical context.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Direct keyword match
        pattern_words = set()
        for key in pattern.retrieval_keys:
            pattern_words.update(key.lower().split())

        keyword_score = len(query_words & pattern_words) / max(1, len(query_words))

        # Vector similarity
        vector_score = 0.0
        if pattern.algorithmic_vector and query_vector:
            sim = _compute_cosine_similarity(query_vector, pattern.algorithmic_vector)
            vector_score = (sim + 1) / 2

        # Quality factors
        quality_score = (
            pattern.semantic_density * 0.5 +
            min(1.0, pattern.usage_frequency / 10) * 0.3 +
            (1.0 if pattern.performance_metrics else 0.0) * 0.2
        )

        # Combine with cluster context
        individual_score = (
            keyword_score * 0.35 +
            vector_score * 0.30 +
            quality_score * 0.20 +
            cluster_score * 0.15  # Inherited from cluster
        )

        return min(1.0, individual_score)
    
    def _adaptive_retrieval(self, query_vector: List[float], query: str,
                           context_length: int, max_results: int) -> List[Dict]:
        """
        Perform adaptive retrieval with intelligent strategy selection.

        Analyzes query characteristics to determine optimal strategy mix:
        - Short queries → favor exact + fuzzy
        - Question queries → favor hierarchical + contextual
        - Technical queries → favor pattern-based + exact
        - Exploratory queries → favor fuzzy + contextual

        Dynamically weights and combines results from multiple strategies.
        """
        # Phase 1: Analyze query characteristics
        query_analysis = self._analyze_query_characteristics(query, query_vector)

        # Phase 2: Determine strategy weights based on query type
        strategy_weights = self._determine_strategy_weights(query_analysis)

        # Phase 3: Execute strategies with weighted allocations
        strategy_results: Dict[str, List[Dict]] = {}

        # Calculate result allocation based on weights
        total_weight = sum(strategy_weights.values())
        allocations = {
            strategy: max(1, int((weight / total_weight) * max_results * 1.5))
            for strategy, weight in strategy_weights.items()
        }

        # Execute each strategy
        if strategy_weights.get('exact', 0) > 0:
            strategy_results['exact'] = self._exact_match_retrieval(
                query_vector, query, context_length, allocations.get('exact', 5)
            )

        if strategy_weights.get('fuzzy', 0) > 0:
            strategy_results['fuzzy'] = self._fuzzy_search_retrieval(
                query_vector, query, context_length, allocations.get('fuzzy', 5)
            )

        if strategy_weights.get('pattern', 0) > 0:
            strategy_results['pattern'] = self._pattern_based_retrieval(
                query_vector, query, context_length, allocations.get('pattern', 5)
            )

        if strategy_weights.get('contextual', 0) > 0:
            strategy_results['contextual'] = self._contextual_retrieval(
                query_vector, query, context_length, allocations.get('contextual', 5)
            )

        if strategy_weights.get('hierarchical', 0) > 0:
            strategy_results['hierarchical'] = self._hierarchical_retrieval(
                query_vector, query, context_length, allocations.get('hierarchical', 5)
            )

        # Phase 4: Merge and score results
        merged_results = self._merge_adaptive_results(
            strategy_results, strategy_weights, query_analysis
        )

        # Phase 5: Final ranking and selection
        merged_results.sort(key=lambda x: x['adaptive_score'], reverse=True)

        return merged_results[:max_results]

    def _analyze_query_characteristics(self, query: str, query_vector: List[float]) -> Dict[str, Any]:
        """
        Analyze query to determine its characteristics for strategy selection.
        """
        query_lower = query.lower()
        words = query_lower.split()

        analysis = {
            "length": len(query),
            "word_count": len(words),
            "is_question": query.strip().endswith('?') or any(
                query_lower.startswith(w) for w in ['what', 'who', 'where', 'when', 'why', 'how']
            ),
            "is_short": len(words) <= 3,
            "is_long": len(words) >= 10,
            "has_technical_terms": False,
            "has_proper_nouns": False,
            "is_exploratory": False,
            "specificity": "medium"
        }

        # Check for technical terms
        technical_patterns = [
            'algorithm', 'function', 'method', 'api', 'implementation', 'code',
            'system', 'database', 'query', 'model', 'process', 'architecture'
        ]
        analysis["has_technical_terms"] = any(term in query_lower for term in technical_patterns)

        # Check for proper nouns (simplified - look for capitalized words)
        proper_nouns = [w for w in query.split() if w and w[0].isupper() and len(w) > 1]
        analysis["has_proper_nouns"] = len(proper_nouns) > 0

        # Check for exploratory queries
        exploratory_patterns = ['about', 'related', 'similar', 'like', 'explore', 'find', 'search']
        analysis["is_exploratory"] = any(p in query_lower for p in exploratory_patterns)

        # Determine specificity
        if analysis["has_proper_nouns"] or len([w for w in words if len(w) > 8]) >= 2:
            analysis["specificity"] = "high"
        elif analysis["is_short"] or analysis["is_exploratory"]:
            analysis["specificity"] = "low"

        # Add vector density analysis
        if query_vector:
            non_zero = sum(1 for v in query_vector if abs(v) > 0.01)
            analysis["vector_density"] = non_zero / len(query_vector)
        else:
            analysis["vector_density"] = 0.0

        return analysis

    def _determine_strategy_weights(self, query_analysis: Dict[str, Any]) -> Dict[str, float]:
        """
        Determine optimal strategy weights based on query analysis.
        """
        weights = {
            'exact': 0.2,
            'fuzzy': 0.2,
            'pattern': 0.2,
            'contextual': 0.2,
            'hierarchical': 0.2
        }

        # Adjust for short queries - exact and fuzzy more effective
        if query_analysis["is_short"]:
            weights['exact'] = 0.35
            weights['fuzzy'] = 0.30
            weights['hierarchical'] = 0.15
            weights['contextual'] = 0.10
            weights['pattern'] = 0.10

        # Adjust for questions - hierarchical and contextual better
        elif query_analysis["is_question"]:
            weights['hierarchical'] = 0.30
            weights['contextual'] = 0.25
            weights['pattern'] = 0.20
            weights['fuzzy'] = 0.15
            weights['exact'] = 0.10

        # Adjust for technical queries - pattern and exact better
        elif query_analysis["has_technical_terms"]:
            weights['pattern'] = 0.30
            weights['exact'] = 0.25
            weights['hierarchical'] = 0.20
            weights['fuzzy'] = 0.15
            weights['contextual'] = 0.10

        # Adjust for exploratory queries - fuzzy and contextual better
        elif query_analysis["is_exploratory"]:
            weights['fuzzy'] = 0.30
            weights['contextual'] = 0.30
            weights['hierarchical'] = 0.20
            weights['pattern'] = 0.10
            weights['exact'] = 0.10

        # Adjust for high specificity - exact more important
        if query_analysis["specificity"] == "high":
            weights['exact'] = min(0.4, weights['exact'] * 1.5)

        # Adjust for low specificity - broader search
        elif query_analysis["specificity"] == "low":
            weights['fuzzy'] = min(0.4, weights['fuzzy'] * 1.3)
            weights['contextual'] = min(0.35, weights['contextual'] * 1.2)

        # Normalize weights
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _merge_adaptive_results(self, strategy_results: Dict[str, List[Dict]],
                                strategy_weights: Dict[str, float],
                                query_analysis: Dict[str, Any]) -> List[Dict]:
        """
        Merge results from multiple strategies with intelligent scoring.
        """
        # Map strategy keys to match result types
        strategy_name_map = {
            'exact': 'exact',
            'fuzzy': 'fuzzy',
            'pattern': 'pattern_based',
            'contextual': 'contextual',
            'hierarchical': 'hierarchical'
        }

        # Collect all results with source tracking
        all_results: Dict[str, Dict] = {}

        for strategy_key, results in strategy_results.items():
            strategy_weight = strategy_weights.get(strategy_key, 0.2)

            for result in results:
                pattern_id = result.get("pattern_id")

                if pattern_id not in all_results:
                    all_results[pattern_id] = {
                        **result,
                        "contributing_strategies": [],
                        "strategy_scores": {},
                        "adaptive_score": 0.0
                    }

                # Track this strategy's contribution
                all_results[pattern_id]["contributing_strategies"].append(strategy_key)
                all_results[pattern_id]["strategy_scores"][strategy_key] = result.get("match_score", 0)

        # Calculate adaptive score for each result
        for pattern_id, result in all_results.items():
            strategy_scores = result["strategy_scores"]

            # Weighted score based on strategy contributions
            weighted_sum = sum(
                strategy_weights.get(strategy, 0.2) * score
                for strategy, score in strategy_scores.items()
            )

            # Bonus for appearing in multiple strategies (cross-validation)
            strategy_count = len(result["contributing_strategies"])
            diversity_bonus = min(0.2, (strategy_count - 1) * 0.05)

            # Bonus for high scores across strategies
            if strategy_scores:
                avg_score = sum(strategy_scores.values()) / len(strategy_scores)
                consistency_bonus = 0.1 if avg_score > 0.5 else 0.0
            else:
                consistency_bonus = 0.0

            # Final adaptive score
            result["adaptive_score"] = min(1.0, weighted_sum + diversity_bonus + consistency_bonus)
            result["match_type"] = "adaptive"
            result["adaptive_details"] = {
                "strategy_count": strategy_count,
                "strategies_used": result["contributing_strategies"],
                "diversity_bonus": diversity_bonus,
                "consistency_bonus": consistency_bonus,
                "query_type": (
                    "question" if query_analysis["is_question"] else
                    "technical" if query_analysis["has_technical_terms"] else
                    "exploratory" if query_analysis["is_exploratory"] else
                    "general"
                )
            }

        return list(all_results.values())
    
    def _calculate_contextual_relevance(self, pattern: VectorizedPattern, query: str, window: 'ContextWindow') -> float:
        """Calculate contextual relevance score."""
        # Simple contextual scoring based on window efficiency and pattern usage
        base_score = 0.0

        # Query keyword match in retrieval keys
        query_words = set(query.lower().split())
        pattern_words = set()
        for key in pattern.retrieval_keys:
            pattern_words.update(key.lower().split())

        keyword_match = len(query_words.intersection(pattern_words)) / max(1, len(query_words))

        # Window context boost
        window_boost = window.retrieval_efficiency * 0.3

        # Pattern quality boost
        quality_boost = pattern.semantic_density * 0.4

        return keyword_match + window_boost + quality_boost

    # ==================== Learning Pathway Methods ====================

    def _vectorize_learning_goal(self, goal: str, user_profile: Dict[str, Any]) -> List[float]:
        """
        Vectorize a learning goal considering user profile context.

        Generates a goal vector that captures:
        - Semantic content of the goal
        - User's current knowledge level
        - Learning style preferences
        """
        # Base semantic vector from goal text
        base_vector = self._generate_semantic_vector(goal, user_profile)

        # Adjust based on user profile
        if user_profile:
            # Knowledge level adjustment (beginners need more foundational vectors)
            knowledge_level = user_profile.get('knowledge_level', 0.5)
            if knowledge_level < 0.3:
                # Emphasize foundational concepts
                for i in range(min(100, len(base_vector))):
                    base_vector[i] *= 1.2

            # Learning style adjustment
            learning_style = user_profile.get('learning_style', 'balanced')
            if learning_style == 'visual':
                # Boost conceptual dimensions
                for i in range(100, min(200, len(base_vector))):
                    base_vector[i] *= 1.1
            elif learning_style == 'practical':
                # Boost procedural dimensions
                for i in range(200, min(300, len(base_vector))):
                    base_vector[i] *= 1.1

        # Normalize
        norm = np.linalg.norm(base_vector)
        if norm > 0:
            base_vector = [v / norm for v in base_vector]

        return base_vector

    def _find_matching_knowledge_patterns(self, goal_vector: List[float],
                                          user_profile: Dict[str, Any]) -> List[Dict]:
        """
        Find knowledge patterns that match a learning goal.

        Considers:
        - Vector similarity to goal
        - Appropriate difficulty for user level
        - Coverage of knowledge gaps
        """
        matches = []
        user_level = user_profile.get('knowledge_level', 0.5) if user_profile else 0.5

        for pattern_id, pattern in self.pattern_vectors.items():
            # Vector similarity
            if pattern.algorithmic_vector:
                similarity = _compute_cosine_similarity(goal_vector, pattern.algorithmic_vector)
                normalized_sim = (similarity + 1) / 2  # 0-1 scale
            else:
                normalized_sim = 0.0

            # Difficulty appropriateness (patterns with density close to user level)
            difficulty_diff = abs(pattern.semantic_density - user_level)
            difficulty_score = max(0, 1.0 - difficulty_diff)

            # Prefer patterns slightly above user level for growth
            if pattern.semantic_density > user_level and pattern.semantic_density < user_level + 0.3:
                difficulty_score *= 1.2

            # Combined score
            match_score = normalized_sim * 0.6 + difficulty_score * 0.4

            if match_score > 0.3:
                matches.append({
                    "pattern_id": pattern_id,
                    "pattern": pattern,
                    "match_score": match_score,
                    "similarity": normalized_sim,
                    "difficulty_score": difficulty_score,
                    "estimated_difficulty": pattern.semantic_density
                })

        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        return matches

    def _create_progressive_sequence(self, matching_patterns: List[Dict],
                                     user_profile: Dict[str, Any]) -> List[Dict]:
        """
        Create a progressive learning sequence from matched patterns.

        Orders patterns to:
        - Start with foundational concepts
        - Build complexity gradually
        - Respect prerequisite relationships
        """
        if not matching_patterns:
            return []

        user_level = user_profile.get('knowledge_level', 0.5) if user_profile else 0.5

        # Group patterns by difficulty tier
        tiers = {
            'foundation': [],
            'intermediate': [],
            'advanced': []
        }

        for match in matching_patterns:
            difficulty = match.get('estimated_difficulty', 0.5)
            if difficulty < 0.4:
                tiers['foundation'].append(match)
            elif difficulty < 0.7:
                tiers['intermediate'].append(match)
            else:
                tiers['advanced'].append(match)

        # Build sequence: foundation → intermediate → advanced
        sequence = []
        sequence_position = 0

        for tier_name in ['foundation', 'intermediate', 'advanced']:
            tier_patterns = tiers[tier_name]

            # Sort within tier by match score
            tier_patterns.sort(key=lambda x: x['match_score'], reverse=True)

            for pattern in tier_patterns:
                sequence_position += 1
                sequence.append({
                    **pattern,
                    "sequence_position": sequence_position,
                    "tier": tier_name,
                    "estimated_time_minutes": self._estimate_pattern_time(pattern),
                    "prerequisites": self._identify_prerequisites(pattern, sequence[:-1])
                })

        return sequence

    def _optimize_for_learning_style(self, sequence: List[Dict],
                                     user_profile: Dict[str, Any]) -> List[Dict]:
        """
        Optimize learning sequence for user's learning style.

        Adjustments:
        - Visual learners: prioritize conceptual patterns
        - Practical learners: prioritize procedural patterns
        - Reading learners: prioritize detailed text patterns
        - Interactive learners: interleave different types
        """
        if not sequence or not user_profile:
            return sequence

        learning_style = user_profile.get('learning_style', 'balanced')

        if learning_style == 'visual':
            # Prioritize patterns with conceptual or relational vector spaces
            sequence.sort(
                key=lambda x: (
                    x['tier'],  # Keep tier ordering
                    -1 if x['pattern'].vector_space in [VectorSpaceType.CONCEPTUAL, VectorSpaceType.RELATIONAL] else 0,
                    -x['match_score']
                )
            )

        elif learning_style == 'practical':
            # Prioritize patterns with higher usage frequency (proven practical value)
            sequence.sort(
                key=lambda x: (
                    x['tier'],
                    -x['pattern'].usage_frequency,
                    -x['match_score']
                )
            )

        elif learning_style == 'interactive':
            # Interleave different vector spaces for variety
            by_space = defaultdict(list)
            for item in sequence:
                by_space[item['pattern'].vector_space.value].append(item)

            interleaved = []
            max_len = max(len(v) for v in by_space.values()) if by_space else 0
            for i in range(max_len):
                for space_items in by_space.values():
                    if i < len(space_items):
                        interleaved.append(space_items[i])

            # Update sequence positions
            for i, item in enumerate(interleaved):
                item['sequence_position'] = i + 1
            sequence = interleaved

        # Re-number sequence positions
        for i, item in enumerate(sequence):
            item['sequence_position'] = i + 1
            item['optimized_for'] = learning_style

        return sequence

    def _estimate_learning_duration(self, sequence: List[Dict],
                                    user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate total learning duration for a sequence.
        """
        if not sequence:
            return {"total_minutes": 0, "breakdown": {}}

        # Base time per pattern based on complexity
        total_minutes = 0
        breakdown = {'foundation': 0, 'intermediate': 0, 'advanced': 0}

        user_speed = user_profile.get('learning_speed', 1.0) if user_profile else 1.0

        for item in sequence:
            base_time = self._estimate_pattern_time(item)
            adjusted_time = base_time / user_speed  # Faster learners need less time

            total_minutes += adjusted_time
            tier = item.get('tier', 'intermediate')
            breakdown[tier] = breakdown.get(tier, 0) + adjusted_time

        return {
            "total_minutes": round(total_minutes),
            "total_hours": round(total_minutes / 60, 1),
            "breakdown": breakdown,
            "estimated_sessions": max(1, int(total_minutes / 45))  # 45-min sessions
        }

    def _calculate_difficulty_progression(self, sequence: List[Dict]) -> Dict[str, Any]:
        """
        Calculate how difficulty progresses through the sequence.
        """
        if not sequence:
            return {"progression_type": "empty", "scores": []}

        difficulties = [item.get('estimated_difficulty', 0.5) for item in sequence]

        # Calculate progression metrics
        if len(difficulties) >= 2:
            # Check if generally increasing
            increases = sum(1 for i in range(len(difficulties)-1) if difficulties[i+1] > difficulties[i])
            increase_ratio = increases / (len(difficulties) - 1)

            if increase_ratio >= 0.7:
                progression_type = "increasing"
            elif increase_ratio <= 0.3:
                progression_type = "decreasing"
            else:
                progression_type = "mixed"

            # Smoothness (lower variance is smoother)
            if len(difficulties) > 1:
                variance = statistics.variance(difficulties)
                smoothness = max(0, 1 - variance * 4)
            else:
                smoothness = 1.0
        else:
            progression_type = "single"
            smoothness = 1.0

        return {
            "progression_type": progression_type,
            "starting_difficulty": difficulties[0] if difficulties else 0,
            "ending_difficulty": difficulties[-1] if difficulties else 0,
            "average_difficulty": statistics.mean(difficulties) if difficulties else 0,
            "smoothness": round(smoothness, 2),
            "scores": difficulties
        }

    def _identify_knowledge_gaps(self, sequence: List[Dict],
                                 user_profile: Dict[str, Any]) -> List[Dict]:
        """
        Identify knowledge gaps based on sequence and user profile.
        """
        gaps = []

        if not user_profile:
            return gaps

        user_level = user_profile.get('knowledge_level', 0.5)
        user_strengths = set(user_profile.get('strengths', []))

        # Identify patterns that are significantly above user level
        for item in sequence:
            pattern = item.get('pattern')
            if not pattern:
                continue

            difficulty = item.get('estimated_difficulty', 0.5)

            # Gap if pattern is much harder than user level
            if difficulty > user_level + 0.3:
                # Check if user has related strengths
                pattern_words = set(" ".join(pattern.retrieval_keys).lower().split())
                overlap = pattern_words & user_strengths

                if not overlap:
                    gaps.append({
                        "pattern_id": item['pattern_id'],
                        "gap_type": "difficulty_gap",
                        "difficulty_delta": round(difficulty - user_level, 2),
                        "related_terms": list(pattern_words)[:5],
                        "recommendation": "Consider foundational material first"
                    })

            # Check for vector space coverage gaps
            space = pattern.vector_space.value
            if space not in user_profile.get('familiar_spaces', []):
                gaps.append({
                    "pattern_id": item['pattern_id'],
                    "gap_type": "space_gap",
                    "unfamiliar_space": space,
                    "recommendation": f"This requires {space} thinking skills"
                })

        return gaps

    def _estimate_pattern_time(self, pattern_match: Dict) -> float:
        """
        Estimate time in minutes to learn a pattern.
        """
        pattern = pattern_match.get('pattern')
        if not pattern:
            return 10.0  # Default 10 minutes

        # Base time by complexity
        base_time = 10.0

        # Adjust by semantic density (more dense = more time)
        density_factor = 1 + pattern.semantic_density
        base_time *= density_factor

        # Adjust by context window size
        if pattern.context_window > 1000:
            base_time *= 1.3
        elif pattern.context_window > 500:
            base_time *= 1.1

        # Adjust by vector space type (some are harder)
        space_factors = {
            VectorSpaceType.SEMANTIC: 1.0,
            VectorSpaceType.SYNTACTIC: 0.8,
            VectorSpaceType.CONCEPTUAL: 1.2,
            VectorSpaceType.TEMPORAL: 1.1,
            VectorSpaceType.RELATIONAL: 1.3
        }
        base_time *= space_factors.get(pattern.vector_space, 1.0)

        return round(base_time, 1)

    def _identify_prerequisites(self, current: Dict, preceding: List[Dict]) -> List[str]:
        """
        Identify prerequisite patterns from preceding sequence items.
        """
        prerequisites = []
        current_pattern = current.get('pattern')

        if not current_pattern or not preceding:
            return prerequisites

        current_words = set()
        for key in current_pattern.retrieval_keys:
            current_words.update(key.lower().split())

        # Find preceding patterns that share vocabulary
        for prev in preceding:
            prev_pattern = prev.get('pattern')
            if not prev_pattern:
                continue

            prev_words = set()
            for key in prev_pattern.retrieval_keys:
                prev_words.update(key.lower().split())

            # If significant overlap and previous is easier, it's a prerequisite
            overlap = len(current_words & prev_words)
            if overlap >= 2 and prev.get('estimated_difficulty', 0.5) < current.get('estimated_difficulty', 0.5):
                prerequisites.append(prev.get('pattern_id', 'unknown'))

        return prerequisites[:3]  # Max 3 prerequisites

    # ==================== Optimization Helper Methods ====================

    def _optimize_pattern_arrangement(self, patterns: List[VectorizedPattern]) -> List[VectorizedPattern]:
        """
        Optimize pattern arrangement in a context window for better retrieval.

        Strategies:
        - Group by vector space
        - Order by semantic density (high to low)
        - Cluster related patterns together
        """
        if not patterns:
            return patterns

        # Group by vector space
        by_space: Dict[VectorSpaceType, List[VectorizedPattern]] = defaultdict(list)
        for pattern in patterns:
            by_space[pattern.vector_space].append(pattern)

        # Sort within each space by semantic density (descending)
        optimized = []
        for space in VectorSpaceType:
            space_patterns = by_space.get(space, [])
            space_patterns.sort(key=lambda p: p.semantic_density, reverse=True)
            optimized.extend(space_patterns)

        return optimized

    def _calculate_window_efficiency(self, patterns: List[VectorizedPattern]) -> float:
        """
        Calculate efficiency score for a context window.
        """
        if not patterns:
            return 0.0

        # Average semantic density
        avg_density = sum(p.semantic_density for p in patterns) / len(patterns)

        # Usage frequency factor
        avg_usage = sum(p.usage_frequency for p in patterns) / len(patterns)
        usage_factor = min(1.0, avg_usage / 10)

        # Diversity factor (different vector spaces is good)
        unique_spaces = len(set(p.vector_space for p in patterns))
        diversity_factor = min(1.0, unique_spaces / len(VectorSpaceType))

        return (avg_density * 0.4 + usage_factor * 0.3 + diversity_factor * 0.3)

    def _calculate_compression_ratio(self, patterns: List[VectorizedPattern]) -> float:
        """
        Calculate compression ratio achieved by pattern organization.
        """
        if not patterns:
            return 0.0

        # Total context window sizes
        total_size = sum(p.context_window for p in patterns)

        # Estimated compressed size (based on semantic density overlap)
        # Higher density patterns compress better
        avg_density = sum(p.semantic_density for p in patterns) / len(patterns)

        # Compression improves with density (more meaning per token)
        compression = avg_density * 0.5

        return min(1.0, compression)
