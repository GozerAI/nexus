
"""
Pattern Recognition Engine for Nexus AI Platform.
Identifies and stores algorithmic patterns for efficient knowledge retrieval.
"""

import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class PatternType(Enum):
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"
    TEMPORAL = "temporal"
    LOGICAL = "logical"
    CAUSAL = "causal"

@dataclass
class KnowledgePattern:
    """Represents a recognized pattern in knowledge."""
    pattern_id: str
    pattern_type: PatternType
    signature: str  # Unique signature for the pattern
    retrieval_keys: List[str]  # Keys to locate similar knowledge
    abstraction_level: int  # 1=specific, 5=highly abstract
    confidence: float
    usage_count: int
    created_at: datetime
    last_used: datetime
    context_domains: Set[str]
    retrieval_algorithm: str  # Algorithm to find related knowledge

class PatternRecognitionEngine:
    """
    Recognizes patterns in knowledge and creates efficient retrieval signatures.
    Instead of storing all data, stores patterns that know where to find answers.
    """
    
    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base
        self.patterns: Dict[str, KnowledgePattern] = {}
        self.pattern_index: Dict[str, List[str]] = {}  # signature -> pattern_ids
        self.domain_patterns: Dict[str, Set[str]] = {}  # domain -> pattern_ids
        self.retrieval_strategies: Dict[str, callable] = {}
        self.max_patterns = 50000  # Limit to control memory usage
        self.initialized = False
        
    def initialize(self):
        """Initialize pattern recognition engine."""
        if self.initialized:
            return
        
        self._setup_retrieval_strategies()
        self._load_core_patterns()
        
        self.initialized = True
        logger.info(f"Pattern Recognition Engine initialized with {len(self.patterns)} patterns")
    
    def recognize_pattern(self, content: Any, context: Dict[str, Any] = None) -> Optional[KnowledgePattern]:
        """Recognize patterns in new knowledge and create retrieval signatures."""
        try:
            # Extract features from content
            features = self._extract_features(content, context)
            
            # Generate pattern signature
            signature = self._generate_signature(features)
            
            # Check if pattern already exists
            existing_pattern = self._find_existing_pattern(signature)
            if existing_pattern:
                existing_pattern.usage_count += 1
                existing_pattern.last_used = datetime.now()
                return existing_pattern
            
            # Create new pattern
            pattern = self._create_new_pattern(signature, features, content, context)
            self._store_pattern(pattern)
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error recognizing pattern: {e}")
            return None
    
    def find_knowledge_by_pattern(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Find knowledge using pattern recognition instead of storing everything."""
        query_features = self._extract_features(query, {})
        query_signature = self._generate_signature(query_features)
        
        # Find matching patterns
        matching_patterns = self._find_matching_patterns(query_signature, max_results)
        
        results = []
        for pattern in matching_patterns:
            # Use pattern's retrieval algorithm to find actual knowledge
            knowledge_items = self._execute_retrieval_strategy(pattern, query)
            results.extend(knowledge_items)
        
        return results[:max_results]
    
    def create_smart_index(self, data_source: str, metadata: Dict[str, Any]) -> str:
        """Create a smart index that knows how to retrieve from a data source without storing it."""
        index_id = f"smart_index_{int(datetime.now().timestamp())}"
        
        # Analyze data source to understand its structure
        source_patterns = self._analyze_data_source(data_source, metadata)
        
        # Create retrieval patterns for this source
        for pattern_info in source_patterns:
            pattern = KnowledgePattern(
                pattern_id=f"{index_id}_{pattern_info['type']}",
                pattern_type=PatternType.STRUCTURAL,
                signature=pattern_info['signature'],
                retrieval_keys=pattern_info['keys'],
                abstraction_level=pattern_info['abstraction'],
                confidence=0.8,
                usage_count=0,
                created_at=datetime.now(),
                last_used=datetime.now(),
                context_domains=set(pattern_info['domains']),
                retrieval_algorithm=f"source_retrieval_{pattern_info['type']}"
            )
            self._store_pattern(pattern)
        
        logger.info(f"Created smart index {index_id} with {len(source_patterns)} patterns")
        return index_id
    
    def _extract_features(self, content: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features from content for pattern recognition."""
        features = {
            'content_type': type(content).__name__,
            'length': len(str(content)),
            'domains': [],
            'entities': [],
            'concepts': [],
            'relationships': []
        }
        
        content_str = str(content).lower()
        
        # Domain detection
        domain_keywords = {
            'science': ['research', 'study', 'experiment', 'hypothesis', 'theory'],
            'technology': ['system', 'algorithm', 'software', 'hardware', 'data'],
            'mathematics': ['equation', 'formula', 'calculate', 'theorem', 'proof'],
            'business': ['market', 'company', 'revenue', 'strategy', 'customer'],
            'history': ['century', 'war', 'empire', 'ancient', 'historical']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_str for keyword in keywords):
                features['domains'].append(domain)
        
        # Entity extraction (simplified)
        if any(char.isupper() for char in str(content)):
            features['entities'] = [word for word in str(content).split() if word[0].isupper()][:5]
        
        # Concept extraction (key terms)
        words = content_str.split()
        features['concepts'] = [word for word in words if len(word) > 5][:10]
        
        return features
    
    def _generate_signature(self, features: Dict[str, Any]) -> str:
        """Generate a unique signature for the pattern."""
        signature_data = {
            'type': features['content_type'],
            'domains': sorted(features['domains']),
            'concepts': sorted(features['concepts'][:5]),  # Top 5 concepts
            'length_category': 'short' if features['length'] < 100 else 'medium' if features['length'] < 500 else 'long'
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:16]
    
    def _find_existing_pattern(self, signature: str) -> Optional[KnowledgePattern]:
        """Find existing pattern by signature."""
        if signature in self.pattern_index:
            pattern_ids = self.pattern_index[signature]
            if pattern_ids:
                return self.patterns.get(pattern_ids[0])
        return None
    
    def _create_new_pattern(self, signature: str, features: Dict[str, Any], 
                           content: Any, context: Dict[str, Any]) -> KnowledgePattern:
        """Create a new knowledge pattern."""
        pattern_id = f"pattern_{len(self.patterns)}_{signature[:8]}"
        
        # Determine retrieval strategy based on features
        retrieval_algorithm = self._select_retrieval_algorithm(features)
        
        # Generate retrieval keys
        retrieval_keys = features['concepts'][:5] + features['domains']
        
        return KnowledgePattern(
            pattern_id=pattern_id,
            pattern_type=self._classify_pattern_type(features),
            signature=signature,
            retrieval_keys=retrieval_keys,
            abstraction_level=self._calculate_abstraction_level(features),
            confidence=0.7,
            usage_count=1,
            created_at=datetime.now(),
            last_used=datetime.now(),
            context_domains=set(features['domains']),
            retrieval_algorithm=retrieval_algorithm
        )
    
    def _store_pattern(self, pattern: KnowledgePattern):
        """Store pattern in the recognition system."""
        # Check storage limits
        if len(self.patterns) >= self.max_patterns:
            self._cleanup_old_patterns()
        
        self.patterns[pattern.pattern_id] = pattern
        
        # Update indices
        if pattern.signature not in self.pattern_index:
            self.pattern_index[pattern.signature] = []
        self.pattern_index[pattern.signature].append(pattern.pattern_id)
        
        # Update domain index
        for domain in pattern.context_domains:
            if domain not in self.domain_patterns:
                self.domain_patterns[domain] = set()
            self.domain_patterns[domain].add(pattern.pattern_id)
    
    def _setup_retrieval_strategies(self):
        """Setup different strategies for knowledge retrieval."""
        self.retrieval_strategies = {
            'semantic_search': self._semantic_search_strategy,
            'structural_search': self._structural_search_strategy,
            'domain_search': self._domain_search_strategy,
            'concept_search': self._concept_search_strategy,
            'source_retrieval_text': self._source_text_retrieval,
            'source_retrieval_structured': self._source_structured_retrieval
        }
    
    def _semantic_search_strategy(self, pattern: KnowledgePattern, query: str) -> List[Dict[str, Any]]:
        """Search using semantic similarity."""
        results = []
        if self.knowledge_base:
            knowledge_items = self.knowledge_base.query_knowledge(query, max_results=5)
            for item in knowledge_items:
                results.append({
                    'content': item.content,
                    'confidence': item.confidence,
                    'source': 'knowledge_base',
                    'pattern_match': pattern.pattern_id
                })
        return results
    
    def _structural_search_strategy(self, pattern: KnowledgePattern, query: str) -> List[Dict[str, Any]]:
        """Search using structural patterns."""
        # Implement structural pattern matching
        return []
    
    def _domain_search_strategy(self, pattern: KnowledgePattern, query: str) -> List[Dict[str, Any]]:
        """Search within specific domains."""
        results = []
        for domain in pattern.context_domains:
            if self.knowledge_base:
                domain_items = self.knowledge_base.query_knowledge(f"{domain} {query}", max_results=3)
                for item in domain_items:
                    results.append({
                        'content': item.content,
                        'confidence': item.confidence * 0.9,  # Slight penalty for domain search
                        'source': f'domain_{domain}',
                        'pattern_match': pattern.pattern_id
                    })
        return results
    
    def _concept_search_strategy(self, pattern: KnowledgePattern, query: str) -> List[Dict[str, Any]]:
        """Search using concept matching."""
        results = []
        for key in pattern.retrieval_keys:
            if self.knowledge_base:
                concept_items = self.knowledge_base.query_knowledge(key, max_results=2)
                for item in concept_items:
                    results.append({
                        'content': item.content,
                        'confidence': item.confidence * 0.8,
                        'source': f'concept_{key}',
                        'pattern_match': pattern.pattern_id
                    })
        return results
    
    def _source_text_retrieval(self, pattern: KnowledgePattern, query: str) -> List[Dict[str, Any]]:
        """Retrieve from text-based external sources."""
        # Placeholder for external text source retrieval
        return []
    
    def _source_structured_retrieval(self, pattern: KnowledgePattern, query: str) -> List[Dict[str, Any]]:
        """Retrieve from structured external sources."""
        # Placeholder for external structured data retrieval
        return []
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about pattern recognition."""
        if not self.patterns:
            return {'total_patterns': 0}
        
        type_distribution = {}
        domain_distribution = {}
        usage_stats = []
        
        for pattern in self.patterns.values():
            # Type distribution
            ptype = pattern.pattern_type.value
            type_distribution[ptype] = type_distribution.get(ptype, 0) + 1
            
            # Domain distribution
            for domain in pattern.context_domains:
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1
            
            # Usage stats
            usage_stats.append(pattern.usage_count)
        
        return {
            'total_patterns': len(self.patterns),
            'type_distribution': type_distribution,
            'domain_distribution': domain_distribution,
            'avg_usage': np.mean(usage_stats) if usage_stats else 0,
            'most_used_patterns': sorted([p.pattern_id for p in self.patterns.values()], 
                                       key=lambda x: self.patterns[x].usage_count, reverse=True)[:5]
        }
    
    def _load_core_patterns(self):
        """Load essential patterns for common knowledge types."""
        core_patterns = [
            {
                'type': PatternType.SEMANTIC,
                'domains': ['general'],
                'concepts': ['definition', 'explanation'],
                'algorithm': 'semantic_search'
            },
            {
                'type': PatternType.STRUCTURAL,
                'domains': ['mathematics', 'science'],
                'concepts': ['formula', 'equation', 'proof'],
                'algorithm': 'concept_search'
            },
            {
                'type': PatternType.TEMPORAL,
                'domains': ['history', 'events'],
                'concepts': ['when', 'time', 'date'],
                'algorithm': 'domain_search'
            }
        ]
        
        for i, pattern_info in enumerate(core_patterns):
            features = {
                'content_type': 'core_pattern',
                'domains': pattern_info['domains'],
                'concepts': pattern_info['concepts'],
                'length': 100
            }
            
            signature = self._generate_signature(features)
            pattern = KnowledgePattern(
                pattern_id=f"core_pattern_{i}",
                pattern_type=pattern_info['type'],
                signature=signature,
                retrieval_keys=pattern_info['concepts'],
                abstraction_level=3,
                confidence=0.9,
                usage_count=0,
                created_at=datetime.now(),
                last_used=datetime.now(),
                context_domains=set(pattern_info['domains']),
                retrieval_algorithm=pattern_info['algorithm']
            )
            
            self._store_pattern(pattern)
    
    def _find_matching_patterns(self, query_signature: str, max_results: int) -> List[KnowledgePattern]:
        """Find patterns that match the query signature."""
        matches = []
        
        # Exact signature match
        if query_signature in self.pattern_index:
            for pattern_id in self.pattern_index[query_signature][:max_results]:
                if pattern_id in self.patterns:
                    matches.append(self.patterns[pattern_id])
        
        # If no exact matches, find similar patterns
        if not matches:
            all_patterns = list(self.patterns.values())
            # Sort by usage count and confidence
            all_patterns.sort(key=lambda x: (x.usage_count, x.confidence), reverse=True)
            matches = all_patterns[:max_results]
        
        return matches
    
    def _execute_retrieval_strategy(self, pattern: KnowledgePattern, query: str) -> List[Dict[str, Any]]:
        """Execute the retrieval strategy for a pattern."""
        strategy_func = self.retrieval_strategies.get(pattern.retrieval_algorithm)
        if strategy_func:
            return strategy_func(pattern, query)
        else:
            return self._semantic_search_strategy(pattern, query)
    
    def _analyze_data_source(self, data_source: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze a data source to understand its patterns."""
        patterns = []
        
        # Basic pattern for any data source
        patterns.append({
            'type': 'basic_access',
            'signature': f"source_{hashlib.md5(data_source.encode()).hexdigest()[:8]}",
            'keys': [data_source, metadata.get('type', 'unknown')],
            'abstraction': 2,
            'domains': [metadata.get('domain', 'general')]
        })
        
        return patterns
    
    def _classify_pattern_type(self, features: Dict[str, Any]) -> PatternType:
        """Classify the type of pattern based on features."""
        if 'mathematics' in features['domains'] or any('formula' in c for c in features['concepts']):
            return PatternType.LOGICAL
        elif 'history' in features['domains'] or any('time' in c for c in features['concepts']):
            return PatternType.TEMPORAL
        elif len(features['domains']) > 1:
            return PatternType.SEMANTIC
        else:
            return PatternType.STRUCTURAL
    
    def _calculate_abstraction_level(self, features: Dict[str, Any]) -> int:
        """Calculate abstraction level (1=specific, 5=highly abstract)."""
        if len(features['concepts']) > 8:
            return 1  # Very specific
        elif len(features['concepts']) > 5:
            return 2
        elif len(features['domains']) > 2:
            return 4  # More abstract
        else:
            return 3  # Medium abstraction
    
    def _select_retrieval_algorithm(self, features: Dict[str, Any]) -> str:
        """Select the best retrieval algorithm for the features."""
        if 'mathematics' in features['domains'] or 'science' in features['domains']:
            return 'concept_search'
        elif len(features['domains']) > 1:
            return 'semantic_search'
        else:
            return 'domain_search'
    
    def _cleanup_old_patterns(self):
        """Remove old, unused patterns to maintain memory limits."""
        # Sort by usage and age, remove least used old patterns
        patterns_by_usage = sorted(
            self.patterns.values(),
            key=lambda x: (x.usage_count, x.last_used),
            reverse=False
        )
        
        # Remove bottom 10%
        to_remove = patterns_by_usage[:len(patterns_by_usage) // 10]
        
        for pattern in to_remove:
            if pattern.pattern_id in self.patterns:
                del self.patterns[pattern.pattern_id]
            
            # Clean up indices
            if pattern.signature in self.pattern_index:
                if pattern.pattern_id in self.pattern_index[pattern.signature]:
                    self.pattern_index[pattern.signature].remove(pattern.pattern_id)
                    
            for domain in pattern.context_domains:
                if domain in self.domain_patterns:
                    self.domain_patterns[domain].discard(pattern.pattern_id)
        
        logger.info(f"Cleaned up {len(to_remove)} old patterns")
