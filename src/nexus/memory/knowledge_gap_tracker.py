
"""
Knowledge Gap Tracking System for Nexus AI Platform.
Tracks what the system doesn't know and triggers learning.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from nexus.memory.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

class GapType(Enum):
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    CONTEXTUAL = "contextual"
    DOMAIN_SPECIFIC = "domain_specific"

class GapPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class KnowledgeGap:
    """Represents a gap in the knowledge base."""
    id: str
    query: str
    gap_type: GapType
    priority: GapPriority
    domain: str
    frequency: int
    first_encountered: datetime
    last_encountered: datetime
    attempted_retrieval: bool = False
    retrieval_success: bool = False
    retrieval_attempts: int = 0
    related_queries: List[str] = None
    context: Dict[str, Any] = None

class KnowledgeGapTracker:
    """
    Tracks knowledge gaps and triggers automatic learning.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase, internet_retriever=None):
        self.knowledge_base = knowledge_base
        self.internet_retriever = internet_retriever
        self.gaps: Dict[str, KnowledgeGap] = {}
        self.query_history: List[Dict[str, Any]] = []
        self.gap_patterns: Dict[str, int] = {}
        self.auto_retrieval_threshold = 3  # Trigger retrieval after 3 encounters
        self.max_retrieval_attempts = 2
        
    def record_failed_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """Record a query that couldn't be answered."""
        gap_id = self._generate_gap_id(query)
        current_time = datetime.now()
        
        # Check if this gap already exists
        if gap_id in self.gaps:
            gap = self.gaps[gap_id]
            gap.frequency += 1
            gap.last_encountered = current_time
            
            # Add to related queries if different
            if gap.related_queries is None:
                gap.related_queries = []
            if query not in gap.related_queries and query != gap.query:
                gap.related_queries.append(query)
        else:
            # Create new gap
            gap_type = self._classify_gap_type(query, context)
            priority = self._calculate_priority(query, gap_type, context)
            domain = self._identify_domain(query)
            
            gap = KnowledgeGap(
                id=gap_id,
                query=query,
                gap_type=gap_type,
                priority=priority,
                domain=domain,
                frequency=1,
                first_encountered=current_time,
                last_encountered=current_time,
                related_queries=[],
                context=context or {}
            )
            self.gaps[gap_id] = gap
        
        # Record in query history
        self.query_history.append({
            'query': query,
            'gap_id': gap_id,
            'timestamp': current_time,
            'context': context
        })
        
        # Update patterns
        self._update_gap_patterns(query, gap.domain, gap.gap_type)
        
        # Check if we should trigger automatic retrieval
        if (gap.frequency >= self.auto_retrieval_threshold and 
            not gap.attempted_retrieval and 
            self.internet_retriever and
            gap.retrieval_attempts < self.max_retrieval_attempts):
            
            self._trigger_automatic_retrieval(gap)
        
        logger.info(f"Recorded knowledge gap: {gap_id} (frequency: {gap.frequency})")
        return gap_id
    
    def get_critical_gaps(self, limit: int = 10) -> List[KnowledgeGap]:
        """Get the most critical knowledge gaps that need attention."""
        gaps = list(self.gaps.values())
        
        # Sort by priority, frequency, and recency
        gaps.sort(key=lambda g: (
            g.priority.value == 'critical',
            g.priority.value == 'high',
            g.frequency,
            g.last_encountered
        ), reverse=True)
        
        return gaps[:limit]
    
    def get_gap_by_domain(self, domain: str) -> List[KnowledgeGap]:
        """Get all gaps in a specific domain."""
        return [gap for gap in self.gaps.values() if gap.domain == domain]
    
    def resolve_gap(self, gap_id: str, knowledge_items: List[str]) -> bool:
        """Mark a gap as resolved with the provided knowledge."""
        if gap_id not in self.gaps:
            return False
        
        gap = self.gaps[gap_id]
        gap.retrieval_success = True
        
        # Add metadata to indicate this gap was resolved
        gap.context['resolved'] = True
        gap.context['resolved_at'] = datetime.now().isoformat()
        gap.context['knowledge_items'] = knowledge_items
        
        logger.info(f"Resolved knowledge gap: {gap_id}")
        return True
    
    def _generate_gap_id(self, query: str) -> str:
        """Generate a unique ID for a knowledge gap."""
        import hashlib
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()[:8]
        return f"gap_{query_hash}"
    
    def _classify_gap_type(self, query: str, context: Dict[str, Any] = None) -> GapType:
        """Classify the type of knowledge gap."""
        query_lower = query.lower()
        
        # Factual questions
        if any(word in query_lower for word in ['what is', 'what are', 'who is', 'when', 'where']):
            return GapType.FACTUAL
        
        # Procedural questions
        elif any(word in query_lower for word in ['how to', 'how do', 'how can', 'process', 'steps']):
            return GapType.PROCEDURAL
        
        # Domain-specific indicators
        elif any(word in query_lower for word in ['technical', 'scientific', 'medical', 'legal', 'academic']):
            return GapType.DOMAIN_SPECIFIC
        
        # Default to contextual
        else:
            return GapType.CONTEXTUAL
    
    def _calculate_priority(self, query: str, gap_type: GapType, context: Dict[str, Any] = None) -> GapPriority:
        """Calculate the priority of addressing this gap."""
        priority_score = 0
        
        # Base priority by type
        type_priorities = {
            GapType.FACTUAL: 2,
            GapType.PROCEDURAL: 3,
            GapType.DOMAIN_SPECIFIC: 4,
            GapType.CONTEXTUAL: 1
        }
        priority_score += type_priorities.get(gap_type, 1)
        
        # Boost for urgent keywords
        urgent_keywords = ['critical', 'urgent', 'important', 'emergency', 'immediately']
        if any(keyword in query.lower() for keyword in urgent_keywords):
            priority_score += 3
        
        # Boost for complex queries
        if len(query.split()) > 10:
            priority_score += 1
        
        # Context-based boosting
        if context:
            if context.get('user_waiting', False):
                priority_score += 2
            if context.get('system_critical', False):
                priority_score += 4
        
        # Convert to priority enum
        if priority_score >= 7:
            return GapPriority.CRITICAL
        elif priority_score >= 5:
            return GapPriority.HIGH
        elif priority_score >= 3:
            return GapPriority.MEDIUM
        else:
            return GapPriority.LOW
    
    def _identify_domain(self, query: str) -> str:
        """Identify the domain/subject area of the query."""
        query_lower = query.lower()
        
        domain_keywords = {
            'science': ['science', 'biology', 'chemistry', 'physics', 'research', 'study', 'experiment'],
            'technology': ['computer', 'software', 'programming', 'algorithm', 'digital', 'tech'],
            'mathematics': ['math', 'calculation', 'equation', 'formula', 'number', 'statistics'],
            'history': ['history', 'historical', 'past', 'ancient', 'century', 'war', 'civilization'],
            'geography': ['geography', 'country', 'capital', 'continent', 'ocean', 'mountain', 'location'],
            'health': ['health', 'medical', 'medicine', 'disease', 'treatment', 'body', 'doctor'],
            'literature': ['literature', 'book', 'author', 'novel', 'poem', 'writing'],
            'arts': ['art', 'music', 'painting', 'artist', 'culture', 'creative'],
            'business': ['business', 'economy', 'finance', 'market', 'company', 'trade'],
            'law': ['law', 'legal', 'court', 'justice', 'rights', 'constitution']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return domain
        
        return 'general'
    
    def _update_gap_patterns(self, query: str, domain: str, gap_type: GapType):
        """Update patterns to identify common knowledge gaps."""
        # Track domain patterns
        domain_key = f"domain:{domain}"
        self.gap_patterns[domain_key] = self.gap_patterns.get(domain_key, 0) + 1
        
        # Track type patterns
        type_key = f"type:{gap_type.value}"
        self.gap_patterns[type_key] = self.gap_patterns.get(type_key, 0) + 1
        
        # Track query patterns (first 3 words)
        query_pattern = " ".join(query.lower().split()[:3])
        pattern_key = f"pattern:{query_pattern}"
        self.gap_patterns[pattern_key] = self.gap_patterns.get(pattern_key, 0) + 1
    
    def _trigger_automatic_retrieval(self, gap: KnowledgeGap):
        """Trigger automatic knowledge retrieval for a gap."""
        if not self.internet_retriever:
            logger.warning(f"Cannot retrieve knowledge for gap {gap.id}: no internet retriever available")
            return
        
        try:
            logger.info(f"Triggering automatic retrieval for gap: {gap.id}")
            gap.attempted_retrieval = True
            gap.retrieval_attempts += 1
            
            # Attempt to retrieve knowledge
            retrieval_result = self.internet_retriever.retrieve_knowledge_for_query(gap.query)
            
            if retrieval_result.get('success') and retrieval_result.get('added_to_kb'):
                gap.retrieval_success = True
                self.resolve_gap(gap.id, retrieval_result['added_to_kb'])
                logger.info(f"Successfully retrieved knowledge for gap: {gap.id}")
            else:
                logger.warning(f"Failed to retrieve knowledge for gap: {gap.id}")
                
        except Exception as e:
            logger.error(f"Error during automatic retrieval for gap {gap.id}: {e}")
    
    def get_gap_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge gaps."""
        if not self.gaps:
            return {"total_gaps": 0}
        
        by_type = {}
        by_priority = {}
        by_domain = {}
        resolved_count = 0
        
        for gap in self.gaps.values():
            # Count by type
            gap_type = gap.gap_type.value
            by_type[gap_type] = by_type.get(gap_type, 0) + 1
            
            # Count by priority
            priority = gap.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1
            
            # Count by domain
            domain = gap.domain
            by_domain[domain] = by_domain.get(domain, 0) + 1
            
            # Count resolved
            if gap.retrieval_success:
                resolved_count += 1
        
        return {
            "total_gaps": len(self.gaps),
            "resolved_gaps": resolved_count,
            "resolution_rate": resolved_count / len(self.gaps) if self.gaps else 0,
            "by_type": by_type,
            "by_priority": by_priority,
            "by_domain": by_domain,
            "patterns": dict(list(self.gap_patterns.items())[:10]),  # Top 10 patterns
            "auto_retrieval_threshold": self.auto_retrieval_threshold,
            "queries_in_history": len(self.query_history)
        }
    
    def suggest_proactive_learning(self) -> List[Dict[str, Any]]:
        """Suggest areas for proactive knowledge expansion."""
        suggestions = []
        
        # Analyze patterns for frequently missing domains
        domain_gaps = {}
        for gap in self.gaps.values():
            if not gap.retrieval_success:
                domain_gaps[gap.domain] = domain_gaps.get(gap.domain, 0) + gap.frequency
        
        # Sort domains by gap frequency
        sorted_domains = sorted(domain_gaps.items(), key=lambda x: x[1], reverse=True)
        
        for domain, gap_count in sorted_domains[:5]:  # Top 5 domains
            suggestions.append({
                'type': 'domain_expansion',
                'domain': domain,
                'gap_count': gap_count,
                'priority': 'high' if gap_count > 10 else 'medium',
                'suggestion': f"Expand {domain} knowledge base - {gap_count} unresolved gaps"
            })
        
        return suggestions
