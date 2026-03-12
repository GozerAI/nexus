
import logging
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType

logger = logging.getLogger(__name__)

class KnowledgeExpander:
    """
    System for expanding knowledge base through interaction and learning.
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.knowledge_base = knowledge_base
        self.learning_domains = {
            'science', 'technology', 'history', 'geography', 'literature',
            'mathematics', 'health', 'language', 'arts', 'sports', 'politics',
            'economics', 'philosophy', 'psychology', 'biology', 'chemistry',
            'physics', 'astronomy', 'geology', 'anthropology', 'sociology'
        }
        self.pending_knowledge = {}  # Knowledge awaiting verification
        self.confidence_threshold = 0.7
        self.initialized = False
    
    def initialize(self):
        """Initialize the knowledge expander."""
        if self.initialized:
            return
        logger.info("Knowledge Expander initialized")
        self.initialized = True
    
    def expand_domain(self, domain: str) -> Dict[str, Any]:
        """Expand knowledge in a specific domain."""
        logger.info(f"Expanding knowledge in domain: {domain}")
        
        result = {
            'domain': domain,
            'expanded': False,
            'facts_added': 0,
            'message': f'Domain expansion for {domain}'
        }
        
        # Use existing expand_domain_knowledge method
        facts_added = self.expand_domain_knowledge(domain)
        
        if facts_added > 0:
            result['expanded'] = True
            result['facts_added'] = facts_added
            result['message'] = f'Successfully added {facts_added} facts to {domain} domain'
        else:
            result['message'] = f'No additional facts available for {domain} domain'
        
        return result
        
    def suggest_knowledge_expansion(self, query: str, context: str = None) -> List[Dict[str, Any]]:
        """Suggest areas where knowledge could be expanded based on queries."""
        suggestions = []
        query_lower = query.lower()
        
        # Identify domain
        domain = self._identify_domain(query_lower)
        
        # Check if we have sufficient knowledge in this domain
        domain_knowledge = self.knowledge_base.query_knowledge(
            domain, max_results=50
        ) if domain else []
        
        if len(domain_knowledge) < 10:  # Threshold for "sufficient" knowledge
            suggestions.append({
                'type': 'domain_expansion',
                'domain': domain,
                'current_count': len(domain_knowledge),
                'priority': 'high' if len(domain_knowledge) < 5 else 'medium',
                'suggestion': f"Expand {domain} knowledge base with more facts and procedures"
            })
        
        # Look for specific knowledge gaps
        gaps = self._identify_knowledge_gaps(query_lower, domain)
        suggestions.extend(gaps)
        
        return suggestions
    
    def learn_from_failed_query(self, query: str, expected_type: str = "factual"):
        """Learn from queries that couldn't be answered."""
        domain = self._identify_domain(query.lower())
        
        # Store as pending knowledge for later expansion
        knowledge_id = f"pending_{domain}_{int(time.time())}"
        self.pending_knowledge[knowledge_id] = {
            'query': query,
            'domain': domain,
            'type': expected_type,
            'timestamp': datetime.now(),
            'confidence': 0.3,  # Low confidence for unverified
            'needs_research': True
        }
        
        logger.info(f"Added pending knowledge item for domain: {domain}")
    
    def add_contextual_knowledge(self, topic: str, facts: List[str], 
                                source: str = "learned", confidence: float = 0.8):
        """Add knowledge learned from context or interaction."""
        domain = self._identify_domain(topic.lower())
        
        for fact in facts:
            knowledge_id = self.knowledge_base.add_knowledge(
                content=fact,
                knowledge_type=KnowledgeType.FACTUAL,
                source=source,
                confidence=confidence,
                context_tags=[domain, topic.lower()]
            )
            logger.info(f"Added contextual knowledge: {knowledge_id}")
    
    def expand_domain_knowledge(self, domain: str) -> int:
        """Expand knowledge in a specific domain with common facts."""
        added_count = 0
        
        domain_expansions = {
            'science': [
                "The human body has 12 systems including circulatory, respiratory, and nervous systems",
                "Atoms are composed of protons, neutrons, and electrons",
                "The Earth is approximately 4.54 billion years old",
                "Vaccines work by training the immune system to recognize pathogens",
                "Evolution is the process by which species change over time through natural selection"
            ],
            'technology': [
                "The first computer bug was literally a bug - a moth found in a computer in 1947",
                "The Internet was originally called ARPANET",
                "CPU stands for Central Processing Unit",
                "JavaScript was created in just 10 days",
                "The first website is still online at info.cern.ch"
            ],
            'history': [
                "The Renaissance period lasted from the 14th to the 17th century",
                "The Great Wall of China was built over many dynasties",
                "The Industrial Revolution began in Britain in the late 18th century",
                "The Cold War lasted from approximately 1947 to 1991",
                "The printing press was invented by Johannes Gutenberg around 1440"
            ],
            'geography': [
                "The Amazon Rainforest is located primarily in Brazil",
                "The Sahara Desert is the largest hot desert in the world",
                "The Nile River is traditionally considered the longest river in the world",
                "Russia is the largest country by land area",
                "The Dead Sea is the lowest point on Earth's surface"
            ],
            'mathematics': [
                "The Fibonacci sequence starts: 0, 1, 1, 2, 3, 5, 8, 13...",
                "A prime number is only divisible by 1 and itself",
                "The Pythagorean theorem states that a² + b² = c² for right triangles",
                "Infinity is not a number but a concept",
                "Zero was invented in ancient India"
            ]
        }
        
        if domain in domain_expansions:
            for fact in domain_expansions[domain]:
                knowledge_id = self.knowledge_base.add_knowledge(
                    content=fact,
                    knowledge_type=KnowledgeType.FACTUAL,
                    source="domain_expansion",
                    confidence=0.9,
                    context_tags=[domain]
                )
                added_count += 1
                
        logger.info(f"Added {added_count} knowledge items to {domain} domain")
        return added_count
    
    def _identify_domain(self, query: str) -> str:
        """Identify the domain/subject area of a query."""
        domain_keywords = {
            'science': ['science', 'biology', 'chemistry', 'physics', 'experiment', 'atom', 'molecule', 'cell', 'dna', 'evolution'],
            'technology': ['computer', 'programming', 'software', 'internet', 'code', 'algorithm', 'data', 'tech', 'digital'],
            'history': ['history', 'war', 'ancient', 'century', 'historical', 'past', 'civilization', 'empire', 'revolution'],
            'geography': ['country', 'capital', 'city', 'mountain', 'river', 'ocean', 'continent', 'geography', 'location'],
            'mathematics': ['math', 'number', 'calculate', 'equation', 'formula', 'geometry', 'algebra', 'statistics'],
            'literature': ['book', 'author', 'novel', 'poem', 'literature', 'write', 'story', 'chapter'],
            'health': ['health', 'medicine', 'doctor', 'disease', 'body', 'medical', 'treatment', 'symptom'],
            'arts': ['art', 'painting', 'music', 'artist', 'sculpture', 'museum', 'gallery', 'creative'],
            'sports': ['sport', 'game', 'team', 'player', 'competition', 'athletic', 'championship', 'tournament']
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in query for keyword in keywords):
                return domain
        
        return 'general'
    
    def _identify_knowledge_gaps(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Identify specific knowledge gaps based on query patterns."""
        gaps = []
        
        # Common question patterns that indicate knowledge gaps
        gap_patterns = {
            'who': 'biographical_information',
            'what': 'definitional_knowledge',
            'when': 'temporal_information',
            'where': 'geographical_information', 
            'why': 'causal_relationships',
            'how': 'procedural_knowledge'
        }
        
        for pattern, gap_type in gap_patterns.items():
            if pattern in query:
                gaps.append({
                    'type': 'specific_gap',
                    'gap_type': gap_type,
                    'domain': domain,
                    'priority': 'medium',
                    'suggestion': f"Add more {gap_type.replace('_', ' ')} in {domain}"
                })
        
        return gaps
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """Get statistics about knowledge expansion."""
        domain_counts = {}
        total_knowledge = len(self.knowledge_base.knowledge_items)
        
        # Count knowledge by domain (approximate based on tags)
        for item in self.knowledge_base.knowledge_items.values():
            if item.context_tags:
                for tag in item.context_tags:
                    if tag in self.learning_domains:
                        domain_counts[tag] = domain_counts.get(tag, 0) + 1
        
        return {
            'total_knowledge_items': total_knowledge,
            'domain_distribution': domain_counts,
            'pending_items': len(self.pending_knowledge),
            'learning_domains': list(self.learning_domains),
            'coverage_score': len(domain_counts) / len(self.learning_domains)
        }
