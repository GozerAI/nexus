
"""
Factual Memory Engine for the Nexus AI Platform.
Handles persistent, factual knowledge with stability and verification.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta, timezone
from nexus.memory.memory_block_manager import MemoryBlockManager, MemoryBlockType

logger = logging.getLogger(__name__)

class FactualMemoryEngine:
    """Engine for managing factual, persistent memory."""
    
    def __init__(self, memory_manager: MemoryBlockManager):
        self.memory_manager = memory_manager
        self.fact_categories: Dict[str, Set[str]] = {}  # Category -> fact_ids
        self.verification_history: Dict[str, List[Dict[str, Any]]] = {}
        self.contradiction_tracker: Dict[str, List[str]] = {}  # fact_id -> conflicting_fact_ids
        self.truth_threshold = 0.8
        self.persistence_factor = 0.95
        
    def store_fact(self, fact_id: str, fact_content: Any, category: str = "general", 
                   source: str = "user", confidence: float = 1.0, metadata: Dict[str, Any] = None) -> str:
        """Store a factual piece of information."""
        block_id = f"fact_{fact_id}_{category}"
        
        fact_metadata = {
            'fact_type': 'declarative',
            'category': category,
            'source': source,
            'initial_confidence': confidence,
            'verification_count': 0,
            'last_verified': datetime.now(timezone.utc).isoformat(),
            'persistence_factor': self.persistence_factor,
            **(metadata or {})
        }
        
        # Check for existing fact
        existing_block = self.memory_manager.retrieve_block(block_id)
        if existing_block:
            # Update existing fact
            existing_block.update_content(fact_content, fact_metadata)
            existing_block.confidence_score = confidence
            logger.info(f"Updated fact: {fact_id} in category: {category}")
        else:
            # Create new factual block
            block = self.memory_manager.store_factual_knowledge(block_id, fact_content, fact_metadata)
            block.confidence_score = confidence
            block.add_tag(category)
            block.add_tag("fact")
            block.add_tag(fact_id)
            logger.info(f"Stored new fact: {fact_id} in category: {category}")
        
        # Update category mapping
        if category not in self.fact_categories:
            self.fact_categories[category] = set()
        self.fact_categories[category].add(fact_id)
        
        return block_id
        
    def verify_fact(self, fact_id: str, category: str, verification_source: str, 
                   verification_result: bool, confidence_adjustment: float = 0.0) -> bool:
        """Verify a fact and update its confidence."""
        block_id = f"fact_{fact_id}_{category}"
        block = self.memory_manager.retrieve_block(block_id)
        
        if not block:
            logger.warning(f"Fact not found for verification: {fact_id}")
            return False
        
        # Record verification
        if fact_id not in self.verification_history:
            self.verification_history[fact_id] = []
            
        verification_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': verification_source,
            'result': verification_result,
            'confidence_adjustment': confidence_adjustment
        }
        self.verification_history[fact_id].append(verification_record)
        
        # Update block metadata
        block.metadata['verification_count'] = block.metadata.get('verification_count', 0) + 1
        block.metadata['last_verified'] = verification_record['timestamp']
        
        # Adjust confidence based on verification
        if verification_result:
            block.confidence_score = min(1.0, block.confidence_score + confidence_adjustment + 0.1)
        else:
            block.confidence_score = max(0.1, block.confidence_score - confidence_adjustment - 0.2)
            
        logger.info(f"Verified fact {fact_id}: {verification_result} (confidence: {block.confidence_score:.2f})")
        return True
        
    def check_contradictions(self, new_fact_id: str, new_fact_content: Any, category: str) -> List[Dict[str, Any]]:
        """Check for contradictions with existing facts."""
        contradictions = []
        
        if category in self.fact_categories:
            for existing_fact_id in self.fact_categories[category]:
                if existing_fact_id == new_fact_id:
                    continue
                    
                existing_block_id = f"fact_{existing_fact_id}_{category}"
                existing_block = self.memory_manager.retrieve_block(existing_block_id)
                
                if existing_block:
                    # Simple contradiction detection (can be enhanced with NLP/semantic analysis)
                    contradiction_detected = self._detect_contradiction(
                        new_fact_content, existing_block.content
                    )
                    
                    if contradiction_detected:
                        contradictions.append({
                            'existing_fact_id': existing_fact_id,
                            'existing_content': existing_block.content,
                            'existing_confidence': existing_block.confidence_score,
                            'contradiction_type': 'semantic'
                        })
                        
        return contradictions
        
    def _detect_contradiction(self, fact1: Any, fact2: Any) -> bool:
        """Simple contradiction detection logic."""
        # This is a placeholder - in a real system, this would use
        # semantic analysis, NLP, or domain-specific logic
        
        # Convert to strings for basic comparison
        str1 = str(fact1).lower()
        str2 = str(fact2).lower()
        
        # Check for direct negation patterns
        negation_patterns = [
            ('true', 'false'),
            ('yes', 'no'),
            ('is', 'is not'),
            ('exists', 'does not exist'),
            ('can', 'cannot')
        ]
        
        for pos, neg in negation_patterns:
            if (pos in str1 and neg in str2) or (neg in str1 and pos in str2):
                return True
                
        return False
        
    def resolve_contradiction(self, fact1_id: str, fact2_id: str, category: str, 
                             resolution: str, winning_fact_id: str = None) -> bool:
        """Resolve a contradiction between two facts."""
        block1_id = f"fact_{fact1_id}_{category}"
        block2_id = f"fact_{fact2_id}_{category}"
        
        block1 = self.memory_manager.retrieve_block(block1_id)
        block2 = self.memory_manager.retrieve_block(block2_id)
        
        if not block1 or not block2:
            logger.warning(f"One or both facts not found: {fact1_id}, {fact2_id}")
            return False
        
        # Record contradiction resolution
        resolution_record = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fact1_id': fact1_id,
            'fact2_id': fact2_id,
            'resolution_method': resolution,
            'winning_fact': winning_fact_id
        }
        
        if winning_fact_id == fact1_id:
            # Boost winner, demote loser
            block1.confidence_score = min(1.0, block1.confidence_score + 0.2)
            block2.confidence_score = max(0.1, block2.confidence_score - 0.3)
        elif winning_fact_id == fact2_id:
            block2.confidence_score = min(1.0, block2.confidence_score + 0.2)
            block1.confidence_score = max(0.1, block1.confidence_score - 0.3)
        else:
            # No clear winner, reduce confidence in both
            block1.confidence_score = max(0.3, block1.confidence_score - 0.1)
            block2.confidence_score = max(0.3, block2.confidence_score - 0.1)
            
        # Update metadata
        for block in [block1, block2]:
            if 'contradiction_resolutions' not in block.metadata:
                block.metadata['contradiction_resolutions'] = []
            block.metadata['contradiction_resolutions'].append(resolution_record)
            
        logger.info(f"Resolved contradiction between {fact1_id} and {fact2_id}")
        return True
        
    def get_facts_by_category(self, category: str, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve all facts in a category above confidence threshold."""
        if category not in self.fact_categories:
            return []
            
        facts = []
        for fact_id in self.fact_categories[category]:
            block_id = f"fact_{fact_id}_{category}"
            block = self.memory_manager.retrieve_block(block_id)
            
            if block and block.confidence_score >= min_confidence:
                facts.append({
                    'fact_id': fact_id,
                    'content': block.content,
                    'confidence': block.confidence_score,
                    'metadata': block.metadata,
                    'verification_count': block.metadata.get('verification_count', 0),
                    'last_verified': block.metadata.get('last_verified')
                })
                
        # Sort by confidence
        facts.sort(key=lambda x: x['confidence'], reverse=True)
        return facts
        
    def get_high_confidence_facts(self, min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """Get all high-confidence facts across categories."""
        high_confidence_facts = []
        
        for category in self.fact_categories:
            category_facts = self.get_facts_by_category(category, min_confidence)
            high_confidence_facts.extend(category_facts)
            
        return high_confidence_facts
        
    def archive_low_confidence_facts(self, confidence_threshold: float = 0.3) -> int:
        """Archive facts below confidence threshold (mark but don't delete)."""
        archived_count = 0
        
        for category, fact_ids in self.fact_categories.items():
            for fact_id in list(fact_ids):
                block_id = f"fact_{fact_id}_{category}"
                block = self.memory_manager.retrieve_block(block_id)
                
                if block and block.confidence_score < confidence_threshold:
                    block.metadata['archived'] = True
                    block.metadata['archive_date'] = datetime.now(timezone.utc).isoformat()
                    block.metadata['archive_reason'] = 'low_confidence'
                    block.add_tag('archived')
                    archived_count += 1
                    
        logger.info(f"Archived {archived_count} low-confidence facts")
        return archived_count
        
    def get_fact_lineage(self, fact_id: str, category: str) -> Dict[str, Any]:
        """Get the lineage and history of a fact."""
        block_id = f"fact_{fact_id}_{category}"
        block = self.memory_manager.retrieve_block(block_id)
        
        if not block:
            return {}
            
        lineage = {
            'fact_id': fact_id,
            'category': category,
            'created': block.created_at.isoformat(),
            'last_updated': block.updated_at.isoformat(),
            'current_confidence': block.confidence_score,
            'access_count': block.access_count,
            'source': block.metadata.get('source', 'unknown'),
            'verification_history': self.verification_history.get(fact_id, []),
            'contradiction_resolutions': block.metadata.get('contradiction_resolutions', []),
            'tags': list(block.tags)
        }
        
        return lineage
