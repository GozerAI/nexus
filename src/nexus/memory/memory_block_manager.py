
"""
Memory Block Manager for the Nexus AI Platform.
Manages partitioned memory for factual vs. skill-based knowledge.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)

class MemoryBlockType(Enum):
    FACTUAL = "factual"
    SKILL = "skill"
    HYBRID = "hybrid"

class MemoryBlock:
    """Represents a memory block with metadata and content."""
    
    def __init__(self, block_id: str, block_type: MemoryBlockType, content: Any, metadata: Dict[str, Any] = None):
        self.block_id = block_id
        self.block_type = block_type
        self.content = content
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self.access_count = 0
        self.confidence_score = 1.0
        self.tags = set()
        
    def update_content(self, new_content: Any, metadata_update: Dict[str, Any] = None):
        """Update block content and metadata."""
        self.content = new_content
        if metadata_update:
            self.metadata.update(metadata_update)
        self.updated_at = datetime.now(timezone.utc)
        
    def add_tag(self, tag: str):
        """Add a contextual tag to the memory block."""
        self.tags.add(tag)
        
    def increment_access(self):
        """Track access for reweighting algorithms."""
        self.access_count += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory block to dictionary representation."""
        return {
            'block_id': self.block_id,
            'block_type': self.block_type.value,
            'content': self.content,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'access_count': self.access_count,
            'confidence_score': self.confidence_score,
            'tags': list(self.tags)
        }

class MemoryBlockManager:
    """Manages partitioned memory blocks for factual and skill-based knowledge."""
    
    def __init__(self):
        self.factual_blocks: Dict[str, MemoryBlock] = {}
        self.skill_blocks: Dict[str, MemoryBlock] = {}
        self.hybrid_blocks: Dict[str, MemoryBlock] = {}
        self.initialized = False
        
    def initialize(self):
        """Initialize the memory block manager."""
        logger.info("Initializing Memory Block Manager...")
        self.initialized = True
        logger.info("Memory Block Manager initialized")
        
    def store_factual_knowledge(self, block_id: str, content: Any, metadata: Dict[str, Any] = None) -> MemoryBlock:
        """Store factual knowledge in persistent memory."""
        block = MemoryBlock(block_id, MemoryBlockType.FACTUAL, content, metadata)
        self.factual_blocks[block_id] = block
        logger.info(f"Stored factual knowledge block: {block_id}")
        return block
        
    def store_skill_knowledge(self, block_id: str, content: Any, metadata: Dict[str, Any] = None) -> MemoryBlock:
        """Store skill-based knowledge in volatile memory."""
        block = MemoryBlock(block_id, MemoryBlockType.SKILL, content, metadata)
        self.skill_blocks[block_id] = block
        logger.info(f"Stored skill knowledge block: {block_id}")
        return block
        
    def store_hybrid_knowledge(self, block_id: str, content: Any, metadata: Dict[str, Any] = None) -> MemoryBlock:
        """Store hybrid knowledge that spans both factual and skill domains."""
        block = MemoryBlock(block_id, MemoryBlockType.HYBRID, content, metadata)
        self.hybrid_blocks[block_id] = block
        logger.info(f"Stored hybrid knowledge block: {block_id}")
        return block
        
    def retrieve_block(self, block_id: str) -> Optional[MemoryBlock]:
        """Retrieve a memory block by ID from any partition."""
        # Check factual blocks first
        if block_id in self.factual_blocks:
            block = self.factual_blocks[block_id]
            block.increment_access()
            return block
            
        # Check skill blocks
        if block_id in self.skill_blocks:
            block = self.skill_blocks[block_id]
            block.increment_access()
            return block
            
        # Check hybrid blocks
        if block_id in self.hybrid_blocks:
            block = self.hybrid_blocks[block_id]
            block.increment_access()
            return block
            
        return None
        
    def search_blocks_by_tag(self, tag: str, block_type: Optional[MemoryBlockType] = None) -> List[MemoryBlock]:
        """Search for memory blocks by tag, optionally filtered by type."""
        results = []
        
        collections = []
        if block_type is None:
            collections = [self.factual_blocks, self.skill_blocks, self.hybrid_blocks]
        elif block_type == MemoryBlockType.FACTUAL:
            collections = [self.factual_blocks]
        elif block_type == MemoryBlockType.SKILL:
            collections = [self.skill_blocks]
        elif block_type == MemoryBlockType.HYBRID:
            collections = [self.hybrid_blocks]
            
        for collection in collections:
            for block in collection.values():
                if tag in block.tags:
                    block.increment_access()
                    results.append(block)
                    
        return results
        
    def reweight_blocks(self, reweight_factor: float = 0.1):
        """Apply reweighting based on access patterns and confidence scores."""
        all_blocks = list(self.factual_blocks.values()) + list(self.skill_blocks.values()) + list(self.hybrid_blocks.values())
        
        for block in all_blocks:
            # Boost confidence for frequently accessed blocks
            access_boost = min(block.access_count * 0.01, 0.2)
            
            # Apply time decay for skill blocks (more volatile)
            if block.block_type == MemoryBlockType.SKILL:
                time_diff = (datetime.now(timezone.utc) - block.updated_at).total_seconds() / 86400  # days
                time_decay = max(0.1, 1.0 - (time_diff * 0.02))  # 2% decay per day
                block.confidence_score = min(1.0, (block.confidence_score + access_boost) * time_decay)
            else:
                # Factual blocks are more persistent
                block.confidence_score = min(1.0, block.confidence_score + access_boost)
                
        logger.info("Applied reweighting to memory blocks")
        
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory block usage."""
        return {
            'factual_blocks': len(self.factual_blocks),
            'skill_blocks': len(self.skill_blocks),
            'hybrid_blocks': len(self.hybrid_blocks),
            'total_blocks': len(self.factual_blocks) + len(self.skill_blocks) + len(self.hybrid_blocks),
            'avg_confidence_factual': sum(b.confidence_score for b in self.factual_blocks.values()) / max(len(self.factual_blocks), 1),
            'avg_confidence_skill': sum(b.confidence_score for b in self.skill_blocks.values()) / max(len(self.skill_blocks), 1),
            'avg_confidence_hybrid': sum(b.confidence_score for b in self.hybrid_blocks.values()) / max(len(self.hybrid_blocks), 1)
        }
        
    def clear_volatile_memory(self):
        """Clear skill-based (volatile) memory blocks."""
        cleared_count = len(self.skill_blocks)
        self.skill_blocks.clear()
        logger.info(f"Cleared {cleared_count} volatile skill memory blocks")
        
    def export_persistent_memory(self) -> Dict[str, Any]:
        """Export factual and hybrid memory blocks for persistence."""
        return {
            'factual_blocks': {k: v.to_dict() for k, v in self.factual_blocks.items()},
            'hybrid_blocks': {k: v.to_dict() for k, v in self.hybrid_blocks.items()},
            'export_timestamp': datetime.now(timezone.utc).isoformat()
        }
