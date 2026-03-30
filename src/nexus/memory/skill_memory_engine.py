
"""
Skill Memory Engine for the Nexus AI Platform.
Handles volatile, skill-based knowledge with rapid adaptation.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from nexus.memory.memory_block_manager import MemoryBlockManager, MemoryBlockType

logger = logging.getLogger(__name__)

class SkillMemoryEngine:
    """Engine for managing skill-based, volatile memory."""
    
    def __init__(self, memory_manager: MemoryBlockManager):
        self.memory_manager = memory_manager
        self.skill_contexts: Dict[str, List[str]] = {}  # Context -> block_ids
        self.learning_sessions: Dict[str, Dict[str, Any]] = {}
        self.adaptation_threshold = 0.7
        self.volatility_factor = 0.1
        
    def learn_skill(self, skill_id: str, skill_data: Any, context: str = "general", metadata: Dict[str, Any] = None) -> str:
        """Learn a new skill or update existing skill knowledge."""
        block_id = f"skill_{skill_id}_{context}"
        
        skill_metadata = {
            'skill_type': 'procedural',
            'context': context,
            'learning_session': datetime.now(timezone.utc).isoformat(),
            'volatility': self.volatility_factor,
            **(metadata or {})
        }
        
        # Check if skill already exists
        existing_block = self.memory_manager.retrieve_block(block_id)
        if existing_block:
            # Update existing skill
            existing_block.update_content(skill_data, skill_metadata)
            logger.info(f"Updated skill: {skill_id} in context: {context}")
        else:
            # Create new skill block
            block = self.memory_manager.store_skill_knowledge(block_id, skill_data, skill_metadata)
            block.add_tag(context)
            block.add_tag("skill")
            block.add_tag(skill_id)
            logger.info(f"Learned new skill: {skill_id} in context: {context}")
        
        # Update context mapping
        if context not in self.skill_contexts:
            self.skill_contexts[context] = []
        if block_id not in self.skill_contexts[context]:
            self.skill_contexts[context].append(block_id)
            
        return block_id
        
    def adapt_skill(self, skill_id: str, feedback: Dict[str, Any], context: str = "general"):
        """Adapt skill based on feedback and performance."""
        block_id = f"skill_{skill_id}_{context}"
        block = self.memory_manager.retrieve_block(block_id)
        
        if not block:
            logger.warning(f"Skill not found for adaptation: {skill_id}")
            return
            
        # Extract feedback signals
        success_rate = feedback.get('success_rate', 0.5)
        performance_score = feedback.get('performance_score', 0.5)
        user_satisfaction = feedback.get('user_satisfaction', 0.5)
        
        # Calculate adaptation strength
        adaptation_strength = (success_rate + performance_score + user_satisfaction) / 3.0
        
        # Update confidence based on adaptation
        if adaptation_strength > self.adaptation_threshold:
            block.confidence_score = min(1.0, block.confidence_score + 0.1)
        else:
            block.confidence_score = max(0.1, block.confidence_score - 0.05)
            
        # Update metadata with adaptation info
        block.metadata.update({
            'last_adaptation': datetime.now(timezone.utc).isoformat(),
            'adaptation_strength': adaptation_strength,
            'feedback_received': feedback
        })
        
        logger.info(f"Adapted skill {skill_id} with strength {adaptation_strength:.2f}")
        
    def get_contextual_skills(self, context: str) -> List[Dict[str, Any]]:
        """Retrieve all skills for a given context."""
        if context not in self.skill_contexts:
            return []
            
        skills = []
        for block_id in self.skill_contexts[context]:
            block = self.memory_manager.retrieve_block(block_id)
            if block:
                skills.append({
                    'block_id': block_id,
                    'content': block.content,
                    'confidence': block.confidence_score,
                    'metadata': block.metadata,
                    'last_accessed': block.updated_at.isoformat()
                })
                
        # Sort by confidence and recency
        skills.sort(key=lambda x: (x['confidence'], x['last_accessed']), reverse=True)
        return skills
        
    def transfer_skill(self, skill_id: str, from_context: str, to_context: str) -> bool:
        """Transfer a skill from one context to another."""
        source_block_id = f"skill_{skill_id}_{from_context}"
        source_block = self.memory_manager.retrieve_block(source_block_id)
        
        if not source_block:
            logger.warning(f"Source skill not found: {skill_id} in {from_context}")
            return False
            
        # Create new skill in target context
        target_block_id = self.learn_skill(skill_id, source_block.content, to_context, source_block.metadata)
        
        # Update metadata to reflect transfer
        target_block = self.memory_manager.retrieve_block(target_block_id)
        if target_block:
            target_block.metadata.update({
                'transferred_from': from_context,
                'transfer_time': datetime.now(timezone.utc).isoformat(),
                'original_confidence': source_block.confidence_score
            })
            target_block.add_tag("transferred")
            
        logger.info(f"Transferred skill {skill_id} from {from_context} to {to_context}")
        return True
        
    def prune_low_confidence_skills(self, confidence_threshold: float = 0.3):
        """Remove skills with confidence below threshold."""
        pruned_count = 0
        
        for context, block_ids in list(self.skill_contexts.items()):
            remaining_blocks = []
            
            for block_id in block_ids:
                block = self.memory_manager.retrieve_block(block_id)
                if block and block.confidence_score >= confidence_threshold:
                    remaining_blocks.append(block_id)
                else:
                    if block:
                        # Remove from skill blocks
                        if block_id in self.memory_manager.skill_blocks:
                            del self.memory_manager.skill_blocks[block_id]
                            pruned_count += 1
                            
            self.skill_contexts[context] = remaining_blocks
            
        logger.info(f"Pruned {pruned_count} low-confidence skills")
        
    def create_learning_session(self, session_id: str, context: str, objectives: List[str]):
        """Create a focused learning session for skill acquisition."""
        self.learning_sessions[session_id] = {
            'context': context,
            'objectives': objectives,
            'start_time': datetime.now(timezone.utc),
            'skills_learned': [],
            'performance_metrics': {}
        }
        
        logger.info(f"Created learning session: {session_id} for context: {context}")
        
    def end_learning_session(self, session_id: str, performance_metrics: Dict[str, Any]):
        """End a learning session and update skill confidences."""
        if session_id not in self.learning_sessions:
            logger.warning(f"Learning session not found: {session_id}")
            return
            
        session = self.learning_sessions[session_id]
        session['end_time'] = datetime.now(timezone.utc)
        session['performance_metrics'] = performance_metrics
        
        # Update skills learned in this session
        for skill_id in session['skills_learned']:
            block_id = f"skill_{skill_id}_{session['context']}"
            block = self.memory_manager.retrieve_block(block_id)
            if block:
                # Boost confidence based on session performance
                session_score = performance_metrics.get('overall_score', 0.5)
                confidence_boost = session_score * 0.2
                block.confidence_score = min(1.0, block.confidence_score + confidence_boost)
                
                block.metadata.update({
                    'learning_session_id': session_id,
                    'session_performance': session_score
                })
        
        logger.info(f"Ended learning session: {session_id}")
        
    def get_skill_evolution_history(self, skill_id: str, context: str = "general") -> List[Dict[str, Any]]:
        """Get the evolution history of a specific skill."""
        block_id = f"skill_{skill_id}_{context}"
        block = self.memory_manager.retrieve_block(block_id)
        
        if not block:
            return []
            
        # Extract evolution data from metadata
        evolution_history = []
        
        # Add creation event
        evolution_history.append({
            'event': 'creation',
            'timestamp': block.created_at.isoformat(),
            'confidence': 1.0,
            'details': 'Skill initially learned'
        })
        
        # Add adaptation events from metadata
        if 'feedback_received' in block.metadata:
            evolution_history.append({
                'event': 'adaptation',
                'timestamp': block.metadata.get('last_adaptation', datetime.now(timezone.utc).isoformat()),
                'confidence': block.confidence_score,
                'details': f"Adapted based on feedback: {block.metadata['feedback_received']}"
            })
            
        return evolution_history
