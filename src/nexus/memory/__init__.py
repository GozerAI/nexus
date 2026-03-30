"""
Nexus Memory System

Advanced memory and knowledge management with 17 specialized modules.

This system provides:
- Factual memory with provenance tracking
- Skill/procedural memory
- Pattern recognition and discovery
- Knowledge validation and verification
- Gap detection and curriculum learning
- Knowledge expansion
- Memory analytics
- 45+ domain knowledge base
"""

from .knowledge_base import KnowledgeBase, KnowledgeType, KnowledgeConfidence, KnowledgeItem
from .factual_memory_engine import FactualMemoryEngine
from .skill_memory_engine import SkillMemoryEngine
from .pattern_recognition_engine import PatternRecognitionEngine
from .memory_block_manager import MemoryBlockManager
from .knowledge_validator import KnowledgeValidator
from .knowledge_gap_tracker import KnowledgeGapTracker
from .knowledge_expander import KnowledgeExpander
from .memory_analytics import MemoryAnalytics
from .knowledge_graph_visualizer import KnowledgeGraphVisualizer

__all__ = [
    # Core classes
    "KnowledgeBase",
    "KnowledgeType",
    "KnowledgeConfidence",
    "KnowledgeItem",

    # Memory engines
    "FactualMemoryEngine",
    "SkillMemoryEngine",
    "PatternRecognitionEngine",

    # Management
    "MemoryBlockManager",
    "KnowledgeValidator",
    "KnowledgeGapTracker",
    "KnowledgeExpander",

    # Analytics and visualization
    "MemoryAnalytics",
    "KnowledgeGraphVisualizer",
]

__version__ = "1.0.0"
