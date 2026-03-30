"""
Expert Personas Module - Panel of Experts System

Provides specialized AI experts that can analyze tasks,
provide opinions, and reach consensus through voting.
"""

from .base import ExpertPersona, BaseExpert, ExpertOpinion, ExpertResult, Task
from .consensus import ConsensusEngine, ConsensusResult

__all__ = [
    "ExpertPersona",
    "BaseExpert", 
    "ExpertOpinion",
    "ExpertResult",
    "Task",
    "ConsensusEngine",
    "ConsensusResult",
]
