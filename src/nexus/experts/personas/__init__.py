"""
Expert Personas - Specialized AI experts for the Panel of Experts system
"""

from .research import ResearchExpert, RESEARCH_PERSONA
from .analyst import AnalystExpert, ANALYST_PERSONA
from .writer import WriterExpert, WRITER_PERSONA
from .engineer import EngineerExpert, ENGINEER_PERSONA
from .critic import CriticExpert, CRITIC_PERSONA
from .strategist import StrategistExpert, STRATEGIST_PERSONA

__all__ = [
    # Expert classes
    "ResearchExpert",
    "AnalystExpert", 
    "WriterExpert",
    "EngineerExpert",
    "CriticExpert",
    "StrategistExpert",
    # Persona definitions
    "RESEARCH_PERSONA",
    "ANALYST_PERSONA",
    "WRITER_PERSONA",
    "ENGINEER_PERSONA",
    "CRITIC_PERSONA",
    "STRATEGIST_PERSONA",
]

# Quick access to all experts
ALL_EXPERTS = [
    ResearchExpert,
    AnalystExpert,
    WriterExpert,
    EngineerExpert,
    CriticExpert,
    StrategistExpert,
]

ALL_PERSONAS = [
    RESEARCH_PERSONA,
    ANALYST_PERSONA,
    WRITER_PERSONA,
    ENGINEER_PERSONA,
    CRITIC_PERSONA,
    STRATEGIST_PERSONA,
]
