"""
Blueprints Module

Parse and work with ebook blueprint specifications.
Generate new blueprints from topics.
"""

from .models import BlueprintSpec, BookSpec, ChapterSpec
from .parser import BlueprintParser
from .pipeline import BlueprintEbookPipeline, PromptBuilder, ChapterOutput, BookOutput
from .output_manager import OutputManager
from .llm_backend import (
    LLMBackend,
    LLMResponse,
    AnthropicBackend,
    OllamaBackend,
    OpenAIBackend,
    get_backend,
    get_preset_backend,
    BACKENDS
)
from .generator import (
    BlueprintGenerator,
    BlueprintGenerationResult,
    generate_blueprint_for_topic,
    get_pending_blueprints,
    get_completed_blueprints,
)

__all__ = [
    # Models
    'BlueprintSpec',
    'BookSpec', 
    'ChapterSpec',
    
    # Parser
    'BlueprintParser',
    
    # Pipeline
    'BlueprintEbookPipeline',
    'PromptBuilder',
    'ChapterOutput',
    'BookOutput',
    
    # Output
    'OutputManager',
    
    # LLM Backends
    'LLMBackend',
    'LLMResponse',
    'AnthropicBackend',
    'OllamaBackend',
    'OpenAIBackend',
    'get_backend',
    'get_preset_backend',
    'BACKENDS',
    
    # Blueprint Generator
    'BlueprintGenerator',
    'BlueprintGenerationResult',
    'generate_blueprint_for_topic',
    'get_pending_blueprints',
    'get_completed_blueprints',
]
