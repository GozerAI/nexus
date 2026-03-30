"""
AI Model providers for TheNexus.

This module provides interfaces and implementations for various AI model providers.
"""

from nexus.core.models.base import BaseModel, ModelResponse, ModelConfig
from nexus.core.models.stub_provider import StubProvider

# Try to import real providers, fallback if not available
try:
    from nexus.core.models.openai_provider import OpenAIProvider
except ImportError:
    OpenAIProvider = None

try:
    from nexus.core.models.anthropic_provider import AnthropicProvider
except ImportError:
    AnthropicProvider = None

from nexus.core.models.model_factory import ModelFactory

__all__ = [
    "BaseModel",
    "ModelResponse",
    "ModelConfig",
    "OpenAIProvider",
    "AnthropicProvider",
    "StubProvider",
    "ModelFactory",
]
