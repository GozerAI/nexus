"""
Model adapters for the Nexus Unified Platform.

This module provides adapters for various AI model providers including:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3, Claude 2)
- Google (Gemini, PaLM)
- Meta (Llama 3, Llama 2)
- Mistral AI
- Cohere
- Local models via Ollama
- Specialized models

Adapters follow a consistent interface defined in BaseModelAdapter.
"""

from .base import (
    BaseModelAdapter,
    ModelCapability,
    ModelInfo,
    ModelSize,
    StubModelAdapter,
)
from .registry import (
    MODEL_REGISTRY,
    get_cheapest_models,
    get_model,
    get_model_count,
    get_models_by_use_case,
    get_supported_count,
    list_models,
    register_model,
)
from .factory import (
    ModelFactory,
    auto_register_providers,
)

# Import specific adapters (these will be implemented)
try:
    from .openai_adapter import OpenAIModelAdapter
except ImportError:
    OpenAIModelAdapter = None

try:
    from .anthropic_adapter import AnthropicModelAdapter
except ImportError:
    AnthropicModelAdapter = None

try:
    from .ollama_adapter import OllamaModelAdapter
except ImportError:
    OllamaModelAdapter = None

try:
    from .google_adapter import GoogleModelAdapter
except ImportError:
    GoogleModelAdapter = None

try:
    from .mistral_adapter import MistralModelAdapter
except ImportError:
    MistralModelAdapter = None

try:
    from .together_adapter import TogetherModelAdapter
except ImportError:
    TogetherModelAdapter = None

try:
    from .cohere_adapter import CohereModelAdapter
except ImportError:
    CohereModelAdapter = None

try:
    from .replicate_adapter import ReplicateAdapter
except ImportError:
    ReplicateAdapter = None

try:
    from .huggingface_adapter import HuggingFaceAdapter
except ImportError:
    HuggingFaceAdapter = None


__all__ = [
    # Base classes
    "BaseModelAdapter",
    "StubModelAdapter",
    "ModelInfo",
    "ModelSize",
    "ModelCapability",
    # Registry functions
    "MODEL_REGISTRY",
    "get_model",
    "list_models",
    "register_model",
    "get_models_by_use_case",
    "get_cheapest_models",
    "get_model_count",
    "get_supported_count",
    # Factory
    "ModelFactory",
    "auto_register_providers",
    "get_adapter_for_model",
    # Specific adapters (if available)
    "OpenAIModelAdapter",
    "AnthropicModelAdapter",
    "GoogleModelAdapter",
    "MistralModelAdapter",
    "CohereModelAdapter",
    "TogetherModelAdapter",
    "OllamaModelAdapter",
]


def get_adapter_for_model(model_name: str, api_key: str = None, **kwargs):
    """
    Get the appropriate adapter for a model.

    Args:
        model_name: Name of the model
        api_key: API key for the provider (if required)
        **kwargs: Additional configuration

    Returns:
        Initialized model adapter

    Raises:
        KeyError: If model not found
        ImportError: If required adapter not available
    """
    from nexus.providers.ensemble.types import ModelProvider

    # Map model names to providers
    provider = None

    if "gpt-" in model_name or "text-davinci" in model_name:
        provider = ModelProvider.OPENAI
    elif "claude-" in model_name:
        provider = ModelProvider.ANTHROPIC
    elif "gemini-" in model_name:
        provider = ModelProvider.GOOGLE
    elif "mistral-" in model_name or ("mixtral-" in model_name and "/" not in model_name):
        provider = ModelProvider.MISTRAL
    elif "command" in model_name:
        provider = ModelProvider.COHERE
    elif "/" in model_name and ("llama" in model_name.lower() or "mixtral" in model_name.lower() or "qwen" in model_name.lower()):
        provider = ModelProvider.TOGETHER

    # Use ModelFactory if we detected a provider
    if provider:
        try:
            return ModelFactory.create_by_provider(provider, model_name, api_key=api_key, **kwargs)
        except Exception as e:
            # If factory fails, fall through to try registry
            pass

    # Fall back to registry-based lookup
    try:
        return ModelFactory.create(model_name, api_key=api_key, **kwargs)
    except ValueError:
        # Model not in registry, raise error
        raise KeyError(f"Model {model_name} not found in registry and could not auto-detect provider")
