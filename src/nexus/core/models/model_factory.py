"""
Factory for creating model instances.
"""

import logging
from typing import Dict, Type

from nexus.core.models.base import BaseModel, ModelConfig, ModelProvider
from nexus.core.models.openai_provider import OpenAIProvider
from nexus.core.models.anthropic_provider import AnthropicProvider
from nexus.core.models.stub_provider import StubProvider

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating model provider instances.
    """
    
    # Registry of provider classes
    _providers: Dict[ModelProvider, Type[BaseModel]] = {
        ModelProvider.OPENAI: OpenAIProvider,
        ModelProvider.ANTHROPIC: AnthropicProvider,
        ModelProvider.STUB: StubProvider,
    }
    
    @classmethod
    def create_model(cls, config: ModelConfig) -> BaseModel:
        """
        Create a model instance from configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            BaseModel instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider_class = cls._providers.get(config.provider)
        
        if not provider_class:
            raise ValueError(
                f"Unsupported provider: {config.provider}. "
                f"Supported: {list(cls._providers.keys())}"
            )
        
        logger.info(f"Creating {config.provider.value} model: {config.name}")
        
        try:
            model = provider_class(config)
            model.validate_config()
            return model
        except Exception as e:
            logger.error(f"Failed to create model {config.name}: {e}")
            raise
    
    @classmethod
    def register_provider(cls, provider: ModelProvider, provider_class: Type[BaseModel]):
        """
        Register a new provider class.
        
        Args:
            provider: Provider enum
            provider_class: Provider class
        """
        cls._providers[provider] = provider_class
        logger.info(f"Registered provider: {provider.value}")
    
    @classmethod
    def list_providers(cls) -> list:
        """
        List all registered providers.
        
        Returns:
            List of provider names
        """
        return [p.value for p in cls._providers.keys()]
