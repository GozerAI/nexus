"""
Model Factory for Dynamic Model Creation

Provides factory pattern for creating model adapters with:
- Provider-based model creation
- Configuration validation
- Dynamic provider registration
- Easy model instantiation
- Support for custom providers

Adapted from: TheNexus/src/thenexus/models/model_factory.py
"""

import logging
from typing import Dict, Type, Optional, Any

from nexus.providers.adapters.base import BaseModelAdapter, ModelInfo
from nexus.providers.adapters.registry import MODEL_REGISTRY
from nexus.providers.ensemble.types import ModelProvider

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory for creating model adapters.

    Supports:
    - Provider-based registration
    - Dynamic adapter creation
    - Configuration validation
    - Provider discovery

    Example:
        >>> from unified_intelligence.models import ModelFactory
        >>> from nexus.providers.adapters.openai_adapter import OpenAIAdapter
        >>> from nexus.providers.adapters.anthropic_adapter import AnthropicAdapter
        >>>
        >>> # Register providers
        >>> ModelFactory.register_provider(ModelProvider.OPENAI, OpenAIAdapter)
        >>> ModelFactory.register_provider(ModelProvider.ANTHROPIC, AnthropicAdapter)
        >>>
        >>> # Create model instances
        >>> gpt4 = ModelFactory.create("gpt-4-turbo", api_key="sk-...")
        >>> claude = ModelFactory.create("claude-3-opus-20240229", api_key="sk-...")
        >>>
        >>> # List available providers
        >>> providers = ModelFactory.list_providers()
    """

    _providers: Dict[ModelProvider, Type[BaseModelAdapter]] = {}

    @classmethod
    def register_provider(
        cls,
        provider: ModelProvider,
        adapter_class: Type[BaseModelAdapter]
    ) -> None:
        """
        Register a model provider adapter.

        Args:
            provider: Provider identifier (ModelProvider enum)
            adapter_class: Adapter class implementing BaseModelAdapter

        Raises:
            ValueError: If adapter_class doesn't inherit from BaseModelAdapter
        """
        if not issubclass(adapter_class, BaseModelAdapter):
            raise ValueError(
                f"Adapter class must inherit from BaseModelAdapter, "
                f"got {adapter_class.__name__}"
            )

        cls._providers[provider] = adapter_class
        logger.info(f"✅ Registered provider: {provider.value} -> {adapter_class.__name__}")

    @classmethod
    def create(
        cls,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> BaseModelAdapter:
        """
        Create a model adapter instance.

        Args:
            model_name: Model identifier (from MODEL_REGISTRY)
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration

        Returns:
            Instantiated model adapter

        Raises:
            ValueError: If model not found or provider not registered

        Example:
            >>> adapter = ModelFactory.create(
            ...     "gpt-4-turbo",
            ...     api_key="sk-...",
            ...     temperature=0.7,
            ...     max_tokens=2048
            ... )
        """
        # Look up model info in registry
        model_info = MODEL_REGISTRY.get(model_name)
        if not model_info:
            raise ValueError(
                f"Model '{model_name}' not found in registry. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

        # Check if model is supported
        if not model_info.supported:
            logger.warning(
                f"⚠️ Model '{model_name}' is registered but not yet fully supported. "
                f"Implementation notes: {model_info.notes}"
            )

        # Look up provider adapter
        provider = model_info.provider
        adapter_class = cls._providers.get(provider)
        if not adapter_class:
            raise ValueError(
                f"Provider '{provider.value}' not registered. "
                f"Call ModelFactory.register_provider() first. "
                f"Available providers: {cls.list_providers()}"
            )

        # Validate API key requirement
        if model_info.requires_api_key and not api_key:
            raise ValueError(
                f"Model '{model_name}' requires an API key. "
                f"Pass api_key parameter to ModelFactory.create()"
            )

        # Create adapter instance
        logger.debug(
            f"🏭 Creating adapter for {model_name} "
            f"using {adapter_class.__name__}"
        )

        adapter = adapter_class(
            model_info=model_info,
            api_key=api_key,
            **kwargs
        )

        logger.info(
            f"✅ Created {provider.value} adapter for {model_name}"
        )

        return adapter

    @classmethod
    def create_by_provider(
        cls,
        provider: ModelProvider,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> BaseModelAdapter:
        """
        Create a model adapter by provider and model name.

        This method allows creating models even if they're not in the registry,
        useful for new models or experimental features.

        Args:
            provider: Provider identifier
            model_name: Model name (doesn't need to be in registry)
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration

        Returns:
            Instantiated model adapter

        Raises:
            ValueError: If provider not registered

        Example:
            >>> adapter = ModelFactory.create_by_provider(
            ...     ModelProvider.OPENAI,
            ...     "gpt-5-preview",  # Not in registry yet
            ...     api_key="sk-...",
            ...     context_window=200000
            ... )
        """
        # Look up provider adapter
        adapter_class = cls._providers.get(provider)
        if not adapter_class:
            raise ValueError(
                f"Provider '{provider.value}' not registered. "
                f"Available providers: {cls.list_providers()}"
            )

        # Try to get model info from registry, or create basic info
        model_info = MODEL_REGISTRY.get(model_name)
        if not model_info:
            logger.warning(
                f"⚠️ Model '{model_name}' not in registry. "
                f"Creating basic ModelInfo. Consider adding to registry."
            )

            # Create basic model info
            from nexus.providers.adapters.base import ModelSize, ModelCapability

            model_info = ModelInfo(
                name=model_name,
                display_name=model_name,
                provider=provider,
                size=ModelSize.MEDIUM,  # Default
                context_window=kwargs.get('context_window', 8192),
                max_output_tokens=kwargs.get('max_output_tokens', 2048),
                capabilities=[ModelCapability.TEXT_GENERATION],  # Default
                cost_per_1k_input=kwargs.get('cost_per_1k_input', 0.0),
                cost_per_1k_output=kwargs.get('cost_per_1k_output', 0.0),
                supported=False,  # Mark as unsupported
                notes="Created dynamically, not in registry"
            )

        # Create adapter instance
        adapter = adapter_class(
            model_info=model_info,
            api_key=api_key,
            **kwargs
        )

        logger.info(
            f"✅ Created {provider.value} adapter for {model_name} (dynamic)"
        )

        return adapter

    @classmethod
    def list_providers(cls) -> list[ModelProvider]:
        """
        List all registered providers.

        Returns:
            List of registered provider identifiers
        """
        return list(cls._providers.keys())

    @classmethod
    def list_provider_names(cls) -> list[str]:
        """
        List all registered provider names.

        Returns:
            List of registered provider names (strings)
        """
        return [provider.value for provider in cls._providers.keys()]

    @classmethod
    def get_provider_adapter(cls, provider: ModelProvider) -> Optional[Type[BaseModelAdapter]]:
        """
        Get adapter class for a provider.

        Args:
            provider: Provider identifier

        Returns:
            Adapter class if registered, None otherwise
        """
        return cls._providers.get(provider)

    @classmethod
    def is_provider_registered(cls, provider: ModelProvider) -> bool:
        """
        Check if a provider is registered.

        Args:
            provider: Provider identifier

        Returns:
            True if provider is registered
        """
        return provider in cls._providers

    @classmethod
    def get_available_models(
        cls,
        provider: Optional[ModelProvider] = None,
        supported_only: bool = True
    ) -> list[ModelInfo]:
        """
        Get list of available models.

        Args:
            provider: Optional provider filter
            supported_only: If True, only return fully supported models

        Returns:
            List of ModelInfo objects
        """
        models = list(MODEL_REGISTRY.values())

        # Filter by provider
        if provider is not None:
            models = [m for m in models if m.provider == provider]

        # Filter by support status
        if supported_only:
            models = [m for m in models if m.supported]

        return models

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[ModelInfo]:
        """
        Get model info from registry.

        Args:
            model_name: Model identifier

        Returns:
            ModelInfo if found, None otherwise
        """
        return MODEL_REGISTRY.get(model_name)

    @classmethod
    def validate_config(cls, model_name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a model.

        Args:
            model_name: Model identifier
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        model_info = cls.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found in registry")

        # Validate max_tokens doesn't exceed limit
        max_tokens = config.get('max_tokens')
        if max_tokens and max_tokens > model_info.max_output_tokens:
            raise ValueError(
                f"max_tokens ({max_tokens}) exceeds model limit "
                f"({model_info.max_output_tokens})"
            )

        # Validate temperature range
        temperature = config.get('temperature')
        if temperature is not None and not (0.0 <= temperature <= 2.0):
            raise ValueError(
                f"temperature ({temperature}) must be between 0.0 and 2.0"
            )

        return True


def auto_register_providers() -> None:
    """
    Automatically register all available provider adapters.

    This function attempts to import and register common providers.
    It silently skips providers that aren't available.
    """
    providers_to_register = [
        (ModelProvider.OPENAI, "nexus.providers.adapters.openai_adapter", "OpenAIModelAdapter"),
        (ModelProvider.ANTHROPIC, "nexus.providers.adapters.anthropic_adapter", "AnthropicModelAdapter"),
        (ModelProvider.GOOGLE, "nexus.providers.adapters.google_adapter", "GoogleModelAdapter"),
        (ModelProvider.MISTRAL, "nexus.providers.adapters.mistral_adapter", "MistralModelAdapter"),
        (ModelProvider.COHERE, "nexus.providers.adapters.cohere_adapter", "CohereModelAdapter"),
        (ModelProvider.TOGETHER, "nexus.providers.adapters.together_adapter", "TogetherModelAdapter"),
    ]

    for provider, module_path, class_name in providers_to_register:
        try:
            # Dynamic import
            import importlib
            module = importlib.import_module(module_path)
            adapter_class = getattr(module, class_name)

            # Register
            ModelFactory.register_provider(provider, adapter_class)

        except ImportError as e:
            logger.debug(
                f"Could not import {class_name} from {module_path}: {e}"
            )
        except AttributeError as e:
            logger.debug(
                f"Could not find {class_name} in {module_path}: {e}"
            )
        except Exception as e:
            logger.debug(
                f"Error registering {provider.value} provider: {e}"
            )

    logger.info(
        f"Auto-registered {len(ModelFactory.list_providers())} providers: "
        f"{ModelFactory.list_provider_names()}"
    )


# Auto-register providers on module import
try:
    auto_register_providers()
except Exception as e:
    logger.warning(f"⚠️ Error during auto-registration: {e}")
