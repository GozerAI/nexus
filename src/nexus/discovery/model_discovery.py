"""
Model Discovery Engine - Auto-discovers and self-registers AI models.

This engine discovers models from:
1. OpenRouter (100+ models from multiple providers)
2. OpenAI API (latest models)
3. Anthropic API (Claude models)
4. HuggingFace (open-source models)

Key capability: Self-registration - discovered models are automatically
added to the Nexus model registry for immediate use.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

from .resource_discovery import (
    DiscoveredResource,
    ResourceDiscovery,
    ResourceSource,
    ResourceType,
)

logger = logging.getLogger(__name__)


class ModelDiscoveryEngine:
    """
    Discovers AI models and self-registers them in Nexus.

    This engine enables the system to:
    1. Automatically find new models as they become available
    2. Register them for immediate use
    3. Track model capabilities and performance
    4. Update the registry when models change
    """

    def __init__(
        self,
        resource_discovery: ResourceDiscovery,
        openrouter_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
    ):
        """
        Initialize model discovery engine.

        Args:
            resource_discovery: Main resource discovery system
            openrouter_api_key: OpenRouter API key (discovers 100+ models)
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            huggingface_token: HuggingFace API token
        """
        self.resource_discovery = resource_discovery

        # Load API keys from environment if not provided
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.huggingface_token = huggingface_token or os.environ.get("HUGGINGFACE_API_TOKEN")

        # API endpoints
        self.endpoints = {
            "openrouter": "https://openrouter.ai/api/v1/models",
            "openai": "https://api.openai.com/v1/models",
            "anthropic": "https://api.anthropic.com/v1/models",
            "huggingface": "https://huggingface.co/api/models",
        }

        # Register with resource discovery
        resource_discovery.register_source(ResourceSource.OPENROUTER, self)

        logger.info("ModelDiscoveryEngine initialized")

    async def discover(self) -> int:
        """
        Discover models from all sources.

        Returns:
            Total number of new models discovered
        """
        total_new = 0

        # Discover from each source
        if self.openrouter_api_key:
            count = await self._discover_openrouter()
            total_new += count
            logger.info(f"OpenRouter: {count} new models")

        if self.openai_api_key:
            count = await self._discover_openai()
            total_new += count
            logger.info(f"OpenAI: {count} new models")

        if self.huggingface_token:
            count = await self._discover_huggingface_models()
            total_new += count
            logger.info(f"HuggingFace: {count} new models")

        logger.info(f"Model discovery complete: {total_new} new models")
        return total_new

    def register_persisted_models(self) -> int:
        """Register all persisted model resources into the Nexus MODEL_REGISTRY.

        Called on startup to populate the registry from previously discovered
        resources that were loaded from disk. Without this, discovered models
        are only in ResourceDiscovery but not in the MODEL_REGISTRY that the
        ensemble and /api/models endpoint use.

        Returns count of models registered.
        """
        count = 0
        # Iterate ResourceDiscovery's internal dict-of-dicts
        for source_resources in self.resource_discovery._resources.values():
            for resource in source_resources.values():
                if resource.resource_type == ResourceType.MODEL and resource.provider:
                    try:
                        self._register_in_nexus_registry(resource)
                        count += 1
                    except Exception:
                        pass
        if count:
            logger.info("Registered %d persisted models into MODEL_REGISTRY", count)
        return count

    async def _discover_openrouter(self) -> int:
        """
        Discover models from OpenRouter.

        OpenRouter provides unified access to 100+ models including:
        - OpenAI (GPT-4, GPT-3.5)
        - Anthropic (Claude 3, Claude 2)
        - Google (Gemini, PaLM)
        - Meta (Llama 3, Llama 2)
        - Mistral, Cohere, and many more
        """
        if not self.openrouter_api_key:
            return 0

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://nexus.ai",
        }

        new_count = 0

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    self.endpoints["openrouter"],
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"OpenRouter API returned {response.status}")
                        return 0

                    data = await response.json()
                    models = data.get("data", [])

                    for model_data in models:
                        resource = self._parse_openrouter_model(model_data)
                        if resource and self.resource_discovery.register_resource(resource):
                            new_count += 1
                            # Also register in Nexus model registry
                            self._register_in_nexus_registry(resource)

        except Exception as e:
            logger.error(f"OpenRouter discovery failed: {e}")

        return new_count

    def _parse_openrouter_model(self, data: Dict[str, Any]) -> Optional[DiscoveredResource]:
        """Parse OpenRouter model data into a DiscoveredResource."""
        try:
            model_id = data.get("id", "")
            if not model_id:
                return None

            # Extract pricing
            pricing = data.get("pricing", {})
            pricing_input = float(pricing.get("prompt", 0)) * 1000  # Convert to per 1k tokens
            pricing_output = float(pricing.get("completion", 0)) * 1000

            # Extract capabilities
            capabilities = []
            if data.get("context_length", 0) > 32000:
                capabilities.append("long_context")
            if "vision" in model_id.lower() or data.get("architecture", {}).get("modality") == "multimodal":
                capabilities.append("vision")
            if "code" in model_id.lower() or "codex" in model_id.lower():
                capabilities.append("code_generation")
            capabilities.extend(["text_generation", "chat"])

            # Determine provider from model ID
            provider = model_id.split("/")[0] if "/" in model_id else "unknown"

            return DiscoveredResource(
                id=f"openrouter:{model_id}",
                name=data.get("name", model_id),
                resource_type=ResourceType.MODEL,
                source=ResourceSource.OPENROUTER,
                description=data.get("description"),
                url=f"https://openrouter.ai/models/{model_id}",
                capabilities=capabilities,
                tags=[provider, "llm", "api"],
                provider=provider,
                context_length=data.get("context_length"),
                pricing_input=pricing_input,
                pricing_output=pricing_output,
                is_available=True,
                quality_score=0.7,  # Default score, will be updated with usage
                raw_metadata=data,
            )
        except Exception as e:
            logger.warning(f"Failed to parse OpenRouter model: {e}")
            return None

    async def _discover_openai(self) -> int:
        """Discover models from OpenAI API."""
        if not self.openai_api_key:
            return 0

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
        }

        new_count = 0

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    self.endpoints["openai"],
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"OpenAI API returned {response.status}")
                        return 0

                    data = await response.json()
                    models = data.get("data", [])

                    for model_data in models:
                        model_id = model_data.get("id", "")
                        # Filter to relevant models
                        if not any(x in model_id for x in ["gpt", "text-embedding", "whisper", "tts", "dall-e"]):
                            continue

                        resource = DiscoveredResource(
                            id=f"openai:{model_id}",
                            name=model_id,
                            resource_type=ResourceType.MODEL,
                            source=ResourceSource.OPENAI,
                            provider="openai",
                            capabilities=self._infer_openai_capabilities(model_id),
                            tags=["openai", "api"],
                            is_available=True,
                            quality_score=0.8,
                            raw_metadata=model_data,
                        )

                        if self.resource_discovery.register_resource(resource):
                            new_count += 1
                            self._register_in_nexus_registry(resource)

        except Exception as e:
            logger.error(f"OpenAI discovery failed: {e}")

        return new_count

    def _infer_openai_capabilities(self, model_id: str) -> List[str]:
        """Infer capabilities from OpenAI model ID."""
        capabilities = []

        if "gpt-4" in model_id:
            capabilities.extend(["text_generation", "chat", "reasoning", "code_generation"])
            if "vision" in model_id or "turbo" in model_id:
                capabilities.append("vision")
            if "32k" in model_id or "128k" in model_id:
                capabilities.append("long_context")
        elif "gpt-3.5" in model_id:
            capabilities.extend(["text_generation", "chat", "code_generation"])
        elif "embedding" in model_id:
            capabilities.append("embeddings")
        elif "whisper" in model_id:
            capabilities.append("speech_to_text")
        elif "tts" in model_id:
            capabilities.append("text_to_speech")
        elif "dall-e" in model_id:
            capabilities.append("image_generation")

        return capabilities

    async def _discover_huggingface_models(self) -> int:
        """Discover top models from HuggingFace."""
        new_count = 0

        # Search for popular LLM models
        search_queries = [
            "text-generation",
            "text2text-generation",
            "conversational",
        ]

        headers = {}
        if self.huggingface_token:
            headers["Authorization"] = f"Bearer {self.huggingface_token}"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                for task in search_queries:
                    url = f"{self.endpoints['huggingface']}?pipeline_tag={task}&sort=downloads&limit=20"

                    async with session.get(
                        url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status != 200:
                            continue

                        models = await response.json()

                        for model_data in models:
                            model_id = model_data.get("modelId", "")
                            if not model_id:
                                continue

                            resource = DiscoveredResource(
                                id=f"huggingface:{model_id}",
                                name=model_id,
                                resource_type=ResourceType.MODEL,
                                source=ResourceSource.HUGGINGFACE,
                                description=model_data.get("description"),
                                url=f"https://huggingface.co/{model_id}",
                                provider="huggingface",
                                capabilities=[task, "open_source"],
                                tags=model_data.get("tags", [])[:10],
                                license=model_data.get("license"),
                                is_available=True,
                                quality_score=min(0.9, model_data.get("downloads", 0) / 1000000),
                                raw_metadata=model_data,
                            )

                            if self.resource_discovery.register_resource(resource):
                                new_count += 1

        except Exception as e:
            logger.error(f"HuggingFace model discovery failed: {e}")

        return new_count

    def _register_in_nexus_registry(self, resource: DiscoveredResource) -> None:
        """
        Register discovered model in Nexus's model registry.

        This enables the model to be used immediately by the ensemble system.
        """
        try:
            from nexus.providers.adapters.base import ModelCapability, ModelInfo, ModelSize
            from nexus.providers.adapters.registry import register_model
            from nexus.providers.ensemble.types import ModelProvider

            # Map provider string to enum
            provider_map = {
                "openai": ModelProvider.OPENAI,
                "anthropic": ModelProvider.ANTHROPIC,
                "google": ModelProvider.GOOGLE,
                "mistral": ModelProvider.MISTRAL,
                "meta-llama": ModelProvider.TOGETHER,
                "cohere": ModelProvider.COHERE,
            }

            provider = provider_map.get(resource.provider, ModelProvider.OPENAI)

            # Map capabilities
            capability_map = {
                "text_generation": ModelCapability.TEXT_GENERATION,
                "chat": ModelCapability.TEXT_GENERATION,
                "code_generation": ModelCapability.CODE_GENERATION,
                "vision": ModelCapability.VISION,
                "embeddings": ModelCapability.EMBEDDINGS,
                "long_context": ModelCapability.LONG_CONTEXT,
                "reasoning": ModelCapability.REASONING,
            }

            capabilities = [
                capability_map[c] for c in resource.capabilities
                if c in capability_map
            ]

            # Enrich with model profile metadata
            profile_kwargs = {}
            try:
                from nexus.providers.adapters.model_profiles import get_model_profile
                model_name = resource.id.split(":")[-1] if ":" in resource.id else resource.id
                profile = get_model_profile(model_name)
                if profile:
                    profile_kwargs = {
                        "strengths": list(profile.strengths),
                        "weaknesses": list(profile.weaknesses),
                        "specializations": list(profile.specializations),
                        "not_recommended_for": list(profile.not_recommended_for),
                        "programming_languages": list(profile.programming_languages),
                        "natural_languages": list(profile.natural_languages),
                        "weak_natural_languages": list(profile.weak_natural_languages),
                        "quality_tier": profile.quality_tier,
                    }
            except ImportError:
                pass

            # Create ModelInfo
            model_info = ModelInfo(
                name=resource.id.split(":")[-1],  # Remove prefix
                display_name=resource.name,
                provider=provider,
                size=ModelSize.MEDIUM,  # Default
                context_window=resource.context_length or 4096,
                max_output_tokens=4096,
                capabilities=capabilities or [ModelCapability.TEXT_GENERATION],
                cost_per_1k_input=resource.pricing_input or 0.0,
                cost_per_1k_output=resource.pricing_output or 0.0,
                supported=True,
                description=resource.description or "",
                use_cases=resource.use_cases,
                **profile_kwargs,
            )

            register_model(model_info)
            logger.debug(f"Registered model in Nexus registry: {resource.name}")

        except Exception as e:
            logger.warning(f"Failed to register model in Nexus registry: {e}")

    async def search_models(
        self,
        query: str,
        capabilities: Optional[List[str]] = None,
        max_price: Optional[float] = None,
        min_context: Optional[int] = None,
    ) -> List[DiscoveredResource]:
        """
        Search for models matching criteria.

        Args:
            query: Search query
            capabilities: Required capabilities
            max_price: Maximum price per 1k tokens
            min_context: Minimum context length

        Returns:
            List of matching models
        """
        models = self.resource_discovery.get_models(
            query=query,
            capabilities=capabilities,
        )

        # Apply additional filters
        if max_price is not None:
            models = [m for m in models if (m.pricing_input or 0) <= max_price]

        if min_context is not None:
            models = [m for m in models if (m.context_length or 0) >= min_context]

        return models
