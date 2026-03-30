"""
Ollama Integration - Discover and manage local AI models.

Enables Nexus to:
1. Discover locally installed Ollama models
2. Pull new models from the Ollama library
3. Monitor model status and resource usage
4. Integrate local models into the ensemble
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


class OllamaIntegration:
    """
    Ollama integration for local model discovery and management.

    Capabilities:
    - List locally installed models
    - Discover available models from Ollama library
    - Pull and manage models
    - Check model status and compatibility
    """

    def __init__(
        self,
        resource_discovery: ResourceDiscovery,
        ollama_host: Optional[str] = None,
    ):
        """
        Initialize Ollama integration.

        Args:
            resource_discovery: Main resource discovery system
            ollama_host: Ollama API host (default: http://localhost:11434)
        """
        self.resource_discovery = resource_discovery
        self.ollama_host = ollama_host or os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        )

        # Popular models to suggest
        self.recommended_models = [
            "llama3.1:8b",
            "llama3.1:70b",
            "qwen2.5:7b",
            "qwen2.5:32b",
            "qwen2.5:72b",
            "mistral:7b",
            "mixtral:8x7b",
            "codellama:13b",
            "deepseek-coder:6.7b",
            "phi3:mini",
            "gemma2:9b",
            "command-r:35b",
            "llava:13b",
            "bakllava:7b",
        ]

        # Register with resource discovery
        resource_discovery.register_source(ResourceSource.MANUAL, self)

        logger.info(f"OllamaIntegration initialized with host: {self.ollama_host}")

    async def discover(self) -> int:
        """
        Discover locally installed Ollama models.

        Returns:
            Number of new models discovered
        """
        if not await self.is_available():
            logger.warning("Ollama is not available")
            return 0

        total_new = 0

        # Get installed models
        models = await self.list_models()
        for model in models:
            resource = self._model_to_resource(model)
            if self.resource_discovery.register_resource(resource):
                total_new += 1

        logger.info(f"Ollama discovery complete: {total_new} local models")
        return total_new

    def _model_to_resource(self, model: Dict[str, Any]) -> DiscoveredResource:
        """Convert Ollama model to DiscoveredResource."""
        name = model.get("name", "")
        size = model.get("size", 0)
        details = model.get("details", {})

        # Calculate quality score based on model family
        quality_score = 0.7  # Base score for local models
        if "llama" in name.lower() or "qwen" in name.lower():
            quality_score = 0.85
        if "70b" in name or "72b" in name or "32b" in name:
            quality_score = 0.9

        # Determine capabilities
        capabilities = ["text_generation", "local", "offline"]
        if "code" in name.lower() or "coder" in name.lower():
            capabilities.append("code_generation")
        if "llava" in name.lower() or "bakllava" in name.lower():
            capabilities.append("vision")

        # Format size
        size_gb = size / (1024 ** 3) if size else 0

        return DiscoveredResource(
            id=f"ollama:{name}",
            name=name,
            resource_type=ResourceType.MODEL,
            source=ResourceSource.MANUAL,
            description=f"Local Ollama model ({size_gb:.1f} GB)",
            url=f"https://ollama.ai/library/{name.split(':')[0]}",
            provider="ollama",
            capabilities=capabilities,
            tags=["local", "ollama", details.get("family", "unknown")],
            use_cases=["local_inference", "offline_use", "privacy"],
            context_length=details.get("context_length"),
            pricing_input=0.0,  # Free - local model
            pricing_output=0.0,
            is_available=True,
            quality_score=quality_score,
            raw_metadata={
                "size_bytes": size,
                "size_gb": round(size_gb, 2),
                "modified_at": model.get("modified_at"),
                "digest": model.get("digest"),
                "details": details,
            },
        )

    async def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    f"{self.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    return response.status == 200
        except Exception:
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List locally installed models.

        Returns:
            List of model information
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    f"{self.ollama_host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("models", [])
                    return []

        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{self.ollama_host}/api/show",
                    json={"name": model_name},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None

    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from the Ollama library.

        Args:
            model_name: Name of model to pull (e.g., "llama3.1:8b")

        Returns:
            True if pull started successfully
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{self.ollama_host}/api/pull",
                    json={"name": model_name, "stream": False},
                    timeout=aiohttp.ClientTimeout(total=3600),  # 1 hour for large models
                ) as response:
                    if response.status == 200:
                        logger.info(f"Successfully pulled model: {model_name}")
                        return True
                    else:
                        error = await response.text()
                        logger.error(f"Failed to pull model: {error}")
                        return False

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    async def delete_model(self, model_name: str) -> bool:
        """Delete a locally installed model."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.delete(
                    f"{self.ollama_host}/api/delete",
                    json={"name": model_name},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    return response.status == 200

        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False

    async def generate(
        self,
        model_name: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Generate a response from a local model.

        Args:
            model_name: Model to use
            prompt: Input prompt
            system: System prompt
            options: Generation options (temperature, etc.)

        Returns:
            Generated response, or None if failed
        """
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
        }

        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{self.ollama_host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response")
                    return None

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    async def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Chat with a local model.

        Args:
            model_name: Model to use
            messages: Chat messages [{"role": "user", "content": "..."}]
            options: Generation options

        Returns:
            Assistant response, or None if failed
        """
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }

        if options:
            payload["options"] = options

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    f"{self.ollama_host}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("message", {}).get("content")
                    return None

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return None

    async def get_running_models(self) -> List[Dict[str, Any]]:
        """Get list of currently running models."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    f"{self.ollama_host}/api/ps",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("models", [])
                    return []

        except Exception as e:
            logger.error(f"Failed to get running models: {e}")
            return []

    def get_recommended_models(self) -> List[str]:
        """Get list of recommended models to install."""
        return self.recommended_models.copy()
