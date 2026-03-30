"""
HuggingFace Integration - Discover and access models, datasets, and spaces.

Enables Nexus to:
1. Discover AI models (LLMs, vision, audio, etc.)
2. Find and access datasets
3. Explore HuggingFace Spaces (demos and apps)
4. Download and use resources
5. Track trending and popular resources
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


class HuggingFaceIntegration:
    """
    HuggingFace Hub integration for discovering and accessing resources.

    Capabilities:
    - Search and discover models by task
    - Find datasets by domain and format
    - Explore Spaces (interactive demos)
    - Access model cards and documentation
    - Track downloads and popularity
    """

    def __init__(
        self,
        resource_discovery: ResourceDiscovery,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize HuggingFace integration.

        Args:
            resource_discovery: Main resource discovery system
            hf_token: HuggingFace API token (optional, for private resources)
        """
        self.resource_discovery = resource_discovery
        self.hf_token = hf_token or os.environ.get("HUGGINGFACE_API_TOKEN")

        self.api_base = "https://huggingface.co/api"

        # Register with resource discovery
        resource_discovery.register_source(ResourceSource.HUGGINGFACE, self)

        logger.info("HuggingFaceIntegration initialized")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for HuggingFace API requests."""
        headers = {
            "Accept": "application/json",
        }
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    async def discover(self) -> int:
        """
        Discover resources from HuggingFace.

        Returns:
            Number of new resources discovered
        """
        total_new = 0

        # Discover models by task
        count = await self._discover_models()
        total_new += count

        # Discover datasets
        count = await self._discover_datasets()
        total_new += count

        # Discover spaces
        count = await self._discover_spaces()
        total_new += count

        logger.info(f"HuggingFace discovery complete: {total_new} new resources")
        return total_new

    async def _discover_models(self) -> int:
        """Discover models by task type."""
        tasks = [
            "text-generation",
            "text2text-generation",
            "conversational",
            "summarization",
            "translation",
            "question-answering",
            "text-classification",
            "token-classification",
            "fill-mask",
            "sentence-similarity",
            "feature-extraction",
            "image-classification",
            "object-detection",
            "image-segmentation",
            "text-to-image",
            "image-to-text",
            "automatic-speech-recognition",
            "text-to-speech",
        ]

        new_count = 0

        for task in tasks:
            models = await self.search_models(pipeline_tag=task, limit=10)
            for model in models:
                resource = self._model_to_resource(model, task)
                if self.resource_discovery.register_resource(resource):
                    new_count += 1

        return new_count

    async def _discover_datasets(self) -> int:
        """Discover popular datasets."""
        # Search for popular datasets
        datasets = await self.search_datasets(sort="downloads", limit=50)

        new_count = 0
        for dataset in datasets:
            resource = self._dataset_to_resource(dataset)
            if self.resource_discovery.register_resource(resource):
                new_count += 1

        return new_count

    async def _discover_spaces(self) -> int:
        """Discover popular HuggingFace Spaces."""
        spaces = await self.search_spaces(sort="likes", limit=30)

        new_count = 0
        for space in spaces:
            resource = self._space_to_resource(space)
            if self.resource_discovery.register_resource(resource):
                new_count += 1

        return new_count

    def _model_to_resource(
        self,
        model: Dict[str, Any],
        task: str
    ) -> DiscoveredResource:
        """Convert HuggingFace model to DiscoveredResource."""
        model_id = model.get("modelId", model.get("id", ""))
        downloads = model.get("downloads", 0)

        # Calculate quality score based on downloads
        quality_score = min(1.0, downloads / 1000000)

        return DiscoveredResource(
            id=f"huggingface:model:{model_id}",
            name=model_id,
            resource_type=ResourceType.MODEL,
            source=ResourceSource.HUGGINGFACE,
            description=model.get("description"),
            url=f"https://huggingface.co/{model_id}",
            documentation_url=f"https://huggingface.co/{model_id}#model-card",
            capabilities=[task],
            tags=model.get("tags", [])[:10],
            use_cases=[task],
            provider="huggingface",
            license=model.get("license"),
            is_available=not model.get("private", False),
            quality_score=quality_score,
            raw_metadata={
                "downloads": downloads,
                "likes": model.get("likes", 0),
                "library_name": model.get("library_name"),
                "pipeline_tag": model.get("pipeline_tag"),
            },
        )

    def _dataset_to_resource(self, dataset: Dict[str, Any]) -> DiscoveredResource:
        """Convert HuggingFace dataset to DiscoveredResource."""
        dataset_id = dataset.get("id", "")
        downloads = dataset.get("downloads", 0)

        # Calculate quality score
        quality_score = min(1.0, downloads / 100000)

        return DiscoveredResource(
            id=f"huggingface:dataset:{dataset_id}",
            name=dataset_id,
            resource_type=ResourceType.DATASET,
            source=ResourceSource.HUGGINGFACE,
            description=dataset.get("description"),
            url=f"https://huggingface.co/datasets/{dataset_id}",
            capabilities=[],
            tags=dataset.get("tags", [])[:10],
            use_cases=[],
            size=dataset.get("size_categories", [None])[0] if dataset.get("size_categories") else None,
            license=dataset.get("license"),
            is_available=not dataset.get("private", False),
            quality_score=quality_score,
            raw_metadata={
                "downloads": downloads,
                "likes": dataset.get("likes", 0),
                "task_categories": dataset.get("task_categories", []),
            },
        )

    def _space_to_resource(self, space: Dict[str, Any]) -> DiscoveredResource:
        """Convert HuggingFace Space to DiscoveredResource."""
        space_id = space.get("id", "")
        likes = space.get("likes", 0)

        # Calculate quality score
        quality_score = min(1.0, likes / 1000)

        return DiscoveredResource(
            id=f"huggingface:space:{space_id}",
            name=space_id,
            resource_type=ResourceType.SPACE,
            source=ResourceSource.HUGGINGFACE,
            description=space.get("cardData", {}).get("short_description"),
            url=f"https://huggingface.co/spaces/{space_id}",
            capabilities=[],
            tags=space.get("tags", [])[:10],
            use_cases=[],
            is_available=space.get("runtime", {}).get("stage") == "RUNNING",
            quality_score=quality_score,
            raw_metadata={
                "likes": likes,
                "sdk": space.get("sdk"),
                "runtime": space.get("runtime"),
            },
        )

    async def search_models(
        self,
        query: Optional[str] = None,
        pipeline_tag: Optional[str] = None,
        library: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search HuggingFace models.

        Args:
            query: Text search query
            pipeline_tag: Filter by task (text-generation, etc.)
            library: Filter by library (transformers, etc.)
            sort: Sort by (downloads, likes, lastModified)
            limit: Maximum results

        Returns:
            List of model data
        """
        url = f"{self.api_base}/models"
        params = {
            "sort": sort,
            "direction": "-1",
            "limit": limit,
        }

        if query:
            params["search"] = query
        if pipeline_tag:
            params["pipeline_tag"] = pipeline_tag
        if library:
            params["library"] = library

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []

        except Exception as e:
            logger.error(f"Model search error: {e}")
            return []

    async def search_datasets(
        self,
        query: Optional[str] = None,
        task: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search HuggingFace datasets.

        Args:
            query: Text search query
            task: Filter by task category
            sort: Sort by (downloads, likes, lastModified)
            limit: Maximum results

        Returns:
            List of dataset data
        """
        url = f"{self.api_base}/datasets"
        params = {
            "sort": sort,
            "direction": "-1",
            "limit": limit,
        }

        if query:
            params["search"] = query
        if task:
            params["task_categories"] = task

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []

        except Exception as e:
            logger.error(f"Dataset search error: {e}")
            return []

    async def search_spaces(
        self,
        query: Optional[str] = None,
        sdk: Optional[str] = None,
        sort: str = "likes",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search HuggingFace Spaces.

        Args:
            query: Text search query
            sdk: Filter by SDK (gradio, streamlit)
            sort: Sort by (likes, lastModified)
            limit: Maximum results

        Returns:
            List of space data
        """
        url = f"{self.api_base}/spaces"
        params = {
            "sort": sort,
            "direction": "-1",
            "limit": limit,
        }

        if query:
            params["search"] = query
        if sdk:
            params["sdk"] = sdk

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []

        except Exception as e:
            logger.error(f"Spaces search error: {e}")
            return []

    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed model information."""
        url = f"{self.api_base}/models/{model_id}"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None

        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None

    async def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed dataset information."""
        url = f"{self.api_base}/datasets/{dataset_id}"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return None

        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            return None

    async def list_dataset_files(self, dataset_id: str) -> List[Dict[str, Any]]:
        """List files in a dataset."""
        url = f"{self.api_base}/datasets/{dataset_id}/tree/main"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    return []

        except Exception as e:
            logger.error(f"Failed to list dataset files: {e}")
            return []

    async def get_inference_api(
        self,
        model_id: str,
        inputs: Any,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Call the HuggingFace Inference API.

        Args:
            model_id: Model to use
            inputs: Input data
            parameters: Model parameters

        Returns:
            Model output, or None if failed
        """
        if not self.hf_token:
            logger.warning("HuggingFace token required for Inference API")
            return None

        url = f"https://api-inference.huggingface.co/models/{model_id}"
        headers = self._get_headers()

        payload = {"inputs": inputs}
        if parameters:
            payload["parameters"] = parameters

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error = await response.text()
                        logger.error(f"Inference API error: {error}")
                        return None

        except Exception as e:
            logger.error(f"Inference API error: {e}")
            return None
