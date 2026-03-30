"""
PyPI Integration - Discover Python packages and tools.

Enables Nexus to:
1. Search for Python packages
2. Find AI/ML libraries and tools
3. Access package metadata and documentation
4. Track popular and trending packages
"""

import asyncio
import logging
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


class PyPIIntegration:
    """
    PyPI integration for discovering Python packages.

    Capabilities:
    - Search packages by name or keyword
    - Find AI/ML related packages
    - Access package metadata, versions, dependencies
    - Track download statistics
    """

    def __init__(self, resource_discovery: ResourceDiscovery):
        """
        Initialize PyPI integration.

        Args:
            resource_discovery: Main resource discovery system
        """
        self.resource_discovery = resource_discovery
        self.api_base = "https://pypi.org/pypi"
        self.search_base = "https://pypi.org/search"

        # AI/ML related packages to track
        self.ai_packages = [
            "transformers",
            "torch",
            "tensorflow",
            "langchain",
            "openai",
            "anthropic",
            "huggingface-hub",
            "sentence-transformers",
            "accelerate",
            "peft",
            "datasets",
            "evaluate",
            "gradio",
            "streamlit",
            "fastapi",
            "chromadb",
            "pinecone-client",
            "weaviate-client",
            "llama-index",
            "autogen",
            "crewai",
            "instructor",
            "outlines",
            "vllm",
            "ollama",
        ]

        # Register with resource discovery
        resource_discovery.register_source(ResourceSource.PYPI, self)

        logger.info("PyPIIntegration initialized")

    async def discover(self) -> int:
        """
        Discover AI/ML Python packages.

        Returns:
            Number of new packages discovered
        """
        total_new = 0

        for package_name in self.ai_packages:
            package_info = await self.get_package_info(package_name)
            if package_info:
                resource = self._package_to_resource(package_info)
                if self.resource_discovery.register_resource(resource):
                    total_new += 1

        logger.info(f"PyPI discovery complete: {total_new} new packages")
        return total_new

    def _package_to_resource(self, package: Dict[str, Any]) -> DiscoveredResource:
        """Convert PyPI package to DiscoveredResource."""
        info = package.get("info", {})
        name = info.get("name", "")

        # Extract relevant metadata
        classifiers = info.get("classifiers", [])
        keywords = info.get("keywords", "")
        keywords_list = [k.strip() for k in keywords.split(",")] if keywords else []

        # Calculate quality score based on various factors
        quality_score = 0.5  # Base score
        if info.get("project_urls"):
            quality_score += 0.1
        if info.get("documentation_url") or info.get("docs_url"):
            quality_score += 0.1
        if len(classifiers) > 5:
            quality_score += 0.1
        if info.get("license"):
            quality_score += 0.1
        quality_score = min(1.0, quality_score)

        # Get URLs
        project_urls = info.get("project_urls") or {}
        doc_url = (
            info.get("documentation_url")
            or info.get("docs_url")
            or project_urls.get("Documentation")
            or project_urls.get("Docs")
        )

        return DiscoveredResource(
            id=f"pypi:{name}",
            name=name,
            resource_type=ResourceType.LIBRARY,
            source=ResourceSource.PYPI,
            description=info.get("summary", ""),
            url=info.get("project_url") or f"https://pypi.org/project/{name}/",
            documentation_url=doc_url,
            capabilities=["python", "library"],
            tags=keywords_list[:10] + self._extract_tags_from_classifiers(classifiers),
            use_cases=[],
            license=info.get("license"),
            is_available=True,
            quality_score=quality_score,
            raw_metadata={
                "version": info.get("version"),
                "author": info.get("author"),
                "author_email": info.get("author_email"),
                "requires_python": info.get("requires_python"),
                "requires_dist": info.get("requires_dist", [])[:20],
                "classifiers": classifiers[:20],
                "home_page": info.get("home_page"),
            },
        )

    def _extract_tags_from_classifiers(self, classifiers: List[str]) -> List[str]:
        """Extract relevant tags from PyPI classifiers."""
        tags = []
        for classifier in classifiers:
            parts = classifier.split(" :: ")
            if len(parts) >= 2:
                # Extract meaningful parts
                if parts[0] == "Topic":
                    tags.append(parts[-1].lower().replace(" ", "-"))
                elif parts[0] == "Intended Audience":
                    tags.append(parts[-1].lower().replace(" ", "-"))
        return tags[:5]

    async def get_package_info(self, package_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed package information.

        Args:
            package_name: Name of the package

        Returns:
            Package metadata, or None if not found
        """
        url = f"{self.api_base}/{package_name}/json"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        logger.debug(f"Package not found: {package_name}")
                    else:
                        logger.error(f"PyPI API returned {response.status} for {package_name}")
                    return None

        except Exception as e:
            logger.error(f"Failed to get package info for {package_name}: {e}")
            return None

    async def get_package_releases(self, package_name: str) -> List[str]:
        """Get list of release versions for a package."""
        package_info = await self.get_package_info(package_name)
        if package_info:
            return list(package_info.get("releases", {}).keys())
        return []

    async def search_packages(
        self,
        query: str,
        max_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Search for packages (uses web scraping as PyPI lacks search API).

        Note: PyPI doesn't have a proper search API, so this fetches
        package info for known AI/ML packages matching the query.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of matching packages
        """
        # Filter known packages by query
        matching_packages = [
            pkg for pkg in self.ai_packages
            if query.lower() in pkg.lower()
        ]

        results = []
        for package_name in matching_packages[:max_results]:
            package_info = await self.get_package_info(package_name)
            if package_info:
                results.append(package_info)

        return results

    async def get_dependencies(self, package_name: str) -> List[str]:
        """Get package dependencies."""
        package_info = await self.get_package_info(package_name)
        if package_info:
            info = package_info.get("info", {})
            requires_dist = info.get("requires_dist", [])
            # Extract just the package names
            dependencies = []
            for req in requires_dist or []:
                # Parse requirement string (e.g., "numpy>=1.0" -> "numpy")
                pkg_name = req.split(">")[0].split("<")[0].split("=")[0].split("[")[0].split(";")[0].strip()
                if pkg_name:
                    dependencies.append(pkg_name)
            return dependencies
        return []

    async def check_package_exists(self, package_name: str) -> bool:
        """Check if a package exists on PyPI."""
        package_info = await self.get_package_info(package_name)
        return package_info is not None

    async def get_latest_version(self, package_name: str) -> Optional[str]:
        """Get the latest version of a package."""
        package_info = await self.get_package_info(package_name)
        if package_info:
            return package_info.get("info", {}).get("version")
        return None
