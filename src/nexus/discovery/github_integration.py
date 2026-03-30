"""
GitHub Integration - Discover and access code, datasets, and tools from GitHub.

Enables Nexus to:
1. Search GitHub for relevant repositories
2. Discover datasets and data files
3. Find tools and libraries
4. Access code for analysis and learning
5. Track trending AI/ML repositories
"""

import asyncio
import base64
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


class GitHubIntegration:
    """
    GitHub integration for discovering and accessing resources.

    Capabilities:
    - Search repositories by topic, language, or query
    - Find datasets (CSV, JSON, Parquet files)
    - Discover AI/ML tools and libraries
    - Access file contents for analysis
    - Track trending repositories
    """

    def __init__(
        self,
        resource_discovery: ResourceDiscovery,
        github_token: Optional[str] = None,
    ):
        """
        Initialize GitHub integration.

        Args:
            resource_discovery: Main resource discovery system
            github_token: GitHub personal access token (optional, increases rate limits)
        """
        self.resource_discovery = resource_discovery
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")

        self.api_base = "https://api.github.com"

        # Register with resource discovery
        resource_discovery.register_source(ResourceSource.GITHUB, self)

        logger.info("GitHubIntegration initialized")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Nexus-AI-Platform",
        }
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"
        return headers

    async def discover(self) -> int:
        """
        Discover resources from GitHub.

        Searches for:
        - AI/ML datasets
        - Popular AI tools
        - Trending ML repositories

        Returns:
            Number of new resources discovered
        """
        total_new = 0

        # Discover AI/ML datasets
        count = await self._discover_datasets()
        total_new += count

        # Discover AI tools and libraries
        count = await self._discover_tools()
        total_new += count

        # Discover trending AI repositories
        count = await self._discover_trending()
        total_new += count

        logger.info(f"GitHub discovery complete: {total_new} new resources")
        return total_new

    async def _discover_datasets(self) -> int:
        """Discover dataset repositories."""
        queries = [
            "topic:dataset language:python stars:>100",
            "topic:machine-learning-dataset stars:>50",
            "topic:nlp-dataset stars:>50",
            "topic:computer-vision-dataset stars:>50",
            "awesome-dataset stars:>100",
        ]

        new_count = 0

        for query in queries:
            repos = await self.search_repositories(query, limit=10)
            for repo in repos:
                resource = self._repo_to_resource(repo, ResourceType.DATASET)
                if self.resource_discovery.register_resource(resource):
                    new_count += 1

        return new_count

    async def _discover_tools(self) -> int:
        """Discover AI/ML tools and libraries."""
        queries = [
            "topic:llm topic:tool stars:>500",
            "topic:machine-learning topic:library stars:>1000",
            "topic:deep-learning topic:framework stars:>500",
            "topic:nlp topic:toolkit stars:>200",
            "topic:ai-agents stars:>100",
        ]

        new_count = 0

        for query in queries:
            repos = await self.search_repositories(query, limit=10)
            for repo in repos:
                resource = self._repo_to_resource(repo, ResourceType.TOOL)
                if self.resource_discovery.register_resource(resource):
                    new_count += 1

        return new_count

    async def _discover_trending(self) -> int:
        """Discover trending AI/ML repositories."""
        queries = [
            "topic:artificial-intelligence stars:>1000 pushed:>2024-01-01",
            "topic:large-language-model stars:>500 pushed:>2024-01-01",
            "topic:generative-ai stars:>200 pushed:>2024-01-01",
        ]

        new_count = 0

        for query in queries:
            repos = await self.search_repositories(query, sort="updated", limit=10)
            for repo in repos:
                resource = self._repo_to_resource(repo, ResourceType.CODE_REPO)
                if self.resource_discovery.register_resource(resource):
                    new_count += 1

        return new_count

    def _repo_to_resource(
        self,
        repo: Dict[str, Any],
        resource_type: ResourceType
    ) -> DiscoveredResource:
        """Convert GitHub repository to DiscoveredResource."""
        full_name = repo.get("full_name", "")
        topics = repo.get("topics", [])

        # Calculate quality score based on stars and activity
        stars = repo.get("stargazers_count", 0)
        quality_score = min(1.0, stars / 10000)

        return DiscoveredResource(
            id=f"github:{full_name}",
            name=repo.get("name", ""),
            resource_type=resource_type,
            source=ResourceSource.GITHUB,
            description=repo.get("description"),
            url=repo.get("html_url"),
            documentation_url=repo.get("homepage"),
            capabilities=[],
            tags=topics[:10],
            use_cases=[],
            license=repo.get("license", {}).get("spdx_id") if repo.get("license") else None,
            is_available=not repo.get("archived", False),
            quality_score=quality_score,
            raw_metadata={
                "stars": stars,
                "forks": repo.get("forks_count", 0),
                "language": repo.get("language"),
                "updated_at": repo.get("updated_at"),
                "open_issues": repo.get("open_issues_count", 0),
            },
        )

    async def search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Search GitHub repositories.

        Args:
            query: Search query (supports GitHub search syntax)
            sort: Sort by (stars, forks, updated)
            order: Order (asc, desc)
            limit: Maximum results

        Returns:
            List of repository data
        """
        url = f"{self.api_base}/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": min(limit, 100),
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        logger.error(f"GitHub search failed: {response.status}")
                        return []

                    data = await response.json()
                    return data.get("items", [])[:limit]

        except Exception as e:
            logger.error(f"GitHub search error: {e}")
            return []

    async def get_repository(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """Get repository details."""
        url = f"{self.api_base}/repos/{owner}/{repo}"

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
            logger.error(f"Failed to get repository: {e}")
            return None

    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: str = "main"
    ) -> Optional[str]:
        """
        Get file content from a repository.

        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            ref: Branch or commit ref

        Returns:
            File content as string, or None if not found
        """
        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return None

                    data = await response.json()
                    if data.get("encoding") == "base64":
                        content = base64.b64decode(data.get("content", ""))
                        return content.decode("utf-8")
                    return None

        except Exception as e:
            logger.error(f"Failed to get file content: {e}")
            return None

    async def search_code(
        self,
        query: str,
        language: Optional[str] = None,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Search for code across GitHub.

        Args:
            query: Code search query
            language: Filter by programming language
            limit: Maximum results

        Returns:
            List of code search results
        """
        search_query = query
        if language:
            search_query += f" language:{language}"

        url = f"{self.api_base}/search/code"
        params = {
            "q": search_query,
            "per_page": min(limit, 100),
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    return data.get("items", [])[:limit]

        except Exception as e:
            logger.error(f"Code search error: {e}")
            return []

    async def list_repository_contents(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: str = "main"
    ) -> List[Dict[str, Any]]:
        """List contents of a repository directory."""
        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
        params = {"ref": ref}

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
            logger.error(f"Failed to list repository contents: {e}")
            return []

    async def find_datasets_in_repo(
        self,
        owner: str,
        repo: str,
    ) -> List[Dict[str, Any]]:
        """
        Find dataset files in a repository.

        Searches for common dataset formats:
        - CSV, TSV
        - JSON, JSONL
        - Parquet
        - Arrow
        """
        dataset_extensions = [".csv", ".tsv", ".json", ".jsonl", ".parquet", ".arrow"]
        datasets = []

        try:
            # Search for data files
            for ext in dataset_extensions[:3]:  # Limit to avoid rate limits
                query = f"extension:{ext[1:]} repo:{owner}/{repo}"
                results = await self.search_code(query, limit=10)

                for result in results:
                    datasets.append({
                        "name": result.get("name"),
                        "path": result.get("path"),
                        "repository": f"{owner}/{repo}",
                        "url": result.get("html_url"),
                        "format": ext[1:],
                    })

        except Exception as e:
            logger.error(f"Failed to find datasets: {e}")

        return datasets
