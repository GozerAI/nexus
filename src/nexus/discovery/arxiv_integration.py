"""
Arxiv Integration - Discover and access research papers.

Enables Nexus to:
1. Search for AI/ML research papers
2. Access paper abstracts and metadata
3. Track trending research topics
4. Find papers by author or subject
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import aiohttp

from .resource_discovery import (
    DiscoveredResource,
    ResourceDiscovery,
    ResourceSource,
    ResourceType,
)

logger = logging.getLogger(__name__)

# Arxiv API namespace
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivIntegration:
    """
    Arxiv integration for discovering research papers.

    Capabilities:
    - Search papers by query, category, or author
    - Access abstracts and metadata
    - Track recent submissions in AI/ML categories
    - Find related papers
    """

    def __init__(self, resource_discovery: ResourceDiscovery):
        """
        Initialize Arxiv integration.

        Args:
            resource_discovery: Main resource discovery system
        """
        self.resource_discovery = resource_discovery
        self.api_base = "http://export.arxiv.org/api/query"

        # AI/ML related categories
        self.ai_categories = [
            "cs.AI",  # Artificial Intelligence
            "cs.LG",  # Machine Learning
            "cs.CL",  # Computation and Language (NLP)
            "cs.CV",  # Computer Vision
            "cs.NE",  # Neural and Evolutionary Computing
            "stat.ML",  # Machine Learning (Statistics)
        ]

        # Register with resource discovery
        resource_discovery.register_source(ResourceSource.MANUAL, self)

        logger.info("ArxivIntegration initialized")

    async def discover(self) -> int:
        """
        Discover recent papers from AI/ML categories.

        Returns:
            Number of new papers discovered
        """
        total_new = 0

        for category in self.ai_categories:
            papers = await self.search_papers(
                query=f"cat:{category}",
                sort_by="submittedDate",
                sort_order="descending",
                max_results=20,
            )

            for paper in papers:
                resource = self._paper_to_resource(paper)
                if self.resource_discovery.register_resource(resource):
                    total_new += 1

        logger.info(f"Arxiv discovery complete: {total_new} new papers")
        return total_new

    def _paper_to_resource(self, paper: Dict[str, Any]) -> DiscoveredResource:
        """Convert Arxiv paper to DiscoveredResource."""
        paper_id = paper.get("id", "")
        # Extract arxiv ID from URL
        arxiv_id = paper_id.split("/abs/")[-1] if "/abs/" in paper_id else paper_id

        # Calculate quality score based on update frequency
        quality_score = 0.7  # Base score for peer research

        return DiscoveredResource(
            id=f"arxiv:{arxiv_id}",
            name=paper.get("title", "Untitled"),
            resource_type=ResourceType.DATASET,  # Using DATASET for papers
            source=ResourceSource.MANUAL,  # Arxiv as manual source
            description=paper.get("summary", "")[:500],
            url=paper.get("id"),
            documentation_url=paper.get("pdf_url"),
            capabilities=["research", "knowledge"],
            tags=paper.get("categories", [])[:10],
            use_cases=["research", "learning", "reference"],
            is_available=True,
            quality_score=quality_score,
            raw_metadata={
                "authors": paper.get("authors", []),
                "published": paper.get("published"),
                "updated": paper.get("updated"),
                "categories": paper.get("categories", []),
                "primary_category": paper.get("primary_category"),
            },
        )

    async def search_papers(
        self,
        query: str,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        max_results: int = 50,
        start: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search Arxiv papers.

        Args:
            query: Search query (supports Arxiv query syntax)
            sort_by: Sort by (relevance, lastUpdatedDate, submittedDate)
            sort_order: Sort order (ascending, descending)
            max_results: Maximum results
            start: Start index for pagination

        Returns:
            List of paper data
        """
        params = {
            "search_query": query,
            "sortBy": sort_by,
            "sortOrder": sort_order,
            "max_results": max_results,
            "start": start,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    self.api_base,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error(f"Arxiv API returned {response.status}")
                        return []

                    xml_content = await response.text()
                    return self._parse_arxiv_response(xml_content)

        except Exception as e:
            logger.error(f"Arxiv search error: {e}")
            return []

    def _parse_arxiv_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse Arxiv API XML response."""
        papers = []

        try:
            root = ElementTree.fromstring(xml_content)

            for entry in root.findall("atom:entry", ARXIV_NS):
                paper = {
                    "id": self._get_text(entry, "atom:id"),
                    "title": self._clean_text(self._get_text(entry, "atom:title")),
                    "summary": self._clean_text(self._get_text(entry, "atom:summary")),
                    "published": self._get_text(entry, "atom:published"),
                    "updated": self._get_text(entry, "atom:updated"),
                    "authors": [],
                    "categories": [],
                    "pdf_url": None,
                }

                # Get authors
                for author in entry.findall("atom:author", ARXIV_NS):
                    name = self._get_text(author, "atom:name")
                    if name:
                        paper["authors"].append(name)

                # Get categories
                for category in entry.findall("atom:category", ARXIV_NS):
                    term = category.get("term")
                    if term:
                        paper["categories"].append(term)

                # Get primary category
                primary = entry.find("arxiv:primary_category", ARXIV_NS)
                if primary is not None:
                    paper["primary_category"] = primary.get("term")

                # Get PDF link
                for link in entry.findall("atom:link", ARXIV_NS):
                    if link.get("title") == "pdf":
                        paper["pdf_url"] = link.get("href")

                papers.append(paper)

        except Exception as e:
            logger.error(f"Failed to parse Arxiv response: {e}")

        return papers

    def _get_text(self, element: ElementTree.Element, path: str) -> str:
        """Get text from XML element."""
        child = element.find(path, ARXIV_NS)
        return child.text if child is not None and child.text else ""

    def _clean_text(self, text: str) -> str:
        """Clean whitespace from text."""
        return re.sub(r"\s+", " ", text).strip()

    async def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific paper by Arxiv ID."""
        papers = await self.search_papers(query=f"id:{arxiv_id}", max_results=1)
        return papers[0] if papers else None

    async def search_by_author(
        self,
        author: str,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search papers by author name."""
        return await self.search_papers(
            query=f'au:"{author}"',
            sort_by="submittedDate",
            max_results=max_results,
        )

    async def search_by_category(
        self,
        category: str,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search papers by Arxiv category."""
        return await self.search_papers(
            query=f"cat:{category}",
            sort_by="submittedDate",
            max_results=max_results,
        )

    async def get_recent_ai_papers(self, max_results: int = 50) -> List[Dict[str, Any]]:
        """Get recent AI/ML papers."""
        # Build query for all AI categories
        category_query = " OR ".join([f"cat:{cat}" for cat in self.ai_categories])

        return await self.search_papers(
            query=f"({category_query})",
            sort_by="submittedDate",
            sort_order="descending",
            max_results=max_results,
        )
