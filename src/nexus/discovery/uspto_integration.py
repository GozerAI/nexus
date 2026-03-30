"""
USPTO Integration - Discover patents and patent applications.

Enables Nexus to:
1. Search patents by keyword, inventor, assignee, classification
2. Look up patents/applications by number
3. Track recent patents from AI/tech companies
4. Access patent metadata (title, abstract, claims, dates)

Uses the USPTO PatentsView API:
https://developer.uspto.gov/api-catalog

Environment variables:
- USPTO_USER_AGENT: User-Agent string for USPTO requests
"""

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

# PatentsView API base URL
PATENTSVIEW_BASE = "https://api.patentsview.org/patents/query"
PATENTSVIEW_SINGLE = "https://api.patentsview.org/patents/query"

# Default fields to retrieve from PatentsView
DEFAULT_PATENT_FIELDS = [
    "patent_number",
    "patent_title",
    "patent_abstract",
    "patent_date",
    "patent_type",
    "patent_num_claims",
    "patent_kind",
]

DEFAULT_INVENTOR_FIELDS = [
    "inventor_first_name",
    "inventor_last_name",
]

DEFAULT_ASSIGNEE_FIELDS = [
    "assignee_organization",
    "assignee_type",
]

DEFAULT_CPC_FIELDS = [
    "cpc_group_id",
    "cpc_group_title",
]

# AI/tech companies to track by assignee name
DEFAULT_TRACKED_ASSIGNEES = [
    "Google LLC",
    "Microsoft Corporation",
    "Apple Inc.",
    "Amazon Technologies, Inc.",
    "Meta Platforms, Inc.",
    "NVIDIA Corporation",
    "International Business Machines Corporation",
    "Intel Corporation",
    "OpenAI",
    "Salesforce, Inc.",
]


class USPTOIntegration:
    """
    USPTO integration for discovering patents and patent applications.

    Capabilities:
    - Search patents by keyword, inventor, assignee, CPC classification
    - Look up specific patents by patent number
    - Track recent patents from AI/tech companies
    - Convert patent data to DiscoveredResource objects
    """

    def __init__(
        self,
        resource_discovery: ResourceDiscovery,
        user_agent: Optional[str] = None,
        tracked_assignees: Optional[List[str]] = None,
    ):
        """
        Initialize USPTO integration.

        Args:
            resource_discovery: Main resource discovery system
            user_agent: User-Agent string for USPTO requests
            tracked_assignees: Company names to monitor for new patents
        """
        self.resource_discovery = resource_discovery
        self.user_agent = (
            user_agent
            or os.getenv("USPTO_USER_AGENT")
            or "Nexus/1.0 (nexus-discovery@gozerai.com)"
        )
        self.tracked_assignees = tracked_assignees or DEFAULT_TRACKED_ASSIGNEES

        # Register as USPTO source
        resource_discovery.register_source(ResourceSource.USPTO, self)
        logger.info("USPTOIntegration initialized")

    @property
    def _headers(self) -> Dict[str, str]:
        """Standard headers for USPTO API requests."""
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def discover(self) -> int:
        """
        Discover recent patents from tracked companies.

        Returns:
            Number of new patents discovered
        """
        total_new = 0

        for assignee in self.tracked_assignees:
            patents = await self.search_patents(assignee=assignee, max_results=10)
            for patent in patents:
                resource = self._patent_to_resource(patent)
                if self.resource_discovery.register_resource(resource):
                    total_new += 1

        logger.info(f"USPTO discovery complete: {total_new} new patents")
        return total_new

    def _patent_to_resource(self, patent: Dict[str, Any]) -> DiscoveredResource:
        """Convert a USPTO patent to a DiscoveredResource."""
        patent_number = patent.get("patent_number", "")
        title = patent.get("patent_title", "Untitled Patent")
        abstract = patent.get("patent_abstract", "")
        patent_date = patent.get("patent_date", "")
        assignee = patent.get("assignee_organization", "")
        inventors = patent.get("inventors", "")
        cpc_group = patent.get("cpc_group_id", "")
        patent_type = patent.get("patent_type", "")

        patent_url = (
            f"https://patents.google.com/patent/US{patent_number}"
            if patent_number
            else ""
        )

        tags = [patent_number, "uspto", "patent"]
        if cpc_group:
            tags.append(cpc_group)
        if patent_type:
            tags.append(patent_type)

        description = abstract[:500] if abstract else f"Patent {patent_number}"
        if assignee:
            description = f"{assignee}: {description}"
        if patent_date:
            description += f" (filed {patent_date})"

        return DiscoveredResource(
            id=f"uspto:{patent_number}" if patent_number else f"uspto:{title[:50]}",
            name=f"{title}" + (f" ({patent_number})" if patent_number else ""),
            resource_type=ResourceType.DATASET,
            source=ResourceSource.USPTO,
            description=description,
            url=patent_url,
            capabilities=["patent_data", "intellectual_property", "research"],
            tags=tags,
            use_cases=["patent_analysis", "research", "competitive_intelligence", "ip_landscape"],
            is_available=True,
            quality_score=0.85,  # USPTO patents are authoritative primary sources
            raw_metadata={
                "patent_number": patent_number,
                "patent_title": title,
                "patent_abstract": abstract,
                "patent_date": patent_date,
                "patent_type": patent_type,
                "assignee_organization": assignee,
                "inventors": inventors,
                "cpc_group_id": cpc_group,
                "num_claims": patent.get("patent_num_claims"),
            },
        )

    async def search_patents(
        self,
        keyword: Optional[str] = None,
        inventor: Optional[str] = None,
        assignee: Optional[str] = None,
        cpc_classification: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_results: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Search patents using the PatentsView API.

        Args:
            keyword: Text search across title and abstract
            inventor: Inventor last name filter
            assignee: Assignee organization name filter
            cpc_classification: CPC classification group filter (e.g. "G06N")
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            max_results: Maximum results to return

        Returns:
            List of patent dicts
        """
        # Build query criteria
        criteria = self._build_query_criteria(
            keyword=keyword,
            inventor=inventor,
            assignee=assignee,
            cpc_classification=cpc_classification,
            date_from=date_from,
            date_to=date_to,
        )

        if not criteria:
            logger.warning("No search criteria provided for USPTO search")
            return []

        # Build request body
        fields = (
            DEFAULT_PATENT_FIELDS
            + DEFAULT_ASSIGNEE_FIELDS
            + DEFAULT_INVENTOR_FIELDS
            + DEFAULT_CPC_FIELDS
        )

        request_body = {
            "q": criteria,
            "f": fields,
            "o": {
                "page": 1,
                "per_page": min(max_results, 100),
            },
            "s": [{"patent_date": "desc"}],
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    PATENTSVIEW_BASE,
                    headers=self._headers,
                    json=request_body,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error(
                            "PatentsView API returned %d",
                            response.status,
                        )
                        return []

                    data = await response.json()
                    return self._parse_patent_results(data)

        except Exception as e:
            logger.error("USPTO search error: %s", e)
            return []

    def _build_query_criteria(
        self,
        keyword: Optional[str] = None,
        inventor: Optional[str] = None,
        assignee: Optional[str] = None,
        cpc_classification: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build PatentsView query criteria from search parameters."""
        conditions = []

        if keyword:
            conditions.append({
                "_or": [
                    {"_text_any": {"patent_title": keyword}},
                    {"_text_any": {"patent_abstract": keyword}},
                ]
            })

        if inventor:
            conditions.append({"_text_any": {"inventor_last_name": inventor}})

        if assignee:
            conditions.append({"_contains": {"assignee_organization": assignee}})

        if cpc_classification:
            conditions.append({"_begins": {"cpc_group_id": cpc_classification}})

        if date_from:
            conditions.append({"_gte": {"patent_date": date_from}})

        if date_to:
            conditions.append({"_lte": {"patent_date": date_to}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"_and": conditions}

    def _parse_patent_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse PatentsView API response into patent dicts."""
        patents_raw = data.get("patents", [])
        if not patents_raw:
            return []

        results = []
        for patent in patents_raw:
            # Extract first assignee organization
            assignees = patent.get("assignees", [])
            assignee_org = ""
            if assignees and isinstance(assignees, list) and len(assignees) > 0:
                assignee_org = assignees[0].get("assignee_organization", "") or ""

            # Extract inventors as comma-separated string
            inventors_list = patent.get("inventors", [])
            inventors_str = ""
            if inventors_list and isinstance(inventors_list, list):
                names = []
                for inv in inventors_list:
                    first = inv.get("inventor_first_name", "")
                    last = inv.get("inventor_last_name", "")
                    name = f"{first} {last}".strip()
                    if name:
                        names.append(name)
                inventors_str = ", ".join(names)

            # Extract first CPC group
            cpcs = patent.get("cpcs", [])
            cpc_group = ""
            if cpcs and isinstance(cpcs, list) and len(cpcs) > 0:
                cpc_group = cpcs[0].get("cpc_group_id", "") or ""

            results.append({
                "patent_number": patent.get("patent_number", ""),
                "patent_title": patent.get("patent_title", ""),
                "patent_abstract": patent.get("patent_abstract", ""),
                "patent_date": patent.get("patent_date", ""),
                "patent_type": patent.get("patent_type", ""),
                "patent_num_claims": patent.get("patent_num_claims"),
                "assignee_organization": assignee_org,
                "inventors": inventors_str,
                "cpc_group_id": cpc_group,
            })

        return results

    async def get_patent(self, patent_number: str) -> Optional[Dict[str, Any]]:
        """
        Look up a specific patent by its number.

        Args:
            patent_number: USPTO patent number (e.g. "11234567")

        Returns:
            Patent dict or None if not found
        """
        fields = (
            DEFAULT_PATENT_FIELDS
            + DEFAULT_ASSIGNEE_FIELDS
            + DEFAULT_INVENTOR_FIELDS
            + DEFAULT_CPC_FIELDS
        )

        request_body = {
            "q": {"patent_number": patent_number},
            "f": fields,
            "o": {"page": 1, "per_page": 1},
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    PATENTSVIEW_BASE,
                    headers=self._headers,
                    json=request_body,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error(
                            "PatentsView API returned %d for patent %s",
                            response.status,
                            patent_number,
                        )
                        return None

                    data = await response.json()
                    results = self._parse_patent_results(data)
                    return results[0] if results else None

        except Exception as e:
            logger.error("USPTO patent lookup error for %s: %s", patent_number, e)
            return None

    async def search_by_classification(
        self,
        cpc_group: str,
        max_results: int = 25,
    ) -> List[Dict[str, Any]]:
        """
        Search patents by CPC classification group.

        Common AI/ML classifications:
        - G06N: Computing arrangements based on specific computational models
        - G06F: Electric digital data processing
        - G06V: Image or video recognition
        - G06Q: Data processing systems for business/financial purposes

        Args:
            cpc_group: CPC group prefix (e.g. "G06N" for AI/ML)
            max_results: Maximum results to return

        Returns:
            List of patent dicts
        """
        return await self.search_patents(
            cpc_classification=cpc_group,
            max_results=max_results,
        )
