"""
SEC EDGAR Integration - Discover SEC filings and company data.

Enables Nexus to:
1. Search recent SEC filings (10-K, 10-Q, 8-K, S-1)
2. Look up filings for specific companies by CIK
3. Access XBRL financial facts (revenue, assets, etc.)
4. Track new filings from AI/tech companies

SEC EDGAR fair access policy requires a descriptive User-Agent header
on every request. See: https://www.sec.gov/os/accessing-edgar-data

Environment variables:
- SEC_EDGAR_USER_AGENT: Required User-Agent string (e.g. "MyApp admin@example.com")
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

# EDGAR REST API base URLs
SUBMISSIONS_BASE = "https://data.sec.gov/submissions"
FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
COMPANY_FACTS = "https://data.sec.gov/api/xbrl/companyfacts"

# AI/tech companies to track by CIK
DEFAULT_TRACKED_CIKS = [
    "0001018724",  # Amazon
    "0000320193",  # Apple
    "0001652044",  # Alphabet/Google
    "0000789019",  # Microsoft
    "0001045810",  # NVIDIA
    "0001326801",  # Meta
    "0001559720",  # Palantir
    "0001805833",  # Snowflake
    "0001535527",  # C3.ai
    "0001819974",  # SoundHound AI
]


class SECEdgarIntegration:
    """
    SEC EDGAR integration for discovering public company filings.

    Capabilities:
    - Search filings by form type (10-K, 10-Q, 8-K, S-1)
    - Look up company filings by CIK number
    - Access XBRL company financial facts
    - Track recent filings from AI/tech companies
    """

    def __init__(
        self,
        resource_discovery: ResourceDiscovery,
        user_agent: Optional[str] = None,
        form_types: Optional[List[str]] = None,
        tracked_ciks: Optional[List[str]] = None,
    ):
        """
        Initialize SEC EDGAR integration.

        Args:
            resource_discovery: Main resource discovery system
            user_agent: Required User-Agent string for EDGAR fair access
            form_types: Filing types to discover (default: 10-K, 10-Q, 8-K, S-1)
            tracked_ciks: Company CIKs to monitor for new filings
        """
        self.resource_discovery = resource_discovery
        self.user_agent = (
            user_agent
            or os.getenv("SEC_EDGAR_USER_AGENT")
            or "Nexus/1.0 (nexus-discovery@gozerai.com)"
        )
        self.form_types = form_types or ["10-K", "10-Q", "8-K"]
        self.tracked_ciks = tracked_ciks or DEFAULT_TRACKED_CIKS

        # Register as SEC_EDGAR source
        resource_discovery.register_source(ResourceSource.SEC_EDGAR, self)
        logger.info("SECEdgarIntegration initialized")

    @property
    def _headers(self) -> Dict[str, str]:
        """EDGAR requires a descriptive User-Agent on every request."""
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

    async def discover(self) -> int:
        """
        Discover recent filings from tracked companies.

        Returns:
            Number of new filings discovered
        """
        total_new = 0

        for cik in self.tracked_ciks:
            filings = await self.get_company_filings(cik)
            for filing in filings:
                resource = self._filing_to_resource(filing)
                if self.resource_discovery.register_resource(resource):
                    total_new += 1

        logger.info(f"SEC EDGAR discovery complete: {total_new} new filings")
        return total_new

    def _filing_to_resource(self, filing: Dict[str, Any]) -> DiscoveredResource:
        """Convert an EDGAR filing to a DiscoveredResource."""
        cik = str(filing.get("cik", "")).zfill(10)
        accession = filing.get("accession_number", "").replace("-", "")
        entity = filing.get("entity_name", "Unknown")
        form_type = filing.get("form_type", "")
        filed_date = filing.get("filing_date", "")

        filing_id = f"{cik}-{accession}" if accession else f"{cik}-{form_type}-{filed_date}"

        # Build SEC viewer URL
        accession_dashes = filing.get("accession_number", "")
        filing_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=include&count=10"
        )
        if accession_dashes:
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{accession}-index.htm"
            )

        return DiscoveredResource(
            id=f"sec_edgar:{filing_id}",
            name=f"{entity} — {form_type}" + (f" ({filed_date})" if filed_date else ""),
            resource_type=ResourceType.DATASET,
            source=ResourceSource.SEC_EDGAR,
            description=(
                f"{form_type} filing by {entity}"
                + (f" filed on {filed_date}" if filed_date else "")
                + (f" for period ending {filing.get('report_date', '')}" if filing.get("report_date") else "")
            ),
            url=filing_url,
            capabilities=["financial_data", "regulatory_filing", "research"],
            tags=[form_type, "sec", "edgar", "public_company"],
            use_cases=["financial_analysis", "research", "compliance", "due_diligence"],
            is_available=True,
            quality_score=0.85,  # SEC filings are authoritative primary sources
            raw_metadata={
                "cik": cik,
                "accession_number": filing.get("accession_number"),
                "form_type": form_type,
                "entity_name": entity,
                "filing_date": filed_date,
                "report_date": filing.get("report_date"),
                "primary_document": filing.get("primary_document"),
            },
        )

    async def get_company_filings(
        self,
        cik: str,
        form_type: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get recent filings for a company by CIK.

        Uses the EDGAR submissions API which returns structured JSON
        for each registrant including all recent filings.

        Args:
            cik: SEC Central Index Key (padded or unpadded)
            form_type: Filter by form type (e.g. "10-K")
            max_results: Maximum filings to return

        Returns:
            List of filing dicts
        """
        cik_padded = str(cik).zfill(10)
        url = f"{SUBMISSIONS_BASE}/CIK{cik_padded}.json"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error(
                            "EDGAR submissions API returned %d for CIK %s",
                            response.status,
                            cik_padded,
                        )
                        return []

                    data = await response.json()
                    return self._parse_submissions(data, form_type, max_results)

        except Exception as e:
            logger.error("EDGAR company filings error for CIK %s: %s", cik, e)
            return []

    def _parse_submissions(
        self,
        data: Dict[str, Any],
        form_type: Optional[str],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Parse the EDGAR submissions JSON response."""
        entity_name = data.get("name", "Unknown")
        cik = data.get("cik", "")
        recent = data.get("filings", {}).get("recent", {})

        # EDGAR returns parallel arrays for each field
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        report_dates = recent.get("reportDate", [])

        filings = []
        for i in range(min(len(forms), len(dates))):
            ft = forms[i]
            if form_type and ft != form_type:
                continue
            if self.form_types and ft not in self.form_types:
                continue

            filings.append({
                "cik": cik,
                "entity_name": entity_name,
                "form_type": ft,
                "filing_date": dates[i] if i < len(dates) else "",
                "accession_number": accessions[i] if i < len(accessions) else "",
                "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                "report_date": report_dates[i] if i < len(report_dates) else "",
            })

            if len(filings) >= max_results:
                break

        return filings

    async def get_company_facts(self, cik: str) -> Optional[Dict[str, Any]]:
        """
        Get XBRL company facts (structured financial data).

        Returns standardized financial data points like revenue,
        total assets, net income, etc. across all filings.

        Args:
            cik: SEC Central Index Key

        Returns:
            Company facts dict or None on error
        """
        cik_padded = str(cik).zfill(10)
        url = f"{COMPANY_FACTS}/CIK{cik_padded}.json"

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error(
                            "EDGAR company facts returned %d for CIK %s",
                            response.status,
                            cik_padded,
                        )
                        return None

                    return await response.json()

        except Exception as e:
            logger.error("EDGAR company facts error for CIK %s: %s", cik, e)
            return None

    async def search_filings(
        self,
        query: str = "",
        form_type: str = "10-K",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across EDGAR filings.

        Args:
            query: Search text (company name, keywords, etc.)
            form_type: Filing form type filter
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            max_results: Maximum results to return

        Returns:
            List of filing result dicts
        """
        url = f"{FULL_TEXT_SEARCH}"
        params: Dict[str, Any] = {
            "q": query,
            "forms": form_type,
            "from": 0,
            "size": max_results,
        }

        if date_from:
            params["startdt"] = date_from
        if date_to:
            params["enddt"] = date_to

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    url,
                    headers=self._headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error("EDGAR full-text search returned %d", response.status)
                        return []

                    data = await response.json()
                    return self._parse_search_results(data)

        except Exception as e:
            logger.error("EDGAR search error: %s", e)
            return []

    def _parse_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse EDGAR full-text search JSON response."""
        hits = data.get("hits", {}).get("hits", [])
        results = []

        for hit in hits:
            src = hit.get("_source", {})
            # entity_name can be a list in search results
            entity = src.get("entity_name", "Unknown")
            if isinstance(entity, list):
                entity = entity[0] if entity else "Unknown"

            results.append({
                "accession_number": hit.get("_id", ""),
                "entity_name": entity,
                "form_type": src.get("form_type", ""),
                "filing_date": src.get("file_date", ""),
                "report_date": src.get("period_of_report", ""),
                "cik": str(src.get("cik", "")),
            })

        return results
