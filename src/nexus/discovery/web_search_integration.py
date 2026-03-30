"""
Web Search Integration - Search the web for information.

Enables Nexus to:
1. Search the web using multiple search providers
2. Access real-time information
3. Find documentation and tutorials
4. Research current events and trends
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import aiohttp

from .resource_discovery import (
    DiscoveredResource,
    ResourceDiscovery,
    ResourceSource,
    ResourceType,
)

logger = logging.getLogger(__name__)


class WebSearchIntegration:
    """
    Web search integration for real-time information retrieval.

    Supports multiple search backends:
    - DuckDuckGo (free, no API key)
    - Serper (Google results, requires API key)
    - Brave Search (requires API key)
    """

    def __init__(
        self,
        resource_discovery: ResourceDiscovery,
        serper_api_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
    ):
        """
        Initialize web search integration.

        Args:
            resource_discovery: Main resource discovery system
            serper_api_key: Serper API key for Google results
            brave_api_key: Brave Search API key
        """
        self.resource_discovery = resource_discovery
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY")
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")

        # API endpoints
        self.endpoints = {
            "duckduckgo": "https://api.duckduckgo.com/",
            "serper": "https://google.serper.dev/search",
            "brave": "https://api.search.brave.com/res/v1/web/search",
        }

        logger.info("WebSearchIntegration initialized")

    async def search(
        self,
        query: str,
        num_results: int = 10,
        provider: str = "auto",
    ) -> List[Dict[str, Any]]:
        """
        Search the web.

        Args:
            query: Search query
            num_results: Number of results
            provider: Search provider (auto, duckduckgo, serper, brave)

        Returns:
            List of search results
        """
        if provider == "auto":
            # Try providers in order of preference
            if self.serper_api_key:
                return await self._search_serper(query, num_results)
            elif self.brave_api_key:
                return await self._search_brave(query, num_results)
            else:
                return await self._search_duckduckgo(query, num_results)
        elif provider == "serper":
            return await self._search_serper(query, num_results)
        elif provider == "brave":
            return await self._search_brave(query, num_results)
        else:
            return await self._search_duckduckgo(query, num_results)

    async def _search_duckduckgo(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo Instant Answer API."""
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    self.endpoints["duckduckgo"],
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    results = []

                    # Process instant answer
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", query),
                            "url": data.get("AbstractURL", ""),
                            "snippet": data.get("Abstract", ""),
                            "source": "duckduckgo",
                        })

                    # Process related topics
                    for topic in data.get("RelatedTopics", [])[:num_results - len(results)]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("Text", "")[:100],
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", ""),
                                "source": "duckduckgo",
                            })

                    return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []

    async def _search_serper(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search using Serper (Google results)."""
        if not self.serper_api_key:
            logger.warning("Serper API key not configured")
            return []

        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "num": num_results,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    self.endpoints["serper"],
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error(f"Serper API returned {response.status}")
                        return []

                    data = await response.json()
                    results = []

                    # Process organic results
                    for item in data.get("organic", [])[:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "source": "serper",
                            "position": item.get("position"),
                        })

                    return results

        except Exception as e:
            logger.error(f"Serper search error: {e}")
            return []

    async def _search_brave(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search using Brave Search API."""
        if not self.brave_api_key:
            logger.warning("Brave API key not configured")
            return []

        headers = {
            "X-Subscription-Token": self.brave_api_key,
            "Accept": "application/json",
        }

        params = {
            "q": query,
            "count": num_results,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(
                    self.endpoints["brave"],
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error(f"Brave API returned {response.status}")
                        return []

                    data = await response.json()
                    results = []

                    # Process web results
                    web_results = data.get("web", {}).get("results", [])
                    for item in web_results[:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("description", ""),
                            "source": "brave",
                        })

                    return results

        except Exception as e:
            logger.error(f"Brave search error: {e}")
            return []

    async def search_news(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for news articles."""
        if self.serper_api_key:
            return await self._search_serper_news(query, num_results)
        else:
            # Fall back to regular search with news-related terms
            return await self.search(f"{query} news latest", num_results)

    async def _search_serper_news(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search news using Serper."""
        if not self.serper_api_key:
            return []

        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "num": num_results,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    "https://google.serper.dev/news",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    results = []

                    for item in data.get("news", [])[:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "source": "serper_news",
                            "date": item.get("date"),
                            "source_name": item.get("source"),
                        })

                    return results

        except Exception as e:
            logger.error(f"Serper news search error: {e}")
            return []

    async def search_images(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for images."""
        if not self.serper_api_key:
            logger.warning("Image search requires Serper API key")
            return []

        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "num": num_results,
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    "https://google.serper.dev/images",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    results = []

                    for item in data.get("images", [])[:num_results]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "image_url": item.get("imageUrl", ""),
                            "thumbnail_url": item.get("thumbnailUrl", ""),
                            "source": "serper_images",
                        })

                    return results

        except Exception as e:
            logger.error(f"Serper image search error: {e}")
            return []

    def _is_url_safe(self, url: str) -> tuple[bool, str]:
        """
        Validate URL to prevent SSRF attacks.

        Checks:
        - Only http/https protocols allowed
        - No private/internal IP addresses
        - No localhost/loopback addresses
        - No cloud metadata endpoints

        Returns:
            Tuple of (is_safe, error_message)
        """
        import ipaddress
        import socket
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)

            # Check protocol
            if parsed.scheme not in ('http', 'https'):
                return False, f"Invalid protocol: {parsed.scheme}"

            hostname = parsed.hostname
            if not hostname:
                return False, "No hostname in URL"

            # Block localhost and loopback
            localhost_patterns = ['localhost', '127.0.0.1', '::1', '0.0.0.0']
            if hostname.lower() in localhost_patterns:
                return False, "Localhost URLs not allowed"

            # Block cloud metadata endpoints
            metadata_hosts = [
                '169.254.169.254',  # AWS/Azure/GCP metadata
                'metadata.google.internal',
                'metadata.goog',
            ]
            if hostname.lower() in metadata_hosts:
                return False, "Cloud metadata endpoints not allowed"

            # Try to resolve hostname and check if it's a private IP
            try:
                resolved_ips = socket.getaddrinfo(hostname, None)
                for family, _, _, _, sockaddr in resolved_ips:
                    ip_str = sockaddr[0]
                    try:
                        ip = ipaddress.ip_address(ip_str)
                        if ip.is_private or ip.is_loopback or ip.is_link_local:
                            return False, f"Private/internal IP address not allowed: {ip_str}"
                    except ValueError:
                        continue
            except socket.gaierror:
                # Can't resolve - could be intentional for SSRF
                logger.warning(f"Could not resolve hostname: {hostname}")

            return True, ""

        except Exception as e:
            return False, f"URL validation error: {e}"

    async def fetch_page_content(
        self,
        url: str,
        max_length: int = 10000,
    ) -> Optional[str]:
        """
        Fetch and extract text content from a web page.

        Includes SSRF protection:
        - Only http/https protocols
        - No private/internal IP addresses
        - No cloud metadata endpoints
        - Redirects are disabled

        Args:
            url: URL to fetch
            max_length: Maximum content length

        Returns:
            Page text content, or None if failed
        """
        # SSRF protection: validate URL before fetching
        is_safe, error = self._is_url_safe(url)
        if not is_safe:
            logger.warning(f"URL blocked by SSRF protection: {error}")
            return None

        try:
            # Disable redirects to prevent SSRF via redirect
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; NexusBot/1.0)",
                }
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                    allow_redirects=False,  # Disable redirects for SSRF protection
                ) as response:
                    # Handle redirects safely
                    if response.status in (301, 302, 303, 307, 308):
                        redirect_url = response.headers.get('Location')
                        if redirect_url:
                            # Validate redirect target
                            is_safe, error = self._is_url_safe(redirect_url)
                            if not is_safe:
                                logger.warning(f"Redirect blocked by SSRF protection: {error}")
                                return None
                            # Allow one level of redirect
                            return await self._fetch_content_internal(redirect_url, max_length)
                        return None

                    if response.status != 200:
                        return None

                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        return None

                    html = await response.text()

                    # Basic HTML to text conversion
                    import re
                    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()

                    return text[:max_length]

        except Exception as e:
            logger.error(f"Failed to fetch page content: {e}")
            return None

    async def _fetch_content_internal(self, url: str, max_length: int) -> Optional[str]:
        """Internal fetch helper for handling redirects."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                headers = {"User-Agent": "Mozilla/5.0 (compatible; NexusBot/1.0)"}
                async with session.get(
                    url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                    allow_redirects=False,
                ) as response:
                    if response.status != 200:
                        return None

                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" not in content_type:
                        return None

                    html = await response.text()
                    import re
                    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text).strip()
                    return text[:max_length]
        except Exception as e:
            logger.error(f"Failed to fetch redirect content: {e}")
            return None
