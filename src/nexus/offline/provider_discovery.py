"""
Provider Configuration Auto-Discovery (Item 928)

Automatically discovers available model providers by probing known
endpoints, checking for API keys in environment, and detecting
local services (Ollama, LM Studio, etc.).
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Types of model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    COHERE = "cohere"
    OLLAMA = "ollama"
    LM_STUDIO = "lm_studio"
    TOGETHER = "together"
    REPLICATE = "replicate"
    LOCAL = "local"


class DiscoveryMethod(str, Enum):
    """How a provider was discovered."""

    API_KEY = "api_key"
    ENDPOINT_PROBE = "endpoint_probe"
    ENVIRONMENT = "environment"
    MANUAL = "manual"
    CACHED = "cached"


@dataclass
class ProviderEndpoint:
    """Configuration for a discovered provider endpoint."""

    provider: ProviderType
    base_url: str
    api_key: Optional[str] = None
    discovery_method: DiscoveryMethod = DiscoveryMethod.MANUAL
    available: bool = False
    models: List[str] = field(default_factory=list)
    latency_ms: Optional[float] = None
    last_checked: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Known environment variable names for API keys
_ENV_KEY_MAP: Dict[ProviderType, List[str]] = {
    ProviderType.OPENAI: ["OPENAI_API_KEY", "OPENAI_KEY"],
    ProviderType.ANTHROPIC: ["ANTHROPIC_API_KEY", "ANTHROPIC_KEY"],
    ProviderType.GOOGLE: ["GOOGLE_API_KEY", "GOOGLE_AI_KEY", "GEMINI_API_KEY"],
    ProviderType.MISTRAL: ["MISTRAL_API_KEY"],
    ProviderType.COHERE: ["COHERE_API_KEY", "CO_API_KEY"],
    ProviderType.TOGETHER: ["TOGETHER_API_KEY"],
    ProviderType.REPLICATE: ["REPLICATE_API_TOKEN"],
}

# Known default base URLs for providers
_DEFAULT_URLS: Dict[ProviderType, str] = {
    ProviderType.OPENAI: "https://api.openai.com/v1",
    ProviderType.ANTHROPIC: "https://api.anthropic.com/v1",
    ProviderType.GOOGLE: "https://generativelanguage.googleapis.com/v1",
    ProviderType.MISTRAL: "https://api.mistral.ai/v1",
    ProviderType.COHERE: "https://api.cohere.ai/v1",
    ProviderType.OLLAMA: "http://localhost:11434",
    ProviderType.LM_STUDIO: "http://localhost:1234/v1",
    ProviderType.TOGETHER: "https://api.together.xyz/v1",
    ProviderType.REPLICATE: "https://api.replicate.com/v1",
}


class ProviderAutoDiscovery:
    """
    Automatically discovers available model providers.

    Discovery methods:
    1. Environment variable scanning for API keys
    2. Local endpoint probing (Ollama, LM Studio)
    3. Custom probe functions for extended discovery
    4. Cached discovery results for fast startup
    """

    PROBE_TIMEOUT = 5.0  # seconds
    REDISCOVERY_INTERVAL = 600.0  # 10 minutes

    def __init__(
        self,
        probe_timeout: float = 5.0,
        rediscovery_interval: float = 600.0,
    ):
        self._timeout = probe_timeout
        self._rediscovery_interval = rediscovery_interval
        self._endpoints: Dict[ProviderType, ProviderEndpoint] = {}
        self._custom_probes: Dict[ProviderType, Callable[[], bool]] = {}
        self._last_discovery = 0.0
        self._discovery_log: List[Dict[str, Any]] = []

    # ── Discovery ───────────────────────────────────────────────────

    def discover_all(self, force: bool = False) -> Dict[ProviderType, ProviderEndpoint]:
        """
        Run full discovery of all providers.
        Returns dict of discovered (available) providers.
        """
        now = time.time()
        if not force and now - self._last_discovery < self._rediscovery_interval:
            return {k: v for k, v in self._endpoints.items() if v.available}

        logger.info("Starting provider auto-discovery...")

        # 1. Scan environment for API keys
        self._discover_from_environment()

        # 2. Probe local endpoints
        self._discover_local_endpoints()

        # 3. Run custom probes
        self._run_custom_probes()

        self._last_discovery = now
        available = {k: v for k, v in self._endpoints.items() if v.available}

        logger.info(
            "Discovery complete: %d providers available out of %d checked",
            len(available),
            len(self._endpoints),
        )
        return available

    def discover_provider(
        self, provider: ProviderType, force: bool = False
    ) -> Optional[ProviderEndpoint]:
        """Discover a specific provider."""
        existing = self._endpoints.get(provider)
        if (
            existing
            and not force
            and time.time() - existing.last_checked < self._rediscovery_interval
        ):
            return existing if existing.available else None

        # Try env key
        api_key = self._find_api_key(provider)
        base_url = _DEFAULT_URLS.get(provider, "")

        endpoint = ProviderEndpoint(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            last_checked=time.time(),
        )

        if api_key:
            endpoint.available = True
            endpoint.discovery_method = DiscoveryMethod.API_KEY
        elif provider in (ProviderType.OLLAMA, ProviderType.LM_STUDIO, ProviderType.LOCAL):
            endpoint.available = self._probe_endpoint(base_url)
            endpoint.discovery_method = DiscoveryMethod.ENDPOINT_PROBE
        elif provider in self._custom_probes:
            try:
                endpoint.available = self._custom_probes[provider]()
                endpoint.discovery_method = DiscoveryMethod.ENDPOINT_PROBE
            except Exception:
                endpoint.available = False

        self._endpoints[provider] = endpoint
        self._log_discovery(provider, endpoint)
        return endpoint if endpoint.available else None

    def register_custom_probe(
        self, provider: ProviderType, probe_fn: Callable[[], bool]
    ) -> None:
        """Register a custom probe function for a provider."""
        self._custom_probes[provider] = probe_fn

    def register_endpoint(
        self,
        provider: ProviderType,
        base_url: str,
        api_key: Optional[str] = None,
        models: Optional[List[str]] = None,
    ) -> ProviderEndpoint:
        """Manually register a provider endpoint."""
        endpoint = ProviderEndpoint(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            discovery_method=DiscoveryMethod.MANUAL,
            available=True,
            models=models or [],
            last_checked=time.time(),
        )
        self._endpoints[provider] = endpoint
        self._log_discovery(provider, endpoint)
        return endpoint

    # ── Internal Discovery ──────────────────────────────────────────

    def _discover_from_environment(self) -> None:
        """Scan environment variables for API keys."""
        for provider, env_names in _ENV_KEY_MAP.items():
            api_key = self._find_api_key(provider)
            if api_key:
                endpoint = ProviderEndpoint(
                    provider=provider,
                    base_url=_DEFAULT_URLS.get(provider, ""),
                    api_key=api_key,
                    discovery_method=DiscoveryMethod.API_KEY,
                    available=True,
                    last_checked=time.time(),
                )
                self._endpoints[provider] = endpoint
                self._log_discovery(provider, endpoint)

    def _discover_local_endpoints(self) -> None:
        """Probe known local endpoints (Ollama, LM Studio)."""
        local_providers = [ProviderType.OLLAMA, ProviderType.LM_STUDIO]
        for provider in local_providers:
            url = _DEFAULT_URLS.get(provider, "")
            available = self._probe_endpoint(url)
            endpoint = ProviderEndpoint(
                provider=provider,
                base_url=url,
                discovery_method=DiscoveryMethod.ENDPOINT_PROBE,
                available=available,
                last_checked=time.time(),
            )
            self._endpoints[provider] = endpoint
            self._log_discovery(provider, endpoint)

    def _run_custom_probes(self) -> None:
        """Run registered custom probe functions."""
        for provider, probe_fn in self._custom_probes.items():
            try:
                available = probe_fn()
                if provider in self._endpoints:
                    self._endpoints[provider].available = available
                    self._endpoints[provider].last_checked = time.time()
                else:
                    endpoint = ProviderEndpoint(
                        provider=provider,
                        base_url=_DEFAULT_URLS.get(provider, ""),
                        discovery_method=DiscoveryMethod.ENDPOINT_PROBE,
                        available=available,
                        last_checked=time.time(),
                    )
                    self._endpoints[provider] = endpoint
            except Exception as e:
                logger.debug("Custom probe for %s failed: %s", provider.value, e)

    def _find_api_key(self, provider: ProviderType) -> Optional[str]:
        """Search environment for a provider's API key."""
        env_names = _ENV_KEY_MAP.get(provider, [])
        for name in env_names:
            val = os.environ.get(name)
            if val:
                return val
        return None

    def _probe_endpoint(self, url: str) -> bool:
        """
        Probe an endpoint to check if it's alive.
        Uses a lightweight approach to avoid import issues in offline mode.
        """
        if not url:
            return False
        try:
            import urllib.request

            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=self._timeout)
            return resp.status < 500
        except Exception:
            return False

    def _log_discovery(
        self, provider: ProviderType, endpoint: ProviderEndpoint
    ) -> None:
        self._discovery_log.append(
            {
                "provider": provider.value,
                "available": endpoint.available,
                "method": endpoint.discovery_method.value,
                "timestamp": time.time(),
            }
        )

    # ── Reporting ───────────────────────────────────────────────────

    def get_available_providers(self) -> List[ProviderEndpoint]:
        return [e for e in self._endpoints.values() if e.available]

    def get_endpoint(self, provider: ProviderType) -> Optional[ProviderEndpoint]:
        return self._endpoints.get(provider)

    def get_discovery_report(self) -> Dict[str, Any]:
        available = [e for e in self._endpoints.values() if e.available]
        unavailable = [e for e in self._endpoints.values() if not e.available]

        return {
            "total_checked": len(self._endpoints),
            "available": len(available),
            "unavailable": len(unavailable),
            "providers": {
                e.provider.value: {
                    "available": e.available,
                    "method": e.discovery_method.value,
                    "base_url": e.base_url,
                    "has_api_key": e.api_key is not None,
                    "models": e.models,
                    "last_checked": e.last_checked,
                }
                for e in self._endpoints.values()
            },
            "last_discovery": self._last_discovery,
            "discovery_log": self._discovery_log[-20:],
        }
