"""
Nexus Discovery System - Auto-discovers models, datasets, and tools.

Provides unified access to:
- AI Models (OpenRouter, HuggingFace, Ollama, provider APIs)
- Datasets (HuggingFace, GitHub)
- Tools and Libraries (GitHub, PyPI)
- Code Resources (GitHub repositories)
- Research Papers (Arxiv)
- Web Search (DuckDuckGo, Serper, Brave)
- Local Machine (files, system info, commands)
- Identity/Access/Security (Zuultimate)
- License Management (Vinzy-Engine)
"""

from .resource_discovery import ResourceDiscovery, DiscoveredResource, ResourceType, ResourceSource
from .model_discovery import ModelDiscoveryEngine
from .github_integration import GitHubIntegration
from .huggingface_integration import HuggingFaceIntegration
from .arxiv_integration import ArxivIntegration
from .pypi_integration import PyPIIntegration
from .ollama_integration import OllamaIntegration
from .web_search_integration import WebSearchIntegration
from .local_machine_integration import LocalMachineIntegration
from .zuultimate_integration import ZuultimateIntegration, ZuultimateConfig
from .sec_edgar_integration import SECEdgarIntegration
from .uspto_integration import USPTOIntegration
from .vinzy_integration import VinzyIntegration, VinzyConfig
from .intelligence_sources import (
    TrendscopeDiscoverySource,
    KHDiscoverySource,
    ContentProductionDiscoverySource,
)

__all__ = [
    "ResourceDiscovery",
    "DiscoveredResource",
    "ResourceType",
    "ResourceSource",
    "ModelDiscoveryEngine",
    "GitHubIntegration",
    "HuggingFaceIntegration",
    "ArxivIntegration",
    "PyPIIntegration",
    "OllamaIntegration",
    "WebSearchIntegration",
    "LocalMachineIntegration",
    "ZuultimateIntegration",
    "ZuultimateConfig",
    "SECEdgarIntegration",
    "USPTOIntegration",
    "VinzyIntegration",
    "VinzyConfig",
    "TrendscopeDiscoverySource",
    "KHDiscoverySource",
    "ContentProductionDiscoverySource",
]
