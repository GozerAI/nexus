"""
Nexus Offline Operation System

Self-sufficient offline operation capabilities for Nexus agents and pipelines.

This package provides:
- Offline agent execution with local fallback
- Offline RAG with local vector store
- Offline blueprint execution
- Offline model inference with quantized models
- Offline agent communication via message queues
- Provider auto-discovery and health-based switching
"""

from .agent_execution import OfflineAgentExecutor
from .local_rag import OfflineRAG
from .blueprint_execution import OfflineBlueprintExecutor
from .model_inference import OfflineModelInference
from .agent_communication import OfflineAgentCommunication
from .provider_discovery import ProviderAutoDiscovery
from .provider_health import ProviderHealthManager
from .cost_switching import ProviderCostSwitcher

__all__ = [
    "OfflineAgentExecutor",
    "OfflineRAG",
    "OfflineBlueprintExecutor",
    "OfflineModelInference",
    "OfflineAgentCommunication",
    "ProviderAutoDiscovery",
    "ProviderHealthManager",
    "ProviderCostSwitcher",
]
