"""
Nexus Self-Sufficiency Layer

Facade module providing self-sufficient operation capabilities by re-exporting
from the core offline system and adding convenience wrappers for provider
discovery and cost comparison.

This package maps user-facing concepts to implementations:
- Offline agent execution (#752)    -> nexus.offline.agent_execution
- Offline RAG (#760)                -> nexus.offline.local_rag
- Offline blueprint execution (#770)-> nexus.offline.blueprint_execution
- Offline model inference (#780)    -> nexus.offline.model_inference
- Provider discovery (#928)         -> nexus.offline.provider_discovery
- Cost comparison (#960)            -> nexus.offline.cost_switching
"""

from nexus.offline.agent_execution import (
    OfflineAgentExecutor,
    OfflineTask,
    ConnectivityStatus,
    TaskPriority,
    CachedTool,
)
from nexus.offline.local_rag import (
    OfflineRAG,
    LocalVectorStore,
    Document,
    SearchResult,
    RAGResponse,
)
from nexus.offline.blueprint_execution import (
    OfflineBlueprintExecutor,
    Blueprint,
    BlueprintStep,
    StepStatus,
    ExecutionResult,
)
from nexus.offline.model_inference import (
    OfflineModelInference,
    InferenceRequest,
    InferenceResult,
    InferenceBackend,
    QuantizationType,
    SimpleCompletionEngine,
    LocalModel,
)
from nexus.offline.provider_discovery import (
    ProviderAutoDiscovery,
    ProviderType,
    DiscoveryMethod,
    ProviderEndpoint,
)
from nexus.offline.cost_switching import (
    ProviderCostSwitcher,
    CostSample,
    ProviderCostProfile,
)
from nexus.offline.provider_health import ProviderHealthManager

__all__ = [
    # Offline agent execution (#752)
    "OfflineAgentExecutor",
    "OfflineTask",
    "ConnectivityStatus",
    "TaskPriority",
    "CachedTool",
    # Offline RAG (#760)
    "OfflineRAG",
    "LocalVectorStore",
    "Document",
    "SearchResult",
    "RAGResponse",
    # Offline blueprint execution (#770)
    "OfflineBlueprintExecutor",
    "Blueprint",
    "BlueprintStep",
    "StepStatus",
    "ExecutionResult",
    # Offline model inference (#780)
    "OfflineModelInference",
    "InferenceRequest",
    "InferenceResult",
    "InferenceBackend",
    "QuantizationType",
    "SimpleCompletionEngine",
    "LocalModel",
    # Provider discovery (#928)
    "ProviderAutoDiscovery",
    "ProviderType",
    "DiscoveryMethod",
    "ProviderEndpoint",
    # Cost comparison (#960)
    "ProviderCostSwitcher",
    "CostSample",
    "ProviderCostProfile",
    # Provider health
    "ProviderHealthManager",
]
