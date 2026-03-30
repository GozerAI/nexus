"""
Unit tests for the Nexus Self-Sufficiency facade layer.

Verifies that self_sufficiency re-exports are correctly wired to the
underlying nexus.offline implementations.
"""

import time
import math
import tempfile
import os
import pytest

from nexus.self_sufficiency import (
    OfflineAgentExecutor, ConnectivityStatus, TaskPriority, CachedTool,
    OfflineRAG, LocalVectorStore, Document, SearchResult, RAGResponse,
    OfflineBlueprintExecutor, Blueprint, BlueprintStep, StepStatus, ExecutionResult,
    OfflineModelInference, InferenceRequest, InferenceResult, InferenceBackend,
    SimpleCompletionEngine, LocalModel,
    ProviderAutoDiscovery, ProviderType, DiscoveryMethod, ProviderEndpoint,
    ProviderCostSwitcher, CostSample, ProviderCostProfile,
    ProviderHealthManager,
)
