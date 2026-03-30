"""Legacy orchestration layer for earlier Nexus-brain experiments.

This package remains available for compatibility with older integrations, but
new deployments should treat Nexus as shared infrastructure rather than as the
system that owns organizational strategy.
"""

from nexus.coo.core import AutonomousCOO, COOConfig, ExecutionMode
from nexus.coo.priority_engine import PriorityEngine, PrioritizedItem, PriorityScore
from nexus.coo.learning import PersistentLearning, LearningRecord, OutcomeType
from nexus.coo.executor import AutonomousExecutor, ExecutionResult, ExecutionStatus
from nexus.coo.csuite_bridge import CSuiteBridgeListener, CSuiteBridgeConfig
from nexus.coo.executive_registry import (
    ExecutiveInfo,
    EXECUTIVE_CAPABILITIES,
    get_executive_for_task,
    get_executive_for_text,
    get_executive_info,
    get_all_executives,
    get_executive_codes,
    get_capabilities_for_executive,
    get_all_task_types,
    find_executives_for_capability,
)

__all__ = [
    "AutonomousCOO",
    "COOConfig",
    "ExecutionMode",
    "PriorityEngine",
    "PrioritizedItem",
    "PriorityScore",
    "PersistentLearning",
    "LearningRecord",
    "OutcomeType",
    "AutonomousExecutor",
    "ExecutionResult",
    "ExecutionStatus",
    # csuite Bridge
    "CSuiteBridgeListener",
    "CSuiteBridgeConfig",
    # Executive Registry
    "ExecutiveInfo",
    "EXECUTIVE_CAPABILITIES",
    "get_executive_for_task",
    "get_executive_for_text",
    "get_executive_info",
    "get_all_executives",
    "get_executive_codes",
    "get_capabilities_for_executive",
    "get_all_task_types",
    "find_executives_for_capability",
]
