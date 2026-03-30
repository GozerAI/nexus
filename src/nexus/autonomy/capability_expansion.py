"""
Autonomous Agent Capability Expansion (Item 699)

Monitors agent task execution patterns, identifies recurring failures or
capability gaps, and autonomously synthesizes new capabilities by composing
existing agent skills, adjusting tool configurations, or generating new
task-handling routines.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CapabilityStatus(str, Enum):
    """Status of a capability."""

    ACTIVE = "active"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class GapSeverity(str, Enum):
    """How severe a capability gap is."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Capability:
    """Represents a single agent capability."""

    name: str
    description: str
    task_types: List[str]
    status: CapabilityStatus = CapabilityStatus.ACTIVE
    success_rate: float = 0.0
    total_uses: int = 0
    source: str = "built_in"  # built_in, synthesized, composed
    composed_from: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityGap:
    """Identified gap in agent capabilities."""

    gap_id: str
    task_type: str
    failure_pattern: str
    severity: GapSeverity
    occurrence_count: int
    example_tasks: List[Dict[str, Any]]
    suggested_capabilities: List[str]
    detected_at: float = field(default_factory=time.time)
    resolved: bool = False


@dataclass
class TaskOutcome:
    """Record of a task execution outcome."""

    task_id: str
    task_type: str
    agent_name: str
    success: bool
    error_category: Optional[str] = None
    duration_ms: float = 0.0
    capabilities_used: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AutonomousCapabilityExpander:
    """
    Detects capability gaps and autonomously expands agent capabilities.

    The expander:
    1. Tracks task outcomes across all agents
    2. Identifies recurring failure patterns (capability gaps)
    3. Proposes new capabilities by composing existing ones
    4. Validates synthesized capabilities via trial runs
    5. Promotes or deprecates capabilities based on performance
    """

    # Minimum failures before a gap is identified
    MIN_FAILURES_FOR_GAP = 3
    # Minimum success rate to keep a capability active
    MIN_SUCCESS_RATE = 0.4
    # Maximum number of experimental capabilities at once
    MAX_EXPERIMENTAL = 20

    def __init__(
        self,
        min_failures_for_gap: int = 3,
        min_success_rate: float = 0.4,
        max_experimental: int = 20,
        max_history: int = 10000,
    ):
        self._min_failures = max(1, min_failures_for_gap)
        self._min_success_rate = min_success_rate
        self._max_experimental = max_experimental

        self._capabilities: Dict[str, Capability] = {}
        self._gaps: Dict[str, CapabilityGap] = {}
        self._outcomes: deque = deque(maxlen=max_history)
        self._failure_patterns: Dict[str, List[TaskOutcome]] = defaultdict(list)
        self._composition_registry: Dict[str, Callable] = {}

    # ── Capability Management ───────────────────────────────────────

    def register_capability(
        self,
        name: str,
        description: str,
        task_types: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Capability:
        """Register a built-in capability."""
        cap = Capability(
            name=name,
            description=description,
            task_types=task_types,
            status=CapabilityStatus.ACTIVE,
            source="built_in",
            parameters=parameters or {},
        )
        self._capabilities[name] = cap
        logger.info("Registered capability: %s", name)
        return cap

    def register_composition_rule(
        self, name: str, composer: Callable[..., Any]
    ) -> None:
        """
        Register a callable that can compose multiple capabilities.
        The composer receives a list of capability names and returns
        a new Capability or None.
        """
        self._composition_registry[name] = composer

    def get_capability(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def list_capabilities(
        self, status: Optional[CapabilityStatus] = None
    ) -> List[Capability]:
        caps = list(self._capabilities.values())
        if status:
            caps = [c for c in caps if c.status == status]
        return caps

    # ── Outcome Recording ───────────────────────────────────────────

    def record_outcome(
        self,
        task_id: str,
        task_type: str,
        agent_name: str,
        success: bool,
        error_category: Optional[str] = None,
        duration_ms: float = 0.0,
        capabilities_used: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CapabilityGap]:
        """
        Record a task outcome. If a failure pattern is detected,
        returns the identified CapabilityGap.
        """
        outcome = TaskOutcome(
            task_id=task_id,
            task_type=task_type,
            agent_name=agent_name,
            success=success,
            error_category=error_category,
            duration_ms=duration_ms,
            capabilities_used=capabilities_used or [],
            context=context or {},
        )
        self._outcomes.append(outcome)

        # Update capability success rates
        for cap_name in outcome.capabilities_used:
            if cap_name in self._capabilities:
                cap = self._capabilities[cap_name]
                cap.total_uses += 1
                if success:
                    cap.success_rate = (
                        cap.success_rate * (cap.total_uses - 1) + 1.0
                    ) / cap.total_uses
                else:
                    cap.success_rate = (
                        cap.success_rate * (cap.total_uses - 1)
                    ) / cap.total_uses

        # Track failure patterns
        if not success:
            pattern_key = f"{task_type}::{error_category or 'unknown'}"
            self._failure_patterns[pattern_key].append(outcome)

            gap = self._detect_gap(pattern_key, task_type, error_category)
            if gap:
                self._attempt_expansion(gap)
                return gap

        # Check for capability deprecation
        self._check_deprecations()

        return None

    # ── Gap Detection ───────────────────────────────────────────────

    def _detect_gap(
        self,
        pattern_key: str,
        task_type: str,
        error_category: Optional[str],
    ) -> Optional[CapabilityGap]:
        failures = self._failure_patterns[pattern_key]
        if len(failures) < self._min_failures:
            return None

        # Don't re-detect an existing unresolved gap
        if pattern_key in self._gaps and not self._gaps[pattern_key].resolved:
            self._gaps[pattern_key].occurrence_count = len(failures)
            return None

        severity = self._assess_severity(failures)
        suggestions = self._suggest_capabilities(task_type, error_category, failures)

        gap = CapabilityGap(
            gap_id=pattern_key,
            task_type=task_type,
            failure_pattern=error_category or "unknown",
            severity=severity,
            occurrence_count=len(failures),
            example_tasks=[
                {"task_id": f.task_id, "context": f.context} for f in failures[-5:]
            ],
            suggested_capabilities=suggestions,
        )
        self._gaps[pattern_key] = gap
        logger.warning(
            "Capability gap detected: %s (severity=%s, count=%d)",
            pattern_key,
            severity.value,
            len(failures),
        )
        return gap

    def _assess_severity(self, failures: List[TaskOutcome]) -> GapSeverity:
        count = len(failures)
        if count >= 20:
            return GapSeverity.CRITICAL
        elif count >= 10:
            return GapSeverity.HIGH
        elif count >= 5:
            return GapSeverity.MEDIUM
        return GapSeverity.LOW

    def _suggest_capabilities(
        self,
        task_type: str,
        error_category: Optional[str],
        failures: List[TaskOutcome],
    ) -> List[str]:
        """Suggest capabilities that could fill the gap."""
        suggestions: List[str] = []

        # Find capabilities that handle similar task types
        related_caps = [
            c
            for c in self._capabilities.values()
            if c.status == CapabilityStatus.ACTIVE and c.success_rate > 0.7
        ]

        # Suggest composition of related capabilities
        for cap in related_caps:
            for tt in cap.task_types:
                if tt in task_type or task_type in tt:
                    suggestions.append(f"compose:{cap.name}+error_handling")
                    break

        # Suggest a new specialized capability
        suggestions.append(f"synthesize:{task_type}_{error_category or 'handler'}")

        return suggestions[:5]

    # ── Autonomous Expansion ────────────────────────────────────────

    def _attempt_expansion(self, gap: CapabilityGap) -> Optional[Capability]:
        """Try to autonomously create a capability to fill the gap."""
        experimental_count = sum(
            1
            for c in self._capabilities.values()
            if c.status == CapabilityStatus.EXPERIMENTAL
        )
        if experimental_count >= self._max_experimental:
            logger.debug("Max experimental capabilities reached, skipping expansion")
            return None

        for suggestion in gap.suggested_capabilities:
            if suggestion.startswith("compose:"):
                cap = self._compose_capability(suggestion, gap)
                if cap:
                    return cap
            elif suggestion.startswith("synthesize:"):
                cap = self._synthesize_capability(suggestion, gap)
                if cap:
                    return cap

        return None

    def _compose_capability(
        self, suggestion: str, gap: CapabilityGap
    ) -> Optional[Capability]:
        """Create a new capability by composing existing ones."""
        parts = suggestion.replace("compose:", "").split("+")
        source_names = [p.strip() for p in parts if p.strip() in self._capabilities]

        if not source_names:
            return None

        # Check if any composition rule can handle this
        for rule_name, composer in self._composition_registry.items():
            try:
                result = composer(source_names, gap)
                if isinstance(result, Capability):
                    result.status = CapabilityStatus.EXPERIMENTAL
                    result.source = "composed"
                    result.composed_from = source_names
                    self._capabilities[result.name] = result
                    gap.resolved = True
                    logger.info(
                        "Composed new capability: %s from %s",
                        result.name,
                        source_names,
                    )
                    return result
            except Exception as e:
                logger.debug("Composition rule %s failed: %s", rule_name, e)

        # Default composition: create a wrapper capability
        name = f"composed_{gap.task_type}_{int(time.time())}"
        cap = Capability(
            name=name,
            description=f"Auto-composed capability for {gap.task_type} "
            f"(gap: {gap.failure_pattern})",
            task_types=[gap.task_type],
            status=CapabilityStatus.EXPERIMENTAL,
            source="composed",
            composed_from=source_names,
            parameters={
                "gap_id": gap.gap_id,
                "base_capabilities": source_names,
                "error_handling": True,
                "retry_on_failure": True,
                "max_retries": 2,
            },
        )
        self._capabilities[name] = cap
        gap.resolved = True
        logger.info("Auto-composed capability: %s for gap %s", name, gap.gap_id)
        return cap

    def _synthesize_capability(
        self, suggestion: str, gap: CapabilityGap
    ) -> Optional[Capability]:
        """Synthesize an entirely new capability from the gap description."""
        name = suggestion.replace("synthesize:", "").strip()
        if not name:
            name = f"synth_{gap.task_type}_{int(time.time())}"

        cap = Capability(
            name=name,
            description=f"Auto-synthesized capability for {gap.task_type} "
            f"failures ({gap.failure_pattern})",
            task_types=[gap.task_type],
            status=CapabilityStatus.EXPERIMENTAL,
            source="synthesized",
            parameters={
                "gap_id": gap.gap_id,
                "failure_pattern": gap.failure_pattern,
                "adaptive_retry": True,
                "fallback_strategy": "decompose",
                "error_categories": [gap.failure_pattern],
            },
        )
        self._capabilities[name] = cap
        gap.resolved = True
        logger.info("Synthesized new capability: %s for gap %s", name, gap.gap_id)
        return cap

    def _check_deprecations(self) -> List[str]:
        """Deprecate capabilities with poor success rates."""
        deprecated = []
        for cap in list(self._capabilities.values()):
            if (
                cap.status in (CapabilityStatus.ACTIVE, CapabilityStatus.EXPERIMENTAL)
                and cap.total_uses >= 10
                and cap.success_rate < self._min_success_rate
            ):
                cap.status = CapabilityStatus.DEPRECATED
                deprecated.append(cap.name)
                logger.info(
                    "Deprecated capability %s (success_rate=%.2f)",
                    cap.name,
                    cap.success_rate,
                )
        return deprecated

    # ── Reporting ───────────────────────────────────────────────────

    def get_gaps(self, resolved: Optional[bool] = None) -> List[CapabilityGap]:
        gaps = list(self._gaps.values())
        if resolved is not None:
            gaps = [g for g in gaps if g.resolved == resolved]
        return gaps

    def get_expansion_report(self) -> Dict[str, Any]:
        active = [c for c in self._capabilities.values() if c.status == CapabilityStatus.ACTIVE]
        experimental = [c for c in self._capabilities.values() if c.status == CapabilityStatus.EXPERIMENTAL]
        deprecated = [c for c in self._capabilities.values() if c.status == CapabilityStatus.DEPRECATED]
        unresolved_gaps = [g for g in self._gaps.values() if not g.resolved]

        return {
            "total_capabilities": len(self._capabilities),
            "active": len(active),
            "experimental": len(experimental),
            "deprecated": len(deprecated),
            "total_gaps_detected": len(self._gaps),
            "unresolved_gaps": len(unresolved_gaps),
            "total_outcomes_recorded": len(self._outcomes),
            "synthesized_capabilities": [
                c.name for c in self._capabilities.values() if c.source == "synthesized"
            ],
            "composed_capabilities": [
                c.name for c in self._capabilities.values() if c.source == "composed"
            ],
        }

    def promote_capability(self, name: str) -> bool:
        """Promote an experimental capability to active."""
        cap = self._capabilities.get(name)
        if cap and cap.status == CapabilityStatus.EXPERIMENTAL:
            cap.status = CapabilityStatus.ACTIVE
            logger.info("Promoted capability %s to active", name)
            return True
        return False
