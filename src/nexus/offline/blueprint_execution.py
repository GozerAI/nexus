"""
Offline Blueprint Execution (Item 770)

Executes blueprints (multi-step task pipelines) entirely offline by
resolving dependencies locally, caching intermediate results, and
using local model inference when cloud providers are unavailable.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class StepStatus(str, Enum):
    """Execution status of a blueprint step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


@dataclass
class BlueprintStep:
    """A single step in a blueprint."""

    step_id: str
    name: str
    step_type: str  # e.g., "llm_call", "transform", "aggregate", "validate"
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # step_ids this depends on
    requires_online: bool = False
    fallback_step_id: Optional[str] = None
    timeout_seconds: float = 60.0
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class Blueprint:
    """A multi-step task pipeline."""

    blueprint_id: str
    name: str
    description: str = ""
    steps: Dict[str, BlueprintStep] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def add_step(self, step: BlueprintStep) -> None:
        self.steps[step.step_id] = step

    def get_execution_order(self) -> List[str]:
        """Topological sort of steps based on dependencies."""
        visited: Set[str] = set()
        order: List[str] = []

        def visit(step_id: str) -> None:
            if step_id in visited:
                return
            visited.add(step_id)
            step = self.steps.get(step_id)
            if step:
                for dep in step.dependencies:
                    visit(dep)
            order.append(step_id)

        for sid in self.steps:
            visit(sid)

        return order


@dataclass
class ExecutionResult:
    """Result of a blueprint execution."""

    blueprint_id: str
    success: bool
    step_results: Dict[str, Any]
    step_statuses: Dict[str, StepStatus]
    total_time_ms: float
    steps_completed: int
    steps_failed: int
    steps_cached: int
    offline_mode: bool


class OfflineBlueprintExecutor:
    """
    Executes blueprints offline by resolving steps locally.

    Capabilities:
    - Topological execution ordering respecting dependencies
    - Step result caching for repeated or incremental runs
    - Local handler registration for each step type
    - Fallback chains when primary execution fails
    - Partial execution resume from cached checkpoints
    """

    def __init__(self, is_online: bool = True):
        self._is_online = is_online
        self._step_handlers: Dict[str, Callable] = {}
        self._result_cache: Dict[str, Any] = {}
        self._execution_history: List[ExecutionResult] = []

    # ── Configuration ───────────────────────────────────────────────

    def set_online(self, online: bool) -> None:
        self._is_online = online

    @property
    def is_online(self) -> bool:
        return self._is_online

    def register_step_handler(
        self, step_type: str, handler: Callable[..., Any]
    ) -> None:
        """
        Register a handler for a step type.
        Handler signature: handler(parameters: Dict, context: Dict) -> Any
        """
        self._step_handlers[step_type] = handler
        logger.info("Registered step handler: %s", step_type)

    # ── Blueprint Creation Helpers ──────────────────────────────────

    def create_blueprint(
        self,
        blueprint_id: str,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Blueprint:
        return Blueprint(
            blueprint_id=blueprint_id,
            name=name,
            description=description,
            metadata=metadata or {},
        )

    # ── Execution ───────────────────────────────────────────────────

    def execute(
        self,
        blueprint: Blueprint,
        initial_context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> ExecutionResult:
        """
        Execute a blueprint, handling offline fallbacks and caching.
        """
        start_time = time.time()
        context = dict(initial_context or {})
        step_results: Dict[str, Any] = {}
        step_statuses: Dict[str, StepStatus] = {}

        completed = 0
        failed = 0
        cached = 0

        execution_order = blueprint.get_execution_order()

        for step_id in execution_order:
            step = blueprint.steps.get(step_id)
            if step is None:
                continue

            # Check cache
            cache_key = self._cache_key(blueprint.blueprint_id, step_id, step.parameters)
            if use_cache and cache_key in self._result_cache:
                step.result = self._result_cache[cache_key]
                step.status = StepStatus.CACHED
                step_results[step_id] = step.result
                step_statuses[step_id] = StepStatus.CACHED
                context[step_id] = step.result
                cached += 1
                continue

            # Check if dependencies are satisfied
            deps_ok = all(
                blueprint.steps.get(d) is not None
                and blueprint.steps[d].status
                in (StepStatus.COMPLETED, StepStatus.CACHED)
                for d in step.dependencies
            )
            if not deps_ok:
                step.status = StepStatus.SKIPPED
                step_statuses[step_id] = StepStatus.SKIPPED
                step.error = "Unmet dependencies"
                continue

            # Check online requirement
            if step.requires_online and not self._is_online:
                # Try fallback
                if step.fallback_step_id and step.fallback_step_id in blueprint.steps:
                    fallback = blueprint.steps[step.fallback_step_id]
                    result = self._execute_step(fallback, context)
                    if fallback.status == StepStatus.COMPLETED:
                        step.result = result
                        step.status = StepStatus.COMPLETED
                        step_results[step_id] = result
                        step_statuses[step_id] = StepStatus.COMPLETED
                        context[step_id] = result
                        completed += 1
                        if use_cache:
                            self._result_cache[cache_key] = result
                        continue

                step.status = StepStatus.FAILED
                step.error = "Requires online but currently offline"
                step_statuses[step_id] = StepStatus.FAILED
                failed += 1
                continue

            # Execute the step
            result = self._execute_step(step, context)
            step_results[step_id] = result
            step_statuses[step_id] = step.status

            if step.status == StepStatus.COMPLETED:
                context[step_id] = result
                completed += 1
                if use_cache:
                    self._result_cache[cache_key] = result
            else:
                failed += 1

        total_ms = (time.time() - start_time) * 1000
        success = failed == 0

        result = ExecutionResult(
            blueprint_id=blueprint.blueprint_id,
            success=success,
            step_results=step_results,
            step_statuses=step_statuses,
            total_time_ms=total_ms,
            steps_completed=completed,
            steps_failed=failed,
            steps_cached=cached,
            offline_mode=not self._is_online,
        )
        self._execution_history.append(result)

        logger.info(
            "Blueprint %s: %d completed, %d cached, %d failed (%.1fms, offline=%s)",
            blueprint.blueprint_id,
            completed,
            cached,
            failed,
            total_ms,
            not self._is_online,
        )
        return result

    def _execute_step(
        self, step: BlueprintStep, context: Dict[str, Any]
    ) -> Optional[Any]:
        """Execute a single step using registered handlers."""
        step.status = StepStatus.RUNNING
        step.started_at = time.time()

        handler = self._step_handlers.get(step.step_type)
        if handler is None:
            step.status = StepStatus.FAILED
            step.error = f"No handler for step type: {step.step_type}"
            step.completed_at = time.time()
            return None

        try:
            result = handler(step.parameters, context)
            step.result = result
            step.status = StepStatus.COMPLETED
            step.completed_at = time.time()
            return result
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.completed_at = time.time()
            logger.warning("Step %s failed: %s", step.step_id, e)
            return None

    def clear_cache(self, blueprint_id: Optional[str] = None) -> int:
        """Clear cached results. If blueprint_id given, only clear for that blueprint."""
        if blueprint_id:
            prefix = f"{blueprint_id}::"
            keys_to_remove = [k for k in self._result_cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._result_cache[k]
            return len(keys_to_remove)
        else:
            count = len(self._result_cache)
            self._result_cache.clear()
            return count

    @staticmethod
    def _cache_key(
        blueprint_id: str, step_id: str, parameters: Dict[str, Any]
    ) -> str:
        import hashlib
        import json

        raw = json.dumps(parameters, sort_keys=True, default=str)
        h = hashlib.md5(raw.encode()).hexdigest()[:12]
        return f"{blueprint_id}::{step_id}::{h}"

    def get_execution_history(self) -> List[ExecutionResult]:
        return list(self._execution_history)

    def get_stats(self) -> Dict[str, Any]:
        total = len(self._execution_history)
        successes = sum(1 for r in self._execution_history if r.success)
        return {
            "total_executions": total,
            "successful": successes,
            "failed": total - successes,
            "cache_entries": len(self._result_cache),
            "registered_handlers": list(self._step_handlers.keys()),
            "is_online": self._is_online,
        }
