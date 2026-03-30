"""Task-type model router for the multi-LLM outcome feedback loop.

Routes each step of the outcome feedback loop (decide, evaluate, diagnose, etc.)
to a specific LLM model, ensuring different model families handle different steps
to avoid self-serving bias.

All default models are free-tier OpenRouter models that do NOT train on user data.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default routing table — 6 steps, 5 model families, all free
# ---------------------------------------------------------------------------

_DEFAULT_ROUTING_TABLE: Dict[str, Dict[str, Any]] = {
    "decide": {
        "primary": "qwen/qwen3-30b-a3b:free",
        "fallbacks": ["qwen/qwen3-8b:free"],
        "description": "Planning and generation — creative synthesis",
    },
    "define_success": {
        "primary": "meta-llama/llama-4-maverick:free",
        "fallbacks": ["meta-llama/llama-4-scout:free"],
        "description": "Success criteria — independent from planner",
    },
    "evaluate": {
        "primary": "deepseek/deepseek-r1-0528:free",
        "fallbacks": ["deepseek/deepseek-v3:free"],
        "description": "Judging outcomes — structured comparison",
    },
    "diagnose": {
        "primary": "google/gemma-3-27b-it:free",
        "fallbacks": ["google/gemma-3-12b-it:free"],
        "description": "Root cause analysis — different perspective",
    },
    "prescribe": {
        "primary": "mistralai/mistral-small-3.1-24b-instruct:free",
        "fallbacks": ["mistralai/mistral-nemo:free"],
        "description": "Recommendations — challenges planner assumptions",
    },
    "challenge": {
        "primary": "meta-llama/llama-4-scout:free",
        "fallbacks": ["meta-llama/llama-4-maverick:free"],
        "description": "Devil's advocate — adversarial check",
    },
}


@dataclass
class ModelSelection:
    """Result of a model routing decision."""

    model_id: str
    task_step: str
    fallback_chain: List[str]
    reasoning: str


@dataclass
class _CircuitState:
    """Per-model circuit breaker state."""

    failures: int = 0
    tripped_at: float = 0.0
    is_open: bool = False


class TaskModelRouter:
    """Routes outcome loop steps to specific LLM models.

    Features:
    - Configurable routing table (defaults to 6-step, 5-family mapping)
    - Circuit breaker per model (3 failures → skip to fallback, resets after 300s)
    - Runtime route overrides
    - Selection stats for observability
    """

    FAILURE_THRESHOLD = 3
    RESET_SECONDS = 300.0

    def __init__(self, routing_table: Optional[Dict[str, Dict[str, Any]]] = None):
        import copy
        self._table = routing_table or copy.deepcopy(_DEFAULT_ROUTING_TABLE)
        self._circuits: Dict[str, _CircuitState] = {}
        self._stats: Dict[str, int] = {
            "selections": 0,
            "fallbacks_used": 0,
            "unknown_steps": 0,
        }
        self._step_counts: Dict[str, int] = {}

    def select_model_for_task(
        self,
        task_step: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelSelection:
        """Select the best model for a given outcome loop step.

        Returns the primary model unless its circuit breaker is tripped,
        in which case it walks the fallback chain.
        """
        route = self._table.get(task_step)
        if route is None:
            self._stats["unknown_steps"] += 1
            # Fall back to the 'decide' model for unknown steps
            route = self._table.get("decide", {
                "primary": "qwen/qwen3-30b-a3b:free",
                "fallbacks": [],
                "description": "fallback for unknown step",
            })
            logger.warning("Unknown task step '%s', falling back to decide model", task_step)

        primary = route["primary"]
        fallbacks = list(route.get("fallbacks", []))
        description = route.get("description", "")

        self._stats["selections"] += 1
        self._step_counts[task_step] = self._step_counts.get(task_step, 0) + 1

        # Check circuit breaker for primary
        if not self._is_circuit_open(primary):
            return ModelSelection(
                model_id=primary,
                task_step=task_step,
                fallback_chain=fallbacks,
                reasoning=f"Primary model for '{task_step}': {description}",
            )

        # Primary is tripped — walk fallbacks
        for fb in fallbacks:
            if not self._is_circuit_open(fb):
                self._stats["fallbacks_used"] += 1
                return ModelSelection(
                    model_id=fb,
                    task_step=task_step,
                    fallback_chain=[m for m in fallbacks if m != fb],
                    reasoning=f"Fallback for '{task_step}': primary {primary} circuit open",
                )

        # All models tripped — return primary anyway (circuit may have reset by call time)
        self._stats["fallbacks_used"] += 1
        return ModelSelection(
            model_id=primary,
            task_step=task_step,
            fallback_chain=fallbacks,
            reasoning=f"All models for '{task_step}' have open circuits, retrying primary",
        )

    def record_failure(self, model_id: str) -> None:
        """Record a failure for a model, potentially tripping its circuit breaker."""
        state = self._circuits.setdefault(model_id, _CircuitState())
        state.failures += 1
        if state.failures >= self.FAILURE_THRESHOLD and not state.is_open:
            state.is_open = True
            state.tripped_at = time.monotonic()
            logger.warning(
                "Circuit breaker tripped for %s after %d failures",
                model_id, state.failures,
            )

    def record_success(self, model_id: str) -> None:
        """Record a success for a model, resetting its circuit breaker."""
        state = self._circuits.get(model_id)
        if state:
            state.failures = 0
            state.is_open = False
            state.tripped_at = 0.0

    def _is_circuit_open(self, model_id: str) -> bool:
        """Check if a model's circuit breaker is open (should be skipped)."""
        state = self._circuits.get(model_id)
        if state is None or not state.is_open:
            return False
        # Auto-reset after RESET_SECONDS
        elapsed = time.monotonic() - state.tripped_at
        if elapsed >= self.RESET_SECONDS:
            state.is_open = False
            state.failures = 0
            state.tripped_at = 0.0
            logger.info("Circuit breaker reset for %s after %.0fs", model_id, elapsed)
            return False
        return True

    def override_route(self, task_step: str, model_id: str) -> None:
        """Override the primary model for a task step at runtime."""
        if task_step in self._table:
            old = self._table[task_step]["primary"]
            self._table[task_step]["primary"] = model_id
            logger.info("Route override: %s → %s (was %s)", task_step, model_id, old)
        else:
            self._table[task_step] = {
                "primary": model_id,
                "fallbacks": [],
                "description": "custom override",
            }

    def get_routing_table(self) -> Dict[str, Dict[str, Any]]:
        """Return the current routing table for inspection."""
        return {
            step: {
                "primary": route["primary"],
                "fallbacks": route.get("fallbacks", []),
                "description": route.get("description", ""),
            }
            for step, route in self._table.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return router statistics."""
        circuit_states = {}
        for model_id, state in self._circuits.items():
            circuit_states[model_id] = {
                "failures": state.failures,
                "is_open": state.is_open,
                "tripped_at": state.tripped_at if state.tripped_at else None,
            }
        return {
            **self._stats,
            "step_counts": dict(self._step_counts),
            "circuit_breakers": circuit_states,
            "registered_steps": list(self._table.keys()),
        }
