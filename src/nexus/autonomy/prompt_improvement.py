"""
Autonomous Prompt Engineering Improvement (Item 707)

Tracks prompt effectiveness across tasks and models, identifies patterns
in prompt success/failure, and autonomously refines prompt templates
through variation testing, structural adjustments, and context optimization.
"""

import logging
import time
import statistics
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class PromptVariationType(str, Enum):
    """Types of prompt variations."""

    ORIGINAL = "original"
    RESTRUCTURED = "restructured"
    CONTEXT_ENRICHED = "context_enriched"
    SIMPLIFIED = "simplified"
    FEW_SHOT_ADDED = "few_shot_added"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_ADJUSTED = "role_adjusted"
    FORMAT_CHANGED = "format_changed"


@dataclass
class PromptTemplate:
    """A tracked prompt template with performance data."""

    template_id: str
    template_text: str
    task_type: str
    variation_type: PromptVariationType
    model_target: Optional[str] = None  # None = universal
    avg_score: float = 0.0
    total_uses: int = 0
    success_count: int = 0
    active: bool = True
    parent_template_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptOutcome:
    """Outcome of a prompt execution."""

    template_id: str
    task_type: str
    model_name: str
    quality_score: float
    response_length: int
    latency_ms: float
    success: bool
    issues: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class PromptRefinement:
    """Record of a prompt refinement action."""

    refinement_id: str
    original_template_id: str
    new_template_id: str
    variation_type: PromptVariationType
    reason: str
    expected_improvement: float
    timestamp: float = field(default_factory=time.time)


class AutonomousPromptEngineer:
    """
    Autonomously improves prompt templates by monitoring outcomes
    and generating optimized variations.

    The engineer:
    1. Tracks prompt template performance across tasks and models
    2. Identifies underperforming templates
    3. Generates variations using proven improvement strategies
    4. A/B tests variations against originals
    5. Promotes winning variations and retires losers
    """

    MIN_USES_FOR_EVALUATION = 5
    IMPROVEMENT_THRESHOLD = 0.08
    MAX_ACTIVE_VARIATIONS = 10

    # Structural improvement patterns applied during refinement
    IMPROVEMENT_PATTERNS = {
        "add_step_by_step": {
            "prefix": "Let's approach this step by step.\n\n",
            "type": PromptVariationType.CHAIN_OF_THOUGHT,
        },
        "add_role": {
            "prefix": "You are an expert assistant. ",
            "type": PromptVariationType.ROLE_ADJUSTED,
        },
        "add_format_instruction": {
            "suffix": "\n\nProvide your answer in a structured format with clear sections.",
            "type": PromptVariationType.FORMAT_CHANGED,
        },
        "simplify": {
            "transform": "simplify",
            "type": PromptVariationType.SIMPLIFIED,
        },
        "add_context_frame": {
            "prefix": "Given the following context, ",
            "suffix": "\n\nBe thorough and precise in your response.",
            "type": PromptVariationType.CONTEXT_ENRICHED,
        },
    }

    def __init__(
        self,
        min_uses: int = 5,
        improvement_threshold: float = 0.08,
        max_variations: int = 10,
        max_history: int = 10000,
    ):
        self._min_uses = max(2, min_uses)
        self._improvement_threshold = improvement_threshold
        self._max_variations = max_variations

        self._templates: Dict[str, PromptTemplate] = {}
        self._outcomes: deque = deque(maxlen=max_history)
        self._refinements: deque = deque(maxlen=5000)
        self._task_best: Dict[str, str] = {}  # task_type -> best template_id

    # ── Template Management ─────────────────────────────────────────

    def register_template(
        self,
        template_text: str,
        task_type: str,
        template_id: Optional[str] = None,
        model_target: Optional[str] = None,
        variation_type: PromptVariationType = PromptVariationType.ORIGINAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptTemplate:
        """Register a prompt template for tracking."""
        if template_id is None:
            template_id = self._generate_id(template_text, task_type)

        template = PromptTemplate(
            template_id=template_id,
            template_text=template_text,
            task_type=task_type,
            variation_type=variation_type,
            model_target=model_target,
            metadata=metadata or {},
        )
        self._templates[template_id] = template
        logger.info("Registered prompt template: %s for %s", template_id, task_type)
        return template

    def get_best_template(
        self, task_type: str, model_name: Optional[str] = None
    ) -> Optional[PromptTemplate]:
        """Get the best-performing template for a task type."""
        candidates = [
            t
            for t in self._templates.values()
            if t.task_type == task_type
            and t.active
            and (t.model_target is None or t.model_target == model_name)
        ]

        if not candidates:
            return None

        # Prefer templates with enough observations
        evaluated = [t for t in candidates if t.total_uses >= self._min_uses]
        if evaluated:
            return max(evaluated, key=lambda t: t.avg_score)

        # Fall back to any active template
        return candidates[0]

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        return self._templates.get(template_id)

    # ── Outcome Recording ───────────────────────────────────────────

    def record_outcome(
        self,
        template_id: str,
        task_type: str,
        model_name: str,
        quality_score: float,
        response_length: int = 0,
        latency_ms: float = 0.0,
        success: bool = True,
        issues: Optional[List[str]] = None,
    ) -> Optional[PromptRefinement]:
        """
        Record a prompt execution outcome and potentially trigger refinement.
        """
        outcome = PromptOutcome(
            template_id=template_id,
            task_type=task_type,
            model_name=model_name,
            quality_score=max(0.0, min(1.0, quality_score)),
            response_length=response_length,
            latency_ms=latency_ms,
            success=success,
            issues=issues or [],
        )
        self._outcomes.append(outcome)

        # Update template stats
        template = self._templates.get(template_id)
        if template:
            template.total_uses += 1
            if success:
                template.success_count += 1
            template.avg_score = (
                template.avg_score * (template.total_uses - 1) + quality_score
            ) / template.total_uses

        # Update best template tracking
        self._update_best_template(task_type)

        # Check if refinement is needed
        return self._maybe_refine(template_id, task_type, model_name)

    # ── Autonomous Refinement ───────────────────────────────────────

    def _maybe_refine(
        self,
        template_id: str,
        task_type: str,
        model_name: str,
    ) -> Optional[PromptRefinement]:
        template = self._templates.get(template_id)
        if not template:
            return None
        if template.total_uses < self._min_uses:
            return None

        # Count active variations for this task type
        active_variations = sum(
            1
            for t in self._templates.values()
            if t.task_type == task_type
            and t.active
            and t.variation_type != PromptVariationType.ORIGINAL
        )
        if active_variations >= self._max_variations:
            # Prune underperformers first
            self._prune_variations(task_type)
            return None

        # Check if template is underperforming
        if template.avg_score >= 0.8:
            return None  # Already performing well

        # Generate a variation
        return self._generate_refinement(template, model_name)

    def _generate_refinement(
        self, template: PromptTemplate, model_name: str
    ) -> Optional[PromptRefinement]:
        """Generate a refined version of an underperforming template."""
        # Select the best improvement pattern that hasn't been tried
        tried_types = {
            t.variation_type
            for t in self._templates.values()
            if t.parent_template_id == template.template_id and t.active
        }

        best_pattern = None
        best_pattern_name = None
        for pattern_name, pattern in self.IMPROVEMENT_PATTERNS.items():
            if pattern["type"] not in tried_types:
                best_pattern = pattern
                best_pattern_name = pattern_name
                break

        if best_pattern is None:
            return None

        # Apply the pattern
        new_text = self._apply_pattern(template.template_text, best_pattern)
        if new_text == template.template_text:
            return None

        new_template = self.register_template(
            template_text=new_text,
            task_type=template.task_type,
            model_target=model_name,
            variation_type=best_pattern["type"],
            metadata={
                "parent_template": template.template_id,
                "pattern_applied": best_pattern_name,
            },
        )
        new_template.parent_template_id = template.template_id

        refinement = PromptRefinement(
            refinement_id=f"ref_{new_template.template_id}",
            original_template_id=template.template_id,
            new_template_id=new_template.template_id,
            variation_type=best_pattern["type"],
            reason=f"Template avg_score={template.avg_score:.2f} below threshold, "
            f"applying {best_pattern_name}",
            expected_improvement=self._improvement_threshold,
        )
        self._refinements.append(refinement)

        logger.info(
            "Generated refinement %s: %s -> %s (%s)",
            refinement.refinement_id,
            template.template_id,
            new_template.template_id,
            best_pattern_name,
        )
        return refinement

    def _apply_pattern(
        self, text: str, pattern: Dict[str, Any]
    ) -> str:
        """Apply an improvement pattern to template text."""
        if "transform" in pattern:
            if pattern["transform"] == "simplify":
                return self._simplify_text(text)
            return text

        result = text
        if "prefix" in pattern:
            result = pattern["prefix"] + result
        if "suffix" in pattern:
            result = result + pattern["suffix"]
        return result

    def _simplify_text(self, text: str) -> str:
        """Simplify a prompt by removing redundancies and trimming."""
        lines = text.strip().split("\n")
        seen: Set[str] = set()
        unique_lines: List[str] = []
        for line in lines:
            normalized = line.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_lines.append(line)
            elif not normalized:
                unique_lines.append(line)

        # Remove excessive whitespace
        result = "\n".join(unique_lines).strip()
        # Collapse multiple blank lines
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")
        return result

    def _prune_variations(self, task_type: str) -> int:
        """Deactivate poorly-performing variations."""
        variations = [
            t
            for t in self._templates.values()
            if t.task_type == task_type
            and t.active
            and t.variation_type != PromptVariationType.ORIGINAL
            and t.total_uses >= self._min_uses
        ]

        pruned = 0
        for v in variations:
            if v.avg_score < 0.3 or (
                v.total_uses >= self._min_uses * 2 and v.success_count / v.total_uses < 0.3
            ):
                v.active = False
                pruned += 1
                logger.info("Pruned variation %s (avg_score=%.2f)", v.template_id, v.avg_score)

        return pruned

    def _update_best_template(self, task_type: str) -> None:
        """Update the best template reference for a task type."""
        best = self.get_best_template(task_type)
        if best:
            self._task_best[task_type] = best.template_id

    def _generate_id(self, text: str, task_type: str) -> str:
        h = hashlib.md5(f"{text}:{task_type}".encode()).hexdigest()[:10]
        return f"tpl_{task_type}_{h}"

    # ── Reporting ───────────────────────────────────────────────────

    def get_improvement_report(self) -> Dict[str, Any]:
        """Generate a comprehensive prompt improvement report."""
        active = [t for t in self._templates.values() if t.active]
        inactive = [t for t in self._templates.values() if not t.active]

        task_groups: Dict[str, List[PromptTemplate]] = defaultdict(list)
        for t in active:
            task_groups[t.task_type].append(t)

        task_reports = {}
        for task_type, templates in task_groups.items():
            evaluated = [t for t in templates if t.total_uses >= self._min_uses]
            if evaluated:
                best = max(evaluated, key=lambda t: t.avg_score)
                avg_score = statistics.mean(t.avg_score for t in evaluated)
            else:
                best = templates[0] if templates else None
                avg_score = 0.0

            task_reports[task_type] = {
                "total_templates": len(templates),
                "evaluated": len(evaluated),
                "avg_score": round(avg_score, 3),
                "best_template": best.template_id if best else None,
                "best_score": round(best.avg_score, 3) if best else 0.0,
            }

        return {
            "total_templates": len(self._templates),
            "active": len(active),
            "inactive": len(inactive),
            "total_outcomes": len(self._outcomes),
            "total_refinements": len(self._refinements),
            "task_reports": task_reports,
            "recent_refinements": [
                {
                    "id": r.refinement_id,
                    "original": r.original_template_id,
                    "new": r.new_template_id,
                    "type": r.variation_type.value,
                    "reason": r.reason,
                }
                for r in list(self._refinements)[-10:]
            ],
        }

    def list_templates(
        self, task_type: Optional[str] = None, active_only: bool = True
    ) -> List[PromptTemplate]:
        templates = list(self._templates.values())
        if task_type:
            templates = [t for t in templates if t.task_type == task_type]
        if active_only:
            templates = [t for t in templates if t.active]
        return templates
