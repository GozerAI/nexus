"""
Priority Engine - Strategic prioritization of work items.

Uses multiple factors to determine what should be worked on next:
- Urgency (deadlines, blockers, dependencies)
- Value (goal alignment, business impact)
- Feasibility (resources, confidence, complexity)
- Learning (historical success, similar tasks)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PriorityFactor(Enum):
    """Factors that influence priority."""
    URGENCY = "urgency"
    VALUE = "value"
    FEASIBILITY = "feasibility"
    DEPENDENCY = "dependency"
    LEARNING = "learning"
    MOMENTUM = "momentum"


@dataclass
class PriorityScore:
    """Detailed priority scoring for an item."""
    item_id: str
    total_score: float
    factors: Dict[PriorityFactor, float] = field(default_factory=dict)
    reasoning: str = ""
    calculated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "total_score": self.total_score,
            "factors": {k.value: v for k, v in self.factors.items()},
            "reasoning": self.reasoning,
            "calculated_at": self.calculated_at.isoformat(),
        }


@dataclass
class PrioritizedItem:
    """An item with its priority information."""
    id: str
    item: Any
    score: PriorityScore
    recommended_action: str
    executor_suggestion: str
    estimated_duration_minutes: float
    estimated_cost_usd: float


class PriorityEngine:
    """
    Strategic Priority Engine.

    Scores and ranks all potential work items to determine
    optimal execution order.

    Factors considered:
    1. URGENCY - Time pressure, deadlines, blocking status
    2. VALUE - Goal alignment, business impact, strategic importance
    3. FEASIBILITY - Resource availability, complexity, confidence
    4. DEPENDENCY - Blocks other work, unblocks goals
    5. LEARNING - Historical success with similar tasks
    6. MOMENTUM - Continue related work, context switching cost
    """

    # Weight configuration for priority factors
    DEFAULT_WEIGHTS = {
        PriorityFactor.URGENCY: 0.25,
        PriorityFactor.VALUE: 0.25,
        PriorityFactor.FEASIBILITY: 0.20,
        PriorityFactor.DEPENDENCY: 0.15,
        PriorityFactor.LEARNING: 0.10,
        PriorityFactor.MOMENTUM: 0.05,
    }

    # Priority value mappings
    PRIORITY_VALUES = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.3,
        "backlog": 0.1,
    }

    def __init__(
        self,
        intelligence=None,
        weights: Dict[PriorityFactor, float] = None,
        learning_system=None,
    ):
        """
        Initialize the Priority Engine.

        Args:
            intelligence: NexusIntelligence for accessing goals/tasks
            weights: Custom factor weights (optional)
            learning_system: PersistentLearning instance for historical data
        """
        self._intel = intelligence
        self._weights = weights or self.DEFAULT_WEIGHTS
        self._learning = learning_system

        # Recent execution context for momentum
        self._recent_items: List[str] = []
        self._current_focus_area: Optional[str] = None

        # Learning cache to avoid repeated database queries
        self._learning_cache: Dict[str, float] = {}
        self._cache_ttl_seconds: int = 300  # 5 minutes

    async def prioritize(self, context: Dict[str, Any]) -> List[PrioritizedItem]:
        """
        Prioritize all work items in the current context.

        Args:
            context: Current state context from COO observation

        Returns:
            List of PrioritizedItem sorted by priority (highest first)
        """
        prioritized = []

        # Score tasks
        for task in context.get("tasks", []):
            try:
                score = await self._score_item(task, context, item_type="task")
                prioritized.append(PrioritizedItem(
                    id=task.id,
                    item=task,
                    score=score,
                    recommended_action=self._recommend_action(task, score),
                    executor_suggestion=self._suggest_executor(task),
                    estimated_duration_minutes=self._estimate_duration(task),
                    estimated_cost_usd=self._estimate_cost(task),
                ))
            except Exception as e:
                logger.error(f"Error scoring task {getattr(task, 'id', 'unknown')}: {e}")

        # Score blockers that need resolution
        for blocker_info in context.get("blockers", []):
            try:
                blocker = blocker_info["blocker"]
                score = await self._score_blocker(blocker_info, context)
                prioritized.append(PrioritizedItem(
                    id=blocker.id,
                    item=blocker_info,
                    score=score,
                    recommended_action="resolve_blocker",
                    executor_suggestion="decision_agent",
                    estimated_duration_minutes=5.0,
                    estimated_cost_usd=0.5,
                ))
            except Exception as e:
                logger.error(f"Error scoring blocker: {e}")

        # Sort by total score descending
        prioritized.sort(key=lambda x: x.score.total_score, reverse=True)

        return prioritized

    async def _score_item(
        self,
        item: Any,
        context: Dict[str, Any],
        item_type: str = "task"
    ) -> PriorityScore:
        """Score a single item across all priority factors."""
        item_id = getattr(item, 'id', str(id(item)))
        factors = {}
        reasoning_parts = []

        # URGENCY
        urgency = self._calculate_urgency(item)
        factors[PriorityFactor.URGENCY] = urgency
        if urgency > 0.7:
            reasoning_parts.append(f"High urgency ({urgency:.2f})")

        # VALUE
        value = await self._calculate_value(item, context)
        factors[PriorityFactor.VALUE] = value
        if value > 0.7:
            reasoning_parts.append(f"High value ({value:.2f})")

        # FEASIBILITY
        feasibility = self._calculate_feasibility(item, context)
        factors[PriorityFactor.FEASIBILITY] = feasibility
        if feasibility < 0.5:
            reasoning_parts.append(f"Low feasibility ({feasibility:.2f})")

        # DEPENDENCY
        dependency = self._calculate_dependency(item, context)
        factors[PriorityFactor.DEPENDENCY] = dependency
        if dependency > 0.7:
            reasoning_parts.append(f"Blocks other work ({dependency:.2f})")

        # LEARNING
        learning = await self._calculate_learning(item)
        factors[PriorityFactor.LEARNING] = learning

        # MOMENTUM
        momentum = self._calculate_momentum(item)
        factors[PriorityFactor.MOMENTUM] = momentum
        if momentum > 0.5:
            reasoning_parts.append("Good momentum")

        # Calculate weighted total
        total = sum(
            factors[factor] * self._weights[factor]
            for factor in PriorityFactor
        )

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Standard priority"

        return PriorityScore(
            item_id=item_id,
            total_score=total,
            factors=factors,
            reasoning=reasoning,
        )

    async def _score_blocker(
        self,
        blocker_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> PriorityScore:
        """Score a blocker resolution task."""
        blocker = blocker_info["blocker"]
        blocker_id = blocker.id

        # Blockers get high urgency boost
        factors = {
            PriorityFactor.URGENCY: 0.9,  # Blockers are urgent
            PriorityFactor.VALUE: 0.7,    # Unblocking has high value
            PriorityFactor.FEASIBILITY: 0.6,
            PriorityFactor.DEPENDENCY: 0.95,  # By definition, blocks work
            PriorityFactor.LEARNING: 0.5,
            PriorityFactor.MOMENTUM: 0.3,
        }

        total = sum(
            factors[factor] * self._weights[factor]
            for factor in PriorityFactor
        )

        return PriorityScore(
            item_id=blocker_id,
            total_score=total,
            factors=factors,
            reasoning=f"Blocker on task: {blocker_info.get('task_title', 'unknown')}",
        )

    def _calculate_urgency(self, item: Any) -> float:
        """Calculate urgency score based on time factors."""
        urgency = 0.5  # Base urgency

        # Priority boost
        priority = getattr(item, 'priority', None)
        if priority:
            priority_value = priority.value if hasattr(priority, 'value') else str(priority)
            urgency = self.PRIORITY_VALUES.get(priority_value, 0.5)

        # Status boost (in_progress items need continuation)
        status = getattr(item, 'status', None)
        if status:
            status_value = status.value if hasattr(status, 'value') else str(status)
            if status_value == 'in_progress':
                urgency = min(1.0, urgency + 0.2)

        # Blocked items have reduced urgency (can't be worked on)
        if getattr(item, 'is_blocked', False):
            urgency *= 0.3

        return urgency

    async def _calculate_value(self, item: Any, context: Dict[str, Any]) -> float:
        """Calculate value score based on goal alignment and impact."""
        value = 0.5  # Base value

        # Check goal alignment
        goals = context.get("goals", [])
        if goals:
            item_text = f"{getattr(item, 'title', '')} {getattr(item, 'description', '')}".lower()

            for goal in goals:
                goal_text = f"{getattr(goal, 'title', '')} {getattr(goal, 'description', '')}".lower()

                # Simple keyword overlap for alignment
                item_words = set(item_text.split())
                goal_words = set(goal_text.split())
                overlap = len(item_words & goal_words)

                if overlap > 2:
                    value = min(1.0, value + 0.2)
                    break

        # Priority affects value perception
        priority = getattr(item, 'priority', None)
        if priority:
            priority_value = priority.value if hasattr(priority, 'value') else str(priority)
            if priority_value in ['critical', 'high']:
                value = min(1.0, value + 0.15)

        return value

    def _calculate_feasibility(self, item: Any, context: Dict[str, Any]) -> float:
        """Calculate feasibility based on resources and complexity."""
        feasibility = 0.7  # Base feasibility

        # Check budget
        budget_remaining = context.get("resources", {}).get("daily_budget_remaining", 50)
        if budget_remaining < 5:
            feasibility *= 0.5
        elif budget_remaining < 20:
            feasibility *= 0.8

        # Blocked items are not feasible
        if getattr(item, 'is_blocked', False):
            feasibility = 0.1

        # Check description length as complexity proxy
        description = getattr(item, 'description', '') or ''
        if len(description) > 500:
            feasibility *= 0.9  # Slightly lower for complex items

        return feasibility

    def _calculate_dependency(self, item: Any, context: Dict[str, Any]) -> float:
        """Calculate how much this item blocks/enables other work."""
        dependency = 0.3  # Base dependency score

        # Check if other tasks reference this one
        item_id = getattr(item, 'id', None)
        if not item_id:
            return dependency

        tasks = context.get("tasks", [])
        for task in tasks:
            # Check if this item is a parent of other tasks
            if getattr(task, 'parent_task_id', None) == item_id:
                dependency = min(1.0, dependency + 0.2)

        return dependency

    async def _calculate_learning(self, item: Any) -> float:
        """
        Calculate learning score from historical performance.

        Uses the PersistentLearning system to find similar past items
        and adjust the score based on their success rates.

        Returns:
            Score from 0.0 (historically fails) to 1.0 (historically succeeds)
        """
        if not self._learning:
            return 0.5  # Default neutral score when no learning system

        item_id = getattr(item, 'id', str(id(item)))

        # Check cache first
        if item_id in self._learning_cache:
            return self._learning_cache[item_id]

        try:
            # Get similar historical outcomes
            similar_outcomes = await self._learning.get_similar_outcomes(item, limit=20)

            if len(similar_outcomes) < 3:
                # Not enough historical data
                return 0.5

            # Calculate success rate from similar items
            successes = sum(1 for r in similar_outcomes if r.success)
            success_rate = successes / len(similar_outcomes)

            # Weight recent outcomes more heavily
            recent_weight = 0.0
            recent_count = 0
            for i, record in enumerate(similar_outcomes[:5]):  # Last 5 outcomes
                weight = 1.0 - (i * 0.15)  # Decay weight for older records
                recent_weight += weight * (1.0 if record.success else 0.0)
                recent_count += weight

            if recent_count > 0:
                recent_success_rate = recent_weight / recent_count
                # Blend overall and recent rates (60% recent, 40% overall)
                learning_score = 0.6 * recent_success_rate + 0.4 * success_rate
            else:
                learning_score = success_rate

            # Get executor-specific performance if available
            executor_suggestion = self._suggest_executor(item)
            executor_stats = await self._learning.get_executor_stats(executor_suggestion)
            if executor_stats.get("total", 0) >= 5:
                executor_rate = executor_stats.get("success_rate", 0.5)
                # Factor in executor performance (20% weight)
                learning_score = 0.8 * learning_score + 0.2 * executor_rate

            # Cache the result
            self._learning_cache[item_id] = learning_score

            return learning_score

        except Exception as e:
            logger.warning(f"Error calculating learning score: {e}")
            return 0.5  # Default on error

    def _calculate_momentum(self, item: Any) -> float:
        """Calculate momentum based on recent work context."""
        momentum = 0.3  # Base momentum

        item_id = getattr(item, 'id', '')

        # Boost if recently worked on similar items
        if item_id in self._recent_items:
            momentum = min(1.0, momentum + 0.3)

        # Check focus area alignment
        if self._current_focus_area:
            item_text = f"{getattr(item, 'title', '')} {getattr(item, 'description', '')}".lower()
            if self._current_focus_area.lower() in item_text:
                momentum = min(1.0, momentum + 0.2)

        return momentum

    def _recommend_action(self, item: Any, score: PriorityScore) -> str:
        """Recommend the best action for an item."""
        if score.total_score >= 0.8:
            return "execute_immediately"
        elif score.total_score >= 0.6:
            return "execute_soon"
        elif score.total_score >= 0.4:
            return "schedule"
        elif score.factors.get(PriorityFactor.FEASIBILITY, 1.0) < 0.3:
            return "resolve_blockers"
        else:
            return "defer"

    def _suggest_executor(self, item: Any) -> str:
        """Suggest the best executor for an item."""
        title = getattr(item, 'title', '').lower()
        description = getattr(item, 'description', '').lower()
        text = f"{title} {description}"

        if any(kw in text for kw in ["research", "find", "discover"]):
            return "research_agent"
        elif any(kw in text for kw in ["write", "create", "draft", "content"]):
            return "content_pipeline"
        elif any(kw in text for kw in ["code", "implement", "build"]):
            return "code_agent"
        elif any(kw in text for kw in ["analyze", "evaluate"]):
            return "analyst_expert"
        else:
            return "expert_router"

    def _estimate_duration(self, item: Any) -> float:
        """Estimate execution duration in minutes."""
        description = getattr(item, 'description', '') or ''

        # Base on description length
        base = 5.0
        if len(description) > 200:
            base = 15.0
        elif len(description) > 500:
            base = 30.0

        # Adjust by priority (higher priority = more thorough)
        priority = getattr(item, 'priority', None)
        if priority:
            priority_value = priority.value if hasattr(priority, 'value') else str(priority)
            if priority_value in ['critical', 'high']:
                base *= 1.5

        return base

    def _estimate_cost(self, item: Any) -> float:
        """Estimate execution cost in USD."""
        duration = self._estimate_duration(item)

        # Rough estimate: $1 per 10 minutes of execution
        return duration * 0.1

    def update_momentum(self, item_id: str, focus_area: str = None):
        """Update momentum context after execution."""
        self._recent_items.append(item_id)

        # Keep only last 10 items
        if len(self._recent_items) > 10:
            self._recent_items = self._recent_items[-10:]

        if focus_area:
            self._current_focus_area = focus_area

    def set_weights(self, weights: Dict[PriorityFactor, float]):
        """Update priority factor weights."""
        self._weights.update(weights)
        logger.info(f"Priority weights updated: {weights}")

    def set_learning_system(self, learning_system):
        """
        Set or update the learning system.

        Args:
            learning_system: PersistentLearning instance
        """
        self._learning = learning_system
        self._learning_cache.clear()
        logger.info("Learning system connected to priority engine")

    def clear_learning_cache(self):
        """Clear the learning score cache."""
        self._learning_cache.clear()
        logger.debug("Learning cache cleared")

    async def get_learning_insights(self, item: Any) -> Dict[str, Any]:
        """
        Get detailed learning insights for an item.

        Args:
            item: The item to analyze

        Returns:
            Dictionary with learning insights
        """
        if not self._learning:
            return {"available": False, "reason": "No learning system configured"}

        try:
            similar_outcomes = await self._learning.get_similar_outcomes(item, limit=10)
            executor_suggestion = self._suggest_executor(item)
            executor_stats = await self._learning.get_executor_stats(executor_suggestion)

            if len(similar_outcomes) < 3:
                return {
                    "available": True,
                    "has_data": False,
                    "reason": "Insufficient historical data",
                    "similar_count": len(similar_outcomes),
                }

            successes = sum(1 for r in similar_outcomes if r.success)
            failures = len(similar_outcomes) - successes

            # Find common error patterns
            error_patterns = {}
            for record in similar_outcomes:
                if not record.success and record.error_message:
                    error_type = record.error_message.split(":")[0][:50]
                    error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

            return {
                "available": True,
                "has_data": True,
                "similar_count": len(similar_outcomes),
                "success_count": successes,
                "failure_count": failures,
                "success_rate": successes / len(similar_outcomes),
                "recommended_executor": executor_suggestion,
                "executor_success_rate": executor_stats.get("success_rate", 0.5),
                "executor_avg_duration": executor_stats.get("avg_duration", 0),
                "common_errors": error_patterns,
                "recent_trend": "improving" if successes > failures else "declining" if failures > successes else "stable",
            }

        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            return {"available": False, "reason": str(e)}
