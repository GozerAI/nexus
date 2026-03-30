"""Legacy AutonomousCOO orchestration layer.

This module reflects an earlier architecture where Nexus attempted to own
product-level strategy. It is retained for compatibility, but Nexus now
defaults to shared infrastructure consumed by an external control plane.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Operating mode for the COO."""
    AUTONOMOUS = "autonomous"      # Full autonomous operation
    SUPERVISED = "supervised"      # Execute but report all actions
    APPROVAL = "approval"          # Request approval for each action
    OBSERVE = "observe"            # Only observe and recommend, no action
    PAUSED = "paused"              # Temporarily halted


class COOState(Enum):
    """Current state of the COO."""
    IDLE = "idle"
    OBSERVING = "observing"
    PRIORITIZING = "prioritizing"
    DELEGATING = "delegating"
    EXECUTING = "executing"
    LEARNING = "learning"
    WAITING_APPROVAL = "waiting_approval"


@dataclass
class COOConfig:
    """Configuration for the Autonomous COO."""
    # Operation mode
    mode: ExecutionMode = ExecutionMode.SUPERVISED

    # Timing
    observation_interval_seconds: int = 30      # How often to check state
    priority_refresh_seconds: int = 60          # How often to re-prioritize
    max_concurrent_executions: int = 3          # Max parallel tasks

    # Autonomy thresholds
    auto_execute_confidence: float = 0.9        # Min confidence for auto-exec
    auto_execute_max_priority: str = "medium"   # Won't auto-exec above this
    require_approval_tags: List[str] = field(default_factory=lambda: [
        "financial", "legal", "public", "destructive", "irreversible"
    ])

    # Learning
    learning_enabled: bool = True
    learning_db_path: str = "data/coo_learning.db"

    # Reporting
    report_all_decisions: bool = True
    notification_callback: Optional[Callable] = None

    # Resource limits
    daily_budget_usd: float = 50.0
    max_tokens_per_task: int = 100000


@dataclass
class COOStatus:
    """Current status of the COO."""
    state: COOState
    mode: ExecutionMode
    uptime_seconds: float
    total_tasks_executed: int
    successful_executions: int
    failed_executions: int
    pending_approvals: int
    current_executions: List[str]
    last_observation: Optional[datetime]
    last_decision: Optional[str]
    learning_effectiveness: float
    daily_spend_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "mode": self.mode.value,
            "uptime_seconds": self.uptime_seconds,
            "total_tasks_executed": self.total_tasks_executed,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "pending_approvals": self.pending_approvals,
            "current_executions": self.current_executions,
            "last_observation": self.last_observation.isoformat() if self.last_observation else None,
            "last_decision": self.last_decision,
            "learning_effectiveness": self.learning_effectiveness,
            "daily_spend_usd": self.daily_spend_usd,
        }


class AutonomousCOO:
    """
    Legacy autonomous orchestration controller.

    Historically this class acted like the strategic brain of Nexus by:
    1. OBSERVES - Continuously monitors goals, tasks, resources, and outcomes
    2. PRIORITIZES - Determines what should be worked on next
    3. DELEGATES - Routes work to appropriate agents, pipelines, or experts
    4. EXECUTES - Runs work autonomously with configurable approval levels
    5. LEARNS - Tracks outcomes and improves decision-making over time

    New deployments should treat this as an optional compatibility layer
    sitting on top of shared Nexus infrastructure, not as the owner of
    organizational strategy.
    """

    def __init__(self, intelligence=None, config: COOConfig = None):
        """
        Initialize the Autonomous COO.

        Args:
            intelligence: NexusIntelligence instance for accessing goals/tasks
            config: COO configuration
        """
        self.config = config or COOConfig()
        self._intel = intelligence

        # State
        self._state = COOState.IDLE
        self._mode = self.config.mode
        self._running = False
        self._started_at: Optional[datetime] = None

        # Metrics
        self._total_executed = 0
        self._successful = 0
        self._failed = 0
        self._daily_spend = 0.0
        self._last_observation: Optional[datetime] = None
        self._last_decision: Optional[str] = None

        # Current work
        self._current_executions: Dict[str, asyncio.Task] = {}
        self._pending_approvals: Dict[str, Any] = {}
        self._execution_queue: asyncio.Queue = asyncio.Queue()

        # Components (initialized lazily)
        self._priority_engine = None
        self._learning = None
        self._executor = None
        self._csuite_bridge = None

        # Main loop task and background thread
        self._main_loop_task: Optional[asyncio.Task] = None
        self._background_thread = None
        self._background_loop = None

        # Event callbacks
        self._on_decision: List[Callable] = []
        self._on_execution_complete: List[Callable] = []
        self._on_approval_needed: List[Callable] = []

        logger.info(f"AutonomousCOO initialized in {self._mode.value} mode")

    async def initialize(self):
        """Initialize COO components."""
        from nexus.coo.priority_engine import PriorityEngine
        from nexus.coo.learning import PersistentLearning
        from nexus.coo.executor import AutonomousExecutor
        from nexus.coo.csuite_bridge import CSuiteBridgeListener, CSuiteBridgeConfig

        # Initialize priority engine
        self._priority_engine = PriorityEngine(self._intel)

        # Initialize learning system
        if self.config.learning_enabled:
            self._learning = PersistentLearning(self.config.learning_db_path)
            await self._learning.initialize()

        # Initialize executor
        self._executor = AutonomousExecutor(
            intelligence=self._intel,
            learning=self._learning,
            config=self.config
        )
        await self._executor.initialize()

        # Initialize csuite bridge listener (for communication with CoS)
        self._csuite_bridge = CSuiteBridgeListener(self)
        # Note: Bridge connection is started separately via connect_to_csuite()
        # to allow for custom configuration

        logger.info("AutonomousCOO components initialized")

    async def start(self) -> bool:
        """Start the COO main loop. Returns True on success."""
        logger.info("COO start() called")

        if self._running:
            logger.warning("COO already running")
            return True  # Already running is success

        self._running = True
        self._started_at = datetime.now()
        self._state = COOState.OBSERVING

        # Start main loop in a background thread with its own event loop
        # This is necessary because the GUI's async bridge creates temporary event loops
        import threading

        def run_main_loop():
            logger.info("COO background thread starting...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._background_loop = loop
            try:
                loop.run_until_complete(self._main_loop())
            except Exception as e:
                logger.error(f"COO main loop error: {e}")
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
                self._background_loop = None
                logger.info("COO background thread ended")

        self._background_thread = threading.Thread(target=run_main_loop, daemon=True, name="COO-MainLoop")
        self._background_thread.start()

        logger.info(f"AutonomousCOO started in {self._mode.value} mode")

        # Notify (optional, don't fail if notification fails)
        try:
            await self._notify(f"COO started in {self._mode.value} mode")
        except Exception:
            pass

        return True

    async def stop(self):
        """Stop the COO gracefully."""
        if not self._running:
            return

        logger.info("Stopping AutonomousCOO...")
        self._running = False
        self._state = COOState.IDLE

        # Wait for the background thread to finish (it will exit because _running is False)
        if hasattr(self, '_background_thread') and self._background_thread:
            self._background_thread.join(timeout=5.0)
            self._background_thread = None

        # Clear current executions (they're on the background loop which is now stopped)
        self._current_executions.clear()

        # Save learning state
        if self._learning:
            try:
                await self._learning.save()
            except Exception as e:
                logger.warning(f"Error saving learning state: {e}")

        logger.info("AutonomousCOO stopped")

    async def _main_loop(self):
        """Main COO observation-decision-execution loop."""
        logger.info("COO main loop started")

        while self._running:
            try:
                # PHASE 1: OBSERVE
                self._state = COOState.OBSERVING
                context = await self._observe()
                self._last_observation = datetime.now()
                logger.info(f"COO observed: {len(context.get('tasks', []))} tasks, {len(context.get('goals', []))} goals")

                if self._mode == ExecutionMode.PAUSED:
                    logger.debug("COO is paused, sleeping...")
                    await asyncio.sleep(self.config.observation_interval_seconds)
                    continue

                # PHASE 2: PRIORITIZE
                self._state = COOState.PRIORITIZING
                prioritized = await self._prioritize(context)
                logger.info(f"COO prioritized: {len(prioritized)} items")

                if not prioritized:
                    # Nothing to do
                    logger.debug("No prioritized items, sleeping...")
                    await asyncio.sleep(self.config.observation_interval_seconds)
                    continue

                # PHASE 3: DECIDE & DELEGATE
                self._state = COOState.DELEGATING
                for item in prioritized[:self.config.max_concurrent_executions]:
                    if len(self._current_executions) >= self.config.max_concurrent_executions:
                        break

                    # Skip items already being executed
                    if item.id in self._current_executions:
                        logger.debug(f"Skipping {item.id}: already executing")
                        continue

                    decision = await self._decide(item, context)

                    if decision["action"] == "execute":
                        # PHASE 4: EXECUTE
                        self._state = COOState.EXECUTING
                        await self._execute(item, decision)

                    elif decision["action"] == "request_approval":
                        self._state = COOState.WAITING_APPROVAL
                        await self._request_approval(item, decision)

                    elif decision["action"] == "skip":
                        logger.debug(f"Skipping {item.id}: {decision['reason']}")

                # Wait before next cycle
                await asyncio.sleep(self.config.observation_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in COO main loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)  # Brief pause on error

        logger.info("COO main loop ended")

    async def _observe(self) -> Dict[str, Any]:
        """
        Observe current state of all systems.

        Returns context dict with:
        - goals: Active goals and their progress
        - tasks: Pending and in-progress tasks
        - blockers: Current blockers needing resolution
        - resources: Available resources (budget, models)
        - recent_outcomes: Recent execution results for learning
        """
        context = {
            "timestamp": datetime.now(),
            "goals": [],
            "tasks": [],
            "blockers": [],
            "decisions_pending": [],
            "resources": {
                "daily_budget_remaining": self.config.daily_budget_usd - self._daily_spend,
                "concurrent_slots_available": (
                    self.config.max_concurrent_executions - len(self._current_executions)
                ),
            },
            "recent_outcomes": [],
        }

        if not self._intel:
            return context

        try:
            # Get active goals
            goals = await self._intel.goals.list_goals(include_completed=False)
            context["goals"] = goals

            # Get pending/in-progress tasks
            tasks = await self._intel.tasks.list_tasks(include_completed=False)
            context["tasks"] = tasks

            # Identify blockers
            for task in tasks:
                if hasattr(task, 'blockers'):
                    for blocker in task.blockers:
                        if not blocker.resolved:
                            context["blockers"].append({
                                "task_id": task.id,
                                "task_title": task.title,
                                "blocker": blocker,
                            })

            # Get recent outcomes from learning
            if self._learning:
                context["recent_outcomes"] = await self._learning.get_recent_outcomes(limit=10)

        except Exception as e:
            logger.error(f"Error observing state: {e}")

        return context

    async def _prioritize(self, context: Dict[str, Any]) -> List[Any]:
        """
        Prioritize work items based on current context.

        Uses the PriorityEngine to score and rank all potential work.
        """
        if not self._priority_engine:
            return []

        return await self._priority_engine.prioritize(context)

    async def _decide(self, item: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide what action to take for a prioritized item.

        Returns decision dict with:
        - action: "execute", "request_approval", "skip", "defer"
        - reason: Explanation for the decision
        - executor: Which agent/pipeline should handle this
        - confidence: How confident we are in this decision
        """
        decision = {
            "action": "skip",
            "reason": "Unknown",
            "executor": None,
            "confidence": 0.0,
            "timestamp": datetime.now(),
        }

        # Check mode
        if self._mode == ExecutionMode.OBSERVE:
            decision["action"] = "skip"
            decision["reason"] = "COO in observe-only mode"
            return decision

        # Check resources
        if context["resources"]["daily_budget_remaining"] <= 0:
            decision["action"] = "skip"
            decision["reason"] = "Daily budget exhausted"
            return decision

        if context["resources"]["concurrent_slots_available"] <= 0:
            decision["action"] = "defer"
            decision["reason"] = "All execution slots in use"
            return decision

        # Check for approval-required tags
        item_tags = getattr(item, 'tags', []) or []
        requires_approval = any(
            tag in self.config.require_approval_tags
            for tag in item_tags
        )

        # Determine executor based on item type
        executor = await self._select_executor(item)
        decision["executor"] = executor

        # Calculate confidence
        confidence = await self._calculate_confidence(item, context)
        decision["confidence"] = confidence

        # Make decision based on mode and confidence
        if self._mode == ExecutionMode.APPROVAL:
            decision["action"] = "request_approval"
            decision["reason"] = "Approval mode - all actions require approval"

        elif requires_approval:
            decision["action"] = "request_approval"
            decision["reason"] = f"Task has approval-required tag"

        elif self._mode == ExecutionMode.AUTONOMOUS:
            if confidence >= self.config.auto_execute_confidence:
                decision["action"] = "execute"
                decision["reason"] = f"High confidence ({confidence:.2f})"
            else:
                decision["action"] = "request_approval"
                decision["reason"] = f"Confidence ({confidence:.2f}) below threshold"

        elif self._mode == ExecutionMode.SUPERVISED:
            # In supervised mode, execute but log everything
            if confidence >= self.config.auto_execute_confidence * 0.8:
                decision["action"] = "execute"
                decision["reason"] = f"Supervised execution (confidence: {confidence:.2f})"
            else:
                decision["action"] = "request_approval"
                decision["reason"] = f"Low confidence ({confidence:.2f}) requires approval"

        # Record decision
        self._last_decision = f"{decision['action']}: {item.id if hasattr(item, 'id') else str(item)[:50]}"

        # Trigger callbacks
        for callback in self._on_decision:
            try:
                await callback(item, decision, context)
            except Exception as e:
                logger.error(f"Decision callback error: {e}")

        return decision

    async def _select_executor(self, item: Any) -> str:
        """
        Select the appropriate executor for an item.

        Routing priority:
        1. Task type -> csuite executive (via executive registry)
        2. Text-based keyword matching -> csuite executive
        3. Fallback to Nexus-native executors
        """
        from nexus.coo.executive_registry import (
            get_executive_for_task,
            get_executive_for_text,
        )

        # Get item characteristics
        task_type = getattr(item, 'task_type', None)
        title = getattr(item, 'title', '') or ''
        description = getattr(item, 'description', '') or ''
        tags = getattr(item, 'tags', []) or []

        text = f"{title} {description} {' '.join(tags)}".lower()

        # === PRIORITY 1: Task type-based routing to csuite ===
        if task_type:
            executive = get_executive_for_task(task_type)
            if executive:
                return f"csuite:{executive}"  # e.g., "csuite:CTO"

        # === PRIORITY 2: Text-based routing to csuite ===
        executive = get_executive_for_text(text)
        if executive:
            return f"csuite:{executive}"

        # === PRIORITY 3: Fallback to Nexus-native executors ===
        # Route to appropriate Nexus executor based on keywords
        if any(kw in text for kw in ["research", "find", "discover", "investigate"]):
            return "research_agent"

        elif any(kw in text for kw in ["write", "create", "draft", "content", "article", "ebook", "blog"]):
            return "content_pipeline"

        elif any(kw in text for kw in ["code", "implement", "build", "fix", "debug", "develop"]):
            return "code_agent"

        elif any(kw in text for kw in ["analyze", "evaluate", "compare", "assess"]):
            return "analyst_expert"

        elif any(kw in text for kw in ["trend", "market", "topic", "popular"]):
            return "trend_analyzer"

        elif any(kw in text for kw in ["blueprint", "outline", "structure", "plan"]):
            return "blueprint_factory"

        else:
            return "expert_router"  # Default to expert routing

    async def _calculate_confidence(self, item: Any, context: Dict[str, Any]) -> float:
        """Calculate confidence score for executing an item."""
        confidence = 0.5  # Base confidence

        # Boost from learning history
        if self._learning:
            historical = await self._learning.get_similar_outcomes(item)
            if historical:
                success_rate = sum(1 for h in historical if h.success) / len(historical)
                confidence += 0.3 * success_rate

        # Boost from clear requirements
        if getattr(item, 'description', None):
            if len(item.description) > 50:
                confidence += 0.1

        # Reduce from blockers
        if hasattr(item, 'blockers') and item.blockers:
            unresolved = sum(1 for b in item.blockers if not b.resolved)
            confidence -= 0.1 * unresolved

        # Reduce from high priority (needs more care)
        priority = getattr(item, 'priority', None)
        if priority and hasattr(priority, 'value'):
            if priority.value in ['critical', 'high']:
                confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    async def _execute(self, item: Any, decision: Dict[str, Any]):
        """Execute an item using the appropriate executor."""
        # Unwrap PrioritizedItem if needed
        from nexus.coo.priority_engine import PrioritizedItem
        actual_item = item.item if isinstance(item, PrioritizedItem) else item
        item_id = getattr(actual_item, 'id', str(id(actual_item)))

        async def execute_task():
            try:
                result = await self._executor.execute(
                    item=actual_item,
                    executor_type=decision["executor"],
                    context={"decision": decision}
                )

                self._total_executed += 1
                if result.success:
                    self._successful += 1
                else:
                    self._failed += 1

                self._daily_spend += result.cost_usd

                # Learn from outcome
                if self._learning:
                    await self._learning.record_outcome(actual_item, result)

                # Trigger callbacks
                for callback in self._on_execution_complete:
                    try:
                        await callback(actual_item, result)
                    except Exception as e:
                        logger.error(f"Execution callback error: {e}")

                return result

            except Exception as e:
                logger.error(f"Execution failed for {item_id}: {e}")
                self._failed += 1
                raise

            finally:
                # Remove from current executions
                if item_id in self._current_executions:
                    del self._current_executions[item_id]

        # Start execution task
        task = asyncio.create_task(execute_task())
        self._current_executions[item_id] = task

        logger.info(f"Started execution: {item_id} via {decision['executor']}")

    async def _request_approval(self, item: Any, decision: Dict[str, Any]):
        """Request approval for an action."""
        item_id = getattr(item, 'id', str(id(item)))

        approval_request = {
            "id": item_id,
            "item": item,
            "decision": decision,
            "requested_at": datetime.now(),
            "status": "pending",
        }

        self._pending_approvals[item_id] = approval_request

        # Trigger callbacks
        for callback in self._on_approval_needed:
            try:
                await callback(item, decision)
            except Exception as e:
                logger.error(f"Approval callback error: {e}")

        # Notify
        title = getattr(item, 'title', str(item)[:50])
        await self._notify(
            f"Approval needed: {title}\n"
            f"Reason: {decision['reason']}\n"
            f"Executor: {decision['executor']}"
        )

        logger.info(f"Approval requested for: {item_id}")

    async def approve(self, item_id: str, approved: bool = True, notes: str = None):
        """Process an approval decision."""
        if item_id not in self._pending_approvals:
            logger.warning(f"No pending approval for: {item_id}")
            return False

        approval = self._pending_approvals[item_id]

        if approved:
            approval["status"] = "approved"
            # Execute the approved item
            await self._execute(approval["item"], approval["decision"])
            logger.info(f"Approved and executing: {item_id}")
        else:
            approval["status"] = "rejected"
            logger.info(f"Rejected: {item_id}")

        approval["processed_at"] = datetime.now()
        approval["notes"] = notes

        # Learn from the approval decision
        if self._learning:
            await self._learning.record_approval(
                item=approval["item"],
                approved=approved,
                notes=notes
            )

        # Remove from pending
        del self._pending_approvals[item_id]

        return True

    def set_mode(self, mode: ExecutionMode):
        """Change the operating mode."""
        old_mode = self._mode
        self._mode = mode
        logger.info(f"COO mode changed: {old_mode.value} -> {mode.value}")

    def get_status(self) -> COOStatus:
        """Get current COO status."""
        uptime = (datetime.now() - self._started_at).total_seconds() if self._started_at else 0

        effectiveness = 0.0
        if self._total_executed > 0:
            effectiveness = self._successful / self._total_executed

        return COOStatus(
            state=self._state,
            mode=self._mode,
            uptime_seconds=uptime,
            total_tasks_executed=self._total_executed,
            successful_executions=self._successful,
            failed_executions=self._failed,
            pending_approvals=len(self._pending_approvals),
            current_executions=list(self._current_executions.keys()),
            last_observation=self._last_observation,
            last_decision=self._last_decision,
            learning_effectiveness=effectiveness,
            daily_spend_usd=self._daily_spend,
        )

    async def _notify(self, message: str):
        """Send notification through configured callback."""
        if self.config.notification_callback:
            try:
                await self.config.notification_callback(message)
            except Exception as e:
                logger.error(f"Notification error: {e}")

    # Event registration
    def on_decision(self, callback: Callable):
        """Register callback for decision events."""
        self._on_decision.append(callback)

    def on_execution_complete(self, callback: Callable):
        """Register callback for execution completion."""
        self._on_execution_complete.append(callback)

    def on_approval_needed(self, callback: Callable):
        """Register callback for approval requests."""
        self._on_approval_needed.append(callback)

    # Manual triggers
    async def execute_now(self, task_id: str) -> bool:
        """Manually trigger execution of a specific task."""
        if not self._intel:
            return False

        try:
            task = await self._intel.tasks.get_task(task_id)

            decision = {
                "action": "execute",
                "reason": "Manual trigger",
                "executor": await self._select_executor(task),
                "confidence": 1.0,
                "timestamp": datetime.now(),
            }

            await self._execute(task, decision)
            return True

        except Exception as e:
            logger.error(f"Manual execution failed: {e}")
            return False

    async def suggest_next_action(self) -> Dict[str, Any]:
        """Get COO's suggestion for the next action without executing."""
        context = await self._observe()
        prioritized = await self._prioritize(context)

        if not prioritized:
            return {
                "suggestion": None,
                "reason": "No actionable items found",
                "context_summary": {
                    "active_goals": len(context["goals"]),
                    "pending_tasks": len(context["tasks"]),
                    "blockers": len(context["blockers"]),
                }
            }

        top_item = prioritized[0]
        decision = await self._decide(top_item, context)

        return {
            "suggestion": top_item,
            "decision": decision,
            "reason": f"Highest priority item (confidence: {decision['confidence']:.2f})",
            "context_summary": {
                "active_goals": len(context["goals"]),
                "pending_tasks": len(context["tasks"]),
                "blockers": len(context["blockers"]),
            }
        }

    # csuite Bridge Integration
    async def connect_to_csuite(
        self,
        redis_url: str = "redis://localhost:6379",
        channel_prefix: str = "csuite:nexus",
    ) -> bool:
        """
        Connect to csuite via Redis bridge.

        This enables bidirectional communication between Nexus (strategic brain)
        and csuite CoS (operational coordinator).

        Args:
            redis_url: Redis connection URL
            channel_prefix: Prefix for Redis channels

        Returns:
            True if connection and listening started successfully
        """
        if not self._csuite_bridge:
            from nexus.coo.csuite_bridge import CSuiteBridgeListener, CSuiteBridgeConfig
            config = CSuiteBridgeConfig(
                redis_url=redis_url,
                channel_prefix=channel_prefix,
            )
            self._csuite_bridge = CSuiteBridgeListener(self, config)

        # Connect and start listening
        connected = await self._csuite_bridge.connect()
        if connected:
            await self._csuite_bridge.start_listening()
            logger.info("Connected to csuite via Redis bridge")
            return True

        logger.error("Failed to connect to csuite bridge")
        return False

    async def disconnect_from_csuite(self) -> None:
        """Disconnect from csuite bridge."""
        if self._csuite_bridge:
            await self._csuite_bridge.disconnect()
            logger.info("Disconnected from csuite bridge")

    def is_csuite_connected(self) -> bool:
        """Check if connected to csuite."""
        if self._csuite_bridge:
            return self._csuite_bridge.is_connected
        return False

    async def send_directive_to_csuite(self, directive: Dict[str, Any]) -> bool:
        """
        Send a directive to csuite CoS.

        Directives are strategic instructions that CoS should execute.

        Args:
            directive: The directive to send. Should include:
                - type: Type of directive (e.g., "update_priorities", "pause_tasks")
                - payload: Directive-specific data

        Returns:
            True if directive was sent successfully
        """
        if not self._csuite_bridge or not self._csuite_bridge.is_connected:
            logger.warning("Cannot send directive: csuite bridge not connected")
            return False

        return await self._csuite_bridge.publish_directive(directive)

    def get_csuite_health(self) -> Optional[Dict[str, Any]]:
        """
        Get the last received health status from csuite.

        Returns:
            Health data dict or None if no health update received
        """
        if self._csuite_bridge:
            return self._csuite_bridge.get_csuite_health()
        return None

    def get_csuite_bridge_status(self) -> Dict[str, Any]:
        """
        Get status of the csuite bridge connection.

        Returns:
            Status dict with connection info and metrics
        """
        if self._csuite_bridge:
            return self._csuite_bridge.get_status()
        return {
            "connected": False,
            "listening": False,
            "message": "Bridge not initialized",
        }
