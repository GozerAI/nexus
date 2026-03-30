"""
Autonomous Executor - Delegates and executes work through appropriate agents.

Routes work to:
- Expert Router (for general tasks)
- Content Pipeline (for content creation)
- Research Agent (for research tasks)
- Code Agent (for technical tasks)
- Ensemble (for direct LLM queries)
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of an execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of an execution."""
    item_id: str
    executor: str
    success: bool
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    confidence: float = 0.0
    duration_minutes: float = 0.0
    cost_usd: float = 0.0
    tokens_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "executor": self.executor,
            "success": self.success,
            "status": self.status.value,
            "output": str(self.output)[:1000] if self.output else None,
            "error": self.error,
            "confidence": self.confidence,
            "duration_minutes": self.duration_minutes,
            "cost_usd": self.cost_usd,
            "tokens_used": self.tokens_used,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AutonomousExecutor:
    """
    Autonomous Executor.

    Routes work items to appropriate execution backends:

    Executors:
    - csuite:* - Routes to csuite executives via Redis bridge (e.g., csuite:CTO)
    - expert_router: Routes to expert personas for analysis/execution
    - content_pipeline: Runs the full content creation pipeline
    - research_agent: Autonomous research with goal decomposition
    - code_agent: Code generation and improvement
    - trend_analyzer: Analyzes trends for content opportunities
    - blueprint_factory: Generates content blueprints
    - ensemble: Direct LLM queries via ensemble orchestration
    - decision_agent: Resolves blockers and makes decisions
    """

    EXECUTORS = {
        "csuite": "Route to csuite executive via Redis bridge",
        "expert_router": "Route to expert personas",
        "content_pipeline": "Content creation pipeline",
        "research_agent": "Autonomous research",
        "code_agent": "Code generation",
        "trend_analyzer": "Trend analysis",
        "blueprint_factory": "Blueprint generation",
        "ensemble": "Direct LLM query",
        "decision_agent": "Decision/blocker resolution",
        "analyst_expert": "Analysis tasks",
    }

    # Timeout for csuite execution requests (seconds)
    CSUITE_EXECUTION_TIMEOUT = 120

    def __init__(self, intelligence=None, learning=None, config=None, csuite_bridge=None):
        """
        Initialize the executor.

        Args:
            intelligence: NexusIntelligence for system access
            learning: PersistentLearning for tracking outcomes
            config: COO configuration
            csuite_bridge: CSuiteBridgeListener for csuite communication
        """
        self._intel = intelligence
        self._learning = learning
        self._config = config
        self._csuite_bridge = csuite_bridge

        # Execution tracking
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._execution_history: List[ExecutionResult] = []

        # Pending csuite execution responses
        self._pending_csuite_executions: Dict[str, asyncio.Future] = {}

        # Lazy-loaded executors
        self._expert_router = None
        self._content_pipeline = None
        self._research_agent = None
        self._ensemble = None

    async def initialize(self):
        """Initialize executor backends."""
        # Expert Router
        try:
            from nexus.orchestration.expert_router import ExpertRouter
            self._expert_router = ExpertRouter(platform=self._intel)
            logger.info("ExpertRouter initialized")
        except Exception as e:
            logger.warning(f"Could not initialize ExpertRouter: {e}")

        # Content Pipeline
        try:
            from nexus.automations.autonomous_pipeline import AutonomousPipeline
            self._content_pipeline = AutonomousPipeline()
            logger.info("AutonomousPipeline initialized")
        except Exception as e:
            logger.warning(f"Could not initialize AutonomousPipeline: {e}")

        # Research Agent
        try:
            self._research_agent = AutonomousResearchAgent()
            logger.info("AutonomousResearchAgent initialized")
        except Exception as e:
            logger.warning(f"Could not initialize AutonomousResearchAgent: {e}")

        # Ensemble (from intelligence layer)
        if self._intel and hasattr(self._intel, '_ensemble'):
            self._ensemble = self._intel._ensemble
            logger.info("Ensemble connected from intelligence layer")

        logger.info("AutonomousExecutor initialized")

    async def execute(
        self,
        item: Any,
        executor_type: str,
        context: Dict[str, Any] = None
    ) -> ExecutionResult:
        """
        Execute an item using the specified executor.

        Args:
            item: The item to execute (task, goal, blocker)
            executor_type: Which executor to use
            context: Additional context for execution

        Returns:
            ExecutionResult with outcome
        """
        item_id = self._item_value(item, 'id', str(id(item)))
        start_time = time.time()

        result = ExecutionResult(
            item_id=item_id,
            executor=executor_type,
            success=False,
            status=ExecutionStatus.RUNNING,
        )

        try:
            logger.info(f"Executing {item_id} via {executor_type}")

            # Route to appropriate executor
            # Check for csuite executors first (format: csuite:EXECUTIVE_CODE)
            if executor_type.startswith("csuite:"):
                output = await self._execute_via_csuite(item, executor_type, context)

            elif executor_type == "expert_router":
                output = await self._execute_expert_router(item, context)

            elif executor_type == "content_pipeline":
                output = await self._execute_content_pipeline(item, context)

            elif executor_type == "research_agent":
                output = await self._execute_research_agent(item, context)

            elif executor_type == "code_agent":
                output = await self._execute_code_agent(item, context)

            elif executor_type == "trend_analyzer":
                output = await self._execute_trend_analyzer(item, context)

            elif executor_type == "blueprint_factory":
                output = await self._execute_blueprint_factory(item, context)

            elif executor_type == "ensemble":
                output = await self._execute_ensemble(item, context)

            elif executor_type == "decision_agent":
                output = await self._execute_decision_agent(item, context)

            elif executor_type == "analyst_expert":
                output = await self._execute_analyst(item, context)

            else:
                # Default to expert router
                output = await self._execute_expert_router(item, context)

            self._raise_if_execution_output_failed(output, executor_type)

            # Success
            result.success = True
            result.status = ExecutionStatus.COMPLETED
            result.output = output
            result.confidence = output.get("confidence", 0.8) if isinstance(output, dict) else 0.8

            # Update task status if applicable
            await self._update_item_status(item, success=True)

        except Exception as e:
            logger.error(f"Execution failed for {item_id}: {e}", exc_info=True)
            result.success = False
            result.status = ExecutionStatus.FAILED
            result.error = str(e)

            await self._update_item_status(item, success=False, error=str(e))

        finally:
            result.completed_at = datetime.now()
            result.duration_minutes = (time.time() - start_time) / 60

            # Record to history
            self._execution_history.append(result)
            if len(self._execution_history) > 1000:
                self._execution_history = self._execution_history[-1000:]

        return result

    async def _execute_expert_router(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute via Expert Router."""
        if not self._expert_router:
            await self._ensure_expert_router()
        if not self._expert_router:
            raise RuntimeError("ExpertRouter not initialized")

        from nexus.experts.base import Task, TaskType

        # Convert item to Task format
        task = Task(
            id=self._item_value(item, 'id', 'task_1'),
            description=f"{self._item_value(item, 'title', '')} {self._item_value(item, 'description', '')}".strip(),
            task_type=self._determine_task_type(item),
            context={"original_item": str(item)},
            priority=self._priority_to_int(self._item_value(item, 'priority', 'medium')),
        )

        result = await self._expert_router.execute_with_experts(task)

        return result

    async def _execute_content_pipeline(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute via Content Pipeline."""
        if not self._content_pipeline:
            await self._ensure_content_pipeline()
        if not self._content_pipeline:
            raise RuntimeError("ContentPipeline not initialized")

        title = getattr(item, 'title', '')
        description = getattr(item, 'description', '')

        # Extract topic from task
        topic = title if title else description[:100]

        # Queue the job
        job_id = await self._content_pipeline.queue_job({
            "topic": topic,
            "description": description,
            "source": "coo_executor",
        })

        return {
            "job_id": job_id,
            "status": "queued",
            "topic": topic,
            "confidence": 0.85,
        }

    async def _execute_research_agent(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute via Research Agent."""
        title = self._item_value(item, 'title', '')
        description = self._item_value(item, 'description', '')
        execution_context = context or {}
        topic = f"{title}: {description}".strip(": ") or description or title or "General research task"

        if self._intel and hasattr(self._intel, "research"):
            result = await self._intel.research(
                topic=topic,
                depth=execution_context.get("depth", "moderate"),
                max_iterations=execution_context.get("max_iterations", 5),
            )
        else:
            if not self._research_agent:
                await self._ensure_research_agent()
            if not self._research_agent:
                raise RuntimeError("ResearchAgent not initialized")

            result = await self._research_agent.research(
                topic=topic,
                depth=execution_context.get("depth", "moderate"),
                max_iterations=execution_context.get("max_iterations", 5),
            )

        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(str(result["error"]))

        return {
            "research_complete": True,
            "findings": result.get("findings", []),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.7),
        }

    async def _execute_code_agent(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute via Code Agent (using self-improving codegen)."""
        try:
            title = self._item_value(item, 'title', '')
            description = self._item_value(item, 'description', '')
            execution_context = context or {}
            language = execution_context.get("language", self._item_value(item, "language", "python"))
            requirements = execution_context.get("requirements", self._item_value(item, "requirements", []))
            constraints = execution_context.get("constraints", self._item_value(item, "constraints", []))
            max_iterations = execution_context.get("max_iterations", 3)
            target_quality = execution_context.get("target_quality")
            request_description = f"{title}\n\n{description}".strip() or description or title or "Generate code"

            if self._intel and hasattr(self._intel, "generate_code"):
                result = await self._intel.generate_code(
                    description=request_description,
                    language=language,
                    requirements=self._coerce_codegen_list(requirements),
                    constraints=self._coerce_codegen_list(constraints),
                    max_iterations=max_iterations,
                    target_quality=target_quality,
                    context={"executor": "coo_executor"},
                )
            else:
                    CodeGenerationRequest,
                    SelfImprovingCodeGenerator,
                )

                codegen = SelfImprovingCodeGenerator()
                request = CodeGenerationRequest(
                    request_id=str(uuid.uuid4()),
                    description=request_description,
                    language=language,
                    requirements=self._coerce_codegen_list(requirements),
                    constraints=self._coerce_codegen_list(constraints),
                    context={"executor": "coo_executor"},
                )

                generation_kwargs = {"max_iterations": max_iterations}
                if target_quality is not None:
                    generation_kwargs["target_quality"] = target_quality

                raw_result = await codegen.generate(request, **generation_kwargs)
                result = {
                    "code": raw_result.code,
                    "quality_score": raw_result.quality_score,
                    "confidence": raw_result.confidence,
                    "iteration": raw_result.iteration,
                    "generation_path": "cog_eng_codegen",
                }

            return {
                "code_generated": True,
                "code": result.get("code", ""),
                "quality_score": result.get("quality_score", 0.7),
                "confidence": result.get("confidence", result.get("quality_score", 0.7)),
                "generation_path": result.get("generation_path", "cog_eng_codegen"),
            }

        except Exception as e:
            logger.warning(f"Code agent error: {e}")
            # Fallback to ensemble for code
            return await self._execute_ensemble(item, {"task_type": "code"})

    async def _execute_trend_analyzer(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute trend analysis."""
        try:
            from nexus.automations.content_orchestrator import ContentOrchestrator

            orchestrator = ContentOrchestrator()

            title = self._item_value(item, 'title', '')
            topics = title.split() if title else ["technology", "AI"]

            trends = await orchestrator.analyze_trends(topics[:3])

            return {
                "trends_analyzed": True,
                "trends": trends,
                "confidence": 0.75,
            }

        except Exception as e:
            logger.warning(f"Trend analyzer error: {e}")
            return {
                "trends_analyzed": False,
                "error": str(e),
                "confidence": 0.3,
            }

    async def _execute_blueprint_factory(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute blueprint generation."""
        if not self._content_pipeline:
            await self._ensure_content_pipeline()
        if not self._content_pipeline:
            raise RuntimeError("ContentPipeline not initialized")

        title = self._item_value(item, 'title', '')
        description = self._item_value(item, 'description', '')

        topic = f"{title}: {description}"

        # Generate blueprint only (not full content)
        blueprint = await self._content_pipeline.generate_blueprint(topic)

        return {
            "blueprint_generated": True,
            "blueprint": blueprint,
            "confidence": 0.8,
        }

    async def _execute_ensemble(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute via Ensemble for direct LLM query."""
        if not self._ensemble:
            await self._ensure_ensemble_backend()
        if not self._ensemble:
            raise RuntimeError("Ensemble not initialized")

        title = self._item_value(item, 'title', '')
        description = self._item_value(item, 'description', '')
        execution_context = context or {}

        prompt = f"""Task: {title}

Description: {description}

Please complete this task and provide a detailed response."""

        if hasattr(self._ensemble, "query"):
            response = await self._ensemble.query(
                prompt,
                strategy=execution_context.get("strategy", "simple_best"),
                task_type=execution_context.get("task_type", "conversation"),
                max_tokens=execution_context.get("max_tokens", 2000),
            )
            self._raise_if_execution_output_failed(response, "ensemble")
            metadata = response.get("metadata", {}) if isinstance(response, dict) else {}
            all_scores = metadata.get("all_scores", {}) if isinstance(metadata, dict) else {}
            models_used = response.get("models_queried")
            if not isinstance(models_used, list):
                models_used = list(all_scores.keys()) or [response.get("model_name", "ensemble")]
            return {
                "response": response.get("content", ""),
                "models_used": models_used,
                "confidence": response.get("confidence", 0.7),
                "cost_usd": response.get("total_cost", 0.0),
                "strategy_used": response.get("strategy_used"),
            }

        if hasattr(self._ensemble, "process"):
            from nexus.providers.ensemble.types import EnsembleRequest
            from uuid import uuid4

            request = EnsembleRequest(
                query=prompt,
                request_id=uuid4(),
                user_id="coo_executor",
                max_models=3,
                temperature=0.7,
            )

            response = await self._ensemble.process(request)

            return {
                "response": response.content,
                "models_used": [r.model_name for r in response.model_responses],
                "confidence": response.confidence,
                "cost_usd": response.total_cost_usd,
            }

        raise RuntimeError("Ensemble backend does not support query or process execution")

    async def _execute_decision_agent(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute decision/blocker resolution."""
        blocker = self._item_value(item, "blocker")
        if blocker is not None:
            task_title = self._item_value(item, "task_title", "Unknown task")
            blocker_type = blocker.blocker_type.value if hasattr(blocker.blocker_type, "value") else blocker.blocker_type

            prompt = f"""Blocker Resolution Request:

Task: {task_title}
Blocker Type: {blocker_type}
Description: {blocker.description}

Please analyze this blocker and provide:
1. Root cause analysis
2. Recommended resolution
3. Alternative approaches
4. Risks to consider
"""
        else:
            title = self._item_value(item, 'title', '')
            description = self._item_value(item, 'description', '')
            prompt = f"Decision needed: {title}\n\nContext: {description}"

        # Use ensemble for decision
        return await self._execute_ensemble(
            type('Item', (), {'title': 'Decision Analysis', 'description': prompt})(),
            context
        )

    async def _execute_analyst(self, item: Any, context: Dict) -> Dict[str, Any]:
        """Execute analysis task."""
        return await self._execute_expert_router(item, context)

    async def _execute_via_csuite(
        self,
        item: Any,
        executor_type: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a task via csuite through the Redis bridge.

        Args:
            item: The item to execute
            executor_type: Format "csuite:EXECUTIVE_CODE" (e.g., "csuite:CTO")
            context: Additional execution context

        Returns:
            Execution result from csuite

        Raises:
            RuntimeError: If csuite bridge not available or execution fails
        """
        # Extract executive code from executor_type
        executive_code = executor_type.split(":", 1)[1] if ":" in executor_type else "CoS"

        # Get item details
        item_id = self._item_value(item, 'id', str(uuid.uuid4()))
        task_type = self._item_value(item, 'task_type', None)
        title = self._item_value(item, 'title', '') or ''
        description = self._item_value(item, 'description', '') or ''

        # Check if bridge is available
        if not self._csuite_bridge:
            logger.warning("csuite bridge not available, falling back to expert_router")
            return await self._execute_expert_router(item, context)

        if not self._csuite_bridge.is_connected:
            logger.warning("csuite bridge not connected, falling back to expert_router")
            return await self._execute_expert_router(item, context)

        # Prepare execution request
        execution_request = {
            "type": "execution_request",
            "request_id": str(uuid.uuid4()),
            "task_id": item_id,
            "task_type": task_type,
            "title": title,
            "description": description,
            "target_executive": executive_code,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }

        # Create a future for the response
        response_future = asyncio.get_event_loop().create_future()
        self._pending_csuite_executions[execution_request["request_id"]] = response_future

        try:
            # Send execution request to csuite
            request_channel = f"{self._csuite_bridge.config.channel_prefix}:execute:request"
            await self._csuite_bridge._redis.publish(
                request_channel,
                json.dumps(execution_request)
            )

            logger.info(
                f"Sent execution request to csuite: {executive_code} "
                f"(request_id={execution_request['request_id']})"
            )

            # Start listening for response if not already listening
            response_channel = f"{self._csuite_bridge.config.channel_prefix}:execute:response"
            asyncio.create_task(
                self._listen_for_csuite_response(response_channel, execution_request["request_id"])
            )

            # Wait for response with timeout
            response = await asyncio.wait_for(
                response_future,
                timeout=self.CSUITE_EXECUTION_TIMEOUT
            )

            logger.info(f"Received csuite execution response: {response.get('success', False)}")

            return {
                "csuite_execution": True,
                "executive": executive_code,
                "success": response.get("success", False),
                "output": response.get("output", {}),
                "error": response.get("error"),
                "confidence": response.get("confidence", 0.8),
                "duration_ms": response.get("duration_ms", 0),
            }

        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for csuite execution response (executive={executive_code})")
            return {
                "csuite_execution": True,
                "executive": executive_code,
                "success": False,
                "error": f"Timeout waiting for csuite response after {self.CSUITE_EXECUTION_TIMEOUT}s",
                "confidence": 0.0,
            }

        except Exception as e:
            logger.error(f"Error executing via csuite: {e}")
            return {
                "csuite_execution": True,
                "executive": executive_code,
                "success": False,
                "error": str(e),
                "confidence": 0.0,
            }

        finally:
            # Clean up pending execution
            self._pending_csuite_executions.pop(execution_request["request_id"], None)

    async def _listen_for_csuite_response(
        self,
        response_channel: str,
        request_id: str,
    ) -> None:
        """
        Listen for csuite execution response on Redis.

        Args:
            response_channel: Redis channel to listen on
            request_id: Request ID to match response to
        """
        try:
            pubsub = self._csuite_bridge._redis.pubsub()
            await pubsub.subscribe(response_channel)

            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        response = json.loads(message["data"])

                        # Check if this is our response
                        if response.get("request_id") == request_id:
                            future = self._pending_csuite_executions.get(request_id)
                            if future and not future.done():
                                future.set_result(response)

                            await pubsub.unsubscribe(response_channel)
                            await pubsub.close()
                            return

                    except json.JSONDecodeError:
                        pass

                # Check if the future is already done (timeout or other)
                future = self._pending_csuite_executions.get(request_id)
                if future and future.done():
                    await pubsub.unsubscribe(response_channel)
                    await pubsub.close()
                    return

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error listening for csuite response: {e}")

    def set_csuite_bridge(self, bridge) -> None:
        """
        Set the csuite bridge for execution routing.

        Args:
            bridge: CSuiteBridgeListener instance
        """
        self._csuite_bridge = bridge
        logger.info("csuite bridge set on executor")

    def _determine_task_type(self, item: Any):
        """Determine TaskType from item characteristics."""
        from nexus.experts.base import TaskType

        title = str(self._item_value(item, 'title', '')).lower()
        desc = str(self._item_value(item, 'description', '')).lower()
        text = f"{title} {desc}"

        if any(kw in text for kw in ["research", "find", "discover"]):
            return TaskType.RESEARCH
        elif any(kw in text for kw in ["write", "create", "draft"]):
            return TaskType.WRITING
        elif any(kw in text for kw in ["code", "implement", "build"]):
            return TaskType.CODING
        elif any(kw in text for kw in ["analyze", "evaluate"]):
            return TaskType.ANALYSIS
        elif any(kw in text for kw in ["review", "check"]):
            return TaskType.REVIEW
        elif any(kw in text for kw in ["plan", "strategy"]):
            return TaskType.STRATEGY
        else:
            return TaskType.GENERAL

    def _priority_to_int(self, priority) -> int:
        """Convert priority to integer."""
        priority_value = priority.value if hasattr(priority, 'value') else str(priority)
        mapping = {
            "critical": 10,
            "high": 8,
            "medium": 5,
            "low": 3,
            "backlog": 1,
        }
        return mapping.get(priority_value, 5)

    def _item_value(self, item: Any, key: str, default: Any = None) -> Any:
        """Return a field from either a dict-like or object-like work item."""
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    async def _ensure_expert_router(self) -> None:
        """Lazily initialize expert routing only when needed."""
        if self._expert_router is not None:
            return
        try:
            from nexus.orchestration.expert_router import ExpertRouter

            self._expert_router = ExpertRouter(platform=self._intel)
        except Exception as e:
            logger.warning(f"Could not initialize ExpertRouter: {e}")

    async def _ensure_content_pipeline(self) -> None:
        """Lazily initialize content pipeline only when needed."""
        if self._content_pipeline is not None:
            return
        try:
            from nexus.automations.autonomous_pipeline import AutonomousPipeline

            self._content_pipeline = AutonomousPipeline()
        except Exception as e:
            logger.warning(f"Could not initialize AutonomousPipeline: {e}")

    async def _ensure_research_agent(self) -> None:
        """Lazily initialize research agent only when needed."""
        if self._research_agent is not None:
            return
        try:

            self._research_agent = AutonomousResearchAgent()
        except Exception as e:
            logger.warning(f"Could not initialize AutonomousResearchAgent: {e}")

    async def _ensure_ensemble_backend(self) -> None:
        """Lazily connect an ensemble backend from the intelligence layer."""
        if self._ensemble is not None:
            return
        if self._intel and hasattr(self._intel, "_ensemble"):
            self._ensemble = self._intel._ensemble

    def _coerce_codegen_list(self, value: Any) -> List[str]:
        """Normalize codegen list inputs from context or item metadata."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value]
        return [str(value)]

    def _raise_if_execution_output_failed(self, output: Any, executor_type: str) -> None:
        """Raise when an executor returns a structured failure payload."""
        if not isinstance(output, dict):
            return

        if output.get("success") is False:
            raise RuntimeError(str(output.get("error") or f"{executor_type} execution failed"))

        if output.get("error") and output.get("success") is not True:
            raise RuntimeError(str(output["error"]))

    async def _update_item_status(self, item: Any, success: bool, error: str = None):
        """Update the status of an executed item."""
        if not self._intel:
            return

        item_id = self._item_value(item, 'id', None)
        if not item_id:
            return

        try:
            # Determine if it's a task
            if self._item_value(item, 'status') is not None:

                if success:
                    new_status = TaskStatus.COMPLETED
                else:
                    new_status = TaskStatus.BLOCKED

                await self._intel.tasks.update_task(item_id, {"status": new_status})

                # Add note about execution
                if error:
                    note = TaskNote(
                        id=TaskNote.generate_id(),
                        task_id=item_id,
                        note=f"COO Execution failed: {error}"
                    )
                    await self._intel.tasks.add_note(item_id, note)

                logger.info(f"Updated task {item_id} status to {new_status.value}")

        except Exception as e:
            logger.error(f"Error updating item status: {e}")

    def get_execution_history(self, limit: int = 50) -> List[ExecutionResult]:
        """Get recent execution history."""
        return self._execution_history[-limit:]

    def get_active_executions(self) -> List[str]:
        """Get list of currently active execution IDs."""
        return list(self._active_executions.keys())

    async def cancel_execution(self, item_id: str) -> bool:
        """Cancel an active execution."""
        if item_id in self._active_executions:
            task = self._active_executions[item_id]
            task.cancel()
            del self._active_executions[item_id]
            logger.info(f"Cancelled execution: {item_id}")
            return True
        return False
