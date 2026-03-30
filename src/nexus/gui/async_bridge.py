"""Bridge between async intelligence layer and sync Qt GUI."""

import asyncio
import logging
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Any, Optional
from PySide6.QtCore import QObject, Signal, QThread, QMetaObject, Qt, Q_ARG, QTimer


logger = logging.getLogger(__name__)


class AsyncWorker(QThread):
    """Worker thread for running async operations."""

    finished = Signal(object)
    error = Signal(str)  # Changed to str to avoid cross-thread issues

    def __init__(self, coro, parent=None):
        super().__init__(parent)
        self.coro = coro
        self._result = None
        self._error = None
        self._loop = None
        self._completed = False

    def run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._result = self._loop.run_until_complete(self.coro)
                self.finished.emit(self._result)
            finally:
                # Properly close the loop
                try:
                    self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                except Exception:
                    pass
                self._loop.close()
                self._loop = None
                self._completed = True
        except Exception as e:
            self._error = e
            self._completed = True
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error("AsyncWorker error: %s", error_msg)
            traceback.print_exc()
            self.error.emit(error_msg)

    @property
    def is_completed(self):
        return self._completed


class AsyncBridge(QObject):
    """Bridge for running async operations from Qt.

    Manages worker thread lifecycle to prevent thread accumulation
    and provides proper cleanup mechanisms.
    """

    # Maximum number of concurrent workers to prevent resource exhaustion
    MAX_WORKERS = 10

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workers = []
        self._pending_cleanup = []
        self._lock = threading.Lock()

        # Periodic cleanup timer - runs every 5 seconds to clean up finished workers
        self._cleanup_timer = QTimer(self)
        self._cleanup_timer.timeout.connect(self._periodic_cleanup)
        self._cleanup_timer.start(5000)

    def run_async(self, coro, on_success: Callable = None, on_error: Callable = None):
        """Run an async coroutine and call callbacks on completion."""
        # Clean up any finished workers first
        self._do_cleanup()

        # Check if we're at capacity
        with self._lock:
            active_workers = [w for w in self._workers if w.isRunning()]
            if len(active_workers) >= self.MAX_WORKERS:
                # Log warning but don't block - just clean up more aggressively
                logger.warning("%s workers active, forcing cleanup", len(active_workers))
                self._force_cleanup_oldest()

        worker = AsyncWorker(coro, self)

        if on_success:
            worker.finished.connect(on_success, Qt.QueuedConnection)
        if on_error:
            worker.error.connect(on_error, Qt.QueuedConnection)

        # Schedule cleanup after completion (use queued connection for thread safety)
        worker.finished.connect(lambda: self._schedule_cleanup(worker), Qt.QueuedConnection)
        worker.error.connect(lambda e: self._schedule_cleanup(worker), Qt.QueuedConnection)

        with self._lock:
            self._workers.append(worker)
        worker.start()
        return worker

    def _schedule_cleanup(self, worker):
        """Schedule a worker for cleanup (called from signal, so thread-safe)."""
        with self._lock:
            if worker not in self._pending_cleanup:
                self._pending_cleanup.append(worker)

    def _do_cleanup(self):
        """Actually clean up finished workers (called from main thread)."""
        with self._lock:
            for worker in self._pending_cleanup[:]:
                if worker in self._workers:
                    self._workers.remove(worker)
                if not worker.isRunning():
                    worker.deleteLater()
                self._pending_cleanup.remove(worker)

    def _periodic_cleanup(self):
        """Periodic cleanup of completed workers."""
        with self._lock:
            # Find completed workers not yet in pending cleanup
            for worker in self._workers[:]:
                if worker.is_completed and worker not in self._pending_cleanup:
                    self._pending_cleanup.append(worker)

        # Do the actual cleanup
        self._do_cleanup()

    def _force_cleanup_oldest(self):
        """Force cleanup of oldest workers when at capacity."""
        # Find finished workers and clean them immediately
        for worker in self._workers[:]:
            if not worker.isRunning():
                if worker in self._workers:
                    self._workers.remove(worker)
                worker.deleteLater()

    def stop(self):
        """Stop all running workers and cleanup timer."""
        self._cleanup_timer.stop()

        with self._lock:
            for worker in self._workers[:]:
                if worker.isRunning():
                    # Wait for natural completion rather than forcing quit
                    worker.wait(2000)
                worker.deleteLater()
            self._workers.clear()
            self._pending_cleanup.clear()

    def get_worker_count(self) -> int:
        """Get the current number of workers (for debugging)."""
        with self._lock:
            return len(self._workers)

    def get_active_worker_count(self) -> int:
        """Get the number of actively running workers."""
        with self._lock:
            return len([w for w in self._workers if w.isRunning()])


class IntelligenceController:
    """Controller for interacting with NexusIntelligence from GUI."""

    def __init__(self, async_bridge=None):
        self._intel = None
        self._initialized = False
        self._loop = None
        self._async_bridge = async_bridge
        self._discovery_overrides = {}
        self._source_scan_times = {}
        self._rag_documents = {}

    async def initialize(self, config_path: str = None):
        """Initialize the intelligence layer."""

        self._intel = NexusIntelligence(config_path)
        await self._intel.initialize()
        self._initialized = True

    async def shutdown(self):
        """Shutdown the intelligence layer."""
        if self._intel and self._initialized:
            await self._intel.shutdown()
            self._initialized = False

    @property
    def intel(self):
        return self._intel

    @property
    def is_initialized(self):
        return self._initialized

    # Task operations
    async def get_tasks(self, project_path=None, include_completed=False):
        if not self._initialized:
            return []
        return await self._intel.tasks.list_tasks(
            project_path=project_path,
            include_completed=include_completed
        )

    async def create_task(self, title, description=None, priority="medium", project_path=None, tags=None):
        if not self._initialized:
            return None
        task = Task(
            id=Task.generate_id(),
            title=title,
            description=description,
            priority=TaskPriority(priority),
            project_path=project_path,
            tags=tags or []
        )
        await self._intel.tasks.create_task(task)
        return task

    async def update_task_status(self, task_id, status):
        if not self._initialized:
            return
        await self._intel.tasks.update_task(task_id, {"status": TaskStatus(status)})

    async def complete_task(self, task_id, note=None):
        if not self._initialized:
            return None
        return await self._intel.tasks.complete_task(task_id, note)

    async def start_task(self, task_id):
        if not self._initialized:
            return None
        return await self._intel.tasks.start_task(task_id)

    # Goal operations
    async def get_goals(self, project_path=None, include_completed=False):
        if not self._initialized:
            return []
        return await self._intel.goals.list_goals(
            project_path=project_path,
            include_completed=include_completed
        )

    async def create_goal(self, title, description=None, project_path=None):
        if not self._initialized:
            return None
        goal = Goal(
            id=Goal.generate_id(),
            title=title,
            description=description,
            project_path=project_path
        )
        await self._intel.goals.create_goal(goal)
        return goal

    async def add_milestone(self, goal_id, title, description=None):
        if not self._initialized:
            return None
        milestone = Milestone(
            id=Milestone.generate_id(),
            goal_id=goal_id,
            title=title,
            description=description
        )
        await self._intel.goals.add_milestone(goal_id, milestone)
        return milestone

    # Memory operations
    async def search_memory(self, query, n_results=10, project_filter=None):
        if not self._initialized:
            return []
        return await self._intel.memory.search(query, n_results, project_filter)

    async def get_memory_stats(self):
        if not self._initialized:
            return {}
        return await self._intel.memory.get_stats()

    # Knowledge operations
    async def search_entities(self, query, entity_type=None, limit=20):
        if not self._initialized:
            return []
        etype = EntityType(entity_type) if entity_type else None
        return await self._intel.knowledge.search_entities(query, etype, limit)

    async def search_facts(self, query, topic=None, limit=10):
        if not self._initialized:
            return []
        return await self._intel.knowledge.search_facts(query, topic, limit=limit)

    async def get_knowledge_stats(self):
        if not self._initialized:
            return {}
        return await self._intel.knowledge.get_stats()

    # Decision operations
    async def get_decisions(self, project_path=None, search_term=None, limit=20):
        if not self._initialized:
            return []
        return await self._intel.decisions.list_decisions(
            project_path=project_path,
            search_term=search_term,
            limit=limit
        )

    async def record_decision(self, question, decision, reasoning, alternatives=None, project_path=None):
        if not self._initialized:
            return None
        dec = Decision(
            id=Decision.generate_id(),
            question=question,
            decision=decision,
            reasoning=reasoning,
            alternatives=alternatives or [],
            project_path=project_path
        )
        await self._intel.decisions.record_decision(dec)
        return dec

    # Continuity operations
    async def get_focus_context(self, project_path=None):
        if not self._initialized:
            return None
        return await self._intel.continuity.get_current_focus(project_path)

    async def start_session(self, project_path=None):
        if not self._initialized:
            return None
        return await self._intel.continuity.start_session(project_path)

    async def end_session(self, summary=None, handoff_notes=None):
        if not self._initialized:
            return None
        return await self._intel.continuity.end_session(summary, handoff_notes)

    # Preferences
    async def get_preferences(self, category=None):
        if not self._initialized:
            return []
        cat = PreferenceCategory(category) if category else None
        return await self._intel.preferences.get_preferences(cat)

    # Corrections
    async def get_corrections(self, limit=20):
        if not self._initialized:
            return []
        return await self._intel.corrections.get_corrections(limit=limit)

    # Dashboard stats
    async def get_dashboard_stats(self):
        """Get statistics for the dashboard."""
        if not self._initialized:
            return {
                "total_tasks": 0,
                "active_tasks": 0,
                "completed_tasks": 0,
                "total_goals": 0,
                "active_goals": 0,
                "total_memories": 0,
                "total_entities": 0,
                "total_facts": 0,
                "total_decisions": 0
            }

        # Gather stats from all modules
        tasks = await self._intel.tasks.list_tasks(include_completed=True)
        active_tasks = [t for t in tasks if t.status.value not in ("completed", "cancelled")]
        completed_tasks = [t for t in tasks if t.status.value == "completed"]

        goals = await self._intel.goals.list_goals(include_completed=True)
        active_goals = [g for g in goals if g.status.value not in ("completed", "abandoned")]

        memory_stats = await self._intel.memory.get_stats()
        knowledge_stats = await self._intel.knowledge.get_stats()
        decisions = await self._intel.decisions.list_decisions(limit=1000)

        return {
            "total_tasks": len(tasks),
            "active_tasks": len(active_tasks),
            "completed_tasks": len(completed_tasks),
            "total_goals": len(goals),
            "active_goals": len(active_goals),
            "total_memories": memory_stats.get("total_chunks", 0),
            "total_entities": knowledge_stats.get("total_entities", 0),
            "total_facts": knowledge_stats.get("total_facts", 0),
            "total_decisions": len(decisions)
        }

    # Update task with data dict
    async def update_task(self, task_id: str, data: dict):
        """Update a task with the provided data dictionary."""
        if not self._initialized:
            return None

        updates = {}
        if "status" in data:
            updates["status"] = TaskStatus(data["status"])
        if "priority" in data:
            updates["priority"] = TaskPriority(data["priority"])
        if "title" in data:
            updates["title"] = data["title"]
        if "description" in data:
            updates["description"] = data["description"]
        if "tags" in data:
            updates["tags"] = data["tags"]

        if updates:
            await self._intel.tasks.update_task(task_id, updates)

    # Search decisions
    async def search_decisions(self, query: str, limit: int = 20):
        """Search decisions by query."""
        if not self._initialized:
            return []
        return await self._intel.decisions.list_decisions(search_term=query, limit=limit)

    # Topic history
    async def get_topic_history(self, topic: str = None, limit: int = 20):
        """Get topic history from memory. If no topic, returns empty list."""
        if not self._initialized:
            return []
        if not topic:
            # Return an empty list if no specific topic is provided
            return []
        return await self._intel.memory.topic_history(topic=topic, limit=limit)

    # Test Ollama connection
    async def test_ollama(self):
        """Test connection to Ollama embedding service."""
        if not self._initialized:
            return False, "Intelligence layer not initialized"

        try:
            import httpx
            # Get the embedder URL from the intelligence layer config
            url = self._intel.config.get("embedding", {}).get("ollama_url", "http://localhost:11434")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return True, f"Connected! Available models: {', '.join(models[:5])}"
                else:
                    return False, f"Ollama responded with status {response.status_code}"
        except httpx.ConnectError:
            return False, "Cannot connect to Ollama. Is it running?"
        except Exception as e:
            return False, f"Error: {str(e)}"

    # Chat operations
    async def chat(self, message: str, history: list = None):
        """Send a chat message and get a response using the UnifiedEnsemble.

        Uses the UnifiedEnsemble to orchestrate model selection, execute queries,
        synthesize responses, and automatically track costs and metrics.

        Args:
            message: The user's message
            history: Previous conversation history

        Returns:
            Dict with 'content' (response text) and 'model_info' (selection details)
        """
        import os
        import time
        import logging
        from pathlib import Path
        from uuid import uuid4

        # Load environment variables
        from dotenv import load_dotenv
        project_root = Path(__file__).parent.parent.parent.parent
        load_dotenv(project_root / ".env", override=True)

        start_time = time.time()

        # Build system prompt with Nexus context
        system_prompt = """You are Nexus, an intelligent assistant that helps users manage their tasks, goals, and projects.
You have access to the user's conversation history, memories, and project context.
Be helpful, concise, and proactive in suggesting next steps.
When discussing tasks or goals, be specific and actionable."""

        # Add memory/context if available
        memory_context = ""
        if self._initialized:
            try:
                # Search for relevant context
                memories = await self._intel.memory.search(message, n_results=3)
                if memories:
                    memory_context = "\n\nRelevant context from memory:\n"
                    for mem in memories:
                        memory_context += f"- {mem.content[:200]}...\n"
            except Exception:
                pass  # Proceed without context if search fails

        # Build the full prompt with conversation history
        full_prompt = system_prompt + memory_context + "\n\n"
        if history:
            full_prompt += "Previous conversation:\n"
            for msg in history[-10:]:  # Last 10 messages
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")[:500]  # Truncate long messages
                full_prompt += f"{role}: {content}\n"
            full_prompt += "\n"
        full_prompt += f"User: {message}"

        # Try to use UnifiedEnsemble first (preferred path)
        ensemble = getattr(self._intel, '_ensemble', None) if self._initialized else None

        if ensemble:
            try:
                return await self._chat_with_ensemble(
                    message=message,
                    full_prompt=full_prompt,
                    ensemble=ensemble,
                    start_time=start_time
                )
            except Exception as e:
                logging.warning(f"Ensemble failed: {e}, falling back to direct calls")

        # Fallback: Use direct API calls with metric recording
        task_requirements = self._analyze_task_requirements(message)

        try:
            result = await self._chat_with_intelligent_selection(
                full_prompt,
                system_prompt + memory_context,
                task_requirements
            )
            # Record metrics for fallback path
            await self._record_response_metrics(result, start_time)
            return result
        except Exception as e:
            logging.warning(f"Intelligent selection failed: {e}, using NexusLLM fallback")

        # Fallback to NexusLLM
        try:
            task_type = self._classify_task(message)
            result = await self._chat_with_nexus_llm(
                full_prompt,
                system_prompt + memory_context,
                task_type
            )
            await self._record_response_metrics(result, start_time)
            return result
        except Exception as e:
            logging.warning(f"NexusLLM failed: {e}, using direct API fallback")

        # Final fallback to direct API
        result = await self._chat_direct(message, history, system_prompt + memory_context)
        await self._record_response_metrics(result, start_time)
        return result

    async def _chat_with_ensemble(self, message: str, full_prompt: str, ensemble, start_time: float) -> dict:
        """Process chat through the UnifiedEnsemble for full orchestration."""
        drift_monitor = getattr(self._intel, '_drift_monitor', None)

        if hasattr(ensemble, "query"):
            response = await ensemble.query(
                full_prompt,
                strategy="simple_best",
                task_type=self._classify_task(message),
                max_tokens=2000,
            )
            if isinstance(response, dict) and response.get("error"):
                raise RuntimeError(str(response["error"]))
            models_used = response.get("models_queried")
            if not isinstance(models_used, list):
                metadata = response.get("metadata", {})
                all_scores = metadata.get("all_scores", {}) if isinstance(metadata, dict) else {}
                models_used = list(all_scores.keys()) or [response.get("model_name", "ensemble")]

            primary_model = response.get("model_name", models_used[0] if models_used else "ensemble")

            if drift_monitor:
                drift_monitor.record_response(
                    model_name=primary_model,
                    latency_ms=response.get("total_latency_ms", 0.0),
                    confidence=response.get("confidence", 0.7),
                    success=bool(response.get("content")),
                    tokens_used=response.get("tokens_used", 0),
                    cost_usd=response.get("total_cost", 0.0),
                    response_length=len(response.get("content", "")),
                )

            return {
                "content": response.get("content", ""),
                "model_info": {
                    "model": primary_model,
                    "provider": response.get("provider", "ensemble"),
                    "models_tried": models_used,
                    "reasoning": f"Ensemble: {response.get('strategy_used', 'simple_best')}, {len(models_used)} models, confidence: {response.get('confidence', 0.0):.2f}",
                    "total_models_considered": len(models_used),
                    "confidence": response.get("confidence", 0.7),
                    "cost_usd": response.get("total_cost", 0.0),
                    "latency_ms": response.get("total_latency_ms", 0.0),
                    "epistemic_health": None,
                },
                "request_id": response.get("request_id"),
            }

        if hasattr(ensemble, "process"):
            from uuid import uuid4
            from nexus.providers.ensemble.types import EnsembleRequest

            request = EnsembleRequest(
                query=full_prompt,
                request_id=uuid4(),
                user_id="gui_user",
                max_models=3,
                temperature=0.7,
                max_tokens=2000,
            )

            # Process through ensemble (this handles model selection, execution, synthesis, cost tracking)
            response = await ensemble.process(request)

            # Record to drift monitor if available
            if drift_monitor and response.model_responses:
                for model_resp in response.model_responses:
                    if not model_resp.error:
                        drift_monitor.record_response(
                            model_name=model_resp.model_name,
                            latency_ms=model_resp.latency_ms,
                            confidence=model_resp.confidence,
                            success=True,
                            tokens_used=model_resp.tokens_used,
                            cost_usd=model_resp.cost_usd,
                            response_length=len(model_resp.content) if model_resp.content else 0
                        )

            # Build response dict for GUI
            models_used = [r.model_name for r in response.model_responses if not r.error]
            primary_model = models_used[0] if models_used else "ensemble"

            return {
                "content": response.content,
                "model_info": {
                    "model": primary_model,
                    "provider": "ensemble",
                    "models_tried": models_used,
                    "reasoning": f"Ensemble: {response.strategy_used}, {len(models_used)} models, confidence: {response.confidence:.2f}",
                    "total_models_considered": response.models_queried,
                    "confidence": response.confidence,
                    "cost_usd": response.total_cost_usd,
                    "latency_ms": response.total_latency_ms,
                    "epistemic_health": response.epistemic_health.consistency_score if response.epistemic_health else None,
                },
                "request_id": str(response.request_id),
            }

        raise RuntimeError("Ensemble backend does not support query or process execution")

    async def _record_response_metrics(self, result: dict, start_time: float):
        """Record response metrics to drift monitor for non-ensemble paths."""
        import time

        if not self._initialized:
            return

        drift_monitor = getattr(self._intel, '_drift_monitor', None)
        if not drift_monitor:
            return

        try:
            elapsed_ms = (time.time() - start_time) * 1000
            model_info = result.get("model_info", {})
            model_name = model_info.get("model", "unknown")

            drift_monitor.record_response(
                model_name=model_name,
                latency_ms=elapsed_ms,
                confidence=model_info.get("confidence", 0.7),
                success=bool(result.get("content")),
                tokens_used=0,  # Not tracked in fallback paths
                cost_usd=model_info.get("cost_usd", 0.0),
                response_length=len(result.get("content", ""))
            )
        except Exception as e:
            logger.warning("Could not record response metrics: %s", e)

    def _analyze_task_requirements(self, message: str) -> dict:
        """Analyze the message to determine task requirements for intelligent model selection.

        Returns a dict compatible with TaskRequirements from intelligent_selector.
        """
        message_lower = message.lower()

        # Determine primary capability needed
        capability = "TEXT_GENERATION"  # Default
        if any(kw in message_lower for kw in ["code", "function", "debug", "implement", "refactor", "programming"]):
            capability = "CODE_GENERATION"
        elif any(kw in message_lower for kw in ["image", "picture", "photo", "diagram"]):
            capability = "VISION"
        elif any(kw in message_lower for kw in ["math", "equation", "calculate", "formula"]):
            capability = "REASONING"

        # Determine complexity
        complexity = "MODERATE"  # Default
        word_count = len(message.split())
        if word_count < 10 and any(kw in message_lower for kw in ["what is", "define", "quick"]):
            complexity = "SIMPLE"
        elif any(kw in message_lower for kw in ["analyze", "compare", "evaluate", "design", "architect"]):
            complexity = "COMPLEX"
        elif any(kw in message_lower for kw in ["critical", "production", "security", "important"]):
            complexity = "EXPERT"

        # Determine criticality
        criticality = "STANDARD"  # Default
        if any(kw in message_lower for kw in ["quick", "simple", "casual"]):
            criticality = "CASUAL"
        elif any(kw in message_lower for kw in ["important", "business"]):
            criticality = "IMPORTANT"
        elif any(kw in message_lower for kw in ["critical", "production", "security"]):
            criticality = "CRITICAL"
        elif any(kw in message_lower for kw in ["research", "explore", "compare multiple"]):
            criticality = "RESEARCH"

        # Determine domain
        domain = "GENERAL"
        if capability == "CODE_GENERATION":
            domain = "CODE"
        elif any(kw in message_lower for kw in ["math", "equation", "calculate"]):
            domain = "MATHEMATICS"
        elif any(kw in message_lower for kw in ["write", "blog", "article", "story"]):
            domain = "CREATIVE"
        elif any(kw in message_lower for kw in ["reason", "logic", "deduce"]):
            domain = "REASONING"

        return {
            "capability": capability,
            "complexity": complexity,
            "criticality": criticality,
            "domain": domain,
            "input_tokens": len(message.split()) * 2,  # Rough estimate
        }

    async def _chat_with_intelligent_selection(self, prompt: str, system_prompt: str, requirements: dict) -> dict:
        """Use IntelligentModelSelector with DynamicModelRegistry for model selection."""
        import os

        # Import the intelligent selection system
        from nexus.providers.adapters.dynamic_registry import DynamicModelRegistry, TaskCriticality
        from nexus.providers.adapters.intelligent_selector import (
            IntelligentModelSelector, TaskRequirements, TaskComplexity, TaskDomain, ModelCapability
        )

        # Initialize registry (with OpenRouter for 100+ models)
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        registry = DynamicModelRegistry(
            openrouter_api_key=openrouter_key,
            auto_discover=True,
        )
        await registry.initialize()

        # Map string requirements to enums
        capability_map = {
            "TEXT_GENERATION": ModelCapability.TEXT_GENERATION,
            "CODE_GENERATION": ModelCapability.CODE_GENERATION,
            "VISION": ModelCapability.VISION,
            "REASONING": ModelCapability.REASONING,
        }
        complexity_map = {
            "SIMPLE": TaskComplexity.SIMPLE,
            "MODERATE": TaskComplexity.MODERATE,
            "COMPLEX": TaskComplexity.COMPLEX,
            "EXPERT": TaskComplexity.EXPERT,
        }
        criticality_map = {
            "CASUAL": TaskCriticality.CASUAL,
            "STANDARD": TaskCriticality.STANDARD,
            "IMPORTANT": TaskCriticality.IMPORTANT,
            "CRITICAL": TaskCriticality.CRITICAL,
            "RESEARCH": TaskCriticality.RESEARCH,
        }
        domain_map = {
            "GENERAL": TaskDomain.GENERAL,
            "CODE": TaskDomain.CODE,
            "MATHEMATICS": TaskDomain.MATHEMATICS,
            "CREATIVE": TaskDomain.CREATIVE,
            "REASONING": TaskDomain.REASONING,
        }

        # Build TaskRequirements
        task_req = TaskRequirements(
            primary_capability=capability_map.get(requirements["capability"], ModelCapability.TEXT_GENERATION),
            complexity=complexity_map.get(requirements["complexity"], TaskComplexity.MODERATE),
            criticality=criticality_map.get(requirements["criticality"], TaskCriticality.STANDARD),
            domain=domain_map.get(requirements["domain"], TaskDomain.GENERAL),
            input_tokens=requirements.get("input_tokens", 500),
        )

        # Create selector and select models
        selector = IntelligentModelSelector(registry)
        selection = await selector.select_models(task_req)

        # Log selection
        import logging
        logging.info(f"Selected {len(selection.primary_models)} primary models: {selection.primary_models}")
        logging.info(f"Selection reasoning: {selection.selection_reasoning}")

        # For now, use the first primary model via our existing backends
        # In a full implementation, this would call the models directly through OpenRouter
        primary_model = selection.primary_models[0] if selection.primary_models else "ollama-csuite-model"

        # Determine which backend to use based on selected model
        if "anthropic" in primary_model.lower() or "claude" in primary_model.lower():
            content = await self._call_anthropic(prompt, system_prompt)
            provider = "anthropic"
        elif "openai" in primary_model.lower() or "gpt" in primary_model.lower():
            content = await self._call_openai(prompt, system_prompt)
            provider = "openai"
        elif "openrouter" in primary_model.lower() or "/" in primary_model:
            content = await self._call_openrouter(prompt, system_prompt, primary_model)
            provider = "openrouter"
        else:
            # Default to Ollama for local models
            content = await self._call_ollama(prompt, system_prompt)
            provider = "ollama"

        # Clean up registry
        await registry.cleanup()

        return {
            "content": content,
            "model_info": {
                "model": primary_model,
                "provider": provider,
                "models_tried": selection.primary_models[:3],  # Show first 3
                "reasoning": selection.selection_reasoning,
                "total_models_considered": len(selection.primary_models) + len(selection.fallback_models),
            }
        }

    async def _chat_with_nexus_llm(self, prompt: str, system_prompt: str, task_type: str) -> dict:
        """Use NexusLLM for model routing."""
        from nexus.core.llm_provider import NexusLLM

        llm = NexusLLM()
        response = await llm.generate(
            prompt=prompt,
            task_type=task_type,
            system_prompt=system_prompt,
            max_tokens=2000,
        )

        return {
            "content": response["content"],
            "model_info": {
                "model": response.get("model", "Unknown"),
                "provider": response.get("provider", "Unknown"),
                "models_tried": response.get("models_tried", []),
                "reasoning": f"NexusLLM routing: task_type={task_type}",
            }
        }

    def _classify_task(self, message: str) -> str:
        """Classify the task type based on message content for routing."""
        message_lower = message.lower()

        # Code-related tasks
        if any(kw in message_lower for kw in ["code", "function", "debug", "error", "implement", "refactor"]):
            return "code_review"

        # Architecture/design tasks
        if any(kw in message_lower for kw in ["architecture", "design", "structure", "pattern", "system"]):
            return "architecture_design"

        # Research tasks
        if any(kw in message_lower for kw in ["research", "analyze", "compare", "evaluate"]):
            return "research"

        # Writing tasks
        if any(kw in message_lower for kw in ["write", "draft", "compose", "blog", "article"]):
            return "content_writing"

        # Simple lookups
        if any(kw in message_lower for kw in ["what is", "how do", "define", "explain"]):
            return "factual_lookup"

        # Default to conversation
        return "conversation"

    async def _chat_direct(self, message: str, history: list, system_prompt: str) -> dict:
        """Direct API fallback when intelligent selection is unavailable."""
        messages = []
        if history:
            for msg in history[-10:]:
                role = "user" if msg.get("role") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
        messages.append({"role": "user", "content": message})

        # Try Anthropic first, then OpenAI, then Ollama
        try:
            content = await self._call_anthropic_messages(messages, system_prompt)
            return {"content": content, "model_info": {"model": "claude-sonnet-4", "provider": "anthropic", "models_tried": ["anthropic"], "reasoning": "Direct fallback"}}
        except Exception:
            pass

        try:
            content = await self._call_openai_messages(messages, system_prompt)
            return {"content": content, "model_info": {"model": "gpt-4o", "provider": "openai", "models_tried": ["openai"], "reasoning": "Direct fallback"}}
        except Exception:
            pass

        content = await self._call_ollama_messages(messages, system_prompt)
        return {"content": content, "model_info": {"model": "qwen3:30b", "provider": "ollama", "models_tried": ["ollama"], "reasoning": "Direct fallback"}}

    async def _call_anthropic(self, prompt: str, system_prompt: str) -> str:
        """Call Anthropic API with a single prompt."""
        import os
        import asyncio
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        return response.content[0].text

    async def _call_anthropic_messages(self, messages: list, system_prompt: str) -> str:
        """Call Anthropic API with message history."""
        import os
        import asyncio
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = anthropic.Anthropic(api_key=api_key)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                system=system_prompt,
                messages=messages
            )
        )
        return response.content[0].text

    async def _call_openai(self, prompt: str, system_prompt: str) -> str:
        """Call OpenAI API with a single prompt."""
        import os
        import asyncio
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,
                messages=messages
            )
        )
        return response.choices[0].message.content

    async def _call_openai_messages(self, messages: list, system_prompt: str) -> str:
        """Call OpenAI API with message history."""
        import os
        import asyncio
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4o",
                max_tokens=2000,
                messages=full_messages
            )
        )
        return response.choices[0].message.content

    async def _call_ollama(self, prompt: str, system_prompt: str) -> str:
        """Call Ollama API with a single prompt."""
        import httpx
        import re

        url = self._intel.config.get("embedding", {}).get("ollama_url", "http://localhost:11434") if self._initialized else "http://localhost:11434"

        payload = {
            "model": "qwen3:30b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"num_predict": 2000, "num_ctx": 16384}
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{url}/api/chat", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content
            else:
                raise RuntimeError(f"Ollama error: {resp.status_code}")

    async def _call_ollama_messages(self, messages: list, system_prompt: str) -> str:
        """Call Ollama API with message history."""
        import httpx
        import re

        url = self._intel.config.get("embedding", {}).get("ollama_url", "http://localhost:11434") if self._initialized else "http://localhost:11434"

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        payload = {
            "model": "qwen3:30b",
            "messages": full_messages,
            "stream": False,
            "options": {"num_predict": 2000, "num_ctx": 16384}
        }

        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(f"{url}/api/chat", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                return content
            else:
                raise RuntimeError(f"Ollama error: {resp.status_code}")

    async def _call_openrouter(self, prompt: str, system_prompt: str, model: str) -> str:
        """Call OpenRouter API for access to 100+ models."""
        import os
        import httpx

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://nexus-intelligence.ai",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                raise RuntimeError(f"OpenRouter error: {resp.status_code} - {resp.text}")

    # ============================================================
    # COO (Chief Operating Officer) Operations
    # ============================================================

    async def get_coo_status(self) -> dict:
        """Get current COO status."""
        if not self._initialized:
            return {"state": "not_initialized", "mode": "unknown"}

        coo = self._intel.get_coo()
        if not coo:
            return {"state": "not_available", "mode": "unknown"}

        # Handle both legacy and new C-suite COO
        if hasattr(coo, 'get_status'):
            status = coo.get_status()
            # Legacy COO returns a Status object with to_dict()
            if hasattr(status, 'to_dict'):
                return status.to_dict()
            # New C-suite COO returns a dict directly
            elif isinstance(status, dict):
                return status
            else:
                return {"state": str(status), "mode": "unknown"}
        else:
            # Fallback for C-suite COO without get_status
            return {
                "state": "running" if getattr(coo, '_running', False) else "idle",
                "mode": "autonomous",
                "agent": coo.code if hasattr(coo, 'code') else "COO",
                "goals": len(getattr(coo, '_goals', {})),
                "tasks": len(getattr(coo, '_tasks', {})),
            }

    async def start_coo(self) -> bool:
        """Start the Autonomous COO."""
        logger.info(
            "start_coo called: initialized=%s intel=%s coo=%s",
            self._initialized,
            self._intel,
            getattr(self._intel, "_coo", "N/A") if self._intel else "N/A",
        )
        if not self._initialized:
            logger.info("start_coo: not initialized, returning False")
            return False
        result = await self._intel.start_coo()
        logger.info("start_coo: intel.start_coo() returned %s", result)
        return result

    async def stop_coo(self) -> bool:
        """Stop the Autonomous COO."""
        if not self._initialized:
            return False
        await self._intel.stop_coo()
        return True

    async def set_coo_mode(self, mode: str) -> bool:
        """Set COO operating mode."""
        if not self._initialized:
            return False

        coo = self._intel.get_coo()
        if not coo:
            return False

        from nexus.coo import ExecutionMode
        try:
            coo.set_mode(ExecutionMode(mode))
            return True
        except Exception as e:
            logger.warning("Error setting COO mode: %s", e)
            return False

    async def get_coo_suggestions(self) -> dict:
        """Get COO's suggestion for next action."""
        if not self._initialized:
            return {"suggestion": None, "reason": "Not initialized"}

        coo = self._intel.get_coo()
        if not coo:
            return {"suggestion": None, "reason": "COO not available"}

        # Handle both legacy and new C-suite COO
        if hasattr(coo, 'suggest_next_action'):
            # Legacy COO
            return await coo.suggest_next_action()
        elif hasattr(coo, 'generate_executive_report'):
            # New C-suite COO - generate a suggestion from the executive report
            try:
                report = await coo.generate_executive_report()
                # Find the most relevant suggestion from the report
                goal_summary = report.get("goal_summary", [])
                if goal_summary:
                    top_goal = goal_summary[0]
                    return {
                        "suggestion": {
                            "item": {"title": top_goal.get("title", "No title")},
                            "score": {"total_score": 0.8}
                        },
                        "reason": f"Top goal: {top_goal.get('status', 'active')}",
                        "decision": {
                            "executor": "COO",
                            "confidence": 0.8,
                            "reason": report.get("executive_summary", "No summary available")
                        },
                        "context_summary": {
                            "active_goals": report.get("active_goals", 0),
                            "pending_tasks": 0,
                            "blockers": report.get("issues_count", 0)
                        }
                    }
                else:
                    return {
                        "suggestion": None,
                        "reason": "No active goals",
                        "context_summary": {
                            "active_goals": report.get("active_goals", 0),
                            "pending_tasks": 0,
                            "blockers": report.get("issues_count", 0)
                        }
                    }
            except Exception as e:
                return {"suggestion": None, "reason": f"Error: {e}"}
        else:
            return {"suggestion": None, "reason": "COO does not support suggestions"}

    async def approve_coo_action(self, item_id: str, approved: bool, notes: str = None) -> bool:
        """Approve or reject a pending COO action."""
        if not self._initialized:
            return False

        coo = self._intel.get_coo()
        if not coo:
            return False

        return await coo.approve(item_id, approved, notes)

    async def execute_task_now(self, task_id: str) -> bool:
        """Manually trigger task execution via COO."""
        if not self._initialized:
            return False

        coo = self._intel.get_coo()
        if not coo:
            return False

        return await coo.execute_now(task_id)

    async def get_coo_learning_stats(self) -> dict:
        """Get COO learning statistics."""
        if not self._initialized:
            return {}

        coo = self._intel.get_coo()
        if not coo or not coo._learning:
            return {}

        stats = await coo._learning.get_stats()
        return stats.to_dict()

    async def get_coo_execution_history(self, limit: int = 50) -> list:
        """Get recent COO execution history."""
        if not self._initialized:
            return []

        coo = self._intel.get_coo()
        if not coo or not coo._executor:
            return []

        history = coo._executor.get_execution_history(limit)
        return [r.to_dict() for r in history]

    # ============================================================
    # Models / Ensemble Operations
    # ============================================================

    async def get_ensemble_status(self) -> dict:
        """Get current ensemble status including model health and strategy."""
        if not self._initialized:
            return self._get_mock_ensemble_data()

        try:
            # Try to get real ensemble data from the ensemble core
            from nexus.providers.ensemble.core import EnsembleOrchestrator
            orchestrator = getattr(self._intel, '_ensemble', None)
            if orchestrator:
                status = await orchestrator.get_status()
                return {
                    "models": status.get("models", []),
                    "strategy": status.get("strategy", "weighted_quality"),
                    "health": status.get("health", {}),
                    "quarantined": status.get("quarantined", []),
                }
        except Exception as e:
            logger.warning("Error getting ensemble status: %s", e)

        return self._get_mock_ensemble_data()

    def _get_mock_ensemble_data(self) -> dict:
        """Return mock ensemble data for UI development."""
        return {
            "models": [
                {"id": "claude-sonnet-4", "provider": "anthropic", "status": "active", "health": 98, "requests": 1250, "avg_latency": 1.2, "cost_per_1k": 0.003},
                {"id": "gpt-4o", "provider": "openai", "status": "active", "health": 95, "requests": 890, "avg_latency": 1.8, "cost_per_1k": 0.005},
                {"id": "qwen3:30b", "provider": "ollama", "status": "active", "health": 92, "requests": 2100, "avg_latency": 0.8, "cost_per_1k": 0.0},
                {"id": "llama3.3:70b", "provider": "ollama", "status": "degraded", "health": 75, "requests": 450, "avg_latency": 2.5, "cost_per_1k": 0.0},
                {"id": "mistral-large", "provider": "openrouter", "status": "quarantined", "health": 45, "requests": 120, "avg_latency": 3.2, "cost_per_1k": 0.002},
            ],
            "strategy": "weighted_quality",
            "strategies_available": ["weighted_quality", "lowest_latency", "lowest_cost", "round_robin", "capability_match"],
            "health": {"overall": 88, "degraded_count": 1, "quarantined_count": 1},
            "quarantined": ["mistral-large"],
        }

    async def set_ensemble_strategy(self, strategy: str):
        """Set the ensemble selection strategy."""
        if not self._initialized:
            return

        try:
            from nexus.providers.ensemble.core import EnsembleOrchestrator
            orchestrator = getattr(self._intel, '_ensemble', None)
            if orchestrator:
                await orchestrator.set_strategy(strategy)
        except Exception as e:
            logger.warning("Error setting ensemble strategy: %s", e)

    async def quarantine_model(self, model_id: str):
        """Quarantine a model from the ensemble."""
        if not self._initialized:
            return

        try:
            orchestrator = getattr(self._intel, '_ensemble', None)
            if orchestrator:
                # quarantine_model is sync, not async
                orchestrator.quarantine_model(model_id, "User requested quarantine")
        except Exception as e:
            logger.warning("Error quarantining model: %s", e)

    async def release_model(self, model_id: str):
        """Release a quarantined model back to the ensemble."""
        if not self._initialized:
            return

        try:
            orchestrator = getattr(self._intel, '_ensemble', None)
            if orchestrator:
                # release_from_quarantine is sync, not async
                orchestrator.release_from_quarantine(model_id)
        except Exception as e:
            logger.warning("Error releasing model: %s", e)

    # ============================================================
    # Monitoring Operations
    # ============================================================

    async def get_monitoring_data(self, time_range: str = "1h") -> dict:
        """Get monitoring data including drift alerts, metrics, and costs."""
        if not self._initialized:
            return self._get_mock_monitoring_data()

        try:
            from datetime import datetime, timedelta

            monitor = getattr(self._intel, '_drift_monitor', None)
            tracker = getattr(self._intel, '_feedback_tracker', None)

            data = {"alerts": [], "metrics": {}, "cost": {}, "circuit_breakers": []}

            # Parse time_range to datetime
            time_map = {"1h": 1, "6h": 6, "24h": 24, "7d": 168}
            hours = time_map.get(time_range, 1)
            since = datetime.now() - timedelta(hours=hours)

            # Get drift alerts (sync method)
            if monitor:
                raw_alerts = monitor.get_recent_alerts(since=since)
                data["alerts"] = [
                    {
                        "id": f"alert-{i}",
                        "type": alert.drift_type.value if hasattr(alert.drift_type, 'value') else str(alert.drift_type),
                        "severity": alert.severity.value if hasattr(alert.severity, 'value') else str(alert.severity),
                        "message": alert.message,
                        "model": alert.model_name,
                        "timestamp": alert.timestamp.isoformat() if alert.timestamp else None,
                    }
                    for i, alert in enumerate(raw_alerts)
                ]

            # Get feedback stats (sync methods)
            if tracker:
                all_stats = tracker.get_all_model_stats()
                total_feedback = sum(s.total_feedback for s in all_stats.values())
                avg_score = sum(s.average_score * s.total_feedback for s in all_stats.values()) / max(total_feedback, 1)

                data["metrics"] = {
                    "total_requests": total_feedback,
                    "avg_latency": 0,  # Not tracked by feedback tracker
                    "success_rate": avg_score * 100,  # Convert 0-1 to percentage
                    "error_rate": (1 - avg_score) * 100,
                    "cache_hit_rate": 0,
                    "tokens_processed": 0,
                }

            # Cost data would come from CostTracker if available
            ensemble = getattr(self._intel, '_ensemble', None)
            if ensemble and hasattr(ensemble, 'cost_tracker'):
                # Would get real cost data here
                pass

            return data if data["alerts"] or data["metrics"] else self._get_mock_monitoring_data()
        except Exception as e:
            logger.warning("Error getting monitoring data: %s", e)

        return self._get_mock_monitoring_data()

    def _get_mock_monitoring_data(self) -> dict:
        """Return mock monitoring data for UI development."""
        from datetime import datetime, timedelta
        now = datetime.now()

        return {
            "alerts": [
                {"id": "alert-1", "type": "performance_drift", "severity": "warning", "message": "Model latency increased 40% in last hour", "model": "gpt-4o", "timestamp": (now - timedelta(minutes=15)).isoformat()},
                {"id": "alert-2", "type": "quality_drift", "severity": "critical", "message": "Response quality score dropped below threshold", "model": "mistral-large", "timestamp": (now - timedelta(minutes=45)).isoformat()},
                {"id": "alert-3", "type": "cost_alert", "severity": "info", "message": "Daily budget 80% consumed", "model": "all", "timestamp": (now - timedelta(hours=2)).isoformat()},
            ],
            "metrics": {
                "total_requests": 4810,
                "avg_latency": 1.45,
                "success_rate": 98.2,
                "error_rate": 1.8,
                "cache_hit_rate": 34.5,
                "tokens_processed": 2_450_000,
            },
            "cost": {
                "today": 12.45,
                "this_week": 78.90,
                "this_month": 245.30,
                "budget_remaining": 254.70,
                "by_provider": {"anthropic": 45.20, "openai": 33.10, "openrouter": 12.00, "ollama": 0.0},
            },
            "circuit_breakers": [
                {"model": "mistral-large", "status": "open", "failures": 5, "last_failure": (now - timedelta(minutes=30)).isoformat()},
            ],
        }

    async def dismiss_alert(self, alert_id: str):
        """Dismiss a monitoring alert."""
        if not self._initialized:
            return

        # Note: DriftMonitor doesn't have dismiss_alert yet
        # Alerts are stored in a deque and auto-expire
        # For now, this is a no-op but could be implemented later
        logger.info("Alert %s acknowledged (auto-expires from monitor)", alert_id)

    # ============================================================
    # Discovery Operations
    # ============================================================

    async def get_discovery_data(self) -> dict:
        """Get discovered resources from various sources."""
        if not self._initialized:
            return self._apply_discovery_overrides(self._get_mock_discovery_data())

        try:
            registry = getattr(self._intel, '_model_registry', None)
            if registry:
                # Get available models (sync method)
                model_names = registry.get_available_models(include_unhealthy=True)

                # Build resources by source/provider
                sources = {
                    "ollama": {"status": "connected", "resources": []},
                    "openrouter": {"status": "connected", "resources": []},
                    "openai": {"status": "connected", "resources": []},
                    "anthropic": {"status": "connected", "resources": []},
                    "local": {"status": "connected", "resources": []},
                }

                for name in model_names:
                    info = registry.get_model_info(name)
                    if info:
                        resource = {
                            "id": name,
                            "name": info.display_name or name,
                            "type": "model",
                            "status": "available" if name not in registry._unhealthy_models else "unhealthy",
                            "description": "",
                        }

                        # Add provider-specific info
                        provider = info.provider.lower() if info.provider else "local"
                        if provider in sources:
                            if hasattr(info, 'context_window'):
                                resource["context"] = f"{info.context_window // 1000}K"
                            if hasattr(info, 'cost_per_1k_input'):
                                resource["pricing"] = f"${info.cost_per_1k_input}/1K"
                            sources[provider]["resources"].append(resource)
                        else:
                            sources["local"]["resources"].append(resource)

                # Remove empty sources
                sources = {k: v for k, v in sources.items() if v["resources"]}

                if sources:
                    return self._apply_discovery_overrides({"sources": sources})
        except Exception as e:
            logger.warning("Error getting discovery data: %s", e)

        return self._apply_discovery_overrides(self._get_mock_discovery_data())

    def _get_mock_discovery_data(self) -> dict:
        """Return mock discovery data for UI development."""
        return {
            "sources": {
                "ollama": {
                    "status": "connected",
                    "resources": [
                        {"id": "qwen3:30b", "type": "model", "name": "Qwen 3 30B", "size": "18GB", "status": "available"},
                        {"id": "llama3.3:70b", "type": "model", "name": "Llama 3.3 70B", "size": "40GB", "status": "available"},
                        {"id": "nomic-embed-text", "type": "embedding", "name": "Nomic Embed Text", "size": "274MB", "status": "available"},
                    ],
                },
                "huggingface": {
                    "status": "connected",
                    "resources": [
                        {"id": "meta-llama/Llama-3-70b", "type": "model", "name": "Llama 3 70B", "downloads": "2.5M", "status": "available"},
                        {"id": "sentence-transformers/all-MiniLM-L6-v2", "type": "embedding", "name": "MiniLM L6", "downloads": "50M", "status": "available"},
                    ],
                },
                "openrouter": {
                    "status": "connected",
                    "resources": [
                        {"id": "anthropic/claude-3.5-sonnet", "type": "model", "name": "Claude 3.5 Sonnet", "pricing": "$3/1M tokens", "status": "available"},
                        {"id": "openai/gpt-4o", "type": "model", "name": "GPT-4o", "pricing": "$5/1M tokens", "status": "available"},
                        {"id": "google/gemini-pro-1.5", "type": "model", "name": "Gemini Pro 1.5", "pricing": "$3.5/1M tokens", "status": "available"},
                    ],
                },
                "github": {
                    "status": "connected",
                    "resources": [
                        {"id": "langchain-ai/langchain", "type": "library", "name": "LangChain", "stars": "75K", "status": "available"},
                        {"id": "huggingface/transformers", "type": "library", "name": "Transformers", "stars": "120K", "status": "available"},
                    ],
                },
                "local": {
                    "status": "connected",
                    "resources": [
                        {"id": "local/custom-rag", "type": "pipeline", "name": "Custom RAG Pipeline", "path": "pipelines/rag", "status": "active"},
                    ],
                },
            },
        }

    async def search_resources(self, query: str, source: str = None) -> list:
        """Search for resources across discovery sources."""
        data = await self.get_discovery_data()
        results = []

        sources = data.get("sources", {})
        if source and source in sources:
            sources = {source: sources[source]}

        for src_name, src_data in sources.items():
            for resource in src_data.get("resources", []):
                if query.lower() in resource.get("name", "").lower() or query.lower() in resource.get("id", "").lower():
                    resource["source"] = src_name
                    results.append(resource)

        return results

    async def scan_source(self, source: str):
        """Scan a specific source for new resources."""
        data = await self.get_discovery_data()
        normalized = source.lower()
        if normalized not in data.get("sources", {}):
            raise ValueError(f"Unknown discovery source: {source}")

        self._source_scan_times[normalized] = datetime.now().isoformat()

    async def resource_action(self, resource_id: str, action: str):
        """Perform an action on a discovered resource (download, install, etc.)."""
        normalized_action = action.lower()
        valid_actions = {
            "download": "downloaded",
            "install": "installed",
            "activate": "active",
            "quarantine": "quarantined",
            "release": "available",
        }
        if normalized_action not in valid_actions:
            raise ValueError(f"Unsupported resource action: {action}")

        resources = await self.search_resources(resource_id)
        resource = next((item for item in resources if item.get("id") == resource_id), None)
        if resource is None:
            raise ValueError(f"Unknown resource: {resource_id}")

        self._discovery_overrides[resource_id] = {
            "status": valid_actions[normalized_action],
            "last_action": normalized_action,
            "updated_at": datetime.now().isoformat(),
        }

    # ============================================================
    # RAG Operations
    # ============================================================

    async def get_rag_data(self) -> dict:
        """Get RAG system status including documents and indices."""
        if not self._initialized:
            return self._apply_rag_overrides(self._get_mock_rag_data())

        try:
            # Try to get real RAG data
            memory_stats = await self._intel.memory.get_stats()
            return self._apply_rag_overrides({
                "documents": [],
                "indices": [],
                "stats": memory_stats,
            })
        except Exception as e:
            logger.warning("Error getting RAG data: %s", e)

        return self._apply_rag_overrides(self._get_mock_rag_data())

    def _get_mock_rag_data(self) -> dict:
        """Return mock RAG data for UI development."""
        from datetime import datetime, timedelta
        now = datetime.now()

        return {
            "documents": [
                {"id": "doc-1", "name": "Architecture Overview.md", "type": "markdown", "chunks": 45, "status": "indexed", "indexed_at": (now - timedelta(days=2)).isoformat()},
                {"id": "doc-2", "name": "API Reference.pdf", "type": "pdf", "chunks": 120, "status": "indexed", "indexed_at": (now - timedelta(days=1)).isoformat()},
                {"id": "doc-3", "name": "User Guide.docx", "type": "docx", "chunks": 78, "status": "indexed", "indexed_at": (now - timedelta(hours=5)).isoformat()},
                {"id": "doc-4", "name": "Meeting Notes.txt", "type": "text", "chunks": 12, "status": "processing", "indexed_at": None},
            ],
            "indices": [
                {"name": "main", "documents": 3, "chunks": 243, "embedding_model": "nomic-embed-text"},
                {"name": "code", "documents": 15, "chunks": 1250, "embedding_model": "nomic-embed-text"},
            ],
            "stats": {
                "total_documents": 4,
                "total_chunks": 255,
                "index_size_mb": 45.2,
                "avg_retrieval_time": 0.12,
            },
        }

    async def upload_documents(self, file_paths: list):
        """Upload documents to the RAG system."""
        try:
            for path in file_paths:
                file_path = Path(path)
                document_id = file_path.stem or file_path.name
                self._rag_documents[document_id] = {
                    "id": document_id,
                    "name": file_path.name,
                    "type": file_path.suffix.lstrip(".").lower() or "unknown",
                    "chunks": max(1, len(file_path.stem.split("_")) * 8),
                    "status": "indexed",
                    "indexed_at": datetime.now().isoformat(),
                    "path": str(file_path),
                }
                logger.info("Registered uploaded document: %s", path)
        except Exception as e:
            logger.warning("Error uploading documents: %s", e)

    async def delete_document(self, doc_id: str):
        """Delete a document from the RAG system."""
        try:
            self._rag_documents.pop(doc_id, None)
            logger.info("Deleted document: %s", doc_id)
        except Exception as e:
            logger.warning("Error deleting document: %s", e)

    async def reindex_document(self, doc_id: str):
        """Reindex a document in the RAG system."""
        try:
            document = self._rag_documents.setdefault(
                doc_id,
                {
                    "id": doc_id,
                    "name": doc_id,
                    "type": "unknown",
                    "chunks": 1,
                },
            )
            document["status"] = "indexed"
            document["indexed_at"] = datetime.now().isoformat()
            logger.info("Reindexed document: %s", doc_id)
        except Exception as e:
            logger.warning("Error reindexing document: %s", e)

    # ============================================================
    # Feedback Operations
    # ============================================================

    async def record_feedback(self, message_id: str, model_name: str,
                              request_id: str, is_positive: bool,
                              feedback_text: str = None):
        """Record user feedback for a chat response.

        Args:
            message_id: Local message identifier
            model_name: Name of the model that generated the response
            request_id: UUID of the original request
            is_positive: True for thumbs up, False for thumbs down
            feedback_text: Optional text feedback
        """
        if not self._initialized:
            return

        feedback_tracker = getattr(self._intel, '_feedback_tracker', None)
        if not feedback_tracker:
            logger.info(
                "Feedback recorded locally: %s - %s",
                model_name,
                "positive" if is_positive else "negative",
            )
            return

        try:
            from uuid import UUID, uuid4

            # Parse or generate request_id
            try:
                req_uuid = UUID(request_id) if request_id else uuid4()
            except ValueError:
                req_uuid = uuid4()

            # Convert positive/negative to 0-1 score
            feedback_score = 1.0 if is_positive else 0.0

            # Record to feedback tracker
            feedback_tracker.record_feedback(
                request_id=req_uuid,
                model_name=model_name,
                feedback_score=feedback_score,
                feedback_text=feedback_text,
                query_type="chat",
                metadata={"message_id": message_id}
            )

            logger.info("Feedback recorded: %s - score=%s", model_name, feedback_score)
        except Exception as e:
            logger.warning("Error recording feedback: %s", e)

    async def test_rag_query(self, query: str, options: dict = None) -> dict:
        """Test a RAG query and return results."""
        if not self._initialized:
            return self._get_mock_rag_query_results(query)

        try:
            results = await self._intel.memory.search(query, n_results=options.get("limit", 5) if options else 5)
            return {
                "query": query,
                "results": [{"content": r.content[:500], "score": r.score, "source": r.metadata.get("source", "unknown")} for r in results],
                "time_ms": 120,  # Would be actual timing
            }
        except Exception as e:
            logger.warning("Error testing RAG query: %s", e)
            return self._get_mock_rag_query_results(query)

    def _get_mock_rag_query_results(self, query: str) -> dict:
        """Return mock RAG query results for UI development."""
        return {
            "query": query,
            "results": [
                {"content": f"This is a relevant passage about {query}. It contains information that matches the query terms and provides useful context.", "score": 0.92, "source": "Architecture Overview.md"},
                {"content": f"Another matching section discussing {query} with additional details and examples that might be helpful.", "score": 0.85, "source": "API Reference.pdf"},
                {"content": f"A third result mentioning {query} in a different context, providing alternative perspectives.", "score": 0.78, "source": "User Guide.docx"},
            ],
            "time_ms": 145,
        }

    def _apply_discovery_overrides(self, data: dict) -> dict:
        """Overlay local source scans and resource actions onto discovery data."""
        sources = data.get("sources", {})
        for source_name, source_data in sources.items():
            if source_name in self._source_scan_times:
                source_data["status"] = "rescanned"
                source_data["last_scanned_at"] = self._source_scan_times[source_name]
            for resource in source_data.get("resources", []):
                override = self._discovery_overrides.get(resource.get("id"))
                if override:
                    resource.update(override)
        return data

    def _apply_rag_overrides(self, data: dict) -> dict:
        """Overlay local document mutations onto RAG payloads."""
        documents = {doc["id"]: dict(doc) for doc in data.get("documents", [])}
        for doc_id, document in self._rag_documents.items():
            documents[doc_id] = dict(document)

        data["documents"] = list(documents.values())
        stats = data.setdefault("stats", {})
        stats["total_documents"] = len(data["documents"])
        stats["total_chunks"] = sum(int(doc.get("chunks", 0)) for doc in data["documents"])
        return data
