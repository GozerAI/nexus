"""Regression tests for COO executor lazy backend initialization."""

import asyncio

from nexus.coo.executor import AutonomousExecutor


class _Item:
    def __init__(self, title="", description="", item_id="task-1"):
        self.id = item_id
        self.title = title
        self.description = description


def test_execute_content_pipeline_lazy_initializes_backend():
    class StubPipeline:
        async def queue_job(self, payload):
            assert payload["topic"] == "Ship release"
            return "job-123"

    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)

        async def ensure_pipeline():
            executor._content_pipeline = StubPipeline()

        executor._ensure_content_pipeline = ensure_pipeline
        result = await executor._execute_content_pipeline(_Item("Ship release", "Finalize launch"), {})

        assert result["job_id"] == "job-123"
        assert result["status"] == "queued"

    asyncio.run(run_execute())


def test_execute_expert_router_lazy_initializes_backend():
    class StubRouter:
        async def execute_with_experts(self, task):
            return {"decision": f"Handled {task.description}", "confidence": 0.8}

    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)

        async def ensure_router():
            executor._expert_router = StubRouter()

        executor._ensure_expert_router = ensure_router
        result = await executor._execute_expert_router(_Item("Analyze", "system health"), {})

        assert result["decision"] == "Handled Analyze system health"

    asyncio.run(run_execute())


def test_execute_expert_router_supports_dict_items():
    class StubRouter:
        async def execute_with_experts(self, task):
            return {"decision": f"Handled {task.description}", "confidence": 0.8}

    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)
        executor._expert_router = StubRouter()
        result = await executor._execute_expert_router(
            {"id": "task-9", "title": "Analyze", "description": "system health", "priority": "high"},
            {},
        )

        assert result["decision"] == "Handled Analyze system health"

    asyncio.run(run_execute())


def test_execute_ensemble_lazy_connects_intelligence_backend():
    class StubEnsemble:
        async def query(self, prompt, **kwargs):
            return {
                "content": "ensemble response",
                "model_name": "stub-model",
                "confidence": 0.75,
                "total_cost": 0.01,
                "strategy_used": "simple_best",
                "metadata": {"all_scores": {"stub-model": 0.75}},
            }

    class StubIntel:
        def __init__(self):
            self._ensemble = StubEnsemble()

    async def run_execute():
        executor = AutonomousExecutor(intelligence=StubIntel())
        result = await executor._execute_ensemble(_Item("Decide", "pick a route"), {})

        assert result["response"] == "ensemble response"
        assert result["models_used"] == ["stub-model"]

    asyncio.run(run_execute())
