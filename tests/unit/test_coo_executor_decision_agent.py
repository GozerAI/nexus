"""Regression tests for COO decision-agent routing."""

import asyncio
import types

from nexus.coo.executor import AutonomousExecutor


def test_execute_decision_agent_supports_object_blocker_shape():
    class StubExecutor(AutonomousExecutor):
        async def _execute_ensemble(self, item, context):
            return {"response": item.description, "confidence": 0.8}

    async def run_execute():
        executor = StubExecutor(intelligence=None)
        blocker = types.SimpleNamespace(
            blocker_type=types.SimpleNamespace(value="dependency"),
            description="Waiting on upstream approval",
        )
        item = types.SimpleNamespace(
            blocker=blocker,
            task_title="Ship release",
        )

        result = await executor._execute_decision_agent(item, {})

        assert "Ship release" in result["response"]
        assert "dependency" in result["response"]
        assert "Waiting on upstream approval" in result["response"]

    asyncio.run(run_execute())


def test_execute_decision_agent_supports_dict_blocker_shape():
    class StubExecutor(AutonomousExecutor):
        async def _execute_ensemble(self, item, context):
            return {"response": item.description, "confidence": 0.8}

    async def run_execute():
        executor = StubExecutor(intelligence=None)
        blocker = types.SimpleNamespace(
            blocker_type=types.SimpleNamespace(value="policy"),
            description="Missing approval",
        )
        item = {
            "blocker": blocker,
            "task_title": "Approve launch",
        }

        result = await executor._execute_decision_agent(item, {})

        assert "Approve launch" in result["response"]
        assert "policy" in result["response"]
        assert "Missing approval" in result["response"]

    asyncio.run(run_execute())
