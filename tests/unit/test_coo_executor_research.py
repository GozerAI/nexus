"""Regression tests for COO executor research routing."""

import asyncio

from nexus.coo.executor import AutonomousExecutor


class _Item:
    def __init__(self, title="", description=""):
        self.title = title
        self.description = description


def test_execute_research_agent_prefers_shared_platform_research():
    class StubIntel:
        def __init__(self):
            self.calls = []

        async def research(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "findings": [{"content": "Market expanding"}],
                "sources": ["simulated"],
                "confidence": 0.72,
            }

    async def run_execute():
        intel = StubIntel()
        executor = AutonomousExecutor(intelligence=intel)
        item = _Item("Market expansion", "Assess AI workflow demand")

        result = await executor._execute_research_agent(
            item,
            {"depth": "deep", "max_iterations": 4},
        )

        assert result["research_complete"] is True
        assert result["confidence"] == 0.72
        assert intel.calls == [
            {
                "topic": "Market expansion: Assess AI workflow demand",
                "depth": "deep",
                "max_iterations": 4,
            }
        ]

    asyncio.run(run_execute())


def test_execute_research_agent_uses_direct_agent_with_correct_topic_signature():
    class StubResearchAgent:
        def __init__(self):
            self.calls = []

        async def research(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "findings": [{"content": "Validated"}],
                "sources": ["simulated"],
                "confidence": 0.68,
            }

    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)
        executor._research_agent = StubResearchAgent()
        item = _Item("Deployment review", "Check release readiness")

        result = await executor._execute_research_agent(item, {"max_iterations": 2})

        assert result["research_complete"] is True
        assert executor._research_agent.calls == [
            {
                "topic": "Deployment review: Check release readiness",
                "depth": "moderate",
                "max_iterations": 2,
            }
        ]

    asyncio.run(run_execute())


def test_execute_research_agent_raises_on_structured_error_payload():
    class StubIntel:
        async def research(self, **kwargs):
            return {"error": "research backend unavailable"}

    async def run_execute():
        executor = AutonomousExecutor(intelligence=StubIntel())
        item = _Item("Market expansion", "Assess AI workflow demand")

        try:
            await executor._execute_research_agent(item, {})
        except RuntimeError as exc:
            assert str(exc) == "research backend unavailable"
        else:
            raise AssertionError("Expected structured error payload to raise")

    asyncio.run(run_execute())
