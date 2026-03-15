"""Tests for agent lifecycle, registry, and orchestration."""

import asyncio
import os
import sys
import pytest
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.agents import AgentIntegration, AgentRegistry, AgentCapability


# ---------------------------------------------------------------------------
# Helpers — a concrete agent implementing the AgentInterface protocol
# ---------------------------------------------------------------------------

class StubAgent:
    """Concrete agent for testing."""

    def __init__(
        self,
        name: str = "stub-agent",
        capabilities: List[AgentCapability] = None,
        version: str = "1.0.0",
        execute_result: Dict[str, Any] = None,
    ):
        self._name = name
        self._capabilities = capabilities or [AgentCapability.ANALYSIS]
        self._version = version
        self._execute_result = execute_result or {
            "status": "success",
            "result": "done",
            "metadata": {},
        }
        self._health = {"status": "healthy"}

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> List[AgentCapability]:
        return self._capabilities

    @property
    def version(self) -> str:
        return self._version

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return self._execute_result

    async def validate_task(self, task: Dict[str, Any]) -> bool:
        return "task_type" in task

    async def health_check(self) -> Dict[str, Any]:
        return self._health


class FailingAgent(StubAgent):
    """Agent whose execute always raises."""

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise RuntimeError("agent failure")

    async def health_check(self) -> Dict[str, Any]:
        raise ConnectionError("unreachable")


# ---------------------------------------------------------------------------
# AgentRegistry — Registration & Lookup
# ---------------------------------------------------------------------------

class TestAgentRegistration:
    def setup_method(self):
        self.registry = AgentRegistry()

    def test_register_single_agent(self):
        agent = StubAgent(name="a1")
        self.registry.register(agent)
        assert "a1" in self.registry.list_all_agents()

    def test_register_multiple_agents(self):
        self.registry.register(StubAgent(name="a1"))
        self.registry.register(StubAgent(name="a2"))
        assert len(self.registry.list_all_agents()) == 2

    def test_get_agent_by_name(self):
        agent = StubAgent(name="lookup")
        self.registry.register(agent)
        found = self.registry.get_agent("lookup")
        assert found is agent

    def test_get_unknown_agent_returns_none(self):
        assert self.registry.get_agent("nonexistent") is None

    def test_unregister_removes_agent(self):
        agent = StubAgent(name="temp")
        self.registry.register(agent)
        self.registry.unregister("temp")
        assert self.registry.get_agent("temp") is None
        assert "temp" not in self.registry.list_all_agents()

    def test_unregister_nonexistent_is_safe(self):
        # Should not raise
        self.registry.unregister("nope")

    def test_register_overwrites_same_name(self):
        a1 = StubAgent(name="dup", version="1.0")
        a2 = StubAgent(name="dup", version="2.0")
        self.registry.register(a1)
        self.registry.register(a2)
        found = self.registry.get_agent("dup")
        assert found.version == "2.0"


# ---------------------------------------------------------------------------
# AgentRegistry — Capability Index
# ---------------------------------------------------------------------------

class TestCapabilityIndex:
    def setup_method(self):
        self.registry = AgentRegistry()

    def test_find_agents_by_capability(self):
        self.registry.register(
            StubAgent(name="analyst", capabilities=[AgentCapability.ANALYSIS])
        )
        self.registry.register(
            StubAgent(name="coder", capabilities=[AgentCapability.CODE_EXECUTION])
        )
        found = self.registry.find_agents_by_capability(AgentCapability.ANALYSIS)
        assert len(found) == 1
        assert found[0].name == "analyst"

    def test_multiple_agents_same_capability(self):
        self.registry.register(
            StubAgent(name="a1", capabilities=[AgentCapability.DATA_PROCESSING])
        )
        self.registry.register(
            StubAgent(name="a2", capabilities=[AgentCapability.DATA_PROCESSING])
        )
        found = self.registry.find_agents_by_capability(AgentCapability.DATA_PROCESSING)
        assert len(found) == 2

    def test_find_no_agents_for_capability(self):
        found = self.registry.find_agents_by_capability(AgentCapability.WEB_SCRAPING)
        assert found == []

    def test_agent_with_multiple_capabilities(self):
        multi = StubAgent(
            name="multi",
            capabilities=[AgentCapability.ANALYSIS, AgentCapability.PLANNING],
        )
        self.registry.register(multi)
        assert len(self.registry.find_agents_by_capability(AgentCapability.ANALYSIS)) == 1
        assert len(self.registry.find_agents_by_capability(AgentCapability.PLANNING)) == 1

    def test_unregister_cleans_capability_index(self):
        agent = StubAgent(
            name="cleanup",
            capabilities=[AgentCapability.VERIFICATION],
        )
        self.registry.register(agent)
        self.registry.unregister("cleanup")
        found = self.registry.find_agents_by_capability(AgentCapability.VERIFICATION)
        assert found == []


# ---------------------------------------------------------------------------
# AgentRegistry — Health Check
# ---------------------------------------------------------------------------

class TestAgentHealthCheck:
    def setup_method(self):
        self.registry = AgentRegistry()

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self):
        self.registry.register(StubAgent(name="h1"))
        self.registry.register(StubAgent(name="h2"))
        results = await self.registry.health_check_all()
        assert results["h1"]["status"] == "healthy"
        assert results["h2"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_with_failure(self):
        self.registry.register(StubAgent(name="good"))
        self.registry.register(FailingAgent(name="bad"))
        results = await self.registry.health_check_all()
        assert results["good"]["status"] == "healthy"
        assert results["bad"]["status"] == "error"
        assert "unreachable" in results["bad"]["error"]

    @pytest.mark.asyncio
    async def test_health_check_empty_registry(self):
        results = await self.registry.health_check_all()
        assert results == {}


# ---------------------------------------------------------------------------
# AgentIntegration — Lifecycle
# ---------------------------------------------------------------------------

class TestAgentIntegration:
    def setup_method(self):
        self.integration = AgentIntegration()

    @pytest.mark.asyncio
    async def test_initialize(self):
        await self.integration.initialize()
        assert self.integration._initialized is True

    @pytest.mark.asyncio
    async def test_double_initialize_is_safe(self):
        await self.integration.initialize()
        await self.integration.initialize()
        assert self.integration._initialized is True

    def test_register_and_list(self):
        agent = StubAgent(name="reg")
        self.integration.register_agent(agent)
        assert "reg" in self.integration.list_agents()

    def test_unregister_agent(self):
        agent = StubAgent(name="bye")
        self.integration.register_agent(agent)
        self.integration.unregister_agent("bye")
        assert "bye" not in self.integration.list_agents()

    @pytest.mark.asyncio
    async def test_health_check_delegates_to_registry(self):
        self.integration.register_agent(StubAgent(name="check"))
        results = await self.integration.health_check()
        assert "check" in results


# ---------------------------------------------------------------------------
# Agent Execution
# ---------------------------------------------------------------------------

class TestAgentExecution:
    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        agent = StubAgent(
            execute_result={"status": "success", "result": 42, "metadata": {}}
        )
        result = await agent.execute({"task_id": "t1", "task_type": "analysis", "parameters": {}})
        assert result["status"] == "success"
        assert result["result"] == 42

    @pytest.mark.asyncio
    async def test_execute_failure_raises(self):
        agent = FailingAgent()
        with pytest.raises(RuntimeError, match="agent failure"):
            await agent.execute({"task_id": "t1", "task_type": "test", "parameters": {}})

    @pytest.mark.asyncio
    async def test_validate_task_true(self):
        agent = StubAgent()
        assert await agent.validate_task({"task_type": "analysis"}) is True

    @pytest.mark.asyncio
    async def test_validate_task_false(self):
        agent = StubAgent()
        assert await agent.validate_task({"no_type": True}) is False


# ---------------------------------------------------------------------------
# Agent Capability Enum
# ---------------------------------------------------------------------------

class TestAgentCapabilityEnum:
    def test_all_capabilities_exist(self):
        expected = {
            "data_processing", "code_execution", "web_scraping",
            "analysis", "verification", "planning", "orchestration",
        }
        actual = {c.value for c in AgentCapability}
        assert expected == actual

    def test_capability_identity(self):
        assert AgentCapability.DATA_PROCESSING is AgentCapability.DATA_PROCESSING
        assert AgentCapability.DATA_PROCESSING != AgentCapability.ANALYSIS


# ---------------------------------------------------------------------------
# Multi-Agent Coordination
# ---------------------------------------------------------------------------

class TestMultiAgentCoordination:
    @pytest.mark.asyncio
    async def test_sequential_agent_execution(self):
        agent_a = StubAgent(
            name="a", execute_result={"status": "success", "result": "step1", "metadata": {}}
        )
        agent_b = StubAgent(
            name="b", execute_result={"status": "success", "result": "step2", "metadata": {}}
        )
        r1 = await agent_a.execute({"task_id": "1", "task_type": "a", "parameters": {}})
        r2 = await agent_b.execute({
            "task_id": "2", "task_type": "b",
            "parameters": {}, "context": {"previous": r1["result"]},
        })
        assert r1["status"] == "success"
        assert r2["status"] == "success"

    @pytest.mark.asyncio
    async def test_parallel_agent_execution(self):
        agents = [
            StubAgent(name=f"p{i}", execute_result={"status": "success", "result": i, "metadata": {}})
            for i in range(5)
        ]
        tasks = [
            a.execute({"task_id": str(i), "task_type": "parallel", "parameters": {}})
            for i, a in enumerate(agents)
        ]
        results = await asyncio.gather(*tasks)
        assert all(r["status"] == "success" for r in results)
        assert [r["result"] for r in results] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        good = StubAgent(name="good")
        bad = FailingAgent(name="bad")
        tasks = [
            good.execute({"task_id": "g", "task_type": "ok", "parameters": {}}),
            bad.execute({"task_id": "b", "task_type": "fail", "parameters": {}}),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert results[0]["status"] == "success"
        assert isinstance(results[1], RuntimeError)

    def test_registry_find_and_dispatch(self):
        registry = AgentRegistry()
        registry.register(StubAgent(name="r1", capabilities=[AgentCapability.ANALYSIS]))
        registry.register(StubAgent(name="r2", capabilities=[AgentCapability.CODE_EXECUTION]))
        analysts = registry.find_agents_by_capability(AgentCapability.ANALYSIS)
        assert len(analysts) == 1
        assert analysts[0].name == "r1"
