"""Regression tests for honest expert failure handling."""

import asyncio

from nexus.experts.base import Task, TaskType
from nexus.experts.personas.research import ResearchExpert
from nexus.experts.personas.writer import WriterExpert


def test_writer_execute_fails_on_structured_query_error_payload():
    class StubPlatform:
        async def query(self, *args, **kwargs):
            return {"error": "query backend unavailable"}

    async def run_execute():
        expert = WriterExpert(platform=StubPlatform())
        task = Task(id="task-1", description="Write release notes", task_type=TaskType.WRITING)

        result = await expert.execute(task)

        assert result.success is False
        assert result.errors == ["query backend unavailable"]

    asyncio.run(run_execute())


def test_research_execute_fails_on_structured_research_error_payload():
    class StubPlatform:
        async def research(self, *args, **kwargs):
            return {"error": "research backend unavailable"}

    async def run_execute():
        expert = ResearchExpert(platform=StubPlatform())
        task = Task(id="task-2", description="Investigate market demand", task_type=TaskType.RESEARCH)

        result = await expert.execute(task)

        assert result.success is False
        assert result.errors == ["research backend unavailable"]

    asyncio.run(run_execute())
