"""Regression tests for engineer expert execution."""

import asyncio

from nexus.experts.base import Task, TaskType
from nexus.experts.personas.engineer import EngineerExpert


def test_engineer_execute_uses_platform_codegen_contract():
    class StubPlatform:
        def __init__(self):
            self.calls = []

        async def generate_code(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "code": "def solution():\n    return 42\n",
                "quality_score": 0.93,
                "iteration": 2,
                "generation_path": "cog_eng_codegen",
            }

    async def run_execute():
        platform = StubPlatform()
        expert = EngineerExpert(platform=platform)
        task = Task(
            id="task-1",
            description="Build a small parser",
            task_type=TaskType.CODING,
            context={"language": "python", "requirements": ["Parse CSV input"]},
            constraints=["No external dependencies"],
        )

        result = await expert.execute(task)

        assert result.success is True
        assert "return 42" in result.output
        assert result.confidence == 0.93
        assert result.metadata["quality_score"] == 0.93
        assert result.metadata["iterations"] == 2
        assert platform.calls == [
            {
                "description": "Build a small parser",
                "language": "python",
                "requirements": ["Parse CSV input"],
                "constraints": ["No external dependencies"],
            }
        ]

    asyncio.run(run_execute())


def test_engineer_execute_returns_failure_when_codegen_raises():
    class StubPlatform:
        async def generate_code(self, **kwargs):
            raise RuntimeError("codegen unavailable")

    async def run_execute():
        expert = EngineerExpert(platform=StubPlatform())
        task = Task(
            id="task-2",
            description="Implement a queue",
            task_type=TaskType.CODING,
        )

        result = await expert.execute(task)

        assert result.success is False
        assert result.confidence == 0.0
        assert result.errors == ["codegen unavailable"]

    asyncio.run(run_execute())
