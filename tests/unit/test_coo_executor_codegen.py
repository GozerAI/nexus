"""Regression tests for COO executor code-agent routing."""

import asyncio
import types
from dataclasses import dataclass, field
from unittest.mock import patch

from nexus.coo.executor import AutonomousExecutor


@dataclass
class _Item:
    title: str = ""
    description: str = ""
    requirements: list = field(default_factory=list)
    constraints: list = field(default_factory=list)


def test_execute_code_agent_prefers_shared_platform_codegen():
    class StubIntel:
        def __init__(self):
            self.calls = []

        async def generate_code(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "code": "def ship():\n    return True\n",
                "quality_score": 0.92,
                "confidence": 0.9,
                "generation_path": "cog_eng_codegen",
            }

    async def run_execute():
        intel = StubIntel()
        executor = AutonomousExecutor(intelligence=intel)
        item = _Item(
            title="Build deployment checker",
            description="Implement readiness validation",
        )

        result = await executor._execute_code_agent(
            item,
            {
                "language": "python",
                "requirements": ["Validate readiness"],
                "constraints": ["No external services"],
                "max_iterations": 4,
            },
        )

        assert result["code_generated"] is True
        assert "return True" in result["code"]
        assert result["confidence"] == 0.9
        assert intel.calls == [
            {
                "description": "Build deployment checker\n\nImplement readiness validation",
                "language": "python",
                "requirements": ["Validate readiness"],
                "constraints": ["No external services"],
                "max_iterations": 4,
                "target_quality": None,
                "context": {"executor": "coo_executor"},
            }
        ]

    asyncio.run(run_execute())


def test_execute_code_agent_uses_direct_codegen_request_when_platform_missing():
    class StubGenerator:
        def __init__(self):
            self.calls = []

        async def generate(self, request, **kwargs):
            self.calls.append((request, kwargs))
            return types.SimpleNamespace(
                code="def fallback():\n    return 'ok'\n",
                quality_score=0.81,
                confidence=0.79,
                iteration=2,
            )

    stub_generator = StubGenerator()

    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)
        item = _Item(
            title="Build fallback parser",
            description="Implement a parser",
            requirements=["Parse text"],
            constraints=["No pandas"],
        )

        fake_module = types.SimpleNamespace(
            SelfImprovingCodeGenerator=lambda: stub_generator,
            CodeGenerationRequest=lambda **kwargs: types.SimpleNamespace(**kwargs),
        )

        with patch.dict(
            "sys.modules",
            {"nexus.cog_eng.capabilities.self_improving_codegen": fake_module},
        ):
            result = await executor._execute_code_agent(
                item,
                {"language": "python", "target_quality": 0.85},
            )

        request, kwargs = stub_generator.calls[0]
        assert request.description == "Build fallback parser\n\nImplement a parser"
        assert request.requirements == ["Parse text"]
        assert request.constraints == ["No pandas"]
        assert kwargs == {"max_iterations": 3, "target_quality": 0.85}
        assert result["code_generated"] is True
        assert result["quality_score"] == 0.81
        assert result["generation_path"] == "cog_eng_codegen"

    asyncio.run(run_execute())
