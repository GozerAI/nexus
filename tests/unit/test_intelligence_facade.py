"""Regression tests for shared-platform delegation in NexusIntelligence."""

import asyncio

from nexus.intelligence import NexusIntelligence


def test_intelligence_delegates_query_research_codegen_and_experts_to_shared_platform():
    class StubPlatform:
        def __init__(self):
            self.calls = []

        async def query(self, prompt, **kwargs):
            self.calls.append(("query", prompt, kwargs))
            return {"content": "query ok"}

        async def research(self, topic, **kwargs):
            self.calls.append(("research", topic, kwargs))
            return {"findings": []}

        async def generate_code(self, description, **kwargs):
            self.calls.append(("generate_code", description, kwargs))
            return {"code": "print('ok')"}

        async def get_expert_opinion(self, task):
            self.calls.append(("get_expert_opinion", task, {}))
            return {"consensus": {"decision": "Proceed"}}

    async def run_checks():
        intelligence = NexusIntelligence()
        platform = StubPlatform()
        intelligence._shared_platform = platform

        assert await intelligence.query("hello", model="stub") == {"content": "query ok"}
        assert await intelligence.research("market mapping", depth="deep") == {"findings": []}
        assert await intelligence.generate_code("build parser", language="python") == {"code": "print('ok')"}
        assert await intelligence.get_expert_opinion("next step") == {"consensus": {"decision": "Proceed"}}

        assert platform.calls == [
            ("query", "hello", {"model": "stub"}),
            ("research", "market mapping", {"depth": "deep"}),
            ("generate_code", "build parser", {"language": "python"}),
            ("get_expert_opinion", "next step", {}),
        ]

    asyncio.run(run_checks())


def test_intelligence_lazily_creates_shared_platform_once():
    async def run_checks():
        intelligence = NexusIntelligence()

        first = await intelligence._get_shared_platform()
        second = await intelligence._get_shared_platform()

        assert first is second

    asyncio.run(run_checks())
