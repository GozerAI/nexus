"""Regression tests for cog_eng LLM client initialization."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from nexus.cog_eng.llm.client import LLMClient


def test_init_openai_uses_environment_without_name_error():
    captured = {}

    class StubAsyncOpenAI:
        def __init__(self, api_key, organization=None):
            captured["api_key"] = api_key
            captured["organization"] = organization

    client = LLMClient.__new__(LLMClient)
    client.config = SimpleNamespace(
        llm=SimpleNamespace(openai_api_key="test-key")
    )
    client.model = "gpt-4o"
    client.client = None

    with patch.dict("sys.modules", {"openai": SimpleNamespace(AsyncOpenAI=StubAsyncOpenAI)}):
        with patch("nexus.cog_eng.llm.client.os.getenv", return_value="org-123"):
            client._init_openai()

    assert isinstance(client.client, StubAsyncOpenAI)
    assert captured == {
        "api_key": "test-key",
        "organization": "org-123",
    }


def test_complete_switches_to_simulation_after_provider_failure():
    class FailingCompletions:
        def __init__(self):
            self.calls = 0

        async def create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("quota exceeded")

    completions = FailingCompletions()
    client = LLMClient.__new__(LLMClient)
    client.provider = "openai"
    client.model = "gpt-4o"
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client._simulate_only = False
    client._simulate_only_reason = None

    first = asyncio.run(client.complete("hello"))
    second = asyncio.run(client.complete("hello again"))

    assert first["model"] == "simulated"
    assert second["model"] == "simulated"
    assert client._simulate_only is True
    assert client._simulate_only_reason == "quota exceeded"
    assert completions.calls == 1
