"""Regression tests for COO executor ensemble compatibility."""

import asyncio
import types
from unittest.mock import patch

from nexus.coo.executor import AutonomousExecutor


class _Item:
    def __init__(self, title="", description=""):
        self.title = title
        self.description = description


def test_execute_ensemble_supports_query_style_backend():
    class StubEnsemble:
        async def query(self, prompt, **kwargs):
            assert "Task: Decision" in prompt
            assert kwargs["strategy"] == "simple_best"
            return {
                "content": "Use the fallback route",
                "models_queried": ["stub-model"],
                "confidence": 0.74,
                "total_cost": 0.02,
                "strategy_used": "simple_best",
            }

    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)
        executor._ensemble = StubEnsemble()
        item = _Item("Decision", "Choose the best plan")

        result = await executor._execute_ensemble(item, {"task_type": "analysis"})

        assert result == {
            "response": "Use the fallback route",
            "models_used": ["stub-model"],
            "confidence": 0.74,
            "cost_usd": 0.02,
            "strategy_used": "simple_best",
        }

    asyncio.run(run_execute())


def test_execute_ensemble_query_style_normalizes_models_from_metadata_scores():
    class StubEnsemble:
        async def query(self, prompt, **kwargs):
            return {
                "content": "Use the fallback route",
                "model_name": "stub-model",
                "models_queried": 2,
                "confidence": 0.74,
                "total_cost": 0.02,
                "strategy_used": "simple_best",
                "metadata": {"all_scores": {"stub-model": 0.74, "backup-model": 0.61}},
            }

    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)
        executor._ensemble = StubEnsemble()
        item = _Item("Decision", "Choose the best plan")

        result = await executor._execute_ensemble(item, {})

        assert result["models_used"] == ["stub-model", "backup-model"]

    asyncio.run(run_execute())


def test_execute_ensemble_supports_process_style_backend():
    class StubEnsemble:
        async def process(self, request):
            assert "Task: Decision" in request.query
            return types.SimpleNamespace(
                content="Use the direct route",
                model_responses=[types.SimpleNamespace(model_name="provider-model")],
                confidence=0.81,
                total_cost_usd=0.03,
            )

    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)
        executor._ensemble = StubEnsemble()
        item = _Item("Decision", "Choose the best plan")

        fake_types_module = types.SimpleNamespace(
            EnsembleRequest=lambda **kwargs: types.SimpleNamespace(**kwargs)
        )
        with patch.dict(
            "sys.modules",
            {"nexus.providers.ensemble.types": fake_types_module},
        ):
            result = await executor._execute_ensemble(item, {})

        assert result == {
            "response": "Use the direct route",
            "models_used": ["provider-model"],
            "confidence": 0.81,
            "cost_usd": 0.03,
        }

    asyncio.run(run_execute())
