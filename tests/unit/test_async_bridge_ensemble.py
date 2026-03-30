"""Regression tests for GUI ensemble compatibility."""

import asyncio

from nexus.gui.async_bridge import IntelligenceController


def test_chat_with_ensemble_supports_query_style_backend():
    class StubDriftMonitor:
        def __init__(self):
            self.calls = []

        def record_response(self, **kwargs):
            self.calls.append(kwargs)

    class StubEnsemble:
        async def query(self, prompt, **kwargs):
            assert prompt == "Full prompt"
            assert kwargs["strategy"] == "simple_best"
            return {
                "content": "ensemble answer",
                "model_name": "ensemble-model",
                "provider": "ensemble-provider",
                "strategy_used": "simple_best",
                "confidence": 0.82,
                "total_cost": 0.04,
                "total_latency_ms": 120.0,
                "metadata": {"all_scores": {"ensemble-model": 0.82, "backup-model": 0.71}},
            }

    class DummyBridge:
        def __init__(self):
            self._intel = type("Intel", (), {"_drift_monitor": StubDriftMonitor()})()

        def _classify_task(self, message):
            assert message == "hello"
            return "conversation"

    async def run_chat():
        bridge = DummyBridge()
        result = await IntelligenceController._chat_with_ensemble(
            bridge,
            "hello",
            "Full prompt",
            StubEnsemble(),
            0.0,
        )

        assert result["content"] == "ensemble answer"
        assert result["model_info"]["model"] == "ensemble-model"
        assert result["model_info"]["models_tried"] == ["ensemble-model", "backup-model"]
        assert bridge._intel._drift_monitor.calls[0]["model_name"] == "ensemble-model"
        assert bridge._intel._drift_monitor.calls[0]["success"] is True

    asyncio.run(run_chat())


def test_chat_with_ensemble_raises_on_structured_query_error():
    class StubEnsemble:
        async def query(self, prompt, **kwargs):
            return {"error": "ensemble unavailable"}

    class DummyBridge:
        def __init__(self):
            self._intel = type("Intel", (), {"_drift_monitor": None})()

        def _classify_task(self, message):
            return "conversation"

    async def run_chat():
        bridge = DummyBridge()
        try:
            await IntelligenceController._chat_with_ensemble(
                bridge,
                "hello",
                "Full prompt",
                StubEnsemble(),
                0.0,
            )
        except RuntimeError as exc:
            assert str(exc) == "ensemble unavailable"
        else:
            raise AssertionError("Expected structured ensemble error to raise")

    asyncio.run(run_chat())
