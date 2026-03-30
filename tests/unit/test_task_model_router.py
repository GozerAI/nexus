"""Tests for the task-type model router and outcome loop service endpoints."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.core.task_model_router import (
    ModelSelection,
    TaskModelRouter,
    _DEFAULT_ROUTING_TABLE,
)


# ---------------------------------------------------------------------------
# TaskModelRouter — selection logic
# ---------------------------------------------------------------------------


class TestRouterSelection:
    def test_select_decide_returns_qwen(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("decide")
        assert sel.model_id == "qwen/qwen3-30b-a3b:free"
        assert sel.task_step == "decide"

    def test_select_define_success_returns_llama(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("define_success")
        assert sel.model_id == "meta-llama/llama-4-maverick:free"

    def test_select_evaluate_returns_deepseek(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("evaluate")
        assert sel.model_id == "deepseek/deepseek-r1-0528:free"

    def test_select_diagnose_returns_gemma(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("diagnose")
        assert sel.model_id == "google/gemma-3-27b-it:free"

    def test_select_prescribe_returns_mistral(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("prescribe")
        assert sel.model_id == "mistralai/mistral-small-3.1-24b-instruct:free"

    def test_select_challenge_returns_llama_scout(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("challenge")
        assert sel.model_id == "meta-llama/llama-4-scout:free"

    def test_all_steps_have_different_primary_families(self):
        """No two adjacent steps should share a model family."""
        router = TaskModelRouter()
        steps = ["decide", "define_success", "evaluate", "diagnose", "prescribe", "challenge"]
        families = []
        for step in steps:
            sel = router.select_model_for_task(step)
            family = sel.model_id.split("/")[0]
            families.append(family)
        # At least 4 distinct families
        assert len(set(families)) >= 4

    def test_unknown_step_falls_back_to_decide(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("nonexistent_step")
        assert sel.model_id == "qwen/qwen3-30b-a3b:free"

    def test_selection_returns_model_selection_type(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("decide")
        assert isinstance(sel, ModelSelection)
        assert isinstance(sel.fallback_chain, list)
        assert isinstance(sel.reasoning, str)

    def test_selection_includes_fallbacks(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("decide")
        assert len(sel.fallback_chain) > 0

    def test_context_is_accepted(self):
        """Context parameter should be accepted without error."""
        router = TaskModelRouter()
        sel = router.select_model_for_task("decide", context={"executive": "CFO"})
        assert sel.model_id == "qwen/qwen3-30b-a3b:free"


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_no_failures_circuit_closed(self):
        router = TaskModelRouter()
        sel = router.select_model_for_task("decide")
        assert sel.model_id == "qwen/qwen3-30b-a3b:free"

    def test_circuit_trips_after_threshold_failures(self):
        router = TaskModelRouter()
        model = "qwen/qwen3-30b-a3b:free"
        for _ in range(3):
            router.record_failure(model)
        sel = router.select_model_for_task("decide")
        # Should use fallback
        assert sel.model_id != model
        assert sel.model_id == "qwen/qwen3-8b:free"

    def test_circuit_resets_after_timeout(self):
        router = TaskModelRouter()
        router.RESET_SECONDS = 0.01  # Very short for testing
        model = "qwen/qwen3-30b-a3b:free"
        for _ in range(3):
            router.record_failure(model)
        time.sleep(0.02)
        sel = router.select_model_for_task("decide")
        assert sel.model_id == model

    def test_success_resets_circuit(self):
        router = TaskModelRouter()
        model = "qwen/qwen3-30b-a3b:free"
        router.record_failure(model)
        router.record_failure(model)
        router.record_success(model)
        # Should still use primary (failures reset)
        router.record_failure(model)
        sel = router.select_model_for_task("decide")
        assert sel.model_id == model  # Only 1 failure, threshold is 3

    def test_all_models_tripped_returns_primary_anyway(self):
        router = TaskModelRouter()
        primary = "qwen/qwen3-30b-a3b:free"
        fallback = "qwen/qwen3-8b:free"
        for _ in range(3):
            router.record_failure(primary)
            router.record_failure(fallback)
        sel = router.select_model_for_task("decide")
        # Returns primary as last resort
        assert sel.model_id == primary

    def test_two_failures_does_not_trip(self):
        router = TaskModelRouter()
        model = "qwen/qwen3-30b-a3b:free"
        router.record_failure(model)
        router.record_failure(model)
        sel = router.select_model_for_task("decide")
        assert sel.model_id == model


# ---------------------------------------------------------------------------
# Override and config
# ---------------------------------------------------------------------------


class TestOverrideAndConfig:
    def test_get_routing_table(self):
        router = TaskModelRouter()
        table = router.get_routing_table()
        assert "decide" in table
        assert "evaluate" in table
        assert "challenge" in table
        assert table["decide"]["primary"] == "qwen/qwen3-30b-a3b:free"

    def test_override_route(self):
        router = TaskModelRouter()
        router.override_route("decide", "custom/model:free")
        sel = router.select_model_for_task("decide")
        assert sel.model_id == "custom/model:free"

    def test_override_new_step(self):
        router = TaskModelRouter()
        router.override_route("custom_step", "custom/model:free")
        sel = router.select_model_for_task("custom_step")
        assert sel.model_id == "custom/model:free"

    def test_custom_routing_table(self):
        custom = {
            "decide": {"primary": "custom/a:free", "fallbacks": [], "description": "test"},
        }
        router = TaskModelRouter(routing_table=custom)
        sel = router.select_model_for_task("decide")
        assert sel.model_id == "custom/a:free"

    def test_default_routing_table_has_all_steps(self):
        assert "decide" in _DEFAULT_ROUTING_TABLE
        assert "define_success" in _DEFAULT_ROUTING_TABLE
        assert "evaluate" in _DEFAULT_ROUTING_TABLE
        assert "diagnose" in _DEFAULT_ROUTING_TABLE
        assert "prescribe" in _DEFAULT_ROUTING_TABLE
        assert "challenge" in _DEFAULT_ROUTING_TABLE


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_initial_stats(self):
        router = TaskModelRouter()
        stats = router.get_stats()
        assert stats["selections"] == 0
        assert stats["fallbacks_used"] == 0

    def test_stats_count_selections(self):
        router = TaskModelRouter()
        router.select_model_for_task("decide")
        router.select_model_for_task("evaluate")
        stats = router.get_stats()
        assert stats["selections"] == 2
        assert stats["step_counts"]["decide"] == 1
        assert stats["step_counts"]["evaluate"] == 1

    def test_stats_count_fallbacks(self):
        router = TaskModelRouter()
        for _ in range(3):
            router.record_failure("qwen/qwen3-30b-a3b:free")
        router.select_model_for_task("decide")
        stats = router.get_stats()
        assert stats["fallbacks_used"] == 1

    def test_stats_track_circuit_breakers(self):
        router = TaskModelRouter()
        for _ in range(3):
            router.record_failure("qwen/qwen3-30b-a3b:free")
        stats = router.get_stats()
        assert "qwen/qwen3-30b-a3b:free" in stats["circuit_breakers"]
        assert stats["circuit_breakers"]["qwen/qwen3-30b-a3b:free"]["is_open"]

    def test_stats_list_registered_steps(self):
        router = TaskModelRouter()
        stats = router.get_stats()
        assert set(stats["registered_steps"]) == {
            "decide", "define_success", "evaluate",
            "diagnose", "prescribe", "challenge",
        }


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------


class TestModelProfiles:
    def test_gemma_3_profile_exists(self):
        from nexus.providers.adapters.model_profiles import get_model_profile
        profile = get_model_profile("google/gemma-3-27b-it")
        assert profile is not None
        assert profile.family == "gemma-3"

    def test_mistral_small_profile_exists(self):
        from nexus.providers.adapters.model_profiles import get_model_profile
        profile = get_model_profile("mistralai/mistral-small-3.1-24b")
        assert profile is not None
        assert profile.family == "mistral-small"

    def test_existing_profiles_still_work(self):
        from nexus.providers.adapters.model_profiles import get_model_profile
        assert get_model_profile("qwen/qwen3-30b") is not None
        assert get_model_profile("deepseek/deepseek-r1-0528") is not None
        assert get_model_profile("meta-llama/llama-4-maverick") is not None

    def test_all_routing_table_models_have_profile_families(self):
        from nexus.providers.adapters.model_profiles import get_model_profile
        for step, route in _DEFAULT_ROUTING_TABLE.items():
            model_id = route["primary"].replace(":free", "")
            profile = get_model_profile(model_id)
            assert profile is not None, f"No profile for {model_id} (step: {step})"


# ---------------------------------------------------------------------------
# Service handler endpoints (mocked)
# ---------------------------------------------------------------------------


class TestServiceHandlerEndpoints:
    def _make_handler(self):
        from nexus.coo.service_handler import NexusServiceHandler
        handler = NexusServiceHandler.__new__(NexusServiceHandler)
        handler._requests_handled = 0
        handler._errors = 0
        handler._cache_hits = 0
        handler._cache = MagicMock()
        handler._cache.size = 0
        handler._listening = False
        # Ensure fresh router per test (no singleton leakage)
        if hasattr(handler, "_task_model_router"):
            del handler._task_model_router
        return handler

    @pytest.mark.asyncio
    async def test_select_model_for_task_returns_selection(self):
        handler = self._make_handler()
        result = await handler._select_model_for_task({"task_step": "evaluate"})
        assert result["model_id"] == "deepseek/deepseek-r1-0528:free"
        assert result["task_step"] == "evaluate"

    @pytest.mark.asyncio
    async def test_select_model_default_step(self):
        handler = self._make_handler()
        result = await handler._select_model_for_task({})
        assert result["task_step"] == "decide"

    @pytest.mark.asyncio
    async def test_outcome_loop_query_missing_prompt(self):
        handler = self._make_handler()
        result = await handler._outcome_loop_query({"task_step": "evaluate"})
        assert result["error"] == "prompt is required"
        assert result["response"] is None

    @pytest.mark.asyncio
    async def test_outcome_loop_query_calls_openrouter(self):
        handler = self._make_handler()
        with patch.object(handler, "_call_openrouter", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "This looks like a successful outcome."
            result = await handler._outcome_loop_query({
                "task_step": "evaluate",
                "prompt": "Evaluate this outcome...",
                "system_prompt": "You are a judge.",
            })
        assert result["response"] == "This looks like a successful outcome."
        assert result["model_used"] == "deepseek/deepseek-r1-0528:free"
        assert result["task_step"] == "evaluate"
        assert result["fallback_used"] is False

    @pytest.mark.asyncio
    async def test_outcome_loop_query_tries_fallback_on_failure(self):
        handler = self._make_handler()
        call_count = 0

        async def mock_call(model_id, prompt, system_prompt="", max_tokens=4000):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Model unavailable")
            return "Fallback response"

        with patch.object(handler, "_call_openrouter", side_effect=mock_call):
            result = await handler._outcome_loop_query({
                "task_step": "evaluate",
                "prompt": "test prompt",
            })
        assert result["response"] == "Fallback response"
        assert result["fallback_used"] is True
        assert result["model_used"] == "deepseek/deepseek-v3:free"

    @pytest.mark.asyncio
    async def test_outcome_loop_query_all_models_fail(self):
        handler = self._make_handler()
        with patch.object(handler, "_call_openrouter", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = RuntimeError("All down")
            result = await handler._outcome_loop_query({
                "task_step": "evaluate",
                "prompt": "test",
            })
        assert result["response"] is None
        assert "All models failed" in result["error"]

    @pytest.mark.asyncio
    async def test_handler_dispatch_table_includes_outcome_endpoints(self):
        handler = self._make_handler()
        handler._platform = MagicMock()
        # Test that _get_handler recognizes the new endpoints
        assert handler._get_handler("select_model_for_task") is not None
        assert handler._get_handler("outcome_loop_query") is not None

    @pytest.mark.asyncio
    async def test_stats_include_router_after_use(self):
        handler = self._make_handler()
        await handler._select_model_for_task({"task_step": "decide"})
        stats = handler.get_stats()
        assert "task_model_router" in stats
        assert stats["task_model_router"]["selections"] == 1
