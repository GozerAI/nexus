"""Regression tests for the primary Nexus query path."""

import asyncio
import os
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

from nexus.blueprints import llm_backend
from nexus.core.llm_provider import NexusLLM
from nexus.platform import NexusPlatform


def test_platform_query_returns_content_for_cli_path():
    async def run_query():
        platform = NexusPlatform()
        result = await platform.query("Hello from regression", model="ollama-qwen3-30b")
        assert isinstance(result, dict)
        assert result["content"]
        assert result["model_name"]
        assert result["strategy_used"]

    asyncio.run(run_query())


def test_platform_query_prefers_direct_llm_when_available():
    class StubLLM:
        async def generate(self, **kwargs):
            return {
                "content": "direct llm response",
                "model": "stub-model",
                "provider": "stub",
                "preset_used": "stub-preset",
                "models_tried": ["stub-preset"],
                "duration_seconds": 0.01,
                "tokens_used": 12,
                "cached": False,
            }

    async def run_query():
        platform = NexusPlatform()
        platform._initialized = True
        platform._query_initialized = True
        platform._llm = StubLLM()
        platform._ensemble = None

        with patch.object(platform, "_ensure_ensemble_component") as ensure_ensemble:
            result = await platform.query("route through llm", model="anything")

        assert result["content"] == "direct llm response"
        assert result["strategy_used"] == "direct_llm"
        assert result["preset_used"] == "stub-preset"
        assert result["preferred_preset"] == "anything"
        assert result["actual_preset"] == "stub-preset"
        assert result["backend_mode"] == "fallback"
        assert result["query_path"] == "direct_llm"
        assert result["fallback_used"] is True
        ensure_ensemble.assert_not_called()

    asyncio.run(run_query())


def test_platform_query_marks_ensemble_fallback_metadata():
    class FailingLLM:
        async def generate(self, **kwargs):
            raise RuntimeError("direct backend unavailable")

    class StubEnsemble:
        async def query(self, prompt, **kwargs):
            return {
                "content": "ensemble response",
                "model_name": "ensemble-model",
                "provider": "ensemble-provider",
                "strategy_used": "simple_best",
            }

    async def run_query():
        platform = NexusPlatform()
        platform._initialized = True
        platform._query_initialized = True
        platform._llm = FailingLLM()
        platform._ensemble = StubEnsemble()

        result = await platform.query("route through ensemble", model="preferred-model")

        assert result["content"] == "ensemble response"
        assert result["query_path"] == "ensemble"
        assert result["backend_mode"] == "ensemble"
        assert result["preferred_preset"] == "preferred-model"
        assert result["actual_preset"] == "ensemble-model"
        assert result["fallback_used"] is True

    asyncio.run(run_query())


def test_initialize_query_path_does_not_boot_full_stack():
    async def run_init():
        platform = NexusPlatform()

        with patch.object(platform, "_ensure_llm_component") as ensure_llm, patch.object(
            platform, "_ensure_ensemble_component"
        ) as ensure_ensemble, patch.object(
            platform, "_ensure_monitoring_components"
        ) as ensure_metrics, patch.object(
            platform, "_ensure_cognitive_components"
        ) as ensure_cog, patch.object(
            platform, "_ensure_expert_components"
        ) as ensure_experts, patch.object(
            platform, "_ensure_insights_components"
        ) as ensure_insights, patch.object(
            platform, "_ensure_discovery_components"
        ) as ensure_discovery, patch.object(
            platform, "_query_backend_ready", return_value=True
        ):
            status = await platform.initialize_query_path()

        ensure_llm.assert_awaited_once()
        ensure_ensemble.assert_not_called()
        ensure_metrics.assert_not_called()
        ensure_cog.assert_not_called()
        ensure_experts.assert_not_called()
        ensure_insights.assert_not_called()
        ensure_discovery.assert_not_called()
        assert status["query_backend"] is True
        assert "ensemble" not in status
        assert "observatory" not in status

    asyncio.run(run_init())


def test_initialize_codegen_path_does_not_boot_unrelated_stack():
    async def run_init():
        platform = NexusPlatform()

        with patch.object(platform, "_ensure_codegen_component") as ensure_codegen, patch.object(
            platform, "_ensure_llm_component"
        ) as ensure_llm, patch.object(
            platform, "_ensure_cognitive_components"
        ) as ensure_cog, patch.object(
            platform, "_ensure_expert_components"
        ) as ensure_experts, patch.object(
            platform, "_ensure_monitoring_components"
        ) as ensure_metrics:
            status = await platform.initialize_codegen_path()

        ensure_codegen.assert_awaited_once()
        ensure_llm.assert_not_called()
        ensure_cog.assert_not_called()
        ensure_experts.assert_not_called()
        ensure_metrics.assert_not_called()
        assert status == {"codegen": False}

    asyncio.run(run_init())


def test_get_status_reports_query_backend():
    async def run_status():
        platform = NexusPlatform()
        platform._status = {"llm": True, "ensemble": False}

        with patch.object(platform, "initialize") as initialize, patch.object(
            platform, "initialize_query_path"
        ) as initialize_query_path, patch.object(
            platform,
            "get_query_backend_status",
            return_value={
                "direct_llm_backend": False,
                "preferred_query_backend": False,
                "fallback_query_backend": True,
                "query_backend": True,
            },
        ):
            status = await platform.get_status()

        initialize.assert_not_called()
        initialize_query_path.assert_awaited_once()
        assert status["direct_llm_backend"] is False
        assert status["preferred_query_backend"] is False
        assert status["fallback_query_backend"] is True
        assert status["query_backend"] is True

    asyncio.run(run_status())


def test_get_status_full_boots_broader_platform():
    async def run_status():
        platform = NexusPlatform()
        platform._status = {"llm": True}

        with patch.object(platform, "initialize") as initialize, patch.object(
            platform, "initialize_query_path"
        ) as initialize_query_path, patch.object(
            platform,
            "get_query_backend_status",
            return_value={
                "direct_llm_backend": True,
                "preferred_query_backend": True,
                "fallback_query_backend": True,
                "query_backend": True,
            },
        ):
            status = await platform.get_status(full=True)

        initialize.assert_awaited_once()
        initialize_query_path.assert_not_called()
        assert status["direct_llm_backend"] is True
        assert status["preferred_query_backend"] is True
        assert status["fallback_query_backend"] is True
        assert status["query_backend"] is True

    asyncio.run(run_status())


def test_get_query_backend_status_reports_preferred_ready():
    class StubLLM:
        async def get_backend_status(self, **kwargs):
            return {
                "preferred_available": True,
                "fallback_available": True,
                "usable": True,
            }

    async def run_status():
        platform = NexusPlatform()
        platform._query_initialized = True
        platform._llm = StubLLM()
        platform._ensemble = object()

        status = await platform.get_query_backend_status(model="preferred-model")

        assert status["direct_llm_backend"] is True
        assert status["preferred_query_backend"] is True
        assert status["fallback_query_backend"] is True
        assert status["query_backend"] is True

    asyncio.run(run_status())


def test_get_query_backend_status_reports_fallback_only():
    class StubLLM:
        async def get_backend_status(self, **kwargs):
            return {
                "preferred_available": False,
                "fallback_available": False,
                "usable": False,
            }

    async def run_status():
        platform = NexusPlatform()
        platform._query_initialized = True
        platform._llm = StubLLM()
        platform._ensemble = object()

        status = await platform.get_query_backend_status(model="preferred-model")

        assert status["direct_llm_backend"] is False
        assert status["preferred_query_backend"] is False
        assert status["fallback_query_backend"] is True
        assert status["query_backend"] is True

    asyncio.run(run_status())


def test_check_specific_backends_uses_short_ttl_cache():
    class StubBackend:
        def __init__(self):
            self.calls = 0

        async def check_available(self):
            self.calls += 1
            return True

    async def run_status():
        llm = NexusLLM(readiness_cache_ttl_seconds=60)
        backend = StubBackend()

        with patch.object(llm, "_get_backend", return_value=backend):
            first = await llm.check_specific_backends(["stub-model"])
            second = await llm.check_specific_backends(["stub-model"])

        assert first == {"stub-model": True}
        assert second == {"stub-model": True}
        assert backend.calls == 1

        llm.invalidate_backend_status_cache()

        with patch.object(llm, "_get_backend", return_value=backend):
            third = await llm.check_specific_backends(["stub-model"])

        assert third == {"stub-model": True}
        assert backend.calls == 2

    asyncio.run(run_status())


def test_llm_backends_bootstrap_environment_without_full_platform_init():
    fake_env_path = Path("F:/Projects/nexus/.env")

    def fake_exists(path):
        return Path(path) == fake_env_path

    def fake_load_dotenv(path, override=False):
        assert Path(path) == fake_env_path
        assert override is False
        os.environ["ANTHROPIC_API_KEY"] = "dotenv-anthropic"
        return True

    with patch.dict(os.environ, {}, clear=True), patch.object(
        llm_backend, "_ENV_BOOTSTRAPPED", False
    ), patch("pathlib.Path.exists", autospec=True, side_effect=fake_exists), patch(
        "dotenv.load_dotenv", side_effect=fake_load_dotenv
    ) as load_dotenv:
        first = llm_backend.AnthropicBackend()
        second = llm_backend.AnthropicBackend()

    assert first.api_key == "dotenv-anthropic"
    assert second.api_key == "dotenv-anthropic"
    load_dotenv.assert_called_once()


def test_importing_llm_provider_does_not_boot_ensemble_package():
    script = (
        "import importlib, json, sys; "
        "importlib.import_module('nexus.core.llm_provider'); "
        "print(json.dumps('nexus.core.ensemble_core' in sys.modules))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.stdout.strip() == "false"


def test_ensure_ensemble_component_reuses_module_singleton():
    async def run_init():
        platform = NexusPlatform()
        sentinel = object()

        with patch.dict(
            sys.modules,
            {"nexus.core.strategic_ensemble": types.SimpleNamespace(strategic_ensemble=sentinel)},
        ):
            await platform._ensure_ensemble_component()

        assert platform._ensemble is sentinel
        assert platform._status["ensemble"] is True

    asyncio.run(run_init())


def test_discover_trends_uses_scan_when_categories_are_provided():
    class StubInsights:
        def __init__(self):
            self.categories = None

        async def scan(self, categories=None, **kwargs):
            self.categories = categories
            return {"topics_analyzed": 1, "top_recommendations": []}

    async def run_discover():
        platform = NexusPlatform()
        platform._initialized = True
        platform._insights = StubInsights()

        result = await platform.discover_trends(categories=["business", "technology"])

        assert result["topics_analyzed"] == 1
        assert [category.value for category in platform._insights.categories] == [
            "business",
            "technology",
        ]

    asyncio.run(run_discover())


def test_discover_trends_returns_error_for_invalid_categories():
    class StubInsights:
        async def discover(self, **kwargs):
            raise AssertionError("discover should not be called for invalid categories")

    async def run_discover():
        platform = NexusPlatform()
        platform._initialized = True
        platform._insights = StubInsights()

        result = await platform.discover_trends(categories=["business", "invalid-category"])

        assert result["error"].startswith("Unsupported trend categories:")
        assert "invalid-category" in result["error"]

    asyncio.run(run_discover())


def test_get_metrics_supports_observatory_dashboard_collector():
    platform = NexusPlatform()

    class StubMetrics:
        def get_dashboard_data(self):
            return {"counters": {"queries.total": 4}}

    platform._metrics = StubMetrics()

    assert platform.get_metrics() == {"counters": {"queries.total": 4}}


def test_generate_code_uses_targeted_codegen_init_and_serializes_result():
    class StubQuality:
        value = "good"

    class StubResult:
        request_id = "req-1"
        code = "print('hi')"
        language = "python"
        quality_score = 0.91
        quality_level = StubQuality()
        test_results = {"passed": True}
        improvements_applied = ["documentation"]
        confidence = 0.88
        generation_time = 0.25
        iteration = 2
        metadata = {"source": "stub"}

    class StubCodeGenerator:
        def __init__(self):
            self.calls = []

        async def generate(self, request, **kwargs):
            self.calls.append((request, kwargs))
            return StubResult()

    async def run_generate():
        platform = NexusPlatform()
        stub_generator = StubCodeGenerator()

        with patch.object(platform, "initialize", side_effect=AssertionError("full initialize should not run")), patch.object(
            platform, "initialize_codegen_path"
        ) as initialize_codegen_path:
            async def side_effect():
                platform._code_generator = stub_generator
                platform._status["codegen"] = True
                return {"codegen": True}

            initialize_codegen_path.side_effect = side_effect
            result = await platform.generate_code(
                description="Build a parser",
                language="python",
                requirements=["Parse CSV"],
                constraints=["No pandas"],
                max_iterations=3,
                target_quality=0.9,
            )

        initialize_codegen_path.assert_awaited_once()
        request, kwargs = stub_generator.calls[0]
        assert request.description == "Build a parser"
        assert request.language == "python"
        assert request.requirements == ["Parse CSV"]
        assert request.constraints == ["No pandas"]
        assert kwargs == {"max_iterations": 3, "target_quality": 0.9}
        assert result["code"] == "print('hi')"
        assert result["quality_level"] == "good"
        assert result["generation_path"] == "cog_eng_codegen"

    asyncio.run(run_generate())


def test_search_github_uses_targeted_discovery_init():
    class StubGitHub:
        async def search_repositories(self, query, limit=30):
            assert query == "nexus"
            assert limit == 5
            return [{"full_name": "example/nexus"}]

    async def run_search():
        platform = NexusPlatform()
        platform._github_integration = None

        with patch.object(platform, "initialize", side_effect=AssertionError("full initialize should not run")), patch.object(
            platform, "_ensure_github_components"
        ) as ensure_github:
            ensure_github.side_effect = lambda: setattr(platform, "_github_integration", StubGitHub())
            results = await platform.search_github("nexus", limit=5)

        assert results == [{"full_name": "example/nexus"}]
        ensure_github.assert_awaited_once()

    asyncio.run(run_search())


def test_list_ollama_models_uses_targeted_discovery_init():
    class StubOllama:
        async def list_models(self):
            return [{"name": "qwen2.5:14b"}]

    async def run_list():
        platform = NexusPlatform()
        platform._ollama_integration = None

        with patch.object(platform, "initialize", side_effect=AssertionError("full initialize should not run")), patch.object(
            platform, "_ensure_ollama_components"
        ) as ensure_ollama:
            ensure_ollama.side_effect = lambda: setattr(platform, "_ollama_integration", StubOllama())
            results = await platform.list_ollama_models()

        assert results == [{"name": "qwen2.5:14b"}]
        ensure_ollama.assert_awaited_once()

    asyncio.run(run_list())


def test_discover_resources_uses_targeted_discovery_init():
    class StubDiscovery:
        async def discover_all(self):
            return {"github": 2}

    async def run_discover():
        platform = NexusPlatform()
        platform._resource_discovery = None

        with patch.object(platform, "initialize", side_effect=AssertionError("full initialize should not run")), patch.object(
            platform, "initialize_discovery_path"
        ) as initialize_discovery_path:
            initialize_discovery_path.side_effect = lambda: setattr(platform, "_resource_discovery", StubDiscovery())
            results = await platform.discover_resources()

        assert results == {"github": 2}
        initialize_discovery_path.assert_awaited_once()

    asyncio.run(run_discover())
