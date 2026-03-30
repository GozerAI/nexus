"""Regression tests for CLI logging behavior."""

import json
import logging
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from nexus.cli.main import (
    _configure_cli_logging,
    _ordered_status_items,
    _query_content_or_raise,
    _query_payload,
    _status_payload,
    _status_exit_code,
    cli,
)


def test_configure_cli_logging_defaults_to_quiet_output():
    root_logger = logging.getLogger()
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    asyncio_logger = logging.getLogger("asyncio")
    original = {
        "root": root_logger.level,
        "httpx": httpx_logger.level,
        "httpcore": httpcore_logger.level,
        "asyncio": asyncio_logger.level,
    }

    try:
        _configure_cli_logging(verbose=False)

        assert root_logger.level == logging.WARNING
        assert httpx_logger.level == logging.WARNING
        assert httpcore_logger.level == logging.WARNING
        assert asyncio_logger.level == logging.WARNING
    finally:
        root_logger.setLevel(original["root"])
        httpx_logger.setLevel(original["httpx"])
        httpcore_logger.setLevel(original["httpcore"])
        asyncio_logger.setLevel(original["asyncio"])


def test_configure_cli_logging_supports_verbose_mode():
    root_logger = logging.getLogger()
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    asyncio_logger = logging.getLogger("asyncio")
    original = {
        "root": root_logger.level,
        "httpx": httpx_logger.level,
        "httpcore": httpcore_logger.level,
        "asyncio": asyncio_logger.level,
    }

    try:
        _configure_cli_logging(verbose=True)

        assert root_logger.level == logging.INFO
        assert httpx_logger.level == logging.INFO
        assert httpcore_logger.level == logging.INFO
        assert asyncio_logger.level == logging.INFO
    finally:
        root_logger.setLevel(original["root"])
        httpx_logger.setLevel(original["httpx"])
        httpcore_logger.setLevel(original["httpcore"])
        asyncio_logger.setLevel(original["asyncio"])


def test_ordered_status_items_prioritizes_query_path_signals():
    status = {
        "discovery": True,
        "query_backend": True,
        "ensemble": True,
        "preferred_query_backend": False,
        "llm": True,
        "fallback_query_backend": True,
        "direct_llm_backend": True,
        "observatory": True,
        "custom_component": False,
    }

    ordered = _ordered_status_items(status)

    assert ordered == [
        ("llm", True),
        ("direct_llm_backend", True),
        ("preferred_query_backend", False),
        ("fallback_query_backend", True),
        ("query_backend", True),
        ("observatory", True),
        ("ensemble", True),
        ("discovery", True),
        ("custom_component", False),
    ]


def test_status_exit_code_fails_only_when_query_backend_is_down():
    assert _status_exit_code({"query_backend": True, "preferred_query_backend": False}) == 0
    assert _status_exit_code({"query_backend": False}) == 1
    assert _status_exit_code({"llm": True}) == 0


def test_status_payload_is_machine_readable():
    payload = _status_payload({"llm": True, "query_backend": True}, full=True)

    assert payload == {
        "ok": True,
        "exit_code": 0,
        "full": True,
        "components": {"llm": True, "query_backend": True},
    }


def test_query_content_or_raise_returns_content():
    assert _query_content_or_raise({"content": "hello", "error": False}) == "hello"


def test_query_content_or_raise_raises_on_error_payload():
    with pytest.raises(Exception) as exc_info:
        _query_content_or_raise({"content": "backend unavailable", "error": True})

    assert "backend unavailable" in str(exc_info.value)


def test_query_content_or_raise_raises_when_content_is_missing():
    with pytest.raises(Exception) as exc_info:
        _query_content_or_raise({"provider": "stub"})

    assert "without content" in str(exc_info.value)


def test_query_payload_returns_full_result_on_success():
    payload = _query_payload({"content": "hello", "backend_mode": "preferred"})
    assert payload == {"content": "hello", "backend_mode": "preferred"}


def test_status_command_supports_json_output():
    class StubPlatform:
        async def get_status(self, full=False):
            assert full is True
            return {
                "llm": True,
                "query_backend": True,
                "preferred_query_backend": False,
            }

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["status", "--full", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["full"] is True
    assert payload["components"]["llm"] is True
    assert payload["components"]["preferred_query_backend"] is False


def test_status_command_fails_when_query_backend_is_down():
    class StubPlatform:
        async def get_status(self, full=False):
            return {
                "llm": False,
                "query_backend": False,
            }

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["status", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["exit_code"] == 1


def test_query_command_returns_non_zero_on_error_payload():
    class StubPlatform:
        async def query(self, prompt, model):
            return {"content": "backend unavailable", "error": True}

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["query", "hello"])

    assert result.exit_code != 0
    assert "backend unavailable" in result.output


def test_query_command_supports_json_output():
    class StubPlatform:
        async def query(self, prompt, model):
            return {
                "content": "hello",
                "query_path": "direct_llm",
                "backend_mode": "fallback",
                "preferred_preset": model,
                "actual_preset": "anthropic-sonnet",
                "fallback_used": True,
            }

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["query", "hello", "--json", "--model", "ollama-qwen3-30b"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["content"] == "hello"
    assert payload["query_path"] == "direct_llm"
    assert payload["backend_mode"] == "fallback"
    assert payload["preferred_preset"] == "ollama-qwen3-30b"


def test_research_command_supports_json_output():
    class StubPlatform:
        async def research(self, topic):
            assert topic == "nexus"
            return {
                "topic": topic,
                "status": "completed",
                "confidence": 0.91,
                "recommendations": ["Ship the fix"],
            }

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["research", "nexus", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "research"
    assert payload["data"]["topic"] == "nexus"
    assert payload["data"]["recommendations"] == ["Ship the fix"]


def test_trends_command_supports_json_output_and_category_forwarding():
    class StubPlatform:
        async def discover_trends(self, categories=None):
            assert categories == ["business"]
            return {
                "topics_analyzed": 2,
                "top_recommendations": [
                    {
                        "topic": "AI workflow automation",
                        "category": "business",
                        "overall_score": 0.88,
                    }
                ],
            }

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["trends", "--categories", "business", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "trends"
    assert payload["data"]["topics_analyzed"] == 2
    assert payload["data"]["top_recommendations"][0]["topic"] == "AI workflow automation"


def test_trends_command_emits_json_error_payload_on_failure():
    class StubPlatform:
        async def discover_trends(self, categories=None):
            return {"error": "Insights not initialized"}

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["trends", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["command"] == "trends"
    assert payload["error"] == "Insights not initialized"


def test_metrics_command_supports_json_output():
    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def initialize_monitoring_path(self):
            return {"observatory": True}

        def get_metrics(self):
            return {"queries.total": 7, "latency.avg_ms": 123.4}

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["metrics", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "metrics"
    assert payload["data"]["queries.total"] == 7


def test_discover_command_supports_json_output_with_stats():
    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def initialize_discovery_path(self):
            return {"discovery": True}

        async def discover_resources(self):
            return {"github": 3, "huggingface": 2}

        def get_discovery_stats(self):
            return {"total_resources": 42, "recent_discoveries": 5}

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["discover", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "discover"
    assert payload["data"]["results"]["github"] == 3
    assert payload["data"]["stats"]["total_resources"] == 42


def test_discover_stats_command_supports_json_output():
    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def initialize_discovery_path(self):
            return {"discovery": True}

        def get_discovery_stats(self):
            return {
                "total_resources": 42,
                "by_type": {"model": 20},
                "by_source": {"github": 10},
                "recent_discoveries": 5,
            }

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["discover-stats", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "discover-stats"
    assert payload["data"]["by_source"]["github"] == 10


def test_search_models_command_serializes_object_results_in_json():
    class StubModel:
        def __init__(self):
            self.name = "qwen3"
            self.provider = "ollama"
            self.quality_score = 0.93

    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def search_models(self, query, capabilities=None, max_price=None, min_context=None):
            assert query == "reasoning"
            return [StubModel()]

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["search-models", "reasoning", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "search-models"
    assert payload["data"][0]["name"] == "qwen3"
    assert payload["data"][0]["provider"] == "ollama"


def test_pypi_info_command_emits_json_error_payload_when_package_missing():
    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def get_pypi_package(self, package):
            assert package == "missing-package"
            return None

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["pypi-info", "missing-package", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["command"] == "pypi-info"
    assert payload["error"] == "Package 'missing-package' not found"


def test_pypi_info_command_returns_concise_json_summary_by_default():
    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def get_pypi_package(self, package):
            assert package == "click"
            return {
                "info": {
                    "name": "click",
                    "version": "8.3.1",
                    "summary": "Composable CLI toolkit",
                    "project_url": "https://pypi.org/project/click/",
                    "requires_python": ">=3.10",
                    "requires_dist": ['colorama; platform_system == "Windows"'],
                    "project_urls": {"Source": "https://github.com/pallets/click/"},
                },
                "releases": {"8.3.1": [{"filename": "click-8.3.1.tar.gz"}]},
                "vulnerabilities": [],
            }

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["pypi-info", "click", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["package"] == "click"
    assert payload["full"] is False
    assert payload["data"]["name"] == "click"
    assert payload["data"]["version"] == "8.3.1"
    assert "releases" not in payload["data"]


def test_ls_command_supports_json_output():
    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def initialize_local_machine_path(self):
            return {"local_machine": True}

        def list_directory(self, path, pattern, recursive):
            assert path == "."
            assert pattern == "*"
            assert recursive is False
            return {
                "path": path,
                "count": 1,
                "entries": [{"name": "README.md", "is_dir": False, "size": 128}],
            }

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["ls", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert payload["command"] == "ls"
    assert payload["data"]["entries"][0]["name"] == "README.md"


def test_processes_command_emits_json_error_payload_from_collection_result():
    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def initialize_local_machine_path(self):
            return {"local_machine": True}

        def get_running_processes(self, limit):
            assert limit == 15
            return [{"error": "Process enumeration unavailable"}]

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["processes", "--json"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False
    assert payload["command"] == "processes"
    assert payload["error"] == "Process enumeration unavailable"


def test_ollama_list_command_does_not_require_full_initialize():
    class StubPlatform:
        async def initialize(self):
            raise AssertionError("full initialize should not be used")

        async def list_ollama_models(self):
            return [{"name": "qwen2.5:14b", "size": 1024**3}]

    runner = CliRunner()
    with patch("nexus.cli.main.NexusPlatform", return_value=StubPlatform()):
        result = runner.invoke(cli, ["ollama-list", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["data"][0]["name"] == "qwen2.5:14b"
