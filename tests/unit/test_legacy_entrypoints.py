"""Regression tests for legacy Nexus entrypoints."""

from unittest.mock import patch

from nexus.api import cli as legacy_cli
from nexus.cli import chat as chat_cli


def test_legacy_cli_translates_query_flags_to_unified_cli():
    translated = legacy_cli._translate_legacy_args(
        ["--verbose", "--query", "hello", "--model", "anthropic-sonnet", "--json"]
    )

    assert translated == [
        "--verbose",
        "query",
        "hello",
        "--model",
        "anthropic-sonnet",
        "--json",
    ]


def test_legacy_cli_delegates_to_maintained_cli():
    with patch("nexus.api.cli.unified_cli.main") as mock_main:
        exit_code = legacy_cli.main(
            ["--verbose", "--query", "hello", "--model", "anthropic-sonnet", "--json"]
        )

    assert exit_code == 0
    mock_main.assert_called_once_with(
        args=[
            "--verbose",
            "query",
            "hello",
            "--model",
            "anthropic-sonnet",
            "--json",
        ],
        prog_name="nexus-cli",
        standalone_mode=False,
    )


def test_legacy_cli_returns_error_for_missing_query_argument(capsys):
    exit_code = legacy_cli.main(["--query"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "requires an argument" in captured.err


def test_chat_main_lists_model_presets(capsys):
    with patch("nexus.cli.chat.list_presets", return_value=["anthropic-sonnet", "ollama-qwen3-30b"]):
        exit_code = chat_cli.main(["--list-models"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Available model presets" in captured.out
    assert "anthropic-sonnet" in captured.out
    assert "ollama-qwen3-30b" in captured.out


def test_chat_status_uses_platform_readiness(capsys):
    class StubPlatform:
        async def initialize_query_path(self):
            return {"llm": True, "query_backend": True}

        async def get_query_backend_status(self, model=None):
            assert model == "anthropic-sonnet"
            return {
                "direct_llm_backend": True,
                "preferred_query_backend": True,
                "fallback_query_backend": True,
                "query_backend": True,
            }

    with patch("nexus.cli.chat.NexusPlatform", return_value=StubPlatform()):
        exit_code = chat_cli.main(["--status", "--model", "anthropic-sonnet"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Preferred preset: anthropic-sonnet" in captured.out
    assert "Overall query path: OK" in captured.out


def test_chat_loop_queries_through_nexus_platform(capsys):
    class StubLlm:
        def get_stats(self):
            return {
                "requests": 1,
                "tokens_generated": 5,
                "cache_hit_rate": 0.0,
                "fallback_rate": 0.0,
                "cost_usd": 0.0,
            }

        def clear_cache(self):
            return None

    class StubPlatform:
        def __init__(self):
            self._llm = StubLlm()

        async def initialize_query_path(self):
            return {"llm": True, "query_backend": True}

        async def get_query_backend_status(self, model=None):
            return {
                "direct_llm_backend": True,
                "preferred_query_backend": True,
                "fallback_query_backend": True,
                "query_backend": True,
            }

        async def query(self, prompt, **kwargs):
            assert "User: hello" in prompt
            assert kwargs["model"] == "ollama-qwen3-30b"
            return {
                "content": "hi there",
                "actual_preset": "ollama-qwen3-30b",
                "backend_mode": "preferred",
                "tokens_used": 5,
                "duration_seconds": 0.2,
            }

    with patch("nexus.cli.chat.NexusPlatform", return_value=StubPlatform()):
        with patch("builtins.input", side_effect=["hello", "/exit"]):
            exit_code = chat_cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Nexus: hi there" in captured.out
    assert "mode=preferred" in captured.out
