"""Interactive chat entrypoint built on the maintained Nexus platform."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from typing import Any, Sequence

from nexus.blueprints.llm_backend import list_presets
from nexus.platform import NexusPlatform


SYSTEM_PROMPT = "You are Nexus, a helpful AI assistant. Be concise and practical."


def _configure_console_encoding() -> None:
    """Use UTF-8 on Windows terminals when available."""
    try:
        import sys

        if sys.platform == "win32":
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            if hasattr(sys.stdin, "reconfigure"):
                sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        return


def print_banner() -> None:
    """Print the chat banner."""
    print(
        """
+--------------------------------------------------------------+
| NEXUS CHAT                                                   |
| Shared-infrastructure chat over the maintained query path.   |
+--------------------------------------------------------------+
"""
    )


def print_help() -> None:
    """Print supported chat commands."""
    print(
        """
Commands:
  /help        Show this help
  /status      Check query backend readiness
  /stats       Show session routing statistics
  /model X     Switch preferred model preset
  /models      List available model presets
  /clear       Clear conversation history and query cache
  /exit        Exit chat

Notes:
  - Model routing may still fall back if the preferred preset is unavailable.
  - Use '/status' to see whether you are on the preferred path or fallback path.
"""
    )


def _build_context(history: list[dict[str, str]]) -> str:
    """Return recent conversation context for the next prompt."""
    if not history:
        return ""

    recent_history = history[-5:]
    lines = []
    for item in recent_history:
        lines.append(f"User: {item['user']}")
        lines.append(f"Assistant: {item['assistant']}")
    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"


def _render_response_footer(result: dict[str, Any]) -> str:
    """Return a compact routing footer for a chat response."""
    actual_preset = result.get("actual_preset") or result.get("preset_used") or "unknown"
    backend_mode = result.get("backend_mode", "unknown")
    tokens = result.get("tokens_used", 0)
    duration = result.get("duration_seconds", 0.0)
    return f"[{actual_preset} | mode={backend_mode} | {tokens} tokens | {duration:.1f}s]"


def _available_presets() -> list[str]:
    """Return known model presets."""
    try:
        return sorted(list_presets())
    except Exception:
        return ["ollama-qwen3-30b", "ollama-qwen3-8b", "anthropic-sonnet"]


def show_model_presets() -> None:
    """Print available model presets."""
    print("\nAvailable model presets:\n")
    for preset in _available_presets():
        print(f"  {preset}")
    print()


async def show_status(platform: NexusPlatform, current_model: str) -> None:
    """Print current query-backend readiness."""
    readiness = await platform.get_query_backend_status(model=current_model)

    print("\nQuery Backend Status:\n")
    print(f"  Preferred preset: {current_model}")
    print(f"  Direct LLM route: {'OK' if readiness['direct_llm_backend'] else 'FAIL'}")
    print(
        f"  Preferred backend: {'OK' if readiness['preferred_query_backend'] else 'FAIL'}"
    )
    print(
        f"  Fallback backend: {'OK' if readiness['fallback_query_backend'] else 'FAIL'}"
    )
    print(f"  Overall query path: {'OK' if readiness['query_backend'] else 'FAIL'}")
    print()


def show_stats(platform: NexusPlatform) -> None:
    """Print routing statistics for this chat session."""
    llm = getattr(platform, "_llm", None)
    if llm is None:
        print("\nSession Statistics:\n")
        print("  No routed query activity yet.\n")
        return

    stats = llm.get_stats()
    print("\nSession Statistics:\n")
    print(f"  Requests: {stats['requests']}")
    print(f"  Tokens generated: {stats['tokens_generated']:,}")
    print(f"  Cache hit rate: {stats['cache_hit_rate'] * 100:.1f}%")
    print(f"  Fallback rate: {stats['fallback_rate'] * 100:.1f}%")
    print(f"  Estimated cost: ${stats['cost_usd']:.4f}")
    print()


async def chat_loop(model_preset: str = "ollama-qwen3-30b") -> int:
    """Run the interactive chat loop."""
    _configure_console_encoding()

    platform = NexusPlatform()
    await platform.initialize_query_path()

    current_model = model_preset
    conversation_history: list[dict[str, str]] = []

    print_banner()
    await show_status(platform, current_model)
    print(f"Using: {current_model}")
    print("Type /help for commands, /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye.")
            return 0
        except EOFError:
            print("\nGoodbye.")
            return 0

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split()
            command = parts[0].lower()
            args = parts[1:]

            if command in {"/exit", "/quit"}:
                print("\nGoodbye.")
                return 0
            if command == "/help":
                print_help()
                continue
            if command == "/status":
                await show_status(platform, current_model)
                continue
            if command == "/stats":
                show_stats(platform)
                continue
            if command == "/clear":
                conversation_history.clear()
                llm = getattr(platform, "_llm", None)
                if llm is not None:
                    llm.clear_cache()
                print("Conversation history cleared.\n")
                continue
            if command == "/model":
                if args:
                    current_model = args[0]
                    print(f"Preferred model preset: {current_model}\n")
                else:
                    print(f"Current preferred model preset: {current_model}\n")
                continue
            if command == "/models":
                show_model_presets()
                continue
            if command == "/think":
                print("Thinking mode is controlled by the selected preset. Use /model to switch presets.\n")
                continue

            print(f"Unknown command: {command}\n")
            continue

        prompt = f"{_build_context(conversation_history)}User: {user_input}"
        result = await platform.query(
            prompt,
            model=current_model,
            task_type="conversation",
            system_prompt=SYSTEM_PROMPT,
        )

        if result.get("error"):
            error_message = result.get("content") or result.get("error") or "Query failed."
            print(f"\nError: {error_message}")
            print("Run /status to inspect backend readiness.\n")
            continue

        content = result.get("content", "").strip()
        if not content:
            print("\nError: Query completed without content.\n")
            continue

        print(f"\nNexus: {content}")
        print(f"  {_render_response_footer(result)}\n")

        conversation_history.append(
            {
                "user": user_input,
                "assistant": content,
                "timestamp": datetime.now().isoformat(),
            }
        )


async def run(argv: Sequence[str] | None = None) -> int:
    """Async runner for the packaged chat command."""
    parser = argparse.ArgumentParser(description="Nexus interactive chat")
    parser.add_argument(
        "--model",
        "-m",
        default="ollama-qwen3-30b",
        help="Preferred model preset for chat",
    )
    parser.add_argument(
        "--status",
        "-s",
        action="store_true",
        help="Check query-backend readiness and exit",
    )
    parser.add_argument(
        "--list-models",
        "-l",
        action="store_true",
        help="List available model presets and exit",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_models:
        show_model_presets()
        return 0

    if args.status:
        platform = NexusPlatform()
        await platform.initialize_query_path()
        await show_status(platform, args.model)
        return 0

    return await chat_loop(model_preset=args.model)


def main(argv: Sequence[str] | None = None) -> int:
    """Synchronous console-script entrypoint."""
    return asyncio.run(run(argv))


if __name__ == "__main__":
    raise SystemExit(main())
