"""Compatibility wrapper for the maintained Nexus CLI."""

from __future__ import annotations

from collections.abc import Sequence
import sys

import click

from nexus.cli.main import cli as unified_cli


def _translate_legacy_args(argv: Sequence[str]) -> list[str]:
    """Map the historical ``nexus.api.cli`` flags to the unified CLI."""
    args = list(argv)
    translated: list[str] = []

    if "--verbose" in args:
        translated.append("--verbose")
        args = [arg for arg in args if arg != "--verbose"]

    if "--status" in args:
        args = [arg for arg in args if arg != "--status"]
        return translated + ["status", *args]

    model: str | None = None
    if "--model" in args:
        index = args.index("--model")
        if index + 1 >= len(args):
            raise click.ClickException("Option '--model' requires an argument.")
        model = args[index + 1]
        del args[index:index + 2]

    query: str | None = None
    for index, arg in enumerate(args):
        if arg.startswith("--query="):
            query = arg.split("=", 1)[1]
            del args[index]
            break
        if arg == "--query":
            if index + 1 >= len(args):
                raise click.ClickException("Option '--query' requires an argument.")
            query = args[index + 1]
            del args[index:index + 2]
            break

    if query is not None:
        translated.extend(["query", query])
        if model:
            translated.extend(["--model", model])
        translated.extend(args)
        return translated

    if model:
        translated.extend(["--model", model])
    translated.extend(args)
    return translated


def main(argv: Sequence[str] | None = None) -> int:
    """Run the maintained CLI through the legacy entrypoint."""
    try:
        command_args = _translate_legacy_args(argv if argv is not None else sys.argv[1:])
        unified_cli.main(args=command_args, prog_name="nexus-cli", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return int(exc.exit_code or 0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
