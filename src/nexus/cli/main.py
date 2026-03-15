"""
Nexus Unified CLI - Single entry point for all capabilities.
"""

import asyncio
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
import json
import logging
from typing import Any, Awaitable, Callable

import click

from nexus.platform import NexusPlatform


_STATUS_DISPLAY_ORDER = [
    "llm",
    "direct_llm_backend",
    "preferred_query_backend",
    "fallback_query_backend",
    "query_backend",
    "observatory",
    "ensemble",
    "cog_eng",
    "experts",
    "insights",
    "discovery",
]


def _configure_cli_logging(verbose: bool = False) -> None:
    """Keep CLI output readable by default while allowing explicit verbose mode."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)

    # Third-party HTTP clients are especially noisy at INFO.
    for logger_name in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(logger_name).setLevel(level if verbose else logging.WARNING)


def _ordered_status_items(status: dict[str, bool]) -> list[tuple[str, bool]]:
    """Return status items in a stable, operator-friendly order."""
    ordered = [(key, status[key]) for key in _STATUS_DISPLAY_ORDER if key in status]
    remaining = sorted((key, value) for key, value in status.items() if key not in _STATUS_DISPLAY_ORDER)
    return ordered + remaining


def _status_exit_code(status: dict[str, bool]) -> int:
    """Return a process exit code suitable for automation."""
    if "query_backend" in status and not status["query_backend"]:
        return 1
    return 0


def _status_payload(status: dict[str, bool], full: bool = False) -> dict:
    """Return a machine-readable status payload."""
    return {
        "ok": _status_exit_code(status) == 0,
        "exit_code": _status_exit_code(status),
        "full": full,
        "components": dict(_ordered_status_items(status)),
    }


def _serialize_cli_value(value: Any) -> Any:
    """Convert command results into JSON-safe structures."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _serialize_cli_value(value.to_dict())
    if is_dataclass(value):
        return _serialize_cli_value(asdict(value))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        public_attrs = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        if public_attrs:
            return _serialize_cli_value(public_attrs)
    if isinstance(value, dict):
        return {str(key): _serialize_cli_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_cli_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _error_message(result: Any, default_message: str = "Command failed.") -> str:
    """Extract a user-facing error message from a structured result."""
    if isinstance(result, dict):
        for key in ("error", "message", "content"):
            value = result.get(key)
            if value:
                return str(value)
    return default_message


def _require_command_success(result: Any, default_message: str = "Command failed.") -> Any:
    """Raise a CLI-friendly error when a command result advertises failure."""
    if isinstance(result, dict) and result.get("error"):
        raise click.ClickException(_error_message(result, default_message))
    return result


def _require_collection_success(result: Any, default_message: str = "Command failed.") -> Any:
    """Raise when collection-shaped results encode failure in the first item."""
    if isinstance(result, list) and result:
        first_item = result[0]
        if isinstance(first_item, dict) and first_item.get("error"):
            raise click.ClickException(_error_message(first_item, default_message))
    return result


def _success_payload(command: str, data: Any, **extra: Any) -> dict[str, Any]:
    """Return a standard machine-readable success payload."""
    payload: dict[str, Any] = {
        "ok": True,
        "command": command,
        "data": _serialize_cli_value(data),
    }
    payload.update(
        {
            key: _serialize_cli_value(value)
            for key, value in extra.items()
            if value is not None
        }
    )
    return payload


def _error_payload(command: str, message: str, exit_code: int = 1, **extra: Any) -> dict[str, Any]:
    """Return a standard machine-readable failure payload."""
    payload: dict[str, Any] = {
        "ok": False,
        "command": command,
        "error": str(message),
        "exit_code": exit_code,
    }
    payload.update(
        {
            key: _serialize_cli_value(value)
            for key, value in extra.items()
            if value is not None
        }
    )
    return payload


def _emit_json(payload: dict[str, Any]) -> None:
    """Emit formatted JSON output."""
    click.echo(json.dumps(payload, indent=2))


def _format_scalar(value: Any) -> str:
    """Format numbers consistently for human-readable output."""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _brief_text(item: Any) -> str:
    """Collapse rich result items into a single readable line."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("action", "recommendation", "title", "summary", "topic", "description"):
            value = item.get(key)
            if value:
                return str(value)
        return json.dumps(item, ensure_ascii=True)
    return str(item)


async def _run_async_command(
    command_name: str,
    as_json: bool,
    operation: Callable[[], Awaitable[Any]],
    render_text: Callable[[Any], None],
    payload_builder: Callable[[Any], dict[str, Any]] | None = None,
) -> None:
    """Run a command with consistent human/JSON output and failure semantics."""
    try:
        result = await operation()
        serialized = _serialize_cli_value(result)
        if as_json:
            payload = payload_builder(serialized) if payload_builder else _success_payload(command_name, serialized)
            _emit_json(payload)
        else:
            render_text(serialized)
    except click.exceptions.Exit:
        raise
    except click.ClickException as exc:
        if as_json:
            _emit_json(_error_payload(command_name, exc.format_message(), exit_code=exc.exit_code))
            raise click.exceptions.Exit(exc.exit_code)
        raise
    except Exception as exc:
        if as_json:
            _emit_json(_error_payload(command_name, str(exc)))
            raise click.exceptions.Exit(1)
        raise click.ClickException(str(exc))


def _query_content_or_raise(result: dict) -> str:
    """Extract query content or raise a CLI-friendly error."""
    if result.get("error"):
        message = result.get("content") or result.get("message") or "Query failed."
        raise click.ClickException(str(message))

    content = result.get("content")
    if not content:
        raise click.ClickException("Query completed without content.")

    return str(content)


def _query_payload(result: dict) -> dict:
    """Return the full query payload or raise on structured failure."""
    _query_content_or_raise(result)
    return result


def _render_research_result(result: Any) -> None:
    """Render a research report for terminal users."""
    if not isinstance(result, dict):
        click.echo(str(result))
        return

    click.echo("\nResearch Report\n")
    if result.get("topic"):
        click.echo(f"  Topic: {result['topic']}")
    if result.get("status"):
        click.echo(f"  Status: {result['status']}")
    if "confidence" in result:
        click.echo(f"  Confidence: {_format_scalar(result['confidence'])}")
    if "duration_seconds" in result:
        click.echo(f"  Duration: {_format_scalar(result['duration_seconds'])}s")

    findings = result.get("findings") or []
    recommendations = result.get("recommendations") or []
    click.echo(f"  Findings: {len(findings)}")
    click.echo(f"  Recommendations: {len(recommendations)}")

    synthesis = result.get("synthesis")
    if isinstance(synthesis, dict):
        summary = synthesis.get("executive_summary") or synthesis.get("summary")
        if summary:
            click.echo(f"\nSummary:\n  {summary}")

    if recommendations:
        click.echo("\nTop Recommendations:")
        for recommendation in recommendations[:5]:
            click.echo(f"  - {_brief_text(recommendation)}")


def _render_trends_result(result: Any) -> None:
    """Render top trend recommendations."""
    if not isinstance(result, dict):
        click.echo(str(result))
        return

    click.echo("\nTrending Topics\n")
    if "topics_analyzed" in result:
        click.echo(f"  Topics analyzed: {result['topics_analyzed']}")

    recommendations = result.get("top_recommendations") or result.get("recommendations") or []
    if not recommendations:
        click.echo("  No trends found")
        return

    click.echo()
    for trend in recommendations[:10]:
        if isinstance(trend, dict):
            topic = trend.get("topic", "unknown")
            details = []
            if trend.get("category"):
                details.append(str(trend["category"]))
            score = trend.get("overall_score", trend.get("score"))
            if score is not None:
                details.append(f"score: {_format_scalar(score)}")
            if trend.get("lifecycle"):
                details.append(str(trend["lifecycle"]))
            suffix = f" ({', '.join(details)})" if details else ""
            click.echo(f"  - {topic}{suffix}")
        else:
            click.echo(f"  - {trend}")


def _render_metrics_result(result: Any) -> None:
    """Render platform metrics."""
    click.echo("\nPlatform Metrics\n")
    if isinstance(result, dict) and result:
        for key, value in result.items():
            click.echo(f"  {key}: {_format_scalar(value)}")
        return
    click.echo("  No metrics collected yet")


def _render_discovery_result(result: Any) -> None:
    """Render discovery run results and summary stats."""
    if not isinstance(result, dict):
        click.echo(str(result))
        return

    click.echo("\nDiscovery Results\n")
    discoveries = result.get("results", {})
    if discoveries:
        for source, count in discoveries.items():
            click.echo(f"  {source}: {count} new resources")
    else:
        click.echo("  No discovery results available")

    stats = result.get("stats", {})
    click.echo(f"\nTotal resources: {stats.get('total_resources', 0)}")


def _render_discovery_stats(result: Any) -> None:
    """Render discovery statistics."""
    if not isinstance(result, dict):
        click.echo(str(result))
        return

    click.echo("\nDiscovery Statistics\n")
    click.echo(f"Total resources: {result.get('total_resources', 0)}")

    by_type = result.get("by_type") or {}
    if by_type:
        click.echo("\nBy Type:")
        for resource_type, count in by_type.items():
            click.echo(f"  {resource_type}: {count}")

    by_source = result.get("by_source") or {}
    if by_source:
        click.echo("\nBy Source:")
        for source, count in by_source.items():
            click.echo(f"  {source}: {count}")

    click.echo(f"\nDiscoveries today: {result.get('recent_discoveries', 0)}")


def _summarize_pypi_package(pkg: dict[str, Any]) -> dict[str, Any]:
    """Return a concise PyPI package summary for CLI/API consumers."""
    info = pkg.get("info", {})
    project_urls = info.get("project_urls") or {}
    return {
        "name": info.get("name"),
        "version": info.get("version"),
        "summary": info.get("summary"),
        "author": info.get("author") or info.get("maintainer_email"),
        "license": info.get("license") or info.get("license_expression"),
        "requires_python": info.get("requires_python"),
        "project_url": info.get("project_url") or info.get("package_url"),
        "homepage": info.get("home_page") or project_urls.get("Homepage") or project_urls.get("Documentation"),
        "documentation_url": project_urls.get("Documentation"),
        "source_url": project_urls.get("Source"),
        "latest_release_url": info.get("release_url"),
        "classifiers": info.get("classifiers", [])[:10],
        "dependencies": info.get("requires_dist") or [],
        "vulnerabilities": pkg.get("vulnerabilities", []),
    }


@click.group()
@click.option("--verbose", is_flag=True, help="Show backend initialization and routing logs.")
@click.pass_context
def cli(ctx, verbose):
    """Nexus Unified AI Platform"""
    _configure_cli_logging(verbose=verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj['platform'] = NexusPlatform()


@cli.command()
@click.option('--full', is_flag=True, help='Initialize and report the broader platform surface.')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def status(ctx, full, as_json):
    """Check platform status."""
    async def _status():
        platform = ctx.obj['platform']
        status = await platform.get_status(full=full)

        if as_json:
            click.echo(json.dumps(_status_payload(status, full=full), indent=2))
        else:
            click.echo("\nNexus Platform Status\n")
            for component, ok in _ordered_status_items(status):
                icon = "[OK]" if ok else "[FAIL]"
                click.echo(f"  {icon} {component}")
            click.echo()
        raise click.exceptions.Exit(_status_exit_code(status))
    
    asyncio.run(_status())


@cli.command()
@click.argument('prompt')
@click.option('--model', '-m', default='ollama-qwen3-30b')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def query(ctx, prompt, model, as_json):
    """Execute a query."""
    async def _query():
        platform = ctx.obj['platform']
        result = await platform.query(prompt, model=model)
        if as_json:
            click.echo(json.dumps(_query_payload(result), indent=2))
        else:
            click.echo(_query_content_or_raise(result))
    
    asyncio.run(_query())


@cli.command()
@click.argument('topic')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def research(ctx, topic, as_json):
    """Run autonomous research on a topic."""
    async def _research():
        platform = ctx.obj['platform']
        result = await platform.research(topic)
        return _require_command_success(result, "Research failed.")

    asyncio.run(
        _run_async_command(
            "research",
            as_json,
            _research,
            _render_research_result,
        )
    )


@cli.command()
@click.option('--categories', '-c', multiple=True)
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def trends(ctx, categories, as_json):
    """Discover trending topics."""
    async def _trends():
        platform = ctx.obj['platform']
        trend_categories = list(categories) if categories else None
        result = await platform.discover_trends(categories=trend_categories)
        return _require_command_success(result, "Trend discovery failed.")

    asyncio.run(
        _run_async_command(
            "trends",
            as_json,
            _trends,
            _render_trends_result,
        )
    )


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def metrics(ctx, as_json):
    """Show platform metrics."""
    async def _metrics():
        platform = ctx.obj['platform']
        await platform.initialize_monitoring_path()
        return _require_command_success(platform.get_metrics(), "Metrics unavailable")

    asyncio.run(
        _run_async_command(
            "metrics",
            as_json,
            _metrics,
            _render_metrics_result,
        )
    )


@cli.command()
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def discover(ctx, as_json):
    """Discover new models and resources from all sources."""
    async def _discover():
        platform = ctx.obj['platform']
        await platform.initialize_discovery_path()
        results = await platform.discover_resources()
        _require_command_success(results, "Discovery failed.")
        stats = platform.get_discovery_stats()
        return {"results": results, "stats": stats}

    asyncio.run(
        _run_async_command(
            "discover",
            as_json,
            _discover,
            _render_discovery_result,
        )
    )


@cli.command('discover-stats')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def discover_stats(ctx, as_json):
    """Show discovery statistics."""
    async def _stats():
        platform = ctx.obj['platform']
        await platform.initialize_discovery_path()
        return _require_command_success(platform.get_discovery_stats(), "Discovery stats unavailable")

    asyncio.run(
        _run_async_command(
            "discover-stats",
            as_json,
            _stats,
            _render_discovery_stats,
        )
    )


@cli.command('search-models')
@click.argument('query')
@click.option('--capability', '-c', multiple=True, help='Required capability')
@click.option('--max-price', type=float, help='Max price per 1k tokens')
@click.option('--min-context', type=int, help='Min context length')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def search_models(ctx, query, capability, max_price, min_context, as_json):
    """Search for AI models."""
    async def _search():
        platform = ctx.obj['platform']
        return await platform.search_models(
            query=query,
            capabilities=list(capability) if capability else None,
            max_price=max_price,
            min_context=min_context,
        )

    def _render(models):
        click.echo(f"\nFound {len(models)} models matching '{query}':\n")
        for model in models[:20]:
            name = getattr(model, 'name', str(model))
            provider = getattr(model, 'provider', 'unknown')
            score = getattr(model, 'quality_score', 0)
            click.echo(f"  {name} ({provider}) - score: {score:.2f}")

    asyncio.run(_run_async_command("search-models", as_json, _search, _render))


@cli.command('search-github')
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Max results')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def search_github(ctx, query, limit, as_json):
    """Search GitHub repositories."""
    async def _search():
        platform = ctx.obj['platform']
        return await platform.search_github(query, limit=limit)

    def _render(repos):
        click.echo(f"\nFound {len(repos)} repositories:\n")
        for repo in repos:
            name = repo.get('full_name', 'unknown')
            stars = repo.get('stargazers_count', 0)
            desc = repo.get('description', '')[:60] if repo.get('description') else ''
            click.echo(f"  {name} ({stars} stars)")
            if desc:
                click.echo(f"    {desc}")

    asyncio.run(_run_async_command("search-github", as_json, _search, _render))


@cli.command('search-huggingface')
@click.argument('query')
@click.option('--type', '-t', 'resource_type', default='models',
              type=click.Choice(['models', 'datasets', 'spaces']))
@click.option('--limit', '-l', default=10, help='Max results')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def search_huggingface(ctx, query, resource_type, limit, as_json):
    """Search HuggingFace resources."""
    async def _search():
        platform = ctx.obj['platform']
        return await platform.search_huggingface(
            query=query,
            resource_type=resource_type,
            limit=limit,
        )

    def _render(results):
        click.echo(f"\nFound {len(results)} {resource_type}:\n")
        for item in results:
            if resource_type == 'models':
                name = item.get('modelId', item.get('id', 'unknown'))
                downloads = item.get('downloads', 0)
                click.echo(f"  {name} ({downloads:,} downloads)")
            elif resource_type == 'datasets':
                name = item.get('id', 'unknown')
                downloads = item.get('downloads', 0)
                click.echo(f"  {name} ({downloads:,} downloads)")
            else:
                name = item.get('id', 'unknown')
                likes = item.get('likes', 0)
                click.echo(f"  {name} ({likes} likes)")

    asyncio.run(_run_async_command("search-huggingface", as_json, _search, _render))


@cli.command('search-arxiv')
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Max results')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def search_arxiv(ctx, query, limit, as_json):
    """Search Arxiv for research papers."""
    async def _search():
        platform = ctx.obj['platform']
        return await platform.search_arxiv(query, max_results=limit)

    def _render(papers):
        click.echo(f"\nFound {len(papers)} papers:\n")
        for paper in papers:
            title = paper.get('title', 'Untitled')[:70]
            authors = ', '.join(paper.get('authors', [])[:3])
            if len(paper.get('authors', [])) > 3:
                authors += ' et al.'
            click.echo(f"  {title}")
            click.echo(f"    Authors: {authors}")
            click.echo(f"    URL: {paper.get('id', 'N/A')}")
            click.echo()

    asyncio.run(_run_async_command("search-arxiv", as_json, _search, _render))


@cli.command('search-pypi')
@click.argument('query')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def search_pypi(ctx, query, as_json):
    """Search PyPI for Python packages."""
    async def _search():
        platform = ctx.obj['platform']
        return await platform.search_pypi(query)

    def _render(packages):
        click.echo(f"\nFound {len(packages)} packages:\n")
        for pkg in packages:
            info = pkg.get('info', {})
            name = info.get('name', 'unknown')
            version = info.get('version', '?')
            summary = info.get('summary', '')[:60] if info.get('summary') else ''
            click.echo(f"  {name} ({version})")
            if summary:
                click.echo(f"    {summary}")

    asyncio.run(_run_async_command("search-pypi", as_json, _search, _render))


@cli.command('pypi-info')
@click.argument('package')
@click.option('--full', is_flag=True, help='Include the full upstream package payload in JSON output.')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def pypi_info(ctx, package, full, as_json):
    """Get detailed PyPI package information."""
    async def _info():
        platform = ctx.obj['platform']
        pkg = await platform.get_pypi_package(package)
        if not pkg:
            raise click.ClickException(f"Package '{package}' not found")
        return _require_command_success(pkg, f"Package '{package}' could not be loaded")

    def _render(pkg):
        info = pkg.get('info', {})
        click.echo(f"\n{info.get('name', package)} ({info.get('version', '?')})")
        click.echo(f"  Summary: {info.get('summary', 'N/A')}")
        click.echo(f"  Author: {info.get('author', 'N/A')}")
        click.echo(f"  License: {info.get('license', 'N/A')}")
        click.echo(f"  Python: {info.get('requires_python', 'N/A')}")
        click.echo(f"  URL: {info.get('project_url', 'N/A')}")

    def _payload(pkg):
        data = pkg if full else _summarize_pypi_package(pkg)
        return _success_payload("pypi-info", data, full=full, package=package)

    asyncio.run(_run_async_command("pypi-info", as_json, _info, _render, payload_builder=_payload))


@cli.command('ollama-list')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def ollama_list(ctx, as_json):
    """List locally installed Ollama models."""
    async def _list():
        platform = ctx.obj['platform']
        return await platform.list_ollama_models()

    def _render(models):
        if not models:
            click.echo("\nNo Ollama models found (is Ollama running?)")
            return

        click.echo(f"\nFound {len(models)} local models:\n")
        for model in models:
            name = model.get('name', 'unknown')
            size = model.get('size', 0) / (1024 ** 3)  # GB
            click.echo(f"  {name} ({size:.1f} GB)")

    asyncio.run(_run_async_command("ollama-list", as_json, _list, _render))


@cli.command('web-search')
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Max results')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def web_search_cmd(ctx, query, limit, as_json):
    """Search the web."""
    async def _search():
        platform = ctx.obj['platform']
        return await platform.web_search(query, num_results=limit)

    def _render(results):
        click.echo(f"\nFound {len(results)} results:\n")
        for result in results:
            title = result.get('title', 'Untitled')[:70]
            url = result.get('url', 'N/A')
            snippet = result.get('snippet', '')[:100] if result.get('snippet') else ''
            click.echo(f"  {title}")
            click.echo(f"    {url}")
            if snippet:
                click.echo(f"    {snippet}...")
            click.echo()

    asyncio.run(_run_async_command("web-search", as_json, _search, _render))


@cli.command('news-search')
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Max results')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def news_search(ctx, query, limit, as_json):
    """Search for news articles."""
    async def _search():
        platform = ctx.obj['platform']
        return await platform.search_news(query, num_results=limit)

    def _render(results):
        click.echo(f"\nFound {len(results)} news articles:\n")
        for result in results:
            title = result.get('title', 'Untitled')[:70]
            source = result.get('source_name', result.get('source', 'Unknown'))
            date = result.get('date', '')
            click.echo(f"  {title}")
            click.echo(f"    Source: {source} | {date}")
            click.echo()

    asyncio.run(_run_async_command("news-search", as_json, _search, _render))


# ==================== Local Machine Commands ====================

@cli.command('system-info')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def system_info(ctx, as_json):
    """Show local system information."""
    async def _info():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()
        return _require_command_success(platform.get_system_info(), "System info unavailable")

    def _render(info):
        click.echo("\nSystem Information\n")

        # Platform info
        plat = info.get('platform', {})
        click.echo(f"  OS: {plat.get('system', 'Unknown')} {plat.get('release', '')}")
        click.echo(f"  Machine: {plat.get('machine', 'Unknown')}")
        click.echo(f"  Hostname: {plat.get('hostname', 'Unknown')}")
        click.echo(f"  Python: {plat.get('python_version', 'Unknown')}")

        # CPU
        cpu = info.get('cpu', {})
        click.echo(f"\n  CPU Cores: {cpu.get('logical_cores', 'Unknown')}")
        if cpu.get('percent'):
            click.echo(f"  CPU Usage: {cpu.get('percent')}%")

        # Memory
        mem = info.get('memory', {})
        if 'total_gb' in mem:
            click.echo(f"\n  Memory: {mem.get('used_gb', 0):.1f} / {mem.get('total_gb', 0):.1f} GB ({mem.get('percent', 0)}%)")

        # Disk
        disk = info.get('disk', {})
        partitions = disk.get('partitions', [])
        if partitions:
            click.echo("\n  Disk:")
            for p in partitions[:3]:
                click.echo(f"    {p.get('mountpoint')}: {p.get('used_gb', 0):.1f} / {p.get('total_gb', 0):.1f} GB ({p.get('percent', 0)}%)")

    asyncio.run(_run_async_command("system-info", as_json, _info, _render))


@cli.command('ls')
@click.argument('path', default='.')
@click.option('--pattern', '-p', default='*', help='File pattern')
@click.option('--recursive', '-r', is_flag=True, help='Include subdirectories')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def list_dir(ctx, path, pattern, recursive, as_json):
    """List directory contents."""
    async def _list():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()
        return _require_command_success(platform.list_directory(path, pattern, recursive), "Directory listing failed")

    def _render(result):
        click.echo(f"\n{result['path']} ({result['count']} items):\n")
        for entry in result.get('entries', [])[:50]:
            icon = "[D]" if entry['is_dir'] else "   "
            size = f"{entry['size']:,}" if entry.get('size') else "-"
            click.echo(f"  {icon} {entry['name']:<40} {size:>12}")

        if result['count'] > 50:
            click.echo(f"\n  ... and {result['count'] - 50} more")

    asyncio.run(_run_async_command("ls", as_json, _list, _render))


@cli.command('cat')
@click.argument('path')
@click.option('--lines', '-n', type=int, help='Number of lines')
@click.option('--start', '-s', type=int, default=1, help='Start line')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def cat_file(ctx, path, lines, start, as_json):
    """Read a file's contents."""
    async def _cat():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()

        if lines:
            return _require_command_success(
                platform.read_file_lines(path, start, lines),
                f"Unable to read lines from '{path}'",
            )
        return _require_command_success(
            platform.read_local_file(path),
            f"Unable to read '{path}'",
        )

    def _render(result):
        if lines:
            click.echo(f"\n{result['path']} (lines {result['start_line']}-{result['end_line']} of {result['total_lines']}):\n")
            for line in result.get('lines', []):
                click.echo(f"{line['num']:>5}  {line['content']}")
            return

        click.echo(f"\n{result['path']} ({result['lines']} lines, {result['size']} bytes):\n")
        click.echo(result['content'][:5000])
        if len(result['content']) > 5000:
            click.echo("\n... (truncated)")

    asyncio.run(_run_async_command("cat", as_json, _cat, _render))


@cli.command('find')
@click.argument('path')
@click.argument('pattern')
@click.option('--content', '-c', help='Search file contents')
@click.option('--limit', '-l', default=20, help='Max results')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def find_files(ctx, path, pattern, content, limit, as_json):
    """Search for files."""
    async def _find():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()
        return _require_command_success(
            platform.search_local_files(path, pattern, content, limit),
            "File search failed",
        )

    def _render(result):
        click.echo(f"\nFound {result['count']} files matching '{pattern}'")
        if content:
            click.echo(f"  with content: '{content}'")
        click.echo()

        for f in result.get('results', []):
            click.echo(f"  {f['path']}")
            if f.get('matches'):
                for m in f['matches'][:3]:
                    click.echo(f"    L{m['line']}: {m['content'][:80]}")

    asyncio.run(_run_async_command("find", as_json, _find, _render))


@cli.command('processes')
@click.option('--limit', '-l', default=15, help='Max processes')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def list_processes(ctx, limit, as_json):
    """List running processes."""
    async def _ps():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()
        return _require_collection_success(platform.get_running_processes(limit), "Process listing failed")

    def _render(processes):
        click.echo(f"\nTop {len(processes)} Processes (by CPU):\n")
        click.echo(f"  {'PID':<8} {'CPU%':<8} {'MEM%':<8} NAME")
        click.echo(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*20}")
        for p in processes:
            click.echo(f"  {p['pid']:<8} {p['cpu_percent'] or 0:<8.1f} {p['memory_percent']:<8.1f} {p['name']}")

    asyncio.run(_run_async_command("processes", as_json, _ps, _render))


@cli.command('exec')
@click.argument('command')
@click.option('--cwd', '-d', help='Working directory')
@click.option('--timeout', '-t', default=60, help='Timeout in seconds')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def exec_command(ctx, command, cwd, timeout, as_json):
    """Execute a shell command."""
    async def _exec():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()
        return _require_command_success(
            await platform.execute_command(command, cwd, timeout),
            f"Command execution failed: {command}",
        )

    def _render(result):
        click.echo(f"\nExecuting: {command}")
        if cwd:
            click.echo(f"  in: {cwd}")
        click.echo()

        if result.get('stdout'):
            click.echo(result['stdout'])
        if result.get('stderr'):
            click.echo(f"STDERR:\n{result['stderr']}")

        click.echo(f"\nExit code: {result['returncode']}")

    asyncio.run(_run_async_command("exec", as_json, _exec, _render))


@cli.command('env')
@click.option('--filter', '-f', 'filter_pattern', help='Filter by pattern')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def show_env(ctx, filter_pattern, as_json):
    """Show environment variables."""
    async def _env():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()
        return _require_command_success(
            platform.get_environment_variables(filter_pattern),
            "Environment lookup failed",
        )

    def _render(env_vars):
        click.echo(f"\nEnvironment Variables ({len(env_vars)} total):\n")
        for k, v in sorted(env_vars.items())[:50]:
            v_display = v[:60] + "..." if len(v) > 60 else v
            click.echo(f"  {k}={v_display}")

        if len(env_vars) > 50:
            click.echo(f"\n  ... and {len(env_vars) - 50} more")

    asyncio.run(_run_async_command("env", as_json, _env, _render))


@cli.command('python-info')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def python_info(ctx, as_json):
    """Show Python environment info."""
    async def _info():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()
        return _require_command_success(platform.get_python_info(), "Python environment info unavailable")

    def _render(info):
        click.echo("\nPython Environment:\n")
        click.echo(f"  Version: {info['version_info']['major']}.{info['version_info']['minor']}.{info['version_info']['micro']}")
        click.echo(f"  Executable: {info['executable']}")
        click.echo(f"  Platform: {info['platform']}")
        click.echo(f"  Virtual Env: {'Yes' if info['is_virtualenv'] else 'No'}")
        click.echo(f"  Prefix: {info['prefix']}")

    asyncio.run(_run_async_command("python-info", as_json, _info, _render))


@cli.command('packages')
@click.option('--limit', '-l', default=30, help='Max packages')
@click.option('--json', 'as_json', is_flag=True, help='Emit machine-readable JSON output.')
@click.pass_context
def list_packages(ctx, limit, as_json):
    """List installed Python packages."""
    async def _list():
        platform = ctx.obj['platform']
        await platform.initialize_local_machine_path()
        return _require_collection_success(platform.get_installed_packages(), "Package listing failed")

    def _render(packages):
        click.echo(f"\nInstalled Python Packages ({len(packages)} total):\n")
        for pkg in packages[:limit]:
            click.echo(f"  {pkg['name']:<30} {pkg['version']}")

        if len(packages) > limit:
            click.echo(f"\n  ... and {len(packages) - limit} more")

    asyncio.run(_run_async_command("packages", as_json, _list, _render))


if __name__ == '__main__':
    cli()
