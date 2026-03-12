"""
Nexus Unified CLI - Single entry point for all capabilities.
"""

import asyncio
import click
from nexus.platform import NexusPlatform, PlatformConfig


@click.group()
@click.pass_context
def cli(ctx):
    """Nexus Unified AI Platform"""
    ctx.ensure_object(dict)
    ctx.obj['platform'] = NexusPlatform()


@cli.command()
@click.pass_context
def status(ctx):
    """Check platform status."""
    async def _status():
        platform = ctx.obj['platform']
        status = await platform.initialize()
        
        click.echo("\nNexus Platform Status\n")
        for component, ok in status.items():
            icon = "[OK]" if ok else "[FAIL]"
            click.echo(f"  {icon} {component}")
        click.echo()
    
    asyncio.run(_status())


@cli.command()
@click.argument('prompt')
@click.option('--model', '-m', default='ollama-qwen3-30b')
@click.pass_context
def query(ctx, prompt, model):
    """Execute a query."""
    async def _query():
        platform = ctx.obj['platform']
        await platform.initialize()
        result = await platform.query(prompt, model=model)
        click.echo(result.get('content', str(result)))
    
    asyncio.run(_query())


@cli.command()
@click.argument('topic')
@click.pass_context
def research(ctx, topic):
    """Run autonomous research on a topic."""
    async def _research():
        platform = ctx.obj['platform']
        await platform.initialize()
        result = await platform.research(topic)
        click.echo(result)
    
    asyncio.run(_research())


@cli.command()
@click.option('--categories', '-c', multiple=True)
@click.pass_context
def trends(ctx, categories):
    """Discover trending topics."""
    async def _trends():
        platform = ctx.obj['platform']
        await platform.initialize()
        result = await platform.discover_trends(categories=list(categories))
        
        click.echo("\nTrending Topics\n")
        recommendations = result.get('recommendations', [])
        if recommendations:
            for trend in recommendations[:10]:
                click.echo(f"  - {trend.get('topic', trend)} (score: {trend.get('score', 'N/A')})")
        else:
            click.echo("  No trends found")
    
    asyncio.run(_trends())


@cli.command()
@click.pass_context
def metrics(ctx):
    """Show platform metrics."""
    async def _metrics():
        platform = ctx.obj['platform']
        await platform.initialize()
        data = platform.get_metrics()

        click.echo("\nPlatform Metrics\n")
        if data:
            for key, value in data.items():
                click.echo(f"  {key}: {value}")
        else:
            click.echo("  No metrics collected yet")

    asyncio.run(_metrics())


@cli.command()
@click.pass_context
def discover(ctx):
    """Discover new models and resources from all sources."""
    async def _discover():
        platform = ctx.obj['platform']
        await platform.initialize()

        click.echo("\nDiscovering resources from all sources...\n")
        results = await platform.discover_resources()

        click.echo("Discovery Results:\n")
        for source, count in results.items():
            click.echo(f"  {source}: {count} new resources")

        # Show stats
        stats = platform.get_discovery_stats()
        click.echo(f"\nTotal resources: {stats.get('total_resources', 0)}")

    asyncio.run(_discover())


@cli.command('discover-stats')
@click.pass_context
def discover_stats(ctx):
    """Show discovery statistics."""
    async def _stats():
        platform = ctx.obj['platform']
        await platform.initialize()

        stats = platform.get_discovery_stats()

        click.echo("\nDiscovery Statistics\n")
        click.echo(f"Total resources: {stats.get('total_resources', 0)}")

        if stats.get('by_type'):
            click.echo("\nBy Type:")
            for rt, count in stats['by_type'].items():
                click.echo(f"  {rt}: {count}")

        if stats.get('by_source'):
            click.echo("\nBy Source:")
            for src, count in stats['by_source'].items():
                click.echo(f"  {src}: {count}")

        click.echo(f"\nDiscoveries today: {stats.get('recent_discoveries', 0)}")

    asyncio.run(_stats())


@cli.command('search-models')
@click.argument('query')
@click.option('--capability', '-c', multiple=True, help='Required capability')
@click.option('--max-price', type=float, help='Max price per 1k tokens')
@click.option('--min-context', type=int, help='Min context length')
@click.pass_context
def search_models(ctx, query, capability, max_price, min_context):
    """Search for AI models."""
    async def _search():
        platform = ctx.obj['platform']
        await platform.initialize()

        models = await platform.search_models(
            query=query,
            capabilities=list(capability) if capability else None,
            max_price=max_price,
            min_context=min_context,
        )

        click.echo(f"\nFound {len(models)} models matching '{query}':\n")
        for model in models[:20]:
            name = getattr(model, 'name', str(model))
            provider = getattr(model, 'provider', 'unknown')
            score = getattr(model, 'quality_score', 0)
            click.echo(f"  {name} ({provider}) - score: {score:.2f}")

    asyncio.run(_search())


@cli.command('search-github')
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Max results')
@click.pass_context
def search_github(ctx, query, limit):
    """Search GitHub repositories."""
    async def _search():
        platform = ctx.obj['platform']
        await platform.initialize()

        repos = await platform.search_github(query, limit=limit)

        click.echo(f"\nFound {len(repos)} repositories:\n")
        for repo in repos:
            name = repo.get('full_name', 'unknown')
            stars = repo.get('stargazers_count', 0)
            desc = repo.get('description', '')[:60] if repo.get('description') else ''
            click.echo(f"  {name} ({stars} stars)")
            if desc:
                click.echo(f"    {desc}")

    asyncio.run(_search())


@cli.command('search-huggingface')
@click.argument('query')
@click.option('--type', '-t', 'resource_type', default='models',
              type=click.Choice(['models', 'datasets', 'spaces']))
@click.option('--limit', '-l', default=10, help='Max results')
@click.pass_context
def search_huggingface(ctx, query, resource_type, limit):
    """Search HuggingFace resources."""
    async def _search():
        platform = ctx.obj['platform']
        await platform.initialize()

        results = await platform.search_huggingface(
            query=query,
            resource_type=resource_type,
            limit=limit,
        )

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

    asyncio.run(_search())


@cli.command('search-arxiv')
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Max results')
@click.pass_context
def search_arxiv(ctx, query, limit):
    """Search Arxiv for research papers."""
    async def _search():
        platform = ctx.obj['platform']
        await platform.initialize()

        papers = await platform.search_arxiv(query, max_results=limit)

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

    asyncio.run(_search())


@cli.command('search-pypi')
@click.argument('query')
@click.pass_context
def search_pypi(ctx, query):
    """Search PyPI for Python packages."""
    async def _search():
        platform = ctx.obj['platform']
        await platform.initialize()

        packages = await platform.search_pypi(query)

        click.echo(f"\nFound {len(packages)} packages:\n")
        for pkg in packages:
            info = pkg.get('info', {})
            name = info.get('name', 'unknown')
            version = info.get('version', '?')
            summary = info.get('summary', '')[:60] if info.get('summary') else ''
            click.echo(f"  {name} ({version})")
            if summary:
                click.echo(f"    {summary}")

    asyncio.run(_search())


@cli.command('pypi-info')
@click.argument('package')
@click.pass_context
def pypi_info(ctx, package):
    """Get detailed PyPI package information."""
    async def _info():
        platform = ctx.obj['platform']
        await platform.initialize()

        pkg = await platform.get_pypi_package(package)

        if not pkg:
            click.echo(f"\nPackage '{package}' not found")
            return

        info = pkg.get('info', {})
        click.echo(f"\n{info.get('name', package)} ({info.get('version', '?')})")
        click.echo(f"  Summary: {info.get('summary', 'N/A')}")
        click.echo(f"  Author: {info.get('author', 'N/A')}")
        click.echo(f"  License: {info.get('license', 'N/A')}")
        click.echo(f"  Python: {info.get('requires_python', 'N/A')}")
        click.echo(f"  URL: {info.get('project_url', 'N/A')}")

    asyncio.run(_info())


@cli.command('ollama-list')
@click.pass_context
def ollama_list(ctx):
    """List locally installed Ollama models."""
    async def _list():
        platform = ctx.obj['platform']
        await platform.initialize()

        models = await platform.list_ollama_models()

        if not models:
            click.echo("\nNo Ollama models found (is Ollama running?)")
            return

        click.echo(f"\nFound {len(models)} local models:\n")
        for model in models:
            name = model.get('name', 'unknown')
            size = model.get('size', 0) / (1024 ** 3)  # GB
            click.echo(f"  {name} ({size:.1f} GB)")

    asyncio.run(_list())


@cli.command('web-search')
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Max results')
@click.pass_context
def web_search_cmd(ctx, query, limit):
    """Search the web."""
    async def _search():
        platform = ctx.obj['platform']
        await platform.initialize()

        results = await platform.web_search(query, num_results=limit)

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

    asyncio.run(_search())


@cli.command('news-search')
@click.argument('query')
@click.option('--limit', '-l', default=10, help='Max results')
@click.pass_context
def news_search(ctx, query, limit):
    """Search for news articles."""
    async def _search():
        platform = ctx.obj['platform']
        await platform.initialize()

        results = await platform.search_news(query, num_results=limit)

        click.echo(f"\nFound {len(results)} news articles:\n")
        for result in results:
            title = result.get('title', 'Untitled')[:70]
            source = result.get('source_name', result.get('source', 'Unknown'))
            date = result.get('date', '')
            click.echo(f"  {title}")
            click.echo(f"    Source: {source} | {date}")
            click.echo()

    asyncio.run(_search())


# ==================== Local Machine Commands ====================

@cli.command('system-info')
@click.pass_context
def system_info(ctx):
    """Show local system information."""
    async def _info():
        platform = ctx.obj['platform']
        await platform.initialize()

        info = platform.get_system_info()

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

    asyncio.run(_info())


@cli.command('ls')
@click.argument('path', default='.')
@click.option('--pattern', '-p', default='*', help='File pattern')
@click.option('--recursive', '-r', is_flag=True, help='Include subdirectories')
@click.pass_context
def list_dir(ctx, path, pattern, recursive):
    """List directory contents."""
    async def _list():
        platform = ctx.obj['platform']
        await platform.initialize()

        result = platform.list_directory(path, pattern, recursive)

        if 'error' in result:
            click.echo(f"\nError: {result['error']}")
            return

        click.echo(f"\n{result['path']} ({result['count']} items):\n")
        for entry in result.get('entries', [])[:50]:
            icon = "[D]" if entry['is_dir'] else "   "
            size = f"{entry['size']:,}" if entry.get('size') else "-"
            click.echo(f"  {icon} {entry['name']:<40} {size:>12}")

        if result['count'] > 50:
            click.echo(f"\n  ... and {result['count'] - 50} more")

    asyncio.run(_list())


@cli.command('cat')
@click.argument('path')
@click.option('--lines', '-n', type=int, help='Number of lines')
@click.option('--start', '-s', type=int, default=1, help='Start line')
@click.pass_context
def cat_file(ctx, path, lines, start):
    """Read a file's contents."""
    async def _cat():
        platform = ctx.obj['platform']
        await platform.initialize()

        if lines:
            result = platform.read_file_lines(path, start, lines)
            if 'error' in result:
                click.echo(f"\nError: {result['error']}")
                return

            click.echo(f"\n{result['path']} (lines {result['start_line']}-{result['end_line']} of {result['total_lines']}):\n")
            for line in result.get('lines', []):
                click.echo(f"{line['num']:>5}  {line['content']}")
        else:
            result = platform.read_local_file(path)
            if 'error' in result:
                click.echo(f"\nError: {result['error']}")
                return

            click.echo(f"\n{result['path']} ({result['lines']} lines, {result['size']} bytes):\n")
            click.echo(result['content'][:5000])
            if len(result['content']) > 5000:
                click.echo("\n... (truncated)")

    asyncio.run(_cat())


@cli.command('find')
@click.argument('path')
@click.argument('pattern')
@click.option('--content', '-c', help='Search file contents')
@click.option('--limit', '-l', default=20, help='Max results')
@click.pass_context
def find_files(ctx, path, pattern, content, limit):
    """Search for files."""
    async def _find():
        platform = ctx.obj['platform']
        await platform.initialize()

        result = platform.search_local_files(path, pattern, content, limit)

        if 'error' in result:
            click.echo(f"\nError: {result['error']}")
            return

        click.echo(f"\nFound {result['count']} files matching '{pattern}'")
        if content:
            click.echo(f"  with content: '{content}'")
        click.echo()

        for f in result.get('results', []):
            click.echo(f"  {f['path']}")
            if f.get('matches'):
                for m in f['matches'][:3]:
                    click.echo(f"    L{m['line']}: {m['content'][:80]}")

    asyncio.run(_find())


@cli.command('processes')
@click.option('--limit', '-l', default=15, help='Max processes')
@click.pass_context
def list_processes(ctx, limit):
    """List running processes."""
    async def _ps():
        platform = ctx.obj['platform']
        await platform.initialize()

        processes = platform.get_running_processes(limit)

        if processes and 'error' in processes[0]:
            click.echo(f"\nError: {processes[0]['error']}")
            return

        click.echo(f"\nTop {len(processes)} Processes (by CPU):\n")
        click.echo(f"  {'PID':<8} {'CPU%':<8} {'MEM%':<8} NAME")
        click.echo(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*20}")
        for p in processes:
            click.echo(f"  {p['pid']:<8} {p['cpu_percent'] or 0:<8.1f} {p['memory_percent']:<8.1f} {p['name']}")

    asyncio.run(_ps())


@cli.command('exec')
@click.argument('command')
@click.option('--cwd', '-d', help='Working directory')
@click.option('--timeout', '-t', default=60, help='Timeout in seconds')
@click.pass_context
def exec_command(ctx, command, cwd, timeout):
    """Execute a shell command."""
    async def _exec():
        platform = ctx.obj['platform']
        await platform.initialize()

        click.echo(f"\nExecuting: {command}")
        if cwd:
            click.echo(f"  in: {cwd}")
        click.echo()

        result = await platform.execute_command(command, cwd, timeout)

        if 'error' in result:
            click.echo(f"Error: {result['error']}")
            return

        if result.get('stdout'):
            click.echo(result['stdout'])
        if result.get('stderr'):
            click.echo(f"STDERR:\n{result['stderr']}")

        click.echo(f"\nExit code: {result['returncode']}")

    asyncio.run(_exec())


@cli.command('env')
@click.option('--filter', '-f', 'filter_pattern', help='Filter by pattern')
@click.pass_context
def show_env(ctx, filter_pattern):
    """Show environment variables."""
    async def _env():
        platform = ctx.obj['platform']
        await platform.initialize()

        env_vars = platform.get_environment_variables(filter_pattern)

        if 'error' in env_vars:
            click.echo(f"\nError: {env_vars['error']}")
            return

        click.echo(f"\nEnvironment Variables ({len(env_vars)} total):\n")
        for k, v in sorted(env_vars.items())[:50]:
            v_display = v[:60] + "..." if len(v) > 60 else v
            click.echo(f"  {k}={v_display}")

        if len(env_vars) > 50:
            click.echo(f"\n  ... and {len(env_vars) - 50} more")

    asyncio.run(_env())


@cli.command('python-info')
@click.pass_context
def python_info(ctx):
    """Show Python environment info."""
    async def _info():
        platform = ctx.obj['platform']
        await platform.initialize()

        info = platform.get_python_info()

        if 'error' in info:
            click.echo(f"\nError: {info['error']}")
            return

        click.echo("\nPython Environment:\n")
        click.echo(f"  Version: {info['version_info']['major']}.{info['version_info']['minor']}.{info['version_info']['micro']}")
        click.echo(f"  Executable: {info['executable']}")
        click.echo(f"  Platform: {info['platform']}")
        click.echo(f"  Virtual Env: {'Yes' if info['is_virtualenv'] else 'No'}")
        click.echo(f"  Prefix: {info['prefix']}")

    asyncio.run(_info())


@cli.command('packages')
@click.option('--limit', '-l', default=30, help='Max packages')
@click.pass_context
def list_packages(ctx, limit):
    """List installed Python packages."""
    async def _list():
        platform = ctx.obj['platform']
        await platform.initialize()

        packages = platform.get_installed_packages()

        if packages and 'error' in packages[0]:
            click.echo(f"\nError: {packages[0]['error']}")
            return

        click.echo(f"\nInstalled Python Packages ({len(packages)} total):\n")
        for pkg in packages[:limit]:
            click.echo(f"  {pkg['name']:<30} {pkg['version']}")

        if len(packages) > limit:
            click.echo(f"\n  ... and {len(packages) - limit} more")

    asyncio.run(_list())


if __name__ == '__main__':
    cli()
