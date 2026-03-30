"""Migrate from Claude Memory MCP to Nexus Intelligence."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_from_claude_memory.py <source_path>")
        print("  source_path: Path to exported Claude Memory data")
        sys.exit(1)

    source_path = sys.argv[1]

    from nexus.intelligence import NexusIntelligence
    from nexus.importers.claude_memory import ClaudeMemoryMigrator

    intel = NexusIntelligence()
    await intel.initialize()

    try:
        migrator = ClaudeMemoryMigrator(source_path, intel)
        await migrator.migrate_all()
    finally:
        await intel.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
