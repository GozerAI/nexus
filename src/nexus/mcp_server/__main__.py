"""Entry point for Nexus MCP Server."""

import asyncio
from nexus.intelligence import NexusIntelligence
from nexus.mcp_server.server import NexusMCPServer


async def main():
    intel = NexusIntelligence()
    await intel.initialize()

    try:
        server = NexusMCPServer(intel)
        await server.run()
    finally:
        await intel.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
