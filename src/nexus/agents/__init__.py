"""
Nexus Agent Integration Layer

This module provides the integration point between Nexus (main platform)
and Nexus-Agents (autonomous agent framework).

Usage:
    from nexus.agents import AgentIntegration, AgentCapability

    # Initialize integration
    agent_integration = AgentIntegration()
    await agent_integration.initialize()

    # Submit a task
    task_id = await agent_integration.submit_task(
        task_type="data_processing",
        capability=AgentCapability.DATA_PROCESSING,
        parameters={"file": "data.csv", "operation": "transform"}
    )
"""

from .registry import AgentRegistry, AgentInterface, AgentCapability

__all__ = [
    "AgentIntegration",
    "AgentRegistry",
    "AgentInterface",
    "AgentCapability",
]


class AgentIntegration:
    """
    Main integration class for Nexus-Agents

    This class provides a unified interface for all agent operations.
    Agents are loaded from the separate Nexus-Agents package.
    """

    def __init__(self):
        self.registry = AgentRegistry()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent integration system"""
        if self._initialized:
            return

        # Agents will be registered when Nexus-Agents package is installed
        # and agents are explicitly registered by the user
        self._initialized = True

    def register_agent(self, agent: AgentInterface) -> None:
        """Register an agent with the platform"""
        self.registry.register(agent)

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent"""
        self.registry.unregister(agent_name)

    def list_agents(self) -> list:
        """List all registered agents"""
        return self.registry.list_all_agents()

    async def health_check(self) -> dict:
        """Check health of all agents"""
        return await self.registry.health_check_all()
