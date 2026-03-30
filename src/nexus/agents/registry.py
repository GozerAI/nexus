"""
Agent Registry

Manages registration and discovery of Nexus-Agents.
"""

from typing import Dict, List, Optional, Any, Protocol
from enum import Enum


class AgentCapability(Enum):
    """Agent capability types"""
    DATA_PROCESSING = "data_processing"
    CODE_EXECUTION = "code_execution"
    WEB_SCRAPING = "web_scraping"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"
    PLANNING = "planning"
    ORCHESTRATION = "orchestration"


class AgentInterface(Protocol):
    """Protocol that all Nexus-Agents must implement"""

    @property
    def name(self) -> str:
        """Agent unique identifier"""
        ...

    @property
    def capabilities(self) -> List[AgentCapability]:
        """List of agent capabilities"""
        ...

    @property
    def version(self) -> str:
        """Agent version string"""
        ...

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task assigned to this agent.

        Args:
            task: Task specification with:
                - task_id: str
                - task_type: str
                - parameters: Dict[str, Any]
                - context: Optional[Dict[str, Any]]

        Returns:
            Result dictionary with:
                - status: str ("success" | "failure" | "partial")
                - result: Any
                - metadata: Dict[str, Any]
                - trace: Optional[List[str]]
        """
        ...

    async def validate_task(self, task: Dict[str, Any]) -> bool:
        """Check if agent can handle this task"""
        ...

    async def health_check(self) -> Dict[str, Any]:
        """Return agent health status"""
        ...


class AgentRegistry:
    """Registry for managing available agents"""

    def __init__(self):
        self._agents: Dict[str, AgentInterface] = {}
        self._capabilities_index: Dict[AgentCapability, List[str]] = {}

    def register(self, agent: AgentInterface) -> None:
        """Register an agent with the platform"""
        self._agents[agent.name] = agent
        for capability in agent.capabilities:
            if capability not in self._capabilities_index:
                self._capabilities_index[capability] = []
            self._capabilities_index[capability].append(agent.name)

    def unregister(self, agent_name: str) -> None:
        """Unregister an agent"""
        if agent_name in self._agents:
            agent = self._agents[agent_name]
            for capability in agent.capabilities:
                if capability in self._capabilities_index:
                    self._capabilities_index[capability].remove(agent_name)
            del self._agents[agent_name]

    def get_agent(self, agent_name: str) -> Optional[AgentInterface]:
        """Retrieve agent by name"""
        return self._agents.get(agent_name)

    def find_agents_by_capability(
        self, capability: AgentCapability
    ) -> List[AgentInterface]:
        """Find all agents with a specific capability"""
        agent_names = self._capabilities_index.get(capability, [])
        return [self._agents[name] for name in agent_names]

    def list_all_agents(self) -> List[str]:
        """List all registered agent names"""
        return list(self._agents.keys())

    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all registered agents"""
        results = {}
        for name, agent in self._agents.items():
            try:
                results[name] = await agent.health_check()
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
        return results
