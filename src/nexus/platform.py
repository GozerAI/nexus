"""
Nexus Unified Platform - Main Entry Point

Combines:
- cog-eng (cognitive core)
- Nexus (orchestration, memory, RAG, reasoning)
- unified-intelligence (provider layer)
- Panel of Experts (decision system)
- Observatory (monitoring)
- Insights (trend detection)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import asyncio


@dataclass
class PlatformConfig:
    """Configuration for the unified platform."""
    default_model: str = "ollama-qwen3-30b"
    enable_monitoring: bool = True
    enable_insights: bool = True
    enable_discovery: bool = True
    auto_discover_on_init: bool = False
    autonomy_level: str = "supervised"


class NexusPlatform:
    """
    Unified AI Platform combining all capabilities.
    """
    
    def __init__(self, config: Optional[PlatformConfig] = None):
        self.config = config or PlatformConfig()
        self._initialized = False
        
        # Components (lazy init)
        self._ensemble = None
        self._consciousness = None
        self._research_agent = None
        self._experts = None
        self._metrics = None
        self._insights = None
        self._cost_tracker = None

        # Discovery components
        self._resource_discovery = None
        self._model_discovery = None
        self._github_integration = None
        self._huggingface_integration = None
        self._arxiv_integration = None
        self._pypi_integration = None
        self._ollama_integration = None
        self._web_search = None
        self._local_machine = None
    
    async def initialize(self) -> Dict[str, bool]:
        """Initialize all platform components."""
        status = {}
        
        # Initialize ensemble core
        try:
            from nexus.core.strategic_ensemble import StrategicEnsemble
            self._ensemble = StrategicEnsemble()
            status["ensemble"] = True
        except Exception as e:
            status["ensemble"] = False
            
        # Initialize cognitive core
        try:
            from nexus.cog_eng import ConsciousnessCore, AutonomousResearchAgent
            self._consciousness = ConsciousnessCore()
            self._research_agent = AutonomousResearchAgent()
            status["cog_eng"] = True
        except Exception as e:
            status["cog_eng"] = False
            
        # Initialize experts
        try:
            from nexus.experts import ConsensusEngine
            self._experts = ConsensusEngine()
            status["experts"] = True
        except Exception as e:
            status["experts"] = False
            
        # Initialize monitoring
        if self.config.enable_monitoring:
            try:
                from nexus.observatory import MetricsCollector
                self._metrics = MetricsCollector()
                status["observatory"] = True
            except Exception as e:
                status["observatory"] = False
                
        # Initialize insights
        if self.config.enable_insights:
            try:
                from nexus.insights import InsightsEngine
                self._insights = InsightsEngine()
                status["insights"] = True
            except Exception as e:
                status["insights"] = False

        # Initialize discovery system
        if self.config.enable_discovery:
            try:
                from nexus.discovery import (
                    ResourceDiscovery,
                    ModelDiscoveryEngine,
                    GitHubIntegration,
                    HuggingFaceIntegration,
                    ArxivIntegration,
                    PyPIIntegration,
                    OllamaIntegration,
                    WebSearchIntegration,
                    LocalMachineIntegration,
                )

                # Initialize core discovery
                self._resource_discovery = ResourceDiscovery()

                # Initialize model discovery with self-registration
                self._model_discovery = ModelDiscoveryEngine(self._resource_discovery)

                # Initialize GitHub integration
                self._github_integration = GitHubIntegration(self._resource_discovery)

                # Initialize HuggingFace integration
                self._huggingface_integration = HuggingFaceIntegration(self._resource_discovery)

                # Initialize Arxiv integration for research papers
                self._arxiv_integration = ArxivIntegration(self._resource_discovery)

                # Initialize PyPI integration for Python packages
                self._pypi_integration = PyPIIntegration(self._resource_discovery)

                # Initialize Ollama integration for local models
                self._ollama_integration = OllamaIntegration(self._resource_discovery)

                # Initialize web search
                self._web_search = WebSearchIntegration(self._resource_discovery)

                # Initialize local machine integration
                self._local_machine = LocalMachineIntegration()

                status["discovery"] = True

                # Auto-discover if configured
                if self.config.auto_discover_on_init:
                    await self.discover_resources()

            except Exception as e:
                status["discovery"] = False

        self._initialized = True
        return status
    
    async def query(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Execute a query through the platform."""
        if not self._initialized:
            await self.initialize()
        
        # Route through ensemble
        if self._ensemble:
            response = await self._ensemble.query(prompt, **kwargs)
        else:
            response = {"content": "Ensemble not initialized", "error": True}
        
        # Track metrics
        if self._metrics:
            self._metrics.increment("queries.total")
            
        return response
    
    async def research(self, topic: str, **kwargs) -> Dict[str, Any]:
        """Execute autonomous research."""
        if not self._initialized:
            await self.initialize()
        
        if self._research_agent:
            return await self._research_agent.research(topic, **kwargs)
        return {"error": "Research agent not initialized"}
    
    async def get_expert_opinion(self, task) -> Dict[str, Any]:
        """Get consensus from expert panel."""
        if not self._initialized:
            await self.initialize()
        
        if self._experts:
            return await self._experts.get_consensus(task)
        return {"error": "Experts not initialized"}
    
    async def discover_trends(self, **kwargs) -> Dict[str, Any]:
        """Discover trending topics."""
        if not self._initialized:
            await self.initialize()
        
        if self._insights:
            return await self._insights.discover(**kwargs)
        return {"error": "Insights not initialized"}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current platform metrics."""
        if self._metrics:
            return self._metrics.get_all()
        return {}

    async def discover_resources(self) -> Dict[str, int]:
        """
        Discover resources from all sources.

        This triggers discovery from:
        - OpenRouter (100+ models)
        - OpenAI (latest models)
        - HuggingFace (models, datasets, spaces)
        - GitHub (datasets, tools, trending repos)

        Discovered resources are automatically registered
        and models are added to the model registry.

        Returns:
            Dictionary of source -> count of new discoveries
        """
        if not self._initialized:
            await self.initialize()

        if not self._resource_discovery:
            return {"error": "Discovery not initialized"}

        return await self._resource_discovery.discover_all()

    async def discover_models(self) -> int:
        """
        Discover and self-register new models.

        Returns:
            Number of new models discovered
        """
        if not self._initialized:
            await self.initialize()

        if self._model_discovery:
            return await self._model_discovery.discover()
        return 0

    async def search_models(
        self,
        query: str,
        capabilities: Optional[List[str]] = None,
        max_price: Optional[float] = None,
        min_context: Optional[int] = None,
    ) -> List[Any]:
        """
        Search for models matching criteria.

        Args:
            query: Search query
            capabilities: Required capabilities (e.g., ["vision", "code_generation"])
            max_price: Maximum price per 1k tokens
            min_context: Minimum context length

        Returns:
            List of matching models
        """
        if not self._initialized:
            await self.initialize()

        if self._model_discovery:
            return await self._model_discovery.search_models(
                query=query,
                capabilities=capabilities,
                max_price=max_price,
                min_context=min_context,
            )
        return []

    async def search_datasets(
        self,
        query: str,
        source: Optional[str] = None,
    ) -> List[Any]:
        """
        Search for datasets.

        Args:
            query: Search query
            source: Filter by source ("github" or "huggingface")

        Returns:
            List of matching datasets
        """
        if not self._initialized:
            await self.initialize()

        if self._resource_discovery:
            return self._resource_discovery.get_datasets(query=query)
        return []

    async def search_github(
        self,
        query: str,
        limit: int = 30,
    ) -> List[Any]:
        """
        Search GitHub repositories.

        Args:
            query: Search query (supports GitHub search syntax)
            limit: Maximum results

        Returns:
            List of matching repositories
        """
        if not self._initialized:
            await self.initialize()

        if self._github_integration:
            return await self._github_integration.search_repositories(query, limit=limit)
        return []

    async def search_huggingface(
        self,
        query: str,
        resource_type: str = "models",
        limit: int = 30,
    ) -> List[Any]:
        """
        Search HuggingFace resources.

        Args:
            query: Search query
            resource_type: "models", "datasets", or "spaces"
            limit: Maximum results

        Returns:
            List of matching resources
        """
        if not self._initialized:
            await self.initialize()

        if self._huggingface_integration:
            if resource_type == "models":
                return await self._huggingface_integration.search_models(query=query, limit=limit)
            elif resource_type == "datasets":
                return await self._huggingface_integration.search_datasets(query=query, limit=limit)
            elif resource_type == "spaces":
                return await self._huggingface_integration.search_spaces(query=query, limit=limit)
        return []

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get statistics about discovered resources."""
        if self._resource_discovery:
            return self._resource_discovery.get_stats()
        return {}

    async def search_arxiv(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[Any]:
        """
        Search Arxiv for research papers.

        Args:
            query: Search query (supports Arxiv query syntax)
            max_results: Maximum results

        Returns:
            List of matching papers
        """
        if not self._initialized:
            await self.initialize()

        if self._arxiv_integration:
            return await self._arxiv_integration.search_papers(query, max_results=max_results)
        return []

    async def search_pypi(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[Any]:
        """
        Search PyPI for Python packages.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of matching packages
        """
        if not self._initialized:
            await self.initialize()

        if self._pypi_integration:
            return await self._pypi_integration.search_packages(query, max_results=max_results)
        return []

    async def get_pypi_package(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a PyPI package."""
        if not self._initialized:
            await self.initialize()

        if self._pypi_integration:
            return await self._pypi_integration.get_package_info(package_name)
        return None

    async def list_ollama_models(self) -> List[Any]:
        """List locally installed Ollama models."""
        if not self._initialized:
            await self.initialize()

        if self._ollama_integration:
            return await self._ollama_integration.list_models()
        return []

    async def ollama_generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
    ) -> Optional[str]:
        """Generate response using a local Ollama model."""
        if not self._initialized:
            await self.initialize()

        if self._ollama_integration:
            return await self._ollama_integration.generate(model, prompt, system=system)
        return None

    async def web_search(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Any]:
        """
        Search the web.

        Args:
            query: Search query
            num_results: Number of results

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        if self._web_search:
            return await self._web_search.search(query, num_results=num_results)
        return []

    async def search_news(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Any]:
        """Search for news articles."""
        if not self._initialized:
            await self.initialize()

        if self._web_search:
            return await self._web_search.search_news(query, num_results=num_results)
        return []

    # ==================== Local Machine Methods ====================

    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive local system information."""
        if self._local_machine:
            return self._local_machine.get_system_info()
        return {"error": "Local machine integration not initialized"}

    def read_local_file(
        self,
        path: str,
        max_size_mb: float = 10,
    ) -> Dict[str, Any]:
        """
        Read a local file.

        Args:
            path: File path
            max_size_mb: Maximum file size in MB

        Returns:
            Dict with file content or error
        """
        if self._local_machine:
            return self._local_machine.read_file(path, max_size_mb=max_size_mb)
        return {"error": "Local machine integration not initialized"}

    def read_file_lines(
        self,
        path: str,
        start_line: int = 1,
        num_lines: int = 100,
    ) -> Dict[str, Any]:
        """Read specific lines from a file."""
        if self._local_machine:
            return self._local_machine.read_file_lines(path, start_line, num_lines)
        return {"error": "Local machine integration not initialized"}

    def list_directory(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False,
    ) -> Dict[str, Any]:
        """List directory contents."""
        if self._local_machine:
            return self._local_machine.list_directory(path, pattern, recursive)
        return {"error": "Local machine integration not initialized"}

    def search_local_files(
        self,
        path: str,
        pattern: str,
        content_pattern: Optional[str] = None,
        max_results: int = 100,
    ) -> Dict[str, Any]:
        """
        Search for files by name and optionally content.

        Args:
            path: Base directory
            pattern: File name pattern (glob)
            content_pattern: Text to search in files
            max_results: Maximum results

        Returns:
            Dict with matching files
        """
        if self._local_machine:
            return self._local_machine.search_files(path, pattern, content_pattern, max_results)
        return {"error": "Local machine integration not initialized"}

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get detailed file information."""
        if self._local_machine:
            return self._local_machine.get_file_info(path)
        return {"error": "Local machine integration not initialized"}

    def get_running_processes(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of running processes."""
        if self._local_machine:
            return self._local_machine.get_running_processes(limit)
        return [{"error": "Local machine integration not initialized"}]

    def get_installed_packages(self) -> List[Dict[str, str]]:
        """Get list of installed Python packages."""
        if self._local_machine:
            return self._local_machine.get_installed_packages()
        return [{"error": "Local machine integration not initialized"}]

    def get_environment_variables(self, filter_pattern: Optional[str] = None) -> Dict[str, str]:
        """Get environment variables (sensitive values masked)."""
        if self._local_machine:
            return self._local_machine.get_environment_variables(filter_pattern)
        return {"error": "Local machine integration not initialized"}

    async def execute_command(
        self,
        command: str,
        cwd: Optional[str] = None,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Execute a shell command.

        Args:
            command: Command to execute
            cwd: Working directory
            timeout: Timeout in seconds

        Returns:
            Dict with output or error
        """
        if self._local_machine:
            return await self._local_machine.execute_command(command, cwd, timeout)
        return {"error": "Local machine integration not initialized"}

    def get_python_info(self) -> Dict[str, Any]:
        """Get Python environment information."""
        if self._local_machine:
            return self._local_machine.get_python_info()
        return {"error": "Local machine integration not initialized"}


# Convenience function
async def get_platform() -> NexusPlatform:
    """Get initialized platform instance."""
    platform = NexusPlatform()
    await platform.initialize()
    return platform
