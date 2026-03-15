"""Nexus shared platform facade.

Combines reusable shared services:
- cog-eng (cognitive core)
- Nexus (providers, memory, RAG, reasoning, discovery)
- unified-intelligence (provider layer)
- Panel of Experts (decision system)
- Observatory (monitoring)
- Insights (trend detection)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging
from uuid import uuid4


logger = logging.getLogger(__name__)


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
    Shared infrastructure facade combining reusable Nexus capabilities.
    """
    
    def __init__(self, config: Optional[PlatformConfig] = None):
        self.config = config or PlatformConfig()
        self._initialized = False
        self._query_initialized = False
        self._status: Dict[str, bool] = {}
        
        # Components (lazy init)
        self._ensemble = None
        self._consciousness = None
        self._research_agent = None
        self._code_generator = None
        self._experts = None
        self._metrics = None
        self._insights = None
        self._cost_tracker = None
        self._llm = None

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
        await self._ensure_llm_component()
        await self._ensure_ensemble_component()
        await self._ensure_cognitive_components()
        await self._ensure_expert_components()
        await self._ensure_monitoring_components()
        await self._ensure_insights_components()
        await self._ensure_discovery_components()
        self._initialized = True
        return dict(self._status)

    async def initialize_query_path(self) -> Dict[str, bool]:
        """Initialize only the components required for prompt execution."""
        await self._ensure_llm_component()
        self._query_initialized = True
        return {
            "llm": self._status.get("llm", False),
            "query_backend": self._query_backend_ready(),
        }

    async def initialize_research_path(self) -> Dict[str, bool]:
        """Initialize only the cognitive research stack."""
        await self._ensure_cognitive_components()
        return {
            "cog_eng": self._status.get("cog_eng", False),
        }

    async def initialize_expert_path(self) -> Dict[str, bool]:
        """Initialize only the expert consensus stack."""
        await self._ensure_expert_components()
        return {
            "experts": self._status.get("experts", False),
        }

    async def initialize_codegen_path(self) -> Dict[str, bool]:
        """Initialize only the code-generation stack."""
        await self._ensure_codegen_component()
        return {
            "codegen": self._status.get("codegen", False),
        }

    async def initialize_monitoring_path(self) -> Dict[str, bool]:
        """Initialize only monitoring/metrics components."""
        await self._ensure_monitoring_components()
        return {
            "observatory": self._status.get("observatory", False),
        }

    async def initialize_insights_path(self) -> Dict[str, bool]:
        """Initialize only the insights engine."""
        await self._ensure_insights_components()
        return {
            "insights": self._status.get("insights", False),
        }

    async def initialize_discovery_path(self) -> Dict[str, bool]:
        """Initialize shared discovery services without unrelated subsystems."""
        await self._ensure_discovery_components()
        return {
            "discovery": self._status.get("discovery", False),
        }

    async def initialize_local_machine_path(self) -> Dict[str, bool]:
        """Initialize only local-machine integration."""
        await self._ensure_local_machine_components()
        return {
            "local_machine": self._local_machine is not None,
        }

    async def get_status(self, full: bool = False) -> Dict[str, bool]:
        """Return status with an explicit query backend signal.

        By default this performs a lightweight readiness check centered on the
        query path. Pass ``full=True`` to eagerly initialize the broader
        platform surface.
        """
        if full:
            await self.initialize()
        else:
            await self.initialize_query_path()
            await self._ensure_monitoring_components()
            await self._ensure_ensemble_component()

        status = dict(self._status)
        query_status = await self.get_query_backend_status()
        status["direct_llm_backend"] = query_status["direct_llm_backend"]
        status["preferred_query_backend"] = query_status["preferred_query_backend"]
        status["fallback_query_backend"] = query_status["fallback_query_backend"]
        status["query_backend"] = query_status["query_backend"]
        return status
    
    async def query(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Execute a query through the platform."""
        if not self._query_initialized:
            await self.initialize_query_path()

        query_options = dict(kwargs)
        strategy = query_options.get("strategy")
        should_use_ensemble = strategy not in (None, "simple_best")
        preferred_preset = query_options.get("model", self.config.default_model)
        direct_path_failed = False
        response = None

        if self._llm and not should_use_ensemble:
            task_type = query_options.get("task_type", "conversation")
            system_prompt = query_options.get("system_prompt", "")
            max_tokens = query_options.get("max_tokens", 8000)
            model = preferred_preset
            skip_cache = query_options.get("skip_cache", False)

            try:
                llm_response = await self._llm.generate(
                    prompt=prompt,
                    task_type=task_type,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    force_model=model,
                    skip_cache=skip_cache,
                )
                actual_preset = llm_response.get("actual_preset", llm_response.get("preset_used", model))
                preferred_route = llm_response.get("preferred_preset", model)
                fallback_used = llm_response.get("fallback_used", actual_preset != preferred_route)
                response = {
                    "content": llm_response["content"],
                    "model_name": llm_response.get("model", model),
                    "provider": llm_response.get("provider", "unknown"),
                    "preset_used": llm_response.get("preset_used", model),
                    "preferred_preset": preferred_route,
                    "actual_preset": actual_preset,
                    "backend_mode": llm_response.get(
                        "backend_mode",
                        "fallback" if fallback_used else "preferred",
                    ),
                    "query_path": "direct_llm",
                    "fallback_used": fallback_used,
                    "models_tried": llm_response.get("models_tried", []),
                    "duration_seconds": llm_response.get("duration_seconds", 0.0),
                    "tokens_used": llm_response.get("tokens_used", 0),
                    "cached": llm_response.get("cached", False),
                    "strategy_used": "direct_llm",
                }
            except Exception as exc:
                direct_path_failed = True
                logger.info(
                    "Direct LLM path failed; falling back to ensemble query: %s",
                    exc,
                )

        # Fall back to ensemble when direct routing is unavailable or strategic mode requested.
        if response is None:
            if self._ensemble is None:
                await self._ensure_ensemble_component()
            if self._ensemble:
                response = await self._ensemble.query(prompt, **query_options)
                response["preferred_preset"] = preferred_preset
                response["actual_preset"] = response.get("model_name", preferred_preset)
                response["backend_mode"] = "ensemble"
                response["query_path"] = "ensemble"
                response["fallback_used"] = direct_path_failed
            else:
                response = {"content": "No query backend initialized", "error": True}
        
        # Track metrics
        if self._metrics:
            self._metrics.increment("queries.total")
            
        return response
    
    async def research(self, topic: str, **kwargs) -> Dict[str, Any]:
        """Execute autonomous research."""
        if self._research_agent is None:
            await self.initialize_research_path()
        
        if self._research_agent:
            return await self._research_agent.research(topic, **kwargs)
        return {"error": "Research agent not initialized"}
    
    async def get_expert_opinion(self, task) -> Dict[str, Any]:
        """Get consensus from expert panel."""
        if self._experts is None:
            await self.initialize_expert_path()
        
        if self._experts:
            return await self._experts.get_consensus(task)
        return {"error": "Experts not initialized"}

    async def generate_code(
        self,
        description: str,
        language: str = "python",
        requirements: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        style_preferences: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: Optional[int] = None,
        target_quality: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate code through the shared self-improving codegen capability."""
        if self._code_generator is None:
            await self.initialize_codegen_path()

        if self._code_generator is None:
            raise RuntimeError("Code generator not initialized")

        from nexus.cog_eng.capabilities.self_improving_codegen import CodeGenerationRequest

        request = CodeGenerationRequest(
            request_id=str(uuid4()),
            description=description,
            language=language,
            requirements=list(requirements or []),
            constraints=list(constraints or []),
            style_preferences=dict(style_preferences or {}),
            context=dict(context or {}),
        )

        generation_kwargs: Dict[str, Any] = {}
        if max_iterations is not None:
            generation_kwargs["max_iterations"] = max_iterations
        if target_quality is not None:
            generation_kwargs["target_quality"] = target_quality

        result = await self._code_generator.generate(request, **generation_kwargs)

        return {
            "request_id": result.request_id,
            "code": result.code,
            "language": result.language,
            "quality_score": result.quality_score,
            "quality_level": result.quality_level.value,
            "test_results": result.test_results,
            "improvements_applied": result.improvements_applied,
            "confidence": result.confidence,
            "generation_time": result.generation_time,
            "iteration": result.iteration,
            "metadata": result.metadata,
            "generation_path": "cog_eng_codegen",
        }
    
    async def discover_trends(self, **kwargs) -> Dict[str, Any]:
        """Discover trending topics."""
        if self._insights is None:
            await self.initialize_insights_path()
        
        if self._insights:
            categories = kwargs.pop("categories", None)
            if categories:
                from nexus.insights.models import TrendCategory

                normalized_categories = []
                invalid_categories = []
                for category in categories:
                    if isinstance(category, TrendCategory):
                        normalized_categories.append(category)
                        continue

                    raw_value = str(category).strip().lower()
                    try:
                        normalized_categories.append(TrendCategory(raw_value))
                    except ValueError:
                        invalid_categories.append(str(category))

                if invalid_categories:
                    valid_options = ", ".join(sorted(category.value for category in TrendCategory))
                    return {
                        "error": (
                            f"Unsupported trend categories: {', '.join(invalid_categories)}. "
                            f"Valid categories: {valid_options}"
                        )
                    }

                return await self._insights.scan(categories=normalized_categories, **kwargs)

            return await self._insights.discover(**kwargs)
        return {"error": "Insights not initialized"}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current platform metrics."""
        if self._metrics:
            if hasattr(self._metrics, "get_all"):
                return self._metrics.get_all()
            if hasattr(self._metrics, "get_dashboard_data"):
                return self._metrics.get_dashboard_data()
            if hasattr(self._metrics, "get_metrics"):
                return {"metrics": self._metrics.get_metrics()}
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
        if self._resource_discovery is None:
            await self.initialize_discovery_path()

        if not self._resource_discovery:
            return {"error": "Discovery not initialized"}

        return await self._resource_discovery.discover_all()

    async def discover_models(self) -> int:
        """
        Discover and self-register new models.

        Returns:
            Number of new models discovered
        """
        if self._model_discovery is None:
            await self._ensure_model_discovery_components()

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
        if self._model_discovery is None:
            await self._ensure_model_discovery_components()

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
        if self._resource_discovery is None:
            await self.initialize_discovery_path()

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
        if self._github_integration is None:
            await self._ensure_github_components()

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
        if self._huggingface_integration is None:
            await self._ensure_huggingface_components()

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
        if self._arxiv_integration is None:
            await self._ensure_arxiv_components()

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
        if self._pypi_integration is None:
            await self._ensure_pypi_components()

        if self._pypi_integration:
            return await self._pypi_integration.search_packages(query, max_results=max_results)
        return []

    async def get_pypi_package(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a PyPI package."""
        if self._pypi_integration is None:
            await self._ensure_pypi_components()

        if self._pypi_integration:
            return await self._pypi_integration.get_package_info(package_name)
        return None

    async def list_ollama_models(self) -> List[Any]:
        """List locally installed Ollama models."""
        if self._ollama_integration is None:
            await self._ensure_ollama_components()

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
        if self._ollama_integration is None:
            await self._ensure_ollama_components()

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
        if self._web_search is None:
            await self._ensure_web_search_components()

        if self._web_search:
            return await self._web_search.search(query, num_results=num_results)
        return []

    async def search_news(
        self,
        query: str,
        num_results: int = 10,
    ) -> List[Any]:
        """Search for news articles."""
        if self._web_search is None:
            await self._ensure_web_search_components()

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

    async def _ensure_llm_component(self) -> None:
        """Initialize direct LLM routing for prompt execution."""
        if self._llm is None:
            try:
                from nexus.core.llm_provider import NexusLLM

                self._llm = NexusLLM()
                self._status["llm"] = True
            except Exception:
                self._status["llm"] = False
        else:
            self._status["llm"] = True

    async def _ensure_ensemble_component(self) -> None:
        """Initialize the ensemble fallback path for prompt execution."""
        if self._ensemble is None:
            try:
                from nexus.core.strategic_ensemble import strategic_ensemble

                self._ensemble = strategic_ensemble
                self._status["ensemble"] = True
            except Exception:
                self._status["ensemble"] = False
        else:
            self._status["ensemble"] = True

    async def _ensure_cognitive_components(self) -> None:
        """Initialize cognitive and research components."""
        if self._consciousness is not None and self._research_agent is not None:
            self._status["cog_eng"] = True
            return
        try:
            from nexus.cog_eng import ConsciousnessCore, AutonomousResearchAgent

            self._consciousness = ConsciousnessCore()
            self._research_agent = AutonomousResearchAgent()
            self._status["cog_eng"] = True
        except Exception:
            self._status["cog_eng"] = False

    async def _ensure_codegen_component(self) -> None:
        """Initialize code generation without booting the full cognitive stack."""
        if self._code_generator is not None:
            self._status["codegen"] = True
            return
        try:
            from nexus.cog_eng.capabilities.self_improving_codegen import SelfImprovingCodeGenerator

            self._code_generator = SelfImprovingCodeGenerator()
            self._status["codegen"] = True
        except Exception:
            self._status["codegen"] = False

    async def _ensure_expert_components(self) -> None:
        """Initialize expert consensus components."""
        if self._experts is not None:
            self._status["experts"] = True
            return
        try:
            from nexus.experts import ConsensusEngine

            self._experts = ConsensusEngine(platform=self)
            self._status["experts"] = True
        except Exception:
            self._status["experts"] = False

    async def _ensure_monitoring_components(self) -> None:
        """Initialize monitoring if enabled."""
        if not self.config.enable_monitoring:
            self._status["observatory"] = False
            return
        if self._metrics is not None:
            self._status["observatory"] = True
            return
        try:
            from nexus.observatory import MetricsCollector

            self._metrics = MetricsCollector()
            self._status["observatory"] = True
        except Exception:
            self._status["observatory"] = False

    async def _ensure_resource_discovery_core(self) -> None:
        """Initialize the shared discovery registry used by discovery integrations."""
        if self._resource_discovery is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import ResourceDiscovery

            self._resource_discovery = ResourceDiscovery()
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_model_discovery_components(self) -> None:
        """Initialize model discovery without unrelated discovery integrations."""
        await self._ensure_resource_discovery_core()
        if self._resource_discovery is None:
            return
        if self._model_discovery is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import ModelDiscoveryEngine

            self._model_discovery = ModelDiscoveryEngine(self._resource_discovery)
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_github_components(self) -> None:
        """Initialize GitHub discovery integration."""
        await self._ensure_resource_discovery_core()
        if self._resource_discovery is None:
            return
        if self._github_integration is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import GitHubIntegration

            self._github_integration = GitHubIntegration(self._resource_discovery)
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_huggingface_components(self) -> None:
        """Initialize HuggingFace discovery integration."""
        await self._ensure_resource_discovery_core()
        if self._resource_discovery is None:
            return
        if self._huggingface_integration is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import HuggingFaceIntegration

            self._huggingface_integration = HuggingFaceIntegration(self._resource_discovery)
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_arxiv_components(self) -> None:
        """Initialize Arxiv discovery integration."""
        await self._ensure_resource_discovery_core()
        if self._resource_discovery is None:
            return
        if self._arxiv_integration is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import ArxivIntegration

            self._arxiv_integration = ArxivIntegration(self._resource_discovery)
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_pypi_components(self) -> None:
        """Initialize PyPI discovery integration."""
        await self._ensure_resource_discovery_core()
        if self._resource_discovery is None:
            return
        if self._pypi_integration is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import PyPIIntegration

            self._pypi_integration = PyPIIntegration(self._resource_discovery)
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_ollama_components(self) -> None:
        """Initialize Ollama discovery/local-model integration."""
        await self._ensure_resource_discovery_core()
        if self._resource_discovery is None:
            return
        if self._ollama_integration is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import OllamaIntegration

            self._ollama_integration = OllamaIntegration(self._resource_discovery)
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_web_search_components(self) -> None:
        """Initialize web search integration."""
        await self._ensure_resource_discovery_core()
        if self._resource_discovery is None:
            return
        if self._web_search is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import WebSearchIntegration

            self._web_search = WebSearchIntegration(self._resource_discovery)
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_local_machine_components(self) -> None:
        """Initialize local-machine integration without broader discovery boot."""
        if self._local_machine is not None:
            self._status["discovery"] = True
            return
        try:
            from nexus.discovery import LocalMachineIntegration

            self._local_machine = LocalMachineIntegration()
            self._status["discovery"] = True
        except Exception:
            self._status["discovery"] = False

    async def _ensure_insights_components(self) -> None:
        """Initialize insights if enabled."""
        if not self.config.enable_insights:
            self._status["insights"] = False
            return
        if self._insights is not None:
            self._status["insights"] = True
            return
        try:
            from nexus.insights import InsightsEngine

            self._insights = InsightsEngine()
            self._status["insights"] = True
        except Exception:
            self._status["insights"] = False

    async def _ensure_discovery_components(self) -> None:
        """Initialize discovery if enabled."""
        if not self.config.enable_discovery:
            self._status["discovery"] = False
            return
        if (
            self._resource_discovery is not None
            and self._model_discovery is not None
            and self._github_integration is not None
            and self._huggingface_integration is not None
            and self._arxiv_integration is not None
            and self._pypi_integration is not None
            and self._ollama_integration is not None
            and self._web_search is not None
            and self._local_machine is not None
        ):
            self._status["discovery"] = True
            return
        try:
            await self._ensure_model_discovery_components()
            await self._ensure_github_components()
            await self._ensure_huggingface_components()
            await self._ensure_arxiv_components()
            await self._ensure_pypi_components()
            await self._ensure_ollama_components()
            await self._ensure_web_search_components()
            await self._ensure_local_machine_components()

            if self.config.auto_discover_on_init:
                await self.discover_resources()
        except Exception:
            self._status["discovery"] = False

    def _query_backend_ready(self) -> bool:
        """Return True when at least one query backend is available."""
        return bool(self._llm or self._ensemble)

    async def get_query_backend_status(
        self,
        model: Optional[str] = None,
        task_type: str = "conversation",
    ) -> Dict[str, bool]:
        """Return truthful readiness for preferred and fallback query paths."""
        if not self._query_initialized:
            await self.initialize_query_path()
        if self._ensemble is None:
            await self._ensure_ensemble_component()

        direct_llm_backend = False
        preferred_query_backend = False
        fallback_query_backend = False

        if self._llm:
            llm_status = await self._llm.get_backend_status(
                task_type=task_type,
                preferred_preset=model or self.config.default_model,
            )
            direct_llm_backend = llm_status["usable"]
            preferred_query_backend = llm_status["preferred_available"]
            fallback_query_backend = llm_status["fallback_available"]

        ensemble_fallback = self._ensemble is not None
        query_backend = direct_llm_backend or ensemble_fallback
        if ensemble_fallback and not preferred_query_backend:
            fallback_query_backend = True

        return {
            "direct_llm_backend": direct_llm_backend,
            "preferred_query_backend": preferred_query_backend,
            "fallback_query_backend": fallback_query_backend,
            "query_backend": query_backend,
        }


# Convenience function
async def get_platform() -> NexusPlatform:
    """Get initialized platform instance."""
    platform = NexusPlatform()
    await platform.initialize()
    return platform
