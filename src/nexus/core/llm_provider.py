"""
Nexus LLM Provider - Unified model routing with Ollama primary.

Routes requests to appropriate LLM backends:
- Ollama (local, free) for most tasks
- Anthropic/OpenAI (cloud, paid) for complex reasoning when needed
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TaskComplexity(Enum):
    """Task complexity levels for routing decisions."""
    SIMPLE = "simple"           # Quick responses, routine tasks
    MODERATE = "moderate"       # Standard reasoning
    COMPLEX = "complex"         # Deep analysis, architecture
    CRITICAL = "critical"       # High-stakes decisions


@dataclass
class RoutingConfig:
    """Configuration for intelligent model routing."""
    
    # Task type to complexity mapping
    task_complexity: Dict[str, TaskComplexity] = field(default_factory=lambda: {
        # Simple tasks - use fastest local model
        "greeting": TaskComplexity.SIMPLE,
        "factual_lookup": TaskComplexity.SIMPLE,
        "format_conversion": TaskComplexity.SIMPLE,
        "meeting_summary": TaskComplexity.SIMPLE,
        "email_triage": TaskComplexity.SIMPLE,
        
        # Moderate tasks - use balanced local model
        "conversation": TaskComplexity.MODERATE,
        "content_writing": TaskComplexity.MODERATE,
        "ebook_chapter": TaskComplexity.MODERATE,
        "blog_post": TaskComplexity.MODERATE,
        "trend_analysis": TaskComplexity.MODERATE,
        "research": TaskComplexity.MODERATE,
        
        # Complex tasks - may use cloud when quality matters
        "code_review": TaskComplexity.COMPLEX,
        "architecture_design": TaskComplexity.COMPLEX,
        "complex_reasoning": TaskComplexity.COMPLEX,
        "legal_analysis": TaskComplexity.COMPLEX,
        "financial_analysis": TaskComplexity.COMPLEX,
        
        # Critical tasks - always use best available
        "production_code": TaskComplexity.CRITICAL,
        "security_audit": TaskComplexity.CRITICAL,
    })
    
    # Complexity to model preset mapping
    complexity_models: Dict[TaskComplexity, str] = field(default_factory=lambda: {
        TaskComplexity.SIMPLE: "ollama-qwen3-8b-fast",
        TaskComplexity.MODERATE: "ollama-qwen3-30b",
        TaskComplexity.COMPLEX: "ollama-qwen3-30b",  # Can override to cloud
        TaskComplexity.CRITICAL: "anthropic-sonnet",
    })
    
    # Whether to use cloud for complex tasks
    use_cloud_for_complex: bool = False
    
    # Fallback chain
    fallback_chain: List[str] = field(default_factory=lambda: [
        "ollama-qwen3-30b",
        "ollama-qwen3-8b",
        "anthropic-sonnet",
    ])


class NexusLLM:
    """
    Unified LLM provider for Nexus with intelligent routing.
    
    Features:
    - Automatic routing based on task complexity
    - Fallback chain for reliability
    - Cost tracking
    - Response caching (optional)
    """
    
    def __init__(
        self,
        routing_config: Optional[RoutingConfig] = None,
        enable_cache: bool = True,
        cache_ttl_seconds: int = 3600,
        readiness_cache_ttl_seconds: int = 15,
    ):
        """
        Initialize the Nexus LLM provider.
        
        Args:
            routing_config: Custom routing configuration
            enable_cache: Enable response caching
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.config = routing_config or RoutingConfig()
        self.enable_cache = enable_cache
        self.cache_ttl = cache_ttl_seconds
        self.readiness_cache_ttl = readiness_cache_ttl_seconds
        
        # Lazy-loaded backends
        self._backends: Dict[str, Any] = {}
        
        # Statistics
        self.stats = {
            "requests": 0,
            "cache_hits": 0,
            "fallbacks_used": 0,
            "cost_usd": 0.0,
            "tokens_generated": 0,
        }
        
        # Simple in-memory cache
        self._cache: Dict[str, tuple] = {}
        self._backend_status_cache: Dict[tuple[str, ...], tuple[float, Dict[str, bool]]] = {}
        
        logger.info("NexusLLM initialized")
    
    def _get_backend(self, preset: str):
        """Lazy-load a backend by preset name."""
        if preset not in self._backends:
            try:
                from nexus.blueprints.llm_backend import get_preset_backend
                self._backends[preset] = get_preset_backend(preset)
                logger.info(f"Loaded backend: {preset}")
            except Exception as e:
                logger.error(f"Failed to load backend {preset}: {e}")
                return None
        return self._backends[preset]
    
    def _get_model_for_task(self, task_type: str) -> str:
        """Determine which model to use for a task."""
        complexity = self.config.task_complexity.get(
            task_type, 
            TaskComplexity.MODERATE
        )
        
        model = self.config.complexity_models.get(complexity)
        
        # Override complex tasks to cloud if configured
        if complexity == TaskComplexity.COMPLEX and self.config.use_cloud_for_complex:
            model = "anthropic-sonnet"
        
        return model or "ollama-qwen3-30b"
    
    def _cache_key(self, prompt: str, task_type: str) -> str:
        """Generate cache key for a request."""
        import hashlib
        content = f"{task_type}:{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    async def generate(
        self,
        prompt: str,
        task_type: str = "conversation",
        system_prompt: str = "",
        max_tokens: int = 8000,
        force_model: Optional[str] = None,
        skip_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a response using intelligent routing.
        
        Args:
            prompt: User prompt
            task_type: Type of task (for routing)
            system_prompt: System prompt
            max_tokens: Maximum tokens
            force_model: Force specific model preset
            skip_cache: Skip cache lookup
            
        Returns:
            Response dict with content, metadata, stats
        """
        self.stats["requests"] += 1
        
        # Check cache
        if self.enable_cache and not skip_cache:
            cache_key = self._cache_key(prompt, task_type)
            if cache_key in self._cache:
                cached_response, cached_time = self._cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    self.stats["cache_hits"] += 1
                    return {**cached_response, "cached": True}
        
        # Determine model
        model_preset = force_model or self._get_model_for_task(task_type)
        
        # Try with fallback chain
        last_error = None
        models_tried = []
        
        fallback_chain = [model_preset] + [
            m for m in self.config.fallback_chain if m != model_preset
        ]
        
        for preset in fallback_chain:
            backend = self._get_backend(preset)
            if backend is None:
                continue
            
            models_tried.append(preset)
            
            try:
                response = await backend.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
                
                # Update stats
                self.stats["tokens_generated"] += response.tokens_used
                self.stats["cost_usd"] += backend.estimate_cost(
                    response.input_tokens, 
                    response.output_tokens
                )
                
                if preset != model_preset:
                    self.stats["fallbacks_used"] += 1
                
                result = {
                    "content": response.content,
                    "model": response.model,
                    "provider": response.provider,
                    "tokens_used": response.tokens_used,
                    "duration_seconds": response.duration_seconds,
                    "task_type": task_type,
                    "preset_used": preset,
                    "preferred_preset": model_preset,
                    "actual_preset": preset,
                    "backend_mode": "preferred" if preset == model_preset else "fallback",
                    "fallback_used": preset != model_preset,
                    "models_tried": models_tried,
                    "cached": False,
                }
                
                # Cache successful response
                if self.enable_cache:
                    self._cache[cache_key] = (result, time.time())
                
                return result
                
            except Exception as e:
                last_error = e
                logger.debug("Backend %s failed during fallback chain: %s", preset, e)
                continue
        
        # All backends failed
        raise RuntimeError(f"All backends failed. Last error: {last_error}")
    
    async def check_backends(self) -> Dict[str, bool]:
        """Check availability of all backends."""
        return await self.check_specific_backends(self.config.fallback_chain)

    async def check_specific_backends(self, presets: List[str]) -> Dict[str, bool]:
        """Check availability of a specific preset list."""
        deduped_presets = tuple(dict.fromkeys(presets))
        cached = self._backend_status_cache.get(deduped_presets)
        now = time.monotonic()
        if cached and now - cached[0] < self.readiness_cache_ttl:
            return dict(cached[1])

        results = {}

        for preset in deduped_presets:
            backend = self._get_backend(preset)
            if backend is None:
                results[preset] = False
                continue
            
            if hasattr(backend, 'check_available'):
                try:
                    results[preset] = await backend.check_available()
                except Exception:
                    results[preset] = False
            else:
                # If the backend exposes no readiness hook, treat construction as availability.
                results[preset] = True

        self._backend_status_cache[deduped_presets] = (now, dict(results))
        return results

    async def get_backend_status(
        self,
        task_type: str = "conversation",
        preferred_preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return readiness information for the preferred route and fallbacks."""
        preferred = preferred_preset or self._get_model_for_task(task_type)
        chain = [preferred] + [preset for preset in self.config.fallback_chain if preset != preferred]
        availability = await self.check_specific_backends(chain)
        fallback_available = any(availability.get(preset, False) for preset in chain[1:])
        preferred_available = availability.get(preferred, False)

        return {
            "preferred_preset": preferred,
            "availability": availability,
            "preferred_available": preferred_available,
            "fallback_available": fallback_available,
            "usable": preferred_available or fallback_available,
            "mode": (
                "preferred"
                if preferred_available
                else "fallback"
                if fallback_available
                else "degraded"
            ),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.stats,
            "cache_hit_rate": (
                self.stats["cache_hits"] / self.stats["requests"]
                if self.stats["requests"] > 0 else 0
            ),
            "fallback_rate": (
                self.stats["fallbacks_used"] / self.stats["requests"]
                if self.stats["requests"] > 0 else 0
            ),
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        self._cache.clear()
        logger.info("Cache cleared")

    def invalidate_backend_status_cache(self):
        """Clear cached backend readiness results."""
        self._backend_status_cache.clear()


# Convenience functions
_default_llm: Optional[NexusLLM] = None


def get_llm() -> NexusLLM:
    """Get the default Nexus LLM instance."""
    global _default_llm
    if _default_llm is None:
        _default_llm = NexusLLM()
    return _default_llm


async def quick_generate(
    prompt: str,
    task_type: str = "conversation",
    system_prompt: str = "",
) -> str:
    """Quick generation with defaults."""
    llm = get_llm()
    response = await llm.generate(
        prompt=prompt,
        task_type=task_type,
        system_prompt=system_prompt,
    )
    return response["content"]


# CLI test
if __name__ == "__main__":
    async def test():
        llm = NexusLLM()
        
        print("Checking backends...")
        availability = await llm.check_backends()
        for backend, available in availability.items():
            status = "✓" if available else "✗"
            print(f"  {status} {backend}")
        
        print("\nTesting generation...")
        response = await llm.generate(
            prompt="What is the capital of France?",
            task_type="factual_lookup"
        )
        
        print(f"Model: {response['model']}")
        print(f"Preset: {response['preset_used']}")
        print(f"Tokens: {response['tokens_used']}")
        print(f"Time: {response['duration_seconds']:.1f}s")
        print(f"\nResponse: {response['content'][:500]}")
    
    asyncio.run(test())
