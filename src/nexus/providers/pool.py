"""
LLM Provider Pool — manages multiple Ollama instances and API connections
for parallel workstream execution.

Supports:
- Multiple Ollama instances on different ports/hosts
- Model-specific routing (code model → code instance)
- Source-based workstream isolation (C-Suite gets flagship, Arclane gets fast)
- Round-robin load balancing within a pool
- Health checking and automatic failover
- Concurrent request limits per instance

Configuration via environment:
    OLLAMA_HOSTS=http://localhost:11434,http://localhost:11435,http://localhost:11436
    OLLAMA_POOL_MODELS=qwen3:30b,codestral:22b,llama3:8b

Or programmatic:
    pool = ProviderPool()
    pool.add_ollama("http://localhost:11434", preferred_model="qwen3:30b", tags=["reasoning"])
    pool.add_ollama("http://localhost:11435", preferred_model="codestral:22b", tags=["code"])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


@dataclass
class ProviderInstance:
    """A single LLM provider instance in the pool."""

    name: str
    url: str
    provider_type: str  # "ollama", "openrouter", "openai", "anthropic"
    preferred_model: Optional[str] = None
    tags: List[str] = field(default_factory=list)  # ["reasoning", "code", "fast", "flagship"]
    max_concurrent: int = 3
    active_requests: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    is_healthy: bool = True
    last_health_check: float = 0.0
    consecutive_failures: int = 0

    @property
    def is_available(self) -> bool:
        return self.is_healthy and self.active_requests < self.max_concurrent

    @property
    def load_factor(self) -> float:
        """0.0 (idle) to 1.0 (at capacity)."""
        if self.max_concurrent <= 0:
            return 1.0
        return self.active_requests / self.max_concurrent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "provider_type": self.provider_type,
            "preferred_model": self.preferred_model,
            "tags": self.tags,
            "max_concurrent": self.max_concurrent,
            "active_requests": self.active_requests,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "is_healthy": self.is_healthy,
            "load_factor": round(self.load_factor, 2),
        }


# Workstream routing rules: source → preferred tags
_DEFAULT_WORKSTREAM_RULES: Dict[str, List[str]] = {
    "csuite:critical": ["flagship", "reasoning"],
    "csuite": ["reasoning", "balanced"],
    "arclane": ["fast", "balanced"],
    "trendscope": ["fast", "balanced"],
    "shopforge": ["fast"],
    "brandguard": ["fast"],
    "taskpilot": ["fast"],
    "sentinel": ["reasoning", "security"],
    "default": ["balanced", "fast"],
}


class ProviderPool:
    """Manages multiple LLM provider instances with intelligent routing."""

    def __init__(self):
        self._instances: List[ProviderInstance] = []
        self._lock = asyncio.Lock()
        self._workstream_rules = dict(_DEFAULT_WORKSTREAM_RULES)

    def add_ollama(
        self,
        url: str,
        *,
        name: Optional[str] = None,
        preferred_model: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_concurrent: int = 3,
    ) -> ProviderInstance:
        """Add an Ollama instance to the pool."""
        instance = ProviderInstance(
            name=name or f"ollama-{len(self._instances)}",
            url=url.rstrip("/"),
            provider_type="ollama",
            preferred_model=preferred_model,
            tags=tags or ["balanced"],
            max_concurrent=max_concurrent,
        )
        self._instances.append(instance)
        logger.info("Pool: added Ollama %s at %s (model=%s, tags=%s)",
                     instance.name, url, preferred_model, tags)
        return instance

    def add_api_provider(
        self,
        url: str,
        provider_type: str,
        *,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_concurrent: int = 10,
    ) -> ProviderInstance:
        """Add an API provider (OpenRouter, OpenAI, Anthropic) to the pool."""
        instance = ProviderInstance(
            name=name or f"{provider_type}-{len(self._instances)}",
            url=url.rstrip("/"),
            provider_type=provider_type,
            tags=tags or ["flagship", "reasoning"],
            max_concurrent=max_concurrent,
        )
        self._instances.append(instance)
        logger.info("Pool: added %s %s at %s", provider_type, instance.name, url)
        return instance

    def configure_workstream(self, source: str, preferred_tags: List[str]) -> None:
        """Set routing preferences for a workstream source."""
        self._workstream_rules[source] = preferred_tags

    def select_instance(
        self,
        *,
        source: str = "default",
        model: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[ProviderInstance]:
        """Select the best available instance for a request.

        Priority:
        1. Explicit model match (if instance has preferred_model matching)
        2. Tag match from workstream rules or explicit tags
        3. Least loaded available instance
        """
        available = [i for i in self._instances if i.is_available]
        if not available:
            return None

        # 1. Explicit model match
        if model:
            for inst in available:
                if inst.preferred_model and model.startswith(inst.preferred_model.split(":")[0]):
                    return inst

        # 2. Tag match from workstream rules
        preferred_tags = tags or self._workstream_rules.get(source, self._workstream_rules["default"])
        tagged = [
            i for i in available
            if any(t in i.tags for t in preferred_tags)
        ]
        if tagged:
            # Return least loaded among tag-matched
            return min(tagged, key=lambda i: i.load_factor)

        # 3. Least loaded
        return min(available, key=lambda i: i.load_factor)

    async def generate(
        self,
        prompt: str,
        *,
        source: str = "default",
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Optional[Dict[str, Any]]:
        """Route a generate request to the best available instance."""
        instance = self.select_instance(source=source, model=model)
        if not instance:
            logger.warning("Pool: no available instances for source=%s", source)
            return None

        instance.active_requests += 1
        t0 = time.perf_counter()

        try:
            if instance.provider_type == "ollama":
                result = await self._generate_ollama(instance, prompt, system_prompt, model, max_tokens)
            else:
                result = await self._generate_api(instance, prompt, system_prompt, model, max_tokens)

            latency = (time.perf_counter() - t0) * 1000
            instance.total_requests += 1
            instance.avg_latency_ms = (
                (instance.avg_latency_ms * (instance.total_requests - 1) + latency)
                / instance.total_requests
            )
            instance.consecutive_failures = 0

            if result:
                result["instance"] = instance.name
                result["latency_ms"] = round(latency, 1)
            return result

        except Exception as e:
            instance.total_errors += 1
            instance.consecutive_failures += 1
            if instance.consecutive_failures >= 3:
                instance.is_healthy = False
                logger.warning("Pool: %s marked unhealthy after %d failures",
                               instance.name, instance.consecutive_failures)
            logger.debug("Pool: %s generate failed: %s", instance.name, e)
            return None

        finally:
            instance.active_requests -= 1

    async def _generate_ollama(
        self, instance: ProviderInstance, prompt: str,
        system_prompt: Optional[str], model: Optional[str], max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        """Generate via Ollama API."""
        import aiohttp

        use_model = model or instance.preferred_model or "qwen3:30b"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{instance.url}/api/chat",
                json={"model": use_model, "messages": messages, "stream": False},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = data.get("message", {}).get("content", "")
                return {
                    "content": content,
                    "model": use_model,
                    "provider": "ollama",
                    "tokens_used": data.get("eval_count", 0),
                }

    async def _generate_api(
        self, instance: ProviderInstance, prompt: str,
        system_prompt: Optional[str], model: Optional[str], max_tokens: int,
    ) -> Optional[Dict[str, Any]]:
        """Generate via OpenAI-compatible API (OpenRouter, etc)."""
        import aiohttp

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {"Content-Type": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{instance.url}/v1/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return {
                    "content": content,
                    "model": model or "unknown",
                    "provider": instance.provider_type,
                    "tokens_used": usage.get("total_tokens", 0),
                }

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all instances."""
        results = {}
        for instance in self._instances:
            try:
                if instance.provider_type == "ollama":
                    import aiohttp
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{instance.url}/api/tags",
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as resp:
                            instance.is_healthy = resp.status == 200
                else:
                    instance.is_healthy = True  # API providers assumed healthy
                instance.last_health_check = time.time()
                instance.consecutive_failures = 0
            except Exception:
                instance.is_healthy = False
            results[instance.name] = instance.is_healthy
        return results

    def get_pool_status(self) -> Dict[str, Any]:
        """Get status of all instances in the pool."""
        return {
            "instances": [i.to_dict() for i in self._instances],
            "total_instances": len(self._instances),
            "healthy_instances": sum(1 for i in self._instances if i.is_healthy),
            "total_requests": sum(i.total_requests for i in self._instances),
            "total_errors": sum(i.total_errors for i in self._instances),
            "active_requests": sum(i.active_requests for i in self._instances),
            "workstream_rules": self._workstream_rules,
        }

    @classmethod
    def from_environment(cls) -> ProviderPool:
        """Create a pool from environment variables.

        OLLAMA_HOSTS: comma-separated Ollama URLs
        OLLAMA_POOL_MODELS: comma-separated models (one per host)
        OLLAMA_POOL_TAGS: comma-separated tag sets (semicolon-separated per host)
        """
        pool = cls()

        hosts = os.environ.get("OLLAMA_HOSTS", os.environ.get("OLLAMA_HOST", ""))
        if not hosts:
            return pool

        host_list = [h.strip() for h in hosts.split(",") if h.strip()]
        model_list = os.environ.get("OLLAMA_POOL_MODELS", "").split(",")
        tag_list = os.environ.get("OLLAMA_POOL_TAGS", "").split(",")

        for i, host in enumerate(host_list):
            model = model_list[i].strip() if i < len(model_list) and model_list[i].strip() else None
            tags = tag_list[i].strip().split(";") if i < len(tag_list) and tag_list[i].strip() else ["balanced"]
            pool.add_ollama(host, preferred_model=model, tags=tags)

        logger.info("Pool: configured %d Ollama instances from environment", len(host_list))
        return pool


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_pool: Optional[ProviderPool] = None


def get_provider_pool() -> ProviderPool:
    global _pool
    if _pool is None:
        _pool = ProviderPool.from_environment()
    return _pool


def reset_provider_pool() -> None:
    global _pool
    _pool = None
