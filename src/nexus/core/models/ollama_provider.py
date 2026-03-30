"""
Ollama model provider implementation.

Supports local and remote Ollama instances (including RunPod deployments).
"""

import os
import time
import logging
from typing import Optional

from nexus.core.models.base import BaseModel, ModelResponse, ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseModel):
    """
    Ollama model provider.

    Connects to Ollama API for inference. Supports both local
    and remote instances (e.g., RunPod GPU pods via Tailscale).
    """

    # Route models to specific Ollama instances (pods).
    # Pod 1 (dedicated tech_core): OLLAMA_POD1_HOST or localhost:11435
    # Pod 2 (remaining models): OLLAMA_POD2_HOST or localhost:11436
    # Fallback: OLLAMA_HOST or localhost:11434
    _POD1_MODELS = {"runpod-csuite-governance"}

    def __init__(self, config: ModelConfig):
        """Initialize Ollama provider."""
        super().__init__(config)

        # Route to the correct pod based on model name
        default_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        pod1_host = os.environ.get("OLLAMA_POD1_HOST", "http://localhost:11435").rstrip("/")
        pod2_host = os.environ.get("OLLAMA_POD2_HOST", "http://localhost:11436").rstrip("/")

        if config.name in self._POD1_MODELS:
            self._base_url = pod1_host
        elif config.name.startswith("runpod-csuite-"):
            self._base_url = pod2_host
        else:
            self._base_url = default_host

        # Map ensemble model names to actual Ollama model tags
        self._model_tag = self._resolve_model_tag(config.name)
        self._session = None
        logger.info(
            f"OllamaProvider initialized: {config.name} -> {self._model_tag} "
            f"at {self._base_url}"
        )

    # Explicit mapping from ensemble config names to Ollama model tags.
    # Models not physically on a pod fall back to csuite-operations (same Qwen2.5-32B base).
    _POD2_AVAILABLE = {
        "csuite-security_compliance", "csuite-operations",
        "csuite-operations_coordination", "csuite-personality",
    }
    _POD2_FALLBACK = "csuite-operations"

    _TAG_MAP = {
        "runpod-csuite-merged": "csuite-merged",
        "runpod-csuite-tech-core": "csuite-tech_core",
        "runpod-csuite-security-compliance": "csuite-security_compliance",
        "runpod-csuite-revenue-finance": "csuite-revenue_finance",
        "runpod-csuite-product-strategy": "csuite-product_strategy",
        "runpod-csuite-data-research": "csuite-data_research",
        "runpod-csuite-operations-coordination": "csuite-operations_coordination",
        "runpod-csuite-governance": "csuite-governance",
        "runpod-csuite-operations": "csuite-operations",
        "runpod-csuite-technical": "csuite-technical",
        "runpod-csuite-personality": "csuite-personality",
    }

    @classmethod
    def _resolve_model_tag(cls, name: str) -> str:
        """Map ensemble config names to Ollama model tags.

        For models not physically on Pod 2, falls back to csuite-operations
        (same Qwen2.5-32B base, different LoRA but close enough for general use).
        """
        if name in cls._TAG_MAP:
            tag = cls._TAG_MAP[name]
            # If this model isn't on Pod 2 and we're routing to Pod 2, use fallback
            if name not in cls._POD1_MODELS and tag not in cls._POD2_AVAILABLE:
                logger.info("Model %s not on pod, falling back to %s", tag, cls._POD2_FALLBACK)
                return cls._POD2_FALLBACK + ":latest"
            return tag + ":latest"
        # Strip provider prefix and use as-is
        tag = name
        for prefix in ("runpod-", "ollama-"):
            if tag.startswith(prefix):
                tag = tag[len(prefix):]
        if ":" not in tag:
            tag += ":latest"
        return tag

    def validate_config(self) -> bool:
        """Validate Ollama configuration."""
        return True

    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a response using Ollama.

        Args:
            prompt: Input prompt

        Returns:
            ModelResponse object
        """
        import httpx

        start_time = time.time()
        url = f"{self._base_url}/api/chat"

        payload = {
            "model": self._model_tag,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }

        try:
            # RunPod model swaps take ~45s; use generous timeout
            timeout = max(self.config.timeout, 120)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

            latency = (time.time() - start_time) * 1000
            content = data.get("message", {}).get("content", "")
            eval_count = data.get("eval_count", 0)
            prompt_count = data.get("prompt_eval_count", 0)
            tokens_used = eval_count + prompt_count

            logger.info(
                f"Ollama response ({self._model_tag}): tokens={tokens_used}, "
                f"latency={latency:.0f}ms"
            )

            return ModelResponse(
                content=content,
                model_name=self.name,
                provider="ollama",
                tokens_used=tokens_used,
                latency_ms=latency,
                cost=0.0,
                metadata={
                    "model_tag": self._model_tag,
                    "ollama_host": self._base_url,
                    "eval_count": eval_count,
                    "prompt_eval_count": prompt_count,
                    "total_duration_ns": data.get("total_duration", 0),
                },
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Ollama error ({self._model_tag}): {e}")

            return ModelResponse(
                content="",
                model_name=self.name,
                provider="ollama",
                latency_ms=latency,
                error=str(e),
            )
