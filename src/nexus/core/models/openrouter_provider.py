"""
OpenRouter model provider implementation.

Routes requests through OpenRouter's unified API, supporting free
and paid models from multiple providers (Qwen, NVIDIA, Meta, etc.).
"""

import os
import time
import logging
from typing import Optional

from nexus.core.models.base import BaseModel, ModelResponse, ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(BaseModel):
    """
    OpenRouter model provider.

    Calls OpenRouter's OpenAI-compatible API for inference.
    Supports free models (Qwen3, Nemotron, etc.) with rate limiting.
    """

    # Minimum delay between requests to avoid 429s on free tier
    _last_request_time: float = 0.0
    _min_request_interval: float = 3.0  # seconds between requests

    def __init__(self, config: ModelConfig):
        """Initialize OpenRouter provider."""
        super().__init__(config)
        self._api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = "https://openrouter.ai/api/v1"
        # model_id can be set via config metadata or derived from name
        self._model_id = getattr(config, "model_id", None) or self._resolve_model_id(config.name)
        logger.info(
            "OpenRouterProvider initialized: %s -> %s",
            config.name, self._model_id,
        )

    @staticmethod
    def _resolve_model_id(name: str) -> str:
        """Map ensemble config names to OpenRouter model IDs."""
        _ID_MAP = {
            "openrouter-qwen3-80b": "qwen/qwen3-next-80b-a3b-instruct:free",
            "openrouter-nemotron-120b": "nvidia/nemotron-3-super-120b-a12b:free",
            "openrouter-qwen3-30b": "qwen/qwen3-30b-a3b:free",
            "openrouter-llama4-scout": "meta-llama/llama-4-scout-17b-16e-instruct:free",
            "openrouter-deepseek-r1": "deepseek/deepseek-r1-0528:free",
        }
        if name in _ID_MAP:
            return _ID_MAP[name]
        # Strip prefix and try as-is
        if name.startswith("openrouter-"):
            return name[len("openrouter-"):]
        return name

    def validate_config(self) -> bool:
        """Validate OpenRouter configuration."""
        return bool(self._api_key)

    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a response using OpenRouter.

        Includes rate limiting to stay within free tier limits.
        """
        import asyncio
        import httpx

        # Rate limiting: wait if too soon since last request
        now = time.time()
        elapsed = now - OpenRouterProvider._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        OpenRouterProvider._last_request_time = time.time()

        start_time = time.time()

        if not self._api_key:
            return ModelResponse(
                content="",
                model_name=self.name,
                provider="openrouter",
                error="OPENROUTER_API_KEY not configured",
            )

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "HTTP-Referer": "https://gozerai.com",
                        "X-Title": "GozerAI Nexus",
                    },
                    json={
                        "model": self._model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    },
                )

                if response.status_code == 429:
                    # Rate limited — back off
                    OpenRouterProvider._min_request_interval = min(
                        OpenRouterProvider._min_request_interval * 1.5, 15.0
                    )
                    latency = (time.time() - start_time) * 1000
                    logger.warning(
                        "OpenRouter rate limited. Backing off to %.1fs interval",
                        OpenRouterProvider._min_request_interval,
                    )
                    return ModelResponse(
                        content="",
                        model_name=self.name,
                        provider="openrouter",
                        latency_ms=latency,
                        error="Rate limited (429). Backed off.",
                    )

                response.raise_for_status()
                data = response.json()

            latency = (time.time() - start_time) * 1000
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            # Successful request — gradually reduce backoff
            OpenRouterProvider._min_request_interval = max(
                OpenRouterProvider._min_request_interval * 0.9, 2.0
            )

            logger.info(
                "OpenRouter response (%s): tokens=%d, latency=%.0fms",
                self._model_id, tokens_used, latency,
            )

            return ModelResponse(
                content=content,
                model_name=self.name,
                provider="openrouter",
                tokens_used=tokens_used,
                latency_ms=latency,
                cost=0.0,  # Free tier
                metadata={
                    "model_id": self._model_id,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                },
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error("OpenRouter error (%s): %s", self._model_id, e)
            return ModelResponse(
                content="",
                model_name=self.name,
                provider="openrouter",
                latency_ms=latency,
                error=str(e),
            )
