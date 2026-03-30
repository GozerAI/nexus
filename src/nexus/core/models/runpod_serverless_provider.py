"""
RunPod Serverless model provider.

Routes inference requests to RunPod serverless endpoints. Each endpoint
hosts a specific csuite fine-tuned model that auto-scales from 0 workers.

API: POST https://api.runpod.io/v2/{endpoint_id}/runsync
"""

import os
import time
import logging
from typing import Optional

from nexus.core.models.base import BaseModel, ModelResponse, ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


class RunPodServerlessProvider(BaseModel):
    """
    RunPod Serverless model provider.

    Each csuite model has its own serverless endpoint that auto-scales.
    Cold start ~30s, warm requests ~10-30s depending on generation length.
    """

    # Map ensemble config names to RunPod endpoint IDs
    _ENDPOINT_MAP = {
        "runpod-csuite-tech-core": "7fz4d0ixkx10xl",
        "runpod-csuite-security-compliance": "72mg4k35hy6qoc",
        "runpod-csuite-revenue-finance": "vlmwywraf7b2ao",
        "runpod-csuite-product-strategy": "2wenoso8a0x9ji",
        "runpod-csuite-data-research": "864bh01k2ytwr7",  # Was csuite-merged, now data-research
    }

    def __init__(self, config: ModelConfig):
        """Initialize RunPod Serverless provider."""
        super().__init__(config)
        self._api_key = os.environ.get("RUNPOD_API_KEY", "")
        self._endpoint_id = self._ENDPOINT_MAP.get(config.name, "")
        self._model_tag = config.name.replace("runpod-", "").replace("-", "_")
        if not self._endpoint_id:
            logger.warning("No serverless endpoint mapped for %s", config.name)
        else:
            logger.info(
                "RunPodServerless initialized: %s -> endpoint %s",
                config.name, self._endpoint_id,
            )

    def validate_config(self) -> bool:
        """Validate RunPod Serverless configuration."""
        return bool(self._api_key and self._endpoint_id)

    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a response using RunPod Serverless.

        Uses the /runsync endpoint for synchronous inference (waits for result).
        """
        import httpx

        if not self._api_key or not self._endpoint_id:
            return ModelResponse(
                content="",
                model_name=self.name,
                provider="runpod_serverless",
                error="RUNPOD_API_KEY or endpoint ID not configured",
            )

        start_time = time.time()
        url = f"https://api.runpod.ai/v2/{self._endpoint_id}/runsync"

        # RunPod serverless vLLM worker uses simple prompt format
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=180) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            latency = (time.time() - start_time) * 1000
            status = data.get("status", "")

            if status == "COMPLETED":
                output = data.get("output", {})
                # vLLM returns various formats — handle all
                if isinstance(output, dict):
                    # Try common output keys
                    content = (
                        output.get("text", "")
                        or output.get("response", "")
                        or output.get("generated_text", "")
                        or output.get("output", "")
                    )
                    # OpenAI format fallback
                    if not content:
                        choices = output.get("choices", [])
                        if choices:
                            choice = choices[0]
                            content = (
                                choice.get("message", {}).get("content", "")
                                or choice.get("text", "")
                            )
                    if not content:
                        content = str(output)
                    usage = output.get("usage", {})
                    tokens_used = usage.get("total_tokens", 0)
                elif isinstance(output, str):
                    content = output
                    tokens_used = 0
                elif isinstance(output, list) and output:
                    # Some vLLM workers return a list of outputs
                    content = str(output[0]) if output else ""
                    tokens_used = 0
                else:
                    content = str(output)
                    tokens_used = 0

                logger.info(
                    "RunPod Serverless response (%s): tokens=%d, latency=%.0fms",
                    self._endpoint_id, tokens_used, latency,
                )

                return ModelResponse(
                    content=content,
                    model_name=self.name,
                    provider="runpod_serverless",
                    tokens_used=tokens_used,
                    latency_ms=latency,
                    cost=0.0,
                    metadata={
                        "endpoint_id": self._endpoint_id,
                        "runpod_id": data.get("id", ""),
                        "status": status,
                    },
                )
            elif status in ("IN_QUEUE", "IN_PROGRESS"):
                # Cold start — poll /status until complete
                import asyncio
                job_id = data.get("id", "")
                if not job_id:
                    return ModelResponse(
                        content="", model_name=self.name, provider="runpod_serverless",
                        latency_ms=(time.time() - start_time) * 1000,
                        error="IN_QUEUE but no job ID returned",
                    )
                status_url = f"https://api.runpod.ai/v2/{self._endpoint_id}/status/{job_id}"
                for _ in range(30):  # Poll up to 150s (30 * 5s)
                    await asyncio.sleep(5)
                    async with httpx.AsyncClient(timeout=10) as poll_client:
                        poll_resp = await poll_client.get(
                            status_url,
                            headers={"Authorization": f"Bearer {self._api_key}"},
                        )
                        poll_data = poll_resp.json()
                        poll_status = poll_data.get("status", "")
                        if poll_status == "COMPLETED":
                            data = poll_data
                            status = "COMPLETED"
                            # Re-parse output
                            output = data.get("output", {})
                            if isinstance(output, dict):
                                choices = output.get("choices", [])
                                if choices:
                                    content = choices[0].get("message", {}).get("content", "")
                                else:
                                    content = output.get("text", str(output))
                                tokens_used = output.get("usage", {}).get("total_tokens", 0)
                            else:
                                content = str(output) if output else ""
                                tokens_used = 0
                            latency = (time.time() - start_time) * 1000
                            logger.info(
                                "RunPod Serverless response (%s): tokens=%d, latency=%.0fms (cold start)",
                                self._endpoint_id, tokens_used, latency,
                            )
                            return ModelResponse(
                                content=content, model_name=self.name,
                                provider="runpod_serverless", tokens_used=tokens_used,
                                latency_ms=latency, cost=0.0,
                                metadata={"endpoint_id": self._endpoint_id, "cold_start": True},
                            )
                        elif poll_status == "FAILED":
                            return ModelResponse(
                                content="", model_name=self.name,
                                provider="runpod_serverless",
                                latency_ms=(time.time() - start_time) * 1000,
                                error=str(poll_data.get("error", "Worker failed")),
                            )
                # Timed out waiting
                return ModelResponse(
                    content="", model_name=self.name, provider="runpod_serverless",
                    latency_ms=(time.time() - start_time) * 1000,
                    error="Timed out waiting for cold start (150s)",
                )
            else:
                latency = (time.time() - start_time) * 1000
                error_msg = data.get("error", f"Unexpected status: {status}")
                return ModelResponse(
                    content="",
                    model_name=self.name,
                    provider="runpod_serverless",
                    latency_ms=latency,
                    error=str(error_msg),
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error("RunPod Serverless error (%s): %s", self._endpoint_id, e)
            return ModelResponse(
                content="",
                model_name=self.name,
                provider="runpod_serverless",
                latency_ms=latency,
                error=str(e),
            )
