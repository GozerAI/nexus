"""
Together AI model adapter implementation.

Supports various open-source models via Together AI platform.
"""

import asyncio
import time
from typing import Any, AsyncIterator, List, Optional

try:
    from together import AsyncTogether
    from together.error import RateLimitError, TogetherException
    TOGETHER_AVAILABLE = True
except ImportError:
    TOGETHER_AVAILABLE = False

from nexus.providers.ensemble.types import ModelProvider, ModelResponse
from nexus.providers.adapters.base import BaseModelAdapter, ModelInfo


class TogetherModelAdapter(BaseModelAdapter):
    """
    Adapter for Together AI models.

    Features:
    - Async API calls
    - Streaming support
    - Automatic retries with exponential backoff
    - Access to many open-source models
    - Cost-effective inference
    """

    def __init__(
        self,
        model_info: ModelInfo,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize Together AI adapter.

        Args:
            model_info: Model information
            api_key: Together AI API key
            base_url: Optional custom base URL
            **kwargs: Additional configuration
        """
        if not TOGETHER_AVAILABLE:
            raise ImportError(
                "Together AI library not available. "
                "Install with: pip install together>=0.2.0"
            )

        super().__init__(model_info, api_key=api_key, **kwargs)
        self.base_url = base_url

    async def initialize(self) -> None:
        """Initialize the Together AI client."""
        if not self.api_key:
            raise ValueError("Together AI API key is required")

        # Initialize async client
        client_kwargs = {"api_key": self.api_key}

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self._client = AsyncTogether(**client_kwargs)
        self._initialized = True

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a response from Together AI model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional Together AI parameters

        Returns:
            Model response with content and metadata
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        input_tokens = self.estimate_tokens(prompt)

        # Retry logic for rate limits
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                # Prepare request parameters
                request_params = {
                    "model": self.model_info.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Add stop sequences if provided
                if stop_sequences:
                    request_params["stop"] = stop_sequences

                # Add additional parameters
                for key in ["top_p", "top_k", "repetition_penalty"]:
                    if key in kwargs:
                        request_params[key] = kwargs[key]

                # Make API call
                response = await self._client.chat.completions.create(**request_params)

                # Extract response
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason

                # Get token usage
                usage = response.usage if hasattr(response, 'usage') else None
                if usage:
                    prompt_tokens = usage.prompt_tokens
                    completion_tokens = usage.completion_tokens
                    total_tokens = usage.total_tokens
                else:
                    prompt_tokens = input_tokens
                    completion_tokens = self.estimate_tokens(content)
                    total_tokens = prompt_tokens + completion_tokens

                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                cost = self.calculate_cost(prompt_tokens, completion_tokens)

                # Estimate confidence based on finish reason
                confidence = 0.85 if finish_reason == "stop" or finish_reason == "eos" else 0.7

                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.TOGETHER,
                    content=content,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    tokens_used=total_tokens,
                    cost_usd=cost,
                    metadata={
                        "finish_reason": finish_reason,
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "model_id": response.model if hasattr(response, 'model') else self.model_info.name,
                    },
                )

            except RateLimitError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    return ModelResponse(
                        model_name=self.model_info.name,
                        provider=ModelProvider.TOGETHER,
                        content="",
                        confidence=0.0,
                        latency_ms=(time.time() - start_time) * 1000,
                        tokens_used=0,
                        cost_usd=0.0,
                        error=f"Rate limit exceeded after {max_retries} retries: {str(e)}",
                    )

                # Wait with exponential backoff
                wait_time = 2 ** retry_count
                await asyncio.sleep(wait_time)

            except TogetherException as e:
                # Other Together AI errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.TOGETHER,
                    content="",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    tokens_used=0,
                    cost_usd=0.0,
                    error=f"Together AI error: {str(e)}",
                )

            except Exception as e:
                # Unexpected errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.TOGETHER,
                    content="",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    tokens_used=0,
                    cost_usd=0.0,
                    error=f"Unexpected error: {str(e)}",
                )

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from Together AI.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Prepare request parameters
            request_params = {
                "model": self.model_info.name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            # Add additional parameters
            for key in ["top_p", "top_k"]:
                if key in kwargs:
                    request_params[key] = kwargs[key]

            # Stream response
            stream = await self._client.chat.completions.create(**request_params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            # On error, yield error message
            yield f"[Error: {str(e)}]"

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return max(1, len(text) // 4)

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        return self.model_info.calculate_cost(input_tokens, output_tokens)

    async def health_check(self) -> bool:
        """
        Check if the model is accessible.

        Returns:
            True if model is healthy, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Try minimal generation
            response = await self.generate("test", max_tokens=5)
            return response.error is None

        except Exception:
            return False
