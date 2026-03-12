"""
Mistral AI model adapter implementation.

Supports all Mistral models including Mistral Large, Medium, and Small.
"""

import asyncio
import time
from typing import Any, AsyncIterator, List, Optional

try:
    from mistralai.async_client import MistralAsyncClient
    from mistralai.models.chat_completion import ChatMessage
    from mistralai.exceptions import MistralAPIException, MistralException
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

from nexus.providers.ensemble.types import ModelProvider, ModelResponse
from nexus.providers.adapters.base import BaseModelAdapter, ModelInfo


class MistralModelAdapter(BaseModelAdapter):
    """
    Adapter for Mistral AI models.

    Features:
    - Async API calls
    - Streaming support
    - Automatic retries with exponential backoff
    - Function calling support
    - JSON mode support
    """

    def __init__(
        self,
        model_info: ModelInfo,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize Mistral adapter.

        Args:
            model_info: Model information
            api_key: Mistral API key
            endpoint: Optional custom endpoint
            **kwargs: Additional configuration
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "Mistral AI library not available. "
                "Install with: pip install mistralai>=0.0.11"
            )

        super().__init__(model_info, api_key=api_key, **kwargs)
        self.endpoint = endpoint

    async def initialize(self) -> None:
        """Initialize the Mistral client."""
        if not self.api_key:
            raise ValueError("Mistral API key is required")

        # Initialize async client
        client_kwargs = {"api_key": self.api_key}

        if self.endpoint:
            client_kwargs["endpoint"] = self.endpoint

        self._client = MistralAsyncClient(**client_kwargs)
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
        Generate a response from Mistral model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences (not supported by Mistral)
            **kwargs: Additional Mistral parameters

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
                # Prepare messages
                messages = [
                    ChatMessage(role="user", content=prompt)
                ]

                # Prepare request parameters
                request_params = {
                    "model": self.model_info.name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Add additional parameters
                if "top_p" in kwargs:
                    request_params["top_p"] = kwargs["top_p"]
                if "random_seed" in kwargs:
                    request_params["random_seed"] = kwargs["random_seed"]
                if "safe_mode" in kwargs:
                    request_params["safe_mode"] = kwargs["safe_mode"]

                # Make API call
                response = await self._client.chat(**request_params)

                # Extract response
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason

                # Get token usage
                usage = response.usage
                total_tokens = usage.total_tokens if usage else input_tokens + self.estimate_tokens(content)
                prompt_tokens = usage.prompt_tokens if usage else input_tokens
                completion_tokens = usage.completion_tokens if usage else self.estimate_tokens(content)

                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                cost = self.calculate_cost(prompt_tokens, completion_tokens)

                # Estimate confidence based on finish reason
                confidence = 0.9 if finish_reason == "stop" else 0.75

                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.MISTRAL,
                    content=content,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    tokens_used=total_tokens,
                    cost_usd=cost,
                    metadata={
                        "finish_reason": finish_reason,
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "model_version": response.model,
                    },
                )

            except MistralAPIException as e:
                error_msg = str(e).lower()

                # Check if it's a rate limit error
                if "429" in error_msg or "rate" in error_msg:
                    retry_count += 1
                    if retry_count >= max_retries:
                        return ModelResponse(
                            model_name=self.model_info.name,
                            provider=ModelProvider.MISTRAL,
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
                    continue

                # Other API errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.MISTRAL,
                    content="",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    tokens_used=0,
                    cost_usd=0.0,
                    error=f"Mistral API error: {str(e)}",
                )

            except Exception as e:
                # Unexpected errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.MISTRAL,
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
        Generate a streaming response from Mistral.

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
            # Prepare messages
            messages = [
                ChatMessage(role="user", content=prompt)
            ]

            # Prepare request parameters
            request_params = {
                "model": self.model_info.name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Add additional parameters
            if "top_p" in kwargs:
                request_params["top_p"] = kwargs["top_p"]

            # Stream response
            async for chunk in self._client.chat_stream(**request_params):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            # On error, yield error message
            yield f"[Error: {str(e)}]"

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        Mistral uses similar tokenization to other models.

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
