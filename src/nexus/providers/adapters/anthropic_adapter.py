"""
Anthropic Claude model adapter implementation.

Supports all Claude models including Claude 3 Opus, Sonnet, and Haiku.
"""

import asyncio
import time
from typing import Any, AsyncIterator, List, Optional

try:
    import anthropic
    from anthropic import AsyncAnthropic, APIError, RateLimitError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from nexus.providers.ensemble.types import ModelProvider
from nexus.providers.adapters.base import BaseModelAdapter, ModelInfo, ModelResponse


class AnthropicModelAdapter(BaseModelAdapter):
    """
    Adapter for Anthropic Claude models.

    Features:
    - Async API calls
    - Streaming support
    - Automatic retries with exponential backoff
    - 200K context window support
    - Vision support for Claude 3
    - System prompt support
    """

    def __init__(
        self,
        model_info: ModelInfo,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize Anthropic adapter.

        Args:
            model_info: Model information
            api_key: Anthropic API key
            base_url: Optional custom API base URL
            **kwargs: Additional configuration
        """
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic library not available. Install with: pip install anthropic>=0.7.0"
            )

        super().__init__(model_info, api_key=api_key, **kwargs)
        self.base_url = base_url

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

        # Initialize async client
        client_kwargs = {"api_key": self.api_key}

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self._client = AsyncAnthropic(**client_kwargs)
        self._initialized = True

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a response from Claude model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            system_prompt: Optional system prompt
            **kwargs: Additional Anthropic parameters

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

                # Add system prompt if provided
                if system_prompt:
                    request_params["system"] = system_prompt

                # Add stop sequences
                if stop_sequences:
                    request_params["stop_sequences"] = stop_sequences

                # Add additional parameters
                for key in ["top_p", "top_k"]:
                    if key in kwargs:
                        request_params[key] = kwargs[key]

                # Make API call
                response = await self._client.messages.create(**request_params)

                # Extract response (Claude returns list of content blocks)
                content_blocks = response.content
                content = ""
                for block in content_blocks:
                    if hasattr(block, 'text'):
                        content += block.text

                # Get stop reason
                stop_reason = response.stop_reason

                # Get token usage (Anthropic provides exact counts)
                usage = response.usage
                input_tokens = usage.input_tokens
                output_tokens = usage.output_tokens
                total_tokens = input_tokens + output_tokens

                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                cost = self.calculate_cost(input_tokens, output_tokens)

                # Estimate confidence based on stop reason
                confidence = 0.9 if stop_reason == "end_turn" else 0.75

                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.ANTHROPIC,
                    content=content,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    tokens_used=total_tokens,
                    cost_usd=cost,
                    metadata={
                        "stop_reason": stop_reason,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "model_version": response.model,
                    },
                )

            except RateLimitError as e:
                retry_count += 1
                if retry_count >= max_retries:
                    # Return error response
                    return ModelResponse(
                        model_name=self.model_info.name,
                        provider=ModelProvider.ANTHROPIC,
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

            except APIError as e:
                # Other Anthropic API errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.ANTHROPIC,
                    content="",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    tokens_used=0,
                    cost_usd=0.0,
                    error=f"Anthropic API error: {str(e)}",
                )

            except Exception as e:
                # Unexpected errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.ANTHROPIC,
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
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from Claude.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            system_prompt: Optional system prompt
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Prepare streaming request
            request_params = {
                "model": self.model_info.name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt

            # Add additional parameters
            for key in ["top_p", "top_k"]:
                if key in kwargs:
                    request_params[key] = kwargs[key]

            # Stream response
            async with self._client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text

        except Exception as e:
            # On error, yield error message
            yield f"[Error: {str(e)}]"

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        Anthropic provides exact token counts in responses,
        but we need estimates for input.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Claude uses similar tokenization to GPT models
        # Rough estimate: ~4 characters per token
        return max(1, len(text) // 4)

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
