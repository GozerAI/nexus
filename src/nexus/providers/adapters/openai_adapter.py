"""
OpenAI model adapter implementation.

Supports all OpenAI models including GPT-4, GPT-3.5, and embeddings.
"""

import asyncio
import time
from typing import Any, AsyncIterator, List, Optional

try:
    import openai
    from openai import AsyncOpenAI, OpenAIError, RateLimitError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from nexus.providers.ensemble.types import ModelProvider
from nexus.providers.adapters.base import BaseModelAdapter, ModelInfo, ModelResponse


class OpenAIModelAdapter(BaseModelAdapter):
    """
    Adapter for OpenAI models (GPT-4, GPT-3.5, etc.).

    Features:
    - Async API calls
    - Streaming support
    - Automatic retries with exponential backoff
    - Token counting with tiktoken
    - Cost calculation
    - Rate limit handling
    """

    def __init__(
        self,
        model_info: ModelInfo,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize OpenAI adapter.

        Args:
            model_info: Model information
            api_key: OpenAI API key
            organization: Optional organization ID
            base_url: Optional custom API base URL
            **kwargs: Additional configuration
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. Install with: pip install openai>=1.3.0"
            )

        super().__init__(model_info, api_key=api_key, **kwargs)
        self.organization = organization
        self.base_url = base_url
        self._tokenizer: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Initialize async client
        client_kwargs = {"api_key": self.api_key}

        if self.organization:
            client_kwargs["organization"] = self.organization

        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self._client = AsyncOpenAI(**client_kwargs)

        # Initialize tokenizer for accurate token counting
        try:
            import tiktoken
            model_name = self.model_info.name
            # Map model names to encoding
            if "gpt-4" in model_name:
                encoding_name = "cl100k_base"
            elif "gpt-3.5-turbo" in model_name:
                encoding_name = "cl100k_base"
            else:
                encoding_name = "cl100k_base"  # Default

            self._tokenizer = tiktoken.get_encoding(encoding_name)
        except ImportError:
            # tiktoken not available, will use estimation
            self._tokenizer = None

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
        Generate a response from OpenAI model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional OpenAI parameters (top_p, frequency_penalty, etc.)

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

                if stop_sequences:
                    request_params["stop"] = stop_sequences

                # Add additional parameters
                for key in ["top_p", "frequency_penalty", "presence_penalty", "n"]:
                    if key in kwargs:
                        request_params[key] = kwargs[key]

                # Make API call
                response = await self._client.chat.completions.create(**request_params)

                # Extract response
                content = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason

                # Get token usage
                usage = response.usage
                total_tokens = usage.total_tokens if usage else input_tokens + self.estimate_tokens(content)
                output_tokens = usage.completion_tokens if usage else self.estimate_tokens(content)

                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                cost = self.calculate_cost(input_tokens, output_tokens)

                # Estimate confidence based on finish reason
                confidence = 0.9 if finish_reason == "stop" else 0.7

                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.OPENAI,
                    content=content,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    tokens_used=total_tokens,
                    cost_usd=cost,
                    metadata={
                        "finish_reason": finish_reason,
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
                        provider=ModelProvider.OPENAI,
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

            except OpenAIError as e:
                # Other OpenAI errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.OPENAI,
                    content="",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    tokens_used=0,
                    cost_usd=0.0,
                    error=f"OpenAI API error: {str(e)}",
                )

            except Exception as e:
                # Unexpected errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.OPENAI,
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
        Generate a streaming response from OpenAI.

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
            # Prepare streaming request
            request_params = {
                "model": self.model_info.name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            # Add additional parameters
            for key in ["top_p", "frequency_penalty", "presence_penalty"]:
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

        Uses tiktoken for accurate counting if available,
        otherwise falls back to character-based estimation.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass

        # Fallback to character-based estimation
        # Rough estimate: ~4 characters per token for English
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
