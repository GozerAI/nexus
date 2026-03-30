"""
Cohere model adapter implementation.

Supports Cohere Command models including Command, Command Light, and Command R.
"""

import asyncio
import time
from typing import Any, AsyncIterator, List, Optional

try:
    import cohere
    from cohere import AsyncClient
    from cohere.errors import CohereAPIError, CohereConnectionError
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from nexus.providers.ensemble.types import ModelProvider, ModelResponse
from nexus.providers.adapters.base import BaseModelAdapter, ModelInfo


class CohereModelAdapter(BaseModelAdapter):
    """
    Adapter for Cohere models.

    Features:
    - Async API calls
    - Streaming support
    - Automatic retries with exponential backoff
    - Retrieval-augmented generation support
    - Conversation history support
    """

    def __init__(
        self,
        model_info: ModelInfo,
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize Cohere adapter.

        Args:
            model_info: Model information
            api_key: Cohere API key
            **kwargs: Additional configuration
        """
        if not COHERE_AVAILABLE:
            raise ImportError(
                "Cohere library not available. "
                "Install with: pip install cohere>=4.37"
            )

        super().__init__(model_info, api_key=api_key, **kwargs)

    async def initialize(self) -> None:
        """Initialize the Cohere client."""
        if not self.api_key:
            raise ValueError("Cohere API key is required")

        # Initialize async client
        self._client = AsyncClient(api_key=self.api_key)
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
        Generate a response from Cohere model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-5.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional Cohere parameters

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
                    "message": prompt,
                    "model": self.model_info.name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                # Add stop sequences if provided
                if stop_sequences:
                    request_params["stop_sequences"] = stop_sequences

                # Add additional parameters
                if "p" in kwargs:  # Cohere uses 'p' instead of 'top_p'
                    request_params["p"] = kwargs["p"]
                if "k" in kwargs:
                    request_params["k"] = kwargs["k"]
                if "frequency_penalty" in kwargs:
                    request_params["frequency_penalty"] = kwargs["frequency_penalty"]
                if "presence_penalty" in kwargs:
                    request_params["presence_penalty"] = kwargs["presence_penalty"]

                # Make API call
                response = await self._client.chat(**request_params)

                # Extract response
                content = response.text if hasattr(response, 'text') else ""

                # Get token usage (Cohere provides token counts)
                meta = response.meta if hasattr(response, 'meta') else None
                if meta and hasattr(meta, 'billed_units'):
                    input_tokens = meta.billed_units.input_tokens
                    output_tokens = meta.billed_units.output_tokens
                    total_tokens = input_tokens + output_tokens
                else:
                    output_tokens = self.estimate_tokens(content)
                    total_tokens = input_tokens + output_tokens

                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                cost = self.calculate_cost(input_tokens, output_tokens)

                # Cohere models are generally reliable
                confidence = 0.85

                # Get finish reason
                finish_reason = response.finish_reason if hasattr(response, 'finish_reason') else "complete"

                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.COHERE,
                    content=content,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    tokens_used=total_tokens,
                    cost_usd=cost,
                    metadata={
                        "finish_reason": finish_reason,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "generation_id": response.generation_id if hasattr(response, 'generation_id') else None,
                    },
                )

            except CohereAPIError as e:
                error_msg = str(e).lower()

                # Check if it's a rate limit error
                if "429" in error_msg or "rate" in error_msg or "too many" in error_msg:
                    retry_count += 1
                    if retry_count >= max_retries:
                        return ModelResponse(
                            model_name=self.model_info.name,
                            provider=ModelProvider.COHERE,
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
                    provider=ModelProvider.COHERE,
                    content="",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    tokens_used=0,
                    cost_usd=0.0,
                    error=f"Cohere API error: {str(e)}",
                )

            except Exception as e:
                # Unexpected errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.COHERE,
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
        Generate a streaming response from Cohere.

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
                "message": prompt,
                "model": self.model_info.name,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # Add additional parameters
            if "p" in kwargs:
                request_params["p"] = kwargs["p"]

            # Stream response
            stream = self._client.chat_stream(**request_params)

            async for event in stream:
                if hasattr(event, 'text'):
                    yield event.text

        except Exception as e:
            # On error, yield error message
            yield f"[Error: {str(e)}]"

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        Cohere uses similar tokenization to other models.

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
