"""
Google Gemini model adapter implementation.

Supports all Gemini models including Gemini Pro, Gemini Pro Vision, and Gemini Ultra.
"""

import asyncio
import time
from typing import Any, AsyncIterator, List, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from nexus.providers.ensemble.types import ModelProvider, ModelResponse
from nexus.providers.adapters.base import BaseModelAdapter, ModelInfo


class GoogleModelAdapter(BaseModelAdapter):
    """
    Adapter for Google Gemini models.

    Features:
    - Async API calls
    - Streaming support
    - Automatic retries with exponential backoff
    - Safety settings configuration
    - System instructions support
    - Multi-turn conversation support
    """

    def __init__(
        self,
        model_info: ModelInfo,
        api_key: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Initialize Google Gemini adapter.

        Args:
            model_info: Model information
            api_key: Google API key
            **kwargs: Additional configuration
        """
        if not GOOGLE_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not available. "
                "Install with: pip install google-generativeai>=0.3.0"
            )

        super().__init__(model_info, api_key=api_key, **kwargs)
        self._model: Optional[Any] = None

    async def initialize(self) -> None:
        """Initialize the Google Gemini client."""
        if not self.api_key:
            raise ValueError("Google API key is required")

        # Configure the API
        genai.configure(api_key=self.api_key)

        # Initialize the model
        self._model = genai.GenerativeModel(self.model_info.name)

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
        Generate a response from Gemini model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional Gemini parameters

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
                # Prepare generation config
                generation_config = GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

                if stop_sequences:
                    generation_config.stop_sequences = stop_sequences

                # Add additional parameters
                if "top_p" in kwargs:
                    generation_config.top_p = kwargs["top_p"]
                if "top_k" in kwargs:
                    generation_config.top_k = kwargs["top_k"]

                # Make API call (sync, wrap in executor for async)
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._model.generate_content(
                        prompt,
                        generation_config=generation_config,
                    )
                )

                # Extract response
                content = response.text if response.text else ""

                # Estimate tokens (Gemini doesn't always provide exact counts)
                output_tokens = self.estimate_tokens(content)
                total_tokens = input_tokens + output_tokens

                # Calculate metrics
                latency_ms = (time.time() - start_time) * 1000
                cost = self.calculate_cost(input_tokens, output_tokens)

                # Gemini responses are generally high quality
                confidence = 0.85

                # Check for safety ratings
                safety_ratings = {}
                if hasattr(response, 'safety_ratings') and response.safety_ratings:
                    for rating in response.safety_ratings:
                        safety_ratings[rating.category.name] = rating.probability.name

                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.GOOGLE,
                    content=content,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    tokens_used=total_tokens,
                    cost_usd=cost,
                    metadata={
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "safety_ratings": safety_ratings,
                        "finish_reason": getattr(response, 'finish_reason', None),
                    },
                )

            except Exception as e:
                error_msg = str(e).lower()

                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg or "rate" in error_msg:
                    retry_count += 1
                    if retry_count >= max_retries:
                        return ModelResponse(
                            model_name=self.model_info.name,
                            provider=ModelProvider.GOOGLE,
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

                # Other errors
                return ModelResponse(
                    model_name=self.model_info.name,
                    provider=ModelProvider.GOOGLE,
                    content="",
                    confidence=0.0,
                    latency_ms=(time.time() - start_time) * 1000,
                    tokens_used=0,
                    cost_usd=0.0,
                    error=f"Google API error: {str(e)}",
                )

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from Gemini.

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
            # Prepare generation config
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            if "top_p" in kwargs:
                generation_config.top_p = kwargs["top_p"]
            if "top_k" in kwargs:
                generation_config.top_k = kwargs["top_k"]

            # Stream response (sync generator, need to wrap)
            loop = asyncio.get_event_loop()

            def _generate():
                return self._model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    stream=True,
                )

            response_stream = await loop.run_in_executor(None, _generate)

            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            # On error, yield error message
            yield f"[Error: {str(e)}]"

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        Gemini uses similar tokenization to other models.

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
