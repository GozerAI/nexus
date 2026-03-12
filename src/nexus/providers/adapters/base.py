"""
Base model adapter interface for all AI providers.

This module defines the abstract base class that all model adapters must implement.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from nexus.providers.ensemble.types import ModelProvider, ModelResponse


class ModelCapability(str, Enum):
    """Model capabilities."""

    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    VISION = "vision"
    AUDIO = "audio"
    MULTILINGUAL = "multilingual"
    FUNCTION_CALLING = "function_calling"
    EMBEDDINGS = "embeddings"
    LONG_CONTEXT = "long_context"
    STREAMING = "streaming"


class ModelSize(str, Enum):
    """Model size categories."""

    TINY = "tiny"  # < 1B parameters
    SMALL = "small"  # 1-7B parameters
    MEDIUM = "medium"  # 7-30B parameters
    LARGE = "large"  # 30-70B parameters
    XLARGE = "xlarge"  # 70B+ parameters
    FLAGSHIP = "flagship"  # Top-tier models


@dataclass
class ModelInfo:
    """
    Information about a model.

    Attributes:
        name: Model identifier
        display_name: Human-readable model name
        provider: Model provider
        size: Model size category
        context_window: Maximum context length in tokens
        max_output_tokens: Maximum output tokens
        capabilities: List of model capabilities
        cost_per_1k_input: Cost per 1K input tokens (USD)
        cost_per_1k_output: Cost per 1K output tokens (USD)
        supported: Whether model is fully implemented
        description: Model description
        use_cases: List of recommended use cases
        requires_api_key: Whether API key is required
        supports_streaming: Whether streaming is supported
        release_date: Model release date
        documentation_url: Link to model documentation
        notes: Additional notes for implementation
    """

    name: str
    display_name: str
    provider: ModelProvider
    size: ModelSize
    context_window: int
    max_output_tokens: int
    capabilities: List[ModelCapability]
    cost_per_1k_input: float
    cost_per_1k_output: float
    supported: bool = False
    description: str = ""
    use_cases: List[str] = None
    requires_api_key: bool = True
    supports_streaming: bool = True
    release_date: Optional[str] = None
    documentation_url: Optional[str] = None
    notes: str = ""

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.use_cases is None:
            self.use_cases = []

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1000) * self.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.cost_per_1k_output
        return input_cost + output_cost


class BaseModelAdapter(ABC):
    """
    Abstract base class for all model adapters.

    All model adapters must implement this interface to be compatible
    with the unified ensemble system.
    """

    def __init__(self, model_info: ModelInfo, api_key: Optional[str] = None, **kwargs: Any):
        """
        Initialize the model adapter.

        Args:
            model_info: Model information
            api_key: API key for the provider (if required)
            **kwargs: Additional provider-specific configuration
        """
        self.model_info = model_info
        self.api_key = api_key
        self.config = kwargs
        self._initialized = False
        self._client: Optional[Any] = None

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the model adapter.

        This method should:
        - Validate API keys
        - Initialize the client library
        - Perform any necessary setup
        - Set self._initialized = True
        """
        pass

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop_sequences: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Generate a response from the model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Optional stop sequences
            **kwargs: Additional model-specific parameters

        Returns:
            Model response with content, confidence, and metadata

        Raises:
            RuntimeError: If model is not initialized
            ValueError: If parameters are invalid
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response from the model.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated

        Raises:
            NotImplementedError: If streaming not supported
        """
        pass

    async def health_check(self) -> bool:
        """
        Check if the model is accessible and healthy.

        Returns:
            True if model is healthy, False otherwise
        """
        try:
            if not self._initialized:
                await self.initialize()

            # Try a minimal generation
            response = await self.generate("test", max_tokens=5)
            return response.error is None
        except Exception:
            return False

    def get_info(self) -> ModelInfo:
        """
        Get model information.

        Returns:
            Model information object
        """
        return self.model_info

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

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate number of tokens in text.

        This is a rough estimate using character count.
        Subclasses should override with actual tokenization.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4

    async def __aenter__(self) -> "BaseModelAdapter":
        """Async context manager entry."""
        if not self._initialized:
            await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        # Cleanup if needed
        pass


class StubModelAdapter(BaseModelAdapter):
    """
    Stub implementation for models that aren't fully integrated yet.

    This adapter returns placeholder responses and allows the system
    to work with model configurations before full implementation.
    """

    async def initialize(self) -> None:
        """Initialize stub adapter (no-op)."""
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
        Generate a stub response.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            stop_sequences: Stop sequences
            **kwargs: Additional parameters

        Returns:
            Stub model response
        """
        # Simulate some latency
        await asyncio.sleep(0.1)

        # Generate stub response
        estimated_tokens = self.estimate_tokens(prompt)
        output_tokens = min(100, max_tokens)

        content = (
            f"[STUB RESPONSE from {self.model_info.display_name}]\n"
            f"This is a placeholder response. The model adapter needs to be fully implemented.\n"
            f"Prompt: {prompt[:100]}..."
        )

        return ModelResponse(
            model_name=self.model_info.name,
            provider=self.model_info.provider,
            content=content,
            confidence=0.5,
            latency_ms=100.0,
            tokens_used=estimated_tokens + output_tokens,
            cost_usd=self.calculate_cost(estimated_tokens, output_tokens),
            metadata={
                "stub": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "note": self.model_info.notes,
            },
        )

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate streaming stub response.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        response_text = f"[STUB STREAM from {self.model_info.display_name}] "
        for char in response_text:
            await asyncio.sleep(0.01)
            yield char
