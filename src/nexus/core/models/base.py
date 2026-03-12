"""
Base classes for AI model providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    STUB = "stub"


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    provider: ModelProvider
    weight: float = 0.5
    api_key: Optional[str] = None
    model_id: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class ModelResponse:
    """Response from a model."""
    content: str
    model_name: str
    provider: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if response was successful."""
        return self.error is None


class BaseModel(ABC):
    """
    Abstract base class for AI model providers.
    
    All model providers should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model provider.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.name = config.name
        self.provider = config.provider
        self.weight = config.weight
        logger.info(f"Initialized {self.provider.value} model: {self.name}")
    
    @abstractmethod
    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            ModelResponse object
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    def generate_sync(self, prompt: str) -> ModelResponse:
        """
        Synchronous wrapper for generate method.
        
        Args:
            prompt: Input prompt
            
        Returns:
            ModelResponse object
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate(prompt))
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate the model configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        pass
    
    def calculate_cost(self, tokens_used: int) -> float:
        """
        Calculate the cost for the given number of tokens.
        
        Args:
            tokens_used: Number of tokens used
            
        Returns:
            Cost in USD
        """
        # Override in subclasses with actual pricing
        return 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name}, provider={self.provider.value})"
