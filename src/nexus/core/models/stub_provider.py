"""
Stub model provider for testing and development.
"""

import time
import logging
import asyncio
from typing import Optional

from nexus.core.models.base import BaseModel, ModelResponse, ModelConfig, ModelProvider

logger = logging.getLogger(__name__)


class StubProvider(BaseModel):
    """
    Stub model provider for testing.
    
    Returns predefined responses without calling external APIs.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize stub provider."""
        super().__init__(config)
        logger.info(f"Stub provider initialized: {self.name}")
    
    def validate_config(self) -> bool:
        """Validate stub configuration."""
        return True
    
    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a stub response.
        
        Args:
            prompt: Input prompt
            
        Returns:
            ModelResponse object
        """
        start_time = time.time()
        
        # Simulate API latency
        await asyncio.sleep(0.1)
        
        content = f"[{self.name}] Response to: '{prompt}'"
        latency = (time.time() - start_time) * 1000
        
        logger.debug(f"Stub response generated in {latency:.2f}ms")
        
        return ModelResponse(
            content=content,
            model_name=self.name,
            provider=self.provider.value,
            tokens_used=len(content.split()),
            latency_ms=latency,
            cost=0.0,
            metadata={"stub": True}
        )
    
    def calculate_cost(self, tokens_used: int) -> float:
        """Stub models are free."""
        return 0.0
