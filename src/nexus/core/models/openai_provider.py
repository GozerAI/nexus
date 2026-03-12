"""
OpenAI model provider implementation.
"""

import time
import logging
from typing import Optional
import asyncio

from nexus.core.models.base import BaseModel, ModelResponse, ModelConfig, ModelProvider
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseModel):
    """
    OpenAI model provider.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    """
    
    # Pricing per 1K tokens (approximate, as of 2024)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }
    
    def __init__(self, config: ModelConfig):
        """Initialize OpenAI provider."""
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            import openai
            if self.config.api_key:
                self.client = openai.AsyncOpenAI(api_key=self.config.api_key)
                logger.info(f"OpenAI client initialized for model: {self.config.model_id}")
            else:
                logger.warning("No API key provided for OpenAI")
        except ImportError:
            logger.error("OpenAI package not installed. Run: pip install openai")
            raise
    
    def validate_config(self) -> bool:
        """Validate OpenAI configuration."""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        
        if not self.config.model_id:
            raise ValueError("OpenAI model_id is required (e.g., 'gpt-4', 'gpt-3.5-turbo')")
        
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a response using OpenAI.
        
        Args:
            prompt: Input prompt
            
        Returns:
            ModelResponse object
        """
        start_time = time.time()
        
        try:
            if not self.client:
                return ModelResponse(
                    content="",
                    model_name=self.name,
                    provider=self.provider.value,
                    error="OpenAI client not initialized"
                )
            
            logger.debug(f"Calling OpenAI with prompt: {prompt[:50]}...")
            
            response = await self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
            )
            
            latency = (time.time() - start_time) * 1000
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            cost = self.calculate_cost(tokens_used)
            
            logger.info(
                f"OpenAI response: tokens={tokens_used}, "
                f"latency={latency:.2f}ms, cost=${cost:.4f}"
            )
            
            return ModelResponse(
                content=content,
                model_name=self.name,
                provider=self.provider.value,
                tokens_used=tokens_used,
                latency_ms=latency,
                cost=cost,
                metadata={
                    "model_id": self.config.model_id,
                    "finish_reason": response.choices[0].finish_reason,
                }
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"OpenAI error: {str(e)}", exc_info=True)
            
            return ModelResponse(
                content="",
                model_name=self.name,
                provider=self.provider.value,
                latency_ms=latency,
                error=str(e)
            )
    
    def calculate_cost(self, tokens_used: int) -> float:
        """
        Calculate cost based on OpenAI pricing.
        
        Args:
            tokens_used: Total tokens used
            
        Returns:
            Cost in USD
        """
        model_base = self.config.model_id.split("-")[0:2]
        model_key = "-".join(model_base)
        
        if model_key in self.PRICING:
            # Approximate: 60% input, 40% output
            input_tokens = int(tokens_used * 0.6)
            output_tokens = int(tokens_used * 0.4)
            
            input_cost = (input_tokens / 1000) * self.PRICING[model_key]["input"]
            output_cost = (output_tokens / 1000) * self.PRICING[model_key]["output"]
            
            return input_cost + output_cost
        
        return 0.0
