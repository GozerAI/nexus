"""
Anthropic Claude model provider implementation.
"""

import time
import logging
from typing import Optional

from nexus.core.models.base import BaseModel, ModelResponse, ModelConfig, ModelProvider
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseModel):
    """
    Anthropic Claude model provider.
    
    Supports Claude 3 Opus, Sonnet, and Haiku models.
    """
    
    # Pricing per 1K tokens (approximate)
    PRICING = {
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    def __init__(self, config: ModelConfig):
        """Initialize Anthropic provider."""
        super().__init__(config)
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        try:
            import anthropic
            if self.config.api_key:
                self.client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
                logger.info(f"Anthropic client initialized for model: {self.config.model_id}")
            else:
                logger.warning("No API key provided for Anthropic")
        except ImportError:
            logger.error("Anthropic package not installed. Run: pip install anthropic")
            raise
    
    def validate_config(self) -> bool:
        """Validate Anthropic configuration."""
        if not self.config.api_key:
            raise ValueError("Anthropic API key is required")
        
        if not self.config.model_id:
            raise ValueError("Anthropic model_id is required (e.g., 'claude-3-opus-20240229')")
        
        return True
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a response using Anthropic Claude.
        
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
                    error="Anthropic client not initialized"
                )
            
            logger.debug(f"Calling Anthropic with prompt: {prompt[:50]}...")
            
            response = await self.client.messages.create(
                model=self.config.model_id,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                timeout=self.config.timeout,
            )
            
            latency = (time.time() - start_time) * 1000
            content = response.content[0].text if response.content else ""
            
            # Calculate tokens
            input_tokens = response.usage.input_tokens if hasattr(response.usage, 'input_tokens') else 0
            output_tokens = response.usage.output_tokens if hasattr(response.usage, 'output_tokens') else 0
            tokens_used = input_tokens + output_tokens
            
            cost = self.calculate_cost(tokens_used)
            
            logger.info(
                f"Anthropic response: tokens={tokens_used}, "
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
                    "stop_reason": response.stop_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            )
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"Anthropic error: {str(e)}", exc_info=True)
            
            return ModelResponse(
                content="",
                model_name=self.name,
                provider=self.provider.value,
                latency_ms=latency,
                error=str(e)
            )
    
    def calculate_cost(self, tokens_used: int) -> float:
        """
        Calculate cost based on Anthropic pricing.
        
        Args:
            tokens_used: Total tokens used
            
        Returns:
            Cost in USD
        """
        # Extract base model name
        model_parts = self.config.model_id.split("-")
        if len(model_parts) >= 3:
            model_key = f"{model_parts[0]}-{model_parts[1]}-{model_parts[2]}"
        else:
            model_key = self.config.model_id
        
        if model_key in self.PRICING:
            # Approximate: 60% input, 40% output
            input_tokens = int(tokens_used * 0.6)
            output_tokens = int(tokens_used * 0.4)
            
            input_cost = (input_tokens / 1000) * self.PRICING[model_key]["input"]
            output_cost = (output_tokens / 1000) * self.PRICING[model_key]["output"]
            
            return input_cost + output_cost
        
        return 0.0
