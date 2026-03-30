"""
Stream manager for coordinating streaming responses.
"""

import logging
import asyncio
from typing import Optional, Callable, AsyncGenerator

logger = logging.getLogger(__name__)


class StreamManager:
    """
    Manages streaming of model responses.
    
    Coordinates token-by-token or chunk-by-chunk streaming
    from AI models to clients.
    """
    
    def __init__(self):
        """Initialize stream manager."""
        self.active_streams = {}
        logger.info("StreamManager initialized")
    
    async def stream_response(
        self,
        prompt: str,
        model_generator: AsyncGenerator,
        callback: Callable[[str, dict], None]
    ):
        """
        Stream response from model generator.
        
        Args:
            prompt: Input prompt
            model_generator: Async generator yielding tokens/chunks
            callback: Function to call with each token
        """
        try:
            logger.info(f"Starting stream for prompt: {prompt[:50]}...")
            
            token_count = 0
            full_response = ""
            
            async for token_data in model_generator:
                token = token_data.get('token', '')
                metadata = token_data.get('metadata', {})
                
                # Accumulate response
                full_response += token
                token_count += 1
                
                # Send token to callback
                callback(token, {
                    **metadata,
                    'token_count': token_count,
                    'total_length': len(full_response)
                })
                
                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.01)
            
            logger.info(f"Stream completed: {token_count} tokens")
            
            return {
                'response': full_response,
                'token_count': token_count,
            }
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            raise
    
    async def stream_ensemble(
        self,
        prompt: str,
        models: list,
        callback: Callable[[str, dict], None],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Stream ensemble inference with progress updates.
        
        Args:
            prompt: Input prompt
            models: List of models to use
            callback: Function to call with response chunks
            progress_callback: Optional progress callback
        """
        try:
            total_models = len(models)
            
            if progress_callback:
                progress_callback(0, "Starting ensemble inference...")
            
            # Generate responses from all models
            responses = []
            for i, model in enumerate(models):
                if progress_callback:
                    progress = (i / total_models) * 50  # First 50%
                    progress_callback(progress, f"Querying {model.name}...")
                
                response = await model.generate(prompt)
                responses.append(response)
            
            if progress_callback:
                progress_callback(50, "Scoring responses...")
            
            # Score and rank (simplified)
            from nexus.core.scoring import ResponseScorer
            scorer = ResponseScorer()
            
            scored = []
            for i, response in enumerate(responses):
                if progress_callback:
                    progress = 50 + (i / total_models) * 30  # Next 30%
                    progress_callback(progress, f"Scoring {response.model_name}...")
                
                score = scorer.score_response(response, prompt)
                scored.append((score, response))
            
            # Sort by score
            scored.sort(reverse=True, key=lambda x: x[0])
            top_response = scored[0][1]
            
            if progress_callback:
                progress_callback(80, "Streaming top response...")
            
            # Stream the top response word by word
            words = top_response.content.split()
            for i, word in enumerate(words):
                progress = 80 + (i / len(words)) * 20  # Final 20%
                
                callback(word + " ", {
                    'model': top_response.model_name,
                    'score': scored[0][0],
                    'progress': progress
                })
                
                await asyncio.sleep(0.05)  # Simulate streaming
            
            if progress_callback:
                progress_callback(100, "Complete!")
            
            return {
                'response': top_response.content,
                'model': top_response.model_name,
                'score': scored[0][0],
                'all_scores': [(r.model_name, s) for s, r in scored]
            }
            
        except Exception as e:
            logger.error(f"Error during ensemble streaming: {e}", exc_info=True)
            raise
