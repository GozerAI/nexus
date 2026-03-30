"""
Enhanced multi-model ensemble inference system with real AI model support.

This module provides functionality to orchestrate multiple real AI models,
score their responses, and select the best output through ensemble ranking.
"""

from collections.abc import Iterator, Sequence
from typing import List, Tuple, Optional
import logging
import asyncio
import yaml

from nexus.core.config_validator import validate_config
from nexus.core.models.base import ModelConfig, ModelProvider, ModelResponse
from nexus.core.models.model_factory import ModelFactory
from nexus.core.scoring import ResponseScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_ensemble(config_path: str = "config/config.yaml") -> List:
    """
    Load model definitions from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        List of BaseModel objects configured from the file

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
        ValueError: If configuration validation fails
    """
    try:
        logger.info(f"Loading model ensemble from {config_path}")

        # Validate configuration
        is_valid, errors = validate_config(config_path)
        if not is_valid:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Configuration validation passed")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        models_cfg = cfg.get("model_ensemble", {}).get("models", [])

        if not models_cfg:
            logger.warning("No models found in configuration")

        models = []
        for m in models_cfg:
            try:
                # Parse provider
                provider_str = m.get("provider", "stub")
                try:
                    provider = ModelProvider(provider_str.lower())
                except ValueError:
                    logger.warning(f"Unknown provider '{provider_str}', using stub")
                    provider = ModelProvider.STUB

                # Create model config
                config = ModelConfig(
                    name=m.get("name"),
                    provider=provider,
                    weight=m.get("weight", 0.5),
                    api_key=m.get("api_key"),
                    model_id=m.get("model_id"),
                    temperature=m.get("temperature", 0.7),
                    max_tokens=m.get("max_tokens", 1000),
                    timeout=m.get("timeout", 30),
                )

                # Create model instance
                model = ModelFactory.create_model(config)
                models.append(model)
                logger.debug(f"Loaded model: {config.name} ({provider.value})")

            except Exception as e:
                logger.error(f"Failed to load model {m.get('name')}: {e}")
                continue

        logger.info(f"Successfully loaded {len(models)} models")
        return models

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading model ensemble: {e}")
        raise


class LazyModelEnsemble(Sequence):
    """Load the configured ensemble only when first used."""

    def __init__(self, loader):
        self._loader = loader
        self._models = None

    def _ensure_loaded(self):
        if self._models is None:
            try:
                self._models = self._loader()
            except Exception as exc:
                logger.error(f"Failed to load model ensemble lazily: {exc}")
                self._models = []
        return self._models

    def reset(self) -> None:
        """Clear cached models so tests can isolate loading behavior."""
        self._models = None

    def __iter__(self) -> Iterator:
        return iter(self._ensure_loaded())

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def __getitem__(self, index):
        return self._ensure_loaded()[index]


model_ensemble = LazyModelEnsemble(load_model_ensemble)


async def rank_responses_async(prompt: str) -> List[Tuple[float, ModelResponse]]:
    """
    Generate and rank responses from all models in the ensemble asynchronously.

    Args:
        prompt: Input prompt to send to all models

    Returns:
        List of tuples (score, ModelResponse) sorted by score descending

    Raises:
        ValueError: If model ensemble is empty
        RuntimeError: If all models fail to generate responses
    """
    logger.info(f"Ranking responses for prompt: {prompt[:50]}...")

    if not model_ensemble:
        logger.error("No models available in ensemble")
        raise ValueError("Model ensemble is empty")

    # Generate responses from all models concurrently
    tasks = [model.generate(prompt) for model in model_ensemble]
    
    try:
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.error(f"Error during concurrent model generation: {e}")
        raise RuntimeError("Failed to generate responses from models")

    # Score responses
    scorer = ResponseScorer()
    scored_responses = []

    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(f"Model {model_ensemble[i].name} failed: {response}")
            continue

        if not response.success:
            logger.warning(f"Model {model_ensemble[i].name} returned error: {response.error}")
            continue

        try:
            score = scorer.score_response(response, prompt)
            scored_responses.append((score, response))
            logger.debug(f"{response.model_name}: score={score:.3f}")
        except Exception as e:
            logger.error(f"Error scoring response from {model_ensemble[i].name}: {e}")
            continue

    if not scored_responses:
        logger.error("No successful responses generated from any model")
        raise RuntimeError("All models failed to generate valid responses")

    sorted_responses = sorted(scored_responses, reverse=True, key=lambda x: x[0])
    
    top_model = sorted_responses[0][1].model_name
    top_score = sorted_responses[0][0]
    logger.info(f"Top model: {top_model} with score {top_score:.3f}")

    return sorted_responses


def rank_responses(prompt: str) -> List[Tuple[float, str, str]]:
    """
    Synchronous wrapper for rank_responses_async.
    
    Args:
        prompt: Input prompt
        
    Returns:
        List of tuples (score, response_text, model_name)
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    ranked = loop.run_until_complete(rank_responses_async(prompt))
    
    # Convert to old format for backward compatibility
    return [(score, resp.content, resp.model_name) for score, resp in ranked]


async def ensemble_inference_async(prompt: str) -> ModelResponse:
    """
    Perform ensemble inference asynchronously.

    Args:
        prompt: Input prompt to process

    Returns:
        Top-ranked ModelResponse

    Raises:
        ValueError: If prompt is empty or ensemble is not configured
        RuntimeError: If all models fail to generate responses
    """
    if not prompt:
        logger.error("Empty prompt provided")
        raise ValueError("Prompt cannot be empty")

    logger.info(f"Starting ensemble inference for: {prompt[:50]}...")

    try:
        ranked = await rank_responses_async(prompt)

        print("\nRanked Responses:")
        for i, (score, response) in enumerate(ranked):
            print(
                f"{i+1}. {response.model_name}: {response.content[:100]}... "
                f"(score={score:.2f}, latency={response.latency_ms:.0f}ms, "
                f"cost=${response.cost:.4f})"
            )

        top_response = ranked[0][1]
        logger.info("Ensemble inference completed successfully")

        return top_response

    except Exception as e:
        logger.error(f"Ensemble inference failed: {e}")
        raise


def ensemble_inference(prompt: str) -> str:
    """
    Synchronous wrapper for ensemble_inference_async.
    
    Args:
        prompt: Input prompt
        
    Returns:
        Top-ranked response text
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    response = loop.run_until_complete(ensemble_inference_async(prompt))
    return response.content


if __name__ == "__main__":
    try:
        user_input = input("Enter your question: ")

        if not user_input.strip():
            print("Error: Question cannot be empty")
            exit(1)

        top_response = ensemble_inference(user_input)

        print("\nTop Response Selected:")
        print(top_response)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nError: {e}")
        exit(1)
