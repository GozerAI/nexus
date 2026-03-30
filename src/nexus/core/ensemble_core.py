"""
Multi-model ensemble inference system for superintelligent AI.

This module provides functionality to orchestrate multiple AI models,
score their responses, and select the best output through ensemble ranking.
"""

from collections.abc import Iterator, Sequence
from typing import List, Tuple
import logging
import yaml
from nexus.core.config_validator import validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_ensemble(config_path: str = "config/config.yaml") -> List['ModelStub']:
    """
    Load model definitions from a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        List of ModelStub objects configured from the file

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
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
            name = m.get("name")
            weight = m.get("weight", 0)
            if name is not None:
                models.append(ModelStub(name, weight))
                logger.debug(f"Loaded model: {name} with weight {weight}")

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


class ModelStub:
    """
    Stub implementation of an AI model.

    This is a placeholder that simulates model behavior for testing
    and demonstration purposes.
    """

    def __init__(self, name: str, weight: float) -> None:
        """
        Initialize a model stub.

        Args:
            name: Name/identifier of the model
            weight: Weight for ensemble voting (higher = more influence)
        """
        self.name = name
        self.weight = weight
        logger.debug(f"Initialized ModelStub: {name}")

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response to the given prompt.

        Args:
            prompt: Input text to respond to

        Returns:
            Generated response string
        """
        logger.debug(f"Model {self.name} generating response for: {prompt[:50]}...")
        return f"[{self.name}] Response to: '{prompt}'"


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


def score_response(response: str, prompt: str = "") -> float:
    """
    Score a model response for quality based on multiple heuristics.

    Scoring factors:
    - Length appropriateness (not too short, not too long)
    - Sentence structure (proper sentences, not fragments)
    - Relevance to prompt (keyword overlap)
    - Coherence indicators (transition words, structure)
    - Content quality (avoids filler, repetition)

    Args:
        response: Response text to score
        prompt: Original prompt for relevance scoring

    Returns:
        Float score between 0 and 1
    """
    import re
    from collections import Counter

    if not response or not response.strip():
        return 0.0

    scores = []

    # 1. Length score (optimal range: 50-500 chars for typical responses)
    length = len(response)
    if length < 20:
        length_score = 0.2
    elif length < 50:
        length_score = 0.5
    elif length <= 500:
        length_score = 1.0
    elif length <= 1000:
        length_score = 0.9
    else:
        length_score = max(0.5, 1.0 - (length - 1000) / 5000)
    scores.append(length_score)

    # 2. Sentence structure score
    sentences = re.split(r'[.!?]+', response)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(valid_sentences) == 0:
        structure_score = 0.3
    else:
        # Check for proper capitalization and length
        proper_count = sum(1 for s in valid_sentences if s[0].isupper())
        structure_score = proper_count / len(valid_sentences)
    scores.append(structure_score)

    # 3. Relevance score (keyword overlap with prompt)
    if prompt:
        prompt_words = set(re.findall(r'\b\w{3,}\b', prompt.lower()))
        response_words = set(re.findall(r'\b\w{3,}\b', response.lower()))
        if prompt_words:
            overlap = len(prompt_words & response_words) / len(prompt_words)
            relevance_score = min(1.0, overlap * 2)  # Scale up, cap at 1.0
        else:
            relevance_score = 0.5
    else:
        relevance_score = 0.5  # Neutral if no prompt
    scores.append(relevance_score)

    # 4. Coherence score (presence of transition/structure words)
    coherence_indicators = [
        r'\b(however|therefore|furthermore|moreover|additionally)\b',
        r'\b(first|second|third|finally|lastly)\b',
        r'\b(because|since|although|while|whereas)\b',
        r'\b(for example|for instance|such as|including)\b',
        r'\b(in conclusion|in summary|overall|to summarize)\b',
    ]
    coherence_count = sum(
        1 for pattern in coherence_indicators
        if re.search(pattern, response.lower())
    )
    coherence_score = min(1.0, 0.5 + coherence_count * 0.15)
    scores.append(coherence_score)

    # 5. Content quality (penalize repetition and filler)
    words = re.findall(r'\b\w+\b', response.lower())
    if len(words) > 5:
        word_counts = Counter(words)
        # High repetition of non-common words is bad
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'and', 'in', 'that', 'it', 'for'}
        significant_words = [w for w in words if w not in common_words and len(w) > 3]
        if significant_words:
            sig_counts = Counter(significant_words)
            max_repeat = max(sig_counts.values()) if sig_counts else 1
            repetition_penalty = min(0.5, (max_repeat - 1) * 0.1)
            quality_score = 1.0 - repetition_penalty
        else:
            quality_score = 0.7
    else:
        quality_score = 0.5
    scores.append(quality_score)

    # Calculate weighted average
    weights = [0.15, 0.20, 0.25, 0.20, 0.20]  # Length, Structure, Relevance, Coherence, Quality
    final_score = sum(s * w for s, w in zip(scores, weights))

    logger.debug(f"Scored response: {final_score:.3f} (len={length_score:.2f}, struct={structure_score:.2f}, rel={relevance_score:.2f}, coh={coherence_score:.2f}, qual={quality_score:.2f})")
    return final_score


def rank_responses(prompt: str) -> List[Tuple[float, str, str]]:
    """
    Generate and rank responses from all models in the ensemble.

    Args:
        prompt: Input prompt to send to all models

    Returns:
        List of tuples (score, response, model_name) sorted by score descending
    """
    logger.info(f"Ranking responses for prompt: {prompt[:50]}...")

    if not model_ensemble:
        logger.error("No models available in ensemble")
        raise ValueError("Model ensemble is empty")

    responses = []
    for model in model_ensemble:
        try:
            reply = model.generate_response(prompt)
            score = score_response(reply, prompt)  # Pass prompt for relevance scoring
            responses.append((score, reply, model.name))
            logger.debug(f"{model.name}: score={score:.3f}")
        except Exception as e:
            logger.error(f"Error generating response from {model.name}: {e}")
            # Continue with other models even if one fails
            continue

    if not responses:
        logger.error("No responses generated from any model")
        raise RuntimeError("All models failed to generate responses")

    sorted_responses = sorted(responses, reverse=True)
    logger.info(f"Top model: {sorted_responses[0][2]} with score {sorted_responses[0][0]:.3f}")

    return sorted_responses


def ensemble_inference(prompt: str) -> str:
    """
    Perform ensemble inference by generating, scoring, and ranking model responses.

    Args:
        prompt: Input prompt to process

    Returns:
        Top-ranked response string

    Raises:
        ValueError: If prompt is empty or ensemble is not configured
        RuntimeError: If all models fail to generate responses
    """
    if not prompt:
        logger.error("Empty prompt provided")
        raise ValueError("Prompt cannot be empty")

    logger.info(f"Starting ensemble inference for: {prompt[:50]}...")

    try:
        ranked = rank_responses(prompt)

        print("\nRanked Responses:")
        for i, (score, reply, name) in enumerate(ranked):
            print(f"{i+1}. {name}: {reply} (score={score:.2f})")

        top_response = ranked[0][1]
        logger.info("Ensemble inference completed successfully")

        return top_response

    except Exception as e:
        logger.error(f"Ensemble inference failed: {e}")
        raise


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
