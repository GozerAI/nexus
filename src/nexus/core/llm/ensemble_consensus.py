"""
Multi-model ensemble with early consensus detection.

Runs inference across multiple models in parallel and terminates early
when a sufficient number of models agree on the answer. This reduces
total inference time compared to waiting for all models to complete.

Consensus is detected via:
1. Exact match (for structured/short outputs)
2. Semantic similarity (for longer text)
3. Majority voting (for classification tasks)
"""

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConsensusMethod(str, Enum):
    EXACT_MATCH = "exact_match"
    SIMILARITY = "similarity"
    MAJORITY_VOTE = "majority_vote"


@dataclass
class ModelResponse:
    """Response from a single model in the ensemble."""
    model_name: str
    response: Any
    latency_ms: float
    token_count: int = 0
    confidence: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class ConsensusResult:
    """Result of ensemble inference with consensus."""
    consensus_reached: bool
    consensus_response: Any = None
    consensus_method: Optional[ConsensusMethod] = None
    agreement_ratio: float = 0.0
    models_completed: int = 0
    models_total: int = 0
    responses: List[ModelResponse] = field(default_factory=list)
    total_latency_ms: float = 0.0
    early_termination: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return " ".join(text.lower().strip().split())


class EnsembleConsensus:
    """
    Multi-model ensemble with early consensus detection.

    Features:
    - Parallel inference across N models
    - Early termination when consensus threshold met
    - Multiple consensus methods (exact, similarity, voting)
    - Configurable quorum (minimum models before checking consensus)
    - Weighted responses by model confidence/quality
    """

    DEFAULT_THRESHOLD = 0.7  # 70% agreement needed
    DEFAULT_QUORUM = 2       # Min models before consensus check
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        consensus_method: ConsensusMethod = ConsensusMethod.EXACT_MATCH,
        threshold: float = DEFAULT_THRESHOLD,
        quorum: int = DEFAULT_QUORUM,
        timeout_seconds: float = DEFAULT_TIMEOUT,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.9,
    ):
        """
        Args:
            consensus_method: Method for detecting consensus
            threshold: Fraction of models that must agree
            quorum: Minimum responses before checking consensus
            timeout_seconds: Max time to wait for all models
            embedding_fn: Function to compute text embeddings (for similarity)
            similarity_threshold: Cosine similarity threshold for "agreement"
        """
        self._method = consensus_method
        self._threshold = threshold
        self._quorum = quorum
        self._timeout = timeout_seconds
        self._embedding_fn = embedding_fn
        self._similarity_threshold = similarity_threshold
        self._stats = {
            "runs": 0,
            "consensus_reached": 0,
            "early_terminations": 0,
            "total_models_used": 0,
            "total_models_needed": 0,
        }

    async def run(
        self,
        models: List[str],
        inference_fn: Callable[..., Coroutine],
        prompt: str,
        **kwargs: Any,
    ) -> ConsensusResult:
        """
        Run ensemble inference with early consensus.

        Args:
            models: List of model names to query
            inference_fn: ``async def fn(model_name, prompt, **kwargs) -> response``
            prompt: Input prompt
            **kwargs: Additional params passed to inference_fn

        Returns:
            ConsensusResult
        """
        self._stats["runs"] += 1
        self._stats["total_models_used"] += len(models)
        start = time.time()

        responses: List[ModelResponse] = []
        pending_tasks: Dict[str, asyncio.Task] = {}
        consensus_result = ConsensusResult(
            consensus_reached=False,
            models_total=len(models),
        )

        # Launch all models
        for model_name in models:
            task = asyncio.create_task(
                self._run_model(model_name, inference_fn, prompt, **kwargs)
            )
            pending_tasks[model_name] = task

        # Collect responses, checking consensus as they arrive
        done_set: set = set()
        try:
            remaining_timeout = self._timeout
            while pending_tasks and remaining_timeout > 0:
                done, _ = await asyncio.wait(
                    pending_tasks.values(),
                    timeout=min(remaining_timeout, 1.0),
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in done:
                    # Find which model this task belongs to
                    for name, t in list(pending_tasks.items()):
                        if t is task and name not in done_set:
                            done_set.add(name)
                            del pending_tasks[name]
                            try:
                                resp = task.result()
                                responses.append(resp)
                            except Exception as e:
                                responses.append(ModelResponse(
                                    model_name=name,
                                    response=None,
                                    latency_ms=0,
                                    error=str(e),
                                ))
                            break

                # Check consensus if quorum met
                successful = [r for r in responses if r.success]
                if len(successful) >= self._quorum:
                    check = self._check_consensus(successful, len(models))
                    if check is not None:
                        consensus_result.consensus_reached = True
                        consensus_result.consensus_response = check[0]
                        consensus_result.consensus_method = self._method
                        consensus_result.agreement_ratio = check[1]
                        consensus_result.early_termination = bool(pending_tasks)

                        # Cancel remaining tasks
                        for task in pending_tasks.values():
                            task.cancel()

                        if consensus_result.early_termination:
                            self._stats["early_terminations"] += 1
                            self._stats["total_models_needed"] += len(successful)

                        break

                remaining_timeout = self._timeout - (time.time() - start)

        except asyncio.TimeoutError:
            for task in pending_tasks.values():
                task.cancel()

        consensus_result.responses = responses
        consensus_result.models_completed = len(responses)
        consensus_result.total_latency_ms = (time.time() - start) * 1000

        # Final consensus check if not yet reached
        if not consensus_result.consensus_reached:
            successful = [r for r in responses if r.success]
            if successful:
                check = self._check_consensus(successful, len(models))
                if check:
                    consensus_result.consensus_reached = True
                    consensus_result.consensus_response = check[0]
                    consensus_result.consensus_method = self._method
                    consensus_result.agreement_ratio = check[1]

        if consensus_result.consensus_reached:
            self._stats["consensus_reached"] += 1

        # If no consensus, return best response by confidence or first success
        if not consensus_result.consensus_reached:
            successful = [r for r in responses if r.success]
            if successful:
                best = max(successful, key=lambda r: r.confidence)
                consensus_result.consensus_response = best.response

        return consensus_result

    async def _run_model(
        self,
        model_name: str,
        inference_fn: Callable,
        prompt: str,
        **kwargs: Any,
    ) -> ModelResponse:
        """Run inference on a single model."""
        start = time.time()
        try:
            result = await inference_fn(model_name, prompt, **kwargs)
            latency_ms = (time.time() - start) * 1000
            return ModelResponse(
                model_name=model_name,
                response=result,
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return ModelResponse(
                model_name=model_name,
                response=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    def _check_consensus(
        self, responses: List[ModelResponse], total_models: int
    ) -> Optional[Tuple[Any, float]]:
        """
        Check if consensus has been reached.

        Returns:
            (consensus_response, agreement_ratio) or None
        """
        if not responses:
            return None

        if self._method == ConsensusMethod.EXACT_MATCH:
            return self._check_exact_match(responses, total_models)
        elif self._method == ConsensusMethod.SIMILARITY:
            return self._check_similarity(responses, total_models)
        elif self._method == ConsensusMethod.MAJORITY_VOTE:
            return self._check_majority_vote(responses, total_models)
        return None

    def _check_exact_match(
        self, responses: List[ModelResponse], total: int
    ) -> Optional[Tuple[Any, float]]:
        """Check for exact text match consensus."""
        normalized: Dict[str, List[ModelResponse]] = {}
        for r in responses:
            key = _normalize_text(str(r.response)) if r.response else ""
            normalized.setdefault(key, []).append(r)

        for key, group in normalized.items():
            ratio = len(group) / total
            if ratio >= self._threshold:
                return group[0].response, ratio
        return None

    def _check_similarity(
        self, responses: List[ModelResponse], total: int
    ) -> Optional[Tuple[Any, float]]:
        """Check for semantic similarity consensus."""
        if not self._embedding_fn:
            # Fallback to exact match if no embedding function
            return self._check_exact_match(responses, total)

        texts = [str(r.response) for r in responses if r.response]
        if len(texts) < 2:
            return None

        embeddings = [self._embedding_fn(t) for t in texts]

        # Find largest cluster of similar responses
        clusters: List[List[int]] = []
        assigned: set = set()

        for i in range(len(embeddings)):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j in range(i + 1, len(embeddings)):
                if j in assigned:
                    continue
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._similarity_threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        # Check if any cluster meets threshold
        for cluster in clusters:
            ratio = len(cluster) / total
            if ratio >= self._threshold:
                return responses[cluster[0]].response, ratio
        return None

    def _check_majority_vote(
        self, responses: List[ModelResponse], total: int
    ) -> Optional[Tuple[Any, float]]:
        """Check for majority vote consensus."""
        votes: Dict[str, int] = {}
        vote_to_response: Dict[str, Any] = {}

        for r in responses:
            key = _normalize_text(str(r.response)) if r.response else ""
            votes[key] = votes.get(key, 0) + 1
            vote_to_response[key] = r.response

        if votes:
            best_key = max(votes, key=votes.get)
            ratio = votes[best_key] / total
            if ratio >= self._threshold:
                return vote_to_response[best_key], ratio
        return None

    def get_stats(self) -> Dict[str, Any]:
        runs = self._stats["runs"]
        return {
            **self._stats,
            "consensus_rate": (
                self._stats["consensus_reached"] / runs if runs > 0 else 0.0
            ),
            "early_termination_rate": (
                self._stats["early_terminations"] / runs if runs > 0 else 0.0
            ),
            "avg_models_needed": (
                self._stats["total_models_needed"]
                / max(self._stats["early_terminations"], 1)
            ),
        }
