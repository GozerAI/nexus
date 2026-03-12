"""
Reranking - Improve retrieval precision with cross-encoders.

Initial retrieval (bi-encoder) is fast but approximate.
Reranking with cross-encoders is slower but much more accurate.

Pipeline:
1. Fast retrieval gets top 50-100 candidates
2. Reranker scores each (query, document) pair
3. Return top 5-10 with much better precision
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result after reranking."""

    doc_id: str
    text: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float
    original_rank: int
    new_rank: int


class Reranker(ABC):
    """Base class for rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank documents for a query."""
        pass


class CrossEncoderReranker(Reranker):
    """
    Reranker using cross-encoder models.

    Cross-encoders process (query, document) pairs together,
    allowing for much richer interaction than bi-encoders.

    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast, good quality
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Better quality
    - BAAI/bge-reranker-base: Excellent quality
    - BAAI/bge-reranker-large: Best quality
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,  # "cpu", "cuda", or None for auto
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _ensure_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self.model_name,
                    device=self.device,
                )
                logger.info(f"Loaded cross-encoder: {self.model_name}")

            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. Install with:\n"
                    "  pip install sentence-transformers"
                )

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: The search query
            documents: List of documents with 'doc_id', 'text', 'metadata', 'score'
            top_k: Number of results to return

        Returns:
            List of RerankResult sorted by rerank score
        """
        if not documents:
            return []

        self._ensure_model()

        # Prepare pairs for cross-encoder
        pairs = [(query, doc["text"]) for doc in documents]

        # Get rerank scores
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > 100,
        )

        # Build results with original and new rankings
        results = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            results.append({
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "original_score": doc.get("score", 0.0),
                "rerank_score": float(score),
                "original_rank": i + 1,
            })

        # Sort by rerank score
        results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Assign new ranks and convert to RerankResult
        final_results = []
        for new_rank, result in enumerate(results[:top_k], 1):
            final_results.append(RerankResult(
                doc_id=result["doc_id"],
                text=result["text"],
                metadata=result["metadata"],
                original_score=result["original_score"],
                rerank_score=result["rerank_score"],
                original_rank=result["original_rank"],
                new_rank=new_rank,
            ))

        return final_results


class CohereReranker(Reranker):
    """
    Reranker using Cohere's Rerank API.

    Excellent quality, requires API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "rerank-english-v3.0",
    ):
        import os
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        self.model = model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError(
                    "cohere not installed. Install with:\n"
                    "  pip install cohere"
                )

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank using Cohere API."""
        if not documents:
            return []

        self._ensure_client()

        # Prepare documents for Cohere
        doc_texts = [doc["text"] for doc in documents]

        # Call Cohere rerank
        response = self._client.rerank(
            query=query,
            documents=doc_texts,
            model=self.model,
            top_n=top_k,
        )

        # Build results
        results = []
        for new_rank, result in enumerate(response.results, 1):
            original_idx = result.index
            doc = documents[original_idx]

            results.append(RerankResult(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                original_score=doc.get("score", 0.0),
                rerank_score=result.relevance_score,
                original_rank=original_idx + 1,
                new_rank=new_rank,
            ))

        return results


class LLMReranker(Reranker):
    """
    Reranker using LLM for relevance scoring.

    Flexible but slower. Good for complex relevance judgments.
    """

    def __init__(
        self,
        llm_client=None,  # Any LLM client with a generate method
        max_docs_per_call: int = 5,  # Batch size for LLM
    ):
        self.llm_client = llm_client
        self.max_docs_per_call = max_docs_per_call

    def _score_batch(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> List[float]:
        """Score a batch of documents using LLM."""
        if not self.llm_client:
            raise ValueError("LLM client not configured")

        # Build prompt
        doc_list = "\n\n".join([
            f"Document {i+1}:\n{doc['text'][:500]}"
            for i, doc in enumerate(documents)
        ])

        prompt = f"""Rate the relevance of each document to the query on a scale of 0-10.
Query: {query}

{doc_list}

Return ONLY a JSON array of scores, e.g., [8, 3, 7, 5, 2]
Scores:"""

        # Call LLM
        response = self.llm_client.generate(prompt)

        # Parse scores
        import json
        try:
            scores = json.loads(response.strip())
            return [float(s) for s in scores]
        except (json.JSONDecodeError, ValueError):
            # Fallback: return original order
            return [10 - i for i in range(len(documents))]

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Rerank using LLM scoring."""
        if not documents:
            return []

        # Score in batches
        all_scores = []
        for i in range(0, len(documents), self.max_docs_per_call):
            batch = documents[i:i + self.max_docs_per_call]
            scores = self._score_batch(query, batch)
            all_scores.extend(scores)

        # Build results
        scored_docs = list(zip(documents, all_scores, range(1, len(documents) + 1)))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (doc, score, original_rank) in enumerate(scored_docs[:top_k], 1):
            results.append(RerankResult(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                original_score=doc.get("score", 0.0),
                rerank_score=score / 10.0,  # Normalize to 0-1
                original_rank=original_rank,
                new_rank=new_rank,
            ))

        return results


class NoOpReranker(Reranker):
    """Pass-through reranker that doesn't change ranking."""

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[RerankResult]:
        """Return documents unchanged."""
        results = []
        for i, doc in enumerate(documents[:top_k], 1):
            results.append(RerankResult(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
                original_score=doc.get("score", 0.0),
                rerank_score=doc.get("score", 0.0),
                original_rank=i,
                new_rank=i,
            ))
        return results


def get_reranker(
    provider: str = "cross-encoder",
    model_name: Optional[str] = None,
    **kwargs,
) -> Reranker:
    """
    Factory function to get a reranker.

    Args:
        provider: "cross-encoder", "cohere", "llm", or "none"
        model_name: Model name for the provider
        **kwargs: Additional provider-specific arguments

    Returns:
        Reranker instance
    """
    if provider == "cross-encoder":
        return CrossEncoderReranker(
            model_name=model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            **kwargs,
        )
    elif provider == "cohere":
        return CohereReranker(
            model=model_name or "rerank-english-v3.0",
            **kwargs,
        )
    elif provider == "llm":
        return LLMReranker(**kwargs)
    elif provider == "none":
        return NoOpReranker()
    else:
        raise ValueError(f"Unknown provider: {provider}")
