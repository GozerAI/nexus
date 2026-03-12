"""
Hybrid Search - Combines semantic and keyword search for best retrieval.

BM25 excels at exact matches and rare terms.
Vector search excels at semantic similarity and concepts.
Combining them gives significantly better retrieval quality.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid search."""

    doc_id: str
    text: str
    metadata: Dict[str, Any]
    vector_score: float
    bm25_score: float
    combined_score: float
    match_type: str  # "vector", "keyword", "both"


class BM25Index:
    """
    BM25 (Best Match 25) index for keyword search.

    BM25 is a probabilistic ranking function that excels at:
    - Exact term matching
    - Rare/specific terms (high IDF)
    - Precise queries like error codes, names, IDs
    """

    def __init__(
        self,
        k1: float = 1.5,  # Term frequency saturation
        b: float = 0.75,  # Length normalization
    ):
        self.k1 = k1
        self.b = b

        # Inverted index: term -> {doc_id: term_freq}
        self._index: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Document data
        self._doc_lengths: Dict[str, int] = {}
        self._doc_texts: Dict[str, str] = {}
        self._doc_metadata: Dict[str, Dict] = {}

        # Corpus stats
        self._avg_doc_length: float = 0.0
        self._num_docs: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric."""
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        return tokens

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the BM25 index."""
        tokens = self._tokenize(text)

        # Store document
        self._doc_texts[doc_id] = text
        self._doc_metadata[doc_id] = metadata or {}
        self._doc_lengths[doc_id] = len(tokens)

        # Build inverted index
        term_freqs = Counter(tokens)
        for term, freq in term_freqs.items():
            self._index[term][doc_id] = freq

        # Update corpus stats
        self._num_docs += 1
        total_length = sum(self._doc_lengths.values())
        self._avg_doc_length = total_length / self._num_docs if self._num_docs > 0 else 0

    def add_batch(self, documents: List[Dict[str, Any]]) -> int:
        """Add multiple documents efficiently."""
        for doc in documents:
            self.add_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc.get("metadata"),
            )
        return len(documents)

    def _idf(self, term: str) -> float:
        """Calculate Inverse Document Frequency."""
        doc_freq = len(self._index.get(term, {}))
        if doc_freq == 0:
            return 0.0

        # IDF formula with smoothing
        return math.log((self._num_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        if self._num_docs == 0:
            return []

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        # Calculate BM25 scores
        scores: Dict[str, float] = defaultdict(float)

        for term in query_tokens:
            if term not in self._index:
                continue

            idf = self._idf(term)

            for doc_id, term_freq in self._index[term].items():
                doc_length = self._doc_lengths[doc_id]

                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / self._avg_doc_length)
                )

                scores[doc_id] += idf * (numerator / denominator)

        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return sorted_results[:k]

    def get_document(self, doc_id: str) -> Optional[Tuple[str, Dict]]:
        """Get document text and metadata by ID."""
        if doc_id not in self._doc_texts:
            return None
        return self._doc_texts[doc_id], self._doc_metadata[doc_id]

    def count(self) -> int:
        """Get number of indexed documents."""
        return self._num_docs

    def clear(self) -> None:
        """Clear the index."""
        self._index.clear()
        self._doc_lengths.clear()
        self._doc_texts.clear()
        self._doc_metadata.clear()
        self._avg_doc_length = 0.0
        self._num_docs = 0


class HybridSearcher:
    """
    Combines vector search with BM25 for hybrid retrieval.

    Fusion strategies:
    - RRF (Reciprocal Rank Fusion): Rank-based, robust
    - Linear: Weighted score combination
    - Cascade: BM25 pre-filter, vector rerank
    """

    def __init__(
        self,
        vector_store,  # FAISSVectorStore
        embedding_model,  # EmbeddingModel
        fusion_strategy: str = "rrf",  # "rrf", "linear", "cascade"
        vector_weight: float = 0.5,  # For linear fusion
        rrf_k: int = 60,  # RRF constant
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.fusion_strategy = fusion_strategy
        self.vector_weight = vector_weight
        self.rrf_k = rrf_k

        self.bm25_index = BM25Index()

        self._synced = False

    def add_document(
        self,
        doc_id: str,
        text: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add document to both indexes."""
        # Add to vector store
        self.vector_store.add(
            doc_id=doc_id,
            vector=vector,
            text=text,
            metadata=metadata,
        )

        # Add to BM25 index
        self.bm25_index.add_document(
            doc_id=doc_id,
            text=text,
            metadata=metadata,
        )

    def add_batch(self, documents: List[Dict[str, Any]]) -> int:
        """Add multiple documents to both indexes."""
        # Add to vector store
        self.vector_store.add_batch(documents)

        # Add to BM25 index
        bm25_docs = [
            {"doc_id": d["doc_id"], "text": d["text"], "metadata": d.get("metadata")}
            for d in documents
        ]
        self.bm25_index.add_batch(bm25_docs)

        return len(documents)

    def search(
        self,
        query: str,
        k: int = 10,
        vector_k: int = 50,  # Candidates from vector search
        bm25_k: int = 50,  # Candidates from BM25
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[HybridResult]:
        """
        Perform hybrid search combining vector and BM25.

        Args:
            query: Search query
            k: Number of final results
            vector_k: Vector search candidates
            bm25_k: BM25 candidates
            filter_metadata: Optional metadata filter

        Returns:
            List of HybridResult objects
        """
        # Get query embedding
        query_vector = self.embedding_model.embed(query)

        # Vector search
        vector_results = self.vector_store.search(
            query_vector=query_vector,
            k=vector_k,
            filter_metadata=filter_metadata,
        )

        # BM25 search
        bm25_results = self.bm25_index.search(query, k=bm25_k)

        # Apply fusion
        if self.fusion_strategy == "rrf":
            return self._rrf_fusion(vector_results, bm25_results, k)
        elif self.fusion_strategy == "linear":
            return self._linear_fusion(vector_results, bm25_results, k)
        elif self.fusion_strategy == "cascade":
            return self._cascade_fusion(vector_results, bm25_results, k, query_vector)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def _rrf_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Tuple[str, float]],
        k: int,
    ) -> List[HybridResult]:
        """
        Reciprocal Rank Fusion - combines rankings, not scores.

        RRF is robust because it uses ranks instead of raw scores,
        making it insensitive to score distribution differences.
        """
        scores: Dict[str, Dict[str, Any]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            doc_id = result["doc_id"]
            if doc_id not in scores:
                scores[doc_id] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "vector_score": result["score"],
                    "bm25_score": 0.0,
                    "vector_rank": rank,
                    "bm25_rank": None,
                }
            scores[doc_id]["rrf_score"] = 1.0 / (self.rrf_k + rank)

        # Process BM25 results
        for rank, (doc_id, bm25_score) in enumerate(bm25_results, 1):
            if doc_id not in scores:
                # Get text and metadata from BM25 index
                doc_data = self.bm25_index.get_document(doc_id)
                if doc_data:
                    text, metadata = doc_data
                    scores[doc_id] = {
                        "text": text,
                        "metadata": metadata,
                        "vector_score": 0.0,
                        "bm25_score": bm25_score,
                        "vector_rank": None,
                        "bm25_rank": rank,
                        "rrf_score": 0.0,
                    }

            if doc_id in scores:
                scores[doc_id]["bm25_score"] = bm25_score
                scores[doc_id]["bm25_rank"] = rank
                scores[doc_id]["rrf_score"] = scores[doc_id].get("rrf_score", 0) + 1.0 / (self.rrf_k + rank)

        # Sort by RRF score
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True,
        )

        # Build results
        results = []
        for doc_id, data in sorted_docs[:k]:
            match_type = "both"
            if data.get("vector_rank") is None:
                match_type = "keyword"
            elif data.get("bm25_rank") is None:
                match_type = "vector"

            results.append(HybridResult(
                doc_id=doc_id,
                text=data["text"],
                metadata=data["metadata"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=data["rrf_score"],
                match_type=match_type,
            ))

        return results

    def _linear_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Tuple[str, float]],
        k: int,
    ) -> List[HybridResult]:
        """
        Linear score fusion - weighted combination of normalized scores.
        """
        scores: Dict[str, Dict[str, Any]] = {}

        # Normalize vector scores (already 0-1 for cosine)
        for result in vector_results:
            doc_id = result["doc_id"]
            scores[doc_id] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "vector_score": result["score"],
                "bm25_score": 0.0,
            }

        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results) or 1.0

            for doc_id, bm25_score in bm25_results:
                normalized_bm25 = bm25_score / max_bm25

                if doc_id not in scores:
                    doc_data = self.bm25_index.get_document(doc_id)
                    if doc_data:
                        text, metadata = doc_data
                        scores[doc_id] = {
                            "text": text,
                            "metadata": metadata,
                            "vector_score": 0.0,
                            "bm25_score": normalized_bm25,
                        }
                else:
                    scores[doc_id]["bm25_score"] = normalized_bm25

        # Calculate combined scores
        for doc_id in scores:
            vec_score = scores[doc_id]["vector_score"]
            bm25_score = scores[doc_id]["bm25_score"]
            scores[doc_id]["combined_score"] = (
                self.vector_weight * vec_score +
                (1 - self.vector_weight) * bm25_score
            )

        # Sort and build results
        sorted_docs = sorted(
            scores.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True,
        )

        results = []
        for doc_id, data in sorted_docs[:k]:
            match_type = "both"
            if data["vector_score"] == 0:
                match_type = "keyword"
            elif data["bm25_score"] == 0:
                match_type = "vector"

            results.append(HybridResult(
                doc_id=doc_id,
                text=data["text"],
                metadata=data["metadata"],
                vector_score=data["vector_score"],
                bm25_score=data["bm25_score"],
                combined_score=data["combined_score"],
                match_type=match_type,
            ))

        return results

    def _cascade_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Tuple[str, float]],
        k: int,
        query_vector: List[float],
    ) -> List[HybridResult]:
        """
        Cascade fusion - use BM25 to pre-filter, then rank by vector similarity.

        Good when you want exact term matches but semantic ranking.
        """
        # Get BM25 candidate set
        bm25_doc_ids = {doc_id for doc_id, _ in bm25_results}

        # Filter vector results to BM25 matches, but keep all if no BM25 matches
        if bm25_doc_ids:
            filtered = [r for r in vector_results if r["doc_id"] in bm25_doc_ids]
            # If no overlap, use vector results
            if not filtered:
                filtered = vector_results
        else:
            filtered = vector_results

        # Convert to HybridResult
        bm25_scores = dict(bm25_results)

        results = []
        for result in filtered[:k]:
            doc_id = result["doc_id"]
            results.append(HybridResult(
                doc_id=doc_id,
                text=result["text"],
                metadata=result["metadata"],
                vector_score=result["score"],
                bm25_score=bm25_scores.get(doc_id, 0.0),
                combined_score=result["score"],  # Use vector score as final
                match_type="both" if doc_id in bm25_doc_ids else "vector",
            ))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "vector_count": self.vector_store.count(),
            "bm25_count": self.bm25_index.count(),
            "fusion_strategy": self.fusion_strategy,
            "vector_weight": self.vector_weight,
        }

    def clear(self) -> None:
        """Clear both indexes."""
        self.vector_store.clear()
        self.bm25_index.clear()
