"""
Memory-efficient sparse vector representation.

For high-dimensional embedding spaces where most dimensions are zero
(e.g., TF-IDF, sparse attention), stores only non-zero elements.
Achieves 10-100x memory savings vs dense representation.
"""

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SparseVector:
    """
    Memory-efficient sparse vector using index-value pairs.

    Only stores non-zero elements, significantly reducing memory
    for high-dimensional sparse data.
    """

    __slots__ = ("_indices", "_values", "_dimension")

    def __init__(
        self,
        indices: Optional[List[int]] = None,
        values: Optional[List[float]] = None,
        dimension: int = 0,
    ):
        """
        Args:
            indices: Sorted list of non-zero indices
            values: Corresponding non-zero values
            dimension: Total vector dimension (0 = auto from max index)
        """
        self._indices = list(indices or [])
        self._values = list(values or [])
        if len(self._indices) != len(self._values):
            raise ValueError("Indices and values must have same length")
        self._dimension = dimension or (max(self._indices) + 1 if self._indices else 0)

    @classmethod
    def from_dense(cls, dense: List[float], threshold: float = 1e-8) -> "SparseVector":
        """Create a sparse vector from a dense vector, dropping near-zero values."""
        indices = []
        values = []
        for i, v in enumerate(dense):
            if abs(v) > threshold:
                indices.append(i)
                values.append(v)
        return cls(indices=indices, values=values, dimension=len(dense))

    @classmethod
    def from_dict(cls, mapping: Dict[int, float], dimension: int = 0) -> "SparseVector":
        """Create from index-to-value mapping."""
        sorted_items = sorted(mapping.items())
        indices = [i for i, _ in sorted_items]
        values = [v for _, v in sorted_items]
        return cls(indices=indices, values=values, dimension=dimension)

    def to_dense(self) -> List[float]:
        """Convert to a dense vector."""
        dense = [0.0] * self._dimension
        for idx, val in zip(self._indices, self._values):
            if idx < self._dimension:
                dense[idx] = val
        return dense

    def to_dict(self) -> Dict[int, float]:
        """Convert to index-to-value mapping."""
        return dict(zip(self._indices, self._values))

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self._indices)

    @property
    def sparsity(self) -> float:
        """Fraction of zero elements."""
        return 1.0 - (self.nnz / max(self._dimension, 1))

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        # Each entry: 4 bytes (int index) + 8 bytes (float value)
        return self.nnz * 12 + 32  # + overhead

    @property
    def dense_memory_bytes(self) -> int:
        """Memory of equivalent dense representation."""
        return self._dimension * 8 + 32

    @property
    def compression_ratio(self) -> float:
        return self.dense_memory_bytes / max(self.memory_bytes, 1)

    def dot(self, other: "SparseVector") -> float:
        """Sparse dot product with another sparse vector."""
        result = 0.0
        i, j = 0, 0
        while i < len(self._indices) and j < len(other._indices):
            if self._indices[i] == other._indices[j]:
                result += self._values[i] * other._values[j]
                i += 1
                j += 1
            elif self._indices[i] < other._indices[j]:
                i += 1
            else:
                j += 1
        return result

    def norm(self) -> float:
        """L2 norm."""
        return math.sqrt(sum(v * v for v in self._values))

    def cosine_similarity(self, other: "SparseVector") -> float:
        """Cosine similarity with another sparse vector."""
        d = self.dot(other)
        n1 = self.norm()
        n2 = other.norm()
        if n1 == 0 or n2 == 0:
            return 0.0
        return d / (n1 * n2)

    def add(self, other: "SparseVector") -> "SparseVector":
        """Element-wise addition."""
        result: Dict[int, float] = {}
        for idx, val in zip(self._indices, self._values):
            result[idx] = result.get(idx, 0.0) + val
        for idx, val in zip(other._indices, other._values):
            result[idx] = result.get(idx, 0.0) + val
        dim = max(self._dimension, other._dimension)
        return SparseVector.from_dict(result, dimension=dim)

    def scale(self, scalar: float) -> "SparseVector":
        """Scalar multiplication."""
        return SparseVector(
            indices=list(self._indices),
            values=[v * scalar for v in self._values],
            dimension=self._dimension,
        )

    def top_k(self, k: int) -> List[Tuple[int, float]]:
        """Return top-k elements by absolute value."""
        pairs = list(zip(self._indices, self._values))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs[:k]

    def __getitem__(self, index: int) -> float:
        """Get value at index."""
        try:
            pos = self._indices.index(index)
            return self._values[pos]
        except ValueError:
            return 0.0

    def __len__(self) -> int:
        return self._dimension

    def __repr__(self) -> str:
        return f"SparseVector(dim={self._dimension}, nnz={self.nnz}, sparsity={self.sparsity:.2%})"


class SparseVectorStore:
    """
    In-memory store for collections of sparse vectors.

    Features:
    - Stores named sparse vectors
    - Batch similarity search
    - Memory usage tracking
    - Bulk import/export
    """

    def __init__(self):
        self._vectors: Dict[str, SparseVector] = {}
        self._stats = {"inserts": 0, "lookups": 0, "searches": 0}

    def put(self, key: str, vector: SparseVector) -> None:
        """Store a sparse vector."""
        self._vectors[key] = vector
        self._stats["inserts"] += 1

    def get(self, key: str) -> Optional[SparseVector]:
        """Retrieve a sparse vector by key."""
        self._stats["lookups"] += 1
        return self._vectors.get(key)

    def remove(self, key: str) -> bool:
        """Remove a vector."""
        return self._vectors.pop(key, None) is not None

    def search(
        self, query: SparseVector, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find the most similar vectors to a query.

        Args:
            query: Query sparse vector
            top_k: Number of results

        Returns:
            List of (key, similarity_score) tuples, sorted by score desc
        """
        self._stats["searches"] += 1
        scores = []
        for key, vec in self._vectors.items():
            sim = query.cosine_similarity(vec)
            scores.append((key, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @property
    def size(self) -> int:
        return len(self._vectors)

    @property
    def total_memory_bytes(self) -> int:
        return sum(v.memory_bytes for v in self._vectors.values())

    @property
    def equivalent_dense_bytes(self) -> int:
        return sum(v.dense_memory_bytes for v in self._vectors.values())

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "stored_vectors": len(self._vectors),
            "total_memory_mb": self.total_memory_bytes / (1024 * 1024),
            "equivalent_dense_mb": self.equivalent_dense_bytes / (1024 * 1024),
            "compression_ratio": (
                self.equivalent_dense_bytes / max(self.total_memory_bytes, 1)
            ),
        }
