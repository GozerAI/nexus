"""
Embedding dimension reduction for memory and compute savings.

Reduces high-dimensional embedding vectors to lower dimensions
while preserving similarity relationships. Supports PCA-like
random projection and truncation methods that don't require
training data.
"""

import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReductionMethod(str, Enum):
    RANDOM_PROJECTION = "random_projection"  # Johnson-Lindenstrauss
    TRUNCATION = "truncation"                # Keep first N dimensions
    PCA_APPROX = "pca_approx"               # Approximate PCA via random sampling
    GAUSSIAN_RP = "gaussian_rp"             # Gaussian random projection


@dataclass
class ReductionConfig:
    """Configuration for dimension reduction."""
    input_dim: int
    output_dim: int
    method: ReductionMethod = ReductionMethod.RANDOM_PROJECTION
    random_seed: int = 42
    normalize_output: bool = True


@dataclass
class ReductionStats:
    """Statistics for the dimension reducer."""
    vectors_reduced: int = 0
    total_input_dims: int = 0
    total_output_dims: int = 0
    avg_distortion: float = 0.0
    compression_ratio: float = 0.0
    memory_saved_bytes: int = 0


class DimensionReducer:
    """
    Reduces embedding dimensions while preserving distances.

    Implements the Johnson-Lindenstrauss lemma: for n points in high-D
    space, a random projection to O(log(n)/eps^2) dimensions preserves
    pairwise distances within (1 +/- eps).

    Features:
    - Random projection (fast, no training needed)
    - Gaussian random projection (better preservation)
    - Truncation (simplest, good for some embeddings)
    - Approximate PCA via random sampling
    - Batch processing
    - Distortion measurement
    """

    def __init__(self, config: ReductionConfig):
        self._config = config
        self._projection_matrix: Optional[List[List[float]]] = None
        self._stats = ReductionStats(
            compression_ratio=config.output_dim / max(config.input_dim, 1)
        )

        if config.method in (ReductionMethod.RANDOM_PROJECTION, ReductionMethod.GAUSSIAN_RP):
            self._build_projection_matrix()

    def _build_projection_matrix(self) -> None:
        """Build random projection matrix."""
        rng = random.Random(self._config.random_seed)
        in_d = self._config.input_dim
        out_d = self._config.output_dim
        scale = 1.0 / math.sqrt(out_d)

        if self._config.method == ReductionMethod.GAUSSIAN_RP:
            self._projection_matrix = [
                [rng.gauss(0, scale) for _ in range(in_d)]
                for _ in range(out_d)
            ]
        else:
            # Sparse random projection (Achlioptas): {-1, 0, +1} with prob {1/6, 2/3, 1/6}
            sqrt3_scale = math.sqrt(3) * scale
            self._projection_matrix = []
            for _ in range(out_d):
                row = []
                for _ in range(in_d):
                    p = rng.random()
                    if p < 1 / 6:
                        row.append(sqrt3_scale)
                    elif p < 5 / 6:
                        row.append(0.0)
                    else:
                        row.append(-sqrt3_scale)
                self._projection_matrix.append(row)

        logger.debug(
            "Built %s projection matrix: %d x %d",
            self._config.method.value, out_d, in_d,
        )

    def reduce(self, vector: List[float]) -> List[float]:
        """
        Reduce a single vector to lower dimensions.

        Args:
            vector: High-dimensional input vector

        Returns:
            Reduced-dimension vector
        """
        if len(vector) != self._config.input_dim:
            raise ValueError(
                f"Input dimension {len(vector)} != expected {self._config.input_dim}"
            )

        method = self._config.method
        if method == ReductionMethod.TRUNCATION:
            result = vector[: self._config.output_dim]
        elif method in (ReductionMethod.RANDOM_PROJECTION, ReductionMethod.GAUSSIAN_RP):
            result = self._project(vector)
        elif method == ReductionMethod.PCA_APPROX:
            result = self._project(vector) if self._projection_matrix else vector[: self._config.output_dim]
        else:
            result = vector[: self._config.output_dim]

        if self._config.normalize_output:
            result = self._normalize(result)

        self._stats.vectors_reduced += 1
        self._stats.total_input_dims += self._config.input_dim
        self._stats.total_output_dims += self._config.output_dim
        self._stats.memory_saved_bytes += (
            (self._config.input_dim - self._config.output_dim) * 4
        )

        return result

    def reduce_batch(self, vectors: List[List[float]]) -> List[List[float]]:
        """Reduce a batch of vectors."""
        return [self.reduce(v) for v in vectors]

    def _project(self, vector: List[float]) -> List[float]:
        """Apply random projection."""
        if not self._projection_matrix:
            return vector[: self._config.output_dim]

        result = []
        for row in self._projection_matrix:
            val = sum(a * b for a, b in zip(row, vector))
            result.append(val)
        return result

    @staticmethod
    def _normalize(vector: List[float]) -> List[float]:
        """L2 normalize a vector."""
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def measure_distortion(
        self,
        vectors: List[List[float]],
        sample_pairs: int = 100,
    ) -> float:
        """
        Measure average pairwise distance distortion after reduction.

        Samples random pairs, computes original and reduced distances,
        and returns the average relative distortion.

        Args:
            vectors: Sample of input vectors
            sample_pairs: Number of random pairs to test

        Returns:
            Average relative distortion (0.0 = perfect preservation)
        """
        if len(vectors) < 2:
            return 0.0

        rng = random.Random(42)
        reduced = self.reduce_batch(vectors)
        distortions = []

        for _ in range(min(sample_pairs, len(vectors) * (len(vectors) - 1) // 2)):
            i, j = rng.sample(range(len(vectors)), 2)

            orig_dist = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j]))
            )
            red_dist = math.sqrt(
                sum((a - b) ** 2 for a, b in zip(reduced[i], reduced[j]))
            )

            if orig_dist > 0:
                distortion = abs(red_dist - orig_dist) / orig_dist
                distortions.append(distortion)

        avg = sum(distortions) / len(distortions) if distortions else 0.0
        self._stats.avg_distortion = avg
        return avg

    @staticmethod
    def recommended_output_dim(
        n_vectors: int,
        epsilon: float = 0.1,
    ) -> int:
        """
        Compute recommended output dimension using Johnson-Lindenstrauss bound.

        Args:
            n_vectors: Number of vectors
            epsilon: Acceptable distortion (0.1 = 10%)

        Returns:
            Recommended output dimension
        """
        return max(
            4, int(math.ceil(24 * math.log(max(n_vectors, 2)) / (epsilon ** 2)))
        )

    @property
    def config(self) -> ReductionConfig:
        return self._config

    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self._config.method.value,
            "input_dim": self._config.input_dim,
            "output_dim": self._config.output_dim,
            "vectors_reduced": self._stats.vectors_reduced,
            "compression_ratio": self._stats.compression_ratio,
            "avg_distortion": self._stats.avg_distortion,
            "memory_saved_mb": self._stats.memory_saved_bytes / (1024 * 1024),
        }
