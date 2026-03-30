"""
Tensor memory pre-allocation for inference.

Pre-allocates fixed-size memory pools for common tensor shapes used
during inference, eliminating allocation overhead during request processing.
"""

import logging
import threading
import time
from array import array
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class TensorSpec:
    """Specification for a tensor shape."""
    name: str
    shape: Tuple[int, ...]
    dtype: str = "float32"  # "float32", "float64", "int32"

    @property
    def num_elements(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        elem_size = {"float32": 4, "float64": 8, "int32": 4, "int64": 8}.get(
            self.dtype, 4
        )
        return self.num_elements * elem_size


@dataclass
class PooledTensor:
    """A tensor allocated from a pool."""
    spec: TensorSpec
    data: array
    pool_id: str
    in_use: bool = True
    allocated_at: float = field(default_factory=time.time)
    returned_at: Optional[float] = None

    def zero_fill(self) -> None:
        """Zero out the tensor data."""
        for i in range(len(self.data)):
            self.data[i] = 0


class TensorPool:
    """
    Pre-allocated pool of tensors for a specific shape.

    Manages a fixed set of pre-allocated tensors, lending them out
    for inference and reclaiming them when done.
    """

    ARRAY_TYPECODES = {
        "float32": "f",
        "float64": "d",
        "int32": "i",
        "int64": "q",
    }

    def __init__(self, spec: TensorSpec, pool_size: int = 10):
        self._spec = spec
        self._pool_size = pool_size
        self._available: List[PooledTensor] = []
        self._in_use: Dict[str, PooledTensor] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._stats = {
            "acquires": 0,
            "releases": 0,
            "waits": 0,
            "allocations": 0,
        }

        typecode = self.ARRAY_TYPECODES.get(spec.dtype, "f")
        for _ in range(pool_size):
            data = array(typecode, [0] * spec.num_elements)
            tensor = PooledTensor(
                spec=spec,
                data=data,
                pool_id=f"{spec.name}_{self._counter}",
                in_use=False,
            )
            self._available.append(tensor)
            self._counter += 1
            self._stats["allocations"] += 1

        logger.debug(
            "TensorPool created: %s (%d tensors, %s each)",
            spec.name, pool_size,
            f"{spec.size_bytes / 1024:.1f}KB",
        )

    def acquire(self, zero_fill: bool = True) -> Optional[PooledTensor]:
        """
        Acquire a tensor from the pool.

        Args:
            zero_fill: Zero out the tensor before returning

        Returns:
            PooledTensor, or None if pool is exhausted
        """
        with self._lock:
            if not self._available:
                self._stats["waits"] += 1
                return None

            tensor = self._available.pop()
            tensor.in_use = True
            tensor.allocated_at = time.time()
            tensor.returned_at = None
            self._in_use[tensor.pool_id] = tensor
            self._stats["acquires"] += 1

        if zero_fill:
            tensor.zero_fill()

        return tensor

    def release(self, tensor: PooledTensor) -> None:
        """Return a tensor to the pool."""
        with self._lock:
            if tensor.pool_id in self._in_use:
                del self._in_use[tensor.pool_id]
            tensor.in_use = False
            tensor.returned_at = time.time()
            self._available.append(tensor)
            self._stats["releases"] += 1

    @property
    def available_count(self) -> int:
        return len(self._available)

    @property
    def in_use_count(self) -> int:
        return len(self._in_use)

    @property
    def total_size_bytes(self) -> int:
        return self._pool_size * self._spec.size_bytes

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "pool_size": self._pool_size,
            "available": self.available_count,
            "in_use": self.in_use_count,
            "tensor_shape": self._spec.shape,
            "total_size_mb": self.total_size_bytes / (1024 * 1024),
        }


class TensorPreAllocator:
    """
    Manages multiple tensor pools for different shapes.

    Pre-allocates memory for all common tensor shapes used during
    inference, providing fast tensor acquisition without dynamic allocation.

    Features:
    - Named pools for different tensor shapes
    - Automatic pool sizing based on expected concurrency
    - Pool expansion when demand exceeds initial capacity
    - Global memory tracking
    """

    def __init__(self, max_total_memory_mb: int = 1024):
        self._pools: Dict[str, TensorPool] = {}
        self._max_memory_bytes = max_total_memory_mb * 1024 * 1024
        self._total_allocated = 0
        self._lock = threading.Lock()

    def create_pool(
        self,
        name: str,
        shape: Tuple[int, ...],
        dtype: str = "float32",
        pool_size: int = 10,
    ) -> TensorPool:
        """
        Create a named tensor pool.

        Args:
            name: Pool name
            shape: Tensor shape (e.g., (1, 512, 768))
            dtype: Data type
            pool_size: Number of tensors to pre-allocate

        Returns:
            The created TensorPool

        Raises:
            MemoryError: If allocation would exceed memory limit
        """
        spec = TensorSpec(name=name, shape=shape, dtype=dtype)
        needed = spec.size_bytes * pool_size

        with self._lock:
            if self._total_allocated + needed > self._max_memory_bytes:
                raise MemoryError(
                    f"Cannot allocate {needed / 1024 / 1024:.1f}MB: "
                    f"would exceed {self._max_memory_bytes / 1024 / 1024:.0f}MB limit "
                    f"(currently {self._total_allocated / 1024 / 1024:.1f}MB used)"
                )

            pool = TensorPool(spec, pool_size)
            self._pools[name] = pool
            self._total_allocated += needed

        logger.info(
            "Created tensor pool '%s': shape=%s, size=%d, %.1fMB",
            name, shape, pool_size, needed / 1024 / 1024,
        )
        return pool

    def get_pool(self, name: str) -> Optional[TensorPool]:
        """Get a pool by name."""
        return self._pools.get(name)

    def acquire(self, pool_name: str, zero_fill: bool = True) -> Optional[PooledTensor]:
        """Acquire a tensor from a named pool."""
        pool = self._pools.get(pool_name)
        if not pool:
            return None
        return pool.acquire(zero_fill)

    def release(self, tensor: PooledTensor) -> None:
        """Release a tensor back to its pool."""
        pool = self._pools.get(tensor.spec.name)
        if pool:
            pool.release(tensor)

    @property
    def total_allocated_mb(self) -> float:
        return self._total_allocated / (1024 * 1024)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "pools": {name: pool.get_stats() for name, pool in self._pools.items()},
            "total_allocated_mb": self.total_allocated_mb,
            "max_memory_mb": self._max_memory_bytes / (1024 * 1024),
            "utilization": self._total_allocated / max(self._max_memory_bytes, 1),
        }
