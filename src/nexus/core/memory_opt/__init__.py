"""Memory optimization modules for Nexus."""

from nexus.core.memory_opt.mmap_embeddings import MMapEmbeddingStore, MMapConfig
from nexus.core.memory_opt.tensor_prealloc import TensorPreAllocator, TensorPool
from nexus.core.memory_opt.sparse_vectors import SparseVector, SparseVectorStore
from nexus.core.memory_opt.memory_profiler import MemoryProfiler, MemorySnapshot
from nexus.core.memory_opt.dimension_reduction import DimensionReducer, ReductionMethod
from nexus.core.memory_opt.auto_unloader import AutoModelUnloader, MemoryPressureMonitor

__all__ = [
    "MMapEmbeddingStore",
    "MMapConfig",
    "TensorPreAllocator",
    "TensorPool",
    "SparseVector",
    "SparseVectorStore",
    "MemoryProfiler",
    "MemorySnapshot",
    "DimensionReducer",
    "ReductionMethod",
    "AutoModelUnloader",
    "MemoryPressureMonitor",
]
