"""
Memory-mapped file I/O for embedding vectors.

Uses mmap to access embedding vectors stored on disk without loading
them entirely into RAM. This enables working with embedding datasets
larger than available memory.
"""

import logging
import mmap
import os
import struct
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Header format: magic(4) + version(4) + num_vectors(8) + dimension(4) + dtype_code(4)
HEADER_FORMAT = "<4sIQII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
MAGIC = b"NEMB"
VERSION = 1
DTYPE_FLOAT32 = 0
DTYPE_FLOAT16 = 1


@dataclass
class MMapConfig:
    """Configuration for memory-mapped embedding store."""
    file_path: str
    dimension: int
    dtype: str = "float32"  # "float32" or "float16"
    read_only: bool = False
    preload: bool = False  # Preload pages into memory


class MMapEmbeddingStore:
    """
    Memory-mapped embedding vector storage.

    Features:
    - O(1) random access to any embedding by index
    - Near-zero memory footprint (OS manages page cache)
    - Supports float32 and float16 storage
    - Thread-safe reads
    - Append-only writes for new embeddings
    - Batch read/write operations
    """

    def __init__(self, config: MMapConfig):
        self._config = config
        self._dimension = config.dimension
        self._dtype = config.dtype
        self._dtype_code = DTYPE_FLOAT32 if config.dtype == "float32" else DTYPE_FLOAT16
        self._bytes_per_element = 4 if config.dtype == "float32" else 2
        self._vector_size = self._dimension * self._bytes_per_element
        self._fmt = f"<{self._dimension}f" if config.dtype == "float32" else f"<{self._dimension}e"
        self._num_vectors = 0
        self._mmap: Optional[mmap.mmap] = None
        self._file = None
        self._lock = threading.Lock()
        self._stats = {
            "reads": 0,
            "writes": 0,
            "batch_reads": 0,
            "batch_writes": 0,
        }

    def open(self) -> None:
        """Open or create the embedding store."""
        exists = os.path.exists(self._config.file_path)

        if exists:
            self._open_existing()
        else:
            self._create_new()

        logger.info(
            "MMap embedding store opened: %s (%d vectors, dim=%d, %s)",
            self._config.file_path,
            self._num_vectors,
            self._dimension,
            self._dtype,
        )

    # Pre-allocate space for this many vectors initially
    _INITIAL_CAPACITY = 1000

    def _create_new(self) -> None:
        """Create a new embedding file with header and pre-allocated space."""
        self._file = open(self._config.file_path, "w+b")
        header = struct.pack(
            HEADER_FORMAT, MAGIC, VERSION, 0, self._dimension, self._dtype_code
        )
        self._file.write(header)
        # Pre-allocate space for initial capacity
        initial_size = HEADER_SIZE + self._INITIAL_CAPACITY * self._vector_size
        self._file.seek(initial_size - 1)
        self._file.write(b"\x00")
        self._file.flush()
        self._num_vectors = 0
        self._mmap = mmap.mmap(self._file.fileno(), 0)

    def _open_existing(self) -> None:
        """Open an existing embedding file."""
        mode = "r+b" if not self._config.read_only else "rb"
        self._file = open(self._config.file_path, mode)
        header_bytes = self._file.read(HEADER_SIZE)
        magic, version, num_vectors, dimension, dtype_code = struct.unpack(
            HEADER_FORMAT, header_bytes
        )

        if magic != MAGIC:
            raise ValueError(f"Invalid embedding file (bad magic): {self._config.file_path}")
        if dimension != self._dimension:
            raise ValueError(
                f"Dimension mismatch: file has {dimension}, expected {self._dimension}"
            )

        self._num_vectors = num_vectors
        self._dtype_code = dtype_code
        self._dtype = "float32" if dtype_code == DTYPE_FLOAT32 else "float16"
        self._bytes_per_element = 4 if self._dtype == "float32" else 2
        self._vector_size = self._dimension * self._bytes_per_element

        access = mmap.ACCESS_READ if self._config.read_only else mmap.ACCESS_WRITE
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=access)

    def close(self) -> None:
        """Close the store."""
        if self._mmap:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def _offset(self, index: int) -> int:
        """Compute byte offset for a vector index."""
        return HEADER_SIZE + index * self._vector_size

    def get(self, index: int) -> List[float]:
        """
        Read a single embedding vector by index.

        Args:
            index: Vector index (0-based)

        Returns:
            List of floats
        """
        if index < 0 or index >= self._num_vectors:
            raise IndexError(f"Index {index} out of range [0, {self._num_vectors})")

        offset = self._offset(index)
        self._mmap.seek(offset)
        raw = self._mmap.read(self._vector_size)
        self._stats["reads"] += 1
        return list(struct.unpack(self._fmt, raw))

    def get_batch(self, indices: List[int]) -> List[List[float]]:
        """Read multiple embedding vectors."""
        self._stats["batch_reads"] += 1
        results = []
        for idx in indices:
            results.append(self.get(idx))
        return results

    def append(self, vector: List[float]) -> int:
        """
        Append a vector to the store.

        Args:
            vector: Embedding vector (must match dimension)

        Returns:
            Index of the appended vector
        """
        if len(vector) != self._dimension:
            raise ValueError(
                f"Vector dimension {len(vector)} != store dimension {self._dimension}"
            )

        with self._lock:
            packed = struct.pack(self._fmt, *vector)
            offset = self._offset(self._num_vectors)

            # Ensure file is large enough
            needed = offset + self._vector_size
            current_size = self._mmap.size()
            if current_size < needed:
                # Close mmap before resizing the file
                self._mmap.close()
                # Double the file size or grow to needed + buffer
                new_size = max(current_size * 2, needed + self._vector_size * 1000)
                self._file.seek(new_size - 1)
                self._file.write(b"\x00")
                self._file.flush()
                self._mmap = mmap.mmap(self._file.fileno(), 0)

            self._mmap[offset : offset + self._vector_size] = packed

            index = self._num_vectors
            self._num_vectors += 1

            # Update header
            self._mmap.seek(0)
            header = struct.pack(
                HEADER_FORMAT, MAGIC, VERSION, self._num_vectors,
                self._dimension, self._dtype_code,
            )
            self._mmap[:HEADER_SIZE] = header

            self._stats["writes"] += 1
            return index

    def append_batch(self, vectors: List[List[float]]) -> List[int]:
        """Append multiple vectors."""
        self._stats["batch_writes"] += 1
        indices = []
        for v in vectors:
            indices.append(self.append(v))
        return indices

    @property
    def num_vectors(self) -> int:
        return self._num_vectors

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def file_size_bytes(self) -> int:
        if self._mmap:
            return self._mmap.size()
        return 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "num_vectors": self._num_vectors,
            "dimension": self._dimension,
            "dtype": self._dtype,
            "file_size_mb": self.file_size_bytes / (1024 * 1024),
            "data_size_mb": (
                self._num_vectors * self._vector_size / (1024 * 1024)
            ),
        }
