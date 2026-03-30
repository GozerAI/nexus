"""
Request/response compression middleware.

Compresses API responses and supports compressed request bodies,
reducing network bandwidth and transfer time for large payloads.

Supports gzip, deflate, and zstandard (optional) compression.
"""

import gzip
import io
import logging
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class CompressionAlgorithm(str, Enum):
    GZIP = "gzip"
    DEFLATE = "deflate"
    ZSTD = "zstd"
    NONE = "none"


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    original_size: int
    compressed_size: int
    algorithm: CompressionAlgorithm
    compression_time_ms: float

    @property
    def ratio(self) -> float:
        return self.compressed_size / max(self.original_size, 1)

    @property
    def savings_percent(self) -> float:
        return (1.0 - self.ratio) * 100


class CompressionMiddleware:
    """
    Transparent compression for API request/response bodies.

    Features:
    - Automatic algorithm selection based on Accept-Encoding
    - Configurable minimum size threshold (skip small payloads)
    - Configurable compression level
    - Decompression of compressed request bodies
    - Statistics tracking
    - Content-type awareness (skip already-compressed types)
    """

    # Content types that are already compressed
    SKIP_TYPES = {
        "image/png", "image/jpeg", "image/gif", "image/webp",
        "application/zip", "application/gzip", "application/pdf",
        "video/mp4", "audio/mp3",
    }

    DEFAULT_MIN_SIZE = 1024  # Don't compress below 1KB
    DEFAULT_LEVEL = 6        # gzip level 6 (good balance)

    def __init__(
        self,
        min_size: int = DEFAULT_MIN_SIZE,
        compression_level: int = DEFAULT_LEVEL,
        preferred_algorithm: CompressionAlgorithm = CompressionAlgorithm.GZIP,
    ):
        """
        Args:
            min_size: Minimum payload size to compress
            compression_level: Compression level (1-9)
            preferred_algorithm: Default algorithm when client accepts multiple
        """
        self._min_size = min_size
        self._level = compression_level
        self._preferred = preferred_algorithm
        self._stats = {
            "compressed": 0,
            "decompressed": 0,
            "skipped": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "total_compress_ms": 0.0,
            "total_decompress_ms": 0.0,
        }

    def select_algorithm(
        self, accept_encoding: str = ""
    ) -> CompressionAlgorithm:
        """
        Select best compression algorithm from Accept-Encoding header.

        Args:
            accept_encoding: Client's Accept-Encoding header value

        Returns:
            Best available algorithm
        """
        if not accept_encoding:
            return CompressionAlgorithm.NONE

        accept = accept_encoding.lower()
        if "zstd" in accept and self._has_zstd():
            return CompressionAlgorithm.ZSTD
        if "gzip" in accept:
            return CompressionAlgorithm.GZIP
        if "deflate" in accept:
            return CompressionAlgorithm.DEFLATE
        return CompressionAlgorithm.NONE

    @staticmethod
    def _has_zstd() -> bool:
        try:
            import zstandard  # noqa: F401
            return True
        except ImportError:
            return False

    def compress(
        self,
        data: bytes,
        algorithm: Optional[CompressionAlgorithm] = None,
        content_type: str = "",
    ) -> Tuple[bytes, CompressionResult]:
        """
        Compress data.

        Args:
            data: Raw bytes to compress
            algorithm: Compression algorithm (None = use preferred)
            content_type: Content-Type for skip detection

        Returns:
            (compressed_bytes, CompressionResult)
        """
        algo = algorithm or self._preferred
        original_size = len(data)

        # Skip small payloads
        if original_size < self._min_size:
            self._stats["skipped"] += 1
            return data, CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                algorithm=CompressionAlgorithm.NONE,
                compression_time_ms=0,
            )

        # Skip already-compressed content types
        if content_type in self.SKIP_TYPES:
            self._stats["skipped"] += 1
            return data, CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                algorithm=CompressionAlgorithm.NONE,
                compression_time_ms=0,
            )

        start = time.time()

        if algo == CompressionAlgorithm.GZIP:
            compressed = gzip.compress(data, compresslevel=self._level)
        elif algo == CompressionAlgorithm.DEFLATE:
            compressed = zlib.compress(data, self._level)
        elif algo == CompressionAlgorithm.ZSTD:
            try:
                import zstandard as zstd
                cctx = zstd.ZstdCompressor(level=self._level)
                compressed = cctx.compress(data)
            except ImportError:
                compressed = gzip.compress(data, compresslevel=self._level)
                algo = CompressionAlgorithm.GZIP
        else:
            return data, CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                algorithm=CompressionAlgorithm.NONE,
                compression_time_ms=0,
            )

        elapsed_ms = (time.time() - start) * 1000
        compressed_size = len(compressed)

        # Don't use compression if it makes payload bigger
        if compressed_size >= original_size:
            self._stats["skipped"] += 1
            return data, CompressionResult(
                original_size=original_size,
                compressed_size=original_size,
                algorithm=CompressionAlgorithm.NONE,
                compression_time_ms=elapsed_ms,
            )

        self._stats["compressed"] += 1
        self._stats["total_original_bytes"] += original_size
        self._stats["total_compressed_bytes"] += compressed_size
        self._stats["total_compress_ms"] += elapsed_ms

        return compressed, CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            algorithm=algo,
            compression_time_ms=elapsed_ms,
        )

    def decompress(
        self, data: bytes, algorithm: CompressionAlgorithm
    ) -> bytes:
        """
        Decompress data.

        Args:
            data: Compressed bytes
            algorithm: Algorithm used for compression

        Returns:
            Decompressed bytes
        """
        start = time.time()

        if algorithm == CompressionAlgorithm.GZIP:
            result = gzip.decompress(data)
        elif algorithm == CompressionAlgorithm.DEFLATE:
            result = zlib.decompress(data)
        elif algorithm == CompressionAlgorithm.ZSTD:
            try:
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                result = dctx.decompress(data)
            except ImportError:
                raise ValueError("zstandard library not installed")
        else:
            return data

        elapsed_ms = (time.time() - start) * 1000
        self._stats["decompressed"] += 1
        self._stats["total_decompress_ms"] += elapsed_ms

        return result

    def detect_encoding(self, content_encoding: str) -> CompressionAlgorithm:
        """Detect compression algorithm from Content-Encoding header."""
        encoding = content_encoding.lower().strip()
        if encoding == "gzip":
            return CompressionAlgorithm.GZIP
        elif encoding == "deflate":
            return CompressionAlgorithm.DEFLATE
        elif encoding == "zstd":
            return CompressionAlgorithm.ZSTD
        return CompressionAlgorithm.NONE

    def get_stats(self) -> Dict[str, Any]:
        compressed = self._stats["compressed"]
        return {
            **self._stats,
            "avg_compression_ratio": (
                self._stats["total_compressed_bytes"]
                / max(self._stats["total_original_bytes"], 1)
            ),
            "avg_compress_ms": (
                self._stats["total_compress_ms"] / max(compressed, 1)
            ),
            "bandwidth_saved_bytes": (
                self._stats["total_original_bytes"]
                - self._stats["total_compressed_bytes"]
            ),
        }
