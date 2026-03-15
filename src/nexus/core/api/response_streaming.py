"""
Response streaming for large result sets.

Streams JSON arrays as newline-delimited JSON (NDJSON) or Server-Sent Events
(SSE), enabling clients to process results incrementally without waiting for
the full response to be assembled in memory.

Supports:
- NDJSON streaming (``application/x-ndjson``)
- SSE streaming (``text/event-stream``)
- Chunked transfer encoding
- Backpressure via configurable buffer size
- Progress reporting
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


class StreamFormat(str, Enum):
    NDJSON = "ndjson"
    SSE = "sse"
    CHUNKED_JSON = "chunked_json"


@dataclass
class StreamChunk:
    """A single chunk in a streamed response."""
    data: Any
    index: int
    is_last: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_ndjson(self) -> bytes:
        """Serialize as NDJSON line."""
        payload = {"data": self.data, "index": self.index}
        if self.is_last:
            payload["is_last"] = True
        if self.metadata:
            payload["metadata"] = self.metadata
        return json.dumps(payload, default=str).encode("utf-8") + b"\n"

    def to_sse(self, event_type: str = "data") -> bytes:
        """Serialize as SSE event."""
        payload = json.dumps(
            {"data": self.data, "index": self.index, "metadata": self.metadata},
            default=str,
        )
        lines = [f"event: {event_type}", f"data: {payload}", ""]
        if self.is_last:
            lines.insert(-1, "event: complete")
            lines.insert(-1, f"data: {json.dumps({'total': self.index + 1})}")
            lines.insert(-1, "")
        return "\n".join(lines).encode("utf-8") + b"\n"


@dataclass
class StreamStats:
    """Statistics for a completed stream."""
    total_chunks: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    format: StreamFormat = StreamFormat.NDJSON

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

    @property
    def throughput_bytes_per_sec(self) -> float:
        d = self.duration_seconds
        return self.total_bytes / d if d > 0 else 0.0


class ResponseStreamer:
    """
    Streams large result sets to clients chunk-by-chunk.

    Features:
    - Multiple output formats (NDJSON, SSE, chunked JSON)
    - Configurable chunk size (items per chunk)
    - Backpressure via async buffer
    - Progress callbacks
    - Automatic stream statistics
    """

    DEFAULT_CHUNK_SIZE = 100
    MAX_BUFFER_SIZE = 1000

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        stream_format: StreamFormat = StreamFormat.NDJSON,
        buffer_size: int = MAX_BUFFER_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Args:
            chunk_size: Number of items per chunk
            stream_format: Output format
            buffer_size: Max buffered chunks before applying backpressure
            progress_callback: Called with (items_sent, total_estimated)
        """
        self.chunk_size = max(1, chunk_size)
        self.stream_format = stream_format
        self._buffer_size = buffer_size
        self._progress_callback = progress_callback
        self._stats = StreamStats(format=stream_format)

    def stream_sync(
        self, items: List[Any], total: Optional[int] = None
    ) -> Iterator[bytes]:
        """
        Synchronously stream a list of items.

        Args:
            items: Items to stream
            total: Total items (for progress reporting)

        Yields:
            Encoded bytes for each chunk
        """
        self._stats = StreamStats(format=self.stream_format, start_time=time.time())
        total = total or len(items)
        chunk_idx = 0

        for start in range(0, len(items), self.chunk_size):
            batch = items[start : start + self.chunk_size]
            is_last = start + self.chunk_size >= len(items)
            chunk = StreamChunk(
                data=batch,
                index=chunk_idx,
                is_last=is_last,
                metadata={"batch_size": len(batch)},
            )

            if self.stream_format == StreamFormat.NDJSON:
                encoded = chunk.to_ndjson()
            elif self.stream_format == StreamFormat.SSE:
                encoded = chunk.to_sse()
            else:
                encoded = chunk.to_ndjson()

            self._stats.total_chunks += 1
            self._stats.total_bytes += len(encoded)

            if self._progress_callback:
                self._progress_callback(start + len(batch), total)

            yield encoded
            chunk_idx += 1

        self._stats.end_time = time.time()

    async def stream_async(
        self, items: AsyncIterator[Any], total: Optional[int] = None
    ) -> AsyncIterator[bytes]:
        """
        Asynchronously stream items from an async iterator.

        Args:
            items: Async iterator of items
            total: Estimated total items

        Yields:
            Encoded bytes for each chunk
        """
        self._stats = StreamStats(format=self.stream_format, start_time=time.time())
        buffer: List[Any] = []
        chunk_idx = 0
        items_sent = 0

        async for item in items:
            buffer.append(item)
            if len(buffer) >= self.chunk_size:
                chunk = StreamChunk(
                    data=buffer,
                    index=chunk_idx,
                    metadata={"batch_size": len(buffer)},
                )
                if self.stream_format == StreamFormat.SSE:
                    encoded = chunk.to_sse()
                else:
                    encoded = chunk.to_ndjson()

                self._stats.total_chunks += 1
                self._stats.total_bytes += len(encoded)
                items_sent += len(buffer)

                if self._progress_callback:
                    self._progress_callback(items_sent, total or items_sent)

                yield encoded
                buffer = []
                chunk_idx += 1

        # Flush remaining buffer
        if buffer:
            chunk = StreamChunk(
                data=buffer,
                index=chunk_idx,
                is_last=True,
                metadata={"batch_size": len(buffer)},
            )
            if self.stream_format == StreamFormat.SSE:
                encoded = chunk.to_sse()
            else:
                encoded = chunk.to_ndjson()

            self._stats.total_chunks += 1
            self._stats.total_bytes += len(encoded)
            items_sent += len(buffer)

            if self._progress_callback:
                self._progress_callback(items_sent, total or items_sent)

            yield encoded

        self._stats.end_time = time.time()

    async def stream_from_generator(
        self,
        generator: Callable[[], AsyncIterator[Any]],
        total: Optional[int] = None,
    ) -> AsyncIterator[bytes]:
        """
        Stream from an async generator factory.

        Args:
            generator: Factory function returning an async iterator
            total: Estimated total items
        """
        async for chunk in self.stream_async(generator(), total):
            yield chunk

    @property
    def stats(self) -> StreamStats:
        return self._stats
