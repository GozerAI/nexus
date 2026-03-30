"""
Response streaming for large result sets.

Streams results as newline-delimited JSON (NDJSON) or Server-Sent Events
(SSE), allowing clients to begin processing data before the full
response is generated.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class StreamFormat(str, Enum):
    NDJSON = "ndjson"
    SSE = "sse"
    JSONL = "jsonl"


@dataclass
class StreamChunk:
    """A single chunk in a streamed response."""
    data: Any
    index: int = 0
    chunk_type: str = "data"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_ndjson(self) -> str:
        payload = {"type": self.chunk_type, "index": self.index, "data": self.data}
        if self.metadata:
            payload["metadata"] = self.metadata
        return json.dumps(payload, default=str) + "\n"

    def to_sse(self, event: str = "message") -> str:
        payload = json.dumps(
            {"type": self.chunk_type, "index": self.index, "data": self.data}, default=str
        )
        return f"event: {event}\ndata: {payload}\n\n"


@dataclass
class StreamMetrics:
    """Metrics for a streaming session."""
    stream_id: str
    started_at: float
    chunks_sent: int = 0
    bytes_sent: int = 0
    completed: bool = False
    duration_seconds: float = 0.0
    error: Optional[str] = None


class ResponseStreamer:
    """
    Streams large result sets to clients incrementally.

    Supports NDJSON and SSE output formats, with backpressure handling,
    chunking of large collections, and stream lifecycle management.
    """

    def __init__(self, chunk_size=100, max_stream_seconds=300.0, flush_interval_seconds=0.1):
        self._chunk_size = chunk_size
        self._max_stream_seconds = max_stream_seconds
        self._flush_interval = flush_interval_seconds
        self._active_streams: Dict[str, StreamMetrics] = {}
        self._stats = {"streams_started": 0, "streams_completed": 0, "total_chunks": 0}

    async def stream_items(self, items, fmt=None, transform=None):
        """Stream a list of items as chunks. Yields encoded bytes."""
        if fmt is None:
            fmt = StreamFormat.NDJSON
        stream_id = uuid.uuid4().hex[:12]
        metrics = StreamMetrics(stream_id=stream_id, started_at=time.time())
        self._active_streams[stream_id] = metrics
        self._stats["streams_started"] += 1

        try:
            header = StreamChunk(
                data={"total_items": len(items), "chunk_size": self._chunk_size},
                index=0, chunk_type="header",
            )
            yield self._format_chunk(header, fmt)
            metrics.chunks_sent += 1

            for i in range(0, len(items), self._chunk_size):
                elapsed = time.time() - metrics.started_at
                if elapsed > self._max_stream_seconds:
                    timeout_chunk = StreamChunk(
                        data={"reason": "stream_timeout", "items_sent": i},
                        index=metrics.chunks_sent, chunk_type="timeout",
                    )
                    yield self._format_chunk(timeout_chunk, fmt)
                    break

                batch = items[i : i + self._chunk_size]
                if transform:
                    batch = [transform(item) for item in batch]

                chunk = StreamChunk(
                    data=batch, index=metrics.chunks_sent, chunk_type="data",
                    metadata={"offset": i, "count": len(batch)},
                )
                encoded = self._format_chunk(chunk, fmt)
                metrics.chunks_sent += 1
                metrics.bytes_sent += len(encoded)
                self._stats["total_chunks"] += 1
                yield encoded

                if self._flush_interval > 0:
                    await asyncio.sleep(self._flush_interval)

            trailer = StreamChunk(
                data={
                    "total_chunks": metrics.chunks_sent,
                    "total_bytes": metrics.bytes_sent,
                    "duration_seconds": round(time.time() - metrics.started_at, 3),
                },
                index=metrics.chunks_sent, chunk_type="trailer",
            )
            yield self._format_chunk(trailer, fmt)
            metrics.completed = True
            self._stats["streams_completed"] += 1

        except Exception as exc:
            metrics.error = str(exc)
            error_chunk = StreamChunk(data={"error": str(exc)}, index=metrics.chunks_sent, chunk_type="error")
            yield self._format_chunk(error_chunk, fmt)
            raise
        finally:
            metrics.duration_seconds = time.time() - metrics.started_at
            self._active_streams.pop(stream_id, None)

    async def stream_generator(self, generator, fmt=None, buffer_size=0):
        """Stream items from an async generator. Yields encoded bytes."""
        if fmt is None:
            fmt = StreamFormat.NDJSON
        stream_id = uuid.uuid4().hex[:12]
        metrics = StreamMetrics(stream_id=stream_id, started_at=time.time())
        self._active_streams[stream_id] = metrics
        self._stats["streams_started"] += 1
        buffer = []

        try:
            async for item in generator:
                elapsed = time.time() - metrics.started_at
                if elapsed > self._max_stream_seconds:
                    break
                if buffer_size > 0:
                    buffer.append(item)
                    if len(buffer) >= buffer_size:
                        chunk = StreamChunk(data=buffer, index=metrics.chunks_sent, chunk_type="data",
                                            metadata={"count": len(buffer)})
                        encoded = self._format_chunk(chunk, fmt)
                        metrics.chunks_sent += 1
                        metrics.bytes_sent += len(encoded)
                        self._stats["total_chunks"] += 1
                        yield encoded
                        buffer = []
                else:
                    chunk = StreamChunk(data=item, index=metrics.chunks_sent, chunk_type="data")
                    encoded = self._format_chunk(chunk, fmt)
                    metrics.chunks_sent += 1
                    metrics.bytes_sent += len(encoded)
                    self._stats["total_chunks"] += 1
                    yield encoded

            if buffer:
                chunk = StreamChunk(data=buffer, index=metrics.chunks_sent, chunk_type="data",
                                    metadata={"count": len(buffer), "final_flush": True})
                encoded = self._format_chunk(chunk, fmt)
                metrics.chunks_sent += 1
                metrics.bytes_sent += len(encoded)
                yield encoded

            metrics.completed = True
            self._stats["streams_completed"] += 1
        except Exception as exc:
            metrics.error = str(exc)
            raise
        finally:
            metrics.duration_seconds = time.time() - metrics.started_at
            self._active_streams.pop(stream_id, None)

    def _format_chunk(self, chunk, fmt):
        if fmt == StreamFormat.SSE:
            return chunk.to_sse(event=chunk.chunk_type).encode()
        else:
            return chunk.to_ndjson().encode()

    def get_active_streams(self):
        now = time.time()
        return {
            sid: {"chunks_sent": m.chunks_sent, "bytes_sent": m.bytes_sent,
                  "elapsed_seconds": round(now - m.started_at, 1)}
            for sid, m in self._active_streams.items()
        }

    def get_stats(self):
        return {**self._stats, "active_streams": len(self._active_streams)}
