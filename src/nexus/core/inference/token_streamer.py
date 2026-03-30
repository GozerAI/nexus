"""
Response streaming with token-by-token delivery.

Provides infrastructure for streaming model output tokens to clients
as they are generated, enabling lower time-to-first-token latency.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenEvent:
    """A single token event in a stream."""
    token: str
    index: int
    logprob: Optional[float] = None
    finish_reason: Optional[str] = None
    model_name: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = {"token": self.token, "index": self.index}
        if self.logprob is not None:
            d["logprob"] = self.logprob
        if self.finish_reason:
            d["finish_reason"] = self.finish_reason
        return d


@dataclass
class StreamSession:
    """Metadata for an active streaming session."""
    session_id: str
    model_name: str
    started_at: float
    tokens_sent: int = 0
    total_text: str = ""
    finished: bool = False
    error: Optional[str] = None


class TokenStreamer:
    """
    Manages token-by-token streaming from model inference.

    Supports multiple concurrent streams, backpressure via async queues,
    and configurable buffer sizes.
    """

    def __init__(self, buffer_size=256, max_concurrent_streams=50):
        self._buffer_size = buffer_size
        self._max_streams = max_concurrent_streams
        self._sessions: Dict[str, StreamSession] = {}
        self._queues: Dict[str, asyncio.Queue] = {}
        self._stats = {"streams_started": 0, "streams_completed": 0, "total_tokens": 0}

    def create_stream(self, session_id: str, model_name: str = "") -> asyncio.Queue:
        """Create a new token stream. Returns the queue to push tokens into."""
        if len(self._sessions) >= self._max_streams:
            raise RuntimeError(f"Max concurrent streams ({self._max_streams}) reached")
        session = StreamSession(session_id=session_id, model_name=model_name, started_at=time.time())
        self._sessions[session_id] = session
        queue = asyncio.Queue(maxsize=self._buffer_size)
        self._queues[session_id] = queue
        self._stats["streams_started"] += 1
        return queue

    async def push_token(self, session_id: str, token: str, logprob: Optional[float] = None,
                         finish_reason: Optional[str] = None):
        """Push a token to a stream."""
        session = self._sessions.get(session_id)
        queue = self._queues.get(session_id)
        if not session or not queue:
            return
        event = TokenEvent(
            token=token, index=session.tokens_sent,
            logprob=logprob, finish_reason=finish_reason,
            model_name=session.model_name,
        )
        await queue.put(event)
        session.tokens_sent += 1
        session.total_text += token
        self._stats["total_tokens"] += 1
        if finish_reason:
            session.finished = True
            await queue.put(None)  # Sentinel

    async def consume_stream(self, session_id: str) -> AsyncIterator[TokenEvent]:
        """Consume tokens from a stream as an async iterator."""
        queue = self._queues.get(session_id)
        if not queue:
            return
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

    def finish_stream(self, session_id: str, error: Optional[str] = None):
        """Mark a stream as finished."""
        session = self._sessions.get(session_id)
        if session:
            session.finished = True
            session.error = error
            self._stats["streams_completed"] += 1

    def close_stream(self, session_id: str):
        """Close and clean up a stream."""
        self._sessions.pop(session_id, None)
        self._queues.pop(session_id, None)

    def get_session(self, session_id: str) -> Optional[StreamSession]:
        return self._sessions.get(session_id)

    def get_stats(self) -> dict:
        return {**self._stats, "active_streams": len(self._sessions)}
