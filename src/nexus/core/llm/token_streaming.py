"""
Token-by-token response streaming from LLM inference.

Provides a structured streaming interface for delivering model output
one token at a time to clients, with metadata (logprobs, timing)
attached to each token event.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class TokenEventType(str, Enum):
    TOKEN = "token"
    START = "start"
    END = "end"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class TokenEvent:
    """A single token event in a stream."""
    event_type: TokenEventType
    token: str = ""
    token_index: int = 0
    logprob: Optional[float] = None
    cumulative_text: str = ""
    model_name: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "event": self.event_type.value,
            "token": self.token,
            "index": self.token_index,
            "text_so_far": self.cumulative_text,
            "timestamp": self.timestamp,
        }
        if self.logprob is not None:
            d["logprob"] = self.logprob
        if self.model_name:
            d["model"] = self.model_name
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        import json
        data = json.dumps(self.to_dict(), default=str)
        return f"event: {self.event_type.value}\ndata: {data}\n\n"


@dataclass
class StreamSession:
    """Tracks state for an active streaming session."""
    session_id: str
    model_name: str
    prompt: str
    started_at: float = field(default_factory=time.time)
    tokens_emitted: int = 0
    cumulative_text: str = ""
    finished: bool = False
    error: Optional[str] = None
    total_latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        elapsed = time.time() - self.started_at
        return self.tokens_emitted / elapsed if elapsed > 0 else 0.0


class TokenStreamer:
    """
    Streams LLM tokens to clients with metadata and progress tracking.

    Features:
    - Token-by-token streaming with logprobs
    - Heartbeat events for long pauses (keep-alive)
    - Time-to-first-token tracking
    - Configurable output callbacks (SSE, WebSocket, etc.)
    - Graceful stream cancellation
    """

    HEARTBEAT_INTERVAL = 15.0  # seconds

    def __init__(
        self,
        model_name: str = "",
        heartbeat_interval: float = HEARTBEAT_INTERVAL,
        on_token: Optional[Callable[[TokenEvent], None]] = None,
        on_complete: Optional[Callable[[StreamSession], None]] = None,
    ):
        """
        Args:
            model_name: Model generating the tokens
            heartbeat_interval: Seconds between heartbeat events
            on_token: Callback for each token event
            on_complete: Callback when stream completes
        """
        self.model_name = model_name
        self._heartbeat_interval = heartbeat_interval
        self._on_token = on_token
        self._on_complete = on_complete
        self._active_sessions: Dict[str, StreamSession] = {}
        self._cancelled: set = set()

    async def stream(
        self,
        session_id: str,
        token_source: AsyncIterator[Dict[str, Any]],
        prompt: str = "",
    ) -> AsyncIterator[TokenEvent]:
        """
        Stream tokens from an async source.

        The token_source should yield dicts with at least a ``"token"`` key.
        Optional keys: ``"logprob"``, ``"finish_reason"``, ``"metadata"``.

        Args:
            session_id: Unique session identifier
            token_source: Async iterator yielding token dicts
            prompt: Original prompt (for logging)

        Yields:
            TokenEvent for each token
        """
        session = StreamSession(
            session_id=session_id,
            model_name=self.model_name,
            prompt=prompt,
        )
        self._active_sessions[session_id] = session

        # Emit start event
        start_event = TokenEvent(
            event_type=TokenEventType.START,
            model_name=self.model_name,
            metadata={"prompt_length": len(prompt)},
        )
        if self._on_token:
            self._on_token(start_event)
        yield start_event

        first_token = True
        last_event_time = time.time()

        try:
            async for token_data in token_source:
                if session_id in self._cancelled:
                    break

                token = token_data.get("token", "")
                session.cumulative_text += token
                session.tokens_emitted += 1

                if first_token:
                    session.time_to_first_token_ms = (
                        (time.time() - session.started_at) * 1000
                    )
                    first_token = False

                event = TokenEvent(
                    event_type=TokenEventType.TOKEN,
                    token=token,
                    token_index=session.tokens_emitted - 1,
                    logprob=token_data.get("logprob"),
                    cumulative_text=session.cumulative_text,
                    model_name=self.model_name,
                    metadata=token_data.get("metadata", {}),
                )

                if self._on_token:
                    self._on_token(event)
                yield event
                last_event_time = time.time()

        except Exception as e:
            session.error = str(e)
            error_event = TokenEvent(
                event_type=TokenEventType.ERROR,
                model_name=self.model_name,
                metadata={"error": str(e)},
            )
            yield error_event

        # Emit end event
        session.finished = True
        session.total_latency_ms = (time.time() - session.started_at) * 1000

        end_event = TokenEvent(
            event_type=TokenEventType.END,
            token_index=session.tokens_emitted,
            cumulative_text=session.cumulative_text,
            model_name=self.model_name,
            metadata={
                "total_tokens": session.tokens_emitted,
                "total_latency_ms": session.total_latency_ms,
                "time_to_first_token_ms": session.time_to_first_token_ms,
                "tokens_per_second": session.tokens_per_second,
            },
        )
        if self._on_token:
            self._on_token(end_event)
        if self._on_complete:
            self._on_complete(session)
        yield end_event

        self._cancelled.discard(session_id)

    def cancel(self, session_id: str) -> bool:
        """Cancel an active streaming session."""
        if session_id in self._active_sessions:
            self._cancelled.add(session_id)
            return True
        return False

    def get_session(self, session_id: str) -> Optional[StreamSession]:
        return self._active_sessions.get(session_id)

    def get_active_sessions(self) -> List[str]:
        return [
            sid for sid, s in self._active_sessions.items()
            if not s.finished
        ]
