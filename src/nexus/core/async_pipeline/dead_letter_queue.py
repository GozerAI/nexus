"""
Dead letter queue for failed pipeline tasks.

Captures tasks that have exhausted all retry attempts, preserving
their full context for later debugging, manual intervention, or
automated reprocessing.
"""

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class DeadLetterReason(str, Enum):
    MAX_RETRIES = "max_retries_exceeded"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"
    DEPENDENCY_FAILED = "dependency_failed"
    POISON_PILL = "poison_pill"  # Causes repeated failures
    MANUAL = "manual"


@dataclass
class DeadLetter:
    """A message in the dead letter queue."""
    letter_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_task_id: str = ""
    task_name: str = ""
    reason: DeadLetterReason = DeadLetterReason.MAX_RETRIES
    error_message: str = ""
    error_type: str = ""
    payload: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 0
    first_attempt_at: float = 0.0
    last_attempt_at: float = 0.0
    dead_lettered_at: float = field(default_factory=time.time)
    reprocessed: bool = False
    reprocess_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        return time.time() - self.dead_lettered_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "letter_id": self.letter_id,
            "original_task_id": self.original_task_id,
            "task_name": self.task_name,
            "reason": self.reason.value,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "dead_lettered_at": self.dead_lettered_at,
            "age_seconds": self.age_seconds,
            "reprocessed": self.reprocessed,
            "reprocess_count": self.reprocess_count,
        }


class DeadLetterQueue:
    """
    Dead letter queue for failed tasks.

    Features:
    - Captures failed tasks with full context
    - Configurable max queue size with overflow handling
    - Task reprocessing (retry from DLQ)
    - Filtering and querying by reason, age, task name
    - Optional callback on dead letter arrival
    - Statistics tracking
    """

    DEFAULT_MAX_SIZE = 10000

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        on_dead_letter: Optional[Callable[[DeadLetter], None]] = None,
        overflow_strategy: str = "drop_oldest",  # "drop_oldest" or "reject"
    ):
        """
        Args:
            max_size: Maximum dead letters to retain
            on_dead_letter: Callback when a new dead letter arrives
            overflow_strategy: What to do when queue is full
        """
        self._max_size = max_size
        self._on_dead_letter = on_dead_letter
        self._overflow_strategy = overflow_strategy
        self._queue: List[DeadLetter] = []
        self._index: Dict[str, DeadLetter] = {}
        self._lock = threading.Lock()
        self._stats = {
            "total_received": 0,
            "total_reprocessed": 0,
            "total_dropped": 0,
            "total_purged": 0,
        }

    def put(
        self,
        task_id: str = "",
        task_name: str = "",
        reason: DeadLetterReason = DeadLetterReason.MAX_RETRIES,
        error_message: str = "",
        error_type: str = "",
        payload: Any = None,
        context: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        max_retries: int = 0,
        first_attempt_at: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeadLetter:
        """
        Add a dead letter to the queue.

        Returns:
            The created DeadLetter
        """
        letter = DeadLetter(
            original_task_id=task_id,
            task_name=task_name,
            reason=reason,
            error_message=error_message,
            error_type=error_type,
            payload=payload,
            context=context or {},
            retry_count=retry_count,
            max_retries=max_retries,
            first_attempt_at=first_attempt_at,
            last_attempt_at=time.time(),
            metadata=metadata or {},
        )

        with self._lock:
            if len(self._queue) >= self._max_size:
                if self._overflow_strategy == "drop_oldest":
                    dropped = self._queue.pop(0)
                    del self._index[dropped.letter_id]
                    self._stats["total_dropped"] += 1
                else:
                    logger.warning("DLQ full, rejecting dead letter for task %s", task_id)
                    return letter

            self._queue.append(letter)
            self._index[letter.letter_id] = letter
            self._stats["total_received"] += 1

        if self._on_dead_letter:
            try:
                self._on_dead_letter(letter)
            except Exception as e:
                logger.error("Dead letter callback failed: %s", e)

        logger.info(
            "Dead letter queued: task=%s reason=%s error=%s",
            task_name, reason.value, error_message[:100],
        )
        return letter

    def get(self, letter_id: str) -> Optional[DeadLetter]:
        """Get a dead letter by ID."""
        return self._index.get(letter_id)

    def peek(self, count: int = 10) -> List[DeadLetter]:
        """Peek at the oldest dead letters."""
        with self._lock:
            return list(self._queue[:count])

    def query(
        self,
        reason: Optional[DeadLetterReason] = None,
        task_name: Optional[str] = None,
        max_age_seconds: Optional[float] = None,
        limit: int = 100,
    ) -> List[DeadLetter]:
        """Query dead letters with filters."""
        with self._lock:
            results = self._queue
            if reason:
                results = [l for l in results if l.reason == reason]
            if task_name:
                results = [l for l in results if l.task_name == task_name]
            if max_age_seconds:
                cutoff = time.time() - max_age_seconds
                results = [l for l in results if l.dead_lettered_at >= cutoff]
            return list(results[:limit])

    def mark_reprocessed(self, letter_id: str) -> bool:
        """Mark a dead letter as reprocessed."""
        letter = self._index.get(letter_id)
        if letter:
            letter.reprocessed = True
            letter.reprocess_count += 1
            self._stats["total_reprocessed"] += 1
            return True
        return False

    def remove(self, letter_id: str) -> Optional[DeadLetter]:
        """Remove and return a dead letter."""
        with self._lock:
            letter = self._index.pop(letter_id, None)
            if letter:
                self._queue.remove(letter)
            return letter

    def purge(
        self,
        max_age_seconds: Optional[float] = None,
        reason: Optional[DeadLetterReason] = None,
        reprocessed_only: bool = False,
    ) -> int:
        """
        Purge dead letters matching criteria.

        Returns:
            Number of letters purged
        """
        with self._lock:
            to_remove = []
            now = time.time()
            for letter in self._queue:
                if max_age_seconds and (now - letter.dead_lettered_at) < max_age_seconds:
                    continue
                if reason and letter.reason != reason:
                    continue
                if reprocessed_only and not letter.reprocessed:
                    continue
                to_remove.append(letter)

            for letter in to_remove:
                self._queue.remove(letter)
                del self._index[letter.letter_id]

            self._stats["total_purged"] += len(to_remove)
            return len(to_remove)

    @property
    def size(self) -> int:
        return len(self._queue)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            reason_counts: Dict[str, int] = {}
            for letter in self._queue:
                r = letter.reason.value
                reason_counts[r] = reason_counts.get(r, 0) + 1

            return {
                **self._stats,
                "current_size": len(self._queue),
                "max_size": self._max_size,
                "reason_breakdown": reason_counts,
                "pending_reprocess": sum(
                    1 for l in self._queue if not l.reprocessed
                ),
            }
