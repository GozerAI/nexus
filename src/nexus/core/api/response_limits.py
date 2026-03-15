"""
Response size limits with automatic pagination/truncation.

Prevents large responses from overwhelming clients or network bandwidth.
Supports configurable per-endpoint limits, automatic truncation with
continuation tokens, and size estimation before serialization.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SizeLimitExceeded(Exception):
    """Raised when a response exceeds configured size limits."""

    def __init__(self, actual_bytes: int, limit_bytes: int, truncated: bool = False):
        self.actual_bytes = actual_bytes
        self.limit_bytes = limit_bytes
        self.truncated = truncated
        super().__init__(
            f"Response size {actual_bytes} exceeds limit {limit_bytes}"
        )


@dataclass
class TruncationResult:
    """Result of truncating a large response."""
    data: Any
    original_size_bytes: int
    truncated_size_bytes: int
    was_truncated: bool
    continuation_token: Optional[str] = None
    total_items: int = 0
    returned_items: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResponseSizeLimiter:
    """
    Enforces response size limits with intelligent truncation.

    Features:
    - Per-endpoint size limits
    - Automatic list truncation with continuation tokens
    - Deep object size estimation without full serialization
    - Configurable overflow strategies (truncate, error, paginate)
    """

    DEFAULT_LIMIT_BYTES = 10 * 1024 * 1024  # 10 MB
    ABSOLUTE_MAX_BYTES = 100 * 1024 * 1024  # 100 MB

    class Strategy:
        TRUNCATE = "truncate"
        ERROR = "error"
        PAGINATE = "paginate"

    def __init__(
        self,
        default_limit_bytes: int = DEFAULT_LIMIT_BYTES,
        default_strategy: str = Strategy.TRUNCATE,
        endpoint_limits: Optional[Dict[str, int]] = None,
    ):
        self._default_limit = min(default_limit_bytes, self.ABSOLUTE_MAX_BYTES)
        self._default_strategy = default_strategy
        self._endpoint_limits: Dict[str, int] = endpoint_limits or {}
        self._stats = {
            "checked": 0,
            "truncated": 0,
            "errors": 0,
            "total_original_bytes": 0,
            "total_returned_bytes": 0,
        }

    def set_endpoint_limit(self, endpoint: str, limit_bytes: int) -> None:
        """Set a custom size limit for a specific endpoint."""
        self._endpoint_limits[endpoint] = min(limit_bytes, self.ABSOLUTE_MAX_BYTES)

    def estimate_size(self, data: Any, depth: int = 0) -> int:
        """
        Estimate serialized JSON size without full serialization.

        Uses sys.getsizeof for primitives and samples collections
        for large datasets. Accurate within ~20% for typical payloads.
        """
        if depth > 50:
            return 8  # Prevent infinite recursion

        if data is None:
            return 4  # "null"
        elif isinstance(data, bool):
            return 5  # "true" / "false"
        elif isinstance(data, (int, float)):
            return len(str(data))
        elif isinstance(data, str):
            return len(data) + 2  # quotes
        elif isinstance(data, list):
            if not data:
                return 2  # "[]"
            # For large lists, sample first few items and extrapolate
            sample_size = min(len(data), 10)
            sample_total = sum(
                self.estimate_size(data[i], depth + 1) for i in range(sample_size)
            )
            avg_item = sample_total / sample_size
            # Items + commas + brackets
            return int(avg_item * len(data) + len(data) - 1 + 2)
        elif isinstance(data, dict):
            if not data:
                return 2  # "{}"
            total = 2  # braces
            for key, value in data.items():
                total += len(str(key)) + 2  # key with quotes
                total += 1  # colon
                total += self.estimate_size(value, depth + 1)
                total += 1  # comma
            return total
        else:
            return len(str(data))

    def check(
        self,
        data: Any,
        endpoint: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> TruncationResult:
        """
        Check response size and apply limits if needed.

        Args:
            data: Response payload
            endpoint: Optional endpoint for per-endpoint limits
            strategy: Override default strategy

        Returns:
            TruncationResult with potentially truncated data

        Raises:
            SizeLimitExceeded: If strategy is ERROR and limit exceeded
        """
        self._stats["checked"] += 1
        limit = self._endpoint_limits.get(endpoint, self._default_limit) if endpoint else self._default_limit
        effective_strategy = strategy or self._default_strategy

        estimated = self.estimate_size(data)
        self._stats["total_original_bytes"] += estimated

        if estimated <= limit:
            self._stats["total_returned_bytes"] += estimated
            return TruncationResult(
                data=data,
                original_size_bytes=estimated,
                truncated_size_bytes=estimated,
                was_truncated=False,
            )

        if effective_strategy == self.Strategy.ERROR:
            self._stats["errors"] += 1
            raise SizeLimitExceeded(estimated, limit)

        # Truncation strategies
        self._stats["truncated"] += 1

        if isinstance(data, list):
            return self._truncate_list(data, limit, estimated)
        elif isinstance(data, dict):
            return self._truncate_dict(data, limit, estimated)
        else:
            return self._truncate_string(data, limit, estimated)

    def _truncate_list(
        self, data: list, limit: int, original_size: int
    ) -> TruncationResult:
        """Truncate a list response to fit within limits."""
        total = len(data)
        if total == 0:
            return TruncationResult(
                data=data,
                original_size_bytes=original_size,
                truncated_size_bytes=2,
                was_truncated=False,
            )

        # Binary search for the right number of items
        low, high = 1, total
        best = 1
        while low <= high:
            mid = (low + high) // 2
            subset = data[:mid]
            size = self.estimate_size(subset)
            if size <= limit:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        truncated = data[:best]
        truncated_size = self.estimate_size(truncated)
        self._stats["total_returned_bytes"] += truncated_size

        # Generate continuation token (offset-based)
        continuation = f"offset:{best}" if best < total else None

        return TruncationResult(
            data=truncated,
            original_size_bytes=original_size,
            truncated_size_bytes=truncated_size,
            was_truncated=best < total,
            continuation_token=continuation,
            total_items=total,
            returned_items=best,
            metadata={"truncation_reason": "response_size_limit"},
        )

    def _truncate_dict(
        self, data: dict, limit: int, original_size: int
    ) -> TruncationResult:
        """Truncate dict by removing large nested values."""
        # Look for list values that can be truncated
        result = dict(data)
        for key, value in sorted(
            data.items(), key=lambda kv: self.estimate_size(kv[1]), reverse=True
        ):
            if isinstance(value, list) and len(value) > 1:
                sub = self._truncate_list(value, limit // 2, self.estimate_size(value))
                result[key] = sub.data
                if self.estimate_size(result) <= limit:
                    break
            elif isinstance(value, str) and len(value) > limit // 4:
                max_chars = limit // 4
                result[key] = value[:max_chars] + "...[truncated]"
                if self.estimate_size(result) <= limit:
                    break

        truncated_size = self.estimate_size(result)
        self._stats["total_returned_bytes"] += truncated_size

        return TruncationResult(
            data=result,
            original_size_bytes=original_size,
            truncated_size_bytes=truncated_size,
            was_truncated=True,
            metadata={"truncation_reason": "response_size_limit"},
        )

    def _truncate_string(
        self, data: Any, limit: int, original_size: int
    ) -> TruncationResult:
        """Truncate string-like data."""
        s = str(data)
        if len(s) > limit:
            s = s[: limit - 20] + "...[truncated]"
        truncated_size = len(s)
        self._stats["total_returned_bytes"] += truncated_size
        return TruncationResult(
            data=s,
            original_size_bytes=original_size,
            truncated_size_bytes=truncated_size,
            was_truncated=True,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get size limiter statistics."""
        checked = self._stats["checked"]
        return {
            **self._stats,
            "truncation_rate": (
                self._stats["truncated"] / checked if checked > 0 else 0.0
            ),
            "avg_compression_ratio": (
                self._stats["total_returned_bytes"]
                / max(self._stats["total_original_bytes"], 1)
            ),
        }
