"""
Response size limits with truncation.

Enforces maximum response sizes to prevent memory exhaustion and
slow transfers. Supports configurable limits per endpoint and
automatic truncation with metadata indicating data was truncated.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_MAX_ITEMS = 10000
DEFAULT_MAX_DEPTH = 20


@dataclass
class TruncationInfo:
    """Information about how a response was truncated."""

    truncated: bool = False
    original_size_bytes: int = 0
    truncated_size_bytes: int = 0
    original_item_count: Optional[int] = None
    truncated_item_count: Optional[int] = None
    truncation_reason: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"truncated": self.truncated}
        if self.truncated:
            d["original_size_bytes"] = self.original_size_bytes
            d["truncated_size_bytes"] = self.truncated_size_bytes
            if self.original_item_count is not None:
                d["original_item_count"] = self.original_item_count
                d["truncated_item_count"] = self.truncated_item_count
            if self.truncation_reason:
                d["reason"] = self.truncation_reason
        return d


@dataclass
class LimitConfig:
    """Configuration for response limits."""

    max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_items: int = DEFAULT_MAX_ITEMS
    max_depth: int = DEFAULT_MAX_DEPTH
    max_string_length: int = 100000
    include_truncation_meta: bool = True


class ResponseLimiter:
    """
    Enforces response size limits with smart truncation.

    Features:
    - Byte-size limits on JSON-serialized responses
    - Item count limits for arrays
    - Depth limits for nested structures
    - String length limits for individual values
    - Truncation metadata in response
    """

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_items: int = DEFAULT_MAX_ITEMS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_string_length: int = 100000,
        include_truncation_meta: bool = True,
    ):
        self._config = LimitConfig(
            max_bytes=max_bytes,
            max_items=max_items,
            max_depth=max_depth,
            max_string_length=max_string_length,
            include_truncation_meta=include_truncation_meta,
        )
        self._stats = {"total_requests": 0, "truncated_requests": 0}

    def apply(self, data: Any) -> tuple:
        """Apply size limits to response data. Returns (limited_data, TruncationInfo)."""
        self._stats["total_requests"] += 1
        original_size = self._estimate_size(data)
        info = TruncationInfo(original_size_bytes=original_size)

        limited = self._limit_depth(data, depth=0)
        limited = self._limit_strings(limited)
        limited, items_truncated = self._limit_items(limited)
        if items_truncated:
            info.truncated = True
            info.truncation_reason = "item_count_exceeded"

        limited_size = self._estimate_size(limited)
        if limited_size > self._config.max_bytes:
            limited = self._truncate_to_bytes(limited, self._config.max_bytes)
            limited_size = self._estimate_size(limited)
            info.truncated = True
            info.truncation_reason = "byte_size_exceeded"

        info.truncated_size_bytes = limited_size
        if info.truncated:
            self._stats["truncated_requests"] += 1
            if self._config.include_truncation_meta and isinstance(limited, dict):
                limited["_truncation"] = info.to_dict()

        return limited, info

    def _estimate_size(self, data: Any) -> int:
        try:
            return len(json.dumps(data, default=str).encode())
        except (TypeError, ValueError, OverflowError):
            return sys.getsizeof(data)

    def _limit_depth(self, data: Any, depth: int) -> Any:
        if depth >= self._config.max_depth:
            if isinstance(data, dict):
                return {"_truncated": "max_depth_exceeded"}
            elif isinstance(data, list):
                return ["...truncated..."]
            return data
        if isinstance(data, dict):
            return {k: self._limit_depth(v, depth + 1) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._limit_depth(item, depth + 1) for item in data]
        return data

    def _limit_strings(self, data: Any) -> Any:
        max_len = self._config.max_string_length
        if isinstance(data, str):
            return data[:max_len] + "...[truncated]" if len(data) > max_len else data
        elif isinstance(data, dict):
            return {k: self._limit_strings(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._limit_strings(item) for item in data]
        return data

    def _limit_items(self, data: Any) -> tuple:
        truncated = False
        if isinstance(data, list):
            if len(data) > self._config.max_items:
                truncated = True
                data = data[: self._config.max_items]
            data = [self._limit_items(item)[0] for item in data]
        elif isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                new_dict[k], child_truncated = self._limit_items(v)
                if child_truncated:
                    truncated = True
            data = new_dict
        return data, truncated

    def _truncate_to_bytes(self, data: Any, max_bytes: int) -> Any:
        if isinstance(data, list) and len(data) > 1:
            lo, hi = 1, len(data)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if self._estimate_size(data[:mid]) <= max_bytes:
                    lo = mid
                else:
                    hi = mid - 1
            return data[:lo]
        elif isinstance(data, dict):
            keys_by_size = sorted(data.keys(), key=lambda k: self._estimate_size(data[k]), reverse=True)
            result = dict(data)
            for key in keys_by_size:
                if self._estimate_size(result) <= max_bytes:
                    break
                if key.startswith("_"):
                    continue
                del result[key]
            return result
        return data

    def get_stats(self) -> dict:
        total = self._stats["total_requests"]
        return {**self._stats, "truncation_rate": self._stats["truncated_requests"] / total if total > 0 else 0}


def truncate_response(data: Any, max_bytes: int = DEFAULT_MAX_RESPONSE_BYTES, max_items: int = DEFAULT_MAX_ITEMS) -> tuple:
    """Convenience function to truncate a response."""
    limiter = ResponseLimiter(max_bytes=max_bytes, max_items=max_items)
    return limiter.apply(data)
