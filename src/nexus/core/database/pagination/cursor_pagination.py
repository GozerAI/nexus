"""
Cursor-based (keyset) pagination replacing OFFSET pagination.

Keyset pagination is more efficient than OFFSET because it does not need to
scan and discard rows. Instead it uses an indexed column (the *cursor key*)
to seek directly to the next page.
"""

import base64
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, List, Optional, TypeVar, Type

from sqlalchemy import Column, asc, desc, and_, or_
from sqlalchemy.orm import Session, Query

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SortDirection(str, Enum):
    ASC = "asc"
    DESC = "desc"


@dataclass
class CursorPage(Generic[T]):
    """A single page of cursor-paginated results."""

    items: List[T]
    next_cursor: Optional[str] = None
    previous_cursor: Optional[str] = None
    has_next: bool = False
    has_previous: bool = False
    page_size: int = 0
    total_estimate: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "items": self.items,
            "next_cursor": self.next_cursor,
            "previous_cursor": self.previous_cursor,
            "has_next": self.has_next,
            "has_previous": self.has_previous,
            "page_size": self.page_size,
            "total_estimate": self.total_estimate,
        }


def _encode_cursor(values: dict) -> str:
    """Encode cursor values to an opaque string."""
    payload = json.dumps(values, default=str, sort_keys=True)
    return base64.urlsafe_b64encode(payload.encode()).decode()


def _decode_cursor(cursor: str) -> dict:
    """Decode an opaque cursor string to values."""
    try:
        payload = base64.urlsafe_b64decode(cursor.encode()).decode()
        return json.loads(payload)
    except Exception:
        raise ValueError(f"Invalid cursor: {cursor}")


class CursorPaginator:
    """
    Cursor-based paginator for SQLAlchemy queries.

    Usage::

        paginator = CursorPaginator(
            session=session,
            model=CostEntryModel,
            cursor_columns=[CostEntryModel.id],
            page_size=50,
        )
        page = paginator.get_page(query, after_cursor=cursor)
    """

    def __init__(
        self,
        session: Session,
        model: Type,
        cursor_columns: Optional[List[Column]] = None,
        page_size: int = 50,
        max_page_size: int = 200,
        direction: SortDirection = SortDirection.ASC,
    ):
        self.session = session
        self.model = model
        self.cursor_columns = cursor_columns or []
        self.page_size = min(page_size, max_page_size)
        self.max_page_size = max_page_size
        self.direction = direction

        if not self.cursor_columns:
            # Default to primary key
            mapper = model.__mapper__ if hasattr(model, "__mapper__") else None
            if mapper and mapper.primary_key:
                self.cursor_columns = list(mapper.primary_key)
            else:
                raise ValueError(
                    "cursor_columns required when model has no detectable primary key"
                )

    def _order_func(self, col: Column, reverse: bool = False):
        """Return asc/desc depending on direction and reverse flag."""
        ascending = (self.direction == SortDirection.ASC) != reverse
        return asc(col) if ascending else desc(col)

    def _build_cursor_filter(self, cursor_values: dict, reverse: bool = False):
        """Build a WHERE clause for keyset pagination."""
        columns = self.cursor_columns
        conditions = []

        for i, col in enumerate(columns):
            col_name = col.key if hasattr(col, "key") else col.name
            value = cursor_values.get(col_name)
            if value is None:
                continue

            if (self.direction == SortDirection.ASC) != reverse:
                cmp = col > value
            else:
                cmp = col < value

            if i > 0:
                # For multi-column cursors, preceding columns must be equal
                eq_parts = []
                for prev_col in columns[:i]:
                    prev_name = prev_col.key if hasattr(prev_col, "key") else prev_col.name
                    prev_val = cursor_values.get(prev_name)
                    if prev_val is not None:
                        eq_parts.append(prev_col == prev_val)
                if eq_parts:
                    cmp = and_(*eq_parts, cmp)

            conditions.append(cmp)

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return or_(*conditions)

    def _extract_cursor(self, item) -> dict:
        """Extract cursor column values from a result item."""
        values = {}
        for col in self.cursor_columns:
            col_name = col.key if hasattr(col, "key") else col.name
            val = getattr(item, col_name, None)
            values[col_name] = val
        return values

    def get_page(
        self,
        query: Optional[Query] = None,
        after_cursor: Optional[str] = None,
        before_cursor: Optional[str] = None,
        page_size: Optional[int] = None,
        include_total: bool = False,
    ) -> CursorPage:
        """
        Fetch a page of results using cursor-based pagination.

        Args:
            query: Base SQLAlchemy query (if None, queries model directly)
            after_cursor: Fetch results after this cursor (forward pagination)
            before_cursor: Fetch results before this cursor (backward pagination)
            page_size: Override default page size
            include_total: Include estimated total count (slower)

        Returns:
            CursorPage with items and pagination metadata
        """
        size = min(page_size or self.page_size, self.max_page_size)

        if query is None:
            query = self.session.query(self.model)

        total = None
        if include_total:
            total = query.count()

        reverse = before_cursor is not None

        # Apply cursor filter
        if after_cursor:
            cursor_values = _decode_cursor(after_cursor)
            cursor_filter = self._build_cursor_filter(cursor_values, reverse=False)
            if cursor_filter is not None:
                query = query.filter(cursor_filter)
        elif before_cursor:
            cursor_values = _decode_cursor(before_cursor)
            cursor_filter = self._build_cursor_filter(cursor_values, reverse=True)
            if cursor_filter is not None:
                query = query.filter(cursor_filter)

        # Apply ordering
        for col in self.cursor_columns:
            query = query.order_by(self._order_func(col, reverse=reverse))

        # Fetch one extra to detect has_next / has_previous
        items = query.limit(size + 1).all()

        has_more = len(items) > size
        if has_more:
            items = items[:size]

        # If we searched backwards, reverse items to restore natural order
        if reverse:
            items = list(reversed(items))

        # Build cursors
        next_cursor = None
        previous_cursor = None

        if items:
            if has_more and not reverse:
                next_cursor = _encode_cursor(self._extract_cursor(items[-1]))
            elif not reverse and after_cursor:
                # Check forward: we have items, might have next
                if has_more:
                    next_cursor = _encode_cursor(self._extract_cursor(items[-1]))

            if after_cursor:
                previous_cursor = _encode_cursor(self._extract_cursor(items[0]))
            if reverse and has_more:
                previous_cursor = _encode_cursor(self._extract_cursor(items[0]))
            if reverse:
                next_cursor = _encode_cursor(self._extract_cursor(items[-1]))

        has_next = next_cursor is not None
        has_previous = previous_cursor is not None

        return CursorPage(
            items=items,
            next_cursor=next_cursor,
            previous_cursor=previous_cursor,
            has_next=has_next,
            has_previous=has_previous,
            page_size=size,
            total_estimate=total,
        )
