"""
Batch loading for related entity queries.

Collects individual lookups and executes them as a single query,
eliminating N+1 query patterns.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar

from sqlalchemy import Column, and_
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")


@dataclass
class BatchQuery:
    """Describes a batch query configuration."""
    model: Type
    key_column: Column
    max_batch_size: int = 500
    preload_columns: Optional[List[Column]] = None


class BatchResult(Generic[T]):
    """Result of a batch load operation."""

    def __init__(self, items: Dict[Any, List[T]]):
        self._items = items

    def get(self, key: Any) -> List[T]:
        return self._items.get(key, [])

    def get_one(self, key: Any) -> Optional[T]:
        items = self._items.get(key, [])
        return items[0] if items else None

    def keys(self) -> Set[Any]:
        return set(self._items.keys())

    def all_items(self) -> List[T]:
        result = []
        for items in self._items.values():
            result.extend(items)
        return result

    def __len__(self) -> int:
        return sum(len(v) for v in self._items.values())


class BatchLoader:
    """
    Batches individual entity lookups into efficient bulk queries.

    Usage::

        loader = BatchLoader(session)
        loader.register("user_keys", BatchQuery(
            model=APIKeyModel,
            key_column=APIKeyModel.user_id,
        ))

        # Queue individual lookups
        loader.queue("user_keys", "user-1")
        loader.queue("user_keys", "user-2")
        loader.queue("user_keys", "user-3")

        # Execute all queued queries in a single batch
        results = loader.execute()

        # Access results
        keys_for_user1 = results["user_keys"].get("user-1")
    """

    def __init__(self, session: Session):
        self.session = session
        self._queries: Dict[str, BatchQuery] = {}
        self._queue: Dict[str, Set[Any]] = defaultdict(set)
        self._filters: Dict[str, list] = defaultdict(list)

    def register(self, name: str, query: BatchQuery) -> "BatchLoader":
        """Register a batch query configuration."""
        self._queries[name] = query
        return self

    def queue(self, name: str, key: Any) -> "BatchLoader":
        """Queue a key for batch loading."""
        if name not in self._queries:
            raise ValueError(f"No batch query registered for '{name}'")
        self._queue[name].add(key)
        return self

    def queue_many(self, name: str, keys: List[Any]) -> "BatchLoader":
        """Queue multiple keys for batch loading."""
        if name not in self._queries:
            raise ValueError(f"No batch query registered for '{name}'")
        self._queue[name].update(keys)
        return self

    def add_filter(self, name: str, filter_clause) -> "BatchLoader":
        """Add an additional filter clause to a batch query."""
        self._filters[name].append(filter_clause)
        return self

    def execute(self) -> Dict[str, BatchResult]:
        """
        Execute all queued batch queries.

        Returns:
            Dictionary mapping query name to BatchResult
        """
        results = {}

        for name, keys in self._queue.items():
            if not keys:
                results[name] = BatchResult({})
                continue

            query_config = self._queries[name]
            grouped = defaultdict(list)
            key_list = list(keys)

            # Process in chunks to respect max_batch_size
            for i in range(0, len(key_list), query_config.max_batch_size):
                chunk = key_list[i : i + query_config.max_batch_size]
                q = self.session.query(query_config.model).filter(
                    query_config.key_column.in_(chunk)
                )

                # Apply additional filters
                for f in self._filters.get(name, []):
                    q = q.filter(f)

                rows = q.all()
                col_name = (
                    query_config.key_column.key
                    if hasattr(query_config.key_column, "key")
                    else query_config.key_column.name
                )
                for row in rows:
                    row_key = getattr(row, col_name)
                    grouped[row_key].append(row)

            results[name] = BatchResult(dict(grouped))
            logger.debug(
                "BatchLoader '%s': loaded %d items for %d keys",
                name,
                sum(len(v) for v in grouped.values()),
                len(keys),
            )

        # Clear the queue after execution
        self._queue.clear()
        self._filters.clear()

        return results

    def execute_one(self, name: str) -> BatchResult:
        """Execute a single named batch query."""
        full = self.execute()
        return full.get(name, BatchResult({}))


class RelationshipBatchLoader:
    """
    Eager-loads relationships for a list of parent entities.

    Usage::

        rbl = RelationshipBatchLoader(session)
        users = session.query(UserModel).limit(50).all()

        # Load all api_keys for these users in one query
        rbl.load_relationship(
            parents=users,
            parent_key=lambda u: u.user_id,
            child_model=APIKeyModel,
            child_fk=APIKeyModel.user_id,
            attach_as="loaded_keys",
        )

        for user in users:
            print(user.loaded_keys)
    """

    def __init__(self, session: Session):
        self.session = session

    def load_relationship(
        self,
        parents: List[Any],
        parent_key: Callable,
        child_model: Type,
        child_fk: Column,
        attach_as: Optional[str] = None,
        extra_filters: Optional[list] = None,
    ) -> Dict[Any, List[Any]]:
        """
        Batch-load a relationship for a list of parent entities.

        Args:
            parents: List of parent objects
            parent_key: Function to extract the join key from parent
            child_model: SQLAlchemy model for the child
            child_fk: Foreign key column on the child model
            attach_as: If provided, attach results to parent objects as this attribute
            extra_filters: Additional filter clauses

        Returns:
            Mapping from parent key to list of child entities
        """
        if not parents:
            return {}

        keys = [parent_key(p) for p in parents]
        unique_keys = list(set(keys))

        query = self.session.query(child_model).filter(child_fk.in_(unique_keys))
        if extra_filters:
            for f in extra_filters:
                query = query.filter(f)

        children = query.all()

        col_name = child_fk.key if hasattr(child_fk, "key") else child_fk.name
        grouped = defaultdict(list)
        for child in children:
            grouped[getattr(child, col_name)].append(child)

        if attach_as:
            for parent in parents:
                pk = parent_key(parent)
                setattr(parent, attach_as, grouped.get(pk, []))

        return dict(grouped)
