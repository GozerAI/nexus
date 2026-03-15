"""
GraphQL-style field selection for REST endpoints.

Allows clients to request only specific fields in API responses,
reducing payload size and serialization overhead.

Usage::

    GET /api/v1/models?fields=name,provider,cost_per_1k_input
    GET /api/v1/blueprints/123?fields=id,name,steps.name,steps.status

Nested field paths use dot notation. Wildcards (*) select all fields
at a given level.
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class FieldSelector:
    """
    Prunes response payloads to include only requested fields.

    Supports:
    - Flat fields: ``name,provider``
    - Nested dot paths: ``steps.name,steps.status``
    - Wildcards: ``steps.*`` (all fields under steps)
    - Array indexing: ``steps.0.name`` (first step's name)
    """

    # Maximum nesting depth to prevent abuse
    MAX_DEPTH = 10
    # Maximum number of field paths per request
    MAX_FIELDS = 100

    def __init__(self, fields_param: str):
        """
        Parse a comma-separated fields string.

        Args:
            fields_param: Comma-separated field paths, e.g. ``"name,steps.id"``
        """
        self._raw = fields_param
        self._paths: List[List[str]] = []
        self._tree: Dict = {}
        self._parse(fields_param)

    def _parse(self, fields_param: str) -> None:
        raw_fields = [f.strip() for f in fields_param.split(",") if f.strip()]
        if len(raw_fields) > self.MAX_FIELDS:
            raise ValueError(
                f"Too many fields requested ({len(raw_fields)} > {self.MAX_FIELDS})"
            )
        for raw in raw_fields:
            parts = raw.split(".")
            if len(parts) > self.MAX_DEPTH:
                raise ValueError(
                    f"Field path too deep ({len(parts)} > {self.MAX_DEPTH}): {raw}"
                )
            # Validate each segment
            for part in parts:
                if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$|^\*$|^\d+$', part):
                    raise ValueError(f"Invalid field path segment: {part!r}")
            self._paths.append(parts)

        # Build selection tree for efficient pruning
        for path in self._paths:
            node = self._tree
            for segment in path:
                if segment not in node:
                    node[segment] = {}
                node = node[segment]

    @property
    def paths(self) -> List[List[str]]:
        return list(self._paths)

    def apply(self, data: Any) -> Any:
        """
        Apply field selection to a response payload.

        Args:
            data: The full response data (dict or list of dicts)

        Returns:
            Pruned data with only selected fields
        """
        if isinstance(data, list):
            return [self._prune(item, self._tree) for item in data]
        elif isinstance(data, dict):
            return self._prune(data, self._tree)
        return data

    def _prune(self, data: Any, tree: Dict, depth: int = 0) -> Any:
        if depth > self.MAX_DEPTH:
            return data

        if not isinstance(data, dict):
            return data

        if not tree:
            # Leaf node in selection tree — include entire value
            return data

        # Check for wildcard
        if "*" in tree:
            # Include all keys but recurse into sub-tree if specified
            result = {}
            for key, value in data.items():
                if isinstance(value, dict) and tree["*"]:
                    result[key] = self._prune(value, tree["*"], depth + 1)
                elif isinstance(value, list) and tree["*"]:
                    result[key] = self._prune_list(value, tree["*"], depth + 1)
                else:
                    result[key] = value
            return result

        result = {}
        for key, subtree in tree.items():
            if key not in data:
                continue

            value = data[key]
            if isinstance(value, dict) and subtree:
                result[key] = self._prune(value, subtree, depth + 1)
            elif isinstance(value, list) and subtree:
                result[key] = self._prune_list(value, subtree, depth + 1)
            else:
                result[key] = value

        return result

    def _prune_list(self, data: list, tree: Dict, depth: int) -> list:
        """Prune each element in a list."""
        # Check if tree has numeric indices
        has_indices = any(k.isdigit() for k in tree)
        if has_indices:
            result = []
            for i, item in enumerate(data):
                idx = str(i)
                if idx in tree:
                    if isinstance(item, dict):
                        result.append(self._prune(item, tree[idx], depth + 1))
                    else:
                        result.append(item)
            return result

        # Apply tree to all list elements
        return [
            self._prune(item, tree, depth + 1) if isinstance(item, dict) else item
            for item in data
        ]


class FieldSelectionMiddleware:
    """
    Middleware that intercepts ``?fields=`` query parameters and prunes
    JSON responses before they are sent to the client.

    Works with any WSGI/ASGI framework by wrapping the response.
    """

    def __init__(
        self,
        param_name: str = "fields",
        always_include: Optional[Set[str]] = None,
    ):
        """
        Args:
            param_name: Query parameter name for field selection
            always_include: Fields to always include (e.g. ``{"id", "type"}``)
        """
        self.param_name = param_name
        self.always_include = always_include or set()

    def process_response(
        self, data: Any, fields_param: Optional[str] = None
    ) -> Any:
        """
        Apply field selection to a response payload.

        Args:
            data: Full response data
            fields_param: Raw ``fields`` query param value

        Returns:
            Pruned data, or original if no field selection requested
        """
        if not fields_param:
            return data

        try:
            selector = FieldSelector(fields_param)
        except ValueError as exc:
            logger.warning("Invalid fields parameter: %s", exc)
            return data

        # Inject always-include fields into the tree
        pruned = selector.apply(data)

        if isinstance(pruned, dict) and self.always_include:
            for key in self.always_include:
                if key in data and key not in pruned:
                    pruned[key] = data[key]
        elif isinstance(pruned, list) and self.always_include:
            for i, item in enumerate(pruned):
                if isinstance(item, dict) and isinstance(data[i], dict):
                    for key in self.always_include:
                        if key in data[i] and key not in item:
                            item[key] = data[i][key]

        return pruned
