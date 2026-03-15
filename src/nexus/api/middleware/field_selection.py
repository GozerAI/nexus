"""
GraphQL-style field selection for REST endpoints.

Allows clients to request only specific fields in API responses,
reducing payload size and network transfer time.

Usage::

    # Client sends: GET /api/models?fields=name,provider,cost
    selector = FieldSelector.from_query_param("name,provider,cost")
    result = selector.apply(full_response_dict)
    # Returns only the requested fields
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class FieldSpec:
    """Parsed field specification with optional nested selection."""

    name: str
    children: Optional["FieldSelection"] = None

    def __repr__(self) -> str:
        if self.children:
            return f"{self.name}{{{self.children}}}"
        return self.name


@dataclass
class FieldSelection:
    """A set of selected fields, possibly with nested selections."""

    fields: List[FieldSpec] = field(default_factory=list)

    @property
    def field_names(self) -> Set[str]:
        return {f.name for f in self.fields}

    def has_field(self, name: str) -> bool:
        return name in self.field_names

    def get_child(self, name: str) -> Optional["FieldSelection"]:
        for f in self.fields:
            if f.name == name:
                return f.children
        return None

    def __repr__(self) -> str:
        return ",".join(repr(f) for f in self.fields)


class FieldSelector:
    """
    Parses and applies GraphQL-style field selections to dict responses.

    Supports:
    - Simple fields: ``name,provider,cost``
    - Nested fields: ``name,config{model,temperature}``
    - Wildcard: ``*`` (all fields at this level)
    - Array item projection: applied uniformly to list items

    Usage::

        selector = FieldSelector.from_query_param("name,config{model,temperature}")
        result = selector.apply({
            "name": "gpt-4",
            "provider": "openai",
            "config": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 4096},
            "metadata": {"created": "2024-01-01"},
        })
        # {"name": "gpt-4", "config": {"model": "gpt-4", "temperature": 0.7}}
    """

    def __init__(self, selection: FieldSelection):
        self._selection = selection

    @classmethod
    def from_query_param(cls, fields_param: str) -> "FieldSelector":
        """
        Parse a fields query parameter string.

        Args:
            fields_param: Comma-separated field names with optional nesting.
                          Examples: ``"name,cost"`` or ``"name,config{model,temp}"``

        Returns:
            FieldSelector instance
        """
        selection = cls._parse(fields_param.strip())
        return cls(selection)

    @classmethod
    def _parse(cls, text: str) -> FieldSelection:
        """Parse field selection text into a FieldSelection tree."""
        fields = []
        i = 0
        current = ""

        while i < len(text):
            ch = text[i]

            if ch == ",":
                if current.strip():
                    fields.append(FieldSpec(name=current.strip()))
                current = ""
                i += 1
            elif ch == "{":
                # Find matching closing brace
                name = current.strip()
                current = ""
                depth = 1
                i += 1
                nested_start = i
                while i < len(text) and depth > 0:
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                    i += 1
                nested_text = text[nested_start : i - 1]
                child_selection = cls._parse(nested_text)
                fields.append(FieldSpec(name=name, children=child_selection))
            else:
                current += ch
                i += 1

        if current.strip():
            fields.append(FieldSpec(name=current.strip()))

        return FieldSelection(fields=fields)

    def apply(self, data: Any) -> Any:
        """
        Apply field selection to a data structure.

        Args:
            data: Dictionary, list, or primitive value

        Returns:
            Filtered data with only selected fields
        """
        return self._apply_selection(data, self._selection)

    def _apply_selection(self, data: Any, selection: FieldSelection) -> Any:
        if isinstance(data, dict):
            return self._apply_to_dict(data, selection)
        elif isinstance(data, list):
            return [self._apply_selection(item, selection) for item in data]
        return data

    def _apply_to_dict(self, data: dict, selection: FieldSelection) -> dict:
        # Wildcard: return all fields
        if selection.has_field("*"):
            return data

        result = {}
        for spec in selection.fields:
            if spec.name not in data:
                continue
            value = data[spec.name]

            if spec.children is not None:
                # Apply nested selection
                value = self._apply_selection(value, spec.children)

            result[spec.name] = value

        return result

    @property
    def selection(self) -> FieldSelection:
        return self._selection


def apply_field_selection(
    data: Any,
    fields_param: Optional[str],
) -> Any:
    """
    Convenience function to apply field selection.

    If fields_param is None or empty, returns data unchanged.

    Args:
        data: Response data (dict or list of dicts)
        fields_param: Comma-separated field spec or None

    Returns:
        Filtered data
    """
    if not fields_param:
        return data

    selector = FieldSelector.from_query_param(fields_param)
    return selector.apply(data)
