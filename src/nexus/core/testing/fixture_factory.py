"""
Test fixture factory with lazy initialization.

Provides a fluent API for building test fixtures with automatic
dependency resolution and lazy initialization. Fixtures are only
created when accessed, reducing test setup overhead.
"""

import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LazyFixture:
    """
    A lazily-initialized test fixture.

    The value is computed on first access and cached thereafter.
    Supports override for test-specific customization.
    """

    def __init__(
        self,
        factory_fn: Callable[..., Any],
        name: str = "",
        dependencies: Optional[List[str]] = None,
    ):
        """
        Args:
            factory_fn: Callable that creates the fixture value
            name: Fixture name
            dependencies: Names of fixtures this depends on
        """
        self._factory_fn = factory_fn
        self._name = name
        self._dependencies = dependencies or []
        self._value: Any = None
        self._initialized = False
        self._override: Any = None
        self._has_override = False

    @property
    def value(self) -> Any:
        """Get the fixture value, creating it if needed."""
        if self._has_override:
            return self._override
        if not self._initialized:
            self._value = self._factory_fn()
            self._initialized = True
        return self._value

    def override(self, value: Any) -> None:
        """Override the fixture with a specific value."""
        self._override = value
        self._has_override = True

    def reset(self) -> None:
        """Reset to uninitialized state."""
        self._value = None
        self._initialized = False
        self._override = None
        self._has_override = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def name(self) -> str:
        return self._name

    @property
    def dependencies(self) -> List[str]:
        return list(self._dependencies)


class FixtureFactory:
    """
    Factory for creating and managing test fixtures.

    Features:
    - Lazy initialization (fixtures created on first access)
    - Dependency resolution between fixtures
    - Fixture inheritance (build on base fixtures)
    - Sequence generation for unique values
    - Override mechanism for test customization
    - Automatic cleanup
    """

    def __init__(self):
        self._fixtures: Dict[str, LazyFixture] = {}
        self._sequences: Dict[str, int] = {}
        self._builders: Dict[str, Callable] = {}
        self._cleanup_fns: List[Callable] = []
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        factory_fn: Callable[..., Any],
        dependencies: Optional[List[str]] = None,
    ) -> "FixtureFactory":
        """
        Register a fixture factory.

        Args:
            name: Fixture name
            factory_fn: Callable that creates the fixture
            dependencies: Names of fixtures this depends on

        Returns:
            self (for chaining)
        """
        self._fixtures[name] = LazyFixture(
            factory_fn=factory_fn,
            name=name,
            dependencies=dependencies or [],
        )
        return self

    def register_builder(
        self,
        name: str,
        builder: Callable[..., Any],
    ) -> "FixtureFactory":
        """
        Register a builder function that accepts kwargs for customization.

        Usage::

            factory.register_builder("user", lambda **kw: User(
                name=kw.get("name", "Test User"),
                email=kw.get("email", f"user{factory.sequence('user')}@test.com"),
            ))

            user = factory.build("user", name="Alice")
        """
        self._builders[name] = builder
        return self

    def get(self, name: str) -> Any:
        """Get a fixture value (creating it lazily if needed)."""
        fixture = self._fixtures.get(name)
        if fixture is None:
            raise KeyError(f"Fixture '{name}' not registered")

        # Resolve dependencies first
        for dep_name in fixture.dependencies:
            self.get(dep_name)

        return fixture.value

    def build(self, builder_name: str, **kwargs: Any) -> Any:
        """
        Build a fixture using a registered builder with custom kwargs.

        Args:
            builder_name: Builder name
            **kwargs: Override parameters

        Returns:
            Built fixture
        """
        builder = self._builders.get(builder_name)
        if builder is None:
            raise KeyError(f"Builder '{builder_name}' not registered")
        return builder(**kwargs)

    def build_many(self, builder_name: str, count: int, **kwargs: Any) -> List[Any]:
        """Build multiple fixtures."""
        return [self.build(builder_name, **kwargs) for _ in range(count)]

    def sequence(self, name: str = "default") -> int:
        """
        Get the next value in a named sequence.

        Useful for generating unique IDs, emails, etc.
        """
        with self._lock:
            current = self._sequences.get(name, 0) + 1
            self._sequences[name] = current
            return current

    def unique_id(self) -> str:
        """Generate a unique ID."""
        return str(uuid.uuid4())[:8]

    def unique_email(self, domain: str = "test.example.com") -> str:
        """Generate a unique email address."""
        seq = self.sequence("email")
        return f"user{seq}@{domain}"

    def unique_name(self, prefix: str = "Test") -> str:
        """Generate a unique name."""
        seq = self.sequence("name")
        return f"{prefix} {seq}"

    def override(self, name: str, value: Any) -> "FixtureFactory":
        """
        Override a fixture with a specific value.

        Args:
            name: Fixture name
            value: Override value

        Returns:
            self (for chaining)
        """
        fixture = self._fixtures.get(name)
        if fixture:
            fixture.override(value)
        return self

    def add_cleanup(self, fn: Callable) -> None:
        """Register a cleanup function to be called on reset."""
        self._cleanup_fns.append(fn)

    def reset(self) -> None:
        """Reset all fixtures and sequences."""
        for fixture in self._fixtures.values():
            fixture.reset()
        self._sequences.clear()
        for fn in self._cleanup_fns:
            try:
                fn()
            except Exception as e:
                logger.warning("Cleanup function failed: %s", e)

    def reset_fixture(self, name: str) -> None:
        """Reset a single fixture."""
        fixture = self._fixtures.get(name)
        if fixture:
            fixture.reset()

    @property
    def registered_names(self) -> List[str]:
        return list(self._fixtures.keys())

    @property
    def builder_names(self) -> List[str]:
        return list(self._builders.keys())

    def get_stats(self) -> Dict[str, Any]:
        return {
            "fixtures_registered": len(self._fixtures),
            "builders_registered": len(self._builders),
            "fixtures_initialized": sum(
                1 for f in self._fixtures.values() if f.is_initialized
            ),
            "sequences": dict(self._sequences),
        }
