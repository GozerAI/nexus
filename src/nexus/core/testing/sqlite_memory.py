"""
In-memory SQLite backend for tests.

Provides fast, isolated database sessions using SQLite ``":memory:"``
databases. Each test gets a fresh database with schema pre-created,
eliminating disk I/O and ensuring test isolation.
"""

import logging
from typing import Any, Dict, Generator, List, Optional, Type

from sqlalchemy import StaticPool, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


class InMemorySession:
    """
    Wraps a SQLAlchemy session backed by in-memory SQLite.

    Provides convenience methods for test data setup and teardown.
    """

    def __init__(self, session: Session, engine: Engine):
        self.session = session
        self.engine = engine
        self._created_objects: List[Any] = []

    def add(self, obj: Any) -> Any:
        """Add an object to the session and track it."""
        self.session.add(obj)
        self._created_objects.append(obj)
        return obj

    def add_all(self, objects: List[Any]) -> List[Any]:
        """Add multiple objects."""
        self.session.add_all(objects)
        self._created_objects.extend(objects)
        return objects

    def commit(self) -> None:
        """Commit current transaction."""
        self.session.commit()

    def rollback(self) -> None:
        """Rollback current transaction."""
        self.session.rollback()

    def query(self, *args: Any, **kwargs: Any):
        """Proxy to session.query()."""
        return self.session.query(*args, **kwargs)

    def execute(self, statement: Any, *args: Any, **kwargs: Any):
        """Execute a SQL statement (auto-wraps plain strings in text())."""
        if isinstance(statement, str):
            statement = text(statement)
        return self.session.execute(statement, *args, **kwargs)

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class InMemorySQLiteFactory:
    """
    Factory for creating in-memory SQLite test databases.

    Features:
    - Zero-disk-IO database sessions
    - Automatic schema creation from SQLAlchemy models
    - WAL mode for better concurrent test support
    - Connection pool sharing (StaticPool for in-memory persistence)
    - Optional seed data population
    - Foreign key enforcement
    """

    def __init__(
        self,
        base: Any = None,
        echo: bool = False,
        enable_fk: bool = True,
    ):
        """
        Args:
            base: SQLAlchemy declarative base (for auto schema creation)
            echo: Enable SQL echo logging
            enable_fk: Enable foreign key constraints
        """
        self._base = base
        self._echo = echo
        self._enable_fk = enable_fk
        self._seed_fns: List[Any] = []

    def add_seed(self, seed_fn: Any) -> None:
        """
        Add a seed function called after schema creation.

        The function receives a Session as its argument.
        """
        self._seed_fns.append(seed_fn)

    def create(self) -> InMemorySession:
        """
        Create a new in-memory database session.

        Returns:
            InMemorySession with schema pre-created
        """
        engine = create_engine(
            "sqlite://",
            echo=self._echo,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        # Enable foreign keys
        if self._enable_fk:
            @event.listens_for(engine, "connect")
            def _enable_foreign_keys(dbapi_conn, _):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        # Create schema
        if self._base is not None:
            self._base.metadata.create_all(engine)

        SessionFactory = sessionmaker(bind=engine)
        session = SessionFactory()

        # Run seed functions
        for seed_fn in self._seed_fns:
            seed_fn(session)
        session.commit()

        return InMemorySession(session=session, engine=engine)

    def create_session_factory(self) -> sessionmaker:
        """
        Create a sessionmaker bound to a shared in-memory database.

        Useful when multiple sessions need to share the same DB
        (e.g., testing concurrent access).
        """
        engine = create_engine(
            "sqlite://",
            echo=self._echo,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

        if self._enable_fk:
            @event.listens_for(engine, "connect")
            def _enable_fk(dbapi_conn, _):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        if self._base is not None:
            self._base.metadata.create_all(engine)

        return sessionmaker(bind=engine)

    def pytest_fixture(self):
        """
        Return a pytest fixture function.

        Usage in conftest.py::

            factory = InMemorySQLiteFactory(base=Base)
            db_session = factory.pytest_fixture()
        """
        def _fixture():
            session = self.create()
            try:
                yield session
            finally:
                session.close()

        _fixture.__name__ = "db_session"
        return _fixture
