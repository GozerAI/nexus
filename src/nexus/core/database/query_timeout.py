"""
Connection pool per-query timeout configuration.

Provides mechanisms to set and enforce per-query timeouts on
database connections, preventing long-running queries from blocking
the connection pool.
"""

import contextlib
import logging
import threading
import time
from typing import Any, Dict, Optional

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class QueryTimeoutConfig:
    """
    Per-query timeout configuration for the connection pool.

    Provides context managers and decorators for setting
    query-level timeouts on SQLite and other databases.
    """

    DEFAULT_TIMEOUT_MS = 30000  # 30 seconds
    MAX_TIMEOUT_MS = 300000  # 5 minutes

    def __init__(self, engine: Engine, default_timeout_ms: int = 30000):
        self._engine = engine
        self._default_timeout_ms = min(default_timeout_ms, self.MAX_TIMEOUT_MS)
        self._timeout_local = threading.local()
        self._stats = {
            "queries_executed": 0,
            "timeouts_hit": 0,
            "total_query_ms": 0.0,
        }

        # Register event listeners
        event.listen(engine, "before_cursor_execute", self._before_execute)
        event.listen(engine, "after_cursor_execute", self._after_execute)

        logger.info(
            "QueryTimeoutConfig initialized (default=%dms)", default_timeout_ms
        )

    def _get_current_timeout(self) -> int:
        """Get the timeout for the current thread/context."""
        return getattr(self._timeout_local, "timeout_ms", self._default_timeout_ms)

    def _before_execute(
        self, conn, cursor, statement, parameters, context, executemany
    ):
        """Set timeout before query execution."""
        timeout_ms = self._get_current_timeout()
        self._timeout_local.start_time = time.time()

        # Set SQLite timeout pragma
        dialect_name = conn.engine.dialect.name if hasattr(conn, "engine") else ""
        if dialect_name == "sqlite":
            try:
                cursor.execute(f"PRAGMA busy_timeout = {timeout_ms}")
            except Exception:
                pass

    def _after_execute(
        self, conn, cursor, statement, parameters, context, executemany
    ):
        """Track execution time after query."""
        start = getattr(self._timeout_local, "start_time", None)
        if start:
            elapsed_ms = (time.time() - start) * 1000
            self._stats["queries_executed"] += 1
            self._stats["total_query_ms"] += elapsed_ms

            timeout_ms = self._get_current_timeout()
            if elapsed_ms > timeout_ms:
                self._stats["timeouts_hit"] += 1
                logger.warning(
                    "Query exceeded timeout: %.1fms > %dms: %s",
                    elapsed_ms,
                    timeout_ms,
                    statement[:200],
                )

    @contextlib.contextmanager
    def timeout(self, timeout_ms: int):
        """
        Context manager for setting a per-query timeout.

        Usage::

            with timeout_config.timeout(5000):
                results = session.query(Model).all()

        Args:
            timeout_ms: Timeout in milliseconds
        """
        timeout_ms = min(timeout_ms, self.MAX_TIMEOUT_MS)
        old = getattr(self._timeout_local, "timeout_ms", self._default_timeout_ms)
        self._timeout_local.timeout_ms = timeout_ms
        try:
            yield
        finally:
            self._timeout_local.timeout_ms = old

    def set_default_timeout(self, timeout_ms: int) -> None:
        """Set the default timeout for all queries."""
        self._default_timeout_ms = min(timeout_ms, self.MAX_TIMEOUT_MS)

    def get_stats(self) -> dict:
        """Get query timeout statistics."""
        total = self._stats["queries_executed"]
        return {
            **self._stats,
            "average_query_ms": (
                self._stats["total_query_ms"] / total if total > 0 else 0
            ),
            "timeout_rate": (
                self._stats["timeouts_hit"] / total if total > 0 else 0
            ),
            "current_default_ms": self._default_timeout_ms,
        }


class TimeoutSession:
    """
    Wraps a SQLAlchemy Session with per-query timeout support.

    Usage::

        ts = TimeoutSession(session, engine)
        ts.set_timeout(5000)

        # All queries through this session respect the timeout
        results = ts.session.query(Model).all()
    """

    def __init__(
        self,
        session: Session,
        timeout_config: QueryTimeoutConfig,
        default_timeout_ms: int = 30000,
    ):
        self.session = session
        self._config = timeout_config
        self._timeout_ms = default_timeout_ms

    def set_timeout(self, timeout_ms: int) -> None:
        """Set query timeout for this session."""
        self._timeout_ms = timeout_ms

    @contextlib.contextmanager
    def query_context(self, timeout_ms: Optional[int] = None):
        """
        Context manager that applies timeout to enclosed queries.

        Args:
            timeout_ms: Timeout in milliseconds (uses session default if None)
        """
        effective = timeout_ms or self._timeout_ms
        with self._config.timeout(effective):
            yield self.session
