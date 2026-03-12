"""
Database connection and session management.
"""

import os
import logging
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Manages database connection and session lifecycle.

    Uses SQLite with Write-Ahead Logging (WAL) for better concurrency.
    """

    def __init__(self, db_path: str = "data/thenexus.db", echo: bool = False):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            echo: If True, log all SQL statements
        """
        self.db_path = db_path

        # Ensure data directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        # Create engine
        # Use check_same_thread=False for Flask compatibility
        if db_path == ":memory:":
            # In-memory database for testing
            self.engine = create_engine(
                f"sqlite:///{db_path}",
                echo=echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            # File-based database
            self.engine = create_engine(
                f"sqlite:///{db_path}",
                echo=echo,
                connect_args={"check_same_thread": False},
            )

            # Enable WAL mode for better concurrency
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        logger.info(f"Database connection initialized: {db_path}")

    def create_tables(self):
        """Create all database tables."""
        from nexus.core.database.models import Base
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def close(self):
        """Close the database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")


# Global database instance
_db: DatabaseConnection = None


def init_db(db_path: str = "data/thenexus.db", echo: bool = False) -> DatabaseConnection:
    """
    Initialize the global database connection.

    Args:
        db_path: Path to SQLite database file
        echo: If True, log all SQL statements

    Returns:
        DatabaseConnection instance
    """
    global _db
    _db = DatabaseConnection(db_path=db_path, echo=echo)
    _db.create_tables()
    return _db


def get_db() -> DatabaseConnection:
    """
    Get the global database connection.

    Returns:
        DatabaseConnection instance

    Raises:
        RuntimeError: If database has not been initialized
    """
    global _db
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db


def get_session() -> Session:
    """
    Get a new database session.

    Convenience function that gets session from global database.

    Returns:
        SQLAlchemy Session
    """
    return get_db().get_session()
