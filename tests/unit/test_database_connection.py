"""Regression tests for database path resolution."""

from pathlib import Path

from nexus.core.database.connection import (
    DEFAULT_DB_PATH,
    LEGACY_DEFAULT_DB_PATH,
    DatabaseConnection,
    resolve_default_db_path,
)


def test_resolve_default_db_path_prefers_environment(monkeypatch, tmp_path):
    configured = tmp_path / "custom.db"
    monkeypatch.setenv("DATABASE_PATH", str(configured))

    assert resolve_default_db_path() == str(configured)


def test_resolve_default_db_path_uses_legacy_file_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    Path(LEGACY_DEFAULT_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(LEGACY_DEFAULT_DB_PATH).write_text("", encoding="utf-8")

    assert resolve_default_db_path() == LEGACY_DEFAULT_DB_PATH


def test_database_connection_defaults_to_new_nexus_db_path(monkeypatch, tmp_path):
    monkeypatch.delenv("DATABASE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    db = DatabaseConnection()
    try:
        assert db.db_path == DEFAULT_DB_PATH
    finally:
        db.close()
