"""SQLite storage for structured data."""

import aiosqlite
from typing import Optional, List, Dict, Any
from pathlib import Path

from nexus.core.exceptions import StorageError


class SQLiteStore:
    """Async SQLite storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection = None

    async def initialize(self):
        """Initialize database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row
        await self._run_migrations()

    async def close(self):
        if self._connection:
            await self._connection.close()

    async def _run_migrations(self):
        """Create all tables."""
        await self._connection.executescript("""
            -- Tasks
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                priority TEXT DEFAULT 'medium',
                project_path TEXT,
                parent_task_id TEXT,
                context TEXT,
                tags TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            -- Goals
            CREATE TABLE IF NOT EXISTS goals (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                project_path TEXT,
                status TEXT DEFAULT 'active',
                target_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            );

            -- Milestones
            CREATE TABLE IF NOT EXISTS milestones (
                id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                target_date TIMESTAMP,
                completed_at TIMESTAMP
            );

            -- Decisions
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                decision TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                alternatives TEXT,
                project_path TEXT,
                task_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Corrections
            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                original TEXT NOT NULL,
                corrected TEXT NOT NULL,
                reason TEXT NOT NULL,
                topic TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Preferences
            CREATE TABLE IF NOT EXISTS preferences (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                observation_count INTEGER DEFAULT 1,
                last_observed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(category, key)
            );

            -- Sessions
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                project_path TEXT,
                focus_task_id TEXT,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                summary TEXT,
                handoff_notes TEXT,
                insights TEXT,
                open_questions TEXT
            );

            -- Blockers
            CREATE TABLE IF NOT EXISTS blockers (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                blocker_type TEXT NOT NULL,
                description TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolution TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            );

            -- Task notes
            CREATE TABLE IF NOT EXISTS task_notes (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                note TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Task files
            CREATE TABLE IF NOT EXISTS task_files (
                task_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (task_id, file_path)
            );

            -- Entities
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Entity relations
            CREATE TABLE IF NOT EXISTS entity_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                properties TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Facts
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                topic TEXT,
                source_type TEXT,
                source_id TEXT,
                confidence REAL DEFAULT 0.5,
                verification_count INTEGER DEFAULT 0,
                last_verified TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Conversations (metadata)
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                project_path TEXT,
                message_count INTEGER,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                metadata TEXT
            );

            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_tasks_project ON tasks(project_path);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_goals_project ON goals(project_path);
            CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_path);
            CREATE INDEX IF NOT EXISTS idx_corrections_topic ON corrections(topic);
            CREATE INDEX IF NOT EXISTS idx_preferences_category ON preferences(category);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_facts_topic ON facts(topic);
        """)
        await self._connection.commit()

    async def insert(self, table: str, data: Dict[str, Any]) -> str:
        """Insert row."""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        await self._connection.execute(query, list(data.values()))
        await self._connection.commit()
        return data.get("id", "")

    async def update(self, table: str, id: str, data: Dict[str, Any]):
        """Update row."""
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
        await self._connection.execute(query, list(data.values()) + [id])
        await self._connection.commit()

    async def get(self, table: str, id: str) -> Optional[Dict[str, Any]]:
        """Get row by ID."""
        async with self._connection.execute(f"SELECT * FROM {table} WHERE id = ?", [id]) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def delete(self, table: str, id: str):
        """Delete row."""
        await self._connection.execute(f"DELETE FROM {table} WHERE id = ?", [id])
        await self._connection.commit()

    async def query(self, table: str, where: Optional[Dict] = None,
                   order_by: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Query rows."""
        query = f"SELECT * FROM {table}"
        values = []

        if where:
            conditions = [f"{k} = ?" for k in where.keys()]
            query += " WHERE " + " AND ".join(conditions)
            values = list(where.values())

        if order_by:
            query += f" ORDER BY {order_by}"
        if limit:
            query += f" LIMIT {limit}"

        async with self._connection.execute(query, values) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def count(self, table: str, where: Optional[Dict] = None) -> int:
        """Count rows."""
        query = f"SELECT COUNT(*) as count FROM {table}"
        values = []
        if where:
            conditions = [f"{k} = ?" for k in where.keys()]
            query += " WHERE " + " AND ".join(conditions)
            values = list(where.values())

        async with self._connection.execute(query, values) as cursor:
            row = await cursor.fetchone()
            return row["count"]

    async def execute_raw(self, query: str, values: List = None) -> List[Dict]:
        """Execute raw SQL."""
        async with self._connection.execute(query, values or []) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
