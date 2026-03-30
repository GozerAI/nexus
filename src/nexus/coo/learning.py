"""
Persistent Learning System - Learns from outcomes to improve decisions.

Stores and analyzes execution outcomes to:
- Improve confidence estimation
- Refine executor selection
- Optimize priority weights
- Track patterns of success/failure
"""

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Types of learning outcomes."""
    EXECUTION = "execution"
    APPROVAL = "approval"
    FEEDBACK = "feedback"
    ERROR = "error"


@dataclass
class LearningRecord:
    """A record of an outcome for learning."""
    id: str
    outcome_type: OutcomeType
    item_id: str
    item_type: str  # task, goal, blocker
    item_title: str
    item_tags: List[str]
    executor_used: str
    success: bool
    confidence_before: float
    actual_duration_minutes: float
    actual_cost_usd: float
    error_message: Optional[str] = None
    human_feedback: Optional[str] = None
    notes: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["outcome_type"] = self.outcome_type.value
        d["created_at"] = self.created_at.isoformat()
        d["item_tags"] = json.dumps(self.item_tags)
        d["context"] = json.dumps(self.context)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LearningRecord":
        d["outcome_type"] = OutcomeType(d["outcome_type"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        if isinstance(d.get("item_tags"), str):
            d["item_tags"] = json.loads(d["item_tags"])
        if isinstance(d.get("context"), str):
            d["context"] = json.loads(d["context"])
        return cls(**d)


@dataclass
class LearningStats:
    """Aggregated learning statistics."""
    total_records: int
    success_rate: float
    avg_confidence_accuracy: float  # How close predictions are to outcomes
    best_executors: Dict[str, float]  # executor -> success rate
    worst_patterns: List[str]  # Patterns that lead to failure
    improvement_trend: float  # Recent vs historical success
    top_learnings: List[str]  # Key insights

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PersistentLearning:
    """
    Persistent Learning System.

    Maintains a database of execution outcomes and uses them to:
    1. Improve confidence estimation for future tasks
    2. Select better executors based on task characteristics
    3. Identify patterns that lead to success or failure
    4. Adjust priority weights based on outcomes
    """

    def __init__(self, db_path: str = "data/coo_learning.db"):
        """
        Initialize the learning system.

        Args:
            db_path: Path to SQLite database for learning records
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()  # Thread safety for SQLite operations

        # In-memory caches
        self._executor_stats: Dict[str, Dict[str, float]] = {}
        self._pattern_stats: Dict[str, Dict[str, float]] = {}

    async def initialize(self):
        """Initialize the learning database."""
        # Use check_same_thread=False to allow cross-thread access
        # This is safe because we're using async/await patterns
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Create tables
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS learning_records (
                id TEXT PRIMARY KEY,
                outcome_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                item_title TEXT,
                item_tags TEXT,
                executor_used TEXT,
                success INTEGER NOT NULL,
                confidence_before REAL,
                actual_duration_minutes REAL,
                actual_cost_usd REAL,
                error_message TEXT,
                human_feedback TEXT,
                notes TEXT,
                context TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_learning_item_id ON learning_records(item_id);
            CREATE INDEX IF NOT EXISTS idx_learning_executor ON learning_records(executor_used);
            CREATE INDEX IF NOT EXISTS idx_learning_success ON learning_records(success);
            CREATE INDEX IF NOT EXISTS idx_learning_created ON learning_records(created_at);

            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_key TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_duration REAL,
                avg_cost REAL,
                last_updated TEXT,
                UNIQUE(pattern_type, pattern_key)
            );

            CREATE TABLE IF NOT EXISTS approval_history (
                id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                item_title TEXT,
                approved INTEGER NOT NULL,
                reason TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            );
        """)
        self._conn.commit()

        # Load caches
        await self._load_caches()

        logger.info(f"PersistentLearning initialized with {self.db_path}")

    async def _load_caches(self):
        """Load frequently accessed data into memory."""
        # Load executor stats
        cursor = self._conn.execute("""
            SELECT executor_used,
                   COUNT(*) as total,
                   SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                   AVG(actual_duration_minutes) as avg_duration,
                   AVG(actual_cost_usd) as avg_cost
            FROM learning_records
            WHERE executor_used IS NOT NULL
            GROUP BY executor_used
        """)

        for row in cursor:
            self._executor_stats[row["executor_used"]] = {
                "total": row["total"],
                "success_rate": row["successes"] / row["total"] if row["total"] > 0 else 0,
                "avg_duration": row["avg_duration"] or 0,
                "avg_cost": row["avg_cost"] or 0,
            }

    async def record_outcome(self, item: Any, result: Any) -> str:
        """
        Record an execution outcome for learning.

        Args:
            item: The item that was executed (task, goal, etc.)
            result: The execution result

        Returns:
            ID of the learning record
        """
        import uuid

        record = LearningRecord(
            id=str(uuid.uuid4()),
            outcome_type=OutcomeType.EXECUTION,
            item_id=getattr(item, 'id', str(id(item))),
            item_type=type(item).__name__,
            item_title=getattr(item, 'title', str(item)[:100]),
            item_tags=getattr(item, 'tags', []) or [],
            executor_used=getattr(result, 'executor', 'unknown'),
            success=getattr(result, 'success', False),
            confidence_before=getattr(result, 'confidence', 0.5),
            actual_duration_minutes=getattr(result, 'duration_minutes', 0),
            actual_cost_usd=getattr(result, 'cost_usd', 0),
            error_message=getattr(result, 'error', None),
            context={"item_priority": str(getattr(item, 'priority', 'unknown'))},
        )

        await self._save_record(record)
        await self._update_patterns(record)

        return record.id

    async def record_approval(
        self,
        item: Any,
        approved: bool,
        notes: str = None
    ) -> str:
        """Record an approval decision for learning."""
        import uuid

        record_id = str(uuid.uuid4())
        item_id = getattr(item, 'id', str(id(item)))
        item_title = getattr(item, 'title', str(item)[:100])

        with self._lock:
            self._conn.execute("""
                INSERT INTO approval_history (id, item_id, item_title, approved, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (record_id, item_id, item_title, 1 if approved else 0, notes, datetime.now().isoformat()))
            self._conn.commit()

        # Also record as learning record
        record = LearningRecord(
            id=record_id,
            outcome_type=OutcomeType.APPROVAL,
            item_id=item_id,
            item_type=type(item).__name__,
            item_title=item_title,
            item_tags=getattr(item, 'tags', []) or [],
            executor_used="human_approval",
            success=approved,
            confidence_before=0.5,
            actual_duration_minutes=0,
            actual_cost_usd=0,
            notes=notes,
        )
        await self._save_record(record)

        return record_id

    async def record_feedback(
        self,
        item_id: str,
        feedback: str,
        rating: float = None
    ) -> str:
        """Record human feedback on an outcome."""
        import uuid

        with self._lock:
            # Find the most recent record for this item
            cursor = self._conn.execute("""
                SELECT id FROM learning_records
                WHERE item_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (item_id,))

            row = cursor.fetchone()
            if row:
                self._conn.execute("""
                    UPDATE learning_records
                    SET human_feedback = ?, notes = COALESCE(notes, '') || ?
                    WHERE id = ?
                """, (feedback, f"\nRating: {rating}" if rating else "", row["id"]))
                self._conn.commit()
                return row["id"]

        return ""

    async def _save_record(self, record: LearningRecord):
        """Save a learning record to the database."""
        data = record.to_dict()

        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO learning_records
                (id, outcome_type, item_id, item_type, item_title, item_tags,
                 executor_used, success, confidence_before, actual_duration_minutes,
                 actual_cost_usd, error_message, human_feedback, notes, context, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["id"], data["outcome_type"], data["item_id"], data["item_type"],
                data["item_title"], data["item_tags"], data["executor_used"],
                1 if data["success"] else 0, data["confidence_before"],
                data["actual_duration_minutes"], data["actual_cost_usd"],
                data["error_message"], data["human_feedback"], data["notes"],
                data["context"], data["created_at"]
            ))
            self._conn.commit()

    async def _update_patterns(self, record: LearningRecord):
        """Update pattern statistics based on a new record."""
        patterns = [
            ("executor", record.executor_used),
            ("item_type", record.item_type),
        ]

        # Add tag patterns
        for tag in record.item_tags:
            patterns.append(("tag", tag))

        with self._lock:
            for pattern_type, pattern_key in patterns:
                if not pattern_key:
                    continue

                self._conn.execute("""
                    INSERT INTO patterns (pattern_type, pattern_key, success_count, failure_count, last_updated)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(pattern_type, pattern_key) DO UPDATE SET
                        success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        last_updated = ?
                """, (
                    pattern_type, pattern_key,
                    1 if record.success else 0,
                    0 if record.success else 1,
                    datetime.now().isoformat(),
                    1 if record.success else 0,
                    0 if record.success else 1,
                    datetime.now().isoformat()
                ))

            self._conn.commit()

    async def get_similar_outcomes(self, item: Any, limit: int = 10) -> List[LearningRecord]:
        """
        Find similar past outcomes for confidence estimation.

        Args:
            item: The item to find similar outcomes for
            limit: Maximum number of records to return

        Returns:
            List of similar learning records
        """
        item_type = type(item).__name__
        tags = getattr(item, 'tags', []) or []

        # Build query based on available attributes
        conditions = ["item_type = ?"]
        params = [item_type]

        if tags:
            tag_conditions = ["item_tags LIKE ?"] * min(len(tags), 3)
            conditions.append(f"({' OR '.join(tag_conditions)})")
            params.extend([f"%{tag}%" for tag in tags[:3]])

        sql = f"""
            SELECT * FROM learning_records
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        records = []
        with self._lock:
            cursor = self._conn.execute(sql, params)
            for row in cursor:
                try:
                    record = LearningRecord.from_dict(dict(row))
                    records.append(record)
                except Exception as e:
                    logger.error(f"Error parsing learning record: {e}")

        return records

    async def get_recent_outcomes(self, limit: int = 20) -> List[LearningRecord]:
        """Get most recent outcomes."""
        with self._lock:
            cursor = self._conn.execute("""
                SELECT * FROM learning_records
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            return [LearningRecord.from_dict(dict(row)) for row in cursor]

    async def get_executor_stats(self, executor: str) -> Dict[str, float]:
        """Get performance statistics for an executor."""
        if executor in self._executor_stats:
            return self._executor_stats[executor]

        with self._lock:
            cursor = self._conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    AVG(actual_duration_minutes) as avg_duration,
                    AVG(actual_cost_usd) as avg_cost
                FROM learning_records
                WHERE executor_used = ?
            """, (executor,))

            row = cursor.fetchone()
            if row and row["total"] > 0:
                stats = {
                    "total": row["total"],
                    "success_rate": row["successes"] / row["total"],
                    "avg_duration": row["avg_duration"] or 0,
                    "avg_cost": row["avg_cost"] or 0,
                }
                self._executor_stats[executor] = stats
                return stats

        return {"total": 0, "success_rate": 0.5, "avg_duration": 5, "avg_cost": 0.5}

    async def get_confidence_adjustment(self, item: Any) -> float:
        """
        Calculate confidence adjustment based on historical data.

        Returns a multiplier to apply to base confidence.
        """
        similar = await self.get_similar_outcomes(item, limit=20)

        if len(similar) < 3:
            return 1.0  # Not enough data

        success_rate = sum(1 for r in similar if r.success) / len(similar)

        # Convert to adjustment factor (0.8 to 1.2)
        return 0.8 + (success_rate * 0.4)

    async def get_stats(self) -> LearningStats:
        """Get aggregated learning statistics."""
        with self._lock:
            cursor = self._conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    AVG(ABS(confidence_before - (CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END))) as confidence_error
                FROM learning_records
            """)

            row = cursor.fetchone()
            total = row["total"] or 0
            success_rate = (row["successes"] or 0) / total if total > 0 else 0
            confidence_accuracy = 1.0 - (row["confidence_error"] or 0.5)

            # Get best executors
            best_executors = {}
            for executor, stats in self._executor_stats.items():
                if stats["total"] >= 5:
                    best_executors[executor] = stats["success_rate"]

            # Get worst patterns
            cursor = self._conn.execute("""
                SELECT pattern_type, pattern_key,
                       failure_count * 1.0 / (success_count + failure_count) as failure_rate
                FROM patterns
                WHERE success_count + failure_count >= 5
                ORDER BY failure_rate DESC
                LIMIT 5
            """)

            worst_patterns = [
                f"{row['pattern_type']}:{row['pattern_key']} ({row['failure_rate']:.0%} failure)"
                for row in cursor
            ]

            # Calculate improvement trend
            cursor = self._conn.execute("""
                SELECT
                    (SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
                     FROM learning_records
                     WHERE created_at > datetime('now', '-7 days')) as recent,
                    (SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
                     FROM learning_records
                     WHERE created_at <= datetime('now', '-7 days')) as older
            """)

            trend_row = cursor.fetchone()
            recent = trend_row["recent"] or 0.5
            older = trend_row["older"] or 0.5
            improvement_trend = recent - older

        # Generate insights
        learnings = []
        if improvement_trend > 0.1:
            learnings.append(f"Performance improving: +{improvement_trend:.0%} in last 7 days")
        elif improvement_trend < -0.1:
            learnings.append(f"Performance declining: {improvement_trend:.0%} in last 7 days")

        if best_executors:
            best = max(best_executors.items(), key=lambda x: x[1])
            learnings.append(f"Best executor: {best[0]} ({best[1]:.0%} success)")

        if worst_patterns:
            learnings.append(f"Watch out for: {worst_patterns[0]}")

        return LearningStats(
            total_records=total,
            success_rate=success_rate,
            avg_confidence_accuracy=confidence_accuracy,
            best_executors=best_executors,
            worst_patterns=worst_patterns,
            improvement_trend=improvement_trend,
            top_learnings=learnings,
        )

    async def save(self):
        """Persist any pending changes."""
        with self._lock:
            if self._conn:
                self._conn.commit()

    async def close(self):
        """Close the database connection."""
        with self._lock:
            if self._conn:
                self._conn.commit()
                self._conn.close()
                self._conn = None
