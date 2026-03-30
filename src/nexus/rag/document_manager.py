"""
Document Manager - Track, version, and manage documents in RAG.

Provides:
- Document versioning and history
- Deduplication (content hash + similarity)
- Update and delete operations
- Source tracking and lineage
- Expiration and retention policies
"""

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentStatus(Enum):
    """Document status in the system."""
    ACTIVE = "active"
    DELETED = "deleted"
    EXPIRED = "expired"
    SUPERSEDED = "superseded"  # Replaced by newer version


@dataclass
class DocumentRecord:
    """Metadata record for a tracked document."""

    doc_id: str
    content_hash: str
    source: Optional[str]
    version: int
    status: DocumentStatus
    chunk_count: int

    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime]

    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_version_id: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.status == DocumentStatus.ACTIVE

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class DocumentManager:
    """
    Manages document lifecycle in RAG system.

    Features:
    - Content-based deduplication
    - Version tracking
    - Source lineage
    - Expiration handling
    - Efficient updates (only re-index changed content)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        dedup_enabled: bool = True,
        default_ttl_days: Optional[int] = None,
    ):
        """
        Initialize document manager.

        Args:
            storage_path: Path for SQLite database (None for in-memory)
            dedup_enabled: Enable content-based deduplication
            default_ttl_days: Default document expiration (None for no expiration)
        """
        self.storage_path = storage_path
        self.dedup_enabled = dedup_enabled
        self.default_ttl_days = default_ttl_days

        self._conn: Optional[sqlite3.Connection] = None
        self._content_hashes: Set[str] = set()

    def initialize(self) -> None:
        """Initialize the database."""
        if self.storage_path:
            Path(self.storage_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.storage_path)
        else:
            self._conn = sqlite3.connect(":memory:")

        self._create_tables()
        self._load_content_hashes()

        logger.info(f"DocumentManager initialized with {len(self._content_hashes)} documents")

    def _create_tables(self) -> None:
        """Create database tables."""
        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                source TEXT,
                version INTEGER DEFAULT 1,
                status TEXT DEFAULT 'active',
                chunk_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                metadata TEXT,
                previous_version_id TEXT,

                FOREIGN KEY (previous_version_id) REFERENCES documents(doc_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_content_hash ON documents(content_hash)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON documents(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON documents(source)
        """)

        self._conn.commit()

    def _load_content_hashes(self) -> None:
        """Load existing content hashes for deduplication."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT content_hash FROM documents WHERE status = 'active'"
        )
        self._content_hashes = {row[0] for row in cursor.fetchall()}

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def is_duplicate(self, content: str) -> bool:
        """Check if content already exists."""
        if not self.dedup_enabled:
            return False

        content_hash = self.compute_content_hash(content)
        return content_hash in self._content_hashes

    def get_duplicate_doc_id(self, content: str) -> Optional[str]:
        """Get doc_id of existing duplicate, if any."""
        if not self.dedup_enabled:
            return None

        content_hash = self.compute_content_hash(content)

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT doc_id FROM documents WHERE content_hash = ? AND status = 'active'",
            (content_hash,)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def register_document(
        self,
        doc_id: str,
        content: str,
        chunk_count: int,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl_days: Optional[int] = None,
    ) -> DocumentRecord:
        """
        Register a new document.

        Args:
            doc_id: Unique document identifier
            content: Document content (for hashing)
            chunk_count: Number of chunks indexed
            source: Source identifier (file path, URL, etc.)
            metadata: Additional metadata
            ttl_days: Days until expiration (None for no expiration)

        Returns:
            DocumentRecord for the registered document
        """
        content_hash = self.compute_content_hash(content)
        now = datetime.now(timezone.utc)

        ttl = ttl_days or self.default_ttl_days
        expires_at = now + timedelta(days=ttl) if ttl else None

        record = DocumentRecord(
            doc_id=doc_id,
            content_hash=content_hash,
            source=source,
            version=1,
            status=DocumentStatus.ACTIVE,
            chunk_count=chunk_count,
            created_at=now,
            updated_at=now,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        # Insert into database
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO documents (
                doc_id, content_hash, source, version, status, chunk_count,
                created_at, updated_at, expires_at, metadata, previous_version_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.doc_id,
            record.content_hash,
            record.source,
            record.version,
            record.status.value,
            record.chunk_count,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            record.expires_at.isoformat() if record.expires_at else None,
            json.dumps(record.metadata),
            record.previous_version_id,
        ))
        self._conn.commit()

        # Update hash set
        self._content_hashes.add(content_hash)

        logger.debug(f"Registered document {doc_id} ({chunk_count} chunks)")
        return record

    def update_document(
        self,
        doc_id: str,
        new_content: str,
        new_chunk_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DocumentRecord]:
        """
        Update an existing document with new content.

        Creates a new version, marks old version as superseded.

        Args:
            doc_id: Document to update
            new_content: New content
            new_chunk_count: New chunk count
            metadata: New metadata (merged with existing)

        Returns:
            New DocumentRecord, or None if document not found
        """
        existing = self.get_document(doc_id)
        if not existing:
            return None

        # Check if content actually changed
        new_hash = self.compute_content_hash(new_content)
        if new_hash == existing.content_hash:
            logger.debug(f"Document {doc_id} content unchanged, skipping update")
            return existing

        # Mark old version as superseded
        self._update_status(doc_id, DocumentStatus.SUPERSEDED)

        # Create new version
        new_version = existing.version + 1
        new_doc_id = f"{doc_id}_v{new_version}"

        now = datetime.now(timezone.utc)
        merged_metadata = {**existing.metadata, **(metadata or {})}

        record = DocumentRecord(
            doc_id=new_doc_id,
            content_hash=new_hash,
            source=existing.source,
            version=new_version,
            status=DocumentStatus.ACTIVE,
            chunk_count=new_chunk_count,
            created_at=now,
            updated_at=now,
            expires_at=existing.expires_at,
            metadata=merged_metadata,
            previous_version_id=doc_id,
        )

        # Insert new version
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO documents (
                doc_id, content_hash, source, version, status, chunk_count,
                created_at, updated_at, expires_at, metadata, previous_version_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.doc_id,
            record.content_hash,
            record.source,
            record.version,
            record.status.value,
            record.chunk_count,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            record.expires_at.isoformat() if record.expires_at else None,
            json.dumps(record.metadata),
            record.previous_version_id,
        ))
        self._conn.commit()

        # Update hash sets
        self._content_hashes.discard(existing.content_hash)
        self._content_hashes.add(new_hash)

        logger.info(f"Updated document {doc_id} -> {new_doc_id} (v{new_version})")
        return record

    def delete_document(self, doc_id: str, hard_delete: bool = False) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document to delete
            hard_delete: If True, remove from database. If False, mark as deleted.

        Returns:
            True if deleted, False if not found
        """
        existing = self.get_document(doc_id)
        if not existing:
            return False

        if hard_delete:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            self._conn.commit()
        else:
            self._update_status(doc_id, DocumentStatus.DELETED)

        self._content_hashes.discard(existing.content_hash)

        logger.info(f"Deleted document {doc_id} (hard={hard_delete})")
        return True

    def _update_status(self, doc_id: str, status: DocumentStatus) -> None:
        """Update document status."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE doc_id = ?",
            (status.value, datetime.now(timezone.utc).isoformat(), doc_id)
        )
        self._conn.commit()

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Get document record by ID."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM documents WHERE doc_id = ?",
            (doc_id,)
        )
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def get_documents_by_source(self, source: str) -> List[DocumentRecord]:
        """Get all documents from a source."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM documents WHERE source = ? AND status = 'active'",
            (source,)
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_expired_documents(self) -> List[DocumentRecord]:
        """Get all expired documents."""
        cursor = self._conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute(
            "SELECT * FROM documents WHERE status = 'active' AND expires_at IS NOT NULL AND expires_at < ?",
            (now,)
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_document_history(self, doc_id: str) -> List[DocumentRecord]:
        """Get version history of a document."""
        # Find the root document ID (strip version suffix)
        base_id = doc_id.split("_v")[0]

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM documents WHERE doc_id LIKE ? ORDER BY version ASC",
            (f"{base_id}%",)
        )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def expire_documents(self) -> int:
        """Mark expired documents and return count."""
        expired = self.get_expired_documents()

        for doc in expired:
            self._update_status(doc.doc_id, DocumentStatus.EXPIRED)
            self._content_hashes.discard(doc.content_hash)

        if expired:
            logger.info(f"Expired {len(expired)} documents")

        return len(expired)

    def get_stats(self) -> Dict[str, Any]:
        """Get document statistics."""
        cursor = self._conn.cursor()

        # Count by status
        cursor.execute(
            "SELECT status, COUNT(*) FROM documents GROUP BY status"
        )
        status_counts = dict(cursor.fetchall())

        # Total chunks
        cursor.execute(
            "SELECT SUM(chunk_count) FROM documents WHERE status = 'active'"
        )
        total_chunks = cursor.fetchone()[0] or 0

        # Sources
        cursor.execute(
            "SELECT COUNT(DISTINCT source) FROM documents WHERE status = 'active'"
        )
        source_count = cursor.fetchone()[0] or 0

        return {
            "total_documents": sum(status_counts.values()),
            "active_documents": status_counts.get("active", 0),
            "deleted_documents": status_counts.get("deleted", 0),
            "expired_documents": status_counts.get("expired", 0),
            "superseded_documents": status_counts.get("superseded", 0),
            "total_chunks": total_chunks,
            "unique_sources": source_count,
            "dedup_enabled": self.dedup_enabled,
        }

    def _row_to_record(self, row) -> DocumentRecord:
        """Convert database row to DocumentRecord."""
        return DocumentRecord(
            doc_id=row[0],
            content_hash=row[1],
            source=row[2],
            version=row[3],
            status=DocumentStatus(row[4]),
            chunk_count=row[5],
            created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]),
            expires_at=datetime.fromisoformat(row[8]) if row[8] else None,
            metadata=json.loads(row[9]) if row[9] else {},
            previous_version_id=row[10],
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
