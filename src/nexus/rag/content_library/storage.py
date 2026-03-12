"""
Pluggable Storage Backends for Content Library.

Provides multiple storage options:
- InMemoryStorage: Fast, ephemeral storage for development/testing
- FileStorage: Persistent JSON/YAML file-based storage
- HybridStorage: In-memory cache with file persistence

All backends implement ContentStorageBackend abstract interface.
"""

import json
import os
import logging
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from collections import OrderedDict

from .models import (
    ContentItem,
    ContentVersion,
    ContentFilters,
    ContentStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Storage Backend
# =============================================================================

class ContentStorageBackend(ABC):
    """
    Abstract interface for content storage backends.

    All storage implementations must implement these methods.
    """

    @abstractmethod
    def save(self, content: ContentItem) -> str:
        """
        Save content item to storage.

        Args:
            content: ContentItem to save

        Returns:
            content_id of saved item
        """
        pass

    @abstractmethod
    def get(self, content_id: str) -> Optional[ContentItem]:
        """
        Retrieve content item by ID.

        Args:
            content_id: ID of content to retrieve

        Returns:
            ContentItem if found, None otherwise
        """
        pass

    @abstractmethod
    def update(self, content_id: str, content: ContentItem) -> bool:
        """
        Update existing content item.

        Args:
            content_id: ID of content to update
            content: Updated ContentItem

        Returns:
            True if successful, False if content not found
        """
        pass

    @abstractmethod
    def delete(self, content_id: str, hard_delete: bool = False) -> bool:
        """
        Delete content item.

        Args:
            content_id: ID of content to delete
            hard_delete: If True, permanently delete; if False, soft delete (archive)

        Returns:
            True if successful, False if content not found
        """
        pass

    @abstractmethod
    def list(self, filters: Optional[ContentFilters] = None) -> List[ContentItem]:
        """
        List content items with optional filtering.

        Args:
            filters: Optional ContentFilters to apply

        Returns:
            List of matching ContentItem objects
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 20) -> List[ContentItem]:
        """
        Search content by text query.

        Args:
            query: Search query string
            limit: Maximum results to return

        Returns:
            List of matching ContentItem objects
        """
        pass

    @abstractmethod
    def exists(self, content_id: str) -> bool:
        """
        Check if content exists.

        Args:
            content_id: ID to check

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    def count(self, filters: Optional[ContentFilters] = None) -> int:
        """
        Count content items with optional filtering.

        Args:
            filters: Optional ContentFilters to apply

        Returns:
            Count of matching items
        """
        pass

    # Version management
    @abstractmethod
    def save_version(self, version: ContentVersion) -> str:
        """
        Save content version.

        Args:
            version: ContentVersion to save

        Returns:
            version_id of saved version
        """
        pass

    @abstractmethod
    def get_versions(self, content_id: str) -> List[ContentVersion]:
        """
        Get all versions of content.

        Args:
            content_id: ID of content

        Returns:
            List of ContentVersion objects, ordered by version number
        """
        pass

    @abstractmethod
    def get_version(self, content_id: str, version_number: int) -> Optional[ContentVersion]:
        """
        Get specific version of content.

        Args:
            content_id: ID of content
            version_number: Version number to retrieve

        Returns:
            ContentVersion if found, None otherwise
        """
        pass


# =============================================================================
# In-Memory Storage
# =============================================================================

class InMemoryStorage(ContentStorageBackend):
    """
    In-memory content storage for development and testing.

    Fast but ephemeral - data is lost when process ends.
    """

    def __init__(self):
        self.content: Dict[str, ContentItem] = {}
        self.versions: Dict[str, List[ContentVersion]] = {}  # content_id -> versions
        self.deleted: Dict[str, ContentItem] = {}  # Soft-deleted items
        logger.info("InMemoryStorage initialized")

    def save(self, content: ContentItem) -> str:
        """Save content to memory."""
        self.content[content.content_id] = content
        logger.debug(f"Saved content: {content.content_id}")
        return content.content_id

    def get(self, content_id: str) -> Optional[ContentItem]:
        """Get content from memory."""
        return self.content.get(content_id)

    def update(self, content_id: str, content: ContentItem) -> bool:
        """Update content in memory."""
        if content_id not in self.content:
            return False
        content.updated_at = datetime.now()
        self.content[content_id] = content
        logger.debug(f"Updated content: {content_id}")
        return True

    def delete(self, content_id: str, hard_delete: bool = False) -> bool:
        """Delete content from memory."""
        if content_id not in self.content:
            return False

        if hard_delete:
            del self.content[content_id]
            # Also delete versions
            if content_id in self.versions:
                del self.versions[content_id]
            logger.debug(f"Hard deleted content: {content_id}")
        else:
            # Soft delete - move to deleted storage
            content = self.content.pop(content_id)
            content.status = ContentStatus.ARCHIVED
            self.deleted[content_id] = content
            logger.debug(f"Soft deleted content: {content_id}")

        return True

    def list(self, filters: Optional[ContentFilters] = None) -> List[ContentItem]:
        """List content with optional filtering."""
        items = list(self.content.values())

        if filters:
            items = [item for item in items if filters.matches(item)]

            # Apply sorting
            reverse = filters.sort_order == "desc"
            if filters.sort_by:
                items.sort(
                    key=lambda x: getattr(x, filters.sort_by, None) or datetime.min,
                    reverse=reverse
                )

            # Apply pagination
            items = items[filters.offset:filters.offset + filters.limit]

        return items

    def search(self, query: str, limit: int = 20) -> List[ContentItem]:
        """Search content by query."""
        query_lower = query.lower()
        results = []

        for content in self.content.values():
            score = 0
            # Title match (highest weight)
            if query_lower in content.title.lower():
                score += 3
            # Description match
            if query_lower in content.description.lower():
                score += 2
            # Body match
            if query_lower in content.content_body.lower():
                score += 1
            # Topic match
            if any(query_lower in t.lower() for t in content.topics):
                score += 2
            # Tag match
            if any(query_lower in t.lower() for t in content.tags):
                score += 1

            if score > 0:
                results.append((score, content))

        # Sort by score
        results.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in results[:limit]]

    def exists(self, content_id: str) -> bool:
        """Check if content exists."""
        return content_id in self.content

    def count(self, filters: Optional[ContentFilters] = None) -> int:
        """Count content items."""
        if filters:
            return len([item for item in self.content.values() if filters.matches(item)])
        return len(self.content)

    def save_version(self, version: ContentVersion) -> str:
        """Save content version."""
        if version.content_id not in self.versions:
            self.versions[version.content_id] = []
        self.versions[version.content_id].append(version)
        logger.debug(f"Saved version {version.version_number} for content {version.content_id}")
        return version.version_id

    def get_versions(self, content_id: str) -> List[ContentVersion]:
        """Get all versions for content."""
        versions = self.versions.get(content_id, [])
        return sorted(versions, key=lambda v: v.version_number)

    def get_version(self, content_id: str, version_number: int) -> Optional[ContentVersion]:
        """Get specific version."""
        versions = self.versions.get(content_id, [])
        for version in versions:
            if version.version_number == version_number:
                return version
        return None

    def clear(self):
        """Clear all storage."""
        self.content.clear()
        self.versions.clear()
        self.deleted.clear()
        logger.info("InMemoryStorage cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_content": len(self.content),
            "total_versions": sum(len(v) for v in self.versions.values()),
            "deleted_content": len(self.deleted),
            "by_status": {
                status.value: len([c for c in self.content.values() if c.status == status])
                for status in ContentStatus
            }
        }


# =============================================================================
# File-Based Storage
# =============================================================================

class FileStorage(ContentStorageBackend):
    """
    File-based content storage with JSON serialization.

    Directory structure:
        base_path/
        ├── content/
        │   ├── {content_id}/
        │   │   ├── current.json
        │   │   └── versions/
        │   │       ├── v1.json
        │   │       ├── v2.json
        │   │       └── ...
        │   └── ...
        ├── index.json
        └── deleted/
            └── {content_id}.json
    """

    def __init__(self, base_path: str, format: str = "json"):
        """
        Initialize file storage.

        Args:
            base_path: Root directory for storage
            format: Serialization format ("json" supported)
        """
        self.base_path = Path(base_path)
        self.format = format

        # Create directory structure
        self.content_path = self.base_path / "content"
        self.deleted_path = self.base_path / "deleted"
        self.index_path = self.base_path / "index.json"

        self._ensure_directories()
        self._load_index()

        logger.info(f"FileStorage initialized at {base_path}")

    def _ensure_directories(self):
        """Create required directories."""
        self.content_path.mkdir(parents=True, exist_ok=True)
        self.deleted_path.mkdir(parents=True, exist_ok=True)

    def _load_index(self):
        """Load content index from file."""
        if self.index_path.exists():
            with open(self.index_path, 'r', encoding='utf-8') as f:
                self.index = json.load(f)
        else:
            self.index = {"content_ids": [], "updated_at": None}

    def _save_index(self):
        """Save content index to file."""
        self.index["updated_at"] = datetime.now().isoformat()
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f, indent=2)

    def _content_dir(self, content_id: str) -> Path:
        """Get directory path for content."""
        return self.content_path / content_id

    def _content_file(self, content_id: str) -> Path:
        """Get current content file path."""
        return self._content_dir(content_id) / "current.json"

    def _versions_dir(self, content_id: str) -> Path:
        """Get versions directory for content."""
        return self._content_dir(content_id) / "versions"

    def _version_file(self, content_id: str, version_number: int) -> Path:
        """Get version file path."""
        return self._versions_dir(content_id) / f"v{version_number}.json"

    def _write_json(self, path: Path, data: Dict[str, Any]):
        """Write JSON to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _read_json(self, path: Path) -> Optional[Dict[str, Any]]:
        """Read JSON from file."""
        if not path.exists():
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save(self, content: ContentItem) -> str:
        """Save content to file."""
        content_dir = self._content_dir(content.content_id)
        content_dir.mkdir(parents=True, exist_ok=True)

        # Save current content
        self._write_json(self._content_file(content.content_id), content.to_dict())

        # Update index
        if content.content_id not in self.index["content_ids"]:
            self.index["content_ids"].append(content.content_id)
            self._save_index()

        logger.debug(f"Saved content to file: {content.content_id}")
        return content.content_id

    def get(self, content_id: str) -> Optional[ContentItem]:
        """Get content from file."""
        data = self._read_json(self._content_file(content_id))
        if data:
            return ContentItem.from_dict(data)
        return None

    def update(self, content_id: str, content: ContentItem) -> bool:
        """Update content in file."""
        if not self._content_file(content_id).exists():
            return False

        content.updated_at = datetime.now()
        self._write_json(self._content_file(content_id), content.to_dict())
        logger.debug(f"Updated content file: {content_id}")
        return True

    def delete(self, content_id: str, hard_delete: bool = False) -> bool:
        """Delete content file."""
        content_dir = self._content_dir(content_id)
        if not content_dir.exists():
            return False

        if hard_delete:
            # Remove directory and all versions
            shutil.rmtree(content_dir)
            if content_id in self.index["content_ids"]:
                self.index["content_ids"].remove(content_id)
                self._save_index()
            logger.debug(f"Hard deleted content: {content_id}")
        else:
            # Move to deleted folder
            content = self.get(content_id)
            if content:
                content.status = ContentStatus.ARCHIVED
                self._write_json(self.deleted_path / f"{content_id}.json", content.to_dict())
                shutil.rmtree(content_dir)
                if content_id in self.index["content_ids"]:
                    self.index["content_ids"].remove(content_id)
                    self._save_index()
            logger.debug(f"Soft deleted content: {content_id}")

        return True

    def list(self, filters: Optional[ContentFilters] = None) -> List[ContentItem]:
        """List content with optional filtering."""
        items = []

        for content_id in self.index.get("content_ids", []):
            content = self.get(content_id)
            if content:
                if filters is None or filters.matches(content):
                    items.append(content)

        if filters:
            # Apply sorting
            reverse = filters.sort_order == "desc"
            if filters.sort_by:
                items.sort(
                    key=lambda x: getattr(x, filters.sort_by, None) or datetime.min,
                    reverse=reverse
                )

            # Apply pagination
            items = items[filters.offset:filters.offset + filters.limit]

        return items

    def search(self, query: str, limit: int = 20) -> List[ContentItem]:
        """Search content by query."""
        query_lower = query.lower()
        results = []

        for content_id in self.index.get("content_ids", []):
            content = self.get(content_id)
            if not content:
                continue

            score = 0
            if query_lower in content.title.lower():
                score += 3
            if query_lower in content.description.lower():
                score += 2
            if query_lower in content.content_body.lower():
                score += 1
            if any(query_lower in t.lower() for t in content.topics):
                score += 2
            if any(query_lower in t.lower() for t in content.tags):
                score += 1

            if score > 0:
                results.append((score, content))

        results.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in results[:limit]]

    def exists(self, content_id: str) -> bool:
        """Check if content exists."""
        return self._content_file(content_id).exists()

    def count(self, filters: Optional[ContentFilters] = None) -> int:
        """Count content items."""
        if filters:
            return len(self.list(filters))
        return len(self.index.get("content_ids", []))

    def save_version(self, version: ContentVersion) -> str:
        """Save content version."""
        versions_dir = self._versions_dir(version.content_id)
        versions_dir.mkdir(parents=True, exist_ok=True)

        self._write_json(
            self._version_file(version.content_id, version.version_number),
            version.to_dict()
        )
        logger.debug(f"Saved version {version.version_number} for {version.content_id}")
        return version.version_id

    def get_versions(self, content_id: str) -> List[ContentVersion]:
        """Get all versions for content."""
        versions = []
        versions_dir = self._versions_dir(content_id)

        if versions_dir.exists():
            for version_file in sorted(versions_dir.glob("v*.json")):
                data = self._read_json(version_file)
                if data:
                    versions.append(ContentVersion.from_dict(data))

        return sorted(versions, key=lambda v: v.version_number)

    def get_version(self, content_id: str, version_number: int) -> Optional[ContentVersion]:
        """Get specific version."""
        data = self._read_json(self._version_file(content_id, version_number))
        if data:
            return ContentVersion.from_dict(data)
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(
            f.stat().st_size
            for f in self.base_path.rglob("*")
            if f.is_file()
        )

        return {
            "total_content": len(self.index.get("content_ids", [])),
            "storage_path": str(self.base_path),
            "total_size_bytes": total_size,
            "format": self.format
        }


# =============================================================================
# Hybrid Storage (In-Memory Cache + File Persistence)
# =============================================================================

class LRUCache:
    """Simple LRU cache implementation."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving to end (most recent)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        """Put item in cache."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def delete(self, key: str):
        """Delete item from cache."""
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear cache."""
        self.cache.clear()

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)


class HybridStorage(ContentStorageBackend):
    """
    Hybrid storage combining in-memory cache with file persistence.

    Provides fast reads via LRU cache while ensuring durability through
    file storage backend.
    """

    def __init__(self, file_storage: FileStorage, cache_size: int = 1000):
        """
        Initialize hybrid storage.

        Args:
            file_storage: FileStorage backend for persistence
            cache_size: Maximum items to keep in cache
        """
        self.file_storage = file_storage
        self.cache = LRUCache(cache_size)
        self.version_cache = LRUCache(cache_size // 2)

        logger.info(f"HybridStorage initialized with cache_size={cache_size}")

    def save(self, content: ContentItem) -> str:
        """Save content to both cache and file."""
        # Save to file first (durability)
        result = self.file_storage.save(content)

        # Update cache
        self.cache.put(content.content_id, content)

        return result

    def get(self, content_id: str) -> Optional[ContentItem]:
        """Get content from cache or file."""
        # Check cache first
        cached = self.cache.get(content_id)
        if cached:
            return cached

        # Load from file
        content = self.file_storage.get(content_id)
        if content:
            self.cache.put(content_id, content)
        return content

    def update(self, content_id: str, content: ContentItem) -> bool:
        """Update content in both cache and file."""
        # Update file first
        result = self.file_storage.update(content_id, content)
        if result:
            # Update cache
            self.cache.put(content_id, content)
        return result

    def delete(self, content_id: str, hard_delete: bool = False) -> bool:
        """Delete content from both cache and file."""
        # Delete from file first
        result = self.file_storage.delete(content_id, hard_delete)
        if result:
            # Remove from cache
            self.cache.delete(content_id)
        return result

    def list(self, filters: Optional[ContentFilters] = None) -> List[ContentItem]:
        """List content (uses file storage for complete listing)."""
        # For listing, we go to file storage to ensure completeness
        items = self.file_storage.list(filters)

        # Populate cache with results
        for item in items:
            self.cache.put(item.content_id, item)

        return items

    def search(self, query: str, limit: int = 20) -> List[ContentItem]:
        """Search content."""
        results = self.file_storage.search(query, limit)

        # Populate cache
        for item in results:
            self.cache.put(item.content_id, item)

        return results

    def exists(self, content_id: str) -> bool:
        """Check if content exists."""
        if content_id in self.cache:
            return True
        return self.file_storage.exists(content_id)

    def count(self, filters: Optional[ContentFilters] = None) -> int:
        """Count content items."""
        return self.file_storage.count(filters)

    def save_version(self, version: ContentVersion) -> str:
        """Save version to both cache and file."""
        result = self.file_storage.save_version(version)

        # Cache version
        cache_key = f"{version.content_id}:v{version.version_number}"
        self.version_cache.put(cache_key, version)

        return result

    def get_versions(self, content_id: str) -> List[ContentVersion]:
        """Get all versions."""
        versions = self.file_storage.get_versions(content_id)

        # Cache versions
        for version in versions:
            cache_key = f"{content_id}:v{version.version_number}"
            self.version_cache.put(cache_key, version)

        return versions

    def get_version(self, content_id: str, version_number: int) -> Optional[ContentVersion]:
        """Get specific version from cache or file."""
        cache_key = f"{content_id}:v{version_number}"

        # Check cache
        cached = self.version_cache.get(cache_key)
        if cached:
            return cached

        # Load from file
        version = self.file_storage.get_version(content_id, version_number)
        if version:
            self.version_cache.put(cache_key, version)
        return version

    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache.clear()
        self.version_cache.clear()
        logger.info("HybridStorage cache cleared")

    def warm_cache(self, content_ids: Optional[List[str]] = None):
        """Pre-populate cache with content."""
        if content_ids is None:
            # Warm with all content
            for content in self.file_storage.list():
                self.cache.put(content.content_id, content)
        else:
            # Warm with specific content
            for content_id in content_ids:
                content = self.file_storage.get(content_id)
                if content:
                    self.cache.put(content_id, content)

        logger.info(f"Warmed cache with {len(self.cache)} items")

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        file_stats = self.file_storage.get_statistics()
        return {
            **file_stats,
            "cache_size": len(self.cache),
            "version_cache_size": len(self.version_cache),
            "cache_max_size": self.cache.max_size
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_storage(
    storage_type: str = "memory",
    base_path: Optional[str] = None,
    cache_size: int = 1000
) -> ContentStorageBackend:
    """
    Factory function to create storage backend.

    Args:
        storage_type: Type of storage ("memory", "file", "hybrid")
        base_path: Base path for file storage (required for file/hybrid)
        cache_size: Cache size for hybrid storage

    Returns:
        ContentStorageBackend instance
    """
    if storage_type == "memory":
        return InMemoryStorage()
    elif storage_type == "file":
        if not base_path:
            raise ValueError("base_path required for file storage")
        return FileStorage(base_path)
    elif storage_type == "hybrid":
        if not base_path:
            raise ValueError("base_path required for hybrid storage")
        file_storage = FileStorage(base_path)
        return HybridStorage(file_storage, cache_size)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
