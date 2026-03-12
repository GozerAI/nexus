"""
Asset Management for Content Library.

Provides full multimedia asset storage and management:
- FileAssetStorage: Local filesystem storage
- AssetManager: High-level asset operations
- Support for images, videos, audio, documents, code files
- Metadata extraction and thumbnail generation
"""

import hashlib
import mimetypes
import os
import shutil
import uuid
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json

from .models import ContentAsset, AssetType

logger = logging.getLogger(__name__)


# =============================================================================
# Abstract Asset Storage Backend
# =============================================================================

class AssetStorageBackend(ABC):
    """
    Abstract interface for asset storage.
    """

    @abstractmethod
    def save_asset(
        self,
        asset_id: str,
        file_data: bytes,
        filename: str,
        asset_type: AssetType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save asset to storage.

        Args:
            asset_id: Unique asset identifier
            file_data: Binary file data
            filename: Original filename
            asset_type: Type of asset
            metadata: Optional metadata

        Returns:
            Path or URL to stored asset
        """
        pass

    @abstractmethod
    def get_asset(self, asset_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """
        Retrieve asset data and metadata.

        Args:
            asset_id: Asset identifier

        Returns:
            Tuple of (file_data, metadata) if found, None otherwise
        """
        pass

    @abstractmethod
    def delete_asset(self, asset_id: str) -> bool:
        """
        Delete asset from storage.

        Args:
            asset_id: Asset identifier

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def get_asset_path(self, asset_id: str) -> Optional[str]:
        """
        Get the file path or URL for an asset.

        Args:
            asset_id: Asset identifier

        Returns:
            Path/URL string if found, None otherwise
        """
        pass

    @abstractmethod
    def exists(self, asset_id: str) -> bool:
        """Check if asset exists."""
        pass

    @abstractmethod
    def get_metadata(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset metadata without loading file data."""
        pass


# =============================================================================
# File-Based Asset Storage
# =============================================================================

class FileAssetStorage(AssetStorageBackend):
    """
    Local filesystem asset storage.

    Directory structure:
        base_path/
        ├── images/
        │   └── {asset_id}/
        │       ├── original.{ext}
        │       ├── metadata.json
        │       └── thumbnails/
        │           ├── small.jpg
        │           └── medium.jpg
        ├── videos/
        ├── audio/
        ├── documents/
        └── ...
    """

    def __init__(self, base_path: str):
        """
        Initialize file asset storage.

        Args:
            base_path: Root directory for asset storage
        """
        self.base_path = Path(base_path)
        self._ensure_directories()
        logger.info(f"FileAssetStorage initialized at {base_path}")

    def _ensure_directories(self):
        """Create required directories for each asset type."""
        for asset_type in AssetType:
            (self.base_path / asset_type.value).mkdir(parents=True, exist_ok=True)

    def _asset_dir(self, asset_type: AssetType, asset_id: str) -> Path:
        """Get directory for specific asset."""
        return self.base_path / asset_type.value / asset_id

    def _metadata_path(self, asset_type: AssetType, asset_id: str) -> Path:
        """Get metadata file path."""
        return self._asset_dir(asset_type, asset_id) / "metadata.json"

    def _file_path(self, asset_type: AssetType, asset_id: str, filename: str) -> Path:
        """Get asset file path."""
        return self._asset_dir(asset_type, asset_id) / filename

    def save_asset(
        self,
        asset_id: str,
        file_data: bytes,
        filename: str,
        asset_type: AssetType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save asset to filesystem."""
        # Create asset directory
        asset_dir = self._asset_dir(asset_type, asset_id)
        asset_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        ext = Path(filename).suffix or ""
        stored_filename = f"original{ext}"

        # Save file
        file_path = asset_dir / stored_filename
        with open(file_path, 'wb') as f:
            f.write(file_data)

        # Calculate checksum
        checksum = hashlib.sha256(file_data).hexdigest()

        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(filename)

        # Build metadata
        full_metadata = {
            "asset_id": asset_id,
            "asset_type": asset_type.value,
            "original_filename": filename,
            "stored_filename": stored_filename,
            "file_path": str(file_path),
            "size_bytes": len(file_data),
            "checksum": checksum,
            "checksum_algorithm": "sha256",
            "mime_type": mime_type or "application/octet-stream",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {})
        }

        # Save metadata
        with open(self._metadata_path(asset_type, asset_id), 'w') as f:
            json.dump(full_metadata, f, indent=2)

        logger.debug(f"Saved asset {asset_id} to {file_path}")
        return str(file_path)

    def get_asset(self, asset_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Retrieve asset data and metadata."""
        # Find asset by searching all types
        for asset_type in AssetType:
            metadata_path = self._metadata_path(asset_type, asset_id)
            if metadata_path.exists():
                # Load metadata
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Load file data
                file_path = Path(metadata.get("file_path", ""))
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    return file_data, metadata

        return None

    def delete_asset(self, asset_id: str) -> bool:
        """Delete asset from filesystem."""
        for asset_type in AssetType:
            asset_dir = self._asset_dir(asset_type, asset_id)
            if asset_dir.exists():
                shutil.rmtree(asset_dir)
                logger.debug(f"Deleted asset {asset_id}")
                return True
        return False

    def get_asset_path(self, asset_id: str) -> Optional[str]:
        """Get file path for asset."""
        for asset_type in AssetType:
            metadata_path = self._metadata_path(asset_type, asset_id)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata.get("file_path")
        return None

    def exists(self, asset_id: str) -> bool:
        """Check if asset exists."""
        for asset_type in AssetType:
            if self._metadata_path(asset_type, asset_id).exists():
                return True
        return False

    def get_metadata(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset metadata."""
        for asset_type in AssetType:
            metadata_path = self._metadata_path(asset_type, asset_id)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        return None

    def list_assets(self, asset_type: Optional[AssetType] = None) -> List[str]:
        """List all asset IDs, optionally filtered by type."""
        asset_ids = []
        types_to_check = [asset_type] if asset_type else list(AssetType)

        for atype in types_to_check:
            type_dir = self.base_path / atype.value
            if type_dir.exists():
                for item in type_dir.iterdir():
                    if item.is_dir():
                        asset_ids.append(item.name)

        return asset_ids

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_assets": 0,
            "total_size_bytes": 0,
            "by_type": {}
        }

        for asset_type in AssetType:
            type_dir = self.base_path / asset_type.value
            if type_dir.exists():
                count = 0
                size = 0
                for item in type_dir.iterdir():
                    if item.is_dir():
                        count += 1
                        for file in item.rglob("*"):
                            if file.is_file():
                                size += file.stat().st_size

                stats["by_type"][asset_type.value] = {
                    "count": count,
                    "size_bytes": size
                }
                stats["total_assets"] += count
                stats["total_size_bytes"] += size

        return stats


# =============================================================================
# In-Memory Asset Storage
# =============================================================================

class InMemoryAssetStorage(AssetStorageBackend):
    """
    In-memory asset storage for testing.
    """

    def __init__(self):
        self.assets: Dict[str, Tuple[bytes, Dict[str, Any]]] = {}
        logger.info("InMemoryAssetStorage initialized")

    def save_asset(
        self,
        asset_id: str,
        file_data: bytes,
        filename: str,
        asset_type: AssetType,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save asset to memory."""
        mime_type, _ = mimetypes.guess_type(filename)

        full_metadata = {
            "asset_id": asset_id,
            "asset_type": asset_type.value,
            "original_filename": filename,
            "size_bytes": len(file_data),
            "checksum": hashlib.sha256(file_data).hexdigest(),
            "checksum_algorithm": "sha256",
            "mime_type": mime_type or "application/octet-stream",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **(metadata or {})
        }

        self.assets[asset_id] = (file_data, full_metadata)
        return f"memory://{asset_id}"

    def get_asset(self, asset_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        """Get asset from memory."""
        return self.assets.get(asset_id)

    def delete_asset(self, asset_id: str) -> bool:
        """Delete asset from memory."""
        if asset_id in self.assets:
            del self.assets[asset_id]
            return True
        return False

    def get_asset_path(self, asset_id: str) -> Optional[str]:
        """Get virtual path."""
        if asset_id in self.assets:
            return f"memory://{asset_id}"
        return None

    def exists(self, asset_id: str) -> bool:
        """Check if asset exists."""
        return asset_id in self.assets

    def get_metadata(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """Get asset metadata."""
        if asset_id in self.assets:
            return self.assets[asset_id][1]
        return None

    def clear(self):
        """Clear all assets."""
        self.assets.clear()


# =============================================================================
# Asset Manager
# =============================================================================

class AssetManager:
    """
    High-level asset management interface.

    Provides convenient methods for:
    - Uploading assets from various sources
    - Metadata extraction
    - Thumbnail generation (if PIL available)
    - Asset validation
    """

    # Supported MIME types by category
    SUPPORTED_TYPES = {
        AssetType.IMAGE: [
            "image/jpeg", "image/png", "image/gif", "image/webp",
            "image/svg+xml", "image/bmp", "image/tiff"
        ],
        AssetType.VIDEO: [
            "video/mp4", "video/webm", "video/ogg", "video/avi",
            "video/quicktime", "video/x-msvideo"
        ],
        AssetType.AUDIO: [
            "audio/mpeg", "audio/wav", "audio/ogg", "audio/webm",
            "audio/aac", "audio/flac"
        ],
        AssetType.DOCUMENT: [
            "application/pdf", "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/plain", "text/markdown", "text/html"
        ],
        AssetType.CODE: [
            "text/x-python", "text/javascript", "application/json",
            "text/x-java", "text/x-c", "text/x-c++", "text/x-rust",
            "text/x-go", "text/x-ruby", "text/x-typescript"
        ],
    }

    # Maximum file sizes by type (bytes)
    MAX_SIZES = {
        AssetType.IMAGE: 50 * 1024 * 1024,      # 50 MB
        AssetType.VIDEO: 500 * 1024 * 1024,     # 500 MB
        AssetType.AUDIO: 100 * 1024 * 1024,     # 100 MB
        AssetType.DOCUMENT: 50 * 1024 * 1024,   # 50 MB
        AssetType.CODE: 10 * 1024 * 1024,       # 10 MB
    }

    def __init__(self, storage: AssetStorageBackend):
        """
        Initialize asset manager.

        Args:
            storage: Asset storage backend
        """
        self.storage = storage
        self._pil_available = self._check_pil()
        logger.info(f"AssetManager initialized (PIL available: {self._pil_available})")

    def _check_pil(self) -> bool:
        """Check if PIL/Pillow is available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False

    def upload_from_file(
        self,
        file_path: str,
        asset_type: Optional[AssetType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContentAsset:
        """
        Upload asset from file path.

        Args:
            file_path: Path to file
            asset_type: Type of asset (auto-detected if not provided)
            metadata: Additional metadata

        Returns:
            ContentAsset object
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'rb') as f:
            file_data = f.read()

        return self.upload_from_bytes(
            file_data,
            path.name,
            asset_type=asset_type,
            metadata=metadata
        )

    def upload_from_bytes(
        self,
        file_data: bytes,
        filename: str,
        asset_type: Optional[AssetType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContentAsset:
        """
        Upload asset from bytes.

        Args:
            file_data: Binary file data
            filename: Original filename
            asset_type: Type of asset (auto-detected if not provided)
            metadata: Additional metadata

        Returns:
            ContentAsset object
        """
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        mime_type = mime_type or "application/octet-stream"

        # Auto-detect asset type
        if asset_type is None:
            asset_type = self._detect_asset_type(mime_type, filename)

        # Validate
        self._validate_asset(file_data, mime_type, asset_type)

        # Generate asset ID
        asset_id = str(uuid.uuid4())

        # Extract metadata
        extracted_metadata = self._extract_metadata(file_data, mime_type, asset_type)
        full_metadata = {**(metadata or {}), **extracted_metadata}

        # Save to storage
        file_path = self.storage.save_asset(
            asset_id=asset_id,
            file_data=file_data,
            filename=filename,
            asset_type=asset_type,
            metadata=full_metadata
        )

        # Build ContentAsset
        asset = ContentAsset(
            asset_id=asset_id,
            asset_type=asset_type,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(file_data),
            file_path=file_path,
            checksum=hashlib.sha256(file_data).hexdigest(),
            dimensions=full_metadata.get("dimensions"),
            duration_seconds=full_metadata.get("duration_seconds"),
            metadata=full_metadata
        )

        logger.info(f"Uploaded asset {asset_id}: {filename} ({asset_type.value})")
        return asset

    def upload_from_url(
        self,
        url: str,
        asset_type: Optional[AssetType] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContentAsset:
        """
        Upload asset from URL.

        Args:
            url: URL to download from
            asset_type: Type of asset
            metadata: Additional metadata

        Returns:
            ContentAsset object
        """
        try:
            import urllib.request
            with urllib.request.urlopen(url) as response:
                file_data = response.read()
                filename = url.split("/")[-1].split("?")[0] or "downloaded_file"

                # Add URL to metadata
                full_metadata = {**(metadata or {}), "source_url": url}

                return self.upload_from_bytes(
                    file_data,
                    filename,
                    asset_type=asset_type,
                    metadata=full_metadata
                )
        except Exception as e:
            logger.error(f"Failed to download asset from {url}: {e}")
            raise

    def get_asset_data(self, asset_id: str) -> Optional[bytes]:
        """
        Get raw asset data.

        Args:
            asset_id: Asset identifier

        Returns:
            Binary file data if found
        """
        result = self.storage.get_asset(asset_id)
        if result:
            return result[0]
        return None

    def get_asset_metadata(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get asset metadata.

        Args:
            asset_id: Asset identifier

        Returns:
            Metadata dict if found
        """
        return self.storage.get_metadata(asset_id)

    def delete_asset(self, asset_id: str) -> bool:
        """
        Delete an asset.

        Args:
            asset_id: Asset identifier

        Returns:
            True if deleted
        """
        return self.storage.delete_asset(asset_id)

    def validate_asset(self, asset_id: str) -> Dict[str, Any]:
        """
        Validate an existing asset (checksum, file integrity).

        Args:
            asset_id: Asset identifier

        Returns:
            Validation result dict
        """
        result = {
            "valid": False,
            "exists": False,
            "checksum_match": False,
            "errors": []
        }

        asset_data = self.storage.get_asset(asset_id)
        if not asset_data:
            result["errors"].append("Asset not found")
            return result

        file_data, metadata = asset_data
        result["exists"] = True

        # Verify checksum
        stored_checksum = metadata.get("checksum")
        if stored_checksum:
            current_checksum = hashlib.sha256(file_data).hexdigest()
            result["checksum_match"] = stored_checksum == current_checksum
            if not result["checksum_match"]:
                result["errors"].append("Checksum mismatch - file may be corrupted")

        # Verify size
        stored_size = metadata.get("size_bytes")
        if stored_size and stored_size != len(file_data):
            result["errors"].append("Size mismatch")

        result["valid"] = len(result["errors"]) == 0
        return result

    def generate_thumbnail(
        self,
        asset_id: str,
        size: Tuple[int, int] = (200, 200)
    ) -> Optional[ContentAsset]:
        """
        Generate thumbnail for image/video asset.

        Args:
            asset_id: Source asset ID
            size: Thumbnail dimensions (width, height)

        Returns:
            ContentAsset for thumbnail if successful
        """
        if not self._pil_available:
            logger.warning("PIL not available, cannot generate thumbnail")
            return None

        from PIL import Image
        import io

        asset_data = self.storage.get_asset(asset_id)
        if not asset_data:
            return None

        file_data, metadata = asset_data
        asset_type = AssetType(metadata.get("asset_type", "image"))

        if asset_type != AssetType.IMAGE:
            logger.warning(f"Thumbnail generation only supports images, got {asset_type}")
            return None

        try:
            # Open image
            img = Image.open(io.BytesIO(file_data))

            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Generate thumbnail
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Save to bytes
            thumb_buffer = io.BytesIO()
            img.save(thumb_buffer, format='JPEG', quality=85)
            thumb_data = thumb_buffer.getvalue()

            # Upload as new asset
            thumb_filename = f"thumbnail_{asset_id}.jpg"
            return self.upload_from_bytes(
                thumb_data,
                thumb_filename,
                asset_type=AssetType.THUMBNAIL,
                metadata={
                    "source_asset_id": asset_id,
                    "thumbnail_size": size
                }
            )

        except Exception as e:
            logger.error(f"Failed to generate thumbnail: {e}")
            return None

    def _detect_asset_type(self, mime_type: str, filename: str) -> AssetType:
        """Auto-detect asset type from MIME type and filename."""
        for asset_type, supported_mimes in self.SUPPORTED_TYPES.items():
            if mime_type in supported_mimes:
                return asset_type

        # Fallback based on extension
        ext = Path(filename).suffix.lower()
        ext_mapping = {
            '.py': AssetType.CODE,
            '.js': AssetType.CODE,
            '.ts': AssetType.CODE,
            '.java': AssetType.CODE,
            '.cpp': AssetType.CODE,
            '.c': AssetType.CODE,
            '.go': AssetType.CODE,
            '.rs': AssetType.CODE,
            '.rb': AssetType.CODE,
            '.jpg': AssetType.IMAGE,
            '.jpeg': AssetType.IMAGE,
            '.png': AssetType.IMAGE,
            '.gif': AssetType.IMAGE,
            '.mp4': AssetType.VIDEO,
            '.webm': AssetType.VIDEO,
            '.mp3': AssetType.AUDIO,
            '.wav': AssetType.AUDIO,
            '.pdf': AssetType.DOCUMENT,
            '.doc': AssetType.DOCUMENT,
            '.docx': AssetType.DOCUMENT,
        }

        return ext_mapping.get(ext, AssetType.ATTACHMENT)

    def _validate_asset(self, file_data: bytes, mime_type: str, asset_type: AssetType):
        """Validate asset before upload."""
        # Check size
        max_size = self.MAX_SIZES.get(asset_type, 50 * 1024 * 1024)
        if len(file_data) > max_size:
            raise ValueError(
                f"File too large: {len(file_data)} bytes "
                f"(max {max_size} bytes for {asset_type.value})"
            )

        # Check if empty
        if len(file_data) == 0:
            raise ValueError("Empty file")

    def _extract_metadata(
        self,
        file_data: bytes,
        mime_type: str,
        asset_type: AssetType
    ) -> Dict[str, Any]:
        """Extract metadata from file data."""
        metadata = {}

        if asset_type == AssetType.IMAGE and self._pil_available:
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(file_data))
                metadata["dimensions"] = img.size
                metadata["mode"] = img.mode
                metadata["format"] = img.format
            except Exception as e:
                logger.debug(f"Could not extract image metadata: {e}")

        # Could add video/audio metadata extraction here with appropriate libraries

        return metadata


# =============================================================================
# Factory Function
# =============================================================================

def create_asset_manager(
    storage_type: str = "memory",
    base_path: Optional[str] = None
) -> AssetManager:
    """
    Factory function to create asset manager.

    Args:
        storage_type: Type of storage ("memory", "file")
        base_path: Base path for file storage

    Returns:
        AssetManager instance
    """
    if storage_type == "memory":
        storage = InMemoryAssetStorage()
    elif storage_type == "file":
        if not base_path:
            raise ValueError("base_path required for file storage")
        storage = FileAssetStorage(base_path)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")

    return AssetManager(storage)
