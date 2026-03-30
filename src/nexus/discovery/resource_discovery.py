"""
Unified Resource Discovery - Core system for discovering and managing resources.

This system enables Nexus to:
1. Automatically discover new AI models from multiple sources
2. Find and utilize datasets from HuggingFace and GitHub
3. Discover tools, libraries, and code resources
4. Self-register discovered resources for future use
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Types of discoverable resources."""
    MODEL = "model"
    DATASET = "dataset"
    TOOL = "tool"
    LIBRARY = "library"
    CODE_REPO = "code_repo"
    API = "api"
    SPACE = "space"  # HuggingFace Spaces


class ResourceSource(str, Enum):
    """Sources for resource discovery."""
    OPENROUTER = "openrouter"
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    PYPI = "pypi"
    MANUAL = "manual"
    ZUULTIMATE = "zuultimate"  # Identity/access/vault services
    VINZY = "vinzy"  # License management
    SEC_EDGAR = "sec_edgar"  # SEC EDGAR filings
    WIKIPEDIA = "wikipedia"  # Wikipedia knowledge articles
    TRENDSCOPE = "trendscope"  # Market trend intelligence
    KNOWLEDGE_HARVESTER = "knowledge_harvester"  # Knowledge artifact library
    CONTENT_PRODUCTION = "content_production"  # Content production pipeline
    USPTO = "uspto"  # USPTO patent data


@dataclass
class DiscoveredResource:
    """A discovered resource that can be registered and used."""

    id: str
    name: str
    resource_type: ResourceType
    source: ResourceSource

    # Metadata
    description: Optional[str] = None
    url: Optional[str] = None
    documentation_url: Optional[str] = None

    # Capabilities and tags
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)

    # For models
    provider: Optional[str] = None
    context_length: Optional[int] = None
    pricing_input: Optional[float] = None
    pricing_output: Optional[float] = None

    # For datasets
    size: Optional[str] = None
    format: Optional[str] = None
    license: Optional[str] = None

    # Status
    is_available: bool = True
    is_verified: bool = False
    quality_score: float = 0.0

    # Timestamps
    discovered_at: datetime = field(default_factory=datetime.now)
    last_verified: Optional[datetime] = None

    # Raw metadata from source
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "resource_type": self.resource_type.value,
            "source": self.source.value,
            "description": self.description,
            "url": self.url,
            "documentation_url": self.documentation_url,
            "capabilities": self.capabilities,
            "tags": self.tags,
            "use_cases": self.use_cases,
            "provider": self.provider,
            "context_length": self.context_length,
            "pricing_input": self.pricing_input,
            "pricing_output": self.pricing_output,
            "size": self.size,
            "format": self.format,
            "license": self.license,
            "is_available": self.is_available,
            "is_verified": self.is_verified,
            "quality_score": self.quality_score,
            "discovered_at": self.discovered_at.isoformat(),
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveredResource":
        """Create from dictionary."""
        data = data.copy()
        data["resource_type"] = ResourceType(data["resource_type"])
        data["source"] = ResourceSource(data["source"])
        data["discovered_at"] = datetime.fromisoformat(data["discovered_at"])
        if data.get("last_verified"):
            data["last_verified"] = datetime.fromisoformat(data["last_verified"])
        data.pop("raw_metadata", None)
        return cls(**data)


class ResourceDiscovery:
    """
    Unified resource discovery system.

    This is the main orchestrator that:
    1. Coordinates discovery from multiple sources
    2. Maintains a persistent registry of discovered resources
    3. Enables the system to self-register new resources
    4. Provides search and filtering capabilities
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_persist: bool = True,
    ):
        """
        Initialize resource discovery.

        Args:
            storage_path: Path to persist discovered resources
            auto_persist: Automatically save discoveries to disk
        """
        self.storage_path = storage_path or self._default_storage_path()
        self.auto_persist = auto_persist

        # Resource registries by type
        self._resources: Dict[ResourceType, Dict[str, DiscoveredResource]] = {
            rt: {} for rt in ResourceType
        }

        # Discovery sources (will be set by integrations)
        self._sources: Dict[ResourceSource, Any] = {}

        # Discovery history
        self._discovery_log: List[Dict[str, Any]] = []

        # Load persisted resources
        self._load_persisted_resources()

        logger.info(f"ResourceDiscovery initialized with {self.total_resources} resources")

    def _default_storage_path(self) -> str:
        """Get default storage path."""
        default_data = os.path.join(str(Path.home()), ".nexus", "data")
        base = os.environ.get("NEXUS_DATA_DIR", default_data)
        return os.path.join(base, "discovered_resources.json")

    def _load_persisted_resources(self) -> None:
        """Load previously discovered resources from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)

            for resource_data in data.get("resources", []):
                resource = DiscoveredResource.from_dict(resource_data)
                self._resources[resource.resource_type][resource.id] = resource

            logger.info(f"Loaded {self.total_resources} persisted resources")
        except Exception as e:
            logger.warning(f"Failed to load persisted resources: {e}")

    def _persist_resources(self) -> None:
        """Save discovered resources to disk with secure file permissions."""
        if not self.auto_persist:
            return

        try:
            storage_dir = os.path.dirname(self.storage_path)
            os.makedirs(storage_dir, exist_ok=True)

            all_resources = []
            for resources in self._resources.values():
                all_resources.extend([r.to_dict() for r in resources.values()])

            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "total_count": len(all_resources),
                "resources": all_resources,
            }

            # Write to temp file first, then atomic rename for safety
            import tempfile
            temp_fd, temp_path = tempfile.mkstemp(
                dir=storage_dir,
                prefix=".resources_",
                suffix=".tmp"
            )

            try:
                # Write with restricted permissions (owner read/write only)
                with os.fdopen(temp_fd, "w") as f:
                    json.dump(data, f, indent=2)

                # Set secure permissions before moving (Unix only)
                if hasattr(os, 'chmod'):
                    try:
                        os.chmod(temp_path, 0o600)  # Owner read/write only
                    except OSError:
                        pass  # Windows doesn't support chmod the same way

                # Atomic rename
                os.replace(temp_path, self.storage_path)

            except Exception:
                # Clean up temp file on error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

            logger.debug(f"Persisted {len(all_resources)} resources")
        except Exception as e:
            logger.error(f"Failed to persist resources: {e}")

    @property
    def total_resources(self) -> int:
        """Get total number of discovered resources."""
        return sum(len(r) for r in self._resources.values())

    def register_resource(self, resource: DiscoveredResource) -> bool:
        """
        Register a discovered resource.

        This is the core method for self-registration - the system
        can call this to add new resources it discovers.

        Args:
            resource: The resource to register

        Returns:
            True if newly registered, False if already exists
        """
        registry = self._resources[resource.resource_type]

        if resource.id in registry:
            # Update existing resource
            existing = registry[resource.id]
            existing.last_verified = datetime.now()
            existing.is_available = resource.is_available
            if resource.quality_score > existing.quality_score:
                existing.quality_score = resource.quality_score
            logger.debug(f"Updated existing resource: {resource.id}")
            return False

        # New resource
        registry[resource.id] = resource

        # Log discovery
        self._discovery_log.append({
            "resource_id": resource.id,
            "resource_type": resource.resource_type.value,
            "source": resource.source.value,
            "discovered_at": datetime.now().isoformat(),
        })

        logger.info(f"Registered new {resource.resource_type.value}: {resource.name}")

        # Persist if enabled
        self._persist_resources()

        return True

    def get_resource(self, resource_id: str, resource_type: Optional[ResourceType] = None) -> Optional[DiscoveredResource]:
        """Get a specific resource by ID."""
        if resource_type:
            return self._resources[resource_type].get(resource_id)

        # Search all types
        for registry in self._resources.values():
            if resource_id in registry:
                return registry[resource_id]
        return None

    def search_resources(
        self,
        query: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
        source: Optional[ResourceSource] = None,
        capabilities: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        limit: int = 50,
    ) -> List[DiscoveredResource]:
        """
        Search for resources matching criteria.

        Args:
            query: Text search in name/description
            resource_type: Filter by type
            source: Filter by source
            capabilities: Filter by required capabilities
            tags: Filter by tags
            min_quality: Minimum quality score
            limit: Maximum results

        Returns:
            List of matching resources
        """
        results = []

        # Get candidate resources
        if resource_type:
            candidates = list(self._resources[resource_type].values())
        else:
            candidates = []
            for registry in self._resources.values():
                candidates.extend(registry.values())

        for resource in candidates:
            # Apply filters
            if source and resource.source != source:
                continue

            if resource.quality_score < min_quality:
                continue

            if capabilities:
                if not all(c in resource.capabilities for c in capabilities):
                    continue

            if tags:
                if not any(t in resource.tags for t in tags):
                    continue

            if query:
                query_lower = query.lower()
                if not (
                    query_lower in resource.name.lower() or
                    (resource.description and query_lower in resource.description.lower())
                ):
                    continue

            results.append(resource)

            if len(results) >= limit:
                break

        # Sort by quality score
        results.sort(key=lambda r: r.quality_score, reverse=True)

        return results

    def get_models(self, **kwargs) -> List[DiscoveredResource]:
        """Get discovered models."""
        return self.search_resources(resource_type=ResourceType.MODEL, **kwargs)

    def get_datasets(self, **kwargs) -> List[DiscoveredResource]:
        """Get discovered datasets."""
        return self.search_resources(resource_type=ResourceType.DATASET, **kwargs)

    def get_tools(self, **kwargs) -> List[DiscoveredResource]:
        """Get discovered tools."""
        return self.search_resources(resource_type=ResourceType.TOOL, **kwargs)

    async def discover_all(self) -> Dict[str, int]:
        """
        Run discovery from all registered sources.

        Returns:
            Dictionary of source -> count of new discoveries
        """
        results = {}

        for source_name, source in self._sources.items():
            try:
                if hasattr(source, "discover"):
                    count = await source.discover()
                    results[source_name.value] = count
                    logger.info(f"Discovered {count} resources from {source_name.value}")
            except Exception as e:
                logger.error(f"Discovery failed for {source_name.value}: {e}")
                results[source_name.value] = 0

        return results

    def register_source(self, source_type: ResourceSource, source: Any) -> None:
        """Register a discovery source."""
        self._sources[source_type] = source
        logger.info(f"Registered discovery source: {source_type.value}")

    def get_stats(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        stats = {
            "total_resources": self.total_resources,
            "by_type": {},
            "by_source": {},
            "recent_discoveries": len([
                log for log in self._discovery_log
                if datetime.fromisoformat(log["discovered_at"]) >
                   datetime.now().replace(hour=0, minute=0, second=0)
            ]),
        }

        for rt, registry in self._resources.items():
            stats["by_type"][rt.value] = len(registry)

        for registry in self._resources.values():
            for resource in registry.values():
                source = resource.source.value
                stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

        return stats
