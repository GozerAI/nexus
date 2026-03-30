"""
Content Library - Main Orchestrator.

Comprehensive content management system that integrates:
- Storage backends (in-memory, file, hybrid)
- Asset management (multimedia files)
- AI content generation
- Content templates and builders
- Analytics and quality tracking
- Content relationship graphs

Provides a unified API for all content operations.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .models import (
    ContentItem,
    ContentFormat,
    ContentType,
    ContentStatus,
    ContentSourceType,
    ContentQualityMetrics,
    ContentAsset,
    ContentVersion,
    ContentFilters,
    ContentInteraction,
    InteractionType,
    DifficultyLevel,
    LearningStyle,
)
from .storage import (
    ContentStorageBackend,
    InMemoryStorage,
    FileStorage,
    HybridStorage,
    create_storage,
)
from .assets import (
    AssetManager,
    AssetStorageBackend,
    FileAssetStorage,
    InMemoryAssetStorage,
    create_asset_manager,
)
from .generator import (
    ContentGenerator,
    ContentGenerationConfig,
    GenerationResult,
    create_content_generator,
)
from .templates import (
    ContentTemplate,
    ContentTemplateLibrary,
    ContentBuilder,
)
from .analytics import (
    ContentAnalytics,
    ContentQualityManager,
    ContentAnalyticsReport,
)
from .graph import (
    ContentGraph,
    RelationshipType,
    ContentRelationship,
    LearningPath,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Import/Export Results
# =============================================================================

@dataclass
class ImportResult:
    """Result of a content import operation."""
    success: bool
    imported_count: int
    failed_count: int
    imported_ids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ExportResult:
    """Result of a content export operation."""
    success: bool
    exported_count: int
    file_path: str
    format: str
    errors: List[str] = field(default_factory=list)


@dataclass
class VersionComparison:
    """Comparison between two content versions."""
    content_id: str
    version1: int
    version2: int
    changes: Dict[str, Any]
    fields_changed: List[str]
    body_diff: Optional[str] = None


# =============================================================================
# Content Library Configuration
# =============================================================================

@dataclass
class ContentLibraryConfig:
    """Configuration for Content Library."""
    # Storage settings
    storage_type: str = "memory"  # memory, file, hybrid
    storage_path: Optional[str] = None
    cache_size: int = 1000

    # Asset settings
    asset_storage_type: str = "memory"  # memory, file
    asset_path: Optional[str] = None

    # Generator settings
    enable_ai_generation: bool = True
    llm_provider: Optional[Callable] = None

    # Quality settings
    auto_update_quality: bool = True
    low_quality_threshold: float = 0.3

    # Version settings
    max_versions_per_content: int = 10

    # Integration settings
    knowledge_base: Any = None
    kag_engine: Any = None


# =============================================================================
# Content Library
# =============================================================================

class ContentLibrary:
    """
    Comprehensive content management system.

    Features:
    - CRUD operations for content
    - Versioning and history
    - Search and filtering
    - Quality tracking
    - Import/export
    - AI content generation
    - Asset management
    - Content relationships
    - Integration with KB and KAG

    Usage:
        # Basic usage
        library = ContentLibrary()

        # Create content
        content = ContentBuilder() \\
            .with_title("Python Basics") \\
            .with_body("Content here...") \\
            .with_topics(["python"]) \\
            .build()
        library.create_content(content)

        # Search content
        results = library.search_content("python", ContentFilters(
            difficulty=[DifficultyLevel.BEGINNER]
        ))

        # Generate AI content
        result = await library.generate_content(
            content_type=ContentType.CONCEPT,
            topic="Machine Learning",
            difficulty=DifficultyLevel.INTERMEDIATE
        )
    """

    def __init__(
        self,
        config: Optional[ContentLibraryConfig] = None,
        knowledge_base=None,
        kag_engine=None,
        storage_backend: Optional[ContentStorageBackend] = None
    ):
        """
        Initialize Content Library.

        Args:
            config: Library configuration
            knowledge_base: Optional knowledge base for grounded generation
            kag_engine: Optional KAG engine for verification
            storage_backend: Optional pre-configured storage backend
        """
        self.config = config or ContentLibraryConfig()
        self.knowledge_base = knowledge_base or self.config.knowledge_base
        self.kag_engine = kag_engine or self.config.kag_engine

        # Initialize storage
        self.storage = storage_backend or self._create_storage()

        # Initialize asset manager
        self.asset_manager = self._create_asset_manager()

        # Initialize generator
        self.generator = self._create_generator()

        # Initialize analytics
        self.analytics = ContentAnalytics(
            storage_backend=self.storage
        )

        # Initialize quality manager
        self.quality_manager = ContentQualityManager(
            storage_backend=self.storage,
            analytics=self.analytics,
            auto_update=self.config.auto_update_quality
        )

        # Initialize content graph
        self.graph = ContentGraph(
            storage_backend=self.storage
        )

        # Template library
        self.templates = ContentTemplateLibrary()

        logger.info("ContentLibrary initialized")

    def _create_storage(self) -> ContentStorageBackend:
        """Create storage backend based on configuration."""
        return create_storage(
            storage_type=self.config.storage_type,
            base_path=self.config.storage_path,
            cache_size=self.config.cache_size
        )

    def _create_asset_manager(self) -> AssetManager:
        """Create asset manager based on configuration."""
        return create_asset_manager(
            storage_type=self.config.asset_storage_type,
            base_path=self.config.asset_path
        )

    def _create_generator(self) -> ContentGenerator:
        """Create content generator."""
        return create_content_generator(
            llm_provider=self.config.llm_provider,
            knowledge_base=self.knowledge_base,
            kag_engine=self.kag_engine
        )

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def create_content(self, content: ContentItem) -> ContentItem:
        """
        Create new content item.

        Args:
            content: Content item to create

        Returns:
            Created content with generated ID if needed
        """
        # Ensure content has an ID
        if not content.content_id:
            content.content_id = str(uuid.uuid4())

        # Set timestamps
        now = datetime.now(timezone.utc)
        content.created_at = now
        content.updated_at = now

        # Set initial version
        content.version = 1

        # Save to storage
        self.storage.save(content)

        # Create initial version record
        version = ContentVersion(
            version_id=str(uuid.uuid4()),
            content_id=content.content_id,
            version_number=1,
            content_snapshot=content.to_dict() if hasattr(content, 'to_dict') else {},
            change_summary="Initial creation",
            created_by="system",
            created_at=now
        )
        self.storage.save_version(version)

        logger.info(f"Created content: {content.content_id}")
        return content

    def get_content(
        self,
        content_id: str,
        version: Optional[int] = None
    ) -> Optional[ContentItem]:
        """
        Get content by ID.

        Args:
            content_id: Content ID
            version: Optional specific version (default: latest)

        Returns:
            ContentItem or None if not found
        """
        if version is not None:
            # Get specific version
            versions = self.storage.get_versions(content_id)
            for v in versions:
                if v.version_number == version:
                    # Reconstruct from snapshot
                    return self._reconstruct_content(v.content_snapshot)
            return None

        return self.storage.get(content_id)

    def update_content(
        self,
        content_id: str,
        updates: Dict[str, Any],
        change_summary: str = "Updated content",
        create_version: bool = True
    ) -> Optional[ContentItem]:
        """
        Update content item.

        Args:
            content_id: Content ID
            updates: Dictionary of fields to update
            change_summary: Summary of changes
            create_version: Whether to create new version

        Returns:
            Updated ContentItem or None if not found
        """
        content = self.storage.get(content_id)
        if not content:
            logger.warning(f"Content not found: {content_id}")
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(content, key):
                setattr(content, key, value)

        # Update timestamps
        now = datetime.now(timezone.utc)
        content.updated_at = now

        # Create new version if requested
        if create_version:
            content.version += 1
            version = ContentVersion(
                version_id=str(uuid.uuid4()),
                content_id=content_id,
                version_number=content.version,
                content_snapshot=content.to_dict() if hasattr(content, 'to_dict') else {},
                change_summary=change_summary,
                created_by="system",
                created_at=now
            )
            self.storage.save_version(version)

        # Save updated content
        self.storage.update(content_id, content)

        logger.info(f"Updated content: {content_id} (v{content.version})")
        return content

    def delete_content(
        self,
        content_id: str,
        soft_delete: bool = True
    ) -> bool:
        """
        Delete content item.

        Args:
            content_id: Content ID
            soft_delete: If True, archive instead of delete

        Returns:
            True if successful
        """
        if soft_delete:
            # Archive the content
            content = self.storage.get(content_id)
            if content:
                content.status = ContentStatus.ARCHIVED
                content.updated_at = datetime.now(timezone.utc)
                self.storage.update(content_id, content)
                logger.info(f"Archived content: {content_id}")
                return True
            return False
        else:
            # Hard delete
            success = self.storage.delete(content_id)
            if success:
                logger.info(f"Deleted content: {content_id}")
            return success

    def publish_content(self, content_id: str) -> Optional[ContentItem]:
        """
        Publish content item.

        Args:
            content_id: Content ID

        Returns:
            Published ContentItem
        """
        return self.update_content(
            content_id,
            {
                "status": ContentStatus.PUBLISHED,
                "published_at": datetime.now(timezone.utc)
            },
            change_summary="Published content"
        )

    def archive_content(self, content_id: str) -> Optional[ContentItem]:
        """
        Archive content item.

        Args:
            content_id: Content ID

        Returns:
            Archived ContentItem
        """
        return self.update_content(
            content_id,
            {"status": ContentStatus.ARCHIVED},
            change_summary="Archived content"
        )

    # =========================================================================
    # Version Management
    # =========================================================================

    def get_version_history(self, content_id: str) -> List[ContentVersion]:
        """
        Get version history for content.

        Args:
            content_id: Content ID

        Returns:
            List of versions, newest first
        """
        versions = self.storage.get_versions(content_id)
        return sorted(versions, key=lambda v: v.version_number, reverse=True)

    def rollback_to_version(
        self,
        content_id: str,
        version: int
    ) -> Optional[ContentItem]:
        """
        Rollback content to a specific version.

        Args:
            content_id: Content ID
            version: Version number to rollback to

        Returns:
            Rolled back ContentItem
        """
        versions = self.storage.get_versions(content_id)
        target_version = None

        for v in versions:
            if v.version_number == version:
                target_version = v
                break

        if not target_version:
            logger.warning(f"Version {version} not found for {content_id}")
            return None

        # Reconstruct content from snapshot
        content = self._reconstruct_content(target_version.content_snapshot)
        if not content:
            return None

        # Save as new current version
        current = self.storage.get(content_id)
        new_version = (current.version + 1) if current else version + 1
        content.version = new_version
        content.updated_at = datetime.now(timezone.utc)

        # Create version record
        version_record = ContentVersion(
            version_id=str(uuid.uuid4()),
            content_id=content_id,
            version_number=new_version,
            content_snapshot=content.to_dict() if hasattr(content, 'to_dict') else {},
            change_summary=f"Rolled back to version {version}",
            created_by="system",
            created_at=datetime.now(timezone.utc)
        )

        self.storage.update(content_id, content)
        self.storage.save_version(version_record)

        logger.info(f"Rolled back {content_id} to version {version}")
        return content

    def compare_versions(
        self,
        content_id: str,
        v1: int,
        v2: int
    ) -> Optional[VersionComparison]:
        """
        Compare two versions of content.

        Args:
            content_id: Content ID
            v1: First version
            v2: Second version

        Returns:
            VersionComparison with differences
        """
        versions = self.storage.get_versions(content_id)

        snapshot1 = None
        snapshot2 = None

        for v in versions:
            if v.version_number == v1:
                snapshot1 = v.content_snapshot
            if v.version_number == v2:
                snapshot2 = v.content_snapshot

        if not snapshot1 or not snapshot2:
            return None

        # Find differences
        changes = {}
        fields_changed = []

        all_keys = set(snapshot1.keys()) | set(snapshot2.keys())
        for key in all_keys:
            val1 = snapshot1.get(key)
            val2 = snapshot2.get(key)
            if val1 != val2:
                changes[key] = {"from": val1, "to": val2}
                fields_changed.append(key)

        return VersionComparison(
            content_id=content_id,
            version1=v1,
            version2=v2,
            changes=changes,
            fields_changed=fields_changed
        )

    def _reconstruct_content(self, snapshot: Dict[str, Any]) -> Optional[ContentItem]:
        """Reconstruct ContentItem from version snapshot."""
        try:
            # Parse enums
            if "content_format" in snapshot:
                snapshot["content_format"] = ContentFormat(snapshot["content_format"])
            if "content_type" in snapshot:
                snapshot["content_type"] = ContentType(snapshot["content_type"])
            if "status" in snapshot:
                snapshot["status"] = ContentStatus(snapshot["status"])
            if "source_type" in snapshot:
                snapshot["source_type"] = ContentSourceType(snapshot["source_type"])
            if "difficulty" in snapshot:
                snapshot["difficulty"] = DifficultyLevel(snapshot["difficulty"])
            if "learning_styles" in snapshot:
                snapshot["learning_styles"] = [
                    LearningStyle(s) for s in snapshot["learning_styles"]
                ]

            return ContentItem(**snapshot)
        except Exception as e:
            logger.error(f"Failed to reconstruct content: {e}")
            return None

    # =========================================================================
    # Search and Discovery
    # =========================================================================

    def search_content(
        self,
        query: str,
        filters: Optional[ContentFilters] = None
    ) -> List[ContentItem]:
        """
        Search content by query and filters.

        Args:
            query: Search query
            filters: Optional content filters

        Returns:
            List of matching content items
        """
        return self.storage.search(query, filters)

    def list_content(
        self,
        filters: Optional[ContentFilters] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ContentItem]:
        """
        List content with optional filters.

        Args:
            filters: Optional content filters
            limit: Maximum items to return
            offset: Offset for pagination

        Returns:
            List of content items
        """
        all_content = self.storage.list(filters)
        return all_content[offset:offset + limit]

    def get_by_topic(
        self,
        topic: str,
        limit: int = 20
    ) -> List[ContentItem]:
        """
        Get content by topic.

        Args:
            topic: Topic to filter by
            limit: Maximum items

        Returns:
            List of matching content
        """
        filters = ContentFilters(topics=[topic])
        return self.list_content(filters, limit=limit)

    def get_by_difficulty(
        self,
        difficulty: DifficultyLevel,
        limit: int = 20
    ) -> List[ContentItem]:
        """
        Get content by difficulty level.

        Args:
            difficulty: Difficulty level
            limit: Maximum items

        Returns:
            List of matching content
        """
        filters = ContentFilters(difficulty=[difficulty])
        return self.list_content(filters, limit=limit)

    def get_by_learning_style(
        self,
        style: LearningStyle,
        limit: int = 20
    ) -> List[ContentItem]:
        """
        Get content by learning style.

        Args:
            style: Learning style
            limit: Maximum items

        Returns:
            List of matching content
        """
        filters = ContentFilters(learning_styles=[style])
        return self.list_content(filters, limit=limit)

    def get_related_content(
        self,
        content_id: str,
        limit: int = 10
    ) -> List[ContentItem]:
        """
        Get content related to a specific item.

        Args:
            content_id: Content ID
            limit: Maximum items

        Returns:
            List of related content
        """
        return self.graph.get_related_content(content_id, limit=limit)

    # =========================================================================
    # Quality Management
    # =========================================================================

    def update_quality_metrics(
        self,
        content_id: str,
        metrics_update: Dict[str, Any]
    ) -> bool:
        """
        Update quality metrics for content.

        Args:
            content_id: Content ID
            metrics_update: Dictionary of metrics to update

        Returns:
            True if successful
        """
        content = self.storage.get(content_id)
        if not content:
            return False

        # Update quality metrics
        for key, value in metrics_update.items():
            if hasattr(content.quality_metrics, key):
                setattr(content.quality_metrics, key, value)

        content.quality_metrics.last_updated = datetime.now(timezone.utc)
        self.storage.update(content_id, content)

        return True

    def record_content_interaction(
        self,
        content_id: str,
        user_id: str,
        interaction_type: InteractionType,
        duration_seconds: int = 0,
        performance_score: Optional[float] = None,
        feedback: Optional[str] = None
    ) -> bool:
        """
        Record an interaction with content.

        Args:
            content_id: Content ID
            user_id: User ID
            interaction_type: Type of interaction
            duration_seconds: Duration of interaction
            performance_score: Optional performance score
            feedback: Optional feedback text

        Returns:
            True if successful
        """
        interaction = ContentInteraction(
            interaction_id=str(uuid.uuid4()),
            content_id=content_id,
            user_id=user_id,
            interaction_type=interaction_type,
            duration_seconds=duration_seconds,
            performance_score=performance_score,
            feedback=feedback,
            timestamp=datetime.now(timezone.utc)
        )

        # Record in analytics
        self.analytics.record_interaction(interaction)

        # Auto-update quality if enabled
        if self.config.auto_update_quality:
            self.quality_manager.update_from_interaction(interaction)

        return True

    def get_top_performing_content(
        self,
        limit: int = 20,
        min_completions: int = 5
    ) -> List[ContentItem]:
        """
        Get top performing content by quality score.

        Args:
            limit: Maximum items
            min_completions: Minimum completions required

        Returns:
            List of top content
        """
        all_content = self.storage.list()

        # Filter by minimum completions
        filtered = [
            c for c in all_content
            if c.quality_metrics.total_completions >= min_completions
        ]

        # Sort by effectiveness
        sorted_content = sorted(
            filtered,
            key=lambda c: c.quality_metrics.effectiveness_score,
            reverse=True
        )

        return sorted_content[:limit]

    def get_content_needing_review(
        self,
        threshold: float = 0.5
    ) -> List[ContentItem]:
        """
        Get content with low quality scores needing review.

        Args:
            threshold: Quality threshold

        Returns:
            List of content needing review
        """
        all_content = self.storage.list()

        return [
            c for c in all_content
            if c.quality_metrics.quality_score < threshold
            or c.quality_metrics.effectiveness_score < threshold
        ]

    # =========================================================================
    # AI Content Generation
    # =========================================================================

    async def generate_content(
        self,
        content_type: ContentType,
        topic: str,
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        config: Optional[ContentGenerationConfig] = None,
        auto_save: bool = True
    ) -> GenerationResult:
        """
        Generate content using AI.

        Args:
            content_type: Type of content to generate
            topic: Topic for content
            difficulty: Difficulty level
            config: Generation configuration
            auto_save: Whether to save generated content

        Returns:
            GenerationResult with generated content
        """
        if content_type == ContentType.CONCEPT:
            result = await self.generator.generate_concept_explanation(
                topic=topic,
                difficulty=difficulty,
                config=config
            )
        elif content_type == ContentType.PROCEDURE:
            result = await self.generator.generate_procedure_guide(
                task=topic,
                difficulty=difficulty,
                config=config
            )
        elif content_type == ContentType.EXERCISE:
            result = await self.generator.generate_code_exercise(
                concept=topic,
                difficulty=difficulty,
                config=config
            )
        elif content_type == ContentType.ASSESSMENT:
            result = await self.generator.generate_quiz(
                topics=[topic],
                difficulty=difficulty,
                config=config
            )
        elif content_type == ContentType.FACT:
            result = await self.generator.generate_flashcards(
                topic=topic,
                difficulty=difficulty,
                config=config
            )
        else:
            # Default to concept explanation
            result = await self.generator.generate_concept_explanation(
                topic=topic,
                difficulty=difficulty,
                config=config
            )

        # Auto-save if requested
        if auto_save and result.success and result.content:
            self.create_content(result.content)

        return result

    async def generate_learning_path(
        self,
        topic: str,
        depth: int = 5,
        config: Optional[ContentGenerationConfig] = None
    ) -> List[GenerationResult]:
        """
        Generate a complete learning path.

        Args:
            topic: Main topic
            depth: Number of content items
            config: Generation configuration

        Returns:
            List of GenerationResults
        """
        results = await self.generator.generate_learning_path_content(
            topic=topic,
            depth=depth,
            config=config
        )

        # Save and link content
        prev_id = None
        for result in results:
            if result.success and result.content:
                if prev_id:
                    result.content.prerequisites = [prev_id]
                self.create_content(result.content)

                # Add to graph
                if prev_id:
                    self.graph.add_prerequisite(
                        result.content.content_id,
                        prev_id
                    )

                prev_id = result.content.content_id

        return results

    async def enhance_content(
        self,
        content_id: str,
        enhancements: List[str]
    ) -> GenerationResult:
        """
        Enhance existing content with AI.

        Args:
            content_id: Content ID to enhance
            enhancements: List of enhancement requests

        Returns:
            GenerationResult with enhanced content
        """
        content = self.storage.get(content_id)
        if not content:
            return GenerationResult(
                success=False,
                content=None,
                error=f"Content not found: {content_id}"
            )

        result = await self.generator.enhance_content(content, enhancements)

        if result.success and result.content:
            self.update_content(
                content_id,
                {"content_body": result.content.content_body},
                change_summary=f"Enhanced: {', '.join(enhancements)}"
            )

        return result

    # =========================================================================
    # Asset Management
    # =========================================================================

    def upload_asset(
        self,
        file_path: str,
        asset_type: str,
        content_id: Optional[str] = None
    ) -> Optional[ContentAsset]:
        """
        Upload an asset file.

        Args:
            file_path: Path to file
            asset_type: Type of asset
            content_id: Optional content to attach to

        Returns:
            ContentAsset or None on failure
        """
        asset = self.asset_manager.upload_asset(file_path, asset_type)

        if asset and content_id:
            content = self.storage.get(content_id)
            if content:
                content.assets.append(asset)
                self.storage.update(content_id, content)

        return asset

    def get_asset(self, asset_id: str) -> Optional[bytes]:
        """
        Get asset data.

        Args:
            asset_id: Asset ID

        Returns:
            Asset bytes or None
        """
        return self.asset_manager.get_asset_data(asset_id)

    def delete_asset(self, asset_id: str) -> bool:
        """
        Delete an asset.

        Args:
            asset_id: Asset ID

        Returns:
            True if successful
        """
        return self.asset_manager.delete_asset(asset_id)

    # =========================================================================
    # Content Graph
    # =========================================================================

    def add_prerequisite(
        self,
        content_id: str,
        prerequisite_id: str
    ) -> bool:
        """
        Add prerequisite relationship.

        Args:
            content_id: Content ID
            prerequisite_id: Prerequisite content ID

        Returns:
            True if successful
        """
        return self.graph.add_prerequisite(content_id, prerequisite_id)

    def add_relationship(
        self,
        content_id: str,
        related_id: str,
        relationship_type: RelationshipType
    ) -> bool:
        """
        Add content relationship.

        Args:
            content_id: Content ID
            related_id: Related content ID
            relationship_type: Type of relationship

        Returns:
            True if successful
        """
        return self.graph.add_relationship(content_id, related_id, relationship_type)

    def get_learning_path(
        self,
        target_content_id: str,
        user_id: Optional[str] = None
    ) -> LearningPath:
        """
        Get learning path to target content.

        Args:
            target_content_id: Target content ID
            user_id: Optional user ID for personalization

        Returns:
            LearningPath with prerequisites
        """
        return self.graph.get_learning_path(target_content_id, user_id)

    def validate_prerequisites(
        self,
        content_id: str,
        user_id: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate user has completed prerequisites.

        Args:
            content_id: Content ID
            user_id: User ID

        Returns:
            Tuple of (valid, missing_prerequisite_ids)
        """
        return self.graph.validate_prerequisites(content_id, user_id)

    # =========================================================================
    # Import/Export
    # =========================================================================

    async def import_from_knowledge_base(
        self,
        topic: str,
        limit: int = 50
    ) -> ImportResult:
        """
        Import content from knowledge base.

        Args:
            topic: Topic to import
            limit: Maximum items

        Returns:
            ImportResult
        """
        if not self.knowledge_base:
            return ImportResult(
                success=False,
                imported_count=0,
                failed_count=0,
                errors=["No knowledge base configured"]
            )

        imported_ids = []
        errors = []

        try:
            # Query knowledge base
            results = self.knowledge_base.query_knowledge(topic, max_results=limit)

            for item in results:
                try:
                    # Convert KB item to content
                    content = ContentItem(
                        content_id=str(uuid.uuid4()),
                        title=str(item.content)[:100],
                        description=f"Imported from knowledge base: {topic}",
                        content_body=str(item.content),
                        content_format=ContentFormat.TEXT,
                        content_type=ContentType.FACT,
                        difficulty=DifficultyLevel.INTERMEDIATE,
                        topics=[topic],
                        tags=[topic.lower(), "imported", "knowledge-base"],
                        source_type=ContentSourceType.KNOWLEDGE_BASE,
                        source_id=str(item.chunk_id) if hasattr(item, 'chunk_id') else None,
                        status=ContentStatus.DRAFT
                    )

                    self.create_content(content)
                    imported_ids.append(content.content_id)

                except Exception as e:
                    errors.append(str(e))

            return ImportResult(
                success=True,
                imported_count=len(imported_ids),
                failed_count=len(errors),
                imported_ids=imported_ids,
                errors=errors
            )

        except Exception as e:
            return ImportResult(
                success=False,
                imported_count=0,
                failed_count=0,
                errors=[str(e)]
            )

    def import_from_file(
        self,
        file_path: str,
        format: str = "json"
    ) -> ImportResult:
        """
        Import content from file.

        Args:
            file_path: Path to file
            format: File format (json, yaml)

        Returns:
            ImportResult
        """
        import json
        from pathlib import Path

        imported_ids = []
        errors = []

        try:
            path = Path(file_path)
            if not path.exists():
                return ImportResult(
                    success=False,
                    imported_count=0,
                    failed_count=0,
                    errors=[f"File not found: {file_path}"]
                )

            with open(path, 'r', encoding='utf-8') as f:
                if format == "json":
                    data = json.load(f)
                else:
                    return ImportResult(
                        success=False,
                        imported_count=0,
                        failed_count=0,
                        errors=[f"Unsupported format: {format}"]
                    )

            # Handle list or single item
            items = data if isinstance(data, list) else [data]

            for item in items:
                try:
                    content = self._reconstruct_content(item)
                    if content:
                        content.source_type = ContentSourceType.IMPORTED
                        self.create_content(content)
                        imported_ids.append(content.content_id)
                except Exception as e:
                    errors.append(str(e))

            return ImportResult(
                success=True,
                imported_count=len(imported_ids),
                failed_count=len(errors),
                imported_ids=imported_ids,
                errors=errors
            )

        except Exception as e:
            return ImportResult(
                success=False,
                imported_count=0,
                failed_count=0,
                errors=[str(e)]
            )

    def export_to_file(
        self,
        content_ids: List[str],
        file_path: str,
        format: str = "json"
    ) -> ExportResult:
        """
        Export content to file.

        Args:
            content_ids: List of content IDs to export
            file_path: Output file path
            format: File format

        Returns:
            ExportResult
        """
        import json
        from pathlib import Path

        try:
            contents = []
            for content_id in content_ids:
                content = self.storage.get(content_id)
                if content:
                    if hasattr(content, 'to_dict'):
                        contents.append(content.to_dict())
                    else:
                        # Manual serialization
                        contents.append({
                            "content_id": content.content_id,
                            "title": content.title,
                            "description": content.description,
                            "content_body": content.content_body,
                            "content_format": content.content_format.value,
                            "content_type": content.content_type.value,
                            "difficulty": content.difficulty.value,
                            "topics": content.topics,
                            "tags": content.tags,
                            "status": content.status.value,
                        })

            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(contents, f, indent=2, default=str)

            return ExportResult(
                success=True,
                exported_count=len(contents),
                file_path=file_path,
                format=format
            )

        except Exception as e:
            return ExportResult(
                success=False,
                exported_count=0,
                file_path=file_path,
                format=format,
                errors=[str(e)]
            )

    def bulk_import(
        self,
        contents: List[Dict[str, Any]]
    ) -> ImportResult:
        """
        Bulk import content from dictionaries.

        Args:
            contents: List of content dictionaries

        Returns:
            ImportResult
        """
        imported_ids = []
        errors = []

        for item in contents:
            try:
                content = self._reconstruct_content(item)
                if content:
                    self.create_content(content)
                    imported_ids.append(content.content_id)
            except Exception as e:
                errors.append(str(e))

        return ImportResult(
            success=len(errors) == 0,
            imported_count=len(imported_ids),
            failed_count=len(errors),
            imported_ids=imported_ids,
            errors=errors
        )

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_content_analytics(
        self,
        content_id: str
    ) -> Optional[ContentAnalyticsReport]:
        """
        Get analytics report for content.

        Args:
            content_id: Content ID

        Returns:
            ContentAnalyticsReport
        """
        return self.analytics.get_content_analytics(content_id)

    def get_library_statistics(self) -> Dict[str, Any]:
        """Get overall library statistics."""
        all_content = self.storage.list()

        stats = {
            "total_content": len(all_content),
            "by_status": {},
            "by_type": {},
            "by_difficulty": {},
            "avg_quality_score": 0.0,
            "total_views": 0,
            "total_completions": 0,
        }

        quality_scores = []

        for content in all_content:
            # By status
            status = content.status.value
            stats["by_status"][status] = stats["by_status"].get(status, 0) + 1

            # By type
            ctype = content.content_type.value
            stats["by_type"][ctype] = stats["by_type"].get(ctype, 0) + 1

            # By difficulty
            diff = content.difficulty.value
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

            # Quality
            quality_scores.append(content.quality_metrics.quality_score)
            stats["total_views"] += content.quality_metrics.total_views
            stats["total_completions"] += content.quality_metrics.total_completions

        if quality_scores:
            stats["avg_quality_score"] = sum(quality_scores) / len(quality_scores)

        return stats

    # =========================================================================
    # Templates and Builders
    # =========================================================================

    def get_builder(
        self,
        template: Optional[ContentTemplate] = None
    ) -> ContentBuilder:
        """
        Get a content builder.

        Args:
            template: Optional template to use

        Returns:
            ContentBuilder instance
        """
        return ContentBuilder(template)

    def get_template(self, template_id: str) -> Optional[ContentTemplate]:
        """
        Get a content template by ID.

        Args:
            template_id: Template ID

        Returns:
            ContentTemplate or None
        """
        return self.templates.get_template(template_id)

    def list_templates(self) -> List[ContentTemplate]:
        """Get all available templates."""
        return self.templates.list_templates()


# =============================================================================
# Factory Function
# =============================================================================

def create_content_library(
    storage_type: str = "memory",
    storage_path: Optional[str] = None,
    knowledge_base=None,
    kag_engine=None,
    llm_provider: Optional[Callable] = None,
    **kwargs
) -> ContentLibrary:
    """
    Factory function to create Content Library.

    Args:
        storage_type: Storage backend type
        storage_path: Path for file storage
        knowledge_base: Optional knowledge base
        kag_engine: Optional KAG engine
        llm_provider: Optional LLM provider for AI generation
        **kwargs: Additional configuration options

    Returns:
        ContentLibrary instance
    """
    config = ContentLibraryConfig(
        storage_type=storage_type,
        storage_path=storage_path,
        llm_provider=llm_provider,
        knowledge_base=knowledge_base,
        kag_engine=kag_engine,
        **{k: v for k, v in kwargs.items() if hasattr(ContentLibraryConfig, k)}
    )

    return ContentLibrary(
        config=config,
        knowledge_base=knowledge_base,
        kag_engine=kag_engine
    )
