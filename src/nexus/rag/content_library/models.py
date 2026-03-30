"""
Content Library Data Models for Nexus AI Platform.

Defines all data structures for the content management system including:
- ContentItem: Core content entity with full metadata
- ContentAsset: Multimedia asset representation
- ContentVersion: Version control for content
- ContentQualityMetrics: Dynamic quality tracking
- Various enums for content classification
"""

import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum

# Re-export existing enums from adaptive_pathways for compatibility
try:
    from nexus.rag.adaptive_pathways import (
        LearningStyle,
        DifficultyLevel,
        ContentType,
        MasteryLevel,
    )
except ImportError:
    # Fallback definitions if import fails
    class LearningStyle(Enum):
        VISUAL = "visual"
        AUDITORY = "auditory"
        KINESTHETIC = "kinesthetic"
        READING = "reading"
        MULTIMODAL = "multimodal"

    class DifficultyLevel(Enum):
        NOVICE = "novice"
        BEGINNER = "beginner"
        INTERMEDIATE = "intermediate"
        ADVANCED = "advanced"
        EXPERT = "expert"

        @property
        def numeric(self) -> int:
            return {"novice": 1, "beginner": 2, "intermediate": 3,
                    "advanced": 4, "expert": 5}[self.value]

    class ContentType(Enum):
        CONCEPT = "concept"
        PROCEDURE = "procedure"
        FACT = "fact"
        PRINCIPLE = "principle"
        EXAMPLE = "example"
        EXERCISE = "exercise"
        ASSESSMENT = "assessment"
        PROJECT = "project"


# =============================================================================
# Content Format and Status Enums
# =============================================================================

class ContentFormat(Enum):
    """Format of content body."""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    VIDEO = "video"
    AUDIO = "audio"
    INTERACTIVE = "interactive"
    QUIZ = "quiz"
    CODE_EXERCISE = "code_exercise"
    SIMULATION = "simulation"
    DIAGRAM = "diagram"
    INFOGRAPHIC = "infographic"
    FLASHCARD = "flashcard"
    PRESENTATION = "presentation"


class ContentStatus(Enum):
    """Lifecycle status of content."""
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ContentSourceType(Enum):
    """Origin/source of content."""
    AUTHORED = "authored"               # Manually created by human
    GENERATED = "generated"             # AI-generated
    IMPORTED = "imported"               # From external source
    KNOWLEDGE_BASE = "knowledge_base"   # Extracted from KB
    USER_CONTRIBUTED = "user_contributed"  # User submissions
    CURATED = "curated"                 # Curated from multiple sources


class AssetType(Enum):
    """Type of multimedia asset."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    CODE = "code"
    DIAGRAM = "diagram"
    ANIMATION = "animation"
    THUMBNAIL = "thumbnail"
    ATTACHMENT = "attachment"


class InteractionType(Enum):
    """Type of user interaction with content."""
    VIEW = "view"
    COMPLETE = "complete"
    SKIP = "skip"
    BOOKMARK = "bookmark"
    SHARE = "share"
    RATE = "rate"
    FEEDBACK = "feedback"
    DOWNLOAD = "download"


# =============================================================================
# Quality Metrics
# =============================================================================

@dataclass
class ContentQualityMetrics:
    """
    Dynamic quality metrics for content, updated based on user interactions.

    All scores are 0.0-1.0 scale unless otherwise noted.
    """
    # Core quality scores
    quality_score: float = 0.8
    engagement_score: float = 0.7
    effectiveness_score: float = 0.75

    # Usage metrics
    completion_rate: float = 0.0
    avg_time_spent_seconds: float = 0.0
    avg_performance_score: float = 0.0

    # Counts
    total_views: int = 0
    total_completions: int = 0
    total_skips: int = 0
    unique_users: int = 0

    # Feedback
    positive_feedback: int = 0
    negative_feedback: int = 0
    avg_rating: float = 0.0
    total_ratings: int = 0

    # Verification scores (from KAG)
    verification_score: float = 0.0
    coherence_score: float = 0.0
    accuracy_score: float = 0.0
    verified: bool = False

    # Timestamps
    last_viewed: Optional[datetime] = None
    last_completed: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    def update_from_view(self, duration_seconds: float):
        """Update metrics after content is viewed."""
        self.total_views += 1
        # Running average
        self.avg_time_spent_seconds = (
            (self.avg_time_spent_seconds * (self.total_views - 1) + duration_seconds)
            / self.total_views
        )
        self.last_viewed = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)
        self._recalculate_scores()

    def update_from_completion(self, performance_score: float, duration_seconds: float):
        """Update metrics after content is completed."""
        self.total_completions += 1
        self.avg_performance_score = (
            (self.avg_performance_score * (self.total_completions - 1) + performance_score)
            / self.total_completions
        )
        self.completion_rate = self.total_completions / max(self.total_views, 1)
        self.last_completed = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)
        self._recalculate_scores()

    def update_from_rating(self, rating: float, positive: bool):
        """Update metrics from user rating/feedback."""
        if positive:
            self.positive_feedback += 1
        else:
            self.negative_feedback += 1

        self.total_ratings += 1
        self.avg_rating = (
            (self.avg_rating * (self.total_ratings - 1) + rating)
            / self.total_ratings
        )
        self.last_updated = datetime.now(timezone.utc)
        self._recalculate_scores()

    def update_from_skip(self):
        """Update metrics when content is skipped."""
        self.total_skips += 1
        self.last_updated = datetime.now(timezone.utc)
        self._recalculate_scores()

    def _recalculate_scores(self):
        """Recalculate aggregate quality scores."""
        # Engagement: based on completion and view patterns
        if self.total_views > 0:
            skip_rate = self.total_skips / self.total_views
            self.engagement_score = max(0.0, min(1.0,
                self.completion_rate * 0.6 + (1 - skip_rate) * 0.4
            ))

        # Effectiveness: based on performance and completion
        if self.total_completions > 0:
            self.effectiveness_score = max(0.0, min(1.0,
                self.avg_performance_score * 0.7 + self.completion_rate * 0.3
            ))

        # Overall quality: weighted combination
        feedback_score = 0.5
        if self.total_ratings > 0:
            feedback_score = self.avg_rating / 5.0  # Assume 5-star rating

        self.quality_score = max(0.0, min(1.0,
            self.effectiveness_score * 0.4 +
            self.engagement_score * 0.3 +
            feedback_score * 0.2 +
            (self.verification_score if self.verified else 0.5) * 0.1
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quality_score": self.quality_score,
            "engagement_score": self.engagement_score,
            "effectiveness_score": self.effectiveness_score,
            "completion_rate": self.completion_rate,
            "avg_time_spent_seconds": self.avg_time_spent_seconds,
            "avg_performance_score": self.avg_performance_score,
            "total_views": self.total_views,
            "total_completions": self.total_completions,
            "total_skips": self.total_skips,
            "unique_users": self.unique_users,
            "positive_feedback": self.positive_feedback,
            "negative_feedback": self.negative_feedback,
            "avg_rating": self.avg_rating,
            "total_ratings": self.total_ratings,
            "verification_score": self.verification_score,
            "coherence_score": self.coherence_score,
            "accuracy_score": self.accuracy_score,
            "verified": self.verified,
            "last_viewed": self.last_viewed.isoformat() if self.last_viewed else None,
            "last_completed": self.last_completed.isoformat() if self.last_completed else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentQualityMetrics':
        """Create from dictionary."""
        metrics = cls()
        for key, value in data.items():
            if hasattr(metrics, key):
                if key in ['last_viewed', 'last_completed', 'last_updated'] and value:
                    value = datetime.fromisoformat(value)
                setattr(metrics, key, value)
        return metrics


# =============================================================================
# Content Asset
# =============================================================================

@dataclass
class ContentAsset:
    """
    Multimedia asset attached to content.

    Supports images, videos, audio, documents, code files, etc.
    """
    asset_id: str
    asset_type: AssetType
    filename: str
    mime_type: str
    size_bytes: int

    # Storage location
    file_path: Optional[str] = None
    url: Optional[str] = None

    # Integrity
    checksum: Optional[str] = None
    checksum_algorithm: str = "sha256"

    # Media-specific metadata
    dimensions: Optional[Tuple[int, int]] = None  # (width, height) for images/video
    duration_seconds: Optional[float] = None      # For audio/video
    bitrate: Optional[int] = None                 # For audio/video
    codec: Optional[str] = None                   # For audio/video

    # Derived assets
    thumbnail_id: Optional[str] = None
    preview_url: Optional[str] = None

    # Metadata
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None

    # Extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.asset_id:
            self.asset_id = str(uuid.uuid4())
        if isinstance(self.asset_type, str):
            self.asset_type = AssetType(self.asset_type)

    @staticmethod
    def calculate_checksum(data: bytes, algorithm: str = "sha256") -> str:
        """Calculate checksum of data."""
        if algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "asset_id": self.asset_id,
            "asset_type": self.asset_type.value if isinstance(self.asset_type, AssetType) else self.asset_type,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "file_path": self.file_path,
            "url": self.url,
            "checksum": self.checksum,
            "checksum_algorithm": self.checksum_algorithm,
            "dimensions": self.dimensions,
            "duration_seconds": self.duration_seconds,
            "bitrate": self.bitrate,
            "codec": self.codec,
            "thumbnail_id": self.thumbnail_id,
            "preview_url": self.preview_url,
            "alt_text": self.alt_text,
            "caption": self.caption,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentAsset':
        """Create from dictionary."""
        data = data.copy()
        if 'asset_type' in data and isinstance(data['asset_type'], str):
            data['asset_type'] = AssetType(data['asset_type'])
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'dimensions' in data and data['dimensions']:
            data['dimensions'] = tuple(data['dimensions'])
        return cls(**data)


# =============================================================================
# Content Version
# =============================================================================

@dataclass
class ContentVersion:
    """
    Version snapshot of content for version control.
    """
    version_id: str
    content_id: str
    version_number: int

    # Snapshot of content at this version
    content_snapshot: Dict[str, Any]

    # Change tracking
    change_summary: str = ""
    changes: List[Dict[str, Any]] = field(default_factory=list)  # List of field changes

    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None

    # Metadata
    is_major_version: bool = False
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.version_id:
            self.version_id = f"v_{self.content_id}_{self.version_number}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "content_id": self.content_id,
            "version_number": self.version_number,
            "content_snapshot": self.content_snapshot,
            "change_summary": self.change_summary,
            "changes": self.changes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by,
            "is_major_version": self.is_major_version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentVersion':
        """Create from dictionary."""
        data = data.copy()
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


# =============================================================================
# Main Content Item
# =============================================================================

@dataclass
class ContentItem:
    """
    Core content entity representing a learning content item.

    This is the primary data structure for all content in the library.
    """
    # Identity
    content_id: str
    version: int = 1

    # Core content
    title: str = ""
    description: str = ""
    content_body: str = ""

    # Classification
    content_format: ContentFormat = ContentFormat.TEXT
    content_type: ContentType = ContentType.CONCEPT
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE

    # Categorization
    topics: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    # Learning attributes
    prerequisites: List[str] = field(default_factory=list)  # Content IDs
    learning_styles: List[LearningStyle] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)

    # Timing
    estimated_time_minutes: int = 15

    # Source tracking
    source_type: ContentSourceType = ContentSourceType.AUTHORED
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    author: Optional[str] = None

    # Quality metrics (dynamic)
    quality_metrics: ContentQualityMetrics = field(default_factory=ContentQualityMetrics)

    # Multimedia assets
    assets: List[ContentAsset] = field(default_factory=list)
    primary_asset_id: Optional[str] = None

    # Relationships
    related_content: List[str] = field(default_factory=list)  # Content IDs
    next_content: Optional[str] = None
    previous_content: Optional[str] = None
    parent_content: Optional[str] = None  # For hierarchical content

    # Status and lifecycle
    status: ContentStatus = ContentStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    published_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None

    # Audit
    created_by: Optional[str] = None
    updated_by: Optional[str] = None

    # Localization
    language: str = "en"
    translations: Dict[str, str] = field(default_factory=dict)  # lang -> content_id

    # Extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and validate content item."""
        if not self.content_id:
            self.content_id = str(uuid.uuid4())

        # Convert enums if needed
        if isinstance(self.content_format, str):
            self.content_format = ContentFormat(self.content_format)
        if isinstance(self.content_type, str):
            self.content_type = ContentType(self.content_type)
        if isinstance(self.difficulty, str):
            self.difficulty = DifficultyLevel(self.difficulty)
        if isinstance(self.source_type, str):
            self.source_type = ContentSourceType(self.source_type)
        if isinstance(self.status, str):
            self.status = ContentStatus(self.status)

        # Convert learning styles
        self.learning_styles = [
            LearningStyle(s) if isinstance(s, str) else s
            for s in self.learning_styles
        ]

        # Convert assets
        self.assets = [
            ContentAsset.from_dict(a) if isinstance(a, dict) else a
            for a in self.assets
        ]

        # Convert quality metrics
        if isinstance(self.quality_metrics, dict):
            self.quality_metrics = ContentQualityMetrics.from_dict(self.quality_metrics)

    def publish(self) -> 'ContentItem':
        """Publish the content."""
        self.status = ContentStatus.PUBLISHED
        self.published_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        return self

    def archive(self) -> 'ContentItem':
        """Archive the content."""
        self.status = ContentStatus.ARCHIVED
        self.archived_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        return self

    def deprecate(self) -> 'ContentItem':
        """Mark content as deprecated."""
        self.status = ContentStatus.DEPRECATED
        self.updated_at = datetime.now(timezone.utc)
        return self

    def add_asset(self, asset: ContentAsset) -> 'ContentItem':
        """Add an asset to the content."""
        self.assets.append(asset)
        if not self.primary_asset_id:
            self.primary_asset_id = asset.asset_id
        self.updated_at = datetime.now(timezone.utc)
        return self

    def get_asset(self, asset_id: str) -> Optional[ContentAsset]:
        """Get asset by ID."""
        for asset in self.assets:
            if asset.asset_id == asset_id:
                return asset
        return None

    def get_primary_asset(self) -> Optional[ContentAsset]:
        """Get the primary asset."""
        if self.primary_asset_id:
            return self.get_asset(self.primary_asset_id)
        return self.assets[0] if self.assets else None

    def create_version_snapshot(self, change_summary: str = "", created_by: str = None) -> ContentVersion:
        """Create a version snapshot of current state."""
        return ContentVersion(
            version_id=f"v_{self.content_id}_{self.version}",
            content_id=self.content_id,
            version_number=self.version,
            content_snapshot=self.to_dict(),
            change_summary=change_summary,
            created_by=created_by or self.updated_by,
        )

    def increment_version(self):
        """Increment version number."""
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "content_id": self.content_id,
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "content_body": self.content_body,
            "content_format": self.content_format.value,
            "content_type": self.content_type.value,
            "difficulty": self.difficulty.value,
            "topics": self.topics,
            "tags": self.tags,
            "categories": self.categories,
            "prerequisites": self.prerequisites,
            "learning_styles": [s.value for s in self.learning_styles],
            "learning_objectives": self.learning_objectives,
            "estimated_time_minutes": self.estimated_time_minutes,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "author": self.author,
            "quality_metrics": self.quality_metrics.to_dict(),
            "assets": [a.to_dict() for a in self.assets],
            "primary_asset_id": self.primary_asset_id,
            "related_content": self.related_content,
            "next_content": self.next_content,
            "previous_content": self.previous_content,
            "parent_content": self.parent_content,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "created_by": self.created_by,
            "updated_by": self.updated_by,
            "language": self.language,
            "translations": self.translations,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentItem':
        """Create ContentItem from dictionary."""
        data = data.copy()

        # Convert datetime strings
        for field in ['created_at', 'updated_at', 'published_at', 'archived_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])

        return cls(**data)


# =============================================================================
# Content Interaction (for analytics)
# =============================================================================

@dataclass
class ContentInteraction:
    """
    Record of user interaction with content.
    Used for analytics and quality metric updates.
    """
    interaction_id: str
    content_id: str
    user_id: str
    interaction_type: InteractionType

    # Interaction details
    duration_seconds: int = 0
    performance_score: Optional[float] = None  # 0-1 scale
    rating: Optional[float] = None             # 0-5 scale
    feedback: Optional[str] = None

    # Context
    session_id: Optional[str] = None
    pathway_id: Optional[str] = None

    # Timestamp
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.interaction_id:
            self.interaction_id = str(uuid.uuid4())
        if isinstance(self.interaction_type, str):
            self.interaction_type = InteractionType(self.interaction_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interaction_id": self.interaction_id,
            "content_id": self.content_id,
            "user_id": self.user_id,
            "interaction_type": self.interaction_type.value,
            "duration_seconds": self.duration_seconds,
            "performance_score": self.performance_score,
            "rating": self.rating,
            "feedback": self.feedback,
            "session_id": self.session_id,
            "pathway_id": self.pathway_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentInteraction':
        """Create from dictionary."""
        data = data.copy()
        if 'interaction_type' in data and isinstance(data['interaction_type'], str):
            data['interaction_type'] = InteractionType(data['interaction_type'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


# =============================================================================
# Content Filters
# =============================================================================

@dataclass
class ContentFilters:
    """
    Filter criteria for content search and listing.
    """
    # Text search
    query: Optional[str] = None

    # Classification filters
    topics: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    difficulty: Optional[List[DifficultyLevel]] = None
    content_types: Optional[List[ContentType]] = None
    formats: Optional[List[ContentFormat]] = None
    learning_styles: Optional[List[LearningStyle]] = None

    # Status filter
    status: Optional[List[ContentStatus]] = None

    # Source filter
    source_types: Optional[List[ContentSourceType]] = None
    authors: Optional[List[str]] = None

    # Quality filters
    min_quality_score: Optional[float] = None
    min_engagement_score: Optional[float] = None
    min_effectiveness_score: Optional[float] = None
    verified_only: bool = False

    # Time filters
    max_estimated_time: Optional[int] = None
    min_estimated_time: Optional[int] = None

    # Date filters
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    updated_after: Optional[datetime] = None
    published_after: Optional[datetime] = None

    # Relationship filters
    has_prerequisites: Optional[bool] = None
    prerequisite_of: Optional[str] = None  # Content ID
    related_to: Optional[str] = None       # Content ID

    # Language filter
    language: Optional[str] = None

    # Pagination
    limit: int = 50
    offset: int = 0

    # Sorting
    sort_by: str = "updated_at"
    sort_order: str = "desc"  # "asc" or "desc"

    def matches(self, content: ContentItem) -> bool:
        """Check if content matches all filters."""
        # Text query
        if self.query:
            query_lower = self.query.lower()
            if not (
                query_lower in content.title.lower() or
                query_lower in content.description.lower() or
                query_lower in content.content_body.lower() or
                any(query_lower in t.lower() for t in content.topics) or
                any(query_lower in t.lower() for t in content.tags)
            ):
                return False

        # Classification filters
        if self.topics and not any(t in content.topics for t in self.topics):
            return False
        if self.tags and not any(t in content.tags for t in self.tags):
            return False
        if self.categories and not any(c in content.categories for c in self.categories):
            return False
        if self.difficulty and content.difficulty not in self.difficulty:
            return False
        if self.content_types and content.content_type not in self.content_types:
            return False
        if self.formats and content.content_format not in self.formats:
            return False
        if self.learning_styles and not any(s in content.learning_styles for s in self.learning_styles):
            return False

        # Status
        if self.status and content.status not in self.status:
            return False

        # Source
        if self.source_types and content.source_type not in self.source_types:
            return False
        if self.authors and content.author not in self.authors:
            return False

        # Quality
        if self.min_quality_score and content.quality_metrics.quality_score < self.min_quality_score:
            return False
        if self.min_engagement_score and content.quality_metrics.engagement_score < self.min_engagement_score:
            return False
        if self.min_effectiveness_score and content.quality_metrics.effectiveness_score < self.min_effectiveness_score:
            return False
        if self.verified_only and not content.quality_metrics.verified:
            return False

        # Time
        if self.max_estimated_time and content.estimated_time_minutes > self.max_estimated_time:
            return False
        if self.min_estimated_time and content.estimated_time_minutes < self.min_estimated_time:
            return False

        # Dates
        if self.created_after and content.created_at < self.created_after:
            return False
        if self.created_before and content.created_at > self.created_before:
            return False
        if self.updated_after and content.updated_at < self.updated_after:
            return False
        if self.published_after and (not content.published_at or content.published_at < self.published_after):
            return False

        # Relationships
        if self.has_prerequisites is not None:
            if self.has_prerequisites and not content.prerequisites:
                return False
            if not self.has_prerequisites and content.prerequisites:
                return False
        if self.prerequisite_of and self.prerequisite_of not in content.prerequisites:
            return False
        if self.related_to and self.related_to not in content.related_content:
            return False

        # Language
        if self.language and content.language != self.language:
            return False

        return True
