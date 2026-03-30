"""
Content Library Package.

Comprehensive content management system for learning content with:
- Pluggable storage backends (in-memory, file, hybrid)
- Full asset management (multimedia files)
- AI-powered content generation
- Content templates and builders
- Analytics and quality tracking
- Content relationship graphs
- Version control

Usage:
    from nexus.rag.content_library import (
        ContentLibrary,
        ContentItem,
        ContentBuilder,
        create_content_library,
    )

    # Create library
    library = create_content_library(
        storage_type="file",
        storage_path="./content"
    )

    # Create content with builder
    content = ContentBuilder() \\
        .with_title("Python Basics") \\
        .with_body("Variables store data...") \\
        .with_topics(["python", "basics"]) \\
        .with_difficulty(DifficultyLevel.BEGINNER) \\
        .build()

    library.create_content(content)

    # Search content
    results = library.search_content("python")

    # Generate AI content
    result = await library.generate_content(
        content_type=ContentType.CONCEPT,
        topic="Machine Learning"
    )
"""

# Core Models
from .models import (
    # Enums
    ContentFormat,
    ContentStatus,
    ContentSourceType,
    ContentType,
    DifficultyLevel,
    LearningStyle,
    AssetType,
    InteractionType,
    # Data classes
    ContentQualityMetrics,
    ContentAsset,
    ContentVersion,
    ContentItem,
    ContentInteraction,
    ContentFilters,
)

# Storage
from .storage import (
    ContentStorageBackend,
    InMemoryStorage,
    FileStorage,
    HybridStorage,
    create_storage,
)

# Assets
from .assets import (
    AssetStorageBackend,
    FileAssetStorage,
    InMemoryAssetStorage,
    AssetManager,
    create_asset_manager,
)

# Generator
from .generator import (
    ContentGenerationConfig,
    GenerationMode,
    GenerationResult,
    ContentGenerator,
    PromptTemplates,
    create_content_generator,
)

# Templates
from .templates import (
    ContentTemplate,
    ContentTemplateLibrary,
    ContentBuilder,
    quick_concept,
    quick_exercise,
    quick_quiz,
)

# Analytics
from .analytics import (
    ContentAnalytics,
    ContentQualityManager,
    ContentAnalyticsReport,
    TrendingContentReport,
    UnderperformingContentReport,
)

# Graph
from .graph import (
    RelationshipType,
    ContentRelationship,
    LearningPath,
    ContentGraph,
)

# Main Library
from .library import (
    ContentLibrary,
    ContentLibraryConfig,
    ImportResult,
    ExportResult,
    VersionComparison,
    create_content_library,
)


__all__ = [
    # Enums
    "ContentFormat",
    "ContentStatus",
    "ContentSourceType",
    "ContentType",
    "DifficultyLevel",
    "LearningStyle",
    "AssetType",
    "InteractionType",
    "RelationshipType",
    "GenerationMode",

    # Core Models
    "ContentQualityMetrics",
    "ContentAsset",
    "ContentVersion",
    "ContentItem",
    "ContentInteraction",
    "ContentFilters",

    # Storage
    "ContentStorageBackend",
    "InMemoryStorage",
    "FileStorage",
    "HybridStorage",
    "create_storage",

    # Assets
    "AssetStorageBackend",
    "FileAssetStorage",
    "InMemoryAssetStorage",
    "AssetManager",
    "create_asset_manager",

    # Generator
    "ContentGenerationConfig",
    "GenerationResult",
    "ContentGenerator",
    "PromptTemplates",
    "create_content_generator",

    # Templates
    "ContentTemplate",
    "ContentTemplateLibrary",
    "ContentBuilder",
    "quick_concept",
    "quick_exercise",
    "quick_quiz",

    # Analytics
    "ContentAnalytics",
    "ContentQualityManager",
    "ContentAnalyticsReport",
    "TrendingContentReport",
    "UnderperformingContentReport",

    # Graph
    "ContentRelationship",
    "LearningPath",
    "ContentGraph",

    # Library
    "ContentLibrary",
    "ContentLibraryConfig",
    "ImportResult",
    "ExportResult",
    "VersionComparison",
    "create_content_library",
]
