"""
Content Templates and Builder for Content Library.

Provides:
- Pre-built templates for common content types
- Fluent ContentBuilder for easy content creation
- Template validation and structure enforcement
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Union
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
    DifficultyLevel,
    LearningStyle,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Content Template
# =============================================================================

@dataclass
class ContentTemplate:
    """
    Template for creating consistent content.

    Defines the structure and defaults for a content type.
    """
    template_id: str
    name: str
    description: str

    # Content type this template is for
    content_type: ContentType
    content_format: ContentFormat

    # Structure definition (JSON schema-like)
    structure: Dict[str, Any] = field(default_factory=dict)

    # Default values
    default_values: Dict[str, Any] = field(default_factory=dict)

    # Required fields
    required_fields: List[str] = field(default_factory=list)

    # Optional AI generation prompt
    ai_prompt_template: Optional[str] = None

    # Template metadata
    tags: List[str] = field(default_factory=list)
    difficulty: Optional[DifficultyLevel] = None
    learning_styles: List[LearningStyle] = field(default_factory=list)
    estimated_time_minutes: Optional[int] = None

    # Audit
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None

    def validate(self, content: Dict[str, Any]) -> List[str]:
        """
        Validate content against template requirements.

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Check required fields
        for field_name in self.required_fields:
            if field_name not in content or not content[field_name]:
                errors.append(f"Missing required field: {field_name}")

        # Validate structure if defined
        if self.structure:
            for field_name, field_def in self.structure.items():
                if field_name in content:
                    value = content[field_name]
                    expected_type = field_def.get("type")

                    if expected_type == "string" and not isinstance(value, str):
                        errors.append(f"Field {field_name} must be a string")
                    elif expected_type == "list" and not isinstance(value, list):
                        errors.append(f"Field {field_name} must be a list")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        errors.append(f"Field {field_name} must be a number")

                    # Check min/max length
                    if isinstance(value, str):
                        min_len = field_def.get("min_length", 0)
                        max_len = field_def.get("max_length", float('inf'))
                        if len(value) < min_len:
                            errors.append(f"Field {field_name} too short (min {min_len})")
                        if len(value) > max_len:
                            errors.append(f"Field {field_name} too long (max {max_len})")

        return errors

    def apply_defaults(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to content."""
        result = {**self.default_values, **content}
        return result


# =============================================================================
# Template Library
# =============================================================================

class ContentTemplateLibrary:
    """
    Pre-built templates for common content types.
    """

    # Concept Explanation Template
    CONCEPT_EXPLANATION = ContentTemplate(
        template_id="concept_explanation",
        name="Concept Explanation",
        description="Template for explaining concepts and ideas",
        content_type=ContentType.CONCEPT,
        content_format=ContentFormat.MARKDOWN,
        structure={
            "title": {"type": "string", "min_length": 5, "max_length": 200},
            "content_body": {"type": "string", "min_length": 100},
            "topics": {"type": "list"},
        },
        required_fields=["title", "content_body", "topics"],
        default_values={
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "estimated_time_minutes": 15,
            "learning_styles": [LearningStyle.READING, LearningStyle.VISUAL],
        },
        ai_prompt_template="Explain {topic} clearly for {difficulty} level learners.",
        tags=["concept", "explanation", "learning"],
    )

    # Step-by-Step Procedure Template
    PROCEDURE_STEPS = ContentTemplate(
        template_id="procedure_steps",
        name="Step-by-Step Procedure",
        description="Template for procedural guides and how-tos",
        content_type=ContentType.PROCEDURE,
        content_format=ContentFormat.MARKDOWN,
        structure={
            "title": {"type": "string", "min_length": 5},
            "content_body": {"type": "string", "min_length": 100},
            "steps": {"type": "list"},
        },
        required_fields=["title", "content_body"],
        default_values={
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "estimated_time_minutes": 20,
            "learning_styles": [LearningStyle.KINESTHETIC],
        },
        ai_prompt_template="Create a step-by-step guide for {task}.",
        tags=["procedure", "how-to", "guide"],
    )

    # Code Exercise Template
    CODE_EXERCISE = ContentTemplate(
        template_id="code_exercise",
        name="Code Exercise",
        description="Template for coding exercises and challenges",
        content_type=ContentType.EXERCISE,
        content_format=ContentFormat.CODE_EXERCISE,
        structure={
            "title": {"type": "string", "min_length": 5},
            "problem_statement": {"type": "string", "min_length": 50},
            "starter_code": {"type": "string"},
            "solution": {"type": "string"},
            "test_cases": {"type": "list"},
        },
        required_fields=["title", "problem_statement"],
        default_values={
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "estimated_time_minutes": 30,
            "learning_styles": [LearningStyle.KINESTHETIC],
        },
        ai_prompt_template="Create a coding exercise for {concept} in {language}.",
        tags=["code", "exercise", "practice"],
    )

    # Multiple Choice Quiz Template
    QUIZ_MULTIPLE_CHOICE = ContentTemplate(
        template_id="quiz_multiple_choice",
        name="Multiple Choice Quiz",
        description="Template for multiple choice assessments",
        content_type=ContentType.ASSESSMENT,
        content_format=ContentFormat.QUIZ,
        structure={
            "title": {"type": "string", "min_length": 5},
            "questions": {"type": "list"},
        },
        required_fields=["title", "questions"],
        default_values={
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "estimated_time_minutes": 15,
        },
        ai_prompt_template="Create a {num_questions} question quiz on {topics}.",
        tags=["quiz", "assessment", "multiple-choice"],
    )

    # Video Lesson Template
    VIDEO_LESSON = ContentTemplate(
        template_id="video_lesson",
        name="Video Lesson",
        description="Template for video-based lessons",
        content_type=ContentType.CONCEPT,
        content_format=ContentFormat.VIDEO,
        structure={
            "title": {"type": "string", "min_length": 5},
            "description": {"type": "string"},
            "video_url": {"type": "string"},
            "transcript": {"type": "string"},
        },
        required_fields=["title", "video_url"],
        default_values={
            "learning_styles": [LearningStyle.VISUAL, LearningStyle.AUDITORY],
        },
        tags=["video", "lesson", "visual"],
    )

    # Interactive Tutorial Template
    INTERACTIVE_TUTORIAL = ContentTemplate(
        template_id="interactive_tutorial",
        name="Interactive Tutorial",
        description="Template for interactive learning experiences",
        content_type=ContentType.PROCEDURE,
        content_format=ContentFormat.INTERACTIVE,
        structure={
            "title": {"type": "string", "min_length": 5},
            "steps": {"type": "list"},
            "checkpoints": {"type": "list"},
        },
        required_fields=["title", "steps"],
        default_values={
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "learning_styles": [LearningStyle.KINESTHETIC],
        },
        tags=["interactive", "tutorial", "hands-on"],
    )

    # Flashcard Set Template
    FLASHCARD_SET = ContentTemplate(
        template_id="flashcard_set",
        name="Flashcard Set",
        description="Template for flashcard-based study materials",
        content_type=ContentType.FACT,
        content_format=ContentFormat.FLASHCARD,
        structure={
            "title": {"type": "string", "min_length": 5},
            "cards": {"type": "list"},
        },
        required_fields=["title", "cards"],
        default_values={
            "learning_styles": [LearningStyle.VISUAL, LearningStyle.READING],
        },
        tags=["flashcards", "study", "memorization"],
    )

    # Principle/Concept Summary Template
    PRINCIPLE_SUMMARY = ContentTemplate(
        template_id="principle_summary",
        name="Principle Summary",
        description="Template for summarizing principles and key concepts",
        content_type=ContentType.PRINCIPLE,
        content_format=ContentFormat.MARKDOWN,
        structure={
            "title": {"type": "string", "min_length": 5},
            "principle": {"type": "string"},
            "explanation": {"type": "string"},
            "applications": {"type": "list"},
        },
        required_fields=["title", "principle"],
        default_values={
            "difficulty": DifficultyLevel.INTERMEDIATE,
            "estimated_time_minutes": 10,
        },
        tags=["principle", "summary", "concept"],
    )

    @classmethod
    def get_all_templates(cls) -> List[ContentTemplate]:
        """Get all available templates."""
        return [
            cls.CONCEPT_EXPLANATION,
            cls.PROCEDURE_STEPS,
            cls.CODE_EXERCISE,
            cls.QUIZ_MULTIPLE_CHOICE,
            cls.VIDEO_LESSON,
            cls.INTERACTIVE_TUTORIAL,
            cls.FLASHCARD_SET,
            cls.PRINCIPLE_SUMMARY,
        ]

    @classmethod
    def get_template(cls, template_id: str) -> Optional[ContentTemplate]:
        """Get template by ID."""
        templates = {t.template_id: t for t in cls.get_all_templates()}
        return templates.get(template_id)

    @classmethod
    def get_templates_for_type(cls, content_type: ContentType) -> List[ContentTemplate]:
        """Get templates for a specific content type."""
        return [t for t in cls.get_all_templates() if t.content_type == content_type]


# =============================================================================
# Content Builder
# =============================================================================

class ContentBuilder:
    """
    Fluent builder for creating ContentItem objects.

    Example:
        content = ContentBuilder() \
            .with_title("Python Basics") \
            .with_body("Introduction to Python...") \
            .with_topics(["python", "programming"]) \
            .with_difficulty(DifficultyLevel.BEGINNER) \
            .build()
    """

    def __init__(self, template: Optional[ContentTemplate] = None):
        """
        Initialize builder.

        Args:
            template: Optional template to use as base
        """
        self.template = template
        self._data: Dict[str, Any] = {}

        # Apply template defaults if provided
        if template:
            self._data = template.apply_defaults(self._data)
            self._data["content_type"] = template.content_type
            self._data["content_format"] = template.content_format

    def with_id(self, content_id: str) -> 'ContentBuilder':
        """Set content ID."""
        self._data["content_id"] = content_id
        return self

    def with_title(self, title: str) -> 'ContentBuilder':
        """Set content title."""
        self._data["title"] = title
        return self

    def with_description(self, description: str) -> 'ContentBuilder':
        """Set content description."""
        self._data["description"] = description
        return self

    def with_body(self, body: str) -> 'ContentBuilder':
        """Set content body."""
        self._data["content_body"] = body
        return self

    def with_format(self, format: ContentFormat) -> 'ContentBuilder':
        """Set content format."""
        self._data["content_format"] = format
        return self

    def with_type(self, content_type: ContentType) -> 'ContentBuilder':
        """Set content type."""
        self._data["content_type"] = content_type
        return self

    def with_difficulty(self, difficulty: DifficultyLevel) -> 'ContentBuilder':
        """Set difficulty level."""
        self._data["difficulty"] = difficulty
        return self

    def with_topics(self, topics: List[str]) -> 'ContentBuilder':
        """Set topics."""
        self._data["topics"] = topics
        return self

    def with_tags(self, tags: List[str]) -> 'ContentBuilder':
        """Set tags."""
        self._data["tags"] = tags
        return self

    def with_categories(self, categories: List[str]) -> 'ContentBuilder':
        """Set categories."""
        self._data["categories"] = categories
        return self

    def with_prerequisites(self, prerequisites: List[str]) -> 'ContentBuilder':
        """Set prerequisites (content IDs)."""
        self._data["prerequisites"] = prerequisites
        return self

    def with_learning_objectives(self, objectives: List[str]) -> 'ContentBuilder':
        """Set learning objectives."""
        self._data["learning_objectives"] = objectives
        return self

    def for_learning_style(self, style: LearningStyle) -> 'ContentBuilder':
        """Add a learning style."""
        if "learning_styles" not in self._data:
            self._data["learning_styles"] = []
        self._data["learning_styles"].append(style)
        return self

    def for_learning_styles(self, styles: List[LearningStyle]) -> 'ContentBuilder':
        """Set learning styles."""
        self._data["learning_styles"] = styles
        return self

    def with_estimated_time(self, minutes: int) -> 'ContentBuilder':
        """Set estimated time in minutes."""
        self._data["estimated_time_minutes"] = minutes
        return self

    def with_source(
        self,
        source_type: ContentSourceType,
        source_id: Optional[str] = None,
        source_url: Optional[str] = None
    ) -> 'ContentBuilder':
        """Set source information."""
        self._data["source_type"] = source_type
        if source_id:
            self._data["source_id"] = source_id
        if source_url:
            self._data["source_url"] = source_url
        return self

    def with_author(self, author: str) -> 'ContentBuilder':
        """Set author."""
        self._data["author"] = author
        return self

    def with_asset(self, asset: ContentAsset) -> 'ContentBuilder':
        """Add an asset."""
        if "assets" not in self._data:
            self._data["assets"] = []
        self._data["assets"].append(asset)
        return self

    def with_assets(self, assets: List[ContentAsset]) -> 'ContentBuilder':
        """Set assets."""
        self._data["assets"] = assets
        return self

    def with_related_content(self, content_ids: List[str]) -> 'ContentBuilder':
        """Set related content IDs."""
        self._data["related_content"] = content_ids
        return self

    def with_next_content(self, content_id: str) -> 'ContentBuilder':
        """Set next content ID."""
        self._data["next_content"] = content_id
        return self

    def with_previous_content(self, content_id: str) -> 'ContentBuilder':
        """Set previous content ID."""
        self._data["previous_content"] = content_id
        return self

    def with_parent_content(self, content_id: str) -> 'ContentBuilder':
        """Set parent content ID."""
        self._data["parent_content"] = content_id
        return self

    def with_status(self, status: ContentStatus) -> 'ContentBuilder':
        """Set status."""
        self._data["status"] = status
        return self

    def as_published(self) -> 'ContentBuilder':
        """Set status to published."""
        self._data["status"] = ContentStatus.PUBLISHED
        self._data["published_at"] = datetime.now(timezone.utc)
        return self

    def as_draft(self) -> 'ContentBuilder':
        """Set status to draft."""
        self._data["status"] = ContentStatus.DRAFT
        return self

    def with_language(self, language: str) -> 'ContentBuilder':
        """Set language."""
        self._data["language"] = language
        return self

    def with_metadata(self, metadata: Dict[str, Any]) -> 'ContentBuilder':
        """Set metadata."""
        self._data["metadata"] = metadata
        return self

    def add_metadata(self, key: str, value: Any) -> 'ContentBuilder':
        """Add a metadata entry."""
        if "metadata" not in self._data:
            self._data["metadata"] = {}
        self._data["metadata"][key] = value
        return self

    def validate(self) -> List[str]:
        """
        Validate the built content.

        Returns list of validation errors.
        """
        errors = []

        # Basic validation
        if not self._data.get("title"):
            errors.append("Title is required")
        if not self._data.get("content_body"):
            errors.append("Content body is required")

        # Template validation
        if self.template:
            template_errors = self.template.validate(self._data)
            errors.extend(template_errors)

        return errors

    def build(self, validate: bool = True) -> ContentItem:
        """
        Build the ContentItem.

        Args:
            validate: Whether to validate before building

        Returns:
            ContentItem object

        Raises:
            ValueError: If validation fails
        """
        if validate:
            errors = self.validate()
            if errors:
                raise ValueError(f"Validation failed: {', '.join(errors)}")

        # Set defaults
        if "content_id" not in self._data:
            self._data["content_id"] = str(uuid.uuid4())
        if "content_format" not in self._data:
            self._data["content_format"] = ContentFormat.TEXT
        if "content_type" not in self._data:
            self._data["content_type"] = ContentType.CONCEPT
        if "difficulty" not in self._data:
            self._data["difficulty"] = DifficultyLevel.INTERMEDIATE
        if "source_type" not in self._data:
            self._data["source_type"] = ContentSourceType.AUTHORED
        if "status" not in self._data:
            self._data["status"] = ContentStatus.DRAFT

        return ContentItem(**self._data)

    def to_dict(self) -> Dict[str, Any]:
        """Get current builder data as dict."""
        return self._data.copy()

    def from_dict(self, data: Dict[str, Any]) -> 'ContentBuilder':
        """Load data from dict."""
        self._data.update(data)
        return self

    def from_content(self, content: ContentItem) -> 'ContentBuilder':
        """Load data from existing ContentItem."""
        self._data = content.to_dict()
        return self

    def copy(self) -> 'ContentBuilder':
        """Create a copy of this builder."""
        new_builder = ContentBuilder(self.template)
        new_builder._data = self._data.copy()
        return new_builder


# =============================================================================
# Quick Content Creators
# =============================================================================

def quick_concept(
    title: str,
    body: str,
    topic: str,
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
) -> ContentItem:
    """Quick create a concept explanation."""
    return ContentBuilder(ContentTemplateLibrary.CONCEPT_EXPLANATION) \
        .with_title(title) \
        .with_body(body) \
        .with_topics([topic]) \
        .with_difficulty(difficulty) \
        .build()


def quick_exercise(
    title: str,
    problem: str,
    topic: str,
    language: str = "python",
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
) -> ContentItem:
    """Quick create a code exercise."""
    return ContentBuilder(ContentTemplateLibrary.CODE_EXERCISE) \
        .with_title(title) \
        .with_body(problem) \
        .with_topics([topic, language]) \
        .with_difficulty(difficulty) \
        .add_metadata("programming_language", language) \
        .build()


def quick_quiz(
    title: str,
    questions_content: str,
    topics: List[str],
    difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE
) -> ContentItem:
    """Quick create a quiz."""
    return ContentBuilder(ContentTemplateLibrary.QUIZ_MULTIPLE_CHOICE) \
        .with_title(title) \
        .with_body(questions_content) \
        .with_topics(topics) \
        .with_difficulty(difficulty) \
        .build()
