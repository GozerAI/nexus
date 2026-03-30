"""
Blueprint Data Models

Dataclasses representing the blueprint schema for ebook generation.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChapterSpec:
    """Specification for a single chapter."""
    position: int
    section_id: str
    title: str
    purpose: str
    required: bool
    min_tokens: int
    max_tokens: int
    content_requirements: list[str]
    
    @property
    def target_tokens(self) -> int:
        """Target token count (midpoint of range)."""
        return (self.min_tokens + self.max_tokens) // 2
    
    def __str__(self) -> str:
        return f"Ch{self.position}: {self.title} ({self.min_tokens}-{self.max_tokens} tokens)"


@dataclass
class BookSpec:
    """Specification for a single book in a series."""
    item_id: str
    position: int
    title: str
    subtitle: str
    primary_outcome: str
    stage: str
    dependencies: list[str]
    tags: list[str]
    chapters: list[ChapterSpec] = field(default_factory=list)
    
    @property
    def total_target_tokens(self) -> int:
        """Total target tokens for the book."""
        return sum(ch.target_tokens for ch in self.chapters)
    
    def __str__(self) -> str:
        return f"{self.item_id}: {self.title}"


@dataclass
class BlueprintSpec:
    """Full blueprint specification for an ebook series."""
    # Meta
    blueprint_id: str
    library_id: str
    library_style: str
    version: str
    
    # Executive summary
    purpose: str
    target_audience: list[str]
    primary_outcomes: list[str]
    scope_includes: list[str]
    scope_excludes: list[str]
    
    # Design principles
    core_principles: list[dict]
    required_framing: str
    tone_guidelines: list[str]
    edition_strategy: str
    
    # Book structure
    book_promise_template: str
    artifact_types: list[str]
    artifact_structure: list[str]
    
    # Catalog
    books: list[BookSpec] = field(default_factory=list)
    
    # Value stream
    value_stream_stages: list[dict] = field(default_factory=list)
    
    # Definition of done
    definition_of_done: list[dict] = field(default_factory=list)
    
    # Artifact packs
    per_book_artifacts: list[str] = field(default_factory=list)
    marketing_artifacts: list[str] = field(default_factory=list)
    sales_artifacts: list[str] = field(default_factory=list)
    
    def get_book(self, item_id: str) -> Optional[BookSpec]:
        """Get a book by item_id."""
        for book in self.books:
            if book.item_id == item_id:
                return book
        return None
    
    def get_book_by_position(self, position: int) -> Optional[BookSpec]:
        """Get a book by position (1-indexed)."""
        for book in self.books:
            if book.position == position:
                return book
        return None
    
    def get_book_by_title(self, title: str) -> Optional[BookSpec]:
        """Get a book by title (partial match)."""
        title_lower = title.lower()
        for book in self.books:
            if title_lower in book.title.lower():
                return book
        return None
    
    @property
    def audience_string(self) -> str:
        """Target audience as comma-separated string."""
        return ", ".join(self.target_audience[:2]) if self.target_audience else "General readers"
    
    @property
    def tone_string(self) -> str:
        """Tone guidelines as string."""
        return self.tone_guidelines[0] if self.tone_guidelines else "Professional"
    
    def __str__(self) -> str:
        return f"{self.blueprint_id} ({self.library_id}): {len(self.books)} books"
