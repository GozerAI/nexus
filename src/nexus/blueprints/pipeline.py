"""
Blueprint Ebook Pipeline

Generates ebooks from blueprint specifications using configurable LLM backends.
Supports resume from checkpoint and real-time progress tracking.
"""

import asyncio
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from pathlib import Path
from datetime import datetime

from .models import BlueprintSpec, BookSpec, ChapterSpec
from .llm_backend import LLMBackend, get_backend, LLMResponse


@dataclass
class ChapterOutput:
    """Output from generating a single chapter."""
    chapter_spec: ChapterSpec
    content: str
    artifacts: list[dict] = field(default_factory=list)
    token_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "position": self.chapter_spec.position,
            "title": self.chapter_spec.title,
            "content": self.content,
            "artifacts": self.artifacts,
            "token_count": self.token_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error
        }


@dataclass 
class BookOutput:
    """Output from generating a complete book."""
    book_spec: BookSpec
    blueprint_spec: BlueprintSpec
    chapters: list[ChapterOutput] = field(default_factory=list)
    total_duration_seconds: float = 0
    provider: str = ""
    model: str = ""
    
    @property
    def success(self) -> bool:
        return all(ch.success for ch in self.chapters)
    
    @property
    def total_tokens(self) -> int:
        return sum(ch.token_count for ch in self.chapters)
    
    @property
    def total_input_tokens(self) -> int:
        return sum(ch.input_tokens for ch in self.chapters)
    
    @property
    def total_output_tokens(self) -> int:
        return sum(ch.output_tokens for ch in self.chapters)
    
    @property
    def successful_chapters(self) -> int:
        return sum(1 for ch in self.chapters if ch.success)
    
    def to_markdown(self) -> str:
        """Assemble book as markdown."""
        lines = []
        
        # Title page
        lines.append(f"# {self.book_spec.title}")
        if self.book_spec.subtitle:
            lines.append(f"## {self.book_spec.subtitle}")
        lines.append("")
        lines.append(f"**Primary Outcome:** {self.book_spec.primary_outcome}")
        lines.append("")
        lines.append(f"**For:** {self.blueprint_spec.audience_string}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Table of contents
        lines.append("## Table of Contents")
        lines.append("")
        for ch_output in self.chapters:
            if ch_output.success:
                lines.append(f"- Chapter {ch_output.chapter_spec.position}: {ch_output.chapter_spec.title}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Chapters
        for ch_output in self.chapters:
            if ch_output.success and ch_output.content:
                lines.append(ch_output.content)
                lines.append("")
                lines.append("---")
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class GenerationCheckpoint:
    """Checkpoint for resuming book generation."""
    blueprint_id: str
    book_position: int
    book_title: str
    completed_chapters: list[int]
    chapter_summaries: dict[int, str]  # position -> summary
    last_updated: str
    provider: str
    model: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "GenerationCheckpoint":
        return cls(**data)
    
    def save(self, path: Path):
        """Save checkpoint to file."""
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
    
    @classmethod
    def load(cls, path: Path) -> "GenerationCheckpoint":
        """Load checkpoint from file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


class PromptBuilder:
    """Builds prompts for chapter generation from blueprint specs."""
    
    def __init__(self, blueprint: BlueprintSpec, book: BookSpec):
        self.blueprint = blueprint
        self.book = book
    
    def build_system_prompt(self) -> str:
        """Build the system prompt with tone and framing."""
        lines = [
            "You are an expert ebook author writing practical, actionable content.",
            "",
            "## Series Context",
            f"Series: {self.blueprint.library_id}",
            f"Book: {self.book.title}",
            f"Primary Outcome: {self.book.primary_outcome}",
            "",
            "## Target Audience",
        ]
        for aud in self.blueprint.target_audience:
            lines.append(f"- {aud}")
        
        lines.extend([
            "",
            "## Tone Guidelines",
        ])
        for tone in self.blueprint.tone_guidelines:
            lines.append(f"- {tone}")
        
        lines.extend([
            "",
            "## Required Framing",
            self.blueprint.required_framing,
            "",
            "## Artifact Types to Include",
        ])
        for art in self.blueprint.artifact_types:
            lines.append(f"- {art}")
        
        lines.extend([
            "",
            "## Content Requirements (per chapter)",
        ])
        if self.book.chapters:
            for req in self.book.chapters[0].content_requirements:
                lines.append(f"- {req}")
        
        return "\n".join(lines)

    
    def build_chapter_prompt(
        self, 
        chapter: ChapterSpec, 
        previous_summary: str = "",
        previous_chapters: list[str] = None
    ) -> str:
        """Build the prompt for generating a specific chapter."""
        word_target = chapter.target_tokens // 4
        word_min = chapter.min_tokens // 4
        word_max = chapter.max_tokens // 4
        
        lines = [
            f"# Write Chapter {chapter.position}: {chapter.title}",
            "",
            "## CRITICAL LENGTH REQUIREMENT",
            f"",
            f"**YOU MUST WRITE AT LEAST {word_min} WORDS ({chapter.min_tokens} tokens).**",
            f"**TARGET: {word_target} WORDS ({chapter.target_tokens} tokens).**",
            f"**MAXIMUM: {word_max} WORDS ({chapter.max_tokens} tokens).**",
            f"",
            f"This is a FULL chapter, not a summary. Write comprehensive, detailed content.",
            f"A typical chapter has 8-12 major sections with multiple paragraphs each.",
            f"If your response is under {word_min} words, you have NOT completed the task.",
            "",
            "## Chapter Purpose",
            chapter.purpose,
            "",
            "## Content Requirements",
        ]
        for req in chapter.content_requirements:
            lines.append(f"- {req}")
        
        lines.extend([
            "",
            "## Book Context",
            f"Book Title: {self.book.title}",
            f"Book Outcome: {self.book.primary_outcome}",
            f"This is chapter {chapter.position} of {len(self.book.chapters)}.",
        ])
        
        # Add previous chapter context for continuity
        if previous_summary:
            lines.extend([
                "",
                "## Previous Chapter Summary",
                previous_summary,
            ])
        
        if previous_chapters:
            lines.extend([
                "",
                "## Chapter Titles So Far",
            ])
            for i, title in enumerate(previous_chapters, 1):
                lines.append(f"{i}. {title}")
        
        lines.extend([
            "",
            "## Structure Requirements",
            f"Your chapter MUST include ALL of the following:",
            f"1. Chapter title as markdown heading: # Chapter {chapter.position}: {chapter.title}",
            f"2. Opening section (2-3 paragraphs) explaining the chapter's importance",
            f"3. At least 6-8 major content sections with ### headings",
            f"4. Each section should have 3-5 paragraphs of detailed explanation",
            f"5. At least 2 practical artifacts (checklists, templates, worksheets, decision trees)",
            f"6. Real-world examples and scenarios throughout",
            f"7. Decision points with safe defaults clearly marked",
            f"8. A comprehensive verification checklist at the end (10+ items)",
            f"9. Brief transition to next chapter topic",
            "",
            f"## FINAL REMINDER: Write {word_target}+ words. This is a complete book chapter, not an outline.",
            "",
            "Write the full chapter now:",
        ])
        
        return "\n".join(lines)
    
    def build_summary_prompt(self, chapter_content: str) -> str:
        """Build prompt to summarize a chapter for continuity."""
        return f"""Summarize this chapter in 2-3 sentences for context continuity. 
Focus on: key concepts introduced, main takeaways, and how it sets up following chapters.

Chapter content:
{chapter_content[:3000]}

Summary:"""


class BlueprintEbookPipeline:
    """Pipeline for generating ebooks from blueprint specifications."""
    
    def __init__(
        self, 
        blueprint: BlueprintSpec,
        backend: LLMBackend = None,
        provider: str = "anthropic",
        model: str = None,
        checkpoint_dir: Path = None
    ):
        self.blueprint = blueprint
        
        # Initialize backend
        if backend:
            self.backend = backend
        else:
            self.backend = get_backend(provider=provider, model=model)
        
        self.provider = provider
        self.model = model or (backend.model if backend else None)
        self.prompt_builder = None
        
        # Checkpoint support
        self.checkpoint_dir = checkpoint_dir or Path(__file__).parent.parent.parent / "output" / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress callback: (current, total, message)
        self.on_progress: Optional[Callable[[int, int, str], None]] = None
        
        # Chapter callback: (chapter_output)
        self.on_chapter_complete: Optional[Callable[[ChapterOutput], None]] = None

    
    def _get_checkpoint_path(self, book: BookSpec) -> Path:
        """Get checkpoint file path for a book."""
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in book.title)[:30]
        return self.checkpoint_dir / f"checkpoint_{self.blueprint.library_id}_{safe_title}.json"
    
    def _report_progress(self, current: int, total: int, message: str):
        """Report progress if callback is set."""
        if self.on_progress:
            self.on_progress(current, total, message)
        print(f"[{current}/{total}] {message}")
    
    async def generate_chapter(
        self, 
        book: BookSpec, 
        chapter: ChapterSpec,
        previous_summary: str = "",
        previous_chapters: list[str] = None
    ) -> ChapterOutput:
        """Generate a single chapter from spec."""
        
        self.prompt_builder = PromptBuilder(self.blueprint, book)
        
        try:
            system_prompt = self.prompt_builder.build_system_prompt()
            chapter_prompt = self.prompt_builder.build_chapter_prompt(
                chapter, 
                previous_summary,
                previous_chapters
            )
            
            # Generate chapter
            response = await self.backend.generate(
                prompt=chapter_prompt,
                system_prompt=system_prompt,
                max_tokens=chapter.max_tokens + 1000  # Allow some overhead
            )
            
            # Extract artifacts
            artifacts = self._extract_artifacts(response.content)
            
            return ChapterOutput(
                chapter_spec=chapter,
                content=response.content,
                artifacts=artifacts,
                token_count=response.tokens_used,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                duration_seconds=response.duration_seconds,
                success=True
            )
            
        except Exception as e:
            return ChapterOutput(
                chapter_spec=chapter,
                content="",
                token_count=0,
                success=False,
                error=str(e)
            )
    
    async def generate_book(
        self,
        book: BookSpec,
        resume_from: int = 0,
        max_chapters: int = None,
        save_checkpoints: bool = True
    ) -> BookOutput:
        """Generate a complete book.
        
        Args:
            book: Book specification
            resume_from: Chapter position to resume from (1-indexed, 0 = start fresh)
            max_chapters: Maximum chapters to generate (None = all)
            save_checkpoints: Save progress after each chapter
        """
        import time
        start_time = time.time()
        
        self.prompt_builder = PromptBuilder(self.blueprint, book)
        
        output = BookOutput(
            book_spec=book,
            blueprint_spec=self.blueprint,
            provider=self.provider or "anthropic",
            model=self.model or self.backend.model
        )
        
        # Try to load checkpoint
        checkpoint_path = self._get_checkpoint_path(book)
        checkpoint = None
        chapter_summaries = {}
        
        if resume_from == 0 and checkpoint_path.exists():
            try:
                checkpoint = GenerationCheckpoint.load(checkpoint_path)
                chapter_summaries = checkpoint.chapter_summaries
                self._report_progress(0, len(book.chapters), 
                    f"Resuming from checkpoint: {len(checkpoint.completed_chapters)} chapters done")
            except Exception as e:
                self._report_progress(0, len(book.chapters), f"Could not load checkpoint: {e}")
        
        chapters_to_generate = book.chapters
        
        # Handle chapter range selection
        if resume_from > 0:
            # Filter to chapters at or after resume_from position
            chapters_to_generate = [ch for ch in chapters_to_generate if ch.position >= resume_from]
        
        if max_chapters:
            chapters_to_generate = chapters_to_generate[:max_chapters]
        
        total = len(chapters_to_generate)
        previous_summary = ""
        previous_chapter_titles = []
        
        # Build list of prior chapter titles for context
        for ch in book.chapters:
            if ch.position < (resume_from if resume_from > 0 else 1):
                previous_chapter_titles.append(ch.title)

        
        for i, chapter in enumerate(chapters_to_generate):
            # Skip if in checkpoint or resuming
            if checkpoint and chapter.position in checkpoint.completed_chapters:
                previous_chapter_titles.append(chapter.title)
                # Use saved summary if available
                if chapter.position in chapter_summaries:
                    previous_summary = chapter_summaries[chapter.position]
                continue
            
            if chapter.position < resume_from:
                previous_chapter_titles.append(chapter.title)
                continue
            
            self._report_progress(
                i + 1, 
                total, 
                f"Generating: {chapter.title}"
            )
            
            # Generate chapter
            ch_output = await self.generate_chapter(
                book=book,
                chapter=chapter,
                previous_summary=previous_summary,
                previous_chapters=previous_chapter_titles if previous_chapter_titles else None
            )
            
            output.chapters.append(ch_output)
            
            if ch_output.success:
                # Generate summary for next chapter's context
                previous_summary = await self._summarize_chapter(ch_output.content)
                previous_chapter_titles.append(chapter.title)
                chapter_summaries[chapter.position] = previous_summary
                
                # Notify callback
                if self.on_chapter_complete:
                    self.on_chapter_complete(ch_output)
                
                self._report_progress(
                    i + 1, 
                    total, 
                    f"Completed: {chapter.title} ({ch_output.token_count:,} tokens, {ch_output.duration_seconds:.1f}s)"
                )
                
                # Save checkpoint
                if save_checkpoints:
                    completed = list(checkpoint.completed_chapters) if checkpoint else []
                    completed.append(chapter.position)
                    
                    new_checkpoint = GenerationCheckpoint(
                        blueprint_id=self.blueprint.library_id,
                        book_position=book.position,
                        book_title=book.title,
                        completed_chapters=completed,
                        chapter_summaries=chapter_summaries,
                        last_updated=datetime.now().isoformat(),
                        provider=self.provider or "anthropic",
                        model=self.model or ""
                    )
                    new_checkpoint.save(checkpoint_path)
            else:
                self._report_progress(
                    i + 1, 
                    total, 
                    f"FAILED: {chapter.title} - {ch_output.error}"
                )
        
        output.total_duration_seconds = time.time() - start_time
        
        # Clean up checkpoint if book completed successfully
        if output.success and checkpoint_path.exists():
            checkpoint_path.unlink()
        
        return output
    
    async def _summarize_chapter(self, content: str) -> str:
        """Generate a brief summary of a chapter for continuity."""
        try:
            prompt = self.prompt_builder.build_summary_prompt(content)
            
            response = await self.backend.generate(
                prompt=prompt,
                system_prompt="You are a helpful assistant that creates concise summaries.",
                max_tokens=200
            )
            
            return response.content.strip()
        except Exception:
            # If summarization fails, use truncated content
            return content[:500] + "..."
    
    def _extract_artifacts(self, content: str) -> list[dict]:
        """Extract artifact sections from content."""
        artifacts = []
        
        artifact_markers = [
            ('checklist', ['## Checklist', '### Checklist', '**Checklist']),
            ('template', ['## Template', '### Template', '**Template']),
            ('worksheet', ['## Worksheet', '### Worksheet', '**Worksheet']),
            ('decision_tree', ['## Decision Tree', '### Decision', '**Decision']),
            ('runbook', ['## Runbook', '### Runbook', '## SOP', '### SOP']),
            ('scorecard', ['## Scorecard', '### Scorecard', '**Scorecard']),
            ('verification', ['## Verification', '### Verification', '**Verification']),
        ]
        
        content_lower = content.lower()
        for artifact_type, markers in artifact_markers:
            for marker in markers:
                if marker.lower() in content_lower:
                    artifacts.append({
                        'type': artifact_type,
                        'found': True,
                        'marker': marker
                    })
                    break
        
        return artifacts

    
    def estimate_book_time(self, book: BookSpec, tokens_per_second: float = 5.0) -> dict:
        """Estimate time to generate a book.
        
        Args:
            book: Book specification
            tokens_per_second: Expected generation speed
        """
        total_tokens = book.total_target_tokens
        chapters = len(book.chapters)
        
        # Account for prompts and summaries
        overhead_tokens = chapters * 2000  # ~2k tokens per chapter for prompts
        total_with_overhead = total_tokens + overhead_tokens
        
        seconds = total_with_overhead / tokens_per_second
        
        return {
            'total_tokens': total_tokens,
            'overhead_tokens': overhead_tokens,
            'chapters': chapters,
            'estimated_seconds': seconds,
            'estimated_minutes': seconds / 60,
            'estimated_hours': seconds / 3600,
            'tokens_per_second': tokens_per_second
        }
    
    def estimate_cost(self, book: BookSpec) -> dict:
        """Estimate cost to generate a book.
        
        Args:
            book: Book specification
        """
        total_output = book.total_target_tokens
        # Estimate input tokens (prompts, system messages)
        chapters = len(book.chapters)
        total_input = chapters * 2000  # ~2k tokens per chapter for prompts
        
        cost = self.backend.estimate_cost(total_input, total_output)
        
        return {
            'estimated_input_tokens': total_input,
            'estimated_output_tokens': total_output,
            'estimated_cost_usd': cost,
            'provider': self.provider or "anthropic"
        }
