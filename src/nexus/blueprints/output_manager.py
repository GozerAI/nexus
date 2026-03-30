"""
Output Manager - Handles saving generated books and chapters to disk.

Directory structure:
  output/books/{library_id}/{book_title}/
    ├── chapters/
    │   ├── 01_quick_start.md
    │   ├── 02_use_case_selection.md
    │   └── ...
    ├── full_book.md
    └── metadata.json
"""

import json
import re
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from typing import Optional

from .pipeline import BookOutput, ChapterOutput


def slugify(text: str) -> str:
    """Convert text to filesystem-safe slug."""
    # Replace special chars with underscore
    text = re.sub(r'[^\w\s-]', '', text)
    # Replace whitespace with underscore
    text = re.sub(r'[\s-]+', '_', text)
    return text.strip('_')


class OutputManager:
    """Manages saving generated content to disk."""
    
    def __init__(self, base_dir: str | Path = "output/books"):
        self.base_dir = Path(base_dir)
    
    def get_book_dir(self, library_id: str, book_title: str) -> Path:
        """Get the output directory for a book."""
        book_slug = slugify(book_title)
        return self.base_dir / library_id / book_slug
    
    def save_book(
        self, 
        output: BookOutput,
        overwrite: bool = False
    ) -> dict:
        """Save a complete book output to disk.
        
        Returns dict with paths to saved files.
        """
        library_id = output.blueprint_spec.library_id
        book_title = output.book_spec.title
        
        # Create directory structure
        book_dir = self.get_book_dir(library_id, book_title)
        chapters_dir = book_dir / "chapters"
        
        if book_dir.exists() and not overwrite:
            # Add timestamp to avoid overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            book_dir = book_dir.parent / f"{book_dir.name}_{timestamp}"
            chapters_dir = book_dir / "chapters"
        
        book_dir.mkdir(parents=True, exist_ok=True)
        chapters_dir.mkdir(exist_ok=True)
        
        saved_files = {
            "book_dir": str(book_dir),
            "chapters": [],
            "full_book": None,
            "metadata": None
        }
        
        # Save individual chapters
        for ch_output in output.chapters:
            if ch_output.success and ch_output.content:
                ch_path = self._save_chapter(chapters_dir, ch_output)
                saved_files["chapters"].append(str(ch_path))
        
        # Save assembled book
        full_book_path = book_dir / "full_book.md"
        full_book_path.write_text(output.to_markdown(), encoding='utf-8')
        saved_files["full_book"] = str(full_book_path)
        
        # Save metadata
        metadata_path = book_dir / "metadata.json"
        metadata = self._build_metadata(output)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding='utf-8'
        )
        saved_files["metadata"] = str(metadata_path)
        
        return saved_files
    
    def _save_chapter(
        self, 
        chapters_dir: Path, 
        ch_output: ChapterOutput
    ) -> Path:
        """Save a single chapter to disk."""
        position = ch_output.chapter_spec.position
        title_slug = slugify(ch_output.chapter_spec.title)
        filename = f"{position:02d}_{title_slug}.md"
        
        ch_path = chapters_dir / filename
        
        # Add metadata header
        header = f"""---
chapter: {position}
title: "{ch_output.chapter_spec.title}"
tokens: {ch_output.token_count}
duration_seconds: {ch_output.duration_seconds:.1f}
generated_at: "{datetime.now().isoformat()}"
---

"""
        content = header + ch_output.content
        ch_path.write_text(content, encoding='utf-8')
        
        return ch_path
    
    def _build_metadata(self, output: BookOutput) -> dict:
        """Build metadata dict for a book output."""
        return {
            "generation": {
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": output.total_duration_seconds,
                "total_tokens": output.total_tokens,
                "successful_chapters": output.successful_chapters,
                "total_chapters": len(output.chapters),
                "success": output.success
            },
            "blueprint": {
                "library_id": output.blueprint_spec.library_id,
                "blueprint_id": output.blueprint_spec.blueprint_id,
                "version": output.blueprint_spec.version
            },
            "book": {
                "item_id": output.book_spec.item_id,
                "title": output.book_spec.title,
                "subtitle": output.book_spec.subtitle,
                "primary_outcome": output.book_spec.primary_outcome,
                "position": output.book_spec.position
            },
            "chapters": [
                {
                    "position": ch.chapter_spec.position,
                    "title": ch.chapter_spec.title,
                    "tokens": ch.token_count,
                    "duration_seconds": ch.duration_seconds,
                    "success": ch.success,
                    "artifacts": ch.artifacts,
                    "error": ch.error
                }
                for ch in output.chapters
            ]
        }
    
    def load_metadata(self, library_id: str, book_title: str) -> Optional[dict]:
        """Load metadata for a previously generated book."""
        book_dir = self.get_book_dir(library_id, book_title)
        metadata_path = book_dir / "metadata.json"
        
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding='utf-8'))
        return None
    
    def list_generated_books(self) -> list[dict]:
        """List all generated books with basic info."""
        books = []
        
        if not self.base_dir.exists():
            return books
        
        for library_dir in self.base_dir.iterdir():
            if not library_dir.is_dir():
                continue
            
            for book_dir in library_dir.iterdir():
                if not book_dir.is_dir():
                    continue
                
                metadata_path = book_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
                        books.append({
                            "library_id": library_dir.name,
                            "book_dir": str(book_dir),
                            "title": meta.get("book", {}).get("title", book_dir.name),
                            "chapters": meta.get("generation", {}).get("successful_chapters", 0),
                            "tokens": meta.get("generation", {}).get("total_tokens", 0),
                            "timestamp": meta.get("generation", {}).get("timestamp", "")
                        })
                    except Exception:
                        pass
        
        return books
