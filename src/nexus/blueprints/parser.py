"""
Blueprint Parser

Loads blueprint JSON and converts to BlueprintSpec objects.
Handles both original blueprint format and generated blueprint format.
"""

import json
from pathlib import Path
from typing import Union

from .models import BlueprintSpec, BookSpec, ChapterSpec


class BlueprintParser:
    """Parser for blueprint JSON files."""
    
    def __init__(self):
        self.blueprints: dict[str, BlueprintSpec] = {}
    
    def load_file(self, path: Union[str, Path]) -> dict[str, BlueprintSpec]:
        """Load blueprints from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Blueprint file not found: {path}")
        
        data = json.loads(path.read_text(encoding='utf-8'))
        
        # Handle both single blueprint and array of blueprints
        if isinstance(data, list):
            for bp_data in data:
                bp = self._parse_blueprint(bp_data)
                self.blueprints[bp.library_id] = bp
        else:
            bp = self._parse_blueprint(data)
            self.blueprints[bp.library_id] = bp
        
        return self.blueprints
    
    def load_directory(self, path: Union[str, Path]) -> dict[str, BlueprintSpec]:
        """Load all blueprint JSON files from a directory."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Blueprint directory not found: {path}")
        
        for json_file in path.glob("*.json"):
            try:
                self.load_file(json_file)
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
        
        return self.blueprints
    
    def load_json(self, data: Union[str, dict, list]) -> dict[str, BlueprintSpec]:
        """Load blueprints from JSON string or dict."""
        if isinstance(data, str):
            data = json.loads(data)
        
        if isinstance(data, list):
            for bp_data in data:
                bp = self._parse_blueprint(bp_data)
                self.blueprints[bp.library_id] = bp
        else:
            bp = self._parse_blueprint(data)
            self.blueprints[bp.library_id] = bp
        
        return self.blueprints
    
    def _parse_blueprint(self, data: dict) -> BlueprintSpec:
        """Parse a single blueprint dict into BlueprintSpec."""
        meta = data.get('blueprint_meta', {})
        exec_summary = data.get('executive_summary', {})
        design = data.get('series_design_principles', {})
        structure = data.get('book_structure', {})
        catalog = data.get('catalog_baseline', {})
        value_stream = data.get('value_stream', {})
        dod = data.get('definition_of_done', {})
        artifacts = data.get('artifact_packs', {})
        
        # Parse chapters from book_structure.sections (original format)
        global_chapters = self._parse_chapters(structure.get('sections', []))
        
        # Parse books from catalog - handles both formats
        books = self._parse_books(catalog.get('items', []), global_chapters)
        
        # Parse artifact types from required_modules
        artifact_types = []
        artifact_structure = []
        for module in structure.get('required_modules', []):
            if module.get('module_type') == 'artifact_types':
                artifact_types = module.get('items', [])
                artifact_structure = module.get('per_item_structure', [])
        
        # Get scope boundaries
        scope = exec_summary.get('scope_boundaries', {})
        
        return BlueprintSpec(
            # Meta
            blueprint_id=meta.get('blueprint_id', ''),
            library_id=meta.get('library_id', ''),
            library_style=meta.get('library_style', 'ebook_series'),
            version=meta.get('version', '1.0.0'),
            
            # Executive summary
            purpose=exec_summary.get('purpose', ''),
            target_audience=exec_summary.get('target_audience', []),
            primary_outcomes=exec_summary.get('primary_outcomes', []),
            scope_includes=scope.get('includes', []),
            scope_excludes=scope.get('excludes', []),
            
            # Design principles
            core_principles=design.get('core_principles', []),
            required_framing=design.get('required_framing', ''),
            tone_guidelines=design.get('tone_guidelines', []),
            edition_strategy=design.get('edition_strategy', 'internal_first'),
            
            # Book structure
            book_promise_template=structure.get('book_promise', ''),
            artifact_types=artifact_types,
            artifact_structure=artifact_structure,
            
            # Catalog
            books=books,
            
            # Value stream
            value_stream_stages=value_stream.get('stages', []),
            
            # Definition of done
            definition_of_done=dod.get('criteria', []),
            
            # Artifact packs
            per_book_artifacts=artifacts.get('per_book_artifacts', []),
            marketing_artifacts=artifacts.get('marketing_artifacts', []),
            sales_artifacts=artifacts.get('sales_artifacts', []),
        )
    
    def _parse_chapters(self, sections: list[dict]) -> list[ChapterSpec]:
        """Parse chapter specifications from sections."""
        chapters = []
        for section in sections:
            token_range = section.get('target_token_range', {})
            chapters.append(ChapterSpec(
                position=section.get('position', 0),
                section_id=section.get('section_id', ''),
                title=section.get('title', ''),
                purpose=section.get('purpose', ''),
                required=section.get('required', True),
                min_tokens=token_range.get('min', 6000),
                max_tokens=token_range.get('max', 8000),
                content_requirements=section.get('content_requirements', []),
            ))
        return chapters
    
    def _parse_chapters_inline(self, chapters_data: list[dict]) -> list[ChapterSpec]:
        """Parse chapters from inline format (generated blueprints)."""
        chapters = []
        for ch in chapters_data:
            chapters.append(ChapterSpec(
                position=ch.get('chapter_number', ch.get('position', 0)),
                section_id=ch.get('section_id', f"CH{ch.get('chapter_number', 0):02d}"),
                title=ch.get('title', ''),
                purpose=ch.get('purpose', ch.get('primary_outcome', '')),
                required=ch.get('required', True),
                min_tokens=ch.get('min_tokens', 6000),
                max_tokens=ch.get('max_tokens', 8000),
                content_requirements=ch.get('key_topics', ch.get('content_requirements', [])),
            ))
        return chapters
    
    def _parse_books(self, items: list[dict], global_chapters: list[ChapterSpec]) -> list[BookSpec]:
        """Parse book specifications from catalog items.
        
        Handles two formats:
        1. Original: chapters come from global book_structure.sections
        2. Generated: chapters are inline in each catalog item
        """
        books = []
        for item in items:
            # Check if book has inline chapters (generated format)
            inline_chapters = item.get('chapters', [])
            
            if inline_chapters:
                # Generated blueprint format - chapters inline
                book_chapters = self._parse_chapters_inline(inline_chapters)
            elif global_chapters:
                # Original blueprint format - use global chapters
                book_chapters = [
                    ChapterSpec(
                        position=ch.position,
                        section_id=ch.section_id,
                        title=ch.title,
                        purpose=ch.purpose,
                        required=ch.required,
                        min_tokens=ch.min_tokens,
                        max_tokens=ch.max_tokens,
                        content_requirements=ch.content_requirements.copy(),
                    )
                    for ch in global_chapters
                ]
            else:
                # No chapters - create placeholder
                book_chapters = [
                    ChapterSpec(
                        position=1,
                        section_id="CH01",
                        title="Introduction",
                        purpose="Introduce the topic",
                        required=True,
                        min_tokens=6000,
                        max_tokens=8000,
                        content_requirements=["Overview", "Key concepts"],
                    )
                ]
            
            books.append(BookSpec(
                item_id=item.get('item_id', ''),
                position=item.get('position', 0),
                title=item.get('title', ''),
                subtitle=item.get('subtitle', ''),
                primary_outcome=item.get('primary_outcome', ''),
                stage=item.get('stage', ''),
                dependencies=item.get('dependencies', []),
                tags=item.get('tags', []),
                chapters=book_chapters,
            ))
        return books
    
    # Convenience methods
    
    def get_blueprint(self, blueprint_id: str) -> BlueprintSpec | None:
        """Get a blueprint by ID."""
        # Try direct lookup first
        if blueprint_id in self.blueprints:
            return self.blueprints[blueprint_id]
        
        # Try by blueprint_id field
        for bp in self.blueprints.values():
            if bp.blueprint_id == blueprint_id:
                return bp
        return None
    
    def get_blueprint_by_library(self, library_id: str) -> BlueprintSpec | None:
        """Get a blueprint by library ID."""
        return self.blueprints.get(library_id)
    
    def list_blueprints(self) -> list[tuple[str, str, int]]:
        """List all blueprints as (id, library_id, book_count)."""
        return [(bp.blueprint_id, bp.library_id, len(bp.books)) for bp in self.blueprints.values()]
    
    def list_books(self, blueprint_id: str = None) -> list[tuple[str, str, str]]:
        """List all books as (blueprint_id, item_id, title)."""
        results = []
        for bp in self.blueprints.values():
            if blueprint_id and bp.blueprint_id != blueprint_id:
                continue
            for book in bp.books:
                results.append((bp.blueprint_id, book.item_id, book.title))
        return results
    
    def __len__(self) -> int:
        return len(self.blueprints)
    
    def __iter__(self):
        return iter(self.blueprints.values())
    
    def __getitem__(self, key: str) -> BlueprintSpec:
        return self.blueprints[key]
