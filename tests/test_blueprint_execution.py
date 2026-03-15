"""Tests for the blueprint system — models, parser, and pipeline."""

import json
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.blueprints.models import BlueprintSpec, BookSpec, ChapterSpec
from nexus.blueprints.parser import BlueprintParser
from nexus.blueprints.pipeline import ChapterOutput, BookOutput


# ---------------------------------------------------------------------------
# Helpers — Minimal Blueprint JSON
# ---------------------------------------------------------------------------

def _chapter_spec(position=1, title="Intro", min_tokens=6000, max_tokens=8000):
    return ChapterSpec(
        position=position,
        section_id=f"CH{position:02d}",
        title=title,
        purpose=f"Purpose of {title}",
        required=True,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        content_requirements=["Overview", "Examples"],
    )


def _book_spec(item_id="BK01", position=1, title="Book One", chapters=None):
    return BookSpec(
        item_id=item_id,
        position=position,
        title=title,
        subtitle="Subtitle",
        primary_outcome="Learn something",
        stage="foundation",
        dependencies=[],
        tags=["test"],
        chapters=chapters or [_chapter_spec()],
    )


def _blueprint_spec(books=None):
    return BlueprintSpec(
        blueprint_id="bp-test-001",
        library_id="lib-test",
        library_style="ebook_series",
        version="1.0.0",
        purpose="Test purpose",
        target_audience=["Developers", "Engineers"],
        primary_outcomes=["Learn testing"],
        scope_includes=["Unit tests"],
        scope_excludes=["E2E tests"],
        core_principles=[{"name": "Clarity"}],
        required_framing="Technical",
        tone_guidelines=["Professional", "Concise"],
        edition_strategy="internal_first",
        book_promise_template="You will learn {topic}.",
        artifact_types=["code_sample"],
        artifact_structure=["title", "content"],
        books=books or [_book_spec()],
    )


def _minimal_blueprint_json():
    """Return a minimal valid blueprint dict matching the parser format."""
    return {
        "blueprint_meta": {
            "blueprint_id": "bp-json-001",
            "library_id": "lib-json",
            "library_style": "ebook_series",
            "version": "1.0.0",
        },
        "executive_summary": {
            "purpose": "Test from JSON",
            "target_audience": ["Testers"],
            "primary_outcomes": ["Validate parser"],
            "scope_boundaries": {
                "includes": ["parsing"],
                "excludes": ["rendering"],
            },
        },
        "series_design_principles": {
            "core_principles": [{"name": "Simplicity"}],
            "required_framing": "Practical",
            "tone_guidelines": ["Friendly"],
            "edition_strategy": "internal_first",
        },
        "book_structure": {
            "book_promise": "You will master {topic}.",
            "sections": [
                {
                    "position": 1,
                    "section_id": "CH01",
                    "title": "Getting Started",
                    "purpose": "Introduction",
                    "required": True,
                    "target_token_range": {"min": 5000, "max": 7000},
                    "content_requirements": ["Overview"],
                },
                {
                    "position": 2,
                    "section_id": "CH02",
                    "title": "Deep Dive",
                    "purpose": "Core content",
                    "required": True,
                    "target_token_range": {"min": 6000, "max": 8000},
                    "content_requirements": ["Details", "Examples"],
                },
            ],
            "required_modules": [],
        },
        "catalog_baseline": {
            "items": [
                {
                    "item_id": "BK-01",
                    "position": 1,
                    "title": "First Book",
                    "subtitle": "The Beginning",
                    "primary_outcome": "Understand basics",
                    "stage": "foundation",
                    "dependencies": [],
                    "tags": ["beginner"],
                },
                {
                    "item_id": "BK-02",
                    "position": 2,
                    "title": "Second Book",
                    "subtitle": "Going Deeper",
                    "primary_outcome": "Master advanced topics",
                    "stage": "intermediate",
                    "dependencies": ["BK-01"],
                    "tags": ["advanced"],
                },
            ]
        },
        "value_stream": {"stages": []},
        "definition_of_done": {"criteria": []},
        "artifact_packs": {
            "per_book_artifacts": [],
            "marketing_artifacts": [],
            "sales_artifacts": [],
        },
    }


# ---------------------------------------------------------------------------
# ChapterSpec — Model Tests
# ---------------------------------------------------------------------------

class TestChapterSpec:
    def test_target_tokens(self):
        ch = _chapter_spec(min_tokens=6000, max_tokens=8000)
        assert ch.target_tokens == 7000

    def test_str_representation(self):
        ch = _chapter_spec(position=3, title="Advanced", min_tokens=4000, max_tokens=6000)
        s = str(ch)
        assert "Ch3" in s
        assert "Advanced" in s
        assert "4000" in s
        assert "6000" in s

    def test_required_default(self):
        ch = _chapter_spec()
        assert ch.required is True

    def test_content_requirements(self):
        ch = _chapter_spec()
        assert "Overview" in ch.content_requirements
        assert "Examples" in ch.content_requirements


# ---------------------------------------------------------------------------
# BookSpec — Model Tests
# ---------------------------------------------------------------------------

class TestBookSpec:
    def test_total_target_tokens(self):
        chapters = [
            _chapter_spec(position=1, min_tokens=4000, max_tokens=6000),
            _chapter_spec(position=2, min_tokens=6000, max_tokens=8000),
        ]
        book = _book_spec(chapters=chapters)
        # (4000+6000)/2 + (6000+8000)/2 = 5000 + 7000 = 12000
        assert book.total_target_tokens == 12000

    def test_str_representation(self):
        book = _book_spec(item_id="BK42", title="My Book")
        assert "BK42" in str(book)
        assert "My Book" in str(book)

    def test_empty_chapters_zero_tokens(self):
        book = BookSpec(
            item_id="BK-EMPTY",
            position=1,
            title="Empty",
            subtitle="No chapters",
            primary_outcome="Nothing",
            stage="foundation",
            dependencies=[],
            tags=[],
            chapters=[],
        )
        assert book.total_target_tokens == 0

    def test_dependencies(self):
        book = BookSpec(
            item_id="BK02",
            position=2,
            title="Sequel",
            subtitle="Part 2",
            primary_outcome="Continue",
            stage="intermediate",
            dependencies=["BK01"],
            tags=["series"],
        )
        assert "BK01" in book.dependencies


# ---------------------------------------------------------------------------
# BlueprintSpec — Model Tests
# ---------------------------------------------------------------------------

class TestBlueprintSpec:
    def test_get_book_by_id(self):
        bp = _blueprint_spec()
        book = bp.get_book("BK01")
        assert book is not None
        assert book.title == "Book One"

    def test_get_book_by_id_not_found(self):
        bp = _blueprint_spec()
        assert bp.get_book("NOPE") is None

    def test_get_book_by_position(self):
        bp = _blueprint_spec()
        book = bp.get_book_by_position(1)
        assert book is not None
        assert book.item_id == "BK01"

    def test_get_book_by_position_not_found(self):
        bp = _blueprint_spec()
        assert bp.get_book_by_position(999) is None

    def test_get_book_by_title_partial(self):
        bp = _blueprint_spec()
        book = bp.get_book_by_title("one")
        assert book is not None
        assert book.item_id == "BK01"

    def test_get_book_by_title_not_found(self):
        bp = _blueprint_spec()
        assert bp.get_book_by_title("nonexistent xyz") is None

    def test_audience_string(self):
        bp = _blueprint_spec()
        assert "Developers" in bp.audience_string

    def test_audience_string_empty(self):
        bp = _blueprint_spec()
        bp.target_audience = []
        assert bp.audience_string == "General readers"

    def test_tone_string(self):
        bp = _blueprint_spec()
        assert bp.tone_string == "Professional"

    def test_tone_string_empty(self):
        bp = _blueprint_spec()
        bp.tone_guidelines = []
        assert bp.tone_string == "Professional"

    def test_str_representation(self):
        bp = _blueprint_spec()
        s = str(bp)
        assert "bp-test-001" in s
        assert "lib-test" in s
        assert "1 books" in s


# ---------------------------------------------------------------------------
# BlueprintParser — Load from JSON dict
# ---------------------------------------------------------------------------

class TestBlueprintParserLoadJson:
    def setup_method(self):
        self.parser = BlueprintParser()

    def test_load_single_blueprint(self):
        data = _minimal_blueprint_json()
        result = self.parser.load_json(data)
        assert "lib-json" in result
        bp = result["lib-json"]
        assert bp.blueprint_id == "bp-json-001"
        assert bp.purpose == "Test from JSON"

    def test_load_blueprint_array(self):
        data = [_minimal_blueprint_json()]
        result = self.parser.load_json(data)
        assert len(result) == 1

    def test_load_json_string(self):
        data_str = json.dumps(_minimal_blueprint_json())
        result = self.parser.load_json(data_str)
        assert "lib-json" in result

    def test_parsed_books(self):
        data = _minimal_blueprint_json()
        result = self.parser.load_json(data)
        bp = result["lib-json"]
        assert len(bp.books) == 2
        assert bp.books[0].item_id == "BK-01"
        assert bp.books[1].item_id == "BK-02"

    def test_parsed_chapters_from_sections(self):
        data = _minimal_blueprint_json()
        result = self.parser.load_json(data)
        bp = result["lib-json"]
        # Global chapters applied to each book
        book = bp.books[0]
        assert len(book.chapters) == 2
        assert book.chapters[0].title == "Getting Started"
        assert book.chapters[1].title == "Deep Dive"

    def test_chapter_token_ranges(self):
        data = _minimal_blueprint_json()
        result = self.parser.load_json(data)
        bp = result["lib-json"]
        ch = bp.books[0].chapters[0]
        assert ch.min_tokens == 5000
        assert ch.max_tokens == 7000
        assert ch.target_tokens == 6000


# ---------------------------------------------------------------------------
# BlueprintParser — Load from File
# ---------------------------------------------------------------------------

class TestBlueprintParserLoadFile:
    def test_load_file(self, tmp_path):
        data = _minimal_blueprint_json()
        filepath = tmp_path / "test_bp.json"
        filepath.write_text(json.dumps(data), encoding="utf-8")

        parser = BlueprintParser()
        result = parser.load_file(filepath)
        assert "lib-json" in result

    def test_load_nonexistent_file_raises(self):
        parser = BlueprintParser()
        with pytest.raises(FileNotFoundError):
            parser.load_file("/nonexistent/path.json")

    def test_load_directory(self, tmp_path):
        # Write two blueprint files
        for i in range(2):
            data = _minimal_blueprint_json()
            data["blueprint_meta"]["library_id"] = f"lib-dir-{i}"
            filepath = tmp_path / f"bp_{i}.json"
            filepath.write_text(json.dumps(data), encoding="utf-8")

        parser = BlueprintParser()
        result = parser.load_directory(tmp_path)
        assert len(result) == 2

    def test_load_nonexistent_directory_raises(self):
        parser = BlueprintParser()
        with pytest.raises(FileNotFoundError):
            parser.load_directory("/nonexistent/dir")


# ---------------------------------------------------------------------------
# BlueprintParser — Convenience Methods
# ---------------------------------------------------------------------------

class TestBlueprintParserConvenience:
    def setup_method(self):
        self.parser = BlueprintParser()
        self.parser.load_json(_minimal_blueprint_json())

    def test_get_blueprint_by_library_id(self):
        bp = self.parser.get_blueprint_by_library("lib-json")
        assert bp is not None
        assert bp.library_id == "lib-json"

    def test_get_blueprint_by_blueprint_id(self):
        bp = self.parser.get_blueprint("bp-json-001")
        assert bp is not None

    def test_get_blueprint_not_found(self):
        assert self.parser.get_blueprint("nope") is None

    def test_list_blueprints(self):
        listing = self.parser.list_blueprints()
        assert len(listing) == 1
        bp_id, lib_id, book_count = listing[0]
        assert bp_id == "bp-json-001"
        assert lib_id == "lib-json"
        assert book_count == 2

    def test_list_books(self):
        books = self.parser.list_books()
        assert len(books) == 2
        assert books[0][1] == "BK-01"
        assert books[1][1] == "BK-02"

    def test_parser_len(self):
        assert len(self.parser) == 1

    def test_parser_iter(self):
        blueprints = list(self.parser)
        assert len(blueprints) == 1

    def test_parser_getitem(self):
        bp = self.parser["lib-json"]
        assert bp.blueprint_id == "bp-json-001"

    def test_parser_getitem_missing_raises(self):
        with pytest.raises(KeyError):
            _ = self.parser["nope"]


# ---------------------------------------------------------------------------
# BlueprintParser — Inline Chapters (Generated Format)
# ---------------------------------------------------------------------------

class TestBlueprintParserInlineChapters:
    def test_inline_chapters_parsed(self):
        data = _minimal_blueprint_json()
        # Override catalog items with inline chapters
        data["catalog_baseline"]["items"] = [
            {
                "item_id": "BK-INLINE",
                "position": 1,
                "title": "Inline Book",
                "subtitle": "With chapters",
                "primary_outcome": "Test inline",
                "stage": "foundation",
                "dependencies": [],
                "tags": [],
                "chapters": [
                    {
                        "chapter_number": 1,
                        "title": "Chapter Inline 1",
                        "purpose": "First inline",
                        "min_tokens": 3000,
                        "max_tokens": 5000,
                        "key_topics": ["Topic A"],
                    },
                    {
                        "chapter_number": 2,
                        "title": "Chapter Inline 2",
                        "purpose": "Second inline",
                        "min_tokens": 4000,
                        "max_tokens": 6000,
                        "key_topics": ["Topic B", "Topic C"],
                    },
                ],
            }
        ]
        # Remove global sections to ensure inline takes precedence
        data["book_structure"]["sections"] = []

        parser = BlueprintParser()
        result = parser.load_json(data)
        bp = result["lib-json"]
        book = bp.books[0]
        assert len(book.chapters) == 2
        assert book.chapters[0].title == "Chapter Inline 1"
        assert book.chapters[0].min_tokens == 3000
        assert book.chapters[1].content_requirements == ["Topic B", "Topic C"]


# ---------------------------------------------------------------------------
# BlueprintSpec — Validation Edge Cases
# ---------------------------------------------------------------------------

class TestBlueprintEdgeCases:
    def test_blueprint_no_books(self):
        bp = BlueprintSpec(
            blueprint_id="bp-empty",
            library_id="lib-empty",
            library_style="ebook_series",
            version="1.0.0",
            purpose="Empty",
            target_audience=[],
            primary_outcomes=[],
            scope_includes=[],
            scope_excludes=[],
            core_principles=[],
            required_framing="",
            tone_guidelines=[],
            edition_strategy="internal_first",
            book_promise_template="",
            artifact_types=[],
            artifact_structure=[],
            books=[],
        )
        assert len(bp.books) == 0
        assert bp.get_book("anything") is None

    def test_empty_catalog_parsed(self):
        data = _minimal_blueprint_json()
        data["catalog_baseline"]["items"] = []
        parser = BlueprintParser()
        result = parser.load_json(data)
        bp = result["lib-json"]
        assert len(bp.books) == 0

    def test_missing_optional_fields_use_defaults(self):
        data = {
            "blueprint_meta": {},
            "executive_summary": {},
            "series_design_principles": {},
            "book_structure": {},
            "catalog_baseline": {},
        }
        parser = BlueprintParser()
        result = parser.load_json(data)
        # Should not raise — defaults fill in
        assert len(result) == 1
        bp = list(result.values())[0]
        assert bp.blueprint_id == ""
        assert bp.purpose == ""
        assert bp.books == []


# ---------------------------------------------------------------------------
# ChapterOutput / BookOutput — Pipeline Output Models
# ---------------------------------------------------------------------------

class TestPipelineOutputModels:
    def test_chapter_output_to_dict(self):
        ch_spec = _chapter_spec()
        out = ChapterOutput(
            chapter_spec=ch_spec,
            content="Chapter content here.",
            token_count=150,
            input_tokens=50,
            output_tokens=100,
            duration_seconds=2.5,
        )
        d = out.to_dict()
        assert d["position"] == 1
        assert d["title"] == "Intro"
        assert d["content"] == "Chapter content here."
        assert d["token_count"] == 150
        assert d["success"] is True
        assert d["error"] is None

    def test_chapter_output_error(self):
        out = ChapterOutput(
            chapter_spec=_chapter_spec(),
            content="",
            success=False,
            error="LLM timeout",
        )
        d = out.to_dict()
        assert d["success"] is False
        assert d["error"] == "LLM timeout"

    def test_book_output_success(self):
        bp = _blueprint_spec()
        book = bp.books[0]
        ch_out = ChapterOutput(
            chapter_spec=book.chapters[0],
            content="Content",
            token_count=100,
        )
        book_out = BookOutput(
            book_spec=book,
            blueprint_spec=bp,
            chapters=[ch_out],
            total_duration_seconds=5.0,
            provider="ollama",
            model="llama3",
        )
        assert book_out.success is True
        assert book_out.total_tokens == 100
        assert book_out.successful_chapters == 1

    def test_book_output_partial_failure(self):
        bp = _blueprint_spec()
        book = bp.books[0]
        ok = ChapterOutput(chapter_spec=_chapter_spec(1), content="OK", token_count=50)
        fail = ChapterOutput(
            chapter_spec=_chapter_spec(2), content="", success=False, error="err"
        )
        book_out = BookOutput(
            book_spec=book,
            blueprint_spec=bp,
            chapters=[ok, fail],
        )
        assert book_out.success is False
        assert book_out.successful_chapters == 1
        assert book_out.total_tokens == 50
