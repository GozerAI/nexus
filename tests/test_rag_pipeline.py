"""Tests for the RAG pipeline — chunking, MVPRAG, and document handling."""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.rag.mvp_rag import MVPRAG, RAGConfig, Document, RetrievalResult
from nexus.rag.chunking import (
    Chunk,
    FixedSizeChunker,
)


# ---------------------------------------------------------------------------
# Document Model
# ---------------------------------------------------------------------------

class TestDocument:
    def test_auto_generates_doc_id(self):
        doc = Document(content="Hello world")
        assert doc.doc_id is not None
        assert len(doc.doc_id) == 16

    def test_explicit_doc_id(self):
        doc = Document(content="Test", doc_id="my-doc")
        assert doc.doc_id == "my-doc"

    def test_default_metadata(self):
        doc = Document(content="Test")
        assert doc.metadata == {}

    def test_metadata_provided(self):
        doc = Document(content="Test", metadata={"lang": "en"})
        assert doc.metadata["lang"] == "en"

    def test_source_field(self):
        doc = Document(content="Test", source="/path/to/file.txt")
        assert doc.source == "/path/to/file.txt"

    def test_same_content_same_id(self):
        d1 = Document(content="identical")
        d2 = Document(content="identical")
        assert d1.doc_id == d2.doc_id

    def test_different_content_different_id(self):
        d1 = Document(content="alpha")
        d2 = Document(content="beta")
        assert d1.doc_id != d2.doc_id


# ---------------------------------------------------------------------------
# RetrievalResult Model
# ---------------------------------------------------------------------------

class TestRetrievalResult:
    def test_creation(self):
        result = RetrievalResult(
            query="What is Python?",
            chunks=[{"text": "Python is...", "metadata": {}, "score": 0.9}],
            total_tokens_estimate=25,
            retrieval_time_ms=1.5,
        )
        assert result.query == "What is Python?"
        assert len(result.chunks) == 1
        assert result.total_tokens_estimate == 25
        assert result.retrieval_time_ms == 1.5

    def test_empty_chunks(self):
        result = RetrievalResult(
            query="Nothing", chunks=[], total_tokens_estimate=0, retrieval_time_ms=0.1
        )
        assert result.chunks == []


# ---------------------------------------------------------------------------
# RAGConfig
# ---------------------------------------------------------------------------

class TestRAGConfig:
    def test_defaults(self):
        cfg = RAGConfig()
        assert cfg.embedding_provider == "sentence-transformers"
        assert cfg.chunk_strategy == "recursive"
        assert cfg.chunk_size == 512
        assert cfg.chunk_overlap == 50
        assert cfg.default_top_k == 5
        assert cfg.max_context_tokens == 4000

    def test_custom_config(self):
        cfg = RAGConfig(
            chunk_size=256,
            chunk_overlap=25,
            embedding_provider="openai",
            default_top_k=10,
        )
        assert cfg.chunk_size == 256
        assert cfg.chunk_overlap == 25
        assert cfg.embedding_provider == "openai"
        assert cfg.default_top_k == 10


# ---------------------------------------------------------------------------
# FixedSizeChunker — Chunk Creation
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_single_chunk_small_text(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("Hello world")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"

    def test_multiple_chunks(self):
        text = "word " * 200  # ~1000 chars
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text)
        assert len(chunks) > 1

    def test_chunk_overlap_works(self):
        text = "A" * 500
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # Overlap means second chunk starts before first chunk ends
        if len(chunks) >= 2:
            assert chunks[1].start_char < chunks[0].end_char

    def test_empty_text(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_chunk_metadata_propagated(self):
        chunker = FixedSizeChunker(chunk_size=1000, chunk_overlap=0)
        meta = {"source": "test.txt", "doc_id": "d1"}
        chunks = chunker.chunk("Some content here.", metadata=meta)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["doc_id"] == "d1"

    def test_chunk_indices_sequential(self):
        text = "word " * 500
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(text)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_positions_cover_text(self):
        text = "A sentence of words. " * 50
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(text)
        # First chunk starts at 0
        assert chunks[0].start_char == 0
        # Text content is non-empty for all chunks
        assert all(len(c.text) > 0 for c in chunks)


# ---------------------------------------------------------------------------
# Chunk Dataclass
# ---------------------------------------------------------------------------

class TestChunkDataclass:
    def test_chunk_fields(self):
        c = Chunk(text="Hello", start_char=0, end_char=5, chunk_index=0)
        assert c.text == "Hello"
        assert c.start_char == 0
        assert c.end_char == 5
        assert c.chunk_index == 0

    def test_chunk_default_metadata(self):
        c = Chunk(text="Test", start_char=0, end_char=4, chunk_index=0)
        assert c.metadata == {}

    def test_chunk_with_metadata(self):
        c = Chunk(text="T", start_char=0, end_char=1, chunk_index=0, metadata={"k": "v"})
        assert c.metadata["k"] == "v"


# ---------------------------------------------------------------------------
# MVPRAG — Initialization (Mocked)
# ---------------------------------------------------------------------------

class TestMVPRAGInit:
    def test_not_initialized_by_default(self):
        rag = MVPRAG()
        assert rag._initialized is False

    def test_config_stored(self):
        cfg = RAGConfig(chunk_size=128)
        rag = MVPRAG(config=cfg)
        assert rag.config.chunk_size == 128

    def test_default_config(self):
        rag = MVPRAG()
        assert rag.config.chunk_size == 512

    def test_initial_stats(self):
        rag = MVPRAG()
        stats = rag._stats
        assert stats["documents_added"] == 0
        assert stats["chunks_indexed"] == 0
        assert stats["queries_processed"] == 0


# ---------------------------------------------------------------------------
# MVPRAG — Document Ingestion (Mocked Components)
# ---------------------------------------------------------------------------

class TestMVPRAGIngestion:
    """Test document ingestion with mocked embedder and store."""

    def _make_initialized_rag(self):
        """Create a MVPRAG with mocked internal components."""
        rag = MVPRAG(config=RAGConfig(chunk_size=200, chunk_overlap=20))

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        mock_embedder.embed.return_value = [0.1] * 384
        mock_embedder.embed_batch.return_value = [[0.1] * 384]

        # Mock chunker — use a real FixedSizeChunker
        real_chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=20)

        # Mock store
        mock_store = MagicMock()
        mock_store.add_batch.return_value = 1
        mock_store.search.return_value = []
        mock_store.count.return_value = 0

        rag._embedder = mock_embedder
        rag._chunker = real_chunker
        rag._store = mock_store
        rag._initialized = True

        return rag

    def test_add_document_returns_chunk_count(self):
        rag = self._make_initialized_rag()
        count = rag.add_document("A short document for testing.")
        assert count >= 1

    def test_add_document_increments_stats(self):
        rag = self._make_initialized_rag()
        rag.add_document("Test content.")
        assert rag._stats["documents_added"] == 1

    def test_add_document_with_id_and_metadata(self):
        rag = self._make_initialized_rag()
        count = rag.add_document(
            "Content",
            doc_id="custom-id",
            metadata={"author": "test"},
            source="manual",
        )
        assert count >= 1

    def test_add_documents_multiple(self):
        rag = self._make_initialized_rag()
        count = rag.add_documents(["Doc one", "Doc two", "Doc three"])
        assert count >= 3
        assert rag._stats["documents_added"] == 3

    def test_add_documents_dict_format(self):
        rag = self._make_initialized_rag()
        docs = [
            {"content": "First", "doc_id": "d1", "metadata": {"type": "a"}},
            {"content": "Second", "doc_id": "d2"},
        ]
        count = rag.add_documents(docs)
        assert count >= 2

    def test_add_documents_document_objects(self):
        rag = self._make_initialized_rag()
        docs = [
            Document(content="Alpha", doc_id="a1"),
            Document(content="Beta", doc_id="b1"),
        ]
        count = rag.add_documents(docs)
        assert count >= 2

    def test_add_file(self, tmp_path):
        rag = self._make_initialized_rag()
        f = tmp_path / "sample.txt"
        f.write_text("This is file content for RAG.", encoding="utf-8")
        count = rag.add_file(str(f))
        assert count >= 1

    def test_add_file_not_found(self):
        rag = self._make_initialized_rag()
        with pytest.raises(FileNotFoundError):
            rag.add_file("/nonexistent/file.txt")

    def test_add_directory(self, tmp_path):
        rag = self._make_initialized_rag()
        for i in range(3):
            (tmp_path / f"doc_{i}.txt").write_text(f"Document {i} content.", encoding="utf-8")
        count = rag.add_directory(str(tmp_path), extensions=[".txt"])
        assert count >= 3

    def test_add_directory_not_a_dir(self, tmp_path):
        rag = self._make_initialized_rag()
        f = tmp_path / "not_a_dir.txt"
        f.write_text("nope")
        with pytest.raises(NotADirectoryError):
            rag.add_directory(str(f))


# ---------------------------------------------------------------------------
# MVPRAG — Query (Mocked Components)
# ---------------------------------------------------------------------------

class TestMVPRAGQuery:
    def _make_rag_with_results(self, results=None):
        rag = MVPRAG()
        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        mock_embedder.embed.return_value = [0.1] * 384

        mock_store = MagicMock()
        mock_store.search.return_value = results or []
        mock_store.count.return_value = len(results or [])

        rag._embedder = mock_embedder
        rag._chunker = FixedSizeChunker()
        rag._store = mock_store
        rag._initialized = True
        return rag

    def test_query_returns_retrieval_result(self):
        rag = self._make_rag_with_results([
            {"text": "Python is a language", "metadata": {}, "score": 0.95},
        ])
        result = rag.query("What is Python?")
        assert isinstance(result, RetrievalResult)
        assert result.query == "What is Python?"
        assert len(result.chunks) == 1

    def test_query_empty_corpus(self):
        rag = self._make_rag_with_results([])
        result = rag.query("Anything")
        assert result.chunks == []
        assert result.total_tokens_estimate == 0

    def test_query_increments_stats(self):
        rag = self._make_rag_with_results([])
        rag.query("Test")
        assert rag._stats["queries_processed"] == 1

    def test_query_with_top_k(self):
        rag = self._make_rag_with_results([
            {"text": "A", "metadata": {}, "score": 0.9},
            {"text": "B", "metadata": {}, "score": 0.8},
        ])
        result = rag.query("test", top_k=1)
        # Store mock returns all, but top_k was passed to search
        rag._store.search.assert_called_once()
        call_kwargs = rag._store.search.call_args
        assert call_kwargs.kwargs.get("k") == 1 or call_kwargs[1].get("k") == 1

    def test_query_with_metadata_filter(self):
        rag = self._make_rag_with_results([])
        rag.query("test", filter_metadata={"source": "wiki"})
        call_kwargs = rag._store.search.call_args
        filter_val = call_kwargs.kwargs.get("filter_metadata") or call_kwargs[1].get("filter_metadata")
        assert filter_val == {"source": "wiki"}

    def test_retrieval_time_positive(self):
        rag = self._make_rag_with_results([])
        result = rag.query("test")
        assert result.retrieval_time_ms >= 0

    def test_token_estimate_calculation(self):
        rag = self._make_rag_with_results([
            {"text": "A" * 100, "metadata": {}, "score": 0.9},
        ])
        result = rag.query("test")
        # 100 chars / 4 = 25 tokens
        assert result.total_tokens_estimate == 25


# ---------------------------------------------------------------------------
# MVPRAG — Context for Prompt
# ---------------------------------------------------------------------------

class TestMVPRAGContext:
    def _make_rag(self, chunks):
        rag = MVPRAG(config=RAGConfig(max_context_tokens=100))
        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        mock_embedder.embed.return_value = [0.1] * 384
        mock_store = MagicMock()
        mock_store.search.return_value = chunks
        mock_store.count.return_value = len(chunks)

        rag._embedder = mock_embedder
        rag._chunker = FixedSizeChunker()
        rag._store = mock_store
        rag._initialized = True
        return rag

    def test_get_context_for_prompt_basic(self):
        rag = self._make_rag([
            {"text": "Python is great.", "metadata": {}, "score": 0.9},
        ])
        ctx = rag.get_context_for_prompt("Python?")
        assert "Python is great." in ctx

    def test_get_context_with_source(self):
        rag = self._make_rag([
            {"text": "Data point.", "metadata": {"source": "wiki.txt"}, "score": 0.9},
        ])
        ctx = rag.get_context_for_prompt("data?")
        assert "[Source: wiki.txt]" in ctx

    def test_context_respects_max_tokens(self):
        # Each chunk ~100 chars = ~25 tokens. Max 100 tokens = ~4 chunks.
        chunks = [
            {"text": "X" * 100, "metadata": {}, "score": 0.9}
            for _ in range(10)
        ]
        rag = self._make_rag(chunks)
        ctx = rag.get_context_for_prompt("test", max_tokens=50)
        # Should not include all 10 chunks
        assert len(ctx) < 10 * 100


# ---------------------------------------------------------------------------
# MVPRAG — Stats & Clear
# ---------------------------------------------------------------------------

class TestMVPRAGStatsAndClear:
    def test_get_stats(self):
        rag = MVPRAG()
        mock_store = MagicMock()
        mock_store.count.return_value = 42
        rag._store = mock_store
        stats = rag.get_stats()
        assert stats["vector_count"] == 42
        assert "config" in stats

    def test_get_stats_no_store(self):
        rag = MVPRAG()
        stats = rag.get_stats()
        assert stats["vector_count"] == 0

    def test_clear_resets_stats(self):
        rag = MVPRAG()
        rag._store = MagicMock()
        rag._stats["documents_added"] = 5
        rag._stats["chunks_indexed"] = 20
        rag.clear()
        assert rag._stats["documents_added"] == 0
        assert rag._stats["chunks_indexed"] == 0
        assert rag._stats["queries_processed"] == 0
        rag._store.clear.assert_called_once()


# ---------------------------------------------------------------------------
# Large Document Handling
# ---------------------------------------------------------------------------

class TestLargeDocumentHandling:
    def test_large_text_chunked(self):
        text = "word " * 10000  # ~50k chars
        chunker = FixedSizeChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(text)
        assert len(chunks) > 50
        assert all(len(c.text) <= 600 for c in chunks)  # Allow some slack for word boundaries


# ---------------------------------------------------------------------------
# RAG Module Imports
# ---------------------------------------------------------------------------

class TestRAGModuleImports:
    """Verify key classes are importable from nexus.rag."""

    def test_mvprag_importable(self):
        from nexus.rag import MVPRAG
        assert MVPRAG is not None

    def test_rag_config_importable(self):
        from nexus.rag import RAGConfig
        assert RAGConfig is not None

    def test_document_importable(self):
        from nexus.rag import Document
        assert Document is not None

    def test_create_rag_importable(self):
        from nexus.rag import create_rag
        assert callable(create_rag)

    def test_chunker_classes_importable(self):
        from nexus.rag import FixedSizeChunker, RecursiveChunker, SentenceChunker
        assert FixedSizeChunker is not None
        assert RecursiveChunker is not None
        assert SentenceChunker is not None

    def test_embedding_model_importable(self):
        from nexus.rag import EmbeddingModel
        assert EmbeddingModel is not None

    def test_document_manager_importable(self):
        from nexus.rag import DocumentManager, DocumentStatus
        assert DocumentManager is not None
        assert DocumentStatus is not None
