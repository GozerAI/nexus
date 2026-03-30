"""
MVP RAG System - Production-ready Retrieval Augmented Generation.

This is a real, working RAG implementation that:
- Ingests documents with smart chunking
- Creates embeddings (local or API)
- Stores in FAISS for fast retrieval
- Retrieves relevant context for queries
"""

import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document to be indexed."""

    content: str
    doc_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def __post_init__(self):
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()[:16]


@dataclass
class RetrievalResult:
    """Result from a retrieval query."""

    query: str
    chunks: List[Dict[str, Any]]
    total_tokens_estimate: int
    retrieval_time_ms: float


@dataclass
class RAGConfig:
    """Configuration for the RAG system."""

    # Embedding settings
    embedding_provider: str = "sentence-transformers"  # or "ollama", "openai"
    embedding_model: Optional[str] = None  # Uses provider default if None

    # Chunking settings
    chunk_strategy: str = "recursive"  # or "fixed", "sentence", "markdown"
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Vector store settings
    index_type: str = "flat"  # or "hnsw" for faster search with more vectors
    storage_path: Optional[str] = None  # For persistence

    # Retrieval settings
    default_top_k: int = 5
    max_context_tokens: int = 4000  # Max tokens to return


class MVPRAG:
    """
    Minimal Viable RAG System.

    A production-ready RAG implementation that actually works.

    Usage:
        rag = MVPRAG()
        rag.initialize()

        # Add documents
        rag.add_document("Python is a programming language...")
        rag.add_documents([doc1, doc2, doc3])

        # Query
        results = rag.query("What is Python?", top_k=3)
        context = rag.get_context_for_prompt("What is Python?")
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._embedder = None
        self._chunker = None
        self._store = None
        self._initialized = False

        self._stats = {
            "documents_added": 0,
            "chunks_indexed": 0,
            "queries_processed": 0,
        }

    def initialize(self) -> None:
        """Initialize all components."""
        if self._initialized:
            return

        logger.info("Initializing MVP RAG system...")

        # Initialize embedding model
        from .embeddings import get_embedding_model
        self._embedder = get_embedding_model(
            provider=self.config.embedding_provider,
            model_name=self.config.embedding_model,
        )
        logger.info(f"Embedding model: {self.config.embedding_provider} (dim={self._embedder.dimension})")

        # Initialize chunker
        from .chunking import get_chunker
        self._chunker = get_chunker(
            strategy=self.config.chunk_strategy,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        logger.info(f"Chunker: {self.config.chunk_strategy}")

        # Initialize vector store
        from .vector.faiss_store import FAISSVectorStore
        self._store = FAISSVectorStore(
            dimension=self._embedder.dimension,
            index_type=self.config.index_type,
            storage_path=self.config.storage_path,
        )
        self._store.initialize()
        logger.info(f"Vector store: FAISS ({self.config.index_type})")

        self._initialized = True
        logger.info("MVP RAG system initialized successfully")

    def add_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> int:
        """
        Add a single document to the index.

        Args:
            content: Document text content
            doc_id: Optional unique identifier
            metadata: Optional metadata dict
            source: Optional source identifier (file path, URL, etc.)

        Returns:
            Number of chunks indexed
        """
        if not self._initialized:
            self.initialize()

        doc = Document(
            content=content,
            doc_id=doc_id,
            metadata=metadata or {},
            source=source,
        )

        return self._index_document(doc)

    def add_documents(self, documents: List[Union[str, Document, Dict]]) -> int:
        """
        Add multiple documents to the index.

        Args:
            documents: List of strings, Document objects, or dicts with content/metadata

        Returns:
            Total number of chunks indexed
        """
        if not self._initialized:
            self.initialize()

        total_chunks = 0

        for doc in documents:
            if isinstance(doc, str):
                doc = Document(content=doc)
            elif isinstance(doc, dict):
                doc = Document(
                    content=doc["content"],
                    doc_id=doc.get("doc_id"),
                    metadata=doc.get("metadata", {}),
                    source=doc.get("source"),
                )

            total_chunks += self._index_document(doc)

        return total_chunks

    def add_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a file to the index.

        Args:
            file_path: Path to the file
            metadata: Optional additional metadata

        Returns:
            Number of chunks indexed
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")

        file_metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "extension": path.suffix,
            **(metadata or {}),
        }

        return self.add_document(
            content=content,
            doc_id=f"file:{path.name}",
            metadata=file_metadata,
            source=str(path),
        )

    def add_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> int:
        """
        Add all files from a directory.

        Args:
            dir_path: Path to directory
            extensions: File extensions to include (e.g., [".txt", ".md"])
            recursive: Whether to search subdirectories

        Returns:
            Total chunks indexed
        """
        path = Path(dir_path)

        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        pattern = "**/*" if recursive else "*"
        files = list(path.glob(pattern))

        total_chunks = 0

        for file_path in files:
            if not file_path.is_file():
                continue

            if extensions and file_path.suffix.lower() not in extensions:
                continue

            try:
                total_chunks += self.add_file(str(file_path))
            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        return total_chunks

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Query the RAG system.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            RetrievalResult with matched chunks
        """
        if not self._initialized:
            self.initialize()

        import time
        start = time.time()

        top_k = top_k or self.config.default_top_k

        # Embed query
        query_vector = self._embedder.embed(query)

        # Search
        results = self._store.search(
            query_vector=query_vector,
            k=top_k,
            filter_metadata=filter_metadata,
        )

        elapsed_ms = (time.time() - start) * 1000
        self._stats["queries_processed"] += 1

        # Estimate tokens (rough: 4 chars per token)
        total_chars = sum(len(r["text"]) for r in results)
        token_estimate = total_chars // 4

        return RetrievalResult(
            query=query,
            chunks=results,
            total_tokens_estimate=token_estimate,
            retrieval_time_ms=elapsed_ms,
        )

    def get_context_for_prompt(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        top_k: int = 10,
    ) -> str:
        """
        Get formatted context string for LLM prompt.

        Args:
            query: The user's question
            max_tokens: Maximum tokens in context
            top_k: Number of chunks to consider

        Returns:
            Formatted context string
        """
        max_tokens = max_tokens or self.config.max_context_tokens

        results = self.query(query, top_k=top_k)

        context_parts = []
        current_tokens = 0

        for chunk in results.chunks:
            chunk_tokens = len(chunk["text"]) // 4

            if current_tokens + chunk_tokens > max_tokens:
                break

            # Format chunk with source if available
            source = chunk["metadata"].get("source", chunk["metadata"].get("filename", ""))
            if source:
                context_parts.append(f"[Source: {source}]\n{chunk['text']}")
            else:
                context_parts.append(chunk["text"])

            current_tokens += chunk_tokens

        return "\n\n---\n\n".join(context_parts)

    def _index_document(self, doc: Document) -> int:
        """Index a single document."""
        # Chunk the document
        chunk_metadata = {
            "doc_id": doc.doc_id,
            "source": doc.source,
            **doc.metadata,
        }

        chunks = self._chunker.chunk(doc.content, metadata=chunk_metadata)

        if not chunks:
            return 0

        # Embed all chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_batch(chunk_texts)

        # Prepare batch for vector store
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            documents.append({
                "doc_id": f"{doc.doc_id}:chunk:{i}",
                "vector": embedding,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })

        # Add to store
        added = self._store.add_batch(documents)

        self._stats["documents_added"] += 1
        self._stats["chunks_indexed"] += added

        logger.debug(f"Indexed document {doc.doc_id}: {added} chunks")
        return added

    def save(self) -> None:
        """Save the index to disk."""
        if self._store and self.config.storage_path:
            self._store.save()
            logger.info(f"Saved index to {self.config.storage_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            **self._stats,
            "vector_count": self._store.count() if self._store else 0,
            "config": {
                "embedding_provider": self.config.embedding_provider,
                "chunk_strategy": self.config.chunk_strategy,
                "chunk_size": self.config.chunk_size,
                "index_type": self.config.index_type,
            },
        }

    def clear(self) -> None:
        """Clear all indexed data."""
        if self._store:
            self._store.clear()
        self._stats = {
            "documents_added": 0,
            "chunks_indexed": 0,
            "queries_processed": 0,
        }
        logger.info("Cleared all indexed data")


# Convenience function for quick setup
def create_rag(
    embedding_provider: str = "sentence-transformers",
    storage_path: Optional[str] = None,
    **kwargs,
) -> MVPRAG:
    """
    Create a RAG system with sensible defaults.

    Args:
        embedding_provider: "sentence-transformers" (local), "ollama", or "openai"
        storage_path: Optional path for persistence
        **kwargs: Additional config options

    Returns:
        Initialized MVPRAG instance
    """
    config = RAGConfig(
        embedding_provider=embedding_provider,
        storage_path=storage_path,
        **kwargs,
    )

    rag = MVPRAG(config)
    rag.initialize()

    return rag
