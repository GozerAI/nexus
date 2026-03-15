"""
Offline RAG with Local Vector Store (Item 760)

Provides a fully offline Retrieval-Augmented Generation pipeline using
a local vector store (numpy-based) and local embedding models. No
external API calls are needed once documents are indexed.
"""

import logging
import time
import json
import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document in the local vector store."""

    doc_id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    source: str = ""
    indexed_at: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    """A search result from the vector store."""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class RAGResponse:
    """Response from the offline RAG pipeline."""

    query: str
    context_documents: List[SearchResult]
    augmented_prompt: str
    total_context_tokens: int
    retrieval_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LocalVectorStore:
    """
    Numpy-based vector store that works entirely offline.
    No external database dependencies.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._documents: Dict[str, Document] = {}
        self._embeddings: List[List[float]] = []
        self._doc_ids: List[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def add(self, doc: Document) -> None:
        """Add a document with its embedding to the store."""
        if doc.embedding is None:
            raise ValueError(f"Document {doc.doc_id} has no embedding")
        if len(doc.embedding) != self._dimension:
            raise ValueError(
                f"Embedding dimension {len(doc.embedding)} != store dimension {self._dimension}"
            )

        if doc.doc_id in self._documents:
            # Update existing
            idx = self._doc_ids.index(doc.doc_id)
            self._embeddings[idx] = doc.embedding
            self._documents[doc.doc_id] = doc
        else:
            self._documents[doc.doc_id] = doc
            self._embeddings.append(doc.embedding)
            self._doc_ids.append(doc.doc_id)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Search for nearest neighbors using cosine similarity.
        Returns list of (doc_id, score) tuples.
        """
        if not self._embeddings:
            return []

        if len(query_embedding) != self._dimension:
            raise ValueError(
                f"Query dimension {len(query_embedding)} != store dimension {self._dimension}"
            )

        scores = []
        for i, emb in enumerate(self._embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            scores.append((self._doc_ids[i], sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self._documents.get(doc_id)

    def remove(self, doc_id: str) -> bool:
        if doc_id not in self._documents:
            return False
        idx = self._doc_ids.index(doc_id)
        del self._documents[doc_id]
        del self._embeddings[idx]
        del self._doc_ids[idx]
        return True

    def save(self, path: str) -> None:
        """Save the vector store to disk."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        emb_data = {
            "dimension": self._dimension,
            "doc_ids": self._doc_ids,
            "embeddings": self._embeddings,
        }
        (p / "embeddings.json").write_text(json.dumps(emb_data))

        # Save documents
        docs_data = {}
        for doc_id, doc in self._documents.items():
            docs_data[doc_id] = {
                "content": doc.content,
                "metadata": doc.metadata,
                "chunk_index": doc.chunk_index,
                "source": doc.source,
                "indexed_at": doc.indexed_at,
            }
        (p / "documents.json").write_text(json.dumps(docs_data))
        logger.info("Saved vector store with %d documents to %s", self.size, path)

    def load(self, path: str) -> None:
        """Load the vector store from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Vector store not found at {path}")

        emb_data = json.loads((p / "embeddings.json").read_text())
        self._dimension = emb_data["dimension"]
        self._doc_ids = emb_data["doc_ids"]
        self._embeddings = emb_data["embeddings"]

        docs_data = json.loads((p / "documents.json").read_text())
        self._documents = {}
        for doc_id, ddata in docs_data.items():
            idx = self._doc_ids.index(doc_id) if doc_id in self._doc_ids else -1
            emb = self._embeddings[idx] if idx >= 0 else None
            self._documents[doc_id] = Document(
                doc_id=doc_id,
                content=ddata["content"],
                embedding=emb,
                metadata=ddata.get("metadata", {}),
                chunk_index=ddata.get("chunk_index", 0),
                source=ddata.get("source", ""),
                indexed_at=ddata.get("indexed_at", 0.0),
            )
        logger.info("Loaded vector store with %d documents from %s", self.size, path)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


class OfflineRAG:
    """
    Offline Retrieval-Augmented Generation pipeline.

    Uses a local vector store and local embedding functions to provide
    RAG capabilities without any network calls.

    Usage:
        rag = OfflineRAG(embed_fn=my_local_embedder)
        rag.index_document("doc1", "The capital of France is Paris.")
        response = rag.query("What is the capital of France?")
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 64

    def __init__(
        self,
        embed_fn: Optional[Any] = None,
        dimension: int = 384,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        store_path: Optional[str] = None,
    ):
        """
        Args:
            embed_fn: Callable that takes a string and returns List[float].
                      If None, a simple bag-of-words fallback is used.
            dimension: Embedding dimension (must match embed_fn output).
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            store_path: Path to persist/load the vector store.
        """
        self._embed_fn = embed_fn or self._fallback_embed
        self._dimension = dimension
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._store = LocalVectorStore(dimension=dimension)
        self._store_path = store_path

        if store_path:
            p = Path(store_path)
            if (p / "embeddings.json").exists():
                try:
                    self._store.load(store_path)
                except Exception as e:
                    logger.warning("Failed to load store from %s: %s", store_path, e)

    # ── Indexing ────────────────────────────────────────────────────

    def index_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        source: str = "",
    ) -> int:
        """
        Index a document by chunking and embedding it.
        Returns the number of chunks indexed.
        """
        chunks = self._chunk_text(content)
        indexed = 0

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self._embed_fn(chunk)

            # Ensure dimension matches
            if len(embedding) != self._dimension:
                embedding = self._normalize_dimension(embedding)

            doc = Document(
                doc_id=chunk_id,
                content=chunk,
                embedding=embedding,
                metadata=metadata or {},
                chunk_index=i,
                source=source,
            )
            self._store.add(doc)
            indexed += 1

        logger.info("Indexed %d chunks for document %s", indexed, doc_id)
        return indexed

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Batch index multiple documents.
        Each dict must have 'doc_id' and 'content', optionally 'metadata' and 'source'.
        """
        total = 0
        for doc in documents:
            count = self.index_document(
                doc_id=doc["doc_id"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                source=doc.get("source", ""),
            )
            total += count
        return total

    # ── Query ───────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        min_score: float = 0.0,
        context_template: Optional[str] = None,
    ) -> RAGResponse:
        """
        Query the offline RAG pipeline.

        Returns a RAGResponse with retrieved context documents and
        an augmented prompt ready for model inference.
        """
        start = time.time()

        query_embedding = self._embed_fn(query_text)
        if len(query_embedding) != self._dimension:
            query_embedding = self._normalize_dimension(query_embedding)

        raw_results = self._store.search(query_embedding, top_k=top_k)

        results = []
        for doc_id, score in raw_results:
            if score < min_score:
                continue
            doc = self._store.get_document(doc_id)
            if doc:
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        content=doc.content,
                        score=score,
                        metadata=doc.metadata,
                        source=doc.source,
                    )
                )

        retrieval_ms = (time.time() - start) * 1000

        # Build augmented prompt
        if context_template:
            context_text = "\n\n".join(r.content for r in results)
            augmented = context_template.format(
                context=context_text, query=query_text
            )
        else:
            augmented = self._build_default_prompt(query_text, results)

        total_tokens = len(augmented) // 4  # rough estimate

        return RAGResponse(
            query=query_text,
            context_documents=results,
            augmented_prompt=augmented,
            total_context_tokens=total_tokens,
            retrieval_time_ms=retrieval_ms,
        )

    def save(self) -> None:
        """Save the vector store to disk."""
        if self._store_path:
            self._store.save(self._store_path)

    # ── Internal ────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self._chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self._chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _normalize_dimension(self, embedding: List[float]) -> List[float]:
        """Pad or truncate embedding to match store dimension."""
        if len(embedding) >= self._dimension:
            return embedding[: self._dimension]
        return embedding + [0.0] * (self._dimension - len(embedding))

    def _fallback_embed(self, text: str) -> List[float]:
        """
        Simple bag-of-words embedding for offline fallback.
        Produces a fixed-dimension vector using hash-based feature hashing.
        """
        embedding = [0.0] * self._dimension
        words = text.lower().split()
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big")
            idx = bucket % self._dimension
            sign = 1 if digest[8] % 2 == 0 else -1
            embedding[idx] += sign * 1.0

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _build_default_prompt(
        self, query: str, results: List[SearchResult]
    ) -> str:
        parts = ["Use the following context to answer the question.\n"]
        parts.append("Context:")
        for i, r in enumerate(results, 1):
            parts.append(f"\n[{i}] {r.content}")
        parts.append(f"\n\nQuestion: {query}")
        parts.append("\nAnswer:")
        return "\n".join(parts)

    @property
    def document_count(self) -> int:
        return self._store.size

    def get_stats(self) -> Dict[str, Any]:
        return {
            "document_count": self._store.size,
            "dimension": self._dimension,
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
            "store_path": str(self._store_path) if self._store_path else None,
        }
