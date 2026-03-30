"""Vector storage with ChromaDB or SQLite fallback."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from nexus.core.exceptions import StorageError, NotFoundError

# Try to import chromadb, use fallback if not available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

import numpy as np


@dataclass
class VectorChunk:
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SearchResult:
    chunk: VectorChunk
    score: float

    @property
    def similarity(self) -> float:
        return 1.0 / (1.0 + self.score)


class SQLiteVectorStore:
    """SQLite + NumPy based vector storage (fallback when ChromaDB unavailable)."""

    def __init__(self, persist_directory: str, collection_name: str = "nexus_memory"):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._data_file = self.persist_directory / f"{collection_name}.json"
        self._embeddings_file = self.persist_directory / f"{collection_name}_embeddings.npy"
        self._chunks: Dict[str, VectorChunk] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

    def initialize(self):
        """Initialize SQLite vector store."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        """Load data from disk."""
        if self._data_file.exists():
            with open(self._data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    chunk = VectorChunk(
                        id=item['id'],
                        text=item['text'],
                        metadata=item.get('metadata', {}),
                        created_at=datetime.fromisoformat(item.get('created_at', lambda: datetime.now(timezone.utc)().isoformat()))
                    )
                    self._chunks[chunk.id] = chunk

        if self._embeddings_file.exists():
            embeddings_data = np.load(self._embeddings_file, allow_pickle=True).item()
            self._embeddings = {k: np.array(v) for k, v in embeddings_data.items()}

    def _save(self):
        """Save data to disk."""
        data = []
        for chunk in self._chunks.values():
            data.append({
                'id': chunk.id,
                'text': chunk.text,
                'metadata': chunk.metadata,
                'created_at': chunk.created_at.isoformat() if isinstance(chunk.created_at, datetime) else chunk.created_at
            })

        with open(self._data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if self._embeddings:
            np.save(self._embeddings_file, self._embeddings)

    def add(self, chunks: List[VectorChunk]) -> List[str]:
        """Add chunks to vector store."""
        if not chunks:
            return []

        ids = []
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            if chunk.embedding:
                self._embeddings[chunk.id] = np.array(chunk.embedding)
            ids.append(chunk.id)

        self._save()
        return ids

    def search(self, query_embedding: List[float], n_results: int = 10,
               where: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar chunks using cosine distance."""
        if not self._embeddings:
            return []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        results = []
        for chunk_id, embedding in self._embeddings.items():
            chunk = self._chunks.get(chunk_id)
            if not chunk:
                continue

            # Apply metadata filter
            if where:
                match = True
                for key, value in where.items():
                    if chunk.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Compute cosine distance (L2 distance for normalized vectors approximates cosine)
            emb_norm = np.linalg.norm(embedding)
            if emb_norm == 0:
                continue

            # Cosine similarity -> distance
            similarity = np.dot(query_vec, embedding) / (query_norm * emb_norm)
            distance = 1.0 - similarity  # Convert to distance

            results.append(SearchResult(chunk=chunk, score=distance))

        # Sort by distance (ascending) and return top n
        results.sort(key=lambda x: x.score)
        return results[:n_results]

    def get(self, chunk_id: str) -> VectorChunk:
        """Get chunk by ID."""
        chunk = self._chunks.get(chunk_id)
        if not chunk:
            raise NotFoundError(f"Chunk not found: {chunk_id}")
        return chunk

    def delete(self, chunk_ids: List[str]) -> int:
        """Delete chunks by ID."""
        deleted = 0
        for chunk_id in chunk_ids:
            if chunk_id in self._chunks:
                del self._chunks[chunk_id]
                deleted += 1
            if chunk_id in self._embeddings:
                del self._embeddings[chunk_id]

        self._save()
        return deleted

    def count(self) -> int:
        """Get total chunks."""
        return len(self._chunks)

    def list_all(self, where: Optional[Dict] = None, limit: int = 1000) -> List[VectorChunk]:
        """List all chunks matching criteria."""
        chunks = []
        for chunk in self._chunks.values():
            if where:
                match = True
                for key, value in where.items():
                    if chunk.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            chunks.append(chunk)
            if len(chunks) >= limit:
                break
        return chunks


class ChromaDBVectorStore:
    """ChromaDB-backed vector storage."""

    def __init__(self, persist_directory: str, collection_name: str = "nexus_memory"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def initialize(self):
        """Initialize ChromaDB."""
        self._client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"}
        )

    def add(self, chunks: List[VectorChunk]) -> List[str]:
        """Add chunks to vector store."""
        if not chunks:
            return []

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        embeddings = [c.embedding for c in chunks if c.embedding]
        metadatas = [{**c.metadata, "created_at": c.created_at.isoformat()} for c in chunks]

        if embeddings and len(embeddings) == len(chunks):
            self._collection.add(ids=ids, documents=documents,
                                 embeddings=embeddings, metadatas=metadatas)
        else:
            self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

        return ids

    def search(self, query_embedding: List[float], n_results: int = 10,
               where: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar chunks."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            chunk = VectorChunk(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            )
            search_results.append(SearchResult(chunk=chunk, score=results["distances"][0][i]))
        return search_results

    def get(self, chunk_id: str) -> VectorChunk:
        """Get chunk by ID."""
        results = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not results["ids"]:
            raise NotFoundError(f"Chunk not found: {chunk_id}")
        return VectorChunk(id=results["ids"][0], text=results["documents"][0],
                          metadata=results["metadatas"][0])

    def delete(self, chunk_ids: List[str]) -> int:
        """Delete chunks by ID."""
        self._collection.delete(ids=chunk_ids)
        return len(chunk_ids)

    def count(self) -> int:
        """Get total chunks."""
        return self._collection.count()

    def list_all(self, where: Optional[Dict] = None, limit: int = 1000) -> List[VectorChunk]:
        """List all chunks matching criteria."""
        results = self._collection.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"]
        )
        chunks = []
        for i in range(len(results["ids"])):
            chunks.append(VectorChunk(
                id=results["ids"][i],
                text=results["documents"][i],
                metadata=results["metadatas"][i]
            ))
        return chunks


def VectorStore(persist_directory: str, collection_name: str = "nexus_memory",
                backend: str = "auto", redis_config: dict = None):
    """Factory function to create appropriate vector store.

    Args:
        persist_directory: Path for local storage (SQLite/ChromaDB)
        collection_name: Name of the collection/index
        backend: "auto", "redis", "chromadb", or "sqlite"
        redis_config: Redis configuration dict (host, port, password, cloud_url)

    Priority (when backend="auto"):
        1. Redis (if configured and available)
        2. ChromaDB (if available)
        3. SQLite+NumPy (fallback)
    """
    # Try Redis if configured
    if backend == "redis" or (backend == "auto" and redis_config):
        try:
            from nexus.storage.redis_store import RedisVectorStore, RedisConfig, REDIS_AVAILABLE
            if REDIS_AVAILABLE:
                config = RedisConfig(
                    host=redis_config.get("host", "localhost") if redis_config else "localhost",
                    port=redis_config.get("port", 6379) if redis_config else 6379,
                    password=redis_config.get("password") if redis_config else None,
                    cloud_url=redis_config.get("cloud_url") if redis_config else None,
                    index_name=collection_name
                )
                store = RedisVectorStore(config)
                print("Using Redis for vector storage")
                return store
        except ImportError:
            if backend == "redis":
                raise
            print("Redis not available, trying other backends...")

    # Try ChromaDB
    if backend in ("auto", "chromadb") and CHROMADB_AVAILABLE:
        print("Using ChromaDB for vector storage")
        return ChromaDBVectorStore(persist_directory, collection_name)

    # Fallback to SQLite
    if backend in ("auto", "sqlite") or not CHROMADB_AVAILABLE:
        print("Using SQLite fallback for vector storage")
        return SQLiteVectorStore(persist_directory, collection_name)

    raise StorageError(f"No vector store backend available for: {backend}")
