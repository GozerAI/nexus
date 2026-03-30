"""
Qdrant Vector Store Adapter for Nexus.

Provides semantic search over the knowledge base using Qdrant as the
vector database. Falls back gracefully to the existing keyword-based
search when Qdrant is unavailable.

Usage:
    store = QdrantVectorStore()
    await store.initialize()

    # Index a knowledge item
    await store.upsert("item_id", "The capital of France is Paris", {"type": "factual"})

    # Semantic search
    results = await store.search("What is the capital of France?", top_k=5)
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Qdrant-backed vector store with optional embedding via Ollama or sentence-transformers."""

    COLLECTION_NAME = "nexus_knowledge"
    VECTOR_SIZE = 384  # Default for all-MiniLM-L6-v2

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: int = 384,
    ):
        self._url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self._collection = collection_name or self.COLLECTION_NAME
        self._vector_size = vector_size
        self._client = None
        self._embedder = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Connect to Qdrant and ensure collection exists.

        Returns True if Qdrant is available and ready.
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(url=self._url, timeout=10)

            # Check connection
            self._client.get_collections()

            # Create collection if needed
            collections = [c.name for c in self._client.get_collections().collections]
            if self._collection not in collections:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("Created Qdrant collection: %s", self._collection)

            # Initialize embedder
            self._embedder = self._create_embedder()

            self._initialized = True
            logger.info("Qdrant vector store initialized (url=%s, collection=%s)", self._url, self._collection)
            return True

        except ImportError:
            logger.debug("qdrant-client not installed — vector search disabled")
            return False
        except Exception as e:
            logger.warning("Qdrant connection failed: %s — vector search disabled", e)
            return False

    def _create_embedder(self):
        """Create an embedding function. Tries sentence-transformers first, then simple hash."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Using sentence-transformers for embeddings")
            return model.encode
        except ImportError:
            pass

        # Fallback: simple hash-based pseudo-embeddings (not great but functional)
        logger.info("Using hash-based pseudo-embeddings (install sentence-transformers for better results)")
        return self._hash_embed

    def _hash_embed(self, texts, **kwargs):
        """Deterministic pseudo-embedding via hashing. Not semantic, but functional for testing."""
        import numpy as np
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            h = hashlib.sha256(text.lower().encode()).digest()
            vec = np.frombuffer(h * (self._vector_size // 32 + 1), dtype=np.float32)[:self._vector_size]
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(vec.tolist())
        return embeddings if len(embeddings) > 1 else embeddings[0]

    @property
    def is_available(self) -> bool:
        return self._initialized and self._client is not None

    async def upsert(
        self,
        item_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Index a knowledge item."""
        if not self.is_available:
            return False

        try:
            from qdrant_client.models import PointStruct

            vector = self._embedder(content)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()

            point = PointStruct(
                id=self._stable_id(item_id),
                vector=vector,
                payload={
                    "item_id": item_id,
                    "content": content[:10000],  # Truncate for storage
                    **(metadata or {}),
                },
            )

            self._client.upsert(
                collection_name=self._collection,
                points=[point],
            )
            return True

        except Exception as e:
            logger.debug("Qdrant upsert failed: %s", e)
            return False

    async def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.3,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search for similar knowledge items."""
        if not self.is_available:
            return []

        try:
            vector = self._embedder(query)
            if hasattr(vector, "tolist"):
                vector = vector.tolist()

            # Build filter if provided
            qdrant_filter = None
            if filters:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                conditions = []
                for key, value in filters.items():
                    conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
                qdrant_filter = Filter(must=conditions)

            results = self._client.search(
                collection_name=self._collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=min_score,
                query_filter=qdrant_filter,
            )

            return [
                {
                    "item_id": hit.payload.get("item_id", ""),
                    "content": hit.payload.get("content", ""),
                    "score": hit.score,
                    "metadata": {k: v for k, v in hit.payload.items() if k not in ("item_id", "content")},
                }
                for hit in results
            ]

        except Exception as e:
            logger.debug("Qdrant search failed: %s", e)
            return []

    async def delete(self, item_id: str) -> bool:
        """Delete a knowledge item from the vector store."""
        if not self.is_available:
            return False

        try:
            from qdrant_client.models import PointIdsList
            self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=[self._stable_id(item_id)]),
            )
            return True
        except Exception as e:
            logger.debug("Qdrant delete failed: %s", e)
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.is_available:
            return {"available": False}

        try:
            info = self._client.get_collection(self._collection)
            return {
                "available": True,
                "collection": self._collection,
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "status": info.status.value if info.status else "unknown",
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    @staticmethod
    def _stable_id(item_id: str) -> int:
        """Convert string ID to stable numeric ID for Qdrant."""
        return int(hashlib.md5(item_id.encode()).hexdigest()[:15], 16)
