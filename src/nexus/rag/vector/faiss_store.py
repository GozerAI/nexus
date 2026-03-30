"""
FAISS-based vector store for production RAG.

Provides efficient similarity search with:
- HNSW indexing for fast approximate nearest neighbor
- Persistence to disk
- Metadata storage
- Batch operations
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Production vector store using FAISS for efficient similarity search.

    Supports millions of vectors with sub-millisecond search times.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",  # "flat", "hnsw", "ivf"
        storage_path: Optional[str] = None,
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Vector dimension (must match embedding model)
            index_type: Type of index - "flat" (exact), "hnsw" (fast approximate), "ivf" (scalable)
            storage_path: Path for persistence (optional)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.storage_path = Path(storage_path) if storage_path else None

        self._faiss = None
        self._index = None
        self._id_map: Dict[int, str] = {}  # FAISS internal ID -> document ID
        self._metadata: Dict[str, Dict[str, Any]] = {}  # document ID -> metadata
        self._documents: Dict[str, str] = {}  # document ID -> original text
        self._next_id = 0

        self._initialized = False

    def _ensure_faiss(self):
        """Lazy load FAISS."""
        if self._faiss is None:
            try:
                import faiss
                self._faiss = faiss
                logger.info("FAISS loaded successfully")
            except ImportError:
                raise ImportError(
                    "FAISS not installed. Install with:\n"
                    "  pip install faiss-cpu\n"
                    "Or for GPU support:\n"
                    "  pip install faiss-gpu"
                )

    def initialize(self) -> None:
        """Initialize the FAISS index."""
        if self._initialized:
            return

        self._ensure_faiss()

        # Try to load existing index
        if self.storage_path and self._load_from_disk():
            self._initialized = True
            return

        # Create new index
        self._create_index()
        self._initialized = True
        logger.info(f"Initialized {self.index_type} FAISS index with dimension {self.dimension}")

    def _create_index(self):
        """Create a new FAISS index based on index_type."""
        if self.index_type == "flat":
            # Exact search - best for small datasets (<100k vectors)
            self._index = self._faiss.IndexFlatIP(self.dimension)  # Inner product (cosine with normalized vectors)

        elif self.index_type == "hnsw":
            # Hierarchical Navigable Small World - fast approximate search
            # Good for 100k-10M vectors
            self._index = self._faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
            self._index.hnsw.efConstruction = 200  # Build quality
            self._index.hnsw.efSearch = 64  # Search quality

        elif self.index_type == "ivf":
            # Inverted File Index - scalable for 1M+ vectors
            # Requires training on sample data
            quantizer = self._faiss.IndexFlatIP(self.dimension)
            self._index = self._faiss.IndexIVFFlat(quantizer, self.dimension, 100)  # 100 clusters
            self._index.nprobe = 10  # Search 10 clusters

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

    def add(
        self,
        doc_id: str,
        vector: List[float],
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a single document to the store.

        Args:
            doc_id: Unique document identifier
            vector: Embedding vector (will be normalized)
            text: Original text content
            metadata: Optional metadata dict
        """
        if not self._initialized:
            self.initialize()

        # Normalize vector for cosine similarity
        vec_array = np.array([vector], dtype=np.float32)
        self._faiss.normalize_L2(vec_array)

        # Train IVF index if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            self._index.train(vec_array)

        # Add to FAISS
        self._index.add(vec_array)

        # Store mappings
        faiss_id = self._next_id
        self._id_map[faiss_id] = doc_id
        self._documents[doc_id] = text
        self._metadata[doc_id] = metadata or {}
        self._next_id += 1

    def add_batch(
        self,
        documents: List[Dict[str, Any]],
    ) -> int:
        """
        Add multiple documents efficiently.

        Args:
            documents: List of dicts with keys: doc_id, vector, text, metadata

        Returns:
            Number of documents added
        """
        if not self._initialized:
            self.initialize()

        if not documents:
            return 0

        # Prepare vectors
        vectors = np.array(
            [doc["vector"] for doc in documents],
            dtype=np.float32
        )
        self._faiss.normalize_L2(vectors)

        # Train IVF if needed
        if self.index_type == "ivf" and not self._index.is_trained:
            self._index.train(vectors)

        # Add to FAISS
        self._index.add(vectors)

        # Store mappings
        for doc in documents:
            faiss_id = self._next_id
            doc_id = doc["doc_id"]
            self._id_map[faiss_id] = doc_id
            self._documents[doc_id] = doc["text"]
            self._metadata[doc_id] = doc.get("metadata", {})
            self._next_id += 1

        logger.info(f"Added batch of {len(documents)} documents")
        return len(documents)

    def search(
        self,
        query_vector: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter_metadata: Optional metadata filter (post-filter)

        Returns:
            List of results with doc_id, text, metadata, score
        """
        if not self._initialized:
            self.initialize()

        if self._index.ntotal == 0:
            return []

        # Normalize query
        query = np.array([query_vector], dtype=np.float32)
        self._faiss.normalize_L2(query)

        # Search (get more if filtering)
        search_k = k * 3 if filter_metadata else k
        scores, indices = self._index.search(query, min(search_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for not found
                continue

            doc_id = self._id_map.get(idx)
            if not doc_id:
                continue

            metadata = self._metadata.get(doc_id, {})

            # Apply metadata filter
            if filter_metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append({
                "doc_id": doc_id,
                "text": self._documents.get(doc_id, ""),
                "metadata": metadata,
                "score": float(score),
            })

            if len(results) >= k:
                break

        return results

    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID.

        Note: FAISS doesn't support efficient deletion. For production,
        consider using IndexIDMap or rebuilding periodically.
        """
        if doc_id in self._documents:
            del self._documents[doc_id]
            del self._metadata[doc_id]
            # Note: Vector remains in FAISS index (marked as deleted)
            logger.warning(f"Soft-deleted {doc_id}. Rebuild index to reclaim space.")
            return True
        return False

    def save(self) -> None:
        """Persist index and metadata to disk."""
        if not self.storage_path:
            raise ValueError("No storage_path configured")

        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = self.storage_path / "faiss.index"
        self._faiss.write_index(self._index, str(index_path))

        # Save metadata
        meta_path = self.storage_path / "metadata.pkl"
        with open(meta_path, "wb") as f:
            pickle.dump({
                "id_map": self._id_map,
                "metadata": self._metadata,
                "documents": self._documents,
                "next_id": self._next_id,
                "dimension": self.dimension,
                "index_type": self.index_type,
            }, f)

        logger.info(f"Saved index with {self._index.ntotal} vectors to {self.storage_path}")

    def _load_from_disk(self) -> bool:
        """Load index and metadata from disk."""
        if not self.storage_path:
            return False

        index_path = self.storage_path / "faiss.index"
        meta_path = self.storage_path / "metadata.pkl"

        if not index_path.exists() or not meta_path.exists():
            return False

        try:
            self._ensure_faiss()

            # Load FAISS index
            self._index = self._faiss.read_index(str(index_path))

            # Load metadata
            with open(meta_path, "rb") as f:
                data = pickle.load(f)

            self._id_map = data["id_map"]
            self._metadata = data["metadata"]
            self._documents = data["documents"]
            self._next_id = data["next_id"]

            logger.info(f"Loaded index with {self._index.ntotal} vectors from {self.storage_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def count(self) -> int:
        """Get total number of vectors."""
        if not self._initialized:
            return 0
        return self._index.ntotal

    def clear(self) -> None:
        """Clear all data and reinitialize."""
        self._id_map.clear()
        self._metadata.clear()
        self._documents.clear()
        self._next_id = 0
        self._create_index()
        logger.info("Cleared vector store")

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "total_vectors": self.count(),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "storage_path": str(self.storage_path) if self.storage_path else None,
            "initialized": self._initialized,
        }
