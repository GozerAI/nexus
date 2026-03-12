"""Redis vector storage with RediSearch/RedisVL."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib

from nexus.core.exceptions import StorageError, NotFoundError

# Try to import redis
try:
    import redis
    from redis.commands.search.field import TextField, VectorField, NumericField, TagField
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

import numpy as np


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    # Connection options
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None

    # Redis Cloud options
    cloud_url: Optional[str] = None  # redis://user:password@host:port

    # Index settings
    index_name: str = "nexus_vectors"
    prefix: str = "nexus:chunk:"

    # Vector settings
    vector_dimensions: int = 384  # all-MiniLM-L6-v2 dimensions
    distance_metric: str = "COSINE"  # COSINE, L2, or IP


class RedisVectorStore:
    """Redis-backed vector storage using RediSearch."""

    def __init__(self, config: Optional[RedisConfig] = None):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Run: pip install redis")

        self.config = config or RedisConfig()
        self._client: Optional[redis.Redis] = None
        self._index_exists = False

    def initialize(self):
        """Initialize Redis connection and create index."""
        # Connect to Redis
        if self.config.cloud_url:
            self._client = redis.from_url(self.config.cloud_url)
        else:
            self._client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password,
                decode_responses=False  # We need bytes for vectors
            )

        # Test connection
        try:
            self._client.ping()
        except redis.ConnectionError as e:
            raise StorageError(f"Cannot connect to Redis: {e}")

        # Create index if it doesn't exist
        self._ensure_index()

    def _ensure_index(self):
        """Create the vector search index if it doesn't exist."""
        try:
            # Check if index exists
            self._client.ft(self.config.index_name).info()
            self._index_exists = True
        except redis.ResponseError:
            # Create new index
            schema = (
                TextField("text"),
                TextField("conversation_id"),
                TextField("conversation_title"),
                TagField("project_path"),
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "FLAT",  # or "HNSW" for larger datasets
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.config.vector_dimensions,
                        "DISTANCE_METRIC": self.config.distance_metric
                    }
                )
            )

            definition = IndexDefinition(
                prefix=[self.config.prefix],
                index_type=IndexType.HASH
            )

            self._client.ft(self.config.index_name).create_index(
                schema, definition=definition
            )
            self._index_exists = True

    def _chunk_key(self, chunk_id: str) -> str:
        """Get Redis key for a chunk."""
        return f"{self.config.prefix}{chunk_id}"

    def add(self, chunks: List[Any]) -> List[str]:
        """Add chunks to Redis."""
        if not chunks:
            return []

        pipe = self._client.pipeline()
        ids = []

        for chunk in chunks:
            key = self._chunk_key(chunk.id)

            # Prepare data
            data = {
                "id": chunk.id,
                "text": chunk.text,
                "conversation_id": chunk.metadata.get("conversation_id", ""),
                "conversation_title": chunk.metadata.get("conversation_title", ""),
                "project_path": chunk.metadata.get("project_path", ""),
                "created_at": chunk.created_at.timestamp() if isinstance(chunk.created_at, datetime) else 0,
                "metadata": json.dumps(chunk.metadata)
            }

            # Add embedding if present
            if chunk.embedding:
                embedding_bytes = np.array(chunk.embedding, dtype=np.float32).tobytes()
                data["embedding"] = embedding_bytes

            pipe.hset(key, mapping=data)
            ids.append(chunk.id)

        pipe.execute()
        return ids

    def search(self, query_embedding: List[float], n_results: int = 10,
               where: Optional[Dict] = None) -> List[Any]:
        """Search for similar chunks."""
        from nexus.storage.vector_store import VectorChunk, SearchResult

        # Convert embedding to bytes
        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        # Build query
        base_query = f"*=>[KNN {n_results} @embedding $vec AS score]"

        # Add filters
        if where:
            filters = []
            for key, value in where.items():
                if key == "project_path":
                    filters.append(f"@project_path:{{{value}}}")
                elif key == "conversation_id":
                    filters.append(f"@conversation_id:{value}")

            if filters:
                filter_str = " ".join(filters)
                base_query = f"({filter_str})=>[KNN {n_results} @embedding $vec AS score]"

        query = (
            Query(base_query)
            .return_fields("id", "text", "metadata", "score", "created_at")
            .sort_by("score")
            .dialect(2)
        )

        try:
            results = self._client.ft(self.config.index_name).search(
                query, {"vec": query_bytes}
            )
        except redis.ResponseError as e:
            print(f"Redis search error: {e}")
            return []

        search_results = []
        for doc in results.docs:
            try:
                metadata = json.loads(doc.metadata) if hasattr(doc, 'metadata') else {}
                created_at = datetime.fromtimestamp(float(doc.created_at)) if hasattr(doc, 'created_at') else datetime.now(timezone.utc)

                chunk = VectorChunk(
                    id=doc.id.replace(self.config.prefix, ""),
                    text=doc.text if hasattr(doc, 'text') else "",
                    metadata=metadata,
                    created_at=created_at
                )

                # Convert distance to score (lower is better for COSINE)
                score = float(doc.score) if hasattr(doc, 'score') else 0.0
                search_results.append(SearchResult(chunk=chunk, score=score))
            except Exception as e:
                print(f"Error parsing result: {e}")
                continue

        return search_results

    def get(self, chunk_id: str) -> Any:
        """Get chunk by ID."""
        from nexus.storage.vector_store import VectorChunk

        key = self._chunk_key(chunk_id)
        data = self._client.hgetall(key)

        if not data:
            raise NotFoundError(f"Chunk not found: {chunk_id}")

        # Decode bytes
        data = {k.decode() if isinstance(k, bytes) else k:
                v.decode() if isinstance(v, bytes) and k != b'embedding' else v
                for k, v in data.items()}

        metadata = json.loads(data.get("metadata", "{}"))
        created_at = datetime.fromtimestamp(float(data.get("created_at", 0)))

        return VectorChunk(
            id=chunk_id,
            text=data.get("text", ""),
            metadata=metadata,
            created_at=created_at
        )

    def delete(self, chunk_ids: List[str]) -> int:
        """Delete chunks by ID."""
        if not chunk_ids:
            return 0

        keys = [self._chunk_key(cid) for cid in chunk_ids]
        return self._client.delete(*keys)

    def count(self) -> int:
        """Get total chunks."""
        try:
            info = self._client.ft(self.config.index_name).info()
            return int(info.get("num_docs", 0))
        except Exception:
            return 0

    def list_all(self, where: Optional[Dict] = None, limit: int = 1000) -> List[Any]:
        """List all chunks matching criteria."""
        from nexus.storage.vector_store import VectorChunk

        # Build query
        if where:
            filters = []
            for key, value in where.items():
                if key == "project_path":
                    filters.append(f"@project_path:{{{value}}}")
            query_str = " ".join(filters) if filters else "*"
        else:
            query_str = "*"

        query = Query(query_str).paging(0, limit)

        try:
            results = self._client.ft(self.config.index_name).search(query)
        except redis.ResponseError:
            return []

        chunks = []
        for doc in results.docs:
            try:
                metadata = json.loads(doc.metadata) if hasattr(doc, 'metadata') else {}
                created_at = datetime.fromtimestamp(float(doc.created_at)) if hasattr(doc, 'created_at') else datetime.now(timezone.utc)

                chunks.append(VectorChunk(
                    id=doc.id.replace(self.config.prefix, ""),
                    text=doc.text if hasattr(doc, 'text') else "",
                    metadata=metadata,
                    created_at=created_at
                ))
            except Exception:
                continue

        return chunks

    def clear(self):
        """Clear all chunks from the index."""
        # Delete all keys with our prefix
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=f"{self.config.prefix}*", count=100)
            if keys:
                self._client.delete(*keys)
            if cursor == 0:
                break


class RedisSemanticCache:
    """Semantic cache for LLM responses using Redis."""

    def __init__(self, redis_client: redis.Redis,
                 index_name: str = "nexus_cache",
                 ttl_seconds: int = 86400,  # 24 hours
                 similarity_threshold: float = 0.95):
        self._client = redis_client
        self.index_name = index_name
        self.prefix = "nexus:cache:"
        self.ttl = ttl_seconds
        self.threshold = similarity_threshold
        self._ensure_index()

    def _ensure_index(self):
        """Create cache index if needed."""
        try:
            self._client.ft(self.index_name).info()
        except redis.ResponseError:
            schema = (
                TextField("query"),
                TextField("response"),
                NumericField("created_at"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 384,
                        "DISTANCE_METRIC": "COSINE"
                    }
                )
            )

            definition = IndexDefinition(
                prefix=[self.prefix],
                index_type=IndexType.HASH
            )

            self._client.ft(self.index_name).create_index(schema, definition=definition)

    def _cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        hash_val = hashlib.sha256(query.encode()).hexdigest()[:16]
        return f"{self.prefix}{hash_val}"

    def get(self, query: str, query_embedding: List[float]) -> Optional[str]:
        """Check cache for similar query."""
        query_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        search_query = (
            Query(f"*=>[KNN 1 @embedding $vec AS score]")
            .return_fields("response", "score")
            .dialect(2)
        )

        try:
            results = self._client.ft(self.index_name).search(
                search_query, {"vec": query_bytes}
            )

            if results.docs:
                doc = results.docs[0]
                score = float(doc.score)
                # COSINE distance: 0 = identical, 2 = opposite
                similarity = 1 - (score / 2)

                if similarity >= self.threshold:
                    return doc.response
        except Exception:
            pass

        return None

    def set(self, query: str, response: str, query_embedding: List[float]):
        """Cache a query-response pair."""
        key = self._cache_key(query)
        embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()

        data = {
            "query": query,
            "response": response,
            "embedding": embedding_bytes,
            "created_at": datetime.now(timezone.utc).timestamp()
        }

        self._client.hset(key, mapping=data)
        self._client.expire(key, self.ttl)
