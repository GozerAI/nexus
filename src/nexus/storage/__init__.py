"""Storage layer for Nexus Intelligence Platform."""

from nexus.storage.embedder import (
    LocalEmbedder, EmbeddingConfig,
    OllamaManager, BuiltinEmbedder
)
from nexus.storage.vector_store import (
    VectorStore, VectorChunk, SearchResult,
    SQLiteVectorStore, CHROMADB_AVAILABLE
)
from nexus.storage.sqlite_store import SQLiteStore

# Optional Redis support
try:
    from nexus.storage.redis_store import (
        RedisVectorStore, RedisConfig, RedisSemanticCache, REDIS_AVAILABLE
    )
except ImportError:
    REDIS_AVAILABLE = False
    RedisVectorStore = None
    RedisConfig = None
    RedisSemanticCache = None

__all__ = [
    "LocalEmbedder", "EmbeddingConfig",
    "OllamaManager", "BuiltinEmbedder",
    "VectorStore", "VectorChunk", "SearchResult",
    "SQLiteVectorStore", "CHROMADB_AVAILABLE",
    "SQLiteStore",
    "RedisVectorStore", "RedisConfig", "RedisSemanticCache", "REDIS_AVAILABLE",
]
