"""
Nexus RAG (Retrieval-Augmented Generation) System

Production-scale RAG with:
- FAISS vector search with HNSW indexing
- Hybrid search (BM25 + semantic)
- Cross-encoder reranking
- Async batch processing
- Document versioning and deduplication
- Query preprocessing and expansion

Quick Start:
    from nexus.rag import create_rag
    rag = create_rag()
    rag.add_document("Your text here...")
    results = rag.query("Your question")
"""

# MVP RAG (production-ready)
from .mvp_rag import MVPRAG, RAGConfig, Document, RetrievalResult, create_rag

# Embeddings
from .embeddings import (
    EmbeddingModel,
    SentenceTransformerEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    get_embedding_model,
)

# Chunking
from .chunking import (
    Chunk,
    Chunker,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    MarkdownChunker,
    get_chunker,
)

# Vector Store
from .vector.faiss_store import FAISSVectorStore

# Hybrid Search
from .hybrid_search import (
    BM25Index,
    HybridSearcher,
    HybridResult,
)

# Reranking
from .reranker import (
    Reranker,
    CrossEncoderReranker,
    CohereReranker,
    RerankResult,
    get_reranker,
)

# Async Operations
from .async_rag import (
    AsyncRAG,
    AsyncRAGConfig,
    AsyncEmbedder,
    BackgroundIndexer,
    create_async_rag,
)

# Document Management
from .document_manager import (
    DocumentManager,
    DocumentRecord,
    DocumentStatus,
)

# Query Processing
from .query_processor import (
    QueryProcessor,
    ProcessedQuery,
    MultiQueryGenerator,
    QueryDecomposer,
    HyDEQueryExpander,
    create_query_processor,
)

# Legacy components
from .rag_vector_engine import RAGVectorEngine
from .adaptive_rag_orchestrator import (
    AdaptiveRAGOrchestrator,
    OrchestrationMode,
    OrchestrationRequest,
    OrchestrationResponse,
    LearningIntensity,
)
from .context_window_manager import ContextWindowManager
from .adaptive_pathways import AdaptiveLearningPathways

# Knowledge-Augmented Generation (KAG)
from .knowledge_augmented_generation import (
    KnowledgeAugmentedGeneration,
    KnowledgeAugmentationMode,
    KnowledgeCoherenceLevel,
    KnowledgeSourceType,
    KnowledgeGrounding,
    AugmentedContext,
    VerifiedResponse,
    KnowledgeGap,
    DomainCoherenceReport,
    KAGConfig,
    create_kag,
)

# Knowledge-Enhanced Pathways (ALP + KAG Integration)
from .knowledge_enhanced_pathways import (
    KnowledgeEnhancedPathways,
    ContentVerificationLevel,
    KnowledgeIntegrationMode,
    VerifiedLearningContent,
    EnhancedStudySession,
    KnowledgeAlignedPathway,
    IntegratedPerformancePrediction,
    create_knowledge_enhanced_pathways,
)

# Content Library
from .content_library import (
    ContentLibrary,
    ContentLibraryConfig,
    ContentItem,
    ContentAsset,
    ContentVersion,
    ContentFormat,
    ContentType,
    ContentStatus,
    ContentSourceType,
    ContentFilters,
    ContentInteraction,
    InteractionType,
    ContentQualityMetrics,
    ContentStorageBackend,
    InMemoryStorage,
    FileStorage,
    HybridStorage,
    AssetManager,
    ContentGenerator,
    ContentGenerationConfig,
    ContentTemplate,
    ContentTemplateLibrary,
    ContentBuilder,
    ContentAnalytics,
    ContentQualityManager,
    ContentGraph,
    RelationshipType,
    LearningPath,
    create_content_library,
    create_storage,
    create_asset_manager,
    create_content_generator,
)

# Alias for backward compatibility
AdaptivePathways = AdaptiveLearningPathways

__all__ = [
    # MVP RAG (recommended)
    "MVPRAG",
    "RAGConfig",
    "Document",
    "RetrievalResult",
    "create_rag",
    # Embeddings
    "EmbeddingModel",
    "SentenceTransformerEmbedding",
    "OllamaEmbedding",
    "OpenAIEmbedding",
    "get_embedding_model",
    # Chunking
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "RecursiveChunker",
    "SentenceChunker",
    "MarkdownChunker",
    "get_chunker",
    # Vector Store
    "FAISSVectorStore",
    # Hybrid Search
    "BM25Index",
    "HybridSearcher",
    "HybridResult",
    # Reranking
    "Reranker",
    "CrossEncoderReranker",
    "CohereReranker",
    "RerankResult",
    "get_reranker",
    # Async
    "AsyncRAG",
    "AsyncRAGConfig",
    "AsyncEmbedder",
    "BackgroundIndexer",
    "create_async_rag",
    # Document Management
    "DocumentManager",
    "DocumentRecord",
    "DocumentStatus",
    # Query Processing
    "QueryProcessor",
    "ProcessedQuery",
    "MultiQueryGenerator",
    "QueryDecomposer",
    "HyDEQueryExpander",
    "create_query_processor",
    # Knowledge-Augmented Generation (KAG)
    "KnowledgeAugmentedGeneration",
    "KnowledgeAugmentationMode",
    "KnowledgeCoherenceLevel",
    "KnowledgeSourceType",
    "KnowledgeGrounding",
    "AugmentedContext",
    "VerifiedResponse",
    "KnowledgeGap",
    "DomainCoherenceReport",
    "KAGConfig",
    "create_kag",
    # Knowledge-Enhanced Pathways (ALP + KAG)
    "KnowledgeEnhancedPathways",
    "ContentVerificationLevel",
    "KnowledgeIntegrationMode",
    "VerifiedLearningContent",
    "EnhancedStudySession",
    "KnowledgeAlignedPathway",
    "IntegratedPerformancePrediction",
    "create_knowledge_enhanced_pathways",
    # Content Library
    "ContentLibrary",
    "ContentLibraryConfig",
    "ContentItem",
    "ContentAsset",
    "ContentVersion",
    "ContentFormat",
    "ContentType",
    "ContentStatus",
    "ContentSourceType",
    "ContentFilters",
    "ContentInteraction",
    "InteractionType",
    "ContentQualityMetrics",
    "ContentStorageBackend",
    "InMemoryStorage",
    "FileStorage",
    "HybridStorage",
    "AssetManager",
    "ContentGenerator",
    "ContentGenerationConfig",
    "ContentTemplate",
    "ContentTemplateLibrary",
    "ContentBuilder",
    "ContentAnalytics",
    "ContentQualityManager",
    "ContentGraph",
    "RelationshipType",
    "LearningPath",
    "create_content_library",
    "create_storage",
    "create_asset_manager",
    "create_content_generator",
    # Legacy / Orchestration
    "RAGVectorEngine",
    "AdaptiveRAGOrchestrator",
    "OrchestrationMode",
    "OrchestrationRequest",
    "OrchestrationResponse",
    "LearningIntensity",
    "ContextWindowManager",
    "AdaptivePathways",
    "AdaptiveLearningPathways",
]

__version__ = "2.0.0"
