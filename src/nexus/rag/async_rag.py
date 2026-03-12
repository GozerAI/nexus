"""
Async RAG Operations - Non-blocking document processing and queries.

Provides:
- Async document ingestion with parallel embedding
- Async batch queries
- Background indexing
- Connection pooling for external services
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from queue import Queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class AsyncRAGConfig:
    """Configuration for async RAG operations."""

    # Thread pool settings
    max_embedding_workers: int = 4
    max_indexing_workers: int = 2

    # Batch settings
    embedding_batch_size: int = 32
    indexing_batch_size: int = 100

    # Queue settings
    max_queue_size: int = 10000
    flush_interval_seconds: float = 1.0


@dataclass
class IndexingTask:
    """A document queued for indexing."""

    doc_id: str
    text: str
    metadata: Dict[str, Any]
    priority: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AsyncEmbedder:
    """
    Async wrapper for embedding models with batching.

    Collects embedding requests and processes them in batches
    for much better throughput.
    """

    def __init__(
        self,
        embedding_model,
        batch_size: int = 32,
        max_workers: int = 4,
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = asyncio.Lock()

    async def embed(self, text: str) -> List[float]:
        """Embed a single text asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.embedding_model.embed,
            text,
        )

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts asynchronously with batching."""
        if not texts:
            return []

        loop = asyncio.get_event_loop()

        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = await loop.run_in_executor(
                self._executor,
                self.embedding_model.embed_batch,
                batch,
            )
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def embed_parallel(self, texts: List[str]) -> List[List[float]]:
        """Embed texts in parallel using multiple batches."""
        if not texts:
            return []

        # Split into batches
        batches = [
            texts[i:i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]

        # Process batches in parallel
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self._executor,
                self.embedding_model.embed_batch,
                batch,
            )
            for batch in batches
        ]

        results = await asyncio.gather(*tasks)

        # Flatten results
        all_embeddings = []
        for batch_result in results:
            all_embeddings.extend(batch_result)

        return all_embeddings

    def shutdown(self):
        """Shutdown the thread pool."""
        self._executor.shutdown(wait=True)


class BackgroundIndexer:
    """
    Background indexing with queuing.

    Documents are queued and indexed in the background,
    allowing fast ingestion without blocking.
    """

    def __init__(
        self,
        vector_store,
        embedding_model,
        batch_size: int = 100,
        flush_interval: float = 1.0,
        on_indexed: Optional[Callable[[int], None]] = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.on_indexed = on_indexed

        self._queue: Queue = Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._indexed_count = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background indexer."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._indexing_loop, daemon=True)
        self._thread.start()
        logger.info("Background indexer started")

    def stop(self, wait: bool = True) -> None:
        """Stop the background indexer."""
        self._running = False
        if wait and self._thread:
            self._thread.join(timeout=10)
        logger.info(f"Background indexer stopped. Indexed {self._indexed_count} documents.")

    def queue_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 0,
    ) -> None:
        """Queue a document for background indexing."""
        task = IndexingTask(
            doc_id=doc_id,
            text=text,
            metadata=metadata or {},
            priority=priority,
        )
        self._queue.put(task)

    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    def indexed_count(self) -> int:
        """Get total indexed document count."""
        return self._indexed_count

    def _indexing_loop(self) -> None:
        """Main indexing loop running in background thread."""
        batch: List[IndexingTask] = []

        while self._running or not self._queue.empty():
            try:
                # Collect batch
                while len(batch) < self.batch_size:
                    try:
                        task = self._queue.get(timeout=self.flush_interval)
                        batch.append(task)
                    except Exception:
                        # Queue.get timeout or empty - normal exit condition
                        break

                if batch:
                    self._process_batch(batch)
                    batch = []

            except Exception as e:
                logger.error(f"Indexing error: {e}")

    def _process_batch(self, batch: List[IndexingTask]) -> None:
        """Process a batch of documents."""
        try:
            # Extract texts
            texts = [task.text for task in batch]

            # Generate embeddings
            embeddings = self.embedding_model.embed_batch(texts)

            # Prepare documents
            documents = [
                {
                    "doc_id": task.doc_id,
                    "vector": embedding,
                    "text": task.text,
                    "metadata": task.metadata,
                }
                for task, embedding in zip(batch, embeddings)
            ]

            # Add to vector store
            self.vector_store.add_batch(documents)

            # Update count
            with self._lock:
                self._indexed_count += len(batch)

            # Callback
            if self.on_indexed:
                self.on_indexed(len(batch))

            logger.debug(f"Indexed batch of {len(batch)} documents")

        except Exception as e:
            logger.error(f"Failed to index batch: {e}")


class AsyncRAG:
    """
    Async RAG system with non-blocking operations.

    Wraps the MVP RAG with async capabilities:
    - Async document ingestion
    - Parallel embedding
    - Background indexing
    - Async queries
    """

    def __init__(
        self,
        mvp_rag,  # MVPRAG instance
        config: Optional[AsyncRAGConfig] = None,
    ):
        self.rag = mvp_rag
        self.config = config or AsyncRAGConfig()

        self._async_embedder: Optional[AsyncEmbedder] = None
        self._background_indexer: Optional[BackgroundIndexer] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize async components."""
        if self._initialized:
            return

        # Ensure base RAG is initialized
        if not self.rag._initialized:
            self.rag.initialize()

        # Create async embedder
        self._async_embedder = AsyncEmbedder(
            embedding_model=self.rag._embedder,
            batch_size=self.config.embedding_batch_size,
            max_workers=self.config.max_embedding_workers,
        )

        # Create background indexer
        self._background_indexer = BackgroundIndexer(
            vector_store=self.rag._store,
            embedding_model=self.rag._embedder,
            batch_size=self.config.indexing_batch_size,
            flush_interval=self.config.flush_interval_seconds,
        )
        self._background_indexer.start()

        self._initialized = True
        logger.info("AsyncRAG initialized")

    async def add_document(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        background: bool = False,
    ) -> int:
        """
        Add a document asynchronously.

        Args:
            content: Document text
            doc_id: Optional document ID
            metadata: Optional metadata
            background: If True, queue for background indexing

        Returns:
            Number of chunks (0 if background)
        """
        if not self._initialized:
            await self.initialize()

        from .mvp_rag import Document
        doc = Document(content=content, doc_id=doc_id, metadata=metadata or {})

        if background:
            # Queue for background indexing
            chunks = self.rag._chunker.chunk(doc.content, metadata={"doc_id": doc.doc_id})
            for chunk in chunks:
                self._background_indexer.queue_document(
                    doc_id=f"{doc.doc_id}:chunk:{chunk.chunk_index}",
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            return 0  # Actual count unknown until indexed

        # Chunk document
        chunk_metadata = {"doc_id": doc.doc_id, **doc.metadata}
        chunks = self.rag._chunker.chunk(doc.content, metadata=chunk_metadata)

        if not chunks:
            return 0

        # Embed chunks in parallel
        chunk_texts = [c.text for c in chunks]
        embeddings = await self._async_embedder.embed_parallel(chunk_texts)

        # Add to store
        documents = [
            {
                "doc_id": f"{doc.doc_id}:chunk:{i}",
                "vector": embedding,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        added = self.rag._store.add_batch(documents)

        self.rag._stats["documents_added"] += 1
        self.rag._stats["chunks_indexed"] += added

        return added

    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        background: bool = False,
        parallel: bool = True,
    ) -> int:
        """
        Add multiple documents asynchronously.

        Args:
            documents: List of document dicts
            background: Queue for background indexing
            parallel: Process documents in parallel

        Returns:
            Total chunks added
        """
        if not self._initialized:
            await self.initialize()

        if parallel:
            # Process all documents in parallel
            tasks = [
                self.add_document(
                    content=doc.get("content", doc) if isinstance(doc, dict) else doc,
                    doc_id=doc.get("doc_id") if isinstance(doc, dict) else None,
                    metadata=doc.get("metadata") if isinstance(doc, dict) else None,
                    background=background,
                )
                for doc in documents
            ]
            results = await asyncio.gather(*tasks)
            return sum(results)
        else:
            # Process sequentially
            total = 0
            for doc in documents:
                if isinstance(doc, str):
                    count = await self.add_document(content=doc, background=background)
                else:
                    count = await self.add_document(
                        content=doc["content"],
                        doc_id=doc.get("doc_id"),
                        metadata=doc.get("metadata"),
                        background=background,
                    )
                total += count
            return total

    async def query(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Query asynchronously.

        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            RetrievalResult
        """
        if not self._initialized:
            await self.initialize()

        import time
        start = time.time()

        # Embed query asynchronously
        query_vector = await self._async_embedder.embed(query)

        # Search (sync, but fast)
        results = self.rag._store.search(
            query_vector=query_vector,
            k=top_k,
            filter_metadata=filter_metadata,
        )

        elapsed_ms = (time.time() - start) * 1000
        self.rag._stats["queries_processed"] += 1

        from .mvp_rag import RetrievalResult
        total_chars = sum(len(r["text"]) for r in results)

        return RetrievalResult(
            query=query,
            chunks=results,
            total_tokens_estimate=total_chars // 4,
            retrieval_time_ms=elapsed_ms,
        )

    async def query_batch(
        self,
        queries: List[str],
        top_k: int = 5,
    ) -> List:
        """
        Run multiple queries in parallel.

        Args:
            queries: List of query strings
            top_k: Results per query

        Returns:
            List of RetrievalResults
        """
        tasks = [self.query(q, top_k=top_k) for q in queries]
        return await asyncio.gather(*tasks)

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = self.rag.get_stats()

        if self._background_indexer:
            stats["background_queue_size"] = self._background_indexer.queue_size()
            stats["background_indexed"] = self._background_indexer.indexed_count()

        return stats

    async def shutdown(self) -> None:
        """Shutdown async components."""
        if self._background_indexer:
            self._background_indexer.stop(wait=True)

        if self._async_embedder:
            self._async_embedder.shutdown()

        logger.info("AsyncRAG shutdown complete")


async def create_async_rag(
    embedding_provider: str = "sentence-transformers",
    storage_path: Optional[str] = None,
    **kwargs,
) -> AsyncRAG:
    """
    Create an async RAG system.

    Args:
        embedding_provider: Embedding provider
        storage_path: Optional persistence path
        **kwargs: Additional config

    Returns:
        Initialized AsyncRAG
    """
    from .mvp_rag import create_rag

    mvp_rag = create_rag(
        embedding_provider=embedding_provider,
        storage_path=storage_path,
        **kwargs,
    )

    async_rag = AsyncRAG(mvp_rag)
    await async_rag.initialize()

    return async_rag
