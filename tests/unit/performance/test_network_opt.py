"""Tests for network optimization modules."""

import asyncio
import gzip
import pytest
import zlib

from nexus.core.network.request_batching import RequestBatcher, BatchedRequest, Batch
from nexus.core.network.compression import (
    CompressionMiddleware, CompressionAlgorithm, CompressionResult,
)
from nexus.core.network.provider_queue import (
    ProviderRequestQueue, QueuedProviderRequest, RateLimitConfig, TokenBucket,
)


# ── RequestBatcher ───────────────────────────────────────────

class TestRequestBatcher:
    @pytest.mark.asyncio
    async def test_batch_on_size(self):
        batches_received = []

        async def batch_fn(requests):
            batches_received.append(requests)
            return [f"result_{i}" for i in range(len(requests))]

        batcher = RequestBatcher(
            batch_fn=batch_fn, max_batch_size=3, batch_timeout=10
        )

        # Submit 3 requests to trigger size-based flush
        tasks = [
            asyncio.create_task(batcher.submit(f"prompt_{i}", "model"))
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        assert len(batches_received) == 1
        assert len(batches_received[0]) == 3

    @pytest.mark.asyncio
    async def test_batch_on_timeout(self):
        batches_received = []

        async def batch_fn(requests):
            batches_received.append(requests)
            return [None] * len(requests)

        batcher = RequestBatcher(
            batch_fn=batch_fn, max_batch_size=100, batch_timeout=0.05
        )

        task = asyncio.create_task(batcher.submit("prompt", "model"))
        result = await task
        assert len(batches_received) == 1

    @pytest.mark.asyncio
    async def test_per_model_batching(self):
        batches = []

        async def batch_fn(requests):
            models = set(r.model_name for r in requests)
            batches.append(models)
            return [None] * len(requests)

        batcher = RequestBatcher(
            batch_fn=batch_fn, max_batch_size=2, per_model=True
        )

        t1 = asyncio.create_task(batcher.submit("p", "model_a"))
        t2 = asyncio.create_task(batcher.submit("p", "model_a"))
        await asyncio.gather(t1, t2)
        # Should batch together since same model
        assert len(batches) >= 1
        assert all(len(b) == 1 for b in batches)

    def test_pending_count(self):
        batcher = RequestBatcher()
        assert batcher.pending_count == 0

    def test_stats(self):
        batcher = RequestBatcher()
        stats = batcher.get_stats()
        assert stats["requests_submitted"] == 0


class TestBatchedRequest:
    def test_defaults(self):
        r = BatchedRequest()
        assert r.request_id
        assert r.prompt == ""


class TestBatch:
    def test_size(self):
        b = Batch()
        b.requests = [BatchedRequest(), BatchedRequest()]
        assert b.size == 2


# ── CompressionMiddleware ────────────────────────────────────

class TestCompressionMiddleware:
    def test_gzip_compress(self):
        mw = CompressionMiddleware()
        data = b"hello world " * 100
        compressed, result = mw.compress(data)
        assert result.algorithm == CompressionAlgorithm.GZIP
        assert len(compressed) < len(data)
        assert result.savings_percent > 0

    def test_gzip_decompress(self):
        mw = CompressionMiddleware()
        original = b"test data repeating content " * 200  # Must exceed min_size
        compressed, result = mw.compress(original)
        assert result.algorithm == CompressionAlgorithm.GZIP
        decompressed = mw.decompress(compressed, CompressionAlgorithm.GZIP)
        assert decompressed == original

    def test_deflate_compress(self):
        mw = CompressionMiddleware(preferred_algorithm=CompressionAlgorithm.DEFLATE)
        data = b"deflate test " * 100
        compressed, result = mw.compress(data)
        assert result.algorithm == CompressionAlgorithm.DEFLATE
        assert len(compressed) < len(data)

    def test_deflate_decompress(self):
        mw = CompressionMiddleware()
        original = b"deflate data " * 100
        compressed = zlib.compress(original)
        decompressed = mw.decompress(compressed, CompressionAlgorithm.DEFLATE)
        assert decompressed == original

    def test_skip_small_payload(self):
        mw = CompressionMiddleware(min_size=1000)
        data = b"small"
        compressed, result = mw.compress(data)
        assert result.algorithm == CompressionAlgorithm.NONE
        assert compressed == data

    def test_skip_compressed_content_type(self):
        mw = CompressionMiddleware()
        data = b"x" * 2000
        compressed, result = mw.compress(data, content_type="image/png")
        assert result.algorithm == CompressionAlgorithm.NONE

    def test_select_algorithm_gzip(self):
        mw = CompressionMiddleware()
        algo = mw.select_algorithm("gzip, deflate, br")
        assert algo == CompressionAlgorithm.GZIP

    def test_select_algorithm_none(self):
        mw = CompressionMiddleware()
        algo = mw.select_algorithm("")
        assert algo == CompressionAlgorithm.NONE

    def test_detect_encoding(self):
        mw = CompressionMiddleware()
        assert mw.detect_encoding("gzip") == CompressionAlgorithm.GZIP
        assert mw.detect_encoding("deflate") == CompressionAlgorithm.DEFLATE
        assert mw.detect_encoding("") == CompressionAlgorithm.NONE

    def test_stats(self):
        mw = CompressionMiddleware()
        data = b"stats test " * 200
        mw.compress(data)
        stats = mw.get_stats()
        assert stats["compressed"] == 1
        assert stats["bandwidth_saved_bytes"] > 0

    def test_compression_result(self):
        r = CompressionResult(
            original_size=1000, compressed_size=300,
            algorithm=CompressionAlgorithm.GZIP,
            compression_time_ms=1.5,
        )
        assert r.ratio == 0.3
        assert r.savings_percent == pytest.approx(70.0)


# ── ProviderRequestQueue ────────────────────────────────────

class TestProviderRequestQueue:
    @pytest.fixture
    def queue(self):
        q = ProviderRequestQueue()
        q.configure_provider(RateLimitConfig(
            provider_name="openai",
            requests_per_second=100,
            tokens_per_minute=1000000,
            concurrent_limit=10,
        ))
        return q

    @pytest.mark.asyncio
    async def test_submit(self, queue):
        req = await queue.submit("openai", "hello", model_name="gpt-4")
        assert req.request_id
        assert req.provider_name == "openai"
        assert queue.get_queue_depth("openai") == 1

    @pytest.mark.asyncio
    async def test_submit_unknown_provider(self, queue):
        with pytest.raises(ValueError, match="not configured"):
            await queue.submit("unknown", "hello")

    @pytest.mark.asyncio
    async def test_process_next(self, queue):
        await queue.submit("openai", "hello")

        async def execute(req):
            return "result"

        processed = await queue.process_next("openai", execute)
        assert processed.result == "result"
        assert processed.completed_at is not None

    @pytest.mark.asyncio
    async def test_process_failure(self, queue):
        await queue.submit("openai", "hello")

        async def failing_execute(req):
            raise RuntimeError("fail")

        processed = await queue.process_next("openai", failing_execute)
        assert processed.error == "fail"

    def test_stats(self, queue):
        stats = queue.get_stats("openai")
        assert stats.get("submitted", 0) == 0


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_acquire_available(self):
        bucket = TokenBucket(rate=10, capacity=10)
        wait = await bucket.acquire(1)
        assert wait == 0.0
        assert bucket.available == 9.0

    @pytest.mark.asyncio
    async def test_acquire_deficit(self):
        bucket = TokenBucket(rate=100, capacity=1)
        await bucket.acquire(1)  # Use up capacity
        wait = await bucket.acquire(1)
        assert wait >= 0  # May need to wait


class TestQueuedProviderRequest:
    def test_wait_time(self):
        import time
        req = QueuedProviderRequest(enqueued_at=time.time() - 1)
        assert req.wait_time_ms >= 900


class TestRateLimitConfig:
    def test_defaults(self):
        config = RateLimitConfig(provider_name="test")
        assert config.requests_per_minute == 60
        assert config.concurrent_limit == 10
