"""Tests for API performance modules."""

import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from nexus.core.api.field_selection import FieldSelector, FieldSelectionMiddleware
from nexus.core.api.response_limits import ResponseSizeLimiter, SizeLimitExceeded
from nexus.core.api.webhook_callback import (
    AsyncWebhookDispatcher, WebhookJob, JobStatus,
)
from nexus.core.api.response_streaming import (
    ResponseStreamer, StreamChunk, StreamFormat,
)


# ── FieldSelector ──────────────────────────────────────────────

class TestFieldSelector:
    def test_flat_fields(self):
        sel = FieldSelector("name,provider,cost")
        data = {"name": "gpt-4", "provider": "openai", "cost": 0.03, "extra": "x"}
        result = sel.apply(data)
        assert result == {"name": "gpt-4", "provider": "openai", "cost": 0.03}

    def test_nested_fields(self):
        sel = FieldSelector("name,steps.name,steps.status")
        data = {
            "name": "pipeline",
            "steps": [
                {"name": "step1", "status": "done", "output": "big"},
                {"name": "step2", "status": "running", "output": "big"},
            ],
            "meta": "ignored",
        }
        result = sel.apply(data)
        assert result["name"] == "pipeline"
        assert "meta" not in result
        for step in result["steps"]:
            assert "output" not in step
            assert "name" in step
            assert "status" in step

    def test_wildcard(self):
        sel = FieldSelector("config.*")
        data = {"config": {"a": 1, "b": 2}, "other": "x"}
        result = sel.apply(data)
        assert result == {"config": {"a": 1, "b": 2}}

    def test_list_of_dicts(self):
        sel = FieldSelector("name")
        data = [{"name": "a", "extra": 1}, {"name": "b", "extra": 2}]
        result = sel.apply(data)
        assert result == [{"name": "a"}, {"name": "b"}]

    def test_too_many_fields(self):
        with pytest.raises(ValueError, match="Too many fields"):
            FieldSelector(",".join(f"f{i}" for i in range(200)))

    def test_too_deep_path(self):
        with pytest.raises(ValueError, match="too deep"):
            FieldSelector(".".join(f"a" for _ in range(20)))

    def test_invalid_segment(self):
        with pytest.raises(ValueError, match="Invalid"):
            FieldSelector("name,inv@lid")

    def test_paths_property(self):
        sel = FieldSelector("a,b.c")
        assert sel.paths == [["a"], ["b", "c"]]


class TestFieldSelectionMiddleware:
    def test_no_fields_passthrough(self):
        mw = FieldSelectionMiddleware()
        data = {"a": 1, "b": 2}
        assert mw.process_response(data) == data

    def test_with_fields(self):
        mw = FieldSelectionMiddleware()
        data = {"a": 1, "b": 2, "c": 3}
        result = mw.process_response(data, "a,c")
        assert result == {"a": 1, "c": 3}

    def test_always_include(self):
        mw = FieldSelectionMiddleware(always_include={"id"})
        data = {"id": 1, "name": "test", "extra": "x"}
        result = mw.process_response(data, "name")
        assert result == {"name": "test", "id": 1}

    def test_always_include_list(self):
        mw = FieldSelectionMiddleware(always_include={"id"})
        data = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
        result = mw.process_response(data, "name")
        assert all("id" in item for item in result)

    def test_invalid_fields_passthrough(self):
        mw = FieldSelectionMiddleware()
        data = {"a": 1}
        result = mw.process_response(data, "a,inv@lid")
        assert result == data  # Falls back to full response


# ── ResponseSizeLimiter ────────────────────────────────────────

class TestResponseSizeLimiter:
    def test_small_response_passes(self):
        limiter = ResponseSizeLimiter(default_limit_bytes=10000)
        result = limiter.check({"key": "value"})
        assert not result.was_truncated

    def test_list_truncation(self):
        limiter = ResponseSizeLimiter(default_limit_bytes=500)
        data = [{"id": i, "content": "x" * 50} for i in range(100)]
        result = limiter.check(data)
        assert result.was_truncated
        assert result.returned_items < 100
        assert result.continuation_token is not None

    def test_error_strategy(self):
        limiter = ResponseSizeLimiter(
            default_limit_bytes=100,
            default_strategy=ResponseSizeLimiter.Strategy.ERROR,
        )
        data = {"content": "x" * 200}
        with pytest.raises(SizeLimitExceeded):
            limiter.check(data)

    def test_per_endpoint_limit(self):
        limiter = ResponseSizeLimiter(default_limit_bytes=10000)
        limiter.set_endpoint_limit("/api/v1/small", 100)
        data = {"content": "x" * 200}
        result = limiter.check(data, endpoint="/api/v1/small")
        assert result.was_truncated

    def test_estimate_size(self):
        limiter = ResponseSizeLimiter()
        assert limiter.estimate_size(None) == 4
        assert limiter.estimate_size(True) == 5
        assert limiter.estimate_size(42) > 0
        assert limiter.estimate_size("hello") == 7
        assert limiter.estimate_size([]) == 2
        assert limiter.estimate_size({}) == 2

    def test_stats(self):
        limiter = ResponseSizeLimiter(default_limit_bytes=10000)
        limiter.check({"a": 1})
        stats = limiter.get_stats()
        assert stats["checked"] == 1


# ── AsyncWebhookDispatcher ────────────────────────────────────

class TestAsyncWebhookDispatcher:
    @pytest.fixture
    def mock_http_post(self):
        return AsyncMock(return_value=(200, b"ok"))

    @pytest.fixture
    def dispatcher(self, mock_http_post):
        return AsyncWebhookDispatcher(http_post=mock_http_post, max_concurrent=5)

    @pytest.mark.asyncio
    async def test_submit_job(self, dispatcher):
        async def my_task():
            return {"result": "done"}

        job = dispatcher.submit(
            my_task(), "https://example.com/hook", webhook_secret="secret"
        )
        assert job.status == JobStatus.QUEUED
        assert job.job_id

    @pytest.mark.asyncio
    async def test_get_job(self, dispatcher):
        async def my_task():
            return "ok"

        job = dispatcher.submit(my_task(), "https://example.com/hook")
        retrieved = dispatcher.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id

    def test_job_to_dict(self):
        job = WebhookJob(job_id="test-1", webhook_url="http://x")
        d = job.to_dict()
        assert d["job_id"] == "test-1"
        assert "status" in d

    def test_stats(self, dispatcher):
        stats = dispatcher.get_stats()
        assert "jobs_submitted" in stats

    def test_cleanup(self, dispatcher):
        # Nothing to clean up
        removed = dispatcher.cleanup(max_age_seconds=0)
        assert removed == 0

    def test_cancel_job_prevents_completion_and_webhook(self):
        async def main():
            deliveries = []

            async def fake_post(url, data, headers, timeout):
                deliveries.append((url, data, headers, timeout))
                return 200, b"ok"

            dispatcher = AsyncWebhookDispatcher(http_post=fake_post, max_concurrent=1)

            async def slow_task():
                await asyncio.sleep(0.2)
                return {"result": "done"}

            job = dispatcher.submit(slow_task(), "https://example.com/hook")
            assert dispatcher.cancel_job(job.job_id) is True
            await asyncio.sleep(0.3)

            final_job = dispatcher.get_job(job.job_id)
            assert final_job.status == JobStatus.CANCELLED
            assert deliveries == []

        asyncio.run(main())


# ── ResponseStreamer ──────────────────────────────────────────

class TestResponseStreamer:
    def test_sync_streaming_ndjson(self):
        streamer = ResponseStreamer(chunk_size=2, stream_format=StreamFormat.NDJSON)
        items = [1, 2, 3, 4, 5]
        chunks = list(streamer.stream_sync(items))
        assert len(chunks) == 3  # [1,2], [3,4], [5]
        # Each chunk is valid NDJSON
        for chunk in chunks:
            parsed = json.loads(chunk.decode("utf-8"))
            assert "data" in parsed
            assert "index" in parsed

    def test_sync_streaming_sse(self):
        streamer = ResponseStreamer(chunk_size=3, stream_format=StreamFormat.SSE)
        items = list(range(6))
        chunks = list(streamer.stream_sync(items))
        assert len(chunks) == 2
        for chunk in chunks:
            text = chunk.decode("utf-8")
            assert "event:" in text
            assert "data:" in text

    def test_progress_callback(self):
        progress = []
        streamer = ResponseStreamer(
            chunk_size=5,
            progress_callback=lambda sent, total: progress.append((sent, total)),
        )
        list(streamer.stream_sync(list(range(10))))
        assert len(progress) == 2
        assert progress[-1] == (10, 10)

    def test_stats(self):
        streamer = ResponseStreamer(chunk_size=5)
        list(streamer.stream_sync(list(range(20))))
        stats = streamer.stats
        assert stats.total_chunks == 4
        assert stats.total_bytes > 0

    def test_stream_chunk_ndjson(self):
        chunk = StreamChunk(data=[1, 2], index=0)
        line = chunk.to_ndjson()
        parsed = json.loads(line)
        assert parsed["data"] == [1, 2]

    def test_stream_chunk_sse_last(self):
        chunk = StreamChunk(data=[], index=5, is_last=True)
        sse = chunk.to_sse()
        text = sse.decode("utf-8")
        assert "complete" in text


# ── StreamChunk unit ─────────────────────────────────────────

class TestStreamChunk:
    def test_to_ndjson_basic(self):
        c = StreamChunk(data={"a": 1}, index=0)
        raw = c.to_ndjson()
        assert raw.endswith(b"\n")

    def test_to_sse_basic(self):
        c = StreamChunk(data="hello", index=1, metadata={"k": "v"})
        raw = c.to_sse()
        assert b"event: data" in raw
