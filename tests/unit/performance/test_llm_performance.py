"""Tests for LLM performance optimization modules."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from nexus.core.llm.model_warmup import (
    ModelWarmupManager, WarmupConfig, WarmupResult, WarmupStatus,
)
from nexus.core.llm.kv_cache import KVCacheManager, KVCacheEntry
from nexus.core.llm.priority_queue import (
    InferencePriorityQueue, InferenceRequest, Priority,
)
from nexus.core.llm.token_streaming import (
    TokenStreamer, TokenEvent, TokenEventType, StreamSession,
)
from nexus.core.llm.fallback_chain import (
    FallbackChain, FallbackModel, SelectionStrategy, LatencyWindow,
)
from nexus.core.llm.deterministic_cache import DeterministicOutputCache
from nexus.core.llm.ensemble_consensus import (
    EnsembleConsensus, ConsensusMethod, ConsensusResult,
)
from nexus.core.llm.cost_tracker import (
    InferenceCostTracker, CostRecord, ModelPricing, BudgetAlert,
)


# ── ModelWarmupManager ────────────────────────────────────────

class TestModelWarmupManager:
    @pytest.mark.asyncio
    async def test_warmup_no_models(self):
        mgr = ModelWarmupManager()
        results = await mgr.warmup_all()
        assert results == {}
        assert mgr.is_ready

    @pytest.mark.asyncio
    async def test_warmup_dry_run(self):
        mgr = ModelWarmupManager()
        mgr.register_model("gpt-4", priority=1)
        mgr.register_model("claude-3", priority=0)
        results = await mgr.warmup_all()
        assert len(results) == 2
        assert all(r.status == WarmupStatus.READY for r in results.values())

    @pytest.mark.asyncio
    async def test_warmup_with_inference_fn(self):
        inference_fn = AsyncMock(return_value="warm")
        mgr = ModelWarmupManager(inference_fn=inference_fn)
        mgr.register_model("test-model")
        results = await mgr.warmup_all()
        assert results["test-model"].status == WarmupStatus.READY
        inference_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_warmup_failure_non_required(self):
        inference_fn = AsyncMock(side_effect=RuntimeError("fail"))
        mgr = ModelWarmupManager(inference_fn=inference_fn, max_retries=1)
        mgr.register_model("bad-model", required=False)
        results = await mgr.warmup_all()
        assert results["bad-model"].status == WarmupStatus.FAILED
        assert mgr.is_ready  # Non-required doesn't block

    @pytest.mark.asyncio
    async def test_warmup_failure_required(self):
        inference_fn = AsyncMock(side_effect=RuntimeError("fail"))
        mgr = ModelWarmupManager(inference_fn=inference_fn, max_retries=1)
        mgr.register_model("critical-model", required=True)
        with pytest.raises(RuntimeError, match="Required models failed"):
            await mgr.warmup_all()

    def test_get_ready_models(self):
        mgr = ModelWarmupManager()
        assert mgr.get_ready_models() == []


# ── KVCacheManager ────────────────────────────────────────────

class TestKVCacheManager:
    def test_put_and_get(self):
        cache = KVCacheManager(max_entries=100)
        cache.put("Hello system", "gpt-4", "kv_state", token_count=10, size_bytes=1000)
        entry = cache.get("Hello system", "gpt-4")
        assert entry is not None
        assert entry.token_count == 10
        assert entry.cache_state == "kv_state"

    def test_miss(self):
        cache = KVCacheManager()
        assert cache.get("nonexistent", "gpt-4") is None

    def test_model_isolation(self):
        cache = KVCacheManager()
        cache.put("prefix", "model-a", "state-a", token_count=5)
        assert cache.get("prefix", "model-a") is not None
        assert cache.get("prefix", "model-b") is None

    def test_ttl_expiry(self):
        cache = KVCacheManager(ttl_seconds=0.01)
        cache.put("prefix", "m", "state", token_count=5)
        time.sleep(0.02)
        assert cache.get("prefix", "m") is None

    def test_lru_eviction(self):
        cache = KVCacheManager(max_entries=2)
        cache.put("p1", "m", "s1", token_count=1)
        cache.put("p2", "m", "s2", token_count=2)
        cache.put("p3", "m", "s3", token_count=3)
        # p1 should be evicted
        stats = cache.get_stats()
        assert stats["total_entries"] == 2
        assert stats["evictions"] >= 1

    def test_longest_prefix(self):
        cache = KVCacheManager()
        cache.put("Hello, how are you?", "m", "long_state", token_count=5)
        cache.put("Hello", "m", "short_state", token_count=2)
        result = cache.get_longest_prefix("Hello, how are you? Fine thanks.", "m")
        assert result is not None
        assert result.prefix_text == "Hello, how are you?"

    def test_invalidate_all(self):
        cache = KVCacheManager()
        cache.put("a", "m1", "s1", token_count=1)
        cache.put("b", "m2", "s2", token_count=1)
        count = cache.invalidate()
        assert count == 2
        assert cache.get_stats()["total_entries"] == 0

    def test_invalidate_model(self):
        cache = KVCacheManager()
        cache.put("a", "m1", "s1", token_count=1)
        cache.put("b", "m2", "s2", token_count=1)
        count = cache.invalidate("m1")
        assert count == 1

    def test_stats(self):
        cache = KVCacheManager()
        cache.get("miss", "m")
        cache.put("hit", "m", "s", token_count=1)
        cache.get("hit", "m")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


# ── InferencePriorityQueue ────────────────────────────────────

class TestInferencePriorityQueue:
    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self):
        q = InferencePriorityQueue()
        await q.enqueue("Hello", priority=Priority.NORMAL)
        req = await q.dequeue()
        assert req.prompt == "Hello"
        assert req.priority == Priority.NORMAL

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        q = InferencePriorityQueue()
        await q.enqueue("low", priority=Priority.LOW)
        await q.enqueue("high", priority=Priority.HIGH)
        await q.enqueue("critical", priority=Priority.CRITICAL)

        r1 = await q.dequeue()
        r2 = await q.dequeue()
        r3 = await q.dequeue()
        assert r1.prompt == "critical"
        assert r2.prompt == "high"
        assert r3.prompt == "low"

    @pytest.mark.asyncio
    async def test_queue_full(self):
        q = InferencePriorityQueue(max_queue_size=2)
        await q.enqueue("a")
        await q.enqueue("b")
        with pytest.raises(asyncio.QueueFull):
            await q.enqueue("c")

    @pytest.mark.asyncio
    async def test_queue_depth(self):
        q = InferencePriorityQueue()
        await q.enqueue("a")
        await q.enqueue("b")
        assert q.queue_depth == 2

    def test_stats_initial(self):
        q = InferencePriorityQueue()
        stats = q.get_stats()
        assert stats["enqueued"] == 0


# ── TokenStreamer ─────────────────────────────────────────────

class TestTokenStreamer:
    @pytest.mark.asyncio
    async def test_basic_streaming(self):
        streamer = TokenStreamer(model_name="test-model")

        async def token_source():
            for word in ["Hello", " ", "world"]:
                yield {"token": word}

        events = []
        async for event in streamer.stream("sess-1", token_source(), prompt="test"):
            events.append(event)

        assert events[0].event_type == TokenEventType.START
        assert events[-1].event_type == TokenEventType.END
        token_events = [e for e in events if e.event_type == TokenEventType.TOKEN]
        assert len(token_events) == 3
        assert token_events[-1].cumulative_text == "Hello world"

    @pytest.mark.asyncio
    async def test_callbacks(self):
        tokens = []
        completed = []

        streamer = TokenStreamer(
            model_name="m",
            on_token=lambda e: tokens.append(e),
            on_complete=lambda s: completed.append(s),
        )

        async def source():
            yield {"token": "hi"}

        async for _ in streamer.stream("s1", source()):
            pass

        assert len(tokens) >= 3  # START + TOKEN + END
        assert len(completed) == 1

    @pytest.mark.asyncio
    async def test_cancel_stream(self):
        streamer = TokenStreamer()

        async def infinite_source():
            while True:
                yield {"token": "x"}

        events = []
        async for event in streamer.stream("cancel-me", infinite_source()):
            events.append(event)
            if len(events) == 3:  # START + 2 tokens
                streamer.cancel("cancel-me")

        assert len(events) <= 5

    def test_token_event_to_dict(self):
        e = TokenEvent(event_type=TokenEventType.TOKEN, token="hi", token_index=0)
        d = e.to_dict()
        assert d["event"] == "token"
        assert d["token"] == "hi"

    def test_token_event_to_sse(self):
        e = TokenEvent(event_type=TokenEventType.TOKEN, token="hi", token_index=0)
        sse = e.to_sse()
        assert "event: token" in sse
        assert "data:" in sse

    def test_session_tps(self):
        s = StreamSession(session_id="s", model_name="m", prompt="p")
        s.tokens_emitted = 10
        assert s.tokens_per_second >= 0


# ── FallbackChain ─────────────────────────────────────────────

class TestFallbackChain:
    def test_add_and_select_latency(self):
        chain = FallbackChain(strategy=SelectionStrategy.LATENCY)
        m1 = FallbackModel(name="fast", cost_per_1k_tokens=0.01)
        m1.latency.record(50)
        m2 = FallbackModel(name="slow", cost_per_1k_tokens=0.005)
        m2.latency.record(200)
        chain.add_model(m1)
        chain.add_model(m2)
        selected = chain.select_model()
        assert selected.name == "fast"

    def test_select_cost(self):
        chain = FallbackChain(strategy=SelectionStrategy.COST)
        chain.add_model(FallbackModel(name="expensive", cost_per_1k_tokens=0.03))
        chain.add_model(FallbackModel(name="cheap", cost_per_1k_tokens=0.001))
        assert chain.select_model().name == "cheap"

    def test_select_priority(self):
        chain = FallbackChain(strategy=SelectionStrategy.PRIORITY)
        chain.add_model(FallbackModel(name="secondary", priority=2))
        chain.add_model(FallbackModel(name="primary", priority=1))
        assert chain.select_model().name == "primary"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        chain = FallbackChain(strategy=SelectionStrategy.PRIORITY)
        chain.add_model(FallbackModel(name="m1", priority=1))
        inference = AsyncMock(return_value="result")
        result = await chain.execute(inference, "hello")
        assert result["result"] == "result"
        assert result["model_used"] == "m1"
        assert result["fallback_count"] == 0

    @pytest.mark.asyncio
    async def test_execute_fallback(self):
        chain = FallbackChain(strategy=SelectionStrategy.PRIORITY, max_retries=2)
        chain.add_model(FallbackModel(name="primary", priority=1))
        chain.add_model(FallbackModel(name="backup", priority=2))

        call_count = 0
        async def flaky_inference(model, prompt):
            nonlocal call_count
            call_count += 1
            if model == "primary":
                raise RuntimeError("Primary down")
            return "backup result"

        result = await chain.execute(flaky_inference, "test")
        assert result["model_used"] == "backup"
        assert result["fallback_count"] == 1

    @pytest.mark.asyncio
    async def test_all_models_fail(self):
        chain = FallbackChain(max_retries=1)
        chain.add_model(FallbackModel(name="m1"))
        inference = AsyncMock(side_effect=RuntimeError("fail"))
        with pytest.raises(RuntimeError, match="All models failed"):
            await chain.execute(inference, "test")

    def test_latency_window(self):
        w = LatencyWindow()
        w.record(100, success=True)
        w.record(200, success=True)
        w.record(300, success=False)
        assert w.p50 == 200
        assert w.success_rate == pytest.approx(2 / 3, abs=0.01)

    def test_remove_model(self):
        chain = FallbackChain()
        chain.add_model(FallbackModel(name="m"))
        chain.remove_model("m")
        assert chain.select_model() is None

    def test_stats(self):
        chain = FallbackChain()
        stats = chain.get_stats()
        assert "requests" in stats


# ── DeterministicOutputCache ──────────────────────────────────

class TestDeterministicOutputCache:
    def test_put_get(self):
        cache = DeterministicOutputCache()
        cache.put("gpt-4", "hello", "world", temperature=0.0)
        result = cache.get("gpt-4", "hello", temperature=0.0)
        assert result is not None
        assert result.response == "world"

    def test_miss(self):
        cache = DeterministicOutputCache()
        assert cache.get("gpt-4", "nonexistent") is None

    def test_non_deterministic_skipped(self):
        cache = DeterministicOutputCache()
        result = cache.put("gpt-4", "hello", "world", temperature=0.8)
        assert result is None

    def test_force_store(self):
        cache = DeterministicOutputCache()
        result = cache.put("gpt-4", "hello", "world", temperature=0.8, force=True)
        assert result is not None

    def test_seed_is_deterministic(self):
        cache = DeterministicOutputCache()
        assert cache.is_deterministic(temperature=1.0, seed=42)

    def test_ttl_expiry(self):
        cache = DeterministicOutputCache(default_ttl=0.01)
        cache.put("m", "p", "r", temperature=0.0)
        time.sleep(0.02)
        assert cache.get("m", "p") is None

    def test_invalidate_all(self):
        cache = DeterministicOutputCache()
        cache.put("m1", "p1", "r1", temperature=0.0)
        cache.put("m2", "p2", "r2", temperature=0.0)
        count = cache.invalidate()
        assert count == 2

    def test_invalidate_model(self):
        cache = DeterministicOutputCache()
        cache.put("m1", "p", "r", temperature=0.0)
        cache.put("m2", "p", "r", temperature=0.0)
        count = cache.invalidate("m1")
        assert count == 1

    def test_lru_eviction(self):
        cache = DeterministicOutputCache(max_entries=2)
        cache.put("m", "a", "r1", temperature=0.0)
        cache.put("m", "b", "r2", temperature=0.0)
        cache.put("m", "c", "r3", temperature=0.0)
        stats = cache.get_stats()
        assert stats["entries"] == 2
        assert stats["evictions"] >= 1

    def test_stats(self):
        cache = DeterministicOutputCache()
        cache.get("m", "miss")
        cache.put("m", "p", "r", temperature=0.0)
        cache.get("m", "p")
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


# ── EnsembleConsensus ─────────────────────────────────────────

class TestEnsembleConsensus:
    @pytest.mark.asyncio
    async def test_exact_match_consensus(self):
        ensemble = EnsembleConsensus(
            consensus_method=ConsensusMethod.EXACT_MATCH,
            threshold=0.5,
            quorum=2,
        )

        async def inference(model, prompt):
            return "42"

        result = await ensemble.run(
            ["m1", "m2", "m3"], inference, "what is the answer?"
        )
        assert result.consensus_reached
        assert result.consensus_response == "42"
        assert result.agreement_ratio >= 0.5

    @pytest.mark.asyncio
    async def test_no_consensus(self):
        ensemble = EnsembleConsensus(
            consensus_method=ConsensusMethod.EXACT_MATCH,
            threshold=1.0,
            quorum=2,
        )

        responses = iter(["a", "b", "c"])

        async def inference(model, prompt):
            return next(responses)

        result = await ensemble.run(
            ["m1", "m2", "m3"], inference, "test"
        )
        assert not result.consensus_reached

    @pytest.mark.asyncio
    async def test_majority_vote(self):
        ensemble = EnsembleConsensus(
            consensus_method=ConsensusMethod.MAJORITY_VOTE,
            threshold=0.5,
            quorum=2,
        )

        votes = iter(["yes", "yes", "no"])

        async def inference(model, prompt):
            return next(votes)

        result = await ensemble.run(
            ["m1", "m2", "m3"], inference, "test"
        )
        assert result.consensus_reached
        assert result.consensus_response == "yes"

    @pytest.mark.asyncio
    async def test_early_termination(self):
        ensemble = EnsembleConsensus(
            consensus_method=ConsensusMethod.EXACT_MATCH,
            threshold=0.5,
            quorum=2,
            timeout_seconds=5.0,
        )

        async def fast_inference(model, prompt):
            if model in ("m1", "m2"):
                return "same"
            await asyncio.sleep(10)  # Would time out
            return "different"

        result = await ensemble.run(
            ["m1", "m2", "m3"], fast_inference, "test"
        )
        assert result.consensus_reached
        # m3 should have been cancelled
        assert result.total_latency_ms < 5000

    def test_stats(self):
        e = EnsembleConsensus()
        stats = e.get_stats()
        assert stats["runs"] == 0


# ── InferenceCostTracker ──────────────────────────────────────

class TestInferenceCostTracker:
    def test_record_cost(self):
        tracker = InferenceCostTracker()
        tracker.set_pricing(ModelPricing("gpt-4", cost_per_1k_input=0.03, cost_per_1k_output=0.06))
        record = tracker.record("r1", "gpt-4", input_tokens=1000, output_tokens=500)
        assert record.cost_usd == pytest.approx(0.03 + 0.03)

    def test_no_pricing(self):
        tracker = InferenceCostTracker()
        record = tracker.record("r1", "unknown", input_tokens=100, output_tokens=50)
        assert record.cost_usd == 0.0

    def test_tenant_tracking(self):
        tracker = InferenceCostTracker()
        tracker.set_pricing(ModelPricing("m", cost_per_1k_input=0.01))
        tracker.record("r1", "m", input_tokens=1000, output_tokens=0, tenant_id="t1")
        tracker.record("r2", "m", input_tokens=2000, output_tokens=0, tenant_id="t2")
        assert tracker.get_total_cost("t1") == pytest.approx(0.01)
        assert tracker.get_total_cost("t2") == pytest.approx(0.02)

    def test_budget_alert(self):
        alerts_fired = []
        tracker = InferenceCostTracker(alert_check_interval=1)
        tracker.set_pricing(ModelPricing("m", cost_per_1k_input=1.0))
        tracker.add_alert(BudgetAlert(
            name="test-alert",
            threshold_usd=0.5,
            callback=lambda name, cur, thresh: alerts_fired.append((name, cur)),
        ))
        tracker.record("r1", "m", input_tokens=1000, output_tokens=0)
        assert len(alerts_fired) == 1
        assert alerts_fired[0][0] == "test-alert"

    def test_model_breakdown(self):
        tracker = InferenceCostTracker()
        tracker.set_pricing(ModelPricing("a", cost_per_1k_input=0.01))
        tracker.set_pricing(ModelPricing("b", cost_per_1k_input=0.02))
        tracker.record("r1", "a", input_tokens=1000, output_tokens=0)
        tracker.record("r2", "b", input_tokens=1000, output_tokens=0)
        breakdown = tracker.get_cost_by_model()
        assert "a" in breakdown
        assert "b" in breakdown

    def test_summary(self):
        tracker = InferenceCostTracker()
        tracker.set_pricing(ModelPricing("m", cost_per_1k_input=0.01))
        tracker.record("r1", "m", input_tokens=1000, output_tokens=0)
        summary = tracker.get_cost_summary()
        assert summary["total_requests"] == 1

    def test_stats(self):
        tracker = InferenceCostTracker()
        stats = tracker.get_stats()
        assert stats["total_requests"] == 0
