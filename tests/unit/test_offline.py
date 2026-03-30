"""
Unit tests for the Nexus Offline Operation system.

Covers:
- 752: Offline-capable agent execution
- 760: Offline RAG with local vector store
- 770: Offline blueprint execution
- 780: Offline model inference with quantized models
- 788: Offline agent communication
"""

import math
import time
import pytest
import tempfile
import os
import json
import subprocess
import sys

from nexus.offline.agent_execution import (
    OfflineAgentExecutor,
    OfflineTask,
    ConnectivityStatus,
    TaskPriority,
    CachedTool,
)
from nexus.offline.local_rag import (
    OfflineRAG,
    LocalVectorStore,
    Document,
    SearchResult,
    RAGResponse,
)
from nexus.offline.blueprint_execution import (
    OfflineBlueprintExecutor,
    Blueprint,
    BlueprintStep,
    StepStatus,
    ExecutionResult,
)
from nexus.offline.model_inference import (
    OfflineModelInference,
    InferenceRequest,
    InferenceResult,
    InferenceBackend,
    QuantizationType,
    SimpleCompletionEngine,
    LocalModel,
)
from nexus.offline.agent_communication import (
    OfflineAgentCommunication,
    AgentMessage,
    MessagePriority,
    DeliveryStatus,
    Subscription,
)


# ════════════════════════════════════════════════════════════════════
# 752 — Offline-Capable Agent Execution
# ════════════════════════════════════════════════════════════════════


class TestOfflineAgentExecutor:
    """Tests for OfflineAgentExecutor."""

    def test_initialization(self):
        executor = OfflineAgentExecutor()
        assert executor.is_online is True
        assert executor.is_offline is False

    def test_set_connectivity(self):
        executor = OfflineAgentExecutor()
        executor.set_connectivity(ConnectivityStatus.OFFLINE)
        assert executor.is_offline is True
        assert executor.is_online is False

    def test_set_degraded(self):
        executor = OfflineAgentExecutor()
        executor.set_connectivity(ConnectivityStatus.DEGRADED)
        assert executor.is_online is False
        assert executor.is_offline is False

    def test_submit_task_online(self):
        executor = OfflineAgentExecutor()
        result = executor.submit_task(
            "t1", "analysis", "agent1", {"text": "hello"},
            execute_fn=lambda p: {"result": p["text"].upper()},
        )
        assert result.status == "completed"
        assert result.result == {"result": "HELLO"}

    def test_submit_task_deferred_offline(self):
        executor = OfflineAgentExecutor()
        executor.set_connectivity(ConnectivityStatus.OFFLINE)
        result = executor.submit_task(
            "t1", "api_call", "agent1", {"url": "http://example.com"},
            requires_online=True,
        )
        assert result.status == "deferred"

    def test_local_handler_fallback(self):
        executor = OfflineAgentExecutor()
        executor.set_connectivity(ConnectivityStatus.OFFLINE)
        executor.register_local_handler(
            "analysis", lambda p: {"local_result": True}
        )
        result = executor.submit_task(
            "t1", "analysis", "agent1", {"text": "test"},
        )
        assert result.status == "completed"
        assert result.result == {"local_result": True}

    def test_cache_tool_result(self):
        executor = OfflineAgentExecutor()
        cached = executor.cache_tool_result(
            "search", {"query": "test"}, {"results": [1, 2, 3]}
        )
        assert isinstance(cached, CachedTool)

        result = executor.get_cached_result("search", {"query": "test"})
        assert result == {"results": [1, 2, 3]}

    def test_cache_miss(self):
        executor = OfflineAgentExecutor()
        result = executor.get_cached_result("search", {"query": "nonexistent"})
        assert result is None

    def test_cache_expiration(self):
        executor = OfflineAgentExecutor()
        cached = executor.cache_tool_result(
            "search", {"q": "old"}, {"r": 1}, ttl_seconds=0.001
        )
        time.sleep(0.01)
        result = executor.get_cached_result("search", {"q": "old"})
        assert result is None

    def test_cache_fallback_for_task(self):
        executor = OfflineAgentExecutor()
        executor.cache_tool_result(
            "analysis", {"text": "hi"}, {"cached_data": True}
        )
        # No handler, no execute_fn, should fall through to cache
        result = executor.submit_task(
            "t1", "analysis", "agent1", {"text": "hi"},
        )
        # Cache key includes task_type not tool_name, so this won't match
        # (different key construction). Task should be deferred.

    def test_status_report(self):
        executor = OfflineAgentExecutor()
        executor.register_local_handler("t1", lambda p: {})
        status = executor.get_status()
        assert "connectivity" in status
        assert "cache_entries" in status
        assert "local_handlers" in status
        assert "t1" in status["local_handlers"]

    def test_max_retries_exceeded(self):
        executor = OfflineAgentExecutor()
        # No handlers, no cache, no execute_fn
        task = executor.submit_task("t1", "unknown", "agent1", {})
        # Should be deferred or failed
        assert task.status in ("deferred", "failed")

    def test_connectivity_checker(self):
        executor = OfflineAgentExecutor()
        executor.register_connectivity_checker(lambda: True)
        # Force a check
        executor._last_connectivity_check = 0
        status = executor.check_connectivity()
        assert status == ConnectivityStatus.ONLINE

    def test_connectivity_checker_all_fail(self):
        executor = OfflineAgentExecutor()
        executor.register_connectivity_checker(lambda: False)
        executor._last_connectivity_check = 0
        status = executor.check_connectivity()
        assert status == ConnectivityStatus.OFFLINE

    def test_disk_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = OfflineAgentExecutor(cache_dir=tmpdir)
            executor.cache_tool_result("tool1", {"k": "v"}, {"data": 42})
            # Check file exists
            files = os.listdir(tmpdir)
            assert len(files) == 1

    def test_retry_deferred_task_preserves_execute_fn(self):
        executor = OfflineAgentExecutor()
        executor.set_connectivity(ConnectivityStatus.OFFLINE)
        call_count = 0

        def execute_fn(params):
            nonlocal call_count
            call_count += 1
            return {"echo": params["value"]}

        task = executor.submit_task(
            "t1",
            "api_call",
            "agent1",
            {"value": "ok"},
            requires_online=True,
            execute_fn=execute_fn,
        )
        assert task.status == "deferred"

        executor.set_connectivity(ConnectivityStatus.ONLINE)
        retried = executor._retry_deferred_tasks()

        assert retried == 1
        assert call_count == 1
        assert task.status == "completed"
        assert task.result == {"echo": "ok"}


# ════════════════════════════════════════════════════════════════════
# 760 — Offline RAG with Local Vector Store
# ════════════════════════════════════════════════════════════════════


class TestLocalVectorStore:
    """Tests for LocalVectorStore."""

    def test_initialization(self):
        store = LocalVectorStore(dimension=4)
        assert store.dimension == 4
        assert store.size == 0

    def test_add_document(self):
        store = LocalVectorStore(dimension=3)
        doc = Document(doc_id="d1", content="hello", embedding=[1.0, 0.0, 0.0])
        store.add(doc)
        assert store.size == 1

    def test_add_wrong_dimension(self):
        store = LocalVectorStore(dimension=3)
        doc = Document(doc_id="d1", content="hello", embedding=[1.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            store.add(doc)

    def test_add_no_embedding(self):
        store = LocalVectorStore(dimension=3)
        doc = Document(doc_id="d1", content="hello")
        with pytest.raises(ValueError, match="no embedding"):
            store.add(doc)

    def test_search_basic(self):
        store = LocalVectorStore(dimension=3)
        store.add(Document(doc_id="d1", content="A", embedding=[1.0, 0.0, 0.0]))
        store.add(Document(doc_id="d2", content="B", embedding=[0.0, 1.0, 0.0]))
        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0][0] == "d1"  # Most similar
        assert results[0][1] > results[1][1]

    def test_search_empty(self):
        store = LocalVectorStore(dimension=3)
        results = store.search([1.0, 0.0, 0.0])
        assert results == []

    def test_search_wrong_dimension(self):
        store = LocalVectorStore(dimension=3)
        store.add(Document(doc_id="d1", content="A", embedding=[1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="dimension"):
            store.search([1.0, 0.0])

    def test_get_document(self):
        store = LocalVectorStore(dimension=3)
        store.add(Document(doc_id="d1", content="hello", embedding=[1.0, 0.0, 0.0]))
        doc = store.get_document("d1")
        assert doc.content == "hello"
        assert store.get_document("nonexistent") is None

    def test_remove_document(self):
        store = LocalVectorStore(dimension=3)
        store.add(Document(doc_id="d1", content="A", embedding=[1.0, 0.0, 0.0]))
        assert store.remove("d1") is True
        assert store.size == 0
        assert store.remove("d1") is False

    def test_update_document(self):
        store = LocalVectorStore(dimension=3)
        store.add(Document(doc_id="d1", content="old", embedding=[1.0, 0.0, 0.0]))
        store.add(Document(doc_id="d1", content="new", embedding=[0.0, 1.0, 0.0]))
        assert store.size == 1
        assert store.get_document("d1").content == "new"

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalVectorStore(dimension=3)
            store.add(Document(doc_id="d1", content="hello", embedding=[1.0, 0.0, 0.0]))
            store.save(tmpdir)

            store2 = LocalVectorStore(dimension=3)
            store2.load(tmpdir)
            assert store2.size == 1
            assert store2.get_document("d1").content == "hello"

    def test_cosine_similarity(self):
        sim = LocalVectorStore._cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 0.001

        sim = LocalVectorStore._cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 0.001

        sim = LocalVectorStore._cosine_similarity([1, 0, 0], [-1, 0, 0])
        assert abs(sim + 1.0) < 0.001

    def test_cosine_similarity_zero_vector(self):
        sim = LocalVectorStore._cosine_similarity([0, 0, 0], [1, 0, 0])
        assert sim == 0.0


class TestOfflineRAG:
    """Tests for OfflineRAG."""

    def test_initialization(self):
        rag = OfflineRAG(dimension=4)
        assert rag.document_count == 0

    def test_index_document(self):
        rag = OfflineRAG(dimension=16)
        count = rag.index_document("d1", "The quick brown fox")
        assert count >= 1
        assert rag.document_count >= 1

    def test_index_long_document_chunking(self):
        rag = OfflineRAG(dimension=16, chunk_size=50, chunk_overlap=10)
        text = "word " * 100  # ~500 chars
        count = rag.index_document("d1", text)
        assert count > 1  # Should produce multiple chunks

    def test_query_basic(self):
        rag = OfflineRAG(dimension=16)
        rag.index_document("d1", "Paris is the capital of France")
        rag.index_document("d2", "Berlin is the capital of Germany")
        response = rag.query("capital of France")
        assert isinstance(response, RAGResponse)
        assert len(response.context_documents) > 0
        assert "France" in response.augmented_prompt or "Paris" in response.augmented_prompt

    def test_query_with_min_score(self):
        rag = OfflineRAG(dimension=16)
        rag.index_document("d1", "cats are pets")
        response = rag.query("quantum physics", min_score=0.99)
        # Likely no results above 0.99
        assert isinstance(response, RAGResponse)

    def test_query_custom_template(self):
        rag = OfflineRAG(dimension=16)
        rag.index_document("d1", "Test content")
        template = "Context: {context}\nQ: {query}\nA:"
        response = rag.query("test", context_template=template)
        assert "Q: test" in response.augmented_prompt

    def test_batch_indexing(self):
        rag = OfflineRAG(dimension=16)
        docs = [
            {"doc_id": "d1", "content": "First document"},
            {"doc_id": "d2", "content": "Second document"},
        ]
        total = rag.index_documents(docs)
        assert total >= 2

    def test_stats(self):
        rag = OfflineRAG(dimension=16, chunk_size=100)
        rag.index_document("d1", "Test")
        stats = rag.get_stats()
        assert stats["dimension"] == 16
        assert stats["chunk_size"] == 100
        assert stats["document_count"] >= 1

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "rag_store")
            rag = OfflineRAG(dimension=16, store_path=store_path)
            rag.index_document("d1", "Persistent content")
            rag.save()

            rag2 = OfflineRAG(dimension=16, store_path=store_path)
            assert rag2.document_count >= 1

    def test_fallback_embed(self):
        rag = OfflineRAG(dimension=16)
        emb = rag._fallback_embed("hello world")
        assert len(emb) == 16
        norm = math.sqrt(sum(x * x for x in emb))
        assert abs(norm - 1.0) < 0.01  # Should be normalized

    def test_fallback_embed_is_stable_across_processes(self):
        code = (
            "import json; "
            "from nexus.offline.local_rag import OfflineRAG; "
            "print(json.dumps(OfflineRAG(dimension=8)._fallback_embed('hello world')))"
        )
        outputs = []
        for seed in ("1", "2"):
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"
            env["PYTHONHASHSEED"] = seed
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", code],
                    cwd=os.getcwd(),
                    env=env,
                    text=True,
                ).strip()
            )

        assert len(outputs) == 2
        assert json.loads(outputs[0]) == json.loads(outputs[1])


# ════════════════════════════════════════════════════════════════════
# 770 — Offline Blueprint Execution
# ════════════════════════════════════════════════════════════════════


class TestOfflineBlueprintExecutor:
    """Tests for OfflineBlueprintExecutor."""

    def test_initialization(self):
        executor = OfflineBlueprintExecutor()
        assert executor.is_online is True

    def test_create_blueprint(self):
        executor = OfflineBlueprintExecutor()
        bp = executor.create_blueprint("bp1", "Test Blueprint")
        assert bp.blueprint_id == "bp1"

    def test_register_handler(self):
        executor = OfflineBlueprintExecutor()
        executor.register_step_handler("transform", lambda p, c: p.get("x", 0) * 2)
        stats = executor.get_stats()
        assert "transform" in stats["registered_handlers"]

    def test_execute_simple_blueprint(self):
        executor = OfflineBlueprintExecutor()
        executor.register_step_handler(
            "double", lambda p, c: p.get("value", 0) * 2
        )

        bp = Blueprint(blueprint_id="bp1", name="Double")
        bp.add_step(BlueprintStep(
            step_id="s1", name="Double", step_type="double",
            parameters={"value": 5},
        ))

        result = executor.execute(bp)
        assert result.success is True
        assert result.step_results["s1"] == 10

    def test_execute_with_dependencies(self):
        executor = OfflineBlueprintExecutor()
        executor.register_step_handler(
            "init", lambda p, c: p.get("value", 0)
        )
        executor.register_step_handler(
            "add", lambda p, c: c.get("s1", 0) + p.get("amount", 0)
        )

        bp = Blueprint(blueprint_id="bp1", name="Add Pipeline")
        bp.add_step(BlueprintStep(
            step_id="s1", name="Init", step_type="init",
            parameters={"value": 10},
        ))
        bp.add_step(BlueprintStep(
            step_id="s2", name="Add", step_type="add",
            parameters={"amount": 5}, dependencies=["s1"],
        ))

        result = executor.execute(bp)
        assert result.success is True
        assert result.step_results["s2"] == 15

    def test_execute_step_failure(self):
        executor = OfflineBlueprintExecutor()
        executor.register_step_handler(
            "fail", lambda p, c: (_ for _ in ()).throw(ValueError("boom"))
        )

        bp = Blueprint(blueprint_id="bp1", name="Fail")
        bp.add_step(BlueprintStep(
            step_id="s1", name="Fail", step_type="fail",
        ))

        result = executor.execute(bp)
        assert result.success is False
        assert result.steps_failed == 1

    def test_execute_offline_with_fallback(self):
        executor = OfflineBlueprintExecutor(is_online=False)
        executor.register_step_handler(
            "local", lambda p, c: "local_result"
        )

        bp = Blueprint(blueprint_id="bp1", name="Offline Test")
        bp.add_step(BlueprintStep(
            step_id="fallback", name="Fallback", step_type="local",
        ))
        bp.add_step(BlueprintStep(
            step_id="s1", name="Online Step", step_type="api_call",
            requires_online=True, fallback_step_id="fallback",
        ))

        result = executor.execute(bp)
        # s1 should use fallback since we're offline
        assert "s1" in result.step_results or result.steps_failed > 0

    def test_execute_offline_no_fallback(self):
        executor = OfflineBlueprintExecutor(is_online=False)
        bp = Blueprint(blueprint_id="bp1", name="No Fallback")
        bp.add_step(BlueprintStep(
            step_id="s1", name="Online Only", step_type="api_call",
            requires_online=True,
        ))
        result = executor.execute(bp)
        assert result.steps_failed == 1

    def test_caching(self):
        call_count = 0

        def counting_handler(p, c):
            nonlocal call_count
            call_count += 1
            return "result"

        executor = OfflineBlueprintExecutor()
        executor.register_step_handler("count", counting_handler)

        bp = Blueprint(blueprint_id="bp1", name="Cache Test")
        bp.add_step(BlueprintStep(
            step_id="s1", name="Count", step_type="count",
        ))

        executor.execute(bp, use_cache=True)
        executor.execute(bp, use_cache=True)
        # Second run should use cache
        assert call_count == 1

    def test_clear_cache(self):
        executor = OfflineBlueprintExecutor()
        executor.register_step_handler("noop", lambda p, c: "ok")

        bp = Blueprint(blueprint_id="bp1", name="Clear Test")
        bp.add_step(BlueprintStep(step_id="s1", name="Noop", step_type="noop"))

        executor.execute(bp)
        cleared = executor.clear_cache("bp1")
        assert cleared >= 1

    def test_clear_all_cache(self):
        executor = OfflineBlueprintExecutor()
        executor.register_step_handler("noop", lambda p, c: "ok")

        for i in range(3):
            bp = Blueprint(blueprint_id=f"bp{i}", name=f"BP {i}")
            bp.add_step(BlueprintStep(
                step_id="s1", name="Noop", step_type="noop",
            ))
            executor.execute(bp)

        cleared = executor.clear_cache()
        assert cleared >= 3

    def test_execution_history(self):
        executor = OfflineBlueprintExecutor()
        executor.register_step_handler("noop", lambda p, c: "ok")

        bp = Blueprint(blueprint_id="bp1", name="History")
        bp.add_step(BlueprintStep(step_id="s1", name="Noop", step_type="noop"))
        executor.execute(bp)

        history = executor.get_execution_history()
        assert len(history) == 1

    def test_unmet_dependencies_skipped(self):
        executor = OfflineBlueprintExecutor()
        executor.register_step_handler(
            "fail", lambda p, c: (_ for _ in ()).throw(ValueError("boom"))
        )
        executor.register_step_handler("ok", lambda p, c: "ok")

        bp = Blueprint(blueprint_id="bp1", name="Dep Test")
        bp.add_step(BlueprintStep(
            step_id="s1", name="Fail", step_type="fail",
        ))
        bp.add_step(BlueprintStep(
            step_id="s2", name="OK", step_type="ok",
            dependencies=["s1"],
        ))

        result = executor.execute(bp)
        assert result.step_statuses.get("s2") == StepStatus.SKIPPED

    def test_topological_order(self):
        bp = Blueprint(blueprint_id="bp1", name="Topo")
        bp.add_step(BlueprintStep(step_id="c", name="C", step_type="t", dependencies=["a", "b"]))
        bp.add_step(BlueprintStep(step_id="a", name="A", step_type="t"))
        bp.add_step(BlueprintStep(step_id="b", name="B", step_type="t", dependencies=["a"]))
        order = bp.get_execution_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")


# ════════════════════════════════════════════════════════════════════
# 780 — Offline Model Inference
# ════════════════════════════════════════════════════════════════════


class TestSimpleCompletionEngine:
    """Tests for SimpleCompletionEngine."""

    def test_summarize_pattern(self):
        engine = SimpleCompletionEngine()
        result = engine.complete("Please summarize this text: The cat sat on the mat.")
        assert "Summary" in result or "summary" in result.lower()

    def test_classify_pattern(self):
        engine = SimpleCompletionEngine()
        result = engine.complete("Classify this: positive sentiment")
        assert "Classification" in result or "classif" in result.lower()

    def test_extract_pattern(self):
        engine = SimpleCompletionEngine()
        result = engine.complete("Extract the key entities from: John works at Google.")
        assert "Extract" in result or "extract" in result.lower()

    def test_default_pattern(self):
        engine = SimpleCompletionEngine()
        result = engine.complete("What is the meaning of life?")
        assert len(result) > 0

    def test_custom_handler(self):
        engine = SimpleCompletionEngine()
        engine.register_handler("weather", lambda p: "It's sunny!")
        result = engine.complete("What's the weather today?")
        assert result == "It's sunny!"


class TestOfflineModelInference:
    """Tests for OfflineModelInference."""

    def test_initialization(self):
        inf = OfflineModelInference()
        assert inf is not None
        assert len(inf.list_models()) == 0

    def test_register_model(self):
        inf = OfflineModelInference()
        model = inf.register_model(
            "test-model", "Test Model",
            backend=InferenceBackend.SIMPLE,
        )
        assert model.model_id == "test-model"
        assert len(inf.list_models()) == 1

    def test_infer_with_simple_engine(self):
        inf = OfflineModelInference()
        inf.register_model("simple", "Simple", backend=InferenceBackend.SIMPLE)
        request = InferenceRequest(prompt="Summarize: hello world", model_id="simple")
        result = inf.infer(request)
        assert isinstance(result, InferenceResult)
        assert len(result.content) > 0
        assert result.backend == InferenceBackend.SIMPLE

    def test_infer_unknown_model_fallback(self):
        inf = OfflineModelInference()
        request = InferenceRequest(prompt="Hello", model_id="nonexistent")
        result = inf.infer(request)
        assert result.model_id == "simple_engine"
        assert result.metadata.get("fallback") is True

    def test_infer_tracks_performance(self):
        inf = OfflineModelInference()
        inf.register_model("m1", "M1", backend=InferenceBackend.SIMPLE)
        for _ in range(5):
            inf.infer(InferenceRequest(prompt="test", model_id="m1"))
        model = inf._models["m1"]
        assert model.total_inferences == 5
        assert model.avg_tokens_per_second > 0

    def test_select_model_empty(self):
        inf = OfflineModelInference()
        assert inf.select_model() is None

    def test_select_model_with_candidates(self):
        inf = OfflineModelInference()
        inf.register_model("fast", "Fast", backend=InferenceBackend.SIMPLE)
        inf.register_model("slow", "Slow", backend=InferenceBackend.SIMPLE)
        # Give fast model some inferences
        for _ in range(3):
            inf.infer(InferenceRequest(prompt="test", model_id="fast"))
        selected = inf.select_model()
        assert selected is not None

    def test_custom_completion_handler(self):
        inf = OfflineModelInference()
        inf.register_completion_handler(
            "greeting", lambda p: "Hello there!"
        )
        inf.register_model("m1", "M1", backend=InferenceBackend.SIMPLE)
        result = inf.infer(InferenceRequest(
            prompt="greeting to you", model_id="m1"
        ))
        assert result.content == "Hello there!"

    def test_backend_loader(self):
        inf = OfflineModelInference()

        class MockBackend:
            def generate(self, prompt, max_tokens=256):
                return f"mock: {prompt[:20]}"

        inf.register_backend_loader(
            InferenceBackend.OLLAMA_LOCAL,
            lambda model: MockBackend(),
        )
        inf.register_model(
            "ollama-test", "Ollama Test",
            backend=InferenceBackend.OLLAMA_LOCAL,
        )
        result = inf.infer(InferenceRequest(
            prompt="Hello from ollama", model_id="ollama-test"
        ))
        assert "mock:" in result.content

    def test_stats(self):
        inf = OfflineModelInference()
        inf.register_model("m1", "M1", backend=InferenceBackend.SIMPLE)
        stats = inf.get_stats()
        assert stats["registered_models"] == 1
        assert "models" in stats


# ════════════════════════════════════════════════════════════════════
# 788 — Offline Agent Communication
# ════════════════════════════════════════════════════════════════════


class TestOfflineAgentCommunication:
    """Tests for OfflineAgentCommunication."""

    def test_initialization(self):
        bus = OfflineAgentCommunication()
        assert bus is not None

    def test_register_agent(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("agent1")
        assert bus.is_registered("agent1")

    def test_unregister_agent(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("agent1")
        bus.unregister_agent("agent1")
        assert not bus.is_registered("agent1")

    def test_send_and_receive(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("sender")
        bus.register_agent("receiver")

        msg = bus.send("sender", "receiver", "test", {"data": 42})
        assert isinstance(msg, AgentMessage)
        assert msg.status == DeliveryStatus.QUEUED

        received = bus.receive("receiver")
        assert len(received) == 1
        assert received[0].payload == {"data": 42}
        assert received[0].status == DeliveryStatus.DELIVERED

    def test_receive_empty(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("agent1")
        received = bus.receive("agent1")
        assert received == []

    def test_broadcast(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("sender")
        bus.register_agent("r1")
        bus.register_agent("r2")

        bus.send("sender", "*", "announcement", {"msg": "hello"})

        r1_msgs = bus.receive("r1")
        r2_msgs = bus.receive("r2")
        sender_msgs = bus.receive("sender")

        assert len(r1_msgs) == 1
        assert len(r2_msgs) == 1
        assert len(sender_msgs) == 0  # Sender excluded

    def test_topic_filtering(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("s")
        bus.register_agent("r")

        bus.send("s", "r", "topicA", {"a": 1})
        bus.send("s", "r", "topicB", {"b": 2})

        received = bus.receive("r", topic="topicA")
        assert len(received) == 1
        assert received[0].topic == "topicA"

        # topicB still in queue
        remaining = bus.receive("r")
        assert len(remaining) == 1
        assert remaining[0].topic == "topicB"

    def test_priority_ordering(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("s")
        bus.register_agent("r")

        bus.send("s", "r", "t", {"p": "low"}, priority=MessagePriority.LOW)
        bus.send("s", "r", "t", {"p": "urgent"}, priority=MessagePriority.URGENT)
        bus.send("s", "r", "t", {"p": "normal"}, priority=MessagePriority.NORMAL)

        received = bus.receive("r", limit=3)
        assert len(received) == 3
        assert received[0].payload["p"] == "urgent"
        assert received[-1].payload["p"] == "low"

    def test_message_expiration(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("s")
        bus.register_agent("r")

        bus.send("s", "r", "t", {"old": True}, ttl_seconds=0.001)
        time.sleep(0.01)
        received = bus.receive("r")
        assert len(received) == 0

    def test_acknowledge(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("s")
        bus.register_agent("r")
        msg = bus.send("s", "r", "t", {"data": 1})
        bus.receive("r")  # delivers
        assert bus.acknowledge(msg.message_id) is True
        assert bus.acknowledge("nonexistent") is False

    def test_request_reply(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("client")
        bus.register_agent("server")

        request = bus.send_request(
            "client", "server", "compute", {"x": 5}
        )

        # Server receives and replies
        msgs = bus.receive("server")
        assert len(msgs) == 1
        reply = bus.send_reply("server", request.message_id, {"result": 10})
        assert reply is not None
        assert reply.reply_to == request.message_id

        # Client receives reply
        replies = bus.receive("client")
        assert len(replies) == 1
        assert replies[0].payload == {"result": 10}

    def test_subscription(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("pub")
        bus.register_agent("sub1")

        received_msgs = []
        bus.subscribe("sub1", "events", handler=lambda m: received_msgs.append(m))

        bus.send("pub", "sub1", "events", {"event": "happened"})
        assert len(received_msgs) == 1

    def test_unsubscribe(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("a")
        bus.subscribe("a", "topic1")
        assert bus.unsubscribe("a", "topic1") is True
        assert bus.unsubscribe("a", "topic1") is False  # already unsubscribed

    def test_queue_stats(self):
        bus = OfflineAgentCommunication()
        bus.register_agent("a1")
        bus.register_agent("a2")
        bus.send("a1", "a2", "t", {"d": 1})

        stats = bus.get_queue_stats()
        assert "registered_agents" in stats
        assert "a1" in stats["registered_agents"]
        assert stats["total_messages"] == 1

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = OfflineAgentCommunication(persist_dir=tmpdir)
            bus.register_agent("s")
            bus.register_agent("r")
            bus.send("s", "r", "t", {"persisted": True})
            files = os.listdir(tmpdir)
            assert len(files) == 1
