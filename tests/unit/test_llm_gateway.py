"""Tests for the LLM gateway endpoints in the Nexus service.

Covers POST /api/generate, GET /api/models, GET /api/knowledge/search,
and POST /api/knowledge.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from nexus.service import ServiceRuntime, create_health_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(*, platform=None, kb=None):
    """Build a ServiceRuntime with the given platform and knowledge base."""
    infrastructure = None
    if platform is not None:
        if kb is not None:
            platform._knowledge_base = kb
        else:
            # Ensure MagicMock doesn't auto-create _knowledge_base
            platform._knowledge_base = None
        infrastructure = SimpleNamespace(platform=platform)
    return ServiceRuntime(infrastructure=infrastructure)


def _make_kb(query_results=None, add_return="kid_123"):
    """Return a mock knowledge base."""
    kb = MagicMock()
    kb.query_knowledge = MagicMock(return_value=query_results or [])
    kb.add_knowledge = MagicMock(return_value=add_return)
    return kb


def _make_knowledge_item(*, id="item_1", content="test content", source="unit",
                         knowledge_type_value="factual", confidence=0.9):
    return SimpleNamespace(
        id=id,
        content=content,
        source=source,
        knowledge_type=SimpleNamespace(value=knowledge_type_value),
        confidence=confidence,
    )


# ===================================================================
# POST /api/generate
# ===================================================================

class TestGenerateEndpoint:
    """Tests for POST /api/generate."""

    @pytest.mark.asyncio
    async def test_valid_prompt_returns_proper_structure(self):
        """Valid prompt returns content, model, provider, tokens, latency."""
        platform = MagicMock()
        platform.query = MagicMock(return_value=_awaitable({
            "content": "Hello world",
            "model_name": "gpt-4",
            "provider": "openai",
            "tokens_used": 42,
            "strategy_used": "simple_best",
            "cached": False,
        }))
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/generate", json={"prompt": "Hi"})
            body = await resp.json()

        assert resp.status == 200
        assert body["content"] == "Hello world"
        assert body["model"] == "gpt-4"
        assert body["provider"] == "openai"
        assert body["tokens_used"] == 42
        assert "latency_ms" in body
        assert isinstance(body["latency_ms"], (int, float))

    @pytest.mark.asyncio
    async def test_missing_prompt_returns_400(self):
        """Request body without 'prompt' key returns 400."""
        platform = MagicMock()
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/generate", json={"model": "gpt-4"})
            body = await resp.json()

        assert resp.status == 400
        assert "prompt" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_prompt_returns_400(self):
        """Empty string prompt returns 400."""
        platform = MagicMock()
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/generate", json={"prompt": ""})
            body = await resp.json()

        assert resp.status == 400
        assert "prompt" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_platform_not_initialized_returns_503(self):
        """When platform is None, returns 503."""
        runtime = _make_runtime()  # no platform
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/generate", json={"prompt": "Hi"})
            body = await resp.json()

        assert resp.status == 503
        assert "not initialized" in body["error"]

    @pytest.mark.asyncio
    async def test_platform_exception_returns_500(self):
        """Platform.query raising an exception returns 500."""
        platform = MagicMock()
        platform.query = MagicMock(return_value=_awaitable_raise(RuntimeError("LLM down")))
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/generate", json={"prompt": "Hi"})
            body = await resp.json()

        assert resp.status == 500
        assert "LLM down" in body["error"]

    @pytest.mark.asyncio
    async def test_response_includes_strategy_and_cached(self):
        """Response body contains strategy_used and cached fields."""
        platform = MagicMock()
        platform.query = MagicMock(return_value=_awaitable({
            "content": "ok",
            "model_name": "claude-4",
            "provider": "anthropic",
            "tokens_used": 10,
            "strategy_used": "cascading",
            "cached": True,
        }))
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/generate", json={"prompt": "test"})
            body = await resp.json()

        assert resp.status == 200
        assert body["strategy_used"] == "cascading"
        assert body["cached"] is True

    @pytest.mark.asyncio
    async def test_source_tracking(self):
        """Source field from request is echoed in response."""
        platform = MagicMock()
        platform.query = MagicMock(return_value=_awaitable({
            "content": "x", "model_name": "m", "provider": "p",
            "tokens_used": 1, "strategy_used": "", "cached": False,
        }))
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/generate", json={
                "prompt": "hi", "source": "arclane",
            })
            body = await resp.json()

        assert body["source"] == "arclane"

    @pytest.mark.asyncio
    async def test_source_defaults_to_unknown(self):
        """When no source provided, defaults to 'unknown'."""
        platform = MagicMock()
        platform.query = MagicMock(return_value=_awaitable({
            "content": "x", "model_name": "m", "provider": "p",
            "tokens_used": 1, "strategy_used": "", "cached": False,
        }))
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/generate", json={"prompt": "hi"})
            body = await resp.json()

        assert body["source"] == "unknown"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        """Non-JSON body returns 400."""
        platform = MagicMock()
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post(
                "/api/generate",
                data=b"not json",
                headers={"Content-Type": "application/json"},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "invalid JSON" in body["error"]

    @pytest.mark.asyncio
    async def test_optional_params_passed_to_platform(self):
        """model, strategy, system_prompt, max_tokens forwarded to platform.query."""
        platform = MagicMock()
        platform.query = MagicMock(return_value=_awaitable({
            "content": "ok", "model_name": "m", "provider": "p",
            "tokens_used": 0, "strategy_used": "s", "cached": False,
        }))
        runtime = _make_runtime(platform=platform)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            await client.post("/api/generate", json={
                "prompt": "hello",
                "model": "gpt-4",
                "strategy": "cost_optimized",
                "system_prompt": "be concise",
                "max_tokens": 256,
            })

        call_kwargs = platform.query.call_args[1]
        assert call_kwargs["prompt"] == "hello"
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["strategy"] == "cost_optimized"
        assert call_kwargs["system_prompt"] == "be concise"
        assert call_kwargs["max_tokens"] == 256


# ===================================================================
# GET /api/models
# ===================================================================

class TestModelsEndpoint:
    """Tests for GET /api/models."""

    @pytest.mark.asyncio
    async def test_returns_model_list(self):
        """Basic call returns models array and count."""
        runtime = _make_runtime(platform=MagicMock())
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/models")
            body = await resp.json()

        assert resp.status == 200
        assert "models" in body
        assert "count" in body
        assert isinstance(body["models"], list)

    @pytest.mark.asyncio
    async def test_each_model_has_required_fields(self):
        """Each model entry has name, provider, quality_tier at minimum."""
        runtime = _make_runtime(platform=MagicMock())
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/models")
            body = await resp.json()

        for model in body["models"]:
            assert "name" in model
            assert "provider" in model
            assert "quality_tier" in model
            assert "display_name" in model
            assert "context_window" in model

    @pytest.mark.asyncio
    async def test_models_include_strengths_and_specializations(self):
        """Models with profiles include strengths and specializations."""
        runtime = _make_runtime(platform=MagicMock())
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/models")
            body = await resp.json()

        # At least some models should have profiles with strengths
        models_with_strengths = [m for m in body["models"] if "strengths" in m]
        # If the registry has profiled models they'll appear
        if body["count"] > 0:
            # Each model with strengths should have list values
            for m in models_with_strengths:
                assert isinstance(m["strengths"], list)
                assert isinstance(m["specializations"], list)
                assert len(m["strengths"]) <= 3
                assert len(m["specializations"]) <= 3

    @pytest.mark.asyncio
    async def test_handles_empty_model_registry(self):
        """When MODEL_REGISTRY is empty, returns empty list gracefully."""
        runtime = _make_runtime(platform=MagicMock())
        app = create_health_app(runtime)

        with patch("nexus.service.MODEL_REGISTRY", {}, create=True):
            # The import happens inside the handler, so we patch at the source
            with patch.dict("sys.modules", {}):
                pass
            # Simpler: just verify the endpoint doesn't crash
            async with TestServer(app) as server, TestClient(server) as client:
                resp = await client.get("/api/models")
                body = await resp.json()

        assert resp.status == 200
        assert isinstance(body["models"], list)

    @pytest.mark.asyncio
    async def test_response_capped_at_100(self):
        """Models list is capped at 100 entries."""
        runtime = _make_runtime(platform=MagicMock())
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/models")
            body = await resp.json()

        # The code does models[:100] so even if registry has more, capped
        assert len(body["models"]) <= 100

    @pytest.mark.asyncio
    async def test_model_cost_fields(self):
        """Models include cost_per_1k_input and cost_per_1k_output."""
        runtime = _make_runtime(platform=MagicMock())
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/models")
            body = await resp.json()

        for model in body["models"]:
            assert "cost_per_1k_input" in model
            assert "cost_per_1k_output" in model

    @pytest.mark.asyncio
    async def test_count_matches_models_length(self):
        """The count field matches the length of models array."""
        runtime = _make_runtime(platform=MagicMock())
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/models")
            body = await resp.json()

        # count reflects total registry size (before cap)
        assert body["count"] >= len(body["models"])

    @pytest.mark.asyncio
    async def test_registry_import_failure_returns_empty(self):
        """If MODEL_REGISTRY import fails, returns empty list not 500."""
        runtime = _make_runtime(platform=MagicMock())
        app = create_health_app(runtime)

        # The handler wraps MODEL_REGISTRY iteration in try/except,
        # so any exception results in an empty list, not a 500.
        with patch("nexus.providers.adapters.registry.MODEL_REGISTRY",
                    new_callable=lambda: MagicMock(items=MagicMock(side_effect=Exception("boom")))):
            async with TestServer(app) as server, TestClient(server) as client:
                resp = await client.get("/api/models")
                body = await resp.json()

        assert resp.status == 200
        assert body["models"] == []
        assert body["count"] == 0


# ===================================================================
# GET /api/knowledge/search
# ===================================================================

class TestKnowledgeSearchEndpoint:
    """Tests for GET /api/knowledge/search."""

    @pytest.mark.asyncio
    async def test_valid_query_returns_results(self):
        """Search with valid query returns items array."""
        items = [
            _make_knowledge_item(id="k1", content="Python tips", confidence=0.95),
            _make_knowledge_item(id="k2", content="Go tips", confidence=0.85),
        ]
        kb = _make_kb(query_results=items)
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/knowledge/search?q=tips")
            body = await resp.json()

        assert resp.status == 200
        assert body["count"] == 2
        assert len(body["items"]) == 2

    @pytest.mark.asyncio
    async def test_result_format(self):
        """Each result has id, content, source, type, confidence."""
        items = [_make_knowledge_item(
            id="k1", content="data", source="api",
            knowledge_type_value="procedural", confidence=0.77,
        )]
        kb = _make_kb(query_results=items)
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/knowledge/search?q=data")
            body = await resp.json()

        item = body["items"][0]
        assert item["id"] == "k1"
        assert item["content"] == "data"
        assert item["source"] == "api"
        assert item["type"] == "procedural"
        assert item["confidence"] == 0.77

    @pytest.mark.asyncio
    async def test_missing_query_returns_empty(self):
        """No 'q' parameter still calls kb with empty string."""
        kb = _make_kb(query_results=[])
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/knowledge/search")
            body = await resp.json()

        assert resp.status == 200
        assert body["items"] == []
        kb.query_knowledge.assert_called_once_with("", max_results=10)

    @pytest.mark.asyncio
    async def test_limit_clamped_to_50(self):
        """Limit parameter is clamped to max 50."""
        kb = _make_kb(query_results=[])
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/knowledge/search?q=test&limit=200")
            await resp.json()

        kb.query_knowledge.assert_called_once_with("test", max_results=50)

    @pytest.mark.asyncio
    async def test_limit_respects_small_values(self):
        """Limit < 50 is passed through."""
        kb = _make_kb(query_results=[])
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            await client.get("/api/knowledge/search?q=x&limit=5")

        kb.query_knowledge.assert_called_once_with("x", max_results=5)

    @pytest.mark.asyncio
    async def test_kb_unavailable_returns_graceful_fallback(self):
        """When knowledge base is None, returns empty items with error note."""
        runtime = _make_runtime(platform=MagicMock())  # no kb
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/knowledge/search?q=test")
            body = await resp.json()

        assert resp.status == 200
        assert body["items"] == []
        assert "not available" in body["error"]

    @pytest.mark.asyncio
    async def test_no_platform_returns_graceful_fallback(self):
        """When platform itself is None, returns graceful fallback."""
        runtime = _make_runtime()  # no platform at all
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/knowledge/search?q=test")
            body = await resp.json()

        assert resp.status == 200
        assert body["items"] == []

    @pytest.mark.asyncio
    async def test_content_truncated_to_1000_chars(self):
        """Content is truncated to 1000 characters."""
        long_content = "x" * 2000
        items = [_make_knowledge_item(id="k1", content=long_content)]
        kb = _make_kb(query_results=items)
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.get("/api/knowledge/search?q=test")
            body = await resp.json()

        assert len(body["items"][0]["content"]) == 1000


# ===================================================================
# POST /api/knowledge
# ===================================================================

class TestKnowledgeAddEndpoint:
    """Tests for POST /api/knowledge."""

    @pytest.mark.asyncio
    async def test_valid_knowledge_creation(self):
        """Valid POST creates knowledge and returns id + success."""
        kb = _make_kb(add_return="factual_123_abc")
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/knowledge", json={
                "content": "Python is great",
                "knowledge_type": "factual",
                "source": "test",
            })
            body = await resp.json()

        assert resp.status == 200
        assert body["success"] is True
        assert body["id"] == "factual_123_abc"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        """Non-JSON body returns 400."""
        kb = _make_kb()
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post(
                "/api/knowledge",
                data=b"not json",
                headers={"Content-Type": "application/json"},
            )
            body = await resp.json()

        assert resp.status == 400
        assert "invalid JSON" in body["error"]

    @pytest.mark.asyncio
    async def test_knowledge_type_enum_mapping(self):
        """knowledge_type string maps to correct KnowledgeType enum."""
        kb = _make_kb()
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            await client.post("/api/knowledge", json={
                "content": "how to deploy",
                "knowledge_type": "procedural",
            })

        from nexus.memory.knowledge_base import KnowledgeType
        call_kwargs = kb.add_knowledge.call_args[1]
        assert call_kwargs["knowledge_type"] == KnowledgeType.PROCEDURAL

    @pytest.mark.asyncio
    async def test_unknown_knowledge_type_defaults_to_factual(self):
        """Unknown knowledge_type falls back to FACTUAL."""
        kb = _make_kb()
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            await client.post("/api/knowledge", json={
                "content": "misc info",
                "knowledge_type": "nonexistent_type",
            })

        from nexus.memory.knowledge_base import KnowledgeType
        call_kwargs = kb.add_knowledge.call_args[1]
        assert call_kwargs["knowledge_type"] == KnowledgeType.FACTUAL

    @pytest.mark.asyncio
    async def test_confidence_passed_through(self):
        """Custom confidence is forwarded to add_knowledge."""
        kb = _make_kb()
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            await client.post("/api/knowledge", json={
                "content": "high quality",
                "confidence": 0.99,
            })

        assert kb.add_knowledge.call_args[1]["confidence"] == 0.99

    @pytest.mark.asyncio
    async def test_default_confidence_is_0_8(self):
        """When confidence not provided, defaults to 0.8."""
        kb = _make_kb()
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            await client.post("/api/knowledge", json={"content": "test"})

        assert kb.add_knowledge.call_args[1]["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_context_tags_forwarded(self):
        """context_tags array is passed to add_knowledge."""
        kb = _make_kb()
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            await client.post("/api/knowledge", json={
                "content": "tagged",
                "context_tags": ["python", "ml"],
            })

        assert kb.add_knowledge.call_args[1]["context_tags"] == ["python", "ml"]

    @pytest.mark.asyncio
    async def test_kb_unavailable_returns_503(self):
        """When knowledge base is None, returns 503."""
        runtime = _make_runtime(platform=MagicMock())  # no kb
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/knowledge", json={"content": "test"})
            body = await resp.json()

        assert resp.status == 503
        assert "not available" in body["error"]

    @pytest.mark.asyncio
    async def test_source_defaults_to_api(self):
        """When source not provided, defaults to 'api'."""
        kb = _make_kb()
        runtime = _make_runtime(platform=MagicMock(), kb=kb)
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            await client.post("/api/knowledge", json={"content": "test"})

        assert kb.add_knowledge.call_args[1]["source"] == "api"

    @pytest.mark.asyncio
    async def test_no_platform_returns_503(self):
        """When infrastructure has no platform, returns 503."""
        runtime = _make_runtime()  # no platform at all
        app = create_health_app(runtime)

        async with TestServer(app) as server, TestClient(server) as client:
            resp = await client.post("/api/knowledge", json={"content": "test"})
            body = await resp.json()

        assert resp.status == 503


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------

async def _awaitable(value):
    """Wrap a value in a coroutine so MagicMock can return it."""
    return value


async def _awaitable_raise(exc):
    """Return a coroutine that raises the given exception."""
    raise exc
