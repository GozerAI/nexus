"""Regression tests for the lightweight Nexus API entrypoint."""

from unittest.mock import patch

from nexus.api.api import create_app, initialize_subsystems


def test_create_app_does_not_eagerly_initialize_engine():
    calls = 0
    auth_calls = 0

    class StubEngine:
        def think(self, input_data):
            return f"processed {input_data}"

    def engine_factory():
        nonlocal calls
        calls += 1
        return StubEngine()

    def auth_factory():
        nonlocal auth_calls
        auth_calls += 1
        return object()

    app = create_app({"TESTING": True}, engine_factory=engine_factory, auth_factory=auth_factory)

    with app.test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["engine_initialized"] is False
    assert calls == 0
    assert auth_calls == 0


def test_think_initializes_engine_lazily_once():
    calls = 0

    class StubEngine:
        def think(self, input_data):
            return f"processed {input_data}"

    def engine_factory():
        nonlocal calls
        calls += 1
        return StubEngine()

    app = create_app({"TESTING": True}, engine_factory=engine_factory)

    with app.test_client() as client:
        first = client.post("/think", json={"input": "hello"})
        second = client.post("/think", json={"input": "again"})
        health = client.get("/health")

    assert first.status_code == 200
    assert first.get_json()["response"] == "processed hello"
    assert second.status_code == 200
    assert second.get_json()["response"] == "processed again"
    assert health.get_json()["engine_initialized"] is True
    assert calls == 1


def test_think_rejects_invalid_json_requests():
    app = create_app({"TESTING": True})

    with app.test_client() as client:
        response = client.post("/think", data="hello", headers={"Content-Type": "text/plain"})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload == {
        "ok": False,
        "error": "Content-Type must be application/json",
    }


def test_think_rejects_missing_input_field():
    app = create_app({"TESTING": True})

    with app.test_client() as client:
        response = client.post("/think", json={"prompt": "hello"})

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert payload["error"] == "Missing required field: 'input'"


def test_initialize_subsystems_runs_only_once_per_app():
    app = create_app({"TESTING": True})

    with patch("nexus.api.api.initialize_memory_system") as memory, patch(
        "nexus.api.api.initialize_rag_system"
    ) as rag, patch("nexus.api.api.initialize_reasoning_system") as reasoning, patch(
        "nexus.api.api.initialize_data_system"
    ) as data:
        initialize_subsystems(app, {"mode": "test"})
        initialize_subsystems(app, {"mode": "test"})

    memory.assert_called_once_with({"mode": "test"})
    rag.assert_called_once_with({"mode": "test"})
    reasoning.assert_called_once_with({"mode": "test"})
    data.assert_called_once_with({"mode": "test"})
