"""Regression tests for the lazy enhanced API runtime."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid


def test_enhanced_api_health_does_not_eagerly_initialize_runtime():
    os.environ["DATABASE_PATH"] = os.path.join(
        tempfile.gettempdir(),
        f"enhanced_api_health_{uuid.uuid4().hex}.db",
    )
    module = importlib.import_module("nexus.core.enhanced_api")
    module.runtime.reset_for_testing()

    with module.app.test_client() as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["runtime"]["initialized"] is False
    assert payload["runtime"]["components"]["api_key_manager"] is False
    module.runtime.reset_for_testing()


def test_enhanced_api_component_proxy_initializes_runtime_on_demand():
    os.environ["DATABASE_PATH"] = os.path.join(
        tempfile.gettempdir(),
        f"enhanced_api_proxy_{uuid.uuid4().hex}.db",
    )
    module = importlib.import_module("nexus.core.enhanced_api")
    module.runtime.reset_for_testing()

    module.cache_manager.get_stats()

    assert module.runtime.initialized is True
    assert module.runtime.status()["components"]["cache_manager"] is True
    module.runtime.reset_for_testing()


def test_enhanced_api_import_does_not_load_ensemble_modules():
    script = (
        "import importlib, json, sys; "
        "m = importlib.import_module('nexus.core.enhanced_api'); "
        "print(json.dumps({"
        "'ensemble_core_v2': 'nexus.core.ensemble_core_v2' in sys.modules, "
        "'strategic_ensemble': 'nexus.core.strategic_ensemble' in sys.modules, "
        "'runtime_initialized': m.runtime.initialized"
        "}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.stdout.strip() == (
        '{"ensemble_core_v2": false, "strategic_ensemble": false, "runtime_initialized": false}'
    )
