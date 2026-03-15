"""Regression tests for lazy ensemble imports."""

from pathlib import Path
import os
import subprocess
import sys


def test_ensemble_core_import_is_lazy():
    script = (
        "import importlib, json; "
        "m = importlib.import_module('nexus.core.ensemble_core'); "
        "print(json.dumps({'loaded': getattr(m.model_ensemble, '_models', 'missing')}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.stdout.strip() == '{"loaded": null}'


def test_ensemble_core_v2_import_is_lazy():
    script = (
        "import importlib, json; "
        "m = importlib.import_module('nexus.core.ensemble_core_v2'); "
        "print(json.dumps({'loaded': getattr(m.model_ensemble, '_models', 'missing')}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.stdout.strip() == '{"loaded": null}'


def test_llm_provider_import_does_not_mutate_sys_path():
    script = (
        "import importlib, json, sys; "
        "before = list(sys.path); "
        "importlib.import_module('nexus.core.llm_provider'); "
        "after = list(sys.path); "
        "print(json.dumps({'path_changed': before != after, 'contains_ai_platform_unified': any('ai-platform-unified' in p for p in after)}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.stdout.strip() == '{"path_changed": false, "contains_ai_platform_unified": false}'
