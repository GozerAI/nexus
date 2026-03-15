"""Regression tests for cog_eng import cleanliness."""

from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "nexus.cog_eng.api.client",
        "nexus.cog_eng.capabilities.autonomous_research_agent",
        "nexus.cog_eng.capabilities.self_improving_codegen",
        "nexus.cog_eng.llm.client",
    ],
)
def test_cog_eng_import_does_not_mutate_sys_path(module_name):
    script = (
        "import importlib, json, sys; "
        "before = list(sys.path); "
        f"importlib.import_module('{module_name}'); "
        "after = list(sys.path); "
        "print(json.dumps({'path_changed': before != after}))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={**os.environ, "PYTHONPATH": "src"},
    )

    assert result.stdout.strip() == '{"path_changed": false}'
