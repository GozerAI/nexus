import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core import ensemble_core


def test_rank_responses_and_ensemble(monkeypatch):
    scores = [0.2, 0.9, 0.4, 0.1]
    iter_scores = iter(scores)
    monkeypatch.setattr(ensemble_core, "score_response", lambda *_args: next(iter_scores))
    ranked = ensemble_core.rank_responses("hello")
    assert [name for _, _, name in ranked] == ["llama-3", "mistral", "gpt-4", "claude"]

    iter_scores = iter(scores)
    monkeypatch.setattr(ensemble_core, "score_response", lambda *_args: next(iter_scores))
    top = ensemble_core.ensemble_inference("hello")
    assert top == "[llama-3] Response to: 'hello'"
