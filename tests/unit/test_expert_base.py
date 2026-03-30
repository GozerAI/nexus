"""Regression tests for expert recommendation extraction helpers."""

from nexus.experts.base import BaseExpert, ExpertPersona, TaskType


class StubExpert(BaseExpert):
    async def analyze(self, task):
        raise NotImplementedError

    async def execute(self, task):
        raise NotImplementedError


def _expert() -> StubExpert:
    return StubExpert(
        ExpertPersona(
            name="StubExpert",
            role="Stub",
            description="stub",
            system_prompt="stub",
            task_types=[TaskType.GENERAL],
        )
    )


def test_extract_recommendation_prefers_explicit_marker_line():
    expert = _expert()
    analysis = """
    Overview paragraph.

    Recommendation: Run a deployment readiness audit before release.
    """

    assert expert._extract_recommendation(analysis, "fallback") == "Run a deployment readiness audit before release."


def test_extract_recommendation_uses_following_line_when_header_has_no_inline_text():
    expert = _expert()
    analysis = """
    Recommended Approach
    Execute a staged rollout with monitoring and rollback guards.
    """

    assert expert._extract_recommendation(analysis, "fallback") == "Execute a staged rollout with monitoring and rollback guards."


def test_extract_recommendation_falls_back_to_first_sentence():
    expert = _expert()
    analysis = "This is the main conclusion. Additional detail follows in later sentences."

    assert expert._extract_recommendation(analysis, "fallback") == "This is the main conclusion."


def test_extract_recommendation_skips_heading_like_marker_lines():
    expert = _expert()
    analysis = """
    Recommendation
    Deployment Readiness Recommendation
    Run a deployment readiness audit before release.
    """

    assert expert._extract_recommendation(analysis, "fallback") == "Run a deployment readiness audit before release."
