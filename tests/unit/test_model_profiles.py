"""Tests for model profiles system."""

from nexus.providers.adapters.model_profiles import (
    get_model_profile,
    get_all_profiles,
    get_profile_for_task,
    ModelProfile,
)


class TestModelProfileLookup:
    def test_exact_match_gpt5(self):
        p = get_model_profile("openai/gpt-5")
        assert p is not None
        assert p.family == "gpt-5"
        assert p.quality_tier == "frontier"

    def test_prefix_match_gpt5_variant(self):
        p = get_model_profile("openai/gpt-5.2-codex")
        assert p is not None
        assert p.family == "gpt-5"

    def test_claude_opus(self):
        p = get_model_profile("anthropic/claude-opus-4.6")
        assert p is not None
        assert p.family == "claude-opus-4"
        assert "code generation" in p.strengths or "code review" in p.specializations

    def test_claude_sonnet(self):
        p = get_model_profile("anthropic/claude-sonnet-4.5")
        assert p is not None
        assert p.family == "claude-sonnet-4"

    def test_deepseek(self):
        p = get_model_profile("deepseek/deepseek-v3.2")
        assert p is not None
        assert "code generation" in p.specializations

    def test_qwen(self):
        p = get_model_profile("qwen/qwen3-235b")
        assert p is not None
        assert "chinese" in [l.lower() for l in p.natural_languages]

    def test_unknown_model_returns_none(self):
        p = get_model_profile("totally-unknown/model-xyz")
        assert p is None

    def test_gemini(self):
        p = get_model_profile("google/gemini-3.1-pro")
        assert p is not None
        assert "multimodal" in p.strengths[0].lower() or any("multimodal" in s.lower() for s in p.specializations)


class TestModelProfileMetadata:
    def test_has_programming_languages(self):
        p = get_model_profile("openai/gpt-5")
        assert "python" in p.programming_languages
        assert "javascript" in p.programming_languages
        assert "rust" in p.programming_languages

    def test_has_natural_languages(self):
        p = get_model_profile("openai/gpt-5")
        assert "english" in p.natural_languages
        assert "spanish" in p.natural_languages
        assert "chinese" in p.natural_languages

    def test_has_strengths_and_weaknesses(self):
        p = get_model_profile("anthropic/claude-opus-4")
        assert len(p.strengths) > 0
        assert len(p.weaknesses) > 0

    def test_has_specializations(self):
        p = get_model_profile("mistral/codestral")
        assert "code generation" in p.specializations

    def test_has_not_recommended_for(self):
        p = get_model_profile("deepseek/deepseek-r1")
        assert len(p.not_recommended_for) > 0

    def test_weak_languages(self):
        p = get_model_profile("mistral/mistral-large")
        assert len(p.weak_natural_languages) > 0

    def test_to_dict(self):
        p = get_model_profile("openai/gpt-5")
        d = p.to_dict()
        assert "family" in d
        assert "strengths" in d
        assert "programming_languages" in d
        assert "quality_tier" in d


class TestTaskMatching:
    def test_find_code_generation_models(self):
        profiles = get_profile_for_task("code generation")
        assert len(profiles) > 0
        families = [p.family for p in profiles]
        assert any("gpt" in f or "claude" in f or "codestral" in f or "deepseek" in f for f in families)

    def test_find_models_for_chinese(self):
        profiles = get_profile_for_task(
            "translation",
            required_languages=["chinese"],
        )
        assert len(profiles) > 0
        # Qwen should be in the results
        assert any("qwen" in p.family for p in profiles)

    def test_find_models_for_rust_code(self):
        profiles = get_profile_for_task(
            "code generation",
            required_prog_languages=["rust"],
        )
        assert len(profiles) > 0

    def test_no_match_returns_empty(self):
        profiles = get_profile_for_task("underwater basket weaving")
        assert len(profiles) == 0


class TestAllProfiles:
    def test_profiles_exist(self):
        profiles = get_all_profiles()
        assert len(profiles) >= 10  # We defined 12+ profiles

    def test_all_profiles_have_required_fields(self):
        for pattern, profile in get_all_profiles().items():
            assert profile.family, f"Missing family for {pattern}"
            assert len(profile.strengths) > 0, f"No strengths for {pattern}"
            assert profile.quality_tier in ("frontier", "strong", "good", "basic"), \
                f"Invalid tier for {pattern}: {profile.quality_tier}"
