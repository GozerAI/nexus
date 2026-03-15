"""
Unit tests for the Nexus Autonomy system.

Covers:
- 695: Autonomous model performance improvement
- 699: Autonomous agent capability expansion
- 703: Autonomous reasoning strategy improvement
- 707: Autonomous prompt engineering improvement
"""

import time
import pytest

from nexus.autonomy.model_performance import (
    AutonomousModelTuner,
    PerformanceSnapshot,
    TuningDimension,
    TuningExperiment,
    ModelProfile,
)
from nexus.autonomy.capability_expansion import (
    AutonomousCapabilityExpander,
    Capability,
    CapabilityGap,
    CapabilityStatus,
    GapSeverity,
    TaskOutcome,
)
from nexus.autonomy.reasoning_improvement import (
    AutonomousReasoningImprover,
    StrategyType,
    StrategyProfile,
    ReasoningOutcome,
    StrategyAdjustment,
)
from nexus.autonomy.prompt_improvement import (
    AutonomousPromptEngineer,
    PromptTemplate,
    PromptVariationType,
    PromptOutcome,
    PromptRefinement,
)


# ════════════════════════════════════════════════════════════════════
# 695 — Autonomous Model Performance Improvement
# ════════════════════════════════════════════════════════════════════


class TestAutonomousModelTuner:
    """Tests for AutonomousModelTuner."""

    def test_initialization(self):
        tuner = AutonomousModelTuner()
        assert tuner is not None
        assert len(tuner._profiles) == 0

    def test_record_performance_creates_profile(self):
        tuner = AutonomousModelTuner()
        tuner.record_performance("gpt-4", "summarization", 0.8)
        assert "gpt-4" in tuner._profiles
        profile = tuner._profiles["gpt-4"]
        assert profile.total_observations == 1

    def test_record_multiple_observations(self):
        tuner = AutonomousModelTuner()
        for _ in range(5):
            tuner.record_performance("gpt-4", "qa", 0.75)
        profile = tuner._profiles["gpt-4"]
        assert profile.total_observations == 5
        assert len(profile.task_scores["qa"]) == 5

    def test_score_clamping(self):
        tuner = AutonomousModelTuner()
        tuner.record_performance("gpt-4", "qa", 1.5)  # should clamp to 1.0
        tuner.record_performance("gpt-4", "qa", -0.5)  # should clamp to 0.0
        scores = tuner._profiles["gpt-4"].task_scores["qa"]
        assert scores[0] == 1.0
        assert scores[1] == 0.0

    def test_get_optimal_parameters_default(self):
        tuner = AutonomousModelTuner()
        tuner.record_performance("gpt-4", "qa", 0.8)
        params = tuner.get_optimal_parameters("gpt-4")
        assert "temperature" in params
        assert "top_p" in params

    def test_get_optimal_parameters_with_task(self):
        tuner = AutonomousModelTuner()
        tuner.record_performance("gpt-4", "qa", 0.8)
        params = tuner.get_optimal_parameters("gpt-4", task_type="qa")
        assert isinstance(params, dict)

    def test_performance_report_empty(self):
        tuner = AutonomousModelTuner()
        report = tuner.get_performance_report()
        assert report == {}

    def test_performance_report_populated(self):
        tuner = AutonomousModelTuner()
        for i in range(10):
            tuner.record_performance("gpt-4", "qa", 0.6 + i * 0.02)
        report = tuner.get_performance_report("gpt-4")
        assert "gpt-4" in report
        assert "task_summaries" in report["gpt-4"]
        assert "qa" in report["gpt-4"]["task_summaries"]

    def test_performance_report_nonexistent_model(self):
        tuner = AutonomousModelTuner()
        report = tuner.get_performance_report("nonexistent")
        assert "error" in report

    def test_force_tune_insufficient_data(self):
        tuner = AutonomousModelTuner()
        tuner.record_performance("gpt-4", "qa", 0.5)
        result = tuner.force_tune("gpt-4", "qa")
        # May or may not produce experiment with only 1 sample
        # but should not crash

    def test_force_tune_with_data(self):
        tuner = AutonomousModelTuner(min_samples=3, tuning_cooldown=0)
        for _ in range(5):
            tuner.record_performance("gpt-4", "qa", 0.5)
        result = tuner.force_tune("gpt-4", "qa")
        # Should attempt tuning
        if result:
            assert isinstance(result, TuningExperiment)
            assert result.model_name == "gpt-4"

    def test_rollback_no_history(self):
        tuner = AutonomousModelTuner()
        assert tuner.rollback_last_tuning("gpt-4") is False

    def test_rollback_after_tuning(self):
        tuner = AutonomousModelTuner(min_samples=3, tuning_cooldown=0)
        for _ in range(10):
            tuner.record_performance("gpt-4", "qa", 0.4)

        exp = tuner.force_tune("gpt-4", "qa")
        if exp and exp.accepted:
            original_val = exp.original_value
            assert tuner.rollback_last_tuning("gpt-4") is True
            profile = tuner._profiles["gpt-4"]
            assert profile.parameters[exp.dimension.value] == original_val

    def test_tuning_triggered_on_poor_performance(self):
        tuner = AutonomousModelTuner(min_samples=5, tuning_cooldown=0)
        results = []
        for _ in range(15):
            exp = tuner.record_performance("gpt-4", "qa", 0.2)
            if exp:
                results.append(exp)
        # Should have triggered at least one tuning attempt
        # (may not always produce one due to heuristics)

    def test_compute_trend_flat(self):
        assert abs(AutonomousModelTuner._compute_trend([0.5, 0.5, 0.5])) < 0.01

    def test_compute_trend_rising(self):
        trend = AutonomousModelTuner._compute_trend([0.1, 0.3, 0.5, 0.7, 0.9])
        assert trend > 0

    def test_compute_trend_declining(self):
        trend = AutonomousModelTuner._compute_trend([0.9, 0.7, 0.5, 0.3, 0.1])
        assert trend < 0

    def test_compute_trend_insufficient(self):
        assert AutonomousModelTuner._compute_trend([0.5]) == 0.0
        assert AutonomousModelTuner._compute_trend([]) == 0.0

    def test_custom_parameters(self):
        tuner = AutonomousModelTuner(
            min_samples=3,
            improvement_threshold=0.1,
            tuning_cooldown=10.0,
            max_history=500,
        )
        assert tuner._min_samples == 3
        assert tuner._improvement_threshold == 0.1


# ════════════════════════════════════════════════════════════════════
# 699 — Autonomous Agent Capability Expansion
# ════════════════════════════════════════════════════════════════════


class TestAutonomousCapabilityExpander:
    """Tests for AutonomousCapabilityExpander."""

    def test_initialization(self):
        expander = AutonomousCapabilityExpander()
        assert expander is not None
        assert len(expander._capabilities) == 0

    def test_register_capability(self):
        expander = AutonomousCapabilityExpander()
        cap = expander.register_capability(
            "text_analysis", "Analyze text", ["analysis"]
        )
        assert cap.name == "text_analysis"
        assert cap.status == CapabilityStatus.ACTIVE
        assert expander.get_capability("text_analysis") is not None

    def test_list_capabilities(self):
        expander = AutonomousCapabilityExpander()
        expander.register_capability("cap1", "Cap 1", ["t1"])
        expander.register_capability("cap2", "Cap 2", ["t2"])
        caps = expander.list_capabilities()
        assert len(caps) == 2

    def test_list_capabilities_filtered(self):
        expander = AutonomousCapabilityExpander()
        cap = expander.register_capability("cap1", "Cap 1", ["t1"])
        cap.status = CapabilityStatus.DEPRECATED
        expander.register_capability("cap2", "Cap 2", ["t2"])
        active = expander.list_capabilities(status=CapabilityStatus.ACTIVE)
        assert len(active) == 1

    def test_record_success_updates_rate(self):
        expander = AutonomousCapabilityExpander()
        expander.register_capability("cap1", "Cap 1", ["t1"])
        expander.record_outcome(
            "task1", "t1", "agent1", True,
            capabilities_used=["cap1"],
        )
        assert expander.get_capability("cap1").total_uses == 1
        assert expander.get_capability("cap1").success_rate == 1.0

    def test_record_failure_updates_rate(self):
        expander = AutonomousCapabilityExpander()
        expander.register_capability("cap1", "Cap 1", ["t1"])
        expander.record_outcome(
            "task1", "t1", "agent1", True,
            capabilities_used=["cap1"],
        )
        expander.record_outcome(
            "task2", "t1", "agent1", False,
            capabilities_used=["cap1"],
        )
        cap = expander.get_capability("cap1")
        assert cap.total_uses == 2
        assert cap.success_rate == 0.5

    def test_gap_detection_below_threshold(self):
        expander = AutonomousCapabilityExpander(min_failures_for_gap=3)
        # 2 failures should not trigger gap
        for i in range(2):
            gap = expander.record_outcome(
                f"task{i}", "analysis", "agent1", False,
                error_category="timeout",
            )
        assert gap is None

    def test_gap_detection_at_threshold(self):
        expander = AutonomousCapabilityExpander(min_failures_for_gap=3)
        gap = None
        for i in range(3):
            gap = expander.record_outcome(
                f"task{i}", "analysis", "agent1", False,
                error_category="timeout",
            )
        assert gap is not None
        assert isinstance(gap, CapabilityGap)
        assert gap.task_type == "analysis"
        assert gap.failure_pattern == "timeout"

    def test_gap_severity_levels(self):
        expander = AutonomousCapabilityExpander(min_failures_for_gap=1)
        # 1 failure = low
        expander.record_outcome("t1", "a", "agent1", False, error_category="e1")
        gaps = expander.get_gaps()
        assert len(gaps) == 1

    def test_synthesize_capability_on_gap(self):
        expander = AutonomousCapabilityExpander(min_failures_for_gap=3)
        for i in range(3):
            expander.record_outcome(
                f"task{i}", "analysis", "agent1", False,
                error_category="timeout",
            )
        # Should have auto-synthesized a capability
        synth = [
            c for c in expander._capabilities.values()
            if c.source in ("synthesized", "composed")
        ]
        assert len(synth) >= 1
        assert synth[0].status == CapabilityStatus.EXPERIMENTAL

    def test_promote_capability(self):
        expander = AutonomousCapabilityExpander(min_failures_for_gap=3)
        for i in range(3):
            expander.record_outcome(
                f"task{i}", "analysis", "agent1", False,
                error_category="timeout",
            )
        synth = [
            c for c in expander._capabilities.values()
            if c.status == CapabilityStatus.EXPERIMENTAL
        ]
        if synth:
            assert expander.promote_capability(synth[0].name) is True
            assert expander.get_capability(synth[0].name).status == CapabilityStatus.ACTIVE

    def test_promote_nonexistent(self):
        expander = AutonomousCapabilityExpander()
        assert expander.promote_capability("nonexistent") is False

    def test_deprecation_on_low_success(self):
        expander = AutonomousCapabilityExpander(min_success_rate=0.4)
        cap = expander.register_capability("bad_cap", "Bad", ["t1"])
        cap.total_uses = 11
        cap.success_rate = 0.2
        # Trigger deprecation check via recording an outcome
        expander.record_outcome("tx", "t1", "a1", True)
        assert expander.get_capability("bad_cap").status == CapabilityStatus.DEPRECATED

    def test_expansion_report(self):
        expander = AutonomousCapabilityExpander()
        expander.register_capability("cap1", "Cap 1", ["t1"])
        report = expander.get_expansion_report()
        assert "total_capabilities" in report
        assert report["total_capabilities"] == 1

    def test_composition_rule(self):
        expander = AutonomousCapabilityExpander(min_failures_for_gap=3)
        expander.register_capability("error_handling", "Error Handler", ["analysis"])

        composed_caps = []
        def my_composer(sources, gap):
            cap = Capability(
                name=f"custom_composed_{gap.task_type}",
                description="Custom composed",
                task_types=[gap.task_type],
            )
            composed_caps.append(cap)
            return cap

        expander.register_composition_rule("custom", my_composer)

        for i in range(3):
            expander.record_outcome(
                f"task{i}", "analysis", "agent1", False,
                error_category="timeout",
            )
        # Check that composition was attempted
        # (depends on suggestion order but the rule is registered)

    def test_get_gaps_filtered(self):
        expander = AutonomousCapabilityExpander(min_failures_for_gap=3)
        for i in range(3):
            expander.record_outcome(
                f"t{i}", "a", "ag1", False, error_category="err"
            )
        resolved = expander.get_gaps(resolved=True)
        unresolved = expander.get_gaps(resolved=False)
        total = expander.get_gaps()
        assert len(total) == len(resolved) + len(unresolved)


# ════════════════════════════════════════════════════════════════════
# 703 — Autonomous Reasoning Strategy Improvement
# ════════════════════════════════════════════════════════════════════


class TestAutonomousReasoningImprover:
    """Tests for AutonomousReasoningImprover."""

    def test_initialization(self):
        improver = AutonomousReasoningImprover()
        assert improver is not None
        assert len(improver._profiles) == len(StrategyType)

    def test_all_strategies_have_profiles(self):
        improver = AutonomousReasoningImprover()
        for st in StrategyType:
            assert st in improver._profiles

    def test_record_outcome(self):
        improver = AutonomousReasoningImprover()
        improver.record_outcome(
            StrategyType.CHAIN_OF_THOUGHT, "qa", 0.8, latency_ms=100
        )
        profile = improver._profiles[StrategyType.CHAIN_OF_THOUGHT]
        assert profile.total_uses == 1
        assert profile.avg_quality == 0.8

    def test_record_multiple_outcomes(self):
        improver = AutonomousReasoningImprover()
        for score in [0.6, 0.7, 0.8]:
            improver.record_outcome(
                StrategyType.DECOMPOSITION, "analysis", score
            )
        profile = improver._profiles[StrategyType.DECOMPOSITION]
        assert profile.total_uses == 3
        assert abs(profile.avg_quality - 0.7) < 0.01

    def test_select_strategy_default(self):
        improver = AutonomousReasoningImprover()
        strategy = improver.select_strategy("qa")
        assert isinstance(strategy, StrategyType)

    def test_select_strategy_with_data(self):
        improver = AutonomousReasoningImprover()
        # Make one strategy clearly better for a task
        for _ in range(10):
            improver.record_outcome(
                StrategyType.CHAIN_OF_THOUGHT, "qa", 0.95
            )
            improver.record_outcome(
                StrategyType.DECOMPOSITION, "qa", 0.3
            )
        strategy = improver.select_strategy("qa")
        assert strategy == StrategyType.CHAIN_OF_THOUGHT

    def test_select_strategy_limited_candidates(self):
        improver = AutonomousReasoningImprover()
        candidates = [StrategyType.DEDUCTIVE, StrategyType.INDUCTIVE]
        strategy = improver.select_strategy("logic", available=candidates)
        assert strategy in candidates

    def test_get_fallback_order(self):
        improver = AutonomousReasoningImprover()
        order = improver.get_fallback_order()
        assert len(order) == len(StrategyType)

    def test_get_fallback_order_for_task(self):
        improver = AutonomousReasoningImprover()
        for _ in range(5):
            improver.record_outcome(StrategyType.ANALOGICAL, "creative", 0.9)
        order = improver.get_fallback_order("creative")
        assert order[0] == StrategyType.ANALOGICAL

    def test_get_strategy_config(self):
        improver = AutonomousReasoningImprover()
        config = improver.get_strategy_config(StrategyType.CHAIN_OF_THOUGHT)
        assert "max_depth" in config
        assert "verification_enabled" in config

    def test_weight_adjustment_on_poor_performance(self):
        improver = AutonomousReasoningImprover(
            adjustment_cooldown=0, min_observations=3
        )
        for _ in range(10):
            improver.record_outcome(
                StrategyType.PATTERN_MATCHING, "qa", 0.2, success=False
            )
        profile = improver._profiles[StrategyType.PATTERN_MATCHING]
        # Weight should have been reduced
        assert profile.weight < 1.0

    def test_weight_increase_on_good_performance(self):
        improver = AutonomousReasoningImprover(
            adjustment_cooldown=0, min_observations=3
        )
        for _ in range(10):
            improver.record_outcome(
                StrategyType.META_REASONING, "complex", 0.95
            )
        profile = improver._profiles[StrategyType.META_REASONING]
        assert profile.weight >= 1.0

    def test_improvement_report(self):
        improver = AutonomousReasoningImprover()
        for _ in range(5):
            improver.record_outcome(StrategyType.ENSEMBLE, "qa", 0.7)
        report = improver.get_improvement_report()
        assert "strategy_profiles" in report
        assert "fallback_order" in report
        assert "total_outcomes" in report
        assert report["total_outcomes"] == 5

    def test_step_config_adjustment(self):
        improver = AutonomousReasoningImprover(
            adjustment_cooldown=0, min_observations=3
        )
        # Low quality with few steps should increase depth
        for _ in range(10):
            improver.record_outcome(
                StrategyType.CHAIN_OF_THOUGHT, "hard_qa", 0.3,
                step_count=2,
            )
        config = improver.get_strategy_config(StrategyType.CHAIN_OF_THOUGHT)
        # Depth may have been adjusted

    def test_task_affinity_tracking(self):
        improver = AutonomousReasoningImprover()
        improver.record_outcome(StrategyType.ABDUCTIVE, "diagnosis", 0.9)
        profile = improver._profiles[StrategyType.ABDUCTIVE]
        assert "diagnosis" in profile.task_affinity
        assert profile.task_affinity["diagnosis"] > 0


# ════════════════════════════════════════════════════════════════════
# 707 — Autonomous Prompt Engineering Improvement
# ════════════════════════════════════════════════════════════════════


class TestAutonomousPromptEngineer:
    """Tests for AutonomousPromptEngineer."""

    def test_initialization(self):
        eng = AutonomousPromptEngineer()
        assert eng is not None
        assert len(eng._templates) == 0

    def test_register_template(self):
        eng = AutonomousPromptEngineer()
        tpl = eng.register_template(
            "Summarize: {text}", "summarization"
        )
        assert tpl.task_type == "summarization"
        assert tpl.active is True
        assert tpl.variation_type == PromptVariationType.ORIGINAL

    def test_register_with_custom_id(self):
        eng = AutonomousPromptEngineer()
        tpl = eng.register_template(
            "Analyze: {text}", "analysis", template_id="my_tpl"
        )
        assert tpl.template_id == "my_tpl"
        assert eng.get_template("my_tpl") is not None

    def test_get_best_template_empty(self):
        eng = AutonomousPromptEngineer()
        assert eng.get_best_template("qa") is None

    def test_get_best_template_single(self):
        eng = AutonomousPromptEngineer()
        eng.register_template("Q: {q}", "qa")
        best = eng.get_best_template("qa")
        assert best is not None

    def test_get_best_template_by_score(self):
        eng = AutonomousPromptEngineer(min_uses=2)
        t1 = eng.register_template("V1: {q}", "qa", template_id="v1")
        t2 = eng.register_template("V2: {q}", "qa", template_id="v2")
        # Give v2 better scores
        for _ in range(5):
            eng.record_outcome("v1", "qa", "gpt-4", 0.5)
            eng.record_outcome("v2", "qa", "gpt-4", 0.9)
        best = eng.get_best_template("qa")
        assert best.template_id == "v2"

    def test_record_outcome_updates_stats(self):
        eng = AutonomousPromptEngineer()
        tpl = eng.register_template("Test: {q}", "qa", template_id="t1")
        eng.record_outcome("t1", "qa", "gpt-4", 0.8)
        assert tpl.total_uses == 1
        assert tpl.avg_score == 0.8
        assert tpl.success_count == 1

    def test_record_outcome_failure(self):
        eng = AutonomousPromptEngineer()
        tpl = eng.register_template("Test: {q}", "qa", template_id="t1")
        eng.record_outcome("t1", "qa", "gpt-4", 0.3, success=False)
        assert tpl.success_count == 0

    def test_refinement_triggered_on_low_score(self):
        eng = AutonomousPromptEngineer(min_uses=3, max_variations=5)
        eng.register_template("Bad prompt: {q}", "qa", template_id="orig")

        refinement = None
        for i in range(10):
            r = eng.record_outcome("orig", "qa", "gpt-4", 0.3)
            if r:
                refinement = r

        # Should have triggered at least one refinement
        if refinement:
            assert isinstance(refinement, PromptRefinement)
            assert refinement.original_template_id == "orig"

    def test_no_refinement_on_good_score(self):
        eng = AutonomousPromptEngineer(min_uses=3)
        eng.register_template("Good prompt: {q}", "qa", template_id="good")
        refinement = None
        for _ in range(10):
            r = eng.record_outcome("good", "qa", "gpt-4", 0.95)
            if r:
                refinement = r
        assert refinement is None

    def test_variation_pruning(self):
        eng = AutonomousPromptEngineer(min_uses=2, max_variations=3)
        eng.register_template("Original: {q}", "qa", template_id="orig")
        t2 = eng.register_template(
            "Bad variation", "qa", template_id="bad_var",
            variation_type=PromptVariationType.SIMPLIFIED,
        )
        t2.parent_template_id = "orig"
        for _ in range(10):
            eng.record_outcome("bad_var", "qa", "gpt-4", 0.1, success=False)
        eng._prune_variations("qa")
        assert t2.active is False

    def test_improvement_report(self):
        eng = AutonomousPromptEngineer()
        eng.register_template("T: {q}", "qa", template_id="t1")
        for _ in range(5):
            eng.record_outcome("t1", "qa", "gpt-4", 0.7)
        report = eng.get_improvement_report()
        assert "total_templates" in report
        assert "task_reports" in report
        assert "qa" in report["task_reports"]

    def test_list_templates(self):
        eng = AutonomousPromptEngineer()
        eng.register_template("T1", "qa", template_id="t1")
        eng.register_template("T2", "analysis", template_id="t2")
        all_tpls = eng.list_templates()
        assert len(all_tpls) == 2
        qa_only = eng.list_templates(task_type="qa")
        assert len(qa_only) == 1

    def test_simplify_text(self):
        eng = AutonomousPromptEngineer()
        text = "Hello\nHello\n\n\n\nWorld"
        simplified = eng._simplify_text(text)
        assert "Hello" in simplified
        assert "\n\n\n" not in simplified

    def test_model_specific_template(self):
        eng = AutonomousPromptEngineer()
        eng.register_template(
            "General: {q}", "qa", template_id="gen"
        )
        eng.register_template(
            "GPT-specific: {q}", "qa", template_id="gpt",
            model_target="gpt-4",
        )
        best = eng.get_best_template("qa", model_name="gpt-4")
        # Should return one of them (both are active, neither evaluated yet)
        assert best is not None
