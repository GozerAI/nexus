"""Tests for the orchestration module — expert routing and pipeline execution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexus.experts.base import Task, TaskType, ExpertOpinion, ExpertResult
from nexus.orchestration.expert_router import ExpertRouter, RoutingDecision
from nexus.orchestration.pipeline_executor import (
    PipelineExecutor,
    PipelineStep,
    PipelineResult,
)
from nexus.orchestration.types import (
    ApprovalRequest,
    AutonomyLevel,
    StepStatus,
    TaskCategory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _task(desc="Analyze market trends", task_type=TaskType.GENERAL, priority=5):
    return Task(id="t1", description=desc, task_type=task_type, priority=priority)


# ---------------------------------------------------------------------------
# ExpertRouter — Task Categorization
# ---------------------------------------------------------------------------

class TestTaskCategorization:
    def setup_method(self):
        self.router = ExpertRouter()

    def test_research_keywords(self):
        t = _task("Research the latest AI trends")
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.RESEARCH

    def test_content_keywords(self):
        t = _task("Write a blog article about machine learning")
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.CONTENT

    def test_technical_keywords(self):
        t = _task("Build a REST API endpoint for user auth")
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.TECHNICAL

    def test_analysis_keywords(self):
        t = _task("Analyze our revenue data for Q1")
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.ANALYSIS

    def test_review_keywords(self):
        t = _task("Review the pull request for quality issues")
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.REVIEW

    def test_strategy_keywords(self):
        t = _task("Plan our go-to-market strategy")
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.STRATEGY

    def test_task_type_overrides_keywords(self):
        t = _task("Build something", task_type=TaskType.RESEARCH)
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.RESEARCH

    def test_unknown_defaults_to_strategy(self):
        t = _task("Do something vague and undefined")
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.STRATEGY

    def test_case_insensitive(self):
        t = _task("RESEARCH the MARKET for new OPPORTUNITIES")
        cat = self.router._categorize_task(t)
        assert cat == TaskCategory.RESEARCH


# ---------------------------------------------------------------------------
# ExpertRouter — Routing Decisions
# ---------------------------------------------------------------------------

class TestRoutingDecisions:
    def setup_method(self):
        self.router = ExpertRouter()

    def test_returns_routing_decision(self):
        t = _task("Analyze our competitors")
        decision = self.router.route_task(t)
        assert isinstance(decision, RoutingDecision)
        assert len(decision.primary_experts) >= 1
        assert decision.reasoning

    def test_research_routes_to_research_expert(self):
        t = _task("Research AI trends", task_type=TaskType.RESEARCH)
        decision = self.router.route_task(t)
        assert "ResearchExpert" in decision.primary_experts

    def test_content_routes_to_writer(self):
        t = _task("Write a blog post", task_type=TaskType.WRITING)
        decision = self.router.route_task(t)
        assert "WriterExpert" in decision.primary_experts

    def test_technical_routes_to_engineer(self):
        t = _task("Code a new feature", task_type=TaskType.CODING)
        decision = self.router.route_task(t)
        assert "EngineerExpert" in decision.primary_experts

    def test_high_priority_adds_critic(self):
        t = _task("Research something important", task_type=TaskType.RESEARCH, priority=8)
        decision = self.router.route_task(t)
        all_experts = decision.primary_experts + decision.supporting_experts
        assert "CriticExpert" in all_experts

    def test_estimated_duration_positive(self):
        t = _task("Analyze the market")
        decision = self.router.route_task(t)
        assert decision.estimated_duration > 0

    def test_high_priority_requires_review(self):
        t = _task("Important task", priority=7)
        decision = self.router.route_task(t)
        assert decision.requires_review is True

    def test_low_priority_no_review(self):
        t = _task("Minor task", priority=3)
        decision = self.router.route_task(t)
        assert decision.requires_review is False


# ---------------------------------------------------------------------------
# ExpertRouter — Autonomy Level
# ---------------------------------------------------------------------------

class TestAutonomyLevel:
    def setup_method(self):
        self.router = ExpertRouter()

    def test_critical_priority_requires_full_approval(self):
        t = _task("Deploy to production", priority=9)
        decision = self.router.route_task(t)
        assert decision.autonomy_level == AutonomyLevel.FULL_APPROVAL

    def test_sensitive_keywords_get_supervised(self):
        t = _task("Process customer payment refund", priority=5)
        decision = self.router.route_task(t)
        assert decision.autonomy_level == AutonomyLevel.SUPERVISED

    def test_legal_gets_supervised(self):
        t = _task("Review the legal contract terms", priority=3)
        decision = self.router.route_task(t)
        assert decision.autonomy_level == AutonomyLevel.SUPERVISED

    def test_research_gets_conditional(self):
        t = _task("Research AI trends", task_type=TaskType.RESEARCH, priority=4)
        decision = self.router.route_task(t)
        assert decision.autonomy_level == AutonomyLevel.CONDITIONAL

    def test_content_gets_supervised(self):
        t = _task("Write a press release", task_type=TaskType.WRITING, priority=4)
        decision = self.router.route_task(t)
        assert decision.autonomy_level == AutonomyLevel.SUPERVISED

    def test_low_priority_technical_conditional(self):
        t = _task("Fix a minor bug", task_type=TaskType.CODING, priority=3)
        decision = self.router.route_task(t)
        assert decision.autonomy_level == AutonomyLevel.CONDITIONAL

    def test_confidence_threshold_matches_autonomy(self):
        t = _task("Research AI", task_type=TaskType.RESEARCH, priority=4)
        decision = self.router.route_task(t)
        # CONDITIONAL should have 0.8 threshold
        assert decision.confidence_threshold == 0.8


# ---------------------------------------------------------------------------
# ExpertRouter — Task Decomposition
# ---------------------------------------------------------------------------

class TestTaskDecomposition:
    def setup_method(self):
        self.router = ExpertRouter()

    def test_compound_task_decomposes(self):
        t = _task("Research the market and write a summary")
        subtasks = self.router.decompose_complex_task(t)
        assert len(subtasks) == 2

    def test_simple_task_no_decomposition(self):
        t = _task("Analyze the revenue")
        subtasks = self.router.decompose_complex_task(t)
        assert len(subtasks) == 1
        assert subtasks[0].id == t.id

    def test_semicolon_decomposition(self):
        t = _task("Check the logs; review the metrics")
        subtasks = self.router.decompose_complex_task(t)
        assert len(subtasks) == 2

    def test_subtask_ids_unique(self):
        t = _task("Do A and do B and do C")
        subtasks = self.router.decompose_complex_task(t)
        ids = [s.id for s in subtasks]
        assert len(ids) == len(set(ids))

    def test_subtasks_inherit_priority(self):
        t = _task("Do A and do B", priority=8)
        subtasks = self.router.decompose_complex_task(t)
        for st in subtasks:
            assert st.priority == 8


# ---------------------------------------------------------------------------
# ExpertRouter — Expert Cache
# ---------------------------------------------------------------------------

class TestExpertCache:
    def setup_method(self):
        self.router = ExpertRouter()

    def test_get_known_expert(self):
        expert = self.router.get_expert("ResearchExpert")
        assert expert is not None

    def test_cache_returns_same_instance(self):
        e1 = self.router.get_expert("AnalystExpert")
        e2 = self.router.get_expert("AnalystExpert")
        assert e1 is e2

    def test_unknown_expert_raises(self):
        with pytest.raises(ValueError, match="Unknown expert"):
            self.router.get_expert("NonExistentExpert")

    def test_all_known_experts_instantiate(self):
        names = ["ResearchExpert", "AnalystExpert", "WriterExpert",
                 "EngineerExpert", "CriticExpert", "StrategistExpert"]
        for name in names:
            expert = self.router.get_expert(name)
            assert expert is not None


# ---------------------------------------------------------------------------
# PipelineExecutor — Step Conversion
# ---------------------------------------------------------------------------

class TestPipelineStep:
    def test_to_task(self):
        step = PipelineStep(
            id="s1", name="Research", description="Research the market",
            task_type=TaskType.RESEARCH, inputs={"scope": "AI"}
        )
        task = step.to_task({"extra": "context"})
        assert task.id == "s1"
        assert task.description == "Research the market"
        assert task.task_type == TaskType.RESEARCH
        assert task.context["scope"] == "AI"
        assert task.context["extra"] == "context"

    def test_to_task_no_context(self):
        step = PipelineStep(id="s1", name="Test", description="Test step")
        task = step.to_task()
        assert task.context == {}


# ---------------------------------------------------------------------------
# PipelineExecutor — Pipeline Execution
# ---------------------------------------------------------------------------

class TestPipelineExecution:
    def setup_method(self):
        self.executor = PipelineExecutor(autonomy=AutonomyLevel.AUTONOMOUS)

    @pytest.mark.asyncio
    async def test_single_step_pipeline(self):
        step = PipelineStep(
            id="s1", name="Analyze", description="Analyze market trends",
            task_type=TaskType.ANALYSIS,
        )
        result = await self.executor.execute_pipeline([step])
        assert isinstance(result, PipelineResult)
        assert result.steps_total == 1

    @pytest.mark.asyncio
    async def test_sequential_pipeline(self):
        steps = [
            PipelineStep(id="s1", name="Research", description="Research trends",
                        task_type=TaskType.RESEARCH),
            PipelineStep(id="s2", name="Analyze", description="Analyze the research",
                        task_type=TaskType.ANALYSIS, dependencies=["s1"]),
        ]
        result = await self.executor.execute_pipeline(steps)
        assert result.steps_total == 2

    @pytest.mark.asyncio
    async def test_pipeline_id_auto_generated(self):
        step = PipelineStep(id="s1", name="Test", description="Test",
                          task_type=TaskType.GENERAL)
        result = await self.executor.execute_pipeline([step])
        assert result.pipeline_id is not None
        assert len(result.pipeline_id) > 0

    @pytest.mark.asyncio
    async def test_pipeline_id_custom(self):
        step = PipelineStep(id="s1", name="Test", description="Test",
                          task_type=TaskType.GENERAL)
        result = await self.executor.execute_pipeline([step], pipeline_id="my-pipe")
        assert result.pipeline_id == "my-pipe"

    @pytest.mark.asyncio
    async def test_duration_tracked(self):
        step = PipelineStep(id="s1", name="Test", description="Test",
                          task_type=TaskType.GENERAL)
        result = await self.executor.execute_pipeline([step])
        assert result.duration_seconds >= 0


# ---------------------------------------------------------------------------
# PipelineExecutor — Status
# ---------------------------------------------------------------------------

class TestPipelineStatus:
    def setup_method(self):
        self.executor = PipelineExecutor(autonomy=AutonomyLevel.AUTONOMOUS)

    @pytest.mark.asyncio
    async def test_get_status_after_execution(self):
        step = PipelineStep(id="s1", name="Test", description="Test",
                          task_type=TaskType.GENERAL)
        result = await self.executor.execute_pipeline([step], pipeline_id="test-pipe")
        status = self.executor.get_pipeline_status("test-pipe")
        assert status["pipeline_id"] == "test-pipe"
        assert status["total"] == 1

    def test_unknown_pipeline_returns_error(self):
        status = self.executor.get_pipeline_status("nonexistent")
        assert "error" in status


# ---------------------------------------------------------------------------
# PipelineExecutor — Approval Workflow
# ---------------------------------------------------------------------------

class TestApprovalWorkflow:
    def setup_method(self):
        self.executor = PipelineExecutor(autonomy=AutonomyLevel.FULL_APPROVAL)

    def test_register_callback(self):
        callback = MagicMock()
        self.executor.register_approval_callback(callback)
        assert callback in self.executor._approval_callbacks

    @pytest.mark.asyncio
    async def test_process_approval_unknown_pipeline(self):
        result = await self.executor.process_approval("nonexistent", "s1", True)
        assert result is False

    @pytest.mark.asyncio
    async def test_process_approval_unknown_step(self):
        # Set up a pipeline first
        step = PipelineStep(id="s1", name="Test", description="Test",
                          task_type=TaskType.GENERAL)
        await self.executor.execute_pipeline([step], pipeline_id="test-pipe")
        result = await self.executor.process_approval("test-pipe", "nonexistent", True)
        assert result is False


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TestTypes:
    def test_autonomy_levels(self):
        assert AutonomyLevel.AUTONOMOUS.value == "autonomous"
        assert AutonomyLevel.FULL_APPROVAL.value == "full_approval"

    def test_step_status(self):
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"

    def test_task_category(self):
        assert TaskCategory.RESEARCH.value == "research"
        assert TaskCategory.TECHNICAL.value == "technical"

    def test_approval_request_creation(self):
        req = ApprovalRequest(
            id="a1", step_id="s1", pipeline_id="p1",
            summary="Test", details={}, expert_opinions=[],
            consensus_confidence=0.8, recommended_action="proceed"
        )
        assert req.id == "a1"
        assert req.consensus_confidence == 0.8

    def test_duration_estimation_scales_with_description(self):
        router = ExpertRouter()
        short = _task("Analyze X", task_type=TaskType.ANALYSIS)
        long = _task("Analyze X " * 50, task_type=TaskType.ANALYSIS)
        d_short = router._estimate_duration(short, TaskCategory.ANALYSIS)
        d_long = router._estimate_duration(long, TaskCategory.ANALYSIS)
        assert d_long > d_short

    def test_duration_scales_with_priority(self):
        router = ExpertRouter()
        low = _task("Analyze X", priority=2)
        high = _task("Analyze X", priority=9)
        d_low = router._estimate_duration(low, TaskCategory.ANALYSIS)
        d_high = router._estimate_duration(high, TaskCategory.ANALYSIS)
        assert d_high > d_low
