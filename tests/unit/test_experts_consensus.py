"""Regression tests for expert consensus orchestration."""

import asyncio

from nexus.experts.base import ExpertOpinion, Task, TaskType
from nexus.experts.consensus import ConsensusEngine, ConsensusStrategy
from nexus.platform import NexusPlatform


class StubExpert:
    def __init__(self, name, role, score, recommendation):
        self.persona = type(
            "Persona",
            (),
            {
                "weight": 1.0,
                "matches_task": lambda self_persona, task: score,
            },
        )()
        self._name = name
        self._role = role
        self._recommendation = recommendation

    async def analyze(self, task):
        return ExpertOpinion(
            expert_name=self._name,
            expert_role=self._role,
            analysis=f"{self._name} analysis",
            recommendation=self._recommendation,
            confidence=0.8,
            reasoning=f"{self._name} reasoning",
        )


def test_consensus_engine_coerces_string_task_and_returns_payload():
    engine = ConsensusEngine(platform=object())
    engine._select_experts = lambda task, max_experts=None: [
        StubExpert("ResearchExpert", "Research", 1.0, "Research further"),
        StubExpert("StrategistExpert", "Strategy", 0.9, "Research further"),
    ]

    result = asyncio.run(engine.get_consensus("Assess AI roadmap"))

    assert result["task"]["description"] == "Assess AI roadmap"
    assert result["task"]["task_type"] == TaskType.GENERAL.value
    assert result["consensus"]["strategy_used"] == ConsensusStrategy.SYNTHESIZED.value
    assert len(result["opinions"]) == 2
    assert "Recommended next step:" in result["consensus"]["decision"]
    assert result["consensus"]["metadata"]["synthesis_method"] == "heuristic"


def test_consensus_engine_accepts_dict_task_and_strategy_string():
    engine = ConsensusEngine(platform=object())
    engine._select_experts = lambda task, max_experts=None: [
        StubExpert("AnalystExpert", "Analysis", 1.0, "Proceed"),
        StubExpert("CriticExpert", "Review", 1.0, "Hold"),
    ]

    result = asyncio.run(
        engine.get_consensus(
            {
                "id": "task-1",
                "description": "Review system metrics",
                "task_type": "analysis",
                "priority": 7,
                "constraints": ["be concise"],
            },
            strategy="majority",
        )
    )

    assert result["task"]["id"] == "task-1"
    assert result["task"]["task_type"] == TaskType.ANALYSIS.value
    assert result["consensus"]["strategy_used"] == ConsensusStrategy.MAJORITY.value


def test_platform_get_expert_opinion_uses_consensus_engine_payload():
    class StubExperts:
        async def get_consensus(self, task):
            assert task == "Recommend next step"
            return {"consensus": {"decision": "Proceed"}, "opinions": []}

    platform = NexusPlatform()
    platform._experts = StubExperts()

    result = asyncio.run(platform.get_expert_opinion("Recommend next step"))

    assert result["consensus"]["decision"] == "Proceed"


def test_platform_initialize_expert_path_creates_platform_aware_engine():
    platform = NexusPlatform()

    asyncio.run(platform.initialize_expert_path())

    assert platform._experts is not None
    assert getattr(platform._experts, "platform", None) is platform


def test_consensus_engine_skips_failed_experts_and_records_errors():
    class FailingExpert:
        def __init__(self):
            self.name = "FailingExpert"
            self.role = "Reviewer"

        async def analyze(self, task):
            raise RuntimeError("backend unavailable")

    engine = ConsensusEngine(platform=object())
    engine._select_experts = lambda task, max_experts=None: [
        StubExpert("ResearchExpert", "Research", 1.0, "Research further"),
        FailingExpert(),
    ]

    result = asyncio.run(engine.get_consensus("Assess AI roadmap"))

    assert len(result["opinions"]) == 1
    assert result["consensus"]["participating_experts"] == ["ResearchExpert"]
    assert result["consensus"]["metadata"]["failed_experts"] == [
        {
            "expert_name": "FailingExpert",
            "expert_role": "Reviewer",
            "error": "backend unavailable",
        }
    ]


def test_synthesized_decision_prefers_actionable_recommendation_and_concern_summary():
    engine = ConsensusEngine(platform=object())
    opinions = [
        ExpertOpinion(
            expert_name="WriterExpert",
            expert_role="Content Writer",
            analysis="Writer analysis",
            recommendation="Document the rollout plan",
            confidence=0.85,
            reasoning="writer",
        ),
        ExpertOpinion(
            expert_name="AnalystExpert",
            expert_role="Data Analyst",
            analysis="Analyst analysis",
            recommendation="Run a deployment readiness audit before release",
            confidence=0.75,
            reasoning="analyst",
        ),
        ExpertOpinion(
            expert_name="CriticExpert",
            expert_role="Quality Reviewer",
            analysis="Critic analysis",
            recommendation="Review rollback readiness",
            confidence=0.8,
            reasoning="critic",
            concerns=["Missing rollback validation"],
        ),
    ]

    result = engine.reach_consensus(
        opinions,
        Task(id="task-1", description="Deploy service", task_type=TaskType.GENERAL),
        strategy=ConsensusStrategy.SYNTHESIZED,
    )

    assert result.decision.startswith("Recommended next step: Run a deployment readiness audit before release.")
    assert "Supporting views:" in result.decision
    assert "Key risk: Missing rollback validation." in result.decision
    assert result.metadata["lead_recommendation"] == "Run a deployment readiness audit before release"


def test_critic_concern_extraction_ignores_headings():
    from nexus.experts.personas.critic import CriticExpert

    expert = CriticExpert(platform=None)
    concerns = expert._extract_concerns(
        """
        ## 1. Potential Issues and Risks
        Major Concerns:
        - Missing rollback validation before production.
        - Risk of incomplete monitoring coverage.
        """
    )

    assert concerns == [
        "Missing rollback validation before production.",
        "Risk of incomplete monitoring coverage.",
    ]


def test_synthesized_decision_supporting_views_stay_actionable():
    engine = ConsensusEngine(platform=object())
    opinions = [
        ExpertOpinion(
            expert_name="AnalystExpert",
            expert_role="Data Analyst",
            analysis="Analyst analysis",
            recommendation="Run a deployment readiness audit before release",
            confidence=0.75,
            reasoning="analyst",
        ),
        ExpertOpinion(
            expert_name="StrategistExpert",
            expert_role="Strategic Advisor",
            analysis="Strategist analysis",
            recommendation="Consider trade-offs between speed and certainty",
            confidence=0.8,
            reasoning="strategist",
        ),
        ExpertOpinion(
            expert_name="CriticExpert",
            expert_role="Quality Reviewer",
            analysis="Critic analysis",
            recommendation="Validate rollback readiness before deployment",
            confidence=0.8,
            reasoning="critic",
            concerns=["Missing rollback validation"],
        ),
    ]

    result = engine.reach_consensus(
        opinions,
        Task(id="task-1", description="Deploy service", task_type=TaskType.GENERAL),
        strategy=ConsensusStrategy.SYNTHESIZED,
    )

    assert "Supporting views: Validate rollback readiness before deployment." in result.decision
    assert "Consider trade-offs between speed and certainty" not in result.decision


def test_synthesized_decision_falls_back_to_task_specific_default_when_recommendations_are_abstract():
    engine = ConsensusEngine(platform=object())
    opinions = [
        ExpertOpinion(
            expert_name="StrategistExpert",
            expert_role="Strategic Advisor",
            analysis="Strategist analysis",
            recommendation="Key Trade-offs",
            confidence=0.8,
            reasoning="strategist",
        ),
        ExpertOpinion(
            expert_name="WriterExpert",
            expert_role="Content Writer",
            analysis="Writer analysis",
            recommendation="Deployment Readiness Recommendation",
            confidence=0.85,
            reasoning="writer",
        ),
    ]

    result = engine.reach_consensus(
        opinions,
        Task(id="task-1", description="Recommend next step for deployment readiness", task_type=TaskType.GENERAL),
        strategy=ConsensusStrategy.SYNTHESIZED,
    )

    assert result.metadata["lead_recommendation"] == "Run a deployment readiness assessment before proceeding"
    assert result.decision.startswith("Recommended next step: Run a deployment readiness assessment before proceeding.")
