"""Regression tests for autonomous research reporting semantics."""

import asyncio
from unittest.mock import AsyncMock, patch

from nexus.cog_eng.capabilities.autonomous_research_agent import (
    AutonomousResearchAgent,
    ResearchGoal,
)


def test_research_sub_goal_marks_simulated_llm_output_honestly():
    agent = AutonomousResearchAgent.__new__(AutonomousResearchAgent)
    agent.llm_client = AsyncMock()
    agent.llm_client.complete = AsyncMock(
        return_value={"content": "simulated content", "model": "simulated"}
    )

    sub_goal = ResearchGoal(
        goal_id="goal-1",
        description="Investigate orchestration",
        priority="normal",
        depth="moderate",
        sub_goals=[],
    )

    with patch(
        "nexus.cog_eng.capabilities.autonomous_research_agent.config.has_llm_key",
        return_value=True,
    ):
        findings = asyncio.run(agent._research_sub_goal(sub_goal))

    assert findings[0]["source"] == "simulated"
    assert findings[0]["verification_status"] == "simulated"
    assert findings[0]["evidence_type"] == "simulated"
    assert findings[0]["confidence"] == 0.35


def test_verify_findings_keeps_simulated_entries_out_of_verified_bucket():
    agent = AutonomousResearchAgent.__new__(AutonomousResearchAgent)
    agent.performance_metrics = {"contradictions_detected": 0}

    findings = [
        {
            "finding_id": "finding-1",
            "content": "simulated content",
            "source": "simulated",
            "confidence": 0.35,
            "verification_status": "simulated",
            "evidence_type": "simulated",
            "supporting_evidence": [],
            "contradictions": [],
            "timestamp": "2026-03-14T08:00:00",
        }
    ]

    verified = asyncio.run(agent._verify_findings(findings))

    assert verified[0]["verification_status"] == "simulated"
    assert verified[0]["supporting_evidence"] == []
    assert verified[0]["confidence"] == 0.35


def test_research_marks_main_goal_completed_when_sub_goals_finish():
    agent = AutonomousResearchAgent.__new__(AutonomousResearchAgent)
    agent.research_history = []
    agent.performance_metrics = {
        "total_research_tasks": 0,
        "successful_tasks": 0,
        "average_confidence": 0.0,
        "contradictions_detected": 0,
        "contradictions_resolved": 0,
        "knowledge_nodes_created": 0,
    }

    main_goal = ResearchGoal(
        goal_id="topic-main",
        description="topic",
        priority="normal",
        depth="moderate",
        sub_goals=[
            ResearchGoal(
                goal_id="topic-sub-1",
                description="sub 1",
                priority="normal",
                depth="moderate",
                sub_goals=[],
                completed=True,
                confidence=0.4,
            ),
            ResearchGoal(
                goal_id="topic-sub-2",
                description="sub 2",
                priority="normal",
                depth="moderate",
                sub_goals=[],
                completed=True,
                confidence=0.6,
            ),
        ],
    )

    findings = [
        {
            "finding_id": "finding-1",
            "content": "simulated content",
            "source": "simulated",
            "confidence": 0.35,
            "verification_status": "simulated",
            "supporting_evidence": [],
            "contradictions": [],
            "timestamp": "2026-03-14T08:00:00",
        }
    ]
    synthesis = {
        "overall_confidence": 0.35,
        "verified_findings_count": 0,
        "simulated_findings_count": 1,
        "unverified_findings_count": 0,
        "contradiction_count": 0,
        "knowledge_nodes": [],
        "key_insights": [],
        "recommendations": [],
        "confidence_breakdown": {
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 1,
        },
    }

    agent._decompose_research_goal = AsyncMock(return_value=main_goal)
    agent._execute_research = AsyncMock(return_value=findings)
    agent._verify_findings = AsyncMock(return_value=findings)
    agent._detect_contradictions = AsyncMock(return_value=[])
    agent._synthesize_knowledge = AsyncMock(return_value=synthesis)
    agent._update_and_learn = AsyncMock(return_value=None)

    report = asyncio.run(agent.research("topic"))

    assert report["main_goal"]["completed"] is True
    assert report["main_goal"]["confidence"] == 0.5
    assert report["confidence"] == 0.35


def test_execute_research_does_not_spawn_recursive_verification_for_simulated_findings():
    agent = AutonomousResearchAgent.__new__(AutonomousResearchAgent)

    async def fake_research_sub_goal(sub_goal):
        return [
            {
                "finding_id": f"{sub_goal.goal_id}-finding",
                "content": "simulated content",
                "source": "simulated",
                "confidence": 0.35,
                "verification_status": "simulated",
                "evidence_type": "simulated",
                "supporting_evidence": [],
                "contradictions": [],
                "timestamp": "2026-03-14T08:00:00",
            }
        ]

    agent._research_sub_goal = fake_research_sub_goal

    main_goal = ResearchGoal(
        goal_id="topic-main",
        description="topic",
        priority="normal",
        depth="moderate",
        sub_goals=[
            ResearchGoal(
                goal_id="topic-sub-1",
                description="sub 1",
                priority="normal",
                depth="moderate",
                sub_goals=[],
            ),
            ResearchGoal(
                goal_id="topic-sub-2",
                description="sub 2",
                priority="normal",
                depth="moderate",
                sub_goals=[],
            ),
        ],
    )

    findings = asyncio.run(agent._execute_research(main_goal, max_iterations=5))

    assert len(findings) == 2
    assert [goal.goal_id for goal in main_goal.sub_goals] == ["topic-sub-1", "topic-sub-2"]
