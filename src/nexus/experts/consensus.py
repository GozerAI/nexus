"""
Consensus Engine - Multi-expert voting and conflict resolution
"""

import asyncio
from dataclasses import asdict
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import statistics
import re

from .base import ExpertOpinion, Task, TaskType
from .personas import ALL_EXPERTS


class ConsensusStrategy(Enum):
    """Strategies for reaching consensus."""
    WEIGHTED_VOTE = "weighted_vote"      # Weight by expert confidence and role fit
    MAJORITY = "majority"                 # Simple majority wins
    UNANIMOUS = "unanimous"               # All must agree
    HIGHEST_CONFIDENCE = "highest_conf"   # Trust most confident expert
    SYNTHESIZED = "synthesized"           # Combine all perspectives


@dataclass
class ConsensusResult:
    """Result of consensus process."""
    decision: str
    confidence: float
    strategy_used: ConsensusStrategy
    participating_experts: List[str]
    agreement_level: float  # 0-1, how much experts agreed
    synthesis: str  # Combined reasoning
    dissenting_views: List[str] = field(default_factory=list)
    voting_breakdown: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusEngine:
    """
    Aggregates expert opinions and reaches consensus through voting.
    
    Supports multiple strategies:
    - Weighted voting based on confidence and task fit
    - Majority voting
    - Unanimous agreement requirement  
    - Highest confidence selection
    - Synthesized combination of all views
    """
    
    def __init__(
        self,
        default_strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_VOTE,
        platform: Any = None,
    ):
        self.default_strategy = default_strategy
        self.platform = platform
        self._history: List[ConsensusResult] = []

    async def get_consensus(
        self,
        task: Task | Dict[str, Any] | str,
        strategy: Optional[ConsensusStrategy | str] = None,
        max_experts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Collect expert analyses and return a machine-friendly consensus payload."""
        normalized_task = self._normalize_task(task)
        selected_strategy = self._normalize_strategy(strategy) or ConsensusStrategy.SYNTHESIZED
        experts = self._select_experts(normalized_task, max_experts=max_experts)

        if not experts:
            empty = self.reach_consensus([], normalized_task, strategy=selected_strategy)
            return self._serialize_result(empty, normalized_task, [])

        gathered = await asyncio.gather(
            *(expert.analyze(normalized_task) for expert in experts),
            return_exceptions=True,
        )
        opinions: List[ExpertOpinion] = []
        failed_experts: List[Dict[str, str]] = []
        for expert, result in zip(experts, gathered):
            if isinstance(result, Exception):
                failed_experts.append({
                    "expert_name": expert.name,
                    "expert_role": expert.role,
                    "error": str(result),
                })
                continue
            opinions.append(result)

        consensus = self.reach_consensus(opinions, normalized_task, strategy=selected_strategy)
        payload = self._serialize_result(consensus, normalized_task, opinions)
        if failed_experts:
            payload["consensus"]["metadata"]["failed_experts"] = failed_experts
        return payload

    def _normalize_task(self, task: Task | Dict[str, Any] | str) -> Task:
        """Coerce public task inputs into a Task object."""
        if isinstance(task, Task):
            return task
        if isinstance(task, str):
            description = task.strip()
            return Task(id=f"task_{int(datetime.now().timestamp())}", description=description or "General task")
        if isinstance(task, dict):
            task_type = task.get("task_type", TaskType.GENERAL)
            if isinstance(task_type, str):
                try:
                    task_type = TaskType(task_type.lower())
                except ValueError:
                    task_type = TaskType.GENERAL
            return Task(
                id=str(task.get("id", f"task_{int(datetime.now().timestamp())}")),
                description=str(task.get("description") or task.get("prompt") or "General task"),
                task_type=task_type,
                context=dict(task.get("context", {})),
                constraints=[str(item) for item in task.get("constraints", [])],
                priority=int(task.get("priority", 5)),
            )
        raise TypeError(f"Unsupported task type for consensus: {type(task)!r}")

    def _normalize_strategy(
        self,
        strategy: Optional[ConsensusStrategy | str],
    ) -> Optional[ConsensusStrategy]:
        """Coerce public strategy inputs into ConsensusStrategy."""
        if strategy is None:
            return None
        if isinstance(strategy, ConsensusStrategy):
            return strategy
        if isinstance(strategy, str):
            try:
                return ConsensusStrategy(strategy.lower())
            except ValueError:
                return self.default_strategy
        return self.default_strategy

    def _select_experts(self, task: Task, max_experts: Optional[int] = None) -> List[Any]:
        """Select the best-matching experts for a task."""
        if self.platform is None:
            return []

        ranked = []
        for expert_cls in ALL_EXPERTS:
            expert = expert_cls(platform=self.platform)
            score = expert.persona.matches_task(task) * expert.persona.weight
            ranked.append((score, expert))

        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [expert for score, expert in ranked if score >= 0.5]
        if not selected:
            selected = [expert for _, expert in ranked[:3]]
        if max_experts is not None:
            selected = selected[:max_experts]
        return selected

    def _serialize_result(
        self,
        result: ConsensusResult,
        task: Task,
        opinions: List[ExpertOpinion],
    ) -> Dict[str, Any]:
        """Return a machine-friendly consensus payload."""
        return {
            "task": {
                "id": task.id,
                "description": task.description,
                "task_type": task.task_type.value,
                "priority": task.priority,
                "constraints": list(task.constraints),
                "context": dict(task.context),
            },
            "consensus": {
                "decision": result.decision,
                "confidence": result.confidence,
                "strategy_used": result.strategy_used.value,
                "participating_experts": list(result.participating_experts),
                "agreement_level": result.agreement_level,
                "synthesis": result.synthesis,
                "dissenting_views": list(result.dissenting_views),
                "voting_breakdown": dict(result.voting_breakdown),
                "timestamp": result.timestamp.isoformat(),
                "metadata": dict(result.metadata),
            },
            "opinions": [asdict(opinion) for opinion in opinions],
            "conflicts": self.detect_conflicts(opinions) if opinions else [],
        }
    
    def reach_consensus(
        self,
        opinions: List[ExpertOpinion],
        task: Task,
        strategy: Optional[ConsensusStrategy] = None
    ) -> ConsensusResult:
        """
        Reach consensus from multiple expert opinions.
        
        Args:
            opinions: List of expert opinions to consider
            task: The original task being decided on
            strategy: Consensus strategy to use (defaults to engine default)
            
        Returns:
            ConsensusResult with decision and metadata
        """
        if not opinions:
            return ConsensusResult(
                decision="No opinions provided",
                confidence=0.0,
                strategy_used=strategy or self.default_strategy,
                participating_experts=[],
                agreement_level=0.0,
                synthesis=""
            )
        
        strategy = strategy or self.default_strategy
        
        if strategy == ConsensusStrategy.WEIGHTED_VOTE:
            result = self._weighted_vote(opinions, task)
        elif strategy == ConsensusStrategy.MAJORITY:
            result = self._majority_vote(opinions)
        elif strategy == ConsensusStrategy.UNANIMOUS:
            result = self._unanimous_vote(opinions)
        elif strategy == ConsensusStrategy.HIGHEST_CONFIDENCE:
            result = self._highest_confidence(opinions)
        elif strategy == ConsensusStrategy.SYNTHESIZED:
            result = self._synthesize(opinions, task)
        else:
            result = self._weighted_vote(opinions, task)
        
        result.strategy_used = strategy
        self._history.append(result)
        return result

    def _weighted_vote(self, opinions: List[ExpertOpinion], task: Task) -> ConsensusResult:
        """Weighted voting based on confidence and task fit."""
        weighted_scores = {}
        
        for opinion in opinions:
            # Base weight from confidence
            weight = opinion.confidence
            
            # Bonus for task-type match (would need persona info)
            # For now, use confidence as primary weight
            weighted_scores[opinion.expert_name] = {
                "weight": weight,
                "recommendation": opinion.recommendation,
                "analysis": opinion.analysis
            }
        
        # Aggregate
        total_weight = sum(s["weight"] for s in weighted_scores.values())
        avg_confidence = total_weight / len(opinions) if opinions else 0
        
        # Find highest weighted recommendation
        best = max(weighted_scores.items(), key=lambda x: x[1]["weight"])
        
        # Calculate agreement
        confidences = [o.confidence for o in opinions]
        agreement = 1.0 - (statistics.stdev(confidences) if len(confidences) > 1 else 0)
        
        return ConsensusResult(
            decision=best[1]["recommendation"],
            confidence=avg_confidence,
            strategy_used=ConsensusStrategy.WEIGHTED_VOTE,
            participating_experts=[o.expert_name for o in opinions],
            agreement_level=agreement,
            synthesis=self._create_synthesis(opinions),
            voting_breakdown={k: v["weight"] for k, v in weighted_scores.items()}
        )
    
    def _majority_vote(self, opinions: List[ExpertOpinion]) -> ConsensusResult:
        """Simple majority voting."""
        # Group by recommendation
        votes = {}
        for opinion in opinions:
            rec = opinion.recommendation
            if rec not in votes:
                votes[rec] = []
            votes[rec].append(opinion)
        
        # Find majority
        majority_rec = max(votes.keys(), key=lambda k: len(votes[k]))
        majority_count = len(votes[majority_rec])
        
        agreement = majority_count / len(opinions) if opinions else 0
        avg_conf = statistics.mean([o.confidence for o in votes[majority_rec]])
        
        # Dissenting views
        dissenting = []
        for rec, ops in votes.items():
            if rec != majority_rec:
                dissenting.extend([f"{o.expert_name}: {o.recommendation}" for o in ops])
        
        return ConsensusResult(
            decision=majority_rec,
            confidence=avg_conf,
            strategy_used=ConsensusStrategy.MAJORITY,
            participating_experts=[o.expert_name for o in opinions],
            agreement_level=agreement,
            synthesis=self._create_synthesis(opinions),
            dissenting_views=dissenting
        )
    
    def _unanimous_vote(self, opinions: List[ExpertOpinion]) -> ConsensusResult:
        """Require unanimous agreement."""
        recommendations = set(o.recommendation for o in opinions)
        
        if len(recommendations) == 1:
            # Unanimous
            return ConsensusResult(
                decision=list(recommendations)[0],
                confidence=statistics.mean([o.confidence for o in opinions]),
                strategy_used=ConsensusStrategy.UNANIMOUS,
                participating_experts=[o.expert_name for o in opinions],
                agreement_level=1.0,
                synthesis=self._create_synthesis(opinions)
            )
        else:
            # No consensus
            return ConsensusResult(
                decision="NO CONSENSUS - Experts disagree",
                confidence=0.0,
                strategy_used=ConsensusStrategy.UNANIMOUS,
                participating_experts=[o.expert_name for o in opinions],
                agreement_level=1.0 / len(recommendations),
                synthesis=self._create_synthesis(opinions),
                dissenting_views=[f"{o.expert_name}: {o.recommendation}" for o in opinions]
            )

    def _highest_confidence(self, opinions: List[ExpertOpinion]) -> ConsensusResult:
        """Trust the most confident expert."""
        best = max(opinions, key=lambda o: o.confidence)
        
        # Others as dissenting
        others = [o for o in opinions if o.expert_name != best.expert_name]
        
        return ConsensusResult(
            decision=best.recommendation,
            confidence=best.confidence,
            strategy_used=ConsensusStrategy.HIGHEST_CONFIDENCE,
            participating_experts=[o.expert_name for o in opinions],
            agreement_level=best.confidence,  # Use confidence as proxy
            synthesis=f"Deferred to {best.expert_name} (highest confidence: {best.confidence:.0%})",
            dissenting_views=[f"{o.expert_name}: {o.recommendation}" for o in others]
        )
    
    def _synthesize(self, opinions: List[ExpertOpinion], task: Task) -> ConsensusResult:
        """Synthesize all perspectives into combined view."""
        synthesis = self._create_synthesis(opinions)
        
        # Average confidence
        avg_conf = statistics.mean([o.confidence for o in opinions])
        
        # Agreement based on confidence variance
        conf_std = statistics.stdev([o.confidence for o in opinions]) if len(opinions) > 1 else 0
        agreement = 1.0 - min(conf_std, 1.0)
        
        combined, decision_metadata = self._build_synthesized_decision(opinions, task)
        
        return ConsensusResult(
            decision=combined,
            confidence=avg_conf,
            strategy_used=ConsensusStrategy.SYNTHESIZED,
            participating_experts=[o.expert_name for o in opinions],
            agreement_level=agreement,
            synthesis=synthesis,
            voting_breakdown={o.expert_name: o.confidence for o in opinions},
            metadata=decision_metadata,
        )

    def _build_synthesized_decision(
        self,
        opinions: List[ExpertOpinion],
        task: Task,
    ) -> tuple[str, Dict[str, Any]]:
        """Build a concise decision summary from expert recommendations."""
        ranked = sorted(
            opinions,
            key=self._decision_priority,
            reverse=True,
        )

        cleaned_recommendations: list[str] = []
        for opinion in ranked:
            cleaned = self._clean_recommendation(opinion.recommendation)
            if cleaned and cleaned not in cleaned_recommendations:
                cleaned_recommendations.append(cleaned)

        actionable_recommendations = [
            recommendation
            for recommendation in cleaned_recommendations
            if self._looks_actionable(recommendation)
        ]

        lead = (
            actionable_recommendations[0]
            if actionable_recommendations
            else self._default_synthesized_recommendation(task, opinions)
        )
        supporting = [
            recommendation
            for recommendation in actionable_recommendations[1:]
            if recommendation != lead
        ][:2]
        top_concerns = self._top_concerns(opinions, limit=2)

        parts = [f"Recommended next step: {lead}."]
        if supporting:
            parts.append("Supporting views: " + "; ".join(supporting) + ".")
        if top_concerns:
            parts.append("Key risk" + ("s" if len(top_concerns) > 1 else "") + ": " + "; ".join(top_concerns) + ".")

        return " ".join(parts), {
            "lead_recommendation": lead,
            "supporting_recommendations": supporting,
            "top_concerns": top_concerns,
            "synthesis_method": "heuristic",
        }

    def _decision_priority(self, opinion: ExpertOpinion) -> tuple[float, float, float]:
        """Rank expert recommendations for synthesized decisions."""
        role_priority = {
            "Strategic Advisor": 1.0,
            "Data Analyst": 0.95,
            "Quality Reviewer": 0.92,
            "Research Specialist": 0.9,
            "Software Engineer": 0.9,
            "Content Writer": 0.6,
        }
        role_score = role_priority.get(opinion.expert_role, 0.75)
        actionability = 1.0 if self._looks_actionable(opinion.recommendation) else 0.0
        return (actionability, role_score, opinion.confidence)

    def _looks_actionable(self, text: str) -> bool:
        """Return True when a recommendation looks like a next action."""
        normalized = text.lower()
        verbs = (
            "run ",
            "define ",
            "conduct ",
            "proceed",
            "halt",
            "delay",
            "assess ",
            "review ",
            "deploy ",
            "prepare ",
            "validate ",
            "gather ",
            "establish ",
            "execute ",
            "create ",
        )
        return any(verb in normalized for verb in verbs)

    def _clean_recommendation(self, text: str) -> str:
        """Normalize markdown-heavy recommendation fragments."""
        cleaned = re.sub(r"[*_`#>\"]", "", text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned.strip(" :-")
        if not cleaned:
            return ""
        if not cleaned.endswith((".", "!", "?")):
            cleaned = cleaned[:220].rstrip(" .")
        return cleaned[:220]

    def _top_concerns(self, opinions: List[ExpertOpinion], limit: int = 2) -> list[str]:
        """Return the most informative concerns across expert opinions."""
        concerns_by_text: Dict[str, float] = defaultdict(float)
        for opinion in opinions:
            for concern in opinion.concerns:
                cleaned = self._clean_recommendation(concern)
                if cleaned and self._looks_like_risk(cleaned):
                    concerns_by_text[cleaned] = max(concerns_by_text[cleaned], opinion.confidence)

        ranked = sorted(
            concerns_by_text.items(),
            key=lambda item: (self._looks_actionable(item[0]), item[1], len(item[0])),
            reverse=True,
        )
        return [text for text, _ in ranked[:limit]]

    def _looks_like_risk(self, text: str) -> bool:
        """Return True when a concern looks like an actual risk statement."""
        normalized = text.lower()
        risk_terms = (
            "risk",
            "missing",
            "lack",
            "insufficient",
            "failure",
            "gap",
            "warning",
            "concern",
            "unacceptable",
        )
        return any(term in normalized for term in risk_terms)

    def _default_synthesized_recommendation(
        self,
        task: Task,
        opinions: List[ExpertOpinion],
    ) -> str:
        """Return a sane fallback recommendation when experts stay too abstract."""
        normalized_description = task.description.lower()
        if "deploy" in normalized_description or "deployment" in normalized_description:
            return "Run a deployment readiness assessment before proceeding"
        if task.task_type == TaskType.RESEARCH:
            return "Define the research scope and gather the missing context before proceeding"
        if task.task_type == TaskType.REVIEW:
            return "Define review criteria and validate the highest-risk areas first"
        if task.task_type == TaskType.CODING:
            return "Define implementation constraints and validate the technical approach before execution"
        return f"Define explicit success criteria and review {task.description} before proceeding"
    
    def _create_synthesis(self, opinions: List[ExpertOpinion]) -> str:
        """Create a synthesis of all expert analyses."""
        parts = []
        for opinion in opinions:
            summary = opinion.analysis[:200] + "..." if len(opinion.analysis) > 200 else opinion.analysis
            parts.append(f"**{opinion.expert_name}** ({opinion.confidence:.0%}): {summary}")
        return "\n\n".join(parts)
    
    def detect_conflicts(self, opinions: List[ExpertOpinion]) -> List[Dict[str, Any]]:
        """Detect conflicts between expert opinions."""
        conflicts = []
        
        for i, op1 in enumerate(opinions):
            for op2 in opinions[i+1:]:
                # Check for significant confidence gap with different recommendations
                conf_diff = abs(op1.confidence - op2.confidence)
                rec_differ = op1.recommendation != op2.recommendation
                
                if rec_differ and conf_diff > 0.3:
                    conflicts.append({
                        "experts": [op1.expert_name, op2.expert_name],
                        "type": "recommendation_conflict",
                        "severity": conf_diff,
                        "details": f"{op1.expert_name} recommends '{op1.recommendation}' vs {op2.expert_name} recommends '{op2.recommendation}'"
                    })
                
                # Check for opposing concerns
                op1_concerns = set(op1.concerns)
                op2_concerns = set(op2.concerns)
                if op1_concerns and op2_concerns and not op1_concerns.intersection(op2_concerns):
                    conflicts.append({
                        "experts": [op1.expert_name, op2.expert_name],
                        "type": "concern_mismatch",
                        "severity": 0.5,
                        "details": "Experts have non-overlapping concerns"
                    })
        
        return conflicts
    
    def get_history(self, limit: int = 10) -> List[ConsensusResult]:
        """Get recent consensus history."""
        return self._history[-limit:]
