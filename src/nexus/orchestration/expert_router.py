"""
Expert Router - Routes tasks to appropriate experts based on task analysis
"""

import asyncio
from typing import List, Dict, Any, Optional, Type
from dataclasses import dataclass, field

from nexus.experts.base import Task, TaskType, BaseExpert, ExpertOpinion
from nexus.experts.personas import (
    ResearchExpert, AnalystExpert, WriterExpert,
    EngineerExpert, CriticExpert, StrategistExpert,
    ALL_PERSONAS
)
from nexus.experts.consensus import ConsensusEngine, ConsensusResult, ConsensusStrategy
from .types import AutonomyLevel, TaskCategory


@dataclass
class RoutingDecision:
    """Decision about how to route a task."""
    task: Task
    primary_experts: List[str]
    supporting_experts: List[str]
    autonomy_level: AutonomyLevel
    confidence_threshold: float
    reasoning: str
    estimated_duration: float  # minutes
    requires_review: bool = True


class ExpertRouter:
    """
    Routes tasks to appropriate experts based on task analysis.
    
    Responsibilities:
    - Analyze incoming tasks
    - Select appropriate experts
    - Decompose complex tasks
    - Determine autonomy level
    """
    
    # Mapping from task categories to expert types
    CATEGORY_EXPERT_MAP = {
        TaskCategory.RESEARCH: [ResearchExpert, AnalystExpert],
        TaskCategory.CONTENT: [WriterExpert, CriticExpert],
        TaskCategory.TECHNICAL: [EngineerExpert, CriticExpert],
        TaskCategory.ANALYSIS: [AnalystExpert, ResearchExpert],
        TaskCategory.REVIEW: [CriticExpert, AnalystExpert],
        TaskCategory.STRATEGY: [StrategistExpert, AnalystExpert],
    }
    
    # Keywords for task categorization
    CATEGORY_KEYWORDS = {
        TaskCategory.RESEARCH: ["research", "find", "discover", "investigate", "learn"],
        TaskCategory.CONTENT: ["write", "create", "draft", "content", "article", "ebook"],
        TaskCategory.TECHNICAL: ["code", "build", "implement", "develop", "fix", "debug"],
        TaskCategory.ANALYSIS: ["analyze", "compare", "evaluate", "assess", "measure"],
        TaskCategory.REVIEW: ["review", "check", "verify", "validate", "quality"],
        TaskCategory.STRATEGY: ["plan", "strategy", "decide", "recommend", "should we"],
    }
    
    def __init__(self, platform=None):
        self.platform = platform
        self.consensus_engine = ConsensusEngine()
        self._expert_cache: Dict[str, BaseExpert] = {}
    
    def route_task(self, task: Task) -> RoutingDecision:
        """
        Analyze task and determine routing.
        
        Args:
            task: The task to route
            
        Returns:
            RoutingDecision with expert assignments and autonomy level
        """
        # Categorize the task
        category = self._categorize_task(task)
        
        # Get experts for this category
        expert_types = self.CATEGORY_EXPERT_MAP.get(category, [StrategistExpert])
        primary = [et.__name__ for et in expert_types[:2]]
        supporting = [et.__name__ for et in expert_types[2:]]
        
        # Add critic for review if high priority
        if task.priority >= 7 and "CriticExpert" not in primary:
            supporting.append("CriticExpert")
        
        # Determine autonomy level
        autonomy = self._determine_autonomy(task, category)
        
        # Set confidence threshold based on autonomy
        thresholds = {
            AutonomyLevel.AUTONOMOUS: 0.9,
            AutonomyLevel.CONDITIONAL: 0.8,
            AutonomyLevel.SUPERVISED: 0.7,
            AutonomyLevel.FULL_APPROVAL: 0.0,
        }
        
        return RoutingDecision(
            task=task,
            primary_experts=primary,
            supporting_experts=supporting,
            autonomy_level=autonomy,
            confidence_threshold=thresholds[autonomy],
            reasoning=f"Categorized as {category.value}, assigned {len(primary)} primary experts",
            estimated_duration=self._estimate_duration(task, category),
            requires_review=task.priority >= 5
        )

    def _categorize_task(self, task: Task) -> TaskCategory:
        """Categorize task based on description and type."""
        description_lower = task.description.lower()
        
        # Check task type first
        type_mapping = {
            TaskType.RESEARCH: TaskCategory.RESEARCH,
            TaskType.ANALYSIS: TaskCategory.ANALYSIS,
            TaskType.WRITING: TaskCategory.CONTENT,
            TaskType.CODING: TaskCategory.TECHNICAL,
            TaskType.REVIEW: TaskCategory.REVIEW,
            TaskType.STRATEGY: TaskCategory.STRATEGY,
        }
        
        if task.task_type in type_mapping:
            return type_mapping[task.task_type]
        
        # Fall back to keyword analysis
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in description_lower)
            scores[category] = score
        
        if scores:
            best = max(scores.items(), key=lambda x: x[1])
            if best[1] > 0:
                return best[0]
        
        return TaskCategory.STRATEGY  # Default
    
    def _determine_autonomy(self, task: Task, category: TaskCategory) -> AutonomyLevel:
        """Determine appropriate autonomy level for task."""
        # High priority always needs approval
        if task.priority >= 8:
            return AutonomyLevel.FULL_APPROVAL
        
        # Financial/legal context needs supervision
        sensitive_keywords = ["money", "payment", "legal", "contract", "delete", "remove"]
        if any(kw in task.description.lower() for kw in sensitive_keywords):
            return AutonomyLevel.SUPERVISED
        
        # Research and analysis can be more autonomous
        if category in [TaskCategory.RESEARCH, TaskCategory.ANALYSIS]:
            return AutonomyLevel.CONDITIONAL
        
        # Content creation needs review
        if category == TaskCategory.CONTENT:
            return AutonomyLevel.SUPERVISED
        
        # Technical tasks depend on priority
        if category == TaskCategory.TECHNICAL:
            return AutonomyLevel.CONDITIONAL if task.priority < 5 else AutonomyLevel.SUPERVISED
        
        return AutonomyLevel.SUPERVISED
    
    def _estimate_duration(self, task: Task, category: TaskCategory) -> float:
        """Estimate task duration in minutes."""
        base_durations = {
            TaskCategory.RESEARCH: 5.0,
            TaskCategory.CONTENT: 3.0,
            TaskCategory.TECHNICAL: 2.0,
            TaskCategory.ANALYSIS: 3.0,
            TaskCategory.REVIEW: 2.0,
            TaskCategory.STRATEGY: 4.0,
        }
        
        base = base_durations.get(category, 3.0)
        
        # Adjust for description length (complexity proxy)
        complexity_factor = min(len(task.description) / 100, 3.0)
        
        # Adjust for priority (higher priority = more thorough)
        priority_factor = 1 + (task.priority / 10)
        
        return base * complexity_factor * priority_factor
    
    def decompose_complex_task(self, task: Task) -> List[Task]:
        """
        Decompose a complex task into subtasks.
        
        Uses heuristics to break down tasks that are too large
        for a single expert to handle effectively.
        """
        subtasks = []
        description = task.description.lower()
        
        # Check for compound tasks (and, then, also)
        compound_markers = [" and ", " then ", " also ", "; "]
        
        for marker in compound_markers:
            if marker in description:
                parts = task.description.split(marker)
                for i, part in enumerate(parts):
                    if part.strip():
                        subtasks.append(Task(
                            id=f"{task.id}_sub_{i}",
                            description=part.strip(),
                            task_type=task.task_type,
                            context=task.context.copy(),
                            constraints=task.constraints.copy(),
                            priority=task.priority
                        ))
                break
        
        # If no decomposition, return original
        if not subtasks:
            return [task]
        
        return subtasks

    def get_expert(self, expert_name: str) -> BaseExpert:
        """Get or create an expert instance."""
        if expert_name not in self._expert_cache:
            expert_classes = {
                "ResearchExpert": ResearchExpert,
                "AnalystExpert": AnalystExpert,
                "WriterExpert": WriterExpert,
                "EngineerExpert": EngineerExpert,
                "CriticExpert": CriticExpert,
                "StrategistExpert": StrategistExpert,
            }
            
            if expert_name in expert_classes:
                self._expert_cache[expert_name] = expert_classes[expert_name](self.platform)
            else:
                raise ValueError(f"Unknown expert: {expert_name}")
        
        return self._expert_cache[expert_name]
    
    async def consult_experts(
        self,
        task: Task,
        routing: RoutingDecision
    ) -> ConsensusResult:
        """
        Consult assigned experts and reach consensus.
        
        Args:
            task: The task to analyze
            routing: The routing decision with expert assignments
            
        Returns:
            ConsensusResult with aggregated expert opinions
        """
        # Get all assigned experts
        all_expert_names = routing.primary_experts + routing.supporting_experts
        
        # Gather opinions in parallel
        opinions = []
        tasks = []
        
        for expert_name in all_expert_names:
            expert = self.get_expert(expert_name)
            tasks.append(expert.analyze(task))
        
        # Execute all analyses concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ExpertOpinion):
                opinions.append(result)
            elif isinstance(result, Exception):
                print(f"Expert analysis failed: {result}")
        
        # Reach consensus
        strategy = ConsensusStrategy.WEIGHTED_VOTE
        if len(opinions) <= 2:
            strategy = ConsensusStrategy.HIGHEST_CONFIDENCE
        
        return self.consensus_engine.reach_consensus(opinions, task, strategy)
    
    async def execute_with_experts(
        self,
        task: Task,
        routing: Optional[RoutingDecision] = None
    ) -> Dict[str, Any]:
        """
        Full execution flow: route, consult, and potentially execute.
        
        Args:
            task: The task to execute
            routing: Optional pre-computed routing decision
            
        Returns:
            Execution result with consensus and any outputs
        """
        # Get routing if not provided
        if routing is None:
            routing = self.route_task(task)
        
        # Consult experts
        consensus = await self.consult_experts(task, routing)
        
        # Check if we should auto-proceed
        auto_proceed = (
            routing.autonomy_level == AutonomyLevel.AUTONOMOUS or
            (routing.autonomy_level == AutonomyLevel.CONDITIONAL and
             consensus.confidence >= routing.confidence_threshold)
        )
        
        result = {
            "task_id": task.id,
            "routing": {
                "primary_experts": routing.primary_experts,
                "supporting_experts": routing.supporting_experts,
                "autonomy_level": routing.autonomy_level.value,
            },
            "consensus": {
                "decision": consensus.decision,
                "confidence": consensus.confidence,
                "agreement": consensus.agreement_level,
                "participating_experts": consensus.participating_experts,
            },
            "auto_proceed": auto_proceed,
            "requires_approval": not auto_proceed,
        }
        
        # If auto-proceed, execute with primary expert
        if auto_proceed and routing.primary_experts:
            primary_expert = self.get_expert(routing.primary_experts[0])
            try:
                execution = await primary_expert.execute(task)
                result["execution"] = {
                    "success": execution.success,
                    "output": execution.output,
                    "confidence": execution.confidence,
                    "duration": execution.execution_time,
                }
            except Exception as e:
                result["execution"] = {
                    "success": False,
                    "error": str(e),
                }
        
        return result
