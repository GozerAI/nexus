"""
Strategist Expert - Planning and recommendation specialist
"""

import time
from typing import Dict, Any
from ..base import BaseExpert, ExpertPersona, ExpertOpinion, ExpertResult, Task, TaskType


STRATEGIST_PERSONA = ExpertPersona(
    name="StrategistExpert",
    role="Strategic Advisor",
    description="High-level planning, decision-making, and strategic recommendations",
    system_prompt="""You are a strategic advisor. Your role is to:
- See the big picture and long-term implications
- Evaluate options and recommend optimal paths
- Consider trade-offs and opportunity costs
- Align tactical decisions with strategic goals
- Anticipate second and third-order effects

Think long-term. Consider what success looks like in 6 months.""",
    strengths=["Big picture thinking", "Decision frameworks", "Trade-off analysis", "Planning"],
    weaknesses=["May overlook implementation details", "Can be too abstract"],
    task_types=[TaskType.STRATEGY, TaskType.ANALYSIS, TaskType.GENERAL],
    preferred_models=["gpt-4-turbo", "claude-3-opus", "gpt-4"],
    temperature=0.6,
    weight=1.1
)


class StrategistExpert(BaseExpert):
    """Expert specialized in strategic planning and recommendations."""
    
    def __init__(self, platform=None):
        super().__init__(STRATEGIST_PERSONA, platform)
    
    async def analyze(self, task: Task) -> ExpertOpinion:
        """Analyze task from a strategic perspective."""
        start = time.time()
        
        prompt = f"""{self.persona.system_prompt}

Task to analyze:
{task.to_prompt()}

Provide your strategic analysis:
1. How does this fit into larger goals?
2. What are the key trade-offs?
3. What's the recommended approach and why?
4. What are the risks of action vs inaction?
5. Success metrics and milestones"""

        response = self._require_successful_payload(await self.platform.query(
            prompt,
            temperature=self.persona.temperature
        ), "Expert analysis query")
        
        opinion = ExpertOpinion(
            expert_name=self.name,
            expert_role=self.role,
            analysis=response.get("content", ""),
            recommendation=self._extract_recommendation(
                response.get("content", ""),
                "Strategic direction provided",
            ),
            confidence=0.8,
            reasoning="Strategic framework applied",
            alternatives=self._extract_alternatives(response.get("content", "")),
            metadata={
                "duration": time.time() - start,
                **self._response_usage(response),
            }
        )
        
        self._record_opinion(opinion)
        return opinion
    
    def _extract_alternatives(self, content: str) -> list:
        """Extract alternative approaches from analysis."""
        alternatives = []
        keywords = ["alternatively", "another option", "could also", "or we could"]
        lines = content.split("\n")
        for line in lines:
            if any(kw in line.lower() for kw in keywords):
                alternatives.append(line.strip())
        return alternatives[:3]
    
    async def execute(self, task: Task) -> ExpertResult:
        """Execute strategic planning task."""
        start = time.time()
        
        try:
            prompt = f"""{self.persona.system_prompt}

Develop a strategic plan for:
{task.to_prompt()}

Provide:
1. Executive summary (2-3 sentences)
2. Recommended strategy
3. Key milestones and timeline
4. Resource requirements
5. Risk mitigation approach
6. Success criteria"""

            response = self._require_successful_payload(await self.platform.query(
                prompt,
                temperature=self.persona.temperature,
                max_tokens=2500
            ), "Expert execution query")
            
            return ExpertResult(
                expert_name=self.name,
                task_id=task.id,
                success=True,
                output=response.get("content", ""),
                confidence=0.8,
                execution_time=time.time() - start,
                tokens_used=response.get("tokens_used", response.get("tokens", 0)),
                metadata=self._response_usage(response),
            )
        except Exception as e:
            return ExpertResult(
                expert_name=self.name,
                task_id=task.id,
                success=False,
                output=None,
                confidence=0.0,
                execution_time=time.time() - start,
                errors=[str(e)]
            )
