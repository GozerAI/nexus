"""
Critic Expert - Quality assurance and review specialist
"""

import time
from typing import Dict, Any
from ..base import BaseExpert, ExpertPersona, ExpertOpinion, ExpertResult, Task, TaskType


CRITIC_PERSONA = ExpertPersona(
    name="CriticExpert",
    role="Quality Reviewer",
    description="Critical review, quality assurance, and improvement identification",
    system_prompt="""You are a thorough quality reviewer. Your role is to:
- Critically evaluate work for quality and correctness
- Identify errors, gaps, and areas for improvement
- Provide constructive feedback with specific suggestions
- Ensure work meets stated requirements
- Catch issues others might miss

Be rigorous but fair. Criticism should be actionable.""",
    strengths=["Error detection", "Quality standards", "Constructive feedback", "Thoroughness"],
    weaknesses=["May be overly critical", "Can slow down iteration"],
    task_types=[TaskType.REVIEW, TaskType.ANALYSIS, TaskType.GENERAL],
    preferred_models=["claude-3-opus", "gpt-4-turbo", "claude-3-sonnet"],
    temperature=0.4,
    weight=0.9
)


class CriticExpert(BaseExpert):
    """Expert specialized in critical review and quality assurance."""
    
    def __init__(self, platform=None):
        super().__init__(CRITIC_PERSONA, platform)
    
    async def analyze(self, task: Task) -> ExpertOpinion:
        """Analyze task from a critical perspective."""
        start = time.time()
        
        prompt = f"""{self.persona.system_prompt}

Task to analyze:
{task.to_prompt()}

Provide your critical analysis:
1. Potential issues or risks with this approach
2. Quality criteria that should be met
3. Common pitfalls to avoid
4. Verification steps needed
5. Overall assessment of feasibility"""

        response = self._require_successful_payload(await self.platform.query(
            prompt,
            temperature=self.persona.temperature
        ), "Expert analysis query")
        
        concerns = self._extract_concerns(response.get("content", ""))
        
        opinion = ExpertOpinion(
            expert_name=self.name,
            expert_role=self.role,
            analysis=response.get("content", ""),
            recommendation=self._extract_recommendation(
                response.get("content", ""),
                "Review checkpoints identified",
            ),
            confidence=0.8,
            reasoning="Critical review perspective",
            concerns=concerns,
            metadata={
                "duration": time.time() - start,
                **self._response_usage(response),
            }
        )
        
        self._record_opinion(opinion)
        return opinion
    
    def _extract_concerns(self, content: str) -> list:
        """Extract concerns from analysis content."""
        concerns = []
        keywords = ["issue", "risk", "problem", "concern", "warning", "careful", "missing", "insufficient"]
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            normalized = stripped.lower()
            if not stripped:
                continue
            if stripped.endswith(":") and len(stripped.split()) <= 8:
                continue
            if normalized.startswith(("##", "###")):
                continue
            if any(kw in normalized for kw in keywords):
                if stripped.startswith("-") or stripped.startswith("*") or "**" in stripped or len(stripped.split()) > 6:
                    concerns.append(stripped.strip("-* ").strip())
        return concerns[:5]  # Top 5 concerns
    
    async def execute(self, task: Task) -> ExpertResult:
        """Execute review task."""
        start = time.time()
        
        try:
            prompt = f"""{self.persona.system_prompt}

Review the following:
{task.to_prompt()}

Provide a comprehensive review:
1. PASS/FAIL assessment with reasoning
2. Critical issues (must fix)
3. Warnings (should fix)
4. Suggestions (nice to have)
5. Specific improvement recommendations"""

            response = self._require_successful_payload(await self.platform.query(
                prompt,
                temperature=self.persona.temperature,
                max_tokens=2000
            ), "Expert execution query")
            
            return ExpertResult(
                expert_name=self.name,
                task_id=task.id,
                success=True,
                output=response.get("content", ""),
                confidence=0.85,
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
