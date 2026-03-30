"""
Engineer Expert - Code and architecture specialist
"""

import time
from typing import Dict, Any
from ..base import BaseExpert, ExpertPersona, ExpertOpinion, ExpertResult, Task, TaskType


ENGINEER_PERSONA = ExpertPersona(
    name="EngineerExpert",
    role="Software Engineer",
    description="Code generation, architecture design, and technical implementation",
    system_prompt="""You are an expert software engineer. Your role is to:
- Write clean, efficient, well-documented code
- Design robust architectures and systems
- Identify technical risks and edge cases
- Apply best practices and design patterns
- Consider performance, security, and maintainability

Code should be production-ready. Think about the next developer.""",
    strengths=["Clean code", "Architecture", "Problem solving", "Best practices"],
    weaknesses=["May over-engineer simple solutions", "Can miss business context"],
    task_types=[TaskType.CODING, TaskType.REVIEW, TaskType.ANALYSIS],
    preferred_models=["gpt-4-turbo", "claude-3-opus", "gpt-4"],
    temperature=0.3,
    weight=1.2
)


class EngineerExpert(BaseExpert):
    """Expert specialized in code and architecture."""
    
    def __init__(self, platform=None):
        super().__init__(ENGINEER_PERSONA, platform)
    
    async def analyze(self, task: Task) -> ExpertOpinion:
        """Analyze task from an engineering perspective."""
        start = time.time()
        
        prompt = f"""{self.persona.system_prompt}

Task to analyze:
{task.to_prompt()}

Provide your technical analysis:
1. Technical approach and architecture
2. Key components needed
3. Potential technical risks
4. Dependencies and requirements
5. Estimated complexity (low/medium/high)"""

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
                "Technical implementation path identified",
            ),
            confidence=0.85,
            reasoning="Engineering analysis applied",
            metadata={
                "duration": time.time() - start,
                **self._response_usage(response),
            }
        )
        
        self._record_opinion(opinion)
        return opinion
    
    async def execute(self, task: Task) -> ExpertResult:
        """Execute coding task using cog-eng code generator."""
        start = time.time()
        
        try:
            requirements = task.context.get("requirements", [])
            if isinstance(requirements, str):
                requirements = [requirements]
            elif not isinstance(requirements, list):
                requirements = [str(requirements)]

            # Use cog-eng's self-improving code generator
            result = await self.platform.generate_code(
                description=task.description,
                language=task.context.get("language", "python"),
                requirements=requirements,
                constraints=task.constraints,
            )
            
            return ExpertResult(
                expert_name=self.name,
                task_id=task.id,
                success=True,
                output=result.get("code", ""),
                confidence=result.get("quality_score", 0.8),
                execution_time=time.time() - start,
                metadata={
                    "quality_score": result.get("quality_score"),
                    "iterations": result.get("iteration", 1)
                }
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
