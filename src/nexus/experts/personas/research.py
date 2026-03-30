"""
Research Expert - Deep research and fact-finding specialist
"""

import time
from typing import Dict, Any
from ..base import BaseExpert, ExpertPersona, ExpertOpinion, ExpertResult, Task, TaskType


RESEARCH_PERSONA = ExpertPersona(
    name="ResearchExpert",
    role="Research Specialist",
    description="Deep research, fact-finding, and knowledge synthesis",
    system_prompt="""You are a meticulous research specialist. Your role is to:
- Conduct thorough research on any topic
- Find and verify facts from multiple angles
- Identify knowledge gaps and uncertainties
- Synthesize information into clear insights
- Always cite your reasoning and confidence levels

Be thorough but concise. Flag any uncertainties clearly.""",
    strengths=["Deep research", "Fact verification", "Source synthesis", "Knowledge gaps"],
    weaknesses=["May over-research simple questions", "Can be slow on time-sensitive tasks"],
    task_types=[TaskType.RESEARCH, TaskType.ANALYSIS, TaskType.GENERAL],
    preferred_models=["claude-3-opus", "claude-3-sonnet", "gpt-4-turbo"],
    temperature=0.5,
    weight=1.2
)


class ResearchExpert(BaseExpert):
    """Expert specialized in deep research and fact-finding."""
    
    def __init__(self, platform=None):
        super().__init__(RESEARCH_PERSONA, platform)
    
    async def analyze(self, task: Task) -> ExpertOpinion:
        """Analyze task from a research perspective."""
        start = time.time()
        
        prompt = f"""{self.persona.system_prompt}

Task to analyze:
{task.to_prompt()}

Provide your analysis including:
1. Key research questions this raises
2. What information would be needed
3. Potential sources or approaches
4. Confidence in being able to answer this
5. Any concerns or caveats"""

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
                "Proceed with research" if task.task_type == TaskType.RESEARCH else "Research may help",
            ),
            confidence=0.8,
            reasoning="Research-focused analysis",
            metadata={
                "duration": time.time() - start,
                **self._response_usage(response),
            }
        )
        
        self._record_opinion(opinion)
        return opinion
    
    async def execute(self, task: Task) -> ExpertResult:
        """Execute research task using cog-eng research agent."""
        start = time.time()
        
        try:
            # Use cog-eng's autonomous research agent
            result = self._require_successful_payload(await self.platform.research(
                topic=task.description,
                depth="moderate",
                max_iterations=5
            ), "Research execution")
            
            return ExpertResult(
                expert_name=self.name,
                task_id=task.id,
                success=True,
                output=result,
                confidence=result.get("confidence", 0.7),
                execution_time=time.time() - start,
                tokens_used=result.get("tokens_used", 0),
                metadata={"findings_count": len(result.get("findings", []))}
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
