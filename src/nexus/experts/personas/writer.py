"""
Writer Expert - Content creation and editing specialist
"""

import time
from typing import Dict, Any
from ..base import BaseExpert, ExpertPersona, ExpertOpinion, ExpertResult, Task, TaskType


WRITER_PERSONA = ExpertPersona(
    name="WriterExpert",
    role="Content Writer",
    description="Content creation, editing, and communication specialist",
    system_prompt="""You are a skilled content writer. Your role is to:
- Create clear, engaging, and well-structured content
- Adapt tone and style to the target audience
- Edit and improve existing content
- Ensure clarity, flow, and impact
- Maintain consistency in voice and messaging

Write with purpose. Every word should earn its place.""",
    strengths=["Clear writing", "Audience adaptation", "Editing", "Storytelling"],
    weaknesses=["May prioritize style over technical accuracy", "Subjective judgments"],
    task_types=[TaskType.WRITING, TaskType.REVIEW, TaskType.GENERAL],
    preferred_models=["claude-3-opus", "claude-3-sonnet", "gpt-4-turbo"],
    temperature=0.8,
    weight=1.0
)


class WriterExpert(BaseExpert):
    """Expert specialized in content creation and editing."""
    
    def __init__(self, platform=None):
        super().__init__(WRITER_PERSONA, platform)
    
    async def analyze(self, task: Task) -> ExpertOpinion:
        """Analyze task from a writing perspective."""
        start = time.time()
        
        prompt = f"""{self.persona.system_prompt}

Task to analyze:
{task.to_prompt()}

Provide your analysis including:
1. Target audience and appropriate tone
2. Key messages to convey
3. Recommended structure/format
4. Potential challenges in communication
5. Success criteria for the content"""

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
                "Content approach defined",
            ),
            confidence=0.85,
            reasoning="Writing expertise applied",
            metadata={
                "duration": time.time() - start,
                **self._response_usage(response),
            }
        )
        
        self._record_opinion(opinion)
        return opinion
    
    async def execute(self, task: Task) -> ExpertResult:
        """Execute writing task."""
        start = time.time()
        
        try:
            prompt = f"""{self.persona.system_prompt}

Create content for this task:
{task.to_prompt()}

Deliver polished, ready-to-use content that:
- Matches the intended audience and purpose
- Is well-structured and flows naturally
- Achieves the communication goals
- Is free of errors and awkward phrasing"""

            response = self._require_successful_payload(await self.platform.query(
                prompt,
                temperature=self.persona.temperature,
                max_tokens=3000
            ), "Expert execution query")
            
            return ExpertResult(
                expert_name=self.name,
                task_id=task.id,
                success=True,
                output=response.get("content", ""),
                confidence=0.85,
                execution_time=time.time() - start,
                tokens_used=response.get("tokens_used", response.get("tokens", 0)),
                cost_usd=response.get("cost_usd", response.get("cost", 0.0)),
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
