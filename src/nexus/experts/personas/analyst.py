"""
Analyst Expert - Data analysis and pattern recognition specialist
"""

import time
from typing import Dict, Any
from ..base import BaseExpert, ExpertPersona, ExpertOpinion, ExpertResult, Task, TaskType


ANALYST_PERSONA = ExpertPersona(
    name="AnalystExpert",
    role="Data Analyst",
    description="Quantitative analysis, pattern recognition, and data-driven insights",
    system_prompt="""You are a sharp analytical expert. Your role is to:
- Analyze data and identify patterns
- Provide quantitative assessments when possible
- Break down complex problems into measurable components
- Identify trends, correlations, and anomalies
- Present findings with supporting evidence

Be precise with numbers. Quantify uncertainty when exact figures aren't available.""",
    strengths=["Quantitative analysis", "Pattern recognition", "Data interpretation", "Metrics"],
    weaknesses=["May miss qualitative nuances", "Can over-rely on available data"],
    task_types=[TaskType.ANALYSIS, TaskType.RESEARCH, TaskType.GENERAL],
    preferred_models=["gpt-4-turbo", "gpt-4", "claude-3-sonnet"],
    temperature=0.4,
    weight=1.1
)


class AnalystExpert(BaseExpert):
    """Expert specialized in data analysis and pattern recognition."""
    
    def __init__(self, platform=None):
        super().__init__(ANALYST_PERSONA, platform)
    
    async def analyze(self, task: Task) -> ExpertOpinion:
        """Analyze task from an analytical perspective."""
        start = time.time()
        
        prompt = f"""{self.persona.system_prompt}

Task to analyze:
{task.to_prompt()}

Provide your analysis including:
1. Key metrics or data points relevant to this task
2. Patterns or trends you can identify
3. Quantitative assessment (with confidence intervals if applicable)
4. Data gaps that would improve analysis
5. Actionable insights based on analysis"""

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
                "Data-driven approach recommended",
            ),
            confidence=0.75,
            reasoning="Analytical framework applied",
            metadata={
                "duration": time.time() - start,
                **self._response_usage(response),
            }
        )
        
        self._record_opinion(opinion)
        return opinion
    
    async def execute(self, task: Task) -> ExpertResult:
        """Execute analysis task."""
        start = time.time()
        
        try:
            prompt = f"""{self.persona.system_prompt}

Execute this analysis task:
{task.to_prompt()}

Provide a comprehensive analysis with:
- Executive summary
- Key findings (numbered)
- Supporting data/evidence
- Recommendations
- Confidence level and caveats"""

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
                confidence=0.8,
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
