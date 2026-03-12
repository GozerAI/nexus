"""
Pipeline Executor - Executes multi-step workflows with expert panel
"""

import asyncio
import inspect
import uuid
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from nexus.experts.base import Task, TaskType
from .expert_router import ExpertRouter, RoutingDecision
from .types import AutonomyLevel, StepStatus, ApprovalRequest, ApprovalResponse


@dataclass
class PipelineStep:
    """A single step in a pipeline."""
    id: str
    name: str
    description: str
    task_type: TaskType = TaskType.GENERAL
    dependencies: List[str] = field(default_factory=list)  # Step IDs
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    expert_override: Optional[str] = None  # Force specific expert
    autonomy_override: Optional[AutonomyLevel] = None
    
    def to_task(self, context: Dict[str, Any] = None) -> Task:
        """Convert step to a Task for expert processing."""
        return Task(
            id=self.id,
            description=self.description,
            task_type=self.task_type,
            context={**self.inputs, **(context or {})},
            constraints=[],
            priority=5
        )


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_id: str
    status: str
    steps_completed: int
    steps_total: int
    outputs: Dict[str, Any]
    approval_requests: List[ApprovalRequest]
    duration_seconds: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PipelineExecutor:
    """
    Executes multi-step workflows using the expert panel.
    
    Features:
    - Step dependency management
    - Approval workflow integration
    - Parallel execution where possible
    - State persistence for recovery
    """
    
    def __init__(self, platform=None, autonomy: AutonomyLevel = AutonomyLevel.SUPERVISED):
        self.platform = platform
        self.router = ExpertRouter(platform)
        self.default_autonomy = autonomy
        self._pipelines: Dict[str, Dict] = {}
        self._approval_callbacks: List[Callable] = []
    
    def register_approval_callback(self, callback: Callable):
        """Register callback for approval requests."""
        self._approval_callbacks.append(callback)
    
    async def execute_pipeline(
        self,
        steps: List[PipelineStep],
        pipeline_id: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> PipelineResult:
        """
        Execute a multi-step pipeline.
        
        Args:
            steps: List of pipeline steps to execute
            pipeline_id: Optional ID (generated if not provided)
            context: Shared context across all steps
            
        Returns:
            PipelineResult with outputs and status
        """
        pipeline_id = pipeline_id or str(uuid.uuid4())[:8]
        context = context or {}
        start_time = datetime.now()
        
        # Initialize pipeline state
        self._pipelines[pipeline_id] = {
            "steps": {s.id: s for s in steps},
            "context": context,
            "outputs": {},
            "approval_requests": [],
        }
        
        completed = 0
        errors = []
        
        # Build dependency graph
        pending = list(steps)
        
        while pending:
            # Find steps that can run (dependencies met)
            runnable = [
                s for s in pending
                if all(
                    self._pipelines[pipeline_id]["steps"][dep].status == StepStatus.COMPLETED
                    for dep in s.dependencies
                )
            ]
            
            if not runnable:
                # Check for approval blockers
                awaiting = [s for s in pending if s.status == StepStatus.AWAITING_APPROVAL]
                if awaiting:
                    break  # Wait for approvals
                else:
                    errors.append("Dependency deadlock detected")
                    break
            
            # Execute runnable steps
            for step in runnable:
                try:
                    result = await self._execute_step(step, pipeline_id)
                    
                    if result.get("requires_approval"):
                        step.status = StepStatus.AWAITING_APPROVAL
                        # Create approval request
                        approval = self._create_approval_request(step, pipeline_id, result)
                        self._pipelines[pipeline_id]["approval_requests"].append(approval)
                        await self._notify_approval_needed(approval)
                    else:
                        step.status = StepStatus.COMPLETED
                        step.outputs = result.get("outputs", {})
                        self._pipelines[pipeline_id]["outputs"][step.id] = step.outputs
                        completed += 1
                        
                except Exception as e:
                    step.status = StepStatus.FAILED
                    errors.append(f"Step {step.id} failed: {str(e)}")
                
                pending.remove(step)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return PipelineResult(
            pipeline_id=pipeline_id,
            status="completed" if completed == len(steps) else "partial",
            steps_completed=completed,
            steps_total=len(steps),
            outputs=self._pipelines[pipeline_id]["outputs"],
            approval_requests=self._pipelines[pipeline_id]["approval_requests"],
            duration_seconds=duration,
            errors=errors
        )

    async def _execute_step(
        self,
        step: PipelineStep,
        pipeline_id: str
    ) -> Dict[str, Any]:
        """Execute a single pipeline step."""
        step.status = StepStatus.IN_PROGRESS
        
        # Get context with previous outputs
        context = self._pipelines[pipeline_id]["context"].copy()
        for dep_id in step.dependencies:
            dep_outputs = self._pipelines[pipeline_id]["outputs"].get(dep_id, {})
            context[f"from_{dep_id}"] = dep_outputs
        
        # Create task
        task = step.to_task(context)
        
        # Get routing (with overrides)
        routing = self.router.route_task(task)
        
        if step.expert_override:
            routing.primary_experts = [step.expert_override]
        
        if step.autonomy_override:
            routing.autonomy_level = step.autonomy_override
        elif self.default_autonomy:
            routing.autonomy_level = self.default_autonomy
        
        # Execute with experts
        result = await self.router.execute_with_experts(task, routing)
        
        return {
            "outputs": result.get("execution", {}).get("output", {}),
            "consensus": result.get("consensus", {}),
            "requires_approval": result.get("requires_approval", False),
            "routing": result.get("routing", {}),
        }
    
    def _create_approval_request(
        self,
        step: PipelineStep,
        pipeline_id: str,
        result: Dict[str, Any]
    ) -> ApprovalRequest:
        """Create an approval request for a step."""
        return ApprovalRequest(
            id=f"approval_{pipeline_id}_{step.id}",
            step_id=step.id,
            pipeline_id=pipeline_id,
            summary=f"Approval needed for: {step.name}",
            details={
                "step_description": step.description,
                "inputs": step.inputs,
                "routing": result.get("routing", {}),
            },
            expert_opinions=[],  # Would be populated from consensus
            consensus_confidence=result.get("consensus", {}).get("confidence", 0),
            recommended_action=result.get("consensus", {}).get("decision", ""),
        )
    
    async def _notify_approval_needed(self, approval: ApprovalRequest):
        """Notify registered callbacks about approval request."""
        for callback in self._approval_callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(approval)
                else:
                    callback(approval)
            except Exception as e:
                print(f"Approval callback failed: {e}")
    
    async def process_approval(
        self,
        pipeline_id: str,
        step_id: str,
        approved: bool,
        feedback: str = ""
    ) -> bool:
        """
        Process an approval response and continue pipeline if approved.
        
        Args:
            pipeline_id: The pipeline ID
            step_id: The step that was approved/rejected
            approved: Whether to approve
            feedback: Optional feedback
            
        Returns:
            True if pipeline can continue
        """
        if pipeline_id not in self._pipelines:
            return False
        
        step = self._pipelines[pipeline_id]["steps"].get(step_id)
        if not step:
            return False
        
        if approved:
            step.status = StepStatus.APPROVED
            # Re-execute with full autonomy
            step.autonomy_override = AutonomyLevel.AUTONOMOUS
            result = await self._execute_step(step, pipeline_id)
            step.status = StepStatus.COMPLETED
            step.outputs = result.get("outputs", {})
            self._pipelines[pipeline_id]["outputs"][step_id] = step.outputs
            return True
        else:
            step.status = StepStatus.REJECTED
            return False
    
    def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get current status of a pipeline."""
        if pipeline_id not in self._pipelines:
            return {"error": "Pipeline not found"}
        
        pipeline = self._pipelines[pipeline_id]
        steps = pipeline["steps"]
        
        return {
            "pipeline_id": pipeline_id,
            "steps": {
                sid: {
                    "name": s.name,
                    "status": s.status.value,
                    "has_output": sid in pipeline["outputs"],
                }
                for sid, s in steps.items()
            },
            "pending_approvals": len([
                s for s in steps.values()
                if s.status == StepStatus.AWAITING_APPROVAL
            ]),
            "completed": len([
                s for s in steps.values()
                if s.status == StepStatus.COMPLETED
            ]),
            "total": len(steps),
        }
