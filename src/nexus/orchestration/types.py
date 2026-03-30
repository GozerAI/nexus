"""
Orchestration Types - Core types for pipeline execution
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


class AutonomyLevel(Enum):
    """Levels of autonomous operation."""
    FULL_APPROVAL = "full_approval"      # Human approves every step
    SUPERVISED = "supervised"             # Human approves key decisions
    CONDITIONAL = "conditional"           # Auto-proceed if confidence > threshold
    AUTONOMOUS = "autonomous"             # Fully autonomous execution


class StepStatus(Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskCategory(Enum):
    """Categories of tasks for routing."""
    RESEARCH = "research"
    CONTENT = "content"
    TECHNICAL = "technical"
    ANALYSIS = "analysis"
    REVIEW = "review"
    STRATEGY = "strategy"


@dataclass
class ApprovalRequest:
    """Request for human approval."""
    id: str
    step_id: str
    pipeline_id: str
    summary: str
    details: Dict[str, Any]
    expert_opinions: List[Dict[str, Any]]
    consensus_confidence: float
    recommended_action: str
    alternatives: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


@dataclass
class ApprovalResponse:
    """Human response to approval request."""
    request_id: str
    approved: bool
    feedback: str = ""
    modifications: Dict[str, Any] = field(default_factory=dict)
    responded_at: datetime = field(default_factory=datetime.now)
    respondent: str = "human"
