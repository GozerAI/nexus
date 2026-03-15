"""
Workflow checkpoint and resume.

Persists workflow execution state at configurable checkpoint boundaries,
enabling recovery from failures without re-executing completed stages.
"""

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class StageCheckpoint:
    """Checkpoint data for a single stage."""
    stage_id: str
    stage_name: str
    status: str  # "completed", "failed", "skipped"
    output: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkflowCheckpoint:
    """Full checkpoint for a workflow execution."""
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    workflow_name: str = ""
    stage_checkpoints: Dict[str, StageCheckpoint] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def completed_stages(self) -> Set[str]:
        return {
            sid for sid, sc in self.stage_checkpoints.items()
            if sc.status == "completed"
        }

    @property
    def failed_stages(self) -> Set[str]:
        return {
            sid for sid, sc in self.stage_checkpoints.items()
            if sc.status == "failed"
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "stages": {
                sid: {
                    "stage_id": sc.stage_id,
                    "stage_name": sc.stage_name,
                    "status": sc.status,
                    "output": sc.output,
                    "error": sc.error,
                    "duration_ms": sc.duration_ms,
                    "timestamp": sc.timestamp,
                }
                for sid, sc in self.stage_checkpoints.items()
            },
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed": self.completed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        stages = {}
        for sid, sd in data.get("stages", {}).items():
            stages[sid] = StageCheckpoint(
                stage_id=sd["stage_id"],
                stage_name=sd["stage_name"],
                status=sd["status"],
                output=sd.get("output"),
                error=sd.get("error"),
                duration_ms=sd.get("duration_ms", 0.0),
                timestamp=sd.get("timestamp", 0.0),
            )
        return cls(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data.get("workflow_id", ""),
            workflow_name=data.get("workflow_name", ""),
            stage_checkpoints=stages,
            context=data.get("context", {}),
            created_at=data.get("created_at", 0.0),
            updated_at=data.get("updated_at", 0.0),
            completed=data.get("completed", False),
            metadata=data.get("metadata", {}),
        )


class CheckpointManager:
    """
    Manages workflow checkpoints for fault-tolerant execution.

    Features:
    - Persist checkpoints to disk (JSON) or memory
    - Automatic checkpoint after each stage completion
    - Resume workflow from last checkpoint
    - Checkpoint cleanup and retention policy
    - Multiple checkpoint backends (memory, file)
    """

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        max_checkpoints_per_workflow: int = 10,
        use_memory_backend: bool = False,
    ):
        """
        Args:
            storage_dir: Directory for file-based checkpoints
            max_checkpoints_per_workflow: Max retained checkpoints per workflow
            use_memory_backend: If True, use in-memory storage only
        """
        self._storage_dir = storage_dir
        self._max_per_workflow = max_checkpoints_per_workflow
        self._use_memory = use_memory_backend or storage_dir is None
        self._memory_store: Dict[str, WorkflowCheckpoint] = {}
        self._workflow_index: Dict[str, List[str]] = {}  # workflow_id -> [checkpoint_ids]
        self._stats = {
            "checkpoints_created": 0,
            "checkpoints_loaded": 0,
            "stages_checkpointed": 0,
            "resumes": 0,
        }

        if self._storage_dir and not self._use_memory:
            os.makedirs(self._storage_dir, exist_ok=True)

    def create_checkpoint(
        self,
        workflow_id: str,
        workflow_name: str = "",
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> WorkflowCheckpoint:
        """Create a new checkpoint for a workflow."""
        checkpoint = WorkflowCheckpoint(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            context=context or {},
            metadata=metadata or {},
        )
        self._save(checkpoint)
        self._stats["checkpoints_created"] += 1

        # Track in index
        if workflow_id not in self._workflow_index:
            self._workflow_index[workflow_id] = []
        self._workflow_index[workflow_id].append(checkpoint.checkpoint_id)

        # Enforce retention
        self._enforce_retention(workflow_id)

        return checkpoint

    def checkpoint_stage(
        self,
        checkpoint_id: str,
        stage_id: str,
        stage_name: str,
        status: str,
        output: Any = None,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> Optional[WorkflowCheckpoint]:
        """
        Record a stage completion in a checkpoint.

        Args:
            checkpoint_id: ID of the checkpoint to update
            stage_id: Stage identifier
            stage_name: Human-readable stage name
            status: "completed", "failed", or "skipped"
            output: Stage output (must be JSON-serializable)
            error: Error message if failed
            duration_ms: Stage execution time

        Returns:
            Updated WorkflowCheckpoint, or None if not found
        """
        checkpoint = self._load(checkpoint_id)
        if not checkpoint:
            logger.warning("Checkpoint %s not found", checkpoint_id)
            return None

        checkpoint.stage_checkpoints[stage_id] = StageCheckpoint(
            stage_id=stage_id,
            stage_name=stage_name,
            status=status,
            output=output,
            error=error,
            duration_ms=duration_ms,
        )
        checkpoint.updated_at = time.time()
        self._save(checkpoint)
        self._stats["stages_checkpointed"] += 1

        return checkpoint

    def mark_completed(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Mark a checkpoint as fully completed."""
        checkpoint = self._load(checkpoint_id)
        if checkpoint:
            checkpoint.completed = True
            checkpoint.updated_at = time.time()
            self._save(checkpoint)
        return checkpoint

    def get_latest_checkpoint(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        """Get the most recent checkpoint for a workflow."""
        cids = self._workflow_index.get(workflow_id, [])
        if not cids:
            return None

        # Return most recent
        latest = None
        for cid in reversed(cids):
            cp = self._load(cid)
            if cp and not cp.completed:
                latest = cp
                break
        return latest or (self._load(cids[-1]) if cids else None)

    def get_resume_state(
        self, workflow_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the state needed to resume a workflow.

        Returns:
            Dict with ``checkpoint``, ``completed_stages``, ``context``,
            ``stage_outputs`` — or None if no checkpoint exists.
        """
        checkpoint = self.get_latest_checkpoint(workflow_id)
        if not checkpoint or checkpoint.completed:
            return None

        self._stats["resumes"] += 1
        stage_outputs = {
            sid: sc.output
            for sid, sc in checkpoint.stage_checkpoints.items()
            if sc.status == "completed"
        }

        return {
            "checkpoint": checkpoint,
            "completed_stages": checkpoint.completed_stages,
            "failed_stages": checkpoint.failed_stages,
            "context": checkpoint.context,
            "stage_outputs": stage_outputs,
        }

    def _save(self, checkpoint: WorkflowCheckpoint) -> None:
        """Save checkpoint to storage backend."""
        if self._use_memory:
            self._memory_store[checkpoint.checkpoint_id] = checkpoint
        else:
            path = self._checkpoint_path(checkpoint.checkpoint_id)
            with open(path, "w") as f:
                json.dump(checkpoint.to_dict(), f, default=str, indent=2)

    def _load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        """Load checkpoint from storage backend."""
        self._stats["checkpoints_loaded"] += 1
        if self._use_memory:
            return self._memory_store.get(checkpoint_id)
        else:
            path = self._checkpoint_path(checkpoint_id)
            if not os.path.exists(path):
                return None
            with open(path) as f:
                data = json.load(f)
            return WorkflowCheckpoint.from_dict(data)

    def _checkpoint_path(self, checkpoint_id: str) -> str:
        return os.path.join(self._storage_dir or ".", f"checkpoint_{checkpoint_id}.json")

    def _enforce_retention(self, workflow_id: str) -> None:
        """Remove old checkpoints exceeding retention limit."""
        cids = self._workflow_index.get(workflow_id, [])
        while len(cids) > self._max_per_workflow:
            old_id = cids.pop(0)
            if self._use_memory:
                self._memory_store.pop(old_id, None)
            else:
                path = self._checkpoint_path(old_id)
                if os.path.exists(path):
                    os.remove(path)

    def cleanup(self, max_age_seconds: float = 86400) -> int:
        """Remove checkpoints older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        removed = 0

        if self._use_memory:
            to_remove = [
                cid for cid, cp in self._memory_store.items()
                if cp.updated_at < cutoff
            ]
            for cid in to_remove:
                del self._memory_store[cid]
                removed += 1
        elif self._storage_dir:
            for fname in os.listdir(self._storage_dir):
                if fname.startswith("checkpoint_"):
                    path = os.path.join(self._storage_dir, fname)
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        removed += 1

        return removed

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "workflows_tracked": len(self._workflow_index),
            "checkpoints_stored": (
                len(self._memory_store) if self._use_memory else "file-based"
            ),
        }
