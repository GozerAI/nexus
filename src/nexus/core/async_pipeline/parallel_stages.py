"""
Pipeline stage parallelization.

Executes independent pipeline stages concurrently while respecting
dependency ordering. Stages that share no data dependencies run in
parallel; dependent stages wait for their inputs.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """A stage in a pipeline."""
    stage_id: str
    name: str
    fn: Callable[..., Coroutine]
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 300.0
    retry_count: int = 0
    max_retries: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageResult:
    """Result from executing a pipeline stage."""
    stage_id: str
    name: str
    status: StageStatus
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    retries: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0


class ParallelPipelineExecutor:
    """
    Executes pipeline stages with maximum parallelism.

    Analyzes stage dependency graphs and schedules independent stages
    to run concurrently. Passes outputs from completed stages as inputs
    to downstream stages.

    Features:
    - Automatic parallelism based on dependency graph
    - Per-stage timeout and retry
    - Configurable concurrency limit
    - Stage output forwarding
    - Execution statistics
    """

    def __init__(self, max_concurrency: int = 10):
        self._max_concurrency = max_concurrency
        self._stages: Dict[str, PipelineStage] = {}
        self._results: Dict[str, StageResult] = {}
        self._stats = {
            "pipelines_executed": 0,
            "stages_executed": 0,
            "stages_failed": 0,
            "total_latency_ms": 0.0,
        }

    def add_stage(
        self,
        stage_id: str,
        name: str,
        fn: Callable[..., Coroutine],
        dependencies: Optional[List[str]] = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 1,
    ) -> None:
        """Add a stage to the pipeline."""
        self._stages[stage_id] = PipelineStage(
            stage_id=stage_id,
            name=name,
            fn=fn,
            dependencies=dependencies or [],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def _get_ready_stages(
        self, completed: Set[str], running: Set[str], failed: Set[str]
    ) -> List[PipelineStage]:
        """Find stages whose dependencies are all satisfied."""
        ready = []
        for sid, stage in self._stages.items():
            if sid in completed or sid in running or sid in failed:
                continue
            if all(d in completed for d in stage.dependencies):
                ready.append(stage)
        return ready

    async def execute(
        self,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, StageResult]:
        """
        Execute the pipeline.

        Args:
            initial_context: Context dict available to all stages

        Returns:
            Dict mapping stage_id to StageResult
        """
        self._stats["pipelines_executed"] += 1
        self._results.clear()
        context = dict(initial_context or {})

        completed: Set[str] = set()
        running: Set[str] = set()
        failed: Set[str] = set()
        semaphore = asyncio.Semaphore(self._max_concurrency)
        tasks: Dict[str, asyncio.Task] = {}

        start = time.time()

        while len(completed) + len(failed) < len(self._stages):
            # Find stages ready to run
            ready = self._get_ready_stages(completed, running, failed)

            if not ready and not running:
                # Deadlock or all dependencies failed
                remaining = set(self._stages.keys()) - completed - failed
                for sid in remaining:
                    self._results[sid] = StageResult(
                        stage_id=sid,
                        name=self._stages[sid].name,
                        status=StageStatus.SKIPPED,
                        error="Dependencies failed",
                    )
                    failed.add(sid)
                break

            # Launch ready stages
            for stage in ready:
                running.add(stage.stage_id)
                dep_outputs = {
                    d: self._results[d].output
                    for d in stage.dependencies
                    if d in self._results
                }
                task = asyncio.create_task(
                    self._run_stage(stage, context, dep_outputs, semaphore)
                )
                tasks[stage.stage_id] = task

            # Wait for at least one task to complete
            if tasks:
                pending_tasks = [
                    t for sid, t in tasks.items() if sid in running
                ]
                if pending_tasks:
                    done, _ = await asyncio.wait(
                        pending_tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        for sid, t in list(tasks.items()):
                            if t is task:
                                running.discard(sid)
                                try:
                                    result = task.result()
                                    self._results[sid] = result
                                    if result.status == StageStatus.COMPLETED:
                                        completed.add(sid)
                                        context[sid] = result.output
                                    else:
                                        failed.add(sid)
                                except Exception as e:
                                    self._results[sid] = StageResult(
                                        stage_id=sid,
                                        name=self._stages[sid].name,
                                        status=StageStatus.FAILED,
                                        error=str(e),
                                    )
                                    failed.add(sid)
                                del tasks[sid]
                                break

        total_ms = (time.time() - start) * 1000
        self._stats["total_latency_ms"] += total_ms
        logger.info(
            "Pipeline complete: %d/%d stages succeeded in %.0fms",
            len(completed), len(self._stages), total_ms,
        )
        return dict(self._results)

    async def _run_stage(
        self,
        stage: PipelineStage,
        context: Dict[str, Any],
        dep_outputs: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> StageResult:
        """Execute a single stage with retry."""
        async with semaphore:
            for attempt in range(stage.max_retries + 1):
                start = time.time()
                try:
                    output = await asyncio.wait_for(
                        stage.fn(context=context, inputs=dep_outputs),
                        timeout=stage.timeout_seconds,
                    )
                    latency = (time.time() - start) * 1000
                    self._stats["stages_executed"] += 1
                    return StageResult(
                        stage_id=stage.stage_id,
                        name=stage.name,
                        status=StageStatus.COMPLETED,
                        output=output,
                        latency_ms=latency,
                        retries=attempt,
                        started_at=start,
                        completed_at=time.time(),
                    )
                except Exception as e:
                    latency = (time.time() - start) * 1000
                    if attempt < stage.max_retries:
                        logger.warning(
                            "Stage %s failed (attempt %d/%d): %s",
                            stage.name, attempt + 1, stage.max_retries + 1, e,
                        )
                        await asyncio.sleep(min(2 ** attempt, 10))
                        continue

                    self._stats["stages_failed"] += 1
                    return StageResult(
                        stage_id=stage.stage_id,
                        name=stage.name,
                        status=StageStatus.FAILED,
                        error=str(e),
                        latency_ms=latency,
                        retries=attempt,
                        started_at=start,
                        completed_at=time.time(),
                    )

        # Should not reach here
        return StageResult(
            stage_id=stage.stage_id,
            name=stage.name,
            status=StageStatus.FAILED,
            error="Unexpected state",
        )

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def clear(self) -> None:
        """Clear all stages."""
        self._stages.clear()
        self._results.clear()
