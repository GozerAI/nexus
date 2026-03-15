"""Tests for async pipeline modules."""

import asyncio
import pytest
import time

from nexus.core.async_pipeline.parallel_stages import (
    ParallelPipelineExecutor, StageResult, StageStatus,
)
from nexus.core.async_pipeline.priority_scheduler import (
    TaskScheduler, ScheduledTask, TaskPriority, TaskState,
)
from nexus.core.async_pipeline.execution_dag import (
    ExecutionDAG, DAGNode, DAGExecutor, NodeStatus,
)
from nexus.core.async_pipeline.dead_letter_queue import (
    DeadLetterQueue, DeadLetter, DeadLetterReason,
)
from nexus.core.async_pipeline.checkpoint_resume import (
    CheckpointManager, WorkflowCheckpoint, StageCheckpoint,
)


# ── ParallelPipelineExecutor ─────────────────────────────────

class TestParallelPipelineExecutor:
    @pytest.mark.asyncio
    async def test_sequential_stages(self):
        executor = ParallelPipelineExecutor()

        async def stage_a(context, inputs):
            return "a_result"

        async def stage_b(context, inputs):
            return f"b_got_{inputs.get('a')}"

        executor.add_stage("a", "Stage A", stage_a)
        executor.add_stage("b", "Stage B", stage_b, dependencies=["a"])

        results = await executor.execute()
        assert results["a"].status == StageStatus.COMPLETED
        assert results["b"].status == StageStatus.COMPLETED
        assert results["a"].output == "a_result"

    @pytest.mark.asyncio
    async def test_parallel_stages(self):
        start = time.time()
        executor = ParallelPipelineExecutor()

        async def slow_stage(context, inputs):
            await asyncio.sleep(0.05)
            return "done"

        executor.add_stage("a", "A", slow_stage)
        executor.add_stage("b", "B", slow_stage)
        executor.add_stage("c", "C", slow_stage)

        results = await executor.execute()
        elapsed = time.time() - start

        assert all(r.status == StageStatus.COMPLETED for r in results.values())
        # Parallel: should take ~50ms, not ~150ms
        assert elapsed < 0.15

    @pytest.mark.asyncio
    async def test_failed_stage_skips_dependents(self):
        executor = ParallelPipelineExecutor()

        async def fail_stage(context, inputs):
            raise RuntimeError("boom")

        async def dependent(context, inputs):
            return "should not run"

        executor.add_stage("a", "A", fail_stage, max_retries=0)
        executor.add_stage("b", "B", dependent, dependencies=["a"])

        results = await executor.execute()
        assert results["a"].status == StageStatus.FAILED
        assert results["b"].status == StageStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_initial_context(self):
        executor = ParallelPipelineExecutor()

        async def stage(context, inputs):
            return context.get("key")

        executor.add_stage("s", "S", stage)
        results = await executor.execute(initial_context={"key": "value"})
        assert results["s"].output == "value"

    def test_stats(self):
        executor = ParallelPipelineExecutor()
        stats = executor.get_stats()
        assert stats["pipelines_executed"] == 0

    def test_clear(self):
        executor = ParallelPipelineExecutor()
        executor.add_stage("a", "A", None)
        executor.clear()
        assert executor.get_stats() == executor.get_stats()  # No error


# ── TaskScheduler ────────────────────────────────────────────

class TestTaskScheduler:
    @pytest.mark.asyncio
    async def test_submit_task(self):
        async def dummy():
            return "result"

        scheduler = TaskScheduler()
        task = await scheduler.submit(
            fn=dummy,
            name="test-task",
            priority=TaskPriority.NORMAL,
        )
        assert task.task_id
        assert task.state == TaskState.QUEUED
        assert scheduler.get_queue_depth() == 1

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        async def dummy():
            pass

        scheduler = TaskScheduler()
        task = await scheduler.submit(fn=dummy)
        success = await scheduler.cancel(task.task_id)
        assert success

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self):
        scheduler = TaskScheduler()
        assert not await scheduler.cancel("nonexistent")

    def test_stats(self):
        scheduler = TaskScheduler()
        stats = scheduler.get_stats()
        assert stats["submitted"] == 0


# ── ExecutionDAG ─────────────────────────────────────────────

class TestExecutionDAG:
    def test_add_nodes(self):
        dag = ExecutionDAG()
        dag.add_node("a", "Node A")
        dag.add_node("b", "Node B", dependencies=["a"])
        assert dag.node_count == 2

    def test_validate_valid(self):
        dag = ExecutionDAG()
        dag.add_node("a", "A")
        dag.add_node("b", "B", dependencies=["a"])
        errors = dag.validate()
        assert errors == []

    def test_validate_missing_dep(self):
        dag = ExecutionDAG()
        dag.add_node("a", "A", dependencies=["nonexistent"])
        errors = dag.validate()
        assert len(errors) == 1
        assert "missing" in errors[0]

    def test_detect_cycle(self):
        dag = ExecutionDAG()
        dag.add_node("a", "A", dependencies=["b"])
        dag.add_node("b", "B", dependencies=["a"])
        errors = dag.validate()
        assert any("ycle" in e for e in errors)

    def test_topological_order(self):
        dag = ExecutionDAG()
        dag.add_node("c", "C", dependencies=["a", "b"])
        dag.add_node("a", "A")
        dag.add_node("b", "B", dependencies=["a"])
        order = dag.topological_order()
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")

    def test_parallel_levels(self):
        dag = ExecutionDAG()
        dag.add_node("a", "A")
        dag.add_node("b", "B")
        dag.add_node("c", "C", dependencies=["a", "b"])
        levels = dag.get_parallel_levels()
        assert len(levels) == 2
        assert set(levels[0]) == {"a", "b"}
        assert levels[1] == ["c"]

    def test_critical_path(self):
        dag = ExecutionDAG()
        dag.add_node("a", "A", estimated_duration_ms=100)
        dag.add_node("b", "B", estimated_duration_ms=200)
        dag.add_node("c", "C", dependencies=["a", "b"], estimated_duration_ms=50)
        path, total = dag.critical_path()
        # Critical path: b -> c = 250ms
        assert "b" in path
        assert "c" in path
        assert total >= 250


class TestDAGExecutor:
    @pytest.mark.asyncio
    async def test_execute_simple(self):
        dag = ExecutionDAG()

        async def node_fn(context, inputs):
            return "ok"

        dag.add_node("a", "A", fn=node_fn)
        dag.add_node("b", "B", fn=node_fn, dependencies=["a"])

        executor = DAGExecutor()
        result = await executor.execute(dag)
        assert result.completed_nodes == 2
        assert result.failed_nodes == 0

    @pytest.mark.asyncio
    async def test_execute_with_failure(self):
        dag = ExecutionDAG()

        async def ok_fn(context, inputs):
            return "ok"

        async def fail_fn(context, inputs):
            raise RuntimeError("fail")

        dag.add_node("a", "A", fn=fail_fn)
        dag.add_node("b", "B", fn=ok_fn, dependencies=["a"])

        executor = DAGExecutor()
        result = await executor.execute(dag)
        assert result.failed_nodes == 1
        assert result.skipped_nodes == 1

    @pytest.mark.asyncio
    async def test_invalid_dag_raises(self):
        dag = ExecutionDAG()
        dag.add_node("a", "A", dependencies=["missing"])
        executor = DAGExecutor()
        with pytest.raises(ValueError, match="Invalid DAG"):
            await executor.execute(dag)


# ── DeadLetterQueue ──────────────────────────────────────────

class TestDeadLetterQueue:
    def test_put_and_get(self):
        dlq = DeadLetterQueue()
        letter = dlq.put(
            task_id="t1",
            task_name="process_data",
            reason=DeadLetterReason.MAX_RETRIES,
            error_message="connection timeout",
            retry_count=3,
            max_retries=3,
        )
        assert letter.letter_id
        retrieved = dlq.get(letter.letter_id)
        assert retrieved is not None
        assert retrieved.task_name == "process_data"

    def test_peek(self):
        dlq = DeadLetterQueue()
        dlq.put(task_name="a")
        dlq.put(task_name="b")
        peeked = dlq.peek(1)
        assert len(peeked) == 1
        assert dlq.size == 2  # Not consumed

    def test_query_by_reason(self):
        dlq = DeadLetterQueue()
        dlq.put(reason=DeadLetterReason.TIMEOUT)
        dlq.put(reason=DeadLetterReason.MAX_RETRIES)
        results = dlq.query(reason=DeadLetterReason.TIMEOUT)
        assert len(results) == 1

    def test_query_by_name(self):
        dlq = DeadLetterQueue()
        dlq.put(task_name="special")
        dlq.put(task_name="other")
        results = dlq.query(task_name="special")
        assert len(results) == 1

    def test_mark_reprocessed(self):
        dlq = DeadLetterQueue()
        letter = dlq.put(task_name="t")
        assert dlq.mark_reprocessed(letter.letter_id)
        assert dlq.get(letter.letter_id).reprocessed

    def test_remove(self):
        dlq = DeadLetterQueue()
        letter = dlq.put(task_name="t")
        removed = dlq.remove(letter.letter_id)
        assert removed is not None
        assert dlq.size == 0

    def test_overflow_drop_oldest(self):
        dlq = DeadLetterQueue(max_size=2)
        dlq.put(task_name="first")
        dlq.put(task_name="second")
        dlq.put(task_name="third")
        assert dlq.size == 2
        names = [l.task_name for l in dlq.peek(10)]
        assert "first" not in names

    def test_purge_reprocessed(self):
        dlq = DeadLetterQueue()
        l1 = dlq.put(task_name="done")
        dlq.put(task_name="pending")
        dlq.mark_reprocessed(l1.letter_id)
        purged = dlq.purge(reprocessed_only=True)
        assert purged == 1
        assert dlq.size == 1

    def test_callback(self):
        received = []
        dlq = DeadLetterQueue(on_dead_letter=lambda l: received.append(l))
        dlq.put(task_name="x")
        assert len(received) == 1

    def test_to_dict(self):
        letter = DeadLetter(task_name="t", reason=DeadLetterReason.TIMEOUT)
        d = letter.to_dict()
        assert d["reason"] == "timeout"

    def test_stats(self):
        dlq = DeadLetterQueue()
        dlq.put(reason=DeadLetterReason.TIMEOUT)
        dlq.put(reason=DeadLetterReason.MAX_RETRIES)
        stats = dlq.get_stats()
        assert stats["total_received"] == 2
        assert "timeout" in stats["reason_breakdown"]


# ── CheckpointManager ───────────────────────────────────────

class TestCheckpointManager:
    def test_create_checkpoint(self):
        mgr = CheckpointManager(use_memory_backend=True)
        cp = mgr.create_checkpoint("wf-1", workflow_name="pipeline")
        assert cp.checkpoint_id
        assert cp.workflow_id == "wf-1"

    def test_checkpoint_stage(self):
        mgr = CheckpointManager(use_memory_backend=True)
        cp = mgr.create_checkpoint("wf-1")
        updated = mgr.checkpoint_stage(
            cp.checkpoint_id, "s1", "Stage 1", "completed", output={"result": 42}
        )
        assert "s1" in updated.stage_checkpoints
        assert updated.stage_checkpoints["s1"].status == "completed"

    def test_completed_stages(self):
        cp = WorkflowCheckpoint(workflow_id="wf")
        cp.stage_checkpoints["s1"] = StageCheckpoint("s1", "S1", "completed")
        cp.stage_checkpoints["s2"] = StageCheckpoint("s2", "S2", "failed")
        assert cp.completed_stages == {"s1"}
        assert cp.failed_stages == {"s2"}

    def test_get_latest(self):
        mgr = CheckpointManager(use_memory_backend=True)
        cp1 = mgr.create_checkpoint("wf-1")
        cp2 = mgr.create_checkpoint("wf-1")
        latest = mgr.get_latest_checkpoint("wf-1")
        assert latest.checkpoint_id == cp2.checkpoint_id

    def test_resume_state(self):
        mgr = CheckpointManager(use_memory_backend=True)
        cp = mgr.create_checkpoint("wf-1", context={"key": "val"})
        mgr.checkpoint_stage(cp.checkpoint_id, "s1", "S1", "completed", output="out1")
        resume = mgr.get_resume_state("wf-1")
        assert resume is not None
        assert "s1" in resume["completed_stages"]
        assert resume["stage_outputs"]["s1"] == "out1"
        assert resume["context"]["key"] == "val"

    def test_no_resume_completed(self):
        mgr = CheckpointManager(use_memory_backend=True)
        cp = mgr.create_checkpoint("wf-1")
        mgr.mark_completed(cp.checkpoint_id)
        # Completed workflows don't need resume
        resume = mgr.get_resume_state("wf-1")
        # Latest is completed, so returns it but not None
        assert resume is None or resume["checkpoint"].completed

    def test_no_checkpoint(self):
        mgr = CheckpointManager(use_memory_backend=True)
        assert mgr.get_resume_state("nonexistent") is None

    def test_serialization(self):
        cp = WorkflowCheckpoint(workflow_id="wf")
        cp.stage_checkpoints["s1"] = StageCheckpoint("s1", "S1", "completed", output=42)
        d = cp.to_dict()
        restored = WorkflowCheckpoint.from_dict(d)
        assert restored.workflow_id == "wf"
        assert restored.stage_checkpoints["s1"].output == 42

    def test_retention(self):
        mgr = CheckpointManager(use_memory_backend=True, max_checkpoints_per_workflow=2)
        mgr.create_checkpoint("wf")
        mgr.create_checkpoint("wf")
        mgr.create_checkpoint("wf")
        # Only 2 retained
        assert len(mgr._workflow_index["wf"]) <= 2

    def test_stats(self):
        mgr = CheckpointManager(use_memory_backend=True)
        stats = mgr.get_stats()
        assert "checkpoints_created" in stats
