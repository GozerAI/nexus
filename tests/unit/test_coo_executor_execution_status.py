"""Regression tests for honest COO executor status reporting."""

import asyncio

from nexus.coo.executor import AutonomousExecutor, ExecutionStatus


class _Item:
    def __init__(self, item_id="task-1", title="", description=""):
        self.id = item_id
        self.title = title
        self.description = description


def test_execute_marks_structured_failure_payload_as_failed():
    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)
        item = _Item(title="Analyze trend", description="Look for anomalies")

        async def failing_executor(target_item, context):
            return {"success": False, "error": "trend backend unavailable"}

        executor._execute_trend_analyzer = failing_executor

        result = await executor.execute(item, "trend_analyzer")

        assert result.success is False
        assert result.status is ExecutionStatus.FAILED
        assert result.error == "trend backend unavailable"

    asyncio.run(run_execute())


def test_execute_marks_error_payload_without_success_flag_as_failed():
    async def run_execute():
        executor = AutonomousExecutor(intelligence=None)
        item = _Item(title="Analyze trend", description="Look for anomalies")

        async def failing_executor(target_item, context):
            return {"error": "trend backend unavailable", "trends_analyzed": False}

        executor._execute_trend_analyzer = failing_executor

        result = await executor.execute(item, "trend_analyzer")

        assert result.success is False
        assert result.status is ExecutionStatus.FAILED
        assert result.error == "trend backend unavailable"

    asyncio.run(run_execute())
