"""
Pipeline execution DAG (Directed Acyclic Graph).

Models pipeline stages as a DAG, enabling topological execution ordering,
critical path analysis, and maximum parallelism scheduling.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """A node in the execution DAG."""
    node_id: str
    name: str
    fn: Optional[Callable[..., Coroutine]] = None
    dependencies: List[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    output: Any = None
    error: Optional[str] = None
    estimated_duration_ms: float = 0.0
    actual_duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DAGExecutionResult:
    """Result of executing a DAG."""
    dag_id: str
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    skipped_nodes: int
    total_duration_ms: float
    critical_path_ms: float
    parallelism_efficiency: float  # actual vs sequential duration
    node_results: Dict[str, DAGNode] = field(default_factory=dict)


class ExecutionDAG:
    """
    Represents a pipeline as a directed acyclic graph.

    Features:
    - Cycle detection on construction
    - Topological ordering for execution
    - Critical path analysis
    - Parallelism calculation
    """

    def __init__(self, dag_id: Optional[str] = None):
        self.dag_id = dag_id or str(uuid.uuid4())[:8]
        self._nodes: Dict[str, DAGNode] = {}
        self._adjacency: Dict[str, List[str]] = defaultdict(list)  # node -> dependents
        self._reverse: Dict[str, List[str]] = defaultdict(list)    # node -> dependencies

    def add_node(
        self,
        node_id: str,
        name: str,
        fn: Optional[Callable[..., Coroutine]] = None,
        dependencies: Optional[List[str]] = None,
        estimated_duration_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DAGNode:
        """Add a node to the DAG."""
        deps = dependencies or []
        node = DAGNode(
            node_id=node_id,
            name=name,
            fn=fn,
            dependencies=deps,
            estimated_duration_ms=estimated_duration_ms,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node

        for dep in deps:
            self._adjacency[dep].append(node_id)
            self._reverse[node_id].append(dep)

        return node

    def validate(self) -> List[str]:
        """
        Validate the DAG structure.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check for missing dependencies
        for nid, node in self._nodes.items():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    errors.append(f"Node '{nid}' depends on missing node '{dep}'")

        # Check for cycles using DFS
        if not errors:
            cycle = self._detect_cycle()
            if cycle:
                errors.append(f"Cycle detected: {' -> '.join(cycle)}")

        return errors

    def _detect_cycle(self) -> Optional[List[str]]:
        """Detect cycles using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in self._nodes}
        parent = {}

        def dfs(node_id: str) -> Optional[List[str]]:
            color[node_id] = GRAY
            for neighbor in self._adjacency.get(node_id, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    # Found cycle, reconstruct it
                    cycle = [neighbor, node_id]
                    return cycle
                if color[neighbor] == WHITE:
                    parent[neighbor] = node_id
                    result = dfs(neighbor)
                    if result:
                        return result
            color[node_id] = BLACK
            return None

        for nid in self._nodes:
            if color[nid] == WHITE:
                result = dfs(nid)
                if result:
                    return result
        return None

    def topological_order(self) -> List[str]:
        """Return nodes in topological order."""
        in_degree = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            for dep in self._reverse.get(nid, []):
                in_degree[nid] += 1

        queue = deque([nid for nid, d in in_degree.items() if d == 0])
        order = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for dependent in self._adjacency.get(nid, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return order

    def critical_path(self) -> Tuple[List[str], float]:
        """
        Find the critical path (longest path) through the DAG.

        Returns:
            (path_node_ids, total_estimated_ms)
        """
        order = self.topological_order()
        dist: Dict[str, float] = {nid: 0.0 for nid in self._nodes}
        prev: Dict[str, Optional[str]] = {nid: None for nid in self._nodes}

        for nid in order:
            node = self._nodes[nid]
            current_end = dist[nid] + node.estimated_duration_ms
            for dependent in self._adjacency.get(nid, []):
                if current_end > dist[dependent]:
                    dist[dependent] = current_end
                    prev[dependent] = nid

        # Find end node with max distance
        if not dist:
            return [], 0.0

        end_node = max(dist, key=lambda n: dist[n] + self._nodes[n].estimated_duration_ms)
        total = dist[end_node] + self._nodes[end_node].estimated_duration_ms

        # Reconstruct path
        path = [end_node]
        current = prev[end_node]
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()

        return path, total

    def get_parallel_levels(self) -> List[List[str]]:
        """
        Group nodes into parallel execution levels.

        Each level contains nodes that can execute concurrently
        (all dependencies in previous levels).
        """
        remaining = set(self._nodes.keys())
        completed: Set[str] = set()
        levels = []

        while remaining:
            level = []
            for nid in list(remaining):
                deps = set(self._nodes[nid].dependencies)
                if deps.issubset(completed):
                    level.append(nid)
            if not level:
                break  # Remaining nodes have unsatisfied deps
            levels.append(level)
            for nid in level:
                remaining.discard(nid)
                completed.add(nid)

        return levels

    @property
    def nodes(self) -> Dict[str, DAGNode]:
        return dict(self._nodes)

    @property
    def node_count(self) -> int:
        return len(self._nodes)


class DAGExecutor:
    """
    Executes an ExecutionDAG with maximum parallelism.

    Uses the DAG's parallel levels to schedule concurrent execution
    while respecting all dependency constraints.
    """

    def __init__(self, max_concurrency: int = 10):
        self._max_concurrency = max_concurrency
        self._stats = {
            "dags_executed": 0,
            "nodes_executed": 0,
            "nodes_failed": 0,
        }

    async def execute(
        self,
        dag: ExecutionDAG,
        context: Optional[Dict[str, Any]] = None,
    ) -> DAGExecutionResult:
        """
        Execute the DAG.

        Args:
            dag: The DAG to execute
            context: Shared context available to all nodes

        Returns:
            DAGExecutionResult
        """
        errors = dag.validate()
        if errors:
            raise ValueError(f"Invalid DAG: {errors}")

        self._stats["dags_executed"] += 1
        ctx = dict(context or {})
        start = time.time()
        levels = dag.get_parallel_levels()
        semaphore = asyncio.Semaphore(self._max_concurrency)

        for level in levels:
            tasks = []
            for nid in level:
                node = dag.nodes[nid]
                # Check if dependencies succeeded
                all_deps_ok = all(
                    dag.nodes[d].status == NodeStatus.COMPLETED
                    for d in node.dependencies
                )
                if not all_deps_ok:
                    node.status = NodeStatus.SKIPPED
                    node.error = "Dependency failed"
                    continue

                dep_outputs = {
                    d: dag.nodes[d].output for d in node.dependencies
                }
                tasks.append(self._execute_node(node, ctx, dep_outputs, semaphore))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        total_ms = (time.time() - start) * 1000
        completed = sum(1 for n in dag.nodes.values() if n.status == NodeStatus.COMPLETED)
        failed = sum(1 for n in dag.nodes.values() if n.status == NodeStatus.FAILED)
        skipped = sum(1 for n in dag.nodes.values() if n.status == NodeStatus.SKIPPED)

        # Compute sequential duration for efficiency
        sequential_ms = sum(n.actual_duration_ms for n in dag.nodes.values())
        _, critical_ms = dag.critical_path()

        return DAGExecutionResult(
            dag_id=dag.dag_id,
            total_nodes=dag.node_count,
            completed_nodes=completed,
            failed_nodes=failed,
            skipped_nodes=skipped,
            total_duration_ms=total_ms,
            critical_path_ms=critical_ms,
            parallelism_efficiency=(
                sequential_ms / total_ms if total_ms > 0 else 1.0
            ),
            node_results=dag.nodes,
        )

    async def _execute_node(
        self,
        node: DAGNode,
        context: Dict[str, Any],
        dep_outputs: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single DAG node."""
        async with semaphore:
            node.status = NodeStatus.RUNNING
            start = time.time()
            try:
                if node.fn:
                    node.output = await node.fn(
                        context=context, inputs=dep_outputs
                    )
                node.status = NodeStatus.COMPLETED
                self._stats["nodes_executed"] += 1
            except Exception as e:
                node.status = NodeStatus.FAILED
                node.error = str(e)
                self._stats["nodes_failed"] += 1
                logger.warning("DAG node %s failed: %s", node.name, e)
            finally:
                node.actual_duration_ms = (time.time() - start) * 1000

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)
