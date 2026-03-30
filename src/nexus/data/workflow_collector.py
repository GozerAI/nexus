"""
Workflow Knowledge Collector for Nexus AI Platform.

Extracts knowledge from the Workflow Harvester database about
automation patterns, tool usage, and workflow best practices.

Unlike the WikipediaCollector and PyPIKnowledgeCollector which pull
from external APIs, this collector works with in-memory workflow data
passed as parameters (for testability and decoupling from asyncpg).

Knowledge types stored:
- FACTUAL: workflow facts (tool, category, quality, nodes)
- PROCEDURAL: how-to knowledge for workflow patterns
"""

import logging
from typing import Any, Dict, List, Optional, Set

from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType

logger = logging.getLogger(__name__)

# Default tool types to collect knowledge about
DEFAULT_TOOL_TYPES = [
    "n8n",
    "zapier",
    "make",
    "langchain",
    "crewai",
    "autogen",
    "ifttt",
    "activepieces",
    "windmill",
    "temporal",
    "airflow",
    "node-red",
    "prefect",
    "dagster",
    "langgraph",
    "comfyui",
    "dify",
    "flowise",
    "pipedream",
    "argo",
    "luigi",
    "tekton",
    "github-actions",
    "home-assistant",
    "mlflow",
    "dbt",
    "camunda",
    "kafka-connect",
    "camel",
]


class WorkflowKnowledgeCollector:
    """
    Collects knowledge from knowledge harvester data.

    Works with in-memory workflow dicts for testability.
    Each workflow dict should have keys like: id, workflow_name,
    tool_type, primary_category, quality_score, tags, node_types,
    original_description, estimated_complexity, trigger_type.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        db_dsn: Optional[str] = None,
        confidence: float = 0.8,
    ):
        self.knowledge_base = knowledge_base
        self.db_dsn = db_dsn
        self.confidence = confidence
        self._collected_workflows: Set[str] = set()

    def extract_pattern_knowledge(
        self, workflows: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract pattern knowledge from a list of workflow dicts.

        Generates factual knowledge items about automation patterns
        and stores them in the KnowledgeBase.

        Args:
            workflows: List of workflow dicts from the harvester.

        Returns:
            List of knowledge IDs created.
        """
        knowledge_ids = []

        for wf in workflows:
            wf_id = str(wf.get("id", ""))
            if not wf_id or wf_id in self._collected_workflows:
                continue

            # Build a factual sentence about this workflow
            fact = build_workflow_fact(wf)
            if fact:
                tool_type = wf.get("tool_type", "unknown")
                category = wf.get("primary_category", "general")
                tags = _build_tags(tool_type, category)

                kid = self.knowledge_base.add_knowledge(
                    content=fact,
                    knowledge_type=KnowledgeType.FACTUAL,
                    source=f"knowledge_harvester:{tool_type}",
                    confidence=self.confidence,
                    context_tags=tags,
                )
                knowledge_ids.append(kid)
                self._collected_workflows.add(wf_id)

            # Extract procedural knowledge if we have enough detail
            procedural = _build_procedural_knowledge(wf)
            if procedural:
                tool_type = wf.get("tool_type", "unknown")
                category = wf.get("primary_category", "general")
                tags = _build_tags(tool_type, category) + ["how-to"]

                kid = self.knowledge_base.add_knowledge(
                    content=procedural,
                    knowledge_type=KnowledgeType.PROCEDURAL,
                    source=f"knowledge_harvester:{tool_type}",
                    confidence=self.confidence - 0.05,
                    context_tags=tags,
                )
                knowledge_ids.append(kid)

        return knowledge_ids

    def extract_tool_knowledge(
        self, workflows: List[Dict[str, Any]], tool_type: Optional[str] = None
    ) -> List[str]:
        """
        Extract tool-specific knowledge from workflows.

        Groups workflows by tool type and creates aggregate knowledge
        about each tool's strengths, common patterns, and categories.

        Args:
            workflows: List of workflow dicts.
            tool_type: Optional filter to a specific tool.

        Returns:
            List of knowledge IDs created.
        """
        knowledge_ids = []

        # Group by tool_type
        by_tool: Dict[str, List[Dict[str, Any]]] = {}
        for wf in workflows:
            tt = wf.get("tool_type", "unknown")
            if tool_type and tt != tool_type:
                continue
            by_tool.setdefault(tt, []).append(wf)

        for tt, tool_workflows in by_tool.items():
            # Aggregate stats
            count = len(tool_workflows)
            categories = set()
            total_quality = 0
            quality_count = 0

            for wf in tool_workflows:
                cat = wf.get("primary_category")
                if cat:
                    categories.add(cat)
                qs = wf.get("quality_score")
                if qs is not None:
                    total_quality += qs
                    quality_count += 1

            avg_quality = round(total_quality / quality_count) if quality_count else 0

            summary = (
                f"{tt} has {count} workflows in the library"
                f" covering categories: {', '.join(sorted(categories)[:5])}."
                f" Average quality score is {avg_quality}/100."
            )

            tags = _build_tags(tt, None) + ["tool_summary"]

            kid = self.knowledge_base.add_knowledge(
                content=summary,
                knowledge_type=KnowledgeType.FACTUAL,
                source=f"knowledge_harvester:{tt}:summary",
                confidence=self.confidence,
                context_tags=tags,
            )
            knowledge_ids.append(kid)

        return knowledge_ids

    def collect_patterns(
        self, workflows: List[Dict[str, Any]], limit: int = 100
    ) -> Dict[str, Any]:
        """
        Collect pattern knowledge from workflows.

        Args:
            workflows: List of workflow dicts.
            limit: Maximum workflows to process.

        Returns:
            Collection statistics.
        """
        stats = {
            "workflows_processed": 0,
            "knowledge_items_created": 0,
            "knowledge_ids": [],
            "errors": 0,
        }

        for wf in workflows[:limit]:
            try:
                ids = self.extract_pattern_knowledge([wf])
                stats["knowledge_items_created"] += len(ids)
                stats["knowledge_ids"].extend(ids)
                stats["workflows_processed"] += 1
            except Exception as e:
                logger.error(f"Error processing workflow {wf.get('id')}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Workflow pattern collection complete: {stats['workflows_processed']} workflows, "
            f"{stats['knowledge_items_created']} knowledge items"
        )
        return stats

    def collect_tool_knowledge(
        self, workflows: List[Dict[str, Any]], tool_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Collect tool-specific knowledge from workflows.

        Args:
            workflows: List of workflow dicts.
            tool_type: Optional filter to a specific tool.

        Returns:
            Collection statistics.
        """
        stats = {
            "tools_processed": 0,
            "knowledge_items_created": 0,
            "knowledge_ids": [],
            "errors": 0,
        }

        try:
            ids = self.extract_tool_knowledge(workflows, tool_type=tool_type)
            stats["knowledge_items_created"] = len(ids)
            stats["knowledge_ids"] = ids
            # Count distinct tools
            tools = set()
            for wf in workflows:
                tt = wf.get("tool_type", "unknown")
                if tool_type and tt != tool_type:
                    continue
                tools.add(tt)
            stats["tools_processed"] = len(tools)
        except Exception as e:
            logger.error(f"Error collecting tool knowledge: {e}")
            stats["errors"] += 1

        return stats

    def extract_graph_knowledge(
        self, graph_data: Dict[str, Any]
    ) -> List[str]:
        """Extract knowledge items from intelligence graph query results.

        Takes graph data containing nodes and edges, identifies clusters
        and bridge nodes, and creates knowledge items about them.

        Args:
            graph_data: Dict with "nodes" (list) and "edges" (list) keys.
                        Each node has "id", "label", "properties".
                        Each edge has "source", "target", "type".

        Returns:
            List of knowledge IDs created.
        """
        knowledge_ids: List[str] = []
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        if not nodes:
            return knowledge_ids

        # --- Cluster detection: group nodes by category ---
        category_nodes: Dict[str, List[Dict[str, Any]]] = {}
        for node in nodes:
            props = node.get("properties", {})
            if isinstance(props, str):
                try:
                    import json as _json
                    props = _json.loads(props)
                except Exception:
                    props = {}
            cat = props.get("category", node.get("label", "unknown"))
            category_nodes.setdefault(cat, []).append(node)

        for cat, cat_nodes in category_nodes.items():
            if len(cat_nodes) >= 2:
                fact = (
                    f"Category '{cat}' has a cluster of {len(cat_nodes)} "
                    f"related workflows in the intelligence graph."
                )
                tags = _build_tags("graph", cat) + ["cluster"]
                kid = self.knowledge_base.add_knowledge(
                    content=fact,
                    knowledge_type=KnowledgeType.FACTUAL,
                    source="intelligence_graph:cluster",
                    confidence=self.confidence,
                    context_tags=tags,
                )
                knowledge_ids.append(kid)

        # --- Bridge node detection: nodes with edges to multiple categories ---
        if edges:
            node_categories: Dict[str, Set[str]] = {}
            node_map = {str(n.get("id", "")): n for n in nodes}

            for edge in edges:
                src = str(edge.get("source", ""))
                tgt = str(edge.get("target", ""))
                for nid in (src, tgt):
                    if nid not in node_categories:
                        node_categories[nid] = set()
                    node = node_map.get(nid, {})
                    props = node.get("properties", {})
                    if isinstance(props, str):
                        try:
                            import json as _json
                            props = _json.loads(props)
                        except Exception:
                            props = {}
                    cat = props.get("category", node.get("label", "unknown"))
                    # Also add the other end's category
                    other = tgt if nid == src else src
                    other_node = node_map.get(other, {})
                    other_props = other_node.get("properties", {})
                    if isinstance(other_props, str):
                        try:
                            import json as _json
                            other_props = _json.loads(other_props)
                        except Exception:
                            other_props = {}
                    other_cat = other_props.get("category", other_node.get("label", "unknown"))
                    node_categories[nid].add(cat)
                    node_categories[nid].add(other_cat)

            for nid, cats in node_categories.items():
                if len(cats) >= 2:
                    node = node_map.get(nid, {})
                    label = node.get("label", nid)
                    fact = (
                        f"'{label}' is a bridge node connecting categories: "
                        f"{', '.join(sorted(cats))}."
                    )
                    tags = _build_tags("graph", None) + ["bridge_node"]
                    kid = self.knowledge_base.add_knowledge(
                        content=fact,
                        knowledge_type=KnowledgeType.FACTUAL,
                        source="intelligence_graph:bridge",
                        confidence=self.confidence - 0.05,
                        context_tags=tags,
                    )
                    knowledge_ids.append(kid)

        return knowledge_ids

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return statistics about collected workflows."""
        return {
            "workflows_collected": len(self._collected_workflows),
            "collected_workflow_ids": sorted(self._collected_workflows),
        }


def build_workflow_fact(workflow: Dict[str, Any]) -> str:
    """
    Build a factual sentence from a workflow dict.

    Args:
        workflow: Dict with workflow metadata.

    Returns:
        A factual sentence describing the workflow, or empty string.
    """
    name = workflow.get("workflow_name", "")
    tool_type = workflow.get("tool_type", "")
    category = workflow.get("primary_category", "")
    quality = workflow.get("quality_score")
    description = workflow.get("original_description", "")
    complexity = workflow.get("estimated_complexity", "")

    if not name or not tool_type:
        return ""

    parts = [f"'{name}' is a {tool_type} workflow"]

    if category:
        parts.append(f"in the {category} category")

    if quality is not None:
        parts.append(f"with a quality score of {quality}/100")

    if complexity:
        parts.append(f"(complexity: {complexity})")

    fact = " ".join(parts) + "."

    if description and len(description) > 20:
        # Add a truncated description
        desc = description[:150].strip()
        if not desc.endswith("."):
            desc += "..."
        fact += f" {desc}"

    return fact


def _build_procedural_knowledge(workflow: Dict[str, Any]) -> str:
    """
    Build procedural (how-to) knowledge from a workflow.

    Returns a how-to sentence if the workflow has enough detail.
    """
    name = workflow.get("workflow_name", "")
    tool_type = workflow.get("tool_type", "")
    node_types = workflow.get("node_types") or []
    trigger_type = workflow.get("trigger_type", "")
    tags = workflow.get("tags") or []

    if not name or not tool_type or not node_types:
        return ""

    parts = [f"To implement '{name}' in {tool_type}"]

    if trigger_type:
        parts.append(f"start with a {trigger_type} trigger")

    if node_types:
        nodes_str = ", ".join(node_types[:5])
        parts.append(f"using nodes: {nodes_str}")

    if tags:
        tags_str = ", ".join(tags[:5])
        parts.append(f"(tags: {tags_str})")

    return ", ".join(parts) + "."


def _build_tags(
    tool_type: str, category: Optional[str] = None
) -> List[str]:
    """Build context tags for workflow knowledge items."""
    tags = ["knowledge_harvester"]
    if tool_type:
        tags.append(tool_type.lower())
    if category:
        tags.append(category.lower().replace(" ", "_"))
    return tags
