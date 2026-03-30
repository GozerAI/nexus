"""
Tests for WorkflowKnowledgeCollector — pattern extraction, tool knowledge, fact building.
"""

import pytest
from unittest.mock import MagicMock

from nexus.data.workflow_collector import (
    WorkflowKnowledgeCollector,
    build_workflow_fact,
    _build_procedural_knowledge,
    _build_tags,
    DEFAULT_TOOL_TYPES,
)
from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType


# ─── Sample Data ──────────────────────────────────────────────

SAMPLE_WORKFLOWS = [
    {
        "id": "wf-001",
        "workflow_name": "Lead Capture to CRM",
        "tool_type": "n8n",
        "primary_category": "lead-gen-crm",
        "quality_score": 85,
        "original_description": "Captures leads from a web form and pushes them into HubSpot CRM with deduplication.",
        "estimated_complexity": "medium",
        "node_types": ["Webhook", "IF", "HubSpot", "Slack"],
        "trigger_type": "webhook",
        "tags": ["crm", "leads", "hubspot"],
    },
    {
        "id": "wf-002",
        "workflow_name": "Daily Data Sync",
        "tool_type": "airflow",
        "primary_category": "data-pipeline",
        "quality_score": 72,
        "original_description": "Syncs data from PostgreSQL to BigQuery on a daily schedule.",
        "estimated_complexity": "low",
        "node_types": ["PostgresOperator", "BigQueryInsertJobOperator"],
        "trigger_type": "schedule",
        "tags": ["etl", "sync", "bigquery"],
    },
    {
        "id": "wf-003",
        "workflow_name": "AI Image Pipeline",
        "tool_type": "comfyui",
        "primary_category": "ai-image-generation",
        "quality_score": 90,
        "original_description": "Generates product images using Stable Diffusion with ControlNet and upscaling.",
        "estimated_complexity": "high",
        "node_types": ["KSampler", "ControlNet", "Upscale"],
        "trigger_type": "manual",
        "tags": ["stable-diffusion", "image-gen"],
    },
]


# ─── build_workflow_fact Tests ────────────────────────────────


class TestBuildWorkflowFact:
    """Test the pure build_workflow_fact function."""

    def test_builds_basic_fact(self):
        fact = build_workflow_fact(SAMPLE_WORKFLOWS[0])
        assert "Lead Capture to CRM" in fact
        assert "n8n" in fact
        assert "lead-gen-crm" in fact

    def test_includes_quality_score(self):
        fact = build_workflow_fact(SAMPLE_WORKFLOWS[0])
        assert "85/100" in fact

    def test_includes_complexity(self):
        fact = build_workflow_fact(SAMPLE_WORKFLOWS[0])
        assert "medium" in fact

    def test_includes_truncated_description(self):
        fact = build_workflow_fact(SAMPLE_WORKFLOWS[0])
        assert "Captures leads" in fact

    def test_returns_empty_for_missing_name(self):
        assert build_workflow_fact({"tool_type": "n8n"}) == ""

    def test_returns_empty_for_missing_tool_type(self):
        assert build_workflow_fact({"workflow_name": "Test"}) == ""

    def test_handles_minimal_workflow(self):
        fact = build_workflow_fact({"workflow_name": "Min", "tool_type": "zapier"})
        assert "Min" in fact
        assert "zapier" in fact


# ─── _build_procedural_knowledge Tests ────────────────────────


class TestBuildProceduralKnowledge:
    """Test procedural knowledge generation."""

    def test_builds_how_to(self):
        result = _build_procedural_knowledge(SAMPLE_WORKFLOWS[0])
        assert "To implement" in result
        assert "n8n" in result
        assert "webhook" in result.lower()

    def test_includes_nodes(self):
        result = _build_procedural_knowledge(SAMPLE_WORKFLOWS[0])
        assert "Webhook" in result
        assert "HubSpot" in result

    def test_returns_empty_without_nodes(self):
        wf = {"workflow_name": "Test", "tool_type": "n8n", "node_types": []}
        assert _build_procedural_knowledge(wf) == ""

    def test_returns_empty_without_name(self):
        wf = {"tool_type": "n8n", "node_types": ["A"]}
        assert _build_procedural_knowledge(wf) == ""


# ─── _build_tags Tests ────────────────────────────────────────


class TestBuildTags:
    """Test tag building."""

    def test_basic_tags(self):
        tags = _build_tags("n8n")
        assert "knowledge_harvester" in tags
        assert "n8n" in tags

    def test_with_category(self):
        tags = _build_tags("airflow", "data-pipeline")
        assert "data-pipeline" in tags

    def test_no_category(self):
        tags = _build_tags("make", None)
        assert len(tags) == 2
        assert "knowledge_harvester" in tags
        assert "make" in tags


# ─── DEFAULT_TOOL_TYPES Tests ────────────────────────────────


class TestDefaultToolTypes:
    """Verify the default tool type list."""

    def test_has_tool_types(self):
        assert len(DEFAULT_TOOL_TYPES) >= 20

    def test_includes_original_tools(self):
        for tool in ["n8n", "zapier", "make", "langchain", "crewai", "autogen", "ifttt"]:
            assert tool in DEFAULT_TOOL_TYPES

    def test_includes_wave2_tools(self):
        for tool in ["tekton", "github-actions", "home-assistant", "mlflow",
                      "dbt", "camunda", "kafka-connect", "camel"]:
            assert tool in DEFAULT_TOOL_TYPES


# ─── WorkflowKnowledgeCollector Init Tests ────────────────────


class TestWorkflowKnowledgeCollectorInit:
    """Test collector initialization."""

    def test_init_with_defaults(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        assert collector.knowledge_base is kb
        assert collector.confidence == 0.8
        assert collector.db_dsn is None

    def test_init_custom_params(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(
            knowledge_base=kb,
            db_dsn="postgresql://localhost/test",
            confidence=0.9,
        )
        assert collector.db_dsn == "postgresql://localhost/test"
        assert collector.confidence == 0.9


# ─── extract_pattern_knowledge Tests ─────────────────────────


class TestExtractPatternKnowledge:
    """Test pattern knowledge extraction."""

    def test_extracts_from_workflows(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_pattern_knowledge(SAMPLE_WORKFLOWS)
        # Each workflow should produce at least a factual item
        assert len(ids) >= 3  # 3 workflows × (fact + optional procedural)
        assert len(kb.knowledge_items) >= 3

    def test_deduplicates_by_workflow_id(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids1 = collector.extract_pattern_knowledge(SAMPLE_WORKFLOWS)
        ids2 = collector.extract_pattern_knowledge(SAMPLE_WORKFLOWS)
        # Second call should produce nothing (already collected)
        assert len(ids2) == 0

    def test_stores_correct_knowledge_type(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        collector.extract_pattern_knowledge([SAMPLE_WORKFLOWS[0]])
        # Should have at least one FACTUAL entry
        factual = [
            item for item in kb.knowledge_items.values()
            if item.knowledge_type == KnowledgeType.FACTUAL
        ]
        assert len(factual) >= 1

    def test_stores_procedural_knowledge(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        collector.extract_pattern_knowledge([SAMPLE_WORKFLOWS[0]])
        # Workflow with node_types should produce procedural knowledge
        procedural = [
            item for item in kb.knowledge_items.values()
            if item.knowledge_type == KnowledgeType.PROCEDURAL
        ]
        assert len(procedural) >= 1

    def test_source_includes_tool_type(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        collector.extract_pattern_knowledge([SAMPLE_WORKFLOWS[0]])
        sources = [item.source for item in kb.knowledge_items.values()]
        assert any("knowledge_harvester:n8n" in s for s in sources)

    def test_tags_include_category(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        collector.extract_pattern_knowledge([SAMPLE_WORKFLOWS[0]])
        all_tags = []
        for item in kb.knowledge_items.values():
            all_tags.extend(item.context_tags)
        assert "lead-gen-crm" in all_tags

    def test_skips_workflow_without_id(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_pattern_knowledge([{"workflow_name": "No ID", "tool_type": "n8n"}])
        assert len(ids) == 0


# ─── extract_tool_knowledge Tests ────────────────────────────


class TestExtractToolKnowledge:
    """Test tool-specific knowledge extraction."""

    def test_creates_tool_summaries(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_tool_knowledge(SAMPLE_WORKFLOWS)
        # 3 distinct tool types → 3 summaries
        assert len(ids) == 3

    def test_filters_by_tool_type(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_tool_knowledge(SAMPLE_WORKFLOWS, tool_type="n8n")
        assert len(ids) == 1
        # Verify content mentions n8n
        item = kb.knowledge_items[ids[0]]
        assert "n8n" in item.content

    def test_summary_includes_count(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        collector.extract_tool_knowledge(SAMPLE_WORKFLOWS, tool_type="n8n")
        items = list(kb.knowledge_items.values())
        assert any("1 workflows" in item.content for item in items)

    def test_summary_includes_categories(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        collector.extract_tool_knowledge(SAMPLE_WORKFLOWS, tool_type="n8n")
        items = list(kb.knowledge_items.values())
        assert any("lead-gen-crm" in item.content for item in items)


# ─── collect_patterns Tests ──────────────────────────────────


class TestCollectPatterns:
    """Test the high-level collect_patterns method."""

    def test_returns_stats(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        stats = collector.collect_patterns(SAMPLE_WORKFLOWS)
        assert stats["workflows_processed"] == 3
        assert stats["knowledge_items_created"] >= 3
        assert stats["errors"] == 0

    def test_respects_limit(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        stats = collector.collect_patterns(SAMPLE_WORKFLOWS, limit=1)
        assert stats["workflows_processed"] == 1


# ─── collect_tool_knowledge Tests ────────────────────────────


class TestCollectToolKnowledge:
    """Test the high-level collect_tool_knowledge method."""

    def test_returns_stats(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        stats = collector.collect_tool_knowledge(SAMPLE_WORKFLOWS)
        assert stats["tools_processed"] == 3
        assert stats["knowledge_items_created"] == 3
        assert stats["errors"] == 0

    def test_filters_by_tool(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        stats = collector.collect_tool_knowledge(SAMPLE_WORKFLOWS, tool_type="airflow")
        assert stats["tools_processed"] == 1
        assert stats["knowledge_items_created"] == 1


# ─── get_collection_stats Tests ──────────────────────────────


class TestCollectionStats:
    """Test collection stats tracking."""

    def test_empty_stats(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        stats = collector.get_collection_stats()
        assert stats["workflows_collected"] == 0
        assert stats["collected_workflow_ids"] == []

    def test_stats_after_collection(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        collector.extract_pattern_knowledge(SAMPLE_WORKFLOWS)
        stats = collector.get_collection_stats()
        assert stats["workflows_collected"] == 3
        assert "wf-001" in stats["collected_workflow_ids"]


# ─── Various Tool Type Tests ─────────────────────────────────


class TestVariousToolTypes:
    """Test knowledge extraction across different tool types."""

    @pytest.mark.parametrize("tool_type", [
        "n8n", "zapier", "make", "langchain", "crewai", "airflow",
        "tekton", "github-actions", "home-assistant", "dbt",
    ])
    def test_tool_type_in_source(self, tool_type):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        wf = {
            "id": f"wf-{tool_type}",
            "workflow_name": f"Test {tool_type} Workflow",
            "tool_type": tool_type,
            "primary_category": "general-productivity",
            "quality_score": 70,
            "original_description": f"A test workflow for {tool_type} automation platform.",
        }
        ids = collector.extract_pattern_knowledge([wf])
        assert len(ids) >= 1
        sources = [item.source for item in kb.knowledge_items.values()]
        assert any(f"knowledge_harvester:{tool_type}" in s for s in sources)


# ─── extract_graph_knowledge Tests ────────────────────────────


SAMPLE_GRAPH_DATA = {
    "nodes": [
        {"id": "n1", "label": "CRM Integration", "properties": {"category": "crm"}},
        {"id": "n2", "label": "Email Parser", "properties": {"category": "crm"}},
        {"id": "n3", "label": "Data Pipeline", "properties": {"category": "etl"}},
        {"id": "n4", "label": "API Gateway", "properties": {"category": "api"}},
    ],
    "edges": [
        {"source": "n1", "target": "n2", "type": "similar_to"},
        {"source": "n1", "target": "n3", "type": "depends_on"},
        {"source": "n3", "target": "n4", "type": "feeds_into"},
    ],
}


class TestExtractGraphKnowledge:
    """Test graph knowledge extraction from nodes and edges."""

    def test_creates_cluster_knowledge(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_graph_knowledge(SAMPLE_GRAPH_DATA)
        # crm has 2 nodes → cluster
        contents = [kb.knowledge_items[kid].content for kid in ids]
        cluster_items = [c for c in contents if "cluster" in c.lower()]
        assert len(cluster_items) >= 1
        assert any("crm" in c.lower() for c in cluster_items)

    def test_creates_bridge_node_knowledge(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_graph_knowledge(SAMPLE_GRAPH_DATA)
        contents = [kb.knowledge_items[kid].content for kid in ids]
        bridge_items = [c for c in contents if "bridge" in c.lower()]
        assert len(bridge_items) >= 1

    def test_empty_nodes_returns_empty(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_graph_knowledge({"nodes": [], "edges": []})
        assert ids == []

    def test_no_edges_still_creates_clusters(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        data = {
            "nodes": [
                {"id": "a", "label": "X", "properties": {"category": "cat1"}},
                {"id": "b", "label": "Y", "properties": {"category": "cat1"}},
            ],
            "edges": [],
        }
        ids = collector.extract_graph_knowledge(data)
        assert len(ids) >= 1  # cluster for cat1
        contents = [kb.knowledge_items[kid].content for kid in ids]
        assert any("cat1" in c for c in contents)

    def test_single_node_no_cluster(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        data = {
            "nodes": [{"id": "solo", "label": "Alone", "properties": {"category": "solo_cat"}}],
            "edges": [],
        }
        ids = collector.extract_graph_knowledge(data)
        # Single node category should not create a cluster
        contents = [kb.knowledge_items[kid].content for kid in ids]
        cluster_items = [c for c in contents if "cluster" in c.lower()]
        assert len(cluster_items) == 0

    def test_knowledge_type_is_factual(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_graph_knowledge(SAMPLE_GRAPH_DATA)
        for kid in ids:
            assert kb.knowledge_items[kid].knowledge_type == KnowledgeType.FACTUAL

    def test_tags_include_graph_markers(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_graph_knowledge(SAMPLE_GRAPH_DATA)
        all_tags = []
        for kid in ids:
            all_tags.extend(kb.knowledge_items[kid].context_tags or [])
        assert "cluster" in all_tags or "bridge_node" in all_tags

    def test_string_properties_parsed(self):
        """Nodes with stringified JSON properties should still work."""
        import json
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        data = {
            "nodes": [
                {"id": "s1", "label": "A", "properties": json.dumps({"category": "json_cat"})},
                {"id": "s2", "label": "B", "properties": json.dumps({"category": "json_cat"})},
            ],
            "edges": [],
        }
        ids = collector.extract_graph_knowledge(data)
        contents = [kb.knowledge_items[kid].content for kid in ids]
        assert any("json_cat" in c for c in contents)

    def test_missing_graph_keys_returns_empty(self):
        kb = KnowledgeBase()
        collector = WorkflowKnowledgeCollector(knowledge_base=kb)
        ids = collector.extract_graph_knowledge({})
        assert ids == []
