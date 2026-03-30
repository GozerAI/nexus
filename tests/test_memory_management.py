"""Tests for memory subsystem — KnowledgeBase, MemoryBlockManager, and engines."""

import os
import sys
import time
import pytest
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.memory import (
    KnowledgeBase,
    KnowledgeType,
    KnowledgeConfidence,
    KnowledgeItem,
    FactualMemoryEngine,
    SkillMemoryEngine,
    MemoryBlockManager,
)
from nexus.memory.memory_block_manager import MemoryBlock, MemoryBlockType


# ---------------------------------------------------------------------------
# MemoryBlockManager — Store & Retrieve
# ---------------------------------------------------------------------------

class TestMemoryBlockManagerStoreRetrieve:
    def setup_method(self):
        self.mgr = MemoryBlockManager()

    def test_store_and_retrieve_factual(self):
        block = self.mgr.store_factual_knowledge("f1", "Earth orbits the Sun")
        assert block.block_id == "f1"
        assert block.block_type == MemoryBlockType.FACTUAL
        retrieved = self.mgr.retrieve_block("f1")
        assert retrieved is block

    def test_store_and_retrieve_skill(self):
        block = self.mgr.store_skill_knowledge("s1", "Ride a bicycle")
        assert block.block_type == MemoryBlockType.SKILL
        assert self.mgr.retrieve_block("s1") is block

    def test_store_and_retrieve_hybrid(self):
        block = self.mgr.store_hybrid_knowledge("h1", "Mixed content")
        assert block.block_type == MemoryBlockType.HYBRID
        assert self.mgr.retrieve_block("h1") is block

    def test_retrieve_nonexistent_returns_none(self):
        assert self.mgr.retrieve_block("nope") is None

    def test_store_with_metadata(self):
        meta = {"source": "textbook", "page": 42}
        block = self.mgr.store_factual_knowledge("m1", "Fact", meta)
        assert block.metadata["source"] == "textbook"
        assert block.metadata["page"] == 42

    def test_access_count_increments_on_retrieve(self):
        self.mgr.store_factual_knowledge("ac1", "data")
        self.mgr.retrieve_block("ac1")
        self.mgr.retrieve_block("ac1")
        block = self.mgr.retrieve_block("ac1")
        assert block.access_count == 3  # 3 retrieves


# ---------------------------------------------------------------------------
# MemoryBlockManager — Search by Tag
# ---------------------------------------------------------------------------

class TestMemoryBlockManagerSearch:
    def setup_method(self):
        self.mgr = MemoryBlockManager()

    def test_search_by_tag(self):
        b1 = self.mgr.store_factual_knowledge("t1", "Physics fact")
        b1.add_tag("science")
        b2 = self.mgr.store_factual_knowledge("t2", "History fact")
        b2.add_tag("history")

        results = self.mgr.search_blocks_by_tag("science")
        assert len(results) == 1
        assert results[0].block_id == "t1"

    def test_search_by_tag_with_type_filter(self):
        b1 = self.mgr.store_factual_knowledge("ft1", "Fact")
        b1.add_tag("shared")
        b2 = self.mgr.store_skill_knowledge("st1", "Skill")
        b2.add_tag("shared")

        results = self.mgr.search_blocks_by_tag("shared", MemoryBlockType.SKILL)
        assert len(results) == 1
        assert results[0].block_id == "st1"

    def test_search_no_match(self):
        results = self.mgr.search_blocks_by_tag("nonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# MemoryBlockManager — Statistics & Maintenance
# ---------------------------------------------------------------------------

class TestMemoryBlockManagerStats:
    def setup_method(self):
        self.mgr = MemoryBlockManager()

    def test_statistics_empty(self):
        stats = self.mgr.get_memory_statistics()
        assert stats["total_blocks"] == 0

    def test_statistics_with_data(self):
        self.mgr.store_factual_knowledge("f1", "A")
        self.mgr.store_skill_knowledge("s1", "B")
        self.mgr.store_hybrid_knowledge("h1", "C")
        stats = self.mgr.get_memory_statistics()
        assert stats["factual_blocks"] == 1
        assert stats["skill_blocks"] == 1
        assert stats["hybrid_blocks"] == 1
        assert stats["total_blocks"] == 3

    def test_clear_volatile_memory(self):
        self.mgr.store_factual_knowledge("f1", "Persists")
        self.mgr.store_skill_knowledge("s1", "Volatile")
        self.mgr.clear_volatile_memory()
        assert self.mgr.retrieve_block("f1") is not None
        assert self.mgr.retrieve_block("s1") is None

    def test_export_persistent_memory(self):
        self.mgr.store_factual_knowledge("f1", "Fact")
        self.mgr.store_hybrid_knowledge("h1", "Hybrid")
        self.mgr.store_skill_knowledge("s1", "Skill")
        exported = self.mgr.export_persistent_memory()
        assert "f1" in exported["factual_blocks"]
        assert "h1" in exported["hybrid_blocks"]
        assert "export_timestamp" in exported

    def test_reweight_blocks(self):
        block = self.mgr.store_factual_knowledge("rw1", "Data")
        initial_confidence = block.confidence_score
        # Simulate some accesses
        self.mgr.retrieve_block("rw1")
        self.mgr.retrieve_block("rw1")
        self.mgr.reweight_blocks()
        # Access boost should increase confidence for factual blocks
        assert block.confidence_score >= initial_confidence


# ---------------------------------------------------------------------------
# MemoryBlock — Unit Tests
# ---------------------------------------------------------------------------

class TestMemoryBlock:
    def test_to_dict(self):
        block = MemoryBlock("b1", MemoryBlockType.FACTUAL, "content", {"key": "val"})
        d = block.to_dict()
        assert d["block_id"] == "b1"
        assert d["block_type"] == "factual"
        assert d["content"] == "content"
        assert d["metadata"]["key"] == "val"
        assert d["access_count"] == 0

    def test_update_content(self):
        block = MemoryBlock("b1", MemoryBlockType.SKILL, "old")
        block.update_content("new", {"updated": True})
        assert block.content == "new"
        assert block.metadata["updated"] is True

    def test_add_tag(self):
        block = MemoryBlock("b1", MemoryBlockType.FACTUAL, "x")
        block.add_tag("science")
        block.add_tag("physics")
        assert "science" in block.tags
        assert "physics" in block.tags

    def test_increment_access(self):
        block = MemoryBlock("b1", MemoryBlockType.FACTUAL, "x")
        assert block.access_count == 0
        block.increment_access()
        block.increment_access()
        assert block.access_count == 2


# ---------------------------------------------------------------------------
# MemoryBlock — Serialization
# ---------------------------------------------------------------------------

class TestMemoryBlockSerialization:
    def test_tags_serialized_as_list(self):
        block = MemoryBlock("s1", MemoryBlockType.HYBRID, "data")
        block.add_tag("a")
        block.add_tag("b")
        d = block.to_dict()
        assert isinstance(d["tags"], list)
        assert set(d["tags"]) == {"a", "b"}

    def test_timestamps_are_iso(self):
        block = MemoryBlock("s2", MemoryBlockType.FACTUAL, "data")
        d = block.to_dict()
        # Should be parseable ISO format strings
        datetime.fromisoformat(d["created_at"])
        datetime.fromisoformat(d["updated_at"])


# ---------------------------------------------------------------------------
# KnowledgeBase — Add & Query
# ---------------------------------------------------------------------------

class TestKnowledgeBaseAddQuery:
    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_add_knowledge_returns_id(self):
        kid = self.kb.add_knowledge(
            "Python is a programming language",
            KnowledgeType.FACTUAL,
            source="test",
        )
        assert kid is not None
        assert kid in self.kb.knowledge_items

    def test_query_finds_matching_content(self):
        self.kb.add_knowledge(
            "The speed of light is 299792458 m/s",
            KnowledgeType.FACTUAL,
            source="test",
        )
        results = self.kb.query_knowledge("speed light")
        assert len(results) >= 1
        assert "speed" in str(results[0].content).lower()

    def test_query_no_match(self):
        self.kb.add_knowledge("Apples are fruit", KnowledgeType.FACTUAL, source="test")
        results = self.kb.query_knowledge("quantum chromodynamics")
        assert results == []

    def test_query_with_type_filter(self):
        self.kb.add_knowledge("Fact A", KnowledgeType.FACTUAL, source="test")
        self.kb.add_knowledge("Skill B", KnowledgeType.PROCEDURAL, source="test")
        results = self.kb.query_knowledge(
            "fact skill", knowledge_types=[KnowledgeType.PROCEDURAL]
        )
        for item in results:
            assert item.knowledge_type == KnowledgeType.PROCEDURAL

    def test_query_with_confidence_filter(self):
        self.kb.add_knowledge("Low conf", KnowledgeType.FACTUAL, source="test", confidence=0.2)
        self.kb.add_knowledge("High conf", KnowledgeType.FACTUAL, source="test", confidence=0.9)
        results = self.kb.query_knowledge("conf", min_confidence=0.5)
        for item in results:
            assert item.confidence >= 0.5

    def test_query_max_results(self):
        for i in range(20):
            self.kb.add_knowledge(f"Item {i} about science", KnowledgeType.FACTUAL, source="test")
        results = self.kb.query_knowledge("science", max_results=5)
        assert len(results) <= 5

    def test_add_knowledge_with_context_tags(self):
        kid = self.kb.add_knowledge(
            "Photosynthesis converts sunlight",
            KnowledgeType.FACTUAL,
            source="textbook",
            context_tags=["biology", "plants"],
        )
        item = self.kb.knowledge_items[kid]
        assert "biology" in item.context_tags
        assert "plants" in item.context_tags


# ---------------------------------------------------------------------------
# KnowledgeBase — Metadata & Statistics
# ---------------------------------------------------------------------------

class TestKnowledgeBaseStatistics:
    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_empty_statistics(self):
        stats = self.kb.get_knowledge_statistics()
        assert stats["total_items"] == 0

    def test_statistics_with_items(self):
        self.kb.add_knowledge("A", KnowledgeType.FACTUAL, source="t")
        self.kb.add_knowledge("B", KnowledgeType.PROCEDURAL, source="t")
        stats = self.kb.get_knowledge_statistics()
        assert stats["total_items"] == 2
        assert "factual" in stats["by_type"]
        assert "procedural" in stats["by_type"]

    def test_access_count_tracked(self):
        kid = self.kb.add_knowledge(
            "Tracked item about quantum",
            KnowledgeType.FACTUAL,
            source="test",
        )
        # First query populates cache and increments access_count
        self.kb.query_knowledge("quantum")
        item = self.kb.knowledge_items[kid]
        assert item.access_count >= 1
        # Second query may serve from cache, so access_count stays the same
        # Just verify the first query did track access
        first_count = item.access_count
        # Clear cache to force re-scan
        self.kb.query_cache.clear()
        self.kb.query_knowledge("quantum")
        assert item.access_count > first_count


# ---------------------------------------------------------------------------
# KnowledgeBase — Learning from Interaction
# ---------------------------------------------------------------------------

class TestKnowledgeBaseLearning:
    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_positive_feedback_increases_confidence(self):
        kid = self.kb.add_knowledge(
            "Test feedback item about learning",
            KnowledgeType.FACTUAL,
            source="test",
            confidence=0.5,
        )
        initial = self.kb.knowledge_items[kid].confidence
        self.kb.learn_from_interaction("learning", kid, feedback_positive=True)
        assert self.kb.knowledge_items[kid].confidence > initial

    def test_negative_feedback_decreases_confidence(self):
        kid = self.kb.add_knowledge(
            "Wrong item for negative feedback",
            KnowledgeType.FACTUAL,
            source="test",
            confidence=0.5,
        )
        initial = self.kb.knowledge_items[kid].confidence
        self.kb.learn_from_interaction("negative", kid, feedback_positive=False)
        assert self.kb.knowledge_items[kid].confidence < initial

    def test_confidence_bounded_at_1(self):
        kid = self.kb.add_knowledge("Max conf", KnowledgeType.FACTUAL, source="t", confidence=0.98)
        self.kb.learn_from_interaction("boost", kid, feedback_positive=True)
        assert self.kb.knowledge_items[kid].confidence <= 1.0

    def test_confidence_bounded_at_minimum(self):
        kid = self.kb.add_knowledge("Min conf", KnowledgeType.FACTUAL, source="t", confidence=0.15)
        self.kb.learn_from_interaction("drop", kid, feedback_positive=False)
        assert self.kb.knowledge_items[kid].confidence >= 0.1


# ---------------------------------------------------------------------------
# KnowledgeBase — Synthesis
# ---------------------------------------------------------------------------

class TestKnowledgeBaseSynthesis:
    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_synthesize_with_matching_content(self):
        self.kb.add_knowledge(
            "Python supports multiple paradigms",
            KnowledgeType.FACTUAL,
            source="docs",
        )
        self.kb.add_knowledge(
            "Python has dynamic typing",
            KnowledgeType.FACTUAL,
            source="wiki",
        )
        result = self.kb.synthesize_knowledge("Python")
        assert result["topic"] == "Python"
        assert result["total_items"] >= 1

    def test_synthesize_no_matches(self):
        result = self.kb.synthesize_knowledge("zzzyyyxxx")
        assert "No relevant knowledge found" in result["synthesis"]


# ---------------------------------------------------------------------------
# KnowledgeBase — Cache Behavior
# ---------------------------------------------------------------------------

class TestKnowledgeBaseCache:
    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_query_cache_populates(self):
        self.kb.add_knowledge("Cache test item about caching", KnowledgeType.FACTUAL, source="t")
        self.kb.query_knowledge("caching")
        assert len(self.kb.query_cache) >= 1

    def test_cache_invalidated_on_learn(self):
        kid = self.kb.add_knowledge(
            "Invalidation test about invalidate",
            KnowledgeType.FACTUAL,
            source="t",
        )
        self.kb.query_knowledge("invalidate")
        assert len(self.kb.query_cache) >= 1
        self.kb.learn_from_interaction("invalidate", kid, feedback_positive=True)
        # Cache entries containing "invalidate" should be cleared
        for key in self.kb.query_cache:
            assert "invalidate" not in key.lower()


# ---------------------------------------------------------------------------
# KnowledgeBase — Initialize with Core Knowledge
# ---------------------------------------------------------------------------

class TestKnowledgeBaseInitialize:
    def test_initialize_loads_core_knowledge(self):
        kb = KnowledgeBase()
        kb.initialize()
        assert kb.initialized is True
        assert len(kb.knowledge_items) > 30  # Should load dozens of core facts

    def test_double_initialize_is_safe(self):
        kb = KnowledgeBase()
        kb.initialize()
        count1 = len(kb.knowledge_items)
        kb.initialize()
        count2 = len(kb.knowledge_items)
        assert count1 == count2


# ---------------------------------------------------------------------------
# FactualMemoryEngine — Basic Operations
# ---------------------------------------------------------------------------

class TestFactualMemoryEngine:
    def setup_method(self):
        self.mgr = MemoryBlockManager()
        self.engine = FactualMemoryEngine(self.mgr)

    def test_store_fact(self):
        block_id = self.engine.store_fact("water_boils", "Water boils at 100C")
        assert block_id is not None
        block = self.mgr.retrieve_block(block_id)
        assert block is not None
        assert block.content == "Water boils at 100C"

    def test_store_fact_with_category(self):
        block_id = self.engine.store_fact("pi_value", "3.14159", category="math")
        block = self.mgr.retrieve_block(block_id)
        assert block.metadata["category"] == "math"

    def test_store_fact_updates_existing(self):
        self.engine.store_fact("update_me", "Version 1", category="test")
        self.engine.store_fact("update_me", "Version 2", category="test")
        block_id = f"fact_update_me_test"
        block = self.mgr.retrieve_block(block_id)
        assert block.content == "Version 2"

    def test_verify_fact_success(self):
        self.engine.store_fact("verifiable", "True statement", category="test")
        result = self.engine.verify_fact(
            "verifiable", "test", "external_source", verification_result=True
        )
        assert result is True

    def test_verify_nonexistent_fact(self):
        result = self.engine.verify_fact(
            "nonexistent", "general", "source", verification_result=True
        )
        assert result is False


# ---------------------------------------------------------------------------
# KnowledgeBase — Related Knowledge
# ---------------------------------------------------------------------------

class TestKnowledgeBaseRelated:
    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_get_related_empty_graph(self):
        kid = self.kb.add_knowledge("Solo item", KnowledgeType.FACTUAL, source="t")
        related = self.kb.get_related_knowledge(kid)
        assert related == []

    def test_get_related_nonexistent(self):
        related = self.kb.get_related_knowledge("nonexistent_id")
        assert related == []

    def test_get_related_with_connections(self):
        kid1 = self.kb.add_knowledge("Item A", KnowledgeType.FACTUAL, source="t")
        kid2 = self.kb.add_knowledge("Item B", KnowledgeType.FACTUAL, source="t")
        # Manually add relationship
        self.kb.knowledge_graph[kid1] = [kid2]
        related = self.kb.get_related_knowledge(kid1)
        assert len(related) == 1
        assert related[0].id == kid2
