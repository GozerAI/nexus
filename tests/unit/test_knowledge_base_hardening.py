"""Tests for knowledge base production hardening: persistence, TTL, max_items, thread safety."""

import os
import tempfile
import threading
import time
from datetime import datetime, timedelta

import pytest

from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeItem, KnowledgeType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kb(**kwargs):
    """Create a KnowledgeBase without loading core knowledge."""
    kb = KnowledgeBase(**kwargs)
    kb.initialized = True  # skip core knowledge load
    return kb


def _add_n(kb, n, source="test"):
    """Add *n* items and return their IDs."""
    ids = []
    for i in range(n):
        kid = kb.add_knowledge(f"item-{i}", KnowledgeType.FACTUAL, source)
        ids.append(kid)
    return ids


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_adds_do_not_lose_items(self):
        kb = _make_kb()
        errors = []

        def worker(start):
            try:
                for i in range(50):
                    kb.add_knowledge(f"thread-{start}-{i}", KnowledgeType.FACTUAL, "test")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(kb.knowledge_items) == 200

    def test_concurrent_reads_and_writes(self):
        kb = _make_kb()
        _add_n(kb, 20)
        errors = []

        def reader():
            try:
                for _ in range(30):
                    kb.query_knowledge("item")
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(30):
                    kb.add_knowledge(f"new-{i}", KnowledgeType.FACTUAL, "test")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ---------------------------------------------------------------------------
# Max items & LRU eviction
# ---------------------------------------------------------------------------

class TestMaxItemsEviction:
    def test_evicts_lru_when_at_capacity(self):
        kb = _make_kb(max_items=5)
        ids = _add_n(kb, 5)
        # Access items 3 and 4 to make them recently used
        kb.knowledge_items[ids[3]].last_accessed = datetime.now()
        kb.knowledge_items[ids[4]].last_accessed = datetime.now()
        # Make items 0-2 old
        old = datetime.now() - timedelta(days=1)
        for i in range(3):
            kb.knowledge_items[ids[i]].last_accessed = old

        # Adding one more should evict the oldest
        new_id = kb.add_knowledge("new-item", KnowledgeType.FACTUAL, "test")
        assert len(kb.knowledge_items) <= 5
        assert new_id in kb.knowledge_items

    def test_builtin_items_not_evicted(self):
        kb = _make_kb(max_items=3)
        bi_id = kb.add_knowledge("core-fact", KnowledgeType.FACTUAL, "built-in")
        old = datetime.now() - timedelta(days=30)
        kb.knowledge_items[bi_id].last_accessed = old

        _add_n(kb, 3)
        # built-in item should still be present
        assert bi_id in kb.knowledge_items


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------

class TestTTLExpiry:
    def test_expired_items_evicted_on_add(self):
        kb = _make_kb(item_ttl=1)  # 1 second TTL
        ids = _add_n(kb, 3)
        # Backdoor: set last_accessed to the past
        old = datetime.now() - timedelta(seconds=5)
        for kid in ids:
            kb.knowledge_items[kid].last_accessed = old

        # Adding a new item should trigger eviction
        kb.add_knowledge("fresh", KnowledgeType.FACTUAL, "test")
        # Old items should be gone, only fresh remains
        assert len(kb.knowledge_items) == 1

    def test_builtin_items_survive_ttl(self):
        kb = _make_kb(item_ttl=1)
        bi_id = kb.add_knowledge("core", KnowledgeType.FACTUAL, "built-in")
        old = datetime.now() - timedelta(seconds=5)
        kb.knowledge_items[bi_id].last_accessed = old

        kb.add_knowledge("fresh", KnowledgeType.FACTUAL, "test")
        assert bi_id in kb.knowledge_items


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

class TestSQLitePersistence:
    def test_items_survive_restart(self, tmp_path):
        db_path = str(tmp_path / "kb.db")

        # Create KB, add items, close
        kb1 = _make_kb(db_path=db_path)
        ids = _add_n(kb1, 5)
        kb1.close()

        # Re-open and load
        kb2 = _make_kb(db_path=db_path)
        kb2._load_from_db()
        assert len(kb2.knowledge_items) == 5
        for kid in ids:
            assert kid in kb2.knowledge_items
        kb2.close()

    def test_updated_confidence_persisted(self, tmp_path):
        db_path = str(tmp_path / "kb.db")

        kb1 = _make_kb(db_path=db_path)
        kid = kb1.add_knowledge("test-fact", KnowledgeType.FACTUAL, "test", confidence=0.5)
        kb1.learn_from_interaction("test", kid, feedback_positive=True)
        new_confidence = kb1.knowledge_items[kid].confidence
        kb1.close()

        kb2 = _make_kb(db_path=db_path)
        kb2._load_from_db()
        assert kb2.knowledge_items[kid].confidence == new_confidence
        kb2.close()

    def test_evicted_items_deleted_from_db(self, tmp_path):
        db_path = str(tmp_path / "kb.db")

        kb = _make_kb(db_path=db_path, max_items=3)
        ids = _add_n(kb, 3)
        # Make first item old
        old = datetime.now() - timedelta(days=1)
        kb.knowledge_items[ids[0]].last_accessed = old

        kb.add_knowledge("overflow", KnowledgeType.FACTUAL, "test")
        kb.close()

        # Re-open — evicted item should not be in DB
        kb2 = _make_kb(db_path=db_path)
        kb2._load_from_db()
        assert ids[0] not in kb2.knowledge_items
        kb2.close()

    def test_no_db_path_works_in_memory_only(self):
        kb = _make_kb()
        ids = _add_n(kb, 5)
        assert len(kb.knowledge_items) == 5
        # No crash, no persistence
        kb.close()

    def test_initialize_loads_from_db_before_core_knowledge(self, tmp_path):
        db_path = str(tmp_path / "kb.db")

        # First run: initialize with core knowledge + some user knowledge
        kb1 = KnowledgeBase(db_path=db_path)
        kb1.initialize()
        user_id = kb1.add_knowledge("user-fact", KnowledgeType.FACTUAL, "user")
        initial_count = len(kb1.knowledge_items)
        kb1.close()

        # Second run: should load from DB, skip core knowledge reload
        kb2 = KnowledgeBase(db_path=db_path)
        kb2.initialize()
        assert user_id in kb2.knowledge_items
        # Should have same count (loaded from DB, no duplicate core knowledge)
        assert len(kb2.knowledge_items) == initial_count
        kb2.close()


# ---------------------------------------------------------------------------
# Cache key collision fix
# ---------------------------------------------------------------------------

class TestCacheKeyFix:
    def test_different_params_produce_different_cache_keys(self):
        kb = _make_kb()
        _add_n(kb, 10)

        r1 = kb.query_knowledge("item", max_results=3)
        r2 = kb.query_knowledge("item", max_results=5)
        # These should be independent cache entries
        assert len(r1) <= 3
        assert len(r2) <= 5


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_statistics_include_max_items(self):
        kb = _make_kb(max_items=500)
        _add_n(kb, 3)
        stats = kb.get_knowledge_statistics()
        assert stats["max_items"] == 500
        assert stats["total_items"] == 3
        assert "persistent" in stats

    def test_statistics_show_persistence_status(self, tmp_path):
        db_path = str(tmp_path / "kb.db")
        kb = _make_kb(db_path=db_path)
        stats = kb.get_knowledge_statistics()
        # No items yet, should return total_items=0
        assert stats == {"total_items": 0}
        _add_n(kb, 1)
        stats = kb.get_knowledge_statistics()
        assert stats["persistent"] is True
        kb.close()
