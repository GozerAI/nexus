"""Tests for DynamicLearner SQLite persistence.

Covers:
- Events are persisted to SQLite when db_path is set
- Rules are persisted to SQLite
- Data is loaded on startup from existing db
"""

import os
import tempfile

import pytest

from nexus.reasoning.dynamic_learner import AdaptationRule, DynamicLearner, LearningEvent


class TestDynamicLearnerPersistence:
    """DynamicLearner persists events and rules to SQLite."""

    @pytest.fixture()
    def db_path(self, tmp_path):
        return str(tmp_path / "learner.db")

    def test_events_persisted_to_sqlite(self, db_path):
        learner = DynamicLearner(db_path=db_path)
        event_id = learner.record_learning_event(
            event_type="test_type",
            input_data="input",
            expected_output="expected",
            actual_output="expected",
            context={"key": "value"},
        )

        # Verify it was written to SQLite
        import sqlite3

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT event_id, event_type FROM learning_events WHERE event_id = ?",
            (event_id,),
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == event_id
        assert rows[0][1] == "test_type"

    def test_rules_persisted_to_sqlite(self, db_path):
        learner = DynamicLearner(db_path=db_path)

        # Base rules should be persisted on init
        import sqlite3

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT rule_id FROM adaptation_rules").fetchall()
        conn.close()

        assert len(rows) >= 3  # 3 base rules
        rule_ids = {r[0] for r in rows}
        assert "increase_confidence_threshold" in rule_ids
        assert "adjust_reasoning_depth" in rule_ids
        assert "enhance_memory_integration" in rule_ids

    def test_custom_rule_persisted(self, db_path):
        learner = DynamicLearner(db_path=db_path)

        rule = AdaptationRule(
            rule_id="custom_rule",
            condition="test_condition",
            action="test_action",
            priority=5,
            success_rate=0.8,
            usage_count=3,
            last_applied=1000.0,
        )
        learner.adaptation_rules["custom_rule"] = rule
        learner._persist_rule(rule)

        import sqlite3

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT rule_id, condition, action, priority, success_rate "
            "FROM adaptation_rules WHERE rule_id = 'custom_rule'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0] == ("custom_rule", "test_condition", "test_action", 5, 0.8)

    def test_data_loaded_on_startup(self, db_path):
        # Create learner, add events, close it
        learner1 = DynamicLearner(db_path=db_path)
        learner1.record_learning_event(
            event_type="type_a",
            input_data="in1",
            expected_output="out1",
            actual_output="out1",
        )
        learner1.record_learning_event(
            event_type="type_b",
            input_data="in2",
            expected_output="out2",
            actual_output="out2",
        )

        # Add a custom rule
        rule = AdaptationRule(
            rule_id="persisted_rule",
            condition="some_cond",
            action="some_action",
            priority=1,
            success_rate=0.5,
            usage_count=2,
            last_applied=500.0,
        )
        learner1.adaptation_rules["persisted_rule"] = rule
        learner1._persist_rule(rule)

        # Create new learner from same db
        learner2 = DynamicLearner(db_path=db_path)

        # Events should be loaded
        assert len(learner2.learning_events) >= 2

        # Custom rule should be loaded
        assert "persisted_rule" in learner2.adaptation_rules
        loaded_rule = learner2.adaptation_rules["persisted_rule"]
        assert loaded_rule.condition == "some_cond"
        assert loaded_rule.action == "some_action"
        assert loaded_rule.success_rate == 0.5

        # Performance history should be reconstructed
        assert "type_a" in learner2.performance_history
        assert "type_b" in learner2.performance_history

    def test_no_db_path_skips_persistence(self):
        learner = DynamicLearner()  # no db_path
        assert learner._db is None

        # Should not raise even when recording events
        event_id = learner.record_learning_event(
            event_type="test",
            input_data="in",
            expected_output="out",
            actual_output="out",
        )
        assert event_id  # Still returns an event_id

    def test_base_rules_not_duplicated_on_reload(self, db_path):
        # First init seeds base rules
        learner1 = DynamicLearner(db_path=db_path)
        initial_count = len(learner1.adaptation_rules)

        # Second init loads from db (rules exist, so base rules not re-seeded)
        learner2 = DynamicLearner(db_path=db_path)
        assert len(learner2.adaptation_rules) == initial_count
