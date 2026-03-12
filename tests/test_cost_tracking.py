"""Tests for cost tracking system."""

import os
import sys
import pytest
import json
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from nexus.core.tracking.cost_tracker import CostTracker, CostEntry, CostSummary


class TestCostEntry:
    """Tests for CostEntry dataclass."""

    def test_cost_entry_creation(self):
        """Test creating a cost entry."""
        entry = CostEntry(
            timestamp=datetime.now(timezone.utc),
            model_name="gpt-4",
            provider="openai",
            tokens_used=1000,
            cost_usd=0.03,
            user_id="user_123",
            request_id="req_456"
        )

        assert entry.model_name == "gpt-4"
        assert entry.provider == "openai"
        assert entry.tokens_used == 1000
        assert entry.cost_usd == 0.03
        assert entry.user_id == "user_123"
        assert entry.request_id == "req_456"

    def test_cost_entry_optional_fields(self):
        """Test cost entry with optional fields."""
        entry = CostEntry(
            timestamp=datetime.now(timezone.utc),
            model_name="claude-3-opus",
            provider="anthropic",
            tokens_used=500,
            cost_usd=0.015
        )

        assert entry.user_id is None
        assert entry.request_id is None


class TestCostSummary:
    """Tests for CostSummary dataclass."""

    def test_cost_summary_creation(self):
        """Test creating a cost summary."""
        summary = CostSummary(
            total_cost=10.5,
            total_requests=100,
            total_tokens=50000,
            cost_by_model={"gpt-4": 8.0, "claude-3": 2.5},
            cost_by_provider={"openai": 8.0, "anthropic": 2.5},
            cost_by_user={"user_1": 6.0, "user_2": 4.5}
        )

        assert summary.total_cost == 10.5
        assert summary.total_requests == 100
        assert summary.total_tokens == 50000
        assert summary.cost_by_model["gpt-4"] == 8.0

    def test_cost_summary_defaults(self):
        """Test cost summary default values."""
        summary = CostSummary()

        assert summary.total_cost == 0.0
        assert summary.total_requests == 0
        assert summary.total_tokens == 0
        assert summary.cost_by_model == {}
        assert summary.cost_by_provider == {}
        assert summary.cost_by_user == {}


class TestCostTracker:
    """Tests for CostTracker."""

    def test_tracker_initialization(self):
        """Test tracker initialization."""
        tracker = CostTracker(budget_limit_usd=100.0, alert_threshold=0.8)

        assert tracker.budget_limit == 100.0
        assert tracker.alert_threshold == 0.8
        assert len(tracker.entries) == 0
        assert len(tracker.alerts_sent) == 0

    def test_record_cost(self):
        """Test recording a cost entry."""
        tracker = CostTracker()

        tracker.record_cost(
            model_name="gpt-4",
            provider="openai",
            tokens_used=1000,
            cost_usd=0.03,
            user_id="user_123"
        )

        assert len(tracker.entries) == 1
        entry = tracker.entries[0]
        assert entry.model_name == "gpt-4"
        assert entry.provider == "openai"
        assert entry.tokens_used == 1000
        assert entry.cost_usd == 0.03
        assert entry.user_id == "user_123"

    def test_record_multiple_costs(self):
        """Test recording multiple cost entries."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 0.03, "user_1")
        tracker.record_cost("claude-3-opus", "anthropic", 500, 0.015, "user_2")
        tracker.record_cost("gpt-3.5-turbo", "openai", 2000, 0.002, "user_1")

        assert len(tracker.entries) == 3

    def test_get_summary_all_data(self):
        """Test getting summary of all data."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 0.03, "user_1")
        tracker.record_cost("claude-3-opus", "anthropic", 500, 0.015, "user_2")
        tracker.record_cost("gpt-4", "openai", 1000, 0.03, "user_1")

        now = datetime.now(timezone.utc)
        start_date = datetime(now.year, now.month, 1)
        summary = tracker.get_summary(start_date=start_date)

        assert summary.total_cost == 0.075
        assert summary.total_requests == 3
        assert summary.total_tokens == 2500
        assert summary.cost_by_model["gpt-4"] == 0.06
        assert summary.cost_by_model["claude-3-opus"] == 0.015
        assert summary.cost_by_provider["openai"] == 0.06
        assert summary.cost_by_provider["anthropic"] == 0.015
        assert summary.cost_by_user["user_1"] == 0.06
        assert summary.cost_by_user["user_2"] == 0.015

    def test_get_summary_with_date_filter(self):
        """Test getting summary with date filtering."""
        tracker = CostTracker()

        # Add entries with different timestamps
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)

        # Manually create entries with specific timestamps
        entry1 = CostEntry(
            timestamp=yesterday,
            model_name="gpt-4",
            provider="openai",
            tokens_used=1000,
            cost_usd=0.03,
            user_id="user_1"
        )

        entry2 = CostEntry(
            timestamp=now,
            model_name="claude-3-opus",
            provider="anthropic",
            tokens_used=500,
            cost_usd=0.015,
            user_id="user_2"
        )

        tracker.entries.append(entry1)
        tracker.entries.append(entry2)

        # Get summary for today only
        start_of_today = datetime(now.year, now.month, now.day)
        summary = tracker.get_summary(start_date=start_of_today)

        assert summary.total_cost == 0.015
        assert summary.total_requests == 1

    def test_get_summary_with_user_filter(self):
        """Test getting summary filtered by user."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 0.03, "user_1")
        tracker.record_cost("claude-3-opus", "anthropic", 500, 0.015, "user_2")
        tracker.record_cost("gpt-4", "openai", 1000, 0.03, "user_1")

        now = datetime.now(timezone.utc)
        start_date = datetime(now.year, now.month, 1)
        summary = tracker.get_summary(start_date=start_date, user_id="user_1")

        assert summary.total_cost == 0.06
        assert summary.total_requests == 2
        assert "user_2" not in summary.cost_by_user

    def test_get_monthly_cost(self):
        """Test getting monthly cost."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 0.03)
        tracker.record_cost("claude-3-opus", "anthropic", 500, 0.015)

        monthly_cost = tracker.get_monthly_cost()
        assert monthly_cost == 0.045

    def test_get_daily_cost(self):
        """Test getting daily cost."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 0.03)
        tracker.record_cost("claude-3-opus", "anthropic", 500, 0.015)

        daily_cost = tracker.get_daily_cost()
        assert daily_cost == 0.045

    def test_is_over_budget_false(self):
        """Test budget check when under budget."""
        tracker = CostTracker(budget_limit_usd=10.0)

        tracker.record_cost("gpt-4", "openai", 1000, 0.03)

        assert not tracker.is_over_budget()

    def test_is_over_budget_true(self):
        """Test budget check when over budget."""
        tracker = CostTracker(budget_limit_usd=0.02)

        tracker.record_cost("gpt-4", "openai", 1000, 0.03)

        assert tracker.is_over_budget()

    def test_get_budget_status(self):
        """Test getting budget status."""
        tracker = CostTracker(budget_limit_usd=10.0, alert_threshold=0.8)

        tracker.record_cost("gpt-4", "openai", 1000, 3.0)
        tracker.record_cost("claude-3-opus", "anthropic", 500, 2.0)

        status = tracker.get_budget_status()

        assert status["budget_limit"] == 10.0
        assert status["current_spend"] == 5.0
        assert status["remaining"] == 5.0
        assert status["percent_used"] == 50.0
        assert status["is_over_budget"] is False
        assert status["alert_threshold"] == 80.0

    def test_get_budget_status_over_budget(self):
        """Test budget status when over budget."""
        tracker = CostTracker(budget_limit_usd=5.0)

        tracker.record_cost("gpt-4", "openai", 1000, 6.0)

        status = tracker.get_budget_status()

        assert status["is_over_budget"] is True
        assert status["remaining"] == 0.0
        assert status["percent_used"] == 120.0

    def test_budget_alert_triggered(self, caplog):
        """Test that budget alert is triggered."""
        import logging

        tracker = CostTracker(budget_limit_usd=10.0, alert_threshold=0.8)

        # Spend just under threshold - no alert
        tracker.record_cost("gpt-4", "openai", 1000, 7.5)
        assert len(tracker.alerts_sent) == 0

        # Spend over threshold - alert should trigger
        with caplog.at_level(logging.WARNING):
            tracker.record_cost("gpt-4", "openai", 1000, 1.0)

        assert len(tracker.alerts_sent) == 1
        assert "BUDGET ALERT" in caplog.text

    def test_budget_alert_only_once_per_month(self, caplog):
        """Test that budget alert is only sent once per month."""
        import logging

        tracker = CostTracker(budget_limit_usd=10.0, alert_threshold=0.8)

        # First alert
        with caplog.at_level(logging.WARNING):
            tracker.record_cost("gpt-4", "openai", 1000, 9.0)

        alert_count = caplog.text.count("BUDGET ALERT")
        assert alert_count == 1

        # Second spend - no new alert
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            tracker.record_cost("gpt-4", "openai", 1000, 0.5)

        assert "BUDGET ALERT" not in caplog.text

    def test_get_top_costs(self):
        """Test getting top costs by model."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 5.0)
        tracker.record_cost("claude-3-opus", "anthropic", 500, 3.0)
        tracker.record_cost("gpt-3.5-turbo", "openai", 2000, 0.5)
        tracker.record_cost("gpt-4", "openai", 1000, 5.0)

        top_costs = tracker.get_top_costs(limit=3)

        assert len(top_costs) == 3
        assert top_costs[0] == ("gpt-4", 10.0)
        assert top_costs[1] == ("claude-3-opus", 3.0)
        assert top_costs[2] == ("gpt-3.5-turbo", 0.5)

    def test_get_top_costs_with_limit(self):
        """Test getting top costs with limit."""
        tracker = CostTracker()

        tracker.record_cost("model1", "provider1", 100, 5.0)
        tracker.record_cost("model2", "provider1", 100, 3.0)
        tracker.record_cost("model3", "provider1", 100, 1.0)

        top_costs = tracker.get_top_costs(limit=2)

        assert len(top_costs) == 2
        assert top_costs[0][0] == "model1"
        assert top_costs[1][0] == "model2"

    def test_export_data_json(self, tmp_path):
        """Test exporting data to JSON."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 0.03, "user_1", "req_1")
        tracker.record_cost("claude-3-opus", "anthropic", 500, 0.015, "user_2", "req_2")

        output_file = tmp_path / "costs.json"
        tracker.export_data(str(output_file), format="json")

        assert output_file.exists()

        with open(output_file, 'r') as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["model_name"] == "gpt-4"
        assert data[0]["provider"] == "openai"
        assert data[0]["tokens_used"] == 1000
        assert data[0]["cost_usd"] == 0.03
        assert data[0]["user_id"] == "user_1"
        assert data[0]["request_id"] == "req_1"

    def test_export_data_csv(self, tmp_path):
        """Test exporting data to CSV."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 0.03, "user_1", "req_1")
        tracker.record_cost("claude-3-opus", "anthropic", 500, 0.015, "user_2", "req_2")

        output_file = tmp_path / "costs.csv"
        tracker.export_data(str(output_file), format="csv")

        assert output_file.exists()

        with open(output_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 3  # Header + 2 data rows
        assert rows[0] == ["timestamp", "model_name", "provider", "tokens_used", "cost_usd", "user_id", "request_id"]
        assert rows[1][1] == "gpt-4"
        assert rows[1][2] == "openai"
        assert rows[1][3] == "1000"
        assert rows[1][4] == "0.03"
        assert rows[1][5] == "user_1"
        assert rows[1][6] == "req_1"

    def test_export_data_csv_with_null_values(self, tmp_path):
        """Test exporting CSV with null optional values."""
        tracker = CostTracker()

        tracker.record_cost("gpt-4", "openai", 1000, 0.03)

        output_file = tmp_path / "costs.csv"
        tracker.export_data(str(output_file), format="csv")

        with open(output_file, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # user_id and request_id should be empty strings
        assert rows[1][5] == ""
        assert rows[1][6] == ""

    def test_empty_tracker_summary(self):
        """Test getting summary from empty tracker."""
        tracker = CostTracker()

        summary = tracker.get_summary()

        assert summary.total_cost == 0.0
        assert summary.total_requests == 0
        assert summary.total_tokens == 0

    def test_zero_budget_handling(self):
        """Test handling of zero budget."""
        tracker = CostTracker(budget_limit_usd=0.0)

        tracker.record_cost("gpt-4", "openai", 1000, 0.03)

        status = tracker.get_budget_status()

        assert status["budget_limit"] == 0.0
        assert status["percent_used"] == 0  # Should handle division by zero
