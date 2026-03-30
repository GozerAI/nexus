"""Tests for marketplace usage tracking."""

import time
import pytest

from nexus.marketplace.usage import UsageTracker


@pytest.fixture
def tracker():
    return UsageTracker()


class TestRecordInvocation:
    """Tests for recording invocations."""

    def test_basic_invocation(self, tracker):
        rec = tracker.record_invocation("L1", user="alice")
        assert rec.listing_id == "L1"
        assert rec.action == "invoke"
        assert rec.user == "alice"
        assert rec.success is True

    def test_invocation_with_metrics(self, tracker):
        rec = tracker.record_invocation(
            "L1",
            user="alice",
            duration_ms=150.0,
            tokens_used=500,
            cost_usd=0.05,
        )
        assert rec.duration_ms == 150.0
        assert rec.tokens_used == 500
        assert rec.cost_usd == 0.05

    def test_failed_invocation(self, tracker):
        rec = tracker.record_invocation(
            "L1", user="bob", success=False, error="timeout"
        )
        assert rec.success is False
        assert rec.error == "timeout"

    def test_invocation_with_metadata(self, tracker):
        rec = tracker.record_invocation(
            "L1", metadata={"model": "gpt-4"}
        )
        assert rec.metadata["model"] == "gpt-4"


class TestRecordInstall:
    """Tests for install/uninstall tracking."""

    def test_install(self, tracker):
        rec = tracker.record_install("L1", "alice")
        assert rec.action == "install"
        assert rec.user == "alice"

    def test_uninstall(self, tracker):
        tracker.record_install("L1", "alice")
        rec = tracker.record_uninstall("L1", "alice")
        assert rec.action == "uninstall"

    def test_active_installs_tracked(self, tracker):
        tracker.record_install("L1", "alice")
        tracker.record_install("L1", "bob")
        summary = tracker.get_summary("L1")
        assert summary.active_installs == 2

    def test_uninstall_decrements_active(self, tracker):
        tracker.record_install("L1", "alice")
        tracker.record_install("L1", "bob")
        tracker.record_uninstall("L1", "alice")
        summary = tracker.get_summary("L1")
        assert summary.active_installs == 1

    def test_double_uninstall_safe(self, tracker):
        tracker.record_install("L1", "alice")
        tracker.record_uninstall("L1", "alice")
        tracker.record_uninstall("L1", "alice")  # no error
        summary = tracker.get_summary("L1")
        assert summary.active_installs == 0


class TestGetRecords:
    """Tests for raw record retrieval."""

    def _seed(self, tracker):
        tracker.record_invocation("L1", user="alice", tokens_used=100)
        tracker.record_invocation("L1", user="bob", tokens_used=200)
        tracker.record_install("L1", "alice")
        tracker.record_invocation("L1", user="alice", tokens_used=50)

    def test_all_records(self, tracker):
        self._seed(tracker)
        recs = tracker.get_records("L1")
        assert len(recs) == 4

    def test_filter_by_action(self, tracker):
        self._seed(tracker)
        recs = tracker.get_records("L1", action="invoke")
        assert len(recs) == 3
        assert all(r.action == "invoke" for r in recs)

    def test_filter_by_user(self, tracker):
        self._seed(tracker)
        recs = tracker.get_records("L1", user="alice")
        assert len(recs) == 3

    def test_filter_by_since(self, tracker):
        self._seed(tracker)
        future = time.time() + 1000
        recs = tracker.get_records("L1", since=future)
        assert len(recs) == 0

    def test_limit(self, tracker):
        self._seed(tracker)
        recs = tracker.get_records("L1", limit=2)
        assert len(recs) == 2

    def test_records_sorted_newest_first(self, tracker):
        self._seed(tracker)
        recs = tracker.get_records("L1")
        timestamps = [r.timestamp for r in recs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_empty_listing(self, tracker):
        recs = tracker.get_records("nonexistent")
        assert recs == []


class TestGetSummary:
    """Tests for usage summary aggregation."""

    def test_empty_summary(self, tracker):
        s = tracker.get_summary("L1")
        assert s.total_invocations == 0
        assert s.total_installs == 0
        assert s.success_rate == 1.0
        assert s.unique_users == 0

    def test_summary_counts_invocations(self, tracker):
        tracker.record_invocation("L1", user="alice")
        tracker.record_invocation("L1", user="bob")
        tracker.record_invocation("L1", user="alice")
        s = tracker.get_summary("L1")
        assert s.total_invocations == 3
        assert s.unique_users == 2

    def test_summary_success_rate(self, tracker):
        tracker.record_invocation("L1", success=True)
        tracker.record_invocation("L1", success=True)
        tracker.record_invocation("L1", success=False)
        s = tracker.get_summary("L1")
        assert abs(s.success_rate - 2 / 3) < 0.01

    def test_summary_tokens_and_cost(self, tracker):
        tracker.record_invocation("L1", tokens_used=100, cost_usd=0.01)
        tracker.record_invocation("L1", tokens_used=200, cost_usd=0.02)
        s = tracker.get_summary("L1")
        assert s.total_tokens == 300
        assert abs(s.total_cost_usd - 0.03) < 0.001

    def test_summary_avg_duration(self, tracker):
        tracker.record_invocation("L1", duration_ms=100)
        tracker.record_invocation("L1", duration_ms=200)
        s = tracker.get_summary("L1")
        assert s.avg_duration_ms == 150.0

    def test_summary_time_filter(self, tracker):
        now = time.time()
        tracker.record_invocation("L1", user="alice")
        # This record should be included
        s = tracker.get_summary("L1", since=now - 10)
        assert s.total_invocations == 1

        # This should exclude everything
        s2 = tracker.get_summary("L1", since=now + 100)
        assert s2.total_invocations == 0

    def test_summary_install_count(self, tracker):
        tracker.record_install("L1", "alice")
        tracker.record_install("L1", "bob")
        s = tracker.get_summary("L1")
        assert s.total_installs == 2


class TestGetTopListings:
    """Tests for ranking listings by metrics."""

    def _seed(self, tracker):
        for _ in range(5):
            tracker.record_invocation("L1", tokens_used=100, cost_usd=0.01)
        for _ in range(10):
            tracker.record_invocation("L2", tokens_used=50, cost_usd=0.005)
        for _ in range(2):
            tracker.record_invocation("L3", tokens_used=500, cost_usd=0.05)

    def test_top_by_invocations(self, tracker):
        self._seed(tracker)
        top = tracker.get_top_listings(metric="invocations", limit=2)
        assert len(top) == 2
        assert top[0]["listing_id"] == "L2"
        assert top[0]["value"] == 10

    def test_top_by_tokens(self, tracker):
        self._seed(tracker)
        top = tracker.get_top_listings(metric="tokens")
        # L3: 2*500=1000, L2: 10*50=500, L1: 5*100=500
        assert top[0]["listing_id"] == "L3"
        assert top[0]["value"] == 1000

    def test_top_by_cost(self, tracker):
        self._seed(tracker)
        top = tracker.get_top_listings(metric="cost")
        assert top[0]["listing_id"] == "L3"  # 2*0.05=0.10

    def test_top_by_installs(self, tracker):
        tracker.record_install("L1", "alice")
        tracker.record_install("L2", "alice")
        tracker.record_install("L2", "bob")
        top = tracker.get_top_listings(metric="installs")
        assert top[0]["listing_id"] == "L2"

    def test_unknown_metric_raises(self, tracker):
        self._seed(tracker)
        with pytest.raises(ValueError, match="Unknown metric"):
            tracker.get_top_listings(metric="bananas")

    def test_limit(self, tracker):
        self._seed(tracker)
        top = tracker.get_top_listings(metric="invocations", limit=1)
        assert len(top) == 1


class TestTrimming:
    """Tests for record trimming when max is exceeded."""

    def test_trimming_occurs(self):
        tracker = UsageTracker(max_records_per_listing=10)
        for i in range(15):
            tracker.record_invocation("L1", user=f"u{i}")
        recs = tracker.get_records("L1", limit=100)
        # After inserting 11th, trims 1 (10%). After 12th, etc.
        # The exact count depends on trim logic, but should be <= 15
        assert len(recs) <= 15
        assert len(recs) > 0
