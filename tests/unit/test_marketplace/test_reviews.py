"""Tests for the marketplace review system."""

import pytest

from nexus.marketplace.reviews import (
    MIN_REVIEW_BODY_LENGTH,
    MAX_REVIEW_BODY_LENGTH,
    RatingSnapshot,
    ReviewManager,
)


@pytest.fixture
def manager():
    return ReviewManager()


class TestReviewSubmission:
    """Tests for submitting reviews."""

    def test_submit_valid_review(self, manager):
        r = manager.submit_review("listing-1", "alice", 5, title="Great!")
        assert r.listing_id == "listing-1"
        assert r.reviewer == "alice"
        assert r.rating == 5
        assert r.title == "Great!"

    def test_submit_with_body(self, manager):
        r = manager.submit_review(
            "listing-1", "alice", 4, body="This is a decent review body."
        )
        assert r.body == "This is a decent review body."

    def test_submit_verified(self, manager):
        r = manager.submit_review("listing-1", "alice", 5, verified=True)
        assert r.verified is True

    def test_duplicate_reviewer_rejected(self, manager):
        manager.submit_review("listing-1", "alice", 5)
        with pytest.raises(ValueError, match="already reviewed"):
            manager.submit_review("listing-1", "alice", 3)

    def test_same_reviewer_different_listing_ok(self, manager):
        manager.submit_review("listing-1", "alice", 5)
        r = manager.submit_review("listing-2", "alice", 3)
        assert r.listing_id == "listing-2"

    def test_empty_listing_id_rejected(self, manager):
        with pytest.raises(ValueError, match="listing_id"):
            manager.submit_review("", "alice", 5)

    def test_empty_reviewer_rejected(self, manager):
        with pytest.raises(ValueError, match="reviewer"):
            manager.submit_review("listing-1", "", 5)

    def test_rating_below_1_rejected(self, manager):
        with pytest.raises(ValueError, match="rating must be 1-5"):
            manager.submit_review("listing-1", "alice", 0)

    def test_rating_above_5_rejected(self, manager):
        with pytest.raises(ValueError, match="rating must be 1-5"):
            manager.submit_review("listing-1", "alice", 6)

    def test_body_too_short_rejected(self, manager):
        with pytest.raises(ValueError, match="at least"):
            manager.submit_review("listing-1", "alice", 5, body="short")

    def test_body_too_long_rejected(self, manager):
        with pytest.raises(ValueError, match="at most"):
            manager.submit_review(
                "listing-1", "alice", 5, body="x" * (MAX_REVIEW_BODY_LENGTH + 1)
            )

    def test_empty_body_allowed(self, manager):
        """Empty body is fine — body is optional."""
        r = manager.submit_review("listing-1", "alice", 5, body="")
        assert r.body == ""


class TestReviewUpdate:
    """Tests for updating reviews."""

    def test_update_rating(self, manager):
        r = manager.submit_review("listing-1", "alice", 3)
        updated = manager.update_review(r.id, rating=5)
        assert updated.rating == 5

    def test_update_title(self, manager):
        r = manager.submit_review("listing-1", "alice", 3, title="OK")
        updated = manager.update_review(r.id, title="Better title")
        assert updated.title == "Better title"

    def test_update_body(self, manager):
        r = manager.submit_review("listing-1", "alice", 3)
        updated = manager.update_review(r.id, body="A much better body for this review")
        assert updated.body == "A much better body for this review"

    def test_update_invalid_rating(self, manager):
        r = manager.submit_review("listing-1", "alice", 3)
        with pytest.raises(ValueError, match="rating must be 1-5"):
            manager.update_review(r.id, rating=0)

    def test_update_short_body(self, manager):
        r = manager.submit_review("listing-1", "alice", 3)
        with pytest.raises(ValueError, match="at least"):
            manager.update_review(r.id, body="tiny")

    def test_update_nonexistent_review(self, manager):
        with pytest.raises(KeyError):
            manager.update_review("nope", rating=5)

    def test_update_changes_updated_at(self, manager):
        r = manager.submit_review("listing-1", "alice", 3)
        old_ts = r.updated_at
        updated = manager.update_review(r.id, rating=4)
        assert updated.updated_at >= old_ts


class TestReviewDeletion:
    """Tests for deleting reviews."""

    def test_delete_existing(self, manager):
        r = manager.submit_review("listing-1", "alice", 5)
        manager.delete_review(r.id)
        with pytest.raises(KeyError):
            manager.get_review_by_id(r.id)

    def test_delete_nonexistent(self, manager):
        with pytest.raises(KeyError):
            manager.delete_review("nope")

    def test_delete_invalidates_snapshot(self, manager):
        r = manager.submit_review("listing-1", "alice", 5)
        snap1 = manager.get_rating_snapshot("listing-1")
        assert snap1.total_reviews == 1
        manager.delete_review(r.id)
        snap2 = manager.get_rating_snapshot("listing-1")
        assert snap2.total_reviews == 0


class TestMarkHelpful:
    """Tests for the helpful counter."""

    def test_mark_helpful(self, manager):
        r = manager.submit_review("listing-1", "alice", 5)
        assert r.helpful_count == 0
        updated = manager.mark_helpful(r.id)
        assert updated.helpful_count == 1

    def test_mark_helpful_multiple(self, manager):
        r = manager.submit_review("listing-1", "alice", 5)
        manager.mark_helpful(r.id)
        manager.mark_helpful(r.id)
        assert r.helpful_count == 2

    def test_mark_helpful_nonexistent(self, manager):
        with pytest.raises(KeyError):
            manager.mark_helpful("nope")


class TestReviewQueries:
    """Tests for retrieving and filtering reviews."""

    def _seed(self, manager):
        manager.submit_review("L1", "alice", 5, verified=True)
        manager.submit_review("L1", "bob", 3)
        manager.submit_review("L1", "carol", 4, verified=True)
        manager.submit_review("L1", "dave", 1)

    def test_get_reviews_default(self, manager):
        self._seed(manager)
        reviews = manager.get_reviews("L1")
        assert len(reviews) == 4

    def test_get_reviews_min_rating(self, manager):
        self._seed(manager)
        reviews = manager.get_reviews("L1", min_rating=4)
        assert all(r.rating >= 4 for r in reviews)
        assert len(reviews) == 2

    def test_get_reviews_verified_only(self, manager):
        self._seed(manager)
        reviews = manager.get_reviews("L1", verified_only=True)
        assert all(r.verified for r in reviews)
        assert len(reviews) == 2

    def test_sort_highest(self, manager):
        self._seed(manager)
        reviews = manager.get_reviews("L1", sort_by="highest")
        ratings = [r.rating for r in reviews]
        assert ratings == sorted(ratings, reverse=True)

    def test_sort_lowest(self, manager):
        self._seed(manager)
        reviews = manager.get_reviews("L1", sort_by="lowest")
        ratings = [r.rating for r in reviews]
        assert ratings == sorted(ratings)

    def test_sort_helpful(self, manager):
        self._seed(manager)
        reviews = manager.get_reviews("L1")
        manager.mark_helpful(reviews[2].id)
        manager.mark_helpful(reviews[2].id)
        helpful = manager.get_reviews("L1", sort_by="helpful")
        assert helpful[0].helpful_count >= helpful[-1].helpful_count

    def test_pagination_limit(self, manager):
        self._seed(manager)
        reviews = manager.get_reviews("L1", limit=2)
        assert len(reviews) == 2

    def test_pagination_offset(self, manager):
        self._seed(manager)
        all_reviews = manager.get_reviews("L1")
        offset_reviews = manager.get_reviews("L1", offset=2)
        assert len(offset_reviews) == 2
        assert offset_reviews[0].id == all_reviews[2].id

    def test_empty_listing(self, manager):
        reviews = manager.get_reviews("nonexistent")
        assert reviews == []


class TestRatingSnapshot:
    """Tests for the rating snapshot."""

    def test_snapshot_empty_listing(self, manager):
        snap = manager.get_rating_snapshot("empty")
        assert snap.average_rating == 0.0
        assert snap.total_reviews == 0
        assert snap.rating_distribution == {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    def test_snapshot_after_reviews(self, manager):
        manager.submit_review("L1", "alice", 5)
        manager.submit_review("L1", "bob", 3)
        snap = manager.get_rating_snapshot("L1")
        assert snap.total_reviews == 2
        assert snap.average_rating == 4.0
        assert snap.rating_distribution[5] == 1
        assert snap.rating_distribution[3] == 1

    def test_snapshot_caching(self, manager):
        manager.submit_review("L1", "alice", 5)
        snap1 = manager.get_rating_snapshot("L1")
        snap2 = manager.get_rating_snapshot("L1")
        assert snap1 is snap2  # Same cached object

    def test_snapshot_invalidated_after_new_review(self, manager):
        manager.submit_review("L1", "alice", 5)
        snap1 = manager.get_rating_snapshot("L1")
        assert snap1.total_reviews == 1
        manager.submit_review("L1", "bob", 3)
        snap2 = manager.get_rating_snapshot("L1")
        assert snap2.total_reviews == 2
        assert snap1 is not snap2

    def test_snapshot_to_dict(self, manager):
        manager.submit_review("L1", "alice", 5)
        snap = manager.get_rating_snapshot("L1")
        d = snap.to_dict()
        assert d["listing_id"] == "L1"
        assert d["average_rating"] == 5.0
        assert d["total_reviews"] == 1

    def test_snapshot_rounding(self, manager):
        manager.submit_review("L1", "alice", 5)
        manager.submit_review("L1", "bob", 4)
        manager.submit_review("L1", "carol", 4)
        snap = manager.get_rating_snapshot("L1")
        d = snap.to_dict()
        # (5+4+4)/3 = 4.333...
        assert d["average_rating"] == 4.33
