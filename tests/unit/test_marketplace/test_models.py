"""Tests for marketplace data models."""

import pytest
import time

from nexus.marketplace.models import (
    ListingStatus,
    MarketplaceCategory,
    MarketplaceListing,
    Review,
    UsageRecord,
    UsageSummary,
)


class TestMarketplaceCategory:
    """Tests for MarketplaceCategory enum."""

    def test_provider_categories_exist(self):
        assert MarketplaceCategory.LLM == "llm"
        assert MarketplaceCategory.EMBEDDING == "embedding"
        assert MarketplaceCategory.IMAGE == "image"
        assert MarketplaceCategory.AUDIO == "audio"
        assert MarketplaceCategory.MULTIMODAL == "multimodal"
        assert MarketplaceCategory.CUSTOM == "custom"

    def test_agent_categories_exist(self):
        assert MarketplaceCategory.DATA_PROCESSING == "data_processing"
        assert MarketplaceCategory.CODE_GENERATION == "code_generation"
        assert MarketplaceCategory.ANALYSIS == "analysis"
        assert MarketplaceCategory.AUTOMATION == "automation"
        assert MarketplaceCategory.RESEARCH == "research"
        assert MarketplaceCategory.MONITORING == "monitoring"


class TestListingStatus:
    """Tests for ListingStatus enum."""

    def test_lifecycle_statuses(self):
        assert ListingStatus.DRAFT == "draft"
        assert ListingStatus.PENDING_REVIEW == "pending_review"
        assert ListingStatus.PUBLISHED == "published"
        assert ListingStatus.SUSPENDED == "suspended"
        assert ListingStatus.DEPRECATED == "deprecated"
        assert ListingStatus.REMOVED == "removed"


class TestMarketplaceListing:
    """Tests for MarketplaceListing dataclass."""

    def test_default_creation(self):
        listing = MarketplaceListing()
        assert listing.id  # auto-generated
        assert listing.name == ""
        assert listing.status == ListingStatus.DRAFT
        assert listing.category == MarketplaceCategory.CUSTOM
        assert listing.tags == []
        assert listing.metadata == {}

    def test_creation_with_fields(self):
        listing = MarketplaceListing(
            name="test-provider",
            display_name="Test Provider",
            description="A test provider",
            version="2.0.0",
            author="tester",
            category=MarketplaceCategory.LLM,
            tags=["llm", "fast"],
        )
        assert listing.name == "test-provider"
        assert listing.display_name == "Test Provider"
        assert listing.version == "2.0.0"
        assert listing.category == MarketplaceCategory.LLM
        assert listing.tags == ["llm", "fast"]

    def test_unique_ids(self):
        a = MarketplaceListing()
        b = MarketplaceListing()
        assert a.id != b.id

    def test_to_dict(self):
        listing = MarketplaceListing(
            name="x", display_name="X", category=MarketplaceCategory.LLM
        )
        d = listing.to_dict()
        assert d["name"] == "x"
        assert d["display_name"] == "X"
        assert d["category"] == "llm"
        assert d["status"] == "draft"
        assert isinstance(d["created_at"], float)

    def test_license_feature_default_none(self):
        listing = MarketplaceListing()
        assert listing.license_feature is None

    def test_license_feature_set(self):
        listing = MarketplaceListing(license_feature="nxs.marketplace.providers")
        assert listing.license_feature == "nxs.marketplace.providers"


class TestReview:
    """Tests for Review dataclass."""

    def test_default_review(self):
        r = Review(listing_id="abc", reviewer="user1")
        assert r.id
        assert r.listing_id == "abc"
        assert r.reviewer == "user1"
        assert r.rating == 5
        assert r.helpful_count == 0
        assert r.verified is False

    def test_rating_validation_too_low(self):
        with pytest.raises(ValueError, match="Rating must be 1-5"):
            Review(rating=0)

    def test_rating_validation_too_high(self):
        with pytest.raises(ValueError, match="Rating must be 1-5"):
            Review(rating=6)

    def test_valid_ratings(self):
        for r in range(1, 6):
            review = Review(rating=r)
            assert review.rating == r

    def test_to_dict(self):
        r = Review(listing_id="abc", reviewer="bob", rating=4, title="Good")
        d = r.to_dict()
        assert d["listing_id"] == "abc"
        assert d["reviewer"] == "bob"
        assert d["rating"] == 4
        assert d["title"] == "Good"
        assert isinstance(d["created_at"], float)


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_default_record(self):
        rec = UsageRecord(listing_id="abc")
        assert rec.listing_id == "abc"
        assert rec.action == "invoke"
        assert rec.success is True
        assert rec.error is None
        assert rec.tokens_used == 0
        assert rec.cost_usd == 0.0

    def test_to_dict(self):
        rec = UsageRecord(listing_id="abc", user="u1", tokens_used=100, cost_usd=0.01)
        d = rec.to_dict()
        assert d["listing_id"] == "abc"
        assert d["user"] == "u1"
        assert d["tokens_used"] == 100
        assert d["cost_usd"] == 0.01

    def test_error_record(self):
        rec = UsageRecord(
            listing_id="abc", success=False, error="timeout"
        )
        assert rec.success is False
        assert rec.error == "timeout"


class TestUsageSummary:
    """Tests for UsageSummary dataclass."""

    def test_default_summary(self):
        s = UsageSummary(listing_id="abc")
        assert s.total_invocations == 0
        assert s.total_installs == 0
        assert s.active_installs == 0
        assert s.success_rate == 1.0
        assert s.unique_users == 0

    def test_to_dict(self):
        s = UsageSummary(listing_id="abc", total_invocations=50, unique_users=10)
        d = s.to_dict()
        assert d["listing_id"] == "abc"
        assert d["total_invocations"] == 50
        assert d["unique_users"] == 10
