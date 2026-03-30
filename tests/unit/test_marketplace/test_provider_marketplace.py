"""Tests for the third-party model provider marketplace."""

import pytest
from unittest.mock import MagicMock

from nexus.marketplace.models import ListingStatus, MarketplaceCategory
from nexus.marketplace.provider_marketplace import (
    MARKETPLACE_FEATURE,
    ProviderListing,
    ProviderMarketplace,
)
from nexus.marketplace.reviews import ReviewManager
from nexus.marketplace.usage import UsageTracker


def _make_listing(**overrides):
    defaults = dict(
        name="acme-llm",
        display_name="Acme LLM",
        description="Fast and cheap LLM provider",
        endpoint_url="https://api.acme.ai/v1",
        category=MarketplaceCategory.LLM,
        tags=["llm", "fast"],
        capabilities=["text_generation", "streaming"],
        supported_models=["acme-7b", "acme-70b"],
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
    )
    defaults.update(overrides)
    return ProviderListing(**defaults)


@pytest.fixture
def mp():
    return ProviderMarketplace()


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

class TestProviderRegistration:

    def test_register_valid(self, mp):
        listing = _make_listing()
        result = mp.register(listing)
        assert result.id == listing.id
        assert mp.get(listing.id) is listing

    def test_register_duplicate_name_rejected(self, mp):
        mp.register(_make_listing(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            mp.register(_make_listing(name="dup"))

    def test_register_empty_name_rejected(self, mp):
        with pytest.raises(ValueError, match="must have a name"):
            mp.register(_make_listing(name=""))

    def test_register_no_endpoint_rejected(self, mp):
        with pytest.raises(ValueError, match="endpoint_url"):
            mp.register(_make_listing(endpoint_url=""))

    def test_register_indexes_category(self, mp):
        mp.register(_make_listing())
        cats = mp.list_categories()
        assert MarketplaceCategory.LLM in cats

    def test_register_indexes_tags(self, mp):
        mp.register(_make_listing(tags=["llm", "fast"]))
        tags = mp.list_tags()
        assert "llm" in tags
        assert "fast" in tags


class TestProviderUpdate:

    def test_update_display_name(self, mp):
        listing = mp.register(_make_listing())
        updated = mp.update(listing.id, display_name="New Name")
        assert updated.display_name == "New Name"

    def test_update_changes_updated_at(self, mp):
        listing = mp.register(_make_listing())
        old_ts = listing.updated_at
        updated = mp.update(listing.id, description="new desc")
        assert updated.updated_at >= old_ts

    def test_update_name_reindexes(self, mp):
        listing = mp.register(_make_listing(name="old-name"))
        mp.update(listing.id, name="new-name")
        assert mp.get_by_name("new-name") is listing
        assert mp.get_by_name("old-name") is None

    def test_update_name_conflict_rejected(self, mp):
        mp.register(_make_listing(name="a"))
        listing_b = mp.register(_make_listing(name="b", endpoint_url="https://b.ai"))
        with pytest.raises(ValueError, match="already taken"):
            mp.update(listing_b.id, name="a")

        assert listing_b.name == "b"
        assert mp.get_by_name("a").name == "a"
        assert mp.get_by_name("b") is listing_b

    def test_update_nonexistent_raises(self, mp):
        with pytest.raises(KeyError):
            mp.update("nope", display_name="x")

    def test_update_ignores_id_and_created_at(self, mp):
        listing = mp.register(_make_listing())
        original_id = listing.id
        original_created = listing.created_at
        mp.update(listing.id, id="hacked", created_at=0.0)
        assert listing.id == original_id
        assert listing.created_at == original_created


class TestProviderUnregister:

    def test_unregister(self, mp):
        listing = mp.register(_make_listing())
        mp.unregister(listing.id)
        assert mp.get(listing.id) is None
        assert mp.get_by_name("acme-llm") is None

    def test_unregister_nonexistent_raises(self, mp):
        with pytest.raises(KeyError):
            mp.unregister("nope")

    def test_unregister_removes_from_indexes(self, mp):
        listing = mp.register(_make_listing(tags=["t1"]))
        mp.unregister(listing.id)
        assert "t1" not in mp.list_tags()


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

class TestProviderLifecycle:

    def test_publish_from_draft(self, mp):
        listing = mp.register(_make_listing())
        assert listing.status == ListingStatus.DRAFT
        result = mp.publish(listing.id)
        assert result.status == ListingStatus.PUBLISHED

    def test_publish_from_pending_review(self, mp):
        listing = mp.register(_make_listing())
        listing.status = ListingStatus.PENDING_REVIEW
        result = mp.publish(listing.id)
        assert result.status == ListingStatus.PUBLISHED

    def test_publish_from_suspended_fails(self, mp):
        listing = mp.register(_make_listing())
        listing.status = ListingStatus.SUSPENDED
        with pytest.raises(ValueError, match="Cannot publish"):
            mp.publish(listing.id)

    def test_suspend(self, mp):
        listing = mp.register(_make_listing())
        result = mp.suspend(listing.id, reason="policy violation")
        assert result.status == ListingStatus.SUSPENDED
        assert result.metadata["suspend_reason"] == "policy violation"

    def test_deprecate(self, mp):
        listing = mp.register(_make_listing())
        result = mp.deprecate(listing.id)
        assert result.status == ListingStatus.DEPRECATED

    def test_deprecate_with_successor(self, mp):
        listing = mp.register(_make_listing())
        result = mp.deprecate(listing.id, successor_id="new-id")
        assert result.metadata["successor_id"] == "new-id"


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------

class TestProviderDiscovery:

    def _seed(self, mp):
        p1 = mp.register(
            _make_listing(
                name="alpha-llm",
                display_name="Alpha LLM",
                description="Fast LLM",
                category=MarketplaceCategory.LLM,
                tags=["fast"],
                capabilities=["text_generation"],
            )
        )
        mp.publish(p1.id)

        p2 = mp.register(
            _make_listing(
                name="beta-embed",
                display_name="Beta Embeddings",
                description="Good embeddings",
                endpoint_url="https://beta.ai",
                category=MarketplaceCategory.EMBEDDING,
                tags=["embedding"],
                capabilities=["embeddings"],
            )
        )
        mp.publish(p2.id)

        p3 = mp.register(
            _make_listing(
                name="gamma-llm",
                display_name="Gamma LLM",
                description="Cheap LLM",
                endpoint_url="https://gamma.ai",
                category=MarketplaceCategory.LLM,
                tags=["cheap", "fast"],
                capabilities=["text_generation", "streaming"],
            )
        )
        mp.publish(p3.id)

        return p1, p2, p3

    def test_search_all_published(self, mp):
        self._seed(mp)
        results = mp.search()
        assert len(results) == 3

    def test_search_by_category(self, mp):
        self._seed(mp)
        results = mp.search(category=MarketplaceCategory.LLM)
        assert len(results) == 2
        assert all(r.category == MarketplaceCategory.LLM for r in results)

    def test_search_by_tag(self, mp):
        self._seed(mp)
        results = mp.search(tags=["fast"])
        assert len(results) == 2

    def test_search_by_capabilities(self, mp):
        self._seed(mp)
        results = mp.search(capabilities=["streaming"])
        assert len(results) == 1
        assert results[0].name == "gamma-llm"

    def test_search_by_text_query(self, mp):
        self._seed(mp)
        results = mp.search(query="cheap")
        assert len(results) == 1
        assert results[0].name == "gamma-llm"

    def test_search_case_insensitive(self, mp):
        self._seed(mp)
        results = mp.search(query="ALPHA")
        assert len(results) == 1

    def test_search_excludes_drafts(self, mp):
        mp.register(_make_listing(name="draft-prov"))
        results = mp.search()
        assert len(results) == 0

    def test_search_by_draft_status(self, mp):
        mp.register(_make_listing(name="draft-prov"))
        results = mp.search(status=ListingStatus.DRAFT)
        assert len(results) == 1

    def test_search_sort_by_name(self, mp):
        self._seed(mp)
        results = mp.search(sort_by="name")
        names = [r.name for r in results]
        assert names == sorted(names)

    def test_search_sort_by_newest(self, mp):
        self._seed(mp)
        results = mp.search(sort_by="newest")
        times = [r.created_at for r in results]
        assert times == sorted(times, reverse=True)

    def test_search_sort_by_rating(self, mp):
        p1, p2, p3 = self._seed(mp)
        mp.reviews.submit_review(p3.id, "alice", 5)
        mp.reviews.submit_review(p1.id, "alice", 2)
        results = mp.search(sort_by="rating")
        assert results[0].id == p3.id

    def test_search_sort_by_popular(self, mp):
        p1, p2, p3 = self._seed(mp)
        mp.usage.record_invocation(p2.id)
        mp.usage.record_invocation(p2.id)
        mp.usage.record_invocation(p1.id)
        results = mp.search(sort_by="popular")
        assert results[0].id == p2.id

    def test_search_min_rating(self, mp):
        p1, p2, p3 = self._seed(mp)
        mp.reviews.submit_review(p1.id, "alice", 5)
        mp.reviews.submit_review(p2.id, "alice", 2)
        results = mp.search(min_rating=4.0)
        assert len(results) == 1
        assert results[0].id == p1.id

    def test_search_pagination(self, mp):
        self._seed(mp)
        page1 = mp.search(limit=2, offset=0, sort_by="name")
        page2 = mp.search(limit=2, offset=2, sort_by="name")
        assert len(page1) == 2
        assert len(page2) == 1
        assert page1[0].id != page2[0].id

    def test_get_by_name(self, mp):
        listing = mp.register(_make_listing())
        assert mp.get_by_name("acme-llm") is listing
        assert mp.get_by_name("nonexistent") is None

    def test_count(self, mp):
        self._seed(mp)
        assert mp.count() == 3
        assert mp.count(status=ListingStatus.PUBLISHED) == 3
        assert mp.count(status=ListingStatus.DRAFT) == 0


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

class TestProviderHealth:

    def test_health_no_fn(self, mp):
        listing = mp.register(_make_listing())
        result = mp.check_health(listing.id)
        assert result["healthy"] is True
        assert "No health check" in result.get("note", "")

    def test_health_fn_passes(self, mp):
        listing = _make_listing()
        listing._health_fn = lambda: True
        mp.register(listing)
        result = mp.check_health(listing.id)
        assert result["healthy"] is True

    def test_health_fn_fails(self, mp):
        listing = _make_listing()
        listing._health_fn = lambda: False
        mp.register(listing)
        result = mp.check_health(listing.id)
        assert result["healthy"] is False

    def test_health_fn_raises(self, mp):
        listing = _make_listing()
        listing._health_fn = lambda: (_ for _ in ()).throw(ConnectionError("down"))
        mp.register(listing)
        result = mp.check_health(listing.id)
        assert result["healthy"] is False
        assert "down" in result["error"]

    def test_health_nonexistent(self, mp):
        result = mp.check_health("nope")
        assert result["healthy"] is False


# ------------------------------------------------------------------
# License integration
# ------------------------------------------------------------------

class TestProviderLicenseIntegration:

    def test_entitlement_check_passes(self):
        gate = MagicMock()
        gate.gate = MagicMock()  # no raise
        mp = ProviderMarketplace(license_gate=gate)
        mp._check_entitlement()
        gate.gate.assert_called_once_with(MARKETPLACE_FEATURE)

    def test_entitlement_check_raises(self):
        gate = MagicMock()
        gate.gate = MagicMock(side_effect=PermissionError("nope"))
        mp = ProviderMarketplace(license_gate=gate)
        with pytest.raises(PermissionError):
            mp._check_entitlement()

    def test_no_gate_means_no_check(self):
        mp = ProviderMarketplace(license_gate=None)
        mp._check_entitlement()  # should not raise

    def test_register_requires_entitlement(self):
        gate = MagicMock()
        gate.gate = MagicMock(side_effect=PermissionError("blocked"))
        mp = ProviderMarketplace(license_gate=gate)

        with pytest.raises(PermissionError, match="blocked"):
            mp.register(_make_listing())


# ------------------------------------------------------------------
# ProviderListing serialization
# ------------------------------------------------------------------

class TestProviderListingSerialization:

    def test_to_dict(self):
        listing = _make_listing()
        d = listing.to_dict()
        assert d["name"] == "acme-llm"
        assert d["endpoint_url"] == "https://api.acme.ai/v1"
        assert d["supported_models"] == ["acme-7b", "acme-70b"]
        assert d["capabilities"] == ["text_generation", "streaming"]
        assert d["cost_per_1k_input"] == 0.001
        assert d["category"] == "llm"

    def test_to_dict_includes_base_fields(self):
        listing = _make_listing()
        d = listing.to_dict()
        assert "id" in d
        assert "status" in d
        assert "created_at" in d


# ------------------------------------------------------------------
# Shared ReviewManager and UsageTracker
# ------------------------------------------------------------------

class TestProviderSharedSystems:

    def test_custom_review_manager(self):
        rm = ReviewManager()
        mp = ProviderMarketplace(review_manager=rm)
        assert mp.reviews is rm

    def test_custom_usage_tracker(self):
        ut = UsageTracker()
        mp = ProviderMarketplace(usage_tracker=ut)
        assert mp.usage is ut

    def test_default_review_manager(self, mp):
        assert isinstance(mp.reviews, ReviewManager)

    def test_default_usage_tracker(self, mp):
        assert isinstance(mp.usage, UsageTracker)
