"""Tests for the community-contributed agent marketplace."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from nexus.marketplace.models import ListingStatus, MarketplaceCategory
from nexus.marketplace.agent_marketplace import (
    AGENT_MARKETPLACE_FEATURE,
    AgentListing,
    AgentMarketplace,
    MarketplaceAgentProtocol,
)
from nexus.marketplace.reviews import ReviewManager
from nexus.marketplace.usage import UsageTracker


def _make_listing(**overrides):
    defaults = dict(
        name="summarizer-agent",
        display_name="Summarizer Agent",
        description="Summarizes documents efficiently",
        category=MarketplaceCategory.ANALYSIS,
        tags=["nlp", "summary"],
        capabilities=["text_summarization", "key_extraction"],
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"summary": {"type": "string"}}},
    )
    defaults.update(overrides)
    return AgentListing(**defaults)


def _make_agent(name="summarizer-agent", version="1.0.0"):
    """Create a mock agent satisfying MarketplaceAgentProtocol."""
    agent = MagicMock()
    agent.name = name
    agent.version = version
    agent.execute = AsyncMock(return_value={"status": "success", "result": "summarized"})
    agent.health_check = AsyncMock(return_value={"status": "healthy"})
    return agent


@pytest.fixture
def mp():
    return AgentMarketplace()


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

class TestAgentRegistration:

    def test_register_valid(self, mp):
        listing = _make_listing()
        result = mp.register(listing)
        assert result.id == listing.id
        assert mp.get(listing.id) is listing

    def test_register_with_agent_instance(self, mp):
        listing = _make_listing()
        agent = _make_agent()
        result = mp.register(listing, agent_instance=agent)
        assert result._agent_instance is agent

    def test_register_invalid_agent_rejected(self, mp):
        listing = _make_listing()
        # Object without the protocol methods
        with pytest.raises(ValueError, match="MarketplaceAgentProtocol"):
            mp.register(listing, agent_instance="not-an-agent")

    def test_register_duplicate_name_rejected(self, mp):
        mp.register(_make_listing(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            mp.register(_make_listing(name="dup"))

    def test_register_empty_name_rejected(self, mp):
        with pytest.raises(ValueError, match="must have a name"):
            mp.register(_make_listing(name=""))

    def test_register_indexes_category(self, mp):
        mp.register(_make_listing())
        cats = mp.list_categories()
        assert MarketplaceCategory.ANALYSIS in cats

    def test_register_indexes_tags(self, mp):
        mp.register(_make_listing(tags=["nlp", "summary"]))
        tags = mp.list_tags()
        assert "nlp" in tags
        assert "summary" in tags

    def test_register_indexes_capabilities(self, mp):
        mp.register(_make_listing(capabilities=["text_summarization"]))
        caps = mp.list_capabilities()
        assert "text_summarization" in caps


class TestAgentUpdate:

    def test_update_display_name(self, mp):
        listing = mp.register(_make_listing())
        updated = mp.update(listing.id, display_name="Better Name")
        assert updated.display_name == "Better Name"

    def test_update_name_reindexes(self, mp):
        listing = mp.register(_make_listing(name="old"))
        mp.update(listing.id, name="new")
        assert mp.get_by_name("new") is listing
        assert mp.get_by_name("old") is None

    def test_update_name_conflict(self, mp):
        mp.register(_make_listing(name="a"))
        b = mp.register(_make_listing(name="b"))
        with pytest.raises(ValueError, match="already taken"):
            mp.update(b.id, name="a")

        assert b.name == "b"
        assert mp.get_by_name("a").name == "a"
        assert mp.get_by_name("b") is b

    def test_update_nonexistent(self, mp):
        with pytest.raises(KeyError):
            mp.update("nope", display_name="x")

    def test_update_capabilities_reindexes(self, mp):
        listing = mp.register(_make_listing(capabilities=["cap_a"]))
        mp.update(listing.id, capabilities=["cap_b"])
        assert "cap_b" in mp.list_capabilities()
        # cap_a should be removed (no other listing has it)
        assert "cap_a" not in mp.list_capabilities()


class TestAgentUnregister:

    def test_unregister(self, mp):
        listing = mp.register(_make_listing())
        mp.unregister(listing.id)
        assert mp.get(listing.id) is None

    def test_unregister_nonexistent(self, mp):
        with pytest.raises(KeyError):
            mp.unregister("nope")

    def test_unregister_cleans_all_indexes(self, mp):
        listing = mp.register(
            _make_listing(tags=["t1"], capabilities=["c1"])
        )
        mp.unregister(listing.id)
        assert "t1" not in mp.list_tags()
        assert "c1" not in mp.list_capabilities()


# ------------------------------------------------------------------
# Lifecycle
# ------------------------------------------------------------------

class TestAgentLifecycle:

    def test_publish_from_draft(self, mp):
        listing = mp.register(_make_listing())
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
        result = mp.suspend(listing.id, reason="spam")
        assert result.status == ListingStatus.SUSPENDED
        assert result.metadata["suspend_reason"] == "spam"

    def test_deprecate(self, mp):
        listing = mp.register(_make_listing())
        result = mp.deprecate(listing.id)
        assert result.status == ListingStatus.DEPRECATED

    def test_deprecate_with_successor(self, mp):
        listing = mp.register(_make_listing())
        result = mp.deprecate(listing.id, successor_id="new-agent-id")
        assert result.metadata["successor_id"] == "new-agent-id"


# ------------------------------------------------------------------
# Invocation
# ------------------------------------------------------------------

class TestAgentInvocation:

    @pytest.mark.asyncio
    async def test_invoke_agent(self, mp):
        listing = mp.register(_make_listing())
        agent = _make_agent()
        listing._agent_instance = agent
        mp.publish(listing.id)

        result = await mp.invoke_agent(listing.id, {"text": "hello"}, user="alice")
        assert result["status"] == "success"
        agent.execute.assert_awaited_once_with({"text": "hello"})

    @pytest.mark.asyncio
    async def test_invoke_records_usage(self, mp):
        listing = mp.register(_make_listing())
        agent = _make_agent()
        listing._agent_instance = agent
        mp.publish(listing.id)

        await mp.invoke_agent(listing.id, {}, user="alice")
        summary = mp.usage.get_summary(listing.id)
        assert summary.total_invocations == 1

    @pytest.mark.asyncio
    async def test_invoke_no_instance_raises(self, mp):
        listing = mp.register(_make_listing())
        mp.publish(listing.id)
        with pytest.raises(RuntimeError, match="no attached instance"):
            await mp.invoke_agent(listing.id, {})

    @pytest.mark.asyncio
    async def test_invoke_unpublished_raises(self, mp):
        listing = mp.register(_make_listing())
        listing._agent_instance = _make_agent()
        # Still DRAFT
        with pytest.raises(RuntimeError, match="not published"):
            await mp.invoke_agent(listing.id, {})

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_raises(self, mp):
        with pytest.raises(KeyError):
            await mp.invoke_agent("nope", {})

    @pytest.mark.asyncio
    async def test_invoke_failure_records_usage(self, mp):
        listing = mp.register(_make_listing())
        agent = _make_agent()
        agent.execute = AsyncMock(side_effect=RuntimeError("boom"))
        listing._agent_instance = agent
        mp.publish(listing.id)

        with pytest.raises(RuntimeError, match="boom"):
            await mp.invoke_agent(listing.id, {})

        summary = mp.usage.get_summary(listing.id)
        assert summary.total_invocations == 1
        assert summary.success_rate == 0.0


# ------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------

class TestAgentDiscovery:

    def _seed(self, mp):
        a1 = mp.register(
            _make_listing(
                name="alpha-agent",
                display_name="Alpha Agent",
                description="Does alpha things",
                category=MarketplaceCategory.ANALYSIS,
                tags=["analysis"],
                capabilities=["summarization"],
            )
        )
        mp.publish(a1.id)

        a2 = mp.register(
            _make_listing(
                name="beta-agent",
                display_name="Beta Automation",
                description="Automates everything",
                category=MarketplaceCategory.AUTOMATION,
                tags=["automation"],
                capabilities=["workflow"],
            )
        )
        mp.publish(a2.id)

        a3 = mp.register(
            _make_listing(
                name="gamma-agent",
                display_name="Gamma Researcher",
                description="Deep research agent",
                category=MarketplaceCategory.RESEARCH,
                tags=["analysis", "research"],
                capabilities=["summarization", "research"],
            )
        )
        mp.publish(a3.id)

        return a1, a2, a3

    def test_search_all(self, mp):
        self._seed(mp)
        assert len(mp.search()) == 3

    def test_search_by_category(self, mp):
        self._seed(mp)
        results = mp.search(category=MarketplaceCategory.ANALYSIS)
        assert len(results) == 1
        assert results[0].name == "alpha-agent"

    def test_search_by_tag(self, mp):
        self._seed(mp)
        results = mp.search(tags=["analysis"])
        assert len(results) == 2

    def test_search_by_capabilities(self, mp):
        self._seed(mp)
        results = mp.search(capabilities=["summarization", "research"])
        assert len(results) == 1
        assert results[0].name == "gamma-agent"

    def test_search_by_query(self, mp):
        self._seed(mp)
        results = mp.search(query="automat")
        assert len(results) == 1
        assert results[0].name == "beta-agent"

    def test_search_excludes_drafts(self, mp):
        mp.register(_make_listing(name="draft"))
        assert len(mp.search()) == 0

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
        a1, a2, a3 = self._seed(mp)
        mp.reviews.submit_review(a3.id, "alice", 5)
        mp.reviews.submit_review(a1.id, "alice", 2)
        results = mp.search(sort_by="rating")
        assert results[0].id == a3.id

    def test_search_sort_by_popular(self, mp):
        a1, a2, a3 = self._seed(mp)
        mp.usage.record_invocation(a2.id)
        mp.usage.record_invocation(a2.id)
        results = mp.search(sort_by="popular")
        assert results[0].id == a2.id

    def test_search_min_rating(self, mp):
        a1, a2, a3 = self._seed(mp)
        mp.reviews.submit_review(a1.id, "alice", 5)
        mp.reviews.submit_review(a2.id, "alice", 2)
        results = mp.search(min_rating=4.0)
        assert len(results) == 1
        assert results[0].id == a1.id

    def test_search_pagination(self, mp):
        self._seed(mp)
        page1 = mp.search(limit=2, offset=0, sort_by="name")
        page2 = mp.search(limit=2, offset=2, sort_by="name")
        assert len(page1) == 2
        assert len(page2) == 1

    def test_find_by_capability(self, mp):
        self._seed(mp)
        results = mp.find_by_capability("summarization")
        assert len(results) == 2

    def test_find_by_capability_excludes_drafts(self, mp):
        listing = mp.register(_make_listing(name="draft", capabilities=["rare_cap"]))
        # Not published
        results = mp.find_by_capability("rare_cap")
        assert len(results) == 0

    def test_get_by_name(self, mp):
        listing = mp.register(_make_listing())
        assert mp.get_by_name("summarizer-agent") is listing
        assert mp.get_by_name("nonexistent") is None

    def test_count(self, mp):
        self._seed(mp)
        assert mp.count() == 3
        assert mp.count(status=ListingStatus.PUBLISHED) == 3
        assert mp.count(status=ListingStatus.DRAFT) == 0


# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------

class TestAgentHealth:

    @pytest.mark.asyncio
    async def test_healthy_agent(self, mp):
        listing = mp.register(_make_listing())
        listing._agent_instance = _make_agent()
        result = await mp.check_agent_health(listing.id)
        assert result["healthy"] is True

    @pytest.mark.asyncio
    async def test_no_instance(self, mp):
        listing = mp.register(_make_listing())
        result = await mp.check_agent_health(listing.id)
        assert result["healthy"] is False
        assert "No agent instance" in result["error"]

    @pytest.mark.asyncio
    async def test_agent_health_error(self, mp):
        listing = mp.register(_make_listing())
        agent = _make_agent()
        agent.health_check = AsyncMock(return_value={"status": "error"})
        listing._agent_instance = agent
        result = await mp.check_agent_health(listing.id)
        assert result["healthy"] is False

    @pytest.mark.asyncio
    async def test_agent_health_exception(self, mp):
        listing = mp.register(_make_listing())
        agent = _make_agent()
        agent.health_check = AsyncMock(side_effect=ConnectionError("down"))
        listing._agent_instance = agent
        result = await mp.check_agent_health(listing.id)
        assert result["healthy"] is False
        assert "down" in result["error"]

    @pytest.mark.asyncio
    async def test_nonexistent_listing(self, mp):
        result = await mp.check_agent_health("nope")
        assert result["healthy"] is False


# ------------------------------------------------------------------
# License integration
# ------------------------------------------------------------------

class TestAgentLicenseIntegration:

    def test_entitlement_check_passes(self):
        gate = MagicMock()
        gate.gate = MagicMock()
        amp = AgentMarketplace(license_gate=gate)
        amp._check_entitlement()
        gate.gate.assert_called_once_with(AGENT_MARKETPLACE_FEATURE)

    def test_entitlement_check_raises(self):
        gate = MagicMock()
        gate.gate = MagicMock(side_effect=PermissionError("nope"))
        amp = AgentMarketplace(license_gate=gate)
        with pytest.raises(PermissionError):
            amp._check_entitlement()

    def test_no_gate_means_no_check(self):
        amp = AgentMarketplace(license_gate=None)
        amp._check_entitlement()  # should not raise

    def test_register_requires_entitlement(self):
        gate = MagicMock()
        gate.gate = MagicMock(side_effect=PermissionError("blocked"))
        amp = AgentMarketplace(license_gate=gate)

        with pytest.raises(PermissionError, match="blocked"):
            amp.register(_make_listing())


# ------------------------------------------------------------------
# AgentListing serialization
# ------------------------------------------------------------------

class TestAgentListingSerialization:

    def test_to_dict(self):
        listing = _make_listing(
            source_url="https://github.com/example/agent",
            documentation_url="https://docs.example.com",
            min_nexus_version="1.5.0",
        )
        d = listing.to_dict()
        assert d["name"] == "summarizer-agent"
        assert d["capabilities"] == ["text_summarization", "key_extraction"]
        assert d["input_schema"]["type"] == "object"
        assert d["source_url"] == "https://github.com/example/agent"
        assert d["min_nexus_version"] == "1.5.0"
        assert d["category"] == "analysis"

    def test_to_dict_includes_base_fields(self):
        listing = _make_listing()
        d = listing.to_dict()
        assert "id" in d
        assert "status" in d
        assert "created_at" in d


# ------------------------------------------------------------------
# Shared sub-systems
# ------------------------------------------------------------------

class TestAgentSharedSystems:

    def test_custom_review_manager(self):
        rm = ReviewManager()
        amp = AgentMarketplace(review_manager=rm)
        assert amp.reviews is rm

    def test_custom_usage_tracker(self):
        ut = UsageTracker()
        amp = AgentMarketplace(usage_tracker=ut)
        assert amp.usage is ut
