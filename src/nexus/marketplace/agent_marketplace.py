"""Community-contributed agent marketplace (item #436).

Allows community members to publish, discover, rate, and track usage of
reusable agents within the Nexus platform.  Agents must conform to the
``AgentInterface`` protocol defined in ``nexus.agents.registry``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .models import (
    ListingStatus,
    MarketplaceCategory,
    MarketplaceListing,
)
from .reviews import ReviewManager
from .usage import UsageTracker

logger = logging.getLogger(__name__)

# License feature flag for community agent marketplace
AGENT_MARKETPLACE_FEATURE = "nxs.marketplace.agents"


@runtime_checkable
class MarketplaceAgentProtocol(Protocol):
    """Minimal protocol that community agents must satisfy."""

    @property
    def name(self) -> str: ...  # noqa: E704

    @property
    def version(self) -> str: ...  # noqa: E704

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]: ...  # noqa: E704

    async def health_check(self) -> Dict[str, Any]: ...  # noqa: E704


@dataclass
class AgentListing(MarketplaceListing):
    """A community-contributed agent listing."""

    # Agent-specific fields
    capabilities: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    max_concurrent: int = 1
    avg_latency_ms: float = 0.0
    supported_platforms: List[str] = field(default_factory=lambda: ["nexus"])
    source_url: str = ""
    documentation_url: str = ""
    min_nexus_version: str = ""
    # The actual agent instance (not serialized)
    _agent_instance: Optional[Any] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "capabilities": list(self.capabilities),
                "input_schema": dict(self.input_schema),
                "output_schema": dict(self.output_schema),
                "max_concurrent": self.max_concurrent,
                "avg_latency_ms": self.avg_latency_ms,
                "supported_platforms": list(self.supported_platforms),
                "source_url": self.source_url,
                "documentation_url": self.documentation_url,
                "min_nexus_version": self.min_nexus_version,
            }
        )
        return base


class AgentMarketplace:
    """Registry and discovery surface for community-contributed agents."""

    def __init__(
        self,
        review_manager: Optional[ReviewManager] = None,
        usage_tracker: Optional[UsageTracker] = None,
        license_gate: Optional[Any] = None,
    ) -> None:
        self._listings: Dict[str, AgentListing] = {}
        self._name_index: Dict[str, str] = {}  # name -> id
        self._category_index: Dict[MarketplaceCategory, List[str]] = {}
        self._tag_index: Dict[str, List[str]] = {}
        self._capability_index: Dict[str, List[str]] = {}
        self.reviews = review_manager or ReviewManager()
        self.usage = usage_tracker or UsageTracker()
        self._license_gate = license_gate

    # ------------------------------------------------------------------
    # Entitlement
    # ------------------------------------------------------------------

    def _check_entitlement(self) -> None:
        """Raise PermissionError when the caller lacks the marketplace entitlement."""
        if self._license_gate is not None:
            try:
                self._license_gate.gate(AGENT_MARKETPLACE_FEATURE)
            except PermissionError:
                raise

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        listing: AgentListing,
        agent_instance: Optional[Any] = None,
    ) -> AgentListing:
        """Register a community agent listing.

        Args:
            listing: The agent listing metadata.
            agent_instance: Optional live agent object that satisfies
                ``MarketplaceAgentProtocol``.  If provided it is validated
                and attached to the listing for direct invocation.

        Raises:
            ValueError: on duplicate name, missing fields, or invalid agent.
        """
        self._check_entitlement()
        if not listing.name:
            raise ValueError("Agent listing must have a name")
        if listing.name in self._name_index:
            raise ValueError(f"Agent {listing.name!r} already registered")

        if agent_instance is not None:
            if not isinstance(agent_instance, MarketplaceAgentProtocol):
                raise ValueError(
                    "agent_instance must satisfy MarketplaceAgentProtocol"
                )
            listing._agent_instance = agent_instance

        listing.created_at = time.time()
        listing.updated_at = listing.created_at

        self._listings[listing.id] = listing
        self._name_index[listing.name] = listing.id
        self._index_category(listing)
        self._index_tags(listing)
        self._index_capabilities(listing)

        logger.info("Agent registered: %s (%s)", listing.name, listing.id)
        return listing

    def update(self, listing_id: str, **fields: Any) -> AgentListing:
        """Update an agent listing in-place.

        Raises:
            KeyError: if listing_id not found.
        """
        self._check_entitlement()
        listing = self._get_or_raise(listing_id)
        old_name = listing.name
        old_category = listing.category
        old_tags = set(listing.tags)
        old_caps = set(listing.capabilities)
        new_name = fields.get("name", listing.name)

        if (
            new_name != old_name
            and new_name in self._name_index
            and self._name_index[new_name] != listing.id
        ):
            raise ValueError(f"Agent name {new_name!r} already taken")

        for key, value in fields.items():
            if not hasattr(listing, key) or key in ("id", "created_at"):
                continue
            setattr(listing, key, value)

        listing.updated_at = time.time()

        # Re-index name
        if listing.name != old_name:
            del self._name_index[old_name]
            self._name_index[listing.name] = listing.id

        # Re-index category
        if listing.category != old_category:
            self._deindex_category(listing.id, old_category)
            self._index_category(listing)

        # Re-index tags
        new_tags = set(listing.tags)
        for tag in old_tags - new_tags:
            self._deindex_tag(listing.id, tag)
        for tag in new_tags - old_tags:
            self._tag_index.setdefault(tag, []).append(listing.id)

        # Re-index capabilities
        new_caps = set(listing.capabilities)
        for cap in old_caps - new_caps:
            self._deindex_capability(listing.id, cap)
        for cap in new_caps - old_caps:
            self._capability_index.setdefault(cap, []).append(listing.id)

        return listing

    def unregister(self, listing_id: str) -> None:
        """Remove an agent from the marketplace.

        Raises:
            KeyError: if listing_id not found.
        """
        self._check_entitlement()
        listing = self._get_or_raise(listing_id)
        del self._listings[listing_id]
        del self._name_index[listing.name]
        self._deindex_category(listing_id, listing.category)
        for tag in listing.tags:
            self._deindex_tag(listing_id, tag)
        for cap in listing.capabilities:
            self._deindex_capability(listing_id, cap)
        logger.info("Agent unregistered: %s", listing.name)

    def publish(self, listing_id: str) -> AgentListing:
        """Move a listing to PUBLISHED status.

        Raises:
            KeyError: if listing_id not found.
            ValueError: if not in a publishable state.
        """
        self._check_entitlement()
        listing = self._get_or_raise(listing_id)
        if listing.status not in (ListingStatus.DRAFT, ListingStatus.PENDING_REVIEW):
            raise ValueError(
                f"Cannot publish listing in status {listing.status.value}"
            )
        listing.status = ListingStatus.PUBLISHED
        listing.updated_at = time.time()
        return listing

    def suspend(self, listing_id: str, reason: str = "") -> AgentListing:
        """Suspend a listing.

        Raises:
            KeyError: if listing_id not found.
        """
        self._check_entitlement()
        listing = self._get_or_raise(listing_id)
        listing.status = ListingStatus.SUSPENDED
        listing.metadata["suspend_reason"] = reason
        listing.updated_at = time.time()
        logger.warning("Agent suspended: %s — %s", listing.name, reason)
        return listing

    def deprecate(self, listing_id: str, successor_id: str = "") -> AgentListing:
        """Deprecate a listing, optionally pointing to a successor.

        Raises:
            KeyError: if listing_id not found.
        """
        self._check_entitlement()
        listing = self._get_or_raise(listing_id)
        listing.status = ListingStatus.DEPRECATED
        if successor_id:
            listing.metadata["successor_id"] = successor_id
        listing.updated_at = time.time()
        return listing

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    async def invoke_agent(
        self,
        listing_id: str,
        task: Dict[str, Any],
        user: str = "",
    ) -> Dict[str, Any]:
        """Invoke a registered agent directly.

        Records usage automatically and returns the agent's result dict.

        Raises:
            KeyError: if listing not found.
            RuntimeError: if no agent instance attached.
        """
        self._check_entitlement()
        listing = self._get_or_raise(listing_id)
        if listing._agent_instance is None:
            raise RuntimeError(
                f"Agent {listing.name!r} has no attached instance"
            )
        if listing.status != ListingStatus.PUBLISHED:
            raise RuntimeError(
                f"Agent {listing.name!r} is not published (status={listing.status.value})"
            )

        start = time.time()
        success = True
        error_msg: Optional[str] = None
        result: Dict[str, Any] = {}

        try:
            result = await listing._agent_instance.execute(task)
        except Exception as exc:
            success = False
            error_msg = str(exc)
            raise
        finally:
            duration_ms = (time.time() - start) * 1000
            self.usage.record_invocation(
                listing_id=listing_id,
                user=user,
                duration_ms=duration_ms,
                success=success,
                error=error_msg,
            )

        return result

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get(self, listing_id: str) -> Optional[AgentListing]:
        """Look up an agent by ID."""
        self._check_entitlement()
        return self._listings.get(listing_id)

    def get_by_name(self, name: str) -> Optional[AgentListing]:
        """Look up an agent by its unique name."""
        self._check_entitlement()
        lid = self._name_index.get(name)
        return self._listings.get(lid) if lid else None

    def search(
        self,
        *,
        query: str = "",
        category: Optional[MarketplaceCategory] = None,
        tags: Optional[List[str]] = None,
        status: Optional[ListingStatus] = None,
        capabilities: Optional[List[str]] = None,
        min_rating: float = 0.0,
        sort_by: str = "name",
        limit: int = 50,
        offset: int = 0,
    ) -> List[AgentListing]:
        """Search and filter agent listings."""
        self._check_entitlement()
        results: List[AgentListing] = list(self._listings.values())

        target_status = status or ListingStatus.PUBLISHED
        results = [r for r in results if r.status == target_status]

        if category:
            results = [r for r in results if r.category == category]

        if tags:
            tag_set = set(tags)
            results = [r for r in results if tag_set & set(r.tags)]

        if capabilities:
            cap_set = set(capabilities)
            results = [r for r in results if cap_set <= set(r.capabilities)]

        if query:
            q = query.lower()
            results = [
                r
                for r in results
                if q in r.name.lower()
                or q in r.display_name.lower()
                or q in r.description.lower()
            ]

        if min_rating > 0:
            results = [
                r
                for r in results
                if self.reviews.get_rating_snapshot(r.id).average_rating >= min_rating
            ]

        if sort_by == "name":
            results.sort(key=lambda r: r.name)
        elif sort_by == "newest":
            results.sort(key=lambda r: r.created_at, reverse=True)
        elif sort_by == "rating":
            results.sort(
                key=lambda r: self.reviews.get_rating_snapshot(r.id).average_rating,
                reverse=True,
            )
        elif sort_by == "popular":
            results.sort(
                key=lambda r: self.usage.get_summary(r.id).total_invocations,
                reverse=True,
            )

        return results[offset : offset + limit]

    def find_by_capability(self, capability: str) -> List[AgentListing]:
        """Find all published agents that advertise a given capability."""
        self._check_entitlement()
        ids = self._capability_index.get(capability, [])
        return [
            self._listings[lid]
            for lid in ids
            if lid in self._listings
            and self._listings[lid].status == ListingStatus.PUBLISHED
        ]

    def list_categories(self) -> List[MarketplaceCategory]:
        """Return categories that have at least one listing."""
        self._check_entitlement()
        return [c for c, ids in self._category_index.items() if ids]

    def list_tags(self) -> List[str]:
        """Return all tags in use."""
        self._check_entitlement()
        return sorted(t for t, ids in self._tag_index.items() if ids)

    def list_capabilities(self) -> List[str]:
        """Return all capabilities advertised by at least one agent."""
        self._check_entitlement()
        return sorted(c for c, ids in self._capability_index.items() if ids)

    def count(self, *, status: Optional[ListingStatus] = None) -> int:
        """Count listings, optionally filtered by status."""
        self._check_entitlement()
        if status is None:
            return len(self._listings)
        return sum(1 for l in self._listings.values() if l.status == status)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def check_agent_health(self, listing_id: str) -> Dict[str, Any]:
        """Run the attached agent's health check.

        Returns:
            dict with ``healthy`` bool and optional error/status detail.
        """
        self._check_entitlement()
        listing = self._listings.get(listing_id)
        if listing is None:
            return {"healthy": False, "error": "Listing not found"}
        if listing._agent_instance is None:
            return {"healthy": False, "error": "No agent instance attached"}
        try:
            result = await listing._agent_instance.health_check()
            status = result.get("status", "unknown")
            return {"healthy": status != "error", "detail": result}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_raise(self, listing_id: str) -> AgentListing:
        listing = self._listings.get(listing_id)
        if listing is None:
            raise KeyError(f"Agent listing {listing_id!r} not found")
        return listing

    def _index_category(self, listing: AgentListing) -> None:
        self._category_index.setdefault(listing.category, []).append(listing.id)

    def _deindex_category(self, listing_id: str, category: MarketplaceCategory) -> None:
        ids = self._category_index.get(category, [])
        if listing_id in ids:
            ids.remove(listing_id)

    def _index_tags(self, listing: AgentListing) -> None:
        for tag in listing.tags:
            self._tag_index.setdefault(tag, []).append(listing.id)

    def _deindex_tag(self, listing_id: str, tag: str) -> None:
        ids = self._tag_index.get(tag, [])
        if listing_id in ids:
            ids.remove(listing_id)

    def _index_capabilities(self, listing: AgentListing) -> None:
        for cap in listing.capabilities:
            self._capability_index.setdefault(cap, []).append(listing.id)

    def _deindex_capability(self, listing_id: str, cap: str) -> None:
        ids = self._capability_index.get(cap, [])
        if listing_id in ids:
            ids.remove(listing_id)
