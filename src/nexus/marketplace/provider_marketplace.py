"""Third-party model provider marketplace (item #427).

Allows external model providers to register, be discovered, rated, and
usage-tracked within the Nexus platform.  Integrates with LicenseGate so
marketplace access can be gated behind entitlements.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import (
    ListingStatus,
    MarketplaceCategory,
    MarketplaceListing,
)
from .reviews import ReviewManager
from .usage import UsageTracker

logger = logging.getLogger(__name__)

# License feature flag for marketplace access
MARKETPLACE_FEATURE = "nxs.marketplace.providers"


@dataclass
class ProviderListing(MarketplaceListing):
    """A third-party model provider listing."""

    # Provider-specific fields
    endpoint_url: str = ""
    api_schema_url: str = ""
    supported_models: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    max_context_window: int = 0
    supports_streaming: bool = False
    requires_api_key: bool = True
    health_check_url: str = ""
    # Optional callable that performs a live health check; not serialized.
    _health_fn: Optional[Callable[[], bool]] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "endpoint_url": self.endpoint_url,
                "api_schema_url": self.api_schema_url,
                "supported_models": list(self.supported_models),
                "capabilities": list(self.capabilities),
                "cost_per_1k_input": self.cost_per_1k_input,
                "cost_per_1k_output": self.cost_per_1k_output,
                "max_context_window": self.max_context_window,
                "supports_streaming": self.supports_streaming,
                "requires_api_key": self.requires_api_key,
                "health_check_url": self.health_check_url,
            }
        )
        return base


class ProviderMarketplace:
    """Registry and discovery surface for third-party model providers."""

    def __init__(
        self,
        review_manager: Optional[ReviewManager] = None,
        usage_tracker: Optional[UsageTracker] = None,
        license_gate: Optional[Any] = None,
    ) -> None:
        self._listings: Dict[str, ProviderListing] = {}
        self._name_index: Dict[str, str] = {}  # name -> id
        self._category_index: Dict[MarketplaceCategory, List[str]] = {}
        self._tag_index: Dict[str, List[str]] = {}
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
                self._license_gate.gate(MARKETPLACE_FEATURE)
            except PermissionError:
                raise

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, listing: ProviderListing) -> ProviderListing:
        """Register a new provider listing.

        Raises:
            ValueError: when name is duplicate or required fields are missing.
        """
        self._check_entitlement()
        if not listing.name:
            raise ValueError("Provider listing must have a name")
        if listing.name in self._name_index:
            raise ValueError(f"Provider {listing.name!r} already registered")
        if not listing.endpoint_url:
            raise ValueError("Provider listing must have an endpoint_url")

        listing.created_at = time.time()
        listing.updated_at = listing.created_at

        self._listings[listing.id] = listing
        self._name_index[listing.name] = listing.id
        self._index_category(listing)
        self._index_tags(listing)

        logger.info("Provider registered: %s (%s)", listing.name, listing.id)
        return listing

    def update(
        self,
        listing_id: str,
        **fields: Any,
    ) -> ProviderListing:
        """Update a provider listing in-place.

        Raises:
            KeyError: if listing_id not found.
        """
        self._check_entitlement()
        listing = self._get_or_raise(listing_id)
        old_name = listing.name
        old_category = listing.category
        old_tags = set(listing.tags)
        new_name = fields.get("name", listing.name)

        if (
            new_name != old_name
            and new_name in self._name_index
            and self._name_index[new_name] != listing.id
        ):
            raise ValueError(f"Provider name {new_name!r} already taken")

        for key, value in fields.items():
            if not hasattr(listing, key) or key in ("id", "created_at"):
                continue
            setattr(listing, key, value)

        listing.updated_at = time.time()

        # Re-index if name changed
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

        return listing

    def unregister(self, listing_id: str) -> None:
        """Remove a provider from the marketplace.

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
        logger.info("Provider unregistered: %s", listing.name)

    def publish(self, listing_id: str) -> ProviderListing:
        """Move a listing to PUBLISHED status.

        Raises:
            KeyError: if listing_id not found.
            ValueError: if listing is not in a publishable state.
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

    def suspend(self, listing_id: str, reason: str = "") -> ProviderListing:
        """Suspend a listing.

        Raises:
            KeyError: if listing_id not found.
        """
        self._check_entitlement()
        listing = self._get_or_raise(listing_id)
        listing.status = ListingStatus.SUSPENDED
        listing.metadata["suspend_reason"] = reason
        listing.updated_at = time.time()
        logger.warning("Provider suspended: %s — %s", listing.name, reason)
        return listing

    def deprecate(self, listing_id: str, successor_id: str = "") -> ProviderListing:
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
    # Discovery
    # ------------------------------------------------------------------

    def get(self, listing_id: str) -> Optional[ProviderListing]:
        """Look up a provider by ID."""
        self._check_entitlement()
        return self._listings.get(listing_id)

    def get_by_name(self, name: str) -> Optional[ProviderListing]:
        """Look up a provider by its unique name."""
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
    ) -> List[ProviderListing]:
        """Search and filter provider listings."""
        self._check_entitlement()
        results: List[ProviderListing] = list(self._listings.values())

        # Status filter (default: published only)
        target_status = status or ListingStatus.PUBLISHED
        results = [r for r in results if r.status == target_status]

        # Category filter
        if category:
            results = [r for r in results if r.category == category]

        # Tag filter (match any)
        if tags:
            tag_set = set(tags)
            results = [r for r in results if tag_set & set(r.tags)]

        # Capability filter (match all)
        if capabilities:
            cap_set = set(capabilities)
            results = [r for r in results if cap_set <= set(r.capabilities)]

        # Text search (name, display_name, description)
        if query:
            q = query.lower()
            results = [
                r
                for r in results
                if q in r.name.lower()
                or q in r.display_name.lower()
                or q in r.description.lower()
            ]

        # Rating filter
        if min_rating > 0:
            results = [
                r
                for r in results
                if self.reviews.get_rating_snapshot(r.id).average_rating >= min_rating
            ]

        # Sort
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

    def list_categories(self) -> List[MarketplaceCategory]:
        """Return categories that have at least one listing."""
        self._check_entitlement()
        return [c for c, ids in self._category_index.items() if ids]

    def list_tags(self) -> List[str]:
        """Return all tags in use."""
        self._check_entitlement()
        return sorted(t for t, ids in self._tag_index.items() if ids)

    def count(self, *, status: Optional[ListingStatus] = None) -> int:
        """Count listings, optionally filtered by status."""
        self._check_entitlement()
        if status is None:
            return len(self._listings)
        return sum(1 for l in self._listings.values() if l.status == status)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def check_health(self, listing_id: str) -> Dict[str, Any]:
        """Run the provider's health check if available.

        Returns:
            dict with ``healthy`` bool and optional error details.
        """
        self._check_entitlement()
        listing = self._listings.get(listing_id)
        if listing is None:
            return {"healthy": False, "error": "Listing not found"}
        if listing._health_fn is None:
            return {"healthy": True, "note": "No health check configured"}
        try:
            ok = listing._health_fn()
            return {"healthy": bool(ok)}
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_or_raise(self, listing_id: str) -> ProviderListing:
        listing = self._listings.get(listing_id)
        if listing is None:
            raise KeyError(f"Provider listing {listing_id!r} not found")
        return listing

    def _index_category(self, listing: ProviderListing) -> None:
        self._category_index.setdefault(listing.category, []).append(listing.id)

    def _deindex_category(self, listing_id: str, category: MarketplaceCategory) -> None:
        ids = self._category_index.get(category, [])
        if listing_id in ids:
            ids.remove(listing_id)

    def _index_tags(self, listing: ProviderListing) -> None:
        for tag in listing.tags:
            self._tag_index.setdefault(tag, []).append(listing.id)

    def _deindex_tag(self, listing_id: str, tag: str) -> None:
        ids = self._tag_index.get(tag, [])
        if listing_id in ids:
            ids.remove(listing_id)
