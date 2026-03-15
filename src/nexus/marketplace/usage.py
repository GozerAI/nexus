"""Usage tracking for marketplace listings.

Records individual usage events and provides aggregated summaries for
both provider and agent marketplace items.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from .models import UsageRecord, UsageSummary

logger = logging.getLogger(__name__)

# Keep the last N records per listing in memory (older are discarded from
# the in-memory store but would normally be persisted externally).
_MAX_RECORDS_PER_LISTING = 10_000


class UsageTracker:
    """Records and aggregates usage data for marketplace items."""

    def __init__(self, max_records_per_listing: int = _MAX_RECORDS_PER_LISTING) -> None:
        self._max_records = max_records_per_listing
        # listing_id -> list[UsageRecord]
        self._records: Dict[str, List[UsageRecord]] = {}
        # listing_id -> set of users that currently have it installed
        self._installed_users: Dict[str, set] = {}

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_invocation(
        self,
        listing_id: str,
        user: str = "",
        duration_ms: float = 0.0,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageRecord:
        """Record a single invocation of a marketplace item."""
        record = UsageRecord(
            listing_id=listing_id,
            user=user,
            action="invoke",
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            success=success,
            error=error,
            metadata=metadata or {},
        )
        self._append(record)
        return record

    def record_install(self, listing_id: str, user: str) -> UsageRecord:
        """Record that a user installed/enabled a marketplace item."""
        record = UsageRecord(
            listing_id=listing_id,
            user=user,
            action="install",
        )
        self._append(record)
        self._installed_users.setdefault(listing_id, set()).add(user)
        return record

    def record_uninstall(self, listing_id: str, user: str) -> UsageRecord:
        """Record that a user uninstalled/disabled a marketplace item."""
        record = UsageRecord(
            listing_id=listing_id,
            user=user,
            action="uninstall",
        )
        self._append(record)
        installed = self._installed_users.get(listing_id, set())
        installed.discard(user)
        return record

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_records(
        self,
        listing_id: str,
        *,
        action: Optional[str] = None,
        user: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[UsageRecord]:
        """Retrieve raw usage records with optional filters."""
        records = list(self._records.get(listing_id, []))
        if action:
            records = [r for r in records if r.action == action]
        if user:
            records = [r for r in records if r.user == user]
        if since is not None:
            records = [r for r in records if r.timestamp >= since]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:limit]

    def get_summary(
        self,
        listing_id: str,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
    ) -> UsageSummary:
        """Compute an aggregated usage summary for a listing."""
        records = list(self._records.get(listing_id, []))

        period_start = since if since is not None else 0.0
        period_end = until if until is not None else time.time()

        filtered = [
            r for r in records if period_start <= r.timestamp <= period_end
        ]

        invocations = [r for r in filtered if r.action == "invoke"]
        installs = [r for r in filtered if r.action == "install"]
        users = {r.user for r in filtered if r.user}

        total_inv = len(invocations)
        successful = sum(1 for r in invocations if r.success)
        total_tokens = sum(r.tokens_used for r in invocations)
        total_cost = sum(r.cost_usd for r in invocations)
        total_duration = sum(r.duration_ms for r in invocations)

        return UsageSummary(
            listing_id=listing_id,
            total_invocations=total_inv,
            total_installs=len(installs),
            active_installs=len(self._installed_users.get(listing_id, set())),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            success_rate=(successful / total_inv) if total_inv else 1.0,
            avg_duration_ms=(total_duration / total_inv) if total_inv else 0.0,
            unique_users=len(users),
            period_start=period_start,
            period_end=period_end,
        )

    def get_top_listings(
        self,
        *,
        metric: str = "invocations",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return top listings ranked by a metric.

        Supported metrics: invocations, installs, tokens, cost.
        """
        ranking: List[Dict[str, Any]] = []
        for listing_id in self._records:
            summary = self.get_summary(listing_id)
            if metric == "invocations":
                value = summary.total_invocations
            elif metric == "installs":
                value = summary.total_installs
            elif metric == "tokens":
                value = summary.total_tokens
            elif metric == "cost":
                value = summary.total_cost_usd
            else:
                raise ValueError(f"Unknown metric: {metric!r}")
            ranking.append({"listing_id": listing_id, "value": value})

        ranking.sort(key=lambda x: x["value"], reverse=True)
        return ranking[:limit]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, record: UsageRecord) -> None:
        bucket = self._records.setdefault(record.listing_id, [])
        bucket.append(record)
        if len(bucket) > self._max_records:
            # Drop oldest 10 %
            trim = max(1, self._max_records // 10)
            del bucket[:trim]
