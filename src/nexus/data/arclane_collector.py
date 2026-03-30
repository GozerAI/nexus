"""
Arclane Collector for Nexus AI Platform.

Fetches business cycle insights from Arclane and creates knowledge items.
Uses urllib with graceful degradation (no new dependencies).
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from nexus.memory.knowledge_base import KnowledgeBase, KnowledgeType

# Optional resilience
try:
    from gozerai_telemetry.resilience import (
        resilient_fetch,
        get_circuit_breaker,
        DEFAULT_RETRY,
    )
    _HAS_RESILIENCE = True
    _arclane_cb = get_circuit_breaker("arclane", failure_threshold=3, recovery_timeout=120)
except ImportError:
    _HAS_RESILIENCE = False
    _arclane_cb = None

logger = logging.getLogger(__name__)

DEFAULT_ARCLANE_BASE_URL = "http://localhost:8012"
ARCLANE_TIMEOUT = 5


class ArclaneCollector:
    """Collects business intelligence from Arclane cycle results.

    Fetches recent cycle activity and content to create knowledge items
    about business strategies, market research, and content patterns.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        arclane_base_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        confidence: float = 0.70,
    ):
        self.knowledge_base = knowledge_base
        self.arclane_base_url = arclane_base_url or os.environ.get(
            "ARCLANE_BASE_URL", DEFAULT_ARCLANE_BASE_URL
        )
        self.bearer_token = bearer_token or os.environ.get("ARCLANE_API_TOKEN", "")
        self.confidence = confidence

    def _fetch_json(self, url: str) -> Optional[Any]:
        """GET url and return parsed JSON, or None on failure."""
        headers = {"Accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if _HAS_RESILIENCE:
            return resilient_fetch(
                url, headers=headers,
                timeout=float(ARCLANE_TIMEOUT), retry_policy=DEFAULT_RETRY, circuit_breaker=_arclane_cb,
            )
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=ARCLANE_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, OSError) as exc:
            logger.warning("ArclaneCollector: failed to fetch %s: %s", url, exc)
            return None
        except Exception as exc:
            logger.warning("ArclaneCollector: unexpected error fetching %s: %s", url, exc)
            return None

    def collect_recent_activity(self) -> List[str]:
        """Fetch recent cycle activity and create knowledge items.

        Calls GET /api/v1/activity/recent (public API on Arclane).
        Returns list of knowledge IDs created.
        """
        data = self._fetch_json(f"{self.arclane_base_url}/api/v1/activity/recent")
        if data is None:
            return []

        knowledge_ids: List[str] = []
        activities = data if isinstance(data, list) else data.get("activities", [])

        # Group by business for summary knowledge
        business_insights: Dict[str, List[str]] = {}
        for activity in activities:
            biz = activity.get("business_name", "unknown")
            action = activity.get("action", "")
            detail = activity.get("detail", "")
            if action and detail:
                business_insights.setdefault(biz, []).append(f"{action}: {detail[:200]}")

        for biz_name, insights in business_insights.items():
            fact = (
                f"Arclane business '{biz_name}' recent cycle activity: "
                + "; ".join(insights[:5])
            )
            tags = ["arclane", "business_cycle", biz_name.lower().replace(" ", "_")]
            kid = self.knowledge_base.add_knowledge(
                content=fact,
                knowledge_type=KnowledgeType.FACTUAL,
                source=f"arclane:activity:{biz_name}",
                confidence=self.confidence,
                context_tags=tags,
            )
            knowledge_ids.append(kid)

        return knowledge_ids

    def collect_content_patterns(self) -> List[str]:
        """Fetch content production data and create knowledge items.

        Calls GET /api/v1/content/recent to understand what content types
        are being produced and their characteristics.
        Returns list of knowledge IDs created.
        """
        data = self._fetch_json(f"{self.arclane_base_url}/api/v1/content/recent")
        if data is None:
            return []

        knowledge_ids: List[str] = []
        items = data if isinstance(data, list) else data.get("content", [])

        # Summarize by content type
        type_counts: Dict[str, int] = {}
        for item in items:
            ct = item.get("content_type", "unknown")
            type_counts[ct] = type_counts.get(ct, 0) + 1

        if type_counts:
            summary = ", ".join(f"{ct}: {count}" for ct, count in sorted(type_counts.items()))
            fact = f"Arclane content production patterns: {summary}."
            tags = ["arclane", "content_patterns"]
            kid = self.knowledge_base.add_knowledge(
                content=fact,
                knowledge_type=KnowledgeType.FACTUAL,
                source="arclane:content_patterns",
                confidence=self.confidence,
                context_tags=tags,
            )
            knowledge_ids.append(kid)

        return knowledge_ids

    def collect_all(self) -> Dict[str, Any]:
        """Run all collection methods and return summary statistics."""
        activity_ids = self.collect_recent_activity()
        content_ids = self.collect_content_patterns()

        return {
            "activities_collected": len(activity_ids),
            "content_patterns_collected": len(content_ids),
            "total_knowledge_items": len(activity_ids) + len(content_ids),
            "knowledge_ids": activity_ids + content_ids,
        }
