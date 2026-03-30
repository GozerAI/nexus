"""
KH Graph Collector for Nexus AI Platform.

Fetches intelligence graph clusters and coverage gaps from the
Knowledge Harvester API and creates knowledge items for Nexus.

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
    _kh_cb = get_circuit_breaker("kh", failure_threshold=3, recovery_timeout=120)
except ImportError:
    _HAS_RESILIENCE = False
    _kh_cb = None

logger = logging.getLogger(__name__)

DEFAULT_KH_BASE_URL = "http://localhost:8011"
KH_TIMEOUT = 5


class KHGraphCollector:
    """Collects knowledge from the Knowledge Harvester graph and coverage APIs.

    Fetches cluster and gap data, then creates Nexus knowledge items
    describing the state of the intelligence graph.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        kh_base_url: Optional[str] = None,
        confidence: float = 0.75,
    ):
        self.knowledge_base = knowledge_base
        self.kh_base_url = kh_base_url or os.environ.get(
            "KH_BASE_URL", DEFAULT_KH_BASE_URL
        )
        self.confidence = confidence

    # ─── HTTP helpers ─────────────────────────────────────────────

    def _fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        """GET *url* and return parsed JSON, or None on failure."""
        if _HAS_RESILIENCE:
            return resilient_fetch(
                url, headers={"Accept": "application/json"},
                timeout=float(KH_TIMEOUT), retry_policy=DEFAULT_RETRY, circuit_breaker=_kh_cb,
            )
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=KH_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, OSError) as exc:
            logger.warning("KHGraphCollector: failed to fetch %s: %s", url, exc)
            return None
        except Exception as exc:
            logger.warning("KHGraphCollector: unexpected error fetching %s: %s", url, exc)
            return None

    # ─── Collection methods ───────────────────────────────────────

    def collect_clusters(self) -> List[str]:
        """Fetch cluster data from KH and create knowledge items.

        Returns:
            List of knowledge IDs created.
        """
        data = self._fetch_json(f"{self.kh_base_url}/api/discover/clusters")
        if data is None:
            return []

        knowledge_ids: List[str] = []
        clusters = data.get("clusters", [])
        for cluster in clusters:
            name = cluster.get("name", "unknown")
            size = cluster.get("size", 0)
            categories = cluster.get("categories", [])

            fact = (
                f"Intelligence graph cluster '{name}' contains {size} nodes"
                f" spanning categories: {', '.join(categories[:5])}."
            )
            tags = ["kh_graph", "cluster", name.lower().replace(" ", "_")]
            kid = self.knowledge_base.add_knowledge(
                content=fact,
                knowledge_type=KnowledgeType.FACTUAL,
                source="kh_graph:cluster",
                confidence=self.confidence,
                context_tags=tags,
            )
            knowledge_ids.append(kid)

        return knowledge_ids

    def collect_coverage_gaps(self) -> List[str]:
        """Fetch coverage gap data from KH and create knowledge items.

        Returns:
            List of knowledge IDs created.
        """
        data = self._fetch_json(f"{self.kh_base_url}/api/coverage/gaps")
        if data is None:
            return []

        knowledge_ids: List[str] = []
        gaps = data.get("gaps", [])
        for gap in gaps:
            category = gap.get("category", "unknown")
            severity = gap.get("severity", "unknown")
            artifact_count = gap.get("artifact_count", 0)

            fact = (
                f"Coverage gap detected in category '{category}' "
                f"(severity: {severity}, current artifacts: {artifact_count})."
            )
            tags = ["kh_graph", "coverage_gap", category.lower().replace(" ", "_")]
            kid = self.knowledge_base.add_knowledge(
                content=fact,
                knowledge_type=KnowledgeType.FACTUAL,
                source="kh_graph:coverage_gap",
                confidence=self.confidence,
                context_tags=tags,
            )
            knowledge_ids.append(kid)

        return knowledge_ids

    def collect_all(self) -> Dict[str, Any]:
        """Run all collection methods and return summary statistics.

        Returns:
            Dict with collection stats.
        """
        cluster_ids = self.collect_clusters()
        gap_ids = self.collect_coverage_gaps()

        return {
            "clusters_collected": len(cluster_ids),
            "gaps_collected": len(gap_ids),
            "total_knowledge_items": len(cluster_ids) + len(gap_ids),
            "knowledge_ids": cluster_ids + gap_ids,
        }
