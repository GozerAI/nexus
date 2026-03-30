"""
Trend Signal Collector for Nexus AI Platform.

Fetches top trends, signals, and lifecycle distribution from the
Trendscope API and creates knowledge items for Nexus.

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
    _ts_cb = get_circuit_breaker("trendscope", failure_threshold=3, recovery_timeout=120)
except ImportError:
    _HAS_RESILIENCE = False
    _ts_cb = None

logger = logging.getLogger(__name__)

DEFAULT_TS_BASE_URL = "http://localhost:8002"
TS_TIMEOUT = 5


class TrendSignalCollector:
    """Collects knowledge from the Trendscope trend and signal APIs.

    Fetches top trends, actionable signals, and lifecycle distribution,
    then creates Nexus knowledge items describing market intelligence.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        ts_base_url: Optional[str] = None,
        bearer_token: Optional[str] = None,
        confidence: float = 0.75,
    ):
        self.knowledge_base = knowledge_base
        self.ts_base_url = ts_base_url or os.environ.get(
            "TRENDSCOPE_BASE_URL", DEFAULT_TS_BASE_URL
        )
        self.bearer_token = bearer_token or os.environ.get(
            "TRENDSCOPE_API_TOKEN", ""
        )
        self.confidence = confidence

    # ─── HTTP helpers ─────────────────────────────────────────────

    def _fetch_json(self, url: str) -> Optional[Any]:
        """GET *url* with auth and return parsed JSON, or None on failure."""
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.bearer_token}",
        }
        if _HAS_RESILIENCE:
            return resilient_fetch(
                url, headers=headers,
                timeout=float(TS_TIMEOUT), retry_policy=DEFAULT_RETRY, circuit_breaker=_ts_cb,
            )
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=TS_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, OSError) as exc:
            logger.warning("TrendSignalCollector: failed to fetch %s: %s", url, exc)
            return None
        except Exception as exc:
            logger.warning("TrendSignalCollector: unexpected error fetching %s: %s", url, exc)
            return None

    # ─── Collection methods ───────────────────────────────────────

    def collect_top_trends(self) -> List[str]:
        """Fetch top trends from Trendscope and create knowledge items.

        Returns:
            List of knowledge IDs created.
        """
        data = self._fetch_json(f"{self.ts_base_url}/v1/trends/top?limit=10")
        if data is None:
            return []

        knowledge_ids: List[str] = []
        for trend in data:
            name = trend.get("name", "unknown")
            score = trend.get("score", 0)
            velocity = trend.get("velocity", 0)
            category = trend.get("category", "unknown")
            status = trend.get("status", "unknown")

            fact = (
                f"Market trend '{name}' (category: {category}) has score "
                f"{score} with velocity {velocity}, status: {status}."
            )
            tags = ["trendscope", "trend", category.lower()]
            kid = self.knowledge_base.add_knowledge(
                content=fact,
                knowledge_type=KnowledgeType.FACTUAL,
                source="trendscope:top_trend",
                confidence=self.confidence,
                context_tags=tags,
            )
            knowledge_ids.append(kid)

        return knowledge_ids

    def collect_signals(self) -> List[str]:
        """Fetch signals from Trendscope and create knowledge items.

        Only processes strong_buy and buy signals (skips hold/sell/strong_sell).

        Returns:
            List of knowledge IDs created.
        """
        data = self._fetch_json(f"{self.ts_base_url}/v1/signals")
        if data is None:
            return []

        knowledge_ids: List[str] = []
        for bucket in ("strong_buy", "buy"):
            items = data.get(bucket, [])
            for item in items:
                name = item.get("name", "unknown")
                score = item.get("score", 0)
                velocity = item.get("velocity", 0)
                momentum = item.get("momentum", 0)
                category = item.get("category", "unknown")
                signal = item.get("signal", bucket)

                fact = (
                    f"Trendscope signal {signal} for '{name}' "
                    f"(category: {category}): score={score}, "
                    f"velocity={velocity}, momentum={momentum}."
                )
                tags = ["trendscope", "signal", signal.lower(), category.lower()]
                kid = self.knowledge_base.add_knowledge(
                    content=fact,
                    knowledge_type=KnowledgeType.FACTUAL,
                    source="trendscope:signal",
                    confidence=self.confidence,
                    context_tags=tags,
                )
                knowledge_ids.append(kid)

        return knowledge_ids

    def collect_lifecycle_distribution(self) -> List[str]:
        """Fetch lifecycle distribution from Trendscope and create a knowledge item.

        Returns:
            List of knowledge IDs created (0 or 1).
        """
        data = self._fetch_json(f"{self.ts_base_url}/v1/lifecycle/distribution")
        if data is None:
            return []

        if not data:
            return []

        summary = ", ".join(f"{stage}: {count}" for stage, count in data.items())
        fact = f"Trendscope lifecycle distribution: {summary}."
        tags = ["trendscope", "lifecycle"]
        kid = self.knowledge_base.add_knowledge(
            content=fact,
            knowledge_type=KnowledgeType.FACTUAL,
            source="trendscope:lifecycle",
            confidence=self.confidence,
            context_tags=tags,
        )
        return [kid]

    def collect_all(self) -> Dict[str, Any]:
        """Run all collection methods and return summary statistics.

        Returns:
            Dict with collection stats.
        """
        trend_ids = self.collect_top_trends()
        signal_ids = self.collect_signals()
        lifecycle_ids = self.collect_lifecycle_distribution()

        return {
            "trends_collected": len(trend_ids),
            "signals_collected": len(signal_ids),
            "lifecycle_collected": len(lifecycle_ids),
            "total_knowledge_items": len(trend_ids) + len(signal_ids) + len(lifecycle_ids),
            "knowledge_ids": trend_ids + signal_ids + lifecycle_ids,
        }
