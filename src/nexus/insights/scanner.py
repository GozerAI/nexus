"""
Trend Scanner - Aggregates signals from multiple sources

Modes:
1. DISCOVERY MODE (default) - Find what's actually trending everywhere
2. CATEGORY MODE - Scan specific categories

Sources:
- Perplexity Sonar - Real-time web analysis (primary)
- Google Trends - Search volume data (supplementary)
- Claude Analysis - AI-powered fallback
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import os
import json
import logging

from .models import TrendSignal, TrendCategory
from .sources import GoogleTrendsSource, PerplexitySource, ClaudeAnalysisSource
from .discovery import TrendDiscovery

logger = logging.getLogger(__name__)


class TrendScanner:
    """
    Multi-source trend scanner.
    
    Two modes:
    1. Discovery Mode - Find what's trending across ALL domains
    2. Category Mode - Scan specific categories
    
    Sources:
    - Perplexity (real-time web analysis) - Primary
    - Google Trends (search data) - Supplementary
    - Claude (AI analysis) - Fallback
    """
    
    def __init__(
        self,
        use_google_trends: bool = True,
        use_perplexity: bool = True,
        use_claude: bool = True,
    ):
        self.use_google_trends = use_google_trends
        self.use_perplexity = use_perplexity
        self.use_claude = use_claude
        
        # Initialize sources
        self._google = GoogleTrendsSource() if use_google_trends else None
        self._perplexity = PerplexitySource() if use_perplexity else None
        self._claude = ClaudeAnalysisSource() if use_claude else None
        self._discovery = TrendDiscovery() if use_perplexity else None
    
    async def discover_all(self, include_seasonal: bool = True) -> List[TrendSignal]:
        """
        DISCOVERY MODE - Find what's actually trending, no category limits.
        
        This is the recommended scan mode. It finds:
        - General trending topics across all domains
        - Audience-specific problems people are searching for
        - Seasonal/timely topics (optional)
        
        Returns deduplicated, scored signals.
        """
        if not self._discovery:
            logger.warning("Discovery requires Perplexity - falling back to category scan")
            return await self.scan_all_categories()
        
        all_signals = []
        
        # Full discovery scan
        logger.info("Running full discovery scan...")
        discovery_signals = await self._discovery.full_discovery_scan()
        all_signals.extend(discovery_signals)
        
        # Optionally add Google Trends daily
        if self._google:
            try:
                logger.info("Adding Google Trends daily trending...")
                daily = await self._google.get_trending_searches()
                all_signals.extend(daily[:10])  # Top 10 daily
            except Exception as e:
                logger.warning(f"Google Trends daily failed: {e}")
        
        # Deduplicate
        deduped = self._deduplicate_signals(all_signals)
        
        logger.info(f"Discovery complete: {len(all_signals)} raw, {len(deduped)} after dedup")
        
        return deduped
    
    async def scan_all_categories(self, topics_per_category: int = 3) -> List[TrendSignal]:
        """
        Scan ALL defined categories, not just a subset.
        """
        categories = TrendCategory.all_categories()
        logger.info(f"Scanning {len(categories)} categories...")
        
        return await self.scan_multiple_categories(
            categories=categories,
            count_per_category=topics_per_category
        )
    
    async def scan_category(
        self, 
        category: TrendCategory,
        count: int = 10
    ) -> List[TrendSignal]:
        """
        Scan for trending topics in a category using all available sources.
        """
        all_signals = []
        source_results = {}
        
        # 1. Google Trends - Rising queries for category seeds
        if self._google:
            try:
                logger.info(f"Scanning Google Trends for {category.value}...")
                google_signals = await self._google.scan_category_seeds(category)
                all_signals.extend(google_signals)
                source_results["google_trends"] = len(google_signals)
                logger.info(f"  Found {len(google_signals)} signals from Google Trends")
            except Exception as e:
                logger.error(f"Google Trends error: {e}")
                source_results["google_trends"] = 0
        
        # 2. Perplexity - Real-time web analysis
        if self._perplexity and os.getenv("PERPLEXITY_API_KEY"):
            try:
                logger.info(f"Scanning Perplexity for {category.value}...")
                perplexity_signals = await self._perplexity.search_trends(category, count)
                all_signals.extend(perplexity_signals)
                source_results["perplexity"] = len(perplexity_signals)
                logger.info(f"  Found {len(perplexity_signals)} signals from Perplexity")
            except Exception as e:
                logger.error(f"Perplexity error: {e}")
                source_results["perplexity"] = 0
        
        # 3. Claude Analysis - Fallback/supplement
        if self._claude:
            # Only use Claude if we got few signals from real sources
            real_signal_count = sum(source_results.get(k, 0) for k in ["google_trends", "perplexity"])
            
            if real_signal_count < 5:
                try:
                    logger.info(f"Supplementing with Claude analysis...")
                    claude_signals = await self._claude.analyze_category(category, count)
                    all_signals.extend(claude_signals)
                    source_results["claude"] = len(claude_signals)
                    logger.info(f"  Found {len(claude_signals)} signals from Claude")
                except Exception as e:
                    logger.error(f"Claude error: {e}")
                    source_results["claude"] = 0
        
        # Deduplicate by topic similarity
        deduped = self._deduplicate_signals(all_signals)
        
        logger.info(f"Total: {len(all_signals)} signals, {len(deduped)} after dedup")
        
        return deduped[:count * 2]  # Return more than requested for scoring
    
    async def scan_topic(self, topic: str) -> TrendSignal:
        """
        Analyze a specific topic using multiple sources.
        """
        signals = []
        
        # Google Trends interest data
        if self._google:
            try:
                interest_data = await self._google.get_interest_over_time([topic])
                if topic in interest_data:
                    data = interest_data[topic]
                    signals.append(TrendSignal(
                        topic=topic,
                        source="google_trends",
                        score=data["current_interest"],
                        velocity=data["velocity"],
                        metadata={
                            "average_interest": data["average_interest"],
                            "peak": data["peak"],
                            "data_points": data["data_points"],
                        }
                    ))
            except Exception as e:
                logger.error(f"Google Trends topic error: {e}")
        
        # Perplexity deep analysis
        if self._perplexity and os.getenv("PERPLEXITY_API_KEY"):
            try:
                perplexity_signal = await self._perplexity.analyze_topic(topic)
                signals.append(perplexity_signal)
            except Exception as e:
                logger.error(f"Perplexity topic error: {e}")
        
        # Combine signals into one
        if signals:
            return self._merge_topic_signals(topic, signals)
        
        # Fallback to Claude
        if self._claude:
            claude_signals = await self._claude.analyze_category(
                TrendCategory.OTHER, 
                count=1
            )
            if claude_signals:
                return claude_signals[0]
        
        # Default signal
        return TrendSignal(
            topic=topic,
            source="default",
            score=50,
            velocity=0,
            metadata={"note": "No data available"}
        )
    
    async def get_daily_trending(self, country: str = 'united_states') -> List[TrendSignal]:
        """Get today's trending searches from Google."""
        if self._google:
            return await self._google.get_trending_searches(country)
        return []
    
    async def scan_multiple_categories(
        self,
        categories: List[TrendCategory] = None,
        count_per_category: int = 5
    ) -> List[TrendSignal]:
        """Scan multiple categories."""
        
        if categories is None:
            categories = [
                TrendCategory.PRODUCTIVITY,
                TrendCategory.TECHNOLOGY,
                TrendCategory.BUSINESS,
            ]
        
        all_signals = []
        
        for category in categories:
            signals = await self.scan_category(category, count_per_category)
            all_signals.extend(signals)
            await asyncio.sleep(0.5)  # Rate limit protection
        
        return all_signals
    
    def _deduplicate_signals(self, signals: List[TrendSignal]) -> List[TrendSignal]:
        """Remove duplicate/similar topics, keeping highest scored."""
        
        # Filter out obvious noise (news headlines, questions, etc.)
        filtered_signals = []
        for signal in signals:
            topic_lower = signal.topic.lower()
            
            # Skip noise patterns
            skip = False
            noise_patterns = [
                "what time", "what is", "how to", "where is", "when is",
                "news today", "news yesterday", 
                "slave", "quotes",  # Random philosophical stuff
                "weather", "sunset", "sunrise",
            ]
            for pattern in noise_patterns:
                if pattern in topic_lower:
                    skip = True
                    break
            
            # Skip very short or very long topics
            if len(signal.topic) < 10 or len(signal.topic) > 80:
                skip = True
            
            if not skip:
                filtered_signals.append(signal)
        
        signals = filtered_signals
        
        # Group by normalized topic
        topic_groups: Dict[str, List[TrendSignal]] = {}
        
        for signal in signals:
            # Normalize topic for comparison
            normalized = signal.topic.lower().strip()
            # Remove common words for better matching
            for word in ["the", "a", "an", "for", "to", "in", "of", "and"]:
                normalized = normalized.replace(f" {word} ", " ")
            normalized = " ".join(normalized.split())  # Normalize whitespace
            
            # Find similar existing group
            matched = False
            for key in topic_groups:
                # Simple similarity: check if one contains the other
                if normalized in key or key in normalized:
                    topic_groups[key].append(signal)
                    matched = True
                    break
                # Check word overlap
                key_words = set(key.split())
                norm_words = set(normalized.split())
                overlap = len(key_words & norm_words) / max(len(key_words), len(norm_words))
                if overlap > 0.6:
                    topic_groups[key].append(signal)
                    matched = True
                    break
            
            if not matched:
                topic_groups[normalized] = [signal]
        
        # Keep best signal from each group
        deduped = []
        for signals in topic_groups.values():
            # Sort by source quality, then score
            # Perplexity has better context, Google Trends has volume data
            def sort_key(s):
                source_priority = {
                    "perplexity_realtime": 5,   # Best - real-time with context
                    "perplexity_deep": 4,       # Good - detailed analysis
                    "google_trends_rising": 3,  # Good - actual search data
                    "google_trends": 2,         # OK - raw interest
                    "google_trends_daily": 1,   # Noisy - daily trending
                    "claude_analysis": 2,       # OK - AI analysis
                }
                return (source_priority.get(s.source, 0), s.score)
            
            signals.sort(key=sort_key, reverse=True)
            best = signals[0]
            
            # Merge metadata from other signals
            all_sources = list(set(s.source for s in signals))
            best.metadata["all_sources"] = all_sources
            best.metadata["signal_count"] = len(signals)
            
            deduped.append(best)
        
        # Sort by score
        deduped.sort(key=lambda s: s.score, reverse=True)
        
        return deduped
    
    def _merge_topic_signals(self, topic: str, signals: List[TrendSignal]) -> TrendSignal:
        """Merge multiple signals for the same topic into one."""
        
        # Weight by source reliability
        weights = {
            "google_trends": 1.5,
            "perplexity_deep": 1.3,
            "perplexity_realtime": 1.2,
            "claude_analysis": 0.8,
        }
        
        total_weight = 0
        weighted_score = 0
        weighted_velocity = 0
        all_metadata = {}
        sources = []
        
        for signal in signals:
            weight = weights.get(signal.source, 1.0)
            total_weight += weight
            weighted_score += signal.score * weight
            weighted_velocity += signal.velocity * weight
            sources.append(signal.source)
            all_metadata[signal.source] = signal.metadata
        
        return TrendSignal(
            topic=topic,
            source="merged",
            score=weighted_score / total_weight if total_weight > 0 else 50,
            velocity=weighted_velocity / total_weight if total_weight > 0 else 0,
            metadata={
                "sources": sources,
                "source_data": all_metadata,
            }
        )
