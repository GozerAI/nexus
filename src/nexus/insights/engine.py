"""
Insights Engine - Main entry point for trend detection and queue feeding

This is the "sensor" that watches the world and decides what content to create.

Usage:
    engine = InsightsEngine()
    
    # Scan for trends in specific categories
    result = await engine.scan(categories=[TrendCategory.PRODUCTIVITY])
    
    # Get top recommendations for production queue
    recommendations = await engine.get_recommendations(min_score=0.6, top_n=3)
    
    # Analyze a specific topic
    analysis = await engine.analyze_topic("AI automation for small business")
    
    # Full pipeline: scan → score → queue
    queued = await engine.scan_and_queue(auto_queue=True, min_score=0.7)
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import asyncio

from .models import (
    TrendSignal, TrendAnalysis, TrendCategory, TrendLifecycle, InsightsResult
)
from .scanner import TrendScanner
from .scorer import TrendScorer


class InsightsEngine:
    """
    Main insights engine - scans, scores, and queues content opportunities.
    
    Two scan modes:
    1. discover() - RECOMMENDED - Find what's trending everywhere
    2. scan() - Scan specific categories only
    """
    
    def __init__(self, content_orchestrator=None):
        self.scanner = TrendScanner()
        self.scorer = TrendScorer()
        self.orchestrator = content_orchestrator
        self._last_scan: Optional[InsightsResult] = None
    
    async def discover(self, include_seasonal: bool = True) -> InsightsResult:
        """
        DISCOVERY MODE - Find what's actually trending across ALL domains.
        
        This is the recommended scan method. It:
        - Searches for trending topics without category limits
        - Finds audience-specific problems
        - Includes seasonal/timely topics
        - Covers health, finance, lifestyle, career, and more
        
        Returns scored and ranked opportunities.
        """
        # Collect signals via discovery
        signals = await self.scanner.discover_all(include_seasonal=include_seasonal)
        
        # Score and analyze
        result = await self.scorer.score_signals(signals)
        self._last_scan = result
        
        return result
    
    async def scan(
        self,
        categories: List[TrendCategory] = None,
        topics_per_category: int = 5
    ) -> InsightsResult:
        """
        Scan for trending topics across categories.
        
        Args:
            categories: Categories to scan (default: productivity, tech, business)
            topics_per_category: How many topics to find per category
            
        Returns:
            InsightsResult with all trends and top recommendations
        """
        if categories is None:
            categories = [
                TrendCategory.PRODUCTIVITY,
                TrendCategory.TECHNOLOGY,
                TrendCategory.BUSINESS,
            ]
        
        # Collect signals
        signals = await self.scanner.scan_multiple_categories(
            categories=categories,
            count_per_category=topics_per_category
        )
        
        # Score and analyze
        result = await self.scorer.score_signals(signals)
        self._last_scan = result
        
        return result
    
    async def analyze_topic(self, topic: str) -> TrendAnalysis:
        """
        Analyze a specific topic for production potential.
        
        Args:
            topic: The topic to analyze
            
        Returns:
            TrendAnalysis with scores and blueprint recommendation
        """
        signal = await self.scanner.scan_topic(topic)
        analysis = await self.scorer.score_signal(signal)
        return analysis
    
    async def get_recommendations(
        self,
        min_score: float = 0.5,
        top_n: int = 5,
        rescan: bool = False,
        categories: List[TrendCategory] = None
    ) -> List[TrendAnalysis]:
        """
        Get top content recommendations.
        
        Args:
            min_score: Minimum overall score threshold (0-1)
            top_n: Maximum number of recommendations
            rescan: Force a new scan even if recent data exists
            categories: Categories to scan if rescanning
            
        Returns:
            List of TrendAnalysis recommendations
        """
        if rescan or self._last_scan is None:
            await self.scan(categories=categories)
        
        recommendations = [
            t for t in self._last_scan.top_recommendations
            if t.overall_score >= min_score
        ][:top_n]
        
        return recommendations
    
    async def scan_and_queue(
        self,
        categories: List[TrendCategory] = None,
        min_score: float = 0.6,
        max_queue: int = 3,
        auto_queue: bool = False
    ) -> List[TrendAnalysis]:
        """
        Scan for trends and optionally add to production queue.
        
        Args:
            categories: Categories to scan
            min_score: Minimum score to consider
            max_queue: Maximum items to queue
            auto_queue: If True, automatically add to orchestrator queue
            
        Returns:
            List of recommendations (queued if auto_queue=True)
        """
        # Scan and get recommendations
        result = await self.scan(categories=categories)
        
        recommendations = [
            t for t in result.top_recommendations
            if t.overall_score >= min_score
        ][:max_queue]
        
        # Auto-queue if orchestrator available and enabled
        if auto_queue and self.orchestrator and recommendations:
            for rec in recommendations:
                if rec.recommended_blueprint:
                    self.orchestrator.queue_ebook(
                        blueprint_id=rec.recommended_blueprint,
                        book_index=rec.recommended_book_index or 1,
                        provider="anthropic",
                    )
        
        return recommendations
    
    def get_last_scan(self) -> Optional[InsightsResult]:
        """Get the most recent scan result."""
        return self._last_scan
    
    async def quick_analysis(self, topics: List[str]) -> List[TrendAnalysis]:
        """
        Quickly analyze a list of topics.
        
        Useful for evaluating manual topic ideas.
        """
        analyses = []
        for topic in topics:
            analysis = await self.analyze_topic(topic)
            analyses.append(analysis)
        
        # Sort by score
        analyses.sort(key=lambda x: x.overall_score, reverse=True)
        return analyses


# Convenience function
async def scan_trends(
    categories: List[TrendCategory] = None,
    top_n: int = 5
) -> List[TrendAnalysis]:
    """Quick scan for trends - convenience function."""
    engine = InsightsEngine()
    await engine.scan(categories=categories)
    return await engine.get_recommendations(top_n=top_n)
