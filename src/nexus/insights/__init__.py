"""
Insights Engine - Trend detection and content opportunity discovery

Signal Sources:
- Google Trends (real search volume data)
- Perplexity Sonar (real-time web analysis)
- Claude Analysis (AI-powered fallback)

Components:
- TrendScanner: Aggregates signals from multiple sources
- TrendScorer: Evaluates trends and matches to blueprints
- InsightsEngine: Main orchestrator

Usage:
    from insights import InsightsEngine, TrendCategory
    
    engine = InsightsEngine()
    
    # Scan for opportunities (uses real data sources)
    result = await engine.scan()
    
    # Get recommendations
    recs = await engine.get_recommendations(min_score=0.6)
    
    # Analyze specific topic
    analysis = await engine.analyze_topic("AI automation")
"""

from .models import (
    TrendSignal,
    TrendAnalysis,
    TrendCategory,
    TrendLifecycle,
    InsightsResult,
)

from .sources import (
    GoogleTrendsSource,
    PerplexitySource,
    ClaudeAnalysisSource,
)

from .discovery import TrendDiscovery

from .scanner import TrendScanner
from .scorer import TrendScorer
from .engine import InsightsEngine, scan_trends

__all__ = [
    # Models
    "TrendSignal",
    "TrendAnalysis", 
    "TrendCategory",
    "TrendLifecycle",
    "InsightsResult",
    # Sources
    "GoogleTrendsSource",
    "PerplexitySource",
    "ClaudeAnalysisSource",
    # Discovery
    "TrendDiscovery",
    # Components
    "TrendScanner",
    "TrendScorer",
    "InsightsEngine",
    # Convenience
    "scan_trends",
]
