"""
Insights Engine - Trend detection and topic recommendation

The "eyes and ears" of the system. Scans external signals to find
topics worth creating content about.

Flow:
1. Signal Collection - Gather data from multiple sources
2. Pattern Recognition - Identify trends, velocity, lifecycle stage
3. Revenue Scoring - Estimate monetization potential
4. Blueprint Matching - Recommend which blueprint fits the topic
5. Queue Decision - Add to production queue or not

Sources (in order of implementation priority):
- Google Trends API (free, reliable)
- Perplexity/Claude for real-time analysis
- RSS feeds for news monitoring
- Future: Social APIs, marketplace scrapers
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import os
import json


class TrendLifecycle(Enum):
    """Where a trend is in its lifecycle."""
    EMERGING = "emerging"      # Just starting, high potential
    GROWING = "growing"        # Building momentum
    PEAK = "peak"              # Maximum interest
    DECLINING = "declining"    # Past peak
    STABLE = "stable"          # Consistent interest over time


class TrendCategory(Enum):
    """Content categories for blueprint matching."""
    PRODUCTIVITY = "productivity"
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    HEALTH = "health"
    FINANCE = "finance"
    LIFESTYLE = "lifestyle"
    EDUCATION = "education"
    CAREER = "career"
    MARKETING = "marketing"
    CREATIVITY = "creativity"
    RELATIONSHIPS = "relationships"
    PARENTING = "parenting"
    HOME = "home"
    FOOD = "food"
    TRAVEL = "travel"
    FITNESS = "fitness"
    MENTAL_HEALTH = "mental_health"
    SPIRITUALITY = "spirituality"
    HOBBIES = "hobbies"
    PETS = "pets"
    GAMING = "gaming"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    OTHER = "other"
    
    @classmethod
    def all_categories(cls) -> list:
        """Return all categories except OTHER and NEWS."""
        skip = {cls.OTHER, cls.NEWS}
        return [c for c in cls if c not in skip]
    
    @classmethod
    def core_categories(cls) -> list:
        """Return high-value content categories."""
        return [
            cls.PRODUCTIVITY, cls.TECHNOLOGY, cls.BUSINESS,
            cls.HEALTH, cls.FINANCE, cls.CAREER, cls.MARKETING,
            cls.EDUCATION, cls.MENTAL_HEALTH, cls.FITNESS,
        ]


@dataclass
class TrendSignal:
    """A single trend signal from any source."""
    topic: str
    source: str  # google_trends, perplexity, rss, etc.
    score: float  # 0-100 relative interest
    velocity: float  # Rate of change (-100 to +100)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Analyzed trend with scoring."""
    topic: str
    category: TrendCategory
    lifecycle: TrendLifecycle
    
    # Scores (0-1)
    interest_score: float      # How much interest exists
    velocity_score: float      # How fast it's growing
    competition_score: float   # Lower = less competition = better
    revenue_potential: float   # Estimated monetization potential
    
    # Combined score
    overall_score: float
    
    # Signals that contributed to this analysis
    signals: List[TrendSignal] = field(default_factory=list)
    
    # Blueprint recommendation
    recommended_blueprint: Optional[str] = None
    recommended_book_index: Optional[int] = None
    
    # Reasoning
    analysis_notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "category": self.category.value,
            "lifecycle": self.lifecycle.value,
            "interest_score": self.interest_score,
            "velocity_score": self.velocity_score,
            "competition_score": self.competition_score,
            "revenue_potential": self.revenue_potential,
            "overall_score": self.overall_score,
            "recommended_blueprint": self.recommended_blueprint,
            "recommended_book_index": self.recommended_book_index,
            "analysis_notes": self.analysis_notes,
            "signal_count": len(self.signals),
        }


@dataclass 
class InsightsResult:
    """Result from insights engine scan."""
    scanned_at: datetime
    topics_analyzed: int
    trends: List[TrendAnalysis]
    top_recommendations: List[TrendAnalysis]  # Top N by overall_score
    
    def to_dict(self) -> dict:
        return {
            "scanned_at": self.scanned_at.isoformat(),
            "topics_analyzed": self.topics_analyzed,
            "trends": [t.to_dict() for t in self.trends],
            "top_recommendations": [t.to_dict() for t in self.top_recommendations],
        }
