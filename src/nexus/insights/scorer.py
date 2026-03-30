"""
Trend Scorer - Evaluates trends and matches to blueprints

Takes raw signals and produces actionable recommendations:
1. Score each trend on multiple dimensions
2. Determine lifecycle stage
3. Match to available blueprints
4. Prioritize for production queue
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio
import os
import json
from pathlib import Path

from .models import (
    TrendSignal, TrendAnalysis, TrendCategory, TrendLifecycle, InsightsResult
)


class TrendScorer:
    """
    Scores trends and matches them to blueprints.
    """
    
    def __init__(self, blueprints_path: Optional[Path] = None):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = None
        self._blueprints = None
        self._blueprints_path = blueprints_path
    
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
    
    def _load_blueprints(self) -> Dict[str, Any]:
        """Load available blueprints for matching."""
        if self._blueprints is not None:
            return self._blueprints
        
        # Default path
        if self._blueprints_path is None:
            self._blueprints_path = Path(__file__).parent.parent.parent / "blueprints" / "all_blueprints.json"
        
        if not self._blueprints_path.exists():
            self._blueprints = {}
            return self._blueprints
        
        with open(self._blueprints_path) as f:
            data = json.load(f)
        
        # Handle both list format and dict format
        if isinstance(data, list):
            blueprints_list = data
        else:
            blueprints_list = data.get("blueprints", [])
        
        # Build lookup by ID and keywords
        self._blueprints = {}
        for bp in blueprints_list:
            # Get blueprint meta
            meta = bp.get("blueprint_meta", {})
            bp_id = meta.get("library_id", "")
            
            if not bp_id:
                continue
            
            # Extract keywords from purpose (in executive_summary) and book titles
            keywords = set()
            
            # Get purpose from executive_summary
            exec_summary = bp.get("executive_summary", {})
            purpose = exec_summary.get("purpose", "").lower()
            for word in purpose.split():
                if len(word) > 4:
                    keywords.add(word)
            
            # Also add target audience keywords
            target_audience = exec_summary.get("target_audience", [])
            for audience in target_audience:
                for word in audience.lower().split():
                    if len(word) > 4:
                        keywords.add(word)
            
            # Get books from catalog_baseline.items (not .books)
            catalog = bp.get("catalog_baseline", {})
            books_data = catalog.get("items", [])  # Fixed: items not books
            
            for book in books_data:
                title = book.get("title", "").lower()
                for word in title.split():
                    if len(word) > 4:
                        keywords.add(word)
                
                # Also add primary outcome keywords
                outcome = book.get("primary_outcome", "").lower()
                for word in outcome.split():
                    if len(word) > 4:
                        keywords.add(word)
                
                # Add tags
                for tag in book.get("tags", []):
                    keywords.add(tag.lower())
            
            self._blueprints[bp_id] = {
                "id": bp_id,
                "purpose": purpose,
                "books": books_data,
                "keywords": keywords,
            }
        
        return self._blueprints
    
    def _determine_lifecycle(self, velocity: float) -> TrendLifecycle:
        """Determine lifecycle stage from velocity."""
        if velocity > 50:
            return TrendLifecycle.EMERGING
        elif velocity > 10:
            return TrendLifecycle.GROWING
        elif velocity > -10:
            return TrendLifecycle.STABLE
        elif velocity > -50:
            return TrendLifecycle.DECLINING
        else:
            return TrendLifecycle.DECLINING
    
    def _calculate_overall_score(
        self,
        interest: float,
        velocity: float,
        competition: float,
        revenue: float
    ) -> float:
        """
        Calculate overall score from component scores.
        
        Weights:
        - Interest: 25% (baseline demand)
        - Velocity: 30% (growth potential - most important)
        - Competition: 20% (lower is better)
        - Revenue: 25% (monetization potential)
        """
        # Normalize velocity to 0-1 (input is -100 to +100)
        velocity_normalized = (velocity + 100) / 200
        
        # Invert competition (lower competition = higher score)
        competition_inverted = 1 - (competition / 100)
        
        score = (
            (interest / 100) * 0.25 +
            velocity_normalized * 0.30 +
            competition_inverted * 0.20 +
            (revenue / 100) * 0.25
        )
        
        return round(score, 3)
    
    def _match_blueprint(self, topic: str, category: TrendCategory) -> Tuple[Optional[str], Optional[int]]:
        """
        Match a topic to the best blueprint.
        
        Returns (blueprint_id, book_index) or (None, None)
        """
        blueprints = self._load_blueprints()
        if not blueprints:
            return None, None
        
        topic_words = set(topic.lower().split())
        
        best_match = None
        best_score = 0
        
        for bp_id, bp_data in blueprints.items():
            # Score based on keyword overlap
            overlap = len(topic_words & bp_data["keywords"])
            
            if overlap > best_score:
                best_score = overlap
                best_match = bp_id
        
        if best_match and best_score >= 1:
            # Find best book within blueprint
            bp = blueprints[best_match]
            books = bp.get("books", [])
            
            best_book_idx = 1  # Default to first book
            best_book_score = 0
            
            for idx, book in enumerate(books, 1):
                title_words = set(book.get("title", "").lower().split())
                book_overlap = len(topic_words & title_words)
                if book_overlap > best_book_score:
                    best_book_score = book_overlap
                    best_book_idx = idx
            
            return best_match, best_book_idx
        
        return None, None
    
    async def score_signal(self, signal: TrendSignal) -> TrendAnalysis:
        """Score a single trend signal."""
        
        # Get metadata scores or defaults
        metadata = signal.metadata
        competition = metadata.get("competition", 50)
        revenue_potential = metadata.get("revenue_potential", 50)
        
        # Determine category from metadata or infer
        category_str = metadata.get("category", "other")
        try:
            category = TrendCategory(category_str)
        except ValueError:
            category = TrendCategory.OTHER
        
        # Calculate scores
        interest_score = signal.score / 100
        velocity_score = (signal.velocity + 100) / 200  # Normalize to 0-1
        competition_score = 1 - (competition / 100)  # Invert
        revenue_score = revenue_potential / 100
        
        overall = self._calculate_overall_score(
            signal.score, signal.velocity, competition, revenue_potential
        )
        
        # Determine lifecycle
        lifecycle = self._determine_lifecycle(signal.velocity)
        
        # Match to blueprint
        blueprint_id, book_idx = self._match_blueprint(signal.topic, category)
        
        # Build analysis notes
        notes = []
        if signal.velocity > 30:
            notes.append("Strong growth momentum")
        if competition < 40:
            notes.append("Low competition - good opportunity")
        if revenue_potential > 70:
            notes.append("High revenue potential")
        if metadata.get("pain_points"):
            notes.append(f"Pain points: {', '.join(metadata['pain_points'][:3])}")
        
        return TrendAnalysis(
            topic=signal.topic,
            category=category,
            lifecycle=lifecycle,
            interest_score=round(interest_score, 2),
            velocity_score=round(velocity_score, 2),
            competition_score=round(competition_score, 2),
            revenue_potential=round(revenue_score, 2),
            overall_score=overall,
            signals=[signal],
            recommended_blueprint=blueprint_id,
            recommended_book_index=book_idx,
            analysis_notes=" | ".join(notes) if notes else "Standard opportunity",
        )
    
    async def score_signals(
        self,
        signals: List[TrendSignal],
        top_n: int = 5
    ) -> InsightsResult:
        """Score multiple signals and return prioritized results."""
        
        # Score each signal
        analyses = []
        for signal in signals:
            analysis = await self.score_signal(signal)
            analyses.append(analysis)
        
        # Sort by overall score
        analyses.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Get top N
        top = analyses[:top_n]
        
        return InsightsResult(
            scanned_at=datetime.now(),
            topics_analyzed=len(signals),
            trends=analyses,
            top_recommendations=top,
        )
    
    async def analyze_and_recommend(
        self,
        signals: List[TrendSignal],
        min_score: float = 0.5,
        top_n: int = 3
    ) -> List[TrendAnalysis]:
        """
        Analyze signals and return recommendations that meet threshold.
        
        This is the main entry point for the production queue.
        """
        result = await self.score_signals(signals, top_n=top_n)
        
        # Filter by minimum score
        recommendations = [
            t for t in result.top_recommendations
            if t.overall_score >= min_score
        ]
        
        return recommendations
