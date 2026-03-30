"""
Signal Sources - Real data sources for trend detection

Sources:
- Google Trends (pytrends) - Search volume data
- Perplexity Sonar - Real-time web analysis
- Claude Analysis - AI-powered trend identification (fallback)

Each source returns TrendSignal objects that get aggregated and scored.
"""

import asyncio
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .models import TrendSignal, TrendCategory

logger = logging.getLogger(__name__)


class GoogleTrendsSource:
    """
    Google Trends data via pytrends library.
    
    Provides:
    - Search interest over time
    - Related queries
    - Rising topics
    - Regional interest
    """
    
    def __init__(self):
        self._pytrends = None
    
    def _get_client(self):
        if self._pytrends is None:
            from pytrends.request import TrendReq
            # Simple init - pytrends handles timeouts internally
            self._pytrends = TrendReq(
                hl='en-US', 
                tz=360,
                timeout=(10, 30),  # (connect, read) timeouts
            )
        return self._pytrends
    
    async def get_trending_searches(self, country: str = 'united_states') -> List[TrendSignal]:
        """Get today's trending searches."""
        try:
            pytrends = self._get_client()
            
            # Run in executor since pytrends is sync
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(
                None,
                lambda: pytrends.trending_searches(pn=country)
            )
            
            signals = []
            for idx, row in df.iterrows():
                topic = row[0] if isinstance(row, (list, tuple)) else str(row.iloc[0]) if hasattr(row, 'iloc') else str(row)
                signals.append(TrendSignal(
                    topic=topic,
                    source="google_trends_daily",
                    score=80,  # Trending = high interest
                    velocity=50,  # Assume growing since it's trending
                    metadata={"country": country, "rank": idx + 1}
                ))
            
            return signals[:20]  # Top 20
            
        except Exception as e:
            logger.error(f"Google Trends daily error: {e}")
            return []
    
    async def get_interest_over_time(
        self, 
        keywords: List[str],
        timeframe: str = 'today 3-m'
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get interest over time for specific keywords.
        
        Returns dict with interest data and velocity calculation.
        """
        try:
            pytrends = self._get_client()
            
            loop = asyncio.get_event_loop()
            
            # Build payload
            await loop.run_in_executor(
                None,
                lambda: pytrends.build_payload(keywords[:5], timeframe=timeframe)  # Max 5 keywords
            )
            
            # Get interest over time
            df = await loop.run_in_executor(
                None,
                lambda: pytrends.interest_over_time()
            )
            
            if df.empty:
                return {}
            
            results = {}
            for keyword in keywords[:5]:
                if keyword in df.columns:
                    values = df[keyword].tolist()
                    
                    # Calculate metrics
                    current = values[-1] if values else 0
                    avg = sum(values) / len(values) if values else 0
                    
                    # Calculate velocity (trend direction)
                    if len(values) >= 4:
                        recent = sum(values[-4:]) / 4
                        older = sum(values[:4]) / 4
                        velocity = ((recent - older) / older * 100) if older > 0 else 0
                    else:
                        velocity = 0
                    
                    results[keyword] = {
                        "current_interest": current,
                        "average_interest": avg,
                        "velocity": velocity,
                        "peak": max(values) if values else 0,
                        "data_points": len(values),
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Google Trends interest error: {e}")
            return {}
    
    async def get_related_queries(self, keyword: str) -> List[TrendSignal]:
        """Get rising related queries for a keyword."""
        try:
            pytrends = self._get_client()
            
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                None,
                lambda: pytrends.build_payload([keyword], timeframe='today 3-m')
            )
            
            related = await loop.run_in_executor(
                None,
                lambda: pytrends.related_queries()
            )
            
            signals = []
            
            if keyword in related and related[keyword]['rising'] is not None:
                rising_df = related[keyword]['rising']
                for idx, row in rising_df.iterrows():
                    query = row['query']
                    value = row['value']  # This is % increase
                    
                    # Filter out noise - skip if it looks like news or questions
                    query_lower = query.lower()
                    if any(x in query_lower for x in [
                        'news', 'today', 'yesterday', 'what ', 'how ', 'where ', 'when ',
                        'yojana', 'scheme', '2026', '2025'  # Government schemes
                    ]):
                        continue
                    
                    # Cap the score - rising queries tend to have inflated values
                    capped_score = min(value / 20, 80)  # More conservative
                    
                    signals.append(TrendSignal(
                        topic=query,
                        source="google_trends_rising",
                        score=capped_score,
                        velocity=min(value / 10, 80),  # Cap velocity too
                        metadata={
                            "parent_keyword": keyword,
                            "rise_percentage": value,
                        }
                    ))
            
            return signals[:10]
            
        except Exception as e:
            logger.error(f"Google Trends related error: {e}")
            return []
    
    async def scan_category_seeds(
        self, 
        category: TrendCategory,
        seed_keywords: List[str] = None
    ) -> List[TrendSignal]:
        """
        Scan a category using seed keywords to find related rising topics.
        """
        # Default seed keywords per category
        default_seeds = {
            TrendCategory.PRODUCTIVITY: ["productivity", "time management", "work from home", "AI tools"],
            TrendCategory.TECHNOLOGY: ["artificial intelligence", "automation", "software tools", "tech trends"],
            TrendCategory.BUSINESS: ["small business", "entrepreneurship", "side hustle", "online business"],
            TrendCategory.HEALTH: ["mental health", "fitness", "wellness", "self care"],
            TrendCategory.FINANCE: ["investing", "personal finance", "passive income", "budgeting"],
            TrendCategory.LIFESTYLE: ["minimalism", "self improvement", "habits", "life hacks"],
            TrendCategory.EDUCATION: ["online learning", "skills", "career development", "courses"],
        }
        
        seeds = seed_keywords or default_seeds.get(category, ["trending"])
        
        all_signals = []
        
        # Only use first seed to avoid rate limits
        for seed in seeds[:1]:
            signals = await self.get_related_queries(seed)
            for s in signals:
                s.metadata["category"] = category.value
            all_signals.extend(signals)
            await asyncio.sleep(3)  # Increased delay for rate limit protection
        
        return all_signals


class PerplexitySource:
    """
    Perplexity Sonar for real-time web analysis.
    
    Searches the web in real-time to find:
    - Current discussions about topics
    - Recent news and articles
    - Emerging trends
    """
    
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
    
    async def search_trends(
        self, 
        category: TrendCategory,
        count: int = 10
    ) -> List[TrendSignal]:
        """
        Search for current trends in a category using Perplexity.
        """
        if not self.api_key:
            logger.warning("PERPLEXITY_API_KEY not set")
            return []
        
        import httpx
        
        prompt = f"""What are the most discussed and emerging topics in {category.value} right now (December 2024/January 2025)?

Focus on:
- Topics getting significant online discussion
- Emerging trends that are growing
- Pain points people are actively seeking solutions for
- Topics suitable for educational content (ebooks, guides, courses)

For each topic, provide:
1. The specific topic name
2. Why it's trending NOW (recent events, changes, or growing interest)
3. Evidence of interest (mentions, searches, discussions)

List {count} specific, actionable topics. Be specific, not generic."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a trend researcher. Identify specific, current trends with evidence. Be concrete and specific, not generic."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 2000
                    }
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    logger.warning("Perplexity returned no choices")
                    return []
                content = choices[0].get("message", {}).get("content", "")
                citations = data.get("citations", [])
                
                # Parse the response to extract topics
                signals = await self._parse_trend_response(content, category, citations)
                return signals
                
        except Exception as e:
            logger.error(f"Perplexity search error: {e}")
            return []
    
    async def analyze_topic(self, topic: str) -> TrendSignal:
        """
        Deep analysis of a specific topic's trend status.
        """
        if not self.api_key:
            return TrendSignal(
                topic=topic,
                source="perplexity",
                score=50,
                velocity=0,
                metadata={"error": "API key not set"}
            )
        
        import httpx
        
        prompt = f"""Analyze the current trend status of "{topic}":

1. Current Interest Level (0-100): How much are people searching/discussing this?
2. Growth Velocity: Is interest increasing, stable, or decreasing? By how much?
3. Recent Developments: What's happened in the last 1-3 months related to this?
4. Competition Level: How saturated is the content/product market?
5. Revenue Potential: Can this be monetized through digital products?

Provide specific data and evidence where possible."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "max_tokens": 1500
                    }
                )
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    logger.warning("Perplexity analyze returned no choices")
                    return TrendSignal(
                        topic=topic, source="perplexity", score=50, velocity=0,
                        metadata={"error": "No choices in response"},
                    )
                content = choices[0].get("message", {}).get("content", "")

                # Extract metrics from response
                metrics = await self._extract_metrics(content)
                
                return TrendSignal(
                    topic=topic,
                    source="perplexity_deep",
                    score=metrics.get("interest", 50),
                    velocity=metrics.get("velocity", 0),
                    metadata={
                        "analysis": content[:500],
                        "competition": metrics.get("competition", 50),
                        "revenue_potential": metrics.get("revenue", 50),
                    }
                )
                
        except Exception as e:
            logger.error(f"Perplexity analyze error: {e}")
            return TrendSignal(
                topic=topic,
                source="perplexity",
                score=50,
                velocity=0,
                metadata={"error": str(e)}
            )
    
    async def _parse_trend_response(
        self, 
        content: str, 
        category: TrendCategory,
        citations: List[str]
    ) -> List[TrendSignal]:
        """Parse Perplexity response into TrendSignals."""
        # Use Claude to structure the response
        import anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return []
        
        client = anthropic.Anthropic(api_key=api_key)
        
        parse_prompt = f"""Extract trending topics from this research:

{content}

Return as JSON array:
[
    {{
        "topic": "Specific topic name",
        "score": 70,
        "velocity": 30,
        "reasoning": "Why it's trending",
        "evidence": "Specific evidence mentioned"
    }}
]

Extract up to 10 topics. Score based on evidence strength (0-100). 
Velocity: positive = growing, negative = declining.
Return ONLY valid JSON."""

        try:
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": parse_prompt}]
                )
            )
            
            if not message.content:
                logger.warning("Claude returned empty content for trend parsing")
                return []
            response_text = message.content[0].text
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            topics = json.loads(response_text.strip())
            
            signals = []
            for t in topics:
                signals.append(TrendSignal(
                    topic=t.get("topic", ""),
                    source="perplexity_realtime",
                    score=float(t.get("score", 50)),
                    velocity=float(t.get("velocity", 0)),
                    metadata={
                        "reasoning": t.get("reasoning", ""),
                        "evidence": t.get("evidence", ""),
                        "category": category.value,
                        "citations": citations[:3],
                    }
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return []
    
    async def _extract_metrics(self, content: str) -> Dict[str, float]:
        """Extract numerical metrics from analysis text."""
        # Simple extraction - look for numbers near keywords
        import re
        
        metrics = {
            "interest": 50,
            "velocity": 0,
            "competition": 50,
            "revenue": 50,
        }
        
        # Look for percentage or score patterns
        content_lower = content.lower()
        
        # Interest level
        interest_match = re.search(r'interest[^\d]*(\d+)', content_lower)
        if interest_match:
            metrics["interest"] = min(int(interest_match.group(1)), 100)
        
        # Growth indicators
        if any(word in content_lower for word in ["rapidly growing", "surging", "explosive", "significant increase"]):
            metrics["velocity"] = 50
        elif any(word in content_lower for word in ["growing", "increasing", "rising"]):
            metrics["velocity"] = 25
        elif any(word in content_lower for word in ["stable", "steady"]):
            metrics["velocity"] = 0
        elif any(word in content_lower for word in ["declining", "decreasing", "falling"]):
            metrics["velocity"] = -25
        
        # Competition
        if any(word in content_lower for word in ["highly competitive", "saturated", "crowded"]):
            metrics["competition"] = 80
        elif any(word in content_lower for word in ["competitive"]):
            metrics["competition"] = 60
        elif any(word in content_lower for word in ["low competition", "underserved", "gap"]):
            metrics["competition"] = 30
        
        # Revenue potential
        if any(word in content_lower for word in ["high potential", "lucrative", "profitable"]):
            metrics["revenue"] = 80
        elif any(word in content_lower for word in ["monetiz", "revenue", "market"]):
            metrics["revenue"] = 60
        
        return metrics


class ClaudeAnalysisSource:
    """
    Claude-based trend analysis (fallback/supplementary).
    
    Uses Claude's knowledge to identify trends when real-time
    sources are unavailable or to supplement real data.
    """
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client
    
    async def analyze_category(
        self, 
        category: TrendCategory,
        count: int = 10
    ) -> List[TrendSignal]:
        """Identify trends using Claude's knowledge."""
        
        prompt = f"""Identify {count} trending topics in {category.value} that would make good digital product content.

For each, assess:
1. Interest level (0-100)
2. Growth velocity (-100 to +100)
3. Why it's relevant now

Focus on specific, actionable topics with clear audience pain points.

Return as JSON array:
[{{"topic": "name", "score": 75, "velocity": 35, "reasoning": "why", "audience": "who", "pain_points": ["p1", "p2"]}}]

Return ONLY valid JSON."""

        try:
            client = self._get_client()
            
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            
            content = message.content[0].text
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            topics = json.loads(content.strip())
            
            signals = []
            for t in topics:
                signals.append(TrendSignal(
                    topic=t.get("topic", ""),
                    source="claude_analysis",
                    score=float(t.get("score", 50)),
                    velocity=float(t.get("velocity", 0)),
                    metadata={
                        "reasoning": t.get("reasoning", ""),
                        "audience": t.get("audience", ""),
                        "pain_points": t.get("pain_points", []),
                        "category": category.value,
                    }
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Claude analysis error: {e}")
            return []
