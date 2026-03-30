"""
Trend Discovery - Find what's actually trending, not just what we expect

Instead of only scanning predefined categories, this module:
1. Gets TODAY's trending searches from Google
2. Asks Perplexity what topics are hot RIGHT NOW across all domains
3. Discovers emerging categories we might not have thought of
4. Scans ALL defined categories, not just a subset
"""

import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import TrendSignal, TrendCategory

logger = logging.getLogger(__name__)


class TrendDiscovery:
    """
    Discovery-first trend detection.
    
    Instead of "what's trending in productivity?", asks:
    "What is trending RIGHT NOW that people would pay to learn about?"
    """
    
    def __init__(self):
        self.perplexity_key = os.getenv("PERPLEXITY_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    async def discover_trending_topics(self, count: int = 20) -> List[TrendSignal]:
        """
        Discover what's actually trending right now - no category limits.
        
        Uses Perplexity to search real-time web for:
        - What people are searching for
        - What's being discussed on social media
        - Emerging problems people need solutions for
        """
        if not self.perplexity_key:
            logger.warning("PERPLEXITY_API_KEY not set")
            return []
        
        import httpx
        
        prompt = f"""What are the {count} most significant trending topics RIGHT NOW (late December 2024) that would make excellent digital products (ebooks, courses, guides)?

Search across ALL domains - not just tech or business. Include:
- Health and wellness trends
- Financial concerns people have
- Lifestyle changes happening
- Skills people want to learn
- Problems people are actively trying to solve
- Emerging technologies affecting everyday life
- Cultural shifts and new behaviors
- Seasonal/timely topics

For each topic provide:
1. The specific topic (be precise, not generic)
2. WHY it's trending now (recent trigger/event/shift)
3. Who cares about this (target audience)
4. The pain point or desire driving interest
5. Estimated interest level (0-100)

Focus on topics where people would PAY for a solution - not just news or entertainment."""

        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a trend researcher finding monetizable topics. Be specific and current. Include topics from health, finance, career, relationships, hobbies - not just tech."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.4,
                        "max_tokens": 3000
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                
                # Parse into signals
                signals = await self._parse_discovery_response(content)
                logger.info(f"Discovered {len(signals)} trending topics")
                return signals
                
        except Exception as e:
            logger.error(f"Discovery error: {e}")
            return []
    
    async def discover_by_audience(self, audiences: List[str] = None) -> List[TrendSignal]:
        """
        Discover trends by target audience segments.
        
        Instead of categories, think about WHO is searching:
        - Remote workers
        - Parents
        - Retirees
        - Students
        - Small business owners
        - etc.
        """
        if audiences is None:
            audiences = [
                "remote workers and digital nomads",
                "parents with young children",
                "people considering career changes",
                "small business owners and solopreneurs",
                "people focused on health and weight loss",
                "investors and people building wealth",
                "people learning new skills",
                "homeowners and renters",
            ]
        
        if not self.perplexity_key:
            return []
        
        import httpx
        
        all_signals = []
        
        for audience in audiences[:4]:  # Limit to avoid rate limits
            prompt = f"""What are the TOP 5 problems or goals that {audience} are actively searching for solutions to RIGHT NOW?

Be specific about:
1. The exact problem or goal
2. Why it's urgent now
3. What solution they're looking for
4. Interest level (0-100)

Focus on topics where a digital product (ebook, course, template) could help."""

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://api.perplexity.ai/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.perplexity_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "sonar",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.3,
                            "max_tokens": 1500
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    content = data["choices"][0]["message"]["content"]
                    signals = await self._parse_discovery_response(content, audience=audience)
                    all_signals.extend(signals)
                    
            except Exception as e:
                logger.error(f"Audience discovery error for {audience}: {e}")
            
            await asyncio.sleep(1)  # Rate limit
        
        return all_signals
    
    async def discover_seasonal_trends(self) -> List[TrendSignal]:
        """
        Find trends specific to the current time of year.
        
        December/January = New Year's resolutions, planning, goal-setting
        """
        if not self.perplexity_key:
            return []
        
        import httpx
        
        now = datetime.now()
        month = now.strftime("%B")
        
        prompt = f"""What topics are trending specifically because it's {month}?

Think about:
- Seasonal behaviors and purchases
- Time-of-year goals (New Year's, tax season, summer, etc.)
- Holiday-related needs
- Annual cycles (school, business quarters, etc.)

List 10 specific, actionable topics that people are searching for RIGHT NOW because of the time of year. Include interest level (0-100) and why it's timely."""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.perplexity_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "sonar",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 2000
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                content = data["choices"][0]["message"]["content"]
                signals = await self._parse_discovery_response(content, source="seasonal")
                return signals
                
        except Exception as e:
            logger.error(f"Seasonal discovery error: {e}")
            return []
    
    async def _parse_discovery_response(
        self, 
        content: str, 
        audience: str = None,
        source: str = "discovery"
    ) -> List[TrendSignal]:
        """Parse discovery response into TrendSignals."""
        
        if not self.anthropic_key:
            return []
        
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.anthropic_key)
        
        parse_prompt = f"""Extract trending topics from this research:

{content}

Return as JSON array:
[
    {{
        "topic": "Specific topic name",
        "score": 75,
        "velocity": 30,
        "category": "best matching category",
        "audience": "target audience",
        "pain_point": "main problem/desire",
        "why_now": "why it's trending"
    }}
]

Categories to choose from: productivity, technology, business, health, finance, lifestyle, education, career, marketing, creativity, relationships, parenting, home, food, travel, fitness, mental_health, spirituality, hobbies, pets, gaming, sports, entertainment

Return ONLY valid JSON, no other text."""

        try:
            loop = asyncio.get_event_loop()
            message = await loop.run_in_executor(
                None,
                lambda: client.messages.create(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": parse_prompt}]
                )
            )
            
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
                    source=f"perplexity_{source}",
                    score=float(t.get("score", 50)),
                    velocity=float(t.get("velocity", 0)),
                    metadata={
                        "category": t.get("category", "other"),
                        "audience": audience or t.get("audience", ""),
                        "pain_point": t.get("pain_point", ""),
                        "why_now": t.get("why_now", ""),
                    }
                ))
            
            return signals
            
        except Exception as e:
            logger.error(f"Parse error: {e}")
            return []
    
    async def full_discovery_scan(self) -> List[TrendSignal]:
        """
        Run all discovery methods and combine results.
        
        This is the comprehensive "find everything trending" scan.
        """
        all_signals = []
        
        # 1. General trending topics
        logger.info("Discovering general trending topics...")
        general = await self.discover_trending_topics(count=20)
        all_signals.extend(general)
        
        # 2. Audience-based discovery
        logger.info("Discovering audience-specific trends...")
        audience = await self.discover_by_audience()
        all_signals.extend(audience)
        
        # 3. Seasonal trends
        logger.info("Discovering seasonal trends...")
        seasonal = await self.discover_seasonal_trends()
        all_signals.extend(seasonal)
        
        logger.info(f"Total discovered: {len(all_signals)} signals")
        
        return all_signals
