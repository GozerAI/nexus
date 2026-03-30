"""
Content Orchestrator - Unified Thought-to-Action Pipeline

Replaces n8n workflows with native Python orchestration.
Monitors trends, generates content, handles marketing, tracks sales.

Integrates with:
- Observatory: Real-time metrics and pattern detection
- Expert Panel: Multi-expert decision making
- Pipeline Executor: Approval workflows

Usage:
    from nexus.automations.content_orchestrator import ContentOrchestrator
    
    orchestrator = ContentOrchestrator(platform)
    await orchestrator.initialize()
    orchestrator.start()
    
    # Manual trigger
    await orchestrator.generate_ebook("AAGP", 1)
    
    # Or let it run on schedule
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json

from .engine import AutomationEngine, Automation, AutomationRun
from .scheduler import Schedule, ScheduleType
from .triggers import TriggerType, TriggerEvent

# Observatory integration
from nexus.observatory.collector import get_collector, MetricType

# Get global metrics collector
metrics = get_collector()


class ContentType(Enum):
    """Types of content we can generate."""
    EBOOK = "ebook"
    VIDEO = "video"
    SOCIAL = "social"
    NEWSLETTER = "newsletter"


class ContentStatus(Enum):
    """Status of a content piece."""
    QUEUED = "queued"
    GENERATING = "generating"
    COMPLETED = "completed"
    MARKETING = "marketing"
    PUBLISHED = "published"
    FAILED = "failed"


@dataclass
class ContentItem:
    """A piece of content in the pipeline."""
    id: str
    content_type: ContentType
    status: ContentStatus = ContentStatus.QUEUED
    
    # Source info
    blueprint_id: Optional[str] = None
    book_index: Optional[int] = None
    topic: Optional[str] = None
    
    # Generation config
    provider: str = "anthropic"
    max_chapters: Optional[int] = None
    
    # Output
    output_path: Optional[str] = None
    marketing_hooks: List[str] = field(default_factory=list)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # Sales
    sales_link: Optional[str] = None
    revenue: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.content_type.value,
            "status": self.status.value,
            "blueprint_id": self.blueprint_id,
            "book_index": self.book_index,
            "topic": self.topic,
            "output_path": self.output_path,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }


class ContentOrchestrator:
    """
    Unified content generation and marketing orchestrator.
    
    Replaces n8n workflows with native Python automation.
    
    Features:
    - Scheduled content generation
    - Trend-based topic selection
    - Multi-format output (ebook, video hooks, social)
    - Marketing automation
    - Sales tracking
    """
    
    def __init__(self, platform=None):
        self.platform = platform
        self._queue: List[ContentItem] = []
        self._history: List[ContentItem] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Paths
        self.output_dir = Path(__file__).parent.parent.parent / "output"
        self.queue_file = self.output_dir / "content_queue.json"
        
        # Components (lazy loaded)
        self._parser = None
        self._pipeline = None
        self._output_manager = None
        
        # Callbacks
        self.on_content_complete: Optional[Callable] = None
        self.on_content_failed: Optional[Callable] = None
    
    def _load_parser(self):
        """Lazy load blueprint parser."""
        if self._parser is None:
            from nexus.blueprints.parser import BlueprintParser
            self._parser = BlueprintParser()
            bp_path = Path(__file__).parent.parent.parent / "blueprints" / "all_blueprints.json"
            if bp_path.exists():
                self._parser.load_file(bp_path)
        return self._parser
    
    def _get_pipeline(self, blueprint, provider: str = "anthropic", model: str = None):
        """Get pipeline for a blueprint."""
        from nexus.blueprints.pipeline import BlueprintEbookPipeline
        from nexus.blueprints.llm_backend import get_backend
        
        backend = get_backend(provider=provider, model=model)
        return BlueprintEbookPipeline(blueprint=blueprint, backend=backend)
    
    def _get_output_manager(self):
        """Get output manager."""
        if self._output_manager is None:
            from nexus.blueprints.output_manager import OutputManager
            self._output_manager = OutputManager(
                base_dir=self.output_dir / "books"
            )
        return self._output_manager
    
    async def initialize(self):
        """Initialize the orchestrator."""
        # Load queue from disk
        self._load_queue()
        
        # Pre-load parser
        self._load_parser()
        
        return True
    
    def _load_queue(self):
        """Load queue from disk."""
        if self.queue_file.exists():
            try:
                data = json.loads(self.queue_file.read_text())
                # Reconstruct ContentItems
                self._queue = []
                for item_data in data.get("queue", []):
                    item = ContentItem(
                        id=item_data["id"],
                        content_type=ContentType(item_data["type"]),
                        status=ContentStatus(item_data["status"]),
                        blueprint_id=item_data.get("blueprint_id"),
                        book_index=item_data.get("book_index"),
                        topic=item_data.get("topic"),
                    )
                    self._queue.append(item)
            except Exception as e:
                print(f"Failed to load queue: {e}")
    
    def _save_queue(self):
        """Save queue to disk."""
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "queue": [item.to_dict() for item in self._queue],
            "updated_at": datetime.now().isoformat()
        }
        self.queue_file.write_text(json.dumps(data, indent=2))
    
    # =========================================================================
    # Queue Management
    # =========================================================================
    
    def queue_ebook(
        self,
        blueprint_id: str,
        book_index: int,
        provider: str = "anthropic",
        max_chapters: int = None
    ) -> ContentItem:
        """Add an ebook to the generation queue."""
        import uuid
        
        item = ContentItem(
            id=str(uuid.uuid4())[:8],
            content_type=ContentType.EBOOK,
            blueprint_id=blueprint_id,
            book_index=book_index,
            provider=provider,
            max_chapters=max_chapters,
        )
        
        self._queue.append(item)
        self._save_queue()
        
        return item
    
    def queue_from_trend(self, trend_topic: str, audience: str = None) -> ContentItem:
        """Queue content based on a trending topic.

        Uses keyword matching to find the best blueprint for the trend.

        Args:
            trend_topic: The trending topic to create content for
            audience: Optional target audience specification

        Returns:
            ContentItem: The queued content item
        """
        # Load blueprints and match to trend
        parser = self._load_parser()

        # Define topic-to-blueprint mappings
        topic_keywords = {
            "AAGP": ["ai", "artificial intelligence", "machine learning", "automation", "future", "technology"],
            "productivity": ["productivity", "efficiency", "workflow", "time management", "organization"],
            "business": ["business", "entrepreneurship", "startup", "marketing", "sales", "growth"],
            "health": ["health", "wellness", "fitness", "nutrition", "mental health", "self-care"],
            "finance": ["finance", "investing", "money", "wealth", "crypto", "trading"],
        }

        # Find best matching blueprint
        trend_lower = trend_topic.lower()
        best_match = "AAGP"  # Default
        best_score = 0

        for blueprint_id, keywords in topic_keywords.items():
            score = sum(1 for kw in keywords if kw in trend_lower)
            if score > best_score:
                best_score = score
                best_match = blueprint_id

        # Queue the first book from the matching blueprint
        return self.queue_ebook(best_match, 1)
    
    def get_queue(self) -> List[ContentItem]:
        """Get current queue."""
        return self._queue.copy()
    
    def get_history(self, limit: int = 50) -> List[ContentItem]:
        """Get generation history."""
        return self._history[-limit:]
    
    def clear_queue(self):
        """Clear the queue."""
        self._queue = []
        self._save_queue()
    
    # =========================================================================
    # Generation
    # =========================================================================
    
    async def generate_ebook(
        self,
        blueprint_id: str,
        book_index: int,
        provider: str = "anthropic",
        model: str = None,
        max_chapters: int = None
    ) -> ContentItem:
        """Generate an ebook immediately (not queued)."""
        import uuid
        
        # Track queue depth
        metrics.gauge("pipeline.queue_depth", len(self._queue))
        metrics.increment("pipeline.generation.started")
        
        item = ContentItem(
            id=str(uuid.uuid4())[:8],
            content_type=ContentType.EBOOK,
            blueprint_id=blueprint_id,
            book_index=book_index,
            provider=provider,
            max_chapters=max_chapters,
            status=ContentStatus.GENERATING,
            started_at=datetime.now()
        )
        
        generation_start = datetime.now()
        
        try:
            # Get blueprint
            parser = self._load_parser()
            bp = parser.get_blueprint_by_library(blueprint_id)
            if not bp:
                raise ValueError(f"Blueprint not found: {blueprint_id}")
            
            book = bp.books[book_index - 1]
            
            # Generate
            pipeline = self._get_pipeline(bp, provider, model)
            result = await pipeline.generate_book(book, max_chapters=max_chapters)
            
            # Save
            output_mgr = self._get_output_manager()
            saved = output_mgr.save_book(result)
            
            # Calculate metrics
            duration = (datetime.now() - generation_start).total_seconds()
            total_tokens = sum(ch.token_count for ch in result.chapters if ch.success)
            
            # Record Observatory metrics
            metrics.timer("ebook.generation.duration", duration, 
                         tags={"blueprint": blueprint_id, "provider": provider})
            metrics.increment("ebook.completed", 
                            tags={"blueprint": blueprint_id})
            metrics.gauge("ebook.tokens.total", total_tokens,
                         tags={"blueprint": blueprint_id})
            metrics.gauge("ebook.chapters.count", len([c for c in result.chapters if c.success]),
                         tags={"blueprint": blueprint_id})
            
            # Update item
            item.status = ContentStatus.COMPLETED
            item.completed_at = datetime.now()
            item.output_path = saved["book_dir"]
            item.topic = book.title
            
            # Extract marketing hooks
            item.marketing_hooks = self._extract_marketing_hooks(result)
            
            # Callback
            if self.on_content_complete:
                self.on_content_complete(item)
            
        except Exception as e:
            # Record failure metrics
            metrics.increment("ebook.failed", 
                            tags={"blueprint": blueprint_id, "error_type": type(e).__name__})
            
            item.status = ContentStatus.FAILED
            item.error = str(e)
            item.completed_at = datetime.now()
            
            if self.on_content_failed:
                self.on_content_failed(item)
        
        self._history.append(item)
        return item
    
    def _extract_marketing_hooks(self, result) -> List[str]:
        """Extract marketing hooks from generated book."""
        hooks = []
        
        # Book title hook
        hooks.append(f"{result.book_spec.title} - Complete Guide")
        
        # Chapter-based hooks
        for ch in result.chapters[:3]:
            if ch.success:
                hooks.append(f"Chapter: {ch.chapter_spec.title}")
        
        # Outcome hook
        if result.book_spec.primary_outcome:
            hooks.append(f"Learn: {result.book_spec.primary_outcome[:100]}")
        
        return hooks
    
    # =========================================================================
    # Background Processing
    # =========================================================================
    
    def start(self):
        """Start background queue processing."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_queue_loop())
    
    def stop(self):
        """Stop background processing."""
        self._running = False
        if self._task:
            self._task.cancel()
    
    async def _process_queue_loop(self):
        """Process queue items in background."""
        while self._running:
            # Get next queued item
            queued = [i for i in self._queue if i.status == ContentStatus.QUEUED]
            
            if queued:
                item = queued[0]
                item.status = ContentStatus.GENERATING
                item.started_at = datetime.now()
                self._save_queue()
                
                try:
                    if item.content_type == ContentType.EBOOK:
                        result = await self.generate_ebook(
                            blueprint_id=item.blueprint_id,
                            book_index=item.book_index,
                            provider=item.provider,
                            max_chapters=item.max_chapters
                        )
                        # Copy result data back to queued item
                        item.status = result.status
                        item.output_path = result.output_path
                        item.marketing_hooks = result.marketing_hooks
                        item.completed_at = result.completed_at
                        item.error = result.error
                    
                except Exception as e:
                    item.status = ContentStatus.FAILED
                    item.error = str(e)
                    item.completed_at = datetime.now()
                
                # Move to history
                self._queue.remove(item)
                self._history.append(item)
                self._save_queue()
            
            # Wait before checking again
            await asyncio.sleep(10)
    
    # =========================================================================
    # Trend Monitoring (Future)
    # =========================================================================
    
    async def check_trends(self) -> List[str]:
        """Check for trending topics using available integrations.

        Aggregates trends from multiple sources:
        - Web search (via platform discovery)
        - News search
        - Arxiv (for AI/tech topics)

        Returns:
            List of trending topic strings
        """
        trends = []

        # Use platform's web search if available
        if self.platform and hasattr(self.platform, '_web_search') and self.platform._web_search:
            try:
                # Search for trending AI topics
                ai_news = await self.platform.search_news("artificial intelligence trends", num_results=5)
                for item in ai_news:
                    if item.get('title'):
                        trends.append(item['title'])

                # Search for trending tech topics
                tech_news = await self.platform.search_news("technology trends 2025", num_results=5)
                for item in tech_news:
                    if item.get('title'):
                        trends.append(item['title'])

            except Exception as e:
                print(f"Error fetching trends from web search: {e}")

        # Use platform's Arxiv integration for research trends
        if self.platform and hasattr(self.platform, '_arxiv_integration') and self.platform._arxiv_integration:
            try:
                recent_papers = await self.platform.search_arxiv("machine learning", max_results=5)
                for paper in recent_papers:
                    if paper.get('title'):
                        trends.append(f"Research: {paper['title']}")
            except Exception as e:
                print(f"Error fetching trends from Arxiv: {e}")

        # Deduplicate and limit
        seen = set()
        unique_trends = []
        for trend in trends:
            normalized = trend.lower().strip()[:100]
            if normalized not in seen:
                seen.add(normalized)
                unique_trends.append(trend)

        return unique_trends[:10]  # Limit to top 10
    
    async def auto_queue_from_trends(self):
        """Automatically queue content based on trends.
        
        Called by scheduler to keep pipeline fed.
        """
        trends = await self.check_trends()
        
        for trend in trends[:3]:  # Limit to top 3
            self.queue_from_trend(trend)
    
    # =========================================================================
    # Marketing (Future)
    # =========================================================================
    
    async def create_marketing_content(self, item: ContentItem) -> Dict[str, Any]:
        """Create marketing content from completed ebook.

        Generates:
        - Video hooks/scripts from chapters
        - Social media posts from marketing hooks
        - Email sequence for promotion

        Args:
            item: Completed ContentItem with marketing_hooks

        Returns:
            Dict with video_hooks, social_posts, and email_sequence
        """
        result = {
            "video_hooks": [],
            "social_posts": [],
            "email_sequence": []
        }

        if not item.marketing_hooks:
            return result

        # Generate video hooks from chapter titles
        for hook in item.marketing_hooks:
            if hook.startswith("Chapter:"):
                chapter_title = hook.replace("Chapter:", "").strip()
                result["video_hooks"].append({
                    "title": f"🎬 {chapter_title}",
                    "hook": f"Want to learn about {chapter_title}? In this video, I'll show you exactly how...",
                    "cta": f"Get the full guide in '{item.topic}' - link in bio!",
                    "duration_suggestion": "60-90 seconds"
                })

        # Generate social posts from hooks
        social_templates = [
            "🚀 Just finished reading about {topic}. Here's what blew my mind: {hook}",
            "💡 Key insight from '{topic}': {hook}",
            "📚 Learning {topic} changed everything. Start here: {hook}",
            "🔥 Hot take: {hook} (from '{topic}')",
        ]

        for i, hook in enumerate(item.marketing_hooks[:4]):
            template = social_templates[i % len(social_templates)]
            result["social_posts"].append({
                "platform": ["twitter", "linkedin", "threads"][i % 3],
                "content": template.format(topic=item.topic or "this book", hook=hook),
                "hashtags": self._generate_hashtags(item.topic, hook)
            })

        # Generate email sequence
        result["email_sequence"] = [
            {
                "subject": f"📖 Your guide to {item.topic} is ready",
                "preview": "Inside: Everything you need to know...",
                "type": "welcome",
                "day": 0
            },
            {
                "subject": f"⚡ Quick win from {item.topic}",
                "preview": item.marketing_hooks[0] if item.marketing_hooks else "Start here...",
                "type": "value",
                "day": 2
            },
            {
                "subject": f"🎯 Most people miss this about {item.topic}",
                "preview": "The hidden insight that changes everything...",
                "type": "insight",
                "day": 5
            },
            {
                "subject": f"Last chance: {item.topic} exclusive",
                "preview": "Don't miss out on this...",
                "type": "urgency",
                "day": 7
            }
        ]

        return result

    def _generate_hashtags(self, topic: str, content: str) -> List[str]:
        """Generate relevant hashtags for social content."""
        hashtags = ["#learning", "#growth"]

        if topic:
            # Convert topic to hashtag
            topic_tag = topic.lower().replace(" ", "")[:20]
            hashtags.append(f"#{topic_tag}")

        # Add content-based hashtags
        content_lower = content.lower() if content else ""
        keyword_hashtags = {
            "ai": "#artificialintelligence",
            "machine learning": "#machinelearning",
            "business": "#business",
            "productivity": "#productivity",
            "success": "#success",
            "mindset": "#mindset",
            "tech": "#technology",
        }

        for keyword, hashtag in keyword_hashtags.items():
            if keyword in content_lower:
                hashtags.append(hashtag)

        return hashtags[:5]  # Limit to 5 hashtags
    
    # =========================================================================
    # Status
    # =========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        parser = self._load_parser()
        
        return {
            "running": self._running,
            "queue_size": len(self._queue),
            "history_size": len(self._history),
            "blueprints_loaded": len(parser) if parser else 0,
            "queue": [item.to_dict() for item in self._queue],
            "recent_completions": [
                item.to_dict() 
                for item in self._history[-5:]
                if item.status == ContentStatus.COMPLETED
            ]
        }


# =============================================================================
# Integration with AutomationEngine
# =============================================================================

def register_content_automations(engine: AutomationEngine, orchestrator: ContentOrchestrator):
    """Register content orchestrator automations with the engine."""
    
    # Scheduled content generation
    async def scheduled_content_check(event: TriggerEvent, platform):
        """Check queue and process pending items."""
        await orchestrator._process_queue_loop()
    
    engine.register_automation(Automation(
        name="content_queue_processor",
        description="Process content generation queue",
        trigger_type=TriggerType.MANUAL,
        action=scheduled_content_check,
        schedule_type=ScheduleType.INTERVAL,
        schedule_config={"interval_minutes": 60}  # Check hourly
    ))
    
    # Trend monitoring
    async def trend_monitor(event: TriggerEvent, platform):
        """Monitor trends and queue content."""
        await orchestrator.auto_queue_from_trends()
    
    engine.register_automation(Automation(
        name="trend_monitor",
        description="Monitor trends and auto-queue content",
        trigger_type=TriggerType.MANUAL,
        action=trend_monitor,
        schedule_type=ScheduleType.DAILY,
        schedule_config={"daily_hour": 6, "daily_minute": 0}  # 6 AM daily
    ))
