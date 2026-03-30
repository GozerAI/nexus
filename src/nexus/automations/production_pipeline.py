"""
Production Pipeline - Connects Discovery to Content Generation

This is the autonomous content machine:
1. Insights Engine discovers trending topics
2. Topics get matched to blueprints or generate custom content
3. Research stage gathers information
4. Marketing stage creates promotional content
5. Blueprint pipeline generates the ebook
6. Distribution stage uploads to platforms

Usage:
    from nexus.automations.production_pipeline import ProductionPipeline
    
    pipeline = ProductionPipeline()
    
    # Process a discovered topic
    result = await pipeline.process_topic(
        topic="AI Productivity Tools",
        category="productivity",
        auto_publish=False
    )
    
    # Process queued items
    await pipeline.process_queue()
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProductionStatus(Enum):
    """Status of a production job."""
    QUEUED = "queued"
    RESEARCHING = "researching"
    PLANNING = "planning"
    GENERATING = "generating"
    MARKETING = "marketing"
    REVIEWING = "reviewing"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProductionJob:
    """A content production job."""
    id: str
    topic: str
    category: str
    status: ProductionStatus = ProductionStatus.QUEUED
    
    # Discovery source
    discovery_score: float = 0.0
    discovery_source: str = ""
    audience: str = ""
    pain_points: List[str] = field(default_factory=list)
    
    # Blueprint matching
    matched_blueprint: Optional[str] = None
    matched_book_index: Optional[int] = None
    custom_outline: Optional[Dict] = None
    
    # Research output
    research_summary: str = ""
    research_insights: List[str] = field(default_factory=list)
    
    # Marketing output
    marketing_hooks: List[str] = field(default_factory=list)
    social_posts: List[Dict] = field(default_factory=list)
    email_subjects: List[str] = field(default_factory=list)
    
    # Generation output
    output_path: Optional[str] = None
    word_count: int = 0
    chapter_count: int = 0
    
    # Distribution
    gumroad_url: Optional[str] = None
    price_usd: float = 9.99
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    # Timing
    research_duration: float = 0.0
    marketing_duration: float = 0.0
    generation_duration: float = 0.0
    total_duration: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "category": self.category,
            "status": self.status.value,
            "discovery_score": self.discovery_score,
            "matched_blueprint": self.matched_blueprint,
            "matched_book_index": self.matched_book_index,
            "research_summary": self.research_summary[:200] if self.research_summary else "",
            "marketing_hooks": self.marketing_hooks[:3],
            "output_path": self.output_path,
            "word_count": self.word_count,
            "gumroad_url": self.gumroad_url,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration": self.total_duration,
            "error": self.error,
        }


class ProductionPipeline:
    """
    End-to-end content production pipeline.
    
    Connects:
    - Insights Engine (discovery)
    - Research Stage (Perplexity)
    - Marketing Stage (hooks, posts)
    - Blueprint Pipeline (ebook generation)
    - Distribution Stage (Gumroad)
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent.parent.parent / "output"
        self.jobs_file = self.output_dir / "production_jobs.json"
        self.queue_file = self.output_dir / "production_queue.json"
        
        self._queue: List[ProductionJob] = []
        self._completed: List[ProductionJob] = []
        
        # Load existing state
        self._load_state()
    
    def _load_state(self):
        """Load queue and history from disk."""
        if self.queue_file.exists():
            try:
                data = json.loads(self.queue_file.read_text())
                for job_data in data.get("queue", []):
                    job = ProductionJob(
                        id=job_data["id"],
                        topic=job_data["topic"],
                        category=job_data.get("category", "other"),
                        status=ProductionStatus(job_data.get("status", "queued")),
                        discovery_score=job_data.get("discovery_score", 0),
                        matched_blueprint=job_data.get("matched_blueprint"),
                        matched_book_index=job_data.get("matched_book_index"),
                    )
                    self._queue.append(job)
            except Exception as e:
                logger.error(f"Failed to load queue: {e}")
    
    def _save_state(self):
        """Save queue to disk."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "queue": [j.to_dict() for j in self._queue],
            "updated_at": datetime.now().isoformat(),
        }
        self.queue_file.write_text(json.dumps(data, indent=2))
    
    def _save_job(self, job: ProductionJob):
        """Save completed job to history."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing jobs
        jobs = []
        if self.jobs_file.exists():
            try:
                jobs = json.loads(self.jobs_file.read_text())
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load existing jobs file: {e}")
        
        # Add new job
        jobs.append(job.to_dict())
        
        # Keep last 100
        jobs = jobs[-100:]
        
        self.jobs_file.write_text(json.dumps(jobs, indent=2))
    
    # =========================================================================
    # Queue Management
    # =========================================================================
    
    def add_to_queue(
        self,
        topic: str,
        category: str,
        score: float = 0.0,
        source: str = "manual",
        audience: str = "",
        pain_points: List[str] = None,
        blueprint_id: str = None,
        book_index: int = None,
    ) -> ProductionJob:
        """Add a topic to the production queue."""
        
        job = ProductionJob(
            id=str(uuid.uuid4())[:8],
            topic=topic,
            category=category,
            discovery_score=score,
            discovery_source=source,
            audience=audience,
            pain_points=pain_points or [],
            matched_blueprint=blueprint_id,
            matched_book_index=book_index,
        )
        
        self._queue.append(job)
        self._save_state()
        
        logger.info(f"Added to queue: {topic} (score: {score:.2f})")
        return job
    
    def get_queue(self) -> List[ProductionJob]:
        """Get current queue."""
        return self._queue.copy()
    
    def clear_queue(self):
        """Clear the queue."""
        self._queue = []
        self._save_state()
    
    # =========================================================================
    # Pipeline Execution
    # =========================================================================
    
    async def process_topic(
        self,
        topic: str,
        category: str,
        score: float = 0.0,
        audience: str = "",
        pain_points: List[str] = None,
        blueprint_id: str = None,
        book_index: int = None,
        auto_publish: bool = False,
        max_chapters: int = None,
    ) -> ProductionJob:
        """
        Process a single topic through the full pipeline.
        
        Steps:
        1. Research the topic
        2. Generate marketing content
        3. Match or create blueprint
        4. Generate ebook
        5. Optionally publish
        """
        
        job = ProductionJob(
            id=str(uuid.uuid4())[:8],
            topic=topic,
            category=category,
            discovery_score=score,
            audience=audience,
            pain_points=pain_points or [],
            matched_blueprint=blueprint_id,
            matched_book_index=book_index,
        )
        
        job.started_at = datetime.now()
        
        try:
            # Step 1: Research
            job.status = ProductionStatus.RESEARCHING
            logger.info(f"[{job.id}] Researching: {topic}")
            await self._run_research(job)
            
            # Step 2: Marketing
            job.status = ProductionStatus.MARKETING
            logger.info(f"[{job.id}] Generating marketing content")
            await self._run_marketing(job)
            
            # Step 3: Match blueprint
            job.status = ProductionStatus.PLANNING
            logger.info(f"[{job.id}] Planning content structure")
            await self._match_blueprint(job)
            
            # Step 4: Generate content
            if job.matched_blueprint:
                job.status = ProductionStatus.GENERATING
                logger.info(f"[{job.id}] Generating ebook using {job.matched_blueprint}")
                await self._run_generation(job, max_chapters=max_chapters)
            else:
                logger.warning(f"[{job.id}] No blueprint matched - skipping generation")
            
            # Step 5: Publish (if enabled)
            if auto_publish and job.output_path:
                job.status = ProductionStatus.PUBLISHING
                logger.info(f"[{job.id}] Publishing to Gumroad")
                await self._run_distribution(job)
            
            job.status = ProductionStatus.COMPLETED
            job.completed_at = datetime.now()
            job.total_duration = (job.completed_at - job.started_at).total_seconds()
            
            logger.info(f"[{job.id}] Completed in {job.total_duration:.1f}s")
            
        except Exception as e:
            job.status = ProductionStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
            logger.error(f"[{job.id}] Failed: {e}")
        
        # Save to history
        self._save_job(job)
        self._completed.append(job)
        
        return job
    
    async def process_queue(
        self,
        max_items: int = 1,
        auto_publish: bool = False,
        max_chapters: int = None,
    ) -> List[ProductionJob]:
        """
        Process items from the queue.
        
        Args:
            max_items: Maximum number of items to process
            auto_publish: Whether to publish to Gumroad
            max_chapters: Limit chapters for testing
        """
        results = []
        
        items_to_process = self._queue[:max_items]
        
        for job in items_to_process:
            # Remove from queue
            self._queue.remove(job)
            self._save_state()
            
            # Process
            result = await self.process_topic(
                topic=job.topic,
                category=job.category,
                score=job.discovery_score,
                audience=job.audience,
                pain_points=job.pain_points,
                blueprint_id=job.matched_blueprint,
                book_index=job.matched_book_index,
                auto_publish=auto_publish,
                max_chapters=max_chapters,
            )
            
            results.append(result)
        
        return results
    
    # =========================================================================
    # Pipeline Steps
    # =========================================================================
    
    async def _run_research(self, job: ProductionJob):
        """Run research stage."""
        from stages import ResearchStage, ResearchInput, ResearchDepth
        
        start = datetime.now()
        
        stage = ResearchStage()
        
        # Build focus areas from pain points
        focus_areas = job.pain_points or []
        if job.audience:
            focus_areas.append(f"Needs of {job.audience}")
        
        input_data = ResearchInput(
            task_id=job.id,
            topic=job.topic,
            depth=ResearchDepth.STANDARD,
            focus_areas=focus_areas or ["practical applications", "common challenges", "best practices"],
            max_sources=5,
        )
        
        output = await stage.run(input_data)
        
        if output.data:
            job.research_summary = output.data.summary
            job.research_insights = [i.insight for i in output.data.key_insights[:5]]
        
        job.research_duration = (datetime.now() - start).total_seconds()
    
    async def _run_marketing(self, job: ProductionJob):
        """Run marketing stage."""
        from stages import MarketingStage, MarketingInput, MarketingTone
        
        start = datetime.now()
        
        stage = MarketingStage()
        
        input_data = MarketingInput(
            task_id=job.id,
            topic=job.topic,
            target_audience=job.audience or "professionals and enthusiasts",
            key_benefits=job.pain_points or ["practical knowledge", "actionable strategies"],
            tone=MarketingTone.PROFESSIONAL,
            include_social=True,
            include_email=True,
        )
        
        output = await stage.run(input_data)
        
        if output.data:
            job.marketing_hooks = output.data.hooks[:5]
            job.social_posts = [{"platform": p.platform, "content": p.content} for p in output.data.social_posts[:5]]
            job.email_subjects = output.data.email_subjects[:5]
        
        job.marketing_duration = (datetime.now() - start).total_seconds()
    
    async def _match_blueprint(self, job: ProductionJob):
        """Match topic to a blueprint or create custom outline."""
        
        # If already matched, skip
        if job.matched_blueprint and job.matched_book_index:
            return
        
        # Try to match using scorer
        from insights import TrendScorer
        
        scorer = TrendScorer()
        
        # Create a signal-like object
        from insights import TrendSignal
        signal = TrendSignal(
            topic=job.topic,
            source="production",
            score=job.discovery_score * 100,
            velocity=0,
            metadata={"category": job.category}
        )
        
        # Score and match
        analysis = await scorer.score_signal(signal)
        
        if analysis.recommended_blueprint:
            job.matched_blueprint = analysis.recommended_blueprint
            job.matched_book_index = analysis.recommended_book_index or 1
            logger.info(f"[{job.id}] Matched to {job.matched_blueprint} book {job.matched_book_index}")
        else:
            # Default to a generic blueprint
            job.matched_blueprint = "AAGP"  # AI-Assisted Goal Planning
            job.matched_book_index = 1
            logger.info(f"[{job.id}] No match found, using default blueprint")
    
    async def _run_generation(self, job: ProductionJob, max_chapters: int = None):
        """Run ebook generation."""
        from nexus.blueprints.parser import BlueprintParser
        from nexus.blueprints.pipeline import BlueprintEbookPipeline
        from nexus.blueprints.llm_backend import get_backend
        from nexus.blueprints.output_manager import OutputManager
        
        start = datetime.now()
        
        # Load blueprint
        parser = BlueprintParser()
        bp_path = Path(__file__).parent.parent.parent / "blueprints" / "all_blueprints.json"
        parser.load_file(bp_path)
        
        blueprint = parser.get_blueprint(job.matched_blueprint)
        if not blueprint:
            raise ValueError(f"Blueprint not found: {job.matched_blueprint}")
        
        book = blueprint.get_book(job.matched_book_index)
        if not book:
            raise ValueError(f"Book not found: {job.matched_blueprint} #{job.matched_book_index}")
        
        # Setup pipeline
        backend = get_backend(provider="anthropic")
        pipeline = BlueprintEbookPipeline(blueprint=blueprint, backend=backend)
        
        output_manager = OutputManager(base_dir=self.output_dir / "books")
        
        # Inject research context
        context = {
            "research_summary": job.research_summary,
            "key_insights": job.research_insights,
            "target_audience": job.audience,
            "topic_focus": job.topic,
        }
        
        # Generate
        chapters_to_gen = max_chapters or len(book.chapters)
        
        ebook = await pipeline.generate_book(
            book_index=job.matched_book_index,
            max_chapters=chapters_to_gen,
            context=context,
        )
        
        # Save
        output_path = output_manager.save_ebook(ebook, blueprint.library_id, job.matched_book_index)
        
        job.output_path = str(output_path)
        job.word_count = sum(len(ch.content.split()) for ch in ebook.chapters)
        job.chapter_count = len(ebook.chapters)
        job.generation_duration = (datetime.now() - start).total_seconds()
    
    async def _run_distribution(self, job: ProductionJob):
        """Run distribution stage."""
        from stages import DistributionStage, DistributionInput, Platform, PricingTier
        
        if not job.output_path:
            return
        
        stage = DistributionStage()
        
        # Build description from marketing
        description = f"{job.topic}\n\n"
        if job.marketing_hooks:
            description += job.marketing_hooks[0] + "\n\n"
        if job.research_summary:
            description += job.research_summary[:500]
        
        input_data = DistributionInput(
            task_id=job.id,
            title=job.topic,
            description=description,
            content_file=Path(job.output_path),
            pricing=PricingTier(price_usd=job.price_usd),
            tags=[job.category, "ebook", "guide"],
            platforms=[Platform.GUMROAD],
        )
        
        output = await stage.run(input_data)
        
        if output.data and output.data.listings:
            listing = output.data.listings[0]
            if listing.success:
                job.gumroad_url = listing.product_url


# Convenience function
async def process_discovered_topics(
    topics: List[Dict],
    max_items: int = 1,
    auto_publish: bool = False,
) -> List[ProductionJob]:
    """
    Process topics discovered by the insights engine.
    
    Args:
        topics: List of dicts with topic, score, category
        max_items: How many to process
        auto_publish: Whether to publish to Gumroad
    """
    pipeline = ProductionPipeline()
    
    # Add to queue
    for t in topics[:max_items]:
        pipeline.add_to_queue(
            topic=t.get("topic", ""),
            category=t.get("category", "other"),
            score=t.get("score", 0),
            source="discovery",
        )
    
    # Process
    return await pipeline.process_queue(
        max_items=max_items,
        auto_publish=auto_publish,
    )
