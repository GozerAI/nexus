"""
Autonomous Pipeline - Connects discovery to content generation

CORRECTED ARCHITECTURE (2025-01-02):
1. Topic discovered (from scheduler)
2. Research stage gathers information  
3. Blueprint Generator creates NEW blueprint (OpenAI/Claude) → blueprints/generated/
4. Content pipeline generates ebooks from blueprint (Ollama - local LLM)
5. Blueprint moves to blueprints/created/
6. Marketing stage creates promotional content

LLM Role Division:
- OpenAI/Claude: Creative work (blueprint generation, research)
- Ollama: Heavy lifting (chapter content - avoids rate limits)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RESEARCHING = "researching"
    GENERATING_BLUEPRINT = "generating_blueprint"
    GENERATING_CONTENT = "generating_content"
    MARKETING = "marketing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineJob:
    """A single pipeline job from topic to product."""
    job_id: str
    topic: str
    category: str = "other"
    score: float = 0.0
    
    # Status tracking
    status: PipelineStatus = PipelineStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Stage outputs
    research_summary: str = ""
    research_insights: List[str] = field(default_factory=list)
    marketing_hooks: List[str] = field(default_factory=list)
    social_posts: List[Dict] = field(default_factory=list)
    email_subjects: List[str] = field(default_factory=list)
    
    # Blueprint generation (NEW - not matching)
    generated_blueprint_id: Optional[str] = None
    generated_blueprint_path: Optional[str] = None
    blueprint_tokens_used: int = 0
    
    # Content generation
    content_tokens_used: int = 0
    output_path: Optional[str] = None
    books_generated: int = 0
    chapters_generated: int = 0
    
    # Error tracking
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "topic": self.topic,
            "category": self.category,
            "score": self.score,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "research_summary": self.research_summary[:200] + "..." if len(self.research_summary) > 200 else self.research_summary,
            "marketing_hooks": self.marketing_hooks[:3],
            "generated_blueprint_id": self.generated_blueprint_id,
            "output_path": self.output_path,
            "books_generated": self.books_generated,
            "chapters_generated": self.chapters_generated,
            "blueprint_tokens_used": self.blueprint_tokens_used,
            "content_tokens_used": self.content_tokens_used,
            "error": self.error,
        }


class AutonomousPipeline:
    """
    Autonomous content pipeline with CORRECT architecture.
    
    Topic → Research → Generate Blueprint → Generate Content (Ollama) → Marketing → Output
    
    Key changes from original:
    - Blueprint GENERATION instead of matching
    - Ollama for long-form content (avoids rate limits)
    - Blueprint lifecycle: generated/ → created/
    
    Usage:
        pipeline = AutonomousPipeline()
        
        # Process a single topic (full pipeline)
        job = await pipeline.process_topic("AI productivity tools", category="technology")
        
        # Process existing blueprints only (content generation)
        jobs = await pipeline.process_pending_blueprints()
    """
    
    def __init__(
        self,
        output_dir: Path = None,
        blueprint_provider: str = "openai",  # For blueprint generation
        content_provider: str = "ollama",     # For chapter content
        content_model: str = "qwen3:30b-a3b",  # Qwen3 MoE - best for creative writing
        max_chapters: int = None,              # None = all chapters
        max_books: int = 1,                    # Books per blueprint to generate
    ):
        self.output_dir = output_dir or Path("output/pipeline_runs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.blueprint_provider = blueprint_provider
        self.content_provider = content_provider
        self.content_model = content_model
        self.max_chapters = max_chapters
        self.max_books = max_books
        
        # Job tracking
        self.jobs: Dict[str, PipelineJob] = {}
        self.completed_jobs: List[PipelineJob] = []
        
        # State file
        self.state_file = self.output_dir / "pipeline_state.json"
        self._load_state()
    
    def _load_state(self):
        """Load pipeline state from disk."""
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                # Could restore jobs here if needed
            except (IOError, json.JSONDecodeError) as e:
                logger.warning(f"Could not load pipeline state: {e}")
    
    def _save_state(self):
        """Save pipeline state to disk."""
        state = {
            "pending_jobs": [j.to_dict() for j in self.jobs.values()],
            "completed_jobs": [j.to_dict() for j in self.completed_jobs[-20:]],
            "updated_at": datetime.now().isoformat(),
            "config": {
                "blueprint_provider": self.blueprint_provider,
                "content_provider": self.content_provider,
                "content_model": self.content_model,
            }
        }
        self.state_file.write_text(json.dumps(state, indent=2))
    
    async def process_topic(
        self,
        topic: str,
        category: str = "other",
        score: float = 0.0,
        skip_content: bool = False,
        skip_marketing: bool = False,
    ) -> PipelineJob:
        """
        Process a single topic through the full pipeline.
        
        Flow:
        1. Research the topic
        2. Generate NEW blueprint (OpenAI/Claude)
        3. Generate content from blueprint (Ollama)
        4. Create marketing materials
        
        Args:
            topic: The topic to process
            category: Topic category
            score: Trend score
            skip_content: If True, only generate blueprint (no ebook content)
            skip_marketing: If True, skip marketing stage
        
        Returns:
            PipelineJob with all outputs
        """
        import uuid
        
        job = PipelineJob(
            job_id=str(uuid.uuid4())[:8],
            topic=topic,
            category=category,
            score=score,
            started_at=datetime.now(),
        )
        
        self.jobs[job.job_id] = job
        logger.info(f"[{job.job_id}] Starting pipeline for: {topic}")
        
        try:
            # Stage 1: Research
            job.status = PipelineStatus.RESEARCHING
            await self._run_research(job)
            logger.info(f"[{job.job_id}] Research complete: {len(job.research_summary)} chars")
            
            # Stage 2: Generate Blueprint (NEW - not matching!)
            job.status = PipelineStatus.GENERATING_BLUEPRINT
            await self._generate_blueprint(job)
            
            if job.generated_blueprint_id:
                logger.info(f"[{job.job_id}] Blueprint generated: {job.generated_blueprint_id}")
            else:
                logger.error(f"[{job.job_id}] Blueprint generation failed")
                raise RuntimeError("Blueprint generation failed")
            
            # Stage 3: Generate Content (Ollama)
            if not skip_content:
                job.status = PipelineStatus.GENERATING_CONTENT
                await self._run_content_generation(job)
                logger.info(f"[{job.job_id}] Content complete: {job.chapters_generated} chapters")
            
            # Stage 4: Marketing
            if not skip_marketing:
                job.status = PipelineStatus.MARKETING
                await self._run_marketing(job)
                logger.info(f"[{job.job_id}] Marketing complete: {len(job.marketing_hooks)} hooks")
            
            # Complete
            job.status = PipelineStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Save outputs
            await self._save_job_outputs(job)
            
        except Exception as e:
            logger.error(f"[{job.job_id}] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            job.status = PipelineStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.now()
        
        # Move to completed
        self.completed_jobs.append(job)
        del self.jobs[job.job_id]
        self._save_state()
        
        return job
    
    async def _run_research(self, job: PipelineJob):
        """Run research stage."""
        from stages import ResearchStage, ResearchInput, ResearchDepth
        
        stage = ResearchStage()
        
        input_data = ResearchInput(
            task_id=job.job_id,
            topic=job.topic,
            depth=ResearchDepth.MEDIUM,
            focus_areas=[
                f"Current trends in {job.topic}",
                f"Common problems and pain points",
                f"Best practices and solutions",
                f"Target audience characteristics",
            ],
        )
        
        result = await stage.run(input_data)
        
        if result.data and not result.error:
            job.research_summary = result.data.summary
            job.research_insights = [i.insight for i in result.data.insights[:5]]
    
    async def _generate_blueprint(self, job: PipelineJob):
        """Generate a NEW blueprint for the topic (not match existing!)."""
        from nexus.blueprints.generator import BlueprintGenerator
        
        generator = BlueprintGenerator(provider=self.blueprint_provider)
        
        result = await generator.generate_blueprint(
            topic=job.topic,
            research_summary=job.research_summary,
            category=job.category,
        )
        
        if result.success:
            job.generated_blueprint_id = result.library_id
            job.generated_blueprint_path = str(result.blueprint_path)
            job.blueprint_tokens_used = result.tokens_used
        else:
            job.error = f"Blueprint generation failed: {result.error}"
    
    async def _run_content_generation(self, job: PipelineJob):
        """Generate content using Ollama (local LLM to avoid rate limits)."""
        if not job.generated_blueprint_id:
            logger.warning(f"[{job.job_id}] No blueprint to generate content from")
            return
        
        from blueprints import BlueprintParser, BlueprintEbookPipeline, OutputManager
        from nexus.blueprints.generator import BlueprintGenerator
        from nexus.blueprints.llm_backend import get_backend
        
        # Load the generated blueprint
        bp_path = Path(job.generated_blueprint_path)
        if not bp_path.exists():
            logger.error(f"[{job.job_id}] Blueprint file not found: {bp_path}")
            return
        
        parser = BlueprintParser()
        parser.load_file(bp_path)
        
        bp = parser.get_blueprint_by_library(job.generated_blueprint_id)
        
        if not bp:
            logger.error(f"[{job.job_id}] Blueprint not loaded: {job.generated_blueprint_id}")
            return
        
        # Use Ollama backend for content generation (avoids rate limits!)
        backend = get_backend(
            provider=self.content_provider,
            model=self.content_model,
        )
        
        logger.info(f"[{job.job_id}] Using {backend.name} for content generation")
        
        # Create pipeline
        pipeline = BlueprintEbookPipeline(
            blueprint=bp,
            backend=backend,
        )
        
        # Generate books
        total_tokens = 0
        total_chapters = 0
        books_generated = 0
        
        output_mgr = OutputManager(self.output_dir / "books")
        
        for book_idx, book in enumerate(bp.books[:self.max_books]):
            logger.info(f"[{job.job_id}] Generating book {book_idx + 1}: {book.title}")
            
            result = await pipeline.generate_book(
                book,
                max_chapters=self.max_chapters,
            )
            
            if result.success:
                books_generated += 1
                total_chapters += len(result.chapters)
                
                for ch in result.chapters:
                    total_tokens += ch.token_count
                
                # Save book
                saved = output_mgr.save_book(result)
                job.output_path = saved.get("book_dir")
            else:
                logger.warning(f"[{job.job_id}] Book {book_idx + 1} generation failed")
        
        job.books_generated = books_generated
        job.chapters_generated = total_chapters
        job.content_tokens_used = total_tokens
        
        # Move blueprint to created/ folder
        if books_generated > 0:
            generator = BlueprintGenerator()
            generator.mark_as_created(job.generated_blueprint_id)
    
    async def _run_marketing(self, job: PipelineJob):
        """Run marketing stage."""
        from stages import MarketingStage, MarketingInput, MarketingTone
        
        stage = MarketingStage()
        
        input_data = MarketingInput(
            task_id=job.job_id,
            topic=job.topic,
            title=job.topic,
            summary=job.research_summary[:1000] if job.research_summary else None,
            key_points=job.research_insights,
            pain_points=[f"Challenges with {job.topic}"],
            tone=MarketingTone.PROFESSIONAL,
            platforms=["twitter", "linkedin", "email"],
        )
        
        result = await stage.run(input_data)
        
        if result.data and not result.error:
            job.marketing_hooks = (
                result.data.primary_hooks + 
                result.data.pain_point_hooks + 
                result.data.benefit_hooks
            )[:5]
            job.social_posts = [
                {"platform": p.platform, "content": p.copy}
                for p in result.data.social_posts[:5]
            ]
            job.email_subjects = [
                e.subject_line for e in result.data.email_hooks[:5]
            ]
    
    async def _save_job_outputs(self, job: PipelineJob):
        """Save all job outputs to a single file."""
        job_dir = self.output_dir / "jobs" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        output = {
            "job": job.to_dict(),
            "research": {
                "summary": job.research_summary,
                "insights": job.research_insights,
            },
            "blueprint": {
                "id": job.generated_blueprint_id,
                "path": job.generated_blueprint_path,
                "tokens_used": job.blueprint_tokens_used,
            },
            "content": {
                "books_generated": job.books_generated,
                "chapters_generated": job.chapters_generated,
                "tokens_used": job.content_tokens_used,
                "output_path": job.output_path,
            },
            "marketing": {
                "hooks": job.marketing_hooks,
                "social_posts": job.social_posts,
                "email_subjects": job.email_subjects,
            },
        }
        
        output_file = job_dir / "job_output.json"
        output_file.write_text(json.dumps(output, indent=2))
        
        logger.info(f"[{job.job_id}] Saved outputs to {job_dir}")
    
    async def process_pending_blueprints(
        self,
        max_blueprints: int = 1,
    ) -> List[PipelineJob]:
        """
        Process blueprints that are waiting in generated/ folder.
        
        Use this to generate content from blueprints that were created
        but not yet converted to ebooks.
        """
        from nexus.blueprints.generator import BlueprintGenerator
        
        generator = BlueprintGenerator()
        pending = generator.list_generated()
        
        if not pending:
            logger.info("No pending blueprints to process")
            return []
        
        results = []
        
        for bp_info in pending[:max_blueprints]:
            logger.info(f"Processing pending blueprint: {bp_info['library_id']}")
            
            # Create a job for content-only generation
            job = PipelineJob(
                job_id=f"bp-{bp_info['library_id'][:4]}",
                topic=bp_info["source_topic"],
                generated_blueprint_id=bp_info["library_id"],
                generated_blueprint_path=bp_info["path"],
                started_at=datetime.now(),
            )
            
            self.jobs[job.job_id] = job
            
            try:
                job.status = PipelineStatus.GENERATING_CONTENT
                await self._run_content_generation(job)
                
                job.status = PipelineStatus.COMPLETED
                job.completed_at = datetime.now()
                
            except Exception as e:
                job.status = PipelineStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()
            
            self.completed_jobs.append(job)
            del self.jobs[job.job_id]
            results.append(job)
        
        self._save_state()
        return results
    
    async def process_queue(
        self,
        topics: List[Dict[str, Any]],
        skip_content: bool = False,
    ) -> List[PipelineJob]:
        """
        Process multiple topics from a queue.
        
        Args:
            topics: List of dicts with 'topic', 'category', 'score'
            skip_content: If True, only generate blueprints
        
        Returns:
            List of completed PipelineJobs
        """
        results = []
        
        for topic_data in topics:
            job = await self.process_topic(
                topic=topic_data.get("topic", ""),
                category=topic_data.get("category", "other"),
                score=topic_data.get("score", 0.0),
                skip_content=skip_content,
            )
            results.append(job)
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status."""
        from nexus.blueprints.generator import BlueprintGenerator
        
        generator = BlueprintGenerator()
        
        return {
            "pending_jobs": len(self.jobs),
            "completed_jobs": len(self.completed_jobs),
            "pending_blueprints": len(generator.list_generated()),
            "completed_blueprints": len(generator.list_created()),
            "config": {
                "blueprint_provider": self.blueprint_provider,
                "content_provider": self.content_provider,
                "content_model": self.content_model,
            },
            "recent_completions": [
                j.to_dict() for j in self.completed_jobs[-5:]
            ],
        }


# Convenience function
async def run_pipeline(
    topic: str,
    category: str = "other",
    skip_content: bool = False,
    content_model: str = "qwen2.5:14b",
) -> PipelineJob:
    """Quick way to run a topic through the pipeline."""
    pipeline = AutonomousPipeline(content_model=content_model)
    return await pipeline.process_topic(topic, category, skip_content=skip_content)


async def generate_blueprint_only(
    topic: str,
    category: str = "other",
) -> PipelineJob:
    """Generate just the blueprint (no content)."""
    pipeline = AutonomousPipeline()
    return await pipeline.process_topic(topic, category, skip_content=True, skip_marketing=True)
