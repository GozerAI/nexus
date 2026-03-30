"""
Blueprint Generator - Creates new blueprints from trending topics

Uses OpenAI/Claude for creative blueprint generation, saves to blueprints/generated/
The generated blueprints become the source for content creation.

Flow:
1. Topic + Research summary → LLM generates 8-book blueprint JSON
2. Save to blueprints/generated/{library_id}.json
3. Content pipeline picks up from generated/ folder
4. After content creation, blueprint moves to blueprints/created/
"""

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class BlueprintGenerationResult:
    """Result of blueprint generation."""
    success: bool
    library_id: Optional[str] = None
    blueprint_path: Optional[Path] = None
    blueprint_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: int = 0
    duration_seconds: float = 0.0


# Blueprint schema template for the LLM prompt
BLUEPRINT_SCHEMA = '''
{
  "blueprint_meta": {
    "blueprint_id": "BP-{LIBRARY_ID}-001",
    "library_id": "{LIBRARY_ID}",
    "library_style": "ebook_series",
    "version": "1.0.0",
    "generated_at": "{TIMESTAMP}",
    "source_topic": "{TOPIC}"
  },
  "executive_summary": {
    "purpose": "Brief description of the series purpose",
    "target_audience": ["Audience 1", "Audience 2", "Audience 3"],
    "primary_outcomes": ["Outcome 1", "Outcome 2", "Outcome 3"]
  },
  "series_design_principles": {
    "core_principles": ["Principle 1", "Principle 2", "Principle 3"],
    "tone_guidelines": ["Guideline 1", "Guideline 2"]
  },
  "catalog_baseline": {
    "items": [
      {
        "position": 1,
        "item_id": "{LIBRARY_ID}-01",
        "title": "Book Title for Audience A",
        "primary_outcome": "What reader will achieve",
        "tags": ["tag1", "tag2"],
        "chapters": [
          {"chapter_number": 1, "title": "Chapter Title", "key_topics": ["topic1", "topic2"]},
          {"chapter_number": 2, "title": "Chapter Title", "key_topics": ["topic1", "topic2"]}
        ]
      }
    ]
  }
}
'''

GENERATION_PROMPT = '''You are an expert content strategist creating an 8-book ebook series blueprint.

TOPIC: {topic}

RESEARCH CONTEXT:
{research_summary}

Create a complete blueprint JSON for an 8-book series on this topic. Each book should target a DIFFERENT audience:
1. Small business owners / SMBs
2. Families / Households
3. Freelancers / Solopreneurs
4. Enterprise / Corporate teams
5. Beginners / Newcomers to the topic
6. Advanced practitioners
7. Specific industry vertical (choose relevant one)
8. Future trends / Strategic planning

REQUIREMENTS:
- Each book needs 5-8 chapters with clear, actionable outcomes
- Titles should be compelling and specific (not generic)
- Each chapter needs 2-4 key_topics
- Tags should be searchable keywords
- The series should comprehensively cover the topic from multiple angles

OUTPUT: Return ONLY valid JSON matching this schema (no markdown, no explanation):
{schema}

Generate the complete blueprint now:'''


class BlueprintGenerator:
    """
    Generates new blueprints from trending topics.
    
    Uses OpenAI (GPT-4o) by default for creative generation.
    Falls back to Claude if OpenAI unavailable.
    """
    
    def __init__(
        self,
        output_dir: Optional[Path] = None,
        provider: str = "openai",
        model: Optional[str] = None,
    ):
        self.output_dir = output_dir or Path(__file__).parent.parent.parent / "blueprints" / "generated"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.created_dir = Path(__file__).parent.parent.parent / "blueprints" / "created"
        self.created_dir.mkdir(parents=True, exist_ok=True)
        
        self.provider = provider
        self.model = model or self._get_default_model(provider)
        
        self._client = None
    
    def _get_default_model(self, provider: str) -> str:
        """Get default model for provider."""
        defaults = {
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-20250514",
        }
        return defaults.get(provider, "gpt-4o")
    
    def _get_backend(self):
        """Get LLM backend for generation."""
        from .llm_backend import get_backend
        return get_backend(provider=self.provider, model=self.model)
    
    def _generate_library_id(self, topic: str) -> str:
        """Generate a unique library ID from topic."""
        # Extract key words and create 4-char ID
        words = re.findall(r'\b[a-zA-Z]+\b', topic.upper())
        
        if len(words) >= 2:
            # Take first letter of first 4 words
            base_id = ''.join(w[0] for w in words[:4])
        else:
            # Take first 4 letters of first word
            base_id = words[0][:4] if words else "GNRC"
        
        # Ensure 4 characters
        base_id = base_id[:4].ljust(4, 'X')
        
        # Check for uniqueness
        existing = list(self.output_dir.glob("*.json")) + list(self.created_dir.glob("*.json"))
        existing_ids = {f.stem.upper() for f in existing}
        
        if base_id not in existing_ids:
            return base_id
        
        # Add numeric suffix if exists
        for i in range(1, 100):
            new_id = f"{base_id[:3]}{i}"
            if new_id not in existing_ids:
                return new_id
        
        # Fallback to UUID-based
        return f"G{uuid.uuid4().hex[:3].upper()}"
    
    async def generate_blueprint(
        self,
        topic: str,
        research_summary: str = "",
        category: str = "other",
    ) -> BlueprintGenerationResult:
        """
        Generate a new blueprint from a topic.
        
        Args:
            topic: The trending topic to create content for
            research_summary: Research context from the research stage
            category: Topic category for organization
        
        Returns:
            BlueprintGenerationResult with blueprint data and path
        """
        import time
        start = time.time()
        
        logger.info(f"Generating blueprint for topic: {topic}")
        
        try:
            # Generate library ID
            library_id = self._generate_library_id(topic)
            timestamp = datetime.now().isoformat()
            
            # Build prompt
            schema = BLUEPRINT_SCHEMA.replace("{LIBRARY_ID}", library_id)
            schema = schema.replace("{TIMESTAMP}", timestamp)
            schema = schema.replace("{TOPIC}", topic)
            
            prompt = GENERATION_PROMPT.format(
                topic=topic,
                research_summary=research_summary or "No additional research provided.",
                schema=schema,
            )
            
            # Generate with LLM
            backend = self._get_backend()
            
            system_prompt = (
                "You are a content strategist creating ebook series blueprints. "
                "Output ONLY valid JSON. No markdown code blocks. No explanations. "
                "Start with { and end with }."
            )
            
            response = await backend.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=8000,
            )
            
            duration = time.time() - start
            
            # Parse JSON response
            content = response.content.strip()
            
            # Clean up common issues
            if content.startswith("```"):
                # Remove markdown code blocks
                content = re.sub(r'^```(?:json)?\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
            
            # Parse JSON
            try:
                blueprint_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse blueprint JSON: {e}")
                logger.error(f"Raw content: {content[:500]}...")
                return BlueprintGenerationResult(
                    success=False,
                    error=f"Invalid JSON from LLM: {e}",
                    tokens_used=response.tokens_used,
                    duration_seconds=duration,
                )
            
            # Validate required fields
            if not self._validate_blueprint(blueprint_data):
                return BlueprintGenerationResult(
                    success=False,
                    error="Blueprint missing required fields",
                    tokens_used=response.tokens_used,
                    duration_seconds=duration,
                )
            
            # Ensure library_id is set correctly
            if "blueprint_meta" not in blueprint_data:
                blueprint_data["blueprint_meta"] = {}
            blueprint_data["blueprint_meta"]["library_id"] = library_id
            blueprint_data["blueprint_meta"]["generated_at"] = timestamp
            blueprint_data["blueprint_meta"]["source_topic"] = topic
            blueprint_data["blueprint_meta"]["source_category"] = category
            
            # Save to generated folder
            output_path = self.output_dir / f"{library_id}.json"
            output_path.write_text(json.dumps(blueprint_data, indent=2))
            
            logger.info(f"Blueprint saved: {output_path}")
            
            return BlueprintGenerationResult(
                success=True,
                library_id=library_id,
                blueprint_path=output_path,
                blueprint_data=blueprint_data,
                tokens_used=response.tokens_used,
                duration_seconds=duration,
            )
            
        except Exception as e:
            logger.error(f"Blueprint generation failed: {e}")
            return BlueprintGenerationResult(
                success=False,
                error=str(e),
                duration_seconds=time.time() - start,
            )
    
    def _validate_blueprint(self, data: Dict[str, Any]) -> bool:
        """Validate blueprint has required structure."""
        required = ["blueprint_meta", "executive_summary", "catalog_baseline"]
        
        for field in required:
            if field not in data:
                logger.warning(f"Blueprint missing field: {field}")
                return False
        
        # Check catalog has items
        catalog = data.get("catalog_baseline", {})
        items = catalog.get("items", [])
        
        if not items or len(items) < 1:
            logger.warning("Blueprint has no books in catalog_baseline.items")
            return False
        
        # Check each book has chapters
        for i, item in enumerate(items):
            if "chapters" not in item or not item["chapters"]:
                logger.warning(f"Book {i+1} has no chapters")
                return False
        
        return True
    
    def list_generated(self) -> List[Dict[str, Any]]:
        """List all generated blueprints awaiting content creation."""
        blueprints = []
        
        for bp_file in self.output_dir.glob("*.json"):
            try:
                data = json.loads(bp_file.read_text())
                meta = data.get("blueprint_meta", {})
                exec_sum = data.get("executive_summary", {})
                catalog = data.get("catalog_baseline", {})
                
                blueprints.append({
                    "library_id": meta.get("library_id", bp_file.stem),
                    "source_topic": meta.get("source_topic", "Unknown"),
                    "generated_at": meta.get("generated_at", "Unknown"),
                    "purpose": exec_sum.get("purpose", "")[:100],
                    "book_count": len(catalog.get("items", [])),
                    "path": str(bp_file),
                })
            except Exception as e:
                logger.warning(f"Error reading {bp_file}: {e}")
        
        return blueprints
    
    def list_created(self) -> List[Dict[str, Any]]:
        """List all blueprints that have been converted to content."""
        blueprints = []
        
        for bp_file in self.created_dir.glob("*.json"):
            try:
                data = json.loads(bp_file.read_text())
                meta = data.get("blueprint_meta", {})
                
                blueprints.append({
                    "library_id": meta.get("library_id", bp_file.stem),
                    "source_topic": meta.get("source_topic", "Unknown"),
                    "created_at": meta.get("content_created_at", "Unknown"),
                    "path": str(bp_file),
                })
            except Exception as e:
                logger.warning(f"Error reading {bp_file}: {e}")
        
        return blueprints
    
    def mark_as_created(self, library_id: str) -> bool:
        """
        Move a blueprint from generated/ to created/ after content generation.
        
        Args:
            library_id: The blueprint library ID to move
        
        Returns:
            True if successful, False otherwise
        """
        source = self.output_dir / f"{library_id}.json"
        
        if not source.exists():
            logger.warning(f"Blueprint not found: {source}")
            return False
        
        try:
            # Load and update metadata
            data = json.loads(source.read_text())
            if "blueprint_meta" not in data:
                data["blueprint_meta"] = {}
            data["blueprint_meta"]["content_created_at"] = datetime.now().isoformat()
            
            # Save to created folder
            dest = self.created_dir / f"{library_id}.json"
            dest.write_text(json.dumps(data, indent=2))
            
            # Remove from generated
            source.unlink()
            
            logger.info(f"Blueprint {library_id} moved to created/")
            return True
            
        except Exception as e:
            logger.error(f"Failed to mark blueprint as created: {e}")
            return False
    
    def get_next_for_content(self) -> Optional[Dict[str, Any]]:
        """
        Get the next blueprint from generated/ folder for content creation.
        
        Returns the oldest generated blueprint (FIFO queue).
        """
        blueprints = []
        
        for bp_file in self.output_dir.glob("*.json"):
            try:
                data = json.loads(bp_file.read_text())
                meta = data.get("blueprint_meta", {})
                generated_at = meta.get("generated_at", "")
                
                blueprints.append({
                    "path": bp_file,
                    "data": data,
                    "generated_at": generated_at,
                })
            except Exception as e:
                logger.warning(f"Error reading {bp_file}: {e}")
        
        if not blueprints:
            return None
        
        # Sort by generated_at (oldest first)
        blueprints.sort(key=lambda x: x["generated_at"])
        
        return blueprints[0]


# Convenience functions
async def generate_blueprint_for_topic(
    topic: str,
    research_summary: str = "",
    provider: str = "openai",
) -> BlueprintGenerationResult:
    """Quick way to generate a blueprint."""
    generator = BlueprintGenerator(provider=provider)
    return await generator.generate_blueprint(topic, research_summary)


def get_pending_blueprints() -> List[Dict[str, Any]]:
    """Get list of blueprints waiting for content generation."""
    generator = BlueprintGenerator()
    return generator.list_generated()


def get_completed_blueprints() -> List[Dict[str, Any]]:
    """Get list of blueprints that have been converted to content."""
    generator = BlueprintGenerator()
    return generator.list_created()
