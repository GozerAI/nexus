"""
AI Content Generation for Content Library.

Provides LLM-powered content generation:
- Concept explanations
- Procedure guides
- Code exercises
- Quizzes and assessments
- Flashcards
- Content enhancement and translation

Integrates with knowledge base for grounded generation.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .models import (
    ContentItem,
    ContentFormat,
    ContentType,
    ContentStatus,
    ContentSourceType,
    ContentQualityMetrics,
    DifficultyLevel,
    LearningStyle,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ContentGenerationConfig:
    """Configuration for AI content generation."""
    # Model settings
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9

    # Content options
    include_examples: bool = True
    include_exercises: bool = True
    include_summary: bool = True
    include_prerequisites: bool = True

    # Target audience
    target_reading_level: str = "intermediate"
    target_audience: str = "general"
    language: str = "en"

    # Style
    tone: str = "educational"  # educational, casual, formal, technical
    verbosity: str = "moderate"  # concise, moderate, detailed

    # Structure
    max_sections: int = 5
    include_headings: bool = True
    use_bullet_points: bool = True


class GenerationMode(Enum):
    """Content generation modes."""
    STANDARD = "standard"           # Basic generation
    KNOWLEDGE_GROUNDED = "knowledge_grounded"  # Uses knowledge base
    TEMPLATE_BASED = "template_based"  # Uses templates
    ENHANCEMENT = "enhancement"     # Enhances existing content


# =============================================================================
# Prompt Templates
# =============================================================================

class PromptTemplates:
    """Pre-built prompt templates for different content types."""

    CONCEPT_EXPLANATION = """
Create an educational explanation of the following concept for {difficulty} level learners:

Topic: {topic}
Target Audience: {audience}
Learning Style Preference: {learning_style}

Requirements:
- Provide a clear, accessible explanation
- Include {example_count} practical examples
- Use analogies to make abstract concepts concrete
- Structure the content with clear sections
- End with a brief summary of key points

{additional_context}

Please generate the content in {language}.
"""

    PROCEDURE_GUIDE = """
Create a step-by-step guide for the following task:

Task: {task}
Difficulty Level: {difficulty}
Prerequisites: {prerequisites}

Requirements:
- Break down into clear, numbered steps
- Include expected outcomes for each step
- Highlight common pitfalls and how to avoid them
- Include {example_count} examples where appropriate
- Provide troubleshooting tips

{additional_context}

Please generate the content in {language}.
"""

    CODE_EXERCISE = """
Create a coding exercise for the following concept:

Concept: {concept}
Programming Language: {language}
Difficulty Level: {difficulty}

Requirements:
- Provide a clear problem statement
- Include starter code with TODO comments
- Add test cases for verification
- Provide hints (hidden by default)
- Include a solution with explanation

Expected Skills:
{skills}

{additional_context}
"""

    QUIZ = """
Create a quiz on the following topics:

Topics: {topics}
Difficulty Level: {difficulty}
Number of Questions: {num_questions}
Question Types: {question_types}

Requirements:
- Mix of question types as specified
- Clear, unambiguous questions
- Plausible distractors for multiple choice
- Include explanations for correct answers
- Vary difficulty within the specified level

{additional_context}

Please generate the content in {language}.
"""

    FLASHCARD_SET = """
Create a set of flashcards for studying:

Topic: {topic}
Number of Cards: {num_cards}
Difficulty Level: {difficulty}

Requirements:
- Clear, concise front (question/prompt)
- Comprehensive but focused back (answer)
- Include mnemonic hints where helpful
- Progress from basic to advanced concepts
- Include key terms and definitions

{additional_context}

Please generate the content in {language}.
"""

    CONTENT_ENHANCEMENT = """
Enhance the following educational content:

Original Content:
{original_content}

Enhancement Requests:
{enhancements}

Requirements:
- Maintain the core message and accuracy
- Improve clarity and engagement
- Add requested elements
- Preserve the appropriate difficulty level
- Ensure consistent tone and style

Please provide the enhanced content in {language}.
"""

    SIMPLIFICATION = """
Simplify the following content for a {target_level} audience:

Original Content:
{original_content}

Requirements:
- Reduce complexity while maintaining accuracy
- Use simpler vocabulary and shorter sentences
- Add clarifying examples
- Break down complex concepts
- Maintain educational value

Please provide the simplified content in {language}.
"""


# =============================================================================
# Generation Result
# =============================================================================

@dataclass
class GenerationResult:
    """Result of content generation."""
    success: bool
    content: Optional[ContentItem]
    raw_output: str = ""
    prompt_used: str = ""
    model_used: str = ""
    tokens_used: int = 0
    generation_time_seconds: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Content Generator
# =============================================================================

class ContentGenerator:
    """
    AI-powered content generation using LLM.

    Generates various types of educational content:
    - Concept explanations
    - Procedure guides
    - Code exercises
    - Quizzes and assessments
    - Flashcard sets

    Can optionally integrate with knowledge base for grounded generation.
    """

    def __init__(
        self,
        llm_provider: Optional[Callable] = None,
        knowledge_base=None,
        kag_engine=None,
        default_config: Optional[ContentGenerationConfig] = None
    ):
        """
        Initialize content generator.

        Args:
            llm_provider: Callable for LLM inference (prompt -> response)
            knowledge_base: Optional knowledge base for grounded generation
            kag_engine: Optional KAG engine for verification
            default_config: Default generation configuration
        """
        self.llm = llm_provider
        self.knowledge_base = knowledge_base
        self.kag_engine = kag_engine
        self.default_config = default_config or ContentGenerationConfig()

        # Statistics
        self.stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_tokens_used": 0,
            "by_type": {}
        }

        logger.info("ContentGenerator initialized")

    def _has_llm(self) -> bool:
        """Check if LLM provider is available."""
        return self.llm is not None

    def _generate_with_llm(self, prompt: str, config: ContentGenerationConfig) -> str:
        """Call LLM provider with prompt."""
        if not self.llm:
            # Return placeholder for testing without LLM
            return f"[AI-Generated Content]\n\nPrompt Summary: {prompt[:200]}..."

        try:
            response = self.llm(
                prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p
            )
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _get_knowledge_context(self, topic: str, limit: int = 5) -> str:
        """Get relevant knowledge from knowledge base."""
        if not self.knowledge_base:
            return ""

        try:
            results = self.knowledge_base.query_knowledge(topic, max_results=limit)
            if results:
                context_parts = []
                for item in results:
                    content = str(item.content)[:500]
                    context_parts.append(f"- {content}")
                return "Relevant Knowledge:\n" + "\n".join(context_parts)
        except Exception as e:
            logger.warning(f"Failed to get knowledge context: {e}")

        return ""

    async def generate_concept_explanation(
        self,
        topic: str,
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        config: Optional[ContentGenerationConfig] = None,
        learning_style: LearningStyle = LearningStyle.MULTIMODAL,
        use_knowledge_base: bool = True
    ) -> GenerationResult:
        """
        Generate a concept explanation.

        Args:
            topic: Topic to explain
            difficulty: Target difficulty level
            config: Generation configuration
            learning_style: Target learning style
            use_knowledge_base: Whether to use KB for grounding

        Returns:
            GenerationResult with generated content
        """
        import time
        start_time = time.time()

        config = config or self.default_config

        # Get knowledge context
        additional_context = ""
        if use_knowledge_base:
            additional_context = self._get_knowledge_context(topic)

        # Build prompt
        prompt = PromptTemplates.CONCEPT_EXPLANATION.format(
            topic=topic,
            difficulty=difficulty.value,
            audience=config.target_audience,
            learning_style=learning_style.value,
            example_count=3 if config.include_examples else 0,
            language=config.language,
            additional_context=additional_context
        )

        try:
            # Generate content
            raw_output = self._generate_with_llm(prompt, config)

            # Create ContentItem
            content = ContentItem(
                content_id=str(uuid.uuid4()),
                title=f"Understanding {topic}",
                description=f"A {difficulty.value} level explanation of {topic}",
                content_body=raw_output,
                content_format=ContentFormat.MARKDOWN,
                content_type=ContentType.CONCEPT,
                difficulty=difficulty,
                topics=[topic],
                tags=[topic.lower(), "concept", "explanation"],
                learning_styles=[learning_style],
                estimated_time_minutes=self._estimate_reading_time(raw_output),
                source_type=ContentSourceType.GENERATED,
                status=ContentStatus.DRAFT
            )

            generation_time = time.time() - start_time
            self._update_stats(ContentType.CONCEPT, True)

            return GenerationResult(
                success=True,
                content=content,
                raw_output=raw_output,
                prompt_used=prompt,
                model_used=config.model,
                generation_time_seconds=generation_time
            )

        except Exception as e:
            self._update_stats(ContentType.CONCEPT, False)
            return GenerationResult(
                success=False,
                content=None,
                error=str(e),
                prompt_used=prompt
            )

    async def generate_procedure_guide(
        self,
        task: str,
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        prerequisites: Optional[List[str]] = None,
        config: Optional[ContentGenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a step-by-step procedure guide.

        Args:
            task: Task to document
            difficulty: Target difficulty
            prerequisites: Required prerequisites
            config: Generation configuration

        Returns:
            GenerationResult with generated content
        """
        import time
        start_time = time.time()

        config = config or self.default_config
        prereqs = ", ".join(prerequisites) if prerequisites else "None specified"

        # Get knowledge context
        additional_context = ""
        if self.knowledge_base:
            additional_context = self._get_knowledge_context(task)

        # Build prompt
        prompt = PromptTemplates.PROCEDURE_GUIDE.format(
            task=task,
            difficulty=difficulty.value,
            prerequisites=prereqs,
            example_count=2 if config.include_examples else 0,
            language=config.language,
            additional_context=additional_context
        )

        try:
            raw_output = self._generate_with_llm(prompt, config)

            content = ContentItem(
                content_id=str(uuid.uuid4()),
                title=f"How to: {task}",
                description=f"Step-by-step guide for {task}",
                content_body=raw_output,
                content_format=ContentFormat.MARKDOWN,
                content_type=ContentType.PROCEDURE,
                difficulty=difficulty,
                topics=[task],
                tags=[task.lower(), "procedure", "guide", "how-to"],
                prerequisites=prerequisites or [],
                estimated_time_minutes=self._estimate_reading_time(raw_output),
                source_type=ContentSourceType.GENERATED,
                status=ContentStatus.DRAFT
            )

            generation_time = time.time() - start_time
            self._update_stats(ContentType.PROCEDURE, True)

            return GenerationResult(
                success=True,
                content=content,
                raw_output=raw_output,
                prompt_used=prompt,
                model_used=config.model,
                generation_time_seconds=generation_time
            )

        except Exception as e:
            self._update_stats(ContentType.PROCEDURE, False)
            return GenerationResult(
                success=False,
                content=None,
                error=str(e),
                prompt_used=prompt
            )

    async def generate_code_exercise(
        self,
        concept: str,
        language: str = "python",
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        skills: Optional[List[str]] = None,
        config: Optional[ContentGenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a coding exercise.

        Args:
            concept: Programming concept to practice
            language: Programming language
            difficulty: Difficulty level
            skills: Skills to practice
            config: Generation configuration

        Returns:
            GenerationResult with generated exercise
        """
        import time
        start_time = time.time()

        config = config or self.default_config
        skills_str = "\n".join(f"- {s}" for s in (skills or [concept]))

        prompt = PromptTemplates.CODE_EXERCISE.format(
            concept=concept,
            language=language,
            difficulty=difficulty.value,
            skills=skills_str,
            additional_context=""
        )

        try:
            raw_output = self._generate_with_llm(prompt, config)

            content = ContentItem(
                content_id=str(uuid.uuid4()),
                title=f"{language.title()} Exercise: {concept}",
                description=f"Practice {concept} in {language}",
                content_body=raw_output,
                content_format=ContentFormat.CODE_EXERCISE,
                content_type=ContentType.EXERCISE,
                difficulty=difficulty,
                topics=[concept, language],
                tags=[concept.lower(), language.lower(), "exercise", "code"],
                learning_styles=[LearningStyle.KINESTHETIC],
                estimated_time_minutes=max(15, self._estimate_reading_time(raw_output) * 2),
                source_type=ContentSourceType.GENERATED,
                status=ContentStatus.DRAFT,
                metadata={"programming_language": language, "skills": skills or []}
            )

            generation_time = time.time() - start_time
            self._update_stats(ContentType.EXERCISE, True)

            return GenerationResult(
                success=True,
                content=content,
                raw_output=raw_output,
                prompt_used=prompt,
                model_used=config.model,
                generation_time_seconds=generation_time
            )

        except Exception as e:
            self._update_stats(ContentType.EXERCISE, False)
            return GenerationResult(
                success=False,
                content=None,
                error=str(e),
                prompt_used=prompt
            )

    async def generate_quiz(
        self,
        topics: List[str],
        num_questions: int = 5,
        question_types: Optional[List[str]] = None,
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        config: Optional[ContentGenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a quiz/assessment.

        Args:
            topics: Topics to cover
            num_questions: Number of questions
            question_types: Types of questions (multiple_choice, true_false, short_answer)
            difficulty: Difficulty level
            config: Generation configuration

        Returns:
            GenerationResult with generated quiz
        """
        import time
        start_time = time.time()

        config = config or self.default_config
        question_types = question_types or ["multiple_choice", "true_false"]

        prompt = PromptTemplates.QUIZ.format(
            topics=", ".join(topics),
            difficulty=difficulty.value,
            num_questions=num_questions,
            question_types=", ".join(question_types),
            language=config.language,
            additional_context=""
        )

        try:
            raw_output = self._generate_with_llm(prompt, config)

            content = ContentItem(
                content_id=str(uuid.uuid4()),
                title=f"Quiz: {', '.join(topics[:2])}",
                description=f"Assessment covering {', '.join(topics)}",
                content_body=raw_output,
                content_format=ContentFormat.QUIZ,
                content_type=ContentType.ASSESSMENT,
                difficulty=difficulty,
                topics=topics,
                tags=[t.lower() for t in topics] + ["quiz", "assessment"],
                estimated_time_minutes=num_questions * 2,
                source_type=ContentSourceType.GENERATED,
                status=ContentStatus.DRAFT,
                metadata={
                    "num_questions": num_questions,
                    "question_types": question_types
                }
            )

            generation_time = time.time() - start_time
            self._update_stats(ContentType.ASSESSMENT, True)

            return GenerationResult(
                success=True,
                content=content,
                raw_output=raw_output,
                prompt_used=prompt,
                model_used=config.model,
                generation_time_seconds=generation_time
            )

        except Exception as e:
            self._update_stats(ContentType.ASSESSMENT, False)
            return GenerationResult(
                success=False,
                content=None,
                error=str(e),
                prompt_used=prompt
            )

    async def generate_flashcards(
        self,
        topic: str,
        num_cards: int = 10,
        difficulty: DifficultyLevel = DifficultyLevel.INTERMEDIATE,
        config: Optional[ContentGenerationConfig] = None
    ) -> GenerationResult:
        """
        Generate a set of flashcards.

        Args:
            topic: Topic to study
            num_cards: Number of cards
            difficulty: Difficulty level
            config: Generation configuration

        Returns:
            GenerationResult with flashcard content
        """
        import time
        start_time = time.time()

        config = config or self.default_config

        prompt = PromptTemplates.FLASHCARD_SET.format(
            topic=topic,
            num_cards=num_cards,
            difficulty=difficulty.value,
            language=config.language,
            additional_context=""
        )

        try:
            raw_output = self._generate_with_llm(prompt, config)

            content = ContentItem(
                content_id=str(uuid.uuid4()),
                title=f"Flashcards: {topic}",
                description=f"Study flashcards for {topic}",
                content_body=raw_output,
                content_format=ContentFormat.FLASHCARD,
                content_type=ContentType.FACT,
                difficulty=difficulty,
                topics=[topic],
                tags=[topic.lower(), "flashcards", "study"],
                learning_styles=[LearningStyle.READING, LearningStyle.VISUAL],
                estimated_time_minutes=num_cards,
                source_type=ContentSourceType.GENERATED,
                status=ContentStatus.DRAFT,
                metadata={"num_cards": num_cards}
            )

            generation_time = time.time() - start_time
            self._update_stats(ContentType.FACT, True)

            return GenerationResult(
                success=True,
                content=content,
                raw_output=raw_output,
                prompt_used=prompt,
                model_used=config.model,
                generation_time_seconds=generation_time
            )

        except Exception as e:
            self._update_stats(ContentType.FACT, False)
            return GenerationResult(
                success=False,
                content=None,
                error=str(e),
                prompt_used=prompt
            )

    async def enhance_content(
        self,
        content: ContentItem,
        enhancements: List[str],
        config: Optional[ContentGenerationConfig] = None
    ) -> GenerationResult:
        """
        Enhance existing content.

        Args:
            content: Content to enhance
            enhancements: List of enhancement requests (e.g., "add examples", "improve clarity")
            config: Generation configuration

        Returns:
            GenerationResult with enhanced content
        """
        import time
        start_time = time.time()

        config = config or self.default_config

        prompt = PromptTemplates.CONTENT_ENHANCEMENT.format(
            original_content=content.content_body,
            enhancements="\n".join(f"- {e}" for e in enhancements),
            language=config.language
        )

        try:
            raw_output = self._generate_with_llm(prompt, config)

            # Create new version
            enhanced = ContentItem(
                content_id=content.content_id,
                version=content.version + 1,
                title=content.title,
                description=content.description,
                content_body=raw_output,
                content_format=content.content_format,
                content_type=content.content_type,
                difficulty=content.difficulty,
                topics=content.topics,
                tags=content.tags + ["enhanced"],
                prerequisites=content.prerequisites,
                learning_styles=content.learning_styles,
                estimated_time_minutes=self._estimate_reading_time(raw_output),
                source_type=ContentSourceType.GENERATED,
                source_id=content.content_id,
                status=ContentStatus.DRAFT,
                metadata={
                    **content.metadata,
                    "enhancements_applied": enhancements,
                    "original_version": content.version
                }
            )

            generation_time = time.time() - start_time

            return GenerationResult(
                success=True,
                content=enhanced,
                raw_output=raw_output,
                prompt_used=prompt,
                model_used=config.model,
                generation_time_seconds=generation_time
            )

        except Exception as e:
            return GenerationResult(
                success=False,
                content=None,
                error=str(e),
                prompt_used=prompt
            )

    async def simplify_content(
        self,
        content: ContentItem,
        target_level: DifficultyLevel = DifficultyLevel.BEGINNER,
        config: Optional[ContentGenerationConfig] = None
    ) -> GenerationResult:
        """
        Simplify content for a lower difficulty level.

        Args:
            content: Content to simplify
            target_level: Target difficulty level
            config: Generation configuration

        Returns:
            GenerationResult with simplified content
        """
        import time
        start_time = time.time()

        config = config or self.default_config

        prompt = PromptTemplates.SIMPLIFICATION.format(
            original_content=content.content_body,
            target_level=target_level.value,
            language=config.language
        )

        try:
            raw_output = self._generate_with_llm(prompt, config)

            simplified = ContentItem(
                content_id=str(uuid.uuid4()),
                title=f"{content.title} (Simplified)",
                description=f"Simplified version of: {content.description}",
                content_body=raw_output,
                content_format=content.content_format,
                content_type=content.content_type,
                difficulty=target_level,
                topics=content.topics,
                tags=content.tags + ["simplified"],
                prerequisites=[],
                learning_styles=content.learning_styles,
                estimated_time_minutes=self._estimate_reading_time(raw_output),
                source_type=ContentSourceType.GENERATED,
                source_id=content.content_id,
                status=ContentStatus.DRAFT,
                metadata={
                    "simplified_from": content.content_id,
                    "original_difficulty": content.difficulty.value
                }
            )

            generation_time = time.time() - start_time

            return GenerationResult(
                success=True,
                content=simplified,
                raw_output=raw_output,
                prompt_used=prompt,
                model_used=config.model,
                generation_time_seconds=generation_time
            )

        except Exception as e:
            return GenerationResult(
                success=False,
                content=None,
                error=str(e),
                prompt_used=prompt
            )

    async def generate_learning_path_content(
        self,
        topic: str,
        depth: int = 5,
        config: Optional[ContentGenerationConfig] = None
    ) -> List[GenerationResult]:
        """
        Generate a complete learning path of content.

        Creates a sequence of content items from beginner to advanced.

        Args:
            topic: Main topic
            depth: Number of content items to generate
            config: Generation configuration

        Returns:
            List of GenerationResults
        """
        results = []

        # Map depth to difficulty progression
        difficulties = [
            DifficultyLevel.NOVICE,
            DifficultyLevel.BEGINNER,
            DifficultyLevel.INTERMEDIATE,
            DifficultyLevel.ADVANCED,
            DifficultyLevel.EXPERT
        ]

        for i in range(min(depth, len(difficulties))):
            difficulty = difficulties[i]

            # Alternate between content types
            if i % 2 == 0:
                result = await self.generate_concept_explanation(
                    topic=f"{topic} - Part {i + 1}",
                    difficulty=difficulty,
                    config=config
                )
            else:
                result = await self.generate_code_exercise(
                    concept=topic,
                    difficulty=difficulty,
                    config=config
                )

            results.append(result)

            # Link content
            if len(results) > 1 and results[-2].content and results[-1].content:
                results[-2].content.next_content = results[-1].content.content_id
                results[-1].content.previous_content = results[-2].content.content_id

        return results

    def _estimate_reading_time(self, content: str) -> int:
        """Estimate reading time in minutes (assuming 200 wpm)."""
        words = len(content.split())
        return max(5, words // 200)

    def _update_stats(self, content_type: ContentType, success: bool):
        """Update generation statistics."""
        self.stats["total_generations"] += 1
        if success:
            self.stats["successful_generations"] += 1
        else:
            self.stats["failed_generations"] += 1

        type_key = content_type.value
        if type_key not in self.stats["by_type"]:
            self.stats["by_type"][type_key] = {"success": 0, "failed": 0}

        if success:
            self.stats["by_type"][type_key]["success"] += 1
        else:
            self.stats["by_type"][type_key]["failed"] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.stats.copy()


# =============================================================================
# Factory Function
# =============================================================================

def create_content_generator(
    llm_provider: Optional[Callable] = None,
    knowledge_base=None,
    kag_engine=None,
    config: Optional[ContentGenerationConfig] = None
) -> ContentGenerator:
    """
    Factory function to create content generator.

    Args:
        llm_provider: Callable for LLM inference
        knowledge_base: Optional knowledge base
        kag_engine: Optional KAG engine
        config: Default generation configuration

    Returns:
        ContentGenerator instance
    """
    return ContentGenerator(
        llm_provider=llm_provider,
        knowledge_base=knowledge_base,
        kag_engine=kag_engine,
        default_config=config
    )
